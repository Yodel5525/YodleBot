"""S08: the simulator seat and the harness (docs/LIVE_SEARCH_PLAN.md S08; tools/rs_search.py THE
SEAT, tools/rs_match.py SEARCH SEATS, tools/rs_league_eval.py SEARCH PLAYERS).

No engine, no torch, no data. What is checked (the plan's S08 list first):
  * AgentSpec.to_dict is unchanged with the defaults (the keys of every spec before S08, so the
    specs, game keys and identities of earlier runs do not change), and round-trips with them;
  * search normalises to its full content (seat_search: SearchConfig and the pinned rebuild
    settings, "name@probes" / "name@R2"), compares equal however it was written, and refuses
    an idle seat, a lead, hold_pending, a bad latency or rebuild level;
  * MatchSpec.engine_config charges each agent's latency_ticks to its own team, plain and
    swapped, with an int, a pair or no spec delay; without latency it is the config it was;
  * build_seat builds a SearchSeat only for an agent with a search, armed with its own and the
    opponent's delay from the spec's engine config; seat_info adds "search" and
    "latency_ticks" only for such seats;
  * a whole match on conftest's fake engine through build_seat: the charged delay reaches the
    battle and the result, the seat's own clock pairs with run_match's seat.sent, the search
    summary lands in the result and the match battle sees no save / load / step from the seat;
  * the command line (rs_match --search-a / --latency-a / --search-verify; rs_league_eval
    search= / latency_ticks= options, --worker-mb), the privileged gate decision for s_true
    (from the names and from the games' records), the warnings, merge_search_info, and that
    importing tools/rs_search imports no rs_match.
The search itself is test_rs_search.py's (its FakePolicy and FakeRebuilder serve here), the leak
test is test_search_no_leak.py.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import DEFAULT_CATALOGUE, make_fake_royalesim

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.pipeline.test_rs_search import FakePolicy, FakeRebuilder, ScriptRng  # noqa: E402

HOG = ("Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons", "Log")
OPP = ("Archer", "Arrows", "BabyDragon", "Giant", "Knight", "MiniPekka", "Valkyrie", "Zap")
#: The fake catalogue's columns by name (conftest keeps the cost in column 1).
FAKE_FIELDS = ("name", "elixir", "c2", "c3", "c4", "flying", "c6", "c7", "card_kind")
#: AgentSpec.to_dict's keys before the presses switch (test_rs_match_v2.OLD_AGENT_KEYS).
OLD_AGENT_KEYS = ["ckpt", "name", "schema", "lead", "noise", "noise_seed", "decide", "visibility",
                  "wait_keep", "spell_aim", "hold_pending", "seed_offset", "allow_drift"]
#: seat_info's keys before S08.
OLD_INFO_KEYS = {"name", "ckpt", "lead", "decide", "visibility", "schema", "obs_delay",
                 "live_compat", "presses", "buttons"}


@pytest.fixture
def env(monkeypatch):
    """rs_rows, rs_eval, rs_match, rs_league_eval and rs_search as the tools import them (plain
    names, tools/ on sys.path), rs_rows' settings at their defaults, no engine config from the
    PC, rs_elixir's constant law and the fake catalogue cached for RSAgent (as
    test_rs_match_v2's fixture)."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    ev = importlib.import_module("rs_eval")
    rm = importlib.import_module("rs_match")
    le = importlib.import_module("rs_league_eval")
    sr = importlib.import_module("rs_search")
    rb = importlib.import_module("rs_rebuild")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "VISIBILITY": rr.VIS_LEGACY, "TRUST": rr.TrustRule(), "OUT": rr.OUT,
                        "CARD_FLY": {}, "CARD_BUILDING": {}, "PRESSES": "off",
                        "BUTTONS": 2, "HAND_SIZE": 4, "DECISION": rr.DecisionRule(),
                        "ROW_SCHEMA": "rs_rows/legacy", "SF": None, "UNIT_FORMS": None,
                        "SPELL_CARDS": None}.items():
        monkeypatch.setattr(rr, name, value)
    law = rr.rs_elixir.ElixirLaw.constants()
    monkeypatch.setattr(rr.rs_elixir, "default_law", lambda: law)
    monkeypatch.setattr(ev.RSAgent, "_CAT", [list(r) for r in DEFAULT_CATALOGUE])
    return SimpleNamespace(rows=rr, eval=ev, match=rm, league=le, search=sr, rebuild=rb,
                           engine=eng, law=law, cfg=eng.EngineConfig.defaults())


def pinned(env, settings=None):
    s = settings or env.rebuild.RebuildSettings()
    return {k: v for k, v in s.to_json().items() if k not in ("probes", "notes")}


# ---------------------------------------------------------------------------
# AgentSpec: the defaults keep every old spec, search and latency_ticks
# ---------------------------------------------------------------------------

def test_agent_spec_to_dict_is_unchanged_without_a_search(env):
    M = env.match
    plain = M.AgentSpec("a.pt", name="x")
    assert plain.search is None and plain.latency_ticks == 0 and not plain.privileged
    assert list(plain.to_dict()) == OLD_AGENT_KEYS
    assert M.AgentSpec.from_dict(plain.to_dict()) == plain
    assert M.AgentSpec("a.pt", name="x", search=None, latency_ticks=0) == plain
    assert M.AgentSpec("a.pt", name="x", search="off") == plain       # off: no search
    spec = M.MatchSpec(decks=(HOG, OPP), agents=(plain, M.AgentSpec(name="idle")), delay=24,
                       engine=env.cfg)
    assert [list(a) for a in spec.to_dict()["agents"]] == [OLD_AGENT_KEYS, OLD_AGENT_KEYS]
    assert "search" not in json.dumps(spec.to_dict()) and "latency" not in json.dumps(spec.to_dict())
    # the league's game key is a sha1 of exactly that dict: unchanged for these specs
    LE = env.league
    assert LE.game_key(spec) == LE.game_key(M.MatchSpec.from_dict(spec.to_dict()))
    assert LE.identity(plain, None) == {**plain.to_dict(), "ckpt_digest": None}


def test_agent_spec_search_holds_its_full_content(env, monkeypatch):
    M, SR, RB = env.match, env.search, env.rebuild
    a = M.AgentSpec("a.pt", name="s", search="s_root", latency_ticks=2)
    full = SR.SearchConfig.named("s_root").to_json()
    assert a.search == {**full, "rebuild": pinned(env)} == SR.seat_search("s_root")
    assert list(a.to_dict()) == OLD_AGENT_KEYS + ["search", "latency_ticks"]
    assert a.to_dict()["search"] == a.search and a.to_dict()["latency_ticks"] == 2
    back = M.AgentSpec.from_dict(json.loads(json.dumps(a.to_dict())))
    assert back == a and back.search == a.search
    # however it was written: a preset, its expanded dict, JSON text, the seat dict itself
    for same in (SR.expand("s_root"), json.dumps(SR.expand("s_root")), a.search,
                 {"preset": "s_root"}, " s_root@defaults "):
        assert M.AgentSpec("a.pt", name="s", search=same, latency_ticks=2) == a
    # the rebuild level and a field override change the content (and so the game key)
    r2 = M.AgentSpec("a.pt", search="s_root@R2")
    assert r2.search["rebuild"]["level"] == "R2" and r2.search["level"] is None
    assert r2.search != M.AgentSpec("a.pt", search="s_root").search
    idle = M.AgentSpec("a.pt", search='{"preset": "s_idle", "horizon": 40}')
    assert (idle.search["preset"], idle.search["horizon"], idle.search["max_trials"]) == (
        "s_idle", 40, 8)
    # @probes pins the S01 report's settings when the spec is made, provenance left out
    report = RB.RebuildSettings(spec_per="formation", probes="reports/x.json", notes=("P1 FAIL",))
    monkeypatch.setattr(RB.RebuildSettings, "from_report",
                        classmethod(lambda cls, *args, **kw: report))
    probed = M.AgentSpec("a.pt", search="s_idle@probes+R2")
    assert probed.search["rebuild"] == pinned(env, report.replace(level="R2"))
    assert "probes" not in probed.search["rebuild"] and "notes" not in probed.search["rebuild"]
    monkeypatch.setattr(RB.RebuildSettings, "from_report", None)   # never read again
    assert M.AgentSpec.from_dict(probed.to_dict()) == probed
    # the privileged ceiling says so
    assert M.AgentSpec("a.pt", search="s_true").privileged and not a.privileged


@pytest.mark.parametrize("kw, match", [
    ({"ckpt": "idle", "search": "s_root"}, "idle"),
    ({"ckpt": "", "search": "s_idle"}, "idle"),
    ({"ckpt": "a.pt", "search": "s_root", "lead": -24}, "lead"),
    ({"ckpt": "a.pt", "search": "s_root", "hold_pending": True}, "hold_pending"),
    ({"ckpt": "a.pt", "latency_ticks": -1}, "latency_ticks"),
    ({"ckpt": "a.pt", "latency_ticks": True}, "latency_ticks"),
    ({"ckpt": "a.pt", "search": "s_fast"}, "preset"),
    ({"ckpt": "a.pt", "search": "s_root@R0"}, "rebuild"),
    ({"ckpt": "a.pt", "search": "s_root@probes+defaults"}, "not both"),
    ({"ckpt": "a.pt", "search": {"preset": "s_root", "rebuild": {"level": "R0"}}}, "R0"),
    ({"ckpt": "a.pt", "search": {"preset": "s_root", "rebuild": {"lvl": "R1"}}}, "unknown"),
    ({"ckpt": "a.pt", "search": {"preset": "s_idle", "horizn": 40}}, "unknown"),
])
def test_agent_spec_refuses_what_a_search_seat_cannot_be(env, kw, match):
    with pytest.raises(ValueError, match=match):
        env.match.AgentSpec(**kw)


# ---------------------------------------------------------------------------
# MatchSpec.engine_config: the latency charge
# ---------------------------------------------------------------------------

def test_engine_config_charges_latency_to_its_own_team(env):
    M = env.match
    srch = M.AgentSpec("a.pt", name="s", search="s_root", latency_ticks=2)
    plain_a, plain_b = M.AgentSpec("a.pt", name="p"), M.AgentSpec("b.pt", name="r")
    spec = M.MatchSpec(decks=(HOG, OPP), agents=(srch, plain_b), delay=24, engine=env.cfg)
    assert spec.latency() == (2, 0) and spec.engine_config().delay == (26, 24)
    sw = spec.swapped()                                 # the charge follows the agent
    assert sw.agents[1] == srch and sw.latency() == (0, 2) and sw.engine_config().delay == (24, 26)
    # without latency: exactly the config it always was (EngineConfig.delayed)
    plain = M.MatchSpec(decks=(HOG, OPP), agents=(plain_a, plain_b), delay=24, engine=env.cfg)
    assert plain.engine_config() == plain.base_engine().delayed(24)
    assert plain.engine_config().to_json() == plain.base_engine().delayed(24).to_json()
    # a (blue, red) delay, and none (the engine config's own)
    pair = spec.replace(delay=(20, 24))
    assert pair.engine_config().delay == (22, 24) and pair.swapped().engine_config().delay == (24, 22)
    none = spec.replace(delay=None)
    assert none.engine_config().delay == (2, 0)
    assert plain.replace(delay=None).engine_config() == plain.base_engine()
    assert M.seat_delays(spec.engine_config()) == (26, 24)
    assert M.seat_delays(plain.replace(delay=None).engine_config()) == (0, 0)


# ---------------------------------------------------------------------------
# build_seat, seat_info, a whole match
# ---------------------------------------------------------------------------

def stub_init(cfg):
    """RSAgent.__init__ without a checkpoint (no torch): what the seat reads of it."""
    def fake_init(self, ckpt, seed=0, wait_keep=0.35, *, engine=None, allow_drift=None):
        self.ckpt, self.seed = ckpt, seed
        self.engine = engine if engine is not None else cfg
        self.meta, self.model, self.torch, self.vocab_drift = {}, None, None, False
        self.visibility, self.live, self.v2 = None, False, False
        self.rng = ScriptRng(0.99)
        self.wait_bias = 0.0
    return fake_init


def test_build_seat_dispatches_on_search(env, monkeypatch):
    M, SR = env.match, env.search
    monkeypatch.setattr(env.eval.RSAgent, "__init__", stub_init(env.cfg))
    srch = M.AgentSpec("a.pt", name="s", search="s_root", latency_ticks=2, seed_offset=3)
    plain = M.AgentSpec("b.pt", name="p")
    spec = M.MatchSpec(decks=(HOG, OPP), agents=(srch, plain), delay=24, engine=env.cfg, seed=11)
    cfg = spec.engine_config()
    seat = M.build_seat(srch, 0, spec, cfg)
    assert isinstance(seat, SR.SearchSeat) and isinstance(seat, M.MatchAgent)
    assert isinstance(seat, SR.SearchSeatMixin) and type(seat).__name__ == "SearchSeat"
    assert (seat.search_team, seat.search_delay, seat.search_opp_delay) == (0, 26, 24)
    assert seat.search_seed == 11 and seat.seed == 14 and seat.cadence == spec.cadence
    assert seat.search_config == SR.SearchConfig.named("s_root")
    assert seat.rebuild_settings == env.rebuild.RebuildSettings()
    assert seat.searcher is None and seat.lead == 0 and seat.decide_mode == "live"
    other = M.build_seat(plain, 1, spec, cfg)
    assert type(other) is M.MatchAgent
    assert M.build_seat(M.AgentSpec(name="idle"), 1, spec, cfg) is None
    # the swapped game: the seat sits on team 1, its delay (with its charge) is red's
    sw = spec.swapped()
    seat1 = M.build_seat(sw.agents[1], 1, sw, sw.engine_config())
    assert (seat1.search_team, seat1.search_delay, seat1.search_opp_delay) == (1, 26, 24)
    # seat_info: "search" and "latency_ticks" only for the search seat
    info = M.seat_info(seat, srch)
    assert set(info) == OLD_INFO_KEYS | {"search", "latency_ticks"}
    assert info["latency_ticks"] == 2 and info["search"]["preset"] == "s_root"
    assert (info["search"]["delay"], info["search"]["privileged"]) == (26, False)
    assert info["search"]["rebuild_level"] == "R1"
    assert set(M.seat_info(other, plain)) == OLD_INFO_KEYS
    assert set(M.seat_info(None, M.AgentSpec(name="idle"))) == OLD_INFO_KEYS
    json.dumps(info)
    # setup_search refuses a lead, hold_pending and s_reply (S07)
    seat.lead = 3
    with pytest.raises(SR.SearchConfigError, match="lead"):
        seat.setup_search("s_root", team=0, delay=24)
    seat.lead, seat.hold_pending = 0, True
    with pytest.raises(SR.SearchConfigError, match="hold_pending"):
        seat.setup_search("s_root", team=0, delay=24)
    seat.hold_pending = False
    with pytest.raises(SR.SearchConfigError, match="S07"):
        seat.setup_search("s_reply", team=0, delay=24)
    with pytest.raises(SR.SearchConfigError, match="off"):
        seat.setup_search("off", team=0, delay=24)
    # a seat acts only for the team it was set up for
    seat.setup_search("s_root", team=1, delay=24)
    with pytest.raises(SR.SearchError, match="team 1"):
        seat.act(None, {}, 0, {}, 100, 18, None)


def test_a_search_seat_plays_a_match_through_build_seat(env, monkeypatch):
    """Delay 24, the search seat's latency 2: its commands run 26 ticks after they are sent;
    the seat decides at 90, 100, 110 (s_idle on FakeRebuilder's root, FakePolicy: the Hog every
    time, the search keeps it) and touches the match battle only through check_deploy."""
    M, SR = env.match, env.search
    mod = make_fake_royalesim()
    mod.CATALOGUE_FIELDS = list(FAKE_FIELDS)
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    monkeypatch.setattr(env.eval.RSAgent, "__init__", stub_init(env.cfg))
    rows, policy, rb = [], FakePolicy(), FakeRebuilder()

    class TestSeat(SR.SearchSeat):
        keep_records = True

        def search_policy(self):
            return policy

        def search_rebuilder(self, royalesim, sub, fields):
            assert royalesim is mod and fields == env.rows.F
            return rb

        def search_row(self, state, team, mem, sub, royalesim, tick):
            rows.append(int(tick))
            return {"tick": int(tick), "state": state}

        def search_law(self):
            return env.law

    monkeypatch.setattr(SR, "_SEAT_CLASS", TestSeat)
    made = []

    def factory(agent, team, spec, cfg):
        seat = M.build_seat(agent, team, spec, cfg)
        made.append(seat)
        return seat

    agents = (M.AgentSpec("a.pt", name="s", search="s_idle", latency_ticks=2, decide="legacy"),
              M.AgentSpec(name="idle"))
    spec = M.MatchSpec(decks=(HOG, OPP), agents=agents, delay=24, engine=env.cfg, max_ticks=120)
    r = M.run_match(spec, seat_factory=factory)
    # rs_rows._setup's battle, then the match battle; the seat built none (FakeRebuilder)
    assert len(mod.battles) == 2
    seat, battle = made[0], mod.battles[-1]
    assert isinstance(seat, TestSeat) and made[1] is None
    # the charge: the battle and the result carry 24 + 2 for the search seat's team
    assert battle.delay == (26, 24) and r.delay == (26, 24)
    assert r.decisions == (3, 0) and r.plays == (3, 0)
    # the seat's own clock (live's rule) pairs with run_match's seat.sent (the true clock); no
    # noise, so they are equal: exec = tick + 26
    assert seat._search_sent == [tuple(s) for s in seat.sent]
    assert [s[0] for s in seat.sent] == [116, 126, 136] and seat.search_stats == Counter()
    # each decision: the root row dated B = tick + 26, then the after-rows of the trials
    assert rows[0] == 116 and len(seat.records) == 3
    assert all(rec["privileged"] is False and rec["delay"] == 26 for rec in seat.records)
    info = r.info[0]["search"]
    assert (info["preset"], info["decisions"], info["delay"], info["opp_delay"]) == (
        "s_idle", 3, 26, 24)
    assert info["outcomes"] == {"searched": 3} and info["decision_ms"]["n"] == 3
    assert r.info[0]["latency_ticks"] == 2 and set(r.info[1]) == OLD_INFO_KEYS
    json.dumps(r.to_dict())
    # the match battle saw only run_match's own calls; the seat reached check_deploy alone
    assert set(battle.names()) <= {"set_command_delay_ticks", "reset", "step"}
    assert battle.deploy_checks
    # the rs_match summary of it
    s = M.search_summary([r], "s")
    assert (s["games"], s["decisions"], s["searches"], s["privileged"]) == (1, 3, 3, False)
    assert "s search (s_idle)" in M._search_line("s", s)


# ---------------------------------------------------------------------------
# the command lines
# ---------------------------------------------------------------------------

def test_rs_match_command_line_search_options(env):
    M, SR = env.match, env.search
    args = M.build_parser().parse_args(["--a", "x.pt", "--b", "y.pt"])
    a, b = M.agents_from_args(args)
    # no search option: the AgentSpecs main always built
    assert a == M.AgentSpec("x.pt", name="a", lead=0, noise=None, noise_seed=0, decide="live",
                            visibility=None, presses=False)
    assert list(a.to_dict()) == OLD_AGENT_KEYS and list(b.to_dict()) == OLD_AGENT_KEYS
    args = M.build_parser().parse_args(["--a", "x.pt", "--b", "y.pt", "--search-a", "s_idle@R2",
                                        "--latency-a", "2", "--search-verify"])
    a, b = M.agents_from_args(args)
    assert a.search["preset"] == "s_idle" and a.search["verify"] is True
    assert a.search["rebuild"]["level"] == "R2" and a.latency_ticks == 2
    assert b.search is None and b.latency_ticks == 0
    # verify only checks: the search's key is the same without it
    assert SR.seat_config(a.search).key == SR.SearchConfig.named("s_idle").key
    for bad in (["--search-a", "s_reply"], ["--search-b", "s_root", "--lead-b", "-24"],
                ["--search-a", "s_fast"]):
        args = M.build_parser().parse_args(["--a", "x.pt", "--b", "y.pt", *bad])
        with pytest.raises(ValueError):
            M.agents_from_args(args)


def test_league_search_players_options_and_identities(env):
    M, LE = env.match, env.league
    srch = LE.parse_player("srch=a.pt,search=s_idle,latency_ticks=2")
    assert srch == M.AgentSpec("a.pt", name="srch", search="s_idle", latency_ticks=2)
    assert LE.parse_player("a.pt,latency-ticks=1,search=s_root@R2").search["rebuild"]["level"] == "R2"
    assert LE.parse_player('{"ckpt": "a.pt", "search": {"preset": "s_idle", "horizon": 40}}'
                           ).search["horizon"] == 40
    with pytest.raises(ValueError, match="an int"):
        LE.parse_player("a.pt,latency_ticks=two")
    plain = LE.parse_player("srch=a.pt")
    ident = LE.identity(srch, "d" * 16)
    assert ident["search"] == srch.search and ident["latency_ticks"] == 2
    assert set(LE.identity(plain, "d" * 16)) == set(OLD_AGENT_KEYS) | {"ckpt_digest"}
    assert LE.identity_digest(ident) != LE.identity_digest(LE.identity(plain, "d" * 16))
    proto = LE.Protocol(noise=None, engine=env.cfg)
    pair = M.DeckPair("t1", "val", (HOG, OPP), ((0,) * 8, (0,) * 8),
                      (tuple(c.lower() for c in HOG), tuple(c.lower() for c in OPP)))
    champ = M.AgentSpec("h.pt", name="H")
    with_search = LE.comparison_tasks([pair], proto, srch, champ, games=2)
    without = LE.comparison_tasks([pair], proto, plain, champ, games=2)
    assert {t.key for t in with_search}.isdisjoint({t.key for t in without})
    assert [t.spec.engine_config().delay for t in with_search] == [(26, 24), (24, 26)]
    assert [t.spec.engine_config().delay for t in without] == [(24, 24), (24, 24)]
    # who is privileged, the S07 warning (none before S07), the s_reply refusal
    players = {"t": M.AgentSpec("a.pt", search="s_true"), "r": srch, "p": plain}
    assert LE.privileged_players(players) == ["t"]
    assert LE.privileged_players({n: LE.identity(a, None) for n, a in players.items()}) == ["t"]
    assert LE.privileged_players({"p": plain}) == []
    assert LE.search_warnings(players, {"t": "x", "r": "y", "p": "z"}) == []
    LE.check_search_players(players)
    with pytest.raises(ValueError, match="S07"):
        LE.check_search_players({"q": M.AgentSpec("a.pt", search="s_reply")})


def test_league_worker_mb(env):
    LE = env.league
    assert LE.worker_cap(8, available=8192, cpus=16, per_worker_mb=1800)[0] == 2
    assert LE.worker_cap(8, available=8192, cpus=16)[0] == 4            # 1200 MB each
    run = ["run", "--candidate", "a.pt", "--champion", "b.pt"]
    assert LE.build_parser().parse_args(run).worker_mb == LE.WORKER_MB
    assert LE.build_parser().parse_args(run + ["--worker-mb", "1800"]).worker_mb == 1800


def games(LE, n, *, privileged=True):
    """n games of candidate T against champion H, T winning all, T flagged privileged."""
    out = []
    for i in range(n):
        t_team = i % 2
        teams = ("T", "H") if t_team == 0 else ("H", "T")
        flags = [False, False]
        flags[t_team] = privileged
        out.append(LE.Game(key=f"k{i}", teams=teams, winner=t_team, pairing=f"p{i // 2}",
                           holder="T", protocol="pd", split="val", privileged=tuple(flags)))
    return out


def test_a_privileged_candidate_is_never_promoted(env, tmp_path):
    LE = env.league
    config = LE.GateConfig(champion_games=10)
    plain = LE.gate(games(LE, 10, privileged=False), "T", "H", config=config)
    assert plain.decision == "pass" and plain.promote
    # flagged by its games
    d = LE.gate(games(LE, 10), "T", "H", config=config)
    assert (d.decision, d.admit, d.promote, d.cycle, d.turned_on()) == (
        "privileged", False, False, False, "privileged")
    assert d.conditions["beats_champion"].passed and d.to_dict()["privileged"] is True
    assert "privileged" not in plain.to_dict()
    # flagged by name only (the run header's players)
    d = LE.gate(games(LE, 10, privileged=False), "T", "H", config=config, privileged=True)
    assert d.decision == LE.PRIVILEGED and LE.PRIVILEGED in LE.DECISIONS
    rep = LE.report(games(LE, 10), candidates=["T"], champion="H", config=config)
    assert rep["gates"]["T"]["decision"] == "privileged" and rep["best"] is None
    assert rep["privileged"] == ["T"] and "PRIVILEGED" in LE.format_report(rep)
    rep = LE.report(games(LE, 10, privileged=False), candidates=["T"], champion="H",
                    config=config, privileged=["T"])
    assert rep["best"] is None and rep["gates"]["T"]["privileged"]
    rep = LE.report(games(LE, 10, privileged=False), candidates=["T"], champion="H", config=config)
    assert rep["best"] == "T" and "privileged" not in rep
    # the game records carry it: read back from a log, the Games say so
    info = [{"name": "T", "search": {"preset": "s_true", "privileged": True}}, {"name": "H"}]
    rec = {"kind": LE.KIND_GAME, "key": "k", "teams": ["T", "H"], "winner": 0, "protocol": "pd",
           "split": "val", "result": {"info": info}}
    assert LE.Game.from_record(rec).privileged == (True, False)
    assert LE.Game.from_record({**rec, "result": {"info": [None, None]}}).privileged == ()
    assert LE.Game.from_record({k: v for k, v in rec.items() if k != "result"}).privileged == ()
    LE.append_records(tmp_path / "log.jsonl", [rec])
    assert LE.read_log(tmp_path / "log.jsonl").games[0].privileged_names() == {"T"}


def test_merge_search_info_sums_games(env):
    SR = env.search
    a = {"preset": "s_idle", "privileged": False, "decisions": 10, "searches": 4, "overrides": 1,
         "vetoes": 3, "outcomes": {"searched": 4, "not_triggered": 6},
         "override_by": {"2>wait": 1}, "ms": {"total": 50.0, "trials": 30.0},
         "decision_ms": {"n": 10, "sum": 50.0, "p50": 4.0, "p95": 9.0, "max": 12.0}}
    b = {"preset": "s_idle", "privileged": False, "decisions": 6, "searches": 2, "overrides": 0,
         "vetoes": 2, "outcomes": {"searched": 2, "low_fidelity": 4}, "fidelity_skips": 4,
         "ms": {"total": 30.0}, "decision_ms": {"n": 6, "sum": 30.0, "p95": 11.0, "max": 11.5}}
    m = SR.merge_search_info([a, None, b, {}])
    assert (m["games"], m["decisions"], m["searches"], m["overrides"], m["vetoes"]) == (
        2, 16, 6, 1, 5)
    assert m["outcomes"] == {"low_fidelity": 4, "not_triggered": 6, "searched": 6}
    assert m["override_by"] == {"2>wait": 1} and m["fidelity_skips"] == 4
    assert m["ms"] == {"total": 80.0, "trials": 30.0}
    assert m["decision_ms"] == {"n": 16, "mean": 5.0, "p95_max": 11.0, "max": 12.0}
    assert m["presets"] == ["s_idle"] and m["privileged"] is False
    assert SR.merge_search_info([None, {}]) is None


def test_importing_rs_search_imports_no_rs_match():
    code = ("import sys\n"
            "import tools.rs_search as SR\n"
            "heavy = ('rs_match', 'rs_rows', 'rs_eval', 'torch', 'royalesim')\n"
            "print(','.join(m for m in heavy if m in sys.modules))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""
