"""L02: tools/rs_match.py, the general match runner.

What is checked (the plan's test list first): results are deterministic by seed (stub seats
seeded from the spec); a seat swap mirrors the assignment (decks, forms, levels, per-seat delays,
elixir and agents change sides, and the same agent wins from either seat); the deck-pair splits
are disjoint on synthetic tags (dev = val, final = test, final refused without allow_final).
Golden: the legacy screen preset is rs_eval.play's reactive screen, call for call, under delay
0 / 21, mem-lag legacy / 1 / 21, with the Hog side at low and at high y. Then the memory of each
seat with perception noise (a missed, a late play, a clock offset), the elixir leak worked by
hand, pair skips, pair sets, the summary, the schema check, the sampling preset, the command
checks and the command line. Everything runs on conftest's fake royalesim with stub seats (no
engine, no corpus, no torch) except the one engine-marked smoke test.

Synthetic tags: of "synthetic-0000" .. "synthetic-0199", rs_rows.split_of (sha1 of the tag,
first 8 hex digits mod 100) puts 178 in train, the 12 in VAL in val and the 10 in TEST in test.
"""
from __future__ import annotations

import copy
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    HOG26_KEYS,
    OPPONENT_KEYS,
    il_payload,
    il_replay_row,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
SUB = 18
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VAL = (4, 16, 40, 56, 74, 91, 114, 132, 140, 164, 169, 193)
TEST = (3, 24, 27, 53, 84, 107, 118, 151, 190, 196)

#: IL_Replay key -> RoyaleSim name (data/cards/replay_cards.json's entries for these keys).
NAMES = {"cannon": "Cannon", "fireball": "Fireball", "hog-rider": "HogRider",
         "ice-golem": "IceGolemite", "ice-spirit": "IceSpirits", "musketeer": "Musketeer",
         "skeletons": "Skeletons", "the-log": "Log", "knight": "Knight", "archers": "Archer",
         "giant": "Giant", "arrows": "Arrows", "valkyrie": "Valkyrie", "mini-pekka": "MiniPekka",
         "baby-dragon": "BabyDragon", "zap": "Zap"}
#: Hog 2.6 and the opponent deck as names in sorted-key order (deck_stats.deck_of).
HOG = ("Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons", "Log")
OPP = ("Archer", "Arrows", "BabyDragon", "Giant", "Knight", "MiniPekka", "Valkyrie", "Zap")
HOG_KEYS, OPP_KEYS = tuple(sorted(HOG26_KEYS)), tuple(sorted(OPPONENT_KEYS))
#: il_payload's default forms in that order: evolved Skeletons, hero Musketeer; evolved Knight.
HOG_FORMS, OPP_FORMS = (0, 0, 0, 0, 0, 2, 1, 0), (0, 0, 0, 0, 1, 0, 0, 0)
IDS = {row[0]: i for i, row in enumerate(DEFAULT_CATALOGUE)}


@pytest.fixture
def tools(monkeypatch):
    """rs_rows, rs_eval and rs_match as the tools import them, the module settings at their
    defaults (restored after the test) and no engine config from the PC."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    ev = importlib.import_module("rs_eval")
    rm = importlib.import_module("rs_match")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "OUT": rr.OUT, "CARD_FLY": {}, "CARD_BUILDING": {}}.items():
        monkeypatch.setattr(rr, name, value)
    return types.SimpleNamespace(rows=rr, eval=ev, match=rm, engine=eng)


def cfg(tools):
    return tools.engine.EngineConfig.defaults()


def V(last0=(), last1=()):
    """A memory view from its "last" lists (every play also counts in "seen")."""
    return {"seen": {0: {c for c, _ in last0}, 1: {c for c, _ in last1}},
            "last": {0: list(last0), 1: list(last1)}}


def synthetic_rows(n=200):
    """IL_Replay replays rows synthetic-0000 .. n-1: even tags record Hog 2.6 as the team
    (il_payload's default forms), odd tags record it as the opponent (hero Musketeer only)."""
    out = []
    for i in range(n):
        tag = f"synthetic-{i:04d}"
        if i % 2:
            p = il_payload(tag=tag, team_deck=OPPONENT_KEYS, opponent_deck=HOG26_KEYS,
                           team_forms={}, opponent_forms={"musketeer": 2}, plays=(), abilities=())
        else:
            p = il_payload(tag=tag, plays=(), abilities=())
        out.append(il_replay_row(p))
    return out


def play_slot0(team, sub):
    return (team, 0, 9000 * sub, (8000 if team == 0 else 24_000) * sub)


class Eager:
    """Plays hand slot 0 at every decision."""

    def act(self, battle, state, team, mem, tick, sub, royalesim):
        return play_slot0(team, sub)


class EveryOther:
    """Plays hand slot 0 at the decisions on multiples of 20 ticks."""

    def act(self, battle, state, team, mem, tick, sub, royalesim):
        return play_slot0(team, sub) if tick % 20 == 0 else None


class RandomStub:
    """Plays a random slot with probability 0.3, from a rng seeded by the runner's seed."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, battle, state, team, mem, tick, sub, royalesim):
        if self.rng.random() < 0.3:
            return (team, int(self.rng.integers(0, 4)), 9000 * sub,
                    (8000 if team == 0 else 24_000) * sub)
        return None


class Watcher:
    """Keeps (tick handed, state tick) of every act and the memory at WATCH, keyed by the true
    tick (the battle's); plays slot 0 at `play_at`."""

    WATCH = (200, 220, 230, 270, 280)

    def __init__(self, play_at=()):
        self.play_at, self.views, self.seen = tuple(play_at), {}, {}

    def act(self, battle, state, team, mem, tick, sub, royalesim):
        t = battle.tick
        self.seen[t] = (tick, state["tick"])
        if t in self.WATCH:
            self.views[t] = copy.deepcopy(mem)
        return play_slot0(team, sub) if t in self.play_at else None


class Fixed:
    def __init__(self, make):
        self.make = make

    def act(self, battle, state, team, mem, tick, sub, royalesim):
        return self.make(team)


def referee(over_at=300):
    """FakeBattle states: the game ends at `over_at`, won by the team that sent more commands."""
    def state(battle):
        n = [0, 0]
        for call in battle.calls:
            if call.name == "step":
                for c in call.args[0]:
                    n[c[0]] += 1
        over = battle.tick >= over_at
        win = (0 if n[0] > n[1] else 1 if n[1] > n[0] else -1) if over else -1
        players = [player_dict(t, crowns=int(over and win == t)) for t in (0, 1)]
        return state_dict(battle.tick, players=players, game_over=over, winner=win)
    return state


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------

def test_spec_defaults_normalise_and_round_trip(tools):
    M = tools.match
    spec = M.MatchSpec(decks=[list(HOG), list(OPP)], agents=[{"name": "A"}, M.AgentSpec(name="B")])
    assert spec.decks == (HOG, OPP)
    assert spec.agents == (M.AgentSpec(name="A"), M.AgentSpec(name="B"))
    assert spec.level_lists() == [[11] * 8, [11] * 8] and spec.tower_level_list() == [11, 11]
    assert spec.form_lists() == [[0] * 8, [0] * 8]
    assert (spec.delay, spec.mem_lag, spec.first, spec.seat, spec.seed) == (None, 1, 0, 0, 0)
    assert (spec.cadence, spec.first_decision, spec.max_ticks) == (10, 90, 6200)
    assert spec.elixir_milli == (6000, 6000) and spec.agents[0].decide == "live"
    assert spec.reset_kwargs(IDS) == {
        "seed": 0, "decks": [list(range(8)), [9, 11, 14, 10, 8, 13, 12, 15]],
        "shuffle": 0, "start_tick": 0, "elixir_milli": [6000, 6000], "tower_hp": None,
        "spawns": [], "forms": [[0] * 8, [0] * 8], "levels": [[11] * 8, [11] * 8],
        "tower_levels": [11, 11]}
    with pytest.raises(ValueError, match="Mortar"):
        spec.replace(decks=(HOG, OPP[:7] + ("Mortar",))).reset_kwargs(IDS)
    assert M.MatchSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    full = spec.replace(engine=cfg(tools), delay=(21, 24), levels=[[11] * 7 + [13], [12] * 8],
                        tower_levels=(11, 12), forms=[[0] * 7 + [1], [2] + [0] * 7],
                        mem_lag="legacy", seed=5)
    assert full.engine_config() == cfg(tools).replace(delay=(21, 24))
    assert full.engine_config().delay == (21, 24)
    assert full.replace(delay=21).engine_config().delay == (21, 21)
    assert spec.replace(engine=cfg(tools)).engine_config() == cfg(tools)
    assert full.level_lists() == [[11] * 7 + [13], [12] * 8] and full.tower_level_list() == [11, 12]
    assert M.MatchSpec.from_dict(json.loads(json.dumps(full.to_dict()))) == full
    with pytest.raises(ValueError):
        M.MatchSpec.from_dict({**spec.to_dict(), "colour": "red"})
    bad = [dict(forms=[[0] * 7, [0] * 8]), dict(forms=[[3] * 8, [0] * 8]), dict(levels=17),
           dict(levels=[[11] * 8]), dict(tower_levels=(11, 0)), dict(first=2), dict(delay=-1),
           dict(seed=True), dict(decks=[list(HOG)]), dict(mem_lag="soon"), dict(cadence=0),
           dict(elixir_milli=(6000,))]
    for kw in bad:
        with pytest.raises(ValueError):
            spec.replace(**kw)
    for kw in (dict(decide="greedy"), dict(visibility="x"), dict(lead=True)):
        with pytest.raises(ValueError):
            M.AgentSpec("a.pt", **kw)


def test_swapped_mirrors_every_per_team_setting(tools):
    M = tools.match
    A, B = M.AgentSpec(name="A"), M.AgentSpec(name="B", seed_offset=7)
    s = M.MatchSpec(decks=(HOG, OPP), agents=(A, B), forms=([0] * 7 + [1], [2] + [0] * 7),
                    levels=([11] * 8, [12] * 8), tower_levels=(11, 12), delay=(21, 24),
                    elixir_milli=(6000, 7000), tag="t", pairing="p", seed=9)
    w = s.swapped()
    assert (w.decks, w.agents) == ((OPP, HOG), (B, A))
    assert w.forms == (tuple([2] + [0] * 7), tuple([0] * 7 + [1]))
    assert (w.levels, w.tower_levels) == (((12,) * 8, (11,) * 8), (12, 11))
    assert (w.delay, w.elixir_milli, w.first, w.seat) == ((24, 21), (7000, 6000), 1, 1)
    assert (w.tag, w.pairing, w.seed, w.mem_lag, w.max_ticks) == ("t", "p", 9, 1, 6200)
    assert w.swapped() == s and w != s
    plain = M.MatchSpec(decks=(HOG, OPP), delay=21)
    assert (plain.swapped().levels, plain.swapped().delay, plain.swapped().forms) == (11, 21, None)


# ---------------------------------------------------------------------------
# the plan's tests: determinism, seat swap, split disjointness
# ---------------------------------------------------------------------------

def test_results_are_deterministic_by_seed(fake_royalesim, tools):
    M = tools.match
    spec = M.MatchSpec(decks=(HOG, OPP), seed=11, engine=cfg(tools), max_ticks=400, delay=21,
                       agents=(M.AgentSpec(name="A"), M.AgentSpec(name="B", seed_offset=5)))
    seeds = []

    def factory(agent, team, spec_, cfg_):
        seeds.append(spec_.seed + agent.seed_offset)
        return RandomStub(spec_.seed + agent.seed_offset)

    first = M.run_match(spec, seat_factory=factory)
    again = M.run_match(spec, seat_factory=factory)
    assert first.to_dict(timings=False) == again.to_dict(timings=False)
    assert seeds == [11, 16, 11, 16]
    assert first.decisions == (31, 31) and first.end_tick == 400 and not first.game_over
    assert len(first.log) == sum(first.plays) and first.refused == (0, 0)
    assert all(e[0] % 10 == 0 and 90 <= e[0] < 400 for e in first.log)
    assert set(first.timings) == {"total_s", "setup_s", "engine_s", "decide_s"}
    assert len(first.timings["decide_s"]) == 2
    assert M.MatchResult.from_dict(json.loads(json.dumps(first.to_dict()))) == first
    M.run_match(spec.replace(seed=12), seat_factory=factory)
    assert seeds[-2:] == [12, 17]
    assert fake_royalesim.battles[-1].reset_kwargs["seed"] == 12


def test_seat_swap_mirrors_the_assignment(tools, monkeypatch):
    M = tools.match
    mod = make_fake_royalesim(states=referee(300))
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    pairs = M.read_pairs(synthetic_rows(16), NAMES, "dev", require=M.HOG26)
    assert pairs.tags == ["synthetic-0004"]
    A, B = M.AgentSpec(name="A"), M.AgentSpec(name="B", seed_offset=7)
    s0, s1 = M.pairing_specs(pairs.pairs, (A, B), seed=0, delay=(21, 24), engine=cfg(tools),
                             max_ticks=400)
    assert s1 == s0.swapped() and s0.seed == 1514628266
    made = []

    def factory(agent, team, spec, cfg_):
        made.append((agent.name, team, spec.seed + agent.seed_offset))
        return Eager() if agent.name == "A" else EveryOther()

    r0 = M.run_match(s0, seat_factory=factory)
    b0 = mod.battles[-1]
    r1 = M.run_match(s1, seat_factory=factory)
    b1 = mod.battles[-1]
    seed = s0.seed
    assert made == [("A", 0, seed), ("B", 1, seed + 7), ("B", 0, seed + 7), ("A", 1, seed)]
    k0, k1 = b0.reset_kwargs, b1.reset_kwargs
    for key in ("decks", "forms", "levels", "tower_levels", "elixir_milli"):
        assert k1[key] == k0[key][::-1], key
    assert k0["seed"] == k1["seed"] == seed
    assert sorted(k0["decks"][0]) == list(range(8))                      # A holds Hog 2.6
    assert b0.calls_named("set_command_delay_ticks")[0].args == (21, 24)
    assert b1.calls_named("set_command_delay_ticks")[0].args == (24, 21)
    # A plays at 90..290 (21), B at 100..280 every 20 ticks (10): A wins from either seat.
    assert (r0.winner, r1.winner) == (0, 1)
    assert r0.score_of("A") == r1.score_of("A") == 1.0
    assert r0.score_of("B") == r1.score_of("B") == 0.0
    assert (r0.plays, r1.plays) == ((21, 10), (10, 21))
    assert (r0.crowns, r1.crowns) == ((1, 0), (0, 1))
    assert r0.decisions == r1.decisions == (21, 21)
    assert (r0.end_tick, r1.end_tick, r0.game_over, r1.game_over) == (300, 300, True, True)
    assert (r0.delay, r1.delay) == ((21, 24), (24, 21))
    assert (r0.agents, r1.agents) == (("A", "B"), ("B", "A"))
    assert (r0.seat, r1.seat, r0.pairing, r1.pairing) == (0, 1, "val:synthetic-0004",
                                                          "val:synthetic-0004")


def test_pair_splits_are_disjoint_and_final_is_held_out(tools):
    M = tools.match
    rows = synthetic_rows()
    dev = M.read_pairs(rows, NAMES, "dev", require=M.HOG26)
    train = M.read_pairs(rows, NAMES, "train", require=M.HOG26)
    with pytest.raises(ValueError, match="held out"):
        M.read_pairs(rows, NAMES, "final")
    final = M.read_pairs(rows, NAMES, "final", require=M.HOG26, allow_final=True)
    assert (dev.split, train.split, final.split) == ("val", "train", "test")
    assert dev.tags == [f"synthetic-{i:04d}" for i in VAL]
    assert final.tags == [f"synthetic-{i:04d}" for i in TEST]
    assert len(train) == 178
    sets = [set(x.tags) for x in (dev, train, final)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sets[0] | sets[1] | sets[2] == {f"synthetic-{i:04d}" for i in range(200)}
    for ps in (dev, train, final):
        assert all(p.split == ps.split == M.deck_stats.split_of(p.tag) for p in ps.pairs)
        assert all(frozenset(p.decks[0]) == M.HOG26 for p in ps.pairs)
    assert dev.meta["other_splits"] == 188 and dev.meta["skipped"] == {}
    by = {p.tag: p for p in dev.pairs}
    assert by["synthetic-0004"] == M.DeckPair("synthetic-0004", "val", (HOG, OPP),
                                              (HOG_FORMS, OPP_FORMS), (HOG_KEYS, OPP_KEYS), 1.0)
    # recorded as the opponent: put on side a, its result turned round
    assert by["synthetic-0091"] == M.DeckPair("synthetic-0091", "val", (HOG, OPP),
                                              ((0, 0, 0, 0, 0, 2, 0, 0), (0,) * 8),
                                              (HOG_KEYS, OPP_KEYS), 0.0)
    plain = M.read_pairs(rows, NAMES, "dev")
    assert plain.tags == dev.tags and plain.pairs[5].decks == (OPP, HOG)
    assert M.read_pairs(rows, NAMES, "dev", limit=3).tags == dev.tags[:3]
    assert (M.corpus_split("dev"), M.corpus_split("val"), M.corpus_split("train")) == \
        ("val", "val", "train")
    assert M.corpus_split("final", allow_final=True) == "test"
    for bad in ("test", "final", "everything"):
        with pytest.raises(ValueError):
            M.corpus_split(bad)


# ---------------------------------------------------------------------------
# golden: the legacy reactive screen is rs_eval.play
# ---------------------------------------------------------------------------

def stub_class(play_at=(), watch=(140, 200, 210, 221, 230, 250, 260, 280, 300, 310, 330)):
    """An rs_eval agent class: keeps (a copy of) the memory each act() is handed at `watch`
    in `records` and plays hand slot 0 at the ticks in `play_at`."""
    class Stub:
        made: list = []

        def __init__(self, ckpt, seed=0, **_):
            self.ckpt, self.seed, self.records = ckpt, seed, {}
            Stub.made.append(self)

        def act(self, battle, state, team, mem, tick, sub, royalesim):
            if tick in watch:
                self.records[tick] = copy.deepcopy(mem)
            if tick in play_at:
                return (team, 0, 9000 * sub, (10_000 if team == 0 else 22_000) * sub)
            return None
    return Stub


@pytest.mark.parametrize("team_low", [True, False])
@pytest.mark.parametrize("lag", ["legacy", 1, 21])
@pytest.mark.parametrize("delay", [0, 21])
def test_legacy_screen_preset_is_rs_eval_play(fake_royalesim, tools, monkeypatch, delay, lag,
                                              team_low):
    payload = json.dumps(il_payload(team_low=team_low))
    args = (payload, "", "agent", 3, "opp.pt", delay, 0, False, False, False,
            cfg(tools).to_json(), lag)
    runs = {}
    for name, play in (("eval", tools.eval.play), ("match", tools.match.play_legacy_screen)):
        agent, opponent = stub_class(play_at=(200, 300)), stub_class(play_at=(200, 250))
        monkeypatch.setattr(tools.eval, "AGENT_CLASS", agent)
        monkeypatch.setattr(tools.eval, "RSAgent", opponent)
        before = len(fake_royalesim.battles)
        out = play(args)
        assert len(fake_royalesim.battles) == before + 2          # _setup's and the match's
        runs[name] = (out, agent.made, opponent.made, fake_royalesim.battles[-1])
    (ev, ev_a, ev_o, ev_b), (ma, ma_a, ma_o, ma_b) = runs["eval"], runs["match"]
    match = ma.pop("match")
    assert ma == ev
    assert ev["hog_team"] == (0 if team_low else 1) and ev["agent_plays"] == 2
    assert [s.seed for s in ma_a + ma_o] == [s.seed for s in ev_a + ev_o] == [3, 10]
    assert ma_o[0].records == ev_o[0].records and ma_a[0].records == ev_a[0].records
    assert ma_b.init_kwargs == ev_b.init_kwargs
    assert ma_b.calls == ev_b.calls                     # delay call, reset kwargs, every step
    assert ma_b.state_reads == ev_b.state_reads
    hog = ev["hog_team"]
    assert match["plays"][hog] == 2 and match["plays"][1 - hog] == 2
    assert match["agents"][hog] == "agent" and match["agents"][1 - hog] == "opponent"
    assert match["delay"] == [delay, delay] and match["end_tick"] == 6200


def test_legacy_screen_skips_as_rs_eval_does(fake_royalesim, tools, monkeypatch):
    garbage = il_payload(tag="garbage-1", plays=(
        (100, "team", "musketeer", 3500, 7500), (160, "team", "hog-rider", 3500, 24_000),
        (180, "team", "ice-golem", 14_500, 24_000), (200, "opponent", "knight", 14_500, 24_500)))
    args = (json.dumps(garbage), "", "agent", 0, "opp.pt", 0, 0, False, False, False,
            cfg(tools).to_json())
    monkeypatch.setattr(tools.eval, "AGENT_CLASS", stub_class())
    monkeypatch.setattr(tools.eval, "RSAgent", stub_class())
    skip = tools.match.play_legacy_screen(args)
    assert skip == tools.eval.play(args)
    assert skip["skipped"] == "orient" and skip["mode"] == "agent"
    assert len(fake_royalesim.battles) == 2                         # only the two _setup's
    for bad in (args[:2] + ("pro",) + args[3:], args[:4] + ("",) + args[5:],
                args[:8] + (True,) + args[9:]):
        with pytest.raises(ValueError, match="reactive screen"):
            tools.match.play_legacy_screen(bad)


# ---------------------------------------------------------------------------
# memory per seat, perception noise
# ---------------------------------------------------------------------------

NOISE = {None: None, "miss": {"opponent_plays": {"miss_p": 1.0}},
         "late": {"opponent_plays": {"late_ticks": 4}}, "clock": {"clock": {"bias_ticks": 3}}}


@pytest.mark.parametrize("noise", [None, "miss", "late", "clock"])
def test_memory_per_seat_and_perception_noise(fake_royalesim, tools, noise):
    """Delay 21, mem-lag 1: A (blue) plays the Cannon (token 2) at 200, which runs at 221; B (red)
    the Knight (token 10) at 250, which runs at 271. B's screen may miss A's play, see it 4 ticks
    late, or read the clock 3 ticks ahead."""
    M = tools.match
    a, b = Watcher((200,)), Watcher((250,))
    spec = M.MatchSpec(decks=(HOG, OPP), engine=cfg(tools), delay=21, max_ticks=400,
                       agents=(M.AgentSpec(name="A"), M.AgentSpec(name="B", noise=NOISE[noise])))
    r = M.run_match(spec, seats=[a, b])
    cannon, knight = (2, 221), (10, 271)
    assert r.plays == (1, 1) and r.decisions == (31, 31)
    assert a.views[220] == V() and a.views[230] == V([cannon]) and a.views[270] == V([cannon])
    assert a.views[280] == V([cannon], [knight])
    assert a.seen[200] == (200, 200)
    assert b.seen[200] == ((203, 203) if noise == "clock" else (200, 200))
    assert r.noise[0] is None and (r.noise[1] is None) == (noise is None)
    if noise == "miss":
        assert b.views[230] == V() and b.views[280] == V((), [knight])
        assert r.noise[1]["stats"] == {"observations": 31, "plays": 1, "plays_missed": 1}
    else:
        seen = (2, 225) if noise == "late" else cannon
        assert b.views[220] == V() and b.views[230] == V([seen])
        assert b.views[280] == V([seen], [knight])
    if noise in ("late", "clock"):
        assert r.noise[1]["stats"] == {"observations": 31, "plays": 1}
        assert r.noise[1]["kinds"] == (["opponent_plays"] if noise == "late" else ["clock"])
    assert M.run_match(spec, seats=[Watcher((200,)), Watcher((250,))]).to_dict(timings=False) \
        == r.to_dict(timings=False)                                 # the noise is seeded


def test_legacy_memory_order_in_the_general_runner(fake_royalesim, tools):
    """mem-lag legacy: plays dated when sent; the seat deciding second (`first` = 0: B) sees
    A's play of the same tick; with first = 1 it does not, and A sees B's."""
    M = tools.match
    for first in (0, 1):
        a, b = Watcher((200,)), Watcher((200,))
        spec = M.MatchSpec(decks=(HOG, OPP), engine=cfg(tools), max_ticks=300,
                           mem_lag="legacy", first=first)
        M.run_match(spec, seats=[a, b])
        if first == 0:
            assert a.views[200] == V() and b.views[200] == V([(2, 200)])
        else:
            assert a.views[200] == V((), [(10, 200)]) and b.views[200] == V()
        assert a.views[220] == b.views[220] == V([(2, 200)], [(10, 200)])


# ---------------------------------------------------------------------------
# elixir leak
# ---------------------------------------------------------------------------

def leak_states(battle):
    """Blue sits at 10 elixir, red at 5; the game ends at 200, blue the winner."""
    over = battle.tick >= 200
    players = [player_dict(0, elixir_milli=10_000, crowns=int(over)),
               player_dict(1, elixir_milli=5_000)]
    return state_dict(battle.tick, players=players, game_over=over, winner=0 if over else -1)


@pytest.mark.parametrize("delay", [0, 21])
def test_elixir_leak_by_hand(tools, monkeypatch, delay):
    """Steps of 10 ticks from 0 to 190 (20 steps). Blue is full at each step start. Its play sent
    at 100 runs at 100 (delay 0): the step from 100 leaks nothing, 19 x 10 = 190 ticks; at 121
    (delay 21): the step from 120 leaks 1 tick, 191. At 1x one tick is 500 of 28000 fine units."""
    M = tools.match
    mod = make_fake_royalesim(states=leak_states)
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    spec = M.MatchSpec(decks=(HOG, OPP), engine=cfg(tools), delay=delay,
                       agents=(M.AgentSpec(name="A"), M.AgentSpec(name="idle")))
    law = M.rs_elixir.ElixirLaw.constants()
    r = M.run_match(spec, seats=[Watcher((100,)), None], law=law)
    ticks = 190 if delay == 0 else 191
    assert r.full_ticks == (ticks, 0)
    assert r.leak[0] == pytest.approx(ticks * 500 / 28_000) and r.leak[1] == 0.0
    assert (r.winner, r.crowns, r.end_tick, r.game_over) == (0, (1, 0), 200, True)
    assert (r.plays, r.decisions, r.refused) == ((1, 0), (11, 0), (0, 0))
    assert r.log == ((100, 0, 0, 0, 9000 * SUB, 8000 * SUB, 0),)
    assert r.delay == (delay, delay) and r.score(0) == 1.0 and r.score(1) == 0.0
    assert r.towers == ((4824, 3052, 3052), (4824, 3052, 3052))


def test_refusals_are_counted(tools, monkeypatch):
    """A step result with a non-zero reason (here 4, NOT_ENOUGH_ELIXIR) is a refused play."""
    M = tools.match
    mod = make_fake_royalesim(step_results=lambda battle, cmds, n: [[c[0], 4] for c in cmds])
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    spec = M.MatchSpec(decks=(HOG, OPP), engine=cfg(tools), max_ticks=120)
    r = M.run_match(spec, seats=[Eager(), None])
    assert r.plays == (3, 0) and r.refused == (3, 0)                 # decisions at 90, 100, 110
    assert [e[6] for e in r.log] == [4, 4, 4]


# ---------------------------------------------------------------------------
# deck pairs: skips, sets, pairing specs
# ---------------------------------------------------------------------------

def test_pair_skips(tools):
    M = tools.match
    two_v_two = il_payload(tag="synthetic-0005", plays=(), abilities=())
    two_v_two["battle"]["team"]["players"].append(dict(two_v_two["battle"]["team"]["players"][0]))
    seven = il_payload(tag="synthetic-0007", plays=(), abilities=())
    seven["battle"]["team"]["players"][0]["deck"].pop()
    rows = [
        il_replay_row(il_payload(tag="synthetic-0000", opponent_deck=HOG26_KEYS, opponent_forms={},
                                 plays=(), abilities=())),                              # mirror
        il_replay_row(il_payload(tag="synthetic-0001", game_mode="Friendly", plays=(),
                                 abilities=())),                                        # mode
        il_replay_row(il_payload(tag="synthetic-0002", plays=(), abilities=())),         # kept
        il_replay_row(il_payload(tag="synthetic-0002", plays=(), abilities=())),         # again
        il_replay_row(two_v_two),                                                       # 2v2
        {"replay_tag": "synthetic-0006", "game_mode": "Ladder", "payload_json": "{not json"},
        il_replay_row(seven),                                                           # 7 cards
        il_replay_row(il_payload(tag="synthetic-0008", opponent_forms={}, plays=(), abilities=(),
                                 opponent_deck=OPPONENT_KEYS[:7] + ("goblin-barrel",))),  # card
        il_replay_row(il_payload(tag="synthetic-0009", team_deck=OPPONENT_KEYS, team_forms={},
                                 opponent_deck=OPPONENT_KEYS[:7] + ("fireball",),
                                 opponent_forms={}, plays=(), abilities=())),          # no Hog
        {"replay_tag": None, "game_mode": "Ladder", "payload_json": "{}"},
    ]
    ps = M.read_pairs(rows, NAMES, "train", require=M.HOG26)
    assert ps.tags == ["synthetic-0002"] and ps.meta["other_splits"] == 0
    assert ps.meta["skipped"] == {"card": 1, "deck": 1, "duplicate": 1, "json": 1, "mirror": 1,
                                  "mode": 1, "no tag": 1, "not 1v1": 1, "require": 1}
    assert M.pair_of(rows[1], NAMES) == (None, "mode")
    assert M.pair_of(rows[1], NAMES, modes=None)[1] == "ok"
    kept = M.read_pairs(rows, NAMES, "train", exclude_mirror=False)
    assert kept.tags == ["synthetic-0000", "synthetic-0002", "synthetic-0009"]


def test_pair_set_sample_save_and_load(tools, tmp_path):
    M = tools.match
    dev = M.read_pairs(synthetic_rows(), NAMES, "dev", require=M.HOG26)
    s1, s2 = dev.sample(5, seed=1), dev.sample(5, seed=1)
    assert s1 == s2 and len(s1) == 5 and s1.split == "val"
    where = [dev.tags.index(t) for t in s1.tags]
    assert where == sorted(where) and len(set(where)) == 5          # corpus order, no repeats
    assert (s1.meta["sampled"], s1.meta["of"], s1.meta["sample_seed"]) == (5, 12, 1)
    assert dev.sample(12) is dev and dev.sample(50, seed=3) is dev
    path = s1.save(tmp_path / "pairs.json")
    back = M.PairSet.load(path)
    assert (back.split, back.pairs, back.digest()) == (s1.split, s1.pairs, s1.digest())
    assert dev.digest() != s1.digest()
    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["pairs"][0]["tag"] = "synthetic-9999"
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        M.PairSet.load(path)


#: pairing_specs' engine seed: sha1("seed:tag") first 12 hex digits, mod 2 ** 31.
SEEDS = {5: (1663230741, 870327283, 610466190), 6: (1357937357, 1694216693, 1408252194)}


def test_pairing_specs_deal_swap_and_seed(tools):
    M = tools.match
    pairs = M.read_pairs(synthetic_rows(), NAMES, "dev", require=M.HOG26).pairs[:3]
    assert [p.tag for p in pairs] == ["synthetic-0004", "synthetic-0016", "synthetic-0040"]
    A, B = M.AgentSpec(name="A"), M.AgentSpec(name="B", seed_offset=7)
    specs = M.pairing_specs(pairs, (A, B), seed=5, delay=(21, 24), max_ticks=400)
    assert len(specs) == 6
    for k, pair in enumerate(pairs):
        s0, s1 = specs[2 * k], specs[2 * k + 1]
        assert s1 == s0.swapped() and s0 == s1.swapped()
        assert (s0.seat, s1.seat) == (0, 1)
        assert s0.pairing == s1.pairing == f"val:{pair.tag}" and s0.tag == pair.tag
        assert s0.seed == s1.seed == SEEDS[5][k]
        assert (s0.agents, s1.agents) == ((A, B), (B, A))
        assert (s0.delay, s1.delay, s0.max_ticks) == ((21, 24), (24, 21), 400)
        for side in (0, 1):
            assert sorted(s0.decks[side]) == sorted(pair.decks[side])
            assert dict(zip(s0.decks[side], s0.forms[side])) == \
                dict(zip(pair.decks[side], pair.forms[side]))
    assert M.pairing_specs(pairs, (A, B), seed=5, delay=(21, 24), max_ticks=400) == specs
    assert [s.seed for s in M.pairing_specs(pairs, (A, B), seed=6)[::2]] == list(SEEDS[6])
    base = M.pairing_specs(pairs, (A, B), seed=5, base_forms=(True, False))
    assert base[0].decks == specs[0].decks and base[0].forms[0] == (0,) * 8
    assert base[0].forms[1] == specs[0].forms[1] and base[1].forms[1] == (0,) * 8
    assert [s.seat for s in M.pairing_specs(pairs, (A, B), seats=1)] == [0, 0, 0]
    with pytest.raises(ValueError):
        M.pairing_specs(pairs, (A, B), seats=3)


# ---------------------------------------------------------------------------
# results, seats, the command line
# ---------------------------------------------------------------------------

def result(M, agents, winner, crowns, plays, leak, full):
    return M.MatchResult(id="m", tag="t", pairing="p", seat=0, seed=0, agents=agents,
                         winner=winner, crowns=crowns, game_over=True, end_tick=3600,
                         plays=plays, refused=(0, 0), decisions=(300, 300), full_ticks=full,
                         leak=leak, towers=((0, 0, 0), (0, 0, 0)), delay=(0, 0))


def test_summarize_and_scores(tools):
    M = tools.match
    from tools.league import stats
    rs = [result(M, ("A", "B"), 0, (2, 1), (30, 25), (1.5, 0.0), (40, 0)),
          result(M, ("B", "A"), 0, (1, 0), (20, 28), (0.0, 0.5), (0, 20)),
          result(M, ("A", "B"), -1, (0, 0), (26, 30), (1.0, 2.0), (20, 40))]
    s = M.summarize(rs, "A")
    assert (s["n"], s["wins"], s["draws"], s["losses"], s["score"]) == (3, 1, 1, 1, 0.5)
    assert (s["lo"], s["hi"]) == stats.wilson(1.5, 3)
    assert s["crown_diff"] == 0.0                                  # +1, -1, 0
    assert s["plays"] == 28.0 and s["leak"] == pytest.approx(1.0)  # (30+28+26)/3, (1.5+.5+1)/3
    assert s["full_s"] == pytest.approx(80 / 3 / 20)
    assert s["by_team"][0]["n"] == 2 and s["by_team"][0]["score"] == 0.75
    assert s["by_team"][1]["score"] == 0.0
    assert [r.score_of("B") for r in rs] == [0.0, 1.0, 0.5]
    with pytest.raises(ValueError):
        rs[0].team_of("C")
    with pytest.raises(ValueError):
        result(M, ("A", "A"), 0, (0, 0), (0, 0), (0, 0), (0, 0)).team_of("A")


def test_commands_are_checked_and_idle_seats_never_play(fake_royalesim, tools):
    M = tools.match
    spec = M.MatchSpec(decks=(HOG, OPP), engine=cfg(tools), max_ticks=200)
    for make, match in ((lambda team: (1 - team, 0, 0, 0), "own team"),
                        (lambda team: (team, 9, 0, 0), "hand slot"),
                        (lambda team: "play", r"not \(team")):
        with pytest.raises(ValueError, match=match):
            M.run_match(spec, seats=[Fixed(make), None])
    idle = M.run_match(spec)                    # AgentSpec(): build_seat gives None, no torch
    assert (idle.plays, idle.decisions, idle.winner, idle.score(0)) == ((0, 0), (0, 0), -1, 0.5)
    assert idle.agents == ("idle", "idle") and idle.end_tick == 200
    assert M.build_seat(M.AgentSpec("idle"), 0, spec, cfg(tools)) is None
    assert M.AgentSpec().label == "idle" and M.AgentSpec("runs/tokens/x/best.pt").label == "x"
    assert M.AgentSpec("best.pt").label == "best.pt"
    assert M.AgentSpec("a.pt", name="cand").label == "cand"


def test_configure_and_the_schema_check(tools):
    M = tools.match
    c = cfg(tools)
    seat = types.SimpleNamespace(meta={})
    M.configure(seat, M.AgentSpec("x.pt", lead=-21, decide="sample", visibility="public",
                                  spell_aim=True), c)
    assert (seat.lead, seat.lead_rebuild, seat.lead_extrapolate) == (21, True, False)
    assert (seat.decide_mode, seat.visibility, seat.spell_aim, seat.hold_pending) == \
        ("sample", "public", True, False)
    assert seat.engine is c
    M.configure(seat, M.AgentSpec("x.pt", lead=1026), c)
    assert (seat.lead, seat.lead_rebuild, seat.lead_extrapolate) == (26, False, True)
    assert M.seat_schema(seat) == "rs_rows/legacy" and M.seat_schema(object()) is None
    with pytest.raises(ValueError, match="rs_rows/v9"):
        M.configure(types.SimpleNamespace(meta={"schema": "rs_rows/v9"}), M.AgentSpec("x.pt"), c)
    with pytest.raises(ValueError, match="agent spec says"):
        M.configure(types.SimpleNamespace(meta={"schema": "rs_rows/legacy"}),
                    M.AgentSpec("x.pt", schema="rs_rows/v9"), c)
    stub = types.SimpleNamespace()
    M.configure(stub, M.AgentSpec("x.pt", schema="rs_rows/v9"), c)    # no meta: not checked
    assert stub.decide_mode == "live"


def test_match_agent_decide_presets(tools):
    M = tools.match
    agent = M.MatchAgent.__new__(M.MatchAgent)                      # no checkpoint, no torch
    agent.wait_bias, agent.decide_mode = 1.25, "legacy"
    assert agent.decide_config() == tools.eval.RSAgent.decide_config(agent)
    agent.decide_mode, agent.cadence = "sample", 20
    assert agent.decide_config() == M.rs_decide.DecideConfig.sampling(1.25, dt=20)
    assert agent.decide_config().sample and agent.decide_config().name == "sample"
    agent.decide_mode = "greedy"
    with pytest.raises(ValueError):
        agent.decide_config()


def test_command_line_helpers(tools):
    M = tools.match
    assert M._require("hog26") is M.HOG26 and M._require("none") is None
    assert M._require(",".join(OPP)) == frozenset(OPP)
    with pytest.raises(ValueError):
        M._require("Knight,Archer")
    assert (M._delay_arg("21"), M._delay_arg("21,24"), M._delay_arg(None),
            M._delay_arg("default")) == (21, (21, 24), None, None)


def test_command_line_on_a_pairs_file(fake_royalesim, tools, tmp_path, capsys):
    M = tools.match
    dev = M.read_pairs(synthetic_rows(), NAMES, "dev", require=M.HOG26)
    path = M.PairSet(dev.split, dev.pairs[:2], dev.meta).save(tmp_path / "pairs.json")
    out = tmp_path / "out.jsonl"
    argv = ["--a", "idle", "--b", "idle", "--name-a", "cand", "--split", "dev",
            "--pairs-file", str(path), "--max-ticks", "200", "--out", str(out), "--delay", "21"]
    assert M.main(argv) == 0
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["split"] == "val" and rec["pairs_digest"] == M.PairSet.load(path).digest()
    assert rec["delay"] == 21 and rec["seats"] == 2
    s = rec["summary"]["cand"]
    assert (s["n"], s["draws"], s["score"]) == (4, 4, 0.5)
    assert [r["seat"] for r in rec["results"]] == [0, 1, 0, 1]
    assert [r["agents"] for r in rec["results"]] == [["cand", "b"], ["b", "cand"]] * 2
    assert all(r["delay"] == [21, 21] and "log" not in r for r in rec["results"])
    assert "cand: score 50.0%" in capsys.readouterr().out
    assert argv[6:8] == ["--split", "dev"]
    with pytest.raises(SystemExit):
        M.main(argv[:7] + ["final"] + argv[8:])                     # final needs --final


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

@pytest.mark.engine
def test_engine_idle_match_smoke(tools):
    """Two idle seats on RoyaleSim: the match runs, nobody plays, both bars fill (at 1x the leak
    is 1/56 elixir a tick: 500 of 28000 fine units)."""
    M = tools.match
    r = M.run_match(M.MatchSpec(decks=(HOG, OPP), engine=cfg(tools), delay=21, max_ticks=400))
    assert (r.plays, r.winner, r.end_tick, r.game_over) == ((0, 0), -1, 400, False)
    assert r.towers[0] == r.towers[1] and r.towers[0][1] == r.towers[0][2] > 0
    assert r.full_ticks[0] == r.full_ticks[1] and 0 < r.full_ticks[0] < 400
    for t in (0, 1):
        assert r.leak[t] == pytest.approx(r.full_ticks[t] / 56)
