"""tools/rs_search_bench.py (S01, docs/LIVE_SEARCH_PLAN.md section 9).

No marker: the statistics, the cost model and the recommendation on canned numbers, the
`report` and `search` subcommands, the probe runner's isolation, and the probes' logic on
FakeBattles: P1 on a scripted state, P8 and P16 (the command watcher: is a delayed play charged
when it is sent or when it runs?) on a FakeBattle that queues plays as RoyaleSim's command delay
does.

engine (royalesim, on the PC): P1, P2, P8 and P9 as assertions, P16's answer, and a --quick run
of the whole `engine` subcommand with no probe erroring.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from tests.pipeline.conftest import (DEFAULT_CATALOGUE, DEPLOY_REASONS, Call, FakeBattle,
                                     entity_row, make_fake_royalesim, player_dict, state_dict)

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"

#: The default fake catalogue in royalesim's column layout (CATALOGUE_FIELDS: name, placement,
#: elixir, count, radius, flying, hitpoints, footprint_tiles, card_kind), so costs read right.
REAL_LAYOUT = [[row[0], 0, row[1], 1, 0, row[5], 0, 0, row[8]] for row in DEFAULT_CATALOGUE]
CARD_PENDING = DEPLOY_REASONS.index("CARD_PENDING")
NOT_ENOUGH_ELIXIR = DEPLOY_REASONS.index("NOT_ENOUGH_ELIXIR")
SKELETONS = [row[0] for row in DEFAULT_CATALOGUE].index("Skeletons")


@pytest.fixture
def sb(monkeypatch):
    """rs_search_bench as the tools import it, with no engine config from the PC."""
    monkeypatch.syspath_prepend(str(TOOLS))
    eng = importlib.import_module("rs_engine")
    for var in (eng.CONFIG_ENV, *eng.ENV.values()):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    return importlib.import_module("rs_search_bench")


# ---------------------------------------------------------------------------
# canned numbers
# ---------------------------------------------------------------------------

def _s(ms: float) -> dict:
    return {"n": 50, "mean_ms": ms, "p50_ms": ms, "p95_ms": ms, "p99_ms": ms, "min_ms": ms,
            "max_ms": ms}


def canned_engine() -> dict:
    """An engine report: reset 20 spawns 2 ms, a busy tick 0.05 ms, save 0.2, load 0.3,
    state_json + loads 1.0 ms; 16 CPUs, 11000 MB free."""
    return {
        "format": 1, "kind": "engine", "created": "2026-10-30T12:00:00",
        "machine": {"system": "Windows", "release": "11", "machine": "AMD64", "cpus": 16,
                    "python": "3.12.7", "available_mb": 11000},
        "engine": {"fingerprint": {"version": "0.1.42", "card_table": "cards-160402017-20261019.1",
                                   "id": "0123456789abcdef"}},
        "timings": {
            "construct": _s(20.0),
            "reset": {"0": _s(0.5), "10": _s(1.0), "20": _s(2.0), "40": _s(4.0)},
            "step": {"busy": {"1": _s(0.05), "10": _s(0.5), "60": _s(3.0)},
                     "empty": {"60": _s(0.6)}},
            "ticks_per_s": {"busy": 20000.0, "empty": 100000.0},
            "state_json": _s(0.4), "json_loads": _s(0.6), "state_bytes": 20000,
            "save": _s(0.2), "save_bytes": 50000, "load": _s(0.3), "state_hash": _s(0.01),
            "check_deploy": _s(0.002), "check_deploy_calls": 1080,
            "check_deploy_reasons": {"OK": 600, "OUT_OF_TERRITORY": 480},
            "building_placement": None,
            "composite": {"root": _s(4.5), "root_spawns": 20, "delay": 24,
                          "trial": {"40": _s(3.3), "60": _s(4.3), "80": _s(5.3)}},
        },
        "threads": [{"battles": 1, "ticks_per_s": 20000.0, "efficiency": 1.0, "errors": []},
                    {"battles": 2, "ticks_per_s": 38000.0, "efficiency": 0.95, "errors": []},
                    {"battles": 4, "ticks_per_s": 60000.0, "efficiency": 0.75, "errors": []},
                    {"battles": 8, "ticks_per_s": 80000.0, "efficiency": 0.5, "errors": []}],
        "memory": {"available_mb": 11000, "rss_mb_per_battle": 12.0},
        "probes": [{"id": "P1", "ok": True, "note": "one spec makes 1 Skeleton"},
                   {"id": "P2", "ok": False, "note": "spawns carry deploy timers"},
                   {"id": "P16", "ok": True, "note": "at RUN",
                    "answer": {"elixir": "run", "hand": "run", "delay": 24, "sent_tick": 600,
                               "ran_tick": 624}}],
    }


def canned_costs(sb):
    return sb.Costs.from_engine(canned_engine(), row_ms=1.5, forward_ms=5.0)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_percentile_and_summary(sb):
    xs = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert sb.percentile(xs, 50) == 3.0
    assert sb.percentile(xs, 95) == pytest.approx(4.8)
    assert (sb.percentile(xs, 0), sb.percentile(xs, 100)) == (1.0, 5.0)
    assert sb.percentile([], 50) is None
    assert sb.summarize(xs) == {"n": 5, "mean_ms": 3.0, "p50_ms": 3.0, "p95_ms": 4.8,
                                "p99_ms": 4.96, "min_ms": 1.0, "max_ms": 5.0}
    assert sb.summarize([]) == {"n": 0}
    half = sb.scaled(sb.summarize(xs), 0.5)
    assert (half["n"], half["p95_ms"], half["max_ms"]) == (5, 2.4, 2.5)


def test_timed_runs_before_untimed(sb):
    seen = []
    out = sb.timed(lambda: seen.append("fn"), 3, before=lambda: seen.append("before"))
    assert len(out) == 3 and all(ms >= 0 for ms in out)
    assert seen == ["before", "fn"] * 3


def test_latency_ticks_rounds_up(sb):
    assert sb.latency_ticks(100.0) == 2
    assert sb.latency_ticks(100.01) == 3
    assert sb.latency_ticks(49.9) == 1
    assert sb.latency_ticks(150.0) == 3
    assert sb.latency_ticks(0) == 0
    assert sb.latency_ticks(None) is None


# ---------------------------------------------------------------------------
# the cost model and the recommendation
# ---------------------------------------------------------------------------

def test_costs_from_an_engine_report(sb):
    c = canned_costs(sb)
    assert c.reset_ms == 2.0                     # the 20-spawn reset, a typical rebuild
    assert c.tick_ms == pytest.approx(0.05)      # step 60 / 60
    assert (c.save_ms, c.load_ms) == (0.2, 0.3)
    assert c.state_ms == pytest.approx(1.0)      # state_json + json.loads
    assert (c.row_ms, c.forward_ms) == (1.5, 5.0)
    plain = sb.Costs.from_engine(canned_engine())
    assert (plain.row_ms, plain.forward_ms) == (sb.ROW_MS_PRIOR, sb.FORWARD_MS_PRIOR)


def test_estimates_by_hand(sb):
    c = canned_costs(sb)
    # root = K (reset 2 + 24 x 0.05 + save 0.2) + state 1 + row 1.5 = 5.9 at K 1
    # s_root: 5.9 + 1 forward 5 = 10.9
    assert sb.estimate_ms(c, sb.PRESETS["s_root"]) == pytest.approx(10.9)
    # s_idle: trial = load 0.3 + 60 x 0.05 + 1 + 1.5 = 5.8; 5.9 + 8 x 5.8 + 2 x 5 = 62.3
    assert sb.estimate_ms(c, sb.PRESETS["s_idle"]) == pytest.approx(62.3)
    # s_reply: root 4 x 3.4 + 2.5 = 16.1; trial 0.3 + 4 + 2.5 = 6.8; 32 trials 217.6;
    # 32 reply rows 48; 4 forwards 20: 301.7
    assert sb.estimate_ms(c, sb.PRESETS["s_reply"]) == pytest.approx(301.7)
    assert sb.estimate_ms(c, sb.PRESETS["s_idle"], trials=2) == pytest.approx(5.9 + 11.6 + 10)
    assert sb.estimate_ms(c, sb.PRESETS["s_root"], delay=0) == pytest.approx(10.9 - 1.2)


def test_a_preset_that_fits_needs_no_cut(sb):
    c = canned_costs(sb)
    idle = sb.plan_preset("s_idle", c)
    assert idle["fits"] and idle["cuts"] == [] and idle["trials"] == 8
    assert idle["config"] == {"trials": 8, "horizon": 60, "worlds": 1}
    assert idle["ms"] == pytest.approx(62.3, abs=0.01) and idle["source"] == "estimate"
    assert idle["latency_ticks"] == 2 and idle["break_even_pp"] == pytest.approx(1.02)
    root = sb.plan_preset("s_root", c)
    assert root["fits"] and root["trials"] == 0 and root["latency_ticks"] == 1
    assert root["searchable"] and root["notes"] == []


def test_cuts_follow_the_plans_order(sb):
    """s_reply at 301.7 ms against 120: H 40 (237.7), then 5 trials (162.1), then K 2 (92.3)."""
    reply = sb.plan_preset("s_reply", canned_costs(sb))
    assert reply["cuts"] == ["H 40", "5 trials", "K 2"]
    assert reply["config"] == {"trials": 5, "horizon": 40, "worlds": 2}
    assert reply["fits"] and reply["trials"] == 5 and reply["most_trials"] == 5
    assert reply["ms"] == pytest.approx(92.3, abs=0.01)
    assert reply["latency_ticks"] == 2 and reply["searchable"]


def test_a_slow_engine_cannot_search(sb):
    slow = sb.Costs(reset_ms=10, tick_ms=1.0, save_ms=1, load_ms=1, state_ms=2, row_ms=1,
                    forward_ms=5)
    idle = sb.plan_preset("s_idle", slow)
    # root 38 + forwards 10; a trial at H 40 is 44 ms: no trial fits 80 ms; K 2 does not apply
    assert idle["cuts"] == ["H 40", "5 trials"]
    assert not idle["fits"] and idle["most_trials"] == 0 and idle["trials"] == 0
    assert not idle["searchable"]
    assert any("procs backend" in n for n in idle["notes"])
    assert any("would play d0" in n for n in idle["notes"])
    assert idle["ms"] == pytest.approx(48.0) and idle["latency_ticks"] == 1


def test_a_measured_time_replaces_the_estimate(sb):
    idle = sb.plan_preset("s_idle", canned_costs(sb), measured_p95=101.0)
    assert idle["source"] == "measured" and idle["ms"] == 101.0
    assert idle["latency_ticks"] == 3 and idle["est_ms"] == pytest.approx(62.3, abs=0.01)


def test_gate_workers_and_torch_threads(sb):
    g = sb.gate_workers(16, 11000)
    assert g["caps"] == {"memory": 4, "cpus": 15, "max": 6}      # (11000 - 3072) // 1800
    assert (g["workers"], g["torch_threads"], g["limited_by"]) == (4, 3, ["memory"])
    g = sb.gate_workers(4, None)
    assert g["available_assumed"] and g["available_mb"] == sb.ASSUMED_AVAILABLE_MB
    assert (g["workers"], g["torch_threads"]) == (2, 1)            # (8192 - 3072) // 1800
    g = sb.gate_workers(2, 64000)
    assert (g["workers"], g["torch_threads"], g["limited_by"]) == (1, 1, ["cpus"])
    assert sb.gate_workers(16, 64000, worker_mb=1200)["workers"] == 6


def test_scaling_limit(sb):
    assert sb.scaling_limit(canned_engine()["threads"]) == 4
    assert sb.scaling_limit([]) is None


def test_recommend_on_canned_numbers(sb):
    rec = sb.recommend(canned_engine(), row_ms=1.5, forward_ms=5.0)
    assert rec["latency_ticks"] == {"s_root": 1, "s_idle": 2, "s_reply": 2}
    assert rec["gate"]["workers"] == 4 and rec["gate"]["available_mb"] == 11000
    assert rec["thread_scaling_limit"] == 4
    assert rec["priors"] == {"row_ms": "given", "forward_ms": "given"}
    assert sb.recommend(canned_engine(), available=6000)["gate"]["workers"] == 1
    assert sb.recommend(canned_engine())["priors"]["row_ms"] == f"prior {sb.ROW_MS_PRIOR}"


def test_a_search_report_replaces_the_priors(sb):
    search = {"kind": "search", "row_ms": {"p95_ms": 0.5}, "forward_ms": {"p95_ms": 3.0},
              "presets": {"s_idle": {"p95_ms": 151.0}}}
    rec = sb.recommend(canned_engine(), search, row_ms=9.0)
    assert rec["costs"]["row_ms"] == 0.5 and rec["costs"]["forward_ms"] == 3.0
    assert rec["priors"] == {"row_ms": "measured", "forward_ms": "measured"}
    assert rec["presets"]["s_idle"]["source"] == "measured"
    assert rec["latency_ticks"]["s_idle"] == 4


# ---------------------------------------------------------------------------
# rendering, files and the command line
# ---------------------------------------------------------------------------

def test_render_engine_and_probe_counts(sb):
    rep = canned_engine()
    text = sb.render_engine(rep)
    assert "step 60, busy board" in text and "trial H 80" in text
    assert "P1   PASS" in text and "P2   FAIL" in text
    assert "3 probes: 2 pass, 1 fail, 0 not run" in text
    assert "pending charge (P16): elixir at run, hand at run" in text
    summary = sb.probe_summary(rep["probes"])
    assert summary["failed"] == ["P2"] and summary["not_run_ids"] == []
    assert sb.pending_charge(rep["probes"])["ran_tick"] == 624
    assert sb.pending_charge([]) is None


def test_write_report_never_overwrites(sb, tmp_path):
    a, ta = sb.write_report({"x": 1}, tmp_path, "engine", "text", when="20261030_120000")
    b, tb = sb.write_report({"x": 2}, tmp_path, "engine", None, when="20261030_120000")
    assert a.name == "engine_20261030_120000.json" and ta.read_text() == "text\n"
    assert b.name == "engine_20261030_120000_2.json" and tb is None
    assert json.loads(a.read_text()) == {"x": 1}
    assert sb.newest(tmp_path, "engine") == b
    assert sb.newest(tmp_path, "search") is None


def test_report_cli_reads_the_newest_engine_report(sb, tmp_path, capsys):
    old = canned_engine()
    old["timings"]["step"]["busy"]["60"] = _s(600.0)          # a slow old run, not read
    (tmp_path / "engine_20261029_100000.json").write_text(json.dumps(old))
    (tmp_path / "engine_20261030_120000.json").write_text(json.dumps(canned_engine()))
    assert sb.main(["report", "--dir", str(tmp_path), "--row-ms", "1.5",
                    "--forward-ms", "5"]) == 0
    out = capsys.readouterr().out
    assert "s_reply" in out and "H 40, 5 trials, K 2" in out
    assert "probes: 2 pass, 1 fail (P2: fix the rebuilder first" in out
    [written] = sorted(tmp_path.glob("report_*.json"))
    rec = json.loads(written.read_text())["recommendation"]
    assert rec["sources"] == {"engine": "engine_20261030_120000.json", "search": None}
    assert rec["latency_ticks"] == {"s_root": 1, "s_idle": 2, "s_reply": 2}


def test_report_cli_json_and_missing_inputs(sb, tmp_path, capsys):
    (tmp_path / "engine_20261030_120000.json").write_text(json.dumps(canned_engine()))
    (tmp_path / "search_20261031_090000.json").write_text(json.dumps(
        {"kind": "search", "presets": {"s_root": {"p95_ms": 30.0}}}))
    assert sb.main(["report", "--dir", str(tmp_path), "--json", "--no-write"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["sources"]["search"] == "search_20261031_090000.json"
    assert rec["presets"]["s_root"]["source"] == "measured" and rec["latency_ticks"]["s_root"] == 1
    assert not list(tmp_path.glob("report_*.json"))
    empty = tmp_path / "empty"
    empty.mkdir()
    assert sb.main(["report", "--dir", str(empty)]) == 1
    probes_only = {k: v for k, v in canned_engine().items() if k != "timings"}
    (empty / "engine_20261030_130000.json").write_text(json.dumps(probes_only))
    assert sb.main(["report", "--dir", str(empty)]) == 1
    assert "no timings" in capsys.readouterr().err


def test_search_says_it_needs_s08(sb, capsys):
    assert sb.main(["search", "--ckpt", "runs/tokens/live_big_hog/best.pt", "--opp-ckpt",
                    "runs/tokens/live_big/best.pt", "--games", "6", "--delay", "24",
                    "--noise", "default"]) == 3
    assert "S08" in capsys.readouterr().err


def test_engine_cli_refuses_unknown_probes(sb, capsys):
    assert sb.main(["engine", "--probes", "P1,P99", "--out", "unused"]) == 2
    assert "P99" in capsys.readouterr().err


def test_probe_ids_are_p1_to_p16(sb):
    assert sb.PROBE_IDS == tuple(f"P{i}" for i in range(1, 17))
    assert all(p.question.endswith("?") or "?" in p.question for p in sb.PROBES)


# ---------------------------------------------------------------------------
# the probe runner
# ---------------------------------------------------------------------------

def test_run_probes_isolates_skips_and_errors(sb):
    def good(lab, s):
        return True, {"x": 1}, "fine"

    def bad(lab, s):
        return False, {}, "contradicts"

    def skip(lab, s):
        raise sb.Skip("no Tesla")

    def boom(lab, s):
        raise RuntimeError("engine said no")

    probes = [sb.Probe("P1", "q?", "e", "p", good), sb.Probe("P2", "q?", "e", "p", bad),
              sb.Probe("P3", "q?", "e", "p", skip), sb.Probe("P4", "q?", "e", "p", boom)]
    lines = []
    recs = sb.run_probes(None, sb.QUICK, probes=probes, log=lines.append)
    assert [r["ok"] for r in recs] == [True, False, None, None]
    assert recs[0]["answer"] == {"x": 1} and recs[1]["note"] == "contradicts"
    assert recs[2]["note"] == "not run: no Tesla" and "error" not in recs[2]
    assert recs[3]["error"] == "RuntimeError: engine said no" and recs[3]["traceback"]
    assert sb.probe_summary(recs) == {"probes": 4, "pass": 1, "fail": 1, "not_run": 2,
                                      "failed": ["P2"], "not_run_ids": ["P3", "P4"]}
    assert len(lines) == 4 and "FAIL" in lines[1]
    only = sb.run_probes(None, sb.QUICK, only=["p2"], probes=probes)
    assert [r["id"] for r in only] == ["P2"]


# ---------------------------------------------------------------------------
# the probes' pure helpers
# ---------------------------------------------------------------------------

def test_first_change_and_moment(sb):
    series = [(601, 5017), (602, 5035), (603, 1053), (604, 1071)]
    assert sb.first_change(5000, series, lambda a, b: b < a - 500) == 602
    assert sb.first_change(5000, series[:2], lambda a, b: b < a - 500) is None
    assert sb.moment(602, 600, 602) == "run"
    assert sb.moment(600, 600, 602) == "send"
    assert sb.moment(None, 600, 602) == "never"
    assert sb.moment(600, 600, 600) == "send=run"
    assert sb.moment(601, 600, 602) == "tick 601"


def test_rate_label(sb):
    unit = 1000 * 500 / 28000                    # one 1x tick, in milli-elixir (17.857)
    assert [sb.rate_label(g, unit) for g in (17, 18, 35, 36, 53, 54)] == [1, 1, 2, 2, 3, 3]
    assert sb.rate_label(17, 0) is None


def test_match_formation(sb):
    preview = [("Skeleton", 1000, 2000, 1000), ("Skeleton", 1500, 2000, 1100),
               ("Skeleton", 1250, 2400, 1200)]
    units = [{"x": 1500, "y": 2000, "deploy_ticks": 21}, {"x": 1000, "y": 2000, "deploy_ticks": 19},
             {"x": 1250, "y": 2400, "deploy_ticks": 23}]
    got = sb.match_formation(preview, units)
    assert got["ok"] and got["max_dist"] == 0 and got["max_tick_gap"] == 0
    assert got["preview_ticks"] == [20, 22, 24] and got["members"] == got["units"] == 3
    assert not sb.match_formation(preview, units[:2])["ok"]                # a member missing
    late = [dict(u, deploy_ticks=0) for u in units]
    assert not sb.match_formation(preview, late)["ok"]                     # already deployed
    far = [dict(units[0], x=1500 + 9000)] + units[1:]
    assert not sb.match_formation(preview, far)["ok"]
    assert not sb.match_formation([], units)["ok"]


# ---------------------------------------------------------------------------
# the probes on FakeBattles
# ---------------------------------------------------------------------------

def _lab(sb, rs):
    return sb.Lab(sb.rs_engine.EngineConfig.legacy(), rs)


def test_lab_on_the_fake_engine(sb):
    lab = _lab(sb, make_fake_royalesim())
    names = [lab.name(c) for c in lab.deck]
    assert names == ["HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons",
                     "Fireball", "Log"]                 # rs_spawn_table.FILLER_NAMES
    assert lab.level == 11 and lab.sub == 18
    assert lab.xy(*sb.SPAWN_MT) == (9000 * 18, 10000 * 18)
    board = lab.board(40)
    blue = [s for s in board if s[0] == 0]
    red = [s for s in board if s[0] == 1]
    assert len(blue) == len(red) == 20
    assert len({(s[2], s[3]) for s in blue}) == 20                      # 20 distinct points
    assert all(s[3] <= 13500 * 18 for s in blue) and all(s[3] >= 18500 * 18 for s in red)
    assert {lab.name(s[1]) for s in blue} == {"Knight", "Musketeer", "Valkyrie", "MiniPekka",
                                              "Giant", "HogRider", "BabyDragon"}
    assert lab.board(0) == []
    assert lab.reason(CARD_PENDING) == "CARD_PENDING" and lab.reason(99) == "99"
    assert lab.bit("hidden") == 4 and lab.bit("evolved") == 8
    with pytest.raises(sb.Skip):
        lab.need("Tesla")


def _skeleton_states(n):
    def states(b):
        ents = [entity_row(uid=100 + i, team=0, card_id=SKELETONS, x=(9000 + 300 * i) * 18,
                           y=10000 * 18, hp=81, unit_type=7) for i in range(n)]
        return state_dict(tick=b.tick, entities=ents)
    return states


@pytest.mark.parametrize("n, ok", [(1, True), (3, False)])
def test_p1_counts_the_entities_of_one_spec(sb, n, ok):
    rs = make_fake_royalesim(states=_skeleton_states(n))
    [rec] = sb.run_probes(_lab(sb, rs), sb.QUICK, only=["P1"])
    assert rec["ok"] is ok
    assert rec["answer"]["Skeletons"] == {"entities_at_reset": n, "entities_after_1": n}
    assert ("per unit" if ok else "per formation") in rec["note"]
    spawned = rs.battles[-1].reset_kwargs["spawns"]
    assert spawned == [(0, SKELETONS, 9000 * 18, 10000 * 18, -1)]


class QueueBattle(FakeBattle):
    """A FakeBattle with RoyaleSim's command delay: set_command_delay_ticks takes effect at the
    next reset; a play accepted on tick T runs on T + k; the hand and the elixir change when it
    runs (at_send=True: when it is sent); step_commands_run reports it (report=False: the
    engine has no such accessor). A pending play's slot reads CARD_PENDING, another slot
    NOT_ENOUGH_ELIXIR. Elixir regenerates 18 milli a tick."""

    def __init__(self, *args, at_send=False, report=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.at_send = at_send
        if not report:
            self.step_commands_run = None         # getattr finds no accessor
        self.k, self.sent, self.ran, self.slot, self.rows = 0, None, None, None, []
        self.hand, self.next, self.queue, self.elixir = [], -1, [], 0

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.k = self.delay[0]
        deck = list(kwargs["decks"][0])
        self.hand, self.next, self.queue = deck[:4], deck[4], deck[5:]
        self.elixir = int(kwargs["elixir_milli"][0])
        self.sent = self.ran = self.slot = None
        self.rows = []

    def waiting(self) -> bool:                     # (FakeBattle.pending is a dict attribute)
        return self.sent is not None and self.ran is None

    def _charge(self):
        played = self.hand[self.slot]
        self.hand[self.slot] = self.next
        self.queue.append(played)
        self.next = self.queue.pop(0)
        self.elixir -= 4000

    def step(self, cmds=(), n=1):
        cmds = [tuple(c) for c in cmds]
        self.calls.append(Call("step", (cmds, n), {}))
        self.rows, out = [], []
        for c in cmds:
            if self.waiting() and c[1] == self.slot:
                out.append([c[0], CARD_PENDING])
                continue
            self.sent, self.ran, self.slot = self.tick, None, int(c[1])
            if self.at_send:
                self._charge()
            out.append([c[0], 0])
        for _ in range(int(n)):
            if self.waiting() and self.tick >= self.sent + self.k:
                self.ran = self.tick
                if not self.at_send:
                    self._charge()
                if self.k:
                    self.rows.append((self.tick, 0, "deploy", 2, 0))
            self.elixir = min(10000, self.elixir + 18)
            self.tick += 1
        self.steps += 1
        return out

    def step_commands_run(self):
        return list(self.rows)

    def pending_commands(self, team):
        if team != 0 or not self.waiting():
            return []
        return [["deploy", 2, 0, 0, self.sent + self.k - self.tick, 4]]

    def check_deploy(self, team, slot, x, y):
        self.deploy_checks.append((int(team), int(slot), int(x), int(y)))
        if team == 0 and self.waiting():
            return CARD_PENDING if slot == self.slot else NOT_ENOUGH_ELIXIR
        return 0

    def state(self):
        pend = self.pending_commands(0)
        players = [player_dict(0, elixir_milli=self.elixir, hand=list(self.hand),
                               next_card=self.next, pending=pend, pending_cost=4 if pend else 0),
                   player_dict(1)]
        return state_dict(tick=self.tick, players=players)


def _queue_lab(sb, **queue_kwargs):
    rs = make_fake_royalesim()

    def battle(*args, **kwargs):
        b = QueueBattle(*args, catalogue=REAL_LAYOUT, **queue_kwargs, **kwargs)
        rs.battles.append(b)
        return b

    rs.Battle = battle
    return _lab(sb, rs), rs


@pytest.mark.parametrize("report", [True, False])
def test_observe_command_finds_the_run_tick(sb, report):
    lab, _ = _queue_lab(sb, report=report)
    b = lab.battle(delay=5)
    lab.reset(b, elixir=(5000, 5000), read=False)
    obs = sb.observe_command(lab, b, 0, 0, sb.LANE_MT, max_wait=12, extra=2, other_slot=1,
                             trace=True)
    assert obs["accepted"] and obs["send_reason"] == "OK" and obs["card"] == "HogRider"
    assert obs["sent_tick"] == 600 and obs["ran_tick"] == 605 and obs["delay"] == 5
    assert obs["ran_reported"] is report
    assert [t["tick"] for t in obs["trace"]] == list(range(601, 609))   # until 2 ticks after
    m = sb.charge_moments(obs)
    assert (m["elixir"], m["hand"]) == ("run", "run")
    assert (m["elixir_tick"], m["hand_tick"]) == (605, 605)
    assert m["pending_after_send"] and m["pending_cost_after_send"] == 4
    assert m["check_slot_while_pending"] == "CARD_PENDING"
    assert m["check_other_while_pending"] == "NOT_ENOUGH_ELIXIR"


def test_observe_command_without_delay(sb):
    lab, _ = _queue_lab(sb)
    b = lab.battle()
    lab.reset(b, elixir=(5000, 5000), read=False)
    obs = sb.observe_command(lab, b, 0, 0, sb.LANE_MT, max_wait=5, trace=True)
    assert obs["ran_tick"] == 600 and obs["delay"] == 0 and not obs["ran_reported"]
    assert sb.charge_moments(obs)["elixir"] == "send=run"
    assert obs["waited"] == 1                      # it ran in the step that sent it


def test_p8_on_a_queueing_fake(sb):
    lab, rs = _queue_lab(sb)
    [rec] = sb.run_probes(lab, sb.BenchSettings(delay=6), only=["P8"])
    assert rec["ok"] is True, rec
    assert [r["delay"] for r in rec["answer"]["runs"]] == [6, 6]
    assert rec["answer"]["set_after_reset"] == {"same_battle_delay": 0, "next_reset_delay": 6}
    delayed = rs.battles[1]
    assert delayed.names()[:2] == ["set_command_delay_ticks", "reset"]   # before the first reset


@pytest.mark.parametrize("at_send, verdict", [(False, "run"), (True, "send")])
def test_p16_tells_send_from_run(sb, at_send, verdict):
    lab, _ = _queue_lab(sb, at_send=at_send)
    [rec] = sb.run_probes(lab, sb.BenchSettings(delay=24), only=["P16"])
    a = rec["answer"]
    assert rec.get("error") is None, rec
    assert (a["elixir"], a["hand"]) == (verdict, verdict)
    assert rec["ok"] is (verdict == "run")
    assert (a["sent_tick"], a["ran_tick"], a["delay"]) == (600, 624, 24)
    assert a["card"] == "HogRider" and a["cost"] == 4 and a["other_slot_card"] == "Musketeer"
    assert verdict.upper() in rec["note"]
    charge = sb.pending_charge([rec])
    assert (charge["elixir"], charge["hand"], charge["ok"]) == (verdict, verdict, verdict == "run")


# ---------------------------------------------------------------------------
# the real engine (marked `engine`; run on the PC)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine_lab():
    royalesim = pytest.importorskip("royalesim")
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    bench = importlib.import_module("rs_search_bench")
    return bench, bench.Lab(bench.rs_engine.active(), royalesim)


def _engine_probe(engine_lab, pid: str) -> dict:
    bench, lab = engine_lab
    [rec] = bench.run_probes(lab, bench.BenchSettings(), only=[pid])
    assert rec.get("error") is None, (f"{pid} errored: {rec.get('error')}\n"
                                      + "\n".join(rec.get("traceback") or []))
    if rec["ok"] is None:
        pytest.skip(f"{pid}: {rec['note']}")
    return rec


@pytest.mark.engine
@pytest.mark.parametrize("pid", ["P1", "P2", "P8", "P9"])
def test_engine_probe_holds(engine_lab, pid):
    """The answers plan section 4 assumes. A failure is a finding: fix the rebuilder (S04) to
    the engine's answer first (plan 7.2 M0), then this test's expectation."""
    rec = _engine_probe(engine_lab, pid)
    assert rec["ok"] is True, f"{pid} {rec['question']} -> {rec['note']}; {rec['answer']}"


@pytest.mark.engine
def test_engine_delayed_play_is_charged_when_it_runs(engine_lab):
    """P16: the Battle docstring says the hand and the elixir change only when a command runs."""
    rec = _engine_probe(engine_lab, "P16")
    a = rec["answer"]
    assert a["ran_tick"] - a["sent_tick"] == a["delay"], rec["note"]
    assert (a["elixir"], a["hand"]) == ("run", "run"), rec["note"]
    assert a["check_slot_while_pending"] == "CARD_PENDING", rec["note"]


@pytest.mark.engine
def test_engine_cli_quick_writes_a_report(tmp_path, capsys):
    pytest.importorskip("royalesim")
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    bench = importlib.import_module("rs_search_bench")
    assert bench.main(["engine", "--quick", "--out", str(tmp_path)]) == 0
    [path] = sorted(tmp_path.glob("engine_*.json"))
    rep = json.loads(path.read_text(encoding="utf-8"))
    assert [p["id"] for p in rep["probes"]] == list(bench.PROBE_IDS)
    errors = {p["id"]: p["error"] for p in rep["probes"] if p.get("error")}
    assert not errors, errors
    assert rep["timings"]["step"]["busy"]["60"]["n"] == bench.QUICK.reps
    assert rep["timings"]["save_bytes"] > 0 and rep["timings"]["state_bytes"] > 0
    assert [r["battles"] for r in rep["threads"]] == list(bench.QUICK.threads)
    assert all(r["finished"] == r["battles"] and not r["errors"] for r in rep["threads"])
    assert (tmp_path / path.name.replace(".json", ".txt")).is_file()
    assert bench.main(["report", "--dir", str(tmp_path), "--no-write"]) == 0
    assert "recommendation" in capsys.readouterr().out
