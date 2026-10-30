"""E11: decision-time rows, tools/rs_rows.py --obs-delay D --obs-delay-jitter J --opp-lag L
--wait-window W.

Pure tests, worked by hand: DecisionRule / decision_rule, tap_delays (seeded by the replay tag,
one rng stream per kind of action), exact_hand (tokens-10), pending_features, DecisionPlan's
taps, pending plays and wait rule, obs_row's two new columns and hand_exact.

rows_for, build and train_tokens on conftest's fake royalesim (no engine, no corpus). The
fixture match (tag decision-0001, the team at low y plays blue, Hog 2.6 against conftest's
opponent deck) has four plays, executing at

    105 red Knight (14500, 24500)    token 10, own tile (3, 24)
    290 red Arrows (3500, 8000)      token 13, own tile (14, 8)
    300 blue HogRider (3500, 14500)  token 4,  own tile (3, 17)
    320 blue Musketeer (3500, 7500)  token 7,  own tile (3, 24)

and the fake's blue hand is HogRider, Cannon, Skeletons, IceGolemite (next Musketeer) up to the
state at 300, then Musketeer in HogRider's slot (the hog runs in the step from 300). Red holds
Knight, Arrows, Archer, Giant throughout. With --obs-delay 21 the plays are tapped at 84, 269,
279 and 299 (the spec's example: plays at 300 and 320 give rows at 279 and 299). The row at
279 does not know the Arrows that run at 290; the row at 299 has the HogRider pending, landing
in 1 tick (1/40), and taps a Musketeer that is not in the engine's hand yet, so the gate target
is -1. Wait rows every 20 ticks from 20 to 520 (the last play + 200), except where the side
taps in [t, t + 10]: red at 80 (tap 84) and 260 (tap 269); 26 + 24 waits + 4 plays = 54 rows.
The rows at execution (D = 0, as before E11) have 52: the plays at 105, 290, 300, 320, and no
wait for red at 100 and 280 or for blue at 300 and 320.

Golden: D = 0, by default and given, gives the rows, info and engine calls of before (the
hand-made order, no new column, hand_exact True, obs_row called with the keywords of before),
and the engine gets the same commands at the same ticks under every delay.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    SUBTILE_PER_MILLITILE,
    il_payload,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
SUB = SUBTILE_PER_MILLITILE
TAG = "decision-0001"                  # split "train" (sha1 % 100 >= 10)
WAIT = (0, 0, 0)
PLAYS = ((105, "opponent", "knight", 14500, 24500), (290, "opponent", "arrows", 3500, 8000),
         (300, "team", "hog-rider", 3500, 14500), (320, "team", "musketeer", 3500, 7500))
#: Blue's hand (catalogue ids) and next card up to the state at 300, and after.
BLUE_EARLY, BLUE_LATE = ([2, 0, 6, 3], 5), ([5, 0, 6, 3], 1)
OPP_HAND, OPP_NEXT = [8, 11, 9, 10], 12
#: The four plays in queue order: (exec, side, action).
QUEUE = ((105, -1, (10, 3, 24)), (290, -1, (13, 14, 8)), (300, 1, (4, 3, 17)),
         (320, 1, (7, 3, 24)))
#: The commands the engine gets under every delay: (tick, commands) of each step with any.
SENT = [(105, [(1, 0, 14500 * SUB, 24500 * SUB)]),     # Knight, red slot 0
        (290, [(1, 1, 3500 * SUB, 8000 * SUB)]),       # Arrows, red slot 1
        (300, [(0, 0, 3500 * SUB, 14500 * SUB)]),      # HogRider, blue slot 0
        (320, [(0, 0, 3500 * SUB, 7500 * SUB)])]       # Musketeer, blue slot 0 (after 300)
ACT = ("act_card", "act_x", "act_y", "match", "split")
DSTAT_A = {"play_rows": 4, "press_rows": 0, "not_in_hand": 1, "clamped": 0, "pending_rows": 6,
           "exact_rows": 0}


def payload_a(**kwargs):
    kwargs.setdefault("tag", TAG)
    kwargs.setdefault("plays", PLAYS)
    kwargs.setdefault("abilities", ())
    return il_payload(**kwargs)


def states_a(battle):
    hand, nxt = BLUE_EARLY if battle.tick <= 300 else BLUE_LATE
    return state_dict(battle.tick, players=[player_dict(0, hand=hand, next_card=nxt),
                                            player_dict(1, hand=OPP_HAND, next_card=OPP_NEXT)])


def decision_order():
    """(tick, side, action) of the --obs-delay 21 rows of payload_a(), in row order."""
    taps = {84: (-1, QUEUE[0][2]), 269: (-1, QUEUE[1][2]), 279: (1, QUEUE[2][2]),
            299: (1, QUEUE[3][2])}
    out = []
    for t in sorted(set(range(20, 521, 20)) | set(taps)):
        if t in taps:
            side, act = taps[t]
            out.append((t, side, act))
            continue
        out.append((t, 1, WAIT))
        if t not in (80, 260):
            out.append((t, -1, WAIT))
    return out


def legacy_order():
    """(tick, side, action) of the rows at execution (D = 0), as before E11: a play row, then
    the other seat's wait on a sample tick; no wait within 10 ticks before a play."""
    plays = {e: (s, a) for e, s, a in QUEUE}
    out = []
    for t in sorted(set(range(20, 521, 20)) | set(plays)):
        if t in plays:
            side, act = plays[t]
            out.append((t, side, act))
            if t % 20 == 0:
                out.append((t, -side, WAIT))
            continue
        out.append((t, 1, WAIT))
        if t not in (100, 280):
            out.append((t, -1, WAIT))
    return out


def waits_for(taps):
    """(tick, side, WAIT) of the decision-time waits of a run to 520, in row order: every 20
    ticks from 20, blue then red, each side with no tap in [t, t + 10]; `taps` {side: ticks}."""
    return [(t, side, WAIT) for t in range(20, 521, 20) for side in (1, -1)
            if not any(t <= x <= t + 10 for x in taps.get(side, ()))]


def keys(rows):
    return [(r["tick"], r["side"], tuple(int(v) for v in r["action"])) for r in rows]


def f16(values):
    return np.asarray(values, dtype=np.float16).tolist()


def same(a, b, names):
    for k in names:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.dtype == y.dtype and np.array_equal(x, y), k


def sent(battle):
    """(tick, commands) of every step of a FakeBattle that carried commands."""
    out, tick = [], int((battle.reset_kwargs or {}).get("start_tick") or 0)
    for call in battle.calls:
        if call.name == "step":
            cmds, n = call.args
            if cmds:
                out.append((tick, list(cmds)))
            tick += int(n)
    return out


def shard(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture
def rr(monkeypatch):
    """tools/rs_rows.py as the tools import it, its module settings at the defaults (restored
    after the test, DECISION too) and no engine config from the PC."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "VISIBILITY": rr.VIS_LEGACY, "TRUST": rr.TrustRule(), "OUT": rr.OUT,
                        "CARD_FLY": {}, "CARD_BUILDING": {}, "PRESSES": "off",
                        "BUTTONS": 2, "HAND_SIZE": 4, "DECISION": rr.DecisionRule()}.items():
        monkeypatch.setattr(rr, name, value)
    return rr


@pytest.fixture
def engine(monkeypatch):
    """install(**battle_kwargs): a fake royalesim in sys.modules for this test."""
    def install(**kwargs):
        mod = make_fake_royalesim(**kwargs)
        monkeypatch.setitem(sys.modules, "royalesim", mod)
        return mod
    return install


def cfg(rr):
    return rr.rs_engine.EngineConfig.defaults()


def run(rr, payload=None, **kwargs):
    """rows_for on `payload` (payload_a()): (rows, info)."""
    info = {}
    rows, why = rr.rows_for(payload_a() if payload is None else payload,
                            rr._setup(cfg(rr)), cfg(rr), info=info, **kwargs)
    assert why == "ok"
    return rows, info


# ---------------------------------------------------------------------------
# settings, by hand
# ---------------------------------------------------------------------------

def test_decision_rule_values_and_errors(rr):
    D = rr.DecisionRule
    assert rr.DECISION == D() == D(0, 0, 10) and not D().on and D(1).on
    assert (rr.WAIT_WINDOW, rr.PENDING_SLOTS, rr.PENDING_SCALE) == (10, 4, 40)
    assert rr.DECISION_FIELDS == ("own_pending", "own_pending_land")
    assert rr.decision_rule() == D() and rr.decision_rule(None) == D()
    assert rr.decision_rule(21) == D(21, 0, 10) and rr.decision_rule("24") == D(24)
    got = rr.decision_rule(np.int64(3))
    assert got == D(3) and type(got.delay) is int
    rule = D(24, 3, 30)
    assert rr.decision_rule(rule) == rule
    assert rr.decision_rule(rule.to_json()) == rule
    assert rr.decision_rule({"delay": 21}) == D(21)
    js = rule.to_json()
    assert (js["delay"], js["jitter"], js["wait_window"]) == (24, 3, 30)
    assert (js["pending_slots"], js["pending_scale"], js["tap_floor"]) == (4, 40, 0)
    assert js["streams"] == {"plays": 0, "presses": 1}
    assert json.loads(json.dumps(js)) == js
    assert rr.decision_rule(D(199)) == D(199)                 # 199 < HORIZON
    for bad, what in ((D(3, 4), "jitter"), (D(-1), "obs-delay"), (D(150, 50), "outcome"),
                      (D(0, 0, 25), "wait-window"), (D(True), "obs-delay"),
                      (D(2.5), "obs-delay"), (D(21, 0, -1), "wait-window"), ("x", "obs-delay"),
                      ({"delay": 21, "lead": 3}, "unknown")):
        with pytest.raises(ValueError, match=what):
            rr.decision_rule(bad)
    assert rr.decision_vocab(0) == {} and rr.decision_vocab(None) == {}
    assert rr.decision_vocab(21) == {"decision": D(21)}
    assert rr.describe_decision(0).startswith("at execution")
    assert rr.describe_decision(D(24, 3)).startswith("decision time, 24 +- 3 ticks")
    assert rr.describe_decision(21).startswith("decision time, 21 ticks")


def test_tap_delays_seeded_by_tag_and_stream(rr):
    assert rr.tap_delays("any", 3, 21) == [21, 21, 21]
    assert rr.tap_delays("any", 0, rr.DecisionRule(24, 3)) == []
    assert rr.tap_delays("any", 2) == [0, 0]                  # DECISION: no delay
    seed = int(hashlib.sha1(TAG.encode()).hexdigest()[:16], 16)
    assert rr.tap_seed(TAG) == seed
    rule = rr.DecisionRule(24, 3)
    for stream in (0, 1):
        want = [24 + int(v) for v in
                np.random.default_rng([seed, stream]).integers(-3, 4, size=12)]
        got = rr.tap_delays(TAG, 12, rule, stream)
        assert got == want and all(type(d) is int and 21 <= d <= 27 for d in got)
        assert rr.tap_delays(TAG, 12, rule, stream) == got   # the same every time
    assert (rr.STREAM_PLAYS, rr.STREAM_PRESSES) == (0, 1)
    tags = [f"m{k}" for k in range(20)]
    assert len({tuple(rr.tap_delays(t, 4, rule)) for t in tags}) > 1


def test_exact_hand(rr):
    E = rr.exact_hand
    assert (rr.EXACT_PLAYS, rr.EXACT_MIRROR_WINDOW, rr.MIRROR) == (4, 8, "Mirror")
    assert not E([]) and not E(["A", "B", "C"])
    assert E(["A", "B", "C", "D"]) and E(("X", "A", "B", "C", "D"))
    assert not E(["A", "B", "C", "A"])                         # the last four not distinct
    assert not E(["Mirror", "A", "B", "C", "D"])
    assert not E(["A", "B", "C", "D", "Mirror"])
    assert E(["Mirror"] + list("ABCDEFGH"))                   # out of the last 8
    assert not E(["Mirror"] + list("ABCDEFG"))


def test_pending_features(rr):
    cards, land = rr.pending_features([(4, 1), (7, 21), (9, 50), (10, 0), (12, 3)])
    assert cards.dtype == np.int16 and land.dtype == np.float16
    assert cards.tolist() == [4, 7, 9, 10]                     # the first four, in order
    assert land.tolist() == f16([1 / 40, 21 / 40, 1.0, 0.0])   # 50 / 40 clipped at 1
    cards, land = rr.pending_features([])
    assert cards.tolist() == [0, 0, 0, 0] and land.tolist() == [0.0] * 4
    assert rr.pending_features([(4, -3)])[1].tolist()[0] == 0.0


def test_decision_plan_by_hand(rr):
    queue = [(105, 1, "Knight", 0, 0), (290, 1, "Arrows", 0, 0), (300, 0, "HogRider", 0, 0),
             (320, 0, "Musketeer", 0, 0)]
    plan = rr.DecisionPlan(queue, [10, 13, 4, 7], [], TAG, 21)
    assert plan.play_taps == [84, 269, 279, 299] and plan.play_clamped == [False] * 4
    assert plan.events == [(84, 0, 0), (269, 0, 1), (279, 0, 2), (299, 0, 3)]
    assert plan.tapped(1, 80, 90) and not plan.tapped(1, 60, 70) and plan.tapped(1, 260, 270)
    assert not plan.tapped(0, 280, 290) and plan.tapped(0, 279, 279) and plan.tapped(0, 0, 999)
    assert plan.pending(0, 279) == []                         # its own tap is no pending play
    assert plan.pending(0, 280) == [(4, 20)]
    assert plan.pending(0, 299) == [(4, 1)]                   # the spec's 1/40
    assert plan.pending(0, 300) == [(4, 0), (7, 20)]          # runs at 300: not on the board yet
    assert plan.pending(0, 320) == [(7, 0)] and plan.pending(0, 321) == []
    assert plan.pending(1, 100) == [(10, 5)] and plan.pending(1, 84) == []
    assert not any(plan.exact(t, k) for t in (0, 1) for k in range(0, 600, 7))
    # a press: its own rng stream, a tap of its side, never a pending play
    press = rr.rs_presses.Press(330, 0, (5,), ("musketeer-hero",), 9)
    with_press = rr.DecisionPlan(queue, [10, 13, 4, 7], [(330, 0, press)], TAG, 21)
    assert with_press.press_taps == [309] and with_press.events[-1] == (309, 1, 0)
    assert with_press.tapped(0, 300, 310) and with_press.pending(0, 310) == [(7, 10)]
    # floored at tick 0; same-tick taps are not each other's pending plays
    early = rr.DecisionPlan([(15, 0, "HogRider", 0, 0)], [4], [], TAG, 21)
    assert early.play_taps == [0] and early.play_clamped == [True]
    assert early.pending(0, 0) == [] and early.pending(0, 1) == [(4, 14)]
    twin = rr.DecisionPlan([(300, 0, "HogRider", 0, 0), (300, 0, "Musketeer", 0, 0)], [4, 7],
                           [], TAG, 21)
    assert twin.play_taps == [279, 279] and twin.pending(0, 279) == []
    assert twin.pending(0, 280) == [(4, 20), (7, 20)]
    # hand_exact: four plays run before the row's tick
    five = [(100 + 60 * k, 0, name, 0, 0) for k, name in
            enumerate(("Musketeer", "HogRider", "Skeletons", "Cannon", "IceGolemite"))]
    exact = rr.DecisionPlan(five, [7, 4, 8, 2, 5], [], TAG, 21)
    assert [exact.exact(0, t) for t in (79, 259, 280, 281, 319)] == [False, False, False,
                                                                     True, True]
    assert not exact.exact(1, 500)


def test_jittered_taps_keep_each_sides_order(rr, monkeypatch):
    """One player taps in order: under jitter a play's tap is never before its side's previous
    tap, so the plays a row sees pending land no later than its own (as the live runner's sent
    plays do). The draws are fixed here: HogRider 21, IceSpirits 27, red's Knight 27."""
    def fixed(delays):
        return lambda tag, n, rule=None, stream=0: (list(delays)[:n] if stream == 0
                                                     else [24] * n)
    queue = [(300, 0, "HogRider", 0, 0), (303, 0, "IceSpirits", 0, 0), (305, 1, "Knight", 0, 0)]
    rule = rr.DecisionRule(24, 3)
    monkeypatch.setattr(rr, "tap_delays", fixed([21, 27, 27]))
    plan = rr.DecisionPlan(queue, [4, 6, 10], [], TAG, rule)
    # raw taps 279, 276, 278: the IceSpirits' 276 would come before the HogRider's 279
    assert plan.play_taps == [279, 279, 278] and plan.play_clamped == [False] * 3
    assert plan.events == [(278, 0, 2), (279, 0, 0), (279, 0, 1)]
    assert plan.pending(0, 279) == [] and plan.pending(0, 276) == []
    assert plan.pending(0, 280) == [(4, 20), (6, 23)]
    assert plan.pending(1, 279) == [(10, 26)]                     # the other side keeps its draw
    # taps already in order are left alone
    monkeypatch.setattr(rr, "tap_delays", fixed([27, 21, 24]))
    plan = rr.DecisionPlan(queue, [4, 6, 10], [], TAG, rule)
    assert plan.play_taps == [273, 282, 281]
    assert plan.pending(0, 282) == [(4, 18)]


def test_obs_row_pending_columns_and_hand_exact(rr, engine):
    engine()
    rr._setup(cfg(rr))
    state = state_dict(100)
    view = rr.rs_memory.empty_view()
    old = rr.obs_row(state, 0, SUB, view, 100)
    assert "own_pending" not in old and old["hand_exact"] is True
    new = rr.obs_row(state, 0, SUB, view, 100, pending=[(4, 1), (7, 20)], hand_exact=False)
    assert list(new) == list(old) + ["own_pending", "own_pending_land"]
    same(old, new, [k for k in old if k != "hand_exact"])
    assert new["hand_exact"] is False
    assert new["own_pending"].tolist() == [4, 7, 0, 0]
    assert new["own_pending_land"].tolist() == f16([1 / 40, 0.5, 0, 0])
    empty = rr.obs_row(state, 1, SUB, view, 100, pending=[], hand_exact=True)
    assert empty["hand_exact"] is True and not empty["own_pending"].any()
    assert rr.obs_row(state, 0, SUB, view, 100, hand_exact=np.bool_(False))["hand_exact"] is False


# ---------------------------------------------------------------------------
# rows_for: the spec's example and the rows at execution
# ---------------------------------------------------------------------------

def test_rows_at_execution_are_the_rows_before(rr, engine, monkeypatch):
    """Golden: D = 0 (default, 0, DecisionRule(), the module setting) gives the hand-made rows
    of before, the same info and engine calls, and calls obs_row with the keywords of before."""
    mod = engine(states=states_a)
    runs = []
    for kwargs in ({}, {"decision": 0}, {"decision": rr.DecisionRule()},
                   {"decision": {"delay": 0}}):
        rows, info = run(rr, **kwargs)
        runs.append((rows, info, mod.battles[-1]))
    base, base_info, base_battle = runs[0]
    assert keys(base) == legacy_order() and len(base) == 52
    assert base_info == {"trust": {"cut": None, "why": "none"}}
    assert sent(base_battle) == SENT
    for rows, info, battle in runs[1:]:
        assert info == base_info and battle.calls == base_battle.calls
        assert len(rows) == len(base)
        for a, b in zip(rows, base):
            assert list(a) == list(b)
            same(a, b, a)
    for r in base:
        assert "own_pending" not in r and r["hand_exact"] is True and r["trusted"]
    by = {(r["tick"], r["side"], int(r["action"][0])): r for r in base}
    hog = by[(300, 1, 4)]                         # memory at BOARD_LAG: both red plays ran
    assert hog["opp_last"].tolist() == [10, 13, 0, 0]
    assert hog["opp_last_age"].tolist() == f16([195 / 400, 10 / 400, 1, 1])
    assert hog["hand"].tolist() == [4, 2, 8, 5]
    musk = by[(320, 1, 7)]
    assert musk["hand"].tolist() == [7, 2, 8, 5]
    assert musk["opp_last_age"].tolist() == f16([215 / 400, 30 / 400, 1, 1])
    assert by[(105, -1, 10)]["opp_last"].tolist() == [0, 0, 0, 0]
    # obs_row gets exactly the keywords of before on this path
    seen = []
    real = rr.obs_row

    def spy(*args, **kwargs):
        seen.append(sorted(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(rr, "obs_row", spy)
    rows, _ = run(rr, decision=0)
    assert len(seen) == len(rows) == 52 and all(kw == [] for kw in seen)
    seen.clear()
    rows, _ = run(rr, decision=21)
    assert len(seen) == len(rows) == 54
    assert all(kw == ["hand_exact", "pending"] for kw in seen)


def test_spec_example_rows_at_279_and_299(rr, engine):
    mod = engine(states=states_a)
    rows, info = run(rr, decision=21)
    assert keys(rows) == decision_order() and len(rows) == 54
    assert info == {"trust": {"cut": None, "why": "none"}, "decision": DSTAT_A}
    assert sent(mod.battles[-1]) == SENT                     # the plays still run when recorded
    by = {(r["tick"], r["side"], int(r["action"][0])): r for r in rows}
    hog, musk = by[(279, 1, 4)], by[(299, 1, 7)]
    # memory at 279: the Knight (105) but not the Arrows that run at 290
    assert hog["opp_seen"].tolist() == [10] + [0] * 7
    assert hog["opp_last"].tolist() == [10, 0, 0, 0]
    assert hog["opp_last_age"].tolist() == f16([174 / 400, 1, 1, 1])
    assert hog["own_pending"].tolist() == [0, 0, 0, 0] and not hog["own_pending_land"].any()
    assert hog["hand"].tolist() == [4, 2, 8, 5] and hog["next"] == 7
    # at 299: both red plays, and the HogRider pending, landing at 300
    assert musk["opp_seen"].tolist() == [10, 13] + [0] * 6
    assert musk["opp_last"].tolist() == [10, 13, 0, 0]
    assert musk["opp_last_age"].tolist() == f16([194 / 400, 9 / 400, 1, 1])
    assert musk["own_pending"].tolist() == [4, 0, 0, 0]
    assert musk["own_pending_land"].tolist() == f16([1 / 40, 0, 0, 0])
    assert 7 not in musk["hand"].tolist()                    # the gate target is -1
    # the other rows with a pending play; every other row has none
    want = {(280, 1, 0): ([4, 0, 0, 0], [20 / 40, 0, 0, 0]),
            (299, 1, 7): ([4, 0, 0, 0], [1 / 40, 0, 0, 0]),
            (300, 1, 0): ([4, 7, 0, 0], [0, 20 / 40, 0, 0]),
            (320, 1, 0): ([7, 0, 0, 0], [0, 0, 0, 0]),
            (100, -1, 0): ([10, 0, 0, 0], [5 / 40, 0, 0, 0]),
            (280, -1, 0): ([13, 0, 0, 0], [10 / 40, 0, 0, 0])}
    for key, r in by.items():
        cards, land = want.get(key, ([0, 0, 0, 0], [0, 0, 0, 0]))
        assert r["own_pending"].tolist() == cards, key
        assert r["own_pending_land"].tolist() == f16(land), key
        assert r["own_pending"].dtype == np.int16 and r["own_pending_land"].dtype == np.float16
        assert r["hand_exact"] is False and r["trusted"] and r["outcome"] == 0.0
    assert by[(280, 1, 0)]["opp_last_age"].tolist() == f16([175 / 400, 1, 1, 1])
    assert by[(280, -1, 0)]["opp_last"].tolist() == [0, 0, 0, 0]
    assert by[(320, 1, 0)]["hand"].tolist() == [7, 2, 8, 5]
    assert by[(84, -1, 10)]["opp_last"].tolist() == [0, 0, 0, 0]
    # no wait row within 10 ticks before a tap: red's at 80 and 260 are gone, at 280 it is back
    assert (80, -1, 0) not in by and (260, -1, 0) not in by and (280, -1, 0) in by
    # a wait row both runs take (same tick, same state, same memory view) differs from the row
    # at execution only in hand_exact and the two new columns
    legacy, _ = run(rr)
    old = {(r["tick"], r["side"], int(r["action"][0])): r for r in legacy}
    shared = [k for k in by if k[2] == 0 and k in old]
    # 50 decision waits less blue's at 300 and 320 and red's at 100 and 280 (no wait at
    # execution there)
    assert len(shared) == 46
    for k in shared:
        new, cols = by[k], list(old[k])
        assert cols[12] == "action"      # obs_row's 12 columns, then the labels
        assert list(new) == cols[:12] + list(rr.DECISION_FIELDS) + cols[12:]
        same(new, old[k], [c for c in cols if c != "hand_exact"])
        assert old[k]["hand_exact"] is True and new["hand_exact"] is False


def test_opp_lag_and_the_module_setting(rr, engine, monkeypatch):
    engine(states=states_a)
    late, _ = run(rr, decision=21, mem_lag=21)
    by = dict(zip(keys(late), late))
    assert by[(299, 1, (7, 3, 24))]["opp_last"].tolist() == [10, 0, 0, 0]     # 290 + 21 > 299
    assert by[(320, 1, WAIT)]["opp_last"].tolist() == [10, 13, 0, 0]         # 290 + 21 <= 320
    one, _ = run(rr, decision=21, mem_lag=1)
    for a, b in zip(one, run(rr, decision=21, mem_lag=0)[0]):              # lag 0 reads as 1
        same(a, b, a)
    with pytest.raises(ValueError, match="mem-lag legacy"):
        run(rr, decision=21, mem_lag="legacy")
    monkeypatch.setattr(rr, "DECISION", rr.DecisionRule(21))
    rows, info = run(rr)
    assert keys(rows) == decision_order() and info["decision"] == DSTAT_A
    rows, info = run(rr, decision=0)
    assert keys(rows) == legacy_order() and "decision" not in info
    with pytest.raises(ValueError, match="jitter"):
        run(rr, decision=rr.DecisionRule(2, 3))


def test_jitter_moves_each_tap_within_its_window(rr, engine):
    mod = engine(states=states_a)
    rule = rr.DecisionRule(24, 3)
    rows, info = run(rr, decision=rule)
    delays = rr.tap_delays(TAG, 4, rule, rr.STREAM_PLAYS)
    taps = [e - d for (e, _, _), d in zip(QUEUE, delays)]
    plays = [(t, s, a) for t, s, a in keys(rows) if a != WAIT]
    assert plays == sorted((t, s, a) for t, (_, s, a) in zip(taps, QUEUE))
    assert all(e - 27 <= t <= e - 21 for t, (e, _, _) in zip(taps, QUEUE))
    # the waits follow the taps (the Knight's, in [78, 84], decides red's wait at 80)
    waits = waits_for({-1: taps[:2], 1: taps[2:]})
    assert waits_for({-1: [84, 269], 1: [279, 299]}) == [k for k in decision_order()
                                                          if k[2] == WAIT]
    assert [k for k in keys(rows) if k[2] == WAIT] == waits
    assert len(rows) == 4 + len(waits) and sent(mod.battles[-1]) == SENT
    musk = next(r for r in rows if int(r["action"][0]) == 7)
    assert musk["own_pending"].tolist() == [4, 0, 0, 0]
    assert musk["own_pending_land"].tolist() == f16([(300 - taps[3]) / 40, 0, 0, 0])
    again, _ = run(rr, decision=rule)
    assert keys(again) == keys(rows)
    other = run(rr, payload_a(tag="decision-0002"), decision=rule)[0]
    other_taps = [e - d for (e, _, _), d in
                  zip(QUEUE, rr.tap_delays("decision-0002", 4, rule))]
    assert sorted(t for t, _, a in keys(other) if a != WAIT) == sorted(other_taps)


def test_taps_floor_at_zero(rr, engine):
    plays = ((15, "team", "hog-rider", 3500, 14500), (200, "opponent", "knight", 14500, 24500))
    mod = engine()
    rows, info = run(rr, payload_a(tag="decision-clamp", plays=plays), decision=21)
    assert keys(rows)[0] == (0, 1, (4, 3, 17))
    # pending: red's waits at 180 and 200 see the Knight tapped at 179; blue's HogRider is
    # pending only in (0, 15], where no row is taken
    assert info["decision"] == {"play_rows": 2, "press_rows": 0, "not_in_hand": 0,
                                "clamped": 1, "pending_rows": 2, "exact_rows": 0}
    by = dict(zip(keys(rows), rows))
    assert by[(180, -1, WAIT)]["own_pending_land"].tolist() == f16([20 / 40, 0, 0, 0])
    assert by[(200, -1, WAIT)]["own_pending"].tolist() == [10, 0, 0, 0]
    assert sent(mod.battles[-1]) == [(15, [(0, 2, 3500 * SUB, 14500 * SUB)]),
                                     (200, [(1, 0, 14500 * SUB, 24500 * SUB)])]
    assert (179, -1, (10, 3, 24)) in keys(rows)


def test_hand_exact_is_honest(rr, engine):
    plays = ((100, "team", "musketeer", 3500, 7500), (160, "team", "hog-rider", 3500, 14500),
             (220, "team", "skeletons", 9000, 5000), (280, "team", "cannon", 9000, 9000),
             (340, "team", "ice-golem", 3500, 14500))
    engine()
    rows, info = run(rr, payload_a(tag="decision-exact", plays=plays), decision=21)
    blue = {r["tick"]: r for r in rows if r["side"] == 1}
    assert [blue[t]["hand_exact"] for t in (79, 139, 199, 259, 319)] == [False] * 4 + [True]
    assert [int(blue[t]["action"][0]) for t in (79, 139, 199, 259, 319)] == [7, 4, 8, 2, 5]
    assert blue[280]["hand_exact"] is False and blue[300]["hand_exact"] is True
    assert all(r["hand_exact"] is (r["tick"] > 280) for r in blue.values())
    assert not any(r["hand_exact"] for r in rows if r["side"] == -1)
    assert info["decision"]["exact_rows"] == 14 == sum(r["hand_exact"] for r in rows)
    legacy, _ = run(rr, payload_a(tag="decision-exact", plays=plays))
    assert all(r["hand_exact"] is True for r in legacy)


def test_presses_at_their_tap(rr, engine):
    mod = engine(states=states_a)
    payload = payload_a(abilities=((330, "team", "musketeer"),))
    rows, info = run(rr, payload, decision=21, presses="on")
    want = []
    for k in decision_order():
        if k == (300, 1, WAIT):                   # the press tapped at 309 is within 10 ticks
            continue
        want.append(k)
        if k == (300, -1, WAIT):
            want.append((309, 1, WAIT))
    assert keys(rows) == want and len(rows) == 54
    assert [r["press"] for r in rows] == [1 if k == (309, 1, WAIT) else 0 for k in want]
    press = rows[want.index((309, 1, WAIT))]
    assert press["own_pending"].tolist() == [7, 0, 0, 0]
    assert press["own_pending_land"].tolist() == f16([11 / 40, 0, 0, 0])
    assert press["ability_card"].tolist() == [7, 0]
    assert info["presses"] == {"events": 1, "sent": 1, "ok": 1, "refused": {}, "skipped": {},
                               "order_mismatch": 0}
    assert info["decision"] == dict(DSTAT_A, press_rows=1)
    assert sent(mod.battles[-1]) == SENT + [(330, [(0, 4, 0, 0)])]   # the press runs at 330
    # with jitter, the presses' own stream leaves the plays' taps where they were
    rule = rr.DecisionRule(24, 3)
    on, _ = run(rr, payload, decision=rule, presses="on")
    off, _ = run(rr, payload, decision=rule)
    assert [k for k in keys(on) if k[2] != WAIT] == [k for k in keys(off) if k[2] != WAIT]
    tap = 330 - rr.tap_delays(TAG, 1, rule, rr.STREAM_PRESSES)[0]
    assert [r["tick"] for r in on if r["press"]] == [tap]


# ---------------------------------------------------------------------------
# build, stats, manifest, vocab.json, main
# ---------------------------------------------------------------------------

def test_build_writes_the_decision_columns_and_counts(rr, engine, tmp_path):
    engine(states=states_a)
    text = json.dumps(payload_a())
    eng = cfg(rr).to_json()
    batch, n, skipped, st = rr.build(1, [text], False, str(tmp_path), eng, decision=21,
                                     stats=True)
    assert (batch, n, skipped) == (1, 54, {}) and rr.DECISION == rr.DecisionRule(21)
    new = shard(tmp_path / "shard-0001.npz")
    assert sorted(new) == sorted(rr.FIELDS + rr.DECISION_FIELDS + ACT)
    assert new["own_pending"].dtype == np.int16 and new["own_pending"].shape == (54, 4)
    assert new["own_pending_land"].dtype == np.float16 and new["own_pending_land"].shape == (54, 4)
    assert new["hand_exact"].dtype == np.bool_ and not new["hand_exact"].any()
    order = decision_order()
    assert new["tick"].tolist() == [t for t, _, _ in order]
    assert new["act_card"].tolist() == [a[0] for _, _, a in order]
    assert set(new["split"].tolist()) == {"train"} and set(new["match"].tolist()) == {TAG}
    assert st["decision"] == DSTAT_A
    assert (st["rows"], st["play_rows"], st["trusted"], st["hog_rows"]) == (54, 4, 54, 28)
    assert rr.build(2, [text], False, str(tmp_path), eng)[1] == 54    # None: keep the setting
    # back to D = 0: the shard of before
    back = rr.build(3, [text], False, str(tmp_path), eng, decision=0, stats=True)
    assert back[:3] == (3, 52, {}) and "decision" not in back[3]
    old = shard(tmp_path / "shard-0003.npz")
    assert sorted(old) == sorted(rr.FIELDS + ACT)
    rows, _ = run(rr)
    for k in rr.FIELDS:
        assert np.array_equal(old[k], np.stack([np.asarray(r[k]) for r in rows])), k
    with pytest.raises(ValueError, match="mem-lag legacy"):
        rr.build(4, [text], False, str(tmp_path), eng, decision=21, mem_lag="legacy")
    with pytest.raises(ValueError, match="jitter"):
        rr.build(5, [text], False, str(tmp_path), eng, decision=rr.DecisionRule(2, 5))
    assert not (tmp_path / "shard-0004.npz").exists() and not (tmp_path / "shard-0005.npz").exists()
    assert rr.DECISION == rr.DecisionRule(0)


def test_trust_stats_and_manifest_sum_decision_counts(rr):
    rows = [{"tick": 20, "action": (0, 0, 0), "trusted": True, "hog": True},
            {"tick": 21, "action": (7, 1, 1), "trusted": True, "hog": True}]
    plain = [{"trust": {"cut": None, "why": "none"}}]
    assert "decision" not in rr.trust_stats(rows, plain)
    infos = [dict(plain[0], decision=DSTAT_A),
             dict(plain[0], decision=dict(DSTAT_A, not_in_hand=2, clamped=1))]
    st = rr.trust_stats(rows, infos)
    assert st["decision"] == {"play_rows": 8, "press_rows": 0, "not_in_hand": 3, "clamped": 1,
                              "pending_rows": 12, "exact_rows": 0}
    totals = rr.merge_stats(rr.merge_stats({}, st), st)
    man = rr.manifest(totals, [], {})
    assert man["decision"]["play_rows"] == 16 and list(man)[-2:] == ["decision", "shards"]
    assert rr.describe_manifest(man) == (
        "trusted 4 of 4 rows (100.0%; Hog 2.6 100.0%); decision time: 6 of 16 play rows tap a "
        "card not yet in hand, 2 taps floored at tick 0")
    legacy = rr.manifest(rr.trust_stats(rows, plain), [], {})
    assert "decision" not in legacy
    assert rr.describe_manifest(legacy) == "trusted 2 of 2 rows (100.0%; Hog 2.6 100.0%)"


def test_vocab_records_the_decision_settings_only_when_on(rr):
    cat = [list(r) for r in DEFAULT_CATALOGUE]
    units, eng = list(DEFAULT_UNIT_TYPES), cfg(rr)
    fp = {"version": "0.0.0+fake", "card_table": None, "id": "f00d"}
    base = rr.vocab_record(cat, units, eng, fp, {}, False)
    for same_vocab in (rr.vocab_record(cat, units, eng, fp, {}, False, decision=0),
                       rr.vocab_record(cat, units, eng, fp, {}, False,
                                       decision=rr.DecisionRule()),
                       rr.vocab_record(cat, units, eng, fp, {}, False, **rr.decision_vocab(0))):
        assert json.dumps(same_vocab) == json.dumps(base)       # the vocab.json of before
    rule = rr.DecisionRule(24, 3)
    on = rr.vocab_record(cat, units, eng, fp, {}, False, **rr.memory_vocab(1),
                         **rr.decision_vocab(rule))
    assert list(on) == list(base) + ["opp_memory", "decision_rows", "extra_columns"]
    assert on["obs_delay"] == {"ticks": 24, "jitter": 3, "opp_lag": 1}
    assert on["decision_rows"] == rule.to_json() and on["schema"] == "rs_rows/legacy"
    assert rr.decision_rule(on["decision_rows"]) == rule
    assert on["extra_columns"] == ["own_pending", "own_pending_land"]
    assert json.loads(json.dumps(on)) == on
    both = rr.vocab_record(cat, units, eng, fp, {}, False, trust="v2", presses="on",
                           decision=21, obs_delay=21, opp_lag=4)
    assert both["extra_columns"] == ["trusted_v2", "act_press", "ability_card", "ability_state",
                                     "own_pending", "own_pending_land"]
    assert both["obs_delay"] == {"ticks": 21, "jitter": 0, "opp_lag": 4}
    with pytest.raises(ValueError, match="obs_delay 21"):
        rr.vocab_record(cat, units, eng, fp, {}, False, decision=24, obs_delay=21)
    with pytest.raises(ValueError, match="jitter"):
        rr.vocab_record(cat, units, eng, fp, {}, False, decision=rr.DecisionRule(1, 2))
    # the old keywords alone still write what they did (tests/pipeline/test_vocab_guard.py)
    later = rr.vocab_record(cat, units, eng, fp, {}, False, obs_delay=21, obs_delay_jitter=2)
    assert later["obs_delay"]["ticks"] == 21 and "decision_rows" not in later


def _main(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def test_main_wires_the_decision_settings(rr):
    text = (TOOLS / "rs_rows.py").read_text(encoding="utf-8")
    main = text[text.index("def main()"):]
    for needle in ('"--obs-delay"', '"--obs-delay-jitter"', '"--wait-window"',
                   'add_mem_lag_arg(ap, aliases=("--opp-lag",))', "DECISION = decision",
                   "decision=DECISION.to_json()", "**decision_vocab(DECISION)",
                   "describe_decision(DECISION)", "args.mem_lag == MEM_LEGACY"):
        assert needle in main, needle
    defaults = {}
    for n in ast.walk(_main(TOOLS / "rs_rows.py")):
        if (isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
                and n.args and getattr(n.args[0], "value", None) in ("--obs-delay",
                                                                      "--obs-delay-jitter")):
            defaults[n.args[0].value] = next(k.value.value for k in n.keywords
                                             if k.arg == "default")
    assert defaults == {"--obs-delay": 0, "--obs-delay-jitter": 0}
    ap = argparse.ArgumentParser()
    rr.add_mem_lag_arg(ap, aliases=("--opp-lag",))
    assert ap.parse_args([]).mem_lag == 1
    assert ap.parse_args(["--opp-lag", "21"]).mem_lag == 21
    assert ap.parse_args(["--mem-lag", "legacy"]).mem_lag == "legacy"
    plain = argparse.ArgumentParser()
    rr.add_mem_lag_arg(plain)                     # tools/rs_eval.py's: no alias
    with pytest.raises(SystemExit):
        plain.parse_args(["--opp-lag", "3"])


# ---------------------------------------------------------------------------
# sim/train_tokens.py trains the legacy model on decision-time rows
# ---------------------------------------------------------------------------

def test_the_legacy_model_trains_on_decision_rows(rr, engine, tmp_path):
    torch = pytest.importorskip("torch")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    tt = importlib.import_module("sim.train_tokens")
    mt = importlib.import_module("sim.model_tokens")
    engine(states=states_a)
    assert rr.split_of(TAG) == "train"
    rr.build(1, [json.dumps(payload_a())], False, str(tmp_path), cfg(rr).to_json(), decision=21)
    data = tt.load("train", 1.0, True, False, 0, tmp_path)
    assert list(data) == list(tt.COLS) and len(data["act_card"]) == 54   # extra columns ignored
    torch.manual_seed(0)
    model = mt.TokenPolicy(len(DEFAULT_CATALOGUE) + 2, len(DEFAULT_UNIT_TYPES) + 2, d=16,
                           layers=1, heads=2)
    b = next(tt.batches(data, 64, False, "cpu"))
    gate = mt.target_slot(b["hand"], b["act_card"].long())
    plays = b["act_card"] > 0
    assert gate[plays].tolist() == [1, 2, 1, -1]      # Knight, Arrows, HogRider; Musketeer -1
    loss, stats = mt.losses(model, b)
    assert torch.isfinite(loss) and 0.0 <= stats["card_acc"] <= 1.0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    counts = tt.eligible_counts(b)
    assert counts == {"rows": 54, "gate": 53, "card": 3, "play": 4}
