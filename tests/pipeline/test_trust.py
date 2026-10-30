"""E09: divergence-based trust, tools/rs_rows.py --trust legacy|v2.

Pure tests, worked by hand: tower_breach / seat_breach / divergence on synthetic tower tracks
and recorded final hp give the expected cut ticks and reasons (a destroyed tower the
recording ends with standing; more damage than the recording by more than the margin, with
the recorded level scaled in; the princess pairings), princess_vote, recorded_towers,
trust_rule, trust_stats / merge_stats / manifest and vocab_record.

rows_for and build on conftest's fake royalesim (no engine, no corpus): conftest's payload
(plays at 100, 160, 240, 300, 420; the team at low y plays blue) with scripted tower hp, game
over, crowns and refusals. Rows after each cut are flagged in trusted_v2, and `trusted` is the
legacy column: legacy rows (the default) have no trusted_v2, are equal key by key to the v2
rows without it, and on a destroyed princess give the hand-worked actions, outcomes and trust.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    KING_HP,
    PRINCESS_HP,
    il_payload,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FULL = (KING_HP, PRINCESS_HP, PRINCESS_HP)                     # 4824, 3052, 3052
#: Every card a side plays in conftest's payload is in its hand (as in test_orient).
HOG_HAND, HOG_NEXT = [5, 2, 6, 0], 1          # Musketeer, HogRider, Skeletons, Cannon
OPP_HAND, OPP_NEXT = [8, 11, 9, 10], 12       # Knight, Arrows, Archer, Giant
#: conftest's play rows (test_orient LEGACY_PLAY_ROWS): (tick, side, action).
PLAY_ROWS = [(100, 1, (7, 3, 24)), (160, -1, (10, 3, 24)), (240, 1, (4, 3, 17)),
             (300, -1, (13, 14, 8)), (420, 1, (8, 9, 27))]
#: The loop's sample ticks: every 20 from 20 to the last play + HORIZON (620). Each has one
#: row per seat (no play falls in a wait row's next 10 ticks), 62 rows in all.
TICKS = list(range(20, 621, 20))
#: conftest's recorded final towers: the team (blue) untouched, the opponent (red) lost its
#: princess_left; crowns (1, 0).
REC_BLUE = {"king": KING_HP, "princess_left": PRINCESS_HP, "princess_right": PRINCESS_HP}
REC_RED = {"king": KING_HP, "princess_left": 0, "princess_right": PRINCESS_HP}


@pytest.fixture
def rr(monkeypatch):
    """tools/rs_rows.py as the tools import it, with its module settings at the defaults
    (restored after the test, TRUST too) and no engine config from the PC."""
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
                        "CARD_FLY": {}, "CARD_BUILDING": {}}.items():
        monkeypatch.setattr(rr, name, value)
    return rr


def cfg(rr):
    return rr.rs_engine.EngineConfig.defaults()


def script(blue=(), red=(), over_at=None, red_crowns_at=None):
    """FakeBattle states: `blue` / `red` are [(from_tick, tower_hp)] (the last step at or
    before the tick applies; FULL before the first), game over from `over_at`, red's crowns 1
    from `red_crowns_at`. Hands hold every card the payload plays."""
    def hp_at(steps, tick):
        hp = FULL
        for start, value in steps:
            if tick >= start:
                hp = value
        return list(hp)

    def states(battle):
        t = battle.tick
        crowns = 1 if red_crowns_at is not None and t >= red_crowns_at else 0
        return state_dict(t, players=[
            player_dict(0, hand=HOG_HAND, next_card=HOG_NEXT, tower_hp=hp_at(blue, t)),
            player_dict(1, hand=OPP_HAND, next_card=OPP_NEXT, tower_hp=hp_at(red, t),
                        crowns=crowns)],
            game_over=over_at is not None and t >= over_at)
    return states


def refuse_hog(battle, cmds, n):
    """Step results: blue's HogRider (hand slot 1) is refused with NOT_ENOUGH_ELIXIR (4)."""
    return [[c[0], 4 if (c[0], c[1]) == (0, 1) else 0] for c in cmds]


@pytest.fixture
def engine(monkeypatch):
    """install(**battle_kwargs): a fake royalesim in sys.modules for this test."""
    def install(**kwargs):
        mod = make_fake_royalesim(**kwargs)
        monkeypatch.setitem(sys.modules, "royalesim", mod)
        return mod
    return install


def run(rr, trust=None, payload=None):
    """rows_for on conftest's payload: (rows, info["trust"])."""
    info = {}
    rows, why = rr.rows_for(payload or il_payload(), rr._setup(cfg(rr)), cfg(rr), trust=trust,
                            info=info)
    assert why == "ok"
    return rows, info["trust"]


def row_order(last_tick=620):
    """(tick, side) of rows_for's rows in order: a play row, then the other seat's wait row;
    two waits, blue first, on the other ticks."""
    plays = {t: s for t, s, _ in PLAY_ROWS}
    out = []
    for t in TICKS:
        if t > last_tick:
            break
        out += [(t, plays[t]), (t, -plays[t])] if t in plays else [(t, 1), (t, -1)]
    return out


# ---------------------------------------------------------------------------
# trust_rule, describe_trust, trust_vocab
# ---------------------------------------------------------------------------

def test_trust_rule_reads_modes_rules_and_their_json(rr):
    assert (rr.TRUST_MODES, rr.TRUST_SIDES) == (("legacy", "v2"), ("unordered", "world", "own"))
    assert (rr.TRUST_HP_MARGIN, rr.WAIT_TAIL, rr.TRUST_FIELDS) == (0.10, 20, ("trusted_v2",))
    assert rr.LEFT_SLOT == {"world": (1, 1), "own": (2, 1)}
    assert rr.trust_rule() == rr.TrustRule() == rr.trust_rule("legacy")
    assert not rr.trust_rule().v2
    v2 = rr.trust_rule("v2")
    assert v2 == rr.TrustRule("v2", "unordered", 0.1) and v2.v2
    own = rr.TrustRule("v2", "own", 0.25)
    assert own.to_json() == {"mode": "v2", "sides": "own", "hp_margin": 0.25, "wait_tail": 20,
                             "horizon": 200}
    assert rr.trust_rule(own.to_json()) == own and rr.trust_rule(own) == own
    assert rr.trust_rule({"mode": "v2"}) == v2
    zero = rr.trust_rule(rr.TrustRule("v2", "world", 0))
    assert zero.hp_margin == 0.0 and type(zero.hp_margin) is float
    with pytest.raises(ValueError, match="--trust: one of"):
        rr.trust_rule("v3")
    with pytest.raises(ValueError, match="--trust: one of"):
        rr.trust_rule(42)
    with pytest.raises(ValueError, match="--trust-sides"):
        rr.trust_rule(rr.TrustRule("v2", "left"))
    for bad in (-0.1, 1.5, math.nan, math.inf, True, "0.1"):
        with pytest.raises(ValueError, match="--trust-hp-margin"):
            rr.trust_rule(rr.TrustRule("v2", "unordered", bad))
    with pytest.raises(ValueError, match="unknown setting"):
        rr.trust_rule({"mode": "v2", "margin": 0.1})


def test_trust_rule_none_is_the_module_setting(rr, monkeypatch):
    monkeypatch.setattr(rr, "TRUST", rr.TrustRule("v2", "world", 0.2))
    assert rr.trust_rule() == rr.TrustRule("v2", "world", 0.2)
    assert rr.trust_vocab() == {"trust": rr.TrustRule("v2", "world", 0.2)}
    assert "world" in rr.describe_trust() and "margin 0.2" in rr.describe_trust()
    assert rr.trust_vocab("legacy") == {}
    assert rr.describe_trust("legacy").startswith("legacy")


# ---------------------------------------------------------------------------
# tower_breach, level_scale, seat_breach, divergence (by hand)
# ---------------------------------------------------------------------------

def test_tower_breach_by_hand(rr):
    lost, hp = rr.CUT_TOWER_LOST, rr.CUT_TOWER_HP
    assert (lost, hp) == ("tower_lost", "tower_hp")
    assert rr.tower_breach(0, PRINCESS_HP, PRINCESS_HP) == lost      # recorded standing
    assert rr.tower_breach(0, 1, PRINCESS_HP) == lost                # standing at 1 hp
    assert rr.tower_breach(0, 0, PRINCESS_HP) is None                # both destroyed
    assert rr.tower_breach(0, None, PRINCESS_HP) is None             # not recorded
    # king, margin 0.1 x 4824 = 482.4: damage 424 is inside, 524 is not
    assert rr.tower_breach(4400, KING_HP, KING_HP) is None
    assert rr.tower_breach(4300, KING_HP, KING_HP) == hp
    # the margin is a strict bound: margin 0.25 of 4000 is 1000 hp of extra damage
    assert rr.tower_breach(3000, 4000, 4000, 1.0, 0.25) is None
    assert rr.tower_breach(2999, 4000, 4000, 1.0, 0.25) == hp
    # recorded damage counts: the recording ended at 2000 of 3052 (1052 damage) -> floor 1357.2
    assert rr.tower_breach(1700, 2000, PRINCESS_HP) is None          # damage 1352
    assert rr.tower_breach(1690, 2000, PRINCESS_HP) == hp            # damage 1362
    assert rr.tower_breach(10, 2000, PRINCESS_HP, 1.0, 1.0) is None  # margin = the whole tower
    assert rr.tower_breach(5000, 4824, 0) is None                    # no max hp: no hp check


def test_level_scale(rr):
    assert rr.level_scale([None], KING_HP) == 1.0
    assert rr.level_scale([], KING_HP) == 1.0
    assert rr.level_scale([3000], KING_HP) == 1.0                    # never below 1
    assert rr.level_scale([5300], KING_HP) == 5300 / 4824
    assert rr.level_scale([None, 3500, 2000], PRINCESS_HP) == 3500 / 3052
    assert rr.level_scale([4824], 0) == 1.0


def test_level_scale_keeps_a_stronger_recorded_tower_from_cutting(rr):
    # An untouched recorded king of 5300 (a higher tower level) bounds the engine's damage by
    # the margin alone, as an untouched tower at the engine's level does.
    s = rr.level_scale([5300], KING_HP)
    assert rr.tower_breach(4400, 5300, KING_HP, s) is None           # damage 424 <= 482.4
    assert rr.tower_breach(4300, 5300, KING_HP, s) == rr.CUT_TOWER_HP
    # Princesses recorded at 3500 (untouched) and 2000: their max was ~3500, so the damaged
    # one took 1500; the engine (max 3052) may take 1500 + 305.2 = 1805.2.
    s = rr.level_scale([3500, 2000], PRINCESS_HP)
    assert rr.tower_breach(1300, 2000, PRINCESS_HP, s) is None       # damage 1752
    assert rr.tower_breach(1200, 2000, PRINCESS_HP, s) == rr.CUT_TOWER_HP   # damage 1852
    assert rr.tower_breach(1300, 2000, PRINCESS_HP) == rr.CUT_TOWER_HP      # unscaled: 1357.2
    rec = {"king": KING_HP, "princess_left": 3500, "princess_right": 2000}
    assert rr.seat_breach((KING_HP, 1300, PRINCESS_HP), FULL, rec, 0, "v2") is None
    assert rr.seat_breach((KING_HP, 1200, PRINCESS_HP), FULL, rec, 0, "v2") == rr.CUT_TOWER_HP


def test_seat_breach_princess_pairings(rr):
    """Blue's recording ends with princess_left 3052 and princess_right 1000; the engine has
    slot 1 (world left) at 1000 and slot 2 at 3052."""
    rec = {"king": KING_HP, "princess_left": PRINCESS_HP, "princess_right": 1000}
    hp = (KING_HP, 1000, PRINCESS_HP)
    unordered, world, own = (rr.TrustRule("v2", s) for s in ("unordered", "world", "own"))
    assert rr.seat_breach(hp, FULL, rec, 0, unordered) is None       # sorted: 1000/1000, 3052
    assert rr.seat_breach(hp, FULL, rec, 0, world) == rr.CUT_TOWER_HP   # slot 1 is left: 2052
    assert rr.seat_breach(hp, FULL, rec, 0, own) is None             # blue's left is slot 2
    assert rr.seat_breach(hp, FULL, rec, 1, unordered) is None
    assert rr.seat_breach(hp, FULL, rec, 1, world) == rr.CUT_TOWER_HP
    assert rr.seat_breach(hp, FULL, rec, 1, own) == rr.CUT_TOWER_HP  # red's left is slot 1
    # a princess destroyed while the recording ends with both standing (3052 and 1000) is a
    # loss under any pairing; with the recording's own princess_left down, only a pairing that
    # matches the engine's standing princess to it misses the loss
    down = (KING_HP, PRINCESS_HP, 0)
    for rule in (unordered, world, own):
        assert rr.seat_breach(down, FULL, rec, 0, rule) == rr.CUT_TOWER_LOST
    rec_down = {"king": KING_HP, "princess_left": 0, "princess_right": PRINCESS_HP}
    assert rr.seat_breach(down, FULL, rec_down, 0, unordered) is None
    assert rr.seat_breach(down, FULL, rec_down, 0, world) == rr.CUT_TOWER_LOST
    assert rr.seat_breach(down, FULL, rec_down, 0, own) is None


def test_seat_breach_loss_wins_and_missing_values_are_not_checked(rr):
    rec = dict(REC_BLUE)
    # the king 824 down (an hp breach) and slot 1 destroyed (a loss): the loss is reported
    assert rr.seat_breach((4000, 0, PRINCESS_HP), FULL, rec, 0) == rr.CUT_TOWER_LOST
    assert rr.seat_breach((4000, PRINCESS_HP, PRINCESS_HP), FULL, rec, 0) == rr.CUT_TOWER_HP
    assert rr.seat_breach((0, 0, 0), FULL, {}, 0) is None
    no_left = {"king": KING_HP, "princess_right": PRINCESS_HP}
    # unordered: the missing value sorts lowest and takes the engine's weakest princess
    assert rr.seat_breach((KING_HP, 0, PRINCESS_HP), FULL, no_left, 0) is None
    assert rr.seat_breach((KING_HP, 0, 2000), FULL, no_left, 0) == rr.CUT_TOWER_HP
    world = rr.TrustRule("v2", "world")
    assert rr.seat_breach((KING_HP, 0, PRINCESS_HP), FULL, no_left, 0, world) is None
    assert rr.seat_breach((KING_HP, PRINCESS_HP, 2000), FULL, no_left, 0, world) == \
        rr.CUT_TOWER_HP


def test_divergence_on_a_synthetic_track(rr):
    track = [(0, (FULL, FULL)), (20, (FULL, FULL)),
             (40, (FULL, (4300, PRINCESS_HP, PRINCESS_HP))),               # red king -524
             (60, ((KING_HP, 0, PRINCESS_HP), (4300, PRINCESS_HP, PRINCESS_HP)))]  # blue slot 1
    maxes = (FULL, FULL)
    rec = {0: dict(REC_BLUE), 1: dict(REC_RED)}
    assert rr.divergence(track, maxes, rec, "v2") == (40, "tower_hp")
    assert rr.divergence(track, maxes, {0: dict(REC_BLUE), 1: {}}, "v2") == (60, "tower_lost")
    assert rr.divergence(track, maxes, {0: dict(REC_BLUE)}, "v2") == (60, "tower_lost")
    assert rr.divergence(track, maxes, {0: {}, 1: {}}, "v2") == (None, None)
    wide = rr.TrustRule("v2", "unordered", 0.2)                      # 964.8 of the king
    assert rr.divergence(track, maxes, rec, wide) == (60, "tower_lost")
    low_king = {0: dict(REC_BLUE), 1: dict(REC_RED, king=4300)}      # recorded at 4300 too
    assert rr.divergence(track, maxes, low_king, "v2") == (60, "tower_lost")
    assert rr.divergence(track[:2], maxes, rec, "v2") == (None, None)


def test_princess_vote(rr):
    rec = dict(REC_RED)                                    # princess_left down
    assert (rr.VOTES, rr.SEAT_NAMES) == (("slot1", "slot2"), ("blue", "red"))
    assert rr.princess_vote((KING_HP, 0, PRINCESS_HP), rec) == "slot1"
    assert rr.princess_vote((KING_HP, PRINCESS_HP, 0), rec) == "slot2"
    assert rr.princess_vote((KING_HP, 0, 0), rec) is None             # the engine took both
    assert rr.princess_vote(FULL, rec) is None                        # the engine took none
    assert rr.princess_vote((KING_HP, 0, PRINCESS_HP), dict(REC_BLUE)) is None
    assert rr.princess_vote((KING_HP, 0, PRINCESS_HP), {"princess_left": 0}) is None


def test_recorded_towers_maps_sides_to_seats_and_skips_bad_values(rr):
    battle = il_payload()["battle"]
    assert rr.recorded_towers(battle, {0: 0, 1: 1}) == {0: REC_BLUE, 1: REC_RED}
    assert rr.recorded_towers(battle, {0: 1, 1: 0}) == {1: REC_BLUE, 0: REC_RED}
    odd = {"team": {"players": [{"final_tower_hitpoints": {
               "king": "x", "princess_left": None, "princess_right": 2000.0, "extra": 5}}]},
           "opponent": {"players": [{"final_tower_hitpoints": {
               "king": True, "princess_left": "1500", "princess_right": math.inf}}]}}
    assert rr.recorded_towers(odd, {0: 0, 1: 1}) == {0: {"princess_right": 2000},
                                                     1: {"princess_left": 1500}}
    bare = {"team": {"players": [{}]}, "opponent": {"players": [{"final_tower_hitpoints": []}]}}
    assert rr.recorded_towers(bare, {0: 0, 1: 1}) == {0: {}, 1: {}}


# ---------------------------------------------------------------------------
# rows_for on the fake engine
# ---------------------------------------------------------------------------

#: name -> (battle kwargs, rule, rows, legacy trusted up to tick, trusted_v2 up to tick,
#: info["trust"] under v2). None "up to": no row; 10**9: every row.
ALL = 10 ** 9
SCENARIOS = {
    # no divergence: only the waits after 420 + WAIT_TAIL lose trust
    "tail": ({"states": script()}, "v2", 62, ALL, 440,
             {"cut": None, "why": "none", "cut_v2": None, "why_v2": "none",
              "votes": [None, None]}),
    # blue's world-left princess falls at 300; the recording ends with it standing
    "lost": ({"states": script(blue=[(300, (KING_HP, 0, PRINCESS_HP))])}, "v2", 62, ALL, 100,
             {"cut": None, "why": "none", "cut_v2": 300, "why_v2": "tower_lost",
              "votes": [None, None]}),
    # red's king 424 down at 300 (inside 482.4), 524 at 400
    "hp": ({"states": script(red=[(300, (4400, PRINCESS_HP, PRINCESS_HP)),
                                  (400, (4300, PRINCESS_HP, PRINCESS_HP))])}, "v2", 62, ALL, 200,
           {"cut": None, "why": "none", "cut_v2": 400, "why_v2": "tower_hp",
            "votes": [None, None]}),
    # the engine ends the game at 300; plays at 300 and 420 are still to come
    "over": ({"states": script(over_at=300)}, "v2", 28, ALL, 100,
             {"cut": None, "why": "none", "cut_v2": 300, "why_v2": "game_over",
              "votes": [None, None]}),
    # a game over after the last play is no divergence
    "over_late": ({"states": script(over_at=500)}, "v2", 48, ALL, 440,
                  {"cut": None, "why": "none", "cut_v2": None, "why_v2": "none",
                   "votes": [None, None]}),
    # red takes a crown at 300 the recording does not give it: the legacy cut, kept in v2
    "crowns": ({"states": script(red_crowns_at=300)}, "v2", 62, 100, 100,
               {"cut": 300, "why": "crowns", "cut_v2": 300, "why_v2": "crowns",
                "votes": [None, None]}),
    # blue's HogRider refused at 240 (legacy cut), its princess down at 200 (earlier)
    "refused_lost": ({"states": script(blue=[(200, (KING_HP, 0, PRINCESS_HP))]),
                      "step_results": refuse_hog}, "v2", 62, 40, None,
                     {"cut": 240, "why": "refused", "cut_v2": 200, "why_v2": "tower_lost",
                      "votes": [None, None]}),
    # the same tick: the legacy reason is kept
    "tie": ({"states": script(blue=[(240, (KING_HP, 0, PRINCESS_HP))]),
             "step_results": refuse_hog}, "v2", 62, 40, 40,
            {"cut": 240, "why": "refused", "cut_v2": 240, "why_v2": "refused",
             "votes": [None, None]}),
    # the fake's own hands: blue's Musketeer is not in hand at 100
    "hand": ({}, "v2", 62, None, None,
             {"cut": 100, "why": "hand", "cut_v2": 100, "why_v2": "hand",
              "votes": [None, None]}),
    # red loses its world-right princess at 500; the recording lost princess_left: unordered
    # sees a princess for a princess, and red votes slot2
    "votes": ({"states": script(red=[(500, (KING_HP, PRINCESS_HP, 0))])}, "v2", 62, ALL, 440,
              {"cut": None, "why": "none", "cut_v2": None, "why_v2": "none",
               "votes": [None, "slot2"]}),
    # ... and under "world" that is a tower the recording ends with standing
    "votes_world": ({"states": script(red=[(500, (KING_HP, PRINCESS_HP, 0))])},
                    {"mode": "v2", "sides": "world"}, 62, ALL, 300,
                    {"cut": None, "why": "none", "cut_v2": 500, "why_v2": "tower_lost",
                     "votes": [None, "slot2"]}),
}


def _flags(rows, key, upto):
    return [r[key] for r in rows], [upto is not None and r["tick"] <= upto for r in rows]


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_rows_for_cuts_trusted_v2_and_keeps_trusted(name, rr, engine):
    kwargs, rule, n, trusted_upto, v2_upto, want = SCENARIOS[name]
    engine(**kwargs)
    legacy, legacy_info = run(rr)                         # the default: TRUST legacy
    rows, info = run(rr, rule)
    assert len(legacy) == len(rows) == n
    assert [(r["tick"], r["side"]) for r in rows] == row_order(TICKS[n // 2 - 1])
    got, expected = _flags(rows, "trusted_v2", v2_upto)
    assert got == expected
    assert all(type(v) is bool for v in got)
    got, expected = _flags(legacy, "trusted", trusted_upto)
    assert got == expected
    assert info == want
    assert legacy_info == {"cut": want["cut"], "why": want["why"]}
    for old, new in zip(legacy, rows):                    # v2 only adds the column
        assert "trusted_v2" not in old
        assert list(new) == list(old) + ["trusted_v2"]
        for key in old:
            assert np.array_equal(np.asarray(old[key]), np.asarray(new[key])), key


def test_rows_for_counts_by_hand(rr, engine):
    """The scenario counts: 62 rows; trusted_v2 up to 440 is 22 ticks x 2 = 44, up to 100 is 10,
    up to 200 is 20, up to 300 is 30; a game over at 300 leaves ticks 20-280, 28 rows."""
    engine(**SCENARIOS["tail"][0])
    rows, _ = run(rr, "v2")
    assert sum(r["trusted_v2"] for r in rows) == 44 and sum(r["trusted"] for r in rows) == 62
    for name, count in (("lost", 10), ("hp", 20), ("over", 10), ("votes_world", 30),
                        ("refused_lost", 0), ("tie", 4), ("crowns", 10)):
        kwargs, rule = SCENARIOS[name][:2]
        engine(**kwargs)
        rows, _ = run(rr, rule)
        assert sum(r["trusted_v2"] for r in rows) == count, name


def test_legacy_rows_golden_on_a_destroyed_princess(rr, engine):
    """Blue's princess falls at 300: every row trusted (crowns 0, every card in hand, every
    step OK); the outcome is the tower swing over 200 ticks: a row at 100-280 sees blue lose
    one princess (fraction 1.0), -1 for blue's rows and +1 for red's, 0 elsewhere."""
    engine(**SCENARIOS["lost"][0])
    rows, info = run(rr)
    assert info == {"cut": None, "why": "none"}
    plays = {(t, s): a for t, s, a in PLAY_ROWS}
    want = []
    for t, s in row_order():
        swing = 1.0 if 100 <= t <= 280 else 0.0
        want.append((t, s, plays.get((t, s), (0, 0, 0)), True, -s * swing, s == 1,
                     1.0 if s == 1 else 0.0))
    got = [(r["tick"], r["side"], tuple(r["action"]), r["trusted"], r["outcome"], r["hog"],
            r["won"]) for r in rows]
    assert got == want
    assert all("trusted_v2" not in r for r in rows)


def test_rows_for_follows_the_module_setting_and_checks_the_rule(rr, engine, monkeypatch):
    engine(**SCENARIOS["lost"][0])
    monkeypatch.setattr(rr, "TRUST", rr.TrustRule("v2"))
    rows, info = run(rr)
    assert sum(r["trusted_v2"] for r in rows) == 10 and info["why_v2"] == "tower_lost"
    with pytest.raises(ValueError, match="--trust: one of"):
        rr.rows_for(il_payload(), rr._setup(cfg(rr)), cfg(rr), trust="v3")
    rows, why = rr.rows_for(il_payload(), rr._setup(cfg(rr)), cfg(rr), trust="legacy")
    assert why == "ok" and all("trusted_v2" not in r for r in rows)


def test_rows_for_without_recorded_towers_checks_only_the_rest(rr, engine):
    engine(**SCENARIOS["lost"][0])
    payload = il_payload()
    for side in ("team", "opponent"):
        del payload["battle"][side]["players"][0]["final_tower_hitpoints"]
    rows, info = run(rr, "v2", payload)
    assert info["cut_v2"] is None and info["votes"] == [None, None]
    assert sum(r["trusted_v2"] for r in rows) == 44                  # the wait tail only


# ---------------------------------------------------------------------------
# build, trust_stats, merge_stats, manifest
# ---------------------------------------------------------------------------

ACT = ("act_card", "act_x", "act_y", "match", "split")
#: trust_stats of the "lost" match under v2: blue (the Hog 2.6 team) has 31 rows, 5 of them
#: trusted_v2 (ticks 20-100); five play rows.
LOST_STATS = {"rows": 62, "play_rows": 5, "trusted": 62, "hog_rows": 31, "hog_trusted": 31,
              "matches": 1, "cuts": {"none": 1}, "trusted_v2": 10, "hog_trusted_v2": 5,
              "cuts_v2": {"tower_lost": 1},
              "princess_votes": {"blue": {"slot1": 0, "slot2": 0},
                                 "red": {"slot1": 0, "slot2": 0}}}


def shard(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def test_build_writes_trusted_v2_only_under_v2(rr, engine, tmp_path):
    engine(**SCENARIOS["lost"][0])
    text = json.dumps(il_payload(tag="trust-1"))
    eng = cfg(rr).to_json()
    assert rr.build(1, [text], False, str(tmp_path), eng) == (1, 62, {})
    old = shard(tmp_path / "shard-0001.npz")
    assert sorted(old) == sorted(rr.FIELDS + ACT) and not rr.TRUST.v2
    assert int(old["trusted"].sum()) == 62
    batch, n, skipped, st = rr.build(2, [text], False, str(tmp_path), eng, trust="v2",
                                     stats=True)
    assert (batch, n, skipped) == (2, 62, {}) and st == LOST_STATS and rr.TRUST.v2
    new = shard(tmp_path / "shard-0002.npz")
    assert sorted(new) == sorted(rr.FIELDS + ACT + ("trusted_v2",))
    assert new["trusted_v2"].dtype == np.bool_ and new["trusted_v2"].shape == (62,)
    assert new["trusted_v2"].tolist() == [t <= 100 for t in new["tick"].tolist()]
    for k in old:
        assert np.array_equal(old[k], new[k]), k
    assert rr.build(3, [text], False, str(tmp_path), eng)[1] == 62    # None: keep v2
    assert "trusted_v2" in shard(tmp_path / "shard-0003.npz")
    back = rr.build(4, [text], False, str(tmp_path), eng, trust="legacy", stats=True)
    assert back[:3] == (4, 62, {})
    assert back[3] == {k: LOST_STATS[k] for k in ("rows", "play_rows", "trusted", "hog_rows",
                                                  "hog_trusted", "matches", "cuts")}
    again = shard(tmp_path / "shard-0004.npz")
    assert sorted(again) == sorted(old) and all(np.array_equal(again[k], old[k]) for k in old)
    with pytest.raises(ValueError, match="--trust: one of"):
        rr.build(5, [text], False, str(tmp_path), eng, trust="v3")
    assert not rr.TRUST.v2 and not (tmp_path / "shard-0005.npz").exists()


def test_build_passes_the_rule_through_its_json(rr, engine, tmp_path):
    engine(**SCENARIOS["votes"][0])
    text = json.dumps(il_payload(tag="trust-2"))
    world = rr.TrustRule("v2", "world").to_json()        # what main sends a worker
    *_, st = rr.build(6, [text], False, str(tmp_path), cfg(rr).to_json(), trust=world,
                      stats=True)
    assert rr.TRUST == rr.TrustRule("v2", "world")
    assert st["trusted_v2"] == 30 and st["cuts_v2"] == {"tower_lost": 1}
    assert st["princess_votes"] == {"blue": {"slot1": 0, "slot2": 0},
                                    "red": {"slot1": 0, "slot2": 1}}


def test_trust_stats_merge_and_manifest(rr):
    other = {"rows": 38, "play_rows": 3, "trusted": 30, "hog_rows": 0, "hog_trusted": 0,
             "matches": 1, "cuts": {"hand": 1}, "trusted_v2": 20, "hog_trusted_v2": 0,
             "cuts_v2": {"hand": 1},
             "princess_votes": {"blue": {"slot1": 1, "slot2": 0},
                                "red": {"slot1": 0, "slot2": 1}}}
    totals = rr.merge_stats(rr.merge_stats({}, LOST_STATS), other)
    assert totals == {"rows": 100, "play_rows": 8, "trusted": 92, "hog_rows": 31,
                      "hog_trusted": 31, "matches": 2, "cuts": {"none": 1, "hand": 1},
                      "trusted_v2": 30, "hog_trusted_v2": 5,
                      "cuts_v2": {"tower_lost": 1, "hand": 1},
                      "princess_votes": {"blue": {"slot1": 1, "slot2": 0},
                                         "red": {"slot1": 0, "slot2": 1}}}
    # merge_stats added into fresh dicts: the parts are unchanged
    assert LOST_STATS["cuts"] == {"none": 1} and other["cuts"] == {"hand": 1}
    assert LOST_STATS["princess_votes"]["blue"] == {"slot1": 0, "slot2": 0}
    shards = [rr.shard_entry(7, other), rr.shard_entry(2, LOST_STATS)]
    man = rr.manifest(totals, shards, {"orient": 3}, "v2")
    assert man == {
        "rows": 100, "play_rows": 8, "wait_rows": 92, "matches": 2, "skipped": {"orient": 3},
        "trust": {"mode": "v2", "sides": "unordered", "hp_margin": 0.1, "wait_tail": 20,
                  "horizon": 200},
        "trusted": 92, "trusted_rate": 0.92, "hog_trusted": 31, "hog_trusted_rate": 1.0,
        "trusted_v2": 30, "trusted_v2_rate": 0.3, "hog_trusted_v2": 5,
        "hog_trusted_v2_rate": 5 / 31, "hog_rows": 31,
        "cuts": {"none": 1, "hand": 1}, "cuts_v2": {"tower_lost": 1, "hand": 1},
        "princess_votes": {"blue": {"slot1": 1, "slot2": 0}, "red": {"slot1": 0, "slot2": 1}},
        "shards": [
            {"shard": "shard-0002.npz", "rows": 62, "play_rows": 5, "trusted": 62,
             "trusted_v2": 10, "hog_rows": 31, "hog_trusted": 31, "hog_trusted_v2": 5,
             "matches": 1},
            {"shard": "shard-0007.npz", "rows": 38, "play_rows": 3, "trusted": 30,
             "trusted_v2": 20, "hog_rows": 0, "hog_trusted": 0, "hog_trusted_v2": 0,
             "matches": 1}]}
    assert json.loads(json.dumps(man)) == man
    assert rr.describe_manifest(man) == ("trusted 92 of 100 rows (92.0%; Hog 2.6 100.0%), "
                                         "trusted_v2 30 (30.0%; Hog 2.6 16.1%)")
    legacy_totals = {k: totals[k] for k in ("rows", "play_rows", "trusted", "hog_rows",
                                            "hog_trusted", "matches", "cuts")}
    legacy = rr.manifest(legacy_totals, [rr.shard_entry(2, legacy_totals)], {}, "legacy")
    assert not {"trusted_v2", "trusted_v2_rate", "cuts_v2", "princess_votes"} & set(legacy)
    assert legacy["trusted_rate"] == 0.92 and legacy["trust"]["mode"] == "legacy"
    assert legacy["shards"] == [{"shard": "shard-0002.npz", "rows": 100, "play_rows": 8,
                                 "trusted": 92, "hog_rows": 31, "hog_trusted": 31,
                                 "matches": 2}]
    assert rr.describe_manifest(legacy) == "trusted 92 of 100 rows (92.0%; Hog 2.6 100.0%)"
    empty = rr.manifest({}, [], {}, "v2")
    assert empty["trusted_rate"] is None and empty["trusted_v2_rate"] is None
    assert empty["princess_votes"] == {"blue": {"slot1": 0, "slot2": 0},
                                       "red": {"slot1": 0, "slot2": 0}}
    assert rr.describe_manifest(empty) == ("trusted 0 of 0 rows (-; Hog 2.6 -), "
                                           "trusted_v2 0 (-; Hog 2.6 -)")


def test_trust_stats_from_rows_and_infos(rr):
    def row(tick, card, trusted, v2, hog):
        return {"tick": tick, "action": (card, 0, 0), "trusted": trusted, "trusted_v2": v2,
                "hog": hog}
    rows = [row(20, 0, True, True, True), row(40, 7, True, False, True),
            row(40, 0, False, False, False)]
    infos = [{"trust": {"cut": None, "why": "none", "cut_v2": 40, "why_v2": "tower_hp",
                        "votes": ["slot1", None]}},
             {"trust": {"cut": 30, "why": "hand", "cut_v2": 30, "why_v2": "hand",
                        "votes": ["slot1", "slot2"]}}]
    assert rr.trust_stats(rows, infos, "v2") == {
        "rows": 3, "play_rows": 1, "trusted": 2, "hog_rows": 2, "hog_trusted": 2,
        "matches": 2, "cuts": {"none": 1, "hand": 1}, "trusted_v2": 1, "hog_trusted_v2": 1,
        "cuts_v2": {"tower_hp": 1, "hand": 1},
        "princess_votes": {"blue": {"slot1": 2, "slot2": 0}, "red": {"slot1": 0, "slot2": 1}}}
    assert rr.trust_stats(rows, infos, "legacy") == {
        "rows": 3, "play_rows": 1, "trusted": 2, "hog_rows": 2, "hog_trusted": 2,
        "matches": 2, "cuts": {"none": 1, "hand": 1}}
    assert rr.trust_stats([], [], "v2")["trusted_v2"] == 0


# ---------------------------------------------------------------------------
# vocab.json, and main's wiring
# ---------------------------------------------------------------------------

class StubClasses(list):
    """The surface of a vision_classes.ClassTable vocab_record reads."""
    digest = "c1a55e5"


def test_vocab_records_the_v2_rule_only(rr):
    cat = [list(r) for r in DEFAULT_CATALOGUE]
    units, eng = list(DEFAULT_UNIT_TYPES), cfg(rr)
    fp = {"version": "0.0.0+fake", "card_table": None, "id": "f00d"}
    base = rr.vocab_record(cat, units, eng, fp, {}, False)
    for same in (rr.vocab_record(cat, units, eng, fp, {}, False, trust="legacy"),
                 rr.vocab_record(cat, units, eng, fp, {}, False, **rr.trust_vocab("legacy")),
                 rr.vocab_record(cat, units, eng, fp, {}, False, trust=rr.TrustRule())):
        assert json.dumps(same) == json.dumps(base)             # the vocab.json of before
    assert "trust" not in base and "extra_columns" not in base
    v2 = rr.vocab_record(cat, units, eng, fp, {}, False, **rr.trust_vocab("v2"))
    assert list(v2) == list(base) + ["trust", "extra_columns"]
    assert v2["trust"] == {"mode": "v2", "sides": "unordered", "hp_margin": 0.1,
                           "wait_tail": 20, "horizon": 200}
    assert v2["extra_columns"] == ["trusted_v2"]
    assert rr.trust_rule(v2["trust"]) == rr.TrustRule("v2")     # a checkpoint's meta reads back
    classes = StubClasses([types.SimpleNamespace(name="Knight", id=0)])
    both = rr.vocab_record(cat, units, eng, fp, {}, False, classes=classes, visibility="public",
                           trust=rr.TrustRule("v2", "own", 0.2))
    assert list(both) == list(base) + ["units_mode", "vision_classes_digest", "visibility",
                                       "visibility_rule", "trust", "extra_columns"]
    assert both["extra_columns"] == ["unit_form", "unit_burrow", "trusted_v2"]
    assert both["trust"]["sides"] == "own" and both["trust"]["hp_margin"] == 0.2
    with pytest.raises(ValueError, match="--trust: one of"):
        rr.vocab_record(cat, units, eng, fp, {}, False, trust="v3")


def test_main_wires_trust_into_workers_vocab_and_manifest():
    text = (TOOLS / "rs_rows.py").read_text(encoding="utf-8")
    main = text[text.index("def main()"):]
    for needle in ('"--trust"', '"--trust-sides"', '"--trust-hp-margin"',
                   "trust=TRUST.to_json()", "stats=True", "**trust_vocab(TRUST)",
                   "merge_stats(totals, st)", "shard_entry(bid, st)",
                   'OUT / "manifest.json"', "describe_trust(TRUST)"):
        assert needle in main, needle
