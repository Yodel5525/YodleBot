"""E08: public memory dated at execution (tools/rs_memory.py).

Pure tests on hand-worked plays: a play sent at t with delay 21 runs at t + 21 and is invisible
before; same-tick plays are absent from each other's rows (by order at lag 0, in any order at
BOARD_LAG), worked through the rs_rows loop the deferred wiring uses; the lag parameter;
ordering, copies and argument checks; Battle.step_commands_run() rows. Golden: the dict
rs_rows / rs_eval fill today, rebuilt by their old appends, equals view(tick, 0) fed in the old
order, and tools/rs_rows.obs_row gives the same row from either (numpy needed; conftest's
FIELD stands in for rs_rows.F, so no royalesim).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from tests.pipeline.conftest import DEPLOY_REASONS, FIELD, entity_row, state_dict

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import rs_memory as M  # noqa: E402

LAG = M.BOARD_LAG


def legacy_mem(plays):
    """The dict as rs_rows / rs_eval fill it today: each (team, card, tick) appended in order."""
    mem = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}
    for team, card, tick in plays:
        mem["seen"][team].add(card)
        mem["last"][team].append((card, tick))
    return mem


class Idx:
    """An int-like that is not an int (numpy integers behave the same way)."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------

def test_empty_view_is_the_dict_a_match_starts_from():
    mem = M.PublicMemory()
    view = mem.view(0)
    assert view == M.empty_view() == {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}
    assert list(view) == ["seen", "last"]
    assert list(view["seen"]) == [0, 1] and list(view["last"]) == [0, 1]
    assert all(type(view["seen"][t]) is set and type(view["last"][t]) is list for t in (0, 1))
    assert len(mem) == 0 and repr(mem) == "PublicMemory(plays={0: 0, 1: 0})"


def test_view_holds_int_pairs_card_then_exec_tick():
    mem = M.PublicMemory()
    mem.record(Idx(1), Idx(7), Idx(40))
    mem.record(0, 3, 41)
    view = mem.view(41)
    assert view == {"seen": {0: {3}, 1: {7}}, "last": {0: [(3, 41)], 1: [(7, 40)]}}
    (card, tick), = view["last"][1]
    assert type(card) is int and type(tick) is int
    assert type(next(iter(view["seen"][1]))) is int
    assert len(mem) == 2 and repr(mem) == "PublicMemory(plays={0: 1, 1: 1})"


# ---------------------------------------------------------------------------
# the rule: exec_tick + lag <= tick
# ---------------------------------------------------------------------------

def test_play_sent_with_delay_21_is_invisible_until_it_runs():
    mem = M.PublicMemory()
    sent, delay = 300, 21
    mem.record(1, 7, sent + delay)                      # rs_eval: send tick + command delay
    for tick in range(sent, sent + delay):              # 300 .. 320
        assert mem.view(tick, 0) == M.empty_view(), tick
    assert mem.view(321, 0) == {"seen": {0: set(), 1: {7}}, "last": {0: [], 1: [(7, 321)]}}
    # The state at 321 does not show it yet (it runs in the step from 321): BOARD_LAG waits.
    assert mem.view(321, LAG) == M.empty_view()
    assert mem.view(322, LAG)["last"][1] == [(7, 321)]
    # Today's dict, filled when the command is sent, shows it from 300: 21 ticks early.
    assert legacy_mem([(1, 7, sent)])["last"][1] == [(7, 300)]


def test_eval_wiring_with_delay_matches_the_board():
    """rs_eval under delay 21, deciding every 10 ticks: the ghost's recorded play (it ran at
    320) is sent at 299, the agent sends at 300 (it runs at 321)."""
    mem, delay = M.PublicMemory(), 21
    mem.record(1, 9, 299 + delay)
    mem.record(0, 3, 300 + delay)
    assert mem.view(310, LAG) == M.empty_view()         # neither has run
    assert mem.view(320, LAG) == M.empty_view()         # the ghost's runs in the step from 320
    assert mem.view(320, 0)["last"] == {0: [], 1: [(9, 320)]}    # lag 0: literally E <= t
    assert mem.view(330, LAG) == {"seen": {0: {3}, 1: {9}},
                                  "last": {0: [(3, 321)], 1: [(9, 320)]}}


def test_lag_shifts_what_is_visible():
    mem = M.PublicMemory()
    for card, tick in ((5, 100), (9, 150), (5, 160)):
        mem.record(1, card, tick)
    everything = [(5, 100), (9, 150), (5, 160)]
    cases = {               # lag -> (last, seen) at tick 170
        0: (everything, {5, 9}),
        10: (everything, {5, 9}),                       # 160 + 10 == 170
        11: ([(5, 100), (9, 150)], {5, 9}),
        20: ([(5, 100), (9, 150)], {5, 9}),             # 150 + 20 == 170
        21: ([(5, 100)], {5}),
        70: ([(5, 100)], {5}),                          # 100 + 70 == 170
        71: ([], set()),
    }
    for lag, (last, seen) in cases.items():
        view = mem.view(170, lag)
        assert view["last"] == {0: [], 1: last}, lag
        assert view["seen"] == {0: set(), 1: seen}, lag
    assert mem.view(165, 10)["last"][1] == [(5, 100), (9, 150)]
    with pytest.raises(ValueError, match="lag"):
        mem.view(170, -21)


def test_seen_counts_from_a_cards_earliest_play():
    mem = M.PublicMemory()
    mem.record(1, 9, 150)
    mem.record(1, 9, 120)                               # recorded later, ran earlier
    assert mem.view(119)["seen"][1] == set()
    assert mem.view(130) == {"seen": {0: set(), 1: {9}}, "last": {0: [], 1: [(9, 120)]}}
    assert mem.view(150)["last"][1] == [(9, 120), (9, 150)]


def test_last_is_ordered_by_exec_tick_ties_in_record_order():
    mem = M.PublicMemory()
    for card, tick in ((3, 50), (4, 30), (5, 50), (6, 40)):
        mem.record(0, card, tick)
    assert mem.plays(0) == [(4, 30), (6, 40), (3, 50), (5, 50)]
    assert mem.view(45) == {"seen": {0: {4, 6}, 1: set()}, "last": {0: [(4, 30), (6, 40)], 1: []}}
    assert mem.view(50)["last"][0] == [(4, 30), (6, 40), (3, 50), (5, 50)]
    assert mem.view(50)["seen"][0] == {3, 4, 5, 6}
    assert mem.plays(1) == []


def test_views_and_plays_are_copies():
    mem = M.PublicMemory()
    mem.record(1, 10, 100)
    view = mem.view(200)
    view["seen"][1].add(99)
    view["last"][1].append((99, 150))
    view["last"][0].append((98, 150))
    mem.plays(1).clear()
    assert mem.view(200) == {"seen": {0: set(), 1: {10}}, "last": {0: [], 1: [(10, 100)]}}
    assert mem.view(200) is not mem.view(200)


def test_arguments_are_checked():
    mem = M.PublicMemory()
    for team in (2, -1):
        with pytest.raises(ValueError, match="team"):
            mem.record(team, 5, 10)
    with pytest.raises(TypeError, match="team"):
        mem.record(True, 5, 10)
    for card in (0, -1):                                # 0 pads the row; a raw id -1 is no card
        with pytest.raises(ValueError, match="card"):
            mem.record(0, card, 10)
    with pytest.raises(TypeError, match="card"):
        mem.record(0, 2.0, 10)
    with pytest.raises(ValueError, match="exec_tick"):
        mem.record(0, 5, -1)
    with pytest.raises(TypeError, match="exec_tick"):
        mem.record(0, 5, "10")
    with pytest.raises(TypeError, match="tick"):
        mem.view(1.5)
    with pytest.raises(ValueError, match="team"):
        mem.plays(2)
    assert len(mem) == 0
    mem.record(0, M.UNKNOWN, 10)                        # a play perception could not name
    assert mem.view(10)["seen"][0] == {1}


# ---------------------------------------------------------------------------
# same-tick plays, through the rs_rows loop the deferred wiring uses
# ---------------------------------------------------------------------------

#: (tick, team, card token): both seats play at 100 and at 200, red at 220 (a wait tick) and 230.
QUEUE = ((100, 0, 4), (100, 1, 10), (200, 0, 5), (200, 1, 12), (220, 1, 11), (230, 1, 13))


def rows_of(queue, lag, *, old_order, wait_every=20, end=240):
    """{(tick, team, "play" | "wait"): the opponent's "last" in that row} for rs_rows' loop:
    a row at each play, and a wait row every `wait_every` ticks for a seat not acting.
    old_order: record each play as it is handled and read the memory at each row, as rs_rows
    does today; otherwise read the view once per tick, before that tick's plays are recorded."""
    mem = M.PublicMemory()
    rows = {}
    ticks = sorted({t for t, _, _ in queue} | set(range(wait_every, end + 1, wait_every)))
    for tick in ticks:
        known = mem.view(tick, lag)

        def opp_last(team):
            return (mem.view(tick, lag) if old_order else known)["last"][1 - team]

        due = [(team, card) for t, team, card in queue if t == tick]
        for team, card in due:
            rows[(tick, team, "play")] = opp_last(team)
            if old_order:
                mem.record(team, card, tick)
        if tick % wait_every == 0:
            for team in (0, 1):
                if team not in {t for t, _ in due}:
                    rows[(tick, team, "wait")] = opp_last(team)
        if not old_order:
            for team, card in due:
                mem.record(team, card, tick)
    return rows


def test_same_tick_plays_are_absent_from_each_others_rows():
    rows = rows_of(QUEUE, 0, old_order=False)
    assert len(rows) == 25           # 8 waits at 20-80, 2 plays at 100, 8 waits at 120-180,
    #                                  2 plays at 200, play + wait at 220, play at 230, 2 waits
    assert {k: rows[k] for k in (
        (100, 0, "play"), (100, 1, "play"), (120, 0, "wait"), (120, 1, "wait"),
        (200, 0, "play"), (200, 1, "play"), (220, 1, "play"), (220, 0, "wait"),
        (230, 1, "play"), (240, 0, "wait"), (240, 1, "wait"))} == {
        (100, 0, "play"): [], (100, 1, "play"): [],
        (120, 0, "wait"): [(10, 100)], (120, 1, "wait"): [(4, 100)],
        (200, 0, "play"): [(10, 100)], (200, 1, "play"): [(4, 100)],
        (220, 1, "play"): [(4, 100), (5, 200)], (220, 0, "wait"): [(10, 100), (12, 200)],
        (230, 1, "play"): [(4, 100), (5, 200)],
        (240, 0, "wait"): [(10, 100), (12, 200), (11, 220), (13, 230)],
        (240, 1, "wait"): [(4, 100), (5, 200)],
    }
    assert all(t < tick for (tick, _, _), last in rows.items() for _, t in last)
    # BOARD_LAG gives the same rows, whatever the order (tokens-17 fixed without it).
    assert rows_of(QUEUE, LAG, old_order=False) == rows
    assert rows_of(QUEUE, LAG, old_order=True) == rows


def test_old_order_at_lag_0_is_todays_leak():
    rows, old = rows_of(QUEUE, 0, old_order=False), rows_of(QUEUE, 0, old_order=True)
    assert set(old) == set(rows)
    assert {k for k in rows if rows[k] != old[k]} == {
        (100, 1, "play"), (200, 1, "play"), (220, 0, "wait")}
    assert old[(100, 1, "play")] == [(4, 100)]                  # red's row shows blue's play
    assert old[(200, 1, "play")] == [(4, 100), (5, 200)]
    assert old[(220, 0, "wait")] == [(10, 100), (12, 200), (11, 220)]


def test_perception_lag_in_the_rows():
    rows = rows_of(QUEUE, 21, old_order=False)
    assert rows[(240, 0, "wait")] == [(10, 100), (12, 200)]     # exec <= 219
    assert rows[(220, 1, "play")] == [(4, 100)]                 # exec <= 199
    assert rows[(120, 0, "wait")] == []                         # 100 + 21 > 120
    assert rows[(140, 0, "wait")] == [(10, 100)]


# ---------------------------------------------------------------------------
# Battle.step_commands_run()
# ---------------------------------------------------------------------------

def test_record_ran_takes_the_deploys_that_ran():
    mem = M.PublicMemory()
    ran = [
        (321, 0, "deploy", 3, 0),
        (321, 1, "ability", 5, 0),                                   # a press is not a play
        (330, 1, "deploy", 8, DEPLOY_REASONS.index("NO_HERO")),      # refused when it ran
        [340, 1, "deploy", 9, 0],
        (350, 0, "deploy", 1, DEPLOY_REASONS.index("GAME_OVER")),    # dropped at the end
    ]
    assert mem.record_ran(ran) == 2
    assert mem.plays(0) == [(5, 321)] and mem.plays(1) == [(11, 340)]   # catalogue id + 2
    assert mem.record_ran([]) == 0 and len(mem) == 2


# ---------------------------------------------------------------------------
# golden: today's dict, and tools/rs_rows.obs_row
# ---------------------------------------------------------------------------

#: (team, card token, tick), in the order rs_rows / rs_eval append them today.
OLD_PLAYS = ((0, 4, 100), (1, 10, 100), (1, 11, 180), (0, 5, 200), (0, 4, 260), (1, 10, 300))


def test_lag_0_in_the_old_order_is_todays_dict():
    mem = M.PublicMemory()
    for k, (team, card, tick) in enumerate(OLD_PLAYS):
        mem.record(team, card, tick)
        assert mem.view(tick, 0) == legacy_mem(OLD_PLAYS[:k + 1]), k   # read right after it
    for t in range(0, 400, 10):                          # and at any later row tick
        assert mem.view(t, 0) == legacy_mem([p for p in OLD_PLAYS if p[2] <= t]), t
    assert mem.view(10 ** 6, 0) == legacy_mem(OLD_PLAYS)
    assert mem.view(10 ** 6, 0)["last"] == {0: [(4, 100), (5, 200), (4, 260)],
                                            1: [(10, 100), (11, 180), (10, 300)]}


@pytest.fixture
def rs_rows(monkeypatch):
    """tools/rs_rows imported as the tools import it, with conftest's FIELD as its field index
    (what _setup would read from royalesim) and no royaleimitate."""
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    monkeypatch.setattr(rr, "F", dict(FIELD))
    monkeypatch.setattr(rr, "LIVE", False)
    return rr


def _state(tick):
    knight = entity_row(uid=50, team=1, kind=0, card_id=8, x=9000 * 18, y=20000 * 18, hp=600,
                        unit_type=8)
    return state_dict(tick=tick, entities=[knight])


def _same_row(a, b):
    import numpy as np
    assert list(a) == list(b)
    for key in a:
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key


#: Red's plays (tokens 10-13) and blue's (2, 5), ticks they ran; red's token 14 is a ghost play
#: rs_eval sends at 499 under delay 21, so it runs at 520.
RED = ((10, 100), (11, 200), (12, 300), (10, 350), (13, 400))
BLUE = ((2, 150), (5, 420))


def _plays(ghost_tick):
    plays = [(1, c, t) for c, t in RED] + [(0, c, t) for c, t in BLUE] + [(1, 14, ghost_tick)]
    return sorted(plays, key=lambda p: p[2])


def test_obs_row_is_the_same_from_todays_dict_and_the_view(rs_rows):
    plays = _plays(499)                         # today: dated when sent
    mem = M.PublicMemory()
    for team, card, tick in plays:
        mem.record(team, card, tick)
    old = legacy_mem(plays)
    assert mem.view(500, 0) == old
    for team in (0, 1):
        _same_row(rs_rows.obs_row(_state(500), team, 18, old, 500, live=False),
                  rs_rows.obs_row(_state(500), team, 18, mem.view(500, 0), 500, live=False))


def test_obs_row_reads_only_plays_that_ran(rs_rows):
    import numpy as np
    mem = M.PublicMemory()
    for team, card, tick in _plays(520):        # dated when they run
        mem.record(np.int16(team), np.int16(card), np.int64(tick))
    blue = rs_rows.obs_row(_state(500), 0, 18, mem.view(500, LAG), 500, live=False)
    assert blue["opp_seen"].tolist() == [10, 11, 12, 13, 0, 0, 0, 0]
    assert blue["opp_last"].tolist() == [11, 12, 10, 13]
    assert np.array_equal(blue["opp_last_age"], np.asarray([0.75, 0.5, 0.375, 0.25], np.float16))
    red = rs_rows.obs_row(_state(500), 1, 18, mem.view(500, LAG), 500, live=False)
    assert red["opp_seen"].tolist() == [2, 5, 0, 0, 0, 0, 0, 0]
    assert red["opp_last"].tolist() == [2, 5, 0, 0]
    assert np.array_equal(red["opp_last_age"], np.asarray([0.875, 0.2, 1.0, 1.0], np.float16))
    later = rs_rows.obs_row(_state(600), 0, 18, mem.view(600, LAG), 600, live=False)
    assert later["opp_seen"].tolist() == [10, 11, 12, 13, 14, 0, 0, 0]
    assert later["opp_last"].tolist() == [12, 10, 13, 14]
    assert np.array_equal(later["opp_last_age"],
                          np.asarray([0.75, 0.625, 0.5, 0.2], np.float16))
    # Today's dict (the ghost's card dated at its send, 499) shows it 21 ticks early.
    leak = rs_rows.obs_row(_state(500), 0, 18, legacy_mem(_plays(499)), 500, live=False)
    assert leak["opp_seen"].tolist() == [10, 11, 12, 13, 14, 0, 0, 0]
    assert leak["opp_last"].tolist() == [12, 10, 13, 14]
    assert np.array_equal(leak["opp_last_age"],
                          np.asarray([0.5, 0.375, 0.25, 0.0025], np.float16))
