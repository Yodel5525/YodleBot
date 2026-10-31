"""FIX E: scripts/vision/tap_refusal.py, the refused-tap guard.

Synthetic HUD values only (hands, elixir readings, OwnLanding records): no images. One
observe() per tick, the capture clock 20 ticks a second (t = 10 + tick / 20), unless a test
says otherwise. The live bot's slot 1 holds the Hog (card 2), tapped at tick 100 on own-frame
model tile (4, 22) (screen tile (13, 22)) unless a test says otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import tap_refusal as TR  # noqa: E402
from scripts.vision.types import EMPTY_CARD, OwnLanding  # noqa: E402

E = EMPTY_CARD
CANNON, FIREBALL, HOG, GOLEM, SPIRITS, MUSK, SKEL, LOG = range(8)   # conftest's catalogue order
COST = {CANNON: 3, FIREBALL: 4, HOG: 4, GOLEM: 2, SPIRITS: 1, MUSK: 4, SKEL: 1, LOG: 2}
HAND = [LOG, HOG, SKEL, SPIRITS]
T0 = 10.0


def t_of(tick: float) -> float:
    return T0 + tick / 20.0


def tap(g, tick=100, card=HOG, slot=1, tile=(4, 22), *, elixir=7000, virtual=False, t=None):
    return g.sent(card, slot, COST[card], tick, t_of(tick) if t is None else t, tile=tile,
                  screen_tile=(17 - tile[0], tile[1]), elixir_milli=elixir, spendable_milli=elixir,
                  virtual=virtual)


def rising(tick):
    """Elixir regenerating from 7.0 at tick 100 (about 1 per 2.8 s), no play."""
    return 7000 + 18 * (tick - 100)


def run(g, ticks, hand=lambda tk: HAND, elixir=rising, landings=lambda tk: (), t=t_of):
    """[(kind, tick, data)] of every event over `ticks`."""
    out = []
    for tk in ticks:
        for kind, data in g.observe(tk, t(tk), hand(tk), elixir(tk), landings(tk)):
            out.append((kind, data.get("tick"), data))
    return out


def kinds(events):
    return [(k, tick) for k, tick, _ in events]


def hand_with(slot, card):
    def fn(tk):
        h = list(HAND)
        h[slot] = card
        return h
    return fn


def away_between(lo, hi, slot=1, card=E):
    """The slot reads `card` (default: empty) on ticks lo..hi, the hand otherwise."""
    def fn(tk):
        h = list(HAND)
        if lo <= tk <= hi:
            h[slot] = card
        return h
    return fn


# ---------------------------------------------------------------------------
# 1. a landed tap is never refused
# ---------------------------------------------------------------------------

def test_a_tap_whose_slot_empties_lands_and_is_never_refused():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    hand = away_between(108, 10 ** 6)                 # the slot empties 8 ticks after the tap
    ev = run(g, range(101, 200), hand=hand)
    assert kinds(ev) == [("tap_landed", 113)]          # 6 ticks away: 108..113
    data = ev[0][2]
    assert (data["evidence"], data["hand_lag"], data["card"], data["slot"], data["tap_tick"]) == (
        "hand", 8, HOG, 1, 100)
    assert data["dry"] is False
    assert g.watches == [] and g.masks == [] and not g.finals
    s = g.summary()
    assert (s["watched"], s["landed"], s["refused"], s["final"]) == (1, 1, 0, 0)
    assert (s["hand_lag_median"], s["hand_lag_max"]) == (8, 8)


def test_a_slot_that_refills_with_another_card_also_lands():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    hand = lambda tk: HAND if tk < 104 else ([LOG, E, SKEL, SPIRITS] if tk < 106 else [LOG, MUSK, SKEL, SPIRITS])
    ev = run(g, range(101, 170), hand=hand)
    assert kinds(ev) == [("tap_landed", 109)] and ev[0][2]["hand_lag"] == 4


def test_an_elixir_drop_of_the_cost_lands_a_tap_whose_slot_still_reads_the_card():
    """The hand stuck on the Hog (a HUD misread): the elixir falls by 4 at tick 110, two reads
    in a row -> landed by the elixir at 111."""
    g = TR.RefusedTapGuard(enforce=True)
    tap(g, elixir=rising(100))
    elixir = lambda tk: rising(tk) - (4000 if tk >= 110 else 0)
    ev = run(g, range(101, 200), elixir=elixir)
    assert kinds(ev) == [("tap_landed", 111)]
    assert (ev[0][2]["evidence"], ev[0][2]["elixir_lag"]) == ("elixir", 11)


def test_one_low_elixir_read_is_not_a_landing():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    elixir = lambda tk: rising(tk) - (4000 if tk == 110 else 0)
    assert kinds(run(g, range(101, 131), elixir=elixir)) == [("tap_refused", 130)]


def test_a_board_landing_matched_to_the_tap_lands_it():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    hog = OwnLanding(card=HOG, slot=1, tap_tick=100, land_tick=124, x_mt=13500.0, y_mt=22500.0)
    other = OwnLanding(card=HOG, slot=1, tap_tick=90, land_tick=124)          # another tap's
    missed = OwnLanding(card=HOG, slot=1, tap_tick=100, land_tick=None)       # a miss: no land tick
    ev = run(g, range(101, 200), landings=lambda tk: [other, missed] if tk == 120 else
             ([hog] if tk == 125 else ()))
    assert kinds(ev) == [("tap_landed", 125)]
    assert (ev[0][2]["evidence"], ev[0][2]["board_lag"]) == ("board", 24)


# ---------------------------------------------------------------------------
# 2. refused at +30, not before
# ---------------------------------------------------------------------------

def test_a_card_that_stays_in_its_slot_is_refused_at_30_ticks():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g, elixir=rising(100))
    assert run(g, range(101, 130)) == []                 # +29: nothing yet
    [(kind, tick, rec)] = run(g, [130])
    assert (kind, tick) == ("tap_refused", 130)
    assert (rec["tap_tick"], rec["age"], rec["card"], rec["slot"], rec["cost"]) == (100, 30, HOG, 1, 4)
    assert rec["model_tile"] == [4, 22] and rec["screen_tile"] == [13, 22] and rec["zone"] == "own"
    assert rec["mask"] == {"kind": "box", "x0": 2, "y0": 20, "x1": 6, "y1": 24, "until_tick": 730}
    assert (rec["toggle_suspect"], rec["present_ticks"], rec["reads"], rec["away_max"]) == (False, 30, 30, 0)
    assert (rec["elixir_at_tap"], rec["elixir_now"], rec["elixir_max"]) == (7000, rising(130), rising(130))
    assert (rec["spendable_at_tap"], rec["refusals"], rec["dry"], rec["applied"]) == (7000, 1, False, True)
    # not counted yet: final at +60 (the pending entry's age), then counted toward k
    assert g.finals == []
    ev = run(g, range(131, 161))
    assert kinds(ev) == [("tap_refusal_final", 160)]
    assert ev[0][2]["finals"] == 1 and len(g.finals) == 1
    assert g.watches == [] and len(g.masks) == 1                 # the mask stays after the final


# ---------------------------------------------------------------------------
# 3. hand misreads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ticks", [1, 2, 5])
def test_a_short_empty_misread_is_not_a_landing(ticks):
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    ev = run(g, range(101, 131), hand=away_between(110, 109 + ticks))
    assert kinds(ev) == [("tap_refused", 130)] and ev[0][2]["away_max"] == ticks


def test_a_six_tick_empty_run_is_a_landing():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    ev = run(g, range(101, 170), hand=away_between(110, 115))
    assert kinds(ev) == [("tap_landed", 115)] and ev[0][2]["hand_lag"] == 10


def test_an_away_run_at_30_ticks_defers_the_verdict_until_it_ends():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    ev = run(g, range(101, 140), hand=away_between(128, 130))
    assert kinds(ev) == [("tap_refused", 131)] and ev[0][2]["away_max"] == 3


def test_an_away_run_at_30_ticks_that_lasts_is_a_landing():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    ev = run(g, range(101, 140), hand=away_between(128, 10 ** 6))
    assert kinds(ev) == [("tap_landed", 133)] and ev[0][2]["hand_lag"] == 28


# ---------------------------------------------------------------------------
# 4. too few reads: unconfirmed; unread hands are skipped
# ---------------------------------------------------------------------------

def test_too_few_reads_of_the_card_give_unconfirmed_and_no_mask():
    """The hand reads on ticks 101-105 only; after that no slot shows a card (a failed read or a
    covered panel): those frames are skipped, not taken as the slot reading away."""
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    hand = lambda tk: HAND if tk <= 105 else [E, E, E, E]
    ev = run(g, range(101, 170), hand=hand)
    assert kinds(ev) == [("tap_unconfirmed", 160)]
    rec = ev[0][2]
    assert (rec["reads"], rec["present_ticks"]) == (5, 5)
    assert g.masks == [] and g.finals == [] and g.summary()["unconfirmed"] == 1
    assert not g.masked(HOG, 4, 22, 170)


def test_sparse_reads_still_refuse_once_there_are_ten():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    ev = run(g, range(101, 170, 3))                     # every third tick: 101, 104, ..., 128, 131
    assert kinds(ev) == [("tap_refused", 131), ("tap_refusal_final", 161)]
    assert ev[0][2]["present_ticks"] == 11


# ---------------------------------------------------------------------------
# 5. retraction
# ---------------------------------------------------------------------------

def test_a_late_landing_retracts_the_refusal_and_its_mask():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    ev = run(g, range(101, 200), hand=away_between(140, 10 ** 6))
    assert kinds(ev) == [("tap_refused", 130), ("tap_refusal_retracted", 145)]
    rec = ev[1][2]
    assert (rec["evidence"], rec["lag"], rec["card"], rec["slot"], rec["tap_tick"]) == ("hand", 40, HOG, 1, 100)
    assert g.masks == [] and g.finals == [] and not g.masked(HOG, 4, 22, 150)
    s = g.summary()
    assert (s["refused"], s["retracted"], s["final"]) == (1, 1, 0)


def test_a_late_board_landing_retracts_but_a_late_elixir_drop_does_not():
    hog = OwnLanding(card=HOG, slot=1, tap_tick=100, land_tick=150)
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    ev = run(g, range(101, 200), landings=lambda tk: [hog] if tk == 152 else ())
    assert kinds(ev) == [("tap_refused", 130), ("tap_refusal_retracted", 152)]
    assert ev[1][2]["evidence"] == "board"
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    ev = run(g, range(101, 200), elixir=lambda tk: rising(tk) - (4000 if tk >= 140 else 0))
    assert kinds(ev) == [("tap_refused", 130), ("tap_refusal_final", 160)]


def test_a_landing_after_60_ticks_is_ignored():
    hog = OwnLanding(card=HOG, slot=1, tap_tick=100, land_tick=158)
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    ev = run(g, range(101, 220), hand=away_between(161, 10 ** 6),
             landings=lambda tk: [hog] if tk == 165 else ())
    assert kinds(ev) == [("tap_refused", 130), ("tap_refusal_final", 160)]
    assert len(g.finals) == 1 and len(g.masks) == 1


def test_a_new_tap_on_the_same_slot_makes_the_refused_one_final():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    assert kinds(run(g, range(101, 141))) == [("tap_refused", 130)]
    tap(g, 141, tile=(10, 20))
    ev = run(g, [142])
    assert kinds(ev) == [("tap_refusal_final", 141)]
    assert ev[0][2]["tap_tick"] == 100 and len(g.finals) == 1
    assert [w.tap_tick for w in g.watches] == [141]


# ---------------------------------------------------------------------------
# 6. masks
# ---------------------------------------------------------------------------

def test_masks_cover_the_enemy_half_of_a_lane_or_a_box_in_our_half():
    m = TR.mask_for(GOLEM, 9, 11, 1234, 1200)
    assert (m.kind, m.x0, m.y0, m.x1, m.y1, m.until_tick) == ("lane", 9, 0, 17, 14, None)
    assert m.covers(GOLEM, 9, 11, 10 ** 6) and m.covers(GOLEM, 17, 0, 5000) and m.covers(GOLEM, 12, 14, 0)
    assert not m.covers(GOLEM, 8, 11, 2000) and not m.covers(GOLEM, 12, 15, 2000)
    assert not m.covers(HOG, 9, 11, 2000)                       # that card only
    m = TR.mask_for(GOLEM, 3, 12, 1234, 1200)
    assert (m.kind, m.x0, m.x1, m.y0, m.y1) == ("lane", 0, 8, 0, 14)
    m = TR.mask_for(HOG, 4, 22, 130, 100)
    assert (m.kind, m.x0, m.y0, m.x1, m.y1, m.until_tick) == ("box", 2, 20, 6, 24, 730)
    assert m.covers(HOG, 6, 24, 729) and not m.covers(HOG, 6, 24, 730) and not m.covers(HOG, 7, 22, 200)
    m = TR.mask_for(HOG, 3, 15, 130, 100)                       # the river: a box, not a lane
    assert (m.kind, m.x0, m.y0, m.x1, m.y1) == ("box", 1, 13, 5, 17)
    m = TR.mask_for(SKEL, 0, 31, 130, 100)                      # clipped at the arena's edge
    assert (m.x0, m.y0, m.x1, m.y1) == (0, 29, 2, 31)
    assert TR.zone_of(9, 11, (8, 11)) == "enemy_left" and TR.zone_of(9, 11) == "enemy_left"
    assert TR.zone_of(3, 12) == "enemy_right" and TR.zone_of(3, 15) == "own"


def test_a_refusal_on_the_enemy_half_masks_that_lane_for_the_battle():
    g = TR.RefusedTapGuard(enforce=True)
    g.sent(GOLEM, 0, 2, 100, t_of(100), tile=(9, 11), screen_tile=(8, 11), elixir_milli=7000)
    hand = lambda tk: [GOLEM, HOG, SKEL, SPIRITS]
    [(kind, _, rec)] = run(g, range(101, 131), hand=hand)
    assert kind == "tap_refused" and rec["zone"] == "enemy_left"
    assert rec["mask"] == {"kind": "lane", "x0": 9, "y0": 0, "x1": 17, "y1": 14, "until_tick": None}
    assert g.masked(GOLEM, 13, 2, 131) and g.masked(GOLEM, 13, 2, 4000)
    assert not g.masked(GOLEM, 4, 2, 131) and not g.masked(GOLEM, 13, 20, 131)
    assert not g.masked(HOG, 13, 2, 131)


def test_a_box_mask_expires_after_600_ticks():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    run(g, range(101, 161))
    assert g.masked(HOG, 4, 22, 729) and g.active(729)
    run(g, [730])
    assert not g.masked(HOG, 4, 22, 730) and g.masks == [] and not g.active(730)


# ---------------------------------------------------------------------------
# 7. an open watch blocks its card everywhere
# ---------------------------------------------------------------------------

def test_an_open_watch_blocks_its_card_at_every_tile_until_its_verdict():
    g = TR.RefusedTapGuard(enforce=True)
    assert not g.active(100)
    tap(g)
    assert g.active(100)
    assert g.masked(HOG, 0, 0, 101) and g.masked(HOG, 15, 28, 101)
    assert not g.masked(LOG, 4, 22, 101) and not g.masked(SKEL, 0, 0, 101)
    run(g, range(101, 130))
    assert g.masked(HOG, 15, 28, 129)
    run(g, [130])                                               # refused: only its box now
    assert not g.masked(HOG, 15, 28, 130) and g.masked(HOG, 4, 22, 130)
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    run(g, range(101, 120), hand=away_between(105, 10 ** 6))    # landed at 110: free again
    assert not g.masked(HOG, 4, 22, 120) and not g.active(120)


# ---------------------------------------------------------------------------
# 8. the selected-card hazard
# ---------------------------------------------------------------------------

def test_a_second_refused_tap_on_the_same_slot_is_toggle_suspect():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g, 100)
    run(g, range(101, 162))                                     # refused at 130, final at 160
    second = tap(g, 162, tile=(10, 20))
    assert second.toggle_suspect
    ev = run(g, range(163, 224))
    assert kinds(ev) == [("tap_refused", 192), ("tap_refusal_final", 222)]
    rec = ev[0][2]
    assert rec["toggle_suspect"] is True and rec["mask"] is None and rec["applied"] is False
    assert len(g.finals) == 2 and len(g.masks) == 1             # counted, not masked
    assert not g.masked(HOG, 10, 20, 200)
    third = tap(g, 224, tile=(10, 20))                          # the selection was cleared
    assert not third.toggle_suspect
    ev = run(g, range(225, 255))
    assert ev[-1][2]["toggle_suspect"] is False and ev[-1][2]["mask"]["kind"] == "box"
    assert g.summary()["toggle"] == 1


def test_a_tap_on_another_slot_in_between_clears_the_selection():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g, 100)
    run(g, range(101, 140))                                     # slot 1 refused at 130
    tap(g, 140, card=LOG, slot=0, tile=(4, 20))                 # slot 0: lands
    run(g, range(141, 170), hand=lambda tk: [E if tk >= 142 else LOG, HOG, SKEL, SPIRITS])
    assert not tap(g, 170, tile=(10, 20)).toggle_suspect


def test_a_refusal_after_a_newer_tap_does_not_remember_a_selection():
    """Slot 1 tapped at 100, slot 0 at 110 (it selected the Log instead): slot 1's refusal at
    130 is not the card selected now, so the next slot-1 tap is no toggle."""
    g = TR.RefusedTapGuard(enforce=True)
    tap(g, 100)
    tap(g, 110, card=LOG, slot=0, tile=(4, 20))
    run(g, range(101, 170), hand=lambda tk: [E if tk >= 112 else LOG, HOG, SKEL, SPIRITS])
    assert not tap(g, 170, tile=(10, 20)).toggle_suspect


# ---------------------------------------------------------------------------
# 9. k refusals -> MANUAL, rearm, new battle
# ---------------------------------------------------------------------------

def three_refusals(g, start=100):
    """Slots 1, 0 and 2 tapped 70 ticks apart, each refused and final; the events."""
    ev = []
    for i, (card, slot, tile) in enumerate([(HOG, 1, (4, 22)), (LOG, 0, (9, 11)), (SKEL, 2, (12, 25))]):
        tk = start + 70 * i
        tap(g, tk, card=card, slot=slot, tile=tile)
        ev += run(g, range(tk + 1, tk + 70))
    return ev


def test_three_final_refusals_switch_to_manual_once():
    g = TR.RefusedTapGuard(enforce=True)
    ev = three_refusals(g)
    manual = [e for e in ev if e[0] == "tap_guard_manual"]
    assert len(manual) == 1 and manual[0][1] == 300
    rec = manual[0][2]
    assert rec["k"] == 3 and rec["dry"] is False
    assert [(f["card"], f["slot"], f["tap_tick"], f["tile"], f["toggle_suspect"]) for f in rec["finals"]] == [
        (HOG, 1, 100, [4, 22], False), (LOG, 0, 170, [9, 11], False), (SKEL, 2, 240, [12, 25], False)]
    assert [e[1] for e in ev if e[0] == "tap_refusal_final"] == [160, 230, 300]
    # a fourth does not switch again
    tap(g, 310, card=SPIRITS, slot=3, tile=(5, 25))
    assert not [e for e in run(g, range(311, 380)) if e[0] == "tap_guard_manual"]
    # rearm (the owner handed back): the count starts again, the masks stay
    masks = list(g.masks)
    g.rearm(380)
    assert g.finals == [] and g.masks == masks and g.masked(LOG, 13, 3, 380)
    tap(g, 380, tile=(10, 25))
    ev = run(g, range(381, 450))
    assert [e[0] for e in ev] == ["tap_refused", "tap_refusal_final"]
    assert len(g.finals) == 1
    # a new battle clears everything
    g.new_battle()
    assert g.watches == [] and g.masks == [] and g.finals == [] and g.selected_slot is None
    assert not g.masked(LOG, 13, 3, 500) and g.summary()["watched"] == 0
    assert g.summary(session=True)["watched"] == 5 and g.summary(session=True)["final"] == 5


def test_one_refusal_counts_one_and_k_is_configurable():
    g = TR.RefusedTapGuard(enforce=True, k=1)
    tap(g)
    ev = run(g, range(101, 170))
    assert [e[0] for e in ev] == ["tap_refused", "tap_refusal_final", "tap_guard_manual"]


# ---------------------------------------------------------------------------
# 10. two watches, one drop
# ---------------------------------------------------------------------------

def test_a_drop_while_one_slot_is_empty_belongs_to_that_slots_tap():
    """A: the Hog (slot 1) at 100; B: the Log (slot 0) at 105. At 120 B's slot reads empty and
    the elixir falls by 2: B landed (elixir, then never refused), A is refused at 130."""
    g = TR.RefusedTapGuard(enforce=True)
    tap(g, 100)
    tap(g, 105, card=LOG, slot=0, tile=(4, 20))
    hand = lambda tk: [E if tk >= 120 else LOG, HOG, SKEL, SPIRITS]
    elixir = lambda tk: 7000 if tk < 120 else 5000
    ev = run(g, range(101, 170), hand=hand, elixir=elixir)
    assert [(k, tk, d["card"]) for k, tk, d in ev] == [
        ("tap_landed", 121, LOG), ("tap_refused", 130, HOG), ("tap_refusal_final", 160, HOG)]
    assert ev[0][2]["evidence"] == "elixir"


def test_a_drop_credits_the_watch_whose_slot_reads_away_and_only_one():
    """B's slot reads another card (not empty) when the elixir falls by 4: both watches fit the
    drop, B is credited (its slot is away), A starts its maximum again and is refused."""
    g = TR.RefusedTapGuard(enforce=True)
    tap(g, 100)
    tap(g, 105, card=MUSK, slot=0, tile=(4, 20))
    hand = lambda tk: [CANNON if tk >= 120 else MUSK, HOG, SKEL, SPIRITS]
    elixir = lambda tk: 7000 if tk < 120 else 3000
    ev = run(g, range(101, 140), hand=hand, elixir=elixir)
    assert [(k, tk, d["card"], d.get("evidence")) for k, tk, d in ev] == [
        ("tap_landed", 121, MUSK, "elixir"), ("tap_refused", 130, HOG, None)]


def test_a_drop_from_another_slots_play_is_not_taken_after_its_refill():
    """A human play (no watch) from slot 2 at 110: its slot empties, the elixir falls by 1, the
    slot refills at 125. The Hog's watch must not see that drop once the slot has its card."""
    g = TR.RefusedTapGuard(enforce=True)
    tap(g, 100)
    hand = lambda tk: [LOG, HOG, E if 110 <= tk < 125 else (CANNON if tk >= 125 else SKEL), SPIRITS]
    elixir = lambda tk: rising(tk) - (1000 if tk >= 110 else 0)
    ev = run(g, range(101, 131), hand=hand, elixir=elixir)
    assert kinds(ev) == [("tap_refused", 130)]


# ---------------------------------------------------------------------------
# 11. not enforced: logs only
# ---------------------------------------------------------------------------

def test_without_enforce_nothing_is_masked_and_every_event_is_dry():
    g = TR.RefusedTapGuard()
    w = tap(g, virtual=True)
    assert w.virtual
    assert tap(g, 105, virtual=True) is None                    # one open watch per slot
    assert tap(g, 105, card=LOG, slot=0, tile=(4, 20), virtual=True) is not None
    assert not g.masked(HOG, 4, 22, 101) and not g.active(101)
    ev = run(g, range(101, 300), hand=lambda tk: [E if tk >= 108 else LOG, HOG, SKEL, SPIRITS])
    refused = [d for k, _, d in ev if k == "tap_refused"]
    assert refused and all(d["dry"] is True and d["applied"] is False for d in refused)
    assert refused[0]["mask"]["kind"] == "box"                  # what it would have masked
    assert all(d["dry"] for _, _, d in ev)
    assert not g.masked(HOG, 4, 22, 140) and g.masks == [] and not g.active(140)
    assert g.summary()["masks"] == 1


def test_a_virtual_watch_is_skipped_while_a_real_one_is_open_on_its_slot():
    g = TR.RefusedTapGuard(enforce=True)
    tap(g, 100)
    assert tap(g, 101, virtual=True) is None
    assert len(g.watches) == 1 and not g.watches[0].virtual


# ---------------------------------------------------------------------------
# 12. the age is capped by the capture clock
# ---------------------------------------------------------------------------

def test_a_forward_tick_jump_does_not_make_a_watch_final():
    """Tick 112 misread as 200 (a clock misread), 0.6 s after the tap: its age is 14, not 100."""
    g = TR.RefusedTapGuard(enforce=True)
    tap(g)
    out = run(g, range(101, 112))
    out += [(k, d["tick"], d) for k, d in g.observe(200, t_of(112), HAND, 7200, ())]
    out += run(g, range(113, 170))
    assert kinds(out) == [("tap_refused", 130), ("tap_refusal_final", 160)]
    assert out[0][2]["age"] == 30
    w = TR.TapWatch(HOG, 1, 4, 100, t_of(100), 4, 22)
    assert w.age(200, t_of(112)) == 14 and w.age(130, t_of(130)) == 30 and w.age(130) == 30


# ---------------------------------------------------------------------------
# 13. the constants
# ---------------------------------------------------------------------------

def test_constants_agree_with_the_pending_window():
    pytest.importorskip("numpy")
    from scripts import vision_play as VP
    import public_state as PS
    assert TR.FINAL_AFTER_TICKS == VP.RD.PENDING_TICKS == PS.TAP_MAX_AGE
    assert TR.REFUSE_AFTER_TICKS < TR.FINAL_AFTER_TICKS
    assert TR.LAND_AWAY_TICKS < TR.REFUSE_AFTER_TICKS and TR.PRESENT_TAIL_TICKS < TR.LAND_AWAY_TICKS
    assert TR.EMPTY_CARD == EMPTY_CARD
    cfg = TR.RefusedTapGuard(enforce=True).config()
    assert cfg["enforce"] is True and cfg["k"] == TR.REFUSALS_TO_MANUAL == 3
    assert (cfg["refuse_after_ticks"], cfg["final_after_ticks"], cfg["land_away_ticks"]) == (30, 60, 6)


# ---------------------------------------------------------------------------
# 14. the HUD's hand tracker: one empty frame is a short away run
# ---------------------------------------------------------------------------

def test_one_empty_frame_through_the_hand_tracker_is_a_short_away_run():
    pytest.importorskip("numpy")
    from scripts.vision.hud import HandReading, HandTracker, SlotRead
    deck = list(range(8))
    nxt = CANNON

    def reading(cards, scores=None):
        slots = tuple(SlotRead(c, 0.9, 0.5, scores=dict((scores or {}).get(i, {c: 0.9})))
                      if c != E else SlotRead(E, 0.0, 0.0, empty=True) for i, c in enumerate(cards))
        return HandReading(slots, SlotRead(nxt, 0.9, 0.5, scores={nxt: 0.9}))

    # (the slot's NCC scores, the ticks it then reads away): with the Hog clearly read, the
    # empty frame alone; with the next card (Cannon) scoring 0.3 there too, the FIFO prior takes
    # the Cannon as the refill until rule 4's three votes fix it in place, 4 ticks in all
    for scores, expect in ((None, [110]), ({1: {HOG: 0.9, CANNON: 0.3}}, [110, 111, 112, 113])):
        tr = HandTracker(deck)
        g = TR.RefusedTapGuard(enforce=True)
        hands = []
        for tk in range(90, 170):
            raw = [LOG, E if tk == 110 else HOG, SKEL, SPIRITS]
            st = tr.update(reading(raw, scores), t_of(tk))
            hands.append((tk, list(st.hand)))
            if tk == 100:
                tap(g, 100)
            elif tk > 100:
                for kind, data in g.observe(tk, t_of(tk), st.hand, 7000, ()):
                    hands.append((kind, data["tick"]))
        assert ("tap_refused", 130) in hands, hands
        assert not [h for h in hands if h[0] == "tap_landed"]
        away = [tk for tk, h in hands if isinstance(tk, int) and h[1] != HOG]
        assert away == expect and len(away) < TR.LAND_AWAY_TICKS


# ---------------------------------------------------------------------------
# 15. RefusalLegality (scripts/vision_play.py)
# ---------------------------------------------------------------------------

def test_refusal_legality_masks_tiles_and_otherwise_returns_the_inner_check():
    pytest.importorskip("numpy")
    from scripts import vision_play as VP
    from tests.pipeline.conftest import DEFAULT_CATALOGUE, player_dict, state_dict
    RD = VP.RD
    sub = RD.SUBTILE_PER_MILLITILE
    me = player_dict(1, hand=list(HAND))
    state = state_dict(140, players=[player_dict(0), me])

    def at(check, card, tx, ty, slot=1):
        """The live bot's own-frame model tile (tx, ty), as decide() asks it."""
        wx, wy = RD.to_world(tx * 1000 + 500, ty * 1000 + 500, 1)
        return check(slot, card, wx * sub, wy * sub)

    g = TR.RefusedTapGuard(enforce=True)
    pure = RD.PureLegality(DEFAULT_CATALOGUE)
    leg = VP.RefusalLegality(pure, g, sub=sub)
    assert leg.name == "pure+refusals"
    plain = leg.bind(state, 1)
    assert plain.name == "pure" and at(plain, HOG, 4, 22) and not at(plain, HOG, 9, 11)

    fixed = RD.Bound("fixed", lambda slot, card, x, y: True)
    inner = type("Inner", (), {"name": "inner", "bind": lambda self, s, t: fixed})()
    assert VP.RefusalLegality(inner, g, sub=sub).bind(state, 1) is fixed       # nothing masked

    tap(g, 100)
    run(g, range(101, 140))                                      # refused at 130: box 2-6 x 20-24
    bound = leg.bind(state, 1)
    assert bound.name == "pure+refusals"
    assert not at(bound, HOG, 4, 22) and not at(bound, HOG, 6, 24)
    assert at(bound, HOG, 7, 22) and at(bound, LOG, 4, 22)
    assert not at(bound, HOG, 9, 11)                             # the inner check still applies

    calls = []

    def inner_check(slot, card, x, y):
        calls.append((slot, card))
        return True

    wrapped = VP.RefusalLegality(inner_check, g, sub=sub).bind(state, 1)
    assert wrapped.name == "callable+refusals"
    assert not at(wrapped, HOG, 4, 22) and calls == []           # masked: the inner is not asked
    assert at(wrapped, HOG, 8, 22) and calls == [(1, HOG)]
    # not enforced (a dry run): the inner check itself, whatever the guard holds
    g.enforce = False
    assert VP.RefusalLegality(inner, g, sub=sub).bind(state, 1) is fixed
