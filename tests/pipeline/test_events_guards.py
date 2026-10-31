"""The opponent-play guards (scripts/vision/events.py module doc, GUARDS; events.LIVE_GUARDS).

Each test replays, with synthetic tracks, one way the owner's 2026-10-29 friendly turned one
enemy unit into several plays (103 plays of 27 cards for one 8-card deck), and checks the guard
that answers it against the rules as they were (EventConfig(), the golden behaviour the other
event tests pin). Then the ways the first version of the guards lost real plays (the review of
e2188e5): a push whose first tracks were flashes and whose later ones were past the river, a
unit deployed where another label had just stopped or died, a unit seen only in short
fragments, units spawned past the river by a spell the detector missed. World millitiles,
battle ticks; we are team 1 at high y, the opponent team 0. Also: Tracker.inherit_team,
public_state.OpponentElixir's unaffordable count, the runners' --legacy-opp-plays, the HUD towers
reaching the deriver and the replay's and battle end's counts.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import events as EV  # noqa: E402
from scripts.vision import tracker as T  # noqa: E402
from scripts.vision import types as VT  # noqa: E402

CANNON, FIREBALL, HOG, ICE_GOLEM, ICE_SPIRITS, MUSKETEER, SKELETONS, LOG = range(8)
KNIGHT, SKELETON_ARMY, ROYAL_RECRUITS, GUARDS, MINER, GOBLINS = 8, 21, 47, 25, 26, 30
HOG26 = range(8)
E, O, U = VT.ENEMY_TEAM, VT.OWN_TEAM, VT.UNKNOWN_TEAM


def tables() -> EV.EventTables:
    return EV.EventTables(
        compositions={CANNON: {"Cannon": 1}, HOG: {"HogRider": 1}, ICE_GOLEM: {"IceGolemite": 1},
                      ICE_SPIRITS: {"IceSpirits": 1}, MUSKETEER: {"Musketeer": 1},
                      SKELETONS: {"Skeleton": 3}, KNIGHT: {"Knight": 1},
                      SKELETON_ARMY: {"Skeleton": 15}, ROYAL_RECRUITS: {"Recruit": 6},
                      GUARDS: {"SkeletonWarrior": 3}, MINER: {"Miner": 1}, GOBLINS: {"Goblin": 3}},
        spells={FIREBALL: {"FireballSpell"}, LOG: {"LogProjectile"}})


def tk(uid, cls, team, x, y, first, last, vx=0.0, vy=0.0) -> VT.Track:
    return VT.Track(uid=uid, cls=cls, team=team, conf=0.9, x_mt=float(x), y_mt=float(y),
                    first_seen_tick=first, last_seen_tick=last, vx=float(vx), vy=float(vy))


def at(track, last, **kw):
    return replace(track, last_seen_tick=last, **kw)


def deriver(config=None, **kw) -> EV.EventDeriver:
    d = EV.EventDeriver(tables(), config, own_deck=HOG26, **kw)
    d.update([], 0)                    # the battle starts on an empty board
    return d


def live(**over) -> EV.EventDeriver:
    return deriver(EV.EventConfig.live(**over))


def plays(d):
    return [(p.card, p.tick) for p in d.opp_plays()]


# ---------------------------------------------------------------------------
# the config
# ---------------------------------------------------------------------------
def test_the_guards_are_off_by_default_and_on_in_the_live_config():
    off = EV.EventConfig()
    assert (off.min_play_age_ticks, off.reid_any_label, off.deploy_zone_mt, off.own_reid_cross_team,
            off.own_recent_ticks) == (0, False, None, False, 0)
    assert EV.guard_settings(None) == EV.guard_settings(off) == {
        "min_play_age_ticks": 0, "reid_any_label": False, "deploy_zone_mt": None,
        "own_reid_cross_team": False, "own_recent_ticks": 0, "on": False}
    on = EV.EventConfig.live()
    assert dict(EV.LIVE_GUARDS) == {"min_play_age_ticks": 10, "reid_any_label": True, "deploy_zone_mt": 1000.0}
    assert (on.min_play_age_ticks, on.reid_any_label, on.deploy_zone_mt) == (10, True, 1000.0)
    assert (on.own_reid_cross_team, on.own_recent_ticks) == (False, 0)            # not supported by the data
    assert EV.guard_settings(on)["on"] is True and on.young_drop_ticks == 20
    assert EV.EventConfig.live(min_play_age_ticks=0, unknown_plays=True).min_play_age_ticks == 0
    assert on.deploy_anywhere == EV.DEPLOY_ANYWHERE_CLASSES == frozenset({"Miner", "GoblinDrill", "GoblinDrillDig"})
    assert EV.EventConfig(deploy_anywhere="Miner").deploy_anywhere == frozenset({"Miner"})
    assert EV.EventConfig(deploy_anywhere=["Miner"]) == EV.EventConfig(deploy_anywhere={"Miner"})
    assert (on.reid_min_speed_mt_per_tick, on.late_reach_ticks, on.late_reach_mt, on.late_reach_mt_per_tick) == (
        40.0, 80, 1500.0, 120.0)
    for bad in ({"min_play_age_ticks": -1}, {"young_drop_ticks": -1}, {"own_recent_ticks": -1},
                {"reid_motion_cos": 1.5}, {"deploy_zone_mt": -1.0}, {"reid_min_speed_mt_per_tick": -1.0},
                {"own_recent_radius_mt": -1.0}, {"late_reach_ticks": -1}, {"late_reach_mt": -1.0},
                {"late_reach_mt_per_tick": -1.0}):
        with pytest.raises(ValueError):
            EV.EventConfig(**bad)


# ---------------------------------------------------------------------------
# min_play_age_ticks: a flash of 1-2 frames is no play; a play is named by its matured label
# ---------------------------------------------------------------------------
def test_a_flash_is_no_play_and_a_unit_becomes_one_once_seen_for_ten_ticks():
    k = tk(1, "Knight", E, 9000, 8000, 100, 101)
    old = deriver()
    assert [p.card for p in old.update([k], 101).opp_plays] == [KNIGHT]        # as it was

    d = live()
    assert d.update([k], 101).opp_plays == [] and d.how_of(1) == "pending"
    assert d.update([at(k, 102)], 102).opp_plays == []
    assert d.update([], 122).opp_plays == [] and d.how_of(1) == "pending"     # unseen 20 ticks
    assert d.update([], 123).opp_plays == []                                    # 21: dropped
    assert d.how_of(1) == "dropped" and d.card_of(1) is None
    assert (d.opp_stats.dropped, d.opp_stats.plays) == (1, 0)
    assert d.opp_stats.summary()["dropped"] == 1

    k2 = tk(2, "Knight", E, 9000, 8000, 200, 201)
    assert d.update([k2], 201).opp_plays == []
    assert d.update([at(k2, 209)], 209).opp_plays == []
    ev = d.update([at(k2, 210, y_mt=8400.0)], 210)        # seen over 10 ticks: reported now ...
    assert [(p.card, p.tick, p.exec_tick, p.x_mt, p.y_mt, p.uids) for p in ev.opp_plays] == [
        (KNIGHT, 200, 200, 9000.0, 8000.0, [2])]           # ... dated at its first sighting
    assert d.how_of(2) == "play" and d.card_of(2) == KNIGHT


def test_a_dropped_flash_is_no_predecessor_for_the_unit_seen_next():
    """A Knight 3000 mt from a flash 25 ticks before is beyond rule 1's reach (1200 + 50 per
    tick): another unit. The flash is neither its play nor the start of its age."""
    k1 = tk(1, "Knight", E, 9000, 8000, 100, 101)
    k2 = tk(2, "Knight", E, 9000, 11000, 126, 127)
    old = deriver()
    old.update([k1], 101)
    old.update([k2], 127)
    assert plays(old) == [(KNIGHT, 100), (KNIGHT, 126)]             # the flash was a play

    d = live()
    d.update([k1], 101)
    d.update([], 125)
    assert d.how_of(1) == "dropped"
    d.update([k2], 127)
    assert d.how_of(2) == "pending"                        # not re-acquired, not continued
    d.update([at(k2, 136)], 136)
    assert plays(d) == [(KNIGHT, 126)] and d.how_of(2) == "play" and d.how_of(1) == "dropped"
    assert (d.opp_stats.dropped, d.opp_stats.fragments) == (1, 0)


def test_one_unit_seen_as_short_fragments_is_one_play_dated_at_its_first_sighting():
    """Review of e2188e5: the age clock restarted on every fragment, so a unit the detector saw
    only in bursts shorter than min_play_age_ticks never became a play (by the same rule two of
    our five Hogs on the friendly would have been lost: tracks of 1-4 ticks). The age is the
    unit's: a young track continues the young one it re-finds by rule 1's reach."""
    f1 = tk(1, "HogRider", E, 14500, 12000, 300, 303, vy=100)
    f2 = tk(2, "HogRider", E, 14500, 13100, 311, 314, vy=100)      # 8 ticks on, 1100 mt on
    f3 = tk(3, "HogRider", E, 14500, 14200, 322, 325, vy=100)
    old = deriver()
    for f in (f1, f2, f3):
        old.update([f], f.last_seen_tick)
    assert plays(old) == [(HOG, 300)] and [old.how_of(u) for u in (2, 3)] == ["reacquired"] * 2

    d = live()
    assert d.update([f1], 303).opp_plays == [] and d.how_of(1) == "pending"
    ev = d.update([f2], 314)                               # 314 - 300 = 14 ticks of the unit
    assert [(p.card, p.tick, p.exec_tick, p.y_mt, p.uids) for p in ev.opp_plays] == [
        (HOG, 300, 300, 12000.0, [1, 2])]                   # dated and placed at its first fragment
    assert (d.how_of(1), d.how_of(2), d.card_of(1)) == ("fragment", "play", HOG)
    d.update([f3], 325)
    assert d.how_of(3) == "reacquired" and len(d.opp_plays()) == 1
    assert (d.opp_stats.plays, d.opp_stats.dropped, d.opp_stats.fragments) == (1, 0, 1)

    # fragments of a unit that never lives 10 ticks in all stay dropped
    d = live()
    d.update([tk(1, "HogRider", E, 14500, 12000, 300, 302, vy=100)], 302)
    d.update([tk(2, "HogRider", E, 14500, 12800, 306, 308, vy=100)], 308)
    d.update([], 330)
    assert plays(d) == [] and (d.how_of(1), d.how_of(2)) == ("dropped", "dropped")
    assert (d.opp_stats.dropped, d.opp_stats.fragments) == (2, 0)


def test_a_play_is_named_by_the_label_its_track_has_once_old_enough():
    """Guards whose track the tracker first labelled Recruit (tick 375 of the friendly: a 7-elixir
    Royal Recruits play from a track labelled SkeletonWarrior in 46 of its 50 frames)."""
    g = tk(1, "Recruit", E, 4000, 9000, 100, 101)
    old = deriver()
    old.update([g], 101)
    assert plays(old) == [(ROYAL_RECRUITS, 100)]

    d = live()
    d.update([g], 101)
    d.update([at(g, 105, cls="SkeletonWarrior")], 105)    # the tracker's vote moved on
    ev = d.update([at(g, 110, cls="SkeletonWarrior")], 110)
    assert [(p.card, p.tick) for p in ev.opp_plays] == [(GUARDS, 100)]


def test_the_units_of_one_play_mature_together_and_count_for_its_card():
    sk = [tk(1 + i, "Skeleton", E, 8000 + 300 * i, 9000, 100, 101) for i in range(3)]
    d = live()
    assert d.update(sk, 101).opp_plays == []
    ev = d.update([at(s, 110) for s in sk], 110)
    assert [(p.card, p.tick, p.uids) for p in ev.opp_plays] == [(SKELETONS, 100, [1, 2, 3])]
    assert [d.how_of(u) for u in (1, 2, 3)] == ["play", "merged", "merged"]
    # a late member joins at once: rule 2 does not wait for the age
    ev = d.update([at(s, 112) for s in sk] + [tk(4, "Skeleton", E, 8600, 9300, 111, 112)], 112)
    assert ev.opp_plays == [] and d.how_of(4) == "merged" and d.card_of(4) == SKELETONS


def test_spells_and_our_own_plays_are_not_held():
    d = live()
    ev = d.update([tk(1, "FireballSpell", E, 9000, 26000, 100, 101)], 101)
    assert [(p.card, p.kind) for p in ev.opp_plays] == [(FIREBALL, "spell")]
    ev = d.update([tk(2, "Musketeer", O, 4000, 24000, 200, 201)], 201)       # ours, no tap
    assert [(x.card, x.slot) for x in ev.own_landings] == [(MUSKETEER, -1)]


def test_an_undecided_team_falls_back_once_while_the_track_waits_for_its_age():
    u = tk(1, "Knight", U, 9000, 8000, 100, 101)
    d = live()
    d.update([u], 101)
    assert d.update([at(u, 105)], 110).opp_plays == []      # fallback (their half); 5 ticks old
    assert d.opp_stats.fallback_team == 1 and d.how_of(1) == "pending"
    ev = d.update([at(u, 111)], 111)
    assert [(p.card, p.tick) for p in ev.opp_plays] == [(KNIGHT, 100)]
    assert d.opp_stats.fallback_team == 1


# ---------------------------------------------------------------------------
# reid_any_label: one unit re-found under another label while it keeps going
# ---------------------------------------------------------------------------
def test_a_lost_unit_comes_back_under_another_label_when_it_keeps_moving_the_same_way():
    """The right lane of the friendly: one enemy unit became HogRider, Knight, LavaHound, ...
    (7 plays) as its track broke and the label flickered."""
    hog = tk(1, "HogRider", E, 14500, 6000, 100, 101, vy=80)
    knight = tk(2, "Knight", E, 14500, 7800, 121, 122, vy=75)    # 1000 mt on, 10 ticks later

    old = deriver()
    old.update([hog], 101)
    old.update([at(hog, 111, y_mt=6800.0)], 111)
    old.update([knight], 122)
    assert plays(old) == [(HOG, 100), (KNIGHT, 121)]

    d = live()
    d.update([hog], 101)
    assert [p.card for p in d.update([at(hog, 111, y_mt=6800.0)], 111).opp_plays] == [HOG]
    assert d.update([knight], 122).opp_plays == []
    assert d.how_of(2) == "reacquired" and d.card_of(2) == HOG
    assert (d.opp_stats.reacquired, d.opp_stats.relabelled) == (1, 1)
    assert [p.uids for p in d.opp_plays()] == [[1, 2]]

    # going back up the lane is another unit: a play once old enough
    d = live()
    d.update([hog], 101)
    d.update([at(hog, 111, y_mt=6800.0)], 111)
    back = replace(knight, vy=-75.0)
    assert d.update([back], 122).opp_plays == [] and d.how_of(2) == "pending"
    d.update([at(back, 131, y_mt=7100.0)], 131)
    assert plays(d) == [(HOG, 100), (KNIGHT, 121)] and d.opp_stats.relabelled == 0


def test_another_label_needs_a_predecessor_with_a_play_and_no_spell():
    d = live()
    d.update([tk(1, "FireballSpell", E, 14500, 7000, 100, 101)], 101)       # a spell's track
    d.update([tk(2, "Knight", E, 14500, 7400, 110, 111)], 111)
    assert d.how_of(2) == "pending"                     # not the spell re-acquired
    d.update([tk(2, "Knight", E, 14500, 7400, 110, 120)], 120)
    assert plays(d) == [(FIREBALL, 100), (KNIGHT, 110)]


def test_a_unit_deployed_where_another_label_stood_or_was_lost_is_a_new_play():
    """Review of e2188e5: a predecessor slower than 40 mt/tick passed the motion check, so a Hog
    dropped where their Knight had just died fighting (standing) was taken for that Knight under
    another label, and the Hog's play was lost. Both must move, the same way."""
    knight = tk(1, "Knight", E, 3500, 13000, 100, 101, vy=50.0)
    fight = at(knight, 111, y_mt=13500.0, vx=0.0, vy=0.0)
    hog = tk(2, "HogRider", E, 3500, 14100, 155, 156)          # 600 mt on, 15 ticks after it died
    seq = [([knight], 101), ([fight], 111), ([at(fight, 140)], 140), ([hog], 156),
           ([at(hog, 166, vy=90.0)], 166)]
    first = EV.EventConfig.live(reid_min_speed_mt_per_tick=0.0)   # nothing standing gets through
    for d in (live(), deriver(first)):
        for tracks, tick in seq:
            d.update(tracks, tick)
        assert plays(d) == [(KNIGHT, 100), (HOG, 155)] and d.opp_stats.relabelled == 0

    # a Knight lost while walking, a Hog deployed by it (standing while it deploys): not the Knight
    walker = tk(1, "Knight", E, 3500, 12000, 100, 101, vy=50.0)
    hog = tk(2, "HogRider", E, 3500, 13300, 121, 122)
    d = live()
    for tracks, tick in [([walker], 101), ([at(walker, 111, y_mt=12500.0)], 111), ([hog], 122),
                         ([at(hog, 131)], 131)]:
        d.update(tracks, tick)
    assert plays(d) == [(KNIGHT, 100), (HOG, 121)] and d.how_of(2) == "play"


# ---------------------------------------------------------------------------
# deploy_zone_mt: a unit first seen past the river walked there, from a reported play
# ---------------------------------------------------------------------------
def _hog_play_then_seen_past_the_river(d, uid, x, t0):
    """Their Hog: a play at their bridge (first seen t0, 11 ticks), lost, and seen again 20 ticks
    later 3500 mt on, past the river (beyond rule 1's 1200 + 50 per tick, within the late reach
    1500 + 120 per tick). Returns the second track's uid."""
    hog = tk(uid, "HogRider", E, x, 13000, t0, t0 + 1, vy=100.0)
    d.update([hog], t0 + 1)
    d.update([at(hog, t0 + 11, y_mt=14000.0)], t0 + 11)
    seen = tk(uid + 1, "HogRider", E, x, 17500, t0 + 31, t0 + 32, vy=100.0)
    d.update([seen], t0 + 32)
    d.update([at(seen, t0 + 42, y_mt=18600.0)], t0 + 42)
    return uid + 1


def test_a_unit_first_seen_past_the_river_is_late_when_a_reported_unit_could_have_walked_there():
    old = deriver()
    _hog_play_then_seen_past_the_river(old, 1, 14500, 300)
    assert plays(old) == [(HOG, 300), (HOG, 331)]            # one Hog, two plays

    d = live()
    assert _hog_play_then_seen_past_the_river(d, 1, 14500, 300) == 2
    assert plays(d) == [(HOG, 300)] and d.how_of(2) == "late" and d.card_of(2) == HOG
    assert d.opp_stats.late == 1 and d.opp_stats.summary()["late"] == 1


def test_a_unit_first_seen_past_the_river_with_no_reported_unit_to_come_from_is_a_play():
    """Review of e2188e5: with the age and zone guards together a real push vanished: its Hog was
    seen 2 frames at the bridge (dropped: a flash) and then past the river (late), and no play was
    reported (the friendly: our left tower lost 2,200 HP to it with no opponent play in 29 s).
    Late needs a reported play's unit that could have walked there; without one the play that put
    the unit there went unseen, and this is it."""
    flash = tk(1, "HogRider", E, 14500, 14000, 600, 601, vy=100.0)
    past = tk(2, "HogRider", E, 14500, 18500, 630, 631, vy=100.0)      # 4500 mt on: no rule 1
    old = deriver()
    old.update([flash], 601)
    old.update([past], 631)
    assert plays(old) == [(HOG, 600), (HOG, 630)]

    d = live()
    d.update([flash], 601)
    d.update([], 622)                                      # unseen 21 ticks: dropped
    d.update([past], 631)
    assert d.how_of(1) == "dropped" and d.how_of(2) == "pending"
    ev = d.update([at(past, 641, y_mt=19500.0)], 641)
    assert [(p.card, p.tick, p.y_mt) for p in ev.opp_plays] == [(HOG, 630, 18500.0)]
    assert d.how_of(2) == "play" and d.opp_stats.late == 0

    # a Hog the detector first sees at our tower (its deploy and run missed) is a play too, and so
    # are its units for a spell the detector missed: a Goblin Barrel's goblins at our tower
    d = live()
    d.update([tk(1, "HogRider", E, 14500, 24000, 640, 641)], 641)
    d.update([tk(1, "HogRider", E, 14500, 24500, 640, 651)], 651)
    assert plays(d) == [(HOG, 640)]
    gob = [tk(10 + i, "Goblin", E, 3000 + 400 * i, 24000, 700, 701) for i in range(3)]
    d.update(gob, 701)
    ev = d.update([at(g, 711) for g in gob], 711)
    assert [(p.card, p.tick, p.uids) for p in ev.opp_plays] == [(GOBLINS, 700, [10, 11, 12])]


def test_a_fallen_princess_tower_opens_its_lane_and_a_miner_or_a_spell_goes_anywhere():
    d = live()
    d.update([], 100, towers={"own_right": VT.TowerHp(0, 3052), "own_left": VT.TowerHp(2000, 3052)})
    assert d.fallen_towers == {"own_right"}
    assert _hog_play_then_seen_past_the_river(d, 1, 3500, 200) == 2       # the left lane: late
    assert plays(d) == [(HOG, 200)] and d.how_of(2) == "late"
    assert _hog_play_then_seen_past_the_river(d, 3, 14500, 400) == 4      # the right lane: a pocket
    assert plays(d) == [(HOG, 200), (HOG, 400), (HOG, 431)] and d.how_of(4) == "play"
    # a Miner goes anywhere; a spell is never held or judged by where it lands
    miner = tk(5, "Miner", E, 3500, 25000, 600, 601)
    d.update([miner], 601)
    d.update([at(miner, 611)], 611)
    assert plays(d)[-1] == (MINER, 600)
    ev = d.update([tk(6, "FireballSpell", E, 3500, 25000, 700, 701)], 701)
    assert [p.card for p in ev.opp_plays] == [FIREBALL]
    # the river's far edge (1000 past the centre) is still in the zone
    edge = tk(7, "Knight", E, 9000, 17000, 800, 801)
    d.update([edge], 801)
    d.update([at(edge, 811)], 811)
    assert plays(d)[-1] == (KNIGHT, 800) and d.opp_stats.late == 1


def test_our_unit_re_found_in_the_enemy_half_is_late_and_one_with_nothing_to_come_from_is_a_play():
    d = live()
    ours = tk(1, "HogRider", O, 4000, 18000, 300, 301)               # our Hog at our bridge, no tap
    assert [(x.card, x.slot) for x in d.update([ours], 301).own_landings] == [(HOG, -1)]
    d.update([at(ours, 320, y_mt=16500.0)], 320)
    again = tk(2, "HogRider", O, 4000, 14000, 340, 341)              # 20 ticks on, 2500 mt on
    assert d.update([again], 341).own_landings == []
    assert d.how_of(2) == "late" and d.card_of(2) == HOG and d.landing_stats.late == 1
    assert d.landing_stats.summary()["late"] == 1 and d.opp_stats.late == 0
    lone = tk(3, "HogRider", O, 14000, 9000, 500, 501)               # nothing of ours near: a play
    assert [(x.card, x.slot) for x in d.update([lone], 501).own_landings] == [(HOG, -1)]
    # their left princess tower down: a pocket for us there
    d.update([], 600, towers={"enemy_left": {"hp": 0, "max": 3052}})
    mine = tk(4, "Musketeer", O, 4000, 18000, 700, 701)
    d.update([mine], 701)
    d.update([at(mine, 720, y_mt=16500.0)], 720)
    ev = d.update([tk(5, "Musketeer", O, 4000, 14000, 740, 741)], 741)
    assert [(x.card, x.slot) for x in ev.own_landings] == [(MUSKETEER, -1)] and d.how_of(5) == "play"


def test_unknown_classes_the_guards_hold_back_still_count_as_unknown():
    """OppStats.unknown (the new-card count) went blind to tracks the guards dropped or called
    late: the class no card puts down is counted whatever the guards did with the track."""
    d = live()
    d.update([tk(1, "Witchling", E, 9000, 8000, 100, 101)], 101)
    d.update([], 123)
    assert d.how_of(1) == "dropped" and d.opp_stats.unknown == {"Witchling": 1}
    hog = tk(2, "HogRider", E, 14500, 13000, 300, 301, vy=100.0)
    d.update([hog], 301)
    d.update([at(hog, 311, y_mt=14000.0)], 311)
    odd = tk(3, "Witchling", E, 14500, 17500, 331, 332)
    d.update([odd], 332)
    d.update([at(odd, 342)], 342)
    assert d.how_of(3) == "late" and d.opp_stats.unknown == {"Witchling": 2} and plays(d) == [(HOG, 300)]


def test_revive_towers_closes_a_pocket_the_hud_misread():
    """HudReader's overtime check revives towers the HUD misread as destroyed (a tower skin):
    their pockets close again, so a unit past the river in that lane is late, not a play."""
    d = live()
    d.update([], 100, towers={"own_right": VT.TowerHp(0, 3052)})
    assert d.fallen_towers == {"own_right"}
    assert d.revive_towers(["own_right", "enemy_left"]) == set() and d.revive_towers([]) == set()
    d.update([], 150, towers={"own_right": VT.TowerHp(671, 3052)})
    assert d.fallen_towers == set()
    assert _hog_play_then_seen_past_the_river(d, 3, 14500, 400) == 4      # the right lane: shut
    assert d.how_of(4) == "late" and plays(d) == [(HOG, 400)]


def test_note_towers_reads_hud_towers_dicts_and_numbers_and_forgets_them_on_reset():
    d = live()
    assert d.note_towers({"own_left": VT.TowerHp(None, 3052), "own_right": 5, "enemy_left": True,
                          "own_king": 0}) == set()
    assert d.note_towers({"own_left": {"hp": 0}, "enemy_right": 0}) == {"own_left", "enemy_right"}
    assert d.note_towers({"own_left": VT.TowerHp(3052, 3052)}) == {"own_left", "enemy_right"}   # stays down
    d.reset()
    assert d.fallen_towers == set()


# ---------------------------------------------------------------------------
# the flags the data did not support (off live)
# ---------------------------------------------------------------------------
class TeamTracker:
    """What the deriver asks a tracker: birth positions (none here) and inherit_team."""

    def __init__(self):
        self.inherited = []

    def state(self, uid):
        return None

    def inherit_team(self, uid, team):
        self.inherited.append((uid, team))
        return True


def _our_hog_then_an_enemy_seen_one(d, tracker=None):
    ours = tk(1, "HogRider", O, 14500, 18000, 100, 101)
    d.update([ours], 101, tracker=tracker)
    d.update([at(ours, 140, y_mt=7500.0)], 140, tracker=tracker)          # at their tower; lost
    theirs = tk(2, "HogRider", E, 14500, 7000, 150, 151)
    d.update([theirs], 151, tracker=tracker)
    d.update([at(theirs, 160)], 160, tracker=tracker)


def test_own_reid_cross_team_keeps_our_unit_ours_and_tells_the_tracker():
    d = live()
    _our_hog_then_an_enemy_seen_one(d)
    assert plays(d) == [(HOG, 150)]                       # live: off, so a play
    trk = TeamTracker()
    d = live(own_reid_cross_team=True)
    _our_hog_then_an_enemy_seen_one(d, trk)
    assert plays(d) == [] and d.how_of(2) == "reacquired" and d.card_of(2) == HOG
    assert trk.inherited == [(2, O)] and d.opp_stats.cross_team == 1 and d.opp_stats.reacquired == 0
    assert [(x.card, x.uid) for x in d.own_landings()] == [(HOG, 1)]


def test_own_recent_takes_an_enemy_seen_class_of_ours_near_our_unit_as_ours():
    def run(d):
        ours = tk(1, "Skeleton", O, 8000, 20000, 100, 101)
        d.update([ours], 101)
        d.update([at(ours, 140, y_mt=9000.0)], 140)
        theirs = tk(2, "Skeleton", E, 8000, 7000, 190, 191)                 # 50 ticks on, 2000 mt
        d.update([theirs], 191)
        d.update([at(theirs, 200)], 200)
        return d

    assert plays(run(live())) == [(SKELETONS, 190)]
    d = run(live(own_recent_ticks=60))
    assert plays(d) == [] and d.how_of(2) == "reacquired" and d.opp_stats.cross_team == 1
    assert plays(run(live(own_recent_ticks=40))) == [(SKELETONS, 190)]       # ours seen 50 before


def test_the_guards_off_reproduce_the_rules_as_they_were():
    """The same tracks through EventConfig() and through every guard switched off by hand."""
    seq = [([tk(1, "Knight", E, 9000, 8000, 100, 101)], 101),
           ([tk(2, "HogRider", E, 14500, 20000, 110, 111, vy=90)], 111),
           ([tk(3, "Recruit", E, 4000, 9000, 120, 121)], 121),
           ([tk(3, "SkeletonWarrior", E, 4000, 9000, 120, 130)], 130)]
    off = EV.EventConfig.live(min_play_age_ticks=0, reid_any_label=False, deploy_zone_mt=None)
    a, b = deriver(), deriver(off)
    for tracks, tick in seq:
        assert a.update(tracks, tick) == b.update(tracks, tick)
    assert plays(a) == [(KNIGHT, 100), (HOG, 110), (ROYAL_RECRUITS, 120)]
    assert a.opp_stats == b.opp_stats


# ---------------------------------------------------------------------------
# Tracker.inherit_team
# ---------------------------------------------------------------------------
def test_inherit_team_pins_a_track_until_twice_flip_llr_of_evidence_against_it():
    trk = T.Tracker(own_classes={"HogRider"})
    o = T.Observation("HogRider", 9000.0, 8000.0, side_prob=0.9)
    trk.update([o], 100)
    [t] = trk.update([o], 101)
    assert t.team == E                                   # born on their half, enemy-side twice
    assert trk.inherit_team(t.uid, O) is True
    assert trk.state(t.uid).cues["inherit"] < 0 and trk.state(t.uid).team_llr == -6.0
    assert [trk.update([o], 102 + i)[0].team for i in range(6)] == [O] * 5 + [E]
    assert trk.inherit_team(999, O) is False and trk.inherit_team(t.uid, U) is False
    # the deck rule still comes first: a label our deck cannot put down stays the enemy's
    k = T.Observation("Knight", 3000.0, 8000.0, side_prob=0.1)
    trk.update([o, k], 108)
    [kt] = [x for x in trk.update([o, k], 109) if x.cls == "Knight"]
    assert trk.inherit_team(kt.uid, O) is True and trk.team_of(trk.state(kt.uid)) == E


# ---------------------------------------------------------------------------
# public_state.OpponentElixir: the unaffordable count (the phantom-play diagnostic)
# ---------------------------------------------------------------------------
def test_opponent_elixir_counts_plays_the_bar_could_not_have_paid():
    pytest.importorskip("numpy")
    PS = pytest.importorskip("tools.public_state")
    law = PS.rs_elixir.ElixirLaw.constants()
    rows = (("Cannon", 3, "BUILDING"), ("Fireball", 4, "SPELL"), ("HogRider", 4, "TROOP"),
            ("IceGolemite", 2, "TROOP"), ("IceSpirits", 1, "TROOP"), ("Musketeer", 4, "TROOP"),
            ("Skeletons", 1, "TROOP"), ("Log", 2, "SPELL"), ("Knight", 3, "TROOP"),
            ("Archer", 3, "TROOP"), ("Giant", 5, "TROOP"), ("Arrows", 3, "SPELL"))
    catalogue = [[n, 0, cost, 1, 0, False, 0, 0, kind] for n, cost, kind in rows]
    giant = 10
    oe = PS.OpponentElixir(catalogue, law=law)
    oe.record([VT.OppPlay(card=giant, tick=10, exec_tick=10)])    # 6 elixir at the start: paid
    assert oe.view(15)[1] is True and oe.unaffordable == 0
    oe.record([VT.OppPlay(card=giant, tick=20, exec_tick=20)])    # ~1.4 left: a phantom
    milli, exact = oe.view(30)
    assert exact is False and oe.unaffordable == 1 and milli >= 0
    assert oe.summary() == {"recorded": 2, "unaffordable": 1, "unpriced": 0, "unknown": 0,
                            "waiting": 0, "exact": False}
    oe.record([VT.OppPlay(card=giant, tick=900, exec_tick=900)])
    assert oe.summary()["waiting"] == 1


# ---------------------------------------------------------------------------
# the runners: guards on by default, --legacy-opp-plays turns them off
# ---------------------------------------------------------------------------
def test_vision_play_event_config_turns_the_guards_on_unless_legacy():
    pytest.importorskip("numpy")
    from scripts import vision_play as VP

    parse = VP.build_parser().parse_args
    assert parse([]).legacy_opp_plays is False
    assert VP.event_config(parse([])) == EV.EventConfig.live()
    assert VP.event_config(parse(["--unknown-plays", "--unknown-cost", "2"])) == EV.EventConfig.live(
        unknown_plays=True, unknown_cost=2)
    assert VP.event_config(parse(["--legacy-opp-plays"])) is None
    assert VP.event_config(parse(["--legacy-opp-plays", "--unknown-plays"])) == EV.EventConfig(unknown_plays=True)
    with pytest.raises(ValueError, match="applies to --unknown-plays"):
        VP.event_config(parse(["--legacy-opp-plays", "--unknown-cost", "2"]))
    assert VP.opp_guard_line(EV.EventConfig.live()) == (
        "guards on: an enemy unit is a play once seen for 10 ticks; a lost one may come back under "
        "another label; no play from a unit first seen over 1000 mt past the river centre where a "
        "reported one could have walked")
    assert VP.opp_guard_line(None).startswith("legacy rules (--legacy-opp-plays)")
    assert VP.opp_guard_line(EV.EventConfig(own_recent_ticks=60)) == "guards on: our classes re-found as ours"


def test_vision_replay_reports_its_guards(tmp_path, monkeypatch, capsys):
    pytest.importorskip("numpy")
    from tests.pipeline import test_latency_and_replay as TLR

    assert TLR.VR.build_parser().parse_args(["rec", "--legacy-opp-plays"]).legacy_opp_plays is True
    rc, _, s = TLR._replay_main(tmp_path, monkeypatch, "guarded")
    assert rc == 0 and s["opp_guards"] == EV.guard_settings(EV.EventConfig.live())
    assert "  opp plays  guards on: an enemy unit is a play once seen for 10 ticks" in capsys.readouterr().out
    rc, _, s = TLR._replay_main(tmp_path, monkeypatch, "legacy", "--legacy-opp-plays")
    assert rc == 0 and s["opp_guards"]["on"] is False and s["opp_guards"]["deploy_zone_mt"] is None
    assert "  opp plays  legacy rules (--legacy-opp-plays)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the runners' wiring: the HUD towers reach the deriver; what the guards did is reported
# ---------------------------------------------------------------------------
class TowerHud:
    """A HUD reader whose every battle frame reads our left princess tower destroyed."""

    def read(self, rgb, t):
        return VT.Hud(hand=[7, 2, 6, 4], next=0, elixir_milli=5000, clock_tick=100,
                      phase=VT.PHASE_REGULATION,
                      towers={"own_left": VT.TowerHp(0, 3052), "enemy_right": VT.TowerHp(1200, 3052)})

    def reset(self, **kw):
        pass


def test_the_runners_pass_the_hud_towers_to_the_deriver():
    """A typo in LivePerception.perceive / ReplayPerception.step's towers= would leave every pocket
    shut, so a real pocket play past the river would be a late sighting."""
    pytest.importorskip("numpy")
    from scripts import vision_play as VP
    from scripts.vision.capture import Frame
    from scripts.vision.labels import load_class_table
    from tests.pipeline import test_latency_and_replay as TLR
    from tests.pipeline import test_vision_play as TVP

    d = EV.EventDeriver(tables(), EV.EventConfig.live(), own_deck=HOG26)
    size = (225, 400)
    p = VP.LivePerception(hud_reader=TowerHud(), tracker=T.Tracker(own_classes={"HogRider"}), deriver=d,
                          geometry=TVP.GEOMETRY, detector=TVP.FakeDetector([], size))
    vf = p.perceive(Frame(2.0, TVP.battle_image(size), 0.0))
    assert vf.screen == VT.SCREEN_BATTLE and p.detect_errors == 0 and d.fallen_towers == {"own_left"}

    d = EV.EventDeriver(tables(), EV.EventConfig.live(), own_deck=HOG26)
    r = TLR.VR.ReplayPerception(hud_reader=TowerHud(), tracker=T.Tracker(own_classes={"HogRider"}), deriver=d,
                                geometry=TLR.GEOMETRY, names=TLR.VR.Names(load_class_table()), recorded=True)
    recorded = VT.VisionFrame(t_capture=2.0, tick=100, detections=[], frame_size=(450, 800),
                              screen=VT.SCREEN_BATTLE)
    step = r.step(TLR.battle_image(0), 2.0, recorded=recorded)
    assert step.det_source == "recorded" and d.fallen_towers == {"own_left"}


class TwoPhaseTowerHud(TowerHud):
    """TowerHud read in two phases (HudReader's surface): the towers come from read_towers,
    which records the evidence it got, and its report revives enemy_left on the first frame."""

    def __init__(self):
        self.evidence = []
        self.report = SimpleNamespace(towers_revived=["enemy_left"])

    def read(self, rgb, t, towers=True):
        hud = super().read(rgb, t)
        if not towers:
            hud.towers, hud.crowns = {}, (0, 0)
        return hud

    def read_towers(self, rgb, t, hud, evidence=None):
        self.evidence.append(evidence)
        hud.towers = super().read(rgb, t).towers
        return hud


def test_the_runners_read_the_towers_after_the_detector():
    """LivePerception and ReplayPerception (--recorded) call read_towers with the frame's tower
    boxes (the recording's, scaled to the image), and take the towers it revived out of the
    deriver's fallen towers before the deriver reads the HUD's."""
    pytest.importorskip("numpy")
    from scripts import vision_play as VP
    from scripts.vision.capture import Frame
    from scripts.vision.labels import load_class_table
    from scripts.vision.tower_labels import LABEL_REF_PX
    from tests.pipeline import test_latency_and_replay as TLR
    from tests.pipeline import test_vision_play as TVP

    def label(size):
        x0, y0, x1, y1 = LABEL_REF_PX["enemy_left"]
        sx, sy = size[0] / 1080.0, size[1] / 1920.0
        return VT.Detection("tower-bar", 0.9, 0.9, (x0 * sx, y0 * sy, x1 * sx, y1 * sy))

    size = (225, 400)
    hud = TwoPhaseTowerHud()
    d = EV.EventDeriver(tables(), EV.EventConfig.live(), own_deck=HOG26)
    d.fallen_towers.add("enemy_left")
    p = VP.LivePerception(hud_reader=hud, tracker=T.Tracker(own_classes={"HogRider"}), deriver=d,
                          geometry=TVP.GEOMETRY, detector=TVP.FakeDetector([label(size)], size))
    vf = p.perceive(Frame(2.0, TVP.battle_image(size), 0.0))
    assert vf.hud.towers["own_left"] == VT.TowerHp(0, 3052) and d.fallen_towers == {"own_left"}
    assert len(hud.evidence) == 1 and hud.evidence[0].seen("enemy_left") and p.detect_errors == 0

    hud = TwoPhaseTowerHud()
    d = EV.EventDeriver(tables(), EV.EventConfig.live(), own_deck=HOG26)
    r = TLR.VR.ReplayPerception(hud_reader=hud, tracker=T.Tracker(own_classes={"HogRider"}), deriver=d,
                                geometry=TLR.GEOMETRY, names=TLR.VR.Names(load_class_table()), recorded=True)
    recorded = VT.VisionFrame(t_capture=2.0, tick=100, detections=[label((450, 800))], frame_size=(450, 800),
                              screen=VT.SCREEN_BATTLE)
    step = r.step(TLR.battle_image(0), 2.0, recorded=recorded)
    assert step.det_source == "recorded" and d.fallen_towers == {"own_left"}
    assert len(hud.evidence) == 1 and hud.evidence[0].labels.keys() == {"enemy_left"}


def test_battle_end_logs_the_deriver_and_elixir_counts():
    pytest.importorskip("numpy")
    from scripts import vision_play as VP

    d = live()
    d.update([tk(1, "Knight", E, 9000, 8000, 100, 101)], 101)
    d.update([], 123)
    errors = []
    me = SimpleNamespace(perception=SimpleNamespace(deriver=d), _error=lambda *a: errors.append(a))
    ctx = SimpleNamespace(observer=SimpleNamespace(opp=SimpleNamespace(summary=lambda: {"unaffordable": 2})))
    out = VP.VisionPlay._opp_diagnostics(me, ctx)
    assert out == {"opp_events": d.opp_stats.summary(), "opp_elixir": {"unaffordable": 2}} and errors == []
    assert out["opp_events"]["dropped"] == 1 and out["opp_events"]["fragments"] == 0
    assert VP.VisionPlay._opp_diagnostics(SimpleNamespace(perception=SimpleNamespace()), None) == {}


def test_vision_replay_sums_how_the_enemy_tracks_were_explained(tmp_path):
    """Replay.summary's events carry the deriver's OppStats over the battles ("opp_tracks"), and
    the printed summary says what the guards held back, so a replay shows what they removed."""
    pytest.importorskip("numpy")
    from scripts.vision.capture import FileReplaySource
    from scripts.vision.labels import load_class_table
    from tests.pipeline import test_latency_and_replay as TLR

    VR = TLR.VR
    session = TLR.write_recording(tmp_path / "rec")
    names = VR.Names(load_class_table())
    deriver = EV.EventDeriver(EV.EventTables({2: {"HogRider": 1}, 8: {"Knight": 1}}), EV.EventConfig.live(),
                              own_deck=range(8))
    perception = VR.ReplayPerception(hud_reader=TLR.ReplayHud(), tracker=T.Tracker(own_classes={"HogRider"}),
                                     deriver=deriver, geometry=TLR.GEOMETRY, detector=TLR.ReplayDetector(),
                                     names=names)
    taps, _ = VR.read_recording_records(session)
    s = VR.Replay(FileReplaySource(session), perception, taps=taps, names=names,
                  card_ids={n: i for i, n in TLR.NAMES.items()}, card_names=TLR.NAMES,
                  live_names=TLR.NAMES).run()
    e = s["events"]
    assert (e["opp_plays"], e["opp_cards"], e["own_landed"]) == (1, {"Knight": 1}, 1)   # the Knight, ten ticks on
    ot = e["opp_tracks"]
    assert set(ot) == set(EV.OppStats().summary()) - {"unknown"}
    assert (ot["plays"], ot["dropped"], ot["fragments"], ot["late"], ot["relabelled"]) == (1, 0, 0, 0, 0)
    assert any(line.startswith("  enemy tracks: 1 plays (0 spells), merged 0, children 0, re-found 0 "
                               "(0 under another label); held back: young dropped 0, fragments 0, late 0")
               for line in VR.format_summary(s))

