"""V04: opponent plays and own landings from tracks (scripts/vision/events.py).

Synthetic tracks in world millitiles (we are team 1 at high y, the opponent team 0),
battle ticks, no detector and no engine. Card ids follow conftest's DEFAULT_CATALOGUE
for Hog 2.6 (0-7) and Knight / Zap; the other cards get ids of their own.
"""
from __future__ import annotations

import ast
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import control as CT  # noqa: E402
from scripts.vision import events as EV  # noqa: E402
from scripts.vision import tracker as T  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from tests.pipeline.conftest import (DEFAULT_CATALOGUE, DEFAULT_UNIT_TYPES, ENTITY_FIELDS,  # noqa: E402
                                     entity_row, state_dict)

CANNON, FIREBALL, HOG, ICE_GOLEM, ICE_SPIRITS, MUSKETEER, SKELETONS, LOG = range(8)
KNIGHT, ZAP = 8, 15
WITCH, SKELETON_ARMY, GOLEM, GOBLINS, GOBLIN_GANG, SPEAR_GOBLINS = 20, 21, 22, 23, 24, 25
HOG26 = range(8)

E, O, U = VT.ENEMY_TEAM, VT.OWN_TEAM, VT.UNKNOWN_TEAM


def tables() -> EV.EventTables:
    return EV.EventTables(
        compositions={CANNON: {"Cannon": 1}, HOG: {"HogRider": 1}, ICE_GOLEM: {"IceGolemite": 1},
                      ICE_SPIRITS: {"IceSpirits": 1}, MUSKETEER: {"Musketeer": 1},
                      SKELETONS: {"Skeleton": 3}, KNIGHT: {"Knight": 1}, WITCH: {"Witch": 1},
                      SKELETON_ARMY: {"Skeleton": 15}, GOLEM: {"Golem": 1}, GOBLINS: {"Goblin": 4},
                      GOBLIN_GANG: {"Goblin": 3, "SpearGoblin": 3}, SPEAR_GOBLINS: {"SpearGoblin": 3}},
        spells={FIREBALL: {"FireballSpell"}, LOG: {"LogProjectile", "LogProjectileRolling"},
                ZAP: {"Zap"}},
        children={"Witch": {"Skeleton"}, "Golem": {"Golemite"},
                  "LogProjectile": {"LogProjectileRolling"}},
        ignore={"TowerPrincess"})


def tk(uid, cls, team, x, y, first, last, hp=None) -> VT.Track:
    return VT.Track(uid=uid, cls=cls, team=team, conf=0.9, x_mt=float(x), y_mt=float(y), hp_frac=hp,
                    first_seen_tick=first, last_seen_tick=last)


def seen(tracks, last):
    return [replace(t, last_seen_tick=last) for t in tracks]


def deriver(**kw) -> EV.EventDeriver:
    config = kw.pop("config", None)
    d = EV.EventDeriver(tables(), config, **kw)
    d.update([], 0)            # the battle starts on an empty board
    return d


# ---------------------------------------------------------------------------
# opponent plays: the four cases of the plan
# ---------------------------------------------------------------------------
def test_witch_spawning_skeletons_is_one_play():
    d = deriver()
    ev = d.update([tk(1, "Witch", E, 9000, 8000, 100, 101)], 101)
    assert [p.to_dict() for p in ev.opp_plays] == [
        {"card": WITCH, "tick": 100, "exec_tick": 100, "x_mt": 9000.0, "y_mt": 8000.0,
         "kind": "unit", "uids": [1]}]
    wave1 = [tk(2, "Skeleton", E, 7000, 8000, 120, 121), tk(3, "Skeleton", E, 11000, 8000, 120, 121),
             tk(4, "Skeleton", E, 9000, 6000, 120, 121), tk(5, "Skeleton", E, 9000, 10000, 120, 121)]
    ev = d.update([tk(1, "Witch", E, 9000, 8200, 100, 121)] + wave1, 121)
    assert ev.opp_plays == []
    # 7 s later, far outside PLAY_WINDOW: the living Witch still explains them
    wave2 = [tk(6 + i, "Skeleton", E, x, y, 260, 261)
             for i, (x, y) in enumerate([(7000, 9000), (11000, 9000), (9000, 7000), (9000, 11000)])]
    ev = d.update([tk(1, "Witch", E, 9000, 9000, 100, 261)] + wave2, 261)
    assert ev.opp_plays == []
    plays = d.opp_plays()
    assert [p.card for p in plays] == [WITCH]
    assert plays[0].uids == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # card-level identity (--live-compat): a Witch's skeleton is the Witch's
    assert [d.card_of(u) for u in range(1, 10)] == [WITCH] * 9
    assert (d.opp_stats.plays, d.opp_stats.children, d.opp_stats.merged) == (1, 8, 0)


def test_skeletons_three_units_are_one_play():
    d = deriver()
    sk = [tk(11, "Skeleton", E, 8500, 9000, 200, 201), tk(12, "Skeleton", E, 9000, 9300, 200, 201),
          tk(13, "Skeleton", E, 9500, 9000, 200, 201)]
    # Skeletons or Skeleton Army? the play waits ambiguity_wait_ticks for the rest
    assert d.update(sk, 201).opp_plays == []
    assert d.card_of(11) is None and d.card_for(sk[0]) == SKELETONS     # the guess meanwhile
    assert d.update(seen(sk, 205), 205).opp_plays == []
    ev = d.update(seen(sk, 206), 206)
    assert [p.to_dict() for p in ev.opp_plays] == [
        {"card": SKELETONS, "tick": 200, "exec_tick": 200, "x_mt": 8500.0, "y_mt": 9000.0,
         "kind": "unit", "uids": [11, 12, 13]}]
    assert d.update(seen(sk, 230), 230).opp_plays == []
    assert [d.card_of(u) for u in (11, 12, 13)] == [SKELETONS] * 3
    assert (d.opp_stats.plays, d.opp_stats.merged, d.opp_stats.ambiguous) == (1, 2, 1)


def test_skeleton_army_is_told_from_skeletons_by_the_count():
    d = deriver()
    pts = [(8000 + 500 * i, 8500 + 500 * j) for j in range(3) for i in range(5)]
    first10 = [tk(100 + k, "Skeleton", E, x, y, 300, 301) for k, (x, y) in enumerate(pts[:10])]
    late5 = [tk(110 + k, "Skeleton", E, x, y, 302, 303) for k, (x, y) in enumerate(pts[10:])]
    assert d.update(first10, 301).opp_plays == []
    assert d.update(seen(first10, 303) + late5, 303).opp_plays == []
    ev = d.update(seen(first10 + late5, 306), 306)
    assert [(p.card, p.tick, p.exec_tick, p.uids) for p in ev.opp_plays] == [
        (SKELETON_ARMY, 300, 300, list(range(100, 115)))]


def test_log_and_its_roller_are_one_play():
    d = deriver()
    ev = d.update([tk(21, "LogProjectile", E, 3500, 20000, 300, 301)], 301)
    assert [(p.card, p.kind, p.tick, p.exec_tick) for p in ev.opp_plays] == [(LOG, "spell", 300, 300)]
    assert d.update([tk(21, "LogProjectile", E, 3500, 21000, 300, 305)], 305).opp_plays == []
    # the roller, under another class and uid
    assert d.update([tk(22, "LogProjectileRolling", E, 3500, 23000, 311, 312)], 312).opp_plays == []
    assert d.update([tk(22, "LogProjectileRolling", E, 3500, 24500, 311, 318)], 318).opp_plays == []
    # lost and found further down: 29 ticks after the first sighting, but 11 after the
    # last one; a spell's window chains from its last sighting
    assert d.update([tk(23, "LogProjectileRolling", E, 3500, 27000, 329, 330)], 330).opp_plays == []
    assert [(p.card, p.tick, p.uids) for p in d.opp_plays()] == [(LOG, 300, [21, 22, 23])]
    assert (d.opp_stats.plays, d.opp_stats.spells, d.opp_stats.merged) == (1, 1, 2)
    # a Log seen 70 ticks after the last sighting is another play
    ev = d.update([tk(24, "LogProjectile", E, 14500, 20000, 400, 401)], 401)
    assert [(p.card, p.tick) for p in ev.opp_plays] == [(LOG, 400)]


def test_a_spell_seen_in_two_flashes_25_ticks_apart_is_one_play():
    """An enemy Log lost mid-roll for 25 ticks and found 3,500 mt further on (a rolling Log moves
    135-175 mt a tick on the owner's 2026-10-29 friendly): rule 1 reaches 1,200 + 50 x 25 = 2,450 mt,
    so it is the spell window from the last sighting (spell_window_ticks, SPELL_WINDOW = 50) that
    keeps it one play: 3,500 mt is within spell_reach_mt 3,000 + the Log's own 100 mt a tick x 25.
    A 20-tick window made it two, and each one is charged in full."""
    assert EV.SPELL_WINDOW == 50
    assert EV.EventConfig().spell_window_ticks == EV.EventConfig.live().spell_window_ticks == 50

    def run(config):
        d = deriver(config=config)
        d.update([tk(31, "LogProjectileRolling", E, 3500, 20000, 300, 301)], 301)
        d.update([tk(31, "LogProjectileRolling", E, 3500, 20300, 300, 303)], 303)
        d.update([], 320)
        d.update([tk(32, "LogProjectileRolling", E, 3500, 23800, 328, 329)], 329)
        return d

    for config in (None, EV.EventConfig.live()):
        d = run(config)
        assert [(p.card, p.tick, p.uids) for p in d.opp_plays()] == [(LOG, 300, [31, 32])]
        assert (d.how_of(32), d.opp_stats.spells, d.opp_stats.reacquired) == ("merged", 1, 0)
    d = run(EV.EventConfig(spell_window_ticks=20))
    assert [(p.card, p.tick) for p in d.opp_plays()] == [(LOG, 300), (LOG, 328)]
    with pytest.raises(ValueError):
        EV.EventConfig(spell_window_ticks=-1)
    # units keep PLAY_WINDOW from the play's first sighting: a Knight 25 ticks on, 4,000 mt away
    # (past rule 1's reach, inside merge_radius_mt), is a new play
    d = deriver()
    d.update([tk(41, "Knight", E, 9000, 8000, 100, 101)], 101)
    d.update([tk(42, "Knight", E, 9000, 12000, 126, 127)], 127)
    assert [(p.card, p.tick) for p in d.opp_plays()] == [(KNIGHT, 100), (KNIGHT, 126)]


# ---------------------------------------------------------------------------
# own taps and landings
# ---------------------------------------------------------------------------
def test_lagged_own_landing_is_matched_to_its_tap():
    d = deriver(own_deck=HOG26)
    d.note_own_tap(HOG, 500, slot=2, aim_x_mt=14500, aim_y_mt=17500, t=10.0)
    for tick in (505, 515, 525):              # the client delay: nothing on the board yet
        assert d.update([], tick).own_landings == []
    ev = d.update([tk(31, "HogRider", O, 14400, 17300, 526, 527)], 527, t=11.35)
    assert ev.opp_plays == []
    (landing,) = ev.own_landings
    assert landing.to_dict() == {"card": HOG, "slot": 2, "tap_tick": 500, "land_tick": 526,
                                 "tap_t": 10.0, "aim_x_mt": 14500.0, "aim_y_mt": 17500.0,
                                 "x_mt": 14400.0, "y_mt": 17300.0, "uid": 31}
    assert landing.lag_ticks == 26
    assert d.card_of(31) == HOG
    st = d.landing_stats
    assert (st.taps, st.landed, st.missed, st.unmatched) == (1, 1, 0, 0)
    assert st.lag_ticks == [26]
    assert st.lag_s == [pytest.approx(1.30)]          # first sighting 11.35 - 1 tick, minus 10.0
    assert st.offsets_mt == [(-100.0, -200.0)]
    assert st.sight_lag_estimate(21) == 5
    assert d.pending_taps == []
    # the human-takeover attributor reads it as a bot play
    assert CT.as_own_play(landing) == CT.OwnPlay(HOG, 526, 14400.0, 17300.0, 31, 500)


def test_a_tap_never_seen_is_a_miss():
    d = deriver(own_deck=HOG26)
    d.note_own_tap(FIREBALL, 600, slot=0, aim_x_mt=9000, aim_y_mt=6000, t=20.0)
    assert d.update([], 670).own_landings == []       # 60 ticks + the 10-tick team wait
    (miss,) = d.update([], 671).own_landings
    assert miss.to_dict() == {"card": FIREBALL, "slot": 0, "tap_tick": 600, "land_tick": None,
                              "tap_t": 20.0, "aim_x_mt": 9000.0, "aim_y_mt": 6000.0,
                              "x_mt": None, "y_mt": None, "uid": None}
    assert CT.as_own_play(miss) is None
    assert (d.landing_stats.missed, d.landing_stats.miss_rate) == (1, 1.0)
    assert d.pending_taps == []


def test_own_skeletons_are_one_landing():
    d = deriver(own_deck=HOG26)
    d.note_own_tap(SKELETONS, 700, slot=1, aim_x_mt=9000, aim_y_mt=20000)
    sk = [tk(41, "Skeleton", O, 8500, 20000, 724, 725), tk(42, "Skeleton", O, 9000, 20400, 724, 725),
          tk(43, "Skeleton", O, 9500, 20000, 724, 725)]
    ev = d.update(sk, 725)
    assert [(x.card, x.slot, x.uid, x.lag_ticks) for x in ev.own_landings] == [(SKELETONS, 1, 41, 24)]
    assert [d.card_of(u) for u in (41, 42, 43)] == [SKELETONS] * 3
    assert d.landing_stats.unmatched == 0 and ev.opp_plays == []


def test_own_plays_without_a_tap_are_reported_with_slot_minus_one():
    d = deriver(own_deck=HOG26)
    ev = d.update([tk(51, "Musketeer", O, 4000, 24000, 800, 801)], 801)
    (own,) = ev.own_landings
    assert (own.card, own.slot, own.tap_tick, own.land_tick, own.uid) == (MUSKETEER, -1, 800, 800, 51)
    assert CT.as_own_play(own) == CT.OwnPlay(MUSKETEER, 800, 4000.0, 24000.0, 51, None)
    # our deck settles Skeletons vs Skeleton Army at once
    ev = d.update([tk(52, "Skeleton", O, 14000, 22000, 900, 901)], 901)
    assert [(x.card, x.slot) for x in ev.own_landings] == [(SKELETONS, -1)]
    assert d.landing_stats.unmatched == 2


def test_note_own_tap_tells_the_tracker_what_to_expect():
    class FakeTracker:
        def __init__(self):
            self.taps = []

        def note_own_tap(self, classes, x, y, tick, *, max_units=None):
            self.taps.append((set(classes), x, y, tick, max_units))

    ft = FakeTracker()
    d = deriver()
    d.note_own_tap(SKELETONS, 700, slot=1, aim_x_mt=9000, aim_y_mt=20000, tracker=ft)
    d.note_own_tap(LOG, 710, slot=3, aim_x_mt=9000, aim_y_mt=12000, tracker=ft)
    assert ft.taps == [({"Skeleton"}, 9000.0, 20000.0, 700, 3),
                       ({"LogProjectile", "LogProjectileRolling"}, 9000.0, 12000.0, 710, None)]
    with pytest.raises(ValueError):
        d.note_own_tap(HOG, 720, slot=-1)


# ---------------------------------------------------------------------------
# the other rules
# ---------------------------------------------------------------------------
def test_golemites_of_a_dead_golem_are_not_plays_and_a_lone_one_is_unknown():
    d = deriver()
    assert [p.card for p in d.update([tk(61, "Golem", E, 9000, 12000, 400, 401)], 401).opp_plays] == [GOLEM]
    assert d.update([tk(61, "Golem", E, 9000, 14000, 400, 450, hp=0.05)], 450).opp_plays == []
    ev = d.update([tk(62, "Golemite", E, 8500, 14000, 454, 455),
                   tk(63, "Golemite", E, 9500, 14000, 454, 455)], 455)    # the Golem is gone
    assert ev.opp_plays == []
    assert (d.card_of(62), d.card_of(63)) == (GOLEM, GOLEM)
    ev = d.update([tk(64, "Golemite", E, 3000, 5000, 599, 600)], 600)     # no parent anywhere
    assert ev.opp_plays == [] and d.card_of(64) is None
    assert d.opp_stats.unknown == {"Golemite": 1}
    assert [p.card for p in d.opp_plays()] == [GOLEM]


def test_a_reacquired_track_is_not_a_play_but_a_new_one_after_a_death_is():
    d = deriver()
    d.update([tk(71, "Knight", E, 9000, 10000, 100, 101, hp=0.8)], 101)
    d.update([tk(71, "Knight", E, 9000, 11000, 100, 110, hp=0.8)], 110)
    ev = d.update([tk(72, "Knight", E, 9000, 11800, 125, 126)], 126)      # lost for 15 ticks
    assert ev.opp_plays == [] and d.card_of(72) == KNIGHT
    assert [p.uids for p in d.opp_plays()] == [[71, 72]]
    assert d.opp_stats.reacquired == 1

    d = deriver()
    d.update([tk(81, "Knight", E, 9000, 10000, 100, 101)], 101)
    d.update([tk(81, "Knight", E, 9000, 11000, 100, 110, hp=0.1)], 110)  # lost nearly dead: died
    ev = d.update([tk(82, "Knight", E, 9000, 11800, 125, 126)], 126)
    assert [(p.card, p.tick) for p in ev.opp_plays] == [(KNIGHT, 125)]


def test_repeats_merge_within_play_window_only_and_only_nearby():
    d = deriver()
    k1 = tk(91, "Knight", E, 3500, 9000, 100, 101)
    d.update([k1], 101)
    k2 = tk(92, "Knight", E, 4000, 9000, 114, 115)
    assert d.update([replace(k1, last_seen_tick=115), k2], 115).opp_plays == []    # 14 ticks: merged
    k3 = tk(93, "Knight", E, 3500, 9500, 124, 125)
    ev = d.update(seen([k1, k2], 125) + [k3], 125)                                  # 24 ticks: new
    assert [(p.card, p.tick) for p in ev.opp_plays] == [(KNIGHT, 124)]
    # the same card in the other lane 5 ticks later is another play
    ev = d.update(seen([k1, k2, k3], 130) + [tk(94, "Knight", E, 14500, 9000, 129, 130)], 130)
    assert [(p.card, p.tick) for p in ev.opp_plays] == [(KNIGHT, 129)]


def test_composition_fit_tells_goblin_gang_from_goblins_and_spear_goblins():
    d = deriver()
    gang = ([tk(201 + i, "Goblin", E, 3000 + 400 * i, 9000, 300, 301) for i in range(3)]
            + [tk(204 + i, "SpearGoblin", E, 3000 + 400 * i, 9800, 300, 301) for i in range(3)])
    assert d.update(gang, 301).opp_plays == []
    ev = d.update(seen(gang, 306), 306)
    assert [(p.card, p.uids) for p in ev.opp_plays] == [(GOBLIN_GANG, list(range(201, 207)))]

    d = deriver()
    spears = [tk(301 + i, "SpearGoblin", E, 14000 + 400 * i, 9000, 400, 401) for i in range(3)]
    d.update(spears, 401)
    ev = d.update(seen(spears, 406), 406)
    assert [p.card for p in ev.opp_plays] == [SPEAR_GOBLINS]


def test_undecided_teams_fall_back_after_the_wait():
    d = deriver()
    t1 = tk(101, "Knight", U, 9000, 8000, 100, 101)
    assert d.update([t1], 101).opp_plays == []
    assert d.update([replace(t1, last_seen_tick=109)], 109).opp_plays == []
    ev = d.update([replace(t1, last_seen_tick=110)], 110)
    assert [(p.card, p.tick) for p in ev.opp_plays] == [(KNIGHT, 100)]       # born on their half
    t2 = tk(102, "Musketeer", U, 4000, 24000, 200, 201)                        # born on ours, no tap
    d.update([t2], 201)
    ev = d.update([replace(t2, last_seen_tick=210)], 210)
    assert [(x.card, x.slot) for x in ev.own_landings] == [(MUSKETEER, -1)] and ev.opp_plays == []
    t3 = tk(103, "FireballSpell", U, 9000, 26000, 300, 301)                    # a spell: theirs
    d.update([t3], 301)
    ev = d.update([replace(t3, last_seen_tick=310)], 310)
    assert [(p.card, p.kind) for p in ev.opp_plays] == [(FIREBALL, "spell")]
    assert d.opp_stats.fallback_team == 3


def test_priming_ignored_classes_and_backwards_ticks():
    d = EV.EventDeriver(tables())
    assert d.update([tk(1, "Knight", E, 9000, 8000, 50, 60)], 60).opp_plays == []   # mid-battle start
    assert d.card_of(1) == KNIGHT
    ev = d.update([tk(1, "Knight", E, 9000, 8500, 50, 70), tk(2, "Witch", E, 14500, 9000, 69, 70)], 70)
    assert [p.card for p in ev.opp_plays] == [WITCH]
    ev = d.update([tk(3, "TowerPrincess", E, 3500, 6500, 79, 80)], 80)
    assert ev.opp_plays == [] and d.opp_stats.unknown == {}
    d.update([], 75)
    assert d.backwards_ticks == 1
    d.reset()
    assert d.opp_plays() == [] and d.update([tk(9, "Knight", E, 9000, 8000, 5, 6)], 6).opp_plays == []


def test_how_of_names_the_rule_that_explained_each_track():
    """N05: EventDeriver.how_of(uid), the hook the harvest tools read (rule 6's "unknown_play"
    is in test_events_unknown.py)."""
    p = EV.EventDeriver(tables())
    p.update([tk(1, "Knight", E, 9000, 8000, 50, 60)], 60)
    assert p.how_of(1) == "primed"
    d = deriver(own_deck=HOG26)
    d.update([tk(2, "Knight", U, 9000, 8000, 100, 101)], 101)
    assert d.how_of(2) == "pending" and d.how_of(99) is None
    d.update([tk(2, "Knight", U, 9000, 8000, 100, 110)], 110)          # the fallback: their half
    d.update([tk(3, "Witch", E, 9000, 8000, 200, 201)], 201)
    d.update([tk(3, "Witch", E, 9000, 8000, 200, 221), tk(4, "Skeleton", E, 8000, 8000, 220, 221)], 221)
    d.update([tk(5, "Knight", E, 3500, 10000, 300, 301, hp=0.8)], 301)
    d.update([tk(5, "Knight", E, 3500, 10500, 300, 305, hp=0.8)], 305)
    d.update([tk(6, "Knight", E, 3500, 11000, 315, 316)], 316)          # 5, lost 10 ticks
    d.update([tk(7 + i, "Skeleton", E, 14000 + 400 * i, 9000, 400, 401) for i in range(3)], 401)
    d.note_own_tap(HOG, 500, slot=2, aim_x_mt=14500, aim_y_mt=17500)
    d.update([tk(10, "HogRider", O, 14400, 17300, 526, 527)], 527)
    d.update([tk(11, "TowerPrincess", E, 3500, 6500, 600, 601)], 601)
    d.update([tk(12, "Golemite", E, 3000, 5000, 700, 701)], 701)
    assert {u: d.how_of(u) for u in range(2, 13)} == {
        2: "play", 3: "play", 4: "child", 5: "play", 6: "reacquired", 7: "play", 8: "merged",
        9: "merged", 10: "landing", 11: "ignored", 12: "unknown"}
    assert [p.card for p in d.opp_plays()] == [KNIGHT, WITCH, KNIGHT, SKELETONS]
    d.update([], 1400)                                                   # 600 ticks unseen: forgotten
    assert d.how_of(2) is None and d.how_of(12) is None


# ---------------------------------------------------------------------------
# execution ticks and public memory
# ---------------------------------------------------------------------------
def test_exec_ticks_and_public_memory():
    d = deriver(config=EV.EventConfig(unit_sight_lag_ticks=3, sight_lag_by_card={LOG: 5}))
    plays = (d.update([tk(1, "Witch", E, 9000, 8000, 100, 101)], 101).opp_plays
             + d.update([tk(2, "LogProjectile", E, 3500, 20000, 300, 301)], 301).opp_plays)
    assert [(p.card, p.tick, p.exec_tick) for p in plays] == [(WITCH, 100, 97), (LOG, 300, 295)]

    class Recorder:
        def __init__(self):
            self.calls = []

        def record(self, team, card, exec_tick):
            self.calls.append((team, card, exec_tick))

    rec = Recorder()
    assert EV.record_plays(rec, plays) == 2
    assert rec.calls == [(E, WITCH + 2, 97), (E, LOG + 2, 295)]

    rs_memory = pytest.importorskip("tools.rs_memory")
    mem = rs_memory.PublicMemory()
    EV.record_plays(mem, plays)
    view = mem.view(301, 0)
    assert view["last"][E] == [(WITCH + 2, 97), (LOG + 2, 295)]
    assert view["seen"][E] == {WITCH + 2, LOG + 2}
    assert mem.view(294, 0)["last"][E] == [(WITCH + 2, 97)]


def test_landing_stats_summary():
    st = EV.LandingStats(taps=6, landed=5, missed=1, lag_ticks=[26, 20, 40, 22, 24],
                         offsets_mt=[(300.0, 400.0)])
    s = st.summary()
    assert s["miss_rate"] == pytest.approx(1 / 6)
    assert s["lag_ticks"] == {"n": 5, "mean": 26.4, "median": 24, "p90": 40, "min": 20, "max": 40}
    assert s["lag_s"] is None
    assert s["offset_mt"]["dist"]["median"] == 500.0
    assert st.sight_lag_estimate() == 0                # median 24 < LEAD 26
    assert EV.LandingStats().summary()["miss_rate"] is None


# ---------------------------------------------------------------------------
# with the real tracker, and on engine states
# ---------------------------------------------------------------------------
def test_with_the_real_tracker():
    own = {"Cannon", "FireballSpell", "HogRider", "IceGolemite", "IceSpirits", "Musketeer",
           "Skeleton", "LogProjectile", "LogProjectileRolling"}
    tracker = T.Tracker(own_classes=own, solver="greedy")

    def ob(cls, x, y):
        return T.Observation(cls, float(x), float(y), 0.9, 0.9)     # side_prob 0.9: enemy

    witch = [ob("Witch", 9000, 8000)]
    spawned = witch + [ob("Skeleton", 7000, 8000), ob("Skeleton", 11000, 8000)]
    played = [ob("Skeleton", 3000, 12000), ob("Skeleton", 3500, 12000), ob("Skeleton", 3000, 12500)]
    frames = {100: witch, 101: witch, 102: spawned, 103: spawned}
    frames.update({tick: played for tick in range(110, 117)})
    d = deriver()
    plays = []
    for tick in sorted(frames):
        tracks = tracker.update(frames[tick], tick)
        plays += d.update(tracks, tick, tracker=tracker).opp_plays
    # the Witch, its two skeletons (children), then a Skeletons card at the other bridge
    assert [(p.card, p.tick) for p in plays] == [(WITCH, 100), (SKELETONS, 110)]
    assert [d.card_of(u) for u in (2, 3)] == [WITCH, WITCH]


# spells through the real Tracker (docs/SEASON_UPDATES.md 8.2): the tables' spell classes are the
# tracker's spell_classes; the live chain passes ClassTable.spell_labels()
SPELL_CLASSES = {"FireballSpell", "LogProjectile", "LogProjectileRolling", "Zap"}
OWN_CLASSES = {"Cannon", "FireballSpell", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeleton",
               "LogProjectile", "LogProjectileRolling"}


def tob(cls, x, y, side, conf=0.9):
    return T.Observation(cls, float(x), float(y), conf, side)


def run_tracked(frames, spell_classes, **kw):
    """frames {tick: [Observation]} through a Tracker and a deriver; (tracker, deriver, opp plays)."""
    assert SPELL_CLASSES == set().union(*tables().spells.values())
    tracker = T.Tracker(own_classes=OWN_CLASSES, spell_classes=spell_classes, solver="greedy")
    d = deriver(**kw)
    plays = []
    for tick in sorted(frames):
        plays += d.update(tracker.update(frames[tick], tick), tick, tracker=tracker).opp_plays
    return tracker, d, plays


def test_an_enemy_zap_over_our_unseen_skeleton_is_one_enemy_play():
    """A Zap flash over our Skeleton while the Skeleton's box is missing: it is the one box in the
    Skeleton's gate, so without spell_classes it joined the Skeleton's track and was no play; with
    them it is a track of its own, reported on its first sighting (conf 0.7 >= spell_confirm_conf)."""
    skeleton = [tob("Skeleton", 5000, 24000, 0.1)]
    frames = {tick: skeleton for tick in range(100, 112, 2)}
    frames[112] = [tob("Zap", 5000, 23700, 0.9, conf=0.7)]
    frames.update({tick: skeleton for tick in (116, 118)})
    tracker, d, plays = run_tracked(frames, SPELL_CLASSES, own_deck=HOG26)
    assert [(p.card, p.kind, p.tick, p.uids) for p in plays] == [(ZAP, "spell", 112, [2])]
    skel = tracker.state(1)
    assert (skel.label, tracker.team_of(skel), skel.hits) == ("Skeleton", O, 8)
    assert [(x.card, x.land_tick) for x in d.own_landings()] == [(SKELETONS, 100)]
    tracker, d, plays = run_tracked(frames, None, own_deck=HOG26)     # as before: no Zap play
    assert plays == [] and tracker.state(1).hits == 9 and len(tracker.states()) == 1


def test_our_log_next_to_our_new_cannon_is_a_log_play_and_a_cannon_play():
    """Our Log rolls out of sight and our Cannon lands within its track's gate (the owner's
    2026-10-29 friendly, tick 3599): without spell_classes the Log's track took the Cannon's boxes
    and the Cannon was never our play."""
    frames = {tick: [tob("LogProjectileRolling", 3500, 24000 - 150 * (tick - 200), 0.1, conf=0.8)]
              for tick in (200, 202, 204, 206)}
    frames.update({tick: [tob("Cannon", 4000, 23300, 0.1)] for tick in range(208, 222, 2)})
    tracker, d, plays = run_tracked(frames, SPELL_CLASSES, own_deck=HOG26)
    assert plays == [] and sorted((x.card, x.land_tick, x.slot) for x in d.own_landings()) == [
        (CANNON, 208, -1), (LOG, 200, -1)]
    assert [(s.uid, s.label) for s in tracker.states()] == [(1, "LogProjectileRolling"), (2, "Cannon")]
    tracker, d, plays = run_tracked(frames, None, own_deck=HOG26)
    assert [(x.card, x.land_tick) for x in d.own_landings()] == [(LOG, 200)]
    assert [(s.uid, s.label) for s in tracker.states()] == [(1, "Cannon")]


def test_an_enemy_log_in_two_flashes_is_one_play_with_the_real_tracker():
    """The two flashes 25 ticks and 3,750 mt apart are two tracks (the first expired, and it is far
    past the gate); the spell window keeps them one play, a 20-tick window made them two."""
    frames = {tick: [tob("LogProjectileRolling", 3500, 20000 + 150 * (tick - 300), 0.9, conf=0.8)]
              for tick in (300, 302, 304, 329, 331)}
    tracker, d, plays = run_tracked(frames, SPELL_CLASSES)
    assert [(p.card, p.tick, p.kind) for p in d.opp_plays()] == [(LOG, 300, "spell")]
    assert d.opp_plays()[0].uids == [1, 2] and d.how_of(2) == "merged"
    tracker, d, plays = run_tracked(frames, SPELL_CLASSES, config=EV.EventConfig(spell_window_ticks=20))
    assert [(p.card, p.tick) for p in d.opp_plays()] == [(LOG, 300), (LOG, 329)]


def test_one_stray_box_of_our_own_log_is_no_enemy_play():
    """One box of our Log at the river (side 0.1, no tap pending, dry run): its team is undecided at
    birth, so the tracker waits for a second sighting, as for a unit, and it dies. Reported at once,
    the deriver's fallback (an undecided spell is the enemy's) made it a 2-elixir enemy Log."""
    frames = {100: [tob("LogProjectileRolling", 9000, 16300, 0.1, conf=0.8)]}
    frames.update({tick: [] for tick in range(102, 140, 2)})
    for config in (None, EV.EventConfig.live()):
        tracker, d, plays = run_tracked(frames, SPELL_CLASSES, own_deck=HOG26, config=config)
        assert plays == [] and d.opp_plays() == [] and d.own_landings() == [] and tracker.states() == []
    frames[100] = [tob("Zap", 9000, 16300, 0.9, conf=0.8)]     # a label only the enemy casts: at once
    tracker, d, plays = run_tracked(frames, SPELL_CLASSES, own_deck=HOG26)
    assert [(p.card, p.tick) for p in plays] == [(ZAP, 100)]


def test_our_skeletons_box_read_as_zap_is_no_enemy_play():
    """Our Skeleton's box read as Zap for two frames, at the Skeleton's size, joins the Skeleton's
    track (scripts/vision/tracker.py, SPELLS: flickers): no enemy play. A Zap-sized box there, or one
    without a box size, is the enemy's Zap."""
    def frames(zap_size):
        out = {}
        for tick in range(100, 130, 2):
            if 112 <= tick < 116:
                w, h = zap_size or (None, None)
                out[tick] = [T.Observation("Zap", 5000.0, 24000.0, 0.7, 0.1, box_w=w, box_h=h)]
            else:
                out[tick] = [T.Observation("Skeleton", 5000.0, 24000.0, 0.9, 0.1, box_w=36.0, box_h=40.0)]
        return out

    for config in (None, EV.EventConfig.live()):
        tracker, d, plays = run_tracked(frames((38.0, 44.0)), SPELL_CLASSES, own_deck=HOG26, config=config)
        assert d.opp_plays() == [] and len(tracker.states()) == 1
        skel = tracker.state(1)
        assert (skel.label, tracker.team_of(skel), skel.hits) == ("Skeleton", O, 15)
        assert [(x.card, x.land_tick) for x in d.own_landings()] == [(SKELETONS, 100)]
    for size in ((175.0, 140.0), None):
        tracker, d, plays = run_tracked(frames(size), SPELL_CLASSES, own_deck=HOG26)
        assert [(p.card, p.tick) for p in d.opp_plays()] == [(ZAP, 112)], size


def test_a_second_cast_elsewhere_is_a_play_of_its_own():
    """Rule 2 for a spell past PLAY_WINDOW: a Zap seen again 30 ticks after the last sighting is the
    same cast only within spell_reach_mt (+ the play's own speed x the gap). An area does not move,
    so a Zap on the other lane (or a Mirrored one) is a second play; 1,500 mt away it is the same
    area. Within PLAY_WINDOW any distance merges, as before."""
    def run(x2, gap, config=None):
        d = deriver(config=config)
        d.update([tk(1, "Zap", E, 4000, 22000, 100, 101)], 101)
        d.update([tk(1, "Zap", E, 4000, 22000, 100, 104)], 104)
        first = 104 + gap
        d.update([tk(2, "Zap", E, x2, 22000, first, first + 1)], first + 1)
        return [(p.card, p.tick) for p in d.opp_plays()]

    for config in (None, EV.EventConfig.live()):
        assert run(14000, 30, config) == [(ZAP, 100), (ZAP, 134)]
        assert run(5500, 30, config) == [(ZAP, 100)]
        assert run(14000, 15, config) == [(ZAP, 100)]
    assert run(14000, 30, EV.EventConfig(spell_reach_mt=None)) == [(ZAP, 100)]
    with pytest.raises(ValueError):
        EV.EventConfig(spell_reach_mt=-1.0)


BARB_LOG, BARBARIANS = 30, 31


def test_a_barbarian_barrel_boxed_as_the_log_is_one_play():
    """A spell that puts a unit down, boxed under a look-alike's label: the enemy's Barbarian Barrel
    boxed as the Log rolls out its Barbarian, which the tracker's kind rule keeps a track of its own.
    The Log's classes explain it (SPELL_LOOKALIKES), so it is the Log's child, not a 5-elixir
    Barbarians play; and a spell track whose label the tracker fixes to BarbLog explains it too."""
    table = {"kinds": {"Log": "spell", "BarbLog": "spell", "Barbarians": "troop"},
             "compositions": {"Barbarians": {"Barbarian": 5}},
             "spell_objects": {"Log": ["LogProjectile", "LogProjectileRolling"], "BarbLog": ["BarbLogProjectile"]},
             "spell_children": {"BarbLog": ["Barbarian"]}, "children": {}}
    ids = {"Log": LOG, "BarbLog": BARB_LOG, "Barbarians": BARBARIANS}
    tb = EV.EventTables.from_names(table, ids)
    assert tb.explains("Log", "Barbarian") and tb.explains("the-log", "Barbarian")
    assert tb.explains("BarbLog", "Barbarian") and tb.unit_cards("Barbarian") == (BARBARIANS,)
    plain = EV.EventTables(compositions={BARBARIANS: {"Barbarian": 5}},
                           spells={LOG: {"Log"}, BARB_LOG: {"BarbLog"}}, children={"BarbLog": {"Barbarian"}})

    def run(tables, labels, config=None):
        d = EV.EventDeriver(tables, config)
        d.update([], 0)
        for k, tick in enumerate(range(100, 130, 2)):
            d.update([tk(1, labels(k), E, 4000, 6000 + 330 * k, 100, tick)], tick)
        for tick in range(130, 170, 2):
            d.update([tk(2, "Barbarian", E, 4000, 11300, 130, tick)], tick)
        return d

    for config in (None, EV.EventConfig.live()):
        d = run(tb, lambda k: "Log", config)
        assert [(p.card, p.tick, p.uids) for p in d.opp_plays()] == [(LOG, 100, [1, 2])]
        assert d.how_of(2) == "child"
        d = run(plain, lambda k: "Log" if k < 3 else "BarbLog", config)     # the label follows the track
        assert [(p.card, p.tick) for p in d.opp_plays()] == [(LOG, 100)] and d.how_of(2) == "child"
    d = run(plain, lambda k: "Log")                 # neither: the Barbarian is a play of its own
    assert [(p.card, p.tick) for p in d.opp_plays()] == [(LOG, 100), (BARBARIANS, 130)]


def test_state_tracks_run_the_deriver_on_engine_states():
    tb = tables()
    spell_fields = ["uid", "team", "card_id", "aim_x", "aim_y", "ticks_flown"]
    st = EV.StateTracks(ENTITY_FIELDS, DEFAULT_UNIT_TYPES, sub=18, spell_fields=spell_fields,
                        spell_class=tb.spell_class)
    knight = entity_row(uid=40, team=0, unit_type=8, x=9000 * 18, y=8000 * 18, hp=500, max_hp=1000)
    dead = entity_row(uid=41, team=0, unit_type=8, x=3000 * 18, y=8000 * 18, hp=0, max_hp=1000)
    tr1 = st.tracks(state_dict(40, entities=[knight, dead]))        # crown towers are left out
    assert [(t.uid, t.cls, t.team, t.x_mt, t.y_mt, t.hp_frac, t.first_seen_tick, t.last_seen_tick)
            for t in tr1] == [(40, "Knight", 0, 9000.0, 8000.0, 0.5, 40, 40)]
    fireball = [5, 0, FIREBALL, 9000 * 18, 26000 * 18, 4]
    tr2 = st.tracks(state_dict(50, entities=[knight], spells=[fireball]))
    assert [(t.uid, t.cls, t.team, t.x_mt, t.y_mt, t.first_seen_tick, t.last_seen_tick) for t in tr2] == [
        (40, "Knight", 0, 9000.0, 8000.0, 40, 50),
        (EV.SPELL_UID_OFFSET + 5, "FireballSpell", 0, 9000.0, 26000.0, 46, 50)]
    d = EV.EventDeriver(tb)
    d.update([], 0)
    plays = d.update(tr1, 40).opp_plays + d.update(tr2, 50).opp_plays
    assert [(p.card, p.tick, p.kind) for p in plays] == [(KNIGHT, 40, "unit"), (FIREBALL, 46, "spell")]
    assert EV.strip_form("Skeleton_EV1") == "Skeleton" and EV.strip_form("MusketeerHero") == "Musketeer"


def test_state_tracks_key_engine_spells_without_a_uid():
    # royalesim's SpellState columns (royalegym protocol.py SpellState): there is no uid
    fields = ["team", "card_id", "motion", "x", "y", "aim_x", "aim_y", "delay_ticks", "travelled",
              "length", "hits", "ticks_flown"]
    tb = tables()
    st = EV.StateTracks(ENTITY_FIELDS, DEFAULT_UNIT_TYPES, sub=18, spell_fields=fields,
                        spell_class=tb.spell_class)

    def spell(card, ax, ay, flown):
        return [0, card, 0, ax * 18, ay * 18, ax * 18, ay * 18, 0, 0, 0, 0, flown]

    log = spell(LOG, 3500, 27000, -1)              # a roll; -1: ticks_flown not reported
    fireball = spell(FIREBALL, 9000, 26000, 4)
    a = st.tracks(state_dict(50, spells=[log, fireball]))
    assert [(t.uid, t.cls, t.team, t.first_seen_tick, t.last_seen_tick) for t in a] == [
        (EV.SPELL_UID_OFFSET + 1, "LogProjectile", 0, 50, 50),
        (EV.SPELL_UID_OFFSET + 2, "FireballSpell", 0, 46, 50)]
    b = st.tracks(state_dict(51, spells=[log, fireball]))       # the same casts, a tick later
    assert [(t.uid, t.first_seen_tick, t.last_seen_tick) for t in b] == [
        (EV.SPELL_UID_OFFSET + 1, 50, 51), (EV.SPELL_UID_OFFSET + 2, 46, 51)]
    assert st.tracks(state_dict(60)) == []                       # both are over
    [again] = st.tracks(state_dict(120, spells=[spell(FIREBALL, 9000, 26000, 0)]))
    assert (again.uid, again.first_seen_tick) == (EV.SPELL_UID_OFFSET + 3, 120)   # a new cast
    d = EV.EventDeriver(tb)
    d.update([], 0)
    assert [(p.card, p.tick) for p in d.update(a, 50).opp_plays] == [(FIREBALL, 46), (LOG, 50)]
    assert [(p.card, p.tick) for p in d.update([again], 120).opp_plays] == [(FIREBALL, 120)]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def test_table_lookups_and_validation():
    tb = tables()
    assert tb.unit_cards("Skeleton") == (SKELETONS, SKELETON_ARMY)
    assert tb.spell_cards("LogProjectileRolling") == (LOG,)
    assert tb.is_spell("FireballSpell") and not tb.is_spell("HogRider")
    assert (tb.kind_of(LOG), tb.kind_of(HOG), tb.kind_of(99)) == ("spell", "unit", None)
    assert (tb.units_per_play(SKELETONS), tb.units_per_play(LOG)) == (3, None)
    assert tb.card_guess("Skeleton") == SKELETONS
    assert tb.card_guess("Goblin", prefer=[GOBLIN_GANG]) == GOBLIN_GANG
    assert tb.card_guess("Golemite") is None
    assert tb.explains("Witch", "Skeleton") and not tb.explains("Skeleton", "Witch")
    with pytest.raises(ValueError):
        EV.EventTables({1: {"A": 1}}, spells={1: {"B"}})        # a card both unit and spell
    with pytest.raises(ValueError):
        EV.EventTables({1: {"A": 1}}, spells={2: {"A"}})        # a class both unit and spell
    with pytest.raises(ValueError):
        EV.EventTables({1: {"A": 0}})
    tj = EV.EventTables({"6": ["Skeleton", "Skeleton", "Skeleton"]}, spells={"7": ["LogProjectile"]},
                        children={"Witch": ["Skeleton"]})        # JSON-shaped
    assert tj.compositions == {6: {"Skeleton": 3}} and tj.spells == {7: frozenset({"LogProjectile"})}
    assert tj.children == {"Witch": frozenset({"Skeleton"})}


def test_tables_from_spawn_primary():
    raw = {"primary": {"6": "Skeleton", "20": "Witch", "2": "HogRider"},
           "all": {"6": ["Skeleton"] * 3, "20": ["Witch", "Skeleton"], "2": "HogRider"},
           "hp": {}, "fingerprint": "f"}
    tb = EV.EventTables.from_spawn_primary(raw)
    assert tb.compositions == {6: {"Skeleton": 3}, 20: {"Witch": 1, "Skeleton": 1}, 2: {"HogRider": 1}}
    assert tb.explains("Witch", "Skeleton") and tb.spells == {}
    base = EV.EventTables({20: {"Witch": 1}, 6: {"Skeleton": 3}}, spells={7: {"LogProjectile"}})
    tb2 = EV.EventTables.from_spawn_primary(raw, base=base)
    assert tb2.compositions == {20: {"Witch": 1}, 6: {"Skeleton": 3}, 2: {"HogRider": 1}}
    assert tb2.explains("Witch", "Skeleton") and tb2.spells == {7: frozenset({"LogProjectile"})}
    assert tb2.unit_cards("Skeleton") == (6,)
    # units a card deploys together are not each other's children when base knows the card:
    # a Spear Goblins play next to a Goblin Gang's living goblin is still a play
    gang = {"primary": {"24": "Goblin"}, "all": {"24": ["Goblin", "SpearGoblin"]}}
    tb3 = EV.EventTables.from_spawn_primary(
        gang, base=EV.EventTables({24: {"Goblin": 3, "SpearGoblin": 3}}))
    assert tb3.compositions == {24: {"Goblin": 3, "SpearGoblin": 3}}
    assert not tb3.explains("Goblin", "SpearGoblin")
    assert EV.EventTables.from_spawn_primary(gang).explains("Goblin", "SpearGoblin")   # no counts


def test_tables_for_the_engine_from_e06s_file():
    # tools/rs_spawn_table.py's layout: unit types are unit_types_json() indexes, "all" is
    # deduplicated (no counts), card names give the catalogue ids
    raw = {"format": 1, "card_names": [row[0] for row in DEFAULT_CATALOGUE],
           "unit_types": list(DEFAULT_UNIT_TYPES),
           "primary": {"0": 2, "2": 3, "6": 7, "8": 8, "9": 9},
           "all": {"0": [2], "2": [3], "6": [7], "8": [8], "9": [9]}}
    tb = EV.EventTables.from_spawn_primary(raw)
    assert tb.compositions == {0: {"Cannon": 1}, 2: {"HogRider": 1}, 6: {"Skeleton": 1},
                               8: {"Knight": 1}, 9: {"Archer": 1}}
    tb = EV.EventTables.for_engine(raw)                 # + data/vision/spawn_children.json
    assert tb.compositions[6] == {"Skeleton": 3} and tb.compositions[9] == {"Archer": 2}
    assert tb.compositions[CANNON] == {"Cannon": 1} and tb.compositions[HOG] == {"HogRider": 1}
    assert tb.unit_cards("Skeleton") == (SKELETONS,)
    assert tb.kind_of(LOG) == "spell" and {"LogProjectile", "the-log"} <= tb.classes_of(LOG)
    assert tb.kind_of(ZAP) == "spell" and tb.kind_of(FIREBALL) == "spell"


IDS = {"Skeletons": 6, "SkeletonArmy": 21, "Witch": 20, "Golem": 22, "Log": 7, "Fireball": 1,
       "GoblinBarrel": 30, "Goblins": 23, "GoblinGang": 24, "Clone": 31, "HogRider": 2}


def test_tables_from_the_generated_spawn_table():
    table = EV.load_spawn_children()
    # N01: generated from the 16.402.22 table data/engine.json selects; 15.535 only adds
    assert table["source"]["version"] == "cards-160402017-20261019.1"
    assert [t["version"] for t in table["source"]["tables"]] == ["cards-160402017-20261019.1",
                                                                  "cards-15535.1"]
    tb = EV.EventTables.from_names(table, IDS)
    assert set(tb.compositions) == {2, 6, 20, 21, 22, 23, 24}         # cards IDS lacks are skipped
    assert tb.compositions[6] == {"Skeleton": 3} and tb.compositions[21] == {"Skeleton": 15}
    assert tb.compositions[24] == {"Goblin_Stab": 3, "SpearGoblin": 3}
    assert tb.unit_cards("Skeleton") == (6, 21)
    assert tb.explains("Witch", "Skeleton") and tb.explains("Golem", "Golemite")
    assert {"Log", "LogProjectile", "LogProjectileRolling", "the-log"} <= tb.classes_of(7)
    assert {"Fireball", "FireballSpell", "fireball"} <= tb.classes_of(1)
    assert tb.explains("goblin-barrel", "Goblin")                       # a spell's spawns
    assert tb.explains("clone", "HogRider")                             # Clone copies anything


def test_tables_with_the_vision_class_list():
    path = ROOT / "data" / "vision" / "classes.json"
    if not path.exists():
        pytest.skip("data/vision/classes.json (E16) is not there")
    doc = json.loads(path.read_text(encoding="utf-8"))
    cmap = EV.unit_class_map(doc)
    assert (cmap["Goblin_Stab"], cmap["Skeleton_EV1"], cmap["HogRider"]) == ("Goblin", "Skeleton", "HogRider")
    tb = EV.EventTables.from_names(EV.load_spawn_children(), IDS, class_of=cmap,
                                   ignore=EV.cardless_classes(doc))
    assert tb.compositions[24] == {"Goblin": 3, "SpearGoblin": 3} and tb.compositions[23] == {"Goblin": 4}
    assert tb.unit_cards("Goblin") == (23, 24)
    assert "TowerPrincess" in tb.ignore and "Skeleton" not in tb.ignore
    # the KataCR spell names agree with the class list's crosswalk
    katacr_spells: dict = {}
    for name, entry in doc.get("katacr", {}).items():
        if entry.get("type") == "spell":
            katacr_spells.setdefault(entry["card"], set()).add(name)
    for card, names in EV.KATACR_SPELL_CLASSES.items():
        assert set(names) <= katacr_spells.get(card, set()), card


# ---------------------------------------------------------------------------
# N01 (docs/NEW_CARDS_PLAN.md 1.7, 2.1): Heal Spirit, the Spirit Empress's forms, the
# Goblin Barrel's goblins, and the generator that writes them
# ---------------------------------------------------------------------------
HEAL, EMPRESS, EMPRESS_NORMAL, EMPRESS_MOUNTED, GOBLIN_BARREL = 40, 41, 42, 43, 30
N01_IDS = {**IDS, "Heal": HEAL, "MergeMaiden": EMPRESS, "MergeMaiden_Normal": EMPRESS_NORMAL,
           "MergeMaiden_Mounted": EMPRESS_MOUNTED}


def started(tb: EV.EventTables) -> EV.EventDeriver:
    d = EV.EventDeriver(tb)
    d.update([], 0)            # the battle starts on an empty board
    return d


def without_n01(table: dict) -> dict:
    """The shipped spawn table as the 15.535 extraction had Heal and the Spirit Empress."""
    old = json.loads(json.dumps(table))
    del old["compositions"]["Heal"], old["variants"]
    return old


def test_a_heal_spirit_track_is_a_heal_play():
    table = EV.load_spawn_children()
    assert table["compositions"]["Heal"] == {"HealSpirit": 1}
    tb = EV.EventTables.from_names(table, N01_IDS)
    assert tb.compositions[HEAL] == {"HealSpirit": 1} and HEAL not in tb.spells
    assert (tb.kind_of(HEAL), tb.unit_cards("HealSpirit"), tb.units_per_play(HEAL)) == ("unit", (HEAL,), 1)
    d = started(tb)
    ev = d.update([tk(1, "HealSpirit", E, 9000, 12000, 100, 101)], 101)
    assert [(p.card, p.kind, p.tick, p.exec_tick, p.uids) for p in ev.opp_plays] == [
        (HEAL, "unit", 100, 100, [1])]
    assert d.card_of(1) == HEAL and d.opp_stats.unknown == {}
    # before N01 the only Heal class was the card's name, which no detector label is
    tb_old = EV.EventTables.from_names(without_n01(table), N01_IDS)
    assert tb_old.spells[HEAL] == frozenset({"Heal"}) and tb_old.unit_cards("HealSpirit") == ()
    d = started(tb_old)
    assert d.update([tk(1, "HealSpirit", E, 9000, 12000, 100, 101)], 101).opp_plays == []
    assert d.opp_stats.unknown == {"HealSpirit": 1}


def test_spirit_empress_forms_are_reported_as_the_card_at_their_cost():
    table = EV.load_spawn_children()
    assert table["variants"] == {"MergeMaiden": {"MergeMaiden_Mounted": 6, "MergeMaiden_Normal": 3}}
    tb = EV.EventTables.from_names(table, N01_IDS)
    assert tb.compositions[EMPRESS] == {"MergeMaiden_Mounted": 1, "MergeMaiden_Normal": 1}
    assert EMPRESS_NORMAL not in tb.compositions and EMPRESS_MOUNTED not in tb.compositions
    assert EMPRESS not in tb.spells and tb.kind_of(EMPRESS) == "unit"
    assert tb.aliases == {EMPRESS_NORMAL: EMPRESS, EMPRESS_MOUNTED: EMPRESS}
    assert (tb.reported_card(EMPRESS_MOUNTED), tb.reported_card(EMPRESS), tb.reported_card(HEAL)) == (
        EMPRESS, EMPRESS, HEAL)
    assert tb.unit_cards("MergeMaiden_Mounted") == tb.unit_cards("MergeMaiden_Normal") == (EMPRESS,)
    assert tb.variant_costs == {EMPRESS: {"MergeMaiden_Mounted": 6, "MergeMaiden_Normal": 3}}
    assert (tb.play_cost(EMPRESS, "MergeMaiden_Mounted"), tb.play_cost(EMPRESS, ["MergeMaiden_Normal"])) == (6, 3)
    assert tb.play_cost(EMPRESS, {"MergeMaiden_Normal", "MergeMaiden_Mounted"}) == 6   # both: the dearer
    assert tb.play_cost(EMPRESS) is None and tb.play_cost(HEAL, "HealSpirit") is None
    assert tb.units_per_play(EMPRESS) is None             # one form, which one unknown before it lands
    assert tb.card_guess("MergeMaiden_Normal") == EMPRESS

    d = started(tb)
    ev = d.update([tk(1, "MergeMaiden_Mounted", E, 9000, 9000, 200, 201)], 201)
    (play,) = ev.opp_plays
    assert (play.card, play.kind, play.tick, play.uids) == (EMPRESS, "unit", 200, [1])
    assert tb.play_cost(play.card, ["MergeMaiden_Mounted"]) == 6
    ev = d.update([tk(2, "MergeMaiden_Normal", E, 3500, 9000, 600, 601)], 601)
    assert [(p.card, p.tick, tb.play_cost(p.card, "MergeMaiden_Normal")) for p in ev.opp_plays] == [
        (EMPRESS, 600, 3)]
    assert [d.card_of(u) for u in (1, 2)] == [EMPRESS, EMPRESS]

    # a catalogue without the card itself: the forms stay cards of their own, as before N01
    ids = {k: v for k, v in N01_IDS.items() if k != "MergeMaiden"}
    tb2 = EV.EventTables.from_names(table, ids)
    assert tb2.compositions[EMPRESS_MOUNTED] == {"MergeMaiden_Mounted": 1} and tb2.aliases == {}
    assert tb2.unit_cards("MergeMaiden_Mounted") == (EMPRESS_MOUNTED,) and tb2.variant_costs == {}


def test_spirit_empress_forms_stay_her_on_the_engine_tables():
    # E06's spawn_primary.json lists the forms as TROOP cards with primary units of their own
    names = [row[0] for row in DEFAULT_CATALOGUE] + [f"Card{i}" for i in range(len(DEFAULT_CATALOGUE), 44)]
    for name, cid in (("Heal", HEAL), ("MergeMaiden", EMPRESS), ("MergeMaiden_Normal", EMPRESS_NORMAL),
                      ("MergeMaiden_Mounted", EMPRESS_MOUNTED)):
        names[cid] = name
    raw = {"card_names": names,
           "primary": {str(EMPRESS_NORMAL): "MergeMaiden_Normal", str(EMPRESS_MOUNTED): "MergeMaiden_Mounted",
                       str(HOG): "HogRider"},
           "all": {str(EMPRESS_NORMAL): ["MergeMaiden_Normal"], str(EMPRESS_MOUNTED): ["MergeMaiden_Mounted"],
                   str(HOG): ["HogRider"]}}
    tb = EV.EventTables.for_engine(raw)                 # + data/vision/spawn_children.json
    assert EMPRESS_NORMAL not in tb.compositions and EMPRESS_MOUNTED not in tb.compositions
    assert tb.compositions[EMPRESS] == {"MergeMaiden_Mounted": 1, "MergeMaiden_Normal": 1}
    assert tb.unit_cards("MergeMaiden_Normal") == (EMPRESS,) and tb.compositions[HEAL] == {"HealSpirit": 1}
    assert tb.aliases == {EMPRESS_NORMAL: EMPRESS, EMPRESS_MOUNTED: EMPRESS}
    assert tb.play_cost(EMPRESS, "MergeMaiden_Normal") == 3 and tb.compositions[HOG] == {"HogRider": 1}
    d = started(tb)
    ev = d.update([tk(1, "MergeMaiden_Normal", E, 14500, 9000, 300, 301)], 301)
    assert [(p.card, p.kind) for p in ev.opp_plays] == [(EMPRESS, "unit")]
    # without base tables nothing is known of forms: each is its own card, as before
    plain = EV.EventTables.from_spawn_primary(raw)
    assert plain.unit_cards("MergeMaiden_Normal") == (EMPRESS_NORMAL,) and plain.aliases == {}


def test_goblin_barrel_goblins_are_its_children_on_screen_and_in_the_engine():
    table = EV.load_spawn_children()
    # (+ the evolved barrel's decoy goblins, GoblinDummy: test_events_spawners.py)
    assert table["spell_children"]["GoblinBarrel"] == ["BarrelGoblin", "Goblin", "GoblinDummy"]
    tb = EV.EventTables.from_names(table, N01_IDS)
    for cls in ("goblin-barrel", "GoblinBarrelSpell", "GoblinBarrel"):
        assert tb.explains(cls, "Goblin") and tb.explains(cls, "BarrelGoblin"), cls
    d = started(tb)
    ev = d.update([tk(1, "goblin-barrel", E, 9000, 26000, 300, 301)], 301)
    assert [(p.card, p.kind) for p in ev.opp_plays] == [(GOBLIN_BARREL, "spell")]
    # KataCR boxes the 16.402 BarrelGoblin as `goblin`; an engine state names it BarrelGoblin
    landed = [tk(2 + i, cls, E, 8500 + 500 * i, 26500, 322, 323)
              for i, cls in enumerate(["Goblin", "Goblin", "BarrelGoblin"])]
    assert d.update([tk(1, "goblin-barrel", E, 9000, 26000, 300, 320)] + landed, 323).opp_plays == []
    assert [d.card_of(u) for u in (2, 3, 4)] == [GOBLIN_BARREL] * 3
    assert d.opp_stats.children == 3 and d.opp_stats.unknown == {}


@pytest.mark.parametrize("vision_classes", [False, True])
def test_n01_changes_nothing_else_in_the_tables(vision_classes):
    path = ROOT / "data" / "vision" / "classes.json"
    if vision_classes and not path.exists():
        pytest.skip("data/vision/classes.json (E16) is not there")
    kw = {}
    if vision_classes:
        doc = json.loads(path.read_text(encoding="utf-8"))
        kw = {"class_of": EV.unit_class_map(doc), "ignore": EV.cardless_classes(doc)}
    new = EV.load_spawn_children()
    ids = {name: i for i, name in enumerate(sorted(new["kinds"]))}     # every card of the table
    a = EV.EventTables.from_names(new, ids, **kw)
    b = EV.EventTables.from_names(without_n01(new), ids, **kw)
    n01 = {ids[n] for n in ("Heal", "MergeMaiden", "MergeMaiden_Normal", "MergeMaiden_Mounted")}

    def but_n01(m):
        return {k: v for k, v in m.items() if k not in n01}

    assert but_n01(a.compositions) == but_n01(b.compositions)
    assert but_n01(a.spells) == but_n01(b.spells)
    assert a.children == {k: v for k, v in b.children.items() if k != "Heal"}   # Heal: no spell now
    assert a.ignore == b.ignore and b.aliases == {} and b.variant_costs == {}
    assert a.aliases == {ids["MergeMaiden_Normal"]: ids["MergeMaiden"],
                         ids["MergeMaiden_Mounted"]: ids["MergeMaiden"]}


def test_aliases_and_variant_costs_are_checked():
    tb = EV.EventTables({41: {"MergeMaiden_Normal": 1, "MergeMaiden_Mounted": 1}}, aliases={"42": "41"},
                        variant_costs={"41": {"MergeMaiden_Normal": 3}})          # JSON-shaped
    assert tb.aliases == {42: 41} and tb.variant_costs == {41: {"MergeMaiden_Normal": 3}}
    assert tb.play_cost(41, "MergeMaiden_Mounted") is None                         # no cost given
    assert EV.EventTables({1: {"A": 1}}) == EV.EventTables({1: {"A": 1}}, aliases={}, variant_costs={})
    with pytest.raises(ValueError):
        EV.EventTables({41: {"A": 1}, 42: {"B": 1}}, aliases={42: 41})   # a form with a composition
    with pytest.raises(ValueError):
        EV.EventTables({41: {"A": 1}}, aliases={41: 41})
    with pytest.raises(ValueError):
        EV.EventTables({40: {"A": 1}}, aliases={42: 41, 41: 40})         # a form of a form
    with pytest.raises(ValueError):
        EV.EventTables({41: {"A": 1}}, variant_costs={41: {"B": 3}})     # a class it does not deploy
    with pytest.raises(ValueError):
        EV.EventTables({41: {"A": 1}}, variant_costs={42: {"A": 3}})     # a card with no composition


def mini_table(version: str, *, barrel_spawns: str = "BarrelGoblin", old: bool = False) -> dict:
    """A card table in RoyaleSim's cards.json layout, just big enough for the generator."""
    projectile = "GoblinBarrelSpellOld" if old else "GoblinBarrelSpell"
    units = {"Witch": {"spawner": {"character": "Skeleton"}}, "Skeleton": {}, "Goblin": {},
             "BarrelGoblin": {}, "HealSpirit": {}, "RageBottle": {}, "MergeMaiden_Normal": {},
             "MergeMaiden_Mounted": {}}
    if old:
        units["Boost"] = {"death_spawn": {"character": "Skeleton"}}       # gone from the new table
    cards = [
        {"name": "Witch", "kind": "troop", "elixir": 5, "summon_character": "Witch", "count": 2 if old else 1},
        {"name": "GoblinBarrel", "kind": "spell", "elixir": 3, "projectile": {"name": projectile}},
        {"name": "Heal", "kind": "spell", "elixir": 1,
         "spell": {"spell_as_deploy": True, "summon": {"character": "HealSpirit", "count": 1}}},
        {"name": "Rage", "kind": "spell", "elixir": 2,
         "spell": {"spell_as_deploy": False, "summon": {"character": "RageBottle", "count": 1}}},
        {"name": "MergeMaiden", "kind": "spell", "elixir": 6,
         "spell": {"variant": {"options": [{"trigger_milli": 6000, "card": "MergeMaiden_Mounted"},
                                           {"trigger_milli": 3000, "card": "MergeMaiden_Normal"}]}}},
        {"name": "MergeMaiden_Normal", "kind": "troop", "elixir": 3, "summon_character": "MergeMaiden_Normal"},
        {"name": "MergeMaiden_Mounted", "kind": "troop", "elixir": 6, "summon_character": "MergeMaiden_Mounted"},
    ]
    return {"version": version, "cards": cards, "units": units, "hero_forms": [],
            "projectiles": {projectile: {"spawn_character": barrel_spawns}}, "area_effect_objects": {}}


def test_generator_reads_heal_the_variants_and_the_barrel_goblin():
    VSC = pytest.importorskip("tools.vision_spawn_children")
    doc = VSC.build(mini_table("cards-new.1"), files=["new.json"])
    assert doc["source"] == {"file": "new.json", "version": "cards-new.1", "package": "royalesim==0.1.42"}
    assert doc["kinds"]["Heal"] == "spell" and doc["compositions"]["Heal"] == {"HealSpirit": 1}
    assert "Rage" not in doc["compositions"] and doc["spell_children"]["Rage"] == ["RageBottle"]
    assert doc["variants"] == {"MergeMaiden": {"MergeMaiden_Mounted": 6, "MergeMaiden_Normal": 3}}
    assert "MergeMaiden" not in doc["compositions"]
    assert doc["compositions"]["MergeMaiden_Normal"] == {"MergeMaiden_Normal": 1}
    assert doc["spell_objects"]["GoblinBarrel"] == ["GoblinBarrelSpell"]
    assert doc["spell_children"]["GoblinBarrel"] == ["BarrelGoblin", "Goblin"]      # LOOKS_LIKE
    assert doc["children"] == {"Witch": ["Skeleton"]}
    # what the generator writes is what the deriver needs
    tb = EV.EventTables.from_names(doc, {"Heal": HEAL, "MergeMaiden": EMPRESS, "GoblinBarrel": GOBLIN_BARREL})
    assert tb.unit_cards("HealSpirit") == (HEAL,) and tb.unit_cards("MergeMaiden_Mounted") == (EMPRESS,)
    assert tb.play_cost(EMPRESS, "MergeMaiden_Mounted") == 6 and tb.explains("goblin-barrel", "Goblin")


def test_generator_keeps_what_an_older_table_has_and_the_newer_lacks():
    VSC = pytest.importorskip("tools.vision_spawn_children")
    new = mini_table("cards-new.1")
    old = mini_table("cards-old.1", barrel_spawns="Goblin", old=True)
    doc = VSC.build([new, old], files=["new.json", "old.json"])
    assert doc["source"]["version"] == "cards-new.1" and doc["source"]["file"] == "new.json"
    assert doc["source"]["tables"] == [{"version": "cards-new.1", "file": "new.json"},
                                       {"version": "cards-old.1", "file": "old.json"}]
    assert doc["compositions"]["Witch"] == {"Witch": 1}                  # the first table's
    assert doc["children"] == {"Boost": ["Skeleton"], "Witch": ["Skeleton"]}   # kept from the old
    assert doc["spell_objects"]["GoblinBarrel"] == ["GoblinBarrelSpell", "GoblinBarrelSpellOld"]
    assert doc["spell_children"]["GoblinBarrel"] == ["BarrelGoblin", "Goblin"]
    assert VSC.build(old)["spell_children"]["GoblinBarrel"] == ["Goblin"]        # 15.535-like alone


def test_generator_finds_a_table_in_derived_or_in_the_extension(tmp_path):
    VSC = pytest.importorskip("tools.vision_spawn_children")
    pkg = tmp_path / "royalesim"
    derived = pkg / "data" / "derived"
    derived.mkdir(parents=True)
    (derived / "cards.json").write_text(json.dumps({"version": "cards-15535.1", "cards": []}), encoding="utf-8")
    (derived / "cards-x.json").write_text(json.dumps({"version": "cards-x.1", "cards": []}), encoding="utf-8")
    doc, where = VSC.find_table("cards-x.1", pkg)
    assert doc["version"] == "cards-x.1" and where == "royalesim/data/derived/cards-x.json"
    # a wheel ships no JSON copy of 16.402.22; its extension carries it as text
    text = json.dumps({"version": "cards-160402017-20261019.1", "cards": [1]}).encode("utf-8")
    (pkg / "royalesim.pyd").write_bytes(b"MZ\x00\x01binary{x}" + text + b"\x00\xffmore")
    doc, where = VSC.find_table("cards-160402017-20261019.1", pkg)
    assert doc["cards"] == [1] and where == "royalesim/royalesim.pyd (its compiled-in copy)"
    with pytest.raises(SystemExit):
        VSC.find_table("cards-none.1", pkg)


# ---------------------------------------------------------------------------
# N01 review fixes: KataCR's `bomb` after a Balloon or a Bomb Tower, the Goblin Drill's
# building, and the units champions' abilities create
# ---------------------------------------------------------------------------
def vision_tables(table: dict | None = None) -> tuple[dict, dict, EV.EventTables]:
    """(the shipped spawn table or `table`, every card of it -> an id, its EventTables with
    data/vision/classes.json's class map): what the live chain builds."""
    path = ROOT / "data" / "vision" / "classes.json"
    if not path.exists():
        pytest.skip("data/vision/classes.json (E16) is not there")
    doc = json.loads(path.read_text(encoding="utf-8"))
    table = EV.load_spawn_children() if table is None else table
    ids = {name: i for i, name in enumerate(sorted(table["kinds"]))}
    tb = EV.EventTables.from_names(table, ids, class_of=EV.unit_class_map(doc),
                                   ignore=EV.cardless_classes(doc))
    return table, ids, tb


@pytest.mark.parametrize("parent", ["Balloon", "BombTower"])
def test_a_death_bomb_boxed_as_the_giant_skeletons_is_its_parents_child(parent):
    table, ids, tb = vision_tables()
    assert "GiantSkeletonBomb" in table["children"][parent]          # LOOKS_LIKE
    assert tb.explains(parent, "GiantSkeletonBomb") and tb.explains("GiantSkeleton", "GiantSkeletonBomb")

    def run(tables):
        d = started(tables)
        ev = d.update([tk(1, parent, E, 9000, 20000, 100, 101)], 101)
        assert [(p.card, p.kind) for p in ev.opp_plays] == [(ids[parent], "unit")]
        d.update([tk(1, parent, E, 9000, 22000, 100, 200, hp=0.05)], 200)        # it dies here
        # KataCR's one `bomb` label is the class GiantSkeletonBomb (classes.json "also")
        return d, d.update([tk(2, "GiantSkeletonBomb", E, 9000, 22000, 205, 206)], 206)

    d, ev = run(tb)
    assert ev.opp_plays == [] and d.card_of(2) == ids[parent] and d.how_of(2) == "child"
    assert d.opp_stats.children == 1 and d.opp_stats.unknown == {}
    # before: only a Giant Skeleton explained the class, so the bomb was an unknown track
    old = json.loads(json.dumps(table))
    old["children"][parent] = [k for k in old["children"][parent] if k != "GiantSkeletonBomb"]
    d, ev = run(vision_tables(old)[2])
    assert ev.opp_plays == [] and d.card_of(2) is None and d.opp_stats.unknown == {"GiantSkeletonBomb": 1}


def test_a_goblin_drill_building_is_a_goblin_drill_play():
    table, ids, tb = vision_tables()
    drill = ids["GoblinDrill"]
    assert table["compositions"]["GoblinDrill"] == {"GoblinDrill": 1, "GoblinDrillDig": 1}
    assert table["children"]["GoblinDrillDig"] == ["Goblin", "GoblinDrill"]      # + the drill's goblins
    assert tb.unit_cards("GoblinDrill") == (drill,) and tb.explains("GoblinDrillDig", "GoblinDrill")
    d = started(tb)
    # the dig is KataCR's `dirt`, an object no track follows: the building is the first track
    ev = d.update([tk(1, "GoblinDrill", E, 3500, 26000, 300, 301)], 301)
    assert [(p.card, p.kind, p.uids) for p in ev.opp_plays] == [(drill, "unit", [1])]
    ev = d.update([tk(1, "GoblinDrill", E, 3500, 26000, 300, 330),
                   tk(2, "Goblin", E, 3500, 26800, 325, 330)], 330)                # its spawner goblin
    assert ev.opp_plays == [] and d.card_of(2) == drill and d.opp_stats.unknown == {}
    # before: the composition was the dig alone, so the building was an unknown track
    old = json.loads(json.dumps(table))
    old["compositions"]["GoblinDrill"] = {"GoblinDrillDig": 1}
    d = started(vision_tables(old)[2])
    assert d.update([tk(1, "GoblinDrill", E, 3500, 26000, 300, 301)], 301).opp_plays == []
    assert d.opp_stats.unknown == {"GoblinDrill": 1}

    # engine names (StateTracks): the dig is the play, and the drill it turns into its child
    raw = EV.EventTables.from_names(table, ids)
    d = started(raw)
    ev = d.update([tk(1, "GoblinDrillDig", E, 9000, 4000, 100, 101)], 101)
    assert [p.card for p in ev.opp_plays] == [drill]
    d.update([tk(1, "GoblinDrillDig", E, 3500, 26000, 100, 158)], 158)             # it digs across
    ev = d.update([tk(2, "GoblinDrill", E, 3500, 26000, 160, 161)], 161)
    assert ev.opp_plays == [] and d.card_of(2) == drill and d.how_of(2) == "child"


def test_a_champions_ability_units_are_its_children():
    table, ids, tb = vision_tables()
    assert table["children"]["LittlePrince"] == ["ChampionGuard"]
    assert table["children"]["SkeletonKing"] == ["SkeletonKingSkeleton"]
    assert table["children"]["MightyMiner"] == ["MightyMinerBomb"]
    assert "goblinstein" not in table["children"]       # its tether names the doctor of the play itself
    prince, king = ids["LittlePrince"], ids["SkeletonKing"]
    d = started(tb)
    ev = d.update([tk(1, "LittlePrince", E, 9000, 20000, 100, 101),
                   tk(2, "SkeletonKing", E, 3500, 20000, 100, 101)], 101)
    assert sorted(p.card for p in ev.opp_plays) == sorted([prince, king])
    # 10 s later both use their abilities: the Royal Guardian, and skeletons around the King
    # (classes.json reads SkeletonKingSkeleton as the class Skeleton, KataCR's `skeleton`)
    alive = [tk(1, "LittlePrince", E, 9000, 21000, 100, 301), tk(2, "SkeletonKing", E, 3500, 21000, 100, 301)]
    made = [tk(3, "ChampionGuard", E, 9000, 19000, 300, 301)] + [
        tk(4 + i, "Skeleton", E, 3500 + dx, 21000 + dy, 300, 301)
        for i, (dx, dy) in enumerate([(2500, 0), (-2500, 0), (0, 2500), (0, -2500)])]
    ev = d.update(alive + made, 301)
    assert ev.opp_plays == []                            # before: a phantom Skeletons-type play
    assert d.card_of(3) == prince and [d.card_of(u) for u in (4, 5, 6, 7)] == [king] * 4
    assert d.opp_stats.unknown == {} and d.opp_stats.children == 5


def test_the_generator_reads_morphs_champion_abilities_and_the_bomb_lookalikes():
    VSC = pytest.importorskip("tools.vision_spawn_children")
    units = {"GoblinDrillDig": {"spawn_pathfind": {"speed": 300, "morph": "GoblinDrill"}},
             "GoblinDrill": {"spawner": {"character": "Goblin"}}, "Goblin": {},
             "Miner": {"spawn_pathfind": {"speed": 650, "morph": None}},
             "LittlePrince": {}, "ChampionGuard": {}, "goblinstein": {}, "goblinstein_doctor": {},
             "Balloon": {"death_spawn": {"character": "BalloonBomb"}}, "BalloonBomb": {}}
    cards = [
        {"name": "GoblinDrill", "kind": "building", "elixir": 4, "summon_character": "GoblinDrillDig"},
        {"name": "Miner", "kind": "troop", "elixir": 3, "summon_character": "Miner"},
        {"name": "LittlePrince", "kind": "troop", "elixir": 3, "summon_character": "LittlePrince",
         "ability": {"effect": {"kind": "guard", "unit": "ChampionGuard"}}},
        {"name": "Goblinstein", "kind": "troop", "elixir": 5, "summon_character": "goblinstein",
         "second_summon": {"character": "goblinstein_doctor", "count": 1},
         "ability": {"effect": {"kind": "tether", "unit": "goblinstein_doctor"}}},
        {"name": "Balloon", "kind": "troop", "elixir": 5, "summon_character": "Balloon"},
    ]
    doc = VSC.build({"version": "cards-mini.1", "cards": cards, "units": units, "hero_forms": [],
                     "projectiles": {}, "area_effect_objects": {}}, files=["mini.json"])
    assert doc["compositions"]["GoblinDrill"] == {"GoblinDrillDig": 1, "GoblinDrill": 1}
    assert doc["compositions"]["Miner"] == {"Miner": 1}                  # its dig turns into nothing
    assert doc["compositions"]["Goblinstein"] == {"goblinstein": 1, "goblinstein_doctor": 1}
    assert doc["children"] == {"Balloon": ["BalloonBomb", "GiantSkeletonBomb"], "GoblinDrill": ["Goblin"],
                               "GoblinDrillDig": ["Goblin", "GoblinDrill"],      # + its descendants
                               "LittlePrince": ["ChampionGuard"]}


def test_every_katacr_lookalike_is_one_of_the_generators():
    VSC = pytest.importorskip("tools.vision_spawn_children")
    path = ROOT / "data" / "vision" / "classes.json"
    if not path.exists():
        pytest.skip("data/vision/classes.json (E16) is not there")
    katacr = json.loads(path.read_text(encoding="utf-8"))["katacr"]
    rides = {"RamRider"}           # rides the Ram its label names: not a look-alike (generator doc)
    also = {(u, e["class"]) for e in katacr.values() for u in e.get("also", ()) if u not in rides}
    assert also == {("BalloonBomb", "GiantSkeletonBomb"), ("BombTowerBomb", "GiantSkeletonBomb")}
    assert all(VSC.LOOKS_LIKE.get(u) == cls for u, cls in also)
    assert set(VSC.LOOKS_LIKE.values()) <= {e.get("class") for e in katacr.values()}


def test_events_imports_only_the_standard_library_and_its_package():
    tree = ast.parse((ROOT / "scripts" / "vision" / "events.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    assert names <= {"__future__", "json", "math", "re", "statistics", "collections", "dataclasses",
                     "pathlib", "typing", ".tracker", ".types"}, names
