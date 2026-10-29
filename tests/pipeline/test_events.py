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
    assert table["source"]["file"].endswith("cards.json")
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
