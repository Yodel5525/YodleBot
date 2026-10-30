"""The opponent-play guards (scripts/vision/events.py module doc, GUARDS; events.LIVE_GUARDS).

Each test replays, with synthetic tracks, one way the owner's 2026-10-29 friendly turned one
enemy unit into several plays (103 plays of 27 cards for one 8-card deck), and checks the guard
that answers it against the rules as they were (EventConfig(), the golden behaviour the other
event tests pin). World millitiles, battle ticks; we are team 1 at high y, the opponent team 0.
Also: Tracker.inherit_team, public_state.OpponentElixir's unaffordable count and the runners'
--legacy-opp-plays.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import events as EV  # noqa: E402
from scripts.vision import tracker as T  # noqa: E402
from scripts.vision import types as VT  # noqa: E402

CANNON, FIREBALL, HOG, ICE_GOLEM, ICE_SPIRITS, MUSKETEER, SKELETONS, LOG = range(8)
KNIGHT, SKELETON_ARMY, ROYAL_RECRUITS, GUARDS, MINER = 8, 21, 47, 25, 26
HOG26 = range(8)
E, O, U = VT.ENEMY_TEAM, VT.OWN_TEAM, VT.UNKNOWN_TEAM


def tables() -> EV.EventTables:
    return EV.EventTables(
        compositions={CANNON: {"Cannon": 1}, HOG: {"HogRider": 1}, ICE_GOLEM: {"IceGolemite": 1},
                      ICE_SPIRITS: {"IceSpirits": 1}, MUSKETEER: {"Musketeer": 1},
                      SKELETONS: {"Skeleton": 3}, KNIGHT: {"Knight": 1},
                      SKELETON_ARMY: {"Skeleton": 15}, ROYAL_RECRUITS: {"Recruit": 6},
                      GUARDS: {"SkeletonWarrior": 3}, MINER: {"Miner": 1}},
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
    for bad in ({"min_play_age_ticks": -1}, {"young_drop_ticks": -1}, {"own_recent_ticks": -1},
                {"reid_motion_cos": 1.5}, {"deploy_zone_mt": -1.0}, {"reid_static_mt_per_tick": -1.0},
                {"own_recent_radius_mt": -1.0}):
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
    k1 = tk(1, "Knight", E, 9000, 8000, 100, 101)
    k2 = tk(2, "Knight", E, 9000, 8500, 126, 127)
    old = deriver()
    old.update([k1], 101)
    old.update([k2], 127)
    assert plays(old) == [(KNIGHT, 100)] and old.how_of(2) == "reacquired"   # the flash's play

    d = live()
    d.update([k1], 101)
    d.update([], 125)
    assert d.how_of(1) == "dropped"
    d.update([k2], 127)
    assert d.how_of(2) == "pending"                        # not re-acquired from the flash
    d.update([at(k2, 136)], 136)
    assert plays(d) == [(KNIGHT, 126)] and d.how_of(2) == "play"


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


# ---------------------------------------------------------------------------
# deploy_zone_mt: a unit first seen past the river walked there
# ---------------------------------------------------------------------------
def test_a_unit_first_seen_past_the_river_is_late_unless_a_pocket_is_open():
    hog = tk(1, "HogRider", E, 14500, 20000, 300, 301, vy=90)
    old = deriver()
    old.update([hog], 301)
    assert plays(old) == [(HOG, 300)]

    d = live()
    d.update([hog], 301)
    assert d.update([at(hog, 311, y_mt=21000.0)], 311).opp_plays == []
    assert d.how_of(1) == "late" and d.card_of(1) == HOG and d.opp_stats.late == 1
    # our right princess tower falls: the right lane is a pocket, the left one is not
    d.update([], 400, towers={"own_right": VT.TowerHp(0, 3052), "own_left": VT.TowerHp(2000, 3052)})
    assert d.fallen_towers == {"own_right"}
    right = tk(2, "HogRider", E, 14000, 19000, 410, 411)
    d.update([right], 411)
    d.update([at(right, 421)], 421)
    assert plays(d) == [(HOG, 410)]
    left = tk(3, "HogRider", E, 3500, 19000, 500, 501)
    d.update([left], 501)
    d.update([at(left, 511)], 511)
    assert plays(d) == [(HOG, 410)] and d.how_of(3) == "late" and d.opp_stats.late == 2
    # a Miner goes anywhere; a spell is never held or judged by where it lands
    miner = tk(4, "Miner", E, 3500, 25000, 600, 601)
    d.update([miner], 601)
    d.update([at(miner, 611)], 611)
    assert plays(d)[-1] == (MINER, 600)
    ev = d.update([tk(5, "FireballSpell", E, 3500, 25000, 700, 701)], 701)
    assert [p.card for p in ev.opp_plays] == [FIREBALL]
    # the river's far edge (1000 past the centre) is still in the zone
    edge = tk(6, "Knight", E, 9000, 17000, 800, 801)
    d.update([edge], 801)
    d.update([at(edge, 811)], 811)
    assert plays(d)[-1] == (KNIGHT, 800)


def test_our_unmatched_unit_first_seen_in_the_enemy_half_is_late_too():
    ours = tk(1, "HogRider", O, 4000, 9000, 300, 301)
    old = deriver()
    assert [(x.card, x.slot) for x in old.update([ours], 301).own_landings] == [(HOG, -1)]
    d = live()
    assert d.update([ours], 301).own_landings == []
    assert d.how_of(1) == "late" and d.card_of(1) == HOG and d.landing_stats.late == 1
    assert d.landing_stats.summary()["late"] == 1 and d.opp_stats.late == 0
    # an enemy princess tower down opens the pocket for us
    d.update([], 350, towers={"enemy_left": {"hp": 0, "max": 3052}})
    ev = d.update([tk(2, "HogRider", O, 4000, 9000, 360, 361)], 361)
    assert [(x.card, x.slot) for x in ev.own_landings] == [(HOG, -1)]


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
        "another label; no play from a unit first seen over 1000 mt past the river centre")
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
