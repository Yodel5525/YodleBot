"""The phantom-play rules (scripts/vision/events.py module doc, PHANTOMS; events.LIVE_PHANTOMS) and
the tracker's capture rules (scripts/vision/tracker.py module doc, CAPTURE; TrackerConfig.live).

Each test replays, with SYNTHETIC tracks, one way the owner's 2026-11-01 --tap friendlies inflated
the bot's view of the opponent: one standing enemy unit relabelled became a new play at every
label change, an evolved Skeleton's clones were plays, a Fireball was logged again as a Rocket, a
track kept the card of its first label for ever, and a lost track took another unit's boxes. Each
rule is checked against the config without it (the rules as they were). World millitiles, battle
ticks; we are team 1 at high y, the opponent team 0.
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
SKELETON_ARMY, GOBLINS, LITTLE_PRINCE, PRINCE, MIGHTY_MINER, ROCKET = 21, 30, 40, 41, 42, 43
CARDS9 = list(range(50, 59))               # nine one-unit cards C0..C8 (the deck cap)
HOG26 = range(8)
E, O, U = VT.ENEMY_TEAM, VT.OWN_TEAM, VT.UNKNOWN_TEAM


def tables() -> EV.EventTables:
    comps = {CANNON: {"Cannon": 1}, HOG: {"HogRider": 1}, ICE_GOLEM: {"IceGolemite": 1},
             ICE_SPIRITS: {"IceSpirits": 1}, MUSKETEER: {"Musketeer": 1}, SKELETONS: {"Skeleton": 3},
             SKELETON_ARMY: {"Skeleton": 15}, GOBLINS: {"Goblin": 3}, LITTLE_PRINCE: {"LittlePrince": 1},
             PRINCE: {"Prince": 1}, MIGHTY_MINER: {"MightyMiner": 1}}
    comps.update({c: {f"C{k}": 1} for k, c in enumerate(CARDS9)})
    return EV.EventTables(comps, spells={FIREBALL: {"Fireball"}, ROCKET: {"Rocket"}, LOG: {"Log"}},
                          buildings={CANNON})


def tk(uid, cls, team, x, y, first, last, vx=0.0, vy=0.0) -> VT.Track:
    return VT.Track(uid=uid, cls=cls, team=team, conf=0.9, x_mt=float(x), y_mt=float(y),
                    first_seen_tick=first, last_seen_tick=last, vx=float(vx), vy=float(vy))


def at(track, last, **kw):
    return replace(track, last_seen_tick=last, **kw)


def deriver(config=None, counter=None) -> EV.EventDeriver:
    d = EV.EventDeriver(tables(), config, own_deck=HOG26)
    d.update([], 0)                    # the battle starts on an empty board
    if counter is not None:
        d.bind_opp_elixir(counter)
    return d


def plays(d):
    return [(p.card, p.tick) for p in d.opp_plays()]


def seen(d, track, ticks, **kw):
    """Feed `track` alone on every tick of `ticks`; returns the events of each update."""
    return [d.update([at(track, t, **kw)], t) for t in ticks]


class Count:
    """A bound opponent elixir count (bind_opp_elixir): the cards in `broke` cannot be paid."""

    def __init__(self, broke=()):
        self.broke = set(broke)
        self.asked = []

    def can_pay(self, card, tick, cost=None, extra=()):
        self.asked.append((card, tick))
        return card not in self.broke


# ---------------------------------------------------------------------------
# the config
# ---------------------------------------------------------------------------
def test_the_phantom_rules_are_off_by_default_and_on_in_the_live_config():
    off = EV.EventConfig()
    assert (off.reid_doubtful, off.self_clones, off.spell_relabel, off.prefer_confirmed,
            off.reguess_ticks, off.reid_ticks, off.opp_reid_ticks) == (False, frozenset(), False, False, 0, 30, None)
    on = EV.EventConfig.live()
    assert dict(EV.LIVE_PHANTOMS) == {"opp_reid_ticks": 50, "reid_doubtful": True,
                                      "self_clones": frozenset({"Skeleton"}), "spell_relabel": True,
                                      "prefer_confirmed": True, "reguess_ticks": 15}
    assert (on.reid_doubtful, on.self_clones, on.reguess_ticks, on.reid_ticks, on.opp_reid_ticks, on.deck_size,
            on.reid_doubt_radius_mt) == (True, frozenset({"Skeleton"}), 15, 30, 50, 8, 2000.0)
    assert EV.EventConfig(self_clones="Skeleton").self_clones == frozenset({"Skeleton"})
    assert {"Skeleton", "Bat", "IceSpirits"} <= EV.ONE_HIT_CLASSES
    with pytest.raises(ValueError, match="deck_size"):
        EV.EventConfig(deck_size=0)
    with pytest.raises(ValueError, match="opp_reid_ticks"):
        EV.EventConfig(opp_reid_ticks=-1)
    with pytest.raises(ValueError, match="reguess_ticks"):
        EV.EventConfig(reguess_ticks=-1)
    # the guards' settings (and the runners' setup line) are those of LIVE_GUARDS alone
    assert set(EV.guard_settings(on)) == {"min_play_age_ticks", "reid_any_label", "deploy_zone_mt",
                                          "own_reid_cross_team", "own_recent_ticks", "on"}


# ---------------------------------------------------------------------------
# reid_doubtful: a standing unit relabelled continues its play when the new one is doubtful
# ---------------------------------------------------------------------------
def _standing_relabel(d, *, old_cls="LittlePrince", new_cls="Prince", gap=15, new_v=(0.0, 0.0),
                      old_v=(0.0, 0.0), dist=(200.0, 300.0)):
    """The owner's hero Musketeer at our bridge: a play of `old_cls` (seen 100..115, standing),
    lost, then `new_cls` standing `dist` from where it was lost, `gap` ticks later."""
    old = tk(1, old_cls, E, 3000, 14000, 100, 100, vx=old_v[0], vy=old_v[1])
    seen(d, old, range(100, 116))
    t0 = 115 + gap
    new = tk(2, new_cls, E, 3000 + dist[0], 14000 + dist[1], t0, t0, vx=new_v[0], vy=new_v[1])
    seen(d, new, range(t0, t0 + 15))
    return d


def test_a_standing_relabel_the_count_cannot_pay_continues_its_play():
    """The hero Musketeer read as Little Prince, lost, then read as Prince where it stood: the
    Prince (5) is beyond the opponent's elixir by the count, so it is the Little Prince again."""
    count = Count(broke={PRINCE})
    d = _standing_relabel(deriver(EV.EventConfig.live(), count))
    assert plays(d) == [(LITTLE_PRINCE, 100)]
    assert d.how_of(2) == "reacquired" and d.card_of(2) == LITTLE_PRINCE
    assert (d.opp_stats.doubtful, d.opp_stats.relabelled) == (1, 1)
    assert (PRINCE, 130) in count.asked                     # asked at the new track's first sighting
    # the rules as they were: no motion, no match: a new play
    for config in (EV.EventConfig.live(reid_doubtful=False), EV.EventConfig()):
        d = _standing_relabel(deriver(config, Count(broke={PRINCE})))
        assert plays(d) == [(LITTLE_PRINCE, 100), (PRINCE, 130)] and d.opp_stats.doubtful == 0


def test_a_standing_relabel_the_count_can_pay_is_a_new_play():
    """Affordable and the opponent's deck not yet full (one confirmed card): a real new play."""
    for counter in (Count(), None):
        d = _standing_relabel(deriver(EV.EventConfig.live(), counter))
        assert plays(d) == [(LITTLE_PRINCE, 100), (PRINCE, 130)] and d.how_of(2) == "play"


def test_a_card_outside_the_first_eight_confirmed_cards_is_not_doubtful():
    """Review P1-2: the first eight cards confirmed hold the early misreads (on the owner's game 2
    they left out the Hog Rider, the Musketeer and the Ice Golem). A ninth card standing where one
    of theirs was just lost is a play, with or without the count: only the count doubts a play."""
    for counter in (None, Count()):
        d = deriver(EV.EventConfig.live(), counter)
        for k, card in enumerate(CARDS9[:8]):          # eight plays, far apart, one after another
            unit = tk(10 + k, f"C{k}", E, 1000 + 2000 * k, 8000, 100 + 20 * k, 100 + 20 * k)
            seen(d, unit, range(100 + 20 * k, 116 + 20 * k))
        assert d.confirmed() == CARDS9[:8]
        ninth = tk(30, "C8", E, 15300, 8000, 265, 265)          # 10 ticks after C7 (at 15000) was lost
        seen(d, ninth, range(265, 280))
        assert d.how_of(30) == "play" and plays(d)[-1] == (CARDS9[8], 265) and d.opp_stats.doubtful == 0


def test_a_doubtful_relabel_needs_the_unit_not_moving_apart_near_and_not_a_spirit():
    count = Count(broke={PRINCE})
    # walking the other way: two units
    d = _standing_relabel(deriver(EV.EventConfig.live(), count), old_v=(0.0, 60.0), new_v=(0.0, -60.0))
    assert plays(d) == [(LITTLE_PRINCE, 100), (PRINCE, 130)]
    # farther than reid_doubt_radius_mt (2,000) from where it was lost
    d = _standing_relabel(deriver(EV.EventConfig.live(), count), dist=(1800.0, 1200.0))
    assert plays(d) == [(LITTLE_PRINCE, 100), (PRINCE, 130)]
    # an Ice Spirit that was lost hit something and is gone (KAMIKAZE_CLASSES)
    d = _standing_relabel(deriver(EV.EventConfig.live(), count), old_cls="IceSpirits")
    assert plays(d) == [(ICE_SPIRITS, 100), (PRINCE, 130)]
    # longer than reid_ticks (50)
    d = _standing_relabel(deriver(EV.EventConfig.live(), count), gap=55)
    assert plays(d) == [(LITTLE_PRINCE, 100), (PRINCE, 170)]


def test_a_real_play_dropped_where_their_skeleton_just_died_is_a_play():
    """Review P1-1: a real Hog Rider dropped 1.5 tiles from where their skeleton (of a Skeletons
    play) was lost 20 ticks before is a play, even when the count cannot pay it and the first
    eight confirmed cards leave the Hog out: a Skeleton dies on the first hit it takes
    (ONE_HIT_CLASSES), so a lost one is gone. Their Musketeer lost there would be continued."""
    def battle(lost_cls, counter, fill=True):
        d = deriver(EV.EventConfig.live(), counter)
        t = 100
        if fill:                                          # eight cards confirmed, no Hog among them
            for k, card in enumerate([SKELETONS, CANNON, MUSKETEER] + CARDS9[:5]):
                cls = sorted(d.tables.classes_of(card))[0]
                seen(d, tk(100 + k, cls, E, 1000 + 1500 * k, 3000, t, t), range(t, t + 14))
                t += 40
        lost = tk(500, lost_cls, E, 4000, 15000, t, t)
        seen(d, lost, range(t, t + 30))
        t2 = t + 30 + 20
        hog = tk(501, "HogRider", E, 4900, 16200, t2, t2)
        seen(d, hog, range(t2, t2 + 20))
        return d, t2
    for counter, fill in ((None, True), (Count(broke={HOG}), False), (Count(broke={HOG}), True)):
        d, t2 = battle("Skeleton", counter, fill)
        assert d.how_of(501) == "play" and plays(d)[-1] == (HOG, t2) and d.opp_stats.doubtful == 0
    d, t2 = battle("Musketeer", Count(broke={HOG}), False)    # the rule's own case: a standing unit
    assert d.how_of(501) == "reacquired" and d.opp_stats.doubtful == 1


def test_a_unit_missed_for_a_frame_does_not_swallow_a_play_landing_next_to_it():
    """Review P1-3: their standing Musketeer (a play) is missed on the one frame the opponent drops
    an Ice Golem 0.9 tiles away, and the count says the Ice Golem cannot be paid. The Musketeer is
    not lost: the tracker still reports it (coasting), or it is seen again before the Ice Golem's
    track is min_play_age_ticks old. The Ice Golem is a play."""
    for coasting in (True, False):
        d = deriver(EV.EventConfig.live(), Count(broke={ICE_GOLEM}))
        mus = tk(1, "Musketeer", E, 3000, 14000, 100, 100)
        seen(d, mus, range(100, 131))
        golem = tk(2, "IceGolemite", E, 3600, 14700, 131, 131)
        d.update([at(mus, 130), golem] if coasting else [golem], 131)
        for t in range(132, 160):
            d.update([at(mus, t), at(golem, t)], t)
        assert plays(d) == [(MUSKETEER, 100), (ICE_GOLEM, 131)] and d.how_of(2) == "play"
        assert d.card_of(1) == MUSKETEER and d.opp_stats.doubtful == 0


def test_a_standing_relabel_is_taken_only_once_the_new_track_is_old_enough():
    """The wait of P1-3: the relabelled track is asked again on every update until it is
    min_play_age_ticks (10) old; a flash lost before that is dropped, no play and no merge."""
    count = Count(broke={PRINCE})
    d = deriver(EV.EventConfig.live(), count)
    old = tk(1, "LittlePrince", E, 3000, 14000, 100, 100)
    seen(d, old, range(100, 116))
    new = tk(2, "Prince", E, 3200, 14300, 130, 130)
    seen(d, new, range(130, 136))
    assert d.how_of(2) == "pending"                      # too young to be taken
    seen(d, new, range(136, 145))
    assert d.how_of(2) == "reacquired" and plays(d) == [(LITTLE_PRINCE, 100)]
    d = deriver(EV.EventConfig.live(), Count(broke={PRINCE}))
    seen(d, old, range(100, 116))
    seen(d, new, range(130, 134))                        # a flash of four frames
    for t in range(134, 160):
        d.update([], t)
    assert d.how_of(2) == "dropped" and plays(d) == [(LITTLE_PRINCE, 100)] and d.opp_stats.doubtful == 0


def test_a_real_play_after_a_phantom_the_count_paid_is_not_doubtful():
    """Review P1-1 with the real count: a phantom Prince (unaffordable) drains the bar; a real Hog
    Rider then stands where their Little Prince was lost. The plain count could not pay the Hog;
    the lenient bar can_pay asks (the phantom left out) can, so the Hog is a play."""
    pytest.importorskip("numpy")
    PS = pytest.importorskip("tools.public_state")
    law = PS.rs_elixir.ElixirLaw.constants()
    catalogue = [[f"X{i}", 0, 1, 1, 0, False, 0, 0, "TROOP"] for i in range(60)]
    for card, cost in ((HOG, 4), (LITTLE_PRINCE, 3), (PRINCE, 7)):    # a Prince beyond the bar
        catalogue[card][2] = cost
    oe = PS.OpponentElixir(catalogue, law=law)
    d = deriver(EV.EventConfig.live(), oe)
    lp = tk(1, "LittlePrince", E, 3000, 14000, 100, 100)
    prince = tk(2, "Prince", E, 14000, 6000, 140, 140)
    hog = tk(3, "HogRider", E, 3300, 14300, 215, 215)
    for t in range(100, 240):
        trs = [at(lp, t)] if t < 200 else []
        if 140 <= t < 170:
            trs.append(at(prince, t))
        if t >= 215:
            trs.append(at(hog, t))
        ev = d.update(trs, t)
        oe.record(ev.opp_plays)
        oe.view(t)
    assert plays(d) == [(LITTLE_PRINCE, 100), (PRINCE, 140), (HOG, 215)] and d.how_of(3) == "play"
    assert oe.unaffordable == 2                            # the Prince, and by the plain count the real Hog
    assert oe.counter.elixir_milli(214) < 4000 <= oe._lenient.elixir_milli(214)   # plain count vs lenient bar


def test_the_reid_window_takes_a_walking_relabel_lost_for_34_ticks():
    """The hero Musketeer walking to the bridge as Mighty Miner, lost for 34 ticks, came back as
    Goblins on its line (#46 -> #50): rule 1 by the motion, inside the live 50 ticks, not 30."""
    def battle(config):
        d = deriver(config)
        walker = tk(1, "MightyMiner", E, 3000, 9000, 100, 100, vy=60.0)
        for t in range(100, 116):
            d.update([at(walker, t, y_mt=9000.0 + 60 * (t - 100))], t)
        goblin = tk(2, "Goblin", E, 3000, 9900 + 60 * 34, 149, 149, vy=60.0)
        for t in range(149, 165):
            d.update([at(goblin, t, y_mt=goblin.y_mt + 60 * (t - 149))], t)
        return d
    d = battle(EV.EventConfig.live())
    assert plays(d) == [(MIGHTY_MINER, 100)] and d.how_of(2) == "reacquired"
    d = battle(EV.EventConfig.live(opp_reid_ticks=30))
    assert plays(d) == [(MIGHTY_MINER, 100), (GOBLINS, 149)]


def test_our_own_lost_units_keep_the_old_reid_window():
    """Review P2: the wider window is the enemy's only. Our Ice Golem's track lost 32 ticks before
    our Hog tap's unit is first seen (misread as an Ice Golem) is not that unit: the tap lands."""
    def battle(config):
        d = deriver(config)
        d.note_own_tap(ICE_GOLEM, 90, slot=0, aim_x_mt=3500.0, aim_y_mt=24000.0)
        golem = tk(1, "IceGolemite", O, 3500, 24000, 100, 100)
        seen(d, golem, range(100, 121))                        # our Ice Golem lands, lost at 120
        d.note_own_tap(HOG, 140, slot=1, aim_x_mt=3800.0, aim_y_mt=24300.0)
        hog = tk(2, "IceGolemite", O, 3800, 24300, 152, 152)   # our Hog, read as an Ice Golem
        seen(d, hog, range(152, 170))
        return d
    d = battle(EV.EventConfig.live())
    landed = [x.card for x in d.own_landings() if x.land_tick is not None]
    assert d.how_of(2) == "landing" and landed == [ICE_GOLEM, HOG]
    d = battle(EV.EventConfig.live(reid_ticks=50))             # the window the review found on our side
    assert d.how_of(2) == "reacquired"


# ---------------------------------------------------------------------------
# self_clones: an evolved Skeleton's clones are not plays
# ---------------------------------------------------------------------------
def _skeletons_then_one_more(config, *, parent_cls="Skeleton", parent_card_cls="Skeleton"):
    """Their Skeletons (3, seen 100..200) and, at 160, one more Skeleton next to them (a clone)."""
    d = deriver(config)
    trio = [tk(1 + k, parent_card_cls, E, 4000 + 300 * k, 9000, 100, 100) for k in range(3)]
    clone = tk(9, "Skeleton", E, 4300, 9400, 160, 160)
    for t in range(100, 201):
        cls = parent_card_cls if t < 120 else parent_cls
        tracks = [at(x, t, cls=cls) for x in trio]
        if 160 <= t <= 180:
            tracks.append(at(clone, t))
        d.update(tracks, t)
    return d


def test_a_skeleton_next_to_their_living_skeletons_is_a_clone():
    d = _skeletons_then_one_more(EV.EventConfig.live())
    assert plays(d) == [(SKELETONS, 100)] and d.how_of(9) == "child" and d.card_of(9) == SKELETONS
    assert (d.opp_stats.clones, d.opp_stats.children) == (1, 1)
    d = _skeletons_then_one_more(EV.EventConfig.live(self_clones=()))        # the rules as they were
    assert plays(d) == [(SKELETONS, 100), (SKELETONS, 160)] and d.opp_stats.clones == 0


def test_a_clone_needs_a_parent_of_a_play_whose_card_puts_the_class_down():
    """A Cannon's track that took skeleton boxes (relabelled Skeleton) clones nothing."""
    d = _skeletons_then_one_more(EV.EventConfig.live(), parent_cls="Skeleton", parent_card_cls="Cannon")
    assert plays(d)[-1] == (SKELETONS, 160) and d.how_of(9) == "play" and d.opp_stats.clones == 0


# ---------------------------------------------------------------------------
# spell_relabel: a look-alike spell label on the same flight is the same cast
# ---------------------------------------------------------------------------
def _flight(config, second="Rocket", first_frames=5, second_frames=18):
    """Their Fireball flying 400 mt a tick from t100, lost, and a box of `second` on its path 4
    ticks later (the owner's Fireball logged again as a Rocket on its landing)."""
    d = deriver(config)
    ball = tk(1, "Fireball", E, 9000, 3000, 100, 100, vy=400.0)
    amended = []
    for t in range(100, 100 + first_frames):
        amended += d.update([at(ball, t, y_mt=3000.0 + 400 * (t - 100))], t).opp_amended
    t0 = 100 + first_frames + 3
    land = tk(2, second, E, 9000, 3000 + 400 * (t0 - 100), t0, t0)
    for t in range(t0, t0 + second_frames):
        amended += d.update([at(land, t)], t).opp_amended
    return d, amended


def test_a_rocket_box_on_a_fireballs_flight_is_the_fireball():
    d, amended = _flight(EV.EventConfig.live(), first_frames=15, second_frames=3)
    assert plays(d) == [(FIREBALL, 100)] and d.how_of(2) == "merged" and amended == []
    assert d.opp_stats.spell_relabels == 1
    # seen longer as a Rocket: the play is the Rocket's (amended once)
    d, amended = _flight(EV.EventConfig.live())
    assert plays(d) == [(ROCKET, 100)]
    assert [(a.old.card, a.new.card, a.new.tick) for a in amended] == [(FIREBALL, ROCKET, 100)]
    # the rules as they were: two casts
    d, _ = _flight(EV.EventConfig.live(spell_relabel=False))
    assert plays(d) == [(FIREBALL, 100), (ROCKET, 108)]


def test_a_spell_of_another_look_alike_group_is_a_cast_of_its_own():
    """Fireball then Log (a Hog 2.6 combo) on the same spot: two casts."""
    d, amended = _flight(EV.EventConfig.live(), second="Log")
    assert plays(d) == [(FIREBALL, 100), (LOG, 108)] and amended == []


def test_a_look_alike_spell_on_another_lane_is_a_cast_of_its_own():
    """Review P1 (live): their Fireball on our left tower, their Rocket on our right tower 12 ticks
    later, 11 tiles apart: two casts. A look-alike label joins a cast on its flight only."""
    for config in (EV.EventConfig.live(), EV.EventConfig.live(spell_relabel=False)):
        d = deriver(config)
        ball = tk(1, "Fireball", E, 3500, 26000, 100, 100)
        rocket = tk(2, "Rocket", E, 14500, 26000, 116, 116)
        amended = []
        for t in range(100, 105):
            amended += d.update([at(ball, t)], t).opp_amended
        for t in range(116, 140):
            amended += d.update([at(rocket, t)], t).opp_amended
        assert plays(d) == [(FIREBALL, 100), (ROCKET, 116)] and d.how_of(2) == "play" and amended == []


def test_a_play_amended_on_the_update_that_reports_it_is_reported_once():
    """Review P2 (live): a Fireball box (2 frames) and a Rocket box (15 frames) on one flight, no
    team yet, both classified on one update: the play is reported as a Rocket, with no amendment,
    so the memory and the count hold one play."""
    rs_memory = pytest.importorskip("tools.rs_memory")
    d = deriver(EV.EventConfig.live())
    mem = rs_memory.PublicMemory()
    ball = tk(1, "Fireball", U, 9000, 3000, 100, 100, vy=400.0)
    rocket = tk(2, "Rocket", U, 9100, 3000, 100, 100, vy=400.0)
    reported, amended = [], []
    for t in range(100, 116):
        trs = [at(rocket, t, y_mt=3000.0 + 400 * (t - 100))]
        if t < 102:
            trs.append(at(ball, t, y_mt=3000.0 + 400 * (t - 100)))
        ev = d.update(trs, t)
        reported += ev.opp_plays
        amended += ev.opp_amended
        EV.record_plays(mem, ev.opp_plays)
        EV.amend_plays(mem, ev.opp_amended)
    assert [(p.card, p.tick) for p in reported] == [(ROCKET, 100)] and amended == []
    assert len(mem) == 1 and plays(d) == [(ROCKET, 100)] and d.opp_stats.amended == 0


# ---------------------------------------------------------------------------
# prefer_confirmed: an ambiguous play takes a card they were seen to play
# ---------------------------------------------------------------------------
def test_an_ambiguous_play_takes_a_confirmed_card():
    def battle(config):
        d = deriver(config)
        trio = [tk(1 + k, "Skeleton", E, 4000 + 300 * k, 9000, 100, 100) for k in range(3)]
        for t in range(100, 131):
            d.update([at(x, t) for x in trio], t)
        five = [tk(10 + k, "Skeleton", E, 12000 + 300 * k, 9000, 500, 500) for k in range(5)]
        for t in range(500, 520):
            d.update([at(x, t) for x in five], t)
        return d
    assert plays(battle(EV.EventConfig.live())) == [(SKELETONS, 100), (SKELETONS, 500)]
    assert plays(battle(EV.EventConfig.live(prefer_confirmed=False))) == [(SKELETONS, 100),
                                                                          (SKELETON_ARMY, 500)]


# ---------------------------------------------------------------------------
# reguess_ticks: a card follows its track's label and team; a play is amended
# ---------------------------------------------------------------------------
def _relabelled(config, *, first="MightyMiner", then="HogRider", switch=111, back=None, upto=141,
                card_cls=None):
    """Their unit first read as `first` (a play at 100), read as `then` from `switch` (and as
    `first` again from `back`). Returns the deriver, the amendments and card_for per tick."""
    d = deriver(config)
    unit = tk(1, card_cls or first, E, 2000, 13000, 100, 100)
    amended, card = [], {}
    for t in range(100, upto):
        cls = first if t < switch or (back is not None and t >= back) else then
        if card_cls is not None and t < switch:
            cls = card_cls
        tr = at(unit, t, cls=cls)
        amended += d.update([tr], t).opp_amended
        card[t] = d.card_for(tr)
    return d, amended, card


def test_their_hog_first_read_as_a_mighty_miner_becomes_a_hog():
    """The Hog misread as Mighty Miner for its first 11 frames (game 2, t1795): 15 ticks of Hog
    Rider labels re-guess the card, and the play is amended once its Hog Rider frames are 15 more
    than its Mighty Miner ones (t136)."""
    d, amended, card = _relabelled(EV.EventConfig.live())
    assert card[125] == MIGHTY_MINER and card[126] == HOG and d.card_of(1) == HOG
    assert plays(d) == [(HOG, 100)]
    [a] = amended
    assert (a.old.card, a.new.card, a.old.tick, a.new.tick, a.old.uids) == (MIGHTY_MINER, HOG, 100, 100, [1])
    assert d.opp_stats.amended == 1 and d.opp_stats.reguessed == 1
    # the rules as they were: the first label's card for ever
    d, amended, card = _relabelled(EV.EventConfig.live(reguess_ticks=0))
    assert card[140] == MIGHTY_MINER and plays(d) == [(MIGHTY_MINER, 100)] and amended == []


def test_a_short_flicker_keeps_the_card():
    d, amended, card = _relabelled(EV.EventConfig.live(), back=121)      # Hog Rider for 10 ticks
    assert set(card.values()) == {MIGHTY_MINER} and amended == [] and d.opp_stats.reguessed == 0


def test_the_board_follows_the_label_before_the_play_is_amended():
    """Seen 30 ticks as Mighty Miner, then as Hog Rider: the board's card follows after 15 ticks
    (t145), the play only once its Hog Rider frames are 15 more than its Mighty Miner ones (45, at
    t174): no flip-flop between two labels seen about as often."""
    d, amended, card = _relabelled(EV.EventConfig.live(), switch=130, upto=174)
    assert card[144] == MIGHTY_MINER and card[145] == HOG and amended == [] and plays(d) == [(MIGHTY_MINER, 100)]
    d, amended, card = _relabelled(EV.EventConfig.live(), switch=130, upto=175)
    assert [a.new.card for a in amended] == [HOG] and plays(d) == [(HOG, 100)]


def test_a_building_play_is_not_amended_to_a_troop():
    """Their Cannon's track took a skeleton's boxes (read as Skeleton for 15 ticks and more): the
    board names the track a Skeleton, the play stays a Cannon."""
    d, amended, card = _relabelled(EV.EventConfig.live(), first="Cannon", then="Skeleton", upto=170)
    assert card[169] == SKELETONS and plays(d) == [(CANNON, 100)] and amended == []


def test_one_unit_of_a_play_relabelled_does_not_amend_the_play():
    d = deriver(EV.EventConfig.live())
    trio = [tk(1 + k, "Skeleton", E, 4000 + 300 * k, 9000, 100, 100) for k in range(3)]
    amended = []
    for t in range(100, 160):
        tracks = [at(x, t, cls="Goblin" if (x.uid == 2 and t >= 120) else "Skeleton") for x in trio]
        amended += d.update(tracks, t).opp_amended
    assert d.card_of(2) == GOBLINS and d.card_of(1) == SKELETONS
    assert plays(d) == [(SKELETONS, 100)] and amended == []


def test_a_track_on_the_other_team_now_takes_that_sides_card_without_amending():
    """Our Hog's track (a landing) taken over by their Skeleton: the tracker's team is theirs now,
    so the board's card is their Skeleton's, not our Hog's (game 2, t444: a second 'enemy Hog')."""
    d = deriver(EV.EventConfig.live())
    d.note_own_tap(HOG, 50, slot=0, aim_x_mt=9500.0, aim_y_mt=26500.0)
    hog = tk(5, "HogRider", O, 9500, 26300, 72, 72)
    amended = []
    for t in range(72, 90):
        amended += d.update([at(hog, t)], t).opp_amended
    assert d.how_of(5) == "landing" and d.card_of(5) == HOG
    taken = at(hog, 90, cls="Skeleton", team=E)
    amended += d.update([taken], 90).opp_amended
    assert d.card_for(taken) == SKELETONS and amended == [] and plays(d) == []
    # the rules as they were: still our Hog
    d = deriver(EV.EventConfig.live(reguess_ticks=0))
    d.note_own_tap(HOG, 50, slot=0, aim_x_mt=9500.0, aim_y_mt=26500.0)
    for t in range(72, 90):
        d.update([at(hog, t)], t)
    d.update([taken], 90)
    assert d.card_for(taken) == HOG


# ---------------------------------------------------------------------------
# the amendments reach the memory, the elixir count and the frames
# ---------------------------------------------------------------------------
def _amend(old_card, new_card, tick=20):
    old = VT.OppPlay(card=old_card, tick=tick, exec_tick=tick, uids=[1])
    return VT.OppAmend(old, replace(old, card=new_card, uids=[1]))


def test_events_carry_amendments_and_old_frames_still_read():
    ev = VT.Events([], [], [_amend(10, 2)])
    d = ev.to_dict()
    assert d["opp_amended"] == [{"old": {"card": 10, "tick": 20, "exec_tick": 20, "x_mt": None, "y_mt": None,
                                         "kind": "unit", "uids": [1]},
                                 "new": {"card": 2, "tick": 20, "exec_tick": 20, "x_mt": None, "y_mt": None,
                                         "kind": "unit", "uids": [1]}}]
    assert VT.Events.from_dict(d) == ev
    assert "opp_amended" not in VT.Events().to_dict()                 # frames as they were
    assert VT.Events.from_dict({"opp_plays": [], "own_landings": []}).opp_amended == []


def test_public_memory_retracts_an_amended_play():
    rs_memory = pytest.importorskip("tools.rs_memory")
    mem = rs_memory.PublicMemory()
    EV.record_plays(mem, [VT.OppPlay(card=10, tick=20, exec_tick=20), VT.OppPlay(card=3, tick=40, exec_tick=40)])
    assert EV.amend_plays(mem, [_amend(10, 2)]) == 1
    view = mem.view(100)
    assert view["last"][0] == [(4, 20), (5, 40)] and view["seen"][0] == {4, 5}
    assert mem.retract(0, 12, 20) is False and len(mem) == 2
    assert EV.opp_cards([VT.OppPlay(card=2, tick=20), VT.OppPlay(card=-1, tick=30), VT.OppPlay(card=3, tick=40),
                         VT.OppPlay(card=2, tick=50)]) == [2, 3]


def test_the_opponent_elixir_count_takes_an_amendment_and_answers_the_deriver():
    pytest.importorskip("numpy")
    PS = pytest.importorskip("tools.public_state")
    law = PS.rs_elixir.ElixirLaw.constants()
    rows = (("Cannon", 3, "BUILDING"), ("Fireball", 4, "SPELL"), ("HogRider", 4, "TROOP"),
            ("IceGolemite", 2, "TROOP"), ("IceSpirits", 1, "TROOP"), ("Musketeer", 4, "TROOP"),
            ("Skeletons", 1, "TROOP"), ("Log", 2, "SPELL"), ("Knight", 3, "TROOP"),
            ("Archer", 3, "TROOP"), ("Giant", 5, "TROOP"))
    catalogue = [[n, 0, cost, 1, 0, False, 0, 0, kind] for n, cost, kind in rows]
    giant, skeletons = 10, 6
    oe = PS.OpponentElixir(catalogue, law=law)
    oe.record([VT.OppPlay(card=giant, tick=10, exec_tick=10), VT.OppPlay(card=giant, tick=20, exec_tick=20)])
    assert oe.view(30)[0] < 1000 and oe.unaffordable == 1      # the second Giant: a phantom, the bar empty
    # can_pay asks the lenient bar, which leaves the phantom out (~1.5 at 30, ~2.4 by 80): with the
    # count itself a real play after a phantom reads as unpaid (review P1-1)
    assert oe.can_pay(skeletons, 30) is True and oe.can_pay(giant, 30) is False
    assert oe.can_pay(giant, 80) is False and oe.can_pay(skeletons, 80) is True
    assert oe.can_pay(skeletons, 80, extra=[(22, 0, None)]) is False      # with a Cannon not yet recorded
    assert oe.can_pay(999, 80) is None and oe.can_pay(999, 80, cost=0) is True
    oe.amend([_amend(giant, skeletons)])              # paid already: the count is rebuilt
    assert oe.unaffordable == 0 and oe.view(30)[1] is True
    assert oe.summary() == {"recorded": 2, "unaffordable": 0, "unpriced": 0, "unknown": 0, "waiting": 0,
                            "exact": True, "amended": 1}
    later = PS.OpponentElixir(catalogue, law=law)
    later.record([VT.OppPlay(card=giant, tick=900, exec_tick=900)])      # not due yet: waiting
    later.amend([_amend(giant, skeletons, tick=900)])
    assert later.summary()["waiting"] == 1 and later._waiting == [(900, skeletons, None)]
    unpriced = PS.OpponentElixir(catalogue, law=law)   # review P2: an unpriced play amended
    unpriced.record([VT.OppPlay(card=999, tick=10, exec_tick=10)])
    assert unpriced.unpriced == 1 and unpriced.exact is False
    unpriced.amend([VT.OppAmend(VT.OppPlay(card=999, tick=10, exec_tick=10),
                                VT.OppPlay(card=skeletons, tick=10, exec_tick=10))])
    assert unpriced.unpriced == 0 and unpriced.recorded == 1 and unpriced.view(30)[1] is True
    obs = PS.LiveObserver(catalogue, law=law)          # the live loop's path: events(vf.events)
    obs.events(VT.Events([VT.OppPlay(card=giant, tick=10, exec_tick=10)], []))
    obs.events(VT.Events([], [], [_amend(giant, skeletons, tick=10)]))
    assert obs.opp.recorded == 1 and obs.opp.amended == 1


def test_the_live_runner_binds_the_count_and_applies_amendments(tmp_path):
    """vision_play: at battle start the deriver is given the battle's count (None for legacy rows,
    which have no observer); an amendment takes the old play out of the memory and puts the new
    one in, and the opponent's cards are rebuilt from the deriver's plays."""
    pytest.importorskip("numpy")
    from tests.pipeline import test_vision_play as TVP

    giant = VT.OppPlay(card=10, tick=104, exec_tick=104, x_mt=4000.0, y_mt=8000.0, uids=[1])
    cannon = replace(giant, card=0)

    class Deriver:
        def __init__(self):
            self.bound = []

        def bind_opp_elixir(self, count):
            self.bound.append(count)

        def opp_plays(self):
            return [cannon]

    script = TVP.standard_script(opp_at={104: [giant]})
    for t, vf in script:
        if getattr(vf, "tick", None) == 110:
            vf.events = VT.Events([], [], [VT.OppAmend(giant, cannon)])
    perception = TVP.FakePerception(script)
    perception.deriver = Deriver()
    captured = {}
    TVP.run_script(tmp_path, script, perception=perception,
                   decider=TVP.make_decider(TVP.FakePolicy(()), captured=captured))
    assert perception.deriver.bound == [None]
    views, known = dict(captured["views"]), dict(captured["opp_cards"])
    assert views[108]["last"][0] == [(12, 104)] and known[108] == [10]
    assert views[110]["last"][0] == [(2, 104)] and views[110]["seen"][0] == {2} and known[110] == [0]


# ---------------------------------------------------------------------------
# the tracker: CAPTURE
# ---------------------------------------------------------------------------
def test_the_capture_rules_are_off_by_default_and_on_in_the_live_config():
    off, on = T.TrackerConfig(), T.TrackerConfig.live()
    assert (off.contra_cost, off.coast_strict, off.first_velocity_alpha) == (0.0, False, 1.0)
    assert T.LIVE_TRACKER == {"contra_cost": 1.0, "coast_strict": True, "first_velocity_alpha": 0.5}
    assert (on.contra_cost, on.coast_strict, on.first_velocity_alpha, on.contra_side_prob) == (1.0, True, 0.5, 0.85)
    assert T.TrackerConfig.live(contra_cost=2.0).contra_cost == 2.0
    with pytest.raises(ValueError):
        T.TrackerConfig(first_velocity_alpha=0.0)
    with pytest.raises(ValueError):
        T.TrackerConfig(contra_side_prob=0.4)


def _obs(cls, x, y, side):
    return T.Observation(cls, float(x), float(y), 0.9, side_prob=side)


def test_two_crossing_units_keep_their_sides_boxes():
    """Our Skeleton and their Barbarian cross (game 1, t2169): with the team penalty at 0.24, each
    track took the other's box, and the Barbarian became ours. contra_cost keeps them."""
    def run(config):
        trk = T.Tracker(config, solver="greedy", own_classes={"Skeleton"})
        for t in range(1, 4):
            trk.update([_obs("Skeleton", 5000, 20000, 0.1), _obs("Barbarian", 5950, 20000, 0.9)], t)
        sk, barb = trk.last_assignment
        assert trk.team_of(trk.state(sk)) == O and trk.team_of(trk.state(barb)) == E
        trk.update([_obs("Skeleton", 5900, 20000, 0.1), _obs("Barbarian", 5050, 20000, 0.9)], 4)
        return trk.last_assignment == [sk, barb]
    assert run(T.TrackerConfig.live()) is True
    assert run(T.TrackerConfig()) is False                      # the swap, as it was


def test_a_coasting_track_takes_no_other_sides_box_of_a_new_label():
    """Our Hog's track, unseen for 13 ticks, and their Ice Golem's box inside its grown gate (game
    2, t2799): the box starts a track of its own. A box of a label the track showed, or one on our
    side, still continues it."""
    def run(config, cls="IceGolemite", side=0.9):
        trk = T.Tracker(config, solver="greedy", own_classes={"HogRider", "IceGolemite"})
        for t in range(1, 6):
            trk.update([_obs("HogRider", 3000, 20000 - 50 * t, 0.1)], t)
        [hog] = trk.last_assignment
        assert trk.team_of(trk.state(hog)) == O
        trk.update([_obs(cls, 3000, 19000, side)], 18)
        return trk.last_assignment == [hog]
    assert run(T.TrackerConfig.live()) is False
    assert run(T.TrackerConfig()) is True                       # as it was: the golem became our Hog
    assert run(T.TrackerConfig.live(), cls="HogRider") is True  # its own label, the other side
    assert run(T.TrackerConfig.live(), side=0.1) is True        # a new label, our side


def test_the_first_velocity_estimate_is_damped():
    for config, vx in ((T.TrackerConfig(), 150.0), (T.TrackerConfig.live(), 75.0)):
        trk = T.Tracker(config, solver="greedy")
        trk.update([_obs("IceGolemite", 1000, 9000, 0.9)], 1)
        [tr] = trk.update([_obs("IceGolemite", 1150, 9000, 0.9)], 2)
        assert tr.vx == pytest.approx(vx)
