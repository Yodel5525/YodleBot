"""V03: the position-first tracker with label and team voting (scripts/vision/tracker.py).

Synthetic trajectories in world millitiles, no detector and no images. Every
association test runs on both solvers: greedy always, Hungarian (scipy) when
scipy is installed. test_auto_falls_back_to_greedy_without_scipy hides scipy
to check the fallback is chosen when it is missing.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import tracker as T  # noqa: E402
from scripts.vision import types as VT  # noqa: E402

SOLVERS = [
    pytest.param("greedy", id="greedy"),
    pytest.param("scipy", id="scipy",
                 marks=pytest.mark.skipif(not T.scipy_available(), reason="scipy is not installed")),
]

HOG26_CLASSES = {"cannon", "fireball", "hog-rider", "ice-golem", "ice-spirit", "musketeer",
                 "skeleton", "the-log"}

# scipy's documented example: the optimum is rows (0, 1, 2) -> columns (1, 0, 2),
# total 5; cheapest-pair-first takes (1, 1) = 0 and ends at the diagonal, total 6.
COST_3X3 = [[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]


def ob(cls: str, x: float, y: float, side: float = 0.5, conf: float = 1.0, **kw) -> T.Observation:
    return T.Observation(cls, float(x), float(y), conf, side, **kw)


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------
def test_assign_greedy_takes_the_cheapest_pair_first():
    assert T.assign(COST_3X3, "greedy") == [(0, 0), (1, 1), (2, 2)]
    # (0, 0) = 1 is taken first, which leaves row 1 only the forbidden pair
    assert T.assign([[1.0, 2.0], [2.0, T.FORBIDDEN]], "greedy") == [(0, 0)]


def test_assign_hungarian_finds_the_optimum():
    pytest.importorskip("scipy")
    assert T.assign(COST_3X3, "scipy") == [(0, 1), (1, 0), (2, 2)]
    # the optimum uses both rows (total 4) rather than the cheapest single pair
    assert T.assign([[1.0, 2.0], [2.0, T.FORBIDDEN]], "scipy") == [(0, 1), (1, 0)]


@pytest.mark.parametrize("solver", SOLVERS)
def test_assign_edge_cases(solver):
    assert T.assign([], solver) == []
    assert T.assign([[], []], solver) == []
    assert T.assign([[T.FORBIDDEN, T.FORBIDDEN]], solver) == []
    assert T.assign([[5.0, 1.0, 9.0]], solver) == [(0, 1)]
    assert T.assign([[3.0], [1.0], [2.0]], solver) == [(1, 0)]


def test_resolve_solver_rejects_unknown_names():
    with pytest.raises(ValueError):
        T.resolve_solver("hungarian")
    assert T.resolve_solver("greedy") == "greedy"


def test_auto_falls_back_to_greedy_without_scipy(monkeypatch):
    # A None entry in sys.modules makes `from scipy.optimize import ...` raise
    # ImportError whether or not scipy is installed.
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    assert not T.scipy_available()
    assert T.resolve_solver("auto") == "greedy"
    with pytest.raises(ImportError):
        T.resolve_solver("scipy")
    assert T.assign(COST_3X3) == [(0, 0), (1, 1), (2, 2)]
    tracker = T.Tracker()
    assert tracker.solver == "greedy"
    tracker.update([ob("knight", 9000, 24000)], 0)
    out = tracker.update([ob("knight", 9000, 23900)], 2)
    assert [t.uid for t in out] == [1]


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("solver", SOLVERS)
def test_label_flicker_keeps_one_track_and_the_majority_label(solver):
    tracker = T.Tracker(solver=solver)
    flicker = {4, 9, 10, 17}            # at most two wrong frames in a row
    uids, labels = set(), []
    for k in range(25):
        cls = "valkyrie" if k in flicker else "knight"
        out = tracker.update([ob(cls, 9000, 24000 - 100 * k)], 2 * k)
        if k == 0:
            assert out == []            # tentative until min_hits = 2
            continue
        assert len(out) == 1
        uids.add(out[0].uid)
        labels.append(out[0].cls)
    assert uids == {1}
    assert labels == ["knight"] * 24


@pytest.mark.parametrize("solver", SOLVERS)
def test_label_changes_only_after_n_agreeing_frames(solver):
    tracker = T.Tracker(solver=solver)
    got = []
    for k, cls in enumerate(["valkyrie", "knight", "knight", "knight", "knight"]):
        out = tracker.update([ob(cls, 9000, 24000)], 2 * k)
        got.append(out[0].cls if out else None)
    # weights after the 3rd knight: knight 1 + .97 + .9409 = 2.9109, valkyrie .97^3 = .9127
    assert got == [None, "valkyrie", "valkyrie", "knight", "knight"]


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_streak_alone_does_not_beat_the_majority(solver):
    tracker = T.Tracker(solver=solver)
    seq = ["knight"] * 10 + ["valkyrie"] * 8
    got = []
    for k, cls in enumerate(seq):
        got.append(tracker.update([ob(cls, 9000, 24000)], 2 * k)[-1:])
    labels = [o[0].cls if o else None for o in got]
    # After m valkyries: valkyrie = (1 - .97^m) / .03, knight = 8.7525 * .97^m.
    # m = 7: 6.401 < 7.072, m = 8: 7.209 > 6.860.
    assert labels[10:17] == ["knight"] * 7
    assert labels[17] == "valkyrie"
    assert {o[0].uid for o in got if o} == {1}


# ---------------------------------------------------------------------------
# association
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("solver", SOLVERS)
def test_crossing_tracks_keep_their_ids(solver):
    """Two knights pass each other 10 mt apart at 100 mt/tick, one frame per 4 ticks.

    At k = 8 they have just crossed: A is at 7200, B at 7000. Matching on the
    last positions alone would swap them (6800 -> 7000 and 7400 -> 7200 are
    200 mt, the true moves are 400 mt); the velocity prediction does not. A's
    label also flickers on that frame.
    """
    tracker = T.Tracker(solver=solver)
    for k in range(16):
        ax, bx = 4000 + 400 * k, 10200 - 400 * k
        a_cls = "valkyrie" if k == 8 else "knight"
        out = tracker.update([ob(a_cls, ax, 20000), ob("knight", bx, 20010)], 4 * k)
        if k == 0:
            assert out == []
            continue
        by_uid = {t.uid: t for t in out}
        assert sorted(by_uid) == [1, 2]
        assert by_uid[1].x_mt == pytest.approx(ax)
        assert by_uid[1].y_mt == pytest.approx(20000)
        assert by_uid[2].x_mt == pytest.approx(bx)
        assert by_uid[2].y_mt == pytest.approx(20010)
        assert by_uid[1].cls == "knight" and by_uid[2].cls == "knight"
    assert tracker.state(1).vx == pytest.approx(100.0)
    assert tracker.state(2).vx == pytest.approx(-100.0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_new_units_next_to_a_tracked_one_start_new_tracks(solver):
    tracker = T.Tracker(solver=solver)
    tracker.update([ob("hog-rider", 3500, 20000)], 0)
    tracker.update([ob("hog-rider", 3500, 19900)], 2)
    skeletons = [ob("skeleton", 9000, 25000), ob("skeleton", 9300, 25000), ob("skeleton", 9150, 25300)]
    out = tracker.update([ob("hog-rider", 3500, 19800)] + skeletons, 4)
    assert [t.uid for t in out] == [1]                  # the skeletons are still tentative
    out = tracker.update([ob("hog-rider", 3500, 19700)] + list(reversed(skeletons)), 6)
    by_uid = {t.uid: (t.cls, t.x_mt, t.y_mt) for t in out}
    assert by_uid == {1: ("hog-rider", pytest.approx(3500), pytest.approx(19700)),
                      2: ("skeleton", pytest.approx(9000), pytest.approx(25000)),
                      3: ("skeleton", pytest.approx(9300), pytest.approx(25000)),
                      4: ("skeleton", pytest.approx(9150), pytest.approx(25300))}


def test_velocity_is_clamped():
    tracker = T.Tracker(solver="greedy")
    tracker.update([ob("knight", 5000, 20000)], 0)
    tracker.update([ob("knight", 6000, 20000)], 1)      # 1000 mt in one tick: inside the gate (1050)
    st = tracker.state(1)
    assert st.vx == pytest.approx(150.0)                # max_speed_mt_per_tick
    assert st.vy == pytest.approx(0.0)


def test_a_backwards_tick_is_clamped():
    tracker = T.Tracker(solver="greedy")
    tracker.update([ob("knight", 5000, 20000)], 10)
    out = tracker.update([ob("knight", 5000, 20000)], 5)
    assert tracker.backwards_ticks == 1
    assert [(t.first_seen_tick, t.last_seen_tick) for t in out] == [(10, 10)]


# ---------------------------------------------------------------------------
# team
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("solver", SOLVERS)
def test_team_is_decided_then_held_with_hysteresis(solver):
    """Born on the river (no board-half cue). The side classifier says enemy
    (0.9, clipped to +2 per frame) seven times, then own (0.1, -2) nine times.
    Evidence: 2, 4 (decided), 6, 8, 10, 12, 12 (cap), 10, 8, 6, 4, 2, 0, -2, -4,
    -6: the team flips only on the last frame, at -flip_llr."""
    tracker = T.Tracker(solver=solver)
    sides = [0.9] * 7 + [0.1] * 9
    teams = []
    for k, p in enumerate(sides):
        out = tracker.update([ob("knight", 9000, 16000, side=p)], 2 * k)
        if out:
            teams.append(out[0].team)
    assert teams == [VT.ENEMY_TEAM] * 14 + [VT.OWN_TEAM]
    st = tracker.state(1)
    assert st.team_llr == pytest.approx(-6.0)
    assert st.cues == {"cnn": pytest.approx(7 * 2.0 - 9 * 2.0)}


@pytest.mark.parametrize("solver", SOLVERS)
def test_noisy_side_classifier_gives_a_stable_team(solver):
    tracker = T.Tracker(solver=solver)
    sides = [0.8, 0.8, 0.3, 0.8, 0.45, 0.8, 0.2, 0.8] * 3
    teams = []
    for k, p in enumerate(sides):
        out = tracker.update([ob("knight", 9000, 10000 + 50 * k, side=p)], 2 * k)
        if out:
            teams.append(out[0].team)
    # birth: +1 (enemy half) + ln 4; the second frame decides; one-frame dissent never flips it
    assert teams == [VT.ENEMY_TEAM] * (len(sides) - 1)


@pytest.mark.parametrize("solver", SOLVERS)
def test_own_tap_claims_the_unit_it_produced(solver):
    tracker = T.Tracker(solver=solver)
    tracker.note_own_tap({"hog-rider"}, 3500, 18000, tick=100)
    tracker.update([ob("hog-rider", 3500, 18200), ob("hog-rider", 14500, 13000)], 124)
    out = tracker.update([ob("hog-rider", 3500, 18000), ob("hog-rider", 14500, 13200)], 126)
    teams = {t.uid: t.team for t in out}
    assert teams == {1: VT.OWN_TEAM, 2: VT.UNKNOWN_TEAM}     # -1 - 6 = -7 vs +1
    assert tracker.state(1).cues == {"spawn": -1.0, "tap": -6.0}
    assert tracker.state(2).tap_claimed is False

    # outside the tap window (100 + 60 < 200): not ours by tap
    tracker.update([ob("hog-rider", 3500, 18100)], 200)
    out = tracker.update([ob("hog-rider", 3500, 18000)], 202)
    assert [(t.uid, t.team) for t in out] == [(3, VT.UNKNOWN_TEAM)]
    assert tracker.state(3).tap_claimed is False


@pytest.mark.parametrize("solver", SOLVERS)
def test_one_tap_claims_at_most_max_units(solver):
    tracker = T.Tracker(solver=solver)
    tracker.note_own_tap({"skeleton"}, 9000, 20000, tick=0, max_units=3)
    pts = [(8800, 20000), (9200, 20000), (9000, 20300), (9000, 19700)]
    tracker.update([ob("skeleton", x, y) for x, y in pts], 22)
    out = tracker.update([ob("skeleton", x, y) for x, y in pts], 24)
    assert [tracker.state(u).tap_claimed for u in (1, 2, 3, 4)] == [True, True, True, False]
    assert [t.team for t in out] == [VT.OWN_TEAM] * 3 + [VT.UNKNOWN_TEAM]


@pytest.mark.parametrize("solver", SOLVERS)
def test_deck_membership_overrides_the_side_classifier(solver):
    tracker = T.Tracker(solver=solver, own_classes=HOG26_CLASSES)
    for k in range(2):
        out = tracker.update([ob("pekka", 4000, 22000, side=0.1), ob("musketeer", 12000, 22000, side=0.2)], 2 * k)
    teams = {t.cls: t.team for t in out}
    # pekka: not in our deck, so the enemy's whatever the classifier says;
    # musketeer: -1 (our half) + 2 ln(0.25) = -3.77 -> ours
    assert teams == {"pekka": VT.ENEMY_TEAM, "musketeer": VT.OWN_TEAM}
    assert tracker.state(1).vote_team == VT.OWN_TEAM       # the vote alone would say ours


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_young_track_is_claimed_once_its_label_is_fixed(solver):
    """Our Ice Golem's first frame reads 'golem' (not in our deck): the deck rule
    calls it an enemy until the label vote switches to ice-golem; the tap claims
    it on its second frame because ice-golem is then seen."""
    tracker = T.Tracker(solver=solver, own_classes=HOG26_CLASSES)
    tracker.note_own_tap({"ice-golem"}, 9000, 20000, tick=0)
    got = []
    for k, cls in enumerate(["golem", "ice-golem", "ice-golem", "ice-golem"]):
        out = tracker.update([ob(cls, 9000, 20000 - 30 * k)], 30 + 2 * k)
        got.append((out[0].cls, out[0].team) if out else None)
    assert got == [None, ("golem", VT.ENEMY_TEAM), ("golem", VT.ENEMY_TEAM), ("ice-golem", VT.OWN_TEAM)]
    assert tracker.state(1).tap_claimed is True


@pytest.mark.parametrize("solver", SOLVERS)
def test_hp_bar_colour_votes_and_hp_is_kept(solver):
    tracker = T.Tracker(solver=solver)
    tracker.update([ob("knight", 4000, 16000, bar_team=VT.ENEMY_TEAM, hp_frac=0.4),
                    ob("knight", 14000, 16000, bar_team=VT.OWN_TEAM)], 0)
    out = tracker.update([ob("knight", 4000, 16000, bar_team=VT.ENEMY_TEAM),
                          ob("knight", 14000, 16000, bar_team=VT.OWN_TEAM, hp_frac=1.5)], 2)
    got = {t.uid: (t.team, t.hp_frac) for t in out}
    assert got == {1: (VT.ENEMY_TEAM, 0.4), 2: (VT.OWN_TEAM, 1.0)}   # 2 x 2.5 = 5 >= 3; hp clipped


# ---------------------------------------------------------------------------
# life cycle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("solver", SOLVERS)
def test_a_one_frame_phantom_is_never_reported(solver):
    tracker = T.Tracker(solver=solver)
    assert tracker.update([ob("knight", 5000, 20000)], 0) == []
    assert tracker.update([], 2) == []
    assert len(tracker.states()) == 1
    assert tracker.update([], 7) == []                  # 7 > tentative_ttl_ticks = 6
    assert tracker.states() == []
    assert tracker.expired == []                        # only confirmed tracks are listed


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_confirmed_track_coasts_then_expires(solver):
    tracker = T.Tracker(solver=solver)
    tracker.update([ob("knight", 5000, 20000)], 0)
    assert [t.uid for t in tracker.update([ob("knight", 5000, 20000)], 2)] == [1]
    assert [t.uid for t in tracker.update([], 8)] == [1]        # unseen 6 <= coast_ticks
    assert tracker.update([], 9) == []                          # unseen 7: kept, not reported
    assert [s.uid for s in tracker.states()] == [1]
    assert tracker.update([], 23) == []                         # unseen 21 > max_age_ticks = 20
    assert [t.uid for t in tracker.expired] == [1]
    assert tracker.expired[0].last_seen_tick == 2
    assert tracker.states() == []
    tracker.update([], 24)
    assert tracker.expired == []
    # a unit landing on the same spot later is a new unit, not a resurrection
    tracker.update([ob("knight", 5000, 20000)], 25)
    assert [t.uid for t in tracker.update([ob("knight", 5000, 20000)], 27)] == [2]


def test_a_coasting_track_is_extrapolated():
    tracker = T.Tracker(solver="greedy")
    tracker.update([ob("hog-rider", 5000, 20000)], 0)
    tracker.update([ob("hog-rider", 5200, 20000)], 2)           # 100 mt/tick
    out = tracker.update([], 6)
    assert out[0].x_mt == pytest.approx(5600.0)
    assert (out[0].vx, out[0].last_seen_tick) == (pytest.approx(100.0), 2)


def test_first_seen_and_deploy_ticks():
    tracker = T.Tracker(solver="greedy")
    for tick in (10, 12, 15):
        out = tracker.update([ob("hog-rider", 5000, 20000)], tick)
    (track,) = out
    assert (track.first_seen_tick, track.last_seen_tick) == (10, 15)
    assert T.deploy_ticks_left(track, 15, 1000) == 15           # 20 ticks - age 5
    assert T.deploy_ticks_left(track, 15, 1100) == 17
    assert T.deploy_ticks_left(track, 40, 1000) == 0


@pytest.mark.parametrize("solver", SOLVERS)
def test_last_assignment_gives_each_observation_its_uid(solver):
    """N05: Tracker.last_assignment, in observation order: the matched track's uid, or the new
    tentative track's (reported once confirmed)."""
    tracker = T.Tracker(solver=solver)
    assert tracker.last_assignment == []
    tracker.update([ob("knight", 3500, 20000), ob("skeleton", 9000, 25000)], 0)
    assert tracker.last_assignment == [1, 2]
    out = tracker.update([ob("skeleton", 9000, 25050), ob("hog-rider", 14500, 10000),
                          ob("knight", 3500, 19900)], 2)
    assert tracker.last_assignment == [2, 3, 1]
    assert sorted(t.uid for t in out) == [1, 2]            # 3 is still tentative
    tracker.update([], 4)
    assert tracker.last_assignment == []
    tracker.reset()
    assert tracker.last_assignment == []


def test_reset_starts_a_new_battle():
    tracker = T.Tracker(solver="greedy")
    tracker.note_own_tap({"knight"}, 0, 0, 0)
    tracker.update([ob("knight", 5000, 20000)], 0)
    tracker.update([ob("knight", 5000, 20000)], 2)
    tracker.reset()
    assert tracker.states() == [] and tracker.tracks() == []
    tracker.update([ob("knight", 5000, 20000)], 0)
    assert tracker.states()[0].uid == 1


# ---------------------------------------------------------------------------
# spells (Tracker(spell_classes=...); docs/SEASON_UPDATES.md 8.2)
# ---------------------------------------------------------------------------
SPELL_LABELS = {"Fireball", "fireball", "Log", "the-log", "Zap", "zap", "zap-evolution", "Freeze", "freeze"}
OWN_LABELS = {"Cannon", "HogRider", "Skeleton", "Fireball", "Log"}


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_spell_track_never_takes_a_units_box(solver):
    """Our Log's track and our new Cannon next to it (the 2026-10-29 friendly, tick 3599): with
    spell_classes the Cannon starts a track of its own; without them it joined the Log's track."""
    def run(**kw):
        tracker = T.Tracker(solver=solver, own_classes=OWN_LABELS, **kw)
        for tick in (0, 2, 4):
            tracker.update([ob("Log", 3500, 24000 - 150 * tick, side=0.1, conf=0.8)], tick)
        out = []
        for tick in (6, 8, 10, 12):                 # the Log's box is gone; the Cannon lands by it
            out = tracker.update([ob("Cannon", 4000, 23700, side=0.1, conf=0.9)], tick)
        return tracker, out

    tracker, out = run(spell_classes=SPELL_LABELS)
    assert sorted((t.uid, t.cls, t.team) for t in out) == [(2, "Cannon", VT.OWN_TEAM)]
    assert tracker.state(1).label == "Log" and tracker.state(1).hits == 3
    tracker, out = run()                            # as before: the Log's track took the Cannon
    assert [t.uid for t in out] == [1] and tracker.state(1).hits == 7 and len(tracker.states()) == 1


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_unit_track_never_takes_a_spells_box(solver):
    """The enemy Hog Rider's track, unseen for a moment, and our rolling Log's boxes passing by:
    with spell_classes the Hog keeps its label and its team (it turned Log, then ours, before)."""
    def run(**kw):
        tracker = T.Tracker(solver=solver, own_classes=OWN_LABELS, **kw)
        for tick in (0, 2, 4, 6):
            tracker.update([ob("HogRider", 4500, 20000 + 100 * tick, side=0.9)], tick)
        for tick in range(8, 26, 2):                # the Hog unseen 18 ticks: still held (max_age 20)
            tracker.update([ob("Log", 3500 + 10 * tick, 21400 - 150 * (tick - 8), side=0.1, conf=0.85)], tick)
        return tracker

    tracker = run(spell_classes=SPELL_LABELS)
    hog = tracker.state(1)
    assert (hog.label, tracker.team_of(hog), hog.hits) == ("HogRider", VT.ENEMY_TEAM, 4)
    assert [(s.label, tracker.team_of(s)) for s in tracker.states()[1:]] == [("Log", VT.OWN_TEAM)]
    tracker = run()
    hog = tracker.state(1)
    assert hog.label == "Log" and tracker.team_of(hog) == VT.OWN_TEAM and len(tracker.states()) == 1


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_confident_spell_is_reported_at_its_first_sighting(solver):
    tracker = T.Tracker(solver=solver, own_classes=OWN_LABELS, spell_classes=SPELL_LABELS)
    out = tracker.update([ob("Zap", 9000, 24000, side=0.9, conf=0.6),          # >= 0.5: at once
                          ob("Freeze", 3000, 20000, side=0.9, conf=0.4),       # below the floor
                          ob("Skeleton", 15000, 24000, side=0.9, conf=0.9)], 100)   # a unit: min_hits
    assert [(t.cls, t.team, t.first_seen_tick) for t in out] == [("Zap", VT.ENEMY_TEAM, 100)]
    out = tracker.update([], 102)
    assert [t.cls for t in out] == ["Zap"]                                     # coasting
    assert [s.label for s in tracker.states()] == ["Zap", "Freeze", "Skeleton"]
    tracker.update([], 107)                         # the tentative Freeze and Skeleton die (> 6)
    assert [s.label for s in tracker.states()] == ["Zap"]
    tracker.update([], 121)                         # the Zap expires as a confirmed track (> 20)
    assert [t.cls for t in tracker.expired] == ["Zap"] and tracker.states() == []
    # a weak spell waits for a second sighting, as a unit does
    out = tracker.update([ob("Freeze", 3000, 20000, side=0.9, conf=0.4)], 200)
    assert out == []
    out = tracker.update([ob("Freeze", 3000, 20000, side=0.9, conf=0.4)], 202)
    assert [(t.cls, t.first_seen_tick) for t in out] == [("Freeze", 200)]
    # spell_confirm_conf=None, no own_classes (no deck rule) or no spell_classes: a one-frame Zap
    # is never reported
    for tracker in (T.Tracker(T.TrackerConfig(spell_confirm_conf=None), solver=solver, own_classes=OWN_LABELS,
                              spell_classes=SPELL_LABELS),
                    T.Tracker(solver=solver, spell_classes=SPELL_LABELS),
                    T.Tracker(solver=solver)):
        assert tracker.update([ob("Zap", 9000, 24000, side=0.9, conf=0.95)], 100) == []
        assert tracker.update([], 102) == [] and tracker.expired == []


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_stray_box_of_our_own_spell_waits_for_a_second_sighting(solver):
    """Only a label the deck rule makes the enemy's is reported at its first sighting: one box of
    our Log or Fireball at the river (side 0.1: -2, no half cue) has no team yet, and the event
    deriver gives an undecided spell to the enemy, so reported at once it was a full-cost enemy play."""
    tracker = T.Tracker(solver=solver, own_classes=OWN_LABELS, spell_classes=SPELL_LABELS)
    for k, label in enumerate(("Log", "Fireball")):
        tick = 100 + 50 * k
        assert tracker.update([ob(label, 9000, 16300, side=0.1, conf=0.9)], tick) == []
        assert tracker.update([], tick + 2) == []
        assert tracker.states()[0].team_llr == pytest.approx(-2.0)
        tracker.update([], tick + 7)                # dies tentative (> 6)
        assert tracker.states() == [] and tracker.expired == []
    # seen twice it is reported as before
    tracker.update([ob("Log", 9000, 16300, side=0.1, conf=0.9)], 200)
    out = tracker.update([ob("Log", 9000, 16000, side=0.1, conf=0.9)], 202)
    assert [(t.cls, t.first_seen_tick) for t in out] == [("Log", 200)]


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_unit_wide_spell_box_waits_for_a_second_sighting(solver):
    """spell_confirm_min_mt: a Zap box no wider than a unit's (box_w_mt; the widest unit box on the
    2026-10-29 friendly is 3,355 mt, a Zap about 4,900) is most likely a unit's box misread where no
    track can take it, so it waits for min_hits; a Zap-wide one, or one of unknown width, is
    reported at once."""
    def first(width, **kw):
        tracker = T.Tracker(solver=solver, own_classes=OWN_LABELS, spell_classes=SPELL_LABELS, **kw)
        return [t.cls for t in tracker.update([ob("Zap", 9000, 24000, side=0.9, conf=0.8, box_w_mt=width)], 100)]

    assert first(2000.0) == [] and first(3999.0) == []
    assert first(4000.0) == first(4900.0) == first(None) == ["Zap"]
    assert first(2000.0, config=T.TrackerConfig(spell_confirm_min_mt=None)) == ["Zap"]
    tracker = T.Tracker(solver=solver, own_classes=OWN_LABELS, spell_classes=SPELL_LABELS)
    tracker.update([ob("Zap", 9000, 24000, side=0.9, conf=0.8, box_w_mt=2000.0)], 100)
    out = tracker.update([ob("Zap", 9000, 24000, side=0.9, conf=0.8, box_w_mt=2000.0)], 102)
    assert [(t.cls, t.first_seen_tick) for t in out] == [("Zap", 100)]          # seen twice: reported


def skel(tick, label="Skeleton", side=0.1, conf=0.8, size=(36.0, 40.0)):
    w, h = size if size else (None, None)
    return ob(label, 5000, 24000 - 20 * tick, side=side, conf=conf, box_w=w, box_h=h)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_units_box_read_as_a_spell_stays_the_units(solver):
    """SPELLS, flickers: our Skeleton's box read as Zap or Rocket for 1-3 frames, at the Skeleton's
    size, joins the Skeleton's track, and the label vote never turns the track into a spell (with the
    kind rule alone each misread was a spell track, reported at once: a full-cost enemy play)."""
    labels = SPELL_LABELS | {"Rocket"}

    def run(flicker, n, size=(38.0, 44.0), extra=False, **kw):
        tracker = T.Tracker(solver=solver, own_classes=OWN_LABELS, spell_classes=labels, **kw)
        seen = []
        for tick in range(0, 30, 2):
            obs = [skel(tick)]
            if 10 <= tick < 10 + 2 * n:
                obs = [skel(tick, flicker, side=0.1, conf=0.7, size=size)] + (obs if extra else [])
            seen += [(t.uid, t.cls, t.team) for t in tracker.update(obs, tick)]
        return tracker, seen

    for flicker in ("Zap", "Rocket"):
        for n in (1, 2, 3):
            tracker, seen = run(flicker, n)
            assert set(seen) == {(1, "Skeleton", VT.OWN_TEAM)}, (flicker, n)
            assert [(s.uid, s.label, s.hits) for s in tracker.states()] == [(1, "Skeleton", 15)]
            assert tracker.state(1).label_weights[flicker] > 0
    # a Zap-sized box where the Skeleton's is missing is a Zap: an enemy track, reported at once
    tracker, seen = run("Zap", 1, size=(175.0, 140.0))
    assert (2, "Zap", VT.ENEMY_TEAM) in seen and tracker.state(1).hits == 14
    # so is a Skeleton-sized one next to the Skeleton's own box, one without a box size, and every
    # one with flicker_size_ratio=None
    for kw in ({"extra": True}, {"size": None}, {"config": T.TrackerConfig(flicker_size_ratio=None)}):
        tracker, seen = run("Zap", 1, **kw)
        assert (2, "Zap", VT.ENEMY_TEAM) in seen, kw
    # a unit track that is still held takes it even unseen for a while (the 1.4 size rule decides);
    # one forgotten (max_age_ticks 20) does not, nor one farther than max_gate_mt
    for tick, x, want in ((12, 5000, [(1, "Skeleton")]), (23, 5000, [(2, "Zap")]), (12, 7500, [(2, "Zap")])):
        tracker = T.Tracker(solver=solver, own_classes=OWN_LABELS, spell_classes=labels)
        tracker.update([skel(0)], 0)
        tracker.update([skel(2)], 2)
        flicker = skel(tick, "Zap", conf=0.7)
        flicker.x_mt = float(x)
        out = tracker.update([flicker], tick)
        assert [(t.uid, t.cls) for t in out if t.last_seen_tick == tick] == want, (tick, x)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_fast_spell_is_one_track(solver):
    """SPELLS, fast spells: a Goblin Barrel flying about 620 mt a tick, boxed every 2 ticks, is one
    track (spell_gate_mt, spell_max_speed_mt_per_tick); with a unit's gate and speed clamp every box
    started a track of its own. A box of another spell label keeps the unit gate."""
    def run(**kw):
        tracker = T.Tracker(solver=solver, own_classes=OWN_LABELS, **kw)
        seen = set()
        for tick in range(100, 132, 2):
            obs = [ob("GoblinBarrel", 4000 + 150 * (tick - 100), 3000 + 600 * (tick - 100), side=0.9, conf=0.8)]
            seen |= {(t.uid, t.cls) for t in tracker.update(obs, tick)}
        return tracker, seen

    tracker, seen = run(spell_classes=SPELL_LABELS | {"GoblinBarrel"})
    assert seen == {(1, "GoblinBarrel")} and tracker.state(1).hits == 16
    assert (tracker.state(1).vx, tracker.state(1).vy) == (pytest.approx(150.0), pytest.approx(600.0))
    # 1,500 mt past the barrel's prediction: a Zap starts its own track, another barrel box joins
    px, py = tracker.state(1).predict(132, 10)
    out = tracker.update([ob("Zap", px + 1500, py, side=0.9)], 132)
    assert sorted((t.uid, t.cls) for t in out) == [(1, "GoblinBarrel"), (2, "Zap")]
    tracker.update([ob("GoblinBarrel", px + 1500, py + 1200, side=0.9)], 134)
    assert tracker.state(1).hits == 17
    tracker, seen = run()                           # as before: every box a new tentative track
    assert seen == set() and tracker.states()[-1].uid == 16


@pytest.mark.parametrize("solver", SOLVERS)
def test_spells_still_pair_with_spells_and_units_with_units(solver):
    """The kind rule leaves the association inside a kind as it was: a Log seen as the-log stays one
    track, and a unit's label flicker still does not split it."""
    tracker = T.Tracker(solver=solver, own_classes=OWN_LABELS, spell_classes=SPELL_LABELS)
    for k, tick in enumerate(range(0, 12, 2)):
        out = tracker.update([ob("Log" if k % 2 else "the-log", 3500, 24000 - 100 * tick, side=0.1),
                              ob("Knight" if k != 3 else "HogRider", 9000, 12000, side=0.9)], tick)
    assert sorted((t.uid, t.cls) for t in out) == [(1, "the-log"), (2, "Knight")]
    assert len(tracker.states()) == 2


def test_spell_classes_are_checked():
    with pytest.raises(TypeError):
        T.Tracker(spell_classes="Zap")
    assert T.Tracker().spell_classes is None and not T.Tracker().is_spell("Zap")
    assert T.Tracker(spell_classes=["Zap"]).spell_classes == frozenset({"Zap"})
    for bad in (-0.1, 1.5, True, float("nan")):
        with pytest.raises(ValueError):
            T.TrackerConfig(spell_confirm_conf=bad)
    assert T.TrackerConfig(spell_confirm_conf=None).spell_confirm_conf is None
    assert T.TrackerConfig().spell_confirm_conf == 0.5
    for bad in (0.99, -1.0, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            T.TrackerConfig(flicker_size_ratio=bad)
    assert T.TrackerConfig(flicker_size_ratio=None).flicker_size_ratio is None
    assert T.TrackerConfig().flicker_size_ratio == 1.4
    for bad in (-1.0, True, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            T.TrackerConfig(spell_confirm_min_mt=bad)
    assert T.TrackerConfig(spell_confirm_min_mt=None).spell_confirm_min_mt is None
    assert T.TrackerConfig().spell_confirm_min_mt == 4000.0
    for bad in ({"spell_gate_mt": 0.0}, {"spell_max_speed_mt_per_tick": -1.0},
                {"spell_gate_mt": 3000.0, "spell_max_gate_mt": 2500.0}):
        with pytest.raises(ValueError):
            T.TrackerConfig(**bad)


def test_the_live_chain_builds_its_tracker_with_the_spell_labels():
    """scripts/vision_play.py and tools/vision_replay.py (the same chain on a recording) give the
    tracker the class table's spell labels (scripts/vision/unit_harvest.py's harvest tracker is
    left as it was: docs/SEASON_UPDATES.md 8.2)."""
    for rel in ("scripts/vision_play.py", "tools/vision_replay.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", getattr(n.func, "id", None)) == "Tracker"]
        assert len(calls) == 1, rel
        kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
        assert kw.get("spell_classes") == "table.spell_labels()", (rel, kw)


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------
def test_tracks_are_vision_types_that_round_trip():
    tracker = T.Tracker(solver="greedy")
    tracker.update([ob("musketeer", 12000, 22000, side=0.2, conf=0.75, form=VT.FORM_EVO)], 0)
    out = tracker.update([ob("musketeer", 12000, 21900, side=0.2, conf=0.75, form=VT.FORM_EVO, hp_frac=0.5)], 2)
    (track,) = out
    assert isinstance(track, VT.Track)
    assert (track.cls, track.form, track.hp_frac, track.conf) == ("musketeer", VT.FORM_EVO, 0.5, 0.75)
    assert VT.Track.from_dict(track.to_dict()) == track
    frame = VT.VisionFrame(t_capture=1.0, tick=2, tracks=out, screen=VT.SCREEN_BATTLE)
    assert VT.VisionFrame.from_json(frame.to_json()).tracks == out


def test_observations_from_detections():
    from scripts.vision import geometry as G

    geo = G.ArenaGeometry.builtin()
    gx, gy = geo.world_to_px(9000.0, 24000.0)
    bx, by = geo.world_to_px(3000.0, 20000.0)
    dets = [VT.Detection("knight", side_prob=0.8, conf=0.9, box=(0.0, 0.0, 1.0, 1.0), ground_px=(gx, gy)),
            VT.Detection("minions", 0.3, 0.7, (bx - 20.0, by - 50.0, bx + 20.0, by), None, VT.FORM_EVO)]
    a, b = T.observations_from_detections(dets, geo)
    assert (a.cls, a.conf, a.side_prob) == ("knight", 0.9, 0.8)
    assert (a.x_mt, a.y_mt) == (pytest.approx(9000.0, abs=1e-6), pytest.approx(24000.0, abs=1e-6))
    assert (b.cls, b.form) == ("minions", VT.FORM_EVO)
    assert (b.x_mt, b.y_mt) == (pytest.approx(3000.0, abs=1e-6), pytest.approx(20000.0, abs=1e-6))
    (fly,) = T.observations_from_detections(dets[1:], geo, flying={"minions"})
    assert fly.y_mt == pytest.approx(20000.0 + G.DEFAULT_FLYING.dy_mt, abs=1e-6)
    # the box size in pixels (the flicker rule), None for a side that is not positive, and its width
    # on the ground in millitiles (the geometry's mapping of its edges at its middle row)
    assert (a.box_w, a.box_h) == (1.0, 1.0)
    assert (b.box_w, b.box_h) == (pytest.approx(40.0), pytest.approx(50.0))
    (lx, ly), (rx, ry) = geo.px_to_world(bx - 20.0, by - 25.0), geo.px_to_world(bx + 20.0, by - 25.0)
    assert b.box_w_mt == pytest.approx(((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5)
    assert 700.0 < b.box_w_mt < 900.0                      # 40 px of the builtin 1080-px frame, ~51 a tile
    (flat,) = T.observations_from_detections([VT.Detection("zap", 0.9, 0.6, (5.0, 5.0, 5.0, 9.0), (gx, gy))], geo)
    assert (flat.box_w, flat.box_h, flat.box_w_mt) == (None, 4.0, None)
    blank = T.Observation("zap", 0.0, 0.0)
    assert (blank.box_w, blank.box_h, blank.box_w_mt) == (None, None, None)


def test_config_is_validated():
    with pytest.raises(ValueError):
        T.TrackerConfig(min_hits=0)
    with pytest.raises(ValueError):
        T.TrackerConfig(decide_llr=7.0, flip_llr=6.0)
    with pytest.raises(ValueError):
        T.TrackerConfig(label_decay=0.0)
    with pytest.raises(ValueError):
        T.TrackerConfig(gate_mt=-1.0)


def test_tracker_never_imports_the_memory_reader():
    tree = ast.parse((ROOT / "scripts" / "vision" / "tracker.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    assert not any("memread" in n or "gamestate" in n or "hcr_reader" in n for n in names), names
