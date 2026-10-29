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
