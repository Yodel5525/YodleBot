"""V11: tools/fit_noise_profile.py fits the E17 perception-noise profile from labelled frames and own taps.

No device, detector, engine or emulator. The shipped class table (data/vision/classes.json) gives the
size buckets: level-11 hit points are the level-1 values x 2.56, floored (Skeleton 32 -> 81 small,
Musketeer 282 -> 721, Hog Rider 663 -> 1697 and Knight 690 -> 1766 medium, Giant 1550 -> 3968 large,
Cannon a building; the prior's bounds are 400 and 2500). The base profile is the shipped prior
(data/noise/default_prior.json), so every value the synthetic logs do not measure must come back
unchanged.

(a) Labelled frames: hand-built label / perception pairs scored at the track and the detection level
(known drops, flips, confusions, phantoms and offsets -> known rates), a replay-style session scored
through the label files and the geometry, and the sparse-replay fallback to detections.
(b) Own taps: a vision_play-style session written with the real VisionRecorder: six taps whose units
are first reported 1, 2, 4, 4, 4 and 6 ticks after their first sighting, two single-unit troops that
stand 100 / 200 mt off their aimed tiles while they deploy, an enemy decoy at an aim, a spell, a tap
never seen and one outside any battle.
"""
from __future__ import annotations

import ast
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "tools"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fit_noise_profile as FNP  # noqa: E402
import latency_probe as LP  # noqa: E402
import perception_noise as PN  # noqa: E402
import vision_replay as VR  # noqa: E402
from scripts.vision.geometry import load_geometry  # noqa: E402
from scripts.vision.labels import load_class_table  # noqa: E402
from scripts.vision.record import VisionRecorder  # noqa: E402
from scripts.vision.types import (ENEMY_TEAM, OWN_TEAM, SCREEN_BATTLE, SCREEN_MENU, UNKNOWN_TEAM,  # noqa: E402
                                  Detection, Track, VisionFrame)

TABLE = load_class_table()
PRIOR = PN.NoiseProfile.load("default")
GEO = load_geometry()                 # what the command line uses without --geometry
SIZE = (747, 1328)                    # the owner's PrintWindow frames (9:16)
ALLY, ENEMY = FNP.ALLY, FNP.ENEMY


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def obj(cls, side, x, y, form=0, box=None, decided=True):
    return FNP.Obj(cls, side, form, float(x), float(y), box, FNP._flying(TABLE, cls, form), 1.0, decided)


def prior_pairs_without(*classes):
    return {k: dict(v) for k, v in PRIOR.confusion.pairs.items() if k not in classes}


# ---------------------------------------------------------------------------
# matching and reading perception
# ---------------------------------------------------------------------------

def test_match_objects_same_class_first_then_confusions():
    def cost(t, p):
        d = math.hypot(p.x_mt - t.x_mt, p.y_mt - t.y_mt)
        return d if d <= 1000.0 else None

    truth = [obj("HogRider", ALLY, 5000, 20000), obj("Musketeer", ENEMY, 9000, 10000)]
    preds = [obj("Musketeer", ENEMY, 5100, 20000),     # a look-alike 100 mt from the Hog ...
             obj("HogRider", ALLY, 5600, 20000),       # ... cannot take it from the Hog 600 mt away
             obj("Wizard", ENEMY, 9200, 10000),        # the Musketeer read as a Wizard: a confusion
             obj("Knight", ENEMY, 3000, 3000)]         # nothing there
    assert FNP.match_objects(truth, preds, cost) == ([(0, 1, True), (1, 2, False)], [], [0, 3])
    two = [obj("Skeleton", ENEMY, 9000, 10000), obj("Skeleton", ENEMY, 9000, 10000)]
    assert FNP.match_objects(two, [obj("Skeleton", ENEMY, 9000, 10300)], cost) == ([(0, 0, True)], [1], [])


def test_names_tracks_and_detections_are_read_as_the_public_state_reads_them():
    canon = FNP.Canon(TABLE)
    assert canon("skeleton-evolution") == ("Skeleton", None, "unit", 1)
    assert canon("Knight_EV1@enemy") == ("Knight", 1, "unit", 1)
    assert canon("MusketeerHero") == ("Musketeer", None, "unit", 2)

    skipped = Counter()
    tracks = [Track(1, "HogRider", OWN_TEAM, 0.9, 5000.0, 20000.0),
              Track(2, "Knight", UNKNOWN_TEAM, 0.8, 9000.0, 20000.0, form=1),   # undecided, our half
              Track(3, "Knight", UNKNOWN_TEAM, 0.8, 9000.0, 12000.0),           # undecided, their half
              Track(4, "Fireball", ENEMY_TEAM, 0.7, 9000.0, 8000.0),            # a spell: not a unit
              Track(5, "no-such-thing", ENEMY_TEAM, 0.5, 1000.0, 1000.0),       # public_state leaves it out
              Track(6, "Bat", ENEMY_TEAM, 0.9, 4000.0, 9000.0)]
    got = FNP.track_objects(tracks, canon, skipped)
    assert [(o.cls, o.side, o.form, o.decided, o.flying) for o in got] == [
        ("HogRider", ALLY, 0, True, False), ("Knight", ALLY, 1, False, False),
        ("Knight", ENEMY, 0, False, False), ("Bat", ENEMY, 0, True, True)]
    assert (got[0].x_mt, got[0].y_mt, got[0].box) == (5000.0, 20000.0, None)
    assert dict(skipped) == {"no-such-thing": 1}

    gx, gy = (float(v) for v in GEO.world_to_px(5000.0, 20000.0, SIZE))
    hx, hy = (float(v) for v in GEO.world_to_px(9000.0, 10000.0, SIZE))
    dets = [Detection("HogRider", 0.1, 0.9, (gx - 10, gy - 30, gx + 10, gy), (gx, gy)),
            Detection("Knight", 0.5, 0.8, (hx - 10, hy - 30, hx + 10, hy)),     # no ground_px: the box's
            Detection("class_17", 0.9, 0.5, (10.0, 10.0, 20.0, 20.0)),
            Detection("bar", 0.5, 0.9, (30.0, 30.0, 40.0, 32.0))]
    skipped = Counter()
    got = FNP.detection_objects(dets, SIZE, GEO, canon, skipped)
    assert [(o.cls, o.side) for o in got] == [("HogRider", ALLY), ("Knight", None)]
    assert (got[0].x_mt, got[0].y_mt) == pytest.approx((5000.0, 20000.0), abs=1e-6)
    assert (got[1].x_mt, got[1].y_mt) == pytest.approx((9000.0, 10000.0), abs=1e-6)
    assert got[0].box == pytest.approx(((gx - 10) / 747, (gy - 30) / 1328, (gx + 10) / 747, gy / 1328))
    assert dict(skipped) == {"class_17": 1}


def test_frame_tick_is_the_trackers_tick():
    def entry(extra, vf):
        return SimpleNamespace(data={"extra": extra}, frame=vf)

    tracks = [Track(1, "HogRider", OWN_TEAM, 0.9, 0.0, 0.0, last_seen_tick=41),
              Track(2, "Knight", ENEMY_TEAM, 0.9, 0.0, 0.0, last_seen_tick=44)]
    assert FNP.frame_tick(entry({"replay": {"track_tick": 57}}, VisionFrame(0.0, tick=60))) == 57
    assert FNP.frame_tick(entry({}, VisionFrame(0.0, tick=60, tracks=tracks))) == 60
    assert FNP.frame_tick(entry({}, VisionFrame(0.0, tracks=tracks))) == 44      # clock unread: extrapolated
    assert FNP.frame_tick(entry({}, VisionFrame(0.0))) is None


# ---------------------------------------------------------------------------
# (a) labelled frames -> rates
# ---------------------------------------------------------------------------

#: One labelled frame, the same in all three: 4 medium units (2 Hogs, an evolved Knight, a
#: Musketeer), 2 small (Skeletons), 1 large (Giant), 1 building (Cannon).
TRUTH = [obj("HogRider", ALLY, 5000, 20000), obj("HogRider", ENEMY, 13000, 12000),
         obj("Skeleton", ENEMY, 9000, 10000), obj("Skeleton", ENEMY, 9500, 10000),
         obj("Giant", ENEMY, 9000, 6000), obj("Cannon", ALLY, 9000, 24000),
         obj("Knight", ENEMY, 3000, 8000, form=1), obj("Musketeer", ENEMY, 15000, 9000)]


def tracks_frame(k):
    """What the tracker reported on labelled frame k: our Hog 100 / 200 mt off, theirs read as ours
    (a flip), the second Skeleton missing (its neighbour's track is the nearer match), the Cannon an
    undecided track on frame 2 (our half: ours), the evolved Knight read as its base on frames 0 and
    1, the Musketeer read as a Wizard on frames 0 and 1, and a phantom Skeleton 600 / 500 mt from
    our Hog (a double box)."""
    last = k == 2
    return [obj("HogRider", ALLY, 5100, 20200), obj("HogRider", ALLY, 13000, 12100),
            obj("Skeleton", ENEMY, 9000, 10000), obj("Giant", ENEMY, 9000, 6000),
            obj("Cannon", ALLY, 9000, 24000, decided=not last), obj("Knight", ENEMY, 3000, 8000, form=1 if last else 0),
            obj("Musketeer" if last else "Wizard", ENEMY, 15000, 9000), obj("Skeleton", ENEMY, 5600, 20500)]


def test_label_errors_at_the_track_level_give_known_rates():
    cfg = FNP.FitConfig(min_samples=3, min_class_samples=3, min_tracks=2)
    E = FNP.LabelErrors(FNP.LEVEL_TRACKS, TABLE, PRIOR.size, cfg)
    for k in range(3):
        assert E.add(TRUTH, tracks_frame(k)) == {"truth": 8, "matched": 7, "dropped": 1, "fp": 1}
    s = E.summary()
    assert s["buckets"] == {"small": {"n": 6, "dropped": 3, "rate": 0.5}, "medium": {"n": 12, "dropped": 0, "rate": 0.0},
                            "large": {"n": 3, "dropped": 0, "rate": 0.0}, "building": {"n": 3, "dropped": 0, "rate": 0.0}}
    assert s["classes"]["Musketeer"] == {"n": 3, "dropped": 0, "matched": 3, "confused": {"Wizard": 2}}
    assert (E.side["decided"], E.side["flipped"], E.side["by_half"], E.side["undecided"]) == (21, 3, 1, 0)
    assert (E.forms["evo"], E.forms["evo_base"], E.forms["hero"], E.forms["unlabelled"]) == (3, 2, 0, 0)
    assert len(E.ground) == 19 and E.flyer == []          # 6 same-class ground matches a frame, 7 on frame 2
    assert s["false_positives"] == {"n": 3, "near": 3, "far": {}} and E.fp_near == [(600.0, 500.0)] * 3

    res = FNP.fit("default", E, None, cfg)
    assert res.measured == ["drop.small", "drop.medium", "drop.large", "drop.building", "drop.by_class",
                            "forms.evo_to_base_p", "confusion.pairs", "side_flip.p", "false_positives.rate",
                            "false_positives.classes", "false_positives.near_sigma_mt",
                            "position.sigma_x_mt", "position.sigma_y_mt"]
    p = res.profile
    assert (p.drop.small, p.drop.medium, p.drop.large, p.drop.building) == (0.5, 0.0, 0.0, 0.0)
    assert p.drop.by_class == {"Cannon": 0.0, "Giant": 0.0, "HogRider": 0.0, "Knight": 0.0, "Musketeer": 0.0,
                               "Skeleton": 0.5}
    # 2 of 3 labelled evolutions (a form KataCR labels) read as the base; heroes kept
    assert (p.forms.evo_to_base_p, p.forms.hero_to_base_p, p.forms.unlabelled_to_base) == (0.6667, 1.0, True)
    # Musketeer -> Wizard 2 of 3, rounded down; HogRider and Giant were measured clean: their prior rows go
    assert p.confusion.pairs == {**prior_pairs_without("HogRider", "Giant"), "Musketeer": {"Wizard": 0.6666}}
    assert p.confusion.unknown_p == PRIOR.confusion.unknown_p == 0.01
    assert p.side_flip.p == 0.1429                         # 3 of 21
    # one phantom a frame, 600 / 500 mt from the Hog: sqrt((600^2 + 500^2) / 2) = sqrt(305000) = 552.27
    assert (p.false_positives.rate, p.false_positives.classes, p.false_positives.near_sigma_mt) == (1.0, {}, 552.3)
    # x errors: 100 three times in 19 -> sqrt(30000 / 19) = 39.74; y: 200 and 100 three times each ->
    # sqrt(150000 / 19) = 88.85; the flyer values are kept
    assert (p.position.sigma_x_mt, p.position.sigma_y_mt) == (39.7, 88.9)
    assert (p.position.flyer_dx_mt, p.position.flyer_dy_mt, p.position.flyer_sigma_mt) == (0.0, 0.0, 400.0)
    for g in ("board", "new_units", "hp", "deploy", "towers", "elixir", "clock", "opponent_plays", "size"):
        assert getattr(p, g) == getattr(PRIOR, g), g

    n = res.notes
    assert n["drop"]["measured"]["small"] == {"from": "labels/tracks", "n": 6, "dropped": 3}
    assert n["drop"]["measured"]["by_class"]["n"] == 24 and "kept" not in n["drop"]
    assert n["confusion"]["measured"]["pairs"]["classes"] == {"Cannon": 3, "Giant": 3, "HogRider": 6, "Knight": 3,
                                                              "Musketeer": 3, "Skeleton": 3}
    assert n["confusion"]["kept"] == ["unknown_p"]
    assert n["confusion"]["why"] == FNP.KEPT_FIELD_WHY[("confusion", "unknown_p")]
    assert n["side_flip"]["measured"]["p"] == {"from": "labels/tracks", "n": 21, "flipped": 3, "by_half": 1,
                                               "undecided": 0}
    assert n["false_positives"]["measured"]["classes"] == {"from": "labels/tracks", "n": 3, "far": 0}
    assert n["position"]["measured"]["sigma_x_mt"]["bias_mt"] == 15.8            # 300 / 19
    assert "lower bound" in n["position"]["measured"]["sigma_x_mt"]["note"]
    assert n["position"]["kept"] == ["flyer_dx_mt", "flyer_dy_mt", "flyer_sigma_mt"]
    assert n["position"]["why"] == FNP.FLYER_WHY
    assert n["forms"]["kept"] == ["hero_to_base_p", "unlabelled_to_base"]
    assert n["forms"]["why"] == FNP.HERO_WHY + "; " + FNP.KEPT_FIELD_WHY[("forms", "unlabelled_to_base")]
    assert n["new_units"]["why"] == FNP.NO_TAPS and n["hp"]["why"] == FNP.KEPT_WHY["hp"]
    assert res.warnings == []

    doc = json.loads(json.dumps(res.document()))           # what --out holds: plain JSON
    assert PN.NoiseProfile.from_dict(doc) == p              # the "_fit" notes are skipped
    assert doc["drop"]["_fit"]["measured"]["small"]["n"] == 6
    assert doc["_fit"]["base"] == "default_prior" and doc["_fit"]["base_digest"] == PRIOR.digest()
    assert doc["_fit"]["measured"] == res.measured and doc["_fit"]["config"]["min_samples"] == 3
    assert doc["name"] == "measured" and "3 labelled frames (tracks)" in doc["description"]
    assert any("39.7" in line for line in res.describe())


def test_label_errors_at_the_detection_level_and_the_warning():
    cfg = FNP.FitConfig(min_samples=1, min_class_samples=1)
    E = FNP.LabelErrors(FNP.LEVEL_DETECTIONS, TABLE, PRIOR.size, cfg)
    truth = [obj("HogRider", ALLY, 5000, 20000, box=(0.1, 0.1, 0.2, 0.2)),
             obj("Skeleton", ENEMY, 9000, 10000, box=(0.5, 0.5, 0.55, 0.55))]
    preds = [obj("HogRider", None, 5040, 20030, box=(0.1, 0.1, 0.2, 0.19)),       # IoU 0.9, side 0.5
             obj("Skeleton", ENEMY, 9000, 10300, box=(0.5, 0.5, 0.55, 0.52))]     # IoU 0.4: a drop and a phantom
    assert E.add(truth, preds) == {"truth": 2, "matched": 1, "dropped": 1, "fp": 1}
    assert (E.side["undecided"], E.side["decided"]) == (1, 0)

    res = FNP.fit(PRIOR, E, None, cfg)
    p = res.profile
    assert res.measured == ["drop.small", "drop.medium", "drop.by_class", "confusion.pairs", "false_positives.rate",
                            "false_positives.classes", "false_positives.near_sigma_mt", "position.sigma_x_mt",
                            "position.sigma_y_mt"]
    assert (p.drop.small, p.drop.medium, p.drop.large, p.drop.building) == (1.0, 0.0, 0.03, 0.02)
    assert p.drop.by_class == {"HogRider": 0.0, "Skeleton": 1.0}
    assert p.confusion.pairs == prior_pairs_without("HogRider")
    # the phantom 300 mt from its Skeleton: sqrt(300^2 / 2) = 212.13
    assert (p.false_positives.rate, p.false_positives.near_sigma_mt) == (1.0, 212.1)
    assert (p.position.sigma_x_mt, p.position.sigma_y_mt) == (40.0, 30.0)
    assert p.side_flip == PRIOR.side_flip                   # no matched unit had a side
    assert res.notes["side_flip"]["why"] == "fewer samples than the minimum (--min-samples 1)"
    assert len(res.warnings) == 1 and res.warnings[0].startswith("labelled frames were scored on raw detections")


def test_far_phantoms_give_classes_and_unlabelled_forms_warn():
    cfg = FNP.FitConfig(min_samples=2, min_class_samples=2)
    E = FNP.LabelErrors(FNP.LEVEL_TRACKS, TABLE, PRIOR.size, cfg)
    truth = [obj("HogRider", ALLY, 5000, 20000), obj("Musketeer", ENEMY, 12000, 9000, form=2)]   # a hero
    seen = [obj("HogRider", ALLY, 5000, 20000), obj("Musketeer", ENEMY, 12000, 9000, form=2)]
    knight, goblin = obj("Knight", ENEMY, 3000, 3000), obj("Goblin", ENEMY, 15000, 3000)  # > 2000 mt from any unit
    E.add(truth, seen + [knight, goblin])
    E.add(truth, seen + [knight])
    assert (E.fp, E.fp_near, dict(E.fp_far)) == (3, [], {"Knight": 2, "Goblin": 1})
    assert (E.forms["unlabelled"], E.forms["unlabelled_base"], E.forms["hero"]) == (2, 0, 0)

    res = FNP.fit("default", E, None, cfg)
    fp = res.profile.false_positives
    assert (fp.rate, fp.classes, fp.near_sigma_mt) == (1.5, {"Goblin": 1.0, "Knight": 2.0}, 700.0)
    assert res.notes["false_positives"]["measured"]["classes"] == {"from": "labels/tracks", "n": 3, "near": 0}
    assert res.notes["false_positives"]["kept"] == ["near_sigma_mt"]
    assert res.profile.forms == PRIOR.forms
    assert res.notes["forms"]["why"] == "; ".join(["fewer samples than the minimum (--min-samples 2)", FNP.HERO_WHY,
                                                    FNP.KEPT_FIELD_WHY[("forms", "unlabelled_to_base")]])
    assert len(res.warnings) == 1 and res.warnings[0].startswith("2 of 2 labelled forms the KataCR table lacks")


# ---------------------------------------------------------------------------
# (a) through files: replay sessions, label files, the geometry
# ---------------------------------------------------------------------------

def foot_box(x, y):
    """A 20 x 30 px box whose bottom centre is world (x, y) on a SIZE frame, and that point."""
    gx, gy = (float(v) for v in GEO.world_to_px(x, y, SIZE))
    return (gx - 10.0, gy - 30.0, gx + 10.0, gy), (gx, gy)


def yolo(k, box, side=None):
    x1, y1, x2, y2 = box
    w, h = SIZE
    cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
    return f"{k} {cx!r} {cy!r} {bw!r} {bh!r}" + ("" if side is None else f" {side}")


def write_replay_session(root, frames, name="rec_replay_20261030_120000"):
    """A tools/vision_replay.py output: per frame (t, tracks, detections), the image it replayed
    (extra.replay.image 0000000k.jpg), the tracker's tick 100 + k and that both stages ran."""
    rec = VisionRecorder(root, session=name, image_every=0, wall_clock=lambda: 0.0)
    for k, (t, tracks, dets) in enumerate(frames):
        vf = VisionFrame(t_capture=t, tracks=list(tracks), screen=SCREEN_BATTLE, detections=list(dets),
                         frame_size=SIZE)
        rec.write(vf, None, battle=1, state="battle",
                  replay={"i": k + 1, "image": f"{k:08d}.jpg", "track_tick": 100 + k, "tracks": {"n": len(tracks)},
                          "det": {"source": "recorded", "n": len(dets)}})
    rec.close()
    return rec.session_dir


HOG_BOX, HOG_FOOT = foot_box(5000.0, 20000.0)
SKEL_BOX, _ = foot_box(9000.0, 10000.0)


def write_labels(folder):
    """classes.txt: HogRider@ally, the KataCR name skeleton (side from the 6th column). Frames 1-3
    show our Hog at (5000, 20000) and an enemy Skeleton at (9000, 10000); frame 0 is unlabelled; a
    second file for frame 3 in a subfolder is a duplicate stem (not read)."""
    folder.mkdir()
    (folder / "classes.txt").write_text("HogRider@ally\nskeleton\n", encoding="utf-8")
    for k in (1, 2, 3):
        (folder / f"{k:08d}.txt").write_text(yolo(0, HOG_BOX) + "\n" + yolo(1, SKEL_BOX, 1) + "\n", encoding="utf-8")
    (folder / "sub").mkdir()
    (folder / "sub" / "00000003.txt").write_text("1 0.5 0.5 0.1 0.1 0\n", encoding="utf-8")
    return folder


def labelled_replay(root):
    """Dense (every 50 ms): our Hog tracked 100 mt right of its label on every frame, the Skeleton
    tracked on frames 0, 2 and 3 (frame 1: a drop)."""
    hog = Track(1, "HogRider", OWN_TEAM, 0.9, 5100.0, 20000.0, first_seen_tick=90, last_seen_tick=100)
    skel = Track(2, "Skeleton", ENEMY_TEAM, 0.9, 9000.0, 10000.0, first_seen_tick=90, last_seen_tick=100)
    frames = [(round(0.05 * k, 4), [hog] + ([] if k == 1 else [skel]), []) for k in range(4)]
    return write_replay_session(root, frames)


def test_a_sparse_replay_is_scored_on_detections(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "classes.txt").write_text("HogRider@ally\n", encoding="utf-8")
    (labels / "00000001.txt").write_text(yolo(0, HOG_BOX) + "\n", encoding="utf-8")
    hog_track = Track(1, "HogRider", OWN_TEAM, 0.9, 5000.0, 20000.0, first_seen_tick=99, last_seen_tick=101)
    hog_det = Detection("HogRider", 0.2, 0.9, HOG_BOX, HOG_FOOT)
    frames = [(0.0, [], []), (2.0, [hog_track], [hog_det]), (4.0, [], [])]      # an image every 2 s
    session = write_replay_session(tmp_path / "rep", frames)
    canon = FNP.Canon(TABLE)
    files = FNP.LabelFiles(labels, VR.read_label_names(VR.find_label_names(labels)), canon)
    cfg = FNP.FitConfig()
    errors = {lv: FNP.LabelErrors(lv, TABLE, PRIOR.size, cfg) for lv in FNP.LEVELS}
    info = FNP.score_session(session, files, GEO, canon, errors, cfg)
    assert info == {"session": str(session), "level": "detections", "sparse": True, "tracked_frames": 3,
                    "median_gap_s": 2.0, "labelled": 1, "scored": 1, "not_scored": 0, "errors": 0, "bad_lines": 0}
    D = errors["detections"]
    assert (D.frames, D.matched["HogRider"], D.side["decided"], D.side["flipped"], errors["tracks"].frames) == (1, 1, 1, 0, 0)
    assert D.ground[0] == pytest.approx((0.0, 0.0), abs=0.05)       # the recorder keeps 4 decimals of a pixel
    assert FNP.pick_errors(errors, cfg) is D

    forced = {lv: FNP.LabelErrors(lv, TABLE, PRIOR.size, cfg) for lv in FNP.LEVELS}
    info = FNP.score_session(session, files, GEO, canon, forced, FNP.FitConfig(level="tracks"))
    assert (info["level"], info["scored"], forced["tracks"].matched["HogRider"]) == ("tracks", 1, 1)
    assert files.warnings() == []                           # the same session twice is one session
    other = write_replay_session(tmp_path / "rep", frames, name="rec2_replay_20261030_130000")
    FNP.score_session(other, files, GEO, canon, {lv: FNP.LabelErrors(lv, TABLE) for lv in FNP.LEVELS}, cfg)
    assert files.warnings() == ["1 label files were scored in more than one session (e.g. 00000001): label files "
                                "are matched by image stem, so give one recording per labels folder"]


# ---------------------------------------------------------------------------
# (b) own taps
# ---------------------------------------------------------------------------

def tt(tick):
    """Capture time of a tick: one frame a tick."""
    return 10.0 + (tick - 100) / 20.0


#: (uid, label, team, first seen, first report, last report, x, y, moves from tick, dy per tick)
SPEC = [
    (7, "HogRider", ENEMY_TEAM, 128, 129, 140, 13500.0, 20500.0, None, 0.0),   # the opponent's, at tap A's aim
    (1, "HogRider", OWN_TEAM, 130, 131, 175, 13600.0, 20300.0, 147, -100.0),   # tap A: +100 / -200 mt off
    (2, "Musketeer", OWN_TEAM, 145, 147, 175, 8400.0, 26700.0, 162, -50.0),    # tap B: -100 / +200 mt off
    (3, "Skeleton", OWN_TEAM, 150, 154, 170, 12500.0, 22500.0, None, 0.0),     # tap C: three units
    (4, "Skeleton", OWN_TEAM, 151, 155, 171, 12800.0, 22500.0, None, 0.0),
    (5, "Skeleton", OWN_TEAM, 152, 156, 172, 12200.0, 22500.0, None, 0.0),
    (6, "Cannon", OWN_TEAM, 162, 168, 200, 9500.0, 24000.0, None, 0.0),        # tap D, a building
]
#: tick -> (card, name, slot, screen tile). E is a spell, F is never seen.
TAPS = {110: (2, "HogRider", 1, (13, 20)), 120: (5, "Musketeer", 0, (8, 26)), 130: (6, "Skeletons", 2, (12, 22)),
        140: (0, "Cannon", 3, (9, 24)), 150: (1, "Fireball", 1, (9, 8)), 160: (3, "IceGolemite", 0, (4, 18))}
CLASSES = LP.CardClasses(
    {"HogRider": {"HogRider"}, "Musketeer": {"Musketeer"}, "Cannon": {"Cannon"}, "IceGolemite": {"IceGolemite"},
     "Skeletons": {"Skeleton"}, "Fireball": {"Fireball", "fireball"}},
    {"Fireball": "spell", "Cannon": "building", "HogRider": "troop", "Musketeer": "troop", "IceGolemite": "troop",
     "Skeletons": "troop"})
COMPOSITIONS = {"HogRider": 1, "Musketeer": 1, "Skeletons": 3, "IceGolemite": 1, "Cannon": 1}


def tracks_at(tick):
    out = []
    for uid, label, team, first, rep0, rep1, x, y, move, dy in SPEC:
        if not rep0 <= tick <= rep1:
            continue
        if move is not None and tick >= move:           # deployed: it walks off
            y += dy * (tick - move + 1)
        if uid == 2 and tick == rep0:
            team = UNKNOWN_TEAM                          # its first report: the team vote undecided
        out.append(Track(uid, label, team, 0.9, x, y, first_seen_tick=first, last_seen_tick=tick))
    return out


def write_tap_session(root, name="vision_play_20261030_200000"):
    """What scripts/vision_play.py --tap writes: battle frames every tick from 100 to 200, a tap
    record (logged 40 ms after its frame) before the frame it was decided on, a menu frame with a
    stray track, and a tap of no battle."""
    clock = FakeClock()
    rec = VisionRecorder(root, session=name, image_every=0, clock=clock, wall_clock=lambda: 0.0)
    for tick in range(100, 201):
        if tick in TAPS:
            card, cname, slot, tile = TAPS[tick]
            clock.t = tt(tick) + 0.04
            rec.log("tap", battle=1, tick=tick, card=card, name=cname, slot=slot, model_tile=[17 - tile[0], tile[1]],
                    screen_tile=list(tile), ms={"capture_to_tap": 40.0})
        vf = VisionFrame(t_capture=tt(tick), tick=tick, tracks=tracks_at(tick), screen=SCREEN_BATTLE, frame_size=SIZE)
        rec.write(vf, None, battle=1, state="battle", ms={"stale": 10.0})
    stray = Track(1, "HogRider", OWN_TEAM, 0.9, 13500.0, 20500.0, first_seen_tick=125, last_seen_tick=126)
    rec.write(VisionFrame(t_capture=20.0, tracks=[stray], screen=SCREEN_MENU, frame_size=SIZE), None,
              battle=None, state="idle")
    clock.t = 100.0
    rec.log("tap", tick=400, card=2, name="HogRider", slot=0, screen_tile=[13, 20], ms={})
    rec.close()
    return rec.session_dir


def test_track_histories_and_compositions(tmp_path):
    hist = FNP.read_track_histories(write_tap_session(tmp_path))
    assert sorted(hist) == [1] and len(hist[1]) == 7      # the menu frame's stray track is no battle's
    a = hist[1][(1, 130)]
    assert (a.reports[0].tick, a.first_xy, a.label, a.form, a.own, len(a.reports)) == (
        131, (13600.0, 20300.0), "HogRider", 0, True, 45)
    assert hist[1][(2, 145)].own and not hist[1][(7, 128)].own

    comps = FNP.load_compositions()
    assert (comps["HogRider"], comps["Skeletons"], comps["Musketeer"]) == (1, 3, 1)
    p = tmp_path / "spawn_children.json"
    p.write_text(json.dumps({"compositions": {"A": {"x": 2, "y": 1}, "B": ["u", "v", "w"]}}), encoding="utf-8")
    assert FNP.load_compositions(p) == {"A": 3, "B": 3}


def test_own_taps_give_the_position_error_and_the_confirmation_delay(tmp_path):
    session = write_tap_session(tmp_path)
    T = FNP.own_tap_errors([session], CLASSES, TABLE, COMPOSITIONS, FNP.FitConfig(min_tracks=2))
    assert (dict(T.taps), T.seen, T.unseen) == ({"troop": 5, "building": 1, "spell": 1}, 4, 2)
    assert T.delays == [1, 2, 4, 4, 4, 6]
    # tap A takes our Hog (first seen 130), not the opponent's at its aim (128): its 16 reports before
    # 130 + 20 - 3 = 147; tap B's 15 reports before 145 + 17 = 162
    assert (T.position_tracks, len(T.position), T.position_outliers) == (2, 31, 0)
    assert T.position == [(100.0, -200.0)] * 16 + [(-100.0, 200.0)] * 15
    plays = {p["tick"]: p for p in T.plays}
    assert plays[110] == {"session": session.name, "battle": 1, "tick": 110, "name": "HogRider", "kind": "troop",
                          "uids": [1], "delays": [1], "position_samples": 16, "label": "HogRider", "lag_ticks": 20}
    assert (plays[130]["uids"], plays[130]["delays"], plays[130]["position_samples"]) == ([3, 4, 5], [4, 4, 4], 0)
    assert (plays[140]["uids"], plays[140]["lag_ticks"]) == ([6], 22)
    assert plays[150]["skipped"] and plays[160]["uids"] == [] and plays[400]["battle"] is None
    assert T.summary()["delays"] == {"n": 6, "mean": 3.5, "median": 4.0, "p10": 1.0, "p90": 6.0, "min": 1.0,
                                     "max": 6.0}

    res = FNP.fit("default", None, T, FNP.FitConfig(min_tracks=2))
    p = res.profile
    assert res.measured == ["position.sigma_x_mt", "position.sigma_y_mt", "new_units.delay_ticks",
                            "new_units.jitter_ticks"]
    assert (p.position.sigma_x_mt, p.position.sigma_y_mt, p.position.flyer_sigma_mt) == (100.0, 200.0, 400.0)
    # the prior's first-sighting lag (1) + the median confirmation 4; half the p10..p90 spread (1..6): 2.5 -> 3
    assert (p.new_units.delay_ticks, p.new_units.jitter_ticks) == (5, 3)
    n = res.notes
    assert n["position"]["measured"]["sigma_x_mt"] == {"from": "own_taps", "n": 2, "samples": 31, "bias_mt": 3.2,
                                                       "std_mt": 99.9}
    assert n["position"]["measured"]["sigma_y_mt"] == {"from": "own_taps", "n": 2, "samples": 31, "bias_mt": -6.5,
                                                       "std_mt": 199.9}
    assert n["new_units"]["measured"]["delay_ticks"] == {"from": "own_taps", "n": 6, "confirm_median": 4.0,
                                                         "age_offset_ticks": 1}
    assert n["new_units"]["measured"]["jitter_ticks"] == {"from": "own_taps", "n": 6, "p10": 1.0, "p90": 6.0}
    assert n["drop"]["why"] == FNP.NO_LABELS and n["position"]["why"] == FNP.FLYER_WHY
    assert res.warnings == []                              # a bias of 7 mt: no geometry warning
    for g in ("drop", "forms", "confusion", "side_flip", "false_positives", "hp", "towers", "elixir"):
        assert getattr(p, g) == getattr(PRIOR, g), g


def test_a_biased_position_warns_to_refit_the_geometry():
    T = FNP.TapErrors()
    T.position, T.position_tracks = [(300.0, 0.0)] * 4, 2
    res = FNP.fit("default", None, T, FNP.FitConfig(min_tracks=2))
    assert (res.profile.position.sigma_x_mt, res.profile.position.sigma_y_mt) == (300.0, 0.0)
    assert len(res.warnings) == 1 and res.warnings[0].startswith("own units read 300 mt off their aimed tiles")
    assert res.notes["new_units"]["why"] == "fewer samples than the minimum (--min-tracks 2)"


# ---------------------------------------------------------------------------
# fallbacks, settings, the command line
# ---------------------------------------------------------------------------

def test_fallbacks_keep_the_base_profile():
    res = FNP.fit("default")
    assert res.measured == [] and res.profile.digest() == PRIOR.digest() and res.profile.name == "measured"
    n = res.notes
    assert n["drop"]["why"] == FNP.NO_LABELS and n["new_units"]["why"] == FNP.NO_TAPS
    assert n["position"]["why"] == "; ".join([FNP.NO_TAPS, FNP.NO_LABELS, FNP.FLYER_WHY])
    assert n["confusion"]["why"] == FNP.KEPT_FIELD_WHY[("confusion", "unknown_p")] + "; " + FNP.NO_LABELS
    assert n["board"]["why"] == FNP.KEPT_WHY["board"] and n["size"]["kept"] == ["small_max_hp", "large_min_hp"]
    assert "from nothing" in res.profile.description

    E = FNP.LabelErrors(FNP.LEVEL_TRACKS, TABLE, PRIOR.size)          # the default minimums: 30 and 50
    E.add(TRUTH, tracks_frame(0))
    res = FNP.fit("default", E)
    assert res.measured == [] and res.profile.digest() == PRIOR.digest()
    assert res.notes["drop"]["why"] == "fewer samples than the minimum (--min-samples 30 / --min-class-samples 50)"
    assert res.notes["side_flip"]["why"] == "fewer samples than the minimum (--min-samples 30)"
    assert FNP.fit("none").profile.digest() == PN.NoiseProfile().digest()

    for bad in ({"level": "frames"}, {"iou": 0.0}, {"min_samples": 0}, {"radius_mt": 0.0}, {"deploy_margin_ticks": -1}):
        with pytest.raises(ValueError):
            FNP.FitConfig(**bad)


def test_command_line_writes_a_profile_that_loads(tmp_path, capsys):
    replay = labelled_replay(tmp_path / "replays")
    labels = write_labels(tmp_path / "labels")
    taps = write_tap_session(tmp_path / "vision_play")
    out, rep = tmp_path / "noise" / "measured.json", tmp_path / "fit.json"
    args = [str(replay), "--labels", str(labels), "--taps", str(taps), "--min-samples", "2",
            "--min-class-samples", "2", "--min-tracks", "2"]
    assert FNP.main(args + ["--out", str(out), "--report", str(rep)]) == 0
    assert "profile -> " in capsys.readouterr().out

    p = PN.NoiseProfile.load(out)
    assert p.name == "measured"
    # labels, 3 frames on tracks: the Skeleton dropped on 1 of 3, the Hog never; no flip in 5; no phantom
    assert (p.drop.small, p.drop.medium, p.drop.large, p.drop.building) == (0.3333, 0.0, 0.03, 0.02)
    assert p.drop.by_class == {"HogRider": 0.0, "Skeleton": 0.3333}
    assert p.confusion.pairs == prior_pairs_without("HogRider")
    assert (p.side_flip.p, p.false_positives.rate, p.false_positives.near_sigma_mt) == (0.0, 0.0, 700.0)
    # own taps (the real class table and spawn_children.json): as measured above
    assert (p.position.sigma_x_mt, p.position.sigma_y_mt) == (100.0, 200.0)
    assert (p.new_units.delay_ticks, p.new_units.jitter_ticks) == (5, 3)
    assert (p.hp, p.towers, p.clock, p.opponent_plays) == (PRIOR.hp, PRIOR.towers, PRIOR.clock, PRIOR.opponent_plays)

    doc = json.loads(out.read_text(encoding="utf-8"))
    fit = doc["_fit"]
    assert fit["measured"] == ["drop.small", "drop.medium", "drop.by_class", "confusion.pairs", "side_flip.p",
                               "false_positives.rate", "position.sigma_x_mt", "position.sigma_y_mt",
                               "new_units.delay_ticks", "new_units.jitter_ticks"]
    assert fit["inputs"] == {"labels": str(labels), "sessions": [str(replay)], "taps": [str(taps)]}
    assert len(fit["warnings"]) == 1 and "share a stem" in fit["warnings"][0]
    assert doc["hp"]["_fit"]["why"] == FNP.KEPT_WHY["hp"]
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert (report["labels"]["frames"], report["labels"]["level"], report["taps"]["seen"]) == (3, "tracks", 4)
    s = report["sessions"][0]
    assert (s["level"], s["sparse"], s["labelled"], s["scored"], s["not_scored"]) == ("tracks", False, 3, 3, 0)
    assert PN.main([str(out)]) == 0                        # the E17 checker reads it

    dry = tmp_path / "dry.json"
    assert FNP.main(args + ["--out", str(dry), "--dry-run"]) == 0 and not dry.exists()
    rec = VisionRecorder(tmp_path / "empty", session="vision_play_20261030_210000", image_every=0)
    rec.close()                                            # a dry run's session: no taps
    empty, nothing = rec.session_dir, tmp_path / "nothing.json"
    assert FNP.main(["--taps", str(empty), "--out", str(nothing)]) == 1 and not nothing.exists()
    assert FNP.main([]) == 2 and FNP.main([str(replay)]) == 2
    assert FNP.main(["--taps", str(tmp_path / "missing")]) == 2


def test_reads_logs_only():
    tree = ast.parse((ROOT / "tools" / "fit_noise_profile.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
    assert not [m for m in mods if "gamestate" in m or "memread" in m or "hcr_reader" in m]
    assert not mods & {"subprocess", "socket", "scripts.vision.control", "scripts.vision.capture"}
