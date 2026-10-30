"""N09: the unit harvest scan (scripts/vision/unit_harvest.py).

No device, GPU or real detector (one `gpu` test runs the pinned detector when it exists). The
pieces are checked on hand-made arrays: the masked median in bands, blobs on a cell grid
(closing, the area filter, the breadth-first labelling against scipy's when it imports), the
foreground mask, the proposal merge (side and class duplicates, live detections, blobs), the
background epochs (tower falls, the length limit), the position-only harvest tracker,
last_assignment back to the live detections, and the harvest detector pass on a fake model.

The whole scan runs on a SYNTHETIC ARENA: a 360x640 recording (npy images, 20 a second, the
built-in geometry, where world millitiles map to pixels as px = 26 + 0.017 x, py = 50.4 +
0.0138 y) of a tiled grass board with noise and the HUD panel's blue edge, and three moving
squares from frame 25 on: a red one the fake model reads as an enemy Knight at 0.8 and an
ally Knight at 0.1 (the live chain at 0.35 sees the enemy one; the harvest pass both), an
orange one read as an enemy Valkyrie at 0.2 (only the harvest pass), and a magenta one no box
covers (a blob only). The fake model finds the squares by colour in the crop it is given, so
the live detector and the harvest pass really run on the pixels; the live chain is
vision_replay's ReplayPerception with a scripted HUD, the real Detector, a RecordingTracker
and the event deriver, driven by vision_replay.Replay.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "tools"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.vision import detector as D  # noqa: E402
from scripts.vision import events as EV  # noqa: E402
from scripts.vision import unit_harvest as UH  # noqa: E402
from scripts.vision.geometry import ArenaGeometry  # noqa: E402
from scripts.vision.labels import load_class_table  # noqa: E402
from scripts.vision.record import VisionRecorder  # noqa: E402
from scripts.vision.tracker import Tracker, observations_from_detections  # noqa: E402
from scripts.vision.types import ENEMY_TEAM, SCREEN_BATTLE, Detection, Hud, VisionFrame  # noqa: E402

GEOM = ArenaGeometry.builtin()
TABLE = load_class_table()
SIZE = (360, 640)
HOG26 = ("HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons", "Fireball", "Log")


# ---------------------------------------------------------------------------
# background median, blobs, foreground
# ---------------------------------------------------------------------------

def test_masked_median_ignores_masked_samples_and_works_in_bands():
    bg = np.full((9, 5, 3), 50, np.uint8)
    bg[..., 1] = 80
    stack = np.stack([bg.copy() for _ in range(7)])
    for k in range(5):                                  # a unit stands there in 5 of 7 frames
        stack[k, 2:5, 1:3] = 200
    masks = np.zeros((7, 9, 5), bool)
    masks[:5, 2:5, 1:3] = True
    assert UH.masked_median(stack)[3, 1].tolist() == [200, 200, 200]     # the plain median keeps it
    med = UH.masked_median(stack, masks, band_rows=2)
    assert np.array_equal(med, bg)
    assert np.array_equal(UH.masked_median(stack, masks, band_rows=64), med)
    even = stack.copy()                                 # two samples left: their mean, half up
    even[0, 0, 0], even[1, 0, 0] = 10, 21
    m2 = masks.copy()
    m2[2:, 0, 0] = True
    assert UH.masked_median(even, m2)[0, 0].tolist() == [16, 16, 16]
    every = stack.copy()                                # masked in every sample: the plain median
    for k in range(7):
        every[k, 8, 4] = k + 1
    m3 = masks.copy()
    m3[:, 8, 4] = True
    assert UH.masked_median(every, m3)[8, 4].tolist() == [4, 4, 4]
    with pytest.raises(ValueError):
        UH.masked_median(stack, masks[:3])


def test_fit_background_only_models_the_rect():
    imgs = [np.full((20, 12, 3), v, np.uint8) for v in (10, 30, 20)]
    out = UH.fit_background(imgs, None, rect=(2, 4, 8, 10), band_rows=3)
    assert (out[4:10, 2:8] == 20).all()
    assert (out[:4] == 10).all() and (out[:, :2] == 10).all()           # outside: the first sample


def test_find_blobs_on_cells_with_closing_and_the_area_filter():
    m = np.zeros((60, 40), bool)
    m[4:14, 6:16] = True                                # a 10x10 unit
    m[30:40, 20:26] = True                              # one unit cut in two by a one-cell gap
    m[30:40, 28:34] = True
    m[50, 2] = True                                     # a speck: a quarter of a cell
    m[50:52, 30:32] = True                              # one full cell: under the area floor
    closed = UH.find_blobs(m, cell=2, min_frac=0.35, min_area_px=10)
    assert [(b.box, b.area_px) for b in closed] == [((6.0, 4.0, 16.0, 14.0), 100), ((20.0, 30.0, 34.0, 40.0), 120)]
    apart = UH.find_blobs(m, cell=2, min_frac=0.35, close=0)
    assert [(b.box, b.area_px) for b in apart] == [
        ((6.0, 4.0, 16.0, 14.0), 100), ((20.0, 30.0, 26.0, 40.0), 60), ((28.0, 30.0, 34.0, 40.0), 60),
        ((30.0, 50.0, 32.0, 52.0), 4)]
    banded = UH.find_blobs(m, cell=2, min_frac=0.35, min_area_px=10, max_area_px=110)
    assert banded == [closed[0]]                        # the joined 120 px unit is over the cap
    capped = UH.find_blobs(m, cell=2, min_frac=0.35, max_area_px=110)
    assert capped == [closed[0], apart[3]]              # no floor by default (as in `apart`): the speck stays
    assert UH.find_blobs(np.zeros((8, 8), bool)) == []


def test_breadth_first_labels_match_scipy():
    rng = np.random.default_rng(1)
    a = rng.random((40, 50)) < 0.3
    labels, n = UH.label_components(a, use_scipy=False)
    assert n > 10 and labels.max() == n and ((labels > 0) == a).all()
    for k in range(1, n + 1):                           # no foreground neighbour of a component is outside it
        grown = UH._dilate(labels == k) & a
        assert (labels[grown] == k).all()
    first = [np.flatnonzero(labels.ravel() == k)[0] for k in range(1, n + 1)]
    assert first == sorted(first)                      # numbered in raster order
    if importlib.util.find_spec("scipy") is None:
        return                                          # label_components falls back to the same fill
    ref, m = UH.label_components(a, use_scipy=True)
    assert m == n and np.array_equal(ref, labels)


def test_foreground_mask_keeps_the_region_and_drops_excluded_boxes():
    bg = np.zeros((20, 20, 3), np.uint8)
    img = bg.copy()
    img[2:6, 2:6] = (100, 0, 0)                         # a unit
    img[2:6, 12:16] = (0, 100, 0)                       # outside the region
    img[10:14, 2:6] = (0, 0, 100)                       # under an HP bar box
    img[16:18, 16:18] = (20, 20, 20)                    # below the threshold
    region = np.zeros((20, 20), bool)
    region[:, :10] = True
    fg = UH.foreground_mask(img, bg, threshold=30, region=region, exclude=[(1.5, 9.5, 6.2, 14.0)])
    assert fg.sum() == 16 and fg[2:6, 2:6].all()
    assert UH.box_mask((20, 20), [(1.5, 9.5, 6.2, 14.0)]).sum() == 6 * 5
    assert UH.box_mask((20, 20), [(5, 5, 6, 6)], pad_px=1).sum() == 9
    with pytest.raises(ValueError):
        UH.foreground_mask(img, bg[:10])


# ---------------------------------------------------------------------------
# the proposal merge
# ---------------------------------------------------------------------------

def test_merge_proposals_groups_sides_and_classes_and_attaches_live_and_blobs():
    R, L, B = UH.RawBox, UH.LiveDet, UH.Blob
    raw = [R("Knight", "Knight", 1, 0.8, (10, 10, 30, 40), "unit", 0, 20, 40),
           R("Knight", "Knight", 0, 0.1, (11, 10, 31, 41), "unit", 0, 21, 41),       # the ally box: no side merge
           R("Valkyrie", "Valkyrie", 1, 0.3, (10, 11, 30, 40), "unit", 0, 20, 40),   # another class, same sprite
           R("Valkyrie", "Valkyrie", 1, 0.2, (100, 100, 120, 130), "unit", 0, 110, 130),
           R("PrincessTower", "PrincessTower", 1, 0.9, (200, 10, 260, 70), "tower", 0, 230, 70),
           R("bar", "bar", 1, 0.7, (12, 5, 28, 8), "ui", 0, 20, 6),
           R("Fireball", "fireball", 1, 0.04, (150, 150, 170, 170), "spell", 0, 160, 160)]   # under prop_conf
    blobs = [B((9, 9, 31, 42), 600), B((300, 300, 320, 330), 500), B((101, 95, 119, 125), 400)]
    live = [L(7, "Knight", "unit", 0.8, 0.9, 0, (10, 10, 30, 40), 1),
            L(8, "Knight", "unit", 0.5, 0.9, 0, (12, 12, 32, 42), 1),      # overlaps the taken group less
            L(9, "HogRider", "unit", 0.6, 0.1, 0, (400, 400, 420, 430), 1),
            L(None, "PrincessTower", "tower", 0.9, 0.9, 0, (200, 10, 260, 70), 1)]
    props = UH.merge_proposals(raw, blobs, live)
    got = [(p.src, p.type, p.conf, p.cands, (p.x_mt, p.y_mt), p.blob, p.live_uid, p.live_seg, p.label)
           for p in props]
    assert got == [
        ("det+blob", "unit", 0.8, [("Knight", 1, 0.8), ("Valkyrie", 1, 0.3), ("Knight", 0, 0.1)], (20, 40), 0, 7, 1,
         "Knight"),
        ("det+blob", "unit", 0.2, [("Valkyrie", 1, 0.2)], (110, 130), 2, None, None, "Valkyrie"),
        ("live", "unit", 0.5, [("Knight", 1, 0.5)], (22.0, 42.0), None, 8, 1, "Knight"),
        ("live", "unit", 0.6, [("HogRider", 0, 0.6)], (410.0, 430.0), None, 9, 1, "HogRider"),
        ("blob", "blob", 0.0, [], (310.0, 330.0), 1, None, None, None)]
    assert props[0].box == (10, 10, 30, 40)            # the strongest box, not the blob's
    for p in props:
        assert UH.Proposal.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    assert props[0].to_dict()["live"] == [1, 7] and props[4].to_dict()["cands"] == []
    # a stricter grouping splits the cross-class box (IoU 0.967) and the ally box (0.877) off
    strict = UH.merge_proposals(raw, (), (), config=UH.ScanConfig(group_iou=0.97))
    assert [(p.src, p.cands) for p in strict] == [("det", [("Knight", 1, 0.8)]), ("det", [("Valkyrie", 1, 0.3)]),
                                                  ("det", [("Valkyrie", 1, 0.2)]), ("det", [("Knight", 0, 0.1)])]


def test_rows_round_trip():
    raw = UH.RawBox("Knight", "Knight", None, 0.81234, (1.04, 2.0, 3.0, 4.06), "unit", 1, 9000.4, 10000.6)
    back = UH.RawBox.from_row(json.loads(json.dumps(raw.to_row())))
    assert back.side is None and back.conf == 0.812 and back.box == (1.0, 2.0, 3.0, 4.1)
    assert (back.x_mt, back.y_mt, back.form) == (9000.0, 10001.0, 1)
    assert len(raw.to_row()) == len(UH.RAW_FIELDS)
    live = UH.LiveDet(None, "bar", "ui", 0.5, 0.5, 0, (1, 2, 3, 4), seg=3)
    assert UH.LiveDet.from_row(live.to_row(), 3) == live and len(live.to_row()) == len(UH.LIVE_FIELDS)
    blob = UH.Blob((1.0, 2.0, 3.0, 4.0), 7)
    assert UH.Blob.from_row(blob.to_row()).to_row() == blob.to_row() and len(blob.to_row()) == len(UH.BLOB_FIELDS)


# ---------------------------------------------------------------------------
# background epochs
# ---------------------------------------------------------------------------

def _entries():
    es = []
    for t in range(300):
        towers = {"enemy_left": 2000 if t < 100 else 0,       # falls at 100 s
                  "enemy_right": 2000 if t < 297 else 0,      # falls 2 s before the end: no new epoch
                  "own_king": 0 if t <= 5 else None}          # a 0 before it was ever read standing
        es.append(UH.FrameEntry(t, float(t), 1, Path(f"{t}.npy"), SIZE, [], towers))
    for j in range(3):                                         # a second battle
        es.append(UH.FrameEntry(300 + j, 400.0 + j, 2, Path("x.npy"), SIZE, [], {}))
    return es


def test_tower_falls_and_epochs():
    es = _entries()
    assert UH.tower_falls(es[:300], 3) == [(100.0, "enemy_left"), (297.0, "enemy_right")]
    assert UH.tower_falls(es[:99] + es[100:102], 3) == []        # two zero reads are not a fall
    cfg = UH.ScanConfig(epoch_s=120.0, min_epoch_s=5.0, bg_samples=4, tower_fall_frames=3)
    eps = UH.plan_epochs(es, cfg)
    got = [(e.id, e.hk, e.reason, e.t0, len(e.members), e.members[0], e.samples) for e in eps]
    assert got == [("b1e0", 1, "start", 0.0, 100, 0, [0, 33, 66, 99]),
                   ("b1e1", 1, "tower enemy_left", 100.0, 100, 100, [100, 133, 166, 199]),
                   ("b1e2", 1, "length", 199.5, 100, 200, [200, 233, 266, 299]),
                   ("b2e0", 2, "start", 400.0, 3, 300, [300, 301, 302])]
    assert eps[-1].to_dict()["t1"] is None and eps[0].to_dict()["t1"] == 100.0
    one = UH.plan_epochs(es, UH.ScanConfig(epoch_s=0.0, bg_samples=1))
    assert [(e.id, len(e.members), e.samples) for e in one] == [("b1e0", 100, [50]), ("b1e1", 200, [200]),
                                                                ("b2e0", 3, [301])]


# ---------------------------------------------------------------------------
# the harvest tracker, last_assignment and the harvest detector pass
# ---------------------------------------------------------------------------

def test_harvest_tracker_follows_position_whatever_the_label():
    ht = UH.HarvestTracker()
    ht.reset(1)
    for k in range(10):
        a, b = ("Knight", "Valkyrie") if k % 2 == 0 else ("Valkyrie", "Knight")
        props = [UH.Proposal((0, 0, 1, 1), "det", "unit", 0.5, [(a, 1, 0.5)], 4000.0 + 100 * k, 10000.0),
                 UH.Proposal((0, 0, 1, 1), "det", "unit", 0.5, [(b, 1, 0.5)], 12000.0, 10000.0 + 100 * k),
                 UH.Proposal((0, 0, 1, 1), "blob", "blob", 0.0, [], 9000.0, 20000.0)]
        assert ht.update(props, t=k * 0.05, n=k) == [1, 2, 3] and [p.h for p in props] == [1, 2, 3]
    one = ht.tracks[1].to_dict()
    assert one["n"] == 10 and one["frames"][:2] == [[0, 0], [1, 0]] and one["hk"] == 1
    assert one["cands"] == {"Knight@enemy": {"sum": 2.5, "max": 0.5, "frames": 5},
                            "Valkyrie@enemy": {"sum": 2.5, "max": 0.5, "frames": 5}}
    assert one["birth"] == [4000, 10000] and one["last"] == [4900, 10000] and one["src"] == {"det": 10}
    assert ht.tracks[3].to_dict()["cands"] == {} and ht.tracks[3].to_dict()["src"] == {"blob": 10}
    ht.reset(2)                                          # a new battle: new ids, the old ones kept
    props = [UH.Proposal((0, 0, 1, 1), "blob", "blob", 0.0, [], 4900.0, 10000.0)]
    assert ht.update(props, t=99.0, n=50) == [4] and ht.tracks[4].hk == 2 and len(ht.tracks) == 4
    assert ht.tracker.config.class_cost == 0.0 and ht.tracker.config.team_cost == 0.0


def det_at(cls, x, y, p_enemy=0.9, conf=0.9, half_w=7.0, height=14.0):
    gx, gy = (float(v) for v in GEOM.world_to_px(x, y, SIZE))
    return Detection(cls, p_enemy, conf, (gx - half_w, gy - height, gx + half_w, gy), (gx, gy))


def test_live_uids_follow_last_assignment():
    k1, k2 = det_at("Knight", 9000.0, 10000.0), det_at("Knight", 12000.0, 10000.0)    # same class and score
    fb = Detection("Fireball", 0.9, 0.6, (150.0, 150.0, 170.0, 170.0))
    bar, tower = Detection("bar", 0.9, 0.5, (150.0, 140.0, 165.0, 143.0)), det_at("PrincessTower", 3500.0, 6500.0)
    dets = [k1, k2, fb, bar, tower]
    types = ["unit", "unit", "spell", "ui", "tower"]
    step = SimpleNamespace(vf=VisionFrame(t_capture=0.0, detections=dets, frame_size=SIZE), det=None)
    rt = UH.RecordingTracker()
    rt.update(observations_from_detections([k2, fb, k1], GEOM, SIZE), 10)      # any order
    assert rt.updates == 1 and rt.last_assignment == [1, 2, 3] and len(rt.last_observations) == 3
    assert UH.live_uids(step, rt, GEOM, types, updated=True) == ([3, 1, 2, None, None], "observations")
    assert UH.match_observations(dets, rt.last_observations, GEOM, SIZE) == [1, 2, 0]
    plain = Tracker()                                   # no observations kept: ReplayPerception's order
    plain.update(observations_from_detections([k1, k2, fb], GEOM, SIZE), 10)
    assert UH.live_uids(step, plain, GEOM, types, updated=True) == ([1, 2, 3, None, None], "order")
    short = Tracker()
    short.update(observations_from_detections([k1], GEOM, SIZE), 10)
    assert UH.live_uids(step, short, GEOM, types, updated=True) == ([None] * 5, "mismatch")
    assert UH.live_uids(step, rt, GEOM, types, updated=False) == ([None] * 5, "none")
    rt.reset()
    assert rt.last_observations == [] and rt.updates == 1


class FakeBoxes:
    def __init__(self, rows):
        a = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = a[:, :4], a[:, 4], a[:, 5]


class ListModel:
    """names + predict(): fixed rows (x1, y1, x2, y2, conf, class) in crop pixels, cut at conf."""

    def __init__(self, names, rows):
        self.names = dict(names)
        self.rows = list(rows)
        self.calls = []

    def predict(self, source=None, **kw):
        self.calls.append((source.shape, kw))
        keep = [r for r in self.rows if r[4] >= kw["conf"]][:kw["max_det"]]
        return [SimpleNamespace(boxes=FakeBoxes(keep))]


def test_low_conf_pass_keeps_both_sides_and_maps_the_crop(monkeypatch):
    # the pinned ultralytics 8.4.192's config: precision goes to predict() as quantize, not half
    monkeypatch.setattr(D, "_ultralytics_cfg_defaults", lambda: {"quantize": None})
    g = ArenaGeometry.builtin()                          # 1080x1920: crop (27, 68, 1047, 1518)
    names = {0: "Knight@ally", 1: "Knight@enemy", 2: "no-such-thing@enemy", 3: "bar@enemy", 4: "Fireball@enemy"}

    def row(frame_box, conf, cls):
        x1, y1, x2, y2 = frame_box
        return (x1 - 27, y1 - 68, x2 - 27, y2 - 68, conf, cls)

    hog = (517.0, 505.2, 557.0, 565.2)                   # stands at (9000, 10000)
    rows = [row(hog, 0.1, 0), row(hog, 0.8, 1), row((127.0, 68.0, 167.0, 89.1), 0.5, 1),   # off the board
            row((313.0, 379.6, 353.0, 419.6), 0.2, 2), row((522.0, 495.2, 552.0, 501.2), 0.3, 3),
            row((517.0, 462.4, 557.0, 502.4), 0.55, 4), row(hog, 0.03, 1)]                # under 0.05
    model = ListModel(names, rows)
    lp = UH.LowConfPass.from_detector(D.Detector(model, geometry=g, table=TABLE))
    assert lp.crop_rect((1080, 1920)) == (27, 68, 1047, 1518) and lp.config.cross_class_iou is None
    boxes = lp.detect(np.zeros((1920, 1080, 3), np.uint8))
    shape, kw = model.calls[-1]
    assert shape == (1450, 1020, 3)
    assert (kw["conf"], kw["max_det"], kw["iou"], kw["imgsz"], kw["device"], kw["quantize"]) == (
        0.05, 300, 0.6, 640, "cpu", None)
    assert "half" not in kw and lp.half is False
    assert UH.LowConfPass(model, geometry=g, table=TABLE, device="0").precision == {"quantize": 16}
    monkeypatch.setattr(D, "_ultralytics_cfg_defaults", lambda: {"half": False})     # before 8.4.80
    assert UH.LowConfPass(model, geometry=g, table=TABLE, device="0").precision == {"half": True}
    monkeypatch.setattr(D, "_ultralytics_cfg_defaults", lambda: {"quantize": None})
    got = [(b.cls, b.side, b.conf, b.type) for b in boxes]
    assert got == [("Knight", 1, 0.8, "unit"), ("Fireball", 1, 0.55, "spell"), ("bar", 1, 0.3, "ui"),
                   ("no-such-thing", 1, 0.2, "unknown"), ("Knight", 0, 0.1, "unit")]
    assert boxes[0].box == pytest.approx(hog) and boxes[4].box == pytest.approx(hog)
    assert (boxes[0].x_mt, boxes[0].y_mt) == pytest.approx((9000.0, 10000.0))
    assert (boxes[1].x_mt, boxes[1].y_mt) == pytest.approx((9000.0, 8000.0))             # a spell: its centre
    assert (boxes[3].x_mt, boxes[3].y_mt) == pytest.approx((5000.0, 6000.0))
    assert lp.dropped == 1 and lp.frames == 1 and lp.boxes == 5
    tight = UH.LowConfPass(model, geometry=g, table=TABLE, conf=0.5, max_det=2, iou=0.4)
    assert [b.conf for b in tight.detect(np.zeros((1920, 1080, 3), np.uint8))] == [0.8]   # 0.5 off board
    assert model.calls[-1][1]["iou"] == 0.4 and model.calls[-1][1]["max_det"] == 2


@pytest.mark.gpu
def test_low_conf_pass_runs_the_pinned_detector():
    try:
        det = D.Detector.load()
    except D.DetectorError as exc:
        pytest.skip(f"no pinned detector: {exc}")
    lp = UH.LowConfPass.from_detector(det)
    boxes = lp.detect(np.zeros((1328, 747, 3), np.uint8))
    assert isinstance(boxes, list) and lp.frames == 1 and lp.crop_rect((747, 1328)) == det.crop_rect((747, 1328))


# ---------------------------------------------------------------------------
# ScanPerception, folders
# ---------------------------------------------------------------------------

class TinyLive:
    def __init__(self):
        self.ticks, self.calls = "time", []

    def new_battle(self):
        self.calls.append("new")

    def end_battle(self, *, keep_towers=False):
        self.calls.append("end kept" if keep_towers else "end")

    def start_clock(self, t):
        self.calls.append(("clock", t))

    def tap_tick(self, tick, t):
        return 7

    def note_own_tap(self, card, tick, **kw):
        self.calls.append(("tap", card, tick, kw["slot"]))


def test_scan_perception_passes_replays_calls_through():
    live = TinyLive()
    sp = UH.ScanPerception(live)
    assert sp.ticks == "time" and sp.live is live
    sp.start_clock(1.5)
    sp.note_own_tap(2, 30, slot=1, aim_x_mt=None, aim_y_mt=None, t=None)
    sp.new_battle()
    sp.end_battle()
    sp.end_battle(keep_towers=True)                  # vision_replay._frame: an end without its end screen
    assert sp.tap_tick(None, 2.0) == 7 and sp.segment == 3
    assert live.calls == [("clock", 1.5), ("tap", 2, 30, 1), "new", "end", "end kept"]
    with pytest.raises(AttributeError):
        sp.no_such_thing


def write_session(root, frames, name="capture_t"):
    rec = VisionRecorder(root, session=name, image_every=1, image_format="npy", clock=lambda: 0.0,
                         wall_clock=lambda: 0.0)
    for k, img in enumerate(frames):
        rec.write(VisionFrame(t_capture=round(k / 20.0, 4), screen=SCREEN_BATTLE, frame_size=SIZE), img)
    rec.close()
    return rec.session_dir


def test_sessions_and_output_folders(tmp_path):
    blank = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    s1 = write_session(tmp_path / "root", [blank])
    assert UH.resolve_session(tmp_path / "root") == s1
    assert UH.resolve_session(s1 / "frames.jsonl") == s1 and UH.resolve_session(s1) == s1
    write_session(tmp_path / "root", [blank], name="capture_u")
    with pytest.raises(UH.ScanError, match="2 recorded sessions"):
        UH.resolve_session(tmp_path / "root")
    with pytest.raises(UH.ScanError, match="no recording"):
        UH.resolve_session(tmp_path / "nothing")
    out = tmp_path / "scan"
    UH.prepare_out_dir(out, s1)
    (out / "scan.json").write_text("{}", encoding="utf-8")
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    (out / "background").mkdir()
    (out / "background" / "b1e0.png").write_bytes(b"x")
    (out / "background" / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(UH.ScanError, match="--force"):
        UH.prepare_out_dir(out, s1)
    UH.prepare_out_dir(out, s1, overwrite=True)
    assert sorted(p.name for p in out.rglob("*")) == ["background", "keep.txt", "notes.txt"]
    with pytest.raises(UH.ScanError, match="mix"):
        UH.prepare_out_dir(s1, s1)
    with pytest.raises(UH.ScanError, match="no finished scan"):
        UH.load_scan(out)


def test_command_line_refuses_bad_input(tmp_path, capsys):
    assert UH.main(["scan", str(tmp_path / "missing"), "--no-detector"]) == 2
    assert "scan: no recording" in capsys.readouterr().out
    assert UH.main(["scan", str(tmp_path), "--low-conf", "1.5"]) == 2
    assert "scan settings" in capsys.readouterr().out
    with pytest.raises(ValueError):
        UH.ScanConfig(blob_min_tiles=5.0, blob_max_tiles=1.0)


# ---------------------------------------------------------------------------
# the scan on a synthetic arena
# ---------------------------------------------------------------------------

RED, ORANGE, MAGENTA = (220, 40, 40), (240, 150, 20), (200, 40, 200)
#: colour -> [(model class index, score)]: what the fake model reads a square of that colour as.
READS = {RED: [(1, 0.8), (0, 0.1)], ORANGE: [(3, 0.2)], MAGENTA: []}
NAMES = {0: "Knight@ally", 1: "Knight@enemy", 2: "Valkyrie@ally", 3: "Valkyrie@enemy"}
FIRST, FRAMES = 25, 60


def arena():
    """The empty board: grass tiles in two shades, the river, dark outside the board, and the
    HUD panel's blue top edge (what makes vision_play's screen cues read a battle)."""
    ys, xs = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    wx, wy = GEOM.px_to_world(xs + 0.5, ys + 0.5, SIZE)
    img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    img[:] = (20, 20, 20)
    board = (wx >= 0) & (wx < 18000) & (wy >= 0) & (wy < 32000)
    light = ((np.floor(wx / 1000) + np.floor(wy / 1000)) % 2) == 0
    img[board & light] = (82, 152, 70)
    img[board & ~light] = (70, 140, 60)
    img[board & (wy >= 15000) & (wy < 17000)] = (70, 120, 150)
    img[int(0.805 * SIZE[1]):int(0.83 * SIZE[1]) + 1] = (30, 144, 230)
    return img


def square_at(x, y):
    gx, gy = GEOM.world_to_px(x, y, SIZE)
    cx, by = int(round(float(gx))), int(round(float(gy)))
    return cx - 7, by - 14, cx + 7, by


def paths(k):
    """(colour, world x, y) of the squares on frame k."""
    if k < FIRST:
        return []
    j = k - FIRST
    return [(RED, 4500.0, 6000.0 + 250.0 * j), (ORANGE, 13500.0, 7000.0 + 200.0 * j),
            (MAGENTA, 3000.0 + 300.0 * j, 21000.0)]


def scene(k, base, rng):
    img = np.clip(base.astype(np.int16) + rng.integers(-2, 3, size=base.shape), 0, 255).astype(np.uint8)
    for colour, x, y in paths(k):
        x0, y0, x1, y1 = square_at(x, y)
        img[y0:y1, x0:x1] = colour
    return img


class ColourModel:
    """A YOLO stand-in that finds the squares by colour in the BGR crop it is given."""

    def __init__(self):
        self.names = dict(NAMES)
        self.calls = []

    def predict(self, source=None, **kw):
        self.calls.append(kw["conf"])
        rows = []
        for colour, reads in READS.items():
            hit = np.all(source == np.array(colour[::-1], np.uint8), axis=2)
            if not hit.any():
                continue
            ys, xs = np.nonzero(hit)
            for cls, conf in reads:
                if conf >= kw["conf"]:
                    rows.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, conf, cls))
        return [SimpleNamespace(boxes=FakeBoxes(rows[:kw["max_det"]]))]


class QuietHud:
    """No clock (the live chain ticks on the capture time), no towers."""

    def read(self, rgb, t):
        return Hud()

    def reset(self):
        pass


def test_scan_of_moving_squares_on_a_synthetic_arena(tmp_path):
    import vision_replay as VR

    base = arena()
    rng = np.random.default_rng(7)
    session = write_session(tmp_path / "rec", [scene(k, base, rng) for k in range(FRAMES)])
    model = ColourModel()
    detector = D.Detector(model, geometry=GEOM, table=TABLE)
    live = VR.ReplayPerception(hud_reader=QuietHud(), tracker=UH.RecordingTracker(own_classes=TABLE.deck_labels(HOG26)),
                               deriver=EV.EventDeriver(EV.EventTables({8: {"Knight": 1}}), own_deck=range(8)),
                               geometry=GEOM, detector=detector, names=VR.Names(TABLE), ticks=VR.TICKS_TIME)
    low = UH.LowConfPass.from_detector(detector)
    lines = []
    s = UH.scan_recording(session, None, live=live, low_pass=low, card_ids={"Knight": 8}, log=lines.append)
    out = session / "scan"
    assert s["out"] == str(out) and (out / "scan.json").is_file() and not (out / UH.PASS1_TMP).exists()
    assert (s["images"], s["frames"], s["battle_frames"]) == (FRAMES, FRAMES, FRAMES)
    assert model.calls == [0.35, 0.05] * FRAMES            # the live pass, then the harvest pass, per image

    # the live chain: the battle starts on frame 20 (a second of clock), the Knight is one play
    assert s["live"]["battles"] == 1 and s["live"]["events"]["opp_cards"] == {"Knight": 1}
    assert s["live_assignment"] == {"observations": FRAMES}
    assert s["harvest_pass"]["conf"] == 0.05 and s["harvest_pass"]["boxes"] == 3 * (FRAMES - FIRST)

    # one background, the empty arena (noise +-2 only)
    assert [(e["id"], e["frames"], e["fitted"]) for e in s["epochs"]] == [("b1e0", FRAMES, 40)]
    scan = UH.load_scan(out)
    bg = scan.background("b1e0")
    assert bg.shape == base.shape and int(np.abs(bg.astype(int) - base.astype(int)).max()) <= 2

    # the records: every image, battle 1 from frame 20, the live uid of the Knight's detection
    recs = list(scan.frames())
    assert [r["n"] for r in recs] == list(range(FRAMES)) and all(r["hk"] == 1 for r in recs)
    assert [r["battle"] for r in recs[19:21]] == [None, 1] and [r["seg"] for r in recs[19:22]] == [0, 0, 1]
    r = recs[FIRST + 5]
    assert r["image"] == f"images/{FIRST + 5:08d}.npy" and r["i"] == FIRST + 5 and r["epoch"] == "b1e0"
    assert [(x[0], x[1], x[2]) for x in r["live"]] == [(1, "Knight", "unit")]
    assert [(t["uid"], t["cls"], t["team"], t["how"], t["card"], t["card_name"]) for t in r["tracks"]] == [
        (1, "Knight", ENEMY_TEAM, "play", 8, "Knight")]
    assert [(x[0], x[2], x[3]) for x in r["raw"]] == [("Knight", 1, 0.8), ("Valkyrie", 1, 0.2), ("Knight", 0, 0.1)]
    assert len(r["blobs"]) == 3
    red_box = square_at(*paths(FIRST + 5)[0][1:])
    props = [UH.Proposal.from_dict(p) for p in r["props"]]
    assert [(p.src, p.label, p.live_uid, p.live_seg, p.h) for p in props] == [
        ("det+blob", "Knight", 1, 1, 1), ("det+blob", "Valkyrie", None, None, 2), ("blob", None, None, None, 3)]
    assert props[0].box == pytest.approx(red_box) and props[0].cands == [("Knight", 1, 0.8), ("Knight", 0, 0.1)]
    assert abs(props[2].y_mt - 21000.0) < 100.0                   # the blob stands where the square does
    assert all(not x.get("props") for x in recs[:FIRST])

    # three harvest tracks over the 35 frames with squares
    h = s["harvest"]
    assert (h["proposals"], h["tracks"], h["tracks_with_live"], h["tracks_blob_only"]) == (105, 3, 1, 1)
    assert h["by_source"] == {"blob": 35, "det+blob": 70}
    tr = {t["h"]: t for t in scan.tracks}
    assert [tr[k]["n"] for k in (1, 2, 3)] == [35, 35, 35] and tr[1]["first_n"] == FIRST
    assert set(tr[1]["cands"]) == {"Knight@enemy", "Knight@ally"} and tr[1]["live"] == {"1:1": 35}
    assert tr[1]["live_cls"] == {"Knight": 35} and tr[2]["live"] == {} and set(tr[2]["cands"]) == {"Valkyrie@enemy"}
    assert tr[3]["cands"] == {} and tr[3]["src"] == {"blob": 35}
    assert abs(tr[1]["birth"][1] - 6000) < 100 and abs(tr[1]["last"][1] - 14500) < 100
    knight = scan.live_tracks["1:1"]
    assert (knight["cls"], knight["team"], knight["how"], knight["card_name"]) == ({"Knight": 34}, ENEMY_TEAM, "play",
                                                                                   "Knight")
    assert any(line.startswith("scan -> ") for line in lines)
    json.dumps(s)
    with pytest.raises(UH.ScanError, match="already holds a scan"):
        UH.scan_recording(session, None, live=live, low_pass=low, log=lines.append)


# ---------------------------------------------------------------------------
# N10: the flags, the look split, clustering, the sheet
# ---------------------------------------------------------------------------

E_TEAM, O_TEAM = 0, 1                               # RoyaleSim teams: ENEMY_TEAM, OWN_TEAM
DECK10 = [("Knight", 0), ("Valkyrie", 0), ("BabyDragon", 1), ("HogRider", 0), ("Zap", 0), ("Goblinstein", 0),
          ("Vines", 0), ("Arrows", 0)]


def unit_box(x, y, half=7.0, height=14.0):
    gx, gy = (float(v) for v in GEOM.world_to_px(x, y, SIZE))
    return [round(gx - half, 1), round(gy - height, 1), round(gx + half, 1), round(gy, 1)]


class Scene:
    """Hand-made scan records (scan_frames.jsonl shape) and the scan's tracks.json, built with
    the real HarvestTrack: harvest tracks as proposals, the live detections that joined them,
    enemy stopwatches (harvest 'clock' boxes) and the live chain's tracks."""

    def __init__(self, frames=40):
        self.recs = [{"n": n, "i": n, "image": f"images/{n:08d}.npy", "t": round(n * 0.05, 4), "screen": "battle",
                      "size": list(SIZE), "seg": 1, "battle": 1, "hk": 1, "epoch": None, "live": [], "raw": [],
                      "plays": [], "props": [], "blobs": [], "tracks": []} for n in range(frames)]
        self.live = {}

    def track(self, h, frames, x, y, *, cands=(), live=None, src="det", dy=0.0, ys=None):
        """live: {cls (or one class per image), team, how, card_name, conf, uid, form}."""
        lk = None
        if live is not None:
            uid = live.get("uid", h)
            lk = f"1:{uid}"
            self.live[lk] = {"first_t": 0.0, "last_t": 1.0, "frames": len(frames), "cls": {}, "team": live["team"],
                             "how": live.get("how"), "card": live.get("card", 1), "card_name": live.get("card_name"),
                             "forms": {"0": len(frames)}}
        for j, n in enumerate(frames):
            yy = (y + dy * j) if ys is None else ys[j]
            box = unit_box(x, yy)
            p = {"box": box, "src": src, "type": "unit" if src != "blob" else "blob", "conf": 0.5,
                 "cands": [list(c) for c in cands], "blob": None, "live": None, "live_cls": None,
                 "x": int(x), "y": int(yy), "h": h}
            if live is not None:
                cls = live["cls"][j] if isinstance(live["cls"], list) else live["cls"]
                p["live"], p["live_cls"] = [1, live.get("uid", h)], cls
                self.recs[n]["live"].append([live.get("uid", h), cls, "unit", live.get("conf", 0.8), 0.9,
                                             live.get("form", 0), *box])
                self.live[lk]["cls"][cls] = self.live[lk]["cls"].get(cls, 0) + 1
            self.recs[n]["props"].append(p)

    def clock(self, frames, x, y, conf=0.6):
        gx, gy = (float(v) for v in GEOM.world_to_px(x, y, SIZE))
        for n in frames:
            self.recs[n]["raw"].append(["clock", "clock", 1, conf, gx - 5, gy - 26, gx + 5, gy - 16, "ui", 0, int(x), int(y)])

    def harvest_tracks(self):
        out = {}
        for r in self.recs:
            for k, p in enumerate(r["props"]):
                ht = out.setdefault(p["h"], UH.HarvestTrack(p["h"], 1))
                ht.add(r["n"], k, r["t"], UH.Proposal.from_dict(p))
        return [out[h].to_dict() for h in sorted(out)]

    def scan(self, out_dir, epochs=()):
        return UH.ScanResult(Path(out_dir), {"session_dir": str(out_dir), "meta": {}, "epochs": list(epochs)},
                             self.harvest_tracks(), self.live)


def flag_scene():
    s = Scene(40)
    s.track(1, range(0, 30), 1500, 4000, cands=[("Knight", 1, 0.8)], dy=50,
            live=dict(cls="Knight", team=E_TEAM, how="play", card_name="Knight"))          # clean: auto-accepted
    s.track(2, range(0, 10), 4000, 24000, src="blob")                                      # a bare blob: S1
    s.track(3, range(0, 12), 6500, 4000, cands=[("Valkyrie", 1, 0.5)],
            live=dict(cls="Valkyrie", team=E_TEAM, how="unknown", card_name="Valkyrie"))   # S2
    s.track(4, range(0, 12), 9000, 4000, cands=[("Giant", 1, 0.5)],
            live=dict(cls="Giant", team=E_TEAM, how="play", card_name="Giant"))            # S3: not the deck
    s.track(5, range(0, 20), 11500, 4000, cands=[("Knight", 1, 0.5)],
            live=dict(cls=["Knight", "Valkyrie"] * 4 + ["Knight"] * 12, team=E_TEAM, how="play",
                      card_name="Knight"))                                                 # S4: 16 / 4
    s.track(6, range(0, 10), 9000, 16000, cands=[("Knight", 1, 0.5)],
            live=dict(cls="Knight", team=E_TEAM, how="play", card_name="Knight"))          # S5: 10 on the water
    s.track(7, range(0, 10), 6500, 16000, cands=[("HogRider", 1, 0.5)],
            live=dict(cls="HogRider", team=E_TEAM, how="play", card_name="HogRider"))      # a river jumper
    s.track(8, range(0, 10), 11500, 16000, cands=[("Knight", 1, 0.5)], ys=[16000] * 5 + [12000] * 5,
            live=dict(cls="Knight", team=E_TEAM, how="play", card_name="Knight"))          # 5 on the water: no S5
    s.track(9, range(0, 10), 1500, 9000, cands=[("Knight", 1, 0.5)],
            live=dict(cls="Knight", team=E_TEAM, how="child", card_name="Knight"))
    s.clock(range(0, 4), 1500, 9000)                                                      # S6: no play matches
    s.track(10, range(0, 10), 6500, 9000, cands=[("Knight", 1, 0.5)],
            live=dict(cls="Knight", team=E_TEAM, how="play", card_name="Knight"))
    s.clock(range(0, 4), 6500, 9000)                                                      # matched: a play
    s.clock(range(20, 24), 16000, 28000)                                                  # nothing under it
    s.track(11, range(0, 12), 9000, 26000, cands=[("HogRider", 0, 0.9)],
            live=dict(cls="HogRider", team=O_TEAM, how="landing", card_name="HogRider"))   # ours: left out
    s.track(12, range(0, 3), 11500, 26000, src="blob")                                     # too short
    s.track(13, range(0, 12), 14000, 9000, cands=[("Giant", 1, 0.9)])                      # enemy look, our live track
    for n in range(6):
        p = next(p for p in s.recs[n]["props"] if p["h"] == 13)
        p["live"], p["live_cls"] = [1, 113], "HogRider"
        s.recs[n]["live"].append([113, "HogRider", "unit", 0.6, 0.2, 0, *p["box"]])
    s.live["1:113"] = {"first_t": 0.0, "last_t": 0.3, "frames": 6, "cls": {"HogRider": 6}, "team": O_TEAM,
                       "how": "landing", "card": 1, "card_name": "HogRider", "forms": {"0": 6}}
    return s


def test_flags_on_hand_made_tracks(tmp_path):
    s = flag_scene()
    grey = np.full((SIZE[1], SIZE[0], 3), 100, np.uint8)
    lines = []
    an = UH.analyze_scan(s.scan(tmp_path), deck=DECK10, table=TABLE, geometry=GEOM, frames=s.recs,
                         load=lambda p: grey, log=lines.append)
    got = {p.key: (p.side, sorted(p.flags), None if p.auto is None else p.auto["cls"]) for p in an.parts}
    assert got == {
        "1": ("enemy", [], "Knight"), "2": ("unknown", ["S1"], None), "3": ("enemy", ["S2"], None),
        "4": ("enemy", ["S3"], None), "5": ("enemy", ["S4"], None), "6": ("enemy", ["S5"], None),
        "7": ("enemy", [], "HogRider"), "8": ("enemy", [], "Knight"), "9": ("enemy", ["S6"], None),
        "10": ("enemy", [], "Knight"), "13": ("enemy", ["S1"], None)}
    by = an.by_key()
    assert by["2"].flags["S1"] == "no live track" and by["13"].flags["S1"] == "the live chain made it ours (HogRider)"
    assert by["3"].flags["S2"] == "Valkyrie unknown" and by["4"].flags["S3"] == "card Giant"
    assert by["5"].flags["S4"] == "Knight 16 / Valkyrie 4" and by["6"].flags["S5"] == "10 images on the water as Knight"
    assert by["9"].flags["S6"].startswith("stopwatch at 0.0 s") and "no live play" in by["9"].flags["S6"]
    assert by["1"].auto == {"cls": "Knight", "form": 0, "type": "unit"} and by["1"].read["cover"] == 1.0
    assert (by["1"].read["live"], by["1"].read["how"], by["1"].read["card"], by["1"].read["conf"]) == (
        "Knight", "play", "Knight", 0.8)
    assert by["1"].cands == [("Knight", 0.8)] and by["5"].live_classes == [("Knight", 0, 16), ("Valkyrie", 0, 4)]
    assert by["1"].crop_n == [0, 20, 14] and by["1"].birth == (1500, 4000) and by["2"].read["cover"] == 0.0
    st = an.stats
    assert (st["ally"], st["short"], st["side_enemy"], st["side_unknown"], st["parts"]) == (1, 1, 10, 1, 11)
    assert (st["auto"], st["flagged"], st["markers"], st["markers_matched"], st["markers_orphan"]) == (4, 7, 3, 1, 1)
    assert [(m["matched"], m["under"]) for m in an.markers] == [(False, [9]), (True, [10]), (False, [])]
    assert an.markers[1]["why"] == "live track 1:10 is a play"
    assert any(line.startswith("  analysis: 11 enemy or unknown-side tracks") for line in lines)
    assert set(an.crops) == set(by) and all(len(c) == 3 and c[0].shape == (72, 72, 3) for c in an.crops.values())
    # a play reported near the stopwatch also matches it; a deck with Giant clears S3
    s.recs[2]["plays"].append({"card": 5, "tick": 40, "x_mt": 1500.0, "y_mt": 9000.0, "kind": "unit", "uids": [],
                               "card_name": "Knight"})
    an2 = UH.analyze_scan(s.scan(tmp_path), deck=DECK10[:-1] + [("Giant", 0)], table=TABLE, geometry=GEOM,
                          frames=s.recs, load=lambda p: grey, log=lines.append)
    assert "S6" not in an2.by_key()["9"].flags and an2.markers[0]["why"] == "play Knight at 0.1 s"
    assert an2.by_key()["4"].flags == {} and an2.by_key()["4"].auto["cls"] == "Giant"


def hand_over(s, h, frames, uid, cls, team, *, how="play", card_name=None):
    """Images `frames` of harvest track h held by another live track (uid, cls, team) instead."""
    lk = f"1:{uid}"
    s.live[lk] = {"first_t": 0.0, "last_t": 1.0, "frames": len(frames), "cls": {cls: len(frames)}, "team": team,
                  "how": how, "card": 1, "card_name": card_name or cls, "forms": {"0": len(frames)}}
    for n in frames:
        p = next(p for p in s.recs[n]["props"] if p["h"] == h)
        p["live"], p["live_cls"] = [1, uid], cls
        s.recs[n]["live"].append([uid, cls, "unit", 0.8, 0.9 if team == E_TEAM else 0.2, 0, *p["box"]])


def test_auto_accept_refuses_form_open_mixed_ours_side_and_our_half(tmp_path):
    s = Scene(40)
    enemy_knight = dict(cls="Knight", team=E_TEAM, how="play", card_name="Knight")
    s.track(1, range(0, 20), 1500, 4000, cands=[("BabyDragon", 1, 0.8)],
            live=dict(cls="BabyDragon", team=E_TEAM, how="play", card_name="BabyDragon"))   # evo slot, base read
    s.track(2, range(0, 20), 4000, 4000, cands=[("BabyDragon_EV1", 1, 0.8)],
            live=dict(cls="BabyDragon", form=1, team=E_TEAM, how="play", card_name="BabyDragon"))   # evo read
    s.track(3, range(0, 30), 6500, 4000, cands=[("Knight", 1, 0.5)], live=enemy_knight)
    hand_over(s, 3, range(0, 5), 103, "HogRider", O_TEAM, how="landing")                   # our Hog first
    s.track(4, range(0, 30), 9000, 4000, cands=[("Knight", 1, 0.5)], live=enemy_knight)
    hand_over(s, 4, range(25, 30), 104, "Valkyrie", E_TEAM)                                # another enemy unit
    s.track(5, range(0, 12), 11500, 4000, cands=[("Knight", 0, 0.8)], live=enemy_knight)  # the side head: ours
    s.track(6, range(0, 12), 14000, 20000, cands=[("Knight", 1, 0.8)], live=enemy_knight)  # born on our half
    s.track(7, range(0, 12), 16500, 4000, cands=[("Knight", 1, 0.8)], live=enemy_knight)   # clean: accepted
    grey = np.full((SIZE[1], SIZE[0], 3), 100, np.uint8)
    an = UH.analyze_scan(s.scan(tmp_path), deck=DECK10, table=TABLE, geometry=GEOM, frames=s.recs,
                         load=lambda p: grey, log=lambda m: None)
    by = an.by_key()
    assert all(by[k].flags == {} for k in by) and sorted(by) == [str(h) for h in range(1, 8)]
    assert {k: (None if p.auto is None else (p.auto["cls"], p.auto["form"])) for k, p in by.items()} == {
        "1": None, "2": ("BabyDragon", 1), "3": None, "4": None, "5": None, "6": None, "7": ("Knight", 0)}
    one = by["1"].read
    assert one["form_open"].startswith("base or evo?") and one["no_auto"] == ["form: " + one["form_open"]]
    assert by["2"].read["form_open"] is None and by["2"].read["no_auto"] == []
    assert by["3"].read["no_auto"] == ["ours: 5 image(s) held by our live track(s)"]
    assert by["3"].read["held"] == [["1:103", O_TEAM, "HogRider", 0, 5, 0, 4], ["1:3", E_TEAM, "Knight", 0, 25, 5, 29]]
    assert by["3"].read["cover"] == pytest.approx(25 / 30, abs=1e-3)
    assert by["4"].read["no_auto"] == ["units: another enemy live track read Valkyrie"]
    assert by["5"].read["no_auto"] == ["side: the detector's side scores are 0% enemy"]
    assert by["5"].read["side_det"] == [0.0, pytest.approx(9.6)]
    assert by["6"].read["no_auto"] == ["half: born on our half"] and by["7"].read["no_auto"] == []
    ctx = UH.DeckContext.build(TABLE, DECK10)
    assert ctx.slot_forms == {"BabyDragon": frozenset({1})}
    assert ctx.form_open("BabyDragon", 1) is None and ctx.form_open("Knight", 0) is None
    assert ctx.form_open("BabyDragon", 0).startswith("base or evo?")


def test_a_cut_is_refined_between_samples_and_each_part_has_its_own_side(tmp_path):
    grass, red, blue = (82, 152, 70), (220, 40, 40), (40, 40, 220)
    s = Scene(60)
    change = 35                                        # images 0-34 our red unit, 35-59 an enemy blue one
    s.track(1, range(0, 60), 3000, 6000, dy=100)
    for j, r in enumerate(s.recs):
        r["epoch"] = "b1e0"
        p = r["props"][0]
        p["cands"] = [["HogRider", 0, 0.5]] if j < change else [["Goblin", 1, 0.9]]
    paint = {j: (unit_box(3000, 6000 + 100 * j), red if j < change else blue) for j in range(60)}
    bg = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    bg[:] = grass

    def load(path):
        img = bg.copy()
        box, colour = paint[int(Path(path).stem)]
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        img[y0:y1, x0:x1] = colour
        return img

    cfg = UH.FlagConfig(sig_max_frames=12)             # samples 0 5 11 16 21 27 32 38 ...: the change is between
    an = UH.analyze_scan(s.scan(tmp_path), deck=DECK10, table=TABLE, geometry=GEOM, frames=s.recs,
                         backgrounds={"b1e0": bg}, load=load, log=lambda m: None, config=cfg)
    assert [(p.key, p.side, p.first_n, p.last_n) for p in an.parts] == [("1.1", "enemy", 35, 59)]
    assert (an.stats["split_tracks"], an.stats["split_cuts_moved"], an.stats["ally_parts"]) == (1, 1, 1)
    assert an.parts[0].flags["S7"] == "part 2 of 2: images 35-59"
    # refine_cut on hand-made signatures: images 33-37 between the samples 32 and 38
    a, b = np.zeros(66), np.zeros(66)
    a[48], b[3] = 1.0, 1.0
    sigs = {i: (a if i < 35 else b) for i in range(33, 38)}
    assert UH.refine_cut(sigs, 32, 38, a, b) == 35
    assert UH.refine_cut({}, 32, 38, a, b) == 38 and UH.refine_cut({}, 5, 6, a, b) == 6
    assert UH.refine_cut({i: b for i in range(33, 38)}, 32, 38, a, b) == 33


def test_track_side_and_the_deck_context():
    own = TABLE.deck_labels(UH.DEFAULT_OUR_DECK)
    live = {"1:1": {"team": E_TEAM}, "1:2": {"team": O_TEAM}}
    assert UH.track_side({"cands": {"Giant@ally": {"sum": 2.0}}, "live": {}}, live, own)[0] == "enemy"
    assert UH.track_side({"cands": {"HogRider@ally": {"sum": 2.0}, "HogRider@enemy": {"sum": 1.0}}}, live, own) == (
        "ally", 1.0, 2.0)
    assert UH.track_side({"cands": {}, "live": {"1:1": 3, "1:2": 2}}, live, own) == ("enemy", 3.0, 2.0)
    assert UH.track_side({"cands": {}, "live": {}}, live, own)[0] == "unknown"
    ctx = UH.DeckContext.build(TABLE, DECK10)
    assert ctx.problem("Knight", 0, "Knight") is None and ctx.problem("BabyDragon", 1, "BabyDragon") is None
    assert ctx.problem("Knight", 1, "Knight") == "the evo form of Knight"            # a base slot
    assert ctx.problem("Giant", 0, None) == "class Giant" and ctx.problem("Knight", 0, "Giant") == "card Giant"
    assert ctx.problem("Zap", 0, "Zap") is None and ctx.problem("Fireball", 0, "Fireball") == "card Fireball"


def test_link_markers_by_time_and_place():
    tile = lambda size: (17.0, 13.8)                    # noqa: E731
    C = UH.Clock
    clocks = [C(0, 0.0, (10, 10, 20, 20), SIZE, 0.6), C(1, 0.05, (11, 10, 21, 20), SIZE, 0.6),
              C(1, 0.05, (100, 10, 110, 20), SIZE, 0.6),           # another place: another event
              C(2, 0.10, (12, 10, 22, 20), SIZE, 0.6), C(30, 1.5, (12, 10, 22, 20), SIZE, 0.6),   # after a gap
              C(31, 1.55, (12, 11, 22, 21), SIZE, 0.6)]
    evs = UH.link_markers(clocks, tile, gap_s=0.5, link_tiles=1.0, min_frames=2)
    assert [[c.n for c in e.clocks] for e in evs] == [[0, 1, 2], [30, 31]]
    assert len(UH.link_markers(clocks, tile, min_frames=1)) == 3


def test_split_on_a_look_change_and_clusters(tmp_path):
    grass, red, blue = (82, 152, 70), (220, 40, 40), (40, 40, 220)
    s = Scene(40)
    paint = {}
    for h, x, colours in ((1, 3000, [red] * 15 + [blue] * 15), (2, 9000, [red] * 30), (3, 15000, [blue] * 30)):
        s.track(h, range(0, 30), x, 6000, src="blob", dy=100)
        for j in range(30):
            paint.setdefault(j, []).append((unit_box(x, 6000 + 100 * j), colours[j]))
    for r in s.recs:
        r["epoch"] = "b1e0"
    bg = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    bg[:] = grass

    def load(path):
        img = bg.copy()
        for box, colour in paint.get(int(Path(path).stem), ()):
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            img[y0:y1, x0:x1] = colour
        return img

    an = UH.analyze_scan(s.scan(tmp_path), deck=DECK10, table=TABLE, geometry=GEOM, frames=s.recs,
                         backgrounds={"b1e0": bg}, load=load, log=lambda m: None)
    assert [(p.key, p.n, p.first_n, p.last_n) for p in an.parts] == [("1.0", 15, 0, 14), ("1.1", 15, 15, 29),
                                                                     ("2", 30, 0, 29), ("3", 30, 0, 29)]
    by = an.by_key()
    assert by["1.0"].flags == {"S1": "no live track", "S7": "part 1 of 2: images 0-14"}
    assert by["1.1"].flags["S7"] == "part 2 of 2: images 15-29" and "S7" not in by["2"].flags
    assert an.stats["split_tracks"] == 1
    red_bin = (3 * 4 + 0) * 4 + 0                        # 220 // 64 = 3: the 64-bin histogram's red corner
    assert by["1.0"].sig[red_bin] == pytest.approx(1.0) and by["1.1"].sig[3] == pytest.approx(1.0)
    assert by["2"].sig[64] == pytest.approx(math.log(14 / 17.0), abs=1e-3)   # log width in tiles
    clusters = UH.cluster_parts(an.parts)
    assert [(c.id, c.keys, c.images) for c in clusters] == [(0, ["2", "1.0"], 45), (1, ["3", "1.1"], 45)]
    # the crops show the square on grass, with the box outlined in yellow
    crop = an.crops["1.1"][0]
    assert (crop == np.array(red, np.uint8)).all(axis=2).sum() == 0 and (crop == np.array(blue, np.uint8)).all(axis=2).sum() > 100
    assert (crop == np.array(UH.BOX_COLOUR, np.uint8)).all(axis=2).sum() > 20
    # the analysis is saved and read back while the scan, deck and settings are the same
    (tmp_path / "scan.json").write_text('{"created": "x"}', encoding="utf-8")
    (tmp_path / "tracks.json").write_text("{}", encoding="utf-8")
    an.identity = UH.analysis_identity(s.scan(tmp_path), deck=DECK10, table=TABLE)
    UH.save_analysis(an, tmp_path / "sheet")
    back = UH.load_analysis(tmp_path / "sheet", UH.analysis_identity(s.scan(tmp_path), deck=DECK10, table=TABLE))
    assert [p.to_dict() for p in back.parts] == [json.loads(json.dumps(p.to_dict())) for p in an.parts]
    assert all(np.array_equal(back.crops[k][i], an.crops[k][i]) for k in an.crops for i in range(3))
    assert back.stats == an.stats and back.markers == an.markers
    assert UH.load_analysis(tmp_path / "sheet", UH.analysis_identity(s.scan(tmp_path), deck=DECK10[:-1],
                                                                       table=TABLE)) is None
    (tmp_path / "tracks.json").write_text('{"scanned": "again"}', encoding="utf-8")       # another scan
    assert UH.load_analysis(tmp_path / "sheet", UH.analysis_identity(s.scan(tmp_path), deck=DECK10,
                                                                       table=TABLE)) is None


def test_find_splits_and_cluster_parts_on_hand_made_signatures():
    def sig(bin_, w=0.0):
        v = np.zeros(66)
        v[bin_], v[64], v[65] = 1.0, w, w
        return v

    a, b, c = sig(48), sig(3), sig(10)
    assert UH.find_splits([a] * 10 + [b] * 10, min_part=3) == [10]
    assert UH.find_splits([a] * 20, min_part=3) == [] and UH.find_splits([a] * 5 + [b], min_part=3) == []
    assert UH.find_splits([a] * 8 + [b] * 8 + [c] * 8, min_part=3) == [8, 16]
    assert UH.find_splits([a] * 8 + [b] * 8 + [c] * 8, min_part=3, max_parts=2) == [8]
    mixed = [a, b] * 10                                  # a flicker, not a change: neither side is tight
    assert UH.find_splits(mixed, min_part=3) == []
    assert UH.sig_distance(a, b) == pytest.approx(1.0) and UH.sig_distance(a, sig(48, math.log(2))) == pytest.approx(
        0.5 * math.log(2))

    def part(h, n, s_):
        return UH.PartInfo(h, 0, 1, 1, "enemy", n, 0, n - 1, 0.0, 1.0, sig=None if s_ is None else list(s_))

    parts = [part(1, 10, a), part(2, 30, a * 0.9 + b * 0.1), part(3, 12, b), part(4, 8, b), part(5, 9, None),
             part(6, 5, a)]
    cl = UH.cluster_parts(parts, max_dist=0.3)
    assert [(c_.id, c_.keys) for c_ in cl] == [(0, ["1", "6", "2"]), (1, ["3", "4"]), (2, ["5"])]
    assert cl[0].dists[0] < cl[0].dists[2] and cl[2].centroid is None
    drift = [part(1, 20, a), part(2, 19, a * 0.62 + b * 0.38), part(3, 18, a * 0.8 + b * 0.2)]
    assert [c_.keys for c_ in UH.cluster_parts(drift, max_dist=0.3)] == [["3", "2", "1"]]    # joined by the merge pass
    assert [c_.keys for c_ in UH.cluster_parts(drift, max_dist=0.25)] == [["2", "3"], ["1"]]
    # two confident reads of different labels never share a cluster; a part without one joins either
    same = [part(1, 10, a), part(2, 9, a), part(3, 8, a)]
    assert [c_.keys for c_ in UH.cluster_parts(same, keys={"1": 5, "2": 6})] == [["1", "3"], ["2"]]
    assert [c_.keys for c_ in UH.cluster_parts(same, keys={"1": 5, "2": 5})] == [["1", "2", "3"]]
    # 3 is nearest 1, but reads another label: it joins 2 (no read), and the merge pass keeps the two apart
    assert [c_.keys for c_ in UH.cluster_parts(drift, max_dist=0.3, keys={"1": 1, "3": 2})] == [["2", "3"], ["1"]]
    assert [c_.keys for c_ in UH.cluster_parts(drift, max_dist=0.3, keys={"1": 1, "2": 1})] == [["3", "2", "1"]]
    assert UH.spread_members(list("abcdefgh"), 4) == ["a", "c", "f", "h"] and UH.spread_members(["a"], 4) == ["a"]
    assert UH.part_key(12) == "12" and UH.part_key(12, 1, 2) == "12.1"


def test_the_contact_sheet_is_drawn(tmp_path):
    img = np.zeros((100, 100, 3), np.uint8)
    img[40:60, 40:60] = (200, 0, 0)
    crop = UH.sheet_crop(img, (40, 40, 60, 60), cell=16, pad_frac=0.3)
    assert crop.shape == (16, 16, 3) and (crop == np.array(UH.BOX_COLOUR, np.uint8)).all(axis=2).sum() > 10
    assert UH.text_mask("AB").shape == (10, 16) and UH.text_mask("ab#", scale=1).shape == (5, 12)
    assert np.array_equal(UH.text_mask("~"), UH.text_mask("?"))            # no such glyph: '?'
    tile = UH.SheetTile("#0 n=2 S1", ["read -", "hv Giant 0.21"], [[crop] * 3] * 2)
    sheet = UH.render_sheet([tile], cell=16, width=200, header=["x"], legend=["0 a", "1 b"])
    # pad 4, line 13 px, tile 64 x (4 + 3 lines + 4 rows of 20): 3 tiles a row; header, legend, then the grid
    assert sheet.shape == (165, 192, 3)
    assert np.array_equal(sheet[81:97, 4:20], crop) and np.array_equal(sheet[101:117, 44:60], crop)
    assert (sheet[121:137, 4:20] == 44).all()                               # a third row: nothing
    assert UH.render_sheet([], cell=16, width=200, header=["x"]).shape == (25, 192, 3)
    out = UH.write_sheet_image(tmp_path / "sheet_1", sheet)
    assert out.exists() and out.suffix in (".png", ".npy") and out.stem == "sheet_1"
