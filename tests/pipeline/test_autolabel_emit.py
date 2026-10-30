"""N11: tools/vision_autolabel.py emit, and its library in scripts/vision/unit_harvest.py.

docs/NEW_CARDS_PLAN.md 2.3 step 5 and N11. Checked:
* the crop round trip: to_yolo / from_yolo against detector.crop_rect_for at the PC's 747x1328,
  clipping, a remnant too small to keep, full-frame rows for gold;
* box smoothing (median sizes, the feet kept), short-gap fill (not over 3 images, not across a
  battle, not over one of our units), spell circles in the frame's geometry;
* the feathered background paste and the drop rules (too many masks, a mask over a kept box,
  no background; a tower is in the background, an HP bar goes with its unit's mask), and gold
  keeping its masks as ignore regions;
* the live layer: our units kept, unvetted units masked, towers on their board half, a live box
  that is a decided part's skipped, an object on an x part kept; leftovers (a detector proposal
  of a track no decided part holds, one of ours the live model missed) masked, gold: ignored;
* emit on the synthetic scan of test_autolabel.py: images in the crop framing, label names and
  YOLO boxes, circles, masks pasted with the background, birth images repeated, emit.json and
  emit_fresh, re-emit only with --force, refused while it would lose images whose raws are gone
  (--drop-missing emits without them);
* gold: nothing is written outside <store>/<sid>/gold/, no image file at all; full-frame labels
  that vision_replay's reader decodes back to gold.jsonl; pending parts refused or ignore
  regions; a training emit folder of the session refuses the gold emit;
* yolo_rows: class indices under the extended label set (thin forms as their base, '*' forms
  as their base, base-set forms as themselves), problems for a label without an index.
Synthetic fixtures only (npy images): no image codec, GPU, engine or emulator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "tools"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vision_autolabel as VA  # noqa: E402
from scripts.vision import labels as L  # noqa: E402
from tests.pipeline.test_autolabel import HOG_CARDS, TABLE, cli, synthetic_scan  # noqa: E402

GRASS, RED, WHITE, MAGENTA = (82, 152, 70), (220, 40, 40), (240, 240, 240), (200, 40, 200)


@pytest.fixture(scope="module")
def keys():
    key_map, note = VA.load_key_map()
    assert note is None, note
    return key_map


def known():
    return VA.known_labels(TABLE, VA.Registry.load(), base=None)       # the preview: the same on every machine


def part_plan(UH, key, decision, label=None, src="owner", typ="unit"):
    return UH.PartPlan(str(key), int(str(key).split(".")[0]), 0, decision, label,
                       1 if decision == UH.DECISION_LABEL else None, src, typ, 0, 10, 0.0, 0.5, 1, None)


def frame_rec(UH, n=5, live=(), size=(360, 640)):
    return UH.FrameRec(n, n, f"images/{n:08d}.npy", round(n * 0.05, 4), 1, 1, "b1e0", size, list(live), {})


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------

def test_crop_round_trip_matches_crop_rect_for():
    pytest.importorskip("numpy")
    from scripts.vision import unit_harvest as UH
    from scripts.vision.detector import crop_rect_for
    from scripts.vision.geometry import ArenaGeometry

    geom = ArenaGeometry.builtin()
    crop = crop_rect_for(geom, None, (747, 1328))
    assert crop == (18, 47, 725, 1050)
    b = (100.0, 200.0, 140.0, 260.0)
    row = UH.to_yolo(b, crop)
    assert row == pytest.approx((0.144272, 0.182453, 0.056577, 0.059821), abs=1e-6)
    assert UH.from_yolo(row, crop) == pytest.approx(b, abs=1e-6)
    # a box half outside the crop is clipped to it; a 2-px remnant is no box
    half = UH.to_yolo((10.0, 100.0, 30.0, 140.0), crop)
    assert UH.from_yolo(half, crop) == pytest.approx((18.0, 100.0, 30.0, 140.0), abs=1e-6)
    assert UH.to_yolo((0.0, 100.0, 20.0, 140.0), crop) is None
    assert UH.to_yolo((0.0, 100.0, 20.0, 140.0), crop, min_px=1.0) is not None
    assert UH.to_yolo((0.0, 0.0, 10.0, 10.0), crop) is None
    # gold: relative to the whole frame
    assert UH.frame_yolo(b, (747, 1328)) == pytest.approx((120 / 747, 230 / 1328, 40 / 747, 60 / 1328))
    # emit frames every image with the same crop; the mask pad is 0.15 of a tile's smaller side
    crop2, pad = UH.emit_framing(geom, None, (747, 1328), UH.EmitConfig())
    tw, th = UH.tile_px(geom, (747, 1328))
    assert crop2 == crop and pad == pytest.approx(0.15 * min(tw, th))


def test_smoothing_fill_and_circles():
    np = pytest.importorskip("numpy")
    from scripts.vision import unit_harvest as UH
    from scripts.vision.geometry import ArenaGeometry

    # widths 10, 10, 30, 10, 10 from the detector: the median sizes the middle one, its feet stay
    seq = [(n, (110.0 - w / 2, 180.0, 110.0 + w / 2, 200.0 + n), "det") for n, w in enumerate((10, 10, 30, 10, 10))]
    sm = dict(UH.smooth_boxes(seq, 5, UH.ANCHOR_BOTTOM))
    x1, y1, x2, y2 = sm[2]
    assert (x2 - x1, (x1 + x2) / 2, y2, y2 - y1) == pytest.approx((10.0, 110.0, 202.0, 22.0))
    # a spell keeps its centre
    c = dict(UH.smooth_boxes(seq, 5, UH.ANCHOR_CENTRE))[2]
    assert ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2) == pytest.approx((110.0, 191.0))
    # blob boxes take the size of the part's detector boxes, else the window's blobs
    mixed = [(0, (0.0, 0.0, 20.0, 10.0), "live")] + [(n, (0.0, 0.0, 40.0, 10.0), "blob") for n in range(1, 5)]
    assert dict(UH.smooth_boxes(mixed, 3))[3] == pytest.approx((10.0, 0.0, 30.0, 10.0))
    blobs = [(0, (0.0, 0.0, 10.0, 10.0), "blob"), (1, (0.0, 0.0, 40.0, 10.0), "blob"), (2, (0.0, 0.0, 10.0, 10.0), "blob")]
    assert dict(UH.smooth_boxes(blobs, 3))[1] == pytest.approx((15.0, 0.0, 25.0, 10.0))

    # two missing images: interpolated by record order (centre x, feet, width, height)
    a, b = (90.0, 180.0, 110.0, 200.0), (100.0, 190.0, 130.0, 230.0)
    got = dict(UH.fill_gaps([(10, a), (13, b)], range(40), 3))
    assert sorted(got) == [11, 12]
    w1, h1 = 20 + 10 / 3, 20 + 20 / 3
    assert got[11] == pytest.approx((105 - w1 / 2, 210 - h1, 105 + w1 / 2, 210.0))
    w2, h2 = 20 + 20 / 3, 20 + 40 / 3
    assert got[12] == pytest.approx((110 - w2 / 2, 220 - h2, 110 + w2 / 2, 220.0))
    assert UH.fill_gaps([(10, a), (15, b)], range(40), 3) == []                       # 4 missing: no fill
    assert UH.fill_gaps([(10, a), (13, b)], [10, 13], 3) == []                        # no record between
    assert UH.fill_gaps([(10, a), (13, b)], range(40), 3, hk_of={n: 1 if n < 12 else 2 for n in range(40)}) == []
    assert UH.fill_gaps([(10, a), (13, b)], range(40), 3, skip=[11]) == []           # one of our units there

    # spell circles: the extremes of world_to_px at x -/+ r and y -/+ r around the centre
    geom, size = ArenaGeometry.builtin(), (747, 1328)
    assert UH.EmitConfig().spell_radius_tiles == (("DarkMagic", 2.5), ("Vines", 2.5), ("GoblinCurse", 3.0))
    centre = tuple(float(v) for v in geom.world_to_px(9000.0, 10000.0, size))
    for r in (2.5, 3.0):
        box = UH.spell_circle(centre, size, geom, r)
        xs, _ = geom.world_to_px(np.array([9000.0 - 1000 * r, 9000.0 + 1000 * r]), np.array([10000.0, 10000.0]), size)
        _, ys = geom.world_to_px(np.array([9000.0, 9000.0]), np.array([10000.0 - 1000 * r, 10000.0 + 1000 * r]), size)
        assert box == pytest.approx((min(xs), min(ys), max(xs), max(ys)), abs=1e-6)
    tw, th = UH.tile_px(geom, size)
    vines = UH.spell_circle(centre, size, geom, 2.5)
    assert (vines[2] - vines[0], vines[3] - vines[1]) == pytest.approx((5 * tw, 5 * th))

    # propagate_parts: the live box when linked, our unit's image left out and not filled over,
    # a gap of one image filled, and a Vines part's circle on every image
    def prop(h, box, live=None, src="blob"):
        return UH.PropRef(h, box, src, "blob", live, [], 9000.0, 10000.0)

    index = {}
    for n in range(8):
        props = {0: prop(7, (100.0, 100.0, 114.0, 114.0)), 1: prop(8, (200.0 + n, 300.0, 214.0 + n, 314.0))}
        live = []
        if n == 1:
            props[1] = prop(8, (201.0, 300.0, 215.0, 314.0), live="1:4", src="det+blob")
            live = [UH.LiveDet(4, "Knight", "unit", 0.8, 0.9, 0, (200.0, 298.0, 216.0, 314.0), 1)]
        if n == 3:
            props[1] = prop(8, (203.0, 300.0, 217.0, 314.0), live="1:9")             # one of ours there
        index[n] = UH.FrameRec(n, n, f"images/{n:08d}.npy", n * 0.05, 1, 1, "b1e0", size, live, props)
    tracks = {7: {"frames": [[0, 0], [1, 0], [3, 0], [4, 0]]}, 8: {"frames": [[n, 1] for n in range(6)]}}
    sheet = {"7": {"h": 7, "first_n": 0, "last_n": 4, "first_t": 0.0, "last_t": 0.2, "hk": 1},
             "8": {"h": 8, "first_n": 0, "last_n": 5, "first_t": 0.0, "last_t": 0.25, "hk": 1, "read": {"uid": "1:4"}}}
    decisions = {"7": (UH.DECISION_LABEL, "Vines", UH.SRC_OWNER), "8": (UH.DECISION_LABEL, "Knight", UH.SRC_OWNER)}
    plans = UH.propagate_parts(decisions, sheet, tracks, index, {"1:4": 0, "1:9": 1}, table=TABLE, geometry=geom)
    v, k = plans["7"], plans["8"]
    assert sorted(v.boxes) == [0, 1, 2, 3, 4] and {how for _b, how in v.boxes.values()} == {"circle"}
    circle = UH.circle_box(geom.px_to_world(107.0, 107.0, size), size, geom, 2.5)
    assert all(b == pytest.approx(circle) for b, _how in v.boxes.values())
    assert (v.type, v.side, k.type, k.uid) == ("spell", 1, "unit", "1:4")
    assert sorted(k.boxes) == [0, 1, 2, 4, 5] and k.skipped == 1 and 3 not in k.props
    assert k.boxes[1][0][3] == 314.0 and k.boxes[1][1] == "smooth"           # the live box's feet
    assert k.birth_xy == (9000.0, 10000.0) and k.to_dict()["boxes"][0][0] == 0


def test_feather_paste_and_drop_rules():
    np = pytest.importorskip("numpy")
    from scripts.vision import unit_harvest as UH

    grass = np.zeros((60, 80, 3), np.uint8)
    grass[:] = GRASS
    img = np.zeros_like(grass)
    img[:] = RED
    out = UH.feather_paste(img, grass, [(30.0, 20.0, 40.0, 30.0)], pad_px=0.0, feather_px=3)
    assert (out[20:30, 30:40] == grass[20:30, 30:40]).all()          # inside: the background exactly
    for x, a in ((26, 0.0), (27, 0.25), (28, 0.5), (29, 0.75)):     # the ramp, 1/4 a pixel
        want = np.rint(np.asarray(RED, float) * (1 - a) + np.asarray(GRASS, float) * a)
        assert (out[25, x] == want).all(), x
    lo, hi = np.minimum(RED, GRASS), np.maximum(RED, GRASS)
    assert ((out[25, 27:30] > lo) & (out[25, 27:30] < hi)).all()      # the ramp lies between
    assert (out[25, :26] == img[25, :26]).all() and (out[:16] == img[:16]).all()
    assert (UH.feather_paste(img, grass, [(30.0, 20.0, 40.0, 30.0)], pad_px=2.0)[25, 28] == GRASS).all()
    with pytest.raises(ValueError):
        UH.feather_paste(img, grass[:10], [])

    have, own = known(), TABLE.deck_labels(HOG_CARDS)
    cfg = UH.EmitConfig(feather_px=0)
    kept_box = (100.0, 100.0, 120.0, 150.0)                       # 1,000 px^2
    lab = part_plan(UH, "1", UH.DECISION_LABEL, "BossBandit")

    def plan(masks, rec=None, **kw):
        at = [(lab, None, kept_box, "smooth")] + [(part_plan(UH, 10 + j, UH.DECISION_MASK), None, m, "smooth")
                                                  for j, m in enumerate(masks)]
        return UH.plan_frame(rec or frame_rec(UH), at, table=TABLE, known=have, own_labels=own, config=cfg,
                             crop=(0, 0, 360, 640), pad_px=0.0, **kw)

    far = [(10.0 + 30 * j, 500.0, 20.0 + 30 * j, 510.0) for j in range(4)]
    assert plan(far[:3]).drop is None and plan(far).drop == UH.DROP_MASKS
    assert plan([(100.0, 100.0, 110.0, 110.0)]).drop == UH.DROP_OVERLAP     # 10% of the kept box
    p4 = plan([(100.0, 100.0, 104.0, 110.0)])                               # 4%
    assert p4.drop is None and p4.masks == [(100.0, 100.0, 104.0, 110.0)]
    assert [(b.label, b.side, b.src) for b in p4.keep] == [("BossBandit", 1, "owner")] and p4.enemy_labels == 1
    assert plan([(10.0, 500.0, 20.0, 510.0)], background=False).drop == UH.DROP_NO_BACKGROUND
    assert plan([], background=False).drop is None
    g = plan(far + [(100.0, 100.0, 110.0, 110.0)], gold=True)
    assert g.drop is None and g.ignore == g.masks == far + [(100.0, 100.0, 110.0, 110.0)]
    assert [b.label for b in g.keep] == ["BossBandit"]
    # the overlap counts the pad and the feather the paste reaches: a mask beside the kept box
    beside = [(lab, None, kept_box, "smooth"),
              (part_plan(UH, 10, UH.DECISION_MASK), None, (90.0, 100.0, 99.0, 150.0), "smooth")]
    for feather, pad, want in ((3, 2.0, UH.DROP_OVERLAP), (0, 0.0, None)):     # 5 px reach: 20% of the kept box
        q = UH.plan_frame(frame_rec(UH), beside, table=TABLE, known=have, own_labels=own,
                          config=UH.EmitConfig(feather_px=feather), crop=(0, 0, 360, 640), pad_px=pad)
        assert q.drop == want, (feather, pad)
    # a tower under a mask is in the background (kept); an HP bar under one goes into it
    bar = UH.LiveDet(None, "bar", "ui", 0.8, 0.9, 0, (100.0, 90.0, 120.0, 96.0), 1)
    tower = UH.LiveDet(None, "PrincessTower", "tower", 0.9, 0.9, 0, (200.0, 60.0, 260.0, 130.0), 1)
    at = [(part_plan(UH, 12, UH.DECISION_MASK), None, (100.0, 92.0, 120.0, 120.0), "smooth"),
          (part_plan(UH, 13, UH.DECISION_PENDING), None, (210.0, 70.0, 240.0, 100.0), "smooth")]
    p = UH.plan_frame(frame_rec(UH, live=[bar, tower]), at, table=TABLE, known=have, own_labels=own, config=cfg,
                      crop=(0, 0, 360, 640), pad_px=0.0)
    assert p.drop is None and [(b.label, b.side) for b in p.keep] == [("PrincessTower", 1)]
    assert (100.0, 90.0, 120.0, 96.0) in p.masks and len(p.masks) == 3 and p.counts["ui_absorbed"] == 1


def test_live_layer_rules():
    pytest.importorskip("numpy")
    from scripts.vision import unit_harvest as UH

    have, own = known(), TABLE.deck_labels(HOG_CARDS)
    LD = UH.LiveDet
    ours = LD(1, "HogRider", "unit", 0.8, 0.1, 0, (50.0, 400.0, 70.0, 430.0), 1)
    gk_ally = LD(2, "GoldenKnight", "unit", 0.8, 0.2, 0, (100.0, 400.0, 120.0, 430.0), 1)
    knight = LD(3, "Knight", "unit", 0.8, 0.9, 0, (150.0, 100.0, 170.0, 130.0), 1)
    tower_top = LD(None, "PrincessTower", "tower", 0.9, 0.5, 0, (40.0, 80.0, 90.0, 140.0), 1)
    tower_low = LD(None, "PrincessTower", "tower", 0.9, 0.5, 0, (40.0, 480.0, 90.0, 540.0), 1)
    linked = LD(5, "GoldenKnight", "unit", 0.7, 0.9, 0, (200.0, 200.0, 220.0, 230.0), 1)
    dirt = LD(6, "dirt", "object", 0.6, 0.9, 0, (250.0, 250.0, 270.0, 270.0), 1)
    on_effect = LD(7, "Knight", "unit", 0.6, 0.9, 0, (249.0, 249.0, 271.0, 271.0), 1)
    musk = LD(8, "Musketeer", "unit", 0.7, 0.8, 0, (300.0, 450.0, 320.0, 480.0), 1)       # its live track is ours
    rec = frame_rec(UH, live=[ours, gk_ally, knight, tower_top, tower_low, linked, dirt, on_effect, musk])
    boss = part_plan(UH, "17", UH.DECISION_LABEL, "BossBandit")
    effect = part_plan(UH, "21", UH.DECISION_EFFECT)
    at = [(boss, UH.PropRef(17, (199.0, 199.0, 221.0, 231.0), "det+blob", "unit", "1:5", [("GoldenKnight", 1, 0.7)]),
           (199.0, 198.0, 221.0, 231.0), "smooth"),
          (effect, UH.PropRef(21, (248.0, 248.0, 272.0, 272.0), "blob", "blob", "1:6"), (248.0, 248.0, 272.0, 272.0),
           "smooth")]
    p = UH.plan_frame(rec, at, table=TABLE, known=have, own_labels=own, live_teams={"1:8": 1}, config=UH.EmitConfig(),
                      crop=(0, 0, 360, 640), pad_px=0.0)
    got = sorted((b.label, b.side, b.src, b.key) for b in p.keep)
    assert got == sorted([("BossBandit", 1, "owner", "17"), ("HogRider", 0, "live", None), ("PrincessTower", 1, "live", None),
                          ("PrincessTower", 0, "live", None), ("dirt", 1, "live", None), ("Musketeer", 0, "live", None)])
    assert next(b for b in p.keep if b.key == "17").box == (199.0, 198.0, 221.0, 231.0)      # the part's box, not the live one
    assert sorted(p.masks) == sorted([gk_ally.box, knight.box])                              # never ours; unvetted
    assert p.counts["live_covered"] == 2 and p.drop is None
    # an o part: its live read when it is one of Yodle2's, else its best candidate, else a mask
    o = part_plan(UH, "30", UH.DECISION_OURS)
    hog = LD(9, "HogRider", "unit", 0.8, 0.9, 0, (10.0, 300.0, 30.0, 330.0), 1)
    misread = LD(10, "GoldenKnight", "unit", 0.8, 0.9, 0, (60.0, 300.0, 80.0, 330.0), 1)
    rec2 = frame_rec(UH, live=[hog, misread])
    at2 = [(o, UH.PropRef(30, hog.box, "det", "unit", "1:9"), hog.box, "smooth"),
           (o, UH.PropRef(31, misread.box, "det", "unit", "1:10", [("GoldenKnight", 1, 0.8), ("IceGolemite", 0, 0.3)]),
            misread.box, "smooth"),
           (o, UH.PropRef(32, (120.0, 300.0, 140.0, 330.0), "blob", "blob", None), (120.0, 300.0, 140.0, 330.0), "smooth")]
    p2 = UH.plan_frame(rec2, at2, table=TABLE, known=have, own_labels=own, config=UH.EmitConfig(), crop=(0, 0, 360, 640))
    assert [(b.label, b.side, b.key) for b in p2.keep] == [("HogRider", 0, "30"), ("IceGolemite", 0, "30")]
    assert p2.masks == [(120.0, 300.0, 140.0, 330.0)] and p2.counts["live_covered"] == 2

    # leftovers: a detector proposal of a track no decided part holds here is masked unless a kept,
    # covered or masked box has it (plan 2.3 step 5: unlabelled leftovers are masked)
    rec3 = frame_rec(UH, live=[hog])
    rec3.props = {
        0: UH.PropRef(40, hog.box, "det", "unit", "1:9", conf=0.6),                          # our live Hog Rider: kept
        1: UH.PropRef(41, (200.0, 300.0, 220.0, 330.0), "det", "unit", None, conf=0.5),      # ours, missed by the live model
        2: UH.PropRef(42, (250.0, 300.0, 270.0, 330.0), "det+blob", "unit", None, conf=0.1),  # too weak
        3: UH.PropRef(43, (300.0, 300.0, 320.0, 330.0), "blob", "blob", None),              # a bare blob
        4: UH.PropRef(17, (199.0, 100.0, 221.0, 131.0), "det", "unit", None, conf=0.9),      # the decided part's own
        5: UH.PropRef(44, (201.0, 101.0, 215.0, 125.0), "det", "unit", None, conf=0.4)}      # inside the part's box
    at3 = [(part_plan(UH, "17", UH.DECISION_LABEL, "BossBandit"), rec3.props[4], (199.0, 100.0, 221.0, 131.0), "smooth")]
    kw3 = dict(table=TABLE, known=have, own_labels=own, live_teams={"1:9": 1}, crop=(0, 0, 360, 640))
    p3 = UH.plan_frame(rec3, at3, config=UH.EmitConfig(), **kw3)
    assert p3.masks == [(200.0, 300.0, 220.0, 330.0)] and p3.counts["leftover_masks"] == 1 and p3.drop is None
    assert sorted((b.label, b.key) for b in p3.keep) == [("BossBandit", "17"), ("HogRider", None)]
    assert UH.plan_frame(rec3, at3, config=UH.EmitConfig(mask_leftovers=False), **kw3).masks == []
    assert UH.plan_frame(rec3, at3, config=UH.EmitConfig(), gold=True, **kw3).ignore == [(200.0, 300.0, 220.0, 330.0)]
    # load_frame_index keeps such proposals of tracks no part decides, and only those
    recs = [{"n": 5, "i": 5, "image": "images/00000005.npy", "t": 0.25, "hk": 1, "seg": 1, "size": [360, 640],
             "live": [], "props": [{"box": [1, 2, 3, 4], "src": "det", "type": "unit", "conf": 0.5, "h": 41},
                                   {"box": [1, 2, 3, 4], "src": "blob", "type": "blob", "conf": 0.0, "h": 43},
                                   {"box": [1, 2, 3, 4], "src": "det", "type": "unit", "conf": 0.1, "h": 42},
                                   {"box": [1, 2, 3, 4], "src": "live", "type": "unit", "conf": 0.9, "h": 17}]}]
    assert sorted(p.h for p in UH.load_frame_index(recs, wanted={17})[5].props.values()) == [17]
    assert sorted(p.h for p in UH.load_frame_index(recs, wanted={17}, leftover_conf=0.15)[5].props.values()) == [17, 41]
    # a flyer's live-only proposal is placed as LowConfPass places its harvest boxes
    fly = UH.flying_classes(TABLE)
    assert fly("Minion") and not fly("HogRider") and not fly(None) and not fly("no-such-thing")


# ---------------------------------------------------------------------------
# emit on the synthetic scan
# ---------------------------------------------------------------------------

def label_synthetic(tmp_path, keys, sid):
    """synthetic_scan, then the labels of test_sheet_and_label_on_a_synthetic_scan: h1 GiantBuffer
    (auto-accepted), h2 and h3 goblinstein, h4 Vines, 5.0 x, 5.1 ?."""
    paths, geom = synthetic_scan(tmp_path, keys, sid)
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE)
    VA.make_sheet(sid, geometry=geom, log=lambda m: None, **kw)
    VA.label_session(sid, "0=s,1=goblinstein,2=x", sheet=1, **kw)
    VA.make_sheet(sid, geometry=geom, log=lambda m: None, **kw)
    VA.label_session(sid, "0=Vines,1=?", sheet=2, **kw)
    return paths, geom


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_emit_a_training_session_on_the_synthetic_scan(tmp_path, keys, capsys):
    np = pytest.importorskip("numpy")
    from scripts.vision import unit_harvest as UH

    sid = "20261101_A_1"
    paths, geom = label_synthetic(tmp_path, keys, sid)
    kwl = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE)
    kw = dict(kwl, geometry=geom, log=lambda m: None)
    cfg = UH.EmitConfig(birth_s=0.3, image_format="npy")
    dry = VA.emit_session(sid, config=cfg, dry_run=True, **kw)
    assert not dry.written and not paths.emit_dir.exists() and not paths.gold_dir.exists()
    assert dry.doc["parts"] == {"label": 4, "mask": 1, "effect": 1, "ours": 0, "pending": 0}
    assert dry.doc["labels"] == {"GiantBuffer": 20, "Vines": 20, "goblinstein": 24}
    assert cli(tmp_path, "emit", sid, "--dry-run") == VA.EXIT_OK
    out = capsys.readouterr().out
    assert "6 part(s): 4 label, 1 mask, 1 effect" in out and "dry run: nothing written" in out
    assert not paths.emit_dir.exists()

    res = VA.emit_session(sid, config=cfg, **kw)
    assert res.written and res.kind == VA.KIND_TRAIN and res.out_dir == paths.emit_dir
    assert not paths.gold_dir.exists()
    rows = jsonl(paths.emit_dir / VA.BOXES_FILE)
    by_n = {r["n"]: r for r in rows}
    births = set(range(0, 7)) | set(range(20, 27))         # 0.3 s after h1, h2, h4 (t 0) and h3 (t 1.0)
    assert set(by_n) == births | {9, 12, 15, 18, 27, 30}   # 1 in 3 of the other labelled images; none of 32-39
    assert {n for n, r in by_n.items() if r["birth"]} == births
    assert all(r["repeat"] == (2 if r["birth"] else 1) for r in rows)
    crop = tuple(rows[0]["crop"])
    assert crop == UH.emit_framing(geom, None, (360, 640))[0] == (9, 22, 349, 506)
    images = paths.emit_dir / VA.IMAGES_DIR
    assert sorted(f.name for f in images.iterdir()) == sorted(r["image"] for r in rows)
    for r in rows:
        assert r["image"] == f"{sid}_{r['i']:08d}.npy" and r["frame_size"] == [360, 640] and list(r["crop"]) == list(crop)
        assert np.load(images / r["image"]).shape == (crop[3] - crop[1], crop[2] - crop[0], 3)
        assert all(len(b) == 8 and b[1] in (0, 1) for b in r["boxes"])
    parts = {p["key"]: p for p in jsonl(paths.emit_dir / VA.PARTS_FILE)}
    assert sorted(parts) == ["1", "2", "3", "4", "5.0", "5.1"]

    # h1: GiantBuffer on the enemy side, auto-accepted, in its proposal's place
    h1 = parts["1"]
    assert (h1["decision"], h1["label"], h1["src"], h1["card"], h1["uid"]) == (
        "label", "GiantBuffer", "auto", "GiantBuffer", "1:1")
    for n in sorted(set(by_n) & set(range(20))):
        gb = [b for b in by_n[n]["boxes"] if b[7] == "1"]
        assert len(gb) == 1 and gb[0][:2] == ["GiantBuffer", 1] and gb[0][6] == "auto"
        box = next(b[1:5] for b in h1["boxes"] if b[0] == n)
        assert UH.from_yolo(gb[0][2:6], crop) == pytest.approx(box, abs=0.06)
    # h4: Vines is one circle (2.5 tiles) on each of its images, centred where its blobs were
    vines = parts["4"]
    assert (vines["label"], vines["type"], vines["card"]) == ("Vines", "spell", "Vines")
    assert [b[0] for b in vines["boxes"]] == list(range(20)) and {b[5] for b in vines["boxes"]} == {"circle"}
    assert len({tuple(b[1:5]) for b in vines["boxes"]}) == 1
    x1, y1, x2, y2 = vines["boxes"][0][1:5]
    centres = [((p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2)
               for r in UH.iter_scan_frames(paths.scan_dir) for p in r.get("props") or () if p["h"] == 4]
    assert ((x1 + x2) / 2, (y1 + y2) / 2) == pytest.approx((np.median([c[0] for c in centres]),
                                                           np.median([c[1] for c in centres])), abs=0.06)
    tw, th = UH.tile_px(geom, (360, 640))
    assert (x2 - x1, y2 - y1) == pytest.approx((5 * tw, 5 * th), abs=0.15)
    for n in sorted(set(by_n) & set(range(20))):
        vb = [b for b in by_n[n]["boxes"] if b[7] == "4"]
        assert len(vb) == 1 and vb[0][:2] == ["Vines", 1]
        assert UH.from_yolo(vb[0][2:6], crop) == pytest.approx((x1, y1, x2, y2), abs=0.06)
    # 5.0 (x) has no box and no mask; 5.1 (?) is the background inside its mask
    assert parts["5.0"]["decision"] == "effect" and parts["5.1"]["decision"] == "mask"
    assert not any(b[7] == "5.0" for r in rows for b in r["boxes"])
    bg = np.load(paths.scan_dir / "background" / "b1e0.npy")[crop[1]:crop[3], crop[0]:crop[2]]
    checked = 0
    for b in parts["5.1"]["boxes"]:
        r = by_n.get(b[0])
        if r is None:
            continue
        img = np.load(images / r["image"])
        x0, y0 = int(np.floor(b[1])) - crop[0], int(np.floor(b[2])) - crop[1]
        x1_, y1_ = int(np.ceil(b[3])) - crop[0], int(np.ceil(b[4])) - crop[1]
        assert (img[y0:y1_, x0:x1_] == bg[y0:y1_, x0:x1_]).all()
        assert len(r["masks"]) == 1 and not any(bb[7] == "5.1" for bb in r["boxes"])
        checked += 1
    assert checked == 7                                   # images 12, 15, 18, 20-23
    first = np.load(images / by_n[0]["image"])
    for key, colour in (("1", MAGENTA), ("5.0", WHITE)):            # left as they were
        b = next(bb for bb in parts[key]["boxes"] if bb[0] == 0)
        cx, cy = int((b[1] + b[3]) / 2) - crop[0], int((b[2] + b[4]) / 2) - crop[1]
        assert tuple(first[cy, cx]) == colour, key
    assert by_n[0]["masks"] == []

    # emit.json, and what makes it stale
    doc = json.loads((paths.emit_dir / VA.EMIT_JSON).read_text(encoding="utf-8"))
    assert doc == json.loads(json.dumps(res.doc))
    assert (doc["schema"], doc["kind"], doc["role"], doc["sheet"]) == (VA.EMIT_SCHEMA, "train", "train", 2)
    assert doc["tracks_sha1"] == VA.sha1_file(paths.tracks_file) and doc["scan"] == VA.scan_identity(paths.scan_dir)
    assert doc["parts"] == dry.doc["parts"] and doc["bad_tracks"] == [] and doc["image_format"] == "npy"
    c = doc["counts"]
    assert (c["records"], c["battle_images"], c["considered"], c["emitted"], c["birth"]) == (40, 40, 20, 20, 14)
    assert sum(c["dropped"].values()) == 0 and c["masks"] == 7
    assert c["boxes"]["auto"] == 11 and c["boxes"]["circle"] == 11 and c["boxes"]["live"] == 0
    assert doc["crops"] == {"360x640": list(crop)} and doc["battle_seconds"] == {"1": 1.95}
    assert c["leftover_masks"] == 0                       # h1 is the scan's only detector track, and it is labelled
    base, real = VA.base_label_set(TABLE)                 # which base the pseudo-labels were checked against
    assert (doc["base_digest"], doc["base_source"]) == (base.digest, "real" if real else "preview")
    session = VA.load_session(sid, tmp_path / "labels")
    assert VA.emit_fresh(paths, session) == []
    st = {r["sid"]: r for r in VA.status(tmp_path / "labels", tmp_path / "store")["sessions"]}
    assert "emit" in st[sid]["stages"] and not any("emit stale" in n for n in st[sid]["notes"])

    # a second emit needs --force; a relabel makes the emit stale until then
    with pytest.raises(VA.EmitError, match="pass --force") as err:
        VA.emit_session(sid, config=cfg, **kw)
    assert err.value.code == VA.EXIT_CHECK
    assert cli(tmp_path, "emit", sid, "--format", "npy") == VA.EXIT_CHECK
    assert "pass --force" in capsys.readouterr().out
    VA.label_session(sid, "t2=GiantBuffer", force=True, **kwl)
    assert VA.TRACKS_CHANGED in VA.emit_fresh(paths, session)
    st = {r["sid"]: r for r in VA.status(tmp_path / "labels", tmp_path / "store")["sessions"]}
    assert any("emit stale: tracks.json changed since emit" in n for n in st[sid]["notes"])
    stray = images / "keep_me.npy"
    stray.write_bytes(b"not emit's")
    assert cli(tmp_path, "emit", sid, "--format", "npy", "--force", "--birth-s", "0.3") == VA.EXIT_OK
    assert "next: & $BB tools\\vision_autolabel.py review" in capsys.readouterr().out
    assert VA.emit_fresh(paths, session) == [] and stray.exists()                # --force deletes only emit's own
    assert {p["key"]: p["label"] for p in jsonl(paths.emit_dir / VA.PARTS_FILE)}["2"] == "GiantBuffer"
    # raws deleted after emit (D14), then a relabel: emit --force would lose the images it cannot redo
    VA.label_session(sid, "t3=GiantBuffer", force=True, **kwl)
    (paths.images / "00000003.npy").unlink()                  # a birth image of the emit
    emitted_before = sorted(f.name for f in images.iterdir())
    with pytest.raises(VA.EmitError, match="no raw image any more") as err:
        VA.emit_session(sid, config=cfg, force=True, **kw)
    assert err.value.code == VA.EXIT_CHECK and sorted(f.name for f in images.iterdir()) == emitted_before
    assert VA.TRACKS_CHANGED in VA.emit_fresh(paths, session)                      # nothing was changed
    again = VA.emit_session(sid, config=cfg, force=True, drop_missing=True, **kw)
    assert again.doc["counts"]["dropped"]["no_image"] == 1 and f"{sid}_00000003.npy" not in {
        f.name for f in images.iterdir()} and any("--drop-missing" in w for w in again.warnings)
    # a training emit never writes a gold image, and a gold session never gets a training one
    with pytest.raises(VA.EmitError, match="gold session"):
        VA._assert_trainable(dict(session, role=VA.ROLE_GOLD))
    VA._assert_trainable(session)


def test_gold_is_never_written_to_train_dirs(tmp_path, keys):
    pytest.importorskip("numpy")
    import vision_replay as VR

    sid = "20261101_A_2"                                   # friendly 2: gold
    paths, geom = synthetic_scan(tmp_path, keys, sid)
    session = VA.load_session(sid, tmp_path / "labels")
    assert VA.is_gold(session) and VA.out_dir_for(paths, session) == paths.gold_dir
    kwl = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", table=TABLE)
    kw = dict(kwl, geometry=geom, log=lambda m: None)
    VA.make_sheet(sid, geometry=geom, log=lambda m: None, **kwl)
    VA.label_session(sid, "t1=GiantBuffer,t2=goblinstein,t3=goblinstein,t4=Vines,t5.0=x", **kwl)   # t5.1 pending
    before = {p for p in paths.store_dir.rglob("*") if p.is_file()}

    with pytest.raises(VA.EmitError, match="not labelled yet") as err:
        VA.emit_session(sid, **kw)
    assert err.value.code == VA.EXIT_CHECK and not paths.gold_dir.exists() and not paths.emit_dir.exists()
    pend = VA.emit_session(sid, allow_pending=True, **kw)
    assert pend.kind == VA.KIND_GOLD and pend.doc["parts"]["pending"] == 1 and pend.doc["pending"] == ["5.1"]
    gold = {g["n"]: g for g in jsonl(paths.gold_dir / VA.GOLD_FILE)}
    parts = {p["key"]: p for p in jsonl(paths.gold_dir / VA.PARTS_FILE)}
    for b in parts["5.1"]["boxes"]:                        # a pending part is an ignore region of each image
        assert any(g[0] - 0.1 <= b[1] and g[1] - 0.1 <= b[2] and b[3] <= g[2] + 0.1 and b[4] <= g[3] + 0.1
                   for g in gold[b[0]]["ignore"])
        assert not any(bb["key"] == "5.1" for bb in gold[b[0]]["boxes"])

    VA.label_session(sid, "t5.1=?", **kwl)
    with pytest.raises(VA.EmitError, match="pass --force"):
        VA.emit_session(sid, **kw)
    res = VA.emit_session(sid, force=True, **kw)
    assert res.out_dir == paths.gold_dir and res.doc["image_format"] is None and res.doc["parts"]["pending"] == 0
    new = {p for p in paths.store_dir.rglob("*") if p.is_file()} - before
    assert new and all(paths.gold_dir in p.parents for p in new)                    # nothing outside gold/
    assert not paths.emit_dir.exists()
    assert not any(p.suffix.lower() in (".jpg", ".jpeg", ".png", ".npy") for p in new)    # and no image at all
    names = {p.relative_to(paths.gold_dir).as_posix() for p in new}
    assert {VA.EMIT_JSON, VA.PARTS_FILE, VA.GOLD_FILE, "labels/detector_labels.json"} <= names
    assert all(n in (VA.EMIT_JSON, VA.PARTS_FILE, VA.GOLD_FILE, "labels/detector_labels.json")
               or (n.startswith("labels/") and n.endswith(".txt")) for n in names)
    stems = {p.stem for p in (paths.gold_dir / "labels").glob("*.txt")}
    assert stems == {f"{n:08d}" for n in range(40)}                            # one per battle image
    gold = jsonl(paths.gold_dir / VA.GOLD_FILE)
    assert [g["stem"] for g in gold] == [f"{n:08d}" for n in range(40)] and gold[0]["image"] == "images/00000000.npy"

    # the labels decode, through the names file, to gold.jsonl's boxes over the frame size
    names_file = VR.find_label_names(paths.gold_dir / "labels")
    assert names_file == paths.gold_dir / "labels" / "detector_labels.json"
    yolo_names = VR.read_label_names(names_file)
    names_doc = json.loads(names_file.read_text(encoding="utf-8"))
    assert yolo_names == dict(enumerate(names_doc["yolo_names"]))
    assert names_doc["labels"] == ["GiantBuffer", "goblinstein", "Vines"] and res.doc["gold_names_digest"] == names_doc["digest"]
    for g in gold:
        rows = [line.split() for line in (paths.gold_dir / "labels" / f"{g['stem']}.txt").read_text().splitlines()]
        assert len(rows) == len(g["boxes"])
        w, h = g["size"]
        for row, b in zip(rows, g["boxes"]):
            assert L.split_side(yolo_names[int(row[0])]) == (b["label"], b["side"])
            x1, y1, x2, y2 = b["box"]
            want = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
            assert [float(v) for v in row[1:]] == pytest.approx(want, abs=5e-4)
        assert {b["src"] for b in g["boxes"]} <= {"owner", "live"}
    assert any(b["label"] == "Vines" and b["how"] == "circle" for b in gold[0]["boxes"])
    assert any(gold[n]["ignore"] for n in range(12, 24)) and not any(gold[n]["ignore"] for n in range(0, 12))
    index = VR.LabelIndex(paths.gold_dir / "labels", yolo_names, TABLE)
    assert len(index) == 40 and len(index.truth_for("images/00000000.npy")) == len(gold[0]["boxes"])
    assert VA.emit_fresh(paths, session) == []
    assert res.doc["battle_seconds"] == {"1": 1.95}

    # gold never reaches a training write, and a training emit folder of the session refuses gold
    with pytest.raises(VA.EmitError, match="never written as training images"):
        VA._assert_trainable(session)
    paths.emit_dir.mkdir()
    with pytest.raises(VA.EmitError, match="holds a training emit") as err:
        VA.emit_session(sid, force=True, **kw)
    assert err.value.code == VA.EXIT_CHECK
    paths.emit_dir.rmdir()
    st = {r["sid"]: r for r in VA.status(tmp_path / "labels", tmp_path / "store")["sessions"]}
    assert "emit-gold" in st[sid]["stages"] and "emit" not in st[sid]["stages"]
    assert any("gold plays not derived yet" in n for n in st[sid]["notes"])


# ---------------------------------------------------------------------------
# names to class indices (shared with N13's mix)
# ---------------------------------------------------------------------------

ENTRIES = [{"label": "Berserker", "card": "Berserker", "type": "unit", "form": "base"},
           {"label": "BerserkerHero", "card": "Berserker", "type": "unit", "form": "hero", "base": "Berserker"},
           {"label": "BossBandit", "card": "BossBandit", "type": "unit", "form": "base"}]


def registry_for(base, entries=ENTRIES):
    return VA.Registry.from_dict({"schema": VA.REGISTRY_SCHEMA, "base": {"digest": base.digest}, "labels": list(entries)})


def test_yolo_rows_indices_against_the_label_set_digest():
    base, _ = L.build_label_set(TABLE, ["hog-rider", "knight", "minion"])
    reg = registry_for(base)
    ext = reg.extend(base, require_pin=True)
    assert ext.labels == ["HogRider", "Knight", "Minion", "Berserker", "BerserkerHero", "BossBandit"]
    digest = ext.digest
    boxes = [["BossBandit", 1, 0.5, 0.5, 0.1, 0.1, "owner", "17"], ["BerserkerHero", 0, 0.2, 0.3, 0.05, 0.06, "owner", "3"],
             ["Minion", 1, 0.1, 0.1, 0.02, 0.02, "live", None], ["MinionHorde_EV1", 1, 0.3, 0.3, 0.02, 0.02, "owner", "9"]]
    lines, problems = VA.yolo_rows(boxes, ext, reg, {"BerserkerHero": 3}, TABLE)
    assert problems == []
    idx = [int(line.split()[0]) for line in lines]
    assert idx == [2 * ext.index("BossBandit") + 1, 2 * ext.index("Berserker"), 2 * ext.index("Minion") + 1,
                   2 * ext.index("Minion") + 1] == [11, 6, 5, 5]
    assert lines[0] == "11 0.500000 0.500000 0.100000 0.100000"
    assert [ext.decode(i) for i in idx] == [("BossBandit", 1), ("Berserker", 0), ("Minion", 1), ("Minion", 1)]
    lines, _ = VA.yolo_rows(boxes[1:2], ext, reg, {"BerserkerHero": 8}, TABLE)       # enough tracks: its own class
    assert int(lines[0].split()[0]) == 2 * ext.index("BerserkerHero") == 8
    lines, problems = VA.yolo_rows([["GoldenKnight", 1, .5, .5, .1, .1], ["Knight", 2, .5, .5, .1, .1]], ext, reg, {}, TABLE)
    assert lines == [] and problems[0] == "no index: GoldenKnight" and problems[1].startswith(VA.YOLO_SIDE + ":")
    assert ext.digest == digest                                                      # nothing appended by the calls
    # trains_as: a form tonight's set has trains as itself (Registry.train_label alone would give its base)
    base5, _ = L.build_label_set(TABLE, ["hog-rider", "knight", "knight-evolution", "minion"])
    reg5 = registry_for(base5)
    ext5 = reg5.extend(base5, require_pin=True)
    assert VA.trains_as("Knight_EV1", ext5, reg5, 0, TABLE) == "Knight_EV1"
    assert reg5.train_label("Knight_EV1", 0, TABLE) == "Knight"
    assert VA.trains_as("KnightHero", ext5, reg5, 0, TABLE) == "Knight"             # a '*' form: its base
    assert VA.trains_as("BerserkerHero", ext5, reg5, 7, TABLE) == "Berserker"
    assert VA.trains_as("BerserkerHero", ext5, reg5, 8, TABLE) == "BerserkerHero"
    # a label the session's client may not teach
    late = [dict(ENTRIES[2], valid_from_client="16.700.0")]
    reg_c = registry_for(base, late)
    ext_c = reg_c.extend(base, require_pin=True)
    lines, problems = VA.yolo_rows(boxes[:1], ext_c, reg_c, {}, TABLE, client="16.635.1")
    assert lines == [] and problems[0].startswith(VA.YOLO_CLIENT + ":")
    assert VA.yolo_rows(boxes[:1], ext_c, reg_c, {}, TABLE, client="16.700.0")[0] == [
        f"{2 * ext_c.index('BossBandit') + 1} 0.500000 0.500000 0.100000 0.100000"]
    # an unpinned registry writes no index (mix extends with require_pin)
    with pytest.raises(VA.RegistryError, match="not pinned"):
        VA.Registry.from_dict({"schema": VA.REGISTRY_SCHEMA, "labels": ENTRIES}).extend(base, require_pin=True)


def test_effective_bad_tracks_and_decisions():
    tracks = {"tracks": [{"key": "17", "label": "BossBandit", "at": "2026-11-01T21:00:00", "source": "owner"},
                         {"key": "18", "code": "x", "at": "2026-11-01T21:00:00"},
                         {"key": "19", "code": "o", "at": "2026-11-01T21:00:00"},
                         {"key": "20", "code": "?", "at": "2026-11-01T21:00:00"},
                         {"key": "21", "label": "Zap", "at": "2026-11-01T22:30:00", "source": "owner"}],
              "split": ["22.0", "22.1"]}
    marks = {"schema": VA.MARKS_SCHEMA, "bad_tracks": {"17": {"label_at": "2026-11-01T21:00:00"},
                                                       "21": {"label_at": "2026-11-01T21:00:00"}}}
    assert VA.effective_bad_tracks(marks, tracks) == {"17"}          # 21 was labelled again since its mark
    assert VA.effective_bad_tracks(None, tracks) == set()
    pytest.importorskip("numpy")
    from scripts.vision import unit_harvest as UH

    sheet = {k: {} for k in ("17", "18", "19", "20", "21", "22.0", "22.1", "23")}
    got = UH.decide_parts(sheet, tracks, VA.effective_bad_tracks(marks, tracks))
    assert got == {"17": ("mask", None, "bad"), "18": ("effect", None, "owner"), "19": ("ours", None, "owner"),
                   "20": ("mask", None, "owner"), "21": ("label", "Zap", "owner"), "22.0": ("pending", None, "pending"),
                   "22.1": ("pending", None, "pending"), "23": ("pending", None, "pending")}
    assert list(got) == ["17", "18", "19", "20", "21", "22.0", "22.1", "23"]
