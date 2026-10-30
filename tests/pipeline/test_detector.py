"""V09: the detector wrapper, its label set and the dataset / training scripts.

scripts/vision/detector.py runs on a FAKE model here: the fake returns boxes in
the coordinates of the crop it was given, and the tests check what the
detector makes of them (frame pixels, class x side through the class list,
side merging, the board filter, ground points, card-level identity). No
ultralytics, torch, GPU or image file is needed; the geometry is the built-in
1080x1920 fit (scripts/vision/geometry.py), where world millitiles map to
pixels as px = 78 + 0.051 x, py = 151.2 + 0.0414 y, and the default crop is
(27, 68, 1047, 1518).

A tiny classes.json document (tiny_doc) stands in for data/vision/classes.json
in the mechanics tests; the shipped file is used for the Hog 2.6 crosswalk.
scripts/prepare_yolo_dataset.py runs on a three-frame synthetic KataCR dataset
in tmp_path, scripts/train_detector.py with a fake YOLO and a fake torch.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import detector as D  # noqa: E402
from scripts.vision import labels as L  # noqa: E402
from scripts.vision.geometry import ArenaGeometry, FrameSizeError  # noqa: E402


def _load_script(name: str):
    """scripts/<name>.py as the module `name` (they are run as files, not imported)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# a tiny class table
# ---------------------------------------------------------------------------

TINY_KATACR = ["king-tower", "bar", "skeleton", "skeleton-evolution", "hog-rider", "baby-dragon",
               "witch", "fireball", "dirt", "golem", "pad_0"]
#: The label set TINY_KATACR becomes (pad_0 left out, golem kept although unmapped).
TINY_LABELS = ["KingTower", "bar", "Skeleton", "Skeleton_EV1", "HogRider", "BabyDragon", "Witch",
               "fireball", "dirt", "golem"]


def _form(unit, flying=False, building=False, deploy=1000, hp=100):
    return {"unit": unit, "flying": flying, "building": building, "deploy_time_ms": deploy,
            "hitpoints": hp}


def _cls(i, name, forms, *, kind="troop", cards=(), katacr=None):
    return {"id": i, "name": name, "kind": kind, "primary_card": cards[0][0] if cards else None,
            "cards": [list(c) for c in cards], "forms": forms, "katacr": katacr or {}}


def tiny_doc() -> dict:
    classes = [
        _cls(0, "KingTower", {"0": _form("KingTower", building=True, deploy=0, hp=2400)},
             kind="tower", katacr={"0": "king-tower"}),
        _cls(1, "Skeleton", {"0": _form("Skeleton"), "1": _form("Skeleton_EV1")},
             cards=[("Skeletons", "summon"), ("Witch", "spawn")],
             katacr={"0": "skeleton", "1": "skeleton-evolution"}),
        _cls(2, "HogRider", {"0": _form("HogRider")}, cards=[("HogRider", "summon")],
             katacr={"0": "hog-rider"}),
        _cls(3, "BabyDragon", {"0": _form("BabyDragon", flying=True)},
             cards=[("BabyDragon", "summon")], katacr={"0": "baby-dragon"}),
        _cls(4, "Witch", {"0": _form("Witch")}, cards=[("Witch", "summon")], katacr={"0": "witch"}),
    ]
    units = {"KingTower": [0, 0], "Skeleton": [1, 0], "Skeleton_EV1": [1, 1], "HogRider": [2, 0],
             "BabyDragon": [3, 0], "Witch": [4, 0]}
    katacr = {
        "king-tower": {"type": "tower", "class": "KingTower", "form": 0},
        "bar": {"type": "ui"},
        "skeleton": {"type": "unit", "class": "Skeleton", "form": 0},
        "skeleton-evolution": {"type": "unit", "class": "Skeleton", "form": 1},
        "hog-rider": {"type": "unit", "class": "HogRider", "form": 0},
        "baby-dragon": {"type": "unit", "class": "BabyDragon", "form": 0},
        "witch": {"type": "unit", "class": "Witch", "form": 0},
        "fireball": {"type": "spell", "card": "Fireball", "form": 0},
        "dirt": {"type": "object", "of": []},
        "golem": {"type": "unmapped"},
        "pad_0": {"type": "pad"},
    }
    cards = {
        "Fireball": {"kind": "spell", "elixir": 4, "flying": False, "forms": [0], "classes": [],
                     "katacr": ["fireball"]},
        "HogRider": {"kind": "troop", "elixir": 4, "flying": False, "forms": [0],
                     "classes": ["HogRider"], "katacr": []},
        "Skeletons": {"kind": "troop", "elixir": 1, "flying": False, "forms": [0, 1],
                      "classes": ["Skeleton"], "katacr": []},
        "Witch": {"kind": "troop", "elixir": 5, "flying": False, "forms": [0],
                  "classes": ["Skeleton", "Witch"], "katacr": []},
        "BabyDragon": {"kind": "troop", "elixir": 4, "flying": True, "forms": [0],
                       "classes": ["BabyDragon"], "katacr": []},
    }
    return {"schema": 1, "katacr_names": list(TINY_KATACR), "classes": classes, "units": units,
            "cards": cards, "katacr": katacr}


TABLE = L.load_class_table(tiny_doc())
GEOM = ArenaGeometry.builtin()
CROP_1080 = (27, 68, 1047, 1518)


def tiny_labels(joint: bool = True) -> L.LabelSet:
    labels, _ = L.build_label_set(TABLE, TINY_KATACR, joint_side=joint)
    return labels


def joint_names() -> dict[int, str]:
    return dict(enumerate(tiny_labels().yolo_names()))


# ---------------------------------------------------------------------------
# a fake ultralytics model
# ---------------------------------------------------------------------------

class FakeBoxes:
    def __init__(self, rows):
        a = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = a[:, :4], a[:, 4], a[:, 5]


class FakeModel:
    """names + predict(): returns `rows` (x1, y1, x2, y2, conf, class) in CROP pixels."""

    def __init__(self, names, rows=(), error: Exception | None = None):
        self.names = dict(names)
        self.rows = list(rows)
        self.error = error
        self.calls: list[tuple[np.ndarray, dict]] = []

    def predict(self, source=None, **kwargs):
        self.calls.append((source, kwargs))
        if self.error is not None:
            raise self.error
        return [SimpleNamespace(boxes=FakeBoxes(self.rows))]


def crop_row(frame_box, conf, cls, origin=CROP_1080[:2]):
    x0, y0 = origin
    x1, y1, x2, y2 = frame_box
    return (x1 - x0, y1 - y0, x2 - x0, y2 - y0, conf, cls)


def detector(model, **cfg) -> D.Detector:
    return D.Detector(model, geometry=GEOM, table=TABLE, config=D.DetectorConfig(**cfg))


def frame(w=1080, h=1920) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# Frame boxes (x1, y1, x2, y2) of the scene, with the world point each stands on.
BOX_KING = (477.0, 237.5, 597.0, 337.5)          # bottom-centre (537, 337.5) = world (9000, 4500)
BOX_HOG = (517.0, 505.2, 557.0, 565.2)           # (537, 565.2) = (9000, 10000)
BOX_DRAGON = (670.0, 588.0, 710.0, 648.0)        # (690, 648) = (12000, 12000), flying: +600 mt
BOX_SKEL = (272.0, 1166.2, 292.0, 1186.2)        # (282, 1186.2) = (4000, 25000)
BOX_BAR = (522.0, 495.2, 552.0, 501.2)           # centre (537, 498.2) = (9000, 8381.64)
BOX_WITCH = (772.0, 919.2, 812.0, 979.2)         # (792, 979.2) = (14000, 20000)
BOX_FIREBALL = (517.0, 462.4, 557.0, 502.4)      # centre (537, 482.4) = (9000, 8000)
BOX_OFFBOARD = (127.0, 68.0, 167.0, 89.1)        # bottom (147, 89.1) = (1352.9, -1500): dropped
BOX_GOLEM = (313.0, 379.6, 353.0, 419.6)         # centre (333, 399.6) = (5000, 6000)


def scene_rows():
    """The scene in a deliberately unsorted order (yolo index = label index * 2 + side)."""
    n = {name: i for i, name in joint_names().items()}
    return [
        crop_row(BOX_WITCH, 0.3, n["Witch@ally"]),           # merged into the enemy Witch
        crop_row(BOX_HOG, 0.8, n["HogRider@enemy"]),
        crop_row(BOX_OFFBOARD, 0.5, n["HogRider@ally"]),     # off the board
        crop_row(BOX_SKEL, 0.4, n["Skeleton@ally"]),         # same sprite as the evo skeleton
        crop_row(BOX_KING, 0.95, n["KingTower@enemy"]),
        crop_row(BOX_FIREBALL, 0.55, n["fireball@enemy"]),
        crop_row(BOX_GOLEM, 0.45, n["golem@enemy"]),         # not in the class list
        crop_row(BOX_SKEL, 0.7, n["Skeleton_EV1@ally"]),
        crop_row(BOX_DRAGON, 0.75, n["BabyDragon@enemy"]),
        crop_row(BOX_BAR, 0.65, n["bar@enemy"]),
        crop_row(BOX_WITCH, 0.6, n["Witch@enemy"]),
    ]


# ---------------------------------------------------------------------------
# labels: sides and the label set
# ---------------------------------------------------------------------------

def test_side_suffix_round_trip():
    assert L.with_side("HogRider", L.ENEMY_SIDE) == "HogRider@enemy"
    assert L.with_side("Skeleton_EV1", L.ALLY_SIDE) == "Skeleton_EV1@ally"
    assert L.with_side("bar", None) == "bar"
    assert L.split_side("HogRider@enemy") == ("HogRider", 1)
    assert L.split_side("the-log@ally") == ("the-log", 0)
    assert L.split_side("hog-rider") == ("hog-rider", None)
    assert L.split_side("odd@name") == ("odd@name", None)
    with pytest.raises(ValueError):
        L.with_side("HogRider", 2)
    assert [L.parse_side(v) for v in ("0", "1", "1.0", "2", "-1", None, "x")] == [0, 1, 1, None, None, None, None]


def test_label_set_follows_the_crosswalk_and_keeps_unmapped_names():
    labels, mapping = L.build_label_set(TABLE, TINY_KATACR)
    assert labels.labels == TINY_LABELS
    assert mapping["pad_0"] is None and mapping["skeleton-evolution"] == "Skeleton_EV1"
    assert mapping["fireball"] == "fireball" and mapping["golem"] == "golem"
    assert labels.status["HogRider"] == "mapped" and labels.status["bar"] == "kept"
    assert labels.unmapped == ["golem"]
    assert labels.sources["Skeleton_EV1"] == ["skeleton-evolution"]
    names = labels.yolo_names()
    assert len(names) == 20
    assert names[:4] == ["KingTower@ally", "KingTower@enemy", "bar@ally", "bar@enemy"]
    assert names[9] == "HogRider@enemy" and names[19] == "golem@enemy"
    assert labels.yolo_index("Skeleton_EV1", 0) == 6
    assert labels.decode(9) == ("HogRider", 1)
    with pytest.raises(ValueError):
        labels.yolo_index("HogRider", None)
    again = L.LabelSet.from_dict(json.loads(json.dumps(labels.to_dict())))
    assert again.yolo_names() == names and again.digest == labels.digest


def test_label_set_without_side_and_in_katacr_mode():
    flat = tiny_labels(joint=False)
    assert flat.yolo_names() == TINY_LABELS and flat.yolo_index("HogRider") == 4
    assert flat.decode(4) == ("HogRider", None)
    kat, mapping = L.build_label_set(TABLE, TINY_KATACR, mode=L.MODE_KATACR)
    assert kat.labels == TINY_KATACR                     # pads too: the dataset's own ids
    assert all(mapping[n] == n for n in TINY_KATACR)
    with pytest.raises(ValueError):
        L.build_label_set(TABLE, TINY_KATACR, mode="other")


def test_extra_labels_are_appended_or_refused():
    labels, _ = L.build_label_set(TABLE, TINY_KATACR, extra=["Skeleton_EV1", "Fireball"])
    assert labels.labels == TINY_LABELS + ["Fireball"]   # Skeleton_EV1 was already there
    assert labels.status["Fireball"] == "extra"
    with pytest.raises(ValueError, match="Nope"):
        L.build_label_set(TABLE, TINY_KATACR, extra=["Nope"])


def test_resolve_label_reads_every_naming():
    assert L.resolve_label(TABLE, "Skeleton_EV1")[:3] == ("unit", 1, 1)
    assert L.resolve_label(TABLE, "skeleton-evolution")[:3] == ("unit", 1, 1)
    assert L.resolve_label(TABLE, "KingTower").type == "tower"
    assert L.resolve_label(TABLE, "fireball").card == "Fireball"
    assert L.resolve_label(TABLE, "bar").type == "ui"
    unknown = L.resolve_label(TABLE, "golem")
    assert unknown.type == "unknown" and unknown.name == "golem"
    assert L.resolve_label(TABLE, "class_250").type == "unknown"


def test_converter_maps_ids_keeps_side_and_reports():
    labels, mapping = L.build_label_set(TABLE, TINY_KATACR)
    conv = L.LabelConverter(labels, dict(enumerate(TINY_KATACR)), mapping)
    assert conv.convert(4, "1", 0.3) == 9                 # hog-rider, enemy
    assert conv.convert(3, "0", 0.8) == 6                 # skeleton-evolution, ally
    assert conv.convert(9, "1", 0.4) == 19                # golem: kept, reported
    assert conv.convert(10, "0", 0.5) is None             # pad_0: dropped, reported
    assert conv.convert(7, None, 0.2) == 15               # no side column: top half -> enemy
    assert conv.convert(7, "x", 0.9) == 14                # bad side column: bottom half -> ally
    assert conv.convert(42, "1", 0.5) == 21               # an id the yaml lacks: class_42, appended
    assert labels.labels[-1] == "class_42" and labels.status["class_42"] == "unmapped"
    r = conv.report.to_dict()
    assert r["boxes"] == 7 and r["written"] == 6
    assert r["pad_boxes"] == {"pad_0": 1} and r["unmapped"] == {"golem": 1, "class_42": 1}
    assert r["unknown_ids"] == {"42": 1} and r["side_guessed"] == 2
    assert r["sides"] == {"ally": 2, "enemy": 4}


#: Hog 2.6 KataCR classes -> (label, class, form), through the shipped classes.json.
HOG26_KATACR = {
    "hog-rider": ("HogRider", "HogRider", 0), "cannon": ("Cannon", "Cannon", 0),
    "ice-golem": ("IceGolemite", "IceGolemite", 0), "ice-spirit": ("IceSpirits", "IceSpirits", 0),
    "ice-spirit-evolution": ("IceSpirits_EV1", "IceSpirits", 1),
    "musketeer": ("Musketeer", "Musketeer", 0), "skeleton": ("Skeleton", "Skeleton", 0),
    "skeleton-evolution": ("Skeleton_EV1", "Skeleton", 1),
}


def test_shipped_class_list_crosswalk_round_trip_for_hog26():
    table = L.load_class_table()
    names = list(table.doc["katacr_names"])
    labels, mapping = L.build_label_set(table, names)
    assert labels.unmapped == []
    assert len(labels) == len([n for n in names if not n.startswith("pad_")])
    for kat, (label, cls, form) in HOG26_KATACR.items():
        assert mapping[kat] == label
        for side in (L.ALLY_SIDE, L.ENEMY_SIDE):
            base, got_side = L.split_side(L.with_side(label, side))
            lab = L.resolve_label(table, base)
            assert (base, got_side, lab.type, lab.name, lab.form) == (label, side, "unit", cls, form)
    for kat, card in (("fireball", "Fireball"), ("the-log", "Log")):
        assert mapping[kat] == kat
        lab = L.resolve_label(table, kat)
        assert (lab.type, lab.card) == ("spell", card)
    # Every label of the set reads back as what it is: nothing turns unknown on the way.
    assert all(L.resolve_label(table, x).type != "unknown" for x in labels.labels)
    assert set(labels.yolo_names()) == {L.with_side(x, s) for x in labels.labels for s in (0, 1)}


# ---------------------------------------------------------------------------
# the detector on a fake model
# ---------------------------------------------------------------------------

def test_detector_crops_the_arena_and_feeds_bgr():
    model = FakeModel(joint_names())
    det = detector(model)
    img = frame()
    img[100, 200] = (10, 20, 30)                          # RGB at frame (x 200, y 100)
    res = det.detect(img)
    assert res.crop == CROP_1080 and res.frame_size == (1080, 1920)
    source, kwargs = model.calls[0]
    assert source.shape == (1450, 1020, 3) and source.dtype == np.uint8
    assert tuple(source[100 - 68, 200 - 27]) == (30, 20, 10)   # BGR, offset by the crop
    assert source.flags["C_CONTIGUOUS"]
    assert kwargs == {"imgsz": 640, "conf": 0.35, "iou": 0.6, "max_det": 150, "device": "cpu",
                      "half": False, "verbose": False}
    assert res.detections == [] and res.raw_boxes == 0


def test_detector_maps_filters_and_places_the_scene():
    model = FakeModel(joint_names(), scene_rows())
    det = detector(model)
    res = det.detect(frame())
    got = [(d.cls, d.form) for d in res.detections]
    assert got == [("KingTower", 0), ("HogRider", 0), ("BabyDragon", 0), ("Skeleton", 1), ("bar", 0),
                   ("Witch", 0), ("Fireball", 0), ("golem", 0)]
    assert [lab.type for lab in res.labels] == ["tower", "unit", "unit", "unit", "ui", "unit",
                                                 "spell", "unknown"]
    assert [d.conf for d in res.detections] == pytest.approx([0.95, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.45])
    assert [d.side_prob for d in res.detections] == pytest.approx(
        [0.9, 0.9, 0.9, 0.1, 0.9, 0.6 / 0.9, 0.9, 0.9])
    # boxes are back in frame pixels
    for d, box in zip(res.detections, (BOX_KING, BOX_HOG, BOX_DRAGON, BOX_SKEL, BOX_BAR, BOX_WITCH,
                                       BOX_FIREBALL, BOX_GOLEM)):
        assert d.box == pytest.approx(box, abs=1e-9)
    # ground points: box bottom-centre for units and towers (the dragon 600 mt lower,
    # it flies), box centre for UI, spells and unknown names
    expect_world = [(9000, 4500), (9000, 10000), (12000, 12600), (4000, 25000),
                    (9000, 347 / 0.0414), (14000, 20000), (9000, 8000), (5000, 6000)]
    expect_px = [(537, 337.5), (537, 565.2), (690, 672.84), (282, 1186.2), (537, 498.2),
                 (792, 979.2), (537, 482.4), (333, 399.6)]
    for (wx, wy), (gx, gy), w, d in zip(expect_world, expect_px, res.world, res.detections):
        assert w == pytest.approx((wx, wy), abs=1e-6)
        assert d.ground_px == pytest.approx((gx, gy), abs=1e-6)
    # the off-board unit is dropped, the duplicate skeleton suppressed, the Witch pair merged
    assert [(d.cls, d.conf) for d in res.dropped] == [("HogRider", 0.5)]
    assert res.suppressed == 1 and res.raw_boxes == 11
    # groups
    assert [d.cls for d in res.units()] == ["HogRider", "BabyDragon", "Skeleton", "Witch"]
    assert [d.cls for d in res.towers()] == ["KingTower"]
    assert [d.cls for d in res.spells()] == ["Fireball"]
    assert [d.cls for d in res.bars()] == ["bar"]
    assert [d.cls for d in res.unknown()] == ["golem"]
    assert res.objects() == []
    assert [d.cls for d in res.for_tracker()] == ["HogRider", "BabyDragon", "Skeleton", "Witch", "Fireball"]
    assert [d.cls for d in res.for_tracker(spells=False, unknown=True)] == [
        "HogRider", "BabyDragon", "Skeleton", "Witch", "golem"]
    # unknown names are reported, not dropped
    assert det.unresolved == Counter({"golem": 1})
    assert det.unresolved_names == ["golem"]
    assert det.side_mode == "joint"


def test_card_level_identity_for_live_compat_rows():
    res = detector(FakeModel(joint_names(), scene_rows())).detect(frame())
    assert res.cards(TABLE) == [None, "HogRider", "BabyDragon", "Skeletons", None, "Witch",
                                "Fireball", None]
    # with a Witch and no Skeletons in the deck, a skeleton is the Witch's
    assert res.cards(TABLE, deck=["Witch", "HogRider", "Fireball"]) == [
        None, "HogRider", "BabyDragon", "Witch", None, "Witch", "Fireball", None]


def test_cross_class_suppression_can_be_turned_off():
    res = detector(FakeModel(joint_names(), scene_rows()), cross_class_iou=None).detect(frame())
    assert res.suppressed == 0
    assert [(d.cls, d.form) for d in res.units()] == [("HogRider", 0), ("BabyDragon", 0), ("Skeleton", 1),
                                                      ("Witch", 0), ("Skeleton", 0)]


def test_one_sided_boxes_use_the_configured_prior():
    n = {name: i for i, name in joint_names().items()}
    rows = [crop_row(BOX_HOG, 0.8, n["HogRider@enemy"]), crop_row(BOX_WITCH, 0.6, n["Witch@ally"])]
    res = detector(FakeModel(joint_names(), rows), single_side_prob=0.75).detect(frame())
    assert [d.side_prob for d in res.detections] == pytest.approx([0.75, 0.25])


def test_side_pairs_merge_only_when_they_overlap():
    n = {name: i for i, name in joint_names().items()}
    far = (BOX_WITCH[0] + 100, BOX_WITCH[1], BOX_WITCH[2] + 100, BOX_WITCH[3])   # same label, apart
    rows = [crop_row(BOX_WITCH, 0.6, n["Witch@enemy"]), crop_row(far, 0.3, n["Witch@ally"])]
    res = detector(FakeModel(joint_names(), rows), cross_class_iou=None).detect(frame())
    assert [(d.cls, d.side_prob) for d in res.detections] == [("Witch", pytest.approx(0.9)),
                                                              ("Witch", pytest.approx(0.1))]


def test_legacy_katacr_model_names_have_no_side():
    names = {0: "hog-rider", 1: "skeleton-evolution", 2: "bar", 3: "pad_0"}
    rows = [crop_row(BOX_HOG, 0.8, 0), crop_row(BOX_SKEL, 0.7, 1)]
    det = detector(FakeModel(names, rows))
    res = det.detect(frame())
    assert det.side_mode == "none" and det.unresolved_names == []
    assert [(d.cls, d.form, d.side_prob) for d in res.detections] == [("HogRider", 0, 0.5),
                                                                      ("Skeleton", 1, 0.5)]
    assert det.name_map()[3]["type"] == "pad"


def test_a_class_index_the_model_does_not_name_is_reported():
    det = detector(FakeModel({0: "HogRider@ally"}, [crop_row(BOX_HOG, 0.8, 7)]))
    res = det.detect(frame())
    assert [d.cls for d in res.unknown()] == ["class_7"]
    assert det.unresolved == Counter({"class_7": 1})


def test_detector_works_at_the_pc_capture_size():
    # 747x1328 (MuMu's window): the same geometry, scaled. HogRider at world (9000, 10000)
    # is at (371.425, 390.93) px; the crop starts at (18, 47).
    n = {name: i for i, name in joint_names().items()}
    box = (361.425, 350.93, 381.425, 390.93)
    model = FakeModel(joint_names(), [crop_row(box, 0.8, n["HogRider@enemy"], origin=(18, 47))])
    res = detector(model).detect(frame(747, 1328))
    assert res.crop == (18, 47, 725, 1050) and res.frame_size == (747, 1328)
    assert model.calls[0][0].shape == (1003, 707, 3)
    assert [d.cls for d in res.detections] == ["HogRider"]
    assert res.detections[0].box == pytest.approx(box, abs=1e-9)
    assert res.world[0] == pytest.approx((9000, 10000), abs=1e-6)
    assert res.detections[0].ground_px == pytest.approx((371.425, 390.93), abs=1e-6)


def test_crop_rect_for_is_the_detectors_crop_without_a_model():
    """N05: detector.crop_rect_for(geometry, config, size), what the label emitters use, is
    Detector.crop_rect: at the PC's 747x1328 and the other sizes, for each kind of crop."""
    assert D.crop_rect_for(GEOM, None, (747, 1328)) == (18, 47, 725, 1050)
    assert D.crop_rect_for(GEOM, D.DetectorConfig(), (1080, 1920)) == CROP_1080
    for cfg in (D.DetectorConfig(), D.DetectorConfig(crop_norm=(0.0, 0.05, 1.0, 0.8)),
                D.DetectorConfig(crop_margin_mt=(0, 0, 0, 0)), D.DetectorConfig(conf=0.05, imgsz=960)):
        det = D.Detector(FakeModel(joint_names()), geometry=GEOM, table=TABLE, config=cfg)
        for size in ((747, 1328), (1080, 1920), (225, 400)):
            assert D.crop_rect_for(GEOM, cfg, size) == det.crop_rect(size), (cfg, size)
    with pytest.raises(FrameSizeError):
        D.crop_rect_for(GEOM, None, (1080, 2400))


def test_detector_refuses_frames_that_are_not_9_16():
    det = detector(FakeModel(joint_names()))
    with pytest.raises(FrameSizeError):
        det.detect(frame(1080, 2400))
    with pytest.raises(D.DetectorError):
        det.detect(np.zeros((1920, 1080), dtype=np.uint8))


def test_crop_from_normalised_box_and_config_checks():
    det = detector(FakeModel(joint_names()), crop_norm=(0.0, 0.05, 1.0, 0.8))
    assert det.crop_rect((1080, 1920)) == (0, 96, 1080, 1536)
    cfg = D.DetectorConfig.from_dict({"conf": 0.5, "crop_margin_mt": [0, 0, 0, 0]})
    assert cfg.conf == 0.5 and cfg.crop_margin_mt == (0.0, 0.0, 0.0, 0.0)
    assert detector(FakeModel(joint_names()), crop_margin_mt=(0, 0, 0, 0)).crop_rect((1080, 1920)) == (
        78, 151, 996, 1476)
    with pytest.raises(ValueError):
        D.DetectorConfig.from_dict({"confidence": 0.5})
    with pytest.raises(ValueError):
        D.DetectorConfig(crop_norm=(0.5, 0.0, 0.4, 1.0))
    with pytest.raises(ValueError):
        D.DetectorConfig(single_side_prob=0.3)


def test_predict_failure_is_a_detector_error():
    det = detector(FakeModel(joint_names(), error=RuntimeError("CUDA out of memory")))
    with pytest.raises(D.DetectorError, match="CUDA out of memory"):
        det.detect(frame())


def test_box_iou():
    assert D.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert D.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert D.box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_bar_classes_match_the_hp_bar_reader():
    from scripts.vision import hpbars
    assert tuple(hpbars.BAR_CLASSES) == D.BAR_CLASSES


# ---------------------------------------------------------------------------
# weights, SHA-256, pin file, device
# ---------------------------------------------------------------------------

def fake_torch(cuda: bool, name="NVIDIA GeForce RTX 4070 SUPER", total=12282 * 1024 ** 2):
    return SimpleNamespace(
        __version__="2.14.1+cu130",
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda i: name,
                             get_device_properties=lambda i: SimpleNamespace(total_memory=total)))


def test_pick_device_follows_torch_cuda():
    dev, desc, vram = D.pick_device(None, torch_module=fake_torch(True))
    assert dev == "0" and "4070 SUPER" in desc and vram == pytest.approx(11.994, abs=1e-3)
    assert D.pick_device("auto", torch_module=fake_torch(False))[0] == "cpu"
    assert D.pick_device("cpu", torch_module=fake_torch(True))[0] == "cpu"
    assert D.pick_device("1", torch_module=fake_torch(True))[0] == "1"


def test_load_hashes_logs_and_checks_the_weights(tmp_path, caplog):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights v1")
    digest = hashlib.sha256(b"weights v1").hexdigest()
    made = []

    def factory(path):
        made.append(path)
        return FakeModel(joint_names())

    caplog.set_level(logging.INFO, logger="hastycr.vision.detector")
    det = D.Detector.load(weights, model_factory=factory, torch_module=fake_torch(True),
                          geometry=GEOM, table=TABLE)
    assert made == [str(weights)]
    assert det.sha256 == digest and det.device == "0" and det.half is True
    info = det.info()
    assert info["sha256"] == digest and info["weights"] == str(weights)
    assert info["model_classes"] == 20 and info["side_mode"] == "joint"
    assert digest in caplog.text and "best.pt" in caplog.text
    with pytest.raises(D.DetectorError, match="SHA-256"):
        D.Detector.load(weights, expected_sha256="0" * 64, model_factory=factory, device="cpu",
                        geometry=GEOM, table=TABLE)
    with pytest.raises(D.DetectorError, match="do not exist"):
        D.Detector.load(tmp_path / "missing.pt", model_factory=factory, device="cpu",
                        geometry=GEOM, table=TABLE)


def test_pin_file_round_trip_and_refusal(tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"abc")
    pin = D.write_pin(weights, path=tmp_path / "pin.json", meta={"epochs": 3})
    doc = D.read_pin(pin)
    assert doc["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert doc["meta"] == {"epochs": 3} and doc["schema"] == D.PIN_SCHEMA
    det = D.Detector.load(pin=pin, model_factory=lambda p: FakeModel(joint_names()), device="cpu",
                          geometry=GEOM, table=TABLE)
    assert det.weights.resolve() == weights.resolve() and det.sha256 == doc["sha256"]
    weights.write_bytes(b"abd")                            # retrained behind the pin's back
    with pytest.raises(D.DetectorError, match="SHA-256"):
        D.Detector.load(pin=pin, model_factory=lambda p: FakeModel(joint_names()), device="cpu",
                        geometry=GEOM, table=TABLE)
    with pytest.raises(D.DetectorError, match="train_detector"):
        D.Detector.load(pin=tmp_path / "none.json", model_factory=lambda p: FakeModel({}),
                        device="cpu", geometry=GEOM, table=TABLE)


def test_default_pin_lives_under_the_machine_data_root():
    from hastycr import machine
    assert D.default_pin_path() == Path(machine.get().data) / "vision" / "detector_pin.json"


# ---------------------------------------------------------------------------
# scripts/prepare_yolo_dataset.py
# ---------------------------------------------------------------------------

def make_dataset(root: Path) -> Path:
    """A three-frame KataCR-style dataset under root/images/part2 (labels: 12 columns)."""
    part2 = root / "images" / "part2"
    part2.mkdir(parents=True)
    lines = ["path: x", "names:"] + [f"  {i}: {n}" for i, n in enumerate(TINY_KATACR)]
    (part2 / "ClashRoyale_detection.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (part2 / "train_annotation.txt").write_text("./ep1/0/00000.jpg\n./ep1/0/00001.jpg\n", encoding="utf-8")
    (part2 / "val_annotation.txt").write_text("./ep2/0/00000.jpg\n", encoding="utf-8")
    labels = {
        "ep1/0/00000": ["4 0.500000 0.300000 0.100000 0.050000 1 0 0 0 0 0 0",    # hog-rider, top
                        "3 0.200000 0.800000 0.050000 0.040000 0 0 0 0 0 0 0",    # skeleton-evolution
                        "9 0.400000 0.400000 0.100000 0.100000 1 0 0 0 0 0 0",    # golem (unmapped)
                        "10 0.5 0.5 0.1 0.1 0 0 0 0 0 0 0",                       # pad_0
                        "7 0.6 0.2 0.1 0.1"],                                     # fireball, no side
        "ep1/0/00001": ["0 0.5 0.1 0.2 0.15 1 0 0 0 0 0 0",                       # king-tower, top
                        "4 1.2 0.5 0.1 0.1 0"],                                   # out of range
        "ep2/0/00000": ["1 0.5 0.5 0.05 0.01 0 0 0 0 0 0 0"],                     # bar, bottom
    }
    for stem, rows in labels.items():
        image = part2 / f"{stem}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\xff\xd8 not decoded")
        image.with_suffix(".txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return part2


def _classes_file(tmp_path: Path) -> Path:
    path = tmp_path / "classes.json"
    path.write_text(json.dumps(tiny_doc()), encoding="utf-8")
    return path


def test_prepare_keeps_the_side_and_maps_names(tmp_path):
    prep = _load_script("prepare_yolo_dataset")
    make_dataset(tmp_path / "ds")
    out = tmp_path / "yolo"
    assert prep.main(["--source", str(tmp_path / "ds"), "--out", str(out),
                      "--classes", str(_classes_file(tmp_path))]) == 0
    train = out / "labels" / "train"
    assert (train / "ep1_0_00000.txt").read_text(encoding="utf-8") == (
        "9 0.500000 0.300000 0.100000 0.050000\n"          # HogRider@enemy = 4 * 2 + 1
        "6 0.200000 0.800000 0.050000 0.040000\n"          # Skeleton_EV1@ally = 3 * 2 + 0
        "19 0.400000 0.400000 0.100000 0.100000\n"         # golem@enemy: kept, reported
        "15 0.6 0.2 0.1 0.1\n")                            # fireball, side from the top half
    assert (train / "ep1_0_00001.txt").read_text(encoding="utf-8") == "1 0.5 0.1 0.2 0.15\n"
    assert (out / "labels" / "val" / "ep2_0_00000.txt").read_text(encoding="utf-8") == "2 0.5 0.5 0.05 0.01\n"
    assert (out / "images" / "train" / "ep1_0_00000.jpg").read_bytes() == b"\xff\xd8 not decoded"
    yaml_lines = (out / "cr.yaml").read_text(encoding="utf-8").splitlines()
    assert yaml_lines[1:4] == ["train: images/train", "val: images/val", "names:"]
    assert yaml_lines[4:] == [f"  {i}: {n}" for i, n in enumerate(tiny_labels().yolo_names())]
    doc = json.loads((out / "detector_labels.json").read_text(encoding="utf-8"))
    assert doc["labels"] == TINY_LABELS and doc["joint_side"] is True
    assert doc["status"]["golem"] == "unmapped"
    assert doc["report"]["unmapped"] == {"golem": 1} and doc["report"]["pad_boxes"] == {"pad_0": 1}
    assert doc["report"]["side_guessed"] == 1 and doc["report"]["sides"] == {"ally": 2, "enemy": 4}
    assert doc["dataset"]["splits"] == {"train": {"frames": 2, "boxes": 5}, "val": {"frames": 1, "boxes": 1}}
    sides = json.loads((out / "sides.json").read_text(encoding="utf-8"))
    assert sides == {"ep1_0_00000": [1, 0, 1, -1], "ep1_0_00001": [1], "ep2_0_00000": [0]}


def test_prepare_can_still_write_the_original_output(tmp_path):
    prep = _load_script("prepare_yolo_dataset")
    part2 = make_dataset(tmp_path / "ds")
    out = tmp_path / "yolo"
    assert prep.main(["--source", str(part2), "--out", str(out), "--labels", "katacr", "--side", "drop"]) == 0
    assert (out / "labels" / "train" / "ep1_0_00000.txt").read_text(encoding="utf-8") == (
        "4 0.500000 0.300000 0.100000 0.050000\n3 0.200000 0.800000 0.050000 0.040000\n"
        "9 0.400000 0.400000 0.100000 0.100000\n10 0.5 0.5 0.1 0.1\n7 0.6 0.2 0.1 0.1\n")
    yaml_lines = (out / "cr.yaml").read_text(encoding="utf-8").splitlines()
    assert yaml_lines[4:] == [f"  {i}: {n}" for i, n in enumerate(TINY_KATACR)]
    assert not (out / "detector_labels.json").exists()
    assert json.loads((out / "sides.json").read_text(encoding="utf-8"))["ep1_0_00000"] == [1, 0, 1, 0, -1]


def test_prepare_finds_the_dataset_under_the_data_root(tmp_path, monkeypatch):
    prep = _load_script("prepare_yolo_dataset")
    data = tmp_path / "data"
    monkeypatch.setattr(prep, "_machine_data_root", lambda: data)
    assert prep.resolve_source(None) == prep.SOURCE                 # nothing there: vendor default
    part2 = make_dataset(data / "CR-Detection-Dataset")
    assert prep.resolve_source(None) == part2
    assert prep.resolve_source(tmp_path / "elsewhere") == tmp_path / "elsewhere" / "images" / "part2"
    assert prep.main(["--source", str(tmp_path / "elsewhere"), "--out", str(tmp_path / "o")]) == 1


# ---------------------------------------------------------------------------
# scripts/train_detector.py
# ---------------------------------------------------------------------------

def test_default_batch_for_a_12gb_card():
    td = _load_script("train_detector")
    assert td.model_size("yolov8s.pt") == "s" and td.model_size("yolo11m.pt") == "m"
    assert td.model_size("runs/best.pt") is None
    assert td.default_batch("yolov8s.pt", 640, 12282 / 1024) == 32     # a 4070 SUPER reports 11.99
    assert td.default_batch("yolov8n.pt", 640, 12) == 64
    assert td.default_batch("yolo11m.pt", 640, 12) == 16
    assert td.default_batch("yolov8x.pt", 640, 12) == 4
    assert td.default_batch("yolov8s.pt", 960, 12) == 8                # 32 * (640/960)^2 = 14.2
    assert td.default_batch("yolov8s.pt", 640, 8) == 16                 # 32 * 8/12 = 21.3
    assert td.default_batch("best.pt", 640, None) == 32


def _fake_yolo_class():
    class FakeYOLO:
        instances: list = []

        def __init__(self, model):
            self.model, self.train_kwargs, self.trainer = model, None, None
            FakeYOLO.instances.append(self)

        def train(self, **kw):
            self.train_kwargs = kw
            wdir = Path(kw["project"]) / kw["name"] / "weights"
            wdir.mkdir(parents=True)
            (wdir / "best.pt").write_bytes(b"best")
            (wdir / "last.pt").write_bytes(b"last")
            self.trainer = SimpleNamespace(best=str(wdir / "best.pt"))
    return FakeYOLO


def test_train_uses_cuda_hashes_and_pins_best(tmp_path, monkeypatch):
    td = _load_script("train_detector")
    yolo = _fake_yolo_class()
    monkeypatch.setattr(td, "_import_torch", lambda: fake_torch(True))
    monkeypatch.setattr(td, "_import_yolo", lambda: yolo)
    data = tmp_path / "cr.yaml"
    data.write_text("names:\n  0: a@ally\n  1: a@enemy\n", encoding="utf-8")
    (tmp_path / "detector_labels.json").write_text(json.dumps({"digest": "abc"}), encoding="utf-8")
    pin = tmp_path / "pin.json"
    rc = td.main(["--data", str(data), "--project", str(tmp_path / "runs"), "--name", "t",
                  "--epochs", "3", "--pin-file", str(pin)])
    assert rc == 0
    kw = yolo.instances[-1].train_kwargs
    assert yolo.instances[-1].model == "yolov8s.pt"
    assert (kw["device"], kw["batch"], kw["epochs"], kw["imgsz"], kw["workers"]) == ("0", 32, 3, 640, 4)
    assert kw["data"] == str(data) and kw["project"] == str(tmp_path / "runs") and kw["name"] == "t"
    assert kw["flipud"] == 0.0 and kw["cache"] is False
    best = tmp_path / "runs" / "t" / "weights" / "best.pt"
    digest = hashlib.sha256(b"best").hexdigest()
    assert best.with_name("best.pt.sha256").read_text(encoding="utf-8") == f"{digest} *best.pt\n"
    manifest = json.loads((tmp_path / "runs" / "t" / "detector_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == digest and manifest["last_sha256"] == hashlib.sha256(b"last").hexdigest()
    assert manifest["label_set_digest"] == "abc" and manifest["batch"] == 32
    doc = D.read_pin(pin)
    assert doc["sha256"] == digest and Path(doc["weights"]).resolve() == best.resolve()


def test_train_refuses_to_run_on_the_cpu_by_accident(tmp_path, monkeypatch):
    td = _load_script("train_detector")
    yolo = _fake_yolo_class()
    monkeypatch.setattr(td, "_import_torch", lambda: fake_torch(False))
    monkeypatch.setattr(td, "_import_yolo", lambda: yolo)
    data = tmp_path / "cr.yaml"
    data.write_text("names:\n  0: a\n", encoding="utf-8")
    args = ["--data", str(data), "--project", str(tmp_path / "runs"), "--name", "t"]
    assert td.main(args + ["--pin-file", str(tmp_path / "pin.json")]) == 2
    assert yolo.instances == []
    assert td.main(args + ["--device", "cpu", "--no-pin", "--pin-file", str(tmp_path / "pin.json")]) == 0
    assert yolo.instances[-1].train_kwargs["device"] == "cpu"
    assert not (tmp_path / "pin.json").exists()


# ---------------------------------------------------------------------------
# hygiene
# ---------------------------------------------------------------------------

def _imports(path: Path, top_level_only: bool = False) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if top_level_only else ast.walk(tree)
    names = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


@pytest.mark.parametrize("rel", ["scripts/vision/detector.py", "scripts/vision/labels.py",
                                 "scripts/prepare_yolo_dataset.py", "scripts/train_detector.py"])
def test_no_memory_reader_and_lazy_heavy_imports(rel):
    names = _imports(ROOT / rel)
    assert not any("memread" in n or "gamestate" in n or "hcr_reader" in n for n in names), names
    top = _imports(ROOT / rel, top_level_only=True)
    assert not top & {"ultralytics", "torch", "cv2", "PIL"}, top
