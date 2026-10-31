"""FIX A: princess tower labels read where they are (scripts/vision/tower_labels.py, hud.py).

The owner's 2026-10-31 friendly: the opponent's chicken tower skin drew the enemy HP labels
about 36 px (at 1080x1920) above the boxes the HUD reads, which then read the chicken's head
and feathers. Synthetic frames only, drawn with numpy at the owner's 747x1328 by
tests/pipeline/skin_synth.py's plain label: a tan floor (neither bar colour nor gold), a crown
badge in gold, the bar's fill in the side's colour with its dim track, at the reference label
position or moved. The detector's boxes are plain SimpleNamespace objects like FakeDetector's.

These tests pin the placement that came before the per-battle label calibration (HudReader
calibrate=False: LabelPlacer, the fixed boxes, reads on every frame), kept so old sessions replay
as they did. The plain label has no UI outline, so the calibration would refuse it; the
calibrated default is tested in tests/pipeline/test_tower_calib.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import hud as H  # noqa: E402
from scripts.vision import templates as TP  # noqa: E402
from scripts.vision import tower_labels as TL  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from tests.pipeline.skin_synth import (  # noqa: E402  (the frame helpers, moved there)
    CENTRES, ENEMY, FILL, GOLD, HT, OWN, REGIONS, SKIN_DV, SY, W, bar_geometry, body_det, draw_label,
    floor, label_centre_v, label_det, rows, skin_frame)


def close(a, b, tol: float = 1e-6) -> bool:
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def skin_dets(own_dv=(1.0, -1.0)) -> list:
    return ([label_det(k, SKIN_DV) for k in ENEMY] + [label_det(k, dv) for k, dv in zip(OWN, own_dv)]
            + [body_det(k) for k in ENEMY + OWN])


def reader(**kw) -> H.HudReader:
    """The pre-calibration HudReader (module doc)."""
    return H.HudReader(TP.TemplateStore.memory(), {}, regions=REGIONS, towers_wait_for_clock=False,
                       calibrate=False, **kw)


def read(r: H.HudReader, img: np.ndarray, t: float, dets=None) -> VT.Hud:
    hud = r.read(img, t, towers=False)
    ev = None if dets is None else r.tower_evidence(dets, (W, HT))
    return r.read_towers(img, t, hud, ev)


# ---------------------------------------------------------------------------
# (a) evidence -> placement
# ---------------------------------------------------------------------------
def test_the_regions_carry_the_reference_labels():
    refs, fallback = REGIONS.label_refs()
    assert not fallback and set(refs) == set(TL.PRINCESS_KEYS)
    assert all(close(refs[k], TL.LABEL_REF_PX[k]) for k in TL.PRINCESS_KEYS)
    bare = H.HudRegions({k: v for k, v in REGIONS.rects.items() if not k.startswith("label_")},
                        hand_centres=CENTRES, name="old_pc")
    refs, fallback = bare.label_refs()
    assert fallback and refs == TL.LABEL_REF_PX
    r = H.HudReader(TP.TemplateStore.memory(), {}, regions=bare)
    assert r.tower_note().startswith("the HUD regions 'old_pc' have no label_* boxes")
    assert reader().tower_note() is None


def _pc_regions(edit=None) -> dict:
    """The runbook's hud_regions_pc.json: `hud regions --size 747x1328 --out ...` saved before
    the label_* boxes existed (so: none), then edit(doc)."""
    doc = REGIONS.to_dict((W, HT))
    doc["name"] = "pc"
    for k in [k for k in doc["regions"] if k.startswith("label_")]:
        del doc["regions"][k]
    if edit is not None:
        edit(doc)
    return doc


def test_a_regions_file_without_labels_needs_no_edit():
    """A 747x1328 file without label_* boxes: the built-in references come back exactly (its
    boxes' rounding is under LABEL_FOLLOW_MIN_PX), the chicken boxes of 10-31 still place the
    enemy labels, and the setup line is a note, not a WARNING."""
    regions = H.HudRegions.from_dict(_pc_regions(), hand_centres=CENTRES)
    refs, fallback = regions.label_refs()
    assert fallback and refs == TL.LABEL_REF_PX
    r = H.HudReader(TP.TemplateStore.memory(), {}, regions=regions)
    assert "nothing to do" in r.tower_note() and r.tower_warning() is None
    chicken = [SimpleNamespace(cls="tower-bar", box=(122, 154, 236, 189), conf=0.9),
               SimpleNamespace(cls="tower-bar", box=(513, 153, 627, 189), conf=0.9)]
    ev = r.tower_evidence(chicken, (W, HT))
    assert set(ev.placement) == set(ENEMY)
    assert all(-37.0 < ev.placement[k][1] < -34.0 for k in ENEMY)


def test_label_boxes_copied_from_the_1080_file_are_reported():
    """The old WARNING's advice, copying the label_* entries of hud_regions_1080x1920.json
    into the 747x1328 file, put them at 1080/747 times their place: no detector box matched
    them and the WARNING went away. Now tower_warning names them."""
    default = H.HudRegions.builtin(hand_centres=CENTRES).to_dict(H.REF_SIZE)

    def copy_labels(doc):
        for k in TL.PRINCESS_KEYS:
            doc["regions"][f"label_{k}"] = default["regions"][f"label_{k}"]

    regions = H.HudRegions.from_dict(_pc_regions(copy_labels), hand_centres=CENTRES)
    r = H.HudReader(TP.TemplateStore.memory(), {}, regions=regions)
    assert r.tower_note() is None
    warn = r.tower_warning()
    assert warn is not None and all(f"label_{k} " in warn for k in TL.PRINCESS_KEYS)
    assert "delete those label_* entries" in warn
    assert H.HudReader(TP.TemplateStore.memory(), {}, regions=REGIONS).tower_warning() is None


def test_the_fallback_label_follows_the_owners_moved_tower_boxes():
    """The owner moved enemy_left's tower_* and bar_* boxes 7 px down at 747 (10.1 px at 1080)
    to fit his frames: its fallback label reference moves with them, so a default-skin label
    there is no shift (the boxes are read where he put them, not moved a second time)."""
    def move(doc):
        for name in ("tower_enemy_left", "bar_enemy_left"):
            b = doc["regions"][name]["px"]
            doc["regions"][name]["px"] = [b[0], b[1] + 7, b[2], b[3] + 7]

    regions = H.HudRegions.from_dict(_pc_regions(move), hand_centres=CENTRES)
    refs, fallback = regions.label_refs()
    dv = 7 * 1920 / 1328
    x0, y0, x1, y1 = TL.LABEL_REF_PX["enemy_left"]
    assert fallback and close(refs["enemy_left"], (x0, y0 + dv, x1, y1 + dv), 0.05)   # 747 rounding
    assert refs["enemy_right"] == TL.LABEL_REF_PX["enemy_right"]
    ev = TL.TowerEvidence.from_detections([label_det("enemy_left", dv)], (W, HT), refs)
    assert abs(ev.placement["enemy_left"][1]) < 0.05
    assert H.HudReader(TP.TemplateStore.memory(), {}, regions=regions).tower_warning() is None


def test_rect_shifted_keeps_its_size_inside_the_frame():
    r = H.Rect(0.2, 0.1, 0.3, 0.15)
    assert r.shifted(0, 0) is r
    s = r.shifted(0.0, -0.2)
    assert (s.v0, s.u0) == (0.0, 0.2) and abs((s.v1 - s.v0) - 0.05) < 1e-12
    e = r.shifted(0.75, 0.0)
    assert e.u1 == 1.0 and abs((e.u1 - e.u0) - 0.1) < 1e-12


def test_detector_labels_move_the_enemy_boxes_and_leave_ours():
    r = reader(search_labels=False)                 # the detector's placement alone
    img = skin_frame()
    for i in range(4):
        read(r, img, i * 0.125, skin_dets())
    assert r.report.tower_shift["enemy_left"] == [0.0, 0.0, "fixed"]       # 4 votes: not yet
    read(r, img, 0.5, skin_dets())
    sh = r.report.tower_shift
    for k in ENEMY:
        assert sh[k][0] == 0.0 and abs(sh[k][1] - SKIN_DV) < 1e-6 and sh[k][2] == "detector"
    assert sh["own_left"] == [0.0, 0.0, "fixed"] and sh["own_right"] == [0.0, 0.0, "fixed"]
    tr = r.tower_reader
    assert tr._region("tower_enemy_left", "enemy_left").px((W, HT)) == (147, 146, 236, 172)
    assert tr._region("bar_enemy_left", "enemy_left").px((W, HT)) == (144, 165, 237, 200)
    assert REGIONS.px("tower_enemy_left", (W, HT)) == (147, 171, 236, 197)        # calibrated
    assert "own_left" not in tr.placement
    assert r.report.tower_bars["enemy_left"] == 1.0 and r.report.tower_bars["enemy_right"] == 1.0
    assert r.report.tower_bars["own_left"] == 1.0
    assert r.report.summary()["tower_shift"]["enemy_left"][2] == "detector"


# ---------------------------------------------------------------------------
# (b) what is not a label
# ---------------------------------------------------------------------------
def test_boxes_that_are_not_labels():
    k = "enemy_left"
    kx = (TL.TOWER_BODY_REF_PX["enemy_king"][0] + TL.TOWER_BODY_REF_PX["enemy_king"][2]) / 2.0
    lx = (TL.LABEL_REF_PX[k][0] + TL.LABEL_REF_PX[k][2]) / 2.0
    cases = {
        "bar class": [label_det(k, SKIN_DV, cls="bar")],
        "twice as tall": [label_det(k, SKIN_DV, tall=2.0)],
        "at the king's x": [label_det(k, SKIN_DV, du=kx - lx)],
        "120 px low": [label_det(k, 120.0)],
    }
    for name, dets in cases.items():
        ev = TL.TowerEvidence.from_detections(dets, (W, HT))
        assert k not in ev.placement and not ev.placement, name
        assert ev.seen(k) is (name == "twice as tall"), name
    ev = TL.TowerEvidence.from_detections([body_det("enemy_king", "Cannoneer")], (W, HT))
    assert ev.bodies == {"enemy_king"} and ev.seen("enemy_king") and not ev.labels
    ev = TL.TowerEvidence.from_detections([body_det("enemy_left", "Cannoneer"), label_det("own_right")], (W, HT))
    assert ev.bodies == {"enemy_left"} and set(ev.labels) == {"own_right"}
    assert close(ev.placement["own_right"], (0.0, 0.0))
    # a dagger-duchess label is wider than the reference: presence only
    dd = TL.TowerEvidence.from_detections([label_det(k, cls="dagger-duchess-tower-bar", wide=1.4)], (W, HT))
    assert dd.seen(k) and dd.placement == {}


# ---------------------------------------------------------------------------
# (c) the median, stickiness, resets
# ---------------------------------------------------------------------------
def test_the_shift_is_a_sticky_median_until_reset_towers():
    r = reader()
    img = skin_frame()
    for i, dv in enumerate([SKIN_DV, SKIN_DV, -80.0, SKIN_DV, SKIN_DV]):
        read(r, img, i * 0.125, [label_det("enemy_left", dv)])
    assert abs(r.report.tower_shift["enemy_left"][1] - SKIN_DV) < 1e-6
    for i in range(5, 10):                                  # no boxes: the shift stays
        read(r, img, i * 0.125, [])
    assert r.report.tower_shift["enemy_left"][2] == "detector"
    r.reset(keep_towers=True)
    read(r, img, 2.0, [])
    assert r.report.tower_shift["enemy_left"][2] == "detector"
    r.reset_towers()
    read(r, floor(), 2.5, [])
    assert r.report.tower_shift["enemy_left"] == [0.0, 0.0, "fixed"] and r.tower_reader.placement == {}


# ---------------------------------------------------------------------------
# (d), (e) the pixel search
# ---------------------------------------------------------------------------
def anchor_u(key: str) -> float:
    return REGIONS.get(f"bar_{key}").u0 + REGIONS.bar_margin


def find(img: np.ndarray, key: str) -> float | None:
    return TL.find_label_dv(img, key, anchor_u(key), TL.LABEL_REF_PX[key])


def test_the_search_finds_a_raised_label_over_red_feathers():
    k = "enemy_left"
    img = draw_label(floor(), k, SKIN_DV)
    cv = label_centre_v(k, SKIN_DV + 60.0)                  # red "feathers" 60 px lower, no badge
    ax, full = bar_geometry(k)
    img[rows(cv - 10, cv + 10), ax - 25:ax + full] = FILL["enemy"]
    dv = find(img, k)
    assert dv is not None and abs(dv - SKIN_DV) <= 2.0
    calibrated = find(draw_label(floor(), k), k)
    assert calibrated is not None and abs(calibrated) < 3.0                  # the dead band
    for key in ("enemy_right", "own_left", "own_right"):
        got = find(draw_label(floor(), key, SKIN_DV), key)
        assert got is not None and abs(got - SKIN_DV) <= 2.0, key


def test_the_search_takes_our_label_not_the_roof_trim_above_it():
    k = "own_left"
    img = draw_label(floor(), k)
    side_bc = label_centre_v(k) + TL.BADGE_DV_PX["own"] - 40.0          # gold trim on a blue roof
    ax, full = bar_geometry(k)
    img[rows(side_bc - 13, side_bc + 13), ax - 25:ax + full] = FILL["own"]
    img[rows(side_bc - 13, side_bc + 13), ax - 21:ax - 3] = GOLD
    dv = find(img, k)
    assert dv is not None and abs(dv) <= 2.0


def test_the_search_refuses_ambiguous_or_badgeless_labels():
    k = "enemy_left"
    two = draw_label(draw_label(floor(), k, -10.0), k, 10.0)              # 20 px apart: one blob
    assert find(two, k) is None
    pair = draw_label(draw_label(floor(), k, -18.0, badge_tall=24.0), k, 10.0, badge_tall=24.0)
    assert find(pair, k) is None                                          # 18 and 10 px away: unsure
    one = draw_label(floor(), k, 10.0, badge_tall=24.0)
    assert abs(find(one, k) - 10.0) <= 2.0
    overlay = floor()
    cv = label_centre_v(k)
    ax, full = bar_geometry(k)
    overlay[rows(cv - 20, cv + 30.6), ax - 30:ax + full] = (240, 120, 170)     # pink, 35 rows
    assert find(overlay, k) is None
    assert find(floor(), k) is None


def test_the_search_places_the_labels_without_a_detector():
    r = reader()
    img = skin_frame()
    for i in range(3):
        read(r, img, i * 0.125)                                  # no detector: evidence None
    sh = r.report.tower_shift
    for k in ENEMY:
        assert sh[k][2] == "search" and abs(sh[k][1] - SKIN_DV) <= 2.0, k
    assert sh["own_left"][2] == "fixed" and r.report.tower_bars["enemy_left"] == 1.0
    assert r.placer.wants_search("enemy_left") is False          # settled: the search stops
    off = reader(search_labels=False)
    read(off, img, 0.0)
    assert off.report.tower_bars["enemy_left"] == 0.0 and off.report.tower_shift["enemy_left"][2] == "fixed"


# ---------------------------------------------------------------------------
# (f) bars, (g) digits, the pixel hold's badge test
# ---------------------------------------------------------------------------
def test_bars_read_at_the_placed_label():
    tr = H.TowerReader(TP.TemplateStore.memory(), REGIONS)
    img = skin_frame()
    assert tr.read_bars(img)["enemy_left"] == 0.0
    tr.placement = {k: (0.0, SKIN_DV / 1920.0) for k in ENEMY}
    bars = tr.read_bars(img)
    assert bars["enemy_left"] == 1.0 and bars["enemy_right"] == 1.0 and bars["own_left"] == 1.0
    half = draw_label(floor(), "enemy_left", SKIN_DV, frac=0.5)
    assert abs(tr.read_bars(half)["enemy_left"] - 0.5) <= 0.01


def test_digits_read_at_the_placed_label():
    from tests.pipeline.test_hud import compose, digit_store
    k = "enemy_left"
    img = floor()
    glyphs = compose("3052")
    top = int(round((label_centre_v(k, SKIN_DV) - 33.0) * SY))
    img[top:top + glyphs.shape[0], 152:152 + glyphs.shape[1]] = glyphs
    tr = H.TowerReader(digit_store(("tower",)), REGIONS)
    assert tr.read_digits(img)[k].hp is None
    tr.placement = {k: (0.0, SKIN_DV / 1920.0)}
    assert tr.read_digits(img)[k].hp == 3052


def test_label_seen_is_the_badge_at_the_placed_row():
    tr = H.TowerReader(TP.TemplateStore.memory(), REGIONS)
    img = skin_frame()
    seen = tr.label_seen(img)
    assert seen["own_left"] is True and seen["enemy_left"] is False and seen["enemy_king"] is None
    tr.placement = {k: (0.0, SKIN_DV / 1920.0) for k in ENEMY}
    assert tr.label_seen(img)["enemy_left"] is True
    empty = draw_label(floor(), "enemy_left", SKIN_DV, frac=0.0)            # any HP shows the badge
    assert tr.label_seen(empty)["enemy_left"] is True
    rubble = draw_label(floor(), "enemy_left", SKIN_DV, badge=False)
    assert tr.label_seen(rubble)["enemy_left"] is False


# ---------------------------------------------------------------------------
# presence end to end: a tower the detector sees is not destroyed; one it lost is
# ---------------------------------------------------------------------------
def test_a_tower_the_detector_sees_is_never_read_destroyed():
    """The incident without placement (place_labels and search off): the fixed boxes read the
    feathers (0.22), then nothing. The detector still sees the tower: it holds at 671."""
    k = "enemy_left"
    r = reader(place_labels=False, search_labels=False)
    feathers = floor()
    ax, full = bar_geometry(k)
    feathers[rows(282.5 + 8, 282.5 + 22), ax:ax + int(round(0.22 * full))] = FILL["enemy"]
    dets = [label_det(k, SKIN_DV), body_det(k)]
    huds = [read(r, feathers, i * 0.125, dets) for i in range(8)]
    huds += [read(r, floor(), i * 0.125, dets) for i in range(8, 60)]
    assert r.report.tower_bars[k] == 0.0 and huds[7].towers[k] == VT.TowerHp(678, 3052)   # 0.222
    assert all(h.towers[k] == VT.TowerHp(678, 3052) for h in huds[8:])
    assert all(h.crowns == (0, 0) for h in huds)
    assert r.towers.holds[k] == 52 and r.report.tower_present[k] is True
    # the detector loses it: destroyed once it has been gone 8 frames and 1 s, and the HUD's 2.5 s
    gone = [read(r, floor(), 7.5 + i * 0.125, []) for i in range(30)]
    first = next(i for i, h in enumerate(gone) if h.towers[k].hp == 0)
    assert first == 20 and gone[first].crowns == (1, 0) and r.towers.destroyed_by[k] == "detector"
    assert r.report.tower_present[k] is False


def test_no_detector_keeps_the_old_rule_and_the_badge_holds():
    k = "enemy_left"
    ax, full = bar_geometry(k)
    feathers = floor()
    feathers[rows(282.5 + 8, 282.5 + 22), ax:ax + int(round(0.22 * full))] = FILL["enemy"]
    old = reader(place_labels=False, search_labels=False, label_hold=False)
    huds = [read(old, feathers, i * 0.125) for i in range(8)]
    huds += [read(old, floor(), i * 0.125) for i in range(8, 40)]
    first = next(i for i, h in enumerate(huds) if h.towers[k].hp == 0)
    assert first == 8 + 20 and huds[-1].crowns == (1, 0)                       # today's 2.5 s
    badge = floor()
    draw_label(badge, k, bar=False)                                          # the badge, no bar
    feathers_badge = draw_label(feathers.copy(), k, bar=False)
    hold = reader(place_labels=False, search_labels=False)
    huds = [read(hold, feathers_badge, i * 0.125) for i in range(8)]
    huds += [read(hold, badge, i * 0.125) for i in range(8, 40)]
    assert all(h.towers[k].hp > 0 for h in huds) and hold.report.tower_present[k] is True
