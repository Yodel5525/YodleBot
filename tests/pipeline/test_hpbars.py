"""V07: HP bars (scripts/vision/hpbars.py): unit bar fill, bar -> unit attachment, tower bar measure.

Bars are drawn with numpy on a 360x640 frame: the fill in a team colour, the
drained part dark, arena green around. Colours are the ones the author
measured for the tower bars (scripts/tower_hp.py:172-174, 177-179, 224).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import hpbars as B  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from scripts.vision.tracker import Observation  # noqa: E402

BG = (90, 160, 70)
BLUE = (60, 120, 230)
RED = (220, 40, 40)
DARK = (25, 25, 25)
WHITE = (255, 255, 255)
# unit bars on the owner's frames (client 160635001, 2026-10-29)
ALLY_FILL, ALLY_DRAINED = (113, 208, 253), (69, 77, 113)
ENEMY_FILL, ENEMY_DRAINED = (225, 36, 100), (96, 45, 60)
RIVER = (211, 95, 202)                  # the purple river behind units crossing it
BADGE = (90, 187, 232)                  # our level badge
OUTLINE = (20, 30, 60)


def canvas() -> np.ndarray:
    f = np.empty((640, 360, 3), dtype=np.uint8)
    f[:] = BG
    return f


def draw_bar(f: np.ndarray, x0: int, y0: int, w: int, h: int, fill_px: int, colour,
             drained=DARK) -> tuple[int, int, int, int]:
    f[y0:y0 + h, x0:x0 + fill_px] = colour
    f[y0:y0 + h, x0 + fill_px:x0 + w] = drained
    return (x0, y0, x0 + w, y0 + h)


def px(colour) -> np.ndarray:
    return np.array([[colour]], dtype=np.uint8)


def test_team_masks_are_the_authors_tower_masks():
    assert B.ally_mask(px(BLUE))[0, 0] and not B.enemy_mask(px(BLUE))[0, 0]
    for c in (RED, (255, 156, 212), (255, 205, 255)):      # deep red, bright pink, light pink fills
        assert B.enemy_mask(px(c))[0, 0] and not B.ally_mask(px(c))[0, 0], c
    for c in ((93, 50, 73), BG, DARK):                      # drained enemy track (tower_hp.py:177-179)
        assert not B.enemy_mask(px(c))[0, 0] and not B.ally_mask(px(c))[0, 0], c
    assert B.dark_mask(px(DARK))[0, 0] and not B.dark_mask(px(BG))[0, 0]


def test_unit_bar_colours_from_the_owners_client():
    for c in (ALLY_DRAINED, ENEMY_DRAINED, (93, 50, 73)):     # drained parts are dim, not black
        assert B.dark_mask(px(c))[0, 0], c
    assert B.ally_mask(px(ALLY_FILL))[0, 0] and B.unit_enemy_mask(px(ENEMY_FILL))[0, 0]
    for c in (RIVER, (233, 162, 100)):                        # the river, an orange sprite
        assert B.enemy_mask(px(c))[0, 0] and not B.unit_enemy_mask(px(c))[0, 0], c
    f = canvas()
    r = B.read_bar(f, draw_bar(f, 100, 300, 60, 8, 45, ALLY_FILL, ALLY_DRAINED))
    assert (r.frac, r.team, r.fill_px, r.track_px) == (0.75, VT.OWN_TEAM, 45, 60)
    f[190:220, 90:180] = RIVER                                # an enemy unit crossing the river
    r = B.read_bar(f, draw_bar(f, 100, 200, 60, 8, 30, ENEMY_FILL, ENEMY_DRAINED))
    assert (r.frac, r.team, r.fill_px, r.track_px) == (0.5, VT.ENEMY_TEAM, 30, 60)


def test_read_bar_skips_the_level_badge():
    f = canvas()
    f[196:212, 100:112] = BADGE
    f[199:209, 103:109] = WHITE                               # the level number
    f[200:208, 112:114] = OUTLINE
    draw_bar(f, 114, 200, 60, 8, 45, ALLY_FILL, ALLY_DRAINED)
    r = B.read_bar(f, (100, 196, 174, 212))                   # a "bar-level" box: badge + bar
    assert (r.frac, r.team, r.fill_px, r.track_px) == (0.75, VT.OWN_TEAM, 45, 60)
    assert B.read_bar(f, (100, 196, 112, 212)).frac is None    # the badge alone is no bar


def test_walk_run_bridges_short_gaps():
    cols = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    assert B.walk_run(cols, 0, 0) == 2
    assert B.walk_run(cols, 0, 1) == 4
    assert B.walk_run(cols, 0, 2) == 7
    assert B.walk_run(cols, 0, 3) == 11


def test_read_bar_fill_fraction_and_team():
    f = canvas()
    own = draw_bar(f, 100, 200, 60, 8, 45, BLUE)
    r = B.read_bar(f, own)
    assert (r.frac, r.team, r.fill_px, r.track_px, r.conf) == (0.75, VT.OWN_TEAM, 45, 60, 1.0)
    enemy = draw_bar(f, 200, 300, 60, 8, 30, RED)
    r = B.read_bar(f, enemy)
    assert (r.frac, r.team) == (0.5, VT.ENEMY_TEAM)
    assert B.read_bar(f, enemy, VT.ENEMY_TEAM).frac == 0.5
    assert B.read_bar(f, enemy, VT.OWN_TEAM).frac is None    # no blue in a red bar
    full = draw_bar(f, 100, 400, 60, 8, 60, BLUE)
    assert B.read_bar(f, full).frac == 1.0
    almost = draw_bar(f, 200, 400, 60, 8, 58, BLUE)           # 58/60 = 0.967 >= 0.95: full
    assert B.read_bar(f, almost).frac == 1.0
    notched = draw_bar(f, 100, 500, 60, 8, 45, BLUE)
    f[500:508, 120] = DARK                                    # a 1 px notch inside the fill
    assert B.read_bar(f, notched).frac == 0.75
    assert B.read_bar(f, (10, 10, 50, 18)) == B.BarRead(None)  # plain arena: no bar
    assert B.read_bar(f, (400, 700, 450, 710)) == B.BarRead(None)   # off the frame


def test_attach_bars_to_the_unit_below():
    units = [(100, 200, 140, 260), (300, 200, 340, 260)]
    bars = [(600, 100, 650, 106), (300, 190, 340, 196), (95, 185, 145, 192)]
    assert B.attach_bars(units, bars) == [2, 1]
    # one bar between two units: it belongs to the one it is centred on
    assert B.attach_bars([(100, 200, 140, 260), (130, 205, 170, 265)], [(125, 190, 165, 196)]) == [None, 0]
    # a bar below a unit, or far above it, is not its bar
    assert B.attach_bars([(100, 200, 140, 260)], [(100, 270, 140, 276)]) == [None]
    assert B.attach_bars([(100, 200, 140, 260)], [(100, 150, 140, 156)]) == [None]
    assert B.attach_bars([], bars) == [] and B.attach_bars(units, []) == [None, None]


def test_read_unit_hp_without_a_bar_is_full():
    f = canvas()
    bar = draw_bar(f, 100, 200, 60, 8, 45, BLUE)
    units = [(100, 210, 160, 270), (300, 210, 340, 270)]
    hp = B.read_unit_hp(f, units, [bar])
    assert hp[0] == B.HpRead(0.75, True, VT.OWN_TEAM, 1.0, 0)
    assert hp[1] == B.HpRead(1.0, False)
    unreadable = B.read_unit_hp(f, [(10, 30, 50, 90)], [(10, 20, 50, 28)])     # attached, but no colour
    assert unreadable == [B.HpRead(1.0, False, VT.UNKNOWN_TEAM, 0.0, 0)]


def test_hp_for_detections_and_tracker_observations():
    f = canvas()
    bar = draw_bar(f, 100, 200, 60, 8, 45, BLUE)
    dets = [VT.Detection("bar", 0.5, 0.9, tuple(float(c) for c in bar)),
            VT.Detection("hog-rider", 0.9, 0.9, (100.0, 210.0, 160.0, 270.0)),
            VT.Detection("skeleton", 0.1, 0.8, (300.0, 210.0, 340.0, 270.0))]
    assert B.split_bars(dets) == ([1, 2], [0])
    reads = B.hp_for_detections(f, dets)
    assert sorted(reads) == [1, 2]
    assert (reads[1].frac, reads[1].has_bar, reads[1].team) == (0.75, True, VT.OWN_TEAM)
    assert (reads[2].frac, reads[2].has_bar) == (1.0, False)
    obs = [Observation("hog-rider", 0.0, 0.0), Observation("skeleton", 0.0, 0.0)]
    B.annotate_observations(obs, [reads[1], reads[2]])
    assert (obs[0].hp_frac, obs[0].bar_team) == (0.75, VT.OWN_TEAM)
    assert (obs[1].hp_frac, obs[1].bar_team) == (None, VT.UNKNOWN_TEAM)   # tracker keeps its value
    B.annotate_observations(obs, [reads[1], reads[2]], missing_frac=1.0)
    assert obs[1].hp_frac == 1.0


def band(rows=range(20, 28), cols=range(8, 68), shape=(50, 135)) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[np.ix_(list(rows), list(cols))] = True
    return m


def test_measure_tower_bar():
    kw = dict(anchor_px=8, tol_px=8)
    assert B.measure_tower_bar(band(), 119, **kw) == 0.504                      # 60 / 119
    assert B.measure_tower_bar(band(cols=range(8, 127)), 119, **kw) == 1.0      # full
    assert B.measure_tower_bar(band(cols=range(8, 122)), 119, **kw) == 1.0      # 114/119 = 0.958: full
    assert B.measure_tower_bar(band(rows=range(20, 23)), 119, **kw) == 0.0      # 3 rows: too thin
    assert B.measure_tower_bar(band(cols=range(40, 100)), 119, **kw) == 0.0     # not at the anchor
    notch = band()
    notch[:, 30:32] = False                                                     # 2 px: bridged, counted
    assert B.measure_tower_bar(notch, 119, **kw) == 0.504
    wide = band()
    wide[:, 30:33] = False                                                      # 3 px: the run stops at 22
    assert B.measure_tower_bar(wide, 119, **kw) == 0.185
    assert B.measure_tower_bar(np.zeros((0, 0), dtype=bool), 119) == 0.0


def test_measure_tower_bar_matches_the_authors_reader():
    pytest.importorskip("PIL")                   # scripts/tower_hp.py imports Pillow at the top
    from scripts import tower_hp
    rng = np.random.default_rng(7)
    for trial in range(40):
        full = np.zeros((50, 1080), dtype=bool)
        for anchor in (216, 782):
            start = anchor + int(rng.integers(-10, 11))
            length = int(rng.integers(0, 130))
            top, thick = int(rng.integers(0, 40)), int(rng.integers(2, 11))
            full[top:top + thick, max(0, start):start + length] = True
        full |= rng.random(full.shape) < 0.01                                   # speckle
        for anchor in (216, 782):
            ours = B.measure_tower_bar(full[:, anchor - 8:anchor + 119 + 8], 119, anchor_px=8, tol_px=8)
            assert ours == tower_hp._measure_side_hp(full, anchor, 119), (trial, anchor)
