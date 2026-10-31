"""tests/pipeline/skin_synth.py: the synthetic frames of the tower-skin and arena tests.

The generator is checked where it does not depend on the readers the skin work will change:
where it draws each part (the label at (du, dv), its badge, bar and digits, kings, ruins, flying
units), the colours that make the grid adversarial (the fix list's findings 7 and 14), the
deploy overlay as measured, the screen cues of battle and end-screen frames, and the resize.
Today's pixel search (tower_labels.find_label_dv), bar measure and badge test find the label
it draws where label_parts says, at both frame sizes.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import vision_play as VP  # noqa: E402
from scripts.vision import hpbars  # noqa: E402
from scripts.vision import hud as H  # noqa: E402
from scripts.vision import tower_labels as TL  # noqa: E402
from tests.pipeline import skin_synth as S  # noqa: E402

PRINCESS = TL.PRINCESS_KEYS


def at(img: np.ndarray, x: float, y: float) -> tuple[int, int, int]:
    """The pixel at (x, y) px at 1080x1920 on a frame of any size."""
    h, w = img.shape[:2]
    return tuple(int(c) for c in img[int(y * h / 1920.0), int(x * w / 1080.0)])


def crop(img: np.ndarray, box) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(round(y0 * h / 1920.0)):int(round(y1 * h / 1920.0)),
               int(round(x0 * w / 1080.0)):int(round(x1 * w / 1080.0))]


@lru_cache(maxsize=None)
def plain(fl: str, size: tuple[int, int]) -> np.ndarray:
    return S.battle_frame(fl, size=size)


# ---------------------------------------------------------------------------
# floors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", S.SIZES)
@pytest.mark.parametrize("fl", S.FLOOR_NAMES)
def test_floors_rivers_and_battle_cues(fl, size):
    img = plain(fl, size)
    floor = S.FLOORS[fl]
    assert img.shape == (size[1], size[0], 3) and img.dtype == np.uint8
    assert at(img, 540.0, 600.0) == floor.colour                      # the enemy half, between the towers
    assert at(img, 540.0, 1000.0) == floor.colour
    mid = sum(S.RIVER_PX) / 2.0
    if floor.river is None:
        assert at(img, 540.0, mid) == floor.colour
    else:
        assert at(img, 540.0, mid) == floor.river
        assert at(img, sum(S.BRIDGES_PX[0]) / 2.0, mid) == floor.colour    # a bridge
    if floor.bank is not None:
        assert at(img, 540.0, S.RIVER_PX[0] - S.BANK_PX / 2.0) == floor.bank
    cues = VP.read_cues(img)
    assert cues.hud_present and not cues.end_screen                     # a battle frame


def test_the_grid_is_adversarial_to_todays_colour_tests():
    """The fix list's findings: the red floor and the comb pass the enemy bar colour, the river's
    cyan passes ours and its brown bank the enemy unit bar's (14), the pad passes the badge gold."""
    px = lambda c: np.array([[c]], dtype=np.uint8)  # noqa: E731
    assert hpbars.enemy_mask(px(S.FLOORS["red"].colour)).all() and hpbars.enemy_mask(px(S.COMB)).all()
    assert hpbars.ally_mask(px(S.FLOORS["sand_river"].river)).all()
    assert hpbars.unit_enemy_mask(px(S.FLOORS["sand_river"].bank)).all()
    assert TL.gold_mask(px(S.PAD)).all() and TL.gold_mask(px(S.BADGE_GOLD)).all()
    assert not hpbars.enemy_mask(px(S.FLOORS["default"].colour)).any()
    assert not TL.gold_mask(px(S.FLOORS["sand"].colour)).any()
    assert set(S.DECORATIONS) == {"comb", "head", "roof", "flowers", "pad"} and len(S.FLOOR_NAMES) == 7


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", S.SIZES)
@pytest.mark.parametrize("dv", [-36.0, 0.0, 12.0])
def test_todays_search_finds_the_label_where_label_parts_says(dv, size):
    labels = {k: S.Label(dv=dv, frac=0.5) for k in PRINCESS}
    img = S.battle_frame("default", labels=labels, size=size)
    tr = H.TowerReader(None, S.REGIONS)
    for k in PRINCESS:
        found = tr.find_label(img, k)
        assert found is not None and abs(found - dv) <= 2.0, (k, found)
        row = TL.ref_badge_row(k, TL.LABEL_REF_PX[k]) + dv
        anchor = S.anchor_x(k) / 1080.0
        assert TL.badge_seen(img, anchor, row) and not TL.badge_seen(img, anchor, row + 30.0), k
    tr.placement = {k: (0.0, dv / 1920.0) for k in PRINCESS}
    bars = tr.read_bars(img)
    assert all(abs(bars[k] - 0.5) <= 0.02 for k in PRINCESS), bars


@pytest.mark.parametrize("size", S.SIZES)
def test_the_parts_of_a_label_moved_sideways(size):
    du, dv = 40.0, -36.0
    img = S.battle_frame("snow", labels={"enemy_right": S.Label(du=du, dv=dv, frac=0.25)}, size=size)
    parts = S.label_parts("enemy_right", du, dv)
    assert parts["bar"][0] == pytest.approx(S.anchor_x("enemy_right") + du)
    assert parts["bar"][2] - parts["bar"][0] == S.BAR_FULL_PX == 119
    x0, y0, x1, y1 = parts["bar"]
    row = (y0 + y1) / 2.0
    assert at(img, x0 + 10, row) == S.FILL["enemy"] and at(img, x1 - 10, row) == S.TRACK["enemy"]
    assert hpbars.dark_mask(np.array([[at(img, x1 - 10, y0 - 1.5)]], dtype=np.uint8)).all()   # the outline
    assert TL.gold_mask(crop(img, parts["badge"])).mean() > 0.4
    assert at(img, S.anchor_x("enemy_right") - 20, row) == S.FLOORS["snow"].colour     # no badge at du 0


@pytest.mark.parametrize("size", S.SIZES)
def test_hp_digits_read_with_the_same_glyphs_templates(size):
    pytest.importorskip("PIL")
    if S.pil_glyph_fn() is None:
        pytest.skip("this Pillow's default font renders a digit in pieces")
    tr = H.TowerReader(S.digit_store(size), S.REGIONS)
    tr.placement = {k: (0.0, S.SKIN_DV / 1920.0) for k in S.ENEMY}
    for hp in (3052, 1526, 4567, 890, 12):
        labels = {k: S.Label(dv=S.SKIN_DV if k in S.ENEMY else 0.0, hp=hp) for k in PRINCESS}
        reads = tr.read_digits(S.battle_frame("forest", labels=labels, size=size))
        for k in PRINCESS:
            assert reads[k].hp == hp and reads[k].conf >= 0.8, (k, hp, reads[k])


def test_kings_full_and_damaged():
    img = S.battle_frame("default", kings={"enemy_king": S.KingLabel(frac=0.6)})
    full = S.king_parts("own_king")
    assert set(full) == {"badge"} and TL.gold_mask(crop(img, full["badge"])).mean() > 0.5
    hurt = S.king_parts("enemy_king", S.KingLabel(frac=0.6))
    assert set(hurt) == {"badge", "bar", "digits"}
    x0, y0, x1, y1 = hurt["bar"]
    row = crop(img, (x0, (y0 + y1) / 2.0 - 2.0, x1, (y0 + y1) / 2.0 + 2.0))
    assert hpbars.enemy_mask(row).mean() == pytest.approx(0.6, abs=0.01)
    assert hurt["badge"][2] < hurt["bar"][0] and (hurt["badge"][0] + hurt["bar"][2]) / 2.0 == pytest.approx(S.KING_X)
    assert S.KingLabel(frac=0.5).shown_hp() == 2412 and S.KingLabel().shown_hp() == 4824


# ---------------------------------------------------------------------------
# decorations, ruins, units
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", S.SIZES)
def test_decorations(size):
    k = "enemy_left"
    img = S.battle_frame("default", labels={k: S.Label(dv=S.SKIN_DV)},
                         decorations=[(k, "comb"), ("own_left", "roof"), ("own_right", "pad", 0.0, -38.0),
                                      ("enemy_right", "head", 0.0, 0.0)], size=size)
    # the comb sits in the fixed bar band, from the anchor: a quarter of the bar there
    comb = S.decoration_box(k, "comb")
    assert comb[0] < S.anchor_x(k) < comb[2]
    assert (comb[2] - S.anchor_x(k)) / S.BAR_FULL_PX == pytest.approx(0.25, abs=0.01)
    assert hpbars.enemy_mask(crop(img, comb)).mean() > 0.9
    # the roof touches the badge: one gold run of rows over the badge columns, too tall for a badge
    bx0, by0, bx1, by1 = S.label_parts("own_left")["badge"]
    rows = TL.gold_mask(crop(img, (bx0 + 4, by0 - 20, bx1 - 4, by1 + 6))).mean(axis=1) >= TL.GOLD_ROW_MIN
    runs = np.diff(np.concatenate([[0], rows.astype(int), [0]]))
    tallest = int(np.max(np.nonzero(runs == -1)[0] - np.nonzero(runs == 1)[0]))
    assert tallest * 1920.0 / size[1] > TL.BADGE_TALL_PX[1]
    # the pad: a 40-px gold group at the badge columns, 38 px above the badge row
    pad = S.decoration_box("own_right", "pad", 0.0, -38.0)
    assert pad[3] - pad[1] == 40.0 and TL.gold_mask(crop(img, pad)).mean() > 0.95
    # the head: white behind enemy_right's digits (the digits are drawn over it)
    head = S.decoration_box("enemy_right", "head")
    digits = S.label_parts("enemy_right")["digits"]
    assert head[0] < digits[0] and head[1] < digits[1] and head[3] > digits[3]
    assert at(img, head[0] + 2, head[1] + 2) == S.HEAD


@pytest.mark.parametrize("size", S.SIZES)
def test_ruins_and_flying_units(size):
    img = S.battle_frame("red", labels={"enemy_right": None},
                         flyers=[S.flyer_box("own_left")], size=size)
    parts = S.label_parts("enemy_right")
    assert TL.gold_mask(crop(img, parts["badge"])).mean() == 0.0                # a ruin has no label
    assert hpbars.dark_mask(crop(img, parts["outline"])).mean() == 0.0
    x0, y0, x1, y1 = TL.TOWER_BODY_REF_PX["enemy_right"]
    assert at(img, (x0 + x1) / 2.0, y1 - 10.0) == S.RUIN
    box = S.flyer_box("own_left")
    for name in ("badge", "bar"):
        b = S.label_parts("own_left")[name]
        assert box[0] <= b[0] and box[1] <= b[1] and box[2] >= b[2] and box[3] >= b[3]
    assert at(img, (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0) == S.FLYER
    d = S.det("Balloon", box, size=size)
    assert d.box == pytest.approx((box[0] * size[0] / 1080, box[1] * size[1] / 1920,
                                   box[2] * size[0] / 1080, box[3] * size[1] / 1920))


# ---------------------------------------------------------------------------
# the deploy overlay
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", S.SIZES)
def test_the_deploy_overlay_tints_the_floor_only(size):
    img = S.battle_frame("forest", overlay=True, size=size)
    base = plain("forest", size)
    fl = np.array(S.FLOORS["forest"].colour, dtype=float)
    want = tuple(int(v) for v in np.rint(fl * 0.55 + np.array(S.OVERLAY) * 0.45))
    assert at(img, 540.0, 600.0) == want == (250, 171, 143)            # the R0 tiles: (250, 172, 146)
    assert at(img, 540.0, 1000.0) == at(base, 540.0, 1000.0)           # our half
    tinted = np.array([[want]], dtype=np.uint8)
    assert hpbars.enemy_mask(tinted).all() and VP.red_mask(tinted.astype(np.int16)).all()      # finding 7
    for k in PRINCESS:                                                 # the labels keep their colours
        parts = S.label_parts(k)
        badge, bar = parts["badge"], parts["bar"]
        body = ((badge[0] + badge[2]) / 2.0 - 8.0, badge[3] - 6.0)     # the badge's gold, left of the level
        assert at(img, *body) == at(base, *body) == S.BADGE_GOLD, k
        assert at(img, bar[0] + 100.0, (bar[1] + bar[3]) / 2.0) == at(base, bar[0] + 100.0, (bar[1] + bar[3]) / 2.0)
    cues = VP.read_cues(img)
    assert cues.red_banner >= 0.65 and cues.hud_present and not cues.end_screen


# ---------------------------------------------------------------------------
# the end screen
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", S.SIZES)
@pytest.mark.parametrize("fl", S.FLOOR_NAMES)
def test_end_screens(fl, size):
    for own, enemy in ((1, 0), (0, 0), (3, 2)):
        img = S.end_screen(fl, own, enemy, size=size)
        cues = VP.read_cues(img)
        assert cues.end_screen and not cues.hud_present
        assert VP.end_screen_crowns(img) == (own, enemy), (own, enemy)


def test_end_screen_crowns_and_decorations():
    plain_end = S.end_screen("sand", 1, 0)
    decor = S.end_screen("sand", 1, 0, decor=("flowers", "king"))
    x0, y0, x1, y1 = S._nbox(S.CROWN_SLOTS_U[0] - S.CROWN_HALF_U, S.CROWN_ROWS_V["own"][0],
                             S.CROWN_SLOTS_U[0] + S.CROWN_HALF_U, S.CROWN_ROWS_V["own"][1])
    assert 1.1 <= (x1 - x0) / (y1 - y0) <= 1.6                         # a crown's shape
    assert at(plain_end, (x0 + x1) / 2.0, y1 - 5.0) == S.CROWN_BAND
    assert at(plain_end, 540.0, 0.18 * 1920) == S.FLOORS["sand"].colour
    assert at(decor, 540.0, 0.18 * 1920) == S.BADGE_GOLD               # the king behind the middle slot
    with pytest.raises(ValueError):
        S.end_screen("sand", decor=("confetti",))


# ---------------------------------------------------------------------------
# resizing
# ---------------------------------------------------------------------------
def test_resize_is_an_area_average():
    rng = np.random.default_rng(3)
    block = rng.integers(0, 256, (8, 6, 3)).astype(np.uint8)
    assert np.array_equal(S.resize(block, (3, 4)), np.rint(block.reshape(4, 2, 3, 2, 3).mean(axis=(1, 3))))
    big = rng.integers(0, 256, (1920, 1080, 3)).astype(np.uint8)
    small = S.resize(big, S.OWNER_SIZE)
    assert small.shape == (1328, 747, 3) and abs(float(small.mean()) - float(big.mean())) < 0.05
    const = np.full((1920, 1080, 3), 77, dtype=np.uint8)
    assert set(np.unique(S.resize(const, S.OWNER_SIZE))) == {77}
    assert set(S.at_sizes(const)) == {S.REF_SIZE, S.OWNER_SIZE}


def test_the_plain_label_keeps_its_747_columns():
    """test_tower_labels.py's plain badge: columns [anchor - 21, anchor - 3) at 747x1328."""
    img = S.draw_label(S.floor(), "enemy_left")
    ax, full = S.bar_geometry("enemy_left")
    gold = np.nonzero((img == S.GOLD).all(axis=2).any(axis=0))[0]
    assert (gold.min(), gold.max() + 1) == (ax - 21, ax - 3) and full == 81
