"""Synthetic battle frames for the tower-skin and arena tests (not a test module).

The owner picks a random tower skin every battle, clanmates use many, and the arena changes
with the season and the game mode (docs: the skin audit's fix list, section 2). Real frames are
never committed, so the tests draw their own: this module renders them with numpy alone
(Pillow, when installed, only draws the HP digits), at the 1080x1920 reference and resized to
the owner's 747x1328.

Two layers:

* The plain label of the first tower-skin tests (tests/pipeline/test_tower_labels.py, moved
  here): floor(), draw_label(), skin_frame(), det(), label_det(), body_det(). A tan floor, a
  gold badge block and the bar's fill and track, drawn straight at the image's size (the
  747x1328 frame by default, exactly as those tests drew it).
* The grid of the skin tests: battle_frame() renders a frame at 1080x1920 from
    - a floor (FLOORS): default, sand, snow, ice, red, forest (a magenta river), sand_river
      (a cyan river with brown banks);
    - tower bodies, or ruins (a body with no label);
    - decorations in a label's area (DECORATIONS): the red comb in the bar band, a white head
      behind the digits, a gold roof touching the badge, a 40-px gold flower bed or orange pad
      at the badge column;
    - each princess tower's HP label at (du, dv) px from its reference (Label): the gold crown
      badge with its level number, the 119-px bar in the side's colour with a dark outline and
      the drained track at any fraction, the HP digits in the tower tints with a dark outline;
    - the kings' labels (KingLabel): the level badge at full HP, badge + bar + digits damaged
      (estimated geometry: KING_*);
    - flying units over a label (flyer_box, the detector's box with det());
    - the deploy overlay (deploy_overlay: an alpha 0.45 blend toward (254, 80, 66) of the floor
      of the enemy half and the river, under the towers and the labels; measured);
    - the bottom HUD panel with the hand's cards and the elixir bar (draw_hud), so the screen
      cues read a battle;
  and end_screen() renders an end screen: banners, the crown cushions with and without crowns,
  the arena behind them, gold decoration behind the slots on request.
  battle_frame(size=OWNER_SIZE) is the same frame at 747x1328: everything but the HP digits is
  drawn at 1080x1920 and area-averaged down (resize(), like a downscaled capture: blended
  edges); the digits, the flying units, the overlay and the HUD panel are then drawn at the
  frame's own size, the digits crisp like a device's render (thin glyph strokes do not survive
  the area average). digit_store(size) builds tower digit templates from the same glyphs at
  the same size. at_sizes() resizes any 1080x1920 frame to both sizes.

Every pixel constant is at 1080x1920 (REF_SIZE) and measured on the owner's frames unless it
says "estimated": the 2026-10-31 chicken frames (labels, badges, digits) and the 2026-10-29
forest frames (floors, river, end screen). The skin tests parametrise over FLOOR_NAMES,
DECORATIONS and SIZES.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import geometry as G  # noqa: E402
from scripts.vision import hud as H  # noqa: E402
from scripts.vision import templates as TP  # noqa: E402
from scripts.vision import tower_labels as TL  # noqa: E402

RGB = tuple  # (r, g, b)

REF_SIZE = (1080, 1920)
OWNER_SIZE = (747, 1328)                   # the owner's frame size
SIZES = (REF_SIZE, OWNER_SIZE)

# ---------------------------------------------------------------------------
# The plain label (test_tower_labels.py's helpers, moved here unchanged at 747x1328)
# ---------------------------------------------------------------------------
W, HT = OWNER_SIZE
SX, SY = W / 1080.0, HT / 1920.0
FLOOR = (214, 196, 160)                    # tan: no bar colour, not gold
GOLD = (230, 190, 40)                      # the crown badge
FILL = {"enemy": (225, 36, 100), "own": (113, 208, 253)}
TRACK = {"enemy": (96, 45, 60), "own": (69, 77, 113)}
#: the bar's fill rows from the label box centre, px at 1080 (the owner's 10-31 frames:
#: enemy fill rows 169-180 under a label centred at 170.5 there, ours 822-834 under 830.6)
FILL_ROWS = {"enemy": (-3.0, 14.0), "own": (-12.0, 5.0)}
BADGE_TALL = 30.0                          # px at 1080 (rows 164-184 there)
SKIN_DV = -36.0                            # the chicken skin's label, px at 1080
CENTRES = [(x / 1080, y / 1920) for x, y in G.DEFAULT_CARD_SLOTS_PX]
REGIONS = H.HudRegions.builtin(hand_centres=CENTRES)
ENEMY = ("enemy_left", "enemy_right")
OWN = ("own_left", "own_right")
#: the plain badge's columns from the bar anchor: [anchor - 21, anchor - 3) px at 747, here at 1080
_PLAIN_BADGE_COLS = (21 * 1080 / 747, 3 * 1080 / 747)


def _size_of(img: np.ndarray) -> tuple[int, int]:
    return int(img.shape[1]), int(img.shape[0])


def floor(size: Sequence[int] = OWNER_SIZE, colour: RGB = FLOOR) -> np.ndarray:
    img = np.empty((int(size[1]), int(size[0]), 3), dtype=np.uint8)
    img[:] = colour
    return img


def bar_geometry(key: str, size: Sequence[int] = OWNER_SIZE) -> tuple[int, int]:
    """(anchor x, full px) of `key`'s bar on a frame of `size`, as TowerReader.read_bars sees it."""
    x0, _, x1, _ = REGIONS.px(f"bar_{key}", size)
    margin = max(1, int(round(8 * size[0] / 1080.0)))
    return x0 + margin, (x1 - x0) - 2 * margin


def label_centre_v(key: str, dv: float = 0.0) -> float:
    """The label box's centre row, px at 1080."""
    x0, y0, x1, y1 = TL.LABEL_REF_PX[key]
    return (y0 + y1) / 2.0 + dv


def rows(v0: float, v1: float, size: Sequence[int] = OWNER_SIZE) -> slice:
    sy = size[1] / 1920.0
    return slice(int(round(v0 * sy)), int(round(v1 * sy)))


def draw_label(img: np.ndarray, key: str, dv: float = 0.0, *, frac: float = 1.0, badge: bool = True,
               bar: bool = True, badge_tall: float = BADGE_TALL) -> np.ndarray:
    """A princess tower's plain HP label (crown badge + bar) with its box centre `dv` px (at
    1080) from the reference; the bar `frac` full. Drawn at the image's own size."""
    size = _size_of(img)
    side = TL.side_of(key)
    cv = label_centre_v(key, dv)
    ax, full = bar_geometry(key, size)
    if badge:
        bc = cv + TL.BADGE_DV_PX[side]
        sx = size[0] / 1080.0
        c0, c1 = ax - int(round(_PLAIN_BADGE_COLS[0] * sx)), ax - int(round(_PLAIN_BADGE_COLS[1] * sx))
        img[rows(bc - badge_tall / 2, bc + badge_tall / 2, size), c0:c1] = GOLD
    if bar:
        r = rows(cv + FILL_ROWS[side][0], cv + FILL_ROWS[side][1], size)
        end = ax + int(round(frac * full))
        img[r, ax:end] = FILL[side]
        img[r, end:ax + full] = TRACK[side]
    return img


def skin_frame(size: Sequence[int] = OWNER_SIZE) -> np.ndarray:
    """Both enemy labels raised by SKIN_DV, our labels where calibrated."""
    img = floor(size)
    for k in ENEMY:
        draw_label(img, k, SKIN_DV)
    for k in OWN:
        draw_label(img, k)
    return img


def det(cls: str, box_1080, conf: float = 0.9, size: Sequence[int] = OWNER_SIZE,
        side_prob: float = 0.5) -> SimpleNamespace:
    """A detector box shaped like FakeDetector's: `box_1080` (px at 1080) scaled to `size`."""
    sx, sy = size[0] / 1080.0, size[1] / 1920.0
    x0, y0, x1, y1 = box_1080
    return SimpleNamespace(cls=cls, box=(x0 * sx, y0 * sy, x1 * sx, y1 * sy), conf=conf, side_prob=side_prob)


def label_det(key: str, dv: float = 0.0, du: float = 0.0, *, tall: float = 1.0, wide: float = 1.0,
              cls: str = "tower-bar", size: Sequence[int] = OWNER_SIZE):
    x0, y0, x1, y1 = TL.LABEL_REF_PX[key]
    cx, cy = (x0 + x1) / 2.0 + du, (y0 + y1) / 2.0 + dv
    hw, hh = (x1 - x0) / 2.0 * wide, (y1 - y0) / 2.0 * tall
    return det(cls, (cx - hw, cy - hh, cx + hw, cy + hh), size=size)


def body_det(key: str, cls: str = "PrincessTower", size: Sequence[int] = OWNER_SIZE):
    return det(cls, TL.TOWER_BODY_REF_PX[key], size=size)


# ---------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Floor:
    """An arena floor: its colour, and a river (with banks) across the middle, or none."""
    name: str
    colour: RGB
    river: RGB | None = None
    bank: RGB | None = None


FLOORS: dict[str, Floor] = {
    "default": Floor("default", (250, 245, 210)),
    "sand": Floor("sand", (246, 227, 170)),            # the 10-31 sand arena: (246, 222, 165) there
    "snow": Floor("snow", (235, 242, 250)),            # passes the enemy digit tint
    "ice": Floor("ice", (200, 220, 240)),              # passes our digit tint
    "red": Floor("red", (200, 60, 60)),                # passes the enemy bar colour (deep red)
    "forest": Floor("forest", (247, 245, 206), river=(150, 60, 170)),   # Round 0: magenta river
    "sand_river": Floor("sand_river", (246, 227, 170), river=(84, 223, 244), bank=(166, 122, 75)),
}
FLOOR_NAMES = tuple(FLOORS)

#: The arena at 1080x1920 (scripts/vision/geometry.py's default homography: x 0-18000 mt,
#: y 0-32000 mt; the river 15000-17000 mt, bridges at x 2500-4500 and 13500-15500 mt).
_HX, _HY = G.DEFAULT_HOMOGRAPHY_PX[0], G.DEFAULT_HOMOGRAPHY_PX[1]


def _wx(x_mt: float) -> float:
    return _HX[0] * x_mt + _HX[2]


def _wy(y_mt: float) -> float:
    return _HY[1] * y_mt + _HY[2]


ARENA_PX = (_wx(0), _wy(0), _wx(18000), _wy(32000))
RIVER_PX = (_wy(G.RIVER_Y_MT[0]), _wy(G.RIVER_Y_MT[1]))
BRIDGES_PX = tuple((_wx(a), _wx(b)) for a, b in G.BRIDGES_X_MT)
BANK_PX = 12.0                             # a brown bank's height above and below the water
GROUT_PX = (51.0, 41.4)                    # one tile across and down
#: The deploy overlay: its box (the enemy half and the river) and its blend, measured on the
#: 10-29 review tiles F1, F5, F6: floor (241, 237, 199) -> (247, 167, 140) and (249, 246, 209) ->
#: (250, 172, 146), alpha 0.45 toward (254, 80, 66). It tints the floor only: the towers, the
#: king and the HP labels keep their colours. (The fix list's (250, 160, 120) is near the tinted
#: floor, not the blend's target; the tinted floor passes hpbars.enemy_mask and
#: vision_play.red_mask, as its finding 7 says.)
ENEMY_HALF_PX = (ARENA_PX[0], ARENA_PX[1], ARENA_PX[2], RIVER_PX[1])
OVERLAY = (254, 80, 66)
OVERLAY_ALPHA = 0.45


def _span(a0: float, a1: float, s: float, n: int) -> slice:
    """[a0, a1) px at the reference scaled by s, clipped to [0, n)."""
    i0 = max(0, min(n, int(round(a0 * s))))
    return slice(i0, max(i0, min(n, int(round(a1 * s)))))


def _scale(img: np.ndarray) -> tuple[float, float]:
    return img.shape[1] / 1080.0, img.shape[0] / 1920.0


def fill_box(img: np.ndarray, box: Sequence[float], colour: RGB) -> np.ndarray:
    """Paint box (x0, y0, x1, y1), px at 1080x1920, at the image's scale."""
    sx, sy = _scale(img)
    h, w = img.shape[:2]
    img[_span(box[1], box[3], sy, h), _span(box[0], box[2], sx, w)] = colour
    return img


def paint_floor(img: np.ndarray, fl: Floor | str = "default", *, grout: bool = False) -> np.ndarray:
    """The whole frame in the floor's colour, its river across the arena (the bridges in the
    floor's colour), tile grout lines 8% darker on request."""
    fl = FLOORS[fl] if isinstance(fl, str) else fl
    img[:] = fl.colour
    if grout:
        sx, sy = _scale(img)
        dark = tuple(int(c * 0.92) for c in fl.colour)
        x = ARENA_PX[0]
        while x <= ARENA_PX[2]:
            fill_box(img, (x - 1, ARENA_PX[1], x + 1, ARENA_PX[3]), dark)
            x += GROUT_PX[0]
        y = ARENA_PX[1]
        while y <= ARENA_PX[3]:
            fill_box(img, (ARENA_PX[0], y - 1, ARENA_PX[2], y + 1), dark)
            y += GROUT_PX[1]
    if fl.river is not None:
        r0, r1 = RIVER_PX
        if fl.bank is not None:
            fill_box(img, (0, r0 - BANK_PX, 1080, r1 + BANK_PX), fl.bank)
        fill_box(img, (0, r0, 1080, r1), fl.river)
        for b0, b1 in BRIDGES_PX:
            fill_box(img, (b0, r0 - BANK_PX, b1, r1 + BANK_PX), fl.colour)
    return img


def new_frame(fl: Floor | str = "default", size: Sequence[int] = REF_SIZE, *, grout: bool = False) -> np.ndarray:
    return paint_floor(floor(size, (0, 0, 0)), fl, grout=grout)


# ---------------------------------------------------------------------------
# Glyphs
# ---------------------------------------------------------------------------
_SEG = {"a": (slice(0, 1), slice(0, 5)), "b": (slice(0, 5), slice(4, 5)), "c": (slice(4, 9), slice(4, 5)),
        "d": (slice(8, 9), slice(0, 5)), "e": (slice(4, 9), slice(0, 1)), "f": (slice(0, 5), slice(0, 1)),
        "g": (slice(4, 5), slice(0, 5))}
_DIGIT_SEGS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd",
               "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg"}


def seg_glyph(ch: str, cell: int = 2) -> np.ndarray:
    """A 9x5-cell seven-segment digit as a bool mask (cell 2: 18 px tall), as test_hud draws them."""
    g = np.zeros((9, 5), dtype=bool)
    for s in _DIGIT_SEGS[ch]:
        g[_SEG[s]] = True
    return np.kron(g, np.ones((cell, cell), dtype=bool)).astype(bool)


@lru_cache(maxsize=1)
def pil_glyph_fn() -> Callable[[str], np.ndarray] | None:
    """Digits rendered by Pillow's default font at size 28 (19 px tall), as test_hud._pil_glyph_fn
    makes them; None without Pillow 10.1+ or when this font renders a digit in pieces."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.load_default(size=28)
    except Exception:
        return None

    def glyph(ch: str) -> np.ndarray:
        im = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(im).text((16, 16), ch, fill=255, font=font)
        a = np.asarray(im) >= 128
        ys, xs = np.nonzero(a)
        return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()

    for ch in "0123456789":
        g = glyph(ch)
        if len(TP.segment_glyphs(g)) != 1 or g.shape[0] < 14:
            return None
    return glyph


def glyph_fn() -> Callable[[str], np.ndarray]:
    """The HP digits' glyphs: Pillow's (pil_glyph_fn), else seven-segment ones 18 px tall. The
    templates (digit_store) read Pillow's at both sizes; the seven-segment ones only at 1080x1920
    (at 747x1328 their 0, 6 and 8 come out alike): a test reading digits at 747 needs Pillow."""
    return pil_glyph_fn() or seg_glyph


def scale_mask(mask: np.ndarray, s: float) -> np.ndarray:
    """A bool mask scaled by `s` (nearest neighbour), at least 1x1."""
    m = np.asarray(mask, dtype=bool)
    if s == 1.0:
        return m
    h, w = max(1, int(round(m.shape[0] * s))), max(1, int(round(m.shape[1] * s)))
    ys = np.minimum(((np.arange(h) + 0.5) / s).astype(int), m.shape[0] - 1)
    xs = np.minimum(((np.arange(w) + 0.5) / s).astype(int), m.shape[1] - 1)
    return m[ys][:, xs]


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """A bool mask grown by `r` px (a square)."""
    m = np.asarray(mask, dtype=bool)
    if r <= 0:
        return m.copy()
    p = np.pad(m, r)
    out = np.zeros_like(p)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(p, dy, axis=0), dx, axis=1)
    return out[r:-r, r:-r]


def text_mask(text: str, glyph: Callable[[str], np.ndarray] | None = None, gap: int = 3,
              bold: int = 0) -> np.ndarray:
    """`text`'s glyphs top-aligned side by side, `gap` px apart, as a bool mask; each glyph grown
    by `bold` px first (the tower digits' font is bold: ~3 px strokes at 747x1328)."""
    glyph = glyph or glyph_fn()
    gs = [dilate(np.pad(glyph(c), bold), bold) if bold else glyph(c) for c in str(text)]
    h = max(g.shape[0] for g in gs)
    w = sum(g.shape[1] for g in gs) + gap * (len(gs) - 1)
    out = np.zeros((h, w), dtype=bool)
    x = 0
    for g in gs:
        out[:g.shape[0], x:x + g.shape[1]] = g
        x += g.shape[1] + gap
    return out


def stamp(img: np.ndarray, mask: np.ndarray, x: float, y: float, colour: RGB, *,
          outline: RGB | None = None, outline_px: int = 2) -> tuple[int, int, int, int]:
    """Paint a bool mask drawn at 1080x1920 with its top-left at (x, y) px there, scaled to the
    image, with an `outline_px` dark outline under it. Returns its box (px at 1080)."""
    sx, sy = _scale(img)
    m = scale_mask(mask, sx)                 # 9:16 frames: one scale for both axes
    pad = int(round(outline_px * sx)) if outline is not None else 0
    big = np.pad(m, pad)
    ring = dilate(big, pad) if pad else big
    h, w = img.shape[:2]
    x0, y0 = int(round(x * sx)) - pad, int(round(y * sy)) - pad
    ys, xs = slice(max(0, y0), min(h, y0 + big.shape[0])), slice(max(0, x0), min(w, x0 + big.shape[1]))
    if ys.stop > ys.start and xs.stop > xs.start:
        cy, cx = slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0)
        region = img[ys, xs]
        if outline is not None:
            region[ring[cy, cx]] = outline
        region[big[cy, cx]] = colour
    return (float(x), float(y), float(x) + m.shape[1] / sx, float(y) + m.shape[0] / sy)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
PRINCESS_HP = H.LEVEL11_TOWER_MAX_HP["enemy_left"]      # 3052
KING_HP = H.LEVEL11_TOWER_MAX_HP["enemy_king"]          # 4824
BAR_FULL_PX = H.BAR_FULL_PX                              # 119
BADGE_GOLD = (232, 188, 44)
BADGE_DARK = (70, 42, 8)                                 # the badge's outline and level number
BADGE_WIDE_PX = (-35.0, -2.0)                            # badge columns from the bar anchor
BADGE_BOX_TALL = 32.0                                    # outline included (gold rows 163-184 at 747)
BAR_OUTLINE = {"enemy": (40, 18, 30), "own": (18, 30, 62)}
OUTLINE_PX = 2.0
#: The HP digits: tint (passes templates.TOWER_GLYPH_STYLE's side tint), outline, the glyphs'
#: top row from the label centre and their first column from the bar anchor (the 10-31 frames:
#: enemy digits rows 156-170 there over a label centred at 170.5, ours over the bar's lower half).
DIGIT = {"enemy": (255, 222, 236), "own": (225, 240, 255)}
DIGIT_OUTLINE = {"enemy": (60, 20, 40), "own": (20, 34, 70)}
DIGIT_TOP_DV = {"enemy": -22.0, "own": -3.0}
DIGIT_DU = 8.0
DIGIT_BOLD = 1                                           # px each glyph is grown (19 -> 21 px tall)
#: The kings (estimated: a full king shows only its level badge, 50x43 px at 1080 centred over the
#: king, rows 30-61 / 992-1021 at 747; a damaged king's badge, bar and digits are laid out like a
#: princess label around the king's x, its bar about 1.4x a princess bar as on the end screen).
KING_X = 540.0
KING_BADGE_ROW = {"enemy": 66.0, "own": 1455.0}
KING_BADGE_PX = (50.0, 43.0)
KING_BAR_PX = 166.0


def anchor_x(key: str, du: float = 0.0) -> float:
    """The bar fill's start of a princess tower, px at 1080 (its bar region + the margin)."""
    return REGIONS.get(f"bar_{key}").u0 * 1080.0 + H.BAR_MARGIN_PX + du


@dataclass(frozen=True)
class Label:
    """A princess tower's HP label `du`, `dv` px (at 1080) from its reference. hp None: the
    digits show round(frac * max_hp); digits False hides them (a frame between digit redraws)."""
    du: float = 0.0
    dv: float = 0.0
    frac: float = 1.0
    hp: int | None = None
    level: int | None = 11
    badge: bool = True
    bar: bool = True
    digits: bool = True
    max_hp: int = PRINCESS_HP

    def shown_hp(self) -> int:
        return int(self.hp) if self.hp is not None else int(round(self.frac * self.max_hp))


@dataclass(frozen=True)
class KingLabel:
    """A king's label: frac None, the full king's level badge alone; else damaged (badge, bar,
    digits)."""
    du: float = 0.0
    dv: float = 0.0
    frac: float | None = None
    hp: int | None = None
    level: int | None = 11
    max_hp: int = KING_HP

    def shown_hp(self) -> int:
        if self.hp is not None:
            return int(self.hp)
        return int(round((1.0 if self.frac is None else float(self.frac)) * self.max_hp))


def label_parts(key: str, du: float = 0.0, dv: float = 0.0) -> dict[str, tuple[float, float, float, float]]:
    """Where a princess label's parts are drawn, boxes (x0, y0, x1, y1) px at 1080:
    "badge" (outline included), "bar" (the fill track, BAR_FULL_PX wide), "outline" (the bar's
    outline box), "digits" (the HP digits' first column and top row, x1/y1 approximate)."""
    side = TL.side_of(key)
    cv = label_centre_v(key, dv)
    ax = anchor_x(key, du)
    f0, f1 = cv + FILL_ROWS[side][0], cv + FILL_ROWS[side][1]
    bc = cv + TL.BADGE_DV_PX[side]
    return {"badge": (ax + BADGE_WIDE_PX[0], bc - BADGE_BOX_TALL / 2, ax + BADGE_WIDE_PX[1], bc + BADGE_BOX_TALL / 2),
            "bar": (ax, f0, ax + BAR_FULL_PX, f1),
            "outline": (ax - OUTLINE_PX, f0 - OUTLINE_PX, ax + BAR_FULL_PX + OUTLINE_PX, f1 + OUTLINE_PX),
            "digits": (ax + DIGIT_DU, cv + DIGIT_TOP_DV[side], ax + DIGIT_DU + 70.0, cv + DIGIT_TOP_DV[side] + 21.0)}


def draw_badge(img: np.ndarray, box: Sequence[float], level: int | None = 11, *,
               gold: RGB = BADGE_GOLD, dark: RGB = BADGE_DARK) -> np.ndarray:
    """The crown badge in `box` (px at 1080): a dark outline, three prongs over the top quarter,
    gold inside and the level number in dark seven-segment digits."""
    x0, y0, x1, y1 = (float(c) for c in box)
    w, h = x1 - x0, y1 - y0
    prong = h * 0.25
    o = 2.0
    for p0, p1 in ((x0, x0 + w * 0.28), (x0 + w * 0.36, x1 - w * 0.36), (x1 - w * 0.28, x1)):
        fill_box(img, (p0, y0, p1, y0 + prong + o), dark)
        fill_box(img, (p0 + o, y0 + o, p1 - o, y0 + prong + o), gold)
    fill_box(img, (x0, y0 + prong, x1, y1), dark)
    fill_box(img, (x0 + o, y0 + prong, x1 - o, y1 - o), gold)
    if level is not None:
        text = text_mask(str(int(level)), seg_glyph, gap=2)
        th = min(18.0, h - prong - 2 * o - 2)
        s = th / text.shape[0]
        m = scale_mask(text, s)
        tx = (x0 + x1) / 2.0 - m.shape[1] / 2.0
        ty = y0 + prong + (h - prong - o - m.shape[0]) / 2.0
        stamp(img, m, tx, ty, dark)
    return img


def draw_bar(img: np.ndarray, box: Sequence[float], frac: float, side: str, *,
             fill: RGB | None = None, track: RGB | None = None, outline: RGB | None = None) -> np.ndarray:
    """An HP bar: the fill track `box` (px at 1080) `frac` full in the side's colour, the drained
    part in its dark track colour, a dark outline OUTLINE_PX around."""
    x0, y0, x1, y1 = (float(c) for c in box)
    fill_box(img, (x0 - OUTLINE_PX, y0 - OUTLINE_PX, x1 + OUTLINE_PX, y1 + OUTLINE_PX),
             outline or BAR_OUTLINE[side])
    end = x0 + max(0.0, min(1.0, float(frac))) * (x1 - x0)
    fill_box(img, (x0, y0, x1, y1), track or TRACK[side])
    if end > x0:
        fill_box(img, (x0, y0, end, y1), fill or FILL[side])
    return img


def draw_digits(img: np.ndarray, text: str, x: float, y: float, side: str, *,
                glyph: Callable[[str], np.ndarray] | None = None) -> tuple[float, float, float, float]:
    """HP digits `text` with their top-left at (x, y) px at 1080, in the side's tint (glyphs
    grown by DIGIT_BOLD px) with a dark outline. Returns their box (px at 1080)."""
    return stamp(img, text_mask(text, glyph, bold=DIGIT_BOLD), x, y, DIGIT[side], outline=DIGIT_OUTLINE[side])


def draw_ui_label(img: np.ndarray, key: str, label: Label | None = None, *,
                  glyph: Callable[[str], np.ndarray] | None = None) -> np.ndarray:
    """A princess tower's full HP label (Label) at the image's scale: badge, bar, digits."""
    lab = label or Label()
    side = TL.side_of(key)
    parts = label_parts(key, lab.du, lab.dv)
    if lab.bar:
        draw_bar(img, parts["bar"], lab.frac, side)
    if lab.badge:
        draw_badge(img, parts["badge"], lab.level)
    if lab.digits:
        x, y = parts["digits"][:2]
        draw_digits(img, str(lab.shown_hp()), x, y, side, glyph=glyph)
    return img


def king_parts(key: str, label: KingLabel | None = None) -> dict[str, tuple[float, float, float, float]]:
    """Where a king's label parts are drawn (px at 1080): "badge" always; "bar" and "digits" when
    damaged (frac not None)."""
    lab = label or KingLabel()
    side = TL.side_of(key)
    bw, bh = KING_BADGE_PX
    row = KING_BADGE_ROW[side] + lab.dv
    x = KING_X + lab.du
    if lab.frac is None:
        return {"badge": (x - bw / 2, row - bh / 2, x + bw / 2, row + bh / 2)}
    left = x - (bw + 4.0 + KING_BAR_PX) / 2.0
    ax = left + bw + 4.0
    f0, f1 = row + FILL_ROWS["enemy"][0], row + FILL_ROWS["enemy"][1]
    return {"badge": (left, row - bh / 2, left + bw, row + bh / 2), "bar": (ax, f0, ax + KING_BAR_PX, f1),
            "digits": (ax + DIGIT_DU, row - 22.0, ax + DIGIT_DU + 70.0, row - 1.0)}


def draw_king_label(img: np.ndarray, key: str, label: KingLabel | None = None, *,
                    glyph: Callable[[str], np.ndarray] | None = None, digits: bool = True) -> np.ndarray:
    """A king's label (KingLabel) at the image's scale; digits False leaves a damaged king's
    digits out (battle_frame draws them after the resize)."""
    lab = label or KingLabel()
    side = TL.side_of(key)
    parts = king_parts(key, lab)
    if "bar" in parts:
        draw_bar(img, parts["bar"], float(lab.frac), side)
        if digits:
            draw_digits(img, str(lab.shown_hp()), parts["digits"][0], parts["digits"][1], side, glyph=glyph)
    draw_badge(img, parts["badge"], lab.level)
    return img


# ---------------------------------------------------------------------------
# Towers, decorations, units
# ---------------------------------------------------------------------------
TOWER_COLOUR = {"enemy": (190, 150, 140), "own": (150, 160, 190)}    # no bar colour, not gold
RUIN = (122, 110, 98)
DECORATIONS = ("comb", "head", "roof", "flowers", "pad")
OVER_LABEL = ("roof",)                     # decorations battle_frame draws over the labels
COMB = (215, 40, 45)                       # a chicken's red comb: passes the enemy bar colour
HEAD = (250, 250, 250)                     # a white head: passes the enemy digit tint
PAD = (240, 170, 60)                       # the sand arena's orange pad: passes the badge gold
FLYER = (95, 70, 50)


def draw_tower(img: np.ndarray, key: str, kind: str = "tower", colour: RGB | None = None) -> np.ndarray:
    """A tower body at its reference box (tower_labels.TOWER_BODY_REF_PX), or kind "ruin": rubble
    over the lower 40% of that box (a destroyed tower: a body with no label)."""
    x0, y0, x1, y1 = TL.TOWER_BODY_REF_PX[key]
    if kind == "ruin":
        top = y1 - 0.4 * (y1 - y0)
        fill_box(img, (x0 + 10, top, x1 - 10, y1), colour or RUIN)
        fill_box(img, (x0 + 30, top - 14, x0 + 70, top), colour or RUIN)
    elif kind == "tower":
        fill_box(img, (x0, y0 + 20, x1, y1), colour or TOWER_COLOUR[TL.side_of(key)])
    else:
        raise ValueError(f"kind is 'tower' or 'ruin', not {kind!r}")
    return img


def decoration_box(key: str, kind: str, du: float = 0.0, dv: float = 0.0) -> tuple[float, float, float, float]:
    """Where decoration `kind` of a label moved (du, dv) px from the reference is drawn (px at 1080).

    comb     the bar band's rows, from 4 px left of the anchor over 34 px (a fixed box over it
             reads a ~0.25 bar);
    head     behind the digits (white over the digit rows and around them);
    roof     gold, 14 px tall over the badge, wider than it, drawn over the badge's top outline
             (battle_frame draws it after the labels: OVER_LABEL) so the skin's gold meets the
             badge's (one gold group of about 46 px: too tall for a badge);
    flowers  a 40-px gold bed at the badge columns (gold stripes 3 px of every 6), centred on the
             badge row;
    pad      the same as a solid orange pad."""
    side = TL.side_of(key)
    parts = label_parts(key, du, dv)
    ax = anchor_x(key, du)
    cv = label_centre_v(key, dv)
    bx0, by0, bx1, by1 = parts["badge"]
    if kind == "comb":
        f0, f1 = cv + FILL_ROWS[side][0], cv + FILL_ROWS[side][1]
        return (ax - 4.0, f0 + 1.0, ax + 30.0, f1 + 4.0)
    if kind == "head":
        d0 = cv + DIGIT_TOP_DV[side]
        return (ax - 6.0, d0 - 8.0, ax + 96.0, d0 + 27.0)
    if kind == "roof":
        return (bx0 - 10.0, by0 - 14.0, ax + 20.0, by0 + 2.5)
    if kind in ("flowers", "pad"):
        mid = (by0 + by1) / 2.0
        return (bx0 - 4.0, mid - 20.0, bx1 + 4.0, mid + 20.0)
    raise ValueError(f"unknown decoration {kind!r} (one of {DECORATIONS})")


def draw_decoration(img: np.ndarray, key: str, kind: str, du: float = 0.0, dv: float = 0.0) -> np.ndarray:
    box = decoration_box(key, kind, du, dv)
    if kind == "comb":
        return fill_box(img, box, COMB)
    if kind == "head":
        return fill_box(img, box, HEAD)
    if kind in ("roof",):
        return fill_box(img, box, BADGE_GOLD)
    if kind == "pad":
        return fill_box(img, box, PAD)
    x0, y0, x1, y1 = box                         # flowers: gold stripes 3 px of every 6
    x = x0
    while x < x1:
        fill_box(img, (x, y0, min(x1, x + 3.0), y1), BADGE_GOLD)
        x += 6.0
    return img


def flyer_box(key: str, du: float = 0.0, dv: float = 0.0) -> tuple[float, float, float, float]:
    """A flying unit's box over a princess label moved (du, dv) (px at 1080): badge, bar and digits."""
    b = label_parts(key, du, dv)
    x0 = b["badge"][0] - 6.0
    y0 = min(b["badge"][1], b["digits"][1]) - 8.0
    return (x0, y0, b["bar"][2] + 6.0, max(b["badge"][3], b["outline"][3]) + 8.0)


def draw_flyer(img: np.ndarray, box: Sequence[float], colour: RGB = FLYER) -> np.ndarray:
    """A unit's body filling an ellipse in `box` (px at 1080)."""
    sx, sy = _scale(img)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (float(c) for c in box)
    ys, xs = _span(y0, y1, sy, h), _span(x0, x1, sx, w)
    yy, xx = np.mgrid[ys, xs]
    cx, cy = (x0 + x1) / 2.0 * sx, (y0 + y1) / 2.0 * sy
    rx, ry = max(1.0, (x1 - x0) / 2.0 * sx), max(1.0, (y1 - y0) / 2.0 * sy)
    inside = ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1.0
    img[ys, xs][inside] = colour                 # a basic slice: a view of img
    return img


def deploy_overlay(img: np.ndarray, alpha: float = OVERLAY_ALPHA, colour: RGB = OVERLAY,
                   box: Sequence[float] = ENEMY_HALF_PX) -> np.ndarray:
    """The card-selection overlay: every pixel in `box` (px at 1080) blended `alpha` toward `colour`
    (battle_frame applies it to the floor before the towers and labels are drawn)."""
    sx, sy = _scale(img)
    h, w = img.shape[:2]
    ys, xs = _span(box[1], box[3], sy, h), _span(box[0], box[2], sx, w)
    a = img[ys, xs].astype(np.float64)
    img[ys, xs] = np.clip(np.rint(a * (1.0 - alpha) + np.asarray(colour, dtype=np.float64) * alpha),
                          0, 255).astype(np.uint8)
    return img


PANEL = (40, 100, 200)                     # the bottom HUD panel: vision_play.blue_mask
PANEL_TOP_V = 0.805
CARD_ART = (150, 140, 130)                 # the hand's cards: no blue (the OK box reads 12-13% there)
CARD_PX = (170.0, 215.0)
ELIXIR_LIT = (204, 35, 212)
ELIXIR_TRACK = (5, 57, 125)


def draw_hud(img: np.ndarray, elixir: float = 5.0) -> np.ndarray:
    """The bottom HUD panel (vision_play.PANEL_BAND reads it), the four cards of the hand, and the
    elixir bar `elixir` / 10 lit."""
    fill_box(img, (0, PANEL_TOP_V * 1920, 1080, 1920), PANEL)
    for x, y in G.DEFAULT_CARD_SLOTS_PX:
        fill_box(img, (x - CARD_PX[0] / 2, y - CARD_PX[1] / 2, x + CARD_PX[0] / 2, y + CARD_PX[1] / 2 - 40),
                 CARD_ART)
    (x0, y0, x1, y1), _ = H.BUILTIN_REGIONS_PX["elixir_bar"]
    fill_box(img, (x0, y0, x1, y1), ELIXIR_TRACK)
    lit = x0 + max(0.0, min(10.0, float(elixir))) / 10.0 * (x1 - x0)
    if lit > x0:
        fill_box(img, (x0, y0, lit, y1), ELIXIR_LIT)
    return img


# ---------------------------------------------------------------------------
# Whole frames
# ---------------------------------------------------------------------------
def resize(img: np.ndarray, size: Sequence[int]) -> np.ndarray:
    """An area-average resize to (w, h): every output pixel the mean of the input area it covers."""
    w, h = int(size[0]), int(size[1])
    if (img.shape[1], img.shape[0]) == (w, h):
        return np.asarray(img).copy()
    a = _area_axis(np.asarray(img), h, 0)                # uint8 in, float32 out
    a = _area_axis(a, w, 1)
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def _area_axis(a: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Area-average `a` along `axis` to n_out cells: output cell i covers [i s, (i + 1) s) of the
    input, s = n_in / n_out, each input cell weighted by its overlap (a downscale: <= ceil(s) + 1
    input cells per output cell)."""
    n_in = a.shape[axis]
    s = n_in / float(n_out)
    lo = np.arange(n_out) * s
    hi = lo + s
    first = np.floor(lo).astype(int)
    out = None
    for j in range(int(np.ceil(s)) + 1):
        k = first + j
        wgt = (np.minimum(hi, k + 1) - np.maximum(lo, k)).clip(0.0, None) / s
        part = np.take(a, np.minimum(k, n_in - 1), axis=axis)
        shape = [1] * a.ndim
        shape[axis] = n_out
        part = part * wgt.reshape(shape).astype(np.float32)
        out = part if out is None else out + part
    return out


def at_sizes(img: np.ndarray, sizes: Iterable[Sequence[int]] = SIZES) -> dict[tuple[int, int], np.ndarray]:
    """The frame (drawn at 1080x1920) at each size: itself, and resized."""
    return {(int(s[0]), int(s[1])): resize(img, s) for s in sizes}


def default_labels() -> dict[str, Label]:
    return {k: Label() for k in TL.PRINCESS_KEYS}


def battle_frame(fl: Floor | str = "default", *, labels: Mapping[str, Label | None] | None = None,
                 kings: Mapping[str, KingLabel | None] | None = None,
                 towers: Mapping[str, str | None] | None = None,
                 decorations: Iterable[tuple] = (), flyers: Iterable[Sequence[float]] = (),
                 overlay: bool | float = False, hud: bool = True, elixir: float = 5.0, grout: bool = False,
                 size: Sequence[int] = REF_SIZE, glyph: Callable[[str], np.ndarray] | None = None) -> np.ndarray:
    """A battle frame drawn at 1080x1920 and resized to `size`, in this order: the floor (and
    river), the deploy overlay (on the floor only), tower bodies, decorations, labels (badges and
    bars), the OVER_LABEL decorations; then at `size`: the labels' HP digits, flying units, the
    HUD panel.

    labels: princess key -> Label (None: no label, e.g. a ruin's); missing keys get Label()
    (default: all four full at their references). kings: king key -> KingLabel (None: none);
    missing keys get the full king's badge. towers: key -> "tower", "ruin" or None (no body);
    missing keys "tower" (a key whose label is None and whose tower is not given is a ruin).
    decorations: (key, kind) or (key, kind, du, dv) tuples (decoration_box). flyers: boxes
    (px at 1080), flyer_box(). overlay: True (alpha 0.45) or an alpha."""
    lab = dict(default_labels())
    if labels:
        lab.update(labels)
    kg: dict[str, KingLabel | None] = {k: KingLabel() for k in TL.KING_KEYS}
    if kings:
        kg.update(kings)
    tw: dict[str, str | None] = {k: ("ruin" if k in lab and lab[k] is None else "tower")
                                 for k in TL.TOWER_BODY_REF_PX}
    if towers:
        tw.update(towers)
    img = new_frame(fl, REF_SIZE, grout=grout)
    if overlay:
        deploy_overlay(img, OVERLAY_ALPHA if overlay is True else float(overlay))
    for key, kind in tw.items():
        if kind:
            draw_tower(img, key, kind)
    decorations = list(decorations)
    for d in decorations:
        if d[1] not in OVER_LABEL:
            draw_decoration(img, d[0], d[1], *(float(x) for x in d[2:4]))
    for key, label in lab.items():
        if label is not None:
            draw_ui_label(img, key, replace(label, digits=False))
    for key, label in kg.items():
        if label is not None:
            draw_king_label(img, key, label, digits=False)
    for d in decorations:
        if d[1] in OVER_LABEL:
            draw_decoration(img, d[0], d[1], *(float(x) for x in d[2:4]))
    out = resize(img, size)
    # The digits are drawn at the frame's own size, crisp like a device's render: area-averaged
    # 1080 glyphs lose their thin strokes at 747 (templates: digit_store at the same size).
    for key, label in lab.items():
        if label is not None and label.digits:
            x, y = label_parts(key, label.du, label.dv)["digits"][:2]
            draw_digits(out, str(label.shown_hp()), x, y, TL.side_of(key), glyph=glyph)
    for key, label in kg.items():
        if label is not None and label.frac is not None:
            x, y = king_parts(key, label)["digits"][:2]
            draw_digits(out, str(label.shown_hp()), x, y, TL.side_of(key), glyph=glyph)
    for box in flyers:
        draw_flyer(out, box)
    if hud:
        draw_hud(out, elixir)
    return out


# ---------------------------------------------------------------------------
# The end screen
# ---------------------------------------------------------------------------
END_RED = (200, 30, 60)
END_BLUE = (40, 100, 220)
OK_BLUE = (60, 140, 240)
CUSHION = {"enemy": (205, 40, 95), "own": (45, 110, 215)}
CROWN_GOLD = (240, 185, 45)
CROWN_BAND = (70, 200, 235)
#: Normalised (u0, v0, u1, v1) of the end screen's parts (the 2026-10-29 end screen, 747x1328):
#: the banners and the OK button around vision_play's cue boxes, the cushions (the lower part of
#: each crown slot's rows: about 45% of a slot box is not cushion) and a crown on its cushion.
END_BOXES = {"red_banner": (0.08, 0.25, 0.92, 0.33), "blue_banner": (0.08, 0.545, 0.92, 0.625),
             "ok": (0.37, 0.875, 0.63, 0.94)}
CROWN_SLOTS_U = (0.26, 0.50, 0.735)
CUSHION_ROWS_V = {"enemy": (0.195, 0.25), "own": (0.485, 0.535)}
CROWN_ROWS_V = {"enemy": (0.165, 0.235), "own": (0.445, 0.515)}
CUSHION_HALF_U = 0.075
CROWN_HALF_U = 0.08
END_DECOR = ("flowers", "king")


def _nbox(u0: float, v0: float, u1: float, v1: float) -> tuple[float, float, float, float]:
    return u0 * 1080.0, v0 * 1920.0, u1 * 1080.0, v1 * 1920.0


def draw_crown(img: np.ndarray, u: float, rows_v: Sequence[float]) -> np.ndarray:
    """A won crown centred on slot u over rows_v: gold with three prongs, a cyan band at its foot
    (about 1.25 times as wide as tall)."""
    x0, y0, x1, y1 = _nbox(u - CROWN_HALF_U, rows_v[0], u + CROWN_HALF_U, rows_v[1])
    h, w = y1 - y0, x1 - x0
    for p0, p1 in ((x0, x0 + w * 0.25), (x0 + w * 0.375, x1 - w * 0.375), (x1 - w * 0.25, x1)):
        fill_box(img, (p0, y0, p1, y0 + 0.3 * h), CROWN_GOLD)
    fill_box(img, (x0, y0 + 0.3 * h, x1, y0 + 0.65 * h), CROWN_GOLD)
    fill_box(img, (x0, y0 + 0.65 * h, x1, y1), CROWN_BAND)
    return img


def end_screen(fl: Floor | str = "forest", own: int = 0, enemy: int = 0, *, decor: Iterable[str] = (),
               size: Sequence[int] = REF_SIZE) -> np.ndarray:
    """The end screen over the arena: our `own` crowns won on our (blue, lower) cushions, the
    enemy's on theirs (red, upper), left to right; decor: "flowers" (gold stripes behind the
    slots, about 10% of their rows' pixels) and "king" (a gold king's crown behind the enemy's
    middle slot, above its cushion)."""
    img = new_frame(fl, REF_SIZE)
    for d in decor:
        if d == "flowers":
            for side in ("enemy", "own"):
                v0 = CROWN_ROWS_V[side][0] - 0.01
                x = 0.0
                while x < 1080.0:
                    fill_box(img, (x, v0 * 1920.0, x + 3.0, CUSHION_ROWS_V[side][1] * 1920.0), BADGE_GOLD)
                    x += 30.0
        elif d == "king":
            fill_box(img, _nbox(0.46, 0.17, 0.54, 0.195), BADGE_GOLD)
        else:
            raise ValueError(f"unknown end-screen decoration {d!r} (one of {END_DECOR})")
    fill_box(img, _nbox(*END_BOXES["red_banner"]), END_RED)
    fill_box(img, _nbox(*END_BOXES["blue_banner"]), END_BLUE)
    fill_box(img, _nbox(*END_BOXES["ok"]), OK_BLUE)
    for side, won in (("enemy", enemy), ("own", own)):
        for i, u in enumerate(CROWN_SLOTS_U):
            fill_box(img, _nbox(u - CUSHION_HALF_U, CUSHION_ROWS_V[side][0], u + CUSHION_HALF_U,
                                CUSHION_ROWS_V[side][1]), CUSHION[side])
            if i < int(won):
                draw_crown(img, u, CROWN_ROWS_V[side])
    return resize(img, size)


# ---------------------------------------------------------------------------
# Digit templates from the same glyphs
# ---------------------------------------------------------------------------
def digit_store(size: Sequence[int] = REF_SIZE, glyph: Callable[[str], np.ndarray] | None = None,
                side: str = "enemy") -> TP.TemplateStore:
    """In-memory tower digit templates (0-9) cut from "0123456789" drawn as battle_frame draws HP
    digits at `size` (draw_digits on a frame of that size), on a dark floor."""
    img = floor(size, (20, 20, 30))
    x0, y0, x1, y1 = draw_digits(img, "0123456789", 100.0, 300.0, side, glyph=glyph)
    sx, sy = size[0] / 1080.0, size[1] / 1920.0
    crop = img[int(y0 * sy) - 4:int(np.ceil(y1 * sy)) + 4, int(x0 * sx) - 4:int(np.ceil(x1 * sx)) + 4]
    store = TP.TemplateStore.memory()
    TP.add_digit_templates(store, "tower", crop, "0123456789")
    return store


__all__ = [
    "ARENA_PX", "BADGE_TALL", "BAR_FULL_PX", "CENTRES", "DECORATIONS", "ENEMY", "ENEMY_HALF_PX", "FILL",
    "FILL_ROWS", "FLOOR", "FLOORS", "FLOOR_NAMES", "Floor", "GOLD", "HT", "KingLabel", "Label", "OWN",
    "OWNER_SIZE", "REF_SIZE", "REGIONS", "RIVER_PX", "SIZES", "SKIN_DV", "SX", "SY", "TRACK", "W",
    "anchor_x", "at_sizes", "bar_geometry", "battle_frame", "body_det", "decoration_box", "deploy_overlay",
    "det", "digit_store", "draw_decoration", "draw_flyer", "draw_hud", "draw_king_label", "draw_label",
    "draw_tower", "draw_ui_label", "end_screen", "floor", "flyer_box", "glyph_fn", "king_parts",
    "label_centre_v", "label_det", "label_parts", "new_frame", "paint_floor", "resize", "rows",
    "seg_glyph", "skin_frame",
]
