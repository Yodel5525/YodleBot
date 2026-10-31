"""The deploy overlay: is the enemy half tinted on this frame? (skin audit P0-1 step 8 / P1-2).

While a card is selected the game tints the half of the arena where it may not be played (the
enemy half, our side's view) toward red: the 2026-10-29 review tiles F1, F5, F6 and the
2026-10-31 dry-run frame 00000420 show floor (241, 237, 199) -> (247, 167, 140). The tint passes
hpbars.enemy_mask and vision_play.red_mask, so on such a frame a tower search sees enemy bar
colour everywhere.

mirror_tint() needs no history: the arena floor is the same on both halves, so it compares small
floor patches of the enemy half with the same patches mirrored about the river in our half. The
overlay blends a pixel a share TINT_ALPHA (0.45) of the way to OVERLAY_RGB (254, 80, 66), so a
patch is tinted when its difference from its mirror, d = enemy - own, points that way: d = k *
(OVERLAY_RGB - own) with k in TINT_K (0.25-0.7) and less than TINT_RESIDUAL left over, for a
mirror at least TINT_MIN_REACH from OVERLAY_RGB (on a floor already that colour nothing shows;
a red floor still moves enough). overlay_on() is True when >= OVERLAY_SHARE of the patches are
tinted. The patches avoid the towers, the kings, every princess label's search window
(tower_calib) and the river, so units and skins cover only a few of them.

Measured (skin_fix/tint2.py on the owner's frames, 747x1328): overlay frames 0.63-0.99 of the
patches tinted, k 0.42-0.43 (the 24 review-1 tiles: F1, F4-F9, F11, F15, F16, F21, F22; DRY
00000420 0.99), every other frame <= 0.01 (sand and forest arenas, the end screen).

OverlayDetector (P1-2) is the per-battle flag HudReader uses. Per frame, an overlay frame is
any of:
  * mirror: mirror_tint() >= OVERLAY_SHARE (no history needed: from the battle's first frame);
  * median: the floor patches against their own running median (the same enemy-half patches;
    per patch the median of its last MEDIAN_KEEP frames that were not flagged, once it has
    MEDIAN_MIN of them): >= MEDIAN_SHARE (0.6) of them moved toward OVERLAY_RGB by the same
    rule as the mirror (k in TINT_K, at most TINT_RESIDUAL off the line). It needs no mirror, so
    units on our half at the mirrored spots do not hide the tint. The spec's "R rose and G and
    B fell by >= 25" is that direction on a pale floor; on a red floor G and B rise toward
    (254, 80, 66), and the blend direction still holds. A slow change of light moves every
    patch along its own colour, not toward OVERLAY_RGB (the residual), and units cover a few
    patches;
  * tap: inside the bot's own tap window (note_tap: from the card tap, OWN_TAP_PAIR_S for the
    adb shell to run the card tap and the tile tap, plus OWN_TAP_TAIL_S 0.3 s), whatever the
    pixels show.
HudReader skips the tower reads, the label calibration and the label-absence clock on flagged
frames (None, not 0), and vision_play.ScreenCues.end_screen is False on a frame mirror_tint()
flags. Pixel constants are at 1080x1920 and scale to any 9:16 frame. Imports numpy only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from . import tower_labels as TL

REF_SIZE = (1080, 1920)
#: The arena (geometry.DEFAULT_HOMOGRAPHY_PX: x 0-18000 mt, y 0-32000 mt) and the river
#: (15000-17000 mt), px at 1080x1920.
ARENA_PX = (78.0, 151.2, 996.0, 1476.0)
RIVER_PX = (772.2, 855.0)
MIRROR_Y_PX = (RIVER_PX[0] + RIVER_PX[1]) / 2.0
PATCH_PX = 8.0                 # a patch: 8x8 px
STEP_PX = (40.0, 36.0)         # the grid's step across and down
TOP_PX = 190.0                 # patches from this row (below the enemy king's row of labels)
RIVER_MARGIN_PX = 25.0         # ... to this far above the river
#: What a patch must not touch (px at 1080): every tower body and a princess label's whole
#: search window (tower_calib: du +-60, dv -120..+60, plus the badge left of the bar), the kings
#: and their labels; grown EXCLUDE_GROW_PX a side.
KING_BOXES_PX = ((440.0, 0.0, 640.0, 360.0), (440.0, 1200.0, 640.0, 1500.0))
EXCLUDE_GROW_PX = 20.0
OVERLAY_RGB = (254.0, 80.0, 66.0)   # the blend's target (the 10-29 tiles: skin_synth.OVERLAY)
TINT_ALPHA = 0.45
TINT_K = (0.25, 0.7)           # the share of the way to OVERLAY_RGB a tinted patch moved
TINT_RESIDUAL = 20.0           # RGB distance left after that move
TINT_MIN_REACH = 40.0          # a mirror this close to OVERLAY_RGB tells nothing
OVERLAY_SHARE = 0.5            # a frame is an overlay frame from this share of tinted patches
MEDIAN_SHARE = 0.6             # ... or this share tinted against their running median (P1-2)
MEDIAN_KEEP = 15               # the unflagged frames a patch's median is taken over
MEDIAN_MIN = 3                 # the frames a patch needs before the median test votes
#: The bot's own taps (vision_play): the adb shell runs "input tap" for the card and then for the
#: tile, a few hundred ms each; the overlay shows from the first to the second (plus the tail).
OWN_TAP_PAIR_S = 1.0
OWN_TAP_TAIL_S = 0.3


def _excluded() -> list[tuple[float, float, float, float]]:
    boxes = list(TL.TOWER_BODY_REF_PX.values())
    for x0, y0, x1, y1 in TL.LABEL_REF_PX.values():
        boxes.append((x0 - 100.0, y0 - 120.0, x1 + 60.0, y1 + 60.0))
    boxes += list(KING_BOXES_PX)
    g = EXCLUDE_GROW_PX
    return [(x0 - g, y0 - g, x1 + g, y1 + g) for x0, y0, x1, y1 in boxes]


def _inside(x: float, y: float, boxes) -> bool:
    return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes)


def mirror_y(y: float) -> float:
    """The row (px at 1080x1920) mirrored about the river's middle."""
    return 2.0 * MIRROR_Y_PX - float(y)


@lru_cache(maxsize=1)
def patch_centres() -> tuple[tuple[float, float], ...]:
    """The enemy-half patch centres (px at 1080x1920) whose patch and mirror are both clear."""
    ex = _excluded()
    out = []
    for y in np.arange(TOP_PX, RIVER_PX[0] - RIVER_MARGIN_PX, STEP_PX[1]):
        for x in np.arange(ARENA_PX[0] + 20.0, ARENA_PX[2] - 20.0, STEP_PX[0]):
            if _inside(x, y, ex) or _inside(x, mirror_y(y), ex):
                continue
            out.append((float(x), float(y)))
    return tuple(out)


def _means(img: np.ndarray, centres: Sequence[tuple[float, float]], mirrored: bool) -> np.ndarray:
    h, w = img.shape[:2]
    sx, sy = w / float(REF_SIZE[0]), h / float(REF_SIZE[1])
    half = PATCH_PX / 2.0
    out = np.zeros((len(centres), 3), dtype=np.float64)
    for i, (x, y) in enumerate(centres):
        if mirrored:
            y = mirror_y(y)
        x0, x1 = int(round((x - half) * sx)), max(int(round((x - half) * sx)) + 1, int(round((x + half) * sx)))
        y0, y1 = int(round((y - half) * sy)), max(int(round((y - half) * sy)) + 1, int(round((y + half) * sy)))
        out[i] = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1), :3].reshape(-1, 3).mean(axis=0)
    return out


def _usable(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[2] >= 3 and img.shape[0] >= 16 and img.shape[1] >= 9


def tinted(now: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Per patch: whether colour `now` (n x 3) is `base` (n x 3) blended toward OVERLAY_RGB, a
    share k in TINT_K of the way, at most TINT_RESIDUAL off that line (module doc), for a base
    at least TINT_MIN_REACH from OVERLAY_RGB."""
    d = np.asarray(now, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    v = np.asarray(OVERLAY_RGB) - np.asarray(base, dtype=np.float64)
    n2 = (v * v).sum(axis=1)
    k = (d * v).sum(axis=1) / np.maximum(n2, 1e-6)
    left = np.sqrt(((d - k[:, None] * v) ** 2).sum(axis=1))
    return (n2 >= TINT_MIN_REACH ** 2) & (k >= TINT_K[0]) & (k <= TINT_K[1]) & (left <= TINT_RESIDUAL)


def mirror_tint(rgb: np.ndarray) -> float:
    """The share of the enemy-half patches tinted toward the overlay against their mirror in our
    half (module doc); 0.0 for a frame too small to hold a patch."""
    img = np.asarray(rgb)
    if not _usable(img):
        return 0.0
    centres = patch_centres()
    if not centres:
        return 0.0
    return float(tinted(_means(img, centres, False), _means(img, centres, True)).mean())


def overlay_on(rgb: np.ndarray, share: float = OVERLAY_SHARE) -> bool:
    """Whether the deploy overlay tints the enemy half of this battle frame (mirror_tint)."""
    return mirror_tint(rgb) >= float(share)


@dataclass(frozen=True)
class OverlayRead:
    """One frame's overlay flag (OverlayDetector.update): on, and why: mirror (the share tinted
    against the mirror), median (the share tinted against the running medians; None before the
    patches have MEDIAN_MIN frames), tap (inside the bot's own tap window)."""
    on: bool
    mirror: float = 0.0
    median: float | None = None
    tap: bool = False

    @property
    def why(self) -> str | None:
        """"tap", "mirror", "median" (the first that holds), None when off."""
        if not self.on:
            return None
        if self.tap:
            return "tap"
        return "mirror" if self.mirror >= OVERLAY_SHARE else "median"

    def to_dict(self) -> dict:
        return {"on": self.on, "mirror": round(self.mirror, 3),
                "median": None if self.median is None else round(self.median, 3), "tap": self.tap}


class OverlayDetector:
    """The deploy overlay per battle frame (module doc: mirror, median, tap). update(rgb, t) once
    a battle frame; note_tap(t) when the bot sends a tap pair; reset() at each battle's start (the
    floor medians are per battle: the arena changes; tap windows are kept)."""

    def __init__(self, *, share: float = OVERLAY_SHARE, median_share: float = MEDIAN_SHARE,
                 keep: int = MEDIAN_KEEP, min_frames: int = MEDIAN_MIN,
                 tap_s: float = OWN_TAP_PAIR_S + OWN_TAP_TAIL_S):
        self.share, self.median_share = float(share), float(median_share)
        self.keep, self.min_frames = max(1, int(keep)), max(1, int(min_frames))
        self.tap_s = float(tap_s)
        self._taps: deque = deque(maxlen=32)
        self.reset()

    def reset(self) -> None:
        self._hist: deque = deque(maxlen=self.keep)
        self._size: tuple[int, int] | None = None
        self.frames = 0
        self.flagged = 0
        self.last: OverlayRead | None = None

    def note_tap(self, t0: float, t1: float | None = None) -> None:
        """The bot tapped a card at `t0` (the vision clock): frames captured from t0 to `t1`
        (default t0 + OWN_TAP_PAIR_S + OWN_TAP_TAIL_S) are overlay frames."""
        t0 = float(t0)
        self._taps.append((t0, t0 + self.tap_s if t1 is None else float(t1)))

    def in_tap_window(self, t: float) -> bool:
        t = float(t)
        return any(a <= t <= b for a, b in self._taps)

    def median_tint(self, enemy: np.ndarray) -> float | None:
        """The share of patches whose colour `enemy` (n x 3) is tinted against their running
        median, None before MEDIAN_MIN unflagged frames."""
        if len(self._hist) < self.min_frames:
            return None
        base = np.median(np.stack(list(self._hist)), axis=0)
        return float(tinted(enemy, base).mean())

    def update(self, rgb: np.ndarray, t: float) -> OverlayRead:
        img = np.asarray(rgb)
        tap = self.in_tap_window(t)
        centres = patch_centres()
        if not _usable(img) or not centres:
            out = OverlayRead(tap, tap=tap)
        else:
            size = (int(img.shape[1]), int(img.shape[0]))
            if size != self._size:
                self._hist.clear()
                self._size = size
            enemy = _means(img, centres, False)
            mirror = float(tinted(enemy, _means(img, centres, True)).mean())
            median = self.median_tint(enemy)
            on = tap or mirror >= self.share or (median is not None and median >= self.median_share)
            out = OverlayRead(bool(on), mirror, median, tap)
            if not on:
                self._hist.append(enemy)
        self.frames += 1
        self.flagged += int(out.on)
        self.last = out
        return out


__all__ = ["MEDIAN_SHARE", "OVERLAY_RGB", "OVERLAY_SHARE", "OWN_TAP_PAIR_S", "OWN_TAP_TAIL_S", "OverlayDetector",
           "OverlayRead", "TINT_ALPHA", "mirror_tint", "mirror_y", "overlay_on", "patch_centres", "tinted"]
