"""Where each princess tower's HP label is, and whether each tower is still there (FIX A / B).

The HUD reads a tower's hit points from fixed boxes (hud.py BUILTIN_REGIONS_PX), measured on
2026-10-29 frames against default-skin towers. A tower skin moves the label: the owner's main
account's chicken skin (2026-10-31 friendly) draws the enemy labels about 36 px (at 1080x1920)
higher, so the fixed boxes read the chicken's head and its red feathers, the bars read 0.0-0.22
and both enemy towers came out destroyed at tick ~266. Clanmates use many skins and some use
tower troops, so nothing here depends on one layout:

* TowerEvidence: one frame's detector boxes per tower key. A label box (LABEL_CLASSES) near a
  princess tower's reference label (LABEL_REF_PX) is that tower's label; a body box (BODY_CLASSES,
  any class: the detector calls a tower body "Cannoneer" now and then) inside a tower's grown
  reference body box (TOWER_BODY_REF_PX) is evidence that the tower stands. Presence is decided
  by position, never by class. A label of the reference size close to the reference also gives
  the shift to read that tower's HP at (placement).
* LabelPlacer: the shift per princess tower, the per-axis median of the last 15 placement votes
  once there are 5 (a 3 px dead band keeps a default-skin tower on its calibrated boxes), else
  the pixel search's result once 3 searches agree within 3 px, else no shift.
* find_label_dv: the pixel search for frames without detections. The crown badge (gold, left of
  the bar's anchor, 22-40 px tall at 1080) with the bar's first 12 px filled in the side's colour;
  the group closest to the reference badge row, None when two are about as close.
* badge_seen: the badge alone at a given row (HudReader's pixel hold: works at any HP; rubble has
  no badge).

All pixel constants are at the 1080x1920 reference frame (REF_SIZE) and scale to any 9:16 frame.
Measured on the owner's frames: the reference label and body boxes are the median detector boxes
of the 2026-10-29 friendly replayed through the detector (default skin), BADGE_DV_PX on the
2026-10-31 frames. Imports numpy only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .hpbars import ally_mask, enemy_mask

REF_SIZE = (1080, 1920)
PRINCESS_KEYS = ("own_left", "own_right", "enemy_left", "enemy_right")
KING_KEYS = ("own_king", "enemy_king")
#: The detector's HP-label classes of a princess tower (KataCR names; the king's own label,
#: "king-tower-bar", is left out: a king counts by its body).
LABEL_CLASSES = ("tower-bar", "dagger-duchess-tower-bar")
#: Tower body classes. Any of them at a tower's position counts for that tower.
BODY_CLASSES = ("PrincessTower", "Cannoneer", "DaggerDuchess", "ChefTower", "KingTower")
#: key -> the label box (x0, y0, x1, y1) at 1080x1920 for which the HUD's fixed tower regions are
#: right: the median detector tower-bar box of the 2026-10-29 friendly (default skin). The HUD
#: regions table carries them as label_<key> (hud.BUILTIN_REGIONS_PX); this is the fallback.
LABEL_REF_PX: dict[str, tuple[float, float, float, float]] = {
    "enemy_left": (176.5, 256.6, 341.1, 308.4),
    "enemy_right": (741.7, 257.1, 903.1, 308.7),
    "own_left": (174.8, 1179.3, 339.6, 1223.0),
    "own_right": (742.2, 1178.2, 906.0, 1223.0),
}
#: key -> the median tower body box at 1080x1920 (2026-10-29 friendly).
TOWER_BODY_REF_PX: dict[str, tuple[float, float, float, float]] = {
    "enemy_left": (181.8, 303.2, 334.5, 473.8),
    "enemy_right": (741.1, 303.9, 899.7, 478.6),
    "own_left": (180.7, 1079.6, 334.8, 1267.7),
    "own_right": (747.3, 1079.6, 901.4, 1268.8),
    "enemy_king": (440.2, 106.1, 631.2, 345.7),
    "own_king": (446.5, 1216.7, 627.8, 1432.0),
}
#: side -> the crown badge's centre row minus the label box's centre row (px at 1080x1920),
#: measured on the 2026-10-31 frames (the enemy's badge hangs below its bar, ours sits higher).
BADGE_DV_PX = {"enemy": 4.9, "own": -5.2}

LABEL_MATCH_DU_PX = 40.0               # a label box belongs to a tower within this |dx| ...
LABEL_MATCH_DV_PX = (-120.0, 60.0)     # ... and this dy of the reference centre
PLACE_DU_PX = 24.0                     # a label gives a placement vote within this |dx| ...
PLACE_DV_PX = (-90.0, 40.0)            # ... this dy ...
PLACE_SIZE_TOL = 0.25                  # ... and with its width and height within +-25% of the reference
BODY_GROW = 0.25                       # a body box's centre in the reference body grown 25% a side

SEARCH_UP_PX, SEARCH_DOWN_PX = 60.0, 25.0     # the search window around the reference badge row
GOLD_LEFT_PX, GOLD_RIGHT_PX = 34.0, 2.0       # badge columns: [anchor - 34, anchor + 2)
GOLD_ROW_MIN = 0.2                     # a badge row: >= 20% of those columns gold
FILL_PX = 12.0                         # the bar's first 12 px after the anchor ...
FILL_ROW_MIN = 0.6                     # ... >= 60% in the side's bar colour on a filled row
BADGE_TALL_PX = (22.0, 40.0)           # a badge group's height
AMBIGUOUS_PX = 10.0                    # a second group this close to as near: no answer
SEEN_TOL_PX = 4.0                      # badge_seen: a badge group centred within this of the row


def side_of(key: str) -> str:
    return "own" if key.startswith("own") else "enemy"


def _centre(box: Sequence[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def _to_ref(box: Sequence[float], frame_size: Sequence[int]) -> tuple[float, float, float, float]:
    sx, sy = REF_SIZE[0] / float(frame_size[0]), REF_SIZE[1] / float(frame_size[1])
    x0, y0, x1, y1 = (float(c) for c in box)
    return x0 * sx, y0 * sy, x1 * sx, y1 * sy


def _grown(box: Sequence[float], frac: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(c) for c in box)
    gx, gy = (x1 - x0) * frac, (y1 - y0) * frac
    return x0 - gx, y0 - gy, x1 + gx, y1 + gy


# ---------------------------------------------------------------------------
# Detector evidence
# ---------------------------------------------------------------------------
@dataclass
class TowerEvidence:
    """One frame's detector boxes per tower key (all at the 1080x1920 reference).

    labels: key -> the label box (presence; any label class at the label's position).
    placement: key -> (du, dv) px, the label box centre minus the reference centre, for labels
    of the reference size near the reference (a dagger-duchess label or a half-hidden one is
    presence only). bodies: the keys a body box stands on."""
    labels: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    placement: dict[str, tuple[float, float]] = field(default_factory=dict)
    bodies: set[str] = field(default_factory=set)

    def seen(self, key: str) -> bool:
        return key in self.labels or key in self.bodies

    @classmethod
    def from_detections(cls, detections: Iterable[Any], frame_size: Sequence[int],
                        label_refs: Mapping[str, Sequence[float]] | None = None) -> "TowerEvidence":
        """detections: objects with .cls, .box (x0, y0, x1, y1 in frame pixels) and optionally
        .conf (Detection, or anything shaped like it); frame_size (w, h) of those pixels."""
        refs = {k: tuple(float(c) for c in v) for k, v in (label_refs or LABEL_REF_PX).items()
                if k in PRINCESS_KEYS}
        ev = cls()
        best_label: dict[str, float] = {}
        best_place: dict[str, float] = {}
        if not frame_size or float(frame_size[0]) <= 0 or float(frame_size[1]) <= 0:
            return ev
        for d in detections:
            name = str(getattr(d, "cls", ""))
            if name not in LABEL_CLASSES and name not in BODY_CLASSES:
                continue
            box = _to_ref(getattr(d, "box"), frame_size)
            cx, cy = _centre(box)
            conf = float(getattr(d, "conf", 0.0) or 0.0)
            if name in BODY_CLASSES:
                for key, ref in TOWER_BODY_REF_PX.items():
                    x0, y0, x1, y1 = _grown(ref, BODY_GROW)
                    if x0 <= cx <= x1 and y0 <= cy <= y1:
                        ev.bodies.add(key)
                continue
            for key, ref in refs.items():
                rx, ry = _centre(ref)
                du, dv = cx - rx, cy - ry
                if abs(du) > LABEL_MATCH_DU_PX or not LABEL_MATCH_DV_PX[0] <= dv <= LABEL_MATCH_DV_PX[1]:
                    continue
                if conf >= best_label.get(key, -1.0):
                    best_label[key] = conf
                    ev.labels[key] = box
                rw, rh = ref[2] - ref[0], ref[3] - ref[1]
                bw, bh = box[2] - box[0], box[3] - box[1]
                if (abs(du) <= PLACE_DU_PX and PLACE_DV_PX[0] <= dv <= PLACE_DV_PX[1]
                        and rw > 0 and rh > 0 and abs(bw / rw - 1.0) <= PLACE_SIZE_TOL
                        and abs(bh / rh - 1.0) <= PLACE_SIZE_TOL
                        and conf >= best_place.get(key, -1.0)):
                    best_place[key] = conf
                    ev.placement[key] = (du, dv)
        return ev

    @classmethod
    def from_result(cls, result: Any, label_refs: Mapping[str, Sequence[float]] | None = None
                    ) -> "TowerEvidence":
        """From a DetectorResult-like object (.detections and .frame_size)."""
        return cls.from_detections(list(getattr(result, "detections", None) or ()),
                                   getattr(result, "frame_size", None) or (0, 0), label_refs)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------
SOURCE_DETECTOR, SOURCE_SEARCH, SOURCE_FIXED = "detector", "search", "fixed"


class LabelPlacer:
    """The shift (du, dv) px at 1080x1920 to read each princess tower's HP at, and where it came
    from: SOURCE_DETECTOR with >= `min_votes` placement votes (the per-axis median of the last
    `votes`), else SOURCE_SEARCH once `search_agree` pixel searches agree within
    `search_agree_px` (the search then stops for that tower), else SOURCE_FIXED (0, 0). A
    per-axis dead band (|d| < `dead_band_px` -> 0) keeps a default-skin tower on its calibrated
    boxes (and reports it SOURCE_FIXED). The search runs only while a tower has fewer than
    `min_votes` votes. The shift is kept while no boxes come (it is sticky) until reset()."""

    def __init__(self, keys: Sequence[str] = PRINCESS_KEYS, *, votes: int = 15, min_votes: int = 5,
                 dead_band_px: float = 3.0, search_agree: int = 3, search_agree_px: float = 3.0):
        self.keys = tuple(keys)
        self.votes = max(1, int(votes))
        self.min_votes = max(1, int(min_votes))
        self.dead_band_px = float(dead_band_px)
        self.search_agree = max(1, int(search_agree))
        self.search_agree_px = float(search_agree_px)
        self.reset()

    def reset(self) -> None:
        self._votes: dict[str, deque] = {k: deque(maxlen=self.votes) for k in self.keys}
        self._search: dict[str, deque] = {k: deque(maxlen=self.search_agree) for k in self.keys}
        self._settled: dict[str, float] = {}

    def _dead(self, d: float) -> float:
        return 0.0 if abs(d) < self.dead_band_px else float(d)

    def note(self, evidence: TowerEvidence | None) -> None:
        if evidence is None:
            return
        for key, (du, dv) in evidence.placement.items():
            if key in self._votes:
                self._votes[key].append((float(du), float(dv)))

    def wants_search(self, key: str) -> bool:
        return (key in self._votes and len(self._votes[key]) < self.min_votes
                and key not in self._settled)

    def note_search(self, key: str, dv: float | None) -> None:
        """A pixel search's result (dv px, None: nothing found or ambiguous: not counted)."""
        if dv is None or not self.wants_search(key):
            return
        run = self._search[key]
        run.append(float(dv))
        if len(run) >= self.search_agree and max(run) - min(run) <= self.search_agree_px:
            self._settled[key] = self._dead(float(np.median(list(run))))

    def shift(self, key: str) -> tuple[float, float, str]:
        """(du, dv, source); a shift inside the dead band is (0, 0, SOURCE_FIXED) whatever
        found it (the calibrated boxes are read)."""
        du, dv, source = 0.0, 0.0, SOURCE_FIXED
        votes = self._votes.get(key)
        if votes is not None and len(votes) >= self.min_votes:
            a = np.asarray(votes, dtype=float)
            du, dv = self._dead(float(np.median(a[:, 0]))), self._dead(float(np.median(a[:, 1])))
            source = SOURCE_DETECTOR
        elif key in self._settled:
            dv, source = self._settled[key], SOURCE_SEARCH
        return (du, dv, source) if (du or dv) else (0.0, 0.0, SOURCE_FIXED)

    def shifts(self) -> dict[str, tuple[float, float, str]]:
        return {k: self.shift(k) for k in self.keys}


# ---------------------------------------------------------------------------
# Pixels
# ---------------------------------------------------------------------------
def gold_mask(rgb: np.ndarray) -> np.ndarray:
    """The crown badge's gold (the level badge left of a princess tower's bar)."""
    a = np.asarray(rgb)
    r, g, b = (a[..., i].astype(np.int16) for i in range(3))
    return (r > 180) & (g > 150) & (b < 90) & (r - b > 120)


def _groups(flags: Sequence[bool], bridge: int) -> list[tuple[int, int]]:
    """Runs of True (first, last index), bridging gaps of <= `bridge` False entries."""
    out: list[tuple[int, int]] = []
    cur: list[int] | None = None
    miss = 0
    for i, f in enumerate(flags):
        if f:
            cur = [i, i] if cur is None else [cur[0], i]
            miss = 0
        elif cur is not None:
            miss += 1
            if miss > bridge:
                out.append((cur[0], cur[1]))
                cur, miss = None, 0
    if cur is not None:
        out.append((cur[0], cur[1]))
    return out


def ref_badge_row(key: str, label_ref: Sequence[float]) -> float:
    """The reference crown badge's centre row (px at 1080x1920) of a princess tower."""
    return _centre(label_ref)[1] + BADGE_DV_PX[side_of(key)]


def _badge_groups(img: np.ndarray, anchor_x: float, r0: int, r1: int, sx: float, sy: float
                  ) -> list[tuple[int, int, float]]:
    """(lo, hi, centre row px at 1080x1920) of the gold groups 22-40 px tall in rows r0..r1-1."""
    h, w = img.shape[:2]
    c0 = min(w, max(0, int(round(anchor_x - GOLD_LEFT_PX * sx))))
    c1 = min(w, max(0, int(round(anchor_x + GOLD_RIGHT_PX * sx))))
    if r1 <= r0 or c1 <= c0:
        return []
    gold = gold_mask(img[r0:r1, c0:c1]).mean(axis=1) >= GOLD_ROW_MIN
    out = []
    for lo, hi in _groups(gold, max(1, int(round(1.5 * sx)))):
        tall = (hi - lo + 1) / sy
        if BADGE_TALL_PX[0] <= tall <= BADGE_TALL_PX[1]:
            out.append((lo, hi, (r0 + (lo + hi) / 2.0) / sy))
    return out


def find_label_dv(rgb: np.ndarray, key: str, anchor_u: float, label_ref: Sequence[float]) -> float | None:
    """The pixel search: dv px (at 1080x1920) of `key`'s crown badge from its reference row, None
    when no badge with a filled bar start is found or two are about as close.

    anchor_u: the bar fill's start (the bar region's u0 + the bar margin, normalised);
    label_ref: the reference label box at 1080x1920. A badge group is 22-40 px tall with
    >= max(3, 4 px) rows whose first 12 px after the anchor are >= 60% bar colour (enemy_mask /
    ally_mask), within 60 px above to 25 px below the reference badge row."""
    img = np.asarray(rgb)
    h, w = img.shape[:2]
    sx, sy = w / float(REF_SIZE[0]), h / float(REF_SIZE[1])
    anchor = float(anchor_u) * w
    ref = ref_badge_row(key, label_ref)
    r0 = max(0, int((ref - SEARCH_UP_PX) * sy))
    r1 = min(h, int((ref + SEARCH_DOWN_PX) * sy) + 1)
    groups = _badge_groups(img, anchor, r0, r1, sx, sy)
    if not groups:
        return None
    f0 = min(w, max(0, int(round(anchor))))
    f1 = min(w, max(0, int(round(anchor + FILL_PX * sx))))
    if f1 <= f0:
        return None
    band = img[r0:r1, f0:f1]
    fill = (ally_mask(band) if side_of(key) == "own" else enemy_mask(band)).mean(axis=1) >= FILL_ROW_MIN
    need = max(3, int(round(4 * sx)))
    cands = sorted((c for lo, hi, c in groups if int(fill[lo:hi + 1].sum()) >= need),
                   key=lambda c: abs(c - ref))
    if not cands:
        return None
    if len(cands) > 1 and abs(cands[1] - ref) - abs(cands[0] - ref) < AMBIGUOUS_PX:
        return None
    return float(cands[0] - ref)


def badge_seen(rgb: np.ndarray, anchor_u: float, row_px: float, *, tol_px: float = SEEN_TOL_PX) -> bool:
    """Whether a crown badge (a gold group 22-40 px tall left of the anchor) is centred within
    `tol_px` of `row_px` (px at 1080x1920). The badge alone: it shows at any HP; rubble has none."""
    img = np.asarray(rgb)
    h, w = img.shape[:2]
    sx, sy = w / float(REF_SIZE[0]), h / float(REF_SIZE[1])
    half = BADGE_TALL_PX[1] / 2.0 + tol_px + 4.0
    r0 = max(0, int((row_px - half) * sy))
    r1 = min(h, int((row_px + half) * sy) + 1)
    return any(abs(c - row_px) <= tol_px for _, _, c in _badge_groups(img, float(anchor_u) * w, r0, r1, sx, sy))


__all__ = [
    "BADGE_DV_PX", "BODY_CLASSES", "KING_KEYS", "LABEL_CLASSES", "LABEL_REF_PX", "LabelPlacer",
    "PRINCESS_KEYS", "SOURCE_DETECTOR", "SOURCE_FIXED", "SOURCE_SEARCH", "TOWER_BODY_REF_PX",
    "TowerEvidence", "badge_seen", "find_label_dv", "gold_mask", "ref_badge_row",
]
