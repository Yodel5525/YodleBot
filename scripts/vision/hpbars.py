"""HP bars: the fill of a unit's bar, attached to the unit box below it (V07, vision-06).

The detector already finds HP bars (KataCR classes "bar" and "bar-level", which
scripts/brain/vision.py:47-52 throws away as UI). Here each bar box is attached
to the unit box it sits on, and its fill is read from the team colours:

* attach_bars   bar -> unit by geometry: the bar's bottom edge just above the
                unit's top edge (or overlapping it a little), horizontally
                centred on it; cheapest pairs first, one bar per unit.
* read_bar      fill fraction inside a bar box: rows holding the team colour,
                runs of coloured columns (gaps of a couple of pixels bridged,
                as tower_hp.py:106-116 does), each followed by the dim drained
                part to its right; the longest run + drained part is the bar
                (a level badge's coloured border makes short runs) and the
                fraction is run / (run + drained). The colour also names the
                team (blue = ours, red/pink = theirs), a cue for the tracker's
                team vote (Observation.bar_team).
* read_unit_hp  both, per unit. A unit WITHOUT a bar is at full health (the
                game draws a unit's bar once it is damaged): frac 1.0.
* measure_tower_bar  the author's princess-tower bar measure
                (tower_hp.py:53-150) with its geometry made parameters, for
                scripts/vision/hud.py's tower fallback.

Colour tests are the author's, measured on live frames: ours is the ally
tower mask (tower_hp.py:224), theirs the union of the enemy masks
(tower_hp.py:172-174, 203), both used for the tower bars. On the owner's frames
(client 160635001) unit bars are ours (113, 208, 253) and theirs (225, 36,
100); the purple river (211, 95, 202) and orange sprites (233, 162, 100) pass
the author's pink terms, so unit bars use the deep-red term with blue at
least 60 below red (unit_enemy_mask). The drained part is dim but not black,
(69, 77, 113) ours and (96, 45, 60) theirs, like the author's drained enemy
tower track (93, 50, 73) (tower_hp.py:177-179), so DIM_MAX is 135. A bar
flashing white when hit matches neither team and reads as no bar for that
frame; so does a level badge alone (bars shorter than MIN_BAR_FRAC of the box).

Imports numpy only. Boxes are (x1, y1, x2, y2) pixels of the frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .types import ENEMY_TEAM, OWN_TEAM, UNKNOWN_TEAM

BAR_CLASSES = ("bar", "bar-level")      # KataCR detector classes for HP bars
DIM_MAX = 135                           # a drained bar track: every channel <= this
FULL_SNAP = 0.95                        # tower_hp.py:148-149: >= 0.95 reads as full
REF_WIDTH = 1080                        # pixel constants below are at this frame width
GAP_PX = 2                              # tower_hp.py:115: "allow 2px border or number notch"
MIN_BAR_FRAC = 0.4                      # a bar spans at least this much of its (grown) box


def _rint(x: float) -> int:
    return int(math.floor(x + 0.5))


def _channels(rgb: np.ndarray):
    a = np.asarray(rgb)[..., :3].astype(np.int16)
    return a[..., 0], a[..., 1], a[..., 2]


def ally_mask(rgb: np.ndarray) -> np.ndarray:
    """Our bar's blue/cyan fill (tower_hp.py:223-224)."""
    r, g, b = _channels(rgb)
    return (b > 145) & (b - r > 45) & (b - g > 20)


def enemy_mask(rgb: np.ndarray) -> np.ndarray:
    """The opponent's bar fill: deep red, bright pink or light pink skins
    (tower_hp.py:172-174, 203; the drained-track 'dark_red' term stays out,
    tower_hp.py:175-188)."""
    r, g, b = _channels(rgb)
    deep_red = (r > 165) & (g < 130) & (b < 210) & (r - g > 35)
    bright_pink = (r > 200) & (g > 100) & (r - g > 70)
    light_pink = (r > 240) & (g > 180) & (b > 180) & (r - g > 15)
    return deep_red | bright_pink | light_pink


def unit_enemy_mask(rgb: np.ndarray) -> np.ndarray:
    """The opponent's UNIT bar fill: the author's deep red with blue at least 60
    below red (keeps the purple river and orange sprites out; see the module doc)."""
    r, g, b = _channels(rgb)
    return (r > 165) & (g < 130) & (r - g > 35) & (r - b > 60)


def team_mask(rgb: np.ndarray, team: int) -> np.ndarray:
    """A unit bar's fill colour for `team`."""
    return ally_mask(rgb) if team == OWN_TEAM else unit_enemy_mask(rgb)


def dark_mask(rgb: np.ndarray) -> np.ndarray:
    """A bar's drained part (and its dark outline): every channel <= DIM_MAX."""
    return np.asarray(rgb)[..., :3].max(axis=2) <= DIM_MAX


def _dim_after(coldark: np.ndarray, colfill: np.ndarray, start: int, gap: int) -> int:
    """One past the last dim column after `start` (a fill column stops the walk;
    at most `gap` unknown columns in a row are stepped over)."""
    end, misses = start, 0
    for i in range(start, len(coldark)):
        if colfill[i]:
            break
        if coldark[i]:
            end, misses = i + 1, 0
        else:
            misses += 1
            if misses > gap:
                break
    return end


def walk_run(cols: np.ndarray, start: int, gap: int) -> int:
    """One past the last True reachable from `start`, stepping over at most
    `gap` consecutive False entries (start itself is assumed True)."""
    end, misses = start, 0
    for i in range(start, len(cols)):
        if cols[i]:
            end, misses = i + 1, 0
        else:
            misses += 1
            if misses > gap:
                break
    return end


def _box_px(box: Sequence[float], width: int, height: int, margin: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(c) for c in box)
    return (max(0, int(math.floor(x1)) - margin), max(0, int(math.floor(y1)) - margin),
            min(width, int(math.ceil(x2)) + margin), min(height, int(math.ceil(y2)) + margin))


@dataclass(frozen=True)
class BarRead:
    """frac: fill in [0, 1], None when the box holds no readable bar. team: the
    fill colour (OWN_TEAM blue, ENEMY_TEAM red/pink, UNKNOWN_TEAM)."""
    frac: float | None
    team: int = UNKNOWN_TEAM
    conf: float = 0.0
    fill_px: int = 0
    track_px: int = 0


def read_bar(rgb: np.ndarray, box: Sequence[float], team: int | None = None, *,
             margin_px: int = 2, gap_px: int | None = None) -> BarRead:
    """Fill fraction of the HP bar inside `box` (grown by `margin_px`).

    team=None picks the colour with more pixels in the box and reports it;
    pass the unit's team when it is already decided. gap_px defaults to
    GAP_PX scaled to the frame width.
    """
    img = np.asarray(rgb)
    if img.ndim != 3 or img.size == 0:
        return BarRead(None)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _box_px(box, w, h, int(margin_px))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return BarRead(None)
    own, enemy = ally_mask(crop), unit_enemy_mask(crop)
    n_own, n_enemy = int(own.sum()), int(enemy.sum())
    if team is None:
        if n_own == 0 and n_enemy == 0:
            return BarRead(None)
        team = OWN_TEAM if n_own >= n_enemy else ENEMY_TEAM
        team_conf = max(n_own, n_enemy) / float(n_own + n_enemy)
    else:
        team_conf = 1.0
    fill = own if team == OWN_TEAM else enemy
    per_row = fill.sum(axis=1)
    if per_row.max(initial=0) == 0:
        return BarRead(None, int(team), 0.0)
    rows = per_row >= 0.5 * per_row.max()
    colfill = fill[rows].mean(axis=0) >= 0.5
    if not colfill.any():
        return BarRead(None, int(team), 0.0)
    coldark = (dark_mask(crop)[rows].mean(axis=0) >= 0.5) & ~colfill
    gap = int(gap_px) if gap_px is not None else max(1, _rint(GAP_PX * w / REF_WIDTH))
    best = None                                   # (bar length, fill length, start)
    i, n = 0, len(colfill)
    while i < n:
        if not colfill[i]:
            i += 1
            continue
        fill_end = walk_run(colfill, i, gap)
        track_end = _dim_after(coldark, colfill, fill_end, gap)
        cand = (track_end - i, fill_end - i, i)
        if best is None or cand[0] > best[0]:
            best = cand
        i = fill_end
    track_px, fill_px, _ = best
    if track_px < MIN_BAR_FRAC * crop.shape[1]:
        return BarRead(None, int(team), 0.0)      # a badge's border or a speck, not a bar
    frac = fill_px / float(track_px)
    if frac >= FULL_SNAP:
        frac = 1.0
    conf = team_conf * (1.0 if track_px >= 4 else 0.5)
    return BarRead(float(frac), int(team), float(conf), fill_px, track_px)


def attach_bars(unit_boxes: Sequence[Sequence[float]], bar_boxes: Sequence[Sequence[float]], *,
                max_gap: float = 0.6, max_overlap: float = 0.4, max_dx: float = 0.5) -> list[int | None]:
    """For each unit box, the index of the bar box drawn on it, or None.

    A bar belongs to a unit when its bottom edge is at most `max_gap` unit
    heights above the unit's top edge (or overlaps it by at most `max_overlap`
    heights) and its centre is within `max_dx` unit widths of the unit's centre.
    Cost = |gap| / height + |dx| / width; pairs are taken cheapest first (ties
    by unit index, then bar index), each unit and bar at most once.
    """
    pairs = []
    for ui, (ux1, uy1, ux2, uy2) in enumerate(unit_boxes):
        uw, uh = max(1.0, float(ux2) - float(ux1)), max(1.0, float(uy2) - float(uy1))
        ucx = (float(ux1) + float(ux2)) / 2.0
        for bi, (bx1, by1, bx2, by2) in enumerate(bar_boxes):
            gap = float(uy1) - float(by2)
            if gap > max_gap * uh or gap < -max_overlap * uh:
                continue
            dx = abs((float(bx1) + float(bx2)) / 2.0 - ucx) / uw
            if dx > max_dx:
                continue
            pairs.append((abs(gap) / uh + dx, ui, bi))
    pairs.sort()
    out: list[int | None] = [None] * len(unit_boxes)
    used: set[int] = set()
    for _, ui, bi in pairs:
        if out[ui] is None and bi not in used:
            out[ui] = bi
            used.add(bi)
    return out


@dataclass(frozen=True)
class HpRead:
    """frac: 1.0 when no bar is attached (full health draws none). team: the bar
    colour's team, UNKNOWN_TEAM without a bar. bar: index of the attached box."""
    frac: float
    has_bar: bool
    team: int = UNKNOWN_TEAM
    conf: float = 1.0
    bar: int | None = None


def read_unit_hp(rgb: np.ndarray, unit_boxes: Sequence[Sequence[float]],
                 bar_boxes: Sequence[Sequence[float]], teams: Sequence[int | None] | None = None,
                 **attach_kw: Any) -> list[HpRead]:
    """HP fraction per unit box. A unit without an attached, readable bar reads 1.0
    (has_bar False; conf 0 when a bar was attached but could not be read)."""
    att = attach_bars(unit_boxes, bar_boxes, **attach_kw)
    out = []
    for i, bi in enumerate(att):
        if bi is None:
            out.append(HpRead(1.0, False))
            continue
        team = None
        if teams is not None and i < len(teams) and teams[i] in (OWN_TEAM, ENEMY_TEAM):
            team = teams[i]
        br = read_bar(rgb, bar_boxes[bi], team)
        if br.frac is None:
            out.append(HpRead(1.0, False, br.team, 0.0, bi))
        else:
            out.append(HpRead(br.frac, True, br.team, br.conf, bi))
    return out


def split_bars(detections: Sequence[Any], bar_classes: Sequence[str] = BAR_CLASSES) -> tuple[list[int], list[int]]:
    """Indices of unit detections and of bar detections (by `cls`)."""
    units, bars = [], []
    for i, d in enumerate(detections):
        (bars if getattr(d, "cls", None) in bar_classes else units).append(i)
    return units, bars


def hp_for_detections(rgb: np.ndarray, detections: Sequence[Any],
                      bar_classes: Sequence[str] = BAR_CLASSES) -> dict[int, HpRead]:
    """{index of each unit detection: HpRead}, using the bar detections among them."""
    units, bars = split_bars(detections, bar_classes)
    reads = read_unit_hp(rgb, [detections[i].box for i in units], [detections[i].box for i in bars])
    return dict(zip(units, reads))


def annotate_observations(observations: Sequence[Any], reads: Sequence[HpRead],
                          missing_frac: float | None = None) -> None:
    """Copy HP reads onto tracker Observations (same order), in place.

    hp_frac becomes the bar fill, or `missing_frac` for a unit without a bar.
    The default None lets the tracker keep a damaged unit's last reading when
    its bar box drops out for a frame; a track whose hp_frac is None is at full
    health (no bar ever seen), which the state converter reads as 1.0.
    bar_team is the bar colour's team (UNKNOWN_TEAM without a bar).
    """
    for obs, r in zip(observations, reads):
        obs.hp_frac = r.frac if r.has_bar else missing_frac
        obs.bar_team = r.team if r.has_bar else UNKNOWN_TEAM


def measure_tower_bar(mask: np.ndarray, full_px: int, *, anchor_px: int = 0, tol_px: int = 8,
                      gap_px: int = 2, min_run: int = 10, min_rows: int = 4) -> float:
    """The author's princess-bar measure (tower_hp.py:53-150), geometry as parameters.

    mask: HxW bool of the bar colour over a search band; the bar's fill starts
    at column `anchor_px` (+-tol_px) and is `full_px` long when full. Per row:
    the run from the first hit near the anchor, bridging gaps of <= gap_px
    (gaps counted into the run); rows with runs >= min_run form clusters, and
    a cluster at least min_rows thick gives the median of its runs. The best
    cluster over full_px is the fraction, >= 0.95 snapped to 1.0, else rounded
    to 3 decimals. No bar gives 0.0 (the occlusion/destroyed reading
    TowerHpFilter handles).
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or m.size == 0 or full_px <= 0:
        return 0.0
    h, w = m.shape
    min_run = max(1, int(min_run))
    x_min = max(0, anchor_px - tol_px)
    x_max = min(w, anchor_px + full_px + tol_px)
    anchor_rel = anchor_px - x_min
    runs = []
    for r in range(h):
        row = m[r, x_min:x_max]
        lo, hi = max(0, anchor_rel - tol_px), min(len(row), anchor_rel + tol_px + 1)
        hits = np.nonzero(row[lo:hi])[0]
        if len(hits) == 0:
            runs.append(0)
            continue
        length = gaps = 0
        for idx in range(lo + int(hits[0]), len(row)):
            if row[idx]:
                length += 1 + gaps
                gaps = 0
            else:
                gaps += 1
                if gaps > gap_px:
                    break
        runs.append(min(full_px, length))
    best = 0.0
    cluster: list[int] = []
    for length in runs + [0]:                     # the sentinel closes the last cluster
        if length >= min_run:
            cluster.append(length)
            continue
        if len(cluster) >= min_rows:
            best = max(best, float(np.median(cluster)))
        cluster = []
    if best == 0:
        return 0.0
    frac = best / float(full_px)
    if frac >= FULL_SNAP:
        return 1.0
    return round(float(np.clip(frac, 0.0, 1.0)), 3)
