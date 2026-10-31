"""Per-battle calibration of each princess tower's HP label, and the read gate (skin audit P0-1, P1-4).

The owner picks a random tower skin every battle, clanmates use many, and the arena changes with
the season and the mode. A skin moves a tower's HP label (the 2026-10-31 chicken: 36 px up at
1080x1920), and the HUD's fixed boxes then read skin pixels: bars of 0.0-0.22 and, from there,
a false destruction. So each battle finds every princess label where it is, from UI that is the
same in every skin, and the HUD reads a tower only on frames where its label is confirmed there.

THE SIGNATURE (find_candidates; px at 1080x1920, scaled to the frame). In a wide window around
the reference label (du -60..+60, dv -120..+60 px):
  (b) the bar: on >= 4 px of rows, a run of the side's bar colour (hpbars.enemy_mask /
      ally_mask) that starts at an anchor and is BAR_FULL_PX 119 +- 8 long (a full bar), or
      that is followed by the dark drained track up to that length (a damaged bar); the fill
      band (the rows where the bar starts at that anchor, its left neighbours not bar colour)
      10-26 px tall, with a dark row (outline_mask: the UI outline) right above and right
      below it at the anchor and, for a full bar, over the bar's right part, which the digits
      never cover. A run that does not end where the bar ends (the deploy overlay's floor,
      skin decorations, a red floor touching the bar) is no bar;
  (a) the crown badge: a gold group (tower_labels.gold_mask, >= 20% of the badge columns
      [anchor - 35, anchor - 2)) 22-40 px tall, centred within 8 px of the fill band's centre,
      with the badge's dark outline row within 6 px under it. The gold rows are counted only
      from 16 px above the band to 16 px below it, and a group never runs across a row of the
      badge's outline (>= 70% dark): that splits a gold roof, pad or flower bed touching the
      badge off it. A candidate whose rows would leave the search window is refused (it
      touches the window edge);
  (c) the digits, when tower templates exist: on a full bar, digits read at conf >= 0.8 that
      are not a legal full HP (hud.LEGAL_TOWER_MAX_HP) refuse the candidate.
du is the anchor minus the reference anchor (the bar region's left edge + its margin), dv the
fill band's centre minus the reference band's (the label box centre + FILL_ROWS_PX). The
candidate nearest the reference wins; two about as near (within 10 px) and apart: none.

THE LOCK (TowerCalibrator). The search runs only while the battle clock runs (HudReader's
towers_clock_running: from the third accepted clock reading; None, a reader that does not
follow the clock, searches from the first frame), on frames without the deploy overlay
(overlay.overlay_on), in the calibration window: the first WINDOW_S (8) s of running clock and
at least WINDOW_FRAMES (8) searched frames (sparse recordings). A tower is locked once LOCK_FRAMES
(8) searches over >= LOCK_SPAN_S (1) s agree within AGREE_PX (3) px, or SPARSE_FRAMES (3) over
>= SPARSE_SPAN_S (3) s (a recording with an image every few seconds); at the median of the
agreeing votes. The lock keeps the badge crop (the template of the read gate and badge_seen),
the fill band's rows and its colour per row, and the digit box's crop (for P1-3 / P1-5). A tower
not locked when the window ends is "expired": calibrated False, no reads at all.

THE READ GATE (CalibRead.confirmed), every frame per locked tower: confirmed when the badge
template matches at the lock (normalised cross-correlation >= NCC_MIN 0.7, best within +-3 px)
or the bar starts at the anchor on the locked rows (+-1 px: bar colour on its first 5 px, not
left of it, and the outline row on the side away from the digits dark there). HudReader reads
the digits and the bar of a confirmed tower at the lock (its tower_* / bar_* boxes moved by the
lock, a shift under DEAD_BAND_PX 3 left at 0: the calibrated boxes) and reads nothing (None)
for any other princess tower: never 0.0 from an unconfirmed label, never the fixed boxes.
badge_seen (CalibRead.seen, P1-4): True when the template matches and the bar starts at the
anchor; None when only part shows (40% of the template from one side matches, or one of the
two holds: a unit covers part of the label) or a unit box covers the label; False when neither
the badge (nor 40% of it) nor the bar start shows. Gold alone never makes a badge: the template
carries the badge's outline and level number (NCC per colour channel, so a flat gold pad or
floor scores 0).

THE LABEL'S ABSENCE (CalibRead.absent_s, skin audit P0-2): per locked tower, the seconds its label
has been absent at the lock: seen False (neither the badge template, nor 40% of it, nor the bar
start) and no detector tower-bar box at the lock (TowerEvidence.placement within VOTE_FAR_PX, or a
label box centred on the lock's label grown LABEL_AT_LOCK_PX). Each absent frame adds the time
since the tower's previous frame (the first absent frame of a count adds nothing). A frame on
which the label or part of it shows, or the detector boxes it there, starts the count again; a
frame on which a unit box covers the label (covered) or the deploy overlay is on neither adds
nor starts it again (paused: the count is kept, and the next absent frame adds only its own
time since that frame). After a covered frame that time counts only while the count's absent
frames are at least twice its covered ones: frames the detector misses under a unit it boxes
on most frames (a Balloon over a standing tower) are no evidence, so there only absent frames
in a row count. So the bot's own taps or a unit passing over the label slow the count down
without ever losing it; a unit boxed on every other frame holds it until it goes. HudReader gives it to TowerHpTracker, which
destroys a tower (evidence "label") once it reaches 2.5 s while the tower was read <= 0.75. A
body box at the tower does not stop it: ruins are boxed as towers (P1-1).

OVERLAY FRAMES (update's overlay, P1-2): nothing is checked, searched or counted: confirmed and
seen None (HudReader reads no tower on them), no vote, no re-check, the absence clock paused.

RE-CHECK AND UNLATCH. A locked tower's label is matched by a confirmed frame or a detector
tower-bar vote (tower_labels.TowerEvidence.placement) within VOTE_AGREE_PX of the lock. A label
never moves within a battle, so LOST_S (3) s without a match while the detector still boxes
the tower's body (or DET_TRIGGER (5) of its tower-bar votes lie elsewhere), no unit box covers
the label and the tower is not destroyed (a destroyed tower is never re-checked, and a re-search
of it stops), means the lock is wrong: the wide window is searched
again (re-search; the old lock still gates the reads, and a confirmed frame there ends the
re-search). Not while the tower may be falling: a tower read <= 0.75 (update's `low`: its
readings were confirmed at this lock) whose absence count runs keeps its lock, and a re-search
of it stops, so the fall is counted to the end (ruins are boxed as tower bodies, so the re-check
would otherwise start 3 s after the label went and race the 2.5 s count). The re-search locks by
the same rule (source "re-search"); MAX_SEARCHES (3) re-search seconds in a row in which a search
ran and found no candidate remove the lock (status "unlatched": no reads); a second in which no
search ran (the label covered, the overlay on) counts neither way. A tower that expired is
searched again the same way once DET_TRIGGER detector votes ask for it (at most MAX_RESEARCHES
times a battle). The detector's votes never lock a tower by themselves. An unlatched lock is
still checked (its "ghost"): once its label is seen whole there (badge template and bar start)
on as many frames as a lock needs (LOCK_FRAMES over LOCK_SPAN_S, or SPARSE_FRAMES over
SPARSE_SPAN_S), it is locked again (source "re-lock"): the unlatch was a long occlusion, not a
wrong lock.

THE LABEL BACK (seen_run): per locked tower, the frames and seconds since the label was last not
seen (seen False) on which it was seen whole (seen True); overlay frames and unsure frames (seen
None) neither count nor end the run. HudReader revives a tower destroyed on label evidence once
its label has been seen whole again for REVIVE_S (1) s on REVIVE_FRAMES (3) frames: a unit the
detector missed hid it, the tower stands.

THE BATTLE'S LAST TOWER (end_crowns_explained). The tower whose fall ends a battle (a king; any
tower in overtime; one falling in the last seconds of regulation) is never counted by the HUD:
the battle screen goes on the next frame, before its label has been absent 2.5 s. So the end
screen may show more crowns than the HUD: three for a side whose crowns are 3 (a king), and one
more for each of the losing side's princess towers that stood with its label seen within
END_FALL_S (2.5) s of the battle's last frame. Anything else is a misread.

Pixel constants are at 1080x1920 (REF_SIZE), measured on the owner's 747x1328 frames (the
2026-10-31 chicken frames and the 2026-10-29 forest frames, default skin): the enemy fill rows
169-180 under a label centred at 170.5 there, the badge's gold 22 rows and its outline. The
synthetic frames of tests/pipeline/skin_synth.py draw the same geometry. Imports numpy only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import tower_labels as TL
from .hpbars import BAR_CLASSES, ally_mask, dark_mask, enemy_mask
from .overlay import overlay_on

REF_SIZE = (1080, 1920)
PRINCESS_KEYS = TL.PRINCESS_KEYS

# -- the signature (px at 1080x1920) ------------------------------------------
BAR_FULL_PX = 119.0            # the fill track of a princess bar (hud.BAR_FULL_PX)
BAR_TOL_PX = 8.0
#: the fill rows (first, end) from the label box centre (tower_labels.LABEL_REF_PX)
FILL_ROWS_PX = {"enemy": (-3.0, 14.0), "own": (-12.0, 5.0)}
BAND_TALL_PX = (10.0, 26.0)    # the fill band's height
RUN_ROWS_PX = 4.0              # >= this many px of rows hold the bar's whole run
START_PX = 5.0                 # the bar's start: its first 5 px (the digits begin further right)
LEFT_PX = (5.0, 2.0)           # ... and [anchor - 5, anchor - 2), not bar colour
START_FILL_MIN = 0.6
LEFT_FILL_MAX = 0.5            # (a blended badge edge can pass the bar colour)
OUTLINE_GAP_PX = 3.0           # the outline row within 3 px above / below the band
OUTLINE_DARK_MIN = 0.5
#: An outline pixel: every channel <= this. Darker than any bar fill or floor; lighter than
#: hpbars.dark_mask's 135, because our bar's top outline is navy, (45, 108, 141)-(50, 113, 154) on
#: the owner's frames, where the enemy's is (111, 14, 21).
OUTLINE_MAX = 165
RIGHT_PART = (0.70, 0.97)      # the bar's right part: no digits over it
TRACK_GAP_PX = 2.0             # fill -> drained track: a blended column or two between them
GOLD_COLS_PX = (-35.0, -2.0)   # the badge's gold columns from the anchor
GOLD_ROW_MIN = 0.2
OUTLINE_ROW_MIN = 0.7          # a row of the badge's outline: >= 70% of its columns dark
BADGE_TALL_PX = (22.0, 40.0)
BADGE_CENTRE_PX = 8.0          # the badge's centre within this of the band's centre
BADGE_REACH_PX = 16.0          # gold rows counted this far above / below the band (the split)
BADGE_OUTLINE_PX = 6.0         # the badge's dark outline row within this under its gold
#: the badge template: columns from the anchor (its right outline in, the bar's left outline out:
#: that one stays when the badge goes), rows around the gold
BADGE_TEMPLATE_PX = (-38.0, -3.0, 3.0, 4.0)
WINDOW_DU_PX = (-60.0, 60.0)
WINDOW_DV_PX = (-120.0, 60.0)
AMBIGUOUS_PX = 10.0
SAME_PX = 4.0                  # two candidates this close are one
MIN_CONF = 0.8                 # (c): the digit confidence that can refuse a candidate

# -- the lock and the gate ------------------------------------------------------
WINDOW_S = 8.0
WINDOW_FRAMES = 8
LOCK_FRAMES = 8
LOCK_SPAN_S = 1.0
SPARSE_FRAMES = 3
SPARSE_SPAN_S = 3.0
AGREE_PX = 3.0
DEAD_BAND_PX = 3.0
NCC_MIN = 0.7
SHIFT_PX = 3.0                 # the template is matched within +-3 px of the lock
ROW_TOL_PX = 1.0               # the bar start on the locked rows +-1 px
RECHECK_S = 1.0
LOST_S = 3.0
VOTE_AGREE_PX = 8.0            # a detector tower-bar vote this close to the lock matches it
VOTE_FAR_PX = 12.0             # ... and this far is a vote for elsewhere
DET_TRIGGER = 5
MAX_SEARCHES = 3
MAX_RESEARCHES = 3
COVER_SHARE = 0.25             # a unit box over >= 25% of the label covers it
LABEL_AT_LOCK_PX = 16.0        # a detector label box centred this close to the lock's label is there
VOTES_KEPT = 40

STATUS_WAITING = "waiting"     # the clock does not run yet
STATUS_SEARCHING = "searching"
STATUS_LOCKED = "locked"
STATUS_RESEARCH = "re-search"
STATUS_EXPIRED = "expired"     # the window ended without a lock
STATUS_UNLATCHED = "unlatched"
SOURCE_SEARCH, SOURCE_RESEARCH, SOURCE_RELOCK = "search", "re-search", "re-lock"
#: the label back (seen_run): a tower destroyed on label evidence stands again once its label is
#: seen whole this long, on this many frames (HudReader)
REVIVE_S, REVIVE_FRAMES = 1.0, 3
#: the battle's last tower (end_crowns_explained): a label seen this recently may fall with it
END_FALL_S = 2.5


def side_of(key: str) -> str:
    return TL.side_of(key)


def _scale(img: np.ndarray) -> tuple[float, float]:
    return img.shape[1] / float(REF_SIZE[0]), img.shape[0] / float(REF_SIZE[1])


def outline_mask(rgb: np.ndarray) -> np.ndarray:
    """A label's outline (and the bar's drained track): every channel <= OUTLINE_MAX."""
    return np.asarray(rgb)[..., :3].max(axis=-1) <= OUTLINE_MAX


def _fill_mask(rgb: np.ndarray, side: str) -> np.ndarray:
    return ally_mask(rgb) if side == "own" else enemy_mask(rgb)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True."""
    f = np.asarray(flags, dtype=np.int8)
    d = np.diff(np.concatenate(([0], f, [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def ref_band_centre(key: str, label_ref: Sequence[float]) -> float:
    """The reference fill band's centre row (px at 1080x1920) of `key`'s label."""
    a, b = FILL_ROWS_PX[side_of(key)]
    return (float(label_ref[1]) + float(label_ref[3])) / 2.0 + (a + b) / 2.0


# ---------------------------------------------------------------------------
# The signature
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One label found by the search. du, dv: px at 1080x1920 from the reference; the rest in
    the frame's pixels: anchor (the bar fill's first column), end (the bar's end: the fill's,
    or the drained track's), rows (the fill band [r0, r1)), gold (the badge's gold rows
    [g0, g1)), full (the fill reaches the end)."""
    du: float
    dv: float
    anchor: int
    end: int
    rows: tuple[int, int]
    gold: tuple[int, int]
    full: bool

    @property
    def length(self) -> int:
        return self.end - self.anchor


def _bar_rows(F: np.ndarray, D: np.ndarray, sx: float, c_lo: int) -> list[tuple[int, int, int, bool]]:
    """(row, start, end, full) of every window row holding a whole bar run (module doc (b))."""
    lo, hi = (BAR_FULL_PX - BAR_TOL_PX) * sx, (BAR_FULL_PX + BAR_TOL_PX) * sx
    gap = max(1, int(round(TRACK_GAP_PX * sx)))
    extra = max(1, int(round(3.0 * sx)))
    out = []
    for r in range(F.shape[0]):
        row = F[r]
        if not row.any():
            continue
        drow = D[r]
        for s, e in _runs(row):
            if s <= c_lo:                      # cut by the window's left edge
                continue
            n = e - s
            if lo <= n <= hi:
                out.append((r, s, e, True))
            elif 2 <= n < lo:
                j = next((e + k for k in range(gap + 1) if e + k < len(drow) and drow[e + k]), None)
                if j is None:
                    continue
                d_end = j
                while d_end < len(drow) and drow[d_end]:
                    d_end += 1
                if lo <= d_end - s <= hi + extra and d_end < len(drow):
                    out.append((r, s, d_end, False))
    return out


def _fill_at(F: np.ndarray, r: int, a: int, sx: float) -> bool:
    """Row r holds the bar's colour on the first START_PX after anchor a."""
    n = max(2, int(round(START_PX * sx)))
    return 0 <= r < F.shape[0] and a + n <= F.shape[1] and F[r, a:a + n].mean() >= START_FILL_MIN


def _start_ok(F: np.ndarray, r: int, a: int, sx: float) -> bool:
    w = F.shape[1]
    n = max(2, int(round(START_PX * sx)))
    l0, l1 = a - int(round(LEFT_PX[0] * sx)), a - max(1, int(round(LEFT_PX[1] * sx)))
    if not 0 <= r < F.shape[0] or a + n > w or l0 < 0:
        return False
    return (F[r, a:a + n].mean() >= START_FILL_MIN
            and (l1 <= l0 or F[r, l0:l1].mean() <= LEFT_FILL_MAX))


def _dark_near(D: np.ndarray, rows: Iterable[int], c0: int, c1: int) -> bool:
    for r in rows:
        if 0 <= r < D.shape[0] and c1 > c0 and D[r, max(0, c0):c1].mean() >= OUTLINE_DARK_MIN:
            return True
    return False


def _split_groups(gold: np.ndarray, hard: np.ndarray, bridge: int) -> list[tuple[int, int]]:
    """Runs of gold rows (first, last), bridging <= `bridge` other rows, never across a row of
    the badge's dark outline (`hard`): gold above or under a badge (a roof, a pad, flowers)
    stays a group of its own."""
    out: list[tuple[int, int]] = []
    start = 0
    for i in list(np.flatnonzero(hard)) + [len(gold)]:
        out += [(start + lo, start + hi) for lo, hi in TL._groups(list(gold[start:i]), bridge)]
        start = i + 1
    return out


def _badge(G: np.ndarray, D: np.ndarray, a: int, b0: int, b1: int, sx: float, sy: float
           ) -> tuple[int, int] | None:
    """The badge's gold rows [g0, g1) left of anchor `a` for the band [b0, b1), or None."""
    c0, c1 = a + int(round(GOLD_COLS_PX[0] * sx)), a + int(round(GOLD_COLS_PX[1] * sx))
    r0, r1 = b0 - int(round(BADGE_REACH_PX * sy)), b1 + int(round(BADGE_REACH_PX * sy))
    if c0 < 0 or r0 < 0 or r1 > G.shape[0] or c1 <= c0:
        return None
    gold = G[r0:r1, c0:c1].mean(axis=1) >= GOLD_ROW_MIN
    hard = D[r0:r1, c0:c1].mean(axis=1) >= OUTLINE_ROW_MIN
    centre = (b0 + b1) / 2.0
    best = None
    below = max(1, int(round(BADGE_OUTLINE_PX * sy)))
    for lo, hi in _split_groups(gold, hard, max(1, int(round(1.5 * sx)))):
        g0, g1 = r0 + lo, r0 + hi + 1
        tall = (g1 - g0) / sy
        if not BADGE_TALL_PX[0] <= tall <= BADGE_TALL_PX[1]:
            continue
        off = abs((g0 + g1) / 2.0 - centre)
        if off > BADGE_CENTRE_PX * sy:
            continue
        if not _dark_near(D[:, c0:c1], range(g1, min(D.shape[0], g1 + below + 1)), 0, c1 - c0):
            continue
        if best is None or off < best[0]:
            best = (off, g0, g1)
    return None if best is None else (best[1], best[2])


def find_candidates(rgb: np.ndarray, key: str, ref_anchor_px: float, label_ref: Sequence[float], *,
                    digits: Callable[[np.ndarray, str, float, float], tuple[int | None, float]] | None = None,
                    legal_full: Iterable[int] | None = None,
                    window_du: Sequence[float] = WINDOW_DU_PX, window_dv: Sequence[float] = WINDOW_DV_PX
                    ) -> list[Candidate]:
    """Every label of `key` the signature finds in its search window (module doc), nearest the
    reference first. ref_anchor_px: the reference bar fill start (px at 1080); label_ref: the
    reference label box (px at 1080); digits(rgb, key, du, dv) -> (hp, conf) reads the HP digits
    at a shift (check (c)), legal_full the full HPs that pass it."""
    img = np.asarray(rgb)
    h, w = img.shape[:2]
    sx, sy = _scale(img)
    side = side_of(key)
    ax_ref = float(ref_anchor_px) * sx
    bc_ref = ref_band_centre(key, label_ref) * sy
    margin_c = (abs(GOLD_COLS_PX[0]) + 10.0) * sx
    c0 = int(np.floor(ax_ref + window_du[0] * sx - margin_c))
    c1 = int(np.ceil(ax_ref + window_du[1] * sx + (BAR_FULL_PX + BAR_TOL_PX + 10.0) * sx))
    reach = (BADGE_REACH_PX + BAND_TALL_PX[1] / 2.0 + 4.0) * sy
    r0 = int(np.floor(bc_ref + window_dv[0] * sy - reach))
    r1 = int(np.ceil(bc_ref + window_dv[1] * sy + reach))
    cc0, cc1, rr0, rr1 = max(0, c0), min(w, c1), max(0, r0), min(h, r1)
    if cc1 - cc0 < 10 or rr1 - rr0 < 10:
        return []
    win = img[rr0:rr1, cc0:cc1, :3]
    F = _fill_mask(win, side)
    D = dark_mask(win)
    O = outline_mask(win)
    G = TL.gold_mask(win)
    rows = _bar_rows(F, D, sx, 0 if c0 >= 0 else -1)
    if not rows:
        return []
    need = max(3, int(round(RUN_ROWS_PX * sy)))
    tol = max(1, int(round(2.0 * sx)))
    found: list[Candidate] = []
    rows.sort(key=lambda x: (x[1], x[0]))
    groups: list[list[tuple[int, int, int, bool]]] = []
    for item in rows:
        if groups and item[1] - groups[-1][-1][1] <= tol:
            groups[-1].append(item)
        else:
            groups.append([item])
    legal = None if legal_full is None else frozenset(int(v) for v in legal_full)
    for grp in groups:
        grp.sort(key=lambda x: x[0])
        runs: list[list[tuple[int, int, int, bool]]] = [[grp[0]]]
        for item in grp[1:]:
            if item[0] - runs[-1][-1][0] <= 2:
                runs[-1].append(item)
            else:
                runs.append([item])
        for run in runs:
            if len({r for r, *_ in run}) < need:
                continue
            a = int(np.median([s for _, s, _, _ in run]))
            end = int(np.median([e for _, _, e, _ in run]))
            full = sum(1 for *_, f in run if f) * 2 >= len(run)
            top, bot = min(r for r, *_ in run), max(r for r, *_ in run)
            limit = int(round(BAND_TALL_PX[1] * sy))
            while top - 1 >= 0 and bot - top < limit and _fill_at(F, top - 1, a, sx):
                top -= 1
            while bot + 1 < F.shape[0] and bot - top < limit and _fill_at(F, bot + 1, a, sx):
                bot += 1
            b0, b1 = top, bot + 1
            if not BAND_TALL_PX[0] <= (b1 - b0) / sy <= BAND_TALL_PX[1]:
                continue
            g = max(1, int(round(OUTLINE_GAP_PX * sy)))
            n = max(2, int(round(START_PX * sx)))
            above, below = range(b0 - g, b0), range(b1, b1 + g)
            if not (_dark_near(O, above, a, a + n) and _dark_near(O, below, a, a + n)):
                continue
            if full:
                q0, q1 = a + int(RIGHT_PART[0] * (end - a)), a + int(RIGHT_PART[1] * (end - a))
                if not (_dark_near(O, above, q0, q1) and _dark_near(O, below, q0, q1)):
                    continue
            gold = _badge(G, D, a, b0, b1, sx, sy)
            if gold is None:
                continue
            du = (cc0 + a) / sx - float(ref_anchor_px)
            dv = (rr0 + (b0 + b1) / 2.0) / sy - ref_band_centre(key, label_ref)
            if not (window_du[0] <= du <= window_du[1] and window_dv[0] <= dv <= window_dv[1]):
                continue
            if digits is not None and full and legal is not None:
                hp, conf = digits(img, key, du, dv)
                if hp is not None and conf >= MIN_CONF and int(hp) not in legal:
                    continue
            cand = Candidate(round(du, 2), round(dv, 2), cc0 + a, cc0 + end, (rr0 + b0, rr0 + b1),
                             (rr0 + gold[0], rr0 + gold[1]), bool(full))
            if any(abs(c.du - cand.du) <= SAME_PX and abs(c.dv - cand.dv) <= SAME_PX for c in found):
                continue
            found.append(cand)
    found.sort(key=lambda c: (c.du ** 2 + c.dv ** 2, c.dv))
    return found


def find_label(rgb: np.ndarray, key: str, ref_anchor_px: float, label_ref: Sequence[float], **kw
               ) -> Candidate | None:
    """The nearest candidate (find_candidates), None when there is none or a second one lies
    about as near (within AMBIGUOUS_PX) somewhere else."""
    cands = find_candidates(rgb, key, ref_anchor_px, label_ref, **kw)
    if not cands:
        return None
    if len(cands) > 1:
        d0 = float(np.hypot(cands[0].du, cands[0].dv))
        d1 = float(np.hypot(cands[1].du, cands[1].dv))
        if d1 - d0 < AMBIGUOUS_PX:
            return None
    return cands[0]


# ---------------------------------------------------------------------------
# Matching at a lock
# ---------------------------------------------------------------------------
def _zero_mean(a: np.ndarray) -> np.ndarray:
    """`a` as float32 minus its mean per colour channel (a flat colour becomes all 0)."""
    x = np.asarray(a, dtype=np.float32)
    if x.ndim == 3:
        return x - x.reshape(-1, x.shape[2]).mean(axis=0)
    return x - x.mean()


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two equal-shape arrays, each channel zero-meaned on its
    own (0.0 when either is flat)."""
    if np.asarray(a).size == 0 or np.shape(a) != np.shape(b):
        return 0.0
    x, y = _zero_mean(a).ravel(), _zero_mean(b).ravel()
    den = float(np.sqrt(float(np.dot(x, x)) * float(np.dot(y, y))))
    return 0.0 if den < 1e-6 else float(np.dot(x, y)) / den


PART_SHARE = 0.4               # a part of the template: its top, bottom, left or right 40%


def _halves(t: np.ndarray) -> list[tuple[slice, slice]]:
    """The template's parts (PART_SHARE of it from each side): what shows when a unit's bar or
    body covers the rest."""
    h, w = t.shape[:2]
    ph, pw = max(2, int(round(h * PART_SHARE))), max(2, int(round(w * PART_SHARE)))
    return [(slice(0, ph), slice(0, w)), (slice(h - ph, h), slice(0, w)),
            (slice(0, h), slice(0, pw)), (slice(0, h), slice(w - pw, w))]


class _Prepared:
    """A template (or one of its halves) zero-meaned once for ncc against many patches."""

    def __init__(self, t: np.ndarray):
        self.z = _zero_mean(t)
        self.norm = float(np.sqrt(float((self.z * self.z).sum())))

    def score(self, patch: np.ndarray) -> float:
        p = _zero_mean(patch)
        den = self.norm * float(np.sqrt(float((p * p).sum())))
        return 0.0 if den < 1e-6 else float((self.z * p).sum()) / den


def match_template(img: np.ndarray, tmpl: np.ndarray, x0: int, y0: int, shift: int, *,
                   halves_below: float = 2.0) -> tuple[float, float]:
    """(best NCC of the whole template, best NCC of any of its parts: _halves) with its top-left
    within +-`shift` px of (x0, y0). The parts are scored only when the whole scores under
    `halves_below` (else the part score is the whole's)."""
    th, tw = tmpl.shape[:2]
    h, w = img.shape[:2]
    whole = _Prepared(tmpl)
    spots = [(y0 + dy, x0 + dx) for dy in range(-shift, shift + 1) for dx in range(-shift, shift + 1)
             if y0 + dy >= 0 and x0 + dx >= 0 and y0 + dy + th <= h and x0 + dx + tw <= w]
    best = max((whole.score(img[y:y + th, x:x + tw, :3]) for y, x in spots), default=0.0)
    if best >= halves_below:
        return max(0.0, best), max(0.0, best)
    parts = [(ys, xs, _Prepared(tmpl[ys, xs])) for ys, xs in _halves(tmpl)]
    half = max((pp.score(img[y:y + th, x:x + tw, :3][ys, xs]) for y, x in spots for ys, xs, pp in parts),
               default=0.0)
    return max(0.0, best), max(0.0, half)


# ---------------------------------------------------------------------------
# The calibrator
# ---------------------------------------------------------------------------
@dataclass
class TowerLock:
    """A princess label locked for this battle. du, dv: px at 1080x1920 from the reference (the
    median of the agreeing votes); the rest in the pixels of a frame of `size`: the bar's anchor
    and end, the fill band's rows, the badge template's top-left and pixels, the fill colour per
    band row (for P1-3), the digit box and its pixels (for P1-5)."""
    key: str
    du: float
    dv: float
    source: str
    locked_t: float
    frames: int
    size: tuple[int, int]
    anchor: int
    end: int
    rows: tuple[int, int]
    gold: tuple[int, int]
    badge_xy: tuple[int, int]
    badge: np.ndarray = field(repr=False)
    fill_rgb: np.ndarray = field(repr=False)
    digit_box: tuple[int, int, int, int] | None = None
    digit_patch: np.ndarray | None = field(default=None, repr=False)

    def bar_box_ref(self) -> tuple[float, float, float, float]:
        """The bar (anchor to end, the fill band grown 6 px) in px at 1080x1920."""
        sx, sy = self.size[0] / float(REF_SIZE[0]), self.size[1] / float(REF_SIZE[1])
        return self.anchor / sx, self.rows[0] / sy - 6.0, self.end / sx, self.rows[1] / sy + 6.0

    def label_box_ref(self) -> tuple[float, float, float, float]:
        """The label (badge to the bar's end, the badge's rows) in px at 1080x1920."""
        sx, sy = self.size[0] / float(REF_SIZE[0]), self.size[1] / float(REF_SIZE[1])
        x0, y0 = self.badge_xy
        th = self.badge.shape[0]
        return x0 / sx, y0 / sy, self.end / sx, (y0 + th) / sy


@dataclass
class CalibRead:
    """One frame's view of one princess tower (TowerCalibrator.update)."""
    status: str
    calibrated: bool
    confirmed: bool | None = None          # the read gate (None: not calibrated)
    seen: bool | None = None               # badge_seen, P1-4 (None: unsure or not calibrated)
    ncc: float | None = None
    bar_start: bool | None = None
    candidate: tuple[float, float] | None = None     # this frame's search result (du, dv)
    covered: bool = False                  # a unit box covers the label
    absent_s: float | None = None          # the label's absence at the lock, s (None: not calibrated)
    overlay: bool = False                  # an overlay frame: nothing checked


@dataclass
class CalibFrame:
    t: float
    running: bool | None
    overlay: bool | None
    reads: dict[str, CalibRead] = field(default_factory=dict)


@dataclass
class _Tower:
    status: str = STATUS_WAITING
    lock: TowerLock | None = None
    votes: deque = field(default_factory=lambda: deque(maxlen=VOTES_KEPT))
    searched: int = 0
    last_match: float | None = None
    research_t0: float | None = None
    research_found: bool = False
    research_tick: float | None = None
    failed: int = 0
    researches: int = 0
    relocks: int = 0
    unlatches: int = 0
    det_far: deque = field(default_factory=lambda: deque(maxlen=VOTES_KEPT))
    det_votes: int = 0
    absent_s: float = 0.0                  # the label's absence at the lock (module doc)
    absent_on: bool = False                # a count runs: an absent frame since the label showed
    absent_n: int = 0                      # ... its absent frames
    covered_n: int = 0                     # ... and its covered ones
    absent_max: float = 0.0                # the longest count this battle
    prev_t: float | None = None            # this tower's previous frame (any kind)
    prev_kind: str | None = None           # "absent", "covered", "overlay" or None (it showed)
    research_searched: bool = False        # a search ran in this re-search second
    ghost: TowerLock | None = None         # the unlatched lock, still checked (re-lock)
    ghost_t0: float | None = None
    ghost_n: int = 0
    seen_t0: float | None = None           # the label seen whole since (seen_run)
    seen_t1: float | None = None
    seen_n: int = 0

    def clear_absence(self) -> None:
        self.absent_s, self.absent_on = 0.0, False
        self.absent_n = self.covered_n = 0

    def clear_seen(self) -> None:
        self.seen_t0 = self.seen_t1 = None
        self.seen_n = 0


class TowerCalibrator:
    """Each princess tower's label lock for one battle (module doc). update() once a frame with
    the frame, its time, whether the battle clock runs and the detector's evidence; reset() at
    each battle's start.

    label_refs: princess key -> the reference label box (px at 1080; HudRegions.label_refs);
    anchors: princess key -> the reference bar fill start (px at 1080: the bar region's left
    edge + its margin); a key missing from either is never calibrated. legal_full: key -> the
    full HPs check (c) accepts (hud.LEGAL_TOWER_MAX_HP)."""

    def __init__(self, label_refs: Mapping[str, Sequence[float]], anchors: Mapping[str, float], *,
                 legal_full: Mapping[str, Iterable[int]] | None = None, window_s: float = WINDOW_S,
                 window_frames: int = WINDOW_FRAMES, lock_frames: int = LOCK_FRAMES,
                 lock_span_s: float = LOCK_SPAN_S, sparse_frames: int = SPARSE_FRAMES,
                 sparse_span_s: float = SPARSE_SPAN_S, agree_px: float = AGREE_PX,
                 dead_band_px: float = DEAD_BAND_PX, ncc_min: float = NCC_MIN,
                 recheck_s: float = RECHECK_S, lost_s: float = LOST_S, max_searches: int = MAX_SEARCHES,
                 max_researches: int = MAX_RESEARCHES, det_trigger: int = DET_TRIGGER,
                 use_overlay: bool = True):
        self.label_refs = {k: tuple(float(c) for c in v) for k, v in label_refs.items() if k in PRINCESS_KEYS}
        self.anchors = {k: float(v) for k, v in anchors.items() if k in PRINCESS_KEYS and v is not None}
        self.keys = tuple(k for k in PRINCESS_KEYS if k in self.label_refs and k in self.anchors)
        self.legal_full = {k: frozenset(int(x) for x in v) for k, v in (legal_full or {}).items()}
        self.window_s, self.window_frames = float(window_s), max(1, int(window_frames))
        self.lock_frames, self.lock_span_s = max(1, int(lock_frames)), float(lock_span_s)
        self.sparse_frames, self.sparse_span_s = max(1, int(sparse_frames)), float(sparse_span_s)
        self.agree_px, self.dead_band_px, self.ncc_min = float(agree_px), float(dead_band_px), float(ncc_min)
        self.recheck_s, self.lost_s = float(recheck_s), float(lost_s)
        self.max_searches, self.max_researches = max(1, int(max_searches)), max(0, int(max_researches))
        self.det_trigger = max(1, int(det_trigger))
        self.use_overlay = bool(use_overlay)
        self.reset()

    def reset(self) -> None:
        self.t_run0: float | None = None
        self.towers: dict[str, _Tower] = {k: _Tower() for k in PRINCESS_KEYS}
        self.last: CalibFrame | None = None

    # -- outputs ----------------------------------------------------------------
    def lock(self, key: str) -> TowerLock | None:
        st = self.towers.get(key)
        return None if st is None else st.lock

    def calibrated(self, key: str) -> bool:
        return self.lock(key) is not None

    def _dead(self, d: float) -> float:
        return 0.0 if abs(d) < self.dead_band_px else float(d)

    def placement(self) -> dict[str, tuple[float, float]]:
        """Calibrated key -> the shift (du, dv) px at 1080 its boxes are read at (the lock, an
        axis under the dead band at 0)."""
        out = {}
        for k in PRINCESS_KEYS:
            lk = self.lock(k)
            if lk is not None:
                out[k] = (self._dead(lk.du), self._dead(lk.dv))
        return out

    def summary(self, frame: CalibFrame | None = None) -> dict[str, dict[str, Any]]:
        """Per princess key the record HudReport writes (tools/tower_audit.py reads it):
        calibrated, du / dv (the lock, px at 1080), source, state (STATUS_*), confirmed (this
        frame's gate), seen (badge_seen), covered, and once locked locked_t (frame time) and
        frames (the votes it took)."""
        fr = frame if frame is not None else self.last
        out: dict[str, dict[str, Any]] = {}
        for k in PRINCESS_KEYS:
            st = self.towers[k]
            lk = st.lock
            r = fr.reads.get(k) if fr is not None else None
            rec: dict[str, Any] = {"calibrated": lk is not None, "state": st.status}
            if lk is not None:
                rec.update(du=round(lk.du, 1), dv=round(lk.dv, 1), source=lk.source,
                           locked_t=round(lk.locked_t, 3), frames=lk.frames)
            else:
                rec.update(du=0.0, dv=0.0, source="none")
            if r is not None:
                if r.confirmed is not None:
                    rec["confirmed"] = r.confirmed
                if r.seen is not None:
                    rec["seen"] = r.seen
                if r.covered:
                    rec["covered"] = True
                if r.absent_s:
                    rec["absent_s"] = round(r.absent_s, 2)
            out[k] = rec
        return out

    def info(self) -> dict[str, dict[str, Any]]:
        """The battle's calibration per princess key (vision_play's battle_end record)."""
        out = {}
        for k in PRINCESS_KEYS:
            st = self.towers[k]
            lk = st.lock
            out[k] = {"calibrated": lk is not None, "state": st.status,
                      "du": None if lk is None else round(lk.du, 1),
                      "dv": None if lk is None else round(lk.dv, 1),
                      "source": None if lk is None else lk.source,
                      "locked_t": None if lk is None else round(lk.locked_t, 3),
                      "frames": None if lk is None else lk.frames,
                      "relocks": st.relocks, "unlatches": st.unlatches, "researches": st.researches,
                      "absent_max_s": round(st.absent_max, 2)}
        return out

    def absent(self) -> dict[str, float | None]:
        """The last frame's label absence per princess key (s at the lock; None: not
        calibrated), what TowerHpTracker.update's label_absent takes."""
        fr = self.last
        return {k: (None if fr is None or k not in fr.reads else fr.reads[k].absent_s) for k in PRINCESS_KEYS}

    def seen(self) -> dict[str, bool | None]:
        """The last frame's badge_seen per princess key (P1-4)."""
        fr = self.last
        return {k: (None if fr is None or k not in fr.reads else fr.reads[k].seen) for k in PRINCESS_KEYS}

    def seen_run(self, key: str) -> tuple[float, int]:
        """(seconds, frames) `key`'s label has been seen whole at its lock since it was last not
        seen (module doc: THE LABEL BACK); (0.0, 0) without a lock or such a frame."""
        st = self.towers.get(key)
        if st is None or st.lock is None or st.seen_t0 is None or st.seen_t1 is None:
            return 0.0, 0
        return st.seen_t1 - st.seen_t0, st.seen_n

    # -- the frame --------------------------------------------------------------
    def update(self, rgb: np.ndarray, t: float, *, running: bool | None = None, evidence: Any = None,
               overlay: bool | None = None, destroyed: Iterable[str] = (), low: Iterable[str] = (),
               digits: Callable[[np.ndarray, str, float, float], tuple[int | None, float]] | None = None
               ) -> CalibFrame:
        """One frame (module doc). running: the battle clock runs (None: not followed, search
        from the first frame; once it ran, False changes nothing: a rejected clock reading
        mid-battle does not stop the gate); evidence: the frame's tower_labels.TowerEvidence (None: no
        detector); overlay: the deploy overlay is on (None: overlay.overlay_on decides);
        destroyed: keys read destroyed (never re-checked); low: keys read <= 0.75 (the label
        rule's limit: while their absence count runs they are not re-checked, module doc);
        digits: check (c)."""
        img = np.asarray(rgb)
        t = float(t)
        size = (int(img.shape[1]), int(img.shape[0]))
        if running is False and self.t_run0 is None:           # it latches, like the tower tracker
            fr = CalibFrame(t, running, None, {k: CalibRead(self.towers[k].status, self.calibrated(k))
                                               for k in PRINCESS_KEYS})
            self.last = fr
            return fr
        if self.t_run0 is None:
            self.t_run0 = t
        ov = overlay
        if ov is None and self.use_overlay:
            ov = overlay_on(img)
        fr = CalibFrame(t, running, ov)
        gone, lows = set(destroyed or ()), set(low or ())
        for key in PRINCESS_KEYS:
            fr.reads[key] = self._update_key(key, img, t, size, evidence, bool(ov), key in gone, digits,
                                             low=key in lows)
            self.towers[key].prev_t = t
        self.last = fr
        return fr

    def _update_key(self, key: str, img: np.ndarray, t: float, size: tuple[int, int], evidence: Any,
                    overlay: bool, destroyed: bool, digits, *, low: bool = False) -> CalibRead:
        st = self.towers[key]
        if key not in self.keys:
            st.status = STATUS_EXPIRED
            return CalibRead(st.status, False)
        if st.lock is not None and st.lock.size != size:          # another frame size: start over
            st.lock, st.status = None, STATUS_SEARCHING
            st.votes.clear()
            st.clear_absence()
            st.clear_seen()
        if st.ghost is not None and st.ghost.size != size:
            st.ghost = None
        if st.status == STATUS_WAITING:
            st.status = STATUS_SEARCHING
        covered = self._covered(key, evidence)
        read = CalibRead(st.status, st.lock is not None, covered=covered)
        if overlay:                            # P1-2: nothing checked, searched or counted (paused)
            read.overlay = True
            st.prev_kind = "overlay"
            if st.lock is not None:
                read.absent_s = st.absent_s
            return read
        if destroyed and st.status == STATUS_RESEARCH:
            st.status, st.research_t0 = STATUS_LOCKED, None          # never re-checked
        if st.lock is not None:
            self._gate(st.lock, img, read)
        vote = None if evidence is None else getattr(evidence, "placement", {}).get(key)
        bodies = set() if evidence is None else set(getattr(evidence, "bodies", ()) or ())
        if st.lock is not None:
            matched = bool(read.confirmed)
            if vote is not None:
                off = float(np.hypot(vote[0] - st.lock.du, vote[1] - st.lock.dv))
                if off <= VOTE_AGREE_PX:
                    matched = True
                elif off >= VOTE_FAR_PX:
                    st.det_far.append(t)
            if st.last_match is None:
                st.last_match = st.lock.locked_t
            if matched:
                st.last_match = t
                if st.status == STATUS_RESEARCH and read.confirmed:
                    st.status, st.research_t0 = STATUS_LOCKED, None          # it shows again
                    st.votes.clear()
        elif vote is not None and st.status in (STATUS_EXPIRED, STATUS_UNLATCHED):
            st.det_votes += 1
        if st.lock is not None:
            self._count_absence(st, key, t, read, evidence, covered)
            self._note_seen(st, t, read)
        # a tower read <= 0.75 whose label is gone may be falling: its lock stays (module doc)
        falling = low and st.lock is not None and st.absent_on
        if falling and st.status == STATUS_RESEARCH:
            st.status, st.research_t0 = STATUS_LOCKED, None
            st.votes.clear()
        # what to search
        since = 0.0 if self.t_run0 is None else t - self.t_run0
        in_window = (st.lock is None and st.status == STATUS_SEARCHING
                     and (since <= self.window_s or st.searched < self.window_frames))
        if st.lock is None and st.status == STATUS_SEARCHING and not in_window:
            st.status = STATUS_EXPIRED
            st.votes.clear()
        if (st.lock is not None and st.status == STATUS_LOCKED and not destroyed and not covered
                and not falling):
            far = sum(1 for x in st.det_far if t - x <= self.lost_s)
            if t - st.last_match >= self.lost_s and (key in bodies or far >= self.det_trigger):
                self._start_research(st, t)
        if st.lock is None and st.status == STATUS_UNLATCHED and not destroyed and not covered:
            self._check_ghost(st, img, t, read)
        if (st.lock is None and st.status in (STATUS_EXPIRED, STATUS_UNLATCHED) and not destroyed
                and st.det_votes >= self.det_trigger and st.researches < self.max_researches):
            st.det_votes = 0
            self._start_research(st, t)
        if in_window or st.status == STATUS_RESEARCH:
            if not (covered and st.status == STATUS_RESEARCH):
                cand = self._search(key, img, digits if st.status != STATUS_RESEARCH else None)
                if in_window:
                    st.searched += 1
                if st.status == STATUS_RESEARCH:
                    st.research_searched = True
                if cand is not None:
                    read.candidate = (cand.du, cand.dv)
                    st.votes.append((t, cand.du, cand.dv, cand))
                    if st.status == STATUS_RESEARCH:
                        st.research_found = True
                    self._try_lock(key, st, img, t, size, cand)
            if st.status == STATUS_RESEARCH:
                self._tick_research(st, t)
        read.status = st.status
        read.calibrated = st.lock is not None
        if st.lock is not None and read.confirmed is None:          # locked on this frame
            self._gate(st.lock, img, read)
        if st.lock is None:
            read.confirmed = read.seen = read.ncc = read.bar_start = None
            st.clear_absence()
            st.clear_seen()
        else:
            read.absent_s = st.absent_s
        return read

    # -- the label's absence (module doc) -------------------------------------------
    def _count_absence(self, st: _Tower, key: str, t: float, read: CalibRead, evidence: Any,
                       covered: bool) -> None:
        """The absence count (module doc). An absent frame adds its time since the tower's previous
        frame; after a covered frame only while the count's absent frames are at least twice its
        covered ones: under a unit the detector boxes on most frames, the frames it misses are
        no evidence (a Balloon over a standing tower), so there only absent frames in a row count."""
        if read.confirmed or self.label_box_at_lock(key, evidence) or (read.seen is None and not covered):
            st.clear_absence()                 # the label, or part of it, shows
            st.prev_kind = None
        elif covered:
            if st.absent_on:                   # paused: a unit over the label (the count is kept)
                st.covered_n += 1
            st.prev_kind = "covered"
        else:
            if st.absent_on:
                st.absent_n += 1
                if st.prev_t is not None and (st.prev_kind != "covered" or st.absent_n >= 2 * st.covered_n):
                    st.absent_s += max(0.0, t - st.prev_t)   # this frame's own time
            else:
                st.absent_on, st.absent_n = True, 1
            st.prev_kind = "absent"
            st.absent_max = max(st.absent_max, st.absent_s)

    def _note_seen(self, st: _Tower, t: float, read: CalibRead) -> None:
        """The label-back run (module doc): seen True counts, False ends it, None waits."""
        if read.seen is True:
            if st.seen_t0 is None:
                st.seen_t0, st.seen_n = t, 0
            st.seen_t1 = t
            st.seen_n += 1
        elif read.seen is False:
            st.clear_seen()

    def _check_ghost(self, st: _Tower, img: np.ndarray, t: float, read: CalibRead) -> None:
        """An unlatched tower's old lock (module doc: re-lock): seen whole there on as many frames
        as a lock needs, it is the tower's lock again."""
        g = st.ghost
        if g is None:
            return
        probe = CalibRead(st.status, False, covered=read.covered)
        self._gate(g, img, probe)
        if probe.seen is True:
            if st.ghost_t0 is None:
                st.ghost_t0, st.ghost_n = t, 0
            st.ghost_n += 1
            span = t - st.ghost_t0
            if ((st.ghost_n >= self.lock_frames and span >= self.lock_span_s)
                    or (st.ghost_n >= self.sparse_frames and span >= self.sparse_span_s)):
                st.lock = replace(g, source=SOURCE_RELOCK, locked_t=t, frames=st.ghost_n)
                st.ghost, st.ghost_t0, st.ghost_n = None, None, 0
                st.status, st.last_match, st.research_t0 = STATUS_LOCKED, t, None
                st.relocks += 1
                st.clear_absence()
                st.clear_seen()
                st.votes.clear()
                st.det_far.clear()
                st.det_votes = 0
        elif probe.seen is False:
            st.ghost_t0, st.ghost_n = None, 0

    def label_box_at_lock(self, key: str, evidence: Any) -> bool:
        """The detector boxes `key`'s label at its lock: a placement vote within VOTE_FAR_PX, or
        a label box centred on the lock's label grown LABEL_AT_LOCK_PX."""
        lk = self.lock(key)
        if lk is None or evidence is None:
            return False
        vote = (getattr(evidence, "placement", None) or {}).get(key)
        if vote is not None and float(np.hypot(vote[0] - lk.du, vote[1] - lk.dv)) < VOTE_FAR_PX:
            return True
        box = (getattr(evidence, "labels", None) or {}).get(key)
        if box is None:
            return False
        x0, y0, x1, y1 = lk.label_box_ref()
        g = LABEL_AT_LOCK_PX
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        return x0 - g <= cx <= x1 + g and y0 - g <= cy <= y1 + g

    # -- searching and locking ----------------------------------------------------
    def _search(self, key: str, img: np.ndarray, digits) -> Candidate | None:
        return find_label(img, key, self.anchors[key], self.label_refs[key], digits=digits,
                          legal_full=self.legal_full.get(key))

    def _start_research(self, st: _Tower, t: float) -> None:
        st.status = STATUS_RESEARCH
        st.research_t0, st.research_tick = t, t
        st.research_found = st.research_searched = False
        st.failed = 0
        st.researches += 1
        st.votes.clear()
        st.det_far.clear()

    def _tick_research(self, st: _Tower, t: float) -> None:
        """One re-search second: MAX_SEARCHES in a row in which a search ran and found no
        candidate remove the lock; a second without a search (covered) counts neither way."""
        if st.status != STATUS_RESEARCH or st.research_tick is None or t - st.research_tick < self.recheck_s:
            return
        st.research_tick = t
        if st.research_found:
            st.failed = 0
        elif st.research_searched:
            st.failed += 1
        st.research_found = st.research_searched = False
        if st.failed >= self.max_searches:
            if st.lock is not None:
                st.unlatches += 1
                st.ghost, st.ghost_t0, st.ghost_n = st.lock, None, 0
            st.lock = None
            st.clear_absence()
            st.clear_seen()
            st.status = STATUS_UNLATCHED
            st.research_t0 = None
            st.votes.clear()

    def _try_lock(self, key: str, st: _Tower, img: np.ndarray, t: float, size: tuple[int, int],
                  cand: Candidate) -> None:
        votes = list(st.votes)
        best: list = []
        for _, du, dv, _ in votes:
            agree = [v for v in votes if abs(v[1] - du) <= self.agree_px and abs(v[2] - dv) <= self.agree_px]
            if len(agree) > len(best):
                best = agree
        if not best or not any(v[3] is cand for v in best):
            return
        span = best[-1][0] - best[0][0]
        if not ((len(best) >= self.lock_frames and span >= self.lock_span_s)
                or (len(best) >= self.sparse_frames and span >= self.sparse_span_s)):
            return
        du = float(np.median([v[1] for v in best]))
        dv = float(np.median([v[2] for v in best]))
        source = SOURCE_RESEARCH if st.status == STATUS_RESEARCH else SOURCE_SEARCH
        if st.lock is not None:
            st.relocks += 1
        st.lock = self._make_lock(key, img, t, size, cand, du, dv, source, len(best))
        st.ghost, st.ghost_t0, st.ghost_n = None, None, 0
        st.clear_absence()
        st.clear_seen()
        st.status = STATUS_LOCKED
        st.last_match = t
        st.research_t0 = None
        st.det_far.clear()
        st.votes.clear()

    def _make_lock(self, key: str, img: np.ndarray, t: float, size: tuple[int, int], cand: Candidate,
                   du: float, dv: float, source: str, frames: int) -> TowerLock:
        sx, sy = _scale(img)
        h, w = img.shape[:2]
        x0 = max(0, cand.anchor + int(round(BADGE_TEMPLATE_PX[0] * sx)))
        x1 = min(w, cand.anchor + int(round(BADGE_TEMPLATE_PX[1] * sx)))
        y0 = max(0, cand.gold[0] - int(round(BADGE_TEMPLATE_PX[2] * sy)))
        y1 = min(h, cand.gold[1] + int(round(BADGE_TEMPLATE_PX[3] * sy)))
        badge = img[y0:y1, x0:x1, :3].astype(np.float32).copy()
        r0, r1 = cand.rows
        q0 = cand.anchor + int(RIGHT_PART[0] * cand.length) if cand.full else cand.anchor
        q1 = (cand.anchor + int(RIGHT_PART[1] * cand.length) if cand.full
              else cand.anchor + max(2, int(round(START_PX * sx))))
        fill = np.median(img[r0:r1, q0:q1, :3].astype(np.float32), axis=1) if q1 > q0 else np.zeros((0, 3))
        digit_box = digit_patch = None
        side = side_of(key)
        top = (-22.0 if side == "enemy" else -3.0) - (FILL_ROWS_PX[side][0] + FILL_ROWS_PX[side][1]) / 2.0
        cy = (r0 + r1) / 2.0
        dy0, dy1 = int(round(cy + (top - 4.0) * sy)), int(round(cy + (top + 25.0) * sy))
        dx0, dx1 = cand.anchor, min(w, cand.anchor + int(round(100.0 * sx)))
        if 0 <= dy0 < dy1 <= h and dx1 > dx0:
            digit_box = (dx0, dy0, dx1, dy1)
            digit_patch = img[dy0:dy1, dx0:dx1, :3].copy()
        return TowerLock(key, du, dv, source, t, frames, size, cand.anchor, cand.end, cand.rows, cand.gold,
                         (x0, y0), badge, fill, digit_box, digit_patch)

    # -- the gate ---------------------------------------------------------------------
    def _gate(self, lk: TowerLock, img: np.ndarray, read: CalibRead) -> None:
        sx, sy = _scale(img)
        shift = max(1, int(round(SHIFT_PX * sx)))
        best, half = match_template(img, lk.badge, lk.badge_xy[0], lk.badge_xy[1], shift,
                                    halves_below=self.ncc_min)
        start = self.bar_start(lk, img)
        read.ncc = round(best, 3)
        read.bar_start = start
        badge = best >= self.ncc_min
        read.confirmed = bool(badge or start)
        if badge and start:
            read.seen = True
        elif badge or start or half >= self.ncc_min or read.covered:
            read.seen = None                   # part of it shows, or a unit covers it
        else:
            read.seen = False

    def bar_start(self, lk: TowerLock, img: np.ndarray) -> bool:
        """The bar starts at the lock's anchor on its rows +-1 px (module doc: the read gate)."""
        sx, sy = _scale(img)
        side = side_of(lk.key)
        tol = max(1, int(round(ROW_TOL_PX * sy)))
        r0, r1 = lk.rows
        a = lk.anchor
        n = max(2, int(round(START_PX * sx)))
        lpx = int(round(LEFT_PX[0] * sx))
        h, w = img.shape[:2]
        reach = tol + int(round(OUTLINE_GAP_PX * sy)) + 1
        y0, y1 = max(0, r0 - reach), min(h, r1 + reach)
        x0, x1 = max(0, a - lpx - 1), min(w, a + n + 1)
        if y1 <= y0 or x1 <= x0:
            return False
        crop = img[y0:y1, x0:x1, :3]
        F = _fill_mask(crop, side)
        O = outline_mask(crop)
        ac = a - x0
        best = 0
        for off in range(-tol, tol + 1):
            rows = range(r0 + off - y0, r1 + off - y0)
            ok = sum(1 for r in rows if _start_ok(F, r, ac, sx))
            best = max(best, ok)
        if best * 2 < (r1 - r0):
            return False
        g = max(1, int(round(OUTLINE_GAP_PX * sy))) + tol
        if side == "enemy":                    # the digits sit over the top: the outline below
            rows = range(r1 - y0, r1 - y0 + g)
        else:
            rows = range(r0 - y0 - g, r0 - y0)
        return _dark_near(O, rows, ac, ac + n)

    def _covered(self, key: str, evidence: Any) -> bool:
        """A unit box (TowerEvidence.others) over >= COVER_SHARE of the label: at the lock, else
        at the reference label."""
        others = list(getattr(evidence, "others", None) or ()) if evidence is not None else []
        if not others:
            return False
        lk = self.lock(key)
        box = lk.label_box_ref() if lk is not None else self.label_refs.get(key)
        if box is None:
            return False
        x0, y0, x1, y1 = box
        area = max(1e-6, (x1 - x0) * (y1 - y0))
        bar = None if lk is None else lk.bar_box_ref()
        for name, b in others:
            if bar is not None and name in BAR_CLASSES and bar[0] <= (b[0] + b[2]) / 2.0 <= bar[2] \
                    and bar[1] <= (b[1] + b[3]) / 2.0 <= bar[3]:
                continue                       # the tower's own bar boxed as a unit bar
            ix = max(0.0, min(x1, b[2]) - max(x0, b[0]))
            iy = max(0.0, min(y1, b[3]) - max(y0, b[1]))
            if ix * iy >= COVER_SHARE * area:
                return True
        return False


# ---------------------------------------------------------------------------
# The battle's last tower (module doc)
# ---------------------------------------------------------------------------
def could_fall_at_end(absent: Mapping[str, float | None], destroyed: Iterable[str] = (), *,
                      end_fall_s: float = END_FALL_S) -> dict[str, list[str]]:
    """{"own": [...], "enemy": [...]}: the princess keys whose fall the end screen may count
    although the HUD did not, per side of the CROWNS they give ("own": the enemy's towers).
    absent: the battle's last frame's label absence per key (TowerCalibrator.absent(); None:
    not calibrated, never counted); destroyed: the keys the HUD destroyed."""
    gone = set(destroyed or ())
    out: dict[str, list[str]] = {"own": [], "enemy": []}
    for key in PRINCESS_KEYS:
        a = absent.get(key)
        if key in gone or a is None or float(a) >= float(end_fall_s):
            continue
        out["own" if side_of(key) == "enemy" else "enemy"].append(key)
    return out


def end_crowns_explained(hud: Sequence[int], end: Sequence[int], could_fall: Mapping[str, Sequence[str]]
                         ) -> tuple[bool, str]:
    """(explained, why) the end screen's crowns `end` against the HUD's `hud` (own, enemy) at
    the battle's end (module doc: THE BATTLE'S LAST TOWER). could_fall: could_fall_at_end().
    why: "" when they agree, else what explains the difference ("a king fell (3 crowns)", "the
    battle ended as enemy_right fell") or, unexplained, what does not fit."""
    whys: list[str] = []
    ok = True
    for i, side in enumerate(("own", "enemy")):
        h, e = int(hud[i]), int(end[i])
        if e == h:
            continue
        cand = list(could_fall.get(side, ()) or ())
        if e < h:
            ok = False
            whys.append(f"{side} crowns: the HUD counted {h}, the end screen shows {e}")
        elif e == 3:
            whys.append(f"{side} crowns: a king fell (3 crowns)")
        elif e - h <= len(cand):
            whys.append(f"{side} crowns: the battle ended as " + " or ".join(cand) + " fell")
        else:
            ok = False
            whys.append(f"{side} crowns: {e - h} more on the end screen than the HUD counted, "
                        + (f"only {', '.join(cand)} could fall at the end" if cand
                           else "and no tower's label was seen in the battle's last 2.5 s"))
    return ok, "; ".join(whys)


__all__ = [
    "BAR_FULL_PX", "Candidate", "CalibFrame", "CalibRead", "END_FALL_S", "FILL_ROWS_PX", "NCC_MIN",
    "REVIVE_FRAMES", "REVIVE_S", "SOURCE_RELOCK", "SOURCE_RESEARCH", "SOURCE_SEARCH", "STATUS_EXPIRED",
    "STATUS_LOCKED", "STATUS_RESEARCH", "STATUS_SEARCHING", "STATUS_UNLATCHED", "STATUS_WAITING",
    "TowerCalibrator", "TowerLock", "could_fall_at_end", "end_crowns_explained", "find_candidates",
    "find_label", "match_template", "ncc", "ref_band_centre",
]
