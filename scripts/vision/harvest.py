"""Harvest HUD templates from a recorded friendly: clock-sweep, harvest, harvest-label (V07).

`python -m scripts.vision.templates digits|hand|...` stores templates from one
screenshot at a time. These subcommands of the same CLI work on a whole
recording (scripts/vision/record.py: <session>/frames.jsonl plus
images/<8 digits>.jpg, one image every 5th frame, about 11 a second at 57 fps):

    python -m scripts.vision.templates clock-sweep --recording REC --anchor IMAGE --text 1:48 [--dry-run]
    python -m scripts.vision.templates harvest --recording REC --what tower --out tmp/harvest_tower.png
    python -m scripts.vision.templates harvest-label --sheet tmp/harvest_tower.json --map "0=3,1=0,4=5"

clock-sweep
    The clock shows one value per second. Each image's clock box is split into
    glyphs with the splitter `digits` uses; two images show the same value when
    they have as many glyphs, the same box colour (black in regulation, red in
    overtime) and every glyph pair differs in less than CHANGE_FRAC of its
    pixels at the best shift of up to 1 px (glyph_change). Runs of images that
    show one value are segments. A run that interrupts one value (the same
    value before and after it, within MAX_INTERRUPT_S) is dropped, and one value
    on both sides of a gap of unreadable images is one segment.
    From the anchor image, whose clock you read, segments are labelled by
    counting: a change between two consecutive images is one second; across a
    gap, the count is the one whole number of seconds the frames.jsonl
    timestamps allow (+-PERIOD_TOL), else the count is lost there and the sweep
    stops on that side. A count check catches a skipped or doubled second: the
    last digit of a segment must look more like the segment 10 s away than
    like those 9 and 11 s away; at the first failure on a side, that segment,
    the unchecked ones before it and everything after it are dropped.
    Regulation runs from 3:00 down to 0:00 and the sweep stops at its end,
    unless --until-overtime is given with a second anchor on an overtime frame
    (overtime runs from 2:00 down to 0:00). Several regulation anchors may be
    given (each with its --text) to bridge a stretch where the count was lost;
    segments they label differently are dropped.
    For each needed character and position (minutes, colon, tens, ones) - by
    default the characters `templates list` reports missing, --all for every
    one - it takes the middle image of the segment showing it that is checked
    (or within 9 s of its anchor), spans about a second, and is nearest the
    anchor, and stores that one glyph exactly as `digits --font clock` would
    (the same crop, digits/clock/<char>/base_sweep-<image>-<position>.png).
harvest
    Crops the region(s) of --what from every image: elixir (elixir_digit),
    tower (the 6 tower_* digit boxes), clock (clock), hand (hand_0..hand_3),
    next (next). Digits are split into glyphs with the `digits` splitter (an
    elixir glyph wider than SPLIT_ASPECT x its height, as '10' would be if its
    two digits touched, is cut at its thinnest column); a glyph touching its
    box's edge is cut off and skipped. Cards are whole crops; flat ones (an
    empty slot, which the reader knows without a template) are skipped.
    Crops are clustered by NCC (the 20x20 glyph signature DigitOCR uses, or the
    32x32 card signature CardMatcher uses; greyed and coloured cards apart,
    GREYED_CHROMA), and the clusters, largest first, are written as a contact
    sheet (per cluster: '#id n=size', a 'g' when greyed, and its most typical,
    median and least typical crop), a sidecar JSON and an .npz of the
    REPS_KEPT most typical crops of each cluster.
harvest-label
    Reads the JSON and stores the --per-cluster most typical crops of every
    cluster the map labels: digits by their value, cards by their RoyaleSim
    name with an optional variant prefix (greyed:Cannon), or "empty" for an
    empty slot. Unlabelled clusters are skipped; a map with any bad entry
    stores nothing.

Numpy only; OpenCV or Pillow read the JPEGs and write the PNGs (record.py), and
without either the sheet is written as .npy. Nothing here raises on a bad
frame: unreadable, missing or odd-sized images are counted, skipped and
reported.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .geometry import check_frame_size
from .hud import OVERTIME_S, REGULATION_S, TOWER_DIGIT_REGIONS, HudRegions, load_regions, parse_clock_text
from .record import FRAMES_FILE, IMAGE_NAME_RE, IMAGES_DIR, list_sessions, load_image, save_image
from .templates import (DEFAULT_ROOT, DIGITS, KIND_EMPTY, KIND_HAND, KIND_NEXT, PATCH, VARIANTS,
                        GlyphStyle, TemplateError, TemplateStore, area_resize, crop_std, digit_kind,
                        drop_specks, font_style, glyph_signature, label_dir, segment_glyphs, signature)
from .types import TOWER_KEYS

Box = tuple[int, int, int, int]

DEFAULT_FPS = 57.0                 # the owner's capture rate (frames.jsonl times are used when present)
#: glyph_change at or above which two images show different values. A digit
#: that changes moves >= 0.7 on the owner's clock and >= 0.125 for
#: seven-segment digits (8 vs 9); an unchanged glyph moves 0.0 between
#: images (static UI) and <= 0.1 when a neighbour's width shifts the text.
CHANGE_FRAC = 0.1
MAX_INTERRUPT_S = 3.0              # a run shorter than this between two runs of one value is dropped
PERIOD_TOL = 0.1                   # one shown value lasts 1 s +- 10 % (counting across a gap)
MIN_FRAMES = 3                     # images a segment needs to give a template
#: Median R - G of the clock box's pixels outside the glyphs: the overtime box
#: is red (background +99 on the owner's frames), the regulation one dark (-33).
RED_BOX_MIN = 40
REGULATION_TOP, OVERTIME_TOP = REGULATION_S, OVERTIME_S       # 3:00 and 2:00
POSITIONS = {4: ("minutes", "colon", "tens", "ones"), 2: ("tens", "ones"), 1: ("ones",)}

#: Elixir digits are at most about as wide as tall (owner: 19-24 px wide, 28
#: tall; '10' is two glyphs of 10 and ~21 px); a wider glyph with a clear
#: waist (a column with at most SPLIT_VALLEY x the median column's pixels)
#: is two touching digits.
SPLIT_ASPECT = 1.0
SPLIT_VALLEY = 0.5

DEFAULT_MAX_CLUSTERS = 80
#: NCC to join a cluster. Different digits of the owner's tower and clock
#: fonts score <= 0.68, the same digit 0.63-1.0 (lower across the two tower
#: tints and sub-pixel positions, which then form separate clusters); 0.9
#: keeps clusters pure (seven-segment 0 vs 8 is 0.92). Different hand cards
#: score <= 0.46, the same card >= 0.84.
DIGIT_MIN_NCC = 0.9
CARD_MIN_NCC = 0.8
#: Mean (max - min channel) of a card crop below which it is greyed out (owner:
#: greyed slots 0.4-0.7, coloured ones >= 44).
GREYED_CHROMA = 12.0
EMPTY_STD = 8.0                    # HandReader's empty_std: flatter crops are empty slots
RESERVOIR = {"digits": 24, "cards": 8}       # crops kept per cluster while clustering
MAX_CLUSTERS_TOTAL = {"digits": 1000, "cards": 400}
REPS_KEPT = 5                      # most typical crops saved per cluster (harvest-label takes --per-cluster)
HOG_26 = ("HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons", "Fireball", "Log")
HARVEST_SCHEMA = "hastycr.vision.harvest/1"

#: --what -> (digit font or None for cards, regions)
WHATS: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "elixir": ("elixir", ("elixir_digit",)),
    "tower": ("tower", tuple(TOWER_DIGIT_REGIONS[k] for k in TOWER_KEYS)),
    "clock": ("clock", ("clock",)),
    "hand": (None, tuple(f"hand_{i}" for i in range(4))),
    "next": (None, ("next",)),
}


def _slug(text: str, default: str = "x") -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-.") or default


def _mmss(seconds: int) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecImage:
    i: int                 # frame index (the image's number)
    t: float               # capture time, seconds
    path: Path


@dataclass
class Recording:
    session: Path
    images: list[RecImage]
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.session.name

    def find(self, ref: str | int) -> int | None:
        """Index in `images` of an image given as a path, a file name or a frame number."""
        s = str(ref).strip()
        name = Path(s).name
        for k, img in enumerate(self.images):
            if img.path.name == name:
                return k
        stem = Path(s).stem
        if stem.isdigit():
            n = int(stem)
            for k, img in enumerate(self.images):
                if img.i == n:
                    return k
        return None

    def interval(self) -> float:
        """Median time between consecutive images (5 frames at DEFAULT_FPS without times)."""
        dts = [b.t - a.t for a, b in zip(self.images, self.images[1:]) if b.t > a.t]
        return float(np.median(dts)) if dts else 5.0 / DEFAULT_FPS


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def open_recording(path: str | Path, fps: float = DEFAULT_FPS) -> Recording:
    """The images of a recorded session in frame order, with their capture times.

    `path` is a session directory, its frames.jsonl, or a root holding sessions
    (the newest is used). Times: the frame's t_capture, else its t_wall, else
    frame index / fps. Without frames.jsonl (or with no image in it) every
    image in images/ is used at index / fps. Bad lines and images missing on
    disk are skipped and counted in `notes`. Raises FileNotFoundError only
    when there is no session at all.
    """
    p = Path(path)
    notes: list[str] = []
    if p.is_file():
        session = p.parent
    elif (p / FRAMES_FILE).is_file() or (p / IMAGES_DIR).is_dir():
        session = p
    else:
        sessions = list_sessions(p)
        if not sessions:
            raise FileNotFoundError(f"no recorded session in {p} (need {FRAMES_FILE} or {IMAGES_DIR}/)")
        session = sessions[-1]
        notes.append(f"using the newest session {session.name}")
    fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
    frames_path = session / FRAMES_FILE
    found: dict[int, RecImage] = {}
    bad = missing = 0
    if frames_path.is_file():
        with open(frames_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    bad += 1                       # a line cut short by a crash
                    continue
                if not isinstance(rec, dict) or rec.get("kind") != "frame" or not rec.get("image"):
                    continue
                try:
                    i = int(rec.get("i"))
                except (TypeError, ValueError):
                    bad += 1
                    continue
                frame = rec.get("frame") if isinstance(rec.get("frame"), dict) else {}
                t = _num(frame.get("t_capture"))
                if t is None:
                    t = _num(rec.get("t_wall"))
                if t is None:
                    t = i / fps
                img = session / str(rec["image"])
                if not img.is_file():
                    missing += 1                   # pruned by the disk budget, or deleted
                    continue
                found[i] = RecImage(i, t, img)
    else:
        notes.append(f"no {FRAMES_FILE}: times from the image numbers at {fps:g} fps")
    if not found and (session / IMAGES_DIR).is_dir():
        for f in sorted((session / IMAGES_DIR).iterdir()):
            if IMAGE_NAME_RE.match(f.name) and f.is_file():
                i = int(f.name[:8])
                found[i] = RecImage(i, i / fps, f)
        if found and frames_path.is_file():
            notes.append(f"{FRAMES_FILE} lists no image on disk: using images/ at {fps:g} fps")
    if bad:
        notes.append(f"{bad} unreadable line(s) in {FRAMES_FILE} skipped")
    if missing:
        notes.append(f"{missing} image(s) listed in {FRAMES_FILE} are missing on disk")
    return Recording(session, [found[i] for i in sorted(found)], notes)


Loader = Callable[[Path], np.ndarray]


def _load(img: RecImage, load: Loader | None = None) -> tuple[np.ndarray | None, str]:
    """(RGB frame, "") or (None, why): never raises."""
    try:
        rgb = np.asarray((load or load_image)(img.path))
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return None, f"not an RGB image (shape {rgb.shape})"
        check_frame_size(rgb.shape[1], rgb.shape[0])
        return rgb[..., :3], ""
    except Exception as exc:                       # a bad frame must not stop the sweep
        return None, f"{type(exc).__name__}: {exc}"


class _Problems:
    def __init__(self) -> None:
        self.count = 0
        self.first: list[str] = []

    def add(self, img: RecImage, why: str) -> None:
        self.count += 1
        if len(self.first) < 3:
            self.first.append(f"{img.path.name}: {why}")

    def line(self, what: str = "unreadable or odd-sized images") -> str:
        return f"{self.count} {what} skipped" + (f" (e.g. {'; '.join(self.first)})" if self.first else "")


# ---------------------------------------------------------------------------
# Glyphs
# ---------------------------------------------------------------------------
def glyph_boxes(region_rgb: np.ndarray, style: GlyphStyle) -> tuple[list[Box], np.ndarray]:
    """The `digits` splitter (templates.add_digit_templates): the style's fill
    mask, cut into column runs, specks dropped. Returns (boxes, mask)."""
    a = np.asarray(region_rgb)
    if a.ndim != 3 or a.size == 0:
        return [], np.zeros(a.shape[:2] if a.ndim >= 2 else (0, 0), dtype=bool)
    mask = style.mask(a)
    return drop_specks(segment_glyphs(mask)), mask


def _tight(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return m[:0, :0]
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def split_wide_glyphs(mask: np.ndarray, boxes: Sequence[Box], max_aspect: float = SPLIT_ASPECT,
                      valley: float = SPLIT_VALLEY) -> list[Box]:
    """Boxes with every glyph wider than `max_aspect` x its height cut in two at
    its thinnest column in the middle half, when that column holds at most
    `valley` x the median filled column's pixels (two touching digits, as '10'
    in the elixir drop would be); other boxes are kept as they are."""
    m = np.asarray(mask, dtype=bool)
    out: list[Box] = []
    for box in boxes:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        if h <= 0 or w < 4 or w <= max_aspect * h:
            out.append(box)
            continue
        cols = m[y0:y1, x0:x1].sum(axis=0)
        lo, hi = w // 4, w - w // 4
        c = lo + int(np.argmin(cols[lo:hi]))
        filled = cols[cols > 0]
        if filled.size == 0 or cols[c] > valley * float(np.median(filled)):
            out.append(box)
            continue
        parts: list[Box] = []
        for xa, xb in ((x0, x0 + c), (x0 + c + 1, x1)):
            ys, xs = np.nonzero(m[y0:y1, xa:xb])
            if len(ys) == 0:
                parts = []
                break
            parts.append((xa + int(xs.min()), y0 + int(ys.min()), xa + int(xs.max()) + 1, y0 + int(ys.max()) + 1))
        out.extend(parts if parts else [box])
    return out


def glyph_change(a: np.ndarray, b: np.ndarray, shift: int = 1) -> float:
    """How different two tight glyph masks are: the fewest differing pixels over
    relative shifts of up to `shift` px, over the smaller glyph's pixel count.
    0.0 for the same glyph (also shifted by a pixel); >= 0.125 between two
    seven-segment digits; >= 0.7 between two digits of the owner's clock."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 or nb == 0:
        return 0.0 if na == nb else math.inf
    s = max(0, int(shift))
    hh = max(a.shape[0], b.shape[0]) + 2 * s
    ww = max(a.shape[1], b.shape[1]) + 2 * s
    big_a = np.zeros((hh, ww), dtype=bool)
    big_a[s:s + a.shape[0], s:s + a.shape[1]] = a
    best = na + nb
    for dy in range(-s, s + 1):
        for dx in range(-s, s + 1):
            big_b = np.zeros((hh, ww), dtype=bool)
            big_b[s + dy:s + dy + b.shape[0], s + dx:s + dx + b.shape[1]] = b
            best = min(best, int(np.count_nonzero(big_a ^ big_b)))
    return best / float(min(na, nb))


# ---------------------------------------------------------------------------
# clock-sweep
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClockObs:
    """One image's clock: glyph boxes and tight masks (left to right), and
    whether the box is the red overtime one."""
    boxes: tuple[Box, ...]
    glyphs: tuple[np.ndarray, ...] = field(repr=False)
    red: bool = False


def clock_obs(region_rgb: np.ndarray, style: GlyphStyle | None = None) -> ClockObs | None:
    """The clock box's glyphs and colour, None when it shows no glyph."""
    a = np.asarray(region_rgb)
    if a.ndim != 3 or a.shape[2] < 3 or a.size == 0:
        return None
    boxes, mask = glyph_boxes(a, style or font_style("clock"))
    if not boxes:
        return None
    glyphs = tuple(_tight(mask[y0:y1, x0:x1]) for x0, y0, x1, y1 in boxes)
    rg = (a[..., 0].astype(np.int16) - a[..., 1].astype(np.int16))[~mask]
    return ClockObs(tuple(boxes), glyphs, bool(rg.size) and float(np.median(rg)) >= RED_BOX_MIN)


def display_change(a: ClockObs, b: ClockObs) -> float:
    """0 when two clock images show the same value; >= CHANGE_FRAC when not."""
    if len(a.glyphs) != len(b.glyphs) or a.red != b.red:
        return math.inf
    return max((glyph_change(x, y) for x, y in zip(a.glyphs, b.glyphs)), default=0.0)


@dataclass
class ClockSegment:
    """A run of images (indexes into Recording.images) showing one clock value."""
    frames: list[int]
    ref: ClockObs
    linked: bool                    # the image just before showed another value: a change between the two
    t_prev: float | None            # time of the image just before the first one
    t0: float
    t1: float
    label: int | None = None        # seconds shown
    phase: str = ""                 # "regulation" | "overtime"
    text: str | None = None
    trusted: bool = False           # count-checked, or within 9 s of its anchor
    dist: int = 0                   # seconds from its anchor

    @property
    def n_glyphs(self) -> int:
        return len(self.ref.glyphs)

    @property
    def mid(self) -> int:
        return self.frames[len(self.frames) // 2]


def clock_segments(times: Sequence[float], obs: Sequence[ClockObs | None], *, change: float = CHANGE_FRAC,
                   max_interrupt_s: float = MAX_INTERRUPT_S) -> tuple[list[ClockSegment], int]:
    """Images -> segments of one value (see the module doc). `obs[k]` is image
    k's ClockObs or None (unreadable, no glyphs). Returns the segments and the
    number of images dropped as interruptions."""
    segs: list[ClockSegment] = []
    cur: ClockSegment | None = None
    for k, ob in enumerate(obs):
        if ob is None:
            cur = None
            continue
        if cur is not None and display_change(ob, cur.ref) < change:
            cur.frames.append(k)
            cur.t1 = float(times[k])
            continue
        cur = ClockSegment([k], ob, cur is not None, float(times[k - 1]) if k > 0 else None,
                           float(times[k]), float(times[k]))
        segs.append(cur)
    dropped = 0
    merged = True
    while merged:
        merged = False
        j = 0
        while j < len(segs) - 1:
            a, b = segs[j], segs[j + 1]
            if display_change(a.ref, b.ref) < change:          # one value on both sides of a gap
                a.frames += b.frames
                a.t1 = b.t1
                del segs[j + 1]
                merged = True
                continue
            if j + 2 < len(segs):
                c = segs[j + 2]
                if c.t0 - a.t1 <= max_interrupt_s and display_change(a.ref, c.ref) < change:
                    dropped += len(b.frames)                    # b interrupted one value
                    a.frames += c.frames
                    a.t1 = c.t1
                    del segs[j + 1:j + 3]
                    merged = True
                    continue
            j += 1
    return segs, dropped


def clock_text(seconds: int, n_glyphs: int) -> str | None:
    """How the clock shows `seconds` in `n_glyphs` glyphs: 'M:SS' (4), or under
    a minute 'SS' (2) / 'S' (1); None when it cannot."""
    s = int(seconds)
    if s < 0:
        return None
    m, ss = divmod(s, 60)
    cands = [f"{m}:{ss:02d}"] + ([f"{s:02d}", str(s)] if s < 60 else [])
    for c in cands:
        if len(c) == n_glyphs:
            return c
    return None


def _start_bounds(seg: ClockSegment, before: ClockSegment | None, period: float, tol: float) -> tuple[float, float]:
    """(earliest, latest) time the value `seg` shows can have appeared."""
    hi = seg.t0
    if seg.linked and seg.t_prev is not None:
        return min(seg.t_prev, hi), hi
    lo = min(hi, seg.t1 - period * (1.0 + tol))
    if before is not None:
        lo = max(lo, before.t1)
    return min(lo, hi), hi


def seconds_between(segs: Sequence[ClockSegment], j: int, period: float = 1.0,
                    tol: float = PERIOD_TOL) -> int | None:
    """Seconds from segment j-1's value to segment j's: 1 for a change between
    two consecutive images; across a gap, the one whole number the start
    times allow at 1 s +- tol per value; None when none or several fit."""
    a, b = segs[j - 1], segs[j]
    if b.linked:
        return 1
    a_lo, a_hi = _start_bounds(a, segs[j - 2] if j >= 2 else None, period, tol)
    b_lo, b_hi = _start_bounds(b, a, period, tol)
    d_lo, d_hi = b_lo - a_hi, b_hi - a_lo
    if not (math.isfinite(d_lo) and math.isfinite(d_hi)) or d_hi <= 0:
        return None
    top = min(100_000, int(d_hi / (period * (1.0 - tol))) + 1)
    fits = [n for n in range(1, top + 1) if n * period * (1.0 - tol) <= d_hi and n * period * (1.0 + tol) >= d_lo]
    return fits[0] if len(fits) == 1 else None


@dataclass
class AnchorRun:
    """What one anchor labelled."""
    ref: str
    k: int                          # anchor image index
    a: int                          # anchor segment index
    seconds: int
    phase: str
    labels: dict[int, int] = field(default_factory=dict)       # segment index -> seconds
    checked: set[int] = field(default_factory=set)
    stops: dict[int, str] = field(default_factory=dict)        # direction -> why the walk stopped
    dropped: list[tuple[int, str]] = field(default_factory=list)


def _walk(segs: Sequence[ClockSegment], run: AnchorRun, top: int, direction: int) -> None:
    red = segs[run.a].ref.red
    sec, j = run.seconds, run.a
    while True:
        k = j + direction
        if not 0 <= k < len(segs):
            run.stops[direction] = "the recording " + ("ends" if direction > 0 else "starts")
            return
        if segs[k].ref.red != red:
            run.stops[direction] = "the clock box changes colour (regulation / overtime)"
            return
        n = seconds_between(segs, max(j, k))
        if n is None:
            lo, hi = segs[min(j, k)], segs[max(j, k)]
            run.stops[direction] = (f"lost the count across {hi.t0 - lo.t1:.2f} s without a readable clock "
                                    f"(after the image at {lo.t1:.2f} s); add an --anchor beyond it")
            return
        sec = sec - n if direction > 0 else sec + n
        if sec < 0:
            run.stops[direction] = f"{run.phase} ends at 0:00"
            return
        if sec > top:
            run.stops[direction] = f"{run.phase} starts at {_mmss(top)}"
            return
        run.labels[k] = sec
        j = k


def _ones_check(segs: Sequence[ClockSegment], labels: Mapping[int, int], k: int, direction: int) -> bool | None:
    """Does segment k's last digit look most like the segment 10 s nearer the
    anchor (vs 9 and 11 s)? None when one of the three is not labelled."""
    by_label = {v: kk for kk, v in labels.items()}
    sec = labels[k]
    refs = []
    for off in (10, 9, 11):
        kk = by_label.get(sec + direction * off)
        if kk is None:
            return None
        refs.append(segs[kk].ref.glyphs[-1])
    g = segs[k].ref.glyphs[-1]
    r10, r9, r11 = (glyph_change(g, r) for r in refs)
    return r10 <= min(r9, r11)


def _verify(segs: Sequence[ClockSegment], run: AnchorRun) -> None:
    """The count check, outward from the anchor on both sides in turn."""
    seqs = {+1: sorted(k for k in run.labels if k > run.a),
            -1: sorted((k for k in run.labels if k < run.a), reverse=True)}
    pos = {+1: 0, -1: 0}
    pending: dict[int, list[int]] = {+1: [], -1: []}
    done = {+1: not seqs[+1], -1: not seqs[-1]}
    while not (done[+1] and done[-1]):
        for d in (+1, -1):
            if done[d]:
                continue
            k = seqs[d][pos[d]]
            pos[d] += 1
            res = _ones_check(segs, run.labels, k, d)
            if res is None:
                pending[d].append(k)
            elif res:
                pending[d] = []
                run.checked.add(k)
            else:
                shown = clock_text(run.labels[k], segs[k].n_glyphs) or _mmss(run.labels[k])
                why = (f"count check failed at {shown}: its last digit is unlike the one 10 s "
                       f"{'earlier' if d > 0 else 'later'}")
                for kk in pending[d] + seqs[d][pos[d] - 1:]:
                    run.dropped.append((kk, why))
                    run.labels.pop(kk, None)
                run.stops[d] = why
                done[d] = True
                continue
            if pos[d] >= len(seqs[d]):
                done[d] = True


@dataclass
class SweepResult:
    ok: bool
    segments: list[ClockSegment] = field(default_factory=list)
    runs: list[AnchorRun] = field(default_factory=list)
    stored: list[tuple[str, str, str, Path | None]] = field(default_factory=list)  # char, position, image, path
    skipped: list[tuple[int, str]] = field(default_factory=list)                     # segment, why
    not_found: list[str] = field(default_factory=list)
    unreadable: int = 0
    no_clock: int = 0
    interruptions: int = 0


def clock_sweep(rec: Recording, regions: HudRegions, store: TemplateStore,
                anchors: Sequence[tuple[str, str]], *, until_overtime: bool = False, all_chars: bool = False,
                dry_run: bool = False, change: float = CHANGE_FRAC, load: Loader | None = None,
                log: Callable[[str], Any] = print) -> SweepResult:
    """Label the recording's clock segments from `anchors` [(image, text)] and
    store the clock glyphs the store lacks (see the module doc)."""
    res = SweepResult(False)
    for note in rec.notes:
        log(f"note: {note}")
    if not rec.images:
        log(f"ERROR: no images in {rec.session}")
        return res
    if regions.get("clock") is None:
        log("ERROR: the HUD regions have no 'clock' box")
        return res
    style = font_style("clock")
    dt = rec.interval()
    problems = _Problems()
    unreadable: set[int] = set()
    obs: list[ClockObs | None] = []
    times = [img.t for img in rec.images]
    for n, img in enumerate(rec.images):
        if n and n % 1000 == 0:
            log(f"   {n}/{len(rec.images)} images")
        rgb, why = _load(img, load)
        if rgb is None:
            problems.add(img, why)
            unreadable.add(n)
            obs.append(None)
            continue
        ob = clock_obs(regions.crop(rgb, "clock"), style)
        res.no_clock += ob is None
        obs.append(ob)
    res.unreadable = problems.count
    segs, res.interruptions = clock_segments(times, obs, change=change)
    res.segments = segs
    durs = [s.t1 - s.t0 + dt for s in segs]
    log(f"clock-sweep {rec.name}: {len(rec.images)} images, {len(rec.images) - problems.count - res.no_clock} "
        f"with a clock, {res.no_clock} without; {len(segs)} segments of one value"
        + (f" (median {float(np.median(durs)):.2f} s)" if durs else "")
        + (f"; {res.interruptions} images dropped as passing interruptions" if res.interruptions else ""))
    if problems.count:
        log("   " + problems.line())

    # -- anchors -----------------------------------------------------------------
    seg_of = {k: j for j, s in enumerate(segs) for k in s.frames}
    pairs = list(anchors)
    phases = ["regulation"] * len(pairs)
    if until_overtime:
        if len(pairs) >= 2:
            phases[-1] = "overtime"
        else:
            log("note: --until-overtime needs a second --anchor/--text on an overtime frame; "
                "stopping at the regulation end")
    for (ref, text), phase in zip(pairs, phases):
        given = str(text).replace(" ", "")
        sec = parse_clock_text(given)
        k = rec.find(ref)
        if sec is None:
            log(f"ERROR: --text {text!r} is not a clock reading like 1:48")
            return res
        if k is None:
            log(f"ERROR: --anchor {ref!r} is not an image of {rec.session}")
            return res
        name = rec.images[k].path.name
        a = seg_of.get(k)
        if a is None:
            why = ("it is unreadable or not 9:16" if k in unreadable else
                   "its clock box shows no digits" if obs[k] is None else "it shows a passing interruption")
            log(f"ERROR: anchor {name} cannot be used: {why}; pick another frame")
            return res
        seg = segs[a]
        if clock_text(sec, seg.n_glyphs) != given:
            log(f"ERROR: the clock of {name} splits into {seg.n_glyphs} glyphs, which cannot show {given!r}; "
                "check the --text, or the clock box with `hud overlay`")
            return res
        top = OVERTIME_TOP if phase == "overtime" else REGULATION_TOP
        if sec > top:
            log(f"ERROR: {given} cannot be an {phase} clock")
            return res
        if seg.ref.red != (phase == "overtime"):
            if seg.ref.red:
                log(f"note: anchor {name} is an overtime frame (red clock box): "
                    + ("give it last, after a regulation anchor" if until_overtime else
                       "pass --until-overtime (after a regulation anchor) to sweep overtime") + "; skipped")
            else:
                log(f"note: anchor {name} is not an overtime frame (no red clock box); skipped")
            continue
        run = AnchorRun(str(ref), k, a, sec, phase, {a: sec})
        _walk(segs, run, top, +1)
        _walk(segs, run, top, -1)
        _verify(segs, run)
        res.runs.append(run)
    if not res.runs:
        log("ERROR: no usable anchor")
        return res
    res.skipped = _combine(segs, res.runs)
    _report_runs(rec, segs, res, log)

    # -- choose and store --------------------------------------------------------
    kind = digit_kind("clock")
    have = set(store.labels(kind))
    wanted = [c for c in DIGITS + ":" if all_chars or c not in have]
    best = _choose(segs, wanted, dt)
    res.not_found = [c for c in wanted if not any(ch == c for ch, _ in best)]
    for (ch, pos_name), (j, p) in sorted(best.items()):
        seg = segs[j]
        img = rec.images[seg.mid]
        rgb, why = _load(img, load)
        if rgb is None:
            log(f"   {ch!r} ({pos_name}): {img.path.name} became unreadable: {why}")
            continue
        crop = regions.crop(rgb, "clock")
        boxes, _ = glyph_boxes(crop, style)
        if len(boxes) != len(seg.text):
            log(f"   {ch!r} ({pos_name}): {img.path.name} splits into {len(boxes)} glyphs now; skipped")
            continue
        x0, y0, x1, y1 = boxes[p]
        name = f"sweep-{_slug(img.path.stem)}-{p}"
        if _stored(store, kind, ch, "base", name):
            log(f"   {ch!r} ({pos_name}) from {img.path.name} ({seg.text}): already stored")
            continue
        path = None if dry_run else store.add(kind, ch, crop[y0:y1, x0:x1], "base", name)
        res.stored.append((ch, pos_name, img.path.name, path))
        log(f"{'would store' if dry_run else 'stored'} {ch!r} ({pos_name}) from {img.path.name} "
            f"({seg.text}, {seg.phase})" + (f" -> {path}" if path is not None else ""))
    if not wanted:
        log("no clock character is missing (pass --all to store more)")
    elif not best:
        log("nothing to store: no checked segment shows a missing character")
    if res.not_found:
        log("not seen in a checked segment: " + " ".join(res.not_found))
    res.ok = any(s.label is not None for s in segs)
    return res


def _combine(segs: Sequence[ClockSegment], runs: Sequence[AnchorRun]) -> list[tuple[int, str]]:
    """Put the anchors' labels on the segments; returns the segments skipped
    (dropped by a count check, labelled differently by two anchors, or with a
    glyph count that cannot show their label)."""
    skipped: list[tuple[int, str]] = []
    final: dict[int, tuple[int, str]] = {}
    conflicts: set[int] = set()
    for run in runs:
        skipped += run.dropped
        for j, sec in run.labels.items():
            if j in final and final[j] != (sec, run.phase):
                conflicts.add(j)
            final.setdefault(j, (sec, run.phase))
    for j in sorted(conflicts):
        del final[j]
        skipped.append((j, "the anchors disagree on it"))
    for j, (sec, phase) in final.items():
        seg = segs[j]
        seg.label, seg.phase = sec, phase
        seg.text = clock_text(sec, seg.n_glyphs)
        if seg.text is None:
            skipped.append((j, f"{seg.n_glyphs} glyphs cannot show {_mmss(sec)}"))
            continue
        mine = [run for run in runs if run.phase == phase and run.labels.get(j) == sec]
        seg.dist = min(abs(sec - run.seconds) for run in mine)
        seg.trusted = any(j == run.a or j in run.checked or abs(sec - run.seconds) <= 9 for run in mine)
    return skipped


def _choose(segs: Sequence[ClockSegment], wanted: Sequence[str], dt: float) -> dict[tuple[str, str], tuple[int, int]]:
    """(char, position name) -> (segment, glyph index) to take each wanted
    character from: trusted segments of >= MIN_FRAMES images, those spanning
    about a second first, then the nearest their anchor, then the earliest."""
    best: dict[tuple[str, str], tuple[tuple[int, int, float], int, int]] = {}
    for j, seg in enumerate(segs):
        if seg.text is None or not seg.trusted or len(seg.frames) < MIN_FRAMES:
            continue
        full = abs((seg.t1 - seg.t0 + dt) - 1.0) <= 0.25
        key = (0 if full else 1, seg.dist, seg.t0)
        names = POSITIONS.get(len(seg.text)) or tuple(str(p) for p in range(len(seg.text)))
        for p, ch in enumerate(seg.text):
            if ch not in wanted:
                continue
            cur = best.get((ch, names[p]))
            if cur is None or key < cur[0]:
                best[(ch, names[p])] = (key, j, p)
    return {k: (j, p) for k, (_, j, p) in best.items()}


def _report_runs(rec: Recording, segs: Sequence[ClockSegment], res: SweepResult, log: Callable[[str], Any]) -> None:
    t_first = rec.images[0].t
    for run in res.runs:
        got = sorted(segs[j].label for j in run.labels if segs[j].label == run.labels[j] and segs[j].phase == run.phase)
        span = f"{_mmss(got[-1])} down to {_mmss(got[0])}" if got else "nothing"
        log(f"anchor {rec.images[run.k].path.name} = {_mmss(run.seconds)} ({run.phase}): labelled {span} "
            f"({len(got)} segments); before it: {run.stops.get(-1, '-')}; after it: {run.stops.get(+1, '-')}")
    if res.skipped:
        log(f"skipped as ambiguous ({len(res.skipped)}):")
        for j, why in sorted(res.skipped)[:40]:
            s = segs[j]
            log(f"   images {rec.images[s.frames[0]].path.name}-{rec.images[s.frames[-1]].path.name} "
                f"(t {s.t0 - t_first:.1f}-{s.t1 - t_first:.1f} s): {why}")
        if len(res.skipped) > 40:
            log(f"   ... and {len(res.skipped) - 40} more")


def _stored(store: TemplateStore, kind: str, label: str, variant: str, name: str) -> bool:
    """Is there already a template file <variant>_<name>.* for this label?"""
    if store.root is None:
        return False
    d = store.root.joinpath(*kind.split("/"), label_dir(label))
    return any((d / f"{variant}_{name}.{ext}").exists() for ext in ("png", "npy"))


# ---------------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------------
def chroma(crop: np.ndarray) -> float:
    """Mean (max - min channel): about 0 for a greyed-out card."""
    a = np.asarray(crop)[..., :3].astype(np.int16)
    return float((a.max(axis=2) - a.min(axis=2)).mean()) if a.size else 0.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


@dataclass
class Member:
    sig: np.ndarray = field(repr=False)
    crop: np.ndarray = field(repr=False)
    image: str
    i: int
    region: str
    box: Box

    def meta(self, score: float | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {"image": self.image, "i": self.i, "region": self.region, "box": list(self.box)}
        if score is not None:
            d["ncc"] = round(float(score), 4)
        return d


@dataclass
class Cluster:
    id: int
    size: int
    greyed: bool
    members: list[Member]           # most typical first (a sample of the cluster)
    scores: list[float]             # their NCC to the cluster's mean signature
    regions: dict[str, int]
    chroma: float = 0.0

    @property
    def examples(self) -> list[Member]:
        """The most typical, the median and the least typical member."""
        n = len(self.members)
        return [self.members[i] for i in sorted({0, n // 2, n - 1})] if n else []


class _Clusterer:
    """Online leader clustering by NCC (dot products of unit signatures) with a
    reservoir sample of members per cluster, then a merge pass."""

    def __init__(self, min_ncc: float, reservoir: int, max_total: int, seed: int = 0):
        self.min_ncc = float(min_ncc)
        self.reservoir = max(1, int(reservoir))
        self.max_total = max(1, int(max_total))
        self.rng = random.Random(seed)
        self.cent: np.ndarray | None = None
        self.key = np.zeros(0, dtype=np.int64)
        self.sums: list[np.ndarray] = []
        self.counts: list[int] = []
        self.members: list[list[Member]] = []
        self.regions: list[dict[str, int]] = []
        self.chroma: list[float] = []
        self.hint: dict[Any, int] = {}
        self.overflow = 0

    def __len__(self) -> int:
        return len(self.counts)

    def add(self, sig: np.ndarray, key: int, crop: np.ndarray, img: RecImage, region: str, box: Box, *,
            hint: Any = None, chroma_v: float = 0.0) -> None:
        sig = np.asarray(sig, dtype=np.float64).ravel()
        if not sig.any():
            return
        n_cl = len(self.counts)
        j = self.hint.get(hint) if hint is not None else None
        if j is not None and not (self.key[j] == key and float(self.cent[j] @ sig) >= self.min_ncc):
            j = None
        if j is None and n_cl:
            d = self.cent[:n_cl] @ sig
            d = np.where(self.key[:n_cl] == key, d, -2.0)
            b = int(np.argmax(d))
            if d[b] >= self.min_ncc:
                j = b
        if j is None:
            if n_cl >= self.max_total:
                self.overflow += 1
                return
            j = self._new(sig, key)
        else:
            self.sums[j] += sig
            self.counts[j] += 1
            self.cent[j] = _unit(self.sums[j])
        self.regions[j][region] = self.regions[j].get(region, 0) + 1
        self.chroma[j] += float(chroma_v)
        res = self.members[j]
        slot = len(res) if len(res) < self.reservoir else self.rng.randrange(self.counts[j])
        if slot < self.reservoir:
            m = Member(sig.astype(np.float32), np.array(crop, dtype=np.uint8, copy=True), img.path.name, img.i,
                       region, tuple(int(c) for c in box))
            if slot < len(res):
                res[slot] = m
            else:
                res.append(m)
        if hint is not None:
            self.hint[hint] = j

    def _new(self, sig: np.ndarray, key: int) -> int:
        j = len(self.counts)
        if self.cent is None:
            self.cent = np.zeros((64, sig.size), dtype=np.float64)
            self.key = np.zeros(64, dtype=np.int64)
        if j >= self.cent.shape[0]:
            self.cent = np.concatenate([self.cent, np.zeros_like(self.cent)])
            self.key = np.concatenate([self.key, np.zeros_like(self.key)])
        self.cent[j] = _unit(sig)
        self.key[j] = key
        self.sums.append(sig.copy())
        self.counts.append(1)
        self.members.append([])
        self.regions.append({})
        self.chroma.append(0.0)
        return j

    def merge(self) -> None:
        """Join clusters (of one key) whose mean signatures score >= min_ncc, closest first."""
        n = len(self.counts)
        if n < 2:
            return
        cent = self.cent[:n]
        key = self.key[:n]
        sim = cent @ cent.T
        sim[key[:, None] != key[None, :]] = -2.0
        np.fill_diagonal(sim, -2.0)
        alive = np.ones(n, dtype=bool)
        while True:
            flat = int(np.argmax(sim))
            i, j = divmod(flat, n)
            if sim[i, j] < self.min_ncc:
                break
            if self.counts[j] > self.counts[i]:
                i, j = j, i
            self.sums[i] = self.sums[i] + self.sums[j]
            self.counts[i] += self.counts[j]
            pool = self.members[i] + self.members[j]
            self.members[i] = pool if len(pool) <= self.reservoir else self.rng.sample(pool, self.reservoir)
            for r, c in self.regions[j].items():
                self.regions[i][r] = self.regions[i].get(r, 0) + c
            self.chroma[i] += self.chroma[j]
            cent[i] = _unit(self.sums[i])
            alive[j] = False
            sim[j, :] = -2.0
            sim[:, j] = -2.0
            row = cent @ cent[i]
            row[~alive] = -2.0
            row[key != key[i]] = -2.0
            row[i] = -2.0
            sim[i, :] = row
            sim[:, i] = row
        keep = [k for k in range(n) if alive[k]]
        self.cent = cent[keep].copy()
        self.key = key[keep].copy()
        self.sums = [self.sums[k] for k in keep]
        self.counts = [self.counts[k] for k in keep]
        self.members = [self.members[k] for k in keep]
        self.regions = [self.regions[k] for k in keep]
        self.chroma = [self.chroma[k] for k in keep]
        self.hint = {}

    def clusters(self) -> list[Cluster]:
        """All clusters, largest first, ids 0.. in that order."""
        order = sorted(range(len(self.counts)), key=lambda k: (-self.counts[k], k))
        out = []
        for cid, k in enumerate(order):
            c = _unit(self.sums[k])
            scored = sorted(((float(m.sig.astype(np.float64) @ c), n, m) for n, m in enumerate(self.members[k])),
                            key=lambda s: (-s[0], s[1]))
            out.append(Cluster(cid, self.counts[k], bool(self.key[k]), [m for _, _, m in scored],
                               [s for s, _, _ in scored], dict(sorted(self.regions[k].items())),
                               self.chroma[k] / max(1, self.counts[k])))
        return out


@dataclass
class HarvestResult:
    what: str
    kind: str
    font: str | None
    min_ncc: float
    clusters: list[Cluster] = field(default_factory=list)
    left_out: tuple[int, int] = (0, 0)          # clusters and crops beyond max_clusters
    session: str = ""
    images: int = 0
    read: int = 0
    unreadable: int = 0
    samples: int = 0
    flat: int = 0
    cut: int = 0
    overflow: int = 0
    problems: list[str] = field(default_factory=list)


def harvest(rec: Recording, what: str, regions: HudRegions, *, min_ncc: float | None = None,
            max_clusters: int = DEFAULT_MAX_CLUSTERS, every: int = 1, load: Loader | None = None,
            log: Callable[[str], Any] | None = None) -> HarvestResult:
    """Crop --what from every `every`-th image and cluster the crops (module doc)."""
    if what not in WHATS:
        raise TemplateError(f"--what must be one of {', '.join(sorted(WHATS))}, not {what!r}")
    font, names = WHATS[what]
    kind = digit_kind(font) if font else (KIND_HAND if what == "hand" else KIND_NEXT)
    group = "digits" if font else "cards"
    thr = float(min_ncc) if min_ncc is not None else (DIGIT_MIN_NCC if font else CARD_MIN_NCC)
    res = HarvestResult(what, kind, font, thr, session=str(rec.session))
    style = font_style(font) if font else None
    clus = _Clusterer(thr, RESERVOIR[group], MAX_CLUSTERS_TOTAL[group])
    problems = _Problems()
    images = rec.images[::max(1, int(every))]
    res.images = len(images)
    for n, img in enumerate(images):
        if log is not None and n and n % 1000 == 0:
            log(f"   {n}/{len(images)} images")
        rgb, why = _load(img, load)
        if rgb is None:
            problems.add(img, why)
            continue
        res.read += 1
        for name in names:
            crop = regions.crop(rgb, name)
            if crop is None or crop.size == 0:
                continue
            if font is not None:
                boxes, mask = glyph_boxes(crop, style)
                if what == "elixir":
                    boxes = split_wide_glyphs(mask, boxes)
                h, w = mask.shape
                for p, (x0, y0, x1, y1) in enumerate(boxes):
                    if x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h:
                        res.cut += 1                 # cut off by the box: not a whole glyph
                        continue
                    res.samples += 1
                    clus.add(glyph_signature(_tight(mask[y0:y1, x0:x1])), 0, crop[y0:y1, x0:x1], img, name,
                             (x0, y0, x1, y1), hint=(name, p, len(boxes)))
            else:
                if crop_std(crop) < EMPTY_STD:
                    res.flat += 1
                    continue
                c = chroma(crop)
                res.samples += 1
                clus.add(signature(crop, PATCH), int(c < GREYED_CHROMA), crop, img, name,
                         (0, 0, crop.shape[1], crop.shape[0]), hint=name, chroma_v=c)
    res.unreadable = problems.count
    res.problems = problems.first
    res.overflow = clus.overflow
    clus.merge()
    allc = clus.clusters()
    keep = max(1, int(max_clusters))
    res.clusters = allc[:keep]
    rest = allc[keep:]
    res.left_out = (len(rest), sum(c.size for c in rest))
    return res


# 3x5 bitmap font for the sheet's labels (numpy only)
_FONT = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001001001",
    "8": "111101111101111", "9": "111101111001111", "#": "101111101111101", "n": "000110101101101",
    "=": "000111000111000", "g": "011101011001110", " ": "000000000000000",
}
SHEET_BG = (32, 32, 32)
CELL_BG = (64, 64, 64)
TEXT = (255, 255, 255)


def text_mask(text: str, scale: int = 2) -> np.ndarray:
    """`text` in the 3x5 font (digits, '#', 'n', '=', 'g', space), `scale` px per dot."""
    parts = []
    for ch in str(text):
        pat = _FONT.get(ch, _FONT[" "])
        parts.append(np.array([c == "1" for c in pat], dtype=bool).reshape(5, 3))
        parts.append(np.zeros((5, 1), dtype=bool))
    m = np.concatenate(parts, axis=1) if parts else np.zeros((5, 1), dtype=bool)
    s = max(1, int(scale))
    return np.kron(m, np.ones((s, s), dtype=bool)).astype(bool)


def _fit(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """An RGB crop scaled (aspect kept, area_resize) and centred in an h x w cell."""
    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:] = CELL_BG
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.ndim != 3 or a.size == 0:
        return out
    ih, iw = a.shape[:2]
    s = min(h / ih, w / iw)
    th, tw = max(1, min(h, int(round(ih * s)))), max(1, min(w, int(round(iw * s))))
    small = np.stack([area_resize(a[..., c].astype(np.float64), tw, th) for c in range(3)], axis=2)
    y, x = (h - th) // 2, (w - tw) // 2
    out[y:y + th, x:x + tw] = np.clip(np.rint(small), 0, 255).astype(np.uint8)
    return out


def contact_sheet(clusters: Sequence[Cluster], cell: tuple[int, int], width: int = 1280) -> np.ndarray:
    """A grid of tiles, one per cluster: '#id n=size' (+ ' g' when greyed) over
    its most typical, median and least typical crop. cell = (height, width)."""
    ch, cw = int(cell[0]), int(cell[1])
    pad, scale = 4, 2
    head = 5 * scale + 2 * pad
    tile_w, tile_h = pad + 3 * (cw + pad), head + ch + pad
    cols = max(1, int(width) // tile_w)
    rows = max(1, -(-len(clusters) // cols))
    sheet = np.empty((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    sheet[:] = SHEET_BG
    for n, cl in enumerate(clusters):
        r, c = divmod(n, cols)
        y0, x0 = r * tile_h, c * tile_w
        tm = text_mask(f"#{cl.id} n={cl.size}" + (" g" if cl.greyed else ""), scale)
        th, tw = tm.shape[0], min(tm.shape[1], tile_w - 2 * pad)
        sheet[y0 + pad:y0 + pad + th, x0 + pad:x0 + pad + tw][tm[:, :tw]] = TEXT
        for e, m in enumerate(cl.examples):
            xx = x0 + pad + e * (cw + pad)
            sheet[y0 + head:y0 + head + ch, xx:xx + cw] = _fit(m.crop, ch, cw)
    return sheet


def write_harvest(res: HarvestResult, out: str | Path) -> tuple[Path, Path, Path]:
    """Write the contact sheet (PNG, or .npy without OpenCV/Pillow), the sidecar
    JSON and the .npz of representative crops; returns their paths."""
    out = Path(out)
    if out.suffix.lower() != ".png":
        out = out.with_name(out.name + ".png") if not out.suffix else out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path, npz_path = out.with_suffix(".json"), out.with_suffix(".npz")
    sheet = contact_sheet(res.clusters, (48, 40) if res.font else (120, 100))
    sheet_path = out
    try:
        save_image(out, sheet, "png")
    except (RuntimeError, OSError, ValueError):
        sheet_path = out.with_suffix(".npy")        # no OpenCV or Pillow: numpy's own format
        np.save(sheet_path, sheet, allow_pickle=False)
    arrays = {f"c{cl.id}_r{k}": np.ascontiguousarray(m.crop, dtype=np.uint8)
              for cl in res.clusters for k, m in enumerate(cl.members[:REPS_KEPT])}
    with open(npz_path, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    doc = {
        "schema": HARVEST_SCHEMA, "what": res.what, "kind": res.kind, "font": res.font,
        "recording": res.session, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "images": res.images, "images_read": res.read, "unreadable": res.unreadable, "samples": res.samples,
        "flat_skipped": res.flat, "cut_skipped": res.cut, "overflow": res.overflow, "min_ncc": res.min_ncc,
        "sheet": sheet_path.name, "crops": npz_path.name,
        "left_out": {"clusters": res.left_out[0], "crops": res.left_out[1]},
        "clusters": [{"id": cl.id, "size": cl.size, "greyed": cl.greyed, "chroma": round(cl.chroma, 2),
                      "worst_ncc": round(cl.scores[-1], 4) if cl.scores else None,
                      "regions": cl.regions, "reps_saved": min(REPS_KEPT, len(cl.members)),
                      "examples": [m.meta() for m in cl.examples],
                      "reps": [m.meta(s) for m, s in zip(cl.members[:REPS_KEPT], cl.scores)]}
                     for cl in res.clusters],
        "label_with": (f'python -m scripts.vision.templates harvest-label --sheet {json_path} '
                       f'--map "ID=VALUE,..."'),
    }
    json_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return sheet_path, json_path, npz_path


# ---------------------------------------------------------------------------
# harvest-label
# ---------------------------------------------------------------------------
def parse_label_map(text: str) -> dict[int, str]:
    """'0=7, 1=3,2=greyed:Cannon' -> {0: '7', 1: '3', 2: 'greyed:Cannon'}."""
    out: dict[int, str] = {}
    bad = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        cid, sep, value = part.partition("=")
        cid, value = cid.strip().lstrip("#"), value.strip()
        if not sep or not cid.isdigit() or not value:
            bad.append(f"{part!r} is not ID=VALUE")
            continue
        if int(cid) in out:
            bad.append(f"cluster {int(cid)} is labelled twice")
            continue
        out[int(cid)] = value
    if bad:
        raise TemplateError("bad --map: " + "; ".join(bad))
    if not out:
        raise TemplateError("--map labels no cluster")
    return out


def resolve_label(what: str, value: str, deck: Sequence[str] = HOG_26) -> tuple[str, str, str]:
    """A map value -> (template kind, label, variant) for a harvest of `what`."""
    if what not in WHATS:
        raise TemplateError(f"unknown harvest kind {what!r}")
    font = WHATS[what][0]
    value = str(value).strip()
    if font is not None:
        chars = DIGITS + (":" if font == "clock" else "")
        if len(value) != 1 or value not in chars:
            raise TemplateError(f"{value!r} is not one {font} glyph ({' '.join(chars)}); a cluster is one "
                                "glyph, so 10 is two clusters, 1 and 0")
        return digit_kind(font), value, "base"
    if value.lower() == "empty":
        return KIND_EMPTY, what, "base"
    variant, sep, card = value.rpartition(":")
    variant = variant if sep else "base"
    if variant not in VARIANTS:
        raise TemplateError(f"variant {variant!r} is not one of {', '.join(VARIANTS)}")
    if card not in deck:
        raise TemplateError(f"{card!r} is not in the deck ({', '.join(deck)}); use RoyaleSim names or --deck")
    label_dir(card)
    return (KIND_HAND if what == "hand" else KIND_NEXT), card, variant


def harvest_label(sheet: str | Path, mapping: str | Mapping[int, str], store: TemplateStore, *,
                  per_cluster: int = 2, deck: Sequence[str] = HOG_26, force: bool = False,
                  dry_run: bool = False, log: Callable[[str], Any] = print) -> list[tuple[int, str, str, str, Path | None]]:
    """Store the labelled clusters' most typical crops; returns (cluster, kind,
    label, variant, path) per template. Raises TemplateError (storing nothing)
    when the sheet or any map entry is bad."""
    path = Path(sheet)
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TemplateError(f"cannot read the harvest sheet {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != HARVEST_SCHEMA:
        raise TemplateError(f"{path} is not a harvest sheet (schema {HARVEST_SCHEMA})")
    what = str(doc.get("what"))
    clusters = {int(c["id"]): c for c in doc.get("clusters", [])}
    labels = parse_label_map(mapping) if isinstance(mapping, str) else {int(k): str(v) for k, v in mapping.items()}
    plan, problems = [], []
    for cid, value in sorted(labels.items()):
        c = clusters.get(cid)
        if c is None:
            problems.append(f"no cluster {cid} on the sheet (ids 0-{max(clusters) if clusters else '?'})")
            continue
        try:
            kind, label, variant = resolve_label(what, value, deck)
        except TemplateError as exc:
            problems.append(f"cluster {cid}: {exc}")
            continue
        greyed = bool(c.get("greyed"))
        if kind in (KIND_HAND, KIND_NEXT) and not force and greyed != variant.endswith("greyed"):
            want = f"greyed:{label}" if greyed else label
            problems.append(f"cluster {cid} looks {'greyed' if greyed else 'coloured'} (chroma {c.get('chroma')}) "
                            f"but is labelled {value!r}: use {want!r}, or --force")
            continue
        plan.append((cid, kind, label, variant))
    if problems:
        raise TemplateError("nothing stored:\n   " + "\n   ".join(problems))
    crops = path.parent / str(doc.get("crops") or path.with_suffix(".npz").name)
    stem = _slug(path.stem, "sheet")
    out: list[tuple[int, str, str, str, Path | None]] = []
    try:
        z = np.load(crops, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise TemplateError(f"cannot read the harvest crops {crops}: {exc}") from exc
    if not hasattr(z, "files"):
        raise TemplateError(f"{crops} is not the .npz a harvest writes")
    with z:
        for cid, kind, label, variant in plan:
            for r in range(max(1, int(per_cluster))):
                key = f"c{cid}_r{r}"
                if key not in z.files:
                    break
                name = f"hv-{stem}-c{cid}-{r}"
                if _stored(store, kind, label, variant, name):
                    log(f"cluster {cid} crop {r}: already stored")
                    continue
                p = None if dry_run else store.add(kind, label, z[key], variant, name)
                out.append((cid, kind, label, variant, p))
                log(f"{'would store' if dry_run else 'stored'} cluster {cid} crop {r} as {kind}/{label} "
                    f"({variant})" + (f" -> {p}" if p is not None else ""))
    return out


# ---------------------------------------------------------------------------
# CLI (subcommands of python -m scripts.vision.templates)
# ---------------------------------------------------------------------------
def _cmd_clock_sweep(args) -> int:
    if len(args.anchor) != len(args.text):
        print("ERROR: give one --text per --anchor")
        return 1
    try:
        rec = open_recording(args.recording, args.fps)
        regions = load_regions(args.regions)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    store = TemplateStore(args.root or DEFAULT_ROOT)
    res = clock_sweep(rec, regions, store, list(zip(args.anchor, args.text)), until_overtime=args.until_overtime,
                      all_chars=args.all, dry_run=args.dry_run, change=args.change)
    return 0 if res.ok else 1


def _cmd_harvest(args) -> int:
    try:
        rec = open_recording(args.recording, args.fps)
        regions = load_regions(args.regions)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    for note in rec.notes:
        print(f"note: {note}")
    res = harvest(rec, args.what, regions, min_ncc=args.min_ncc, max_clusters=args.max_clusters,
                  every=args.every, log=print)
    sheet, js, _ = write_harvest(res, args.out or Path("tmp") / f"harvest_{args.what}.png")
    print(f"harvest {args.what} from {rec.name}: {res.read}/{res.images} images read, {res.samples} crops "
          f"({res.flat} flat and {res.cut} cut-off skipped), {len(res.clusters)} clusters "
          f"at NCC >= {res.min_ncc:g}" + (f", {res.left_out[0]} smaller ones ({res.left_out[1]} crops) left off"
                                          if res.left_out[0] else ""))
    if res.unreadable:
        print(f"   {res.unreadable} unreadable or odd-sized images skipped"
              + (f" (e.g. {'; '.join(res.problems)})" if res.problems else ""))
    if res.overflow:
        print(f"   {res.overflow} crops matched no cluster after {len(res.clusters) + res.left_out[0]} clusters")
    print(f"wrote {sheet}\nwrote {js}")
    print(f'next: open the sheet, then  harvest-label --sheet {js} --map "ID=VALUE,..."')
    return 0 if res.clusters else 1


def _cmd_harvest_label(args) -> int:
    deck = tuple(c.strip() for c in args.deck.split(",") if c.strip())
    store = TemplateStore(args.root or DEFAULT_ROOT)
    try:
        out = harvest_label(args.sheet, args.map, store, per_cluster=args.per_cluster, deck=deck,
                            force=args.force, dry_run=args.dry_run)
    except (TemplateError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"{len(out)} templates {'would be ' if args.dry_run else ''}stored in {store.root}; "
          "check with `python -m scripts.vision.templates list`")
    return 0


def add_parsers(sub, common: argparse.ArgumentParser) -> None:
    """Add clock-sweep, harvest and harvest-label to the templates CLI."""
    rec = argparse.ArgumentParser(add_help=False, parents=[common])
    rec.add_argument("--recording", required=True,
                     help="a recorded session (reports/vision/recordings/capture_...), or the folder holding "
                          "them (the newest is used)")
    rec.add_argument("--regions", help="HUD regions JSON (default: hud.load_regions())")
    rec.add_argument("--fps", type=float, default=DEFAULT_FPS,
                     help="capture rate, used only when frames.jsonl has no times (default 57)")

    p = sub.add_parser("clock-sweep", parents=[rec],
                       help="store clock digits from a whole recording, counting seconds from one frame you read")
    p.add_argument("--anchor", action="append", required=True,
                   help="an image of the recording (path, file name or frame number) whose clock you read; "
                        "repeat for more anchors")
    p.add_argument("--text", action="append", required=True,
                   help="what that anchor's clock shows, e.g. 1:48 (one per --anchor, in the same order)")
    p.add_argument("--until-overtime", action="store_true",
                   help="also sweep overtime; the last --anchor/--text must then be an overtime frame")
    p.add_argument("--all", action="store_true", help="store every character, not only the missing ones")
    p.add_argument("--dry-run", action="store_true", help="report what would be stored, store nothing")
    p.add_argument("--change", type=float, default=CHANGE_FRAC,
                   help=f"glyph difference that counts as a new value (default {CHANGE_FRAC})")
    p.set_defaults(func=_cmd_clock_sweep)

    p = sub.add_parser("harvest", parents=[rec],
                       help="cluster the digits or cards of a whole recording into a contact sheet to label")
    p.add_argument("--what", required=True, choices=sorted(WHATS))
    p.add_argument("--out", help="contact sheet PNG (default tmp/harvest_<what>.png); the .json and .npz go "
                                 "beside it")
    p.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_CLUSTERS,
                   help=f"largest clusters on the sheet (default {DEFAULT_MAX_CLUSTERS})")
    p.add_argument("--min-ncc", type=float,
                   help=f"NCC to join a cluster (default {DIGIT_MIN_NCC} digits, {CARD_MIN_NCC} cards); "
                        "higher splits more")
    p.add_argument("--every", type=int, default=1, help="use every Nth image (default 1: all)")
    p.set_defaults(func=_cmd_harvest)

    p = sub.add_parser("harvest-label", parents=[common], help="store the clusters of a harvest sheet you labelled")
    p.add_argument("--sheet", required=True, help="the harvest JSON (or its PNG)")
    p.add_argument("--map", required=True,
                   help='cluster id=value, comma-separated: "0=7,1=3" for digits; "2=HogRider,3=greyed:Cannon,'
                        '4=empty" for cards; unlisted clusters are skipped')
    p.add_argument("--per-cluster", type=int, default=2, help="templates per labelled cluster (default 2)")
    p.add_argument("--deck", default=",".join(HOG_26), help="card names the map may use (default Hog 2.6)")
    p.add_argument("--force", action="store_true",
                   help="store a greyed-looking cluster as base (or a coloured one as greyed) anyway")
    p.add_argument("--dry-run", action="store_true", help="report what would be stored, store nothing")
    p.set_defaults(func=_cmd_harvest_label)


__all__ = [
    "CHANGE_FRAC", "ClockObs", "ClockSegment", "Cluster", "HarvestResult", "Member", "RecImage", "Recording",
    "SweepResult", "WHATS", "add_parsers", "chroma", "clock_obs", "clock_segments", "clock_sweep", "clock_text",
    "contact_sheet", "display_change", "glyph_boxes", "glyph_change", "harvest", "harvest_label",
    "open_recording", "parse_label_map", "resolve_label", "seconds_between", "split_wide_glyphs", "text_mask",
    "write_harvest",
]
