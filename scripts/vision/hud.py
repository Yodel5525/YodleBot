"""HUD readers: battle clock, elixir, tower HP, hand and next card (V07).

Every reader works on fixed screen regions (HudRegions, normalised to the 9:16
frame so one set serves 1080x1920, a 560x996 PrintWindow grab and 540x960) and
on templates captured from YOUR client (scripts/vision/templates.py). Without
templates a reader reports "unread" with confidence 0: nothing here raises in
the live loop (HudReader.read catches and reports every error).

Clock (vision-03). Template OCR of the M:SS timer, plus the label above it
    (regulation / overtime) when label templates exist. The tick a display
    implies is the START of its one-second window (20 ticks a second):
        regulation  tick = offset + (180 - s) * 20        (3:00 -> 0, 0:00 -> 3600)
        overtime    tick = offset + 3600 + (120 - s) * 20  (2:00 -> 3600, 0:00 -> 6000)
    offset_ticks (default 0) is the unmeasured lag between RoyaleSim tick 0 and
    the moment the timer leaves 3:00 (the tokens study flags it; RoyaleSim takes
    no plays before tick 90). ClockFilter adds the sub-second part from the
    time the digits changed (the midpoint of the two frames around a change,
    when they are <= 0.3 s apart), keeps the tick monotonic, rejects readings
    more than 1.5 s off the running estimate unless 3 consistent readings agree
    (a re-sync), and, without a label, tells overtime from regulation by which
    reading fits the running estimate.
Elixir (vision-10). The count next to the drop (OCR) and the bar. On the
    owner's client the bar shows whole elixir in purple and the elixir still
    charging as a lighter blue segment after it, over a dark-blue track, so
    the end of purple + light blue is the exact amount; the count and the
    "Max: 10" text cover the bar's left end (bar_fraction reports None when
    the end is hidden there). elixir_milli = the bar when it agrees with the
    digit (+-150 milli); else digit*1000 + the regeneration since the digit
    last went up (2.8 / 1.4 / 0.93 s per elixir from ticks 0 / 2400 / 4800,
    the constants tools/rs_elixir.py documents), clamped to the digit's
    window. A digit seen without either is taken as digit*1000 (conservative).
Tower HP (vision-09). OCR of the hit points by each tower (TOWER_GLYPH_STYLE:
    the opponent's digits are pinkish white, ours light blue), max HP = the
    level-11 prior (LEVEL11_TOWER_MAX_HP; friendly battles), raised by a higher
    reading confirmed twice, or learned the same way without a prior; then the
    author's TowerHpFilter rules (scripts/tower_hp.py:232-338, ported as
    TowerHpFilter below): never rise, a fall must hold 0.6 s, a zero 2.5 s and
    only from <= 0.75. Without digit templates the princess bars' colour fill
    (tower_hp.py's masks and measure; checked on the owner's frames) gives the
    fraction. A king shows no number until damaged: never read, it is full.
Hand (vision-15). NCC of each slot and the next box against your card art
    (base / evo / hero / greyed), 32x32 like scripts/brain/cards.py; an empty
    slot (flat crop, or an "empty" template) is EMPTY_CARD = -1. HandTracker
    enforces 4 distinct cards and a FIFO cycle prior: a played card goes to
    the back of the queue, an emptied slot is refilled with the card that was
    next, and when the art is ambiguous the prior wins. A changed slot without
    an empty frame between is accepted at once only if it shows the old next
    card, otherwise after 3 agreeing frames (the flicker that
    scripts/brain/hand.py:1-22 describes).

    from scripts.vision.hud import HudReader
    from scripts.vision.templates import TemplateStore
    hud_reader = HudReader(TemplateStore(), card_ids={"HogRider": 26, ...})
    hud = hud_reader.read(rgb, t_capture)     # a scripts.vision.types.Hud
    hud_reader.report                          # confidences, raw readings, errors

CLI:
    python -m scripts.vision.hud regions [--out FILE]      the boxes in use, in pixels
    python -m scripts.vision.hud overlay --image shot.png --out check.png
    python -m scripts.vision.hud read --image shot.png --deck HogRider,Musketeer,...

The default regions (data/vision/hud_regions_1080x1920.json, BUILTIN_REGIONS_PX)
are measured on the owner's 747x1328 frames of 2026-10-29 (clock, its label,
elixir count and bar, tower digits), checked there (the author's tower bar
bands, tower_hp.py:19-49; the next box, a recalled BuildABot constant; the hand
boxes from the card size, scripts/brain/cards.py:41-44, and ArenaGeometry's
slot centres) or estimated (the kings' digits). Check them on a screenshot of
your client with `overlay`; save your own boxes as
data/vision/hud_regions_pc.json (same format), which load_regions() prefers.
Do not edit the default file (a test pins it to BUILTIN_REGIONS_PX).

Imports numpy only (Pillow for `overlay`).
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__" and not __package__:
    # Run as a file, sys.path[0] is scripts/vision, whose types.py would shadow
    # the standard library's `types`: put the repo root there instead and
    # import as part of the package (PEP 366), so the relative imports work.
    _here = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if sys.path and os.path.normcase(os.path.abspath(sys.path[0] or os.curdir)) == _here:
        sys.path[0] = _root
    else:
        sys.path.insert(0, _root)
    __import__("scripts.vision")
    __package__ = "scripts.vision"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from .geometry import check_frame_size  # noqa: E402
from .hpbars import ally_mask, enemy_mask, measure_tower_bar  # noqa: E402
from .templates import (KIND_CLOCK_LABEL, KIND_EMPTY, KIND_HAND, KIND_NEXT, LABEL_PATCH,  # noqa: E402
                        MIN_MARGIN, MIN_SCORE, PATCH, VARIANT_FORM, CardMatcher, DigitOCR,
                        DigitRead, GlyphStyle, TemplateStore, crop_std)
from .types import (EMPTY_CARD, FORM_BASE, HAND_SIZE, PHASE_ENDED, PHASE_OVERTIME,  # noqa: E402
                    PHASE_REGULATION, PHASE_UNKNOWN, TOWER_KEYS, Hud, TowerHp, now)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGIONS_PATH = ROOT / "data" / "vision" / "hud_regions_1080x1920.json"
PC_REGIONS_PATH = ROOT / "data" / "vision" / "hud_regions_pc.json"
REGIONS_SCHEMA = "hastycr.vision.hud_regions/1"
REF_SIZE = (1080, 1920)

# Match constants (RoyaleSim calibration.json, as tools/rs_elixir.py documents them).
TICKS_PER_S = 20
REGULATION_S = 180
OVERTIME_S = 120
REGULATION_TICKS = REGULATION_S * TICKS_PER_S                   # 3600
END_TICK = REGULATION_TICKS + OVERTIME_S * TICKS_PER_S          # 6000
DOUBLE_FROM_TICK = 2400                                         # 2x from 120 s
TRIPLE_FROM_TICK = 4800                                         # 3x from 60 s into overtime
REGEN_MS_1X, REGEN_MS_2X, REGEN_MS_3X = 2800, 1400, 930         # per elixir
MAX_ELIXIR_MILLI = 10_000
MAX_TOWER_HP = 10_000

#: The hand card box size, BuildABot CARD_CONFIG's 61x73 on its 368x652
#: screenshot (scripts/brain/cards.py:41-44), as a fraction of the frame.
HAND_BOX_NORM = (61 / 368, 73 / 652)
#: tower_hp.py:49 BAR_X_TOLERANCE: the princess bar's fill starts within 8 px of
#: its anchor; the bar regions below include that margin on both sides.
BAR_MARGIN_PX = 8
BAR_FULL_PX = 119            # tower_hp.py:42, 46: the fill track of a princess bar

_MEASURED = "measured on the owner's 747x1328 frames (client 160635001, 2026-10-29), scaled"
_RECALLED = ("recalled from BuildABot constants.py (vendor/ClashRoyaleBuildABot is not in this "
             "clone), scaled from its 368x652 screenshot")

#: name -> ((x0, y0, x1, y1) pixels at 1080x1920, end-exclusive; where it comes from).
#: "there" in a source means the 747x1328 frames.
BUILTIN_REGIONS_PX: dict[str, tuple[tuple[int, int, int, int], str]] = {
    "clock": ((920, 45, 1064, 101), _MEASURED + ": the M:SS digits (x 646-723, rows 36-66 there) +-7 px"),
    "clock_label": ((920, 12, 1064, 45), _MEASURED + ": the 'Time left:' / 'Overtime' text (rows 13-26 there)"),
    "elixir_digit": ((283, 1827, 379, 1881), _MEASURED + ": the count right of the drop (x 211-235, rows "
                                                         "1269-1297 there), room for '10'; the drop's "
                                                         "highlight and 'Max: 10' left out"),
    "elixir_bar": ((296, 1855, 1042, 1894), _MEASURED + ": the track, 10 segments of 51.6 px from x 205 "
                                                        "there (whole elixir ends at 308 / 360 / 618 for "
                                                        "2 / 3 / 8)"),
    "next": ((62, 1793, 138, 1891), "CARD_CONFIG[0] next card (21, 609, 47, 642) " + _RECALLED
             + "; inside the next card (x 40-100, rows 1238-1313) on the owner's frames"),
    "tower_enemy_left": ((213, 247, 341, 285), _MEASURED + ": the HP digits above the enemy bar (x 150-209, "
                                                           "rows 175-197 there), TOWER_GLYPH_STYLE"),
    "tower_enemy_right": ((778, 247, 907, 285), _MEASURED + ": tower_enemy_left + 391 px there"),
    "tower_own_left": ((213, 1186, 341, 1226), _MEASURED + ": our HP digits over the bar's lower half "
                                                           "(x 154-203, rows 822-845 there)"),
    "tower_own_right": ((778, 1186, 907, 1226), _MEASURED + ": tower_own_left + 391 px there"),
    "tower_enemy_king": ((499, 30, 636, 69), "estimated: a damaged king's label sits on the row of its "
                                             "full-HP level badge (rows 30-62 at 747x1328; the end-screen "
                                             "frame shows it zoomed out); verify on a frame with it damaged"),
    "tower_own_king": ((499, 1439, 636, 1492), "estimated like tower_enemy_king, from our king's level badge "
                                               "(rows 990-1025 at 747x1328); verify"),
    "bar_enemy_left": ((208, 275, 343, 325), "tower_hp.py:20,42-43,49: anchor 216 +-8, fill 119, band 275-325; "
                                             "reads 0.683 / 0.427 for 2101 / 1315 of 3052 on the owner's frames"),
    "bar_enemy_right": ((774, 275, 909, 325), "tower_hp.py:20,42-43,49: anchor 782 +-8"),
    "bar_own_left": ((208, 1145, 343, 1215), "tower_hp.py:45-47,49: anchor 216 +-8, band 1145-1215"),
    "bar_own_right": ((774, 1145, 909, 1215), "tower_hp.py:45-47,49: anchor 782 +-8"),
}
#: Level-11 tower hit points (friendly battles, both sides), seen on the owner's
#: frames: princess 3052 at the start, king 4824 on the end screen. HudReader
#: uses them as the prior maxima (digit readings can raise, not lower, them).
LEVEL11_TOWER_MAX_HP = {"own_king": 4824, "own_left": 3052, "own_right": 3052,
                        "enemy_king": 4824, "enemy_left": 3052, "enemy_right": 3052}
TOWER_DIGIT_REGIONS = {k: f"tower_{k}" for k in TOWER_KEYS}
TOWER_BAR_REGIONS = {k: f"bar_{k}" for k in TOWER_KEYS if not k.endswith("king")}


def _rint(x: float) -> int:
    return int(math.floor(x + 0.5))


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rect:
    """A box in normalised frame coordinates (0..1), end-exclusive."""
    u0: float
    v0: float
    u1: float
    v1: float

    def __post_init__(self):
        if not (0.0 <= self.u0 < self.u1 <= 1.0 and 0.0 <= self.v0 < self.v1 <= 1.0):
            raise ValueError(f"bad region {self}: need 0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1")

    @classmethod
    def from_px(cls, box: Sequence[float], size: Sequence[int]) -> "Rect":
        w, h = float(size[0]), float(size[1])
        x0, y0, x1, y1 = (float(c) for c in box)
        return cls(_clamp(x0 / w, 0.0, 1.0), _clamp(y0 / h, 0.0, 1.0),
                   _clamp(x1 / w, 0.0, 1.0), _clamp(y1 / h, 0.0, 1.0))

    def px(self, size: Sequence[int]) -> tuple[int, int, int, int]:
        w, h = int(size[0]), int(size[1])
        return (min(w, max(0, _rint(self.u0 * w))), min(h, max(0, _rint(self.v0 * h))),
                min(w, max(0, _rint(self.u1 * w))), min(h, max(0, _rint(self.v1 * h))))

    def crop(self, rgb: np.ndarray) -> np.ndarray:
        h, w = np.asarray(rgb).shape[:2]
        x0, y0, x1, y1 = self.px((w, h))
        return np.asarray(rgb)[y0:y1, x0:x1]


class HudRegions:
    """Named HUD boxes. Hand slots come from `hand_N` boxes when given, else from
    the card-slot centres (ArenaGeometry.card_slots_norm) and HAND_BOX_NORM."""

    def __init__(self, rects: Mapping[str, Rect], *,
                 hand_centres: Sequence[Sequence[float]] | None = None,
                 hand_size: Sequence[float] = HAND_BOX_NORM,
                 sources: Mapping[str, str] | None = None,
                 bar_margin: float = BAR_MARGIN_PX / REF_SIZE[0], name: str = "builtin"):
        self.rects = dict(rects)
        if hand_centres is None:
            from .geometry import ArenaGeometry
            hand_centres = ArenaGeometry.default().card_slots_norm
        self.hand_centres = tuple((float(u), float(v)) for u, v in hand_centres)
        if len(self.hand_centres) != HAND_SIZE:
            raise ValueError(f"need {HAND_SIZE} hand centres, got {len(self.hand_centres)}")
        self.hand_size = (float(hand_size[0]), float(hand_size[1]))
        self.sources = dict(sources or {})
        self.bar_margin = float(bar_margin)
        self.name = str(name)

    def get(self, name: str) -> Rect | None:
        if name in self.rects:
            return self.rects[name]
        if name.startswith("hand_") and name[5:].isdigit() and int(name[5:]) < HAND_SIZE:
            return self.hand(int(name[5:]))
        return None

    def hand(self, slot: int) -> Rect:
        key = f"hand_{int(slot)}"
        if key in self.rects:
            return self.rects[key]
        cu, cv = self.hand_centres[int(slot)]
        su, sv = self.hand_size
        return Rect(_clamp(cu - su / 2, 0.0, 1.0), _clamp(cv - sv / 2, 0.0, 1.0),
                    _clamp(cu + su / 2, 0.0, 1.0), _clamp(cv + sv / 2, 0.0, 1.0))

    def names(self) -> list[str]:
        return sorted(set(self.rects) | {f"hand_{i}" for i in range(HAND_SIZE)})

    def crop(self, rgb: np.ndarray, name: str) -> np.ndarray | None:
        r = self.get(name)
        return None if r is None else r.crop(rgb)

    def px(self, name: str, size: Sequence[int]) -> tuple[int, int, int, int] | None:
        r = self.get(name)
        return None if r is None else r.px(size)

    def to_dict(self, size: Sequence[int] = REF_SIZE) -> dict[str, Any]:
        w, h = int(size[0]), int(size[1])
        regions = {}
        for k in sorted(self.rects):
            r = self.rects[k]
            box = [round(r.u0 * w, 2), round(r.v0 * h, 2), round(r.u1 * w, 2), round(r.v1 * h, 2)]
            regions[k] = {"px": [int(c) if float(c).is_integer() else c for c in box],
                          "source": self.sources.get(k, "")}
        return {"schema": REGIONS_SCHEMA, "name": self.name, "frame_size": [w, h],
                "bar_margin_px": round(self.bar_margin * w, 3),
                "note": ("Pixel boxes (x0, y0, x1, y1), end-exclusive, at frame_size; they scale to any "
                         "9:16 frame. Hand slots come from the geometry's card-slot centres unless "
                         "hand_0..hand_3 are given here. See scripts/vision/hud.py."),
                "regions": regions}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, hand_centres: Sequence[Sequence[float]] | None = None) -> "HudRegions":
        schema = str(d.get("schema", REGIONS_SCHEMA))
        if not schema.startswith("hastycr.vision.hud_regions/"):
            raise ValueError(f"not a HUD regions file (schema {schema!r})")
        size = d.get("frame_size") or REF_SIZE
        check_frame_size(size[0], size[1])
        rects, sources = {}, {}
        for k, v in (d.get("regions") or {}).items():
            rects[str(k)] = Rect.from_px(v["px"], size)
            sources[str(k)] = str(v.get("source", ""))
        margin = float(d.get("bar_margin_px", BAR_MARGIN_PX)) / float(size[0])
        return cls(rects, hand_centres=hand_centres, sources=sources, bar_margin=margin,
                   name=str(d.get("name", "")))

    @classmethod
    def builtin(cls, *, hand_centres: Sequence[Sequence[float]] | None = None) -> "HudRegions":
        rects = {k: Rect.from_px(box, REF_SIZE) for k, (box, _) in BUILTIN_REGIONS_PX.items()}
        return cls(rects, hand_centres=hand_centres,
                   sources={k: src for k, (_, src) in BUILTIN_REGIONS_PX.items()}, name="builtin_default")

    @classmethod
    def load(cls, path: str | os.PathLike, **kw) -> "HudRegions":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), **kw)

    def save(self, path: str | os.PathLike, size: Sequence[int] = REF_SIZE) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(size), indent=1) + "\n", encoding="utf-8")
        return p


def load_regions(path: str | os.PathLike | None = None, geometry: Any = None) -> HudRegions:
    """`path`, else data/vision/hud_regions_pc.json, else the default file, else
    the built-in boxes. Hand centres come from `geometry` (default geometry)."""
    centres = None if geometry is None else geometry.card_slots_norm
    for p in ([Path(path)] if path else [PC_REGIONS_PATH, DEFAULT_REGIONS_PATH]):
        if p.exists():
            return HudRegions.load(p, hand_centres=centres)
        if path:
            raise FileNotFoundError(f"no HUD regions file {p}")
    return HudRegions.builtin(hand_centres=centres)


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------
def clock_tick(seconds_left: int, phase: str = PHASE_REGULATION, offset_ticks: int = 0) -> int:
    """The tick at which the timer starts showing `seconds_left` (see module doc)."""
    s = int(seconds_left)
    if phase == PHASE_OVERTIME:
        return int(offset_ticks) + REGULATION_TICKS + (OVERTIME_S - s) * TICKS_PER_S
    return int(offset_ticks) + (REGULATION_S - s) * TICKS_PER_S


def parse_clock_text(text: str | None) -> int | None:
    """'2:59' -> 179, '0:05' -> 5, '259' (colon unread) -> 179, '9' / '59' -> seconds; None if invalid."""
    if not text:
        return None
    t = str(text).replace(" ", "")
    digits = t.replace(":", "")
    if not digits.isdigit() or t.count(":") > 1:
        return None
    if len(digits) == 3:
        m, ss = int(digits[0]), int(digits[1:])
        if ss >= 60:
            return None
        s = 60 * m + ss
    elif len(digits) in (1, 2) and ":" not in t:
        s = int(digits)
        if s >= 60:
            return None
    else:
        return None
    return s if s <= REGULATION_S else None


@dataclass(frozen=True)
class ClockRead:
    seconds_left: int | None
    phase_hint: str | None = None          # from label templates; None = unknown
    conf: float = 0.0
    text: str | None = None
    reason: str = ""


class ClockReader:
    def __init__(self, store: TemplateStore | None, regions: HudRegions, style: GlyphStyle | None = None):
        self.regions = regions
        self.ocr = DigitOCR(store, "clock", style)
        entries = store.entries(KIND_CLOCK_LABEL) if store is not None else []
        self.label = CardMatcher(entries, LABEL_PATCH)

    @property
    def ready(self) -> bool:
        return self.ocr.ready

    def read(self, rgb: np.ndarray) -> ClockRead:
        crop = self.regions.crop(rgb, "clock")
        if crop is None:
            return ClockRead(None, None, 0.0, None, "no clock region")
        dr = self.ocr.read(crop)
        hint = None
        if self.label.ready:
            lab = self.regions.crop(rgb, "clock_label")
            if lab is not None and lab.size:
                m = self.label.match(lab)
                hint = {"regulation": PHASE_REGULATION, "overtime": PHASE_OVERTIME}.get(m.label or "")
        s = parse_clock_text(dr.text)
        if s is None:
            return ClockRead(None, hint, 0.0, dr.text, dr.reason or f"not a clock: {dr.text!r}")
        return ClockRead(s, hint, dr.conf, dr.text)


@dataclass(frozen=True)
class ClockState:
    """tick: filtered battle tick, None when unknown. source: read | resync |
    rejected | unread | lost (no reading for longer than max_extrapolate_s)."""
    tick: int | None
    phase: str = PHASE_UNKNOWN
    conf: float = 0.0
    anchored: bool = False
    source: str = "unread"


def _window_dist(base: float, x: float) -> float:
    """How far x lies outside the display window [base, base + 19]."""
    return max(0.0, base - x, x - (base + TICKS_PER_S - 1))


class ClockFilter:
    """Clock readings -> a monotonic battle tick with sub-second resolution.

    update(seconds_left, t, phase_hint) per frame (t = capture time, seconds):
    * A reading is accepted when its display window [base, base + 19] is within
      `tolerance_ticks` of the running estimate (last tick + elapsed * 20). With
      no phase hint and <= 120 s shown, both regulation and overtime are tried
      and the closer one wins (so 0:00 -> 2:00 enters overtime).
    * When the display moves to a new base and the previous accepted frame was
      <= max_anchor_gap_s earlier, the change is dated at the midpoint of the
      two frames: tick = base + round((t - t_change) * 20) while it shows.
      Otherwise the estimate continues from the last tick, clamped into the
      window. The first reading ever is taken at base + unanchored_offset.
    * The output never goes backwards (a high estimate waits for the display).
    * A rejected reading is kept; `confirm_frames` consecutive rejected readings
      that agree with each other re-sync the filter (a jump, or a new battle).
    * With no (accepted) reading, the tick coasts on elapsed time for up to
      max_extrapolate_s since the last accepted reading, then goes None.
    """

    def __init__(self, *, offset_ticks: int = 0, tolerance_ticks: float = 30, confirm_frames: int = 3,
                 max_extrapolate_s: float = 2.0, max_anchor_gap_s: float = 0.3,
                 unanchored_offset: int = 10):
        self.offset_ticks = int(offset_ticks)
        self.tolerance_ticks = float(tolerance_ticks)
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_extrapolate_s = float(max_extrapolate_s)
        self.max_anchor_gap_s = float(max_anchor_gap_s)
        self.unanchored_offset = int(unanchored_offset)
        self.reset()

    def reset(self) -> None:
        self._tick: int | None = None      # last output
        self._t: float | None = None       # time of the last output
        self._read_t: float | None = None  # time of the last accepted reading
        self._base: int | None = None      # display base of the last accepted reading
        self._phase = PHASE_UNKNOWN
        self._anchor: tuple[int, float] | None = None
        self._pending: list[tuple[int, str, float, int]] = []
        self.rejected = 0

    @property
    def tick(self) -> int | None:
        return self._tick

    @property
    def phase(self) -> str:
        return self._phase

    def _range(self, phase: str) -> tuple[int, int]:
        o = self.offset_ticks
        if phase == PHASE_OVERTIME:
            return o + REGULATION_TICKS, o + END_TICK
        return o, o + REGULATION_TICKS

    def _candidates(self, s: int, hint: str | None) -> list[tuple[int, str]]:
        reg = (clock_tick(s, PHASE_REGULATION, self.offset_ticks), PHASE_REGULATION)
        ot = (clock_tick(s, PHASE_OVERTIME, self.offset_ticks), PHASE_OVERTIME)
        if hint == PHASE_OVERTIME and s <= OVERTIME_S:
            return [ot]
        if hint == PHASE_REGULATION or s > OVERTIME_S:
            return [reg]
        return [reg, ot]

    def _expected(self, t: float) -> float:
        return self._tick + (t - self._t) * TICKS_PER_S

    def update(self, seconds_left: int | None, t: float, phase_hint: str | None = None) -> ClockState:
        t = float(t)
        if seconds_left is None or not 0 <= int(seconds_left) <= REGULATION_S:
            return self._coast(t, "unread")
        s = int(seconds_left)
        cands = self._candidates(s, phase_hint)
        if self._tick is None:
            return self._accept(cands[0][0], cands[0][1], s, t, resync=True)
        exp = self._expected(t)
        base, phase = min(cands, key=lambda c: _window_dist(c[0], exp))
        if _window_dist(base, exp) <= self.tolerance_ticks:
            self._pending.clear()
            return self._accept(base, phase, s, t)
        self.rejected += 1
        self._pending.append((base, phase, t, s))
        del self._pending[:-self.confirm_frames]
        if len(self._pending) >= self.confirm_frames and self._consistent():
            base, phase, _, s = self._pending[-1]
            self._pending.clear()
            return self._accept(base, phase, s, t, resync=True)
        return self._coast(t, "rejected")

    def _consistent(self) -> bool:
        last_base, _, last_t, _ = self._pending[-1]
        return all(_window_dist(last_base, b + (last_t - ti) * TICKS_PER_S) <= self.tolerance_ticks
                   for b, _, ti, _ in self._pending)

    def _accept(self, base: int, phase: str, s: int, t: float, resync: bool = False) -> ClockState:
        if resync:
            self._anchor = None
            tick = base + self.unanchored_offset
            source = "resync" if self._tick is not None else "read"
        else:
            if self._base is not None and base > self._base:
                gap = t - self._read_t
                self._anchor = (base, (self._read_t + t) / 2.0) if gap <= self.max_anchor_gap_s else None
            if self._anchor is not None and self._anchor[0] == base:
                est = base + _rint((t - self._anchor[1]) * TICKS_PER_S)
            else:
                est = _rint(self._expected(t))
            est = int(_clamp(est, base, base + TICKS_PER_S - 1))
            tick = max(est, self._tick)
            source = "read"
        lo, hi = self._range(phase)
        tick = int(_clamp(tick, lo, hi))
        self._tick, self._t, self._read_t = tick, t, t
        if resync or self._base is None or base >= self._base:
            self._base = base          # a reading slightly behind (a misread) never moves the base back
        self._phase = PHASE_ENDED if (phase == PHASE_OVERTIME and s == 0) else phase
        anchored = self._anchor is not None and self._anchor[0] == base
        return ClockState(tick, self._phase, 1.0 if anchored else 0.8, anchored, source)

    def _coast(self, t: float, source: str) -> ClockState:
        if self._tick is None:
            return ClockState(None, self._phase, 0.0, False, source)
        if t - self._read_t > self.max_extrapolate_s:
            return ClockState(None, self._phase, 0.0, False, "lost")
        lo, hi = self._range(PHASE_OVERTIME if self._phase in (PHASE_OVERTIME, PHASE_ENDED) else PHASE_REGULATION)
        tick = int(_clamp(max(self._tick, _rint(self._expected(t))), lo, hi))
        self._tick, self._t = tick, t
        return ClockState(tick, self._phase, 0.5, False, source)


# ---------------------------------------------------------------------------
# Elixir
# ---------------------------------------------------------------------------
def regen_ms(tick: int | None) -> int:
    """Milliseconds per elixir for the tick that runs from `tick` (1x when unknown)."""
    if tick is None or tick < DOUBLE_FROM_TICK:
        return REGEN_MS_1X
    if tick < TRIPLE_FROM_TICK:
        return REGEN_MS_2X
    return REGEN_MS_3X


def _rgb16(rgb: np.ndarray):
    a = np.asarray(rgb)[..., :3].astype(np.int16)
    return a[..., 0], a[..., 1], a[..., 2]


def elixir_fill_mask(rgb: np.ndarray) -> np.ndarray:
    """Whole elixir: the bar's magenta/purple, blue >= 150, red >= 110, green at
    least 60 below both (owner's frames: (204, 35, 212), highlight (243, 136, 242))."""
    r, g, b = _rgb16(rgb)
    return (b >= 150) & (r >= 110) & (g + 60 <= r) & (g + 60 <= b)


def elixir_charge_mask(rgb: np.ndarray) -> np.ndarray:
    """The elixir still charging: a lighter blue segment right after the purple
    (owner's frames: (45-52, 78-88, 155-165))."""
    r, g, b = _rgb16(rgb)
    return (r >= 25) & (r < 110) & (b >= 140) & (b - g >= 50) & (b - r >= 80)


def elixir_lit_mask(rgb: np.ndarray) -> np.ndarray:
    return elixir_fill_mask(rgb) | elixir_charge_mask(rgb)


def elixir_track_mask(rgb: np.ndarray) -> np.ndarray:
    """The empty track: dark blue (owner's frames: (0-8, 19-60, 80-127)). The
    HUD panel around the bar, (7, 71, 143), is not track."""
    r, g, b = _rgb16(rgb)
    return (r < 25) & (b >= 70) & (b < 140) & (b - g >= 40)


def bar_fraction(crop: np.ndarray, lit_fn=elixir_lit_mask, track_fn=elixir_track_mask, *,
                 gap_frac: float = 0.01) -> float | None:
    """Fill of a horizontal bar whose track spans the whole crop.

    A column is lit (or track) when at least half its rows are; anything else
    (the elixir count and "Max: 10" drawn over the bar's left end, a divider,
    an anti-aliased edge) is unknown. The fill ends after the last lit
    column, and that end must be visible: track within max(2, gap_frac *
    width) columns after it and mostly track beyond. Returns end / width; 1.0
    when lit to the right edge; 0.0 for a bare track; None when the end is
    hidden, when track shows inside the lit part (not filled from the left),
    or when the crop is no bar at all.
    """
    a = np.asarray(crop)
    if a.ndim != 3 or a.shape[2] < 3 or a.size == 0:
        return None
    lit = lit_fn(a).mean(axis=0) >= 0.5
    trk = (track_fn(a).mean(axis=0) >= 0.5) & ~lit
    w = int(lit.size)
    gap = max(2, _rint(gap_frac * w))
    lit_idx = np.nonzero(lit)[0]
    if lit_idx.size == 0:
        return 0.0 if trk.mean() >= 0.5 else None
    end = int(lit_idx[-1]) + 1
    if int(trk[:end].sum()) > gap:
        return None
    after = np.nonzero(trk[end:])[0]
    if after.size == 0:
        return 1.0 if end >= w - gap else None
    if int(after[0]) > gap or float(trk[end:].mean()) < 0.5:
        return None
    return end / float(w)


@dataclass(frozen=True)
class ElixirRead:
    digit: int | None
    bar_frac: float | None = None
    conf: float = 0.0
    text: str | None = None
    reason: str = ""


class ElixirReader:
    def __init__(self, store: TemplateStore | None, regions: HudRegions, style: GlyphStyle | None = None):
        self.regions = regions
        self.ocr = DigitOCR(store, "elixir", style)

    def read(self, rgb: np.ndarray) -> ElixirRead:
        digit, text, reason, conf = None, None, "", 0.0
        crop = self.regions.crop(rgb, "elixir_digit")
        if crop is not None:
            dr: DigitRead = self.ocr.read(crop)
            text, reason = dr.text, dr.reason
            if dr.text is not None and dr.text.isdigit() and int(dr.text) <= 10:
                digit, conf = int(dr.text), dr.conf
            elif dr.text is not None:
                reason = f"not an elixir count: {dr.text!r}"
        bar = self.regions.crop(rgb, "elixir_bar")
        frac = bar_fraction(bar) if bar is not None else None
        return ElixirRead(digit, frac, conf, text, reason)


@dataclass(frozen=True)
class ElixirState:
    """milli: 0..10000, None when unknown. source: digit+bar | digit-change |
    digit+regen | digit | bar | regen | rejected | unread."""
    milli: int | None
    digit: int | None = None
    conf: float = 0.0
    source: str = "unread"


class ElixirFilter:
    """Digit + bar + regeneration -> elixir_milli (vision-10). See update()."""

    def __init__(self, *, bar_tolerance_milli: float = 150.0, max_anchor_gap_s: float = 0.3,
                 max_extrapolate_s: float = 1.5, drift_milli: float = 200.0):
        self.bar_tolerance_milli = float(bar_tolerance_milli)
        self.max_anchor_gap_s = float(max_anchor_gap_s)
        self.max_extrapolate_s = float(max_extrapolate_s)
        self.drift_milli = float(drift_milli)
        self.reset()

    def reset(self) -> None:
        self._est: float | None = None
        self._t: float | None = None
        self._read_t: float | None = None
        self._digit: int | None = None
        self._jump: int | None = None
        self._anchored = False

    def update(self, digit: int | None, t: float, bar_frac: float | None = None,
               tick: int | None = None) -> ElixirState:
        """One frame. A digit two or more above the estimate (regeneration adds
        one elixir per >= 0.93 s) is a misread unless the bar agrees or the next
        frame shows it again: the estimate continues ("rejected"). Otherwise, in
        order of preference:
        1. digit 10                         -> 10000
        2. digit and a bar within +-tolerance of the digit's window -> the bar
        3. digit one above the last digit, frames <= max_anchor_gap_s apart
           -> digit*1000 + regen since the midpoint of the two frames
        4. digit and an anchored estimate  -> the estimate + regen: inside the
           window as is; below it -> digit*1000; above it -> spent elixir, so
           digit*1000 + the estimate's fraction, unless it ran just past the
           next boundary (fraction < drift_milli) -> digit*1000 + 999
        5. digit alone                      -> digit*1000 (fraction unknown)
        6. bar alone                        -> the bar
        7. nothing                          -> the estimate + regen for up to
           max_extrapolate_s since the last reading, else None
        """
        t = float(t)
        rate = 1e6 / regen_ms(tick)                       # milli per second
        pred = None
        if self._est is not None:
            pred = min(float(MAX_ELIXIR_MILLI), self._est + max(0.0, t - self._t) * rate)
        bar = None if bar_frac is None else float(bar_frac) * MAX_ELIXIR_MILLI
        if digit is not None:
            d = int(digit)
            lo = min(d, 10) * 1000.0
            hi = float(MAX_ELIXIR_MILLI) if d >= 10 else lo + 999.0
            tol = self.bar_tolerance_milli
            bar_agrees = bar is not None and lo - tol <= bar <= hi + tol
            jump = pred is not None and d > math.floor(pred / 1000.0) + 1
            if jump and self._jump != d and not bar_agrees:
                self._jump = d
                est, src, anchored, conf = pred, "rejected", self._anchored, 0.3
            else:
                self._jump = None
                if d >= 10:
                    est, src, anchored, conf = float(MAX_ELIXIR_MILLI), "digit", True, 1.0
                elif bar_agrees:
                    est, src, anchored, conf = bar, "digit+bar", True, 1.0
                elif (self._digit is not None and d == self._digit + 1 and self._t is not None
                      and t - self._t <= self.max_anchor_gap_s):
                    t_change = (self._t + t) / 2.0
                    est, src, anchored, conf = lo + (t - t_change) * rate, "digit-change", True, 0.9
                elif pred is not None and self._anchored:
                    k = math.floor(pred / 1000.0)
                    if k == d:
                        est = pred
                    elif k < d:
                        est = lo
                    else:
                        frac = pred - 1000.0 * k
                        est = hi if (k == d + 1 and frac < self.drift_milli) else lo + frac
                    src, anchored, conf = "digit+regen", True, 0.8
                else:
                    est, src, anchored, conf = lo, "digit", False, 0.5
                est = _clamp(est, lo, hi)
                self._digit, self._read_t = d, t
        elif bar is not None:
            est, src, anchored, conf = bar, "bar", True, 0.6
            self._digit, self._read_t = None, t
        elif pred is not None and self._read_t is not None and t - self._read_t <= self.max_extrapolate_s:
            est, src, anchored, conf = pred, "regen", self._anchored, 0.4
        else:
            return ElixirState(None, None, 0.0, "unread")
        self._est, self._t, self._anchored = est, t, anchored
        milli = int(_clamp(_rint(est), 0, MAX_ELIXIR_MILLI))
        return ElixirState(milli, None if digit is None else int(digit), conf, src)


# ---------------------------------------------------------------------------
# Tower HP
# ---------------------------------------------------------------------------
class TowerHpFilter:
    """Port of scripts/tower_hp.py:232-338 TowerHpFilter for the six TOWER_KEYS.

    update(key, raw, now) takes a raw fraction (0 = bar/digits not visible) and
    returns the accepted one: a reading never rises (within `tolerance`); a fall
    is accepted after `confirm_seconds` and >= 2 readings, at the (upper) median
    of the confirming run; a zero only after `zero_confirm_seconds`, and only
    from an accepted value <= `zero_max_accepted` (occlusion looks like zero);
    a destroyed tower stays destroyed. Every tower starts full.
    """

    ZERO = 0.01

    def __init__(self, confirm_seconds: float = 0.6, zero_confirm_seconds: float = 2.5,
                 tolerance: float = 0.02, zero_max_accepted: float = 0.75) -> None:
        self.confirm_seconds = confirm_seconds
        self.zero_confirm_seconds = zero_confirm_seconds
        self.tolerance = tolerance
        self.zero_max_accepted = zero_max_accepted
        self.reset()

    def reset(self) -> None:
        self._accepted: dict[str, float] = {k: 1.0 for k in TOWER_KEYS}
        self._pending: dict[str, list[tuple[float, float]]] = {}
        self._zero_run: dict[str, float] = {}

    def accepted(self, key: str) -> float | None:
        return self._accepted.get(key)

    def update(self, key: str, raw: float, now: float) -> float:
        accepted = self._accepted.get(key)
        if accepted is None:
            self._accepted[key] = raw
            return raw
        if accepted <= self.ZERO:
            return 0.0
        if raw <= self.ZERO:
            if accepted > self.zero_max_accepted:
                self._zero_run.pop(key, None)
                return accepted
            run = self._zero_run.setdefault(key, now)
            if now - run >= self.zero_confirm_seconds:
                self._accepted[key] = 0.0
            return self._accepted[key]
        self._zero_run.pop(key, None)
        if raw >= accepted - self.tolerance:
            self._pending.pop(key, None)
            return accepted
        run = self._pending.setdefault(key, [])
        run.append((now, raw))
        if now - run[0][0] >= self.confirm_seconds and len(run) >= 2:
            values = sorted(v for _, v in run)
            self._accepted[key] = min(accepted, values[len(values) // 2])
            self._pending.pop(key, None)
        return self._accepted[key]


@dataclass(frozen=True)
class TowerRead:
    hp: int | None
    conf: float = 0.0
    text: str | None = None
    reason: str = ""


class TowerReader:
    """HP digits by each tower (font 'tower', TOWER_GLYPH_STYLE unless `style`
    is given) and, for the princess towers, the bar's colour fill as a
    fallback fraction."""

    def __init__(self, store: TemplateStore | None, regions: HudRegions, style: GlyphStyle | None = None,
                 *, use_bars: bool = True):
        self.regions = regions
        self.ocr = DigitOCR(store, "tower", style)
        self.use_bars = bool(use_bars)

    @property
    def keys_with_source(self) -> set[str]:
        keys: set[str] = set()
        if self.ocr.ready:
            keys |= {k for k, r in TOWER_DIGIT_REGIONS.items() if self.regions.get(r) is not None}
        if self.use_bars:
            keys |= {k for k, r in TOWER_BAR_REGIONS.items() if self.regions.get(r) is not None}
        return keys

    def read_digits(self, rgb: np.ndarray) -> dict[str, TowerRead]:
        out = {}
        for key, name in TOWER_DIGIT_REGIONS.items():
            crop = self.regions.crop(rgb, name)
            if crop is None or not self.ocr.ready:
                out[key] = TowerRead(None, 0.0, None, "no region" if crop is None else "no tower templates")
                continue
            dr = self.ocr.read(crop)
            text = None if dr.text is None else dr.text.replace(":", "")
            if text and text.isdigit() and 0 < int(text) <= MAX_TOWER_HP:
                out[key] = TowerRead(int(text), dr.conf, dr.text)
            else:
                out[key] = TowerRead(None, 0.0, dr.text, dr.reason or f"not a tower HP: {dr.text!r}")
        return out

    def read_bars(self, rgb: np.ndarray) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        if not self.use_bars:
            return out
        h, w = np.asarray(rgb).shape[:2]
        for key, name in TOWER_BAR_REGIONS.items():
            crop = self.regions.crop(rgb, name)
            if crop is None or crop.size == 0:
                out[key] = None
                continue
            margin = max(1, _rint(self.regions.bar_margin * w))
            full = crop.shape[1] - 2 * margin
            if full <= 0:
                out[key] = None
                continue
            mask = ally_mask(crop) if key.startswith("own") else enemy_mask(crop)
            out[key] = measure_tower_bar(mask, full, anchor_px=margin, tol_px=margin,
                                         gap_px=max(1, _rint(2 * w / REF_SIZE[0])),
                                         min_run=max(2, _rint(10 * w / REF_SIZE[0])),
                                         min_rows=max(2, _rint(4 * h / REF_SIZE[1])))
        return out


class TowerHpTracker:
    """Tower readings -> TowerHp per TOWER_KEYS, through TowerHpFilter.

    max HP: `max_hp[key]` when given (fixed), else `prior_max_hp[key]` (e.g.
    LEVEL11_TOWER_MAX_HP) or nothing, raised by a higher digit reading seen
    `confirm_max_reads` times in a row (so without a prior the first reading
    confirmed twice sets it: towers are full at the start). Per frame a
    tower's raw fraction is digits / max, else the bar fraction, else 0.0
    (nothing visible: TowerHpFilter's occlusion case) once the tower has been
    read at least once. Before its first reading a tower is TowerHp(None, max),
    except a king with a known max, which is TowerHp(max, max) while
    `kings_full_until_read`: the game shows a king's number only once it is
    damaged.
    """

    def __init__(self, max_hp: Mapping[str, int] | None = None, *, confirm_max_reads: int = 2,
                 filt: TowerHpFilter | None = None, prior_max_hp: Mapping[str, int] | None = None,
                 kings_full_until_read: bool = True):
        self.known_max = {k: int(v) for k, v in (max_hp or {}).items() if k in TOWER_KEYS and v}
        self.prior_max = {k: int(v) for k, v in (prior_max_hp or {}).items() if k in TOWER_KEYS and v}
        self.confirm_max_reads = max(1, int(confirm_max_reads))
        self.kings_full_until_read = bool(kings_full_until_read)
        self.filter = filt or TowerHpFilter()
        self.reset()

    def reset(self) -> None:
        self.filter.reset()
        self.max: dict[str, int] = {**self.prior_max, **self.known_max}
        self._cand: dict[str, tuple[int, int]] = {}
        self.frac: dict[str, float | None] = {k: None for k in TOWER_KEYS}

    def _learn_max(self, key: str, hp: int) -> None:
        if key in self.known_max:
            return
        mx = self.max.get(key)
        if mx is not None and hp <= mx:
            self._cand.pop(key, None)
            return
        c = self._cand.get(key)
        n = c[1] + 1 if c is not None and c[0] == hp else 1
        self._cand[key] = (hp, n)
        if n >= self.confirm_max_reads:
            self.max[key] = hp
            self._cand.pop(key, None)

    def update(self, hp_reads: Mapping[str, int | None], now: float,
               bar_fracs: Mapping[str, float | None] | None = None) -> dict[str, TowerHp]:
        bars = bar_fracs or {}
        out: dict[str, TowerHp] = {}
        for key in TOWER_KEYS:
            hp = hp_reads.get(key)
            if hp is not None:
                self._learn_max(key, int(hp))
            mx = self.max.get(key)
            if hp is not None and mx:
                raw = min(1.0, int(hp) / float(mx))
            elif bars.get(key) is not None:
                raw = float(bars[key])
            elif self.frac[key] is not None:
                raw = 0.0
            else:
                raw = None
            if raw is not None:
                self.frac[key] = self.filter.update(key, raw, float(now))
            acc = self.frac[key]
            if acc is not None and acc <= TowerHpFilter.ZERO:
                out[key] = TowerHp(0, mx)
            elif acc is not None and mx:
                out[key] = TowerHp(int(round(acc * mx)), mx)
            elif acc is None and hp is None and mx and self.kings_full_until_read and key.endswith("king"):
                out[key] = TowerHp(mx, mx)
            else:
                out[key] = TowerHp(None if hp is None else int(hp), mx)
        return out

    def crowns(self) -> tuple[int, int]:
        """(own, enemy) crowns from destroyed towers: a king is worth all three."""
        def count(side: str) -> int:
            if self.frac[f"{side}_king"] == 0.0:
                return 3
            return sum(1 for s in ("left", "right") if self.frac[f"{side}_{s}"] == 0.0)
        return count("enemy"), count("own")


# ---------------------------------------------------------------------------
# Hand
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlotRead:
    """card: catalogue id, EMPTY_CARD for an empty slot, None when unsure.
    scores: card id -> best NCC (every deck card with a template)."""
    card: int | None
    score: float = 0.0
    margin: float = 0.0
    variant: str = "base"
    scores: Mapping[int, float] = field(default_factory=dict)
    empty: bool = False

    @property
    def form(self) -> int:
        return VARIANT_FORM.get(self.variant, FORM_BASE)

    @property
    def greyed(self) -> bool:
        return self.variant.endswith("greyed")


@dataclass(frozen=True)
class HandReading:
    slots: tuple[SlotRead, ...]
    next: SlotRead


class HandReader:
    """Per-frame NCC of the 4 hand slots and the next box (cards.py:89-103 rules).

    card_ids maps template labels (RoyaleSim card names) to catalogue ids; only
    those labels are matched. A crop flatter than `empty_std` grey levels, or
    one an "empty" template beats every card on, is EMPTY_CARD.
    """

    def __init__(self, store: TemplateStore | None, card_ids: Mapping[str, int], regions: HudRegions, *,
                 min_score: float = MIN_SCORE, min_margin: float = MIN_MARGIN, empty_std: float = 8.0):
        self.regions = regions
        self.card_ids = {str(k): int(v) for k, v in card_ids.items()}
        self.min_score = float(min_score)
        self.empty_std = float(empty_std)
        labels = set(self.card_ids)
        hand = store.entries(KIND_HAND) if store is not None else []
        nxt = store.entries(KIND_NEXT) if store is not None else []
        self.hand = CardMatcher(hand, PATCH, labels=labels, min_score=min_score, min_margin=min_margin)
        own_next = CardMatcher(nxt, PATCH, labels=labels, min_score=min_score, min_margin=min_margin)
        self.next = own_next if own_next.ready else self.hand
        self.empty_hand = CardMatcher(store.entries(KIND_EMPTY, "hand") if store is not None else [], PATCH)
        self.empty_next = CardMatcher(store.entries(KIND_EMPTY, "next") if store is not None else [], PATCH)

    @property
    def ready(self) -> bool:
        return self.hand.ready

    def read_crop(self, crop: np.ndarray | None, matcher: CardMatcher, empty: CardMatcher) -> SlotRead:
        if crop is None or np.asarray(crop).size == 0:
            return SlotRead(None)
        if crop_std(crop) < self.empty_std:
            return SlotRead(EMPTY_CARD, 1.0, 1.0, empty=True)
        m = matcher.match(crop)
        scores = {self.card_ids[lab]: s for lab, s in m.scores.items() if lab in self.card_ids}
        if empty.ready:
            e = empty.match(crop)
            if e.score >= self.min_score and e.score > m.score:
                return SlotRead(EMPTY_CARD, e.score, e.score - m.score, "base", scores, True)
        card = self.card_ids.get(m.label) if m.label is not None else None
        return SlotRead(card, m.score, m.margin, m.variant, scores)

    def read(self, rgb: np.ndarray) -> HandReading:
        slots = tuple(self.read_crop(self.regions.hand(i).crop(rgb), self.hand, self.empty_hand)
                      for i in range(HAND_SIZE))
        return HandReading(slots, self.read_crop(self.regions.crop(rgb, "next"), self.next, self.empty_next))


@dataclass(frozen=True)
class HandState:
    hand: tuple[int, ...]               # EMPTY_CARD for an empty or unread slot
    next: int                           # EMPTY_CARD when unread
    known: tuple[bool, ...]             # slot read (a card, or an empty slot)
    cycle: tuple[int | None, ...] = ()  # cards not in hand, in the order they return; [0] is next
    corrections: int = 0                # times the FIFO prior overrode the art


class HandTracker:
    """Hand readings over time -> 4 distinct cards, empty slots, the next card.

    State: the hand (card id, EMPTY_CARD while a played slot waits for its
    refill, None unknown), the cycle (cards not in hand in the order they come
    back; cycle[0] is the next card; None for a card not yet seen) and the
    pending refills. With `deck` (our 8 ids) a single unseen card is inferred.

    Per frame:
    1. Duplicates among the slot readings: the weaker one takes its best unused
       alternative (if it clears min_score/min_margin) or becomes unsure.
    2. A slot that showed card c and now reads empty: c was played; it goes to
       the back of the cycle and the slot waits for a refill.
    3. A waiting slot that reads a card x: the prior says the k-th lowest
       waiting slot gets cycle[k]; when x is not that card and the expected
       card still scores >= prior_min_score (or beats x by less than
       prior_override_margin), the expected card is taken: the FIFO prior
       corrects the misread.
    4. A slot that changes card without an empty frame: accepted at once when
       the new card is the (old) next card, else after `change_votes` frames.
    5. Unsure readings hold the last value. The next box is never a card in
       hand; a new next card fills the cycle's first unknown place.
    """

    def __init__(self, deck: Sequence[int] | None = None, *, prior_min_score: float = 0.25,
                 prior_override_margin: float = 0.15, change_votes: int = 3,
                 min_score: float = MIN_SCORE, min_margin: float = MIN_MARGIN):
        self.deck = [int(c) for c in deck] if deck else None
        self.prior_min_score = float(prior_min_score)
        self.prior_override_margin = float(prior_override_margin)
        self.change_votes = max(1, int(change_votes))
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self.reset()

    def reset(self) -> None:
        self.hand: list[int | None] = [None] * HAND_SIZE
        self.next: int | None = None
        self.cycle: list[int | None] = []
        self.pending: dict[int, float] = {}
        self._votes: dict[int, tuple[int, int]] = {}
        self.started = False
        self.plays: list[tuple[float, int, int]] = []       # (t, slot, card) seen played
        self.corrections = 0

    # -- helpers ---------------------------------------------------------------
    def _alternative(self, sr: SlotRead, used: set[int]) -> int | None:
        ranked = sorted(((s, c) for c, s in sr.scores.items() if c not in used), key=lambda sc: (-sc[0], sc[1]))
        if not ranked or ranked[0][0] < self.min_score:
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.min_margin:
            return None
        return ranked[0][1]

    def _distinct(self, slots: Sequence[SlotRead]) -> list[int | None]:
        cards = [s.card for s in slots]
        while True:
            seen: dict[int, int] = {}
            dup = None
            for i, c in enumerate(cards):
                if c is None or c < 0:
                    continue
                if c in seen:
                    dup = (seen[c], i)
                    break
                seen[c] = i
            if dup is None:
                return cards
            i, j = dup
            weak = i if slots[i].score < slots[j].score else j
            used = {c for k, c in enumerate(cards) if k != weak and c is not None and c >= 0}
            cards[weak] = self._alternative(slots[weak], used)

    def _infer(self) -> None:
        if not self.deck:
            return
        missing = [c for c in self.deck if c not in self.hand and c not in self.cycle]
        holes = [i for i, c in enumerate(self.cycle) if c is None]
        if len(missing) == 1 and len(holes) == 1:
            self.cycle[holes[0]] = missing[0]

    def _start(self, raw: list[int | None], nxt: int | None, t: float) -> None:
        self.hand = list(raw)
        self.started = True
        for slot, c in enumerate(raw):
            if c == EMPTY_CARD:
                self.pending[slot] = t
        cards = sum(1 for c in raw if c is not None and c >= 0)
        size = (len(self.deck) if self.deck else 8) - cards
        self.cycle = [None] * max(1, size)
        if nxt is not None and nxt not in raw:
            self.next = nxt
            self.cycle[0] = nxt
        self._infer()

    def _play(self, slot: int, card: int, t: float) -> None:
        self.hand[slot] = EMPTY_CARD
        self.pending[slot] = t
        self.cycle.append(card)
        self.plays.append((t, slot, card))
        self._votes.pop(slot, None)

    def _refill(self, slot: int, card: int) -> None:
        self.hand[slot] = card
        self.pending.pop(slot, None)
        self._votes.pop(slot, None)
        if card in self.cycle:
            self.cycle.remove(card)
        elif None in self.cycle:
            self.cycle.remove(None)
        elif self.deck:                                   # inconsistent: rebuild from the deck
            self.cycle = [c for c in self.deck if c not in self.hand]
        self._infer()

    def _prefer(self, expected: int | None, r: int, sr: SlotRead) -> int:
        if expected is None or expected < 0 or r == expected:
            return r
        s_exp = sr.scores.get(expected, -1.0)
        s_r = sr.scores.get(r, sr.score)
        if s_exp >= self.prior_min_score or s_r - s_exp < self.prior_override_margin:
            self.corrections += 1
            return expected
        return r

    def _others(self, slot: int) -> set[int]:
        return {c for i, c in enumerate(self.hand) if i != slot and c is not None and c >= 0}

    # -- update ----------------------------------------------------------------
    def update(self, reading: HandReading, t: float) -> HandState:
        raw = self._distinct(reading.slots)
        nraw = reading.next.card if reading.next is not None else None
        if nraw is not None and nraw < 0:
            nraw = None
        if not self.started:
            if any(c is not None and c >= 0 for c in raw):
                self._start(raw, nraw, t)
            return self.state()
        old_next = self.next
        for slot in range(HAND_SIZE):
            r, prev, sr = raw[slot], self.hand[slot], reading.slots[slot]
            if slot in self.pending:
                if r is None or r == EMPTY_CARD:
                    continue
                # Refills go to the lowest empty slot first: the k-th lowest slot
                # still waiting gets cycle[k] (cycle[0] already left on a refill
                # earlier in this loop).
                k = sorted(self.pending).index(slot)
                expected = self.cycle[k] if k < len(self.cycle) else None
                card = self._prefer(expected, r, sr)
                if card in self._others(slot):
                    if expected is None or expected in self._others(slot):
                        continue
                    card = expected
                self._refill(slot, card)
                continue
            if r is None:
                continue
            if r == EMPTY_CARD:
                if prev is not None and prev >= 0:
                    self._play(slot, prev, t)
                else:
                    self.hand[slot] = EMPTY_CARD
                    self.pending[slot] = t
                continue
            if r in self._others(slot):
                continue
            if prev is None:
                self.hand[slot] = r
                if r in self.cycle:
                    self.cycle.remove(r)
                elif None in self.cycle and len(self.cycle) > 1:
                    self.cycle.remove(None)
                continue
            if r == prev:
                self._votes.pop(slot, None)
                continue
            if old_next is not None and r == old_next:
                self._play(slot, prev, t)
                self._refill(slot, r)
                continue
            v = self._votes.get(slot)
            n = v[1] + 1 if v is not None and v[0] == r else 1
            self._votes[slot] = (r, n)
            if n >= self.change_votes:
                self._play(slot, prev, t)
                self._refill(slot, r)
        # The next box, checked against the hand AFTER the slots were settled (a
        # misread slot may have shown the next card's art).
        in_hand = self._others(-1)
        if nraw is not None and nraw not in in_hand:
            self.next = nraw
            if nraw not in self.cycle and None in self.cycle:
                self.cycle[self.cycle.index(None)] = nraw
            self._infer()
        elif self.next is None or self.next in in_hand:
            self.next = self.cycle[0] if self.cycle and self.cycle[0] is not None else None
        return self.state()

    def state(self) -> HandState:
        hand = tuple(EMPTY_CARD if c is None else int(c) for c in self.hand)
        known = tuple(c is not None for c in self.hand)
        nxt = EMPTY_CARD if self.next is None else int(self.next)
        return HandState(hand, nxt, known, tuple(self.cycle), self.corrections)


# ---------------------------------------------------------------------------
# The facade
# ---------------------------------------------------------------------------
@dataclass
class HudReport:
    """What HudReader.read saw, for logging and debugging."""
    t: float = 0.0
    clock_read: ClockRead | None = None
    clock: ClockState | None = None
    elixir_read: ElixirRead | None = None
    elixir: ElixirState | None = None
    tower_digits: dict[str, TowerRead] = field(default_factory=dict)
    tower_bars: dict[str, float | None] = field(default_factory=dict)
    tower_frac: dict[str, float | None] = field(default_factory=dict)
    hand_reading: HandReading | None = None
    hand: HandState | None = None
    errors: list[str] = field(default_factory=list)
    ms: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        c, e, h = self.clock, self.elixir, self.hand
        return {"t": round(self.t, 4),
                "clock": None if c is None else {"tick": c.tick, "phase": c.phase, "conf": c.conf,
                                                 "source": c.source,
                                                 "text": None if self.clock_read is None else self.clock_read.text},
                "elixir": None if e is None else {"milli": e.milli, "digit": e.digit, "source": e.source,
                                                  "bar": None if self.elixir_read is None else self.elixir_read.bar_frac},
                "towers": {k: v for k, v in self.tower_frac.items()},
                "hand": None if h is None else {"hand": list(h.hand), "next": h.next, "known": list(h.known),
                                                "corrections": h.corrections},
                "errors": list(self.errors), "ms": {k: round(v, 2) for k, v in self.ms.items()}}


class HudReader:
    """All HUD readers and their filters: read(rgb, t) -> scripts.vision.types.Hud.

    Never raises: a failing reader leaves its fields unread and adds a line to
    report.errors. Call reset() at the start of each battle and feed it only
    in-battle frames (VisionFrame.screen == "battle"): the filters assume one
    continuous battle, and menu pixels would read as bars.

    card_ids maps the hand template labels (RoyaleSim card names) to catalogue
    ids; with exactly 8 ids they are also the deck the hand tracker infers
    unseen cards from (or pass `deck`). tower_max_hp (TOWER_KEYS -> HP) fixes
    the maxima; tower_prior_max_hp (default LEVEL11_TOWER_MAX_HP, friendly
    battles) is where they start, so the bar fractions give hit points before
    any digit template exists; None learns them from the digits alone.
    """

    def __init__(self, store: TemplateStore | None = None, card_ids: Mapping[str, int] | None = None, *,
                 regions: HudRegions | None = None, deck: Sequence[int] | None = None,
                 tower_max_hp: Mapping[str, int] | None = None, clock_offset_ticks: int = 0,
                 glyph_styles: Mapping[str, GlyphStyle] | None = None, use_tower_bars: bool = True,
                 tower_prior_max_hp: Mapping[str, int] | None = LEVEL11_TOWER_MAX_HP):
        self.store = store if store is not None else TemplateStore()
        self.regions = regions if regions is not None else load_regions()
        ids = {str(k): int(v) for k, v in (card_ids or {}).items()}
        styles = dict(glyph_styles or {})
        self.clock_reader = ClockReader(self.store, self.regions, styles.get("clock"))
        self.elixir_reader = ElixirReader(self.store, self.regions, styles.get("elixir"))
        self.tower_reader = TowerReader(self.store, self.regions, styles.get("tower"), use_bars=use_tower_bars)
        self.hand_reader = HandReader(self.store, ids, self.regions)
        if deck is None and len(set(ids.values())) == 8:
            deck = sorted(set(ids.values()))
        self.clock = ClockFilter(offset_ticks=clock_offset_ticks)
        self.elixir = ElixirFilter()
        self.towers = TowerHpTracker(tower_max_hp, prior_max_hp=tower_prior_max_hp)
        self.hand = HandTracker(deck)
        self.report = HudReport()

    def reset(self) -> None:
        self.clock.reset()
        self.elixir.reset()
        self.towers.reset()
        self.hand.reset()

    def _run(self, rep: HudReport, name: str, fn) -> Any:
        t0 = time.perf_counter()
        try:
            return fn()
        except Exception as exc:                       # never raise into the live loop
            rep.errors.append(f"{name}: {type(exc).__name__}: {exc}")
            return None
        finally:
            rep.ms[name] = (time.perf_counter() - t0) * 1000.0

    def read(self, rgb: np.ndarray, t: float | None = None) -> Hud:
        t = now() if t is None else float(t)
        rep = HudReport(t=t)
        self.report = rep
        hud = Hud()
        try:
            img = np.asarray(rgb)
            if img.ndim != 3 or img.shape[2] < 3:
                raise ValueError(f"expected an HxWx3 RGB frame, got {img.shape}")
            check_frame_size(img.shape[1], img.shape[0])
        except Exception as exc:
            rep.errors.append(f"frame: {type(exc).__name__}: {exc}")
            return hud

        def clock():
            cr = self.clock_reader.read(img)
            rep.clock_read = cr
            rep.clock = self.clock.update(cr.seconds_left, t, cr.phase_hint)
            hud.clock_tick, hud.phase = rep.clock.tick, rep.clock.phase

        def elixir():
            er = self.elixir_reader.read(img)
            rep.elixir_read = er
            rep.elixir = self.elixir.update(er.digit, t, er.bar_frac, hud.clock_tick)
            hud.elixir_milli = rep.elixir.milli

        def towers():
            rep.tower_digits = self.tower_reader.read_digits(img)
            rep.tower_bars = self.tower_reader.read_bars(img)
            hud.towers = self.towers.update({k: r.hp for k, r in rep.tower_digits.items()}, t, rep.tower_bars)
            rep.tower_frac = dict(self.towers.frac)
            hud.crowns = self.towers.crowns()

        def hand():
            hr = self.hand_reader.read(img)
            rep.hand_reading = hr
            rep.hand = self.hand.update(hr, t)
            hud.hand, hud.next = list(rep.hand.hand), rep.hand.next

        self._run(rep, "clock", clock)
        self._run(rep, "elixir", elixir)
        self._run(rep, "towers", towers)
        self._run(rep, "hand", hand)
        return hud


__all__ = [
    "BUILTIN_REGIONS_PX", "ClockFilter", "ClockRead", "ClockReader", "ClockState", "ElixirFilter",
    "ElixirRead", "ElixirReader", "ElixirState", "HandReader", "HandReading", "HandState", "HandTracker",
    "HudReader", "HudRegions", "HudReport", "LEVEL11_TOWER_MAX_HP", "Rect", "SlotRead", "TowerHpFilter",
    "TowerHpTracker", "TowerRead", "TowerReader", "bar_fraction", "clock_tick", "elixir_charge_mask",
    "elixir_fill_mask", "elixir_lit_mask", "elixir_track_mask", "load_regions", "parse_clock_text",
    "regen_ms",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"size must look like 1080x1920, not {text!r}") from exc


def _cmd_regions(args) -> int:
    regions = load_regions(args.regions)
    size = args.size or REF_SIZE
    doc = regions.to_dict(size)
    if args.out:
        regions.save(args.out, size)
        print(f"wrote {args.out}")
        return 0
    print(f"{regions.name} at {size[0]}x{size[1]}:")
    for name in regions.names():
        print(f"   {name:18s} {list(regions.px(name, size))}  {doc['regions'].get(name, {}).get('source', '')}")
    return 0


def _cmd_overlay(args) -> int:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("overlay needs Pillow (pip install pillow)")
        return 1
    from .record import load_image
    regions = load_regions(args.regions)
    rgb = load_image(args.image)
    size = check_frame_size(rgb.shape[1], rgb.shape[0])
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for name in regions.names():
        x0, y0, x1, y1 = regions.px(name, size)
        colour = (255, 0, 255) if name.startswith("hand") or name == "next" else (
            (0, 255, 255) if name.startswith(("tower", "bar")) else (255, 255, 0))
        draw.rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], outline=colour, width=2)
        draw.text((x0 + 2, max(0, y0 - 12)), name, fill=colour)
    image.save(args.out)
    print(f"wrote {args.out}")
    return 0


def _cmd_read(args) -> int:
    from .record import load_image
    deck = [c.strip() for c in (args.deck or "").split(",") if c.strip()]
    ids = {name: i for i, name in enumerate(deck)}
    store = TemplateStore(args.root) if args.root else TemplateStore()
    reader = HudReader(store, ids, regions=load_regions(args.regions))
    names = {i: n for n, i in ids.items()}
    for k, path in enumerate(args.image):
        hud = reader.read(load_image(path), float(k) * args.dt)
        hand = [names.get(c, "-") for c in hud.hand]
        print(f"{path}: tick {hud.clock_tick} ({hud.phase})  elixir {hud.elixir_milli}  hand {hand} "
              f"next {names.get(hud.next, '-')}  crowns {hud.crowns}")
        print("   towers " + "  ".join(f"{k2} {v.hp}/{v.max}" for k2, v in hud.towers.items()))
        print("   " + json.dumps(reader.report.summary(), default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.vision.hud",
                                 description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("regions", help="print the HUD boxes in use (or write them with --out)")
    p.add_argument("--regions", help="regions JSON (default: load_regions())")
    p.add_argument("--size", type=_parse_size, help="pixel size to print/write at (default 1080x1920)")
    p.add_argument("--out", help="write a regions file (e.g. data/vision/hud_regions_pc.json) to edit")
    p.set_defaults(func=_cmd_regions)
    p = sub.add_parser("overlay", help="draw the HUD boxes on a screenshot")
    p.add_argument("--image", required=True)
    p.add_argument("--regions")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_overlay)
    p = sub.add_parser("read", help="run the HUD readers on screenshots (in order, --dt s apart)")
    p.add_argument("--image", required=True, nargs="+")
    p.add_argument("--regions")
    p.add_argument("--root", help="template folder (default data/vision/templates)")
    p.add_argument("--deck", help="card names (template labels) to match, comma-separated")
    p.add_argument("--dt", type=float, default=0.25, help="seconds between the screenshots")
    p.set_defaults(func=_cmd_read)
    args = ap.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
