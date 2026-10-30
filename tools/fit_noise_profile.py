r"""Fit the perception-noise profile (E17) to what perception gets wrong on this rig (V11).

    .venvs\buildabot\Scripts\python.exe tools\fit_noise_profile.py --labels D:\labels\friendly1 reports\vision\replays\<recording>_replay_<time>
    .venvs\buildabot\Scripts\python.exe tools\fit_noise_profile.py --taps reports\vision_play --last 3
    .venvs\buildabot\Scripts\python.exe tools\fit_noise_profile.py --labels D:\labels\friendly1 <replay> --taps reports\vision_play --last 3 --report reports\vision\noise_fit.json
    then  tools\perception_noise.py data\noise\measured.json           describe and check it
          tools\rs_rows.py ... --noise data\noise\measured.json        rows (and rs_eval --noise)

Writes data/noise/measured.json (--out): a complete tools/perception_noise.py NoiseProfile.
Every value this tool can measure is replaced by the measured one; every other value is copied
from the --base profile (default: the shipped prior, data/noise/default_prior.json). Each group
carries a "_fit" note (which values were measured, from what, on how many samples; which were
kept and why) and the file a top-level "_fit" (base, inputs, settings, warnings).
NoiseProfile.load skips "_" keys, so the file loads like any other profile. This tool only reads
logs: nothing here taps, captures or touches the emulator, and nothing reads the game's memory.

TWO KINDS OF GROUND TRUTH

(a) LABELLED FRAMES: the positional SESSIONS, with --labels FOLDER. A session is anything in the
recorder's format (scripts/vision/record.py): a tools/vision_replay.py output (its frames name
the image they replayed, extra.replay.image), or a recorded vision_play / capture session whose
frames saved an image (the frame's "image"). The labels are what vision_replay --labels reads:
YOLO txt files named like the images (00000012.txt; found at any depth), class names from
--label-names or a classes.txt / detector_labels.json / cr.yaml / data.yaml in the folder,
every name through the detector's crosswalk (scripts/vision/labels.resolve_label, the form
kept: Skeleton_EV1 is a Skeleton, form evo), the side from an @ally / @enemy suffix or the
KataCR 6th column (0 ally, the bottom player; 1 enemy). Only UNIT labels count: towers, spells,
effects and UI are not units in the noise model, and a label name the crosswalk cannot place is
skipped (and counted). Each label's world ground point comes from its box through the
ArenaGeometry (--geometry: give the replay's), exactly as a detection's does (box bottom-centre;
the flyer offset for flyers).

What it is compared with, per labelled frame (--level, auto by default):
  tracks       the frame's TRACKS: the units the public state is built from, after the
               tracker's coasting, min_hits confirmation and team vote. A track matches a label
               within --radius-mt (1000) in the world. auto picks it when the tracker ran on
               the labelled frames and the session's tracked frames are dense (median gap
               <= 1 s, the tracker's max_age).
  detections   the frame's raw DETECTIONS, matched by IoU >= --iou (0.5) of the normalised
               boxes. auto falls back to it for a sparse replay (images every N frames: the
               tracks expire between them) or labelled frames the tracker did not run on.
               Per-frame detector errors OVERSTATE what reaches the model (the tracker bridges
               short gaps, drops one-frame ghosts and votes the side over a track's life);
               the fit warns. Record densely (vision_play --record-images 1, capture
               --image-every 1) and replay for track-level figures.
Perceived objects are read as tools/public_state.py reads them: a name the crosswalk cannot
place is left out of the state, so it is left out here (counted; the labelled unit under it is
a drop), and a track whose team vote is undecided takes the side of the board half it stands
in (public_state unknown_team "half", vision_play's setting; counted as by_half). A detection
with side_prob 0.5 has no side (counted as undecided, left out of the side figures).
Matching is greedy and in two rounds: same class first, nearest (or best IoU) first, so a
look-alike next to a unit cannot take its correct match; then any class (a confusion). Ties go
to the earlier label, then the earlier prediction. A level-tracks session uses only labelled
frames the tracker ran on; a level-detections one, frames the detector ran on (a live frame:
one that holds detections).

(b) OWN TAPS: --taps, vision_play sessions run with --tap (or replays of them, which copy the
taps). A tap record says the card and the tile aimed at (its centre; world = screen for the
bot, team 1) and the decision tick. Its play is the earliest OWN track (team 1 in more of the
frames it was reported in) that carries one of the card's labels (tools/latency_probe
.CardClasses), was first seen -2..60 ticks after the decision and, for troops and buildings,
within 4000 mt of the aim (tools/latency_probe.ProbeConfig), claimed with the other units of
the same play (Skeletons' three), exactly as tools/latency_probe.py matches it. Spells are not
units here and are skipped. A report's tick is the tracker's: a replay's extra.replay.track_tick,
else the frame's tick, else (the clock unread) the newest last_seen_tick of the frame's tracks.

WHAT IS MEASURED, AND THE FALLBACKS (the --base profile's value is kept, and the note says why,
when a source is missing or has fewer samples than the minimum):

  drop             (a) unmatched labelled units / labelled units, per size bucket (its class's
                   level-11 hit points against the base profile's size bounds; buildings
                   apart): >= --min-samples (30) units in the bucket. by_class: classes with
                   >= --min-class-samples (50) labelled units get their own rate (the base
                   profile's other by_class entries are kept). At level tracks this includes
                   units too young to be confirmed (new_units), a small overstatement.
  forms            (a) evo_to_base_p / hero_to_base_p: labelled evolutions / heroes of a form
                   the detector has a label for (classes.json "katacr"), matched to their own
                   class, read as the base form; >= --min-samples of them. unlabelled_to_base
                   is kept (the fit warns when unlabelled forms are mostly read as their form).
                   KataCR labels no hero, so hero_to_base_p stays at the base value until a
                   detector learns one.
  confusion        (a) pairs: for each class with >= --min-class-samples matched units, its
                   confusions with other unit classes / its matched units (rounded DOWN to 4
                   decimals, so a row never adds up above 1); a measured class with no
                   confusion loses its base row; other classes keep theirs. unknown_p is kept:
                   an unplaceable name never reaches the state (it is a drop here), and a unit
                   reads as token 1 only when its class has no engine unit type, a property of
                   the class table, not a perception error.
  side_flip        (a) matched units of a known side read as the other side, over those with a
                   side (undecided tracks by their half, see above); >= --min-samples. Own-tap
                   units are not used: our taps vote our own units' team (tap_llr), so they say
                   nothing of the opponent's flips.
  position         sigma_x_mt / sigma_y_mt, the root mean square error per screen axis (the
                   profile has no ground bias, so bias and spread go in together; the bias is
                   reported, and a large one is a warning to refit the geometry):
                   (b) first: single-unit troop plays (data/vision/spawn_children.json
                   compositions; one unit seen), every report while the unit still deploys
                   (it does not move: tick < first sighting + deploy time - 3) against the
                   aimed tile's centre; >= --min-tracks (10) such tracks. Buildings are left
                   out (an even footprint snaps half a tile from the aim), and so are tracks
                   first reported more than 2500 mt from the aim (counted as outliers).
                   else (a): same-class matches of ground units, perceived minus labelled
                   ground point (>= --min-samples). This is a LOWER bound: the label's ground
                   point goes through the same geometry, so a wrong pixel map is not in it.
                   flyer_dx_mt / flyer_dy_mt / flyer_sigma_mt are always kept: the bot's deck
                   has no flyers, and labels pass through the detector's own flyer offset.
  new_units        (b) delay_ticks = the base profile's deploy.age_offset_ticks (the first
                   sighting's lag after the unit appears, which the logs cannot separate from
                   D) + the median of first report - first sighting (the tracker's
                   confirmation), rounded half up, over every claimed own unit (at least 0: a
                   negative one, reports dated before the first sighting, warns); jitter_ticks =
                   half the p10..p90 spread of the latter, rounded half up; >= --min-tracks.
  false_positives  (a) rate: unmatched perceived units per scored frame (>= --min-samples
                   frames). Each phantom is "near" when a labelled unit is within 2000 mt (a
                   double box) else "far". With >= --min-samples phantoms their shape is fitted
                   too: mostly near (the default model) gives classes {} and, with
                   >= --min-samples near phantoms, near_sigma_mt = the root mean square offset
                   per axis from the nearest labelled unit; mostly far gives classes = the far
                   phantoms' classes as weights.
  KEPT             board (detection delay is inside D: tools/latency_probe.py measures it and
                   E11 decision-time rows model it), hp (labels carry no hit points, and own
                   units are seen only full), deploy (inside D, see new_units), towers, elixir,
                   clock (the logs hold no true tower hit points, elixir or tick: no memory
                   reading), opponent_plays (an opponent's true card and tick are unknown
                   offline), size (a setting, not a noise).

Rounding: probabilities to 4 decimals (Python's round), confusion entries down to 4 decimals,
millitiles to 0.1, ticks to whole ticks half up. If nothing could be measured the file is not
written (exit 1).

API: FitConfig, Obj, match_objects, Canon, LabelFiles, LabelErrors, track_objects,
detection_objects, score_session, pick_errors, Report, TrackHistory, frame_tick,
read_track_histories, match_tap, load_compositions, TapErrors, own_tap_errors, fit ->
FitResult (.profile, .document(), .describe(), .measured, .notes, .warnings), build_parser,
main.

Imports numpy (through scripts/vision and tools/perception_noise), tools/latency_probe,
tools/vision_replay (its label reading and crosswalk) and tools/perception_noise.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:          # latency_probe, perception_noise, vision_replay
    sys.path.insert(0, str(ROOT / "tools"))
if str(ROOT) not in sys.path:                    # scripts.vision
    sys.path.append(str(ROOT))

import latency_probe as LP  # noqa: E402
import perception_noise as PN  # noqa: E402
import vision_replay as VR  # noqa: E402
from scripts.vision import events as EV  # noqa: E402
from scripts.vision.detector import box_iou  # noqa: E402
from scripts.vision.geometry import load_geometry  # noqa: E402
from scripts.vision.labels import (TYPE_UNIT, TYPE_UNKNOWN, load_class_table, parse_side,  # noqa: E402
                                   resolve_label, split_side)
from scripts.vision.record import VisionLogReader  # noqa: E402
from scripts.vision.types import (ENEMY_TEAM, FORM_BASE, FORM_EVO, FORM_HERO, OWN_TEAM,  # noqa: E402
                                  SCREEN_BATTLE)

FIT_SCHEMA = "hastycr.fit_noise_profile/1"
DEFAULT_OUT = ROOT / "data" / "noise" / "measured.json"
LEVEL_AUTO, LEVEL_TRACKS, LEVEL_DETECTIONS = "auto", "tracks", "detections"
LEVELS = (LEVEL_TRACKS, LEVEL_DETECTIONS)
ALLY, ENEMY = 0, 1                       # label sides (scripts/vision/labels: 0 bottom, 1 top)
MID_Y_MT = 16_000.0                      # the river: the bot (team 1, ALLY) owns y above it
KIND_TROOP, KIND_BUILDING = "troop", "building"
FRIENDLY_LEVEL = 11
SRC_TAPS = "own_taps"
#: Why a group keeps the base profile's values when no source measures it.
KEPT_WHY = {
    "board": "detection delay is inside D (tools/latency_probe.py measures it; E11 decision-time "
             "rows model it), so the board lag stays at the base value",
    "hp": "labels carry no hit points and own units are only seen full: nothing to compare a bar "
          "read with",
    "deploy": "the first sighting's lag after a unit appears is inside D; the logs cannot "
              "separate it",
    "towers": "the logs hold no true tower hit points (no memory reading)",
    "elixir": "the logs hold no true elixir (no memory reading)",
    "clock": "the logs hold no true tick (no memory reading)",
    "opponent_plays": "an opponent play's true card and tick are unknown offline",
    "size": "a setting (the drop buckets' hit-point bounds), not a noise",
}
FLYER_WHY = ("the bot's deck has no flyers, and labelled flyers pass through the detector's own "
             "flyer offset")
#: Why single values keep the base profile's value whatever the sources.
KEPT_FIELD_WHY = {
    ("position", "flyer_dx_mt"): FLYER_WHY,
    ("position", "flyer_dy_mt"): FLYER_WHY,
    ("position", "flyer_sigma_mt"): FLYER_WHY,
    ("forms", "unlabelled_to_base"): ("a property of the detector's label set (classes.json \"katacr\"), "
                                      "not measured; see the warnings"),
    ("confusion", "unknown_p"): ("an unplaceable name never reaches the state (public_state leaves it "
                                 "out: a drop here); token 1 comes from classes without an engine "
                                 "unit type, a property of the class table"),
}
HERO_WHY = ("no labelled hero of a form the detector has a label for (KataCR labels none: every "
            "hero counts as unlabelled)")
NO_LABELS = "no labelled frames scored (positional sessions with --labels)"
NO_TAPS = "no own-tap units matched (--taps: vision_play sessions run with --tap)"


# ---------------------------------------------------------------------------
# configuration and small helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FitConfig:
    """Matching settings and the sample minimums (module doc). Millitiles, ticks, seconds."""
    level: str = LEVEL_AUTO
    iou: float = 0.5                   # detections: a label and a box match at IoU >= this
    radius_mt: float = 1000.0          # tracks: ... within this world distance
    near_radius_mt: float = 2000.0     # a phantom this close to a labelled unit is a double box
    sparse_gap_s: float = 1.0          # tracked frames further apart (median): a sparse replay
    min_samples: int = 30              # per bucket, pooled rates, FP frames, label positions
    min_class_samples: int = 50        # per class (drop.by_class, confusion rows)
    min_tracks: int = 10               # own-tap tracks (position) and units (new_units)
    position_max_mt: float = 2500.0    # own-tap first report further from the aim: an outlier
    deploy_margin_ticks: int = 3       # position samples end this long before the deploy does
    bias_warn_mt: float = 200.0        # an own-tap position bias above this: refit the geometry
    level_hp: int = FRIENDLY_LEVEL     # hit points for the size buckets at this level
    probe: LP.ProbeConfig = field(default_factory=LP.ProbeConfig)

    def __post_init__(self):
        if self.level not in (LEVEL_AUTO,) + LEVELS:
            raise ValueError(f"FitConfig.level must be one of {(LEVEL_AUTO,) + LEVELS}, not {self.level!r}")
        if not 0.0 < self.iou <= 1.0:
            raise ValueError("FitConfig.iou must be in (0, 1]")
        for name in ("radius_mt", "near_radius_mt", "sparse_gap_s", "position_max_mt"):
            if not getattr(self, name) > 0:
                raise ValueError(f"FitConfig.{name} must be > 0")
        for name in ("min_samples", "min_class_samples", "min_tracks"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"FitConfig.{name} must be >= 1")
        if self.deploy_margin_ticks < 0:
            raise ValueError("FitConfig.deploy_margin_ticks must be >= 0")


def _p4(x: float) -> float:
    return round(float(x), 4)


def _floor4(x: float) -> float:
    """Down to 4 decimals (the epsilon keeps 0.29 at 0.29 despite binary floats)."""
    return math.floor(float(x) * 10_000 + 1e-9) / 10_000


def _mt(x: float) -> float:
    return round(float(x), 1)


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def axis_stats(values: Sequence[float]) -> dict[str, Any] | None:
    """n, mean (the bias), population standard deviation and root mean square, millitiles."""
    vals = [float(v) for v in values]
    if not vals:
        return None
    return {"n": len(vals), "bias_mt": _mt(statistics.fmean(vals)), "std_mt": _mt(statistics.pstdev(vals)),
            "rms_mt": _mt(_rms(vals))}


def _team_side(team: int) -> int | None:
    """A RoyaleSim team (the bot is 1) as a label side: ours ALLY, theirs ENEMY, else None."""
    return ALLY if team == OWN_TEAM else ENEMY if team == ENEMY_TEAM else None


def _flying(table: Any, cls: str, form: int) -> bool:
    c = table.get(cls)
    return bool(c is not None and c.form(form).flying)


# ---------------------------------------------------------------------------
# (a) labelled frames
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Obj:
    """A labelled or perceived unit on one frame: class, side (ALLY / ENEMY / None), form,
    world ground point, normalised box (x1, y1, x2, y2; None for a track), flying, confidence,
    and whether the side was decided (False: an undecided track placed by its board half)."""
    cls: str
    side: int | None
    form: int
    x_mt: float
    y_mt: float
    box: tuple[float, float, float, float] | None = None
    flying: bool = False
    conf: float = 1.0
    decided: bool = True


def match_objects(truth: Sequence[Obj], preds: Sequence[Obj], cost: Callable[[Obj, Obj], float | None]
                  ) -> tuple[list[tuple[int, int, bool]], list[int], list[int]]:
    """(matches as (truth index, prediction index, same class), unmatched truth, unmatched
    predictions). cost(t, p) -> a cost (lower is better) or None (no match). Greedy, same class
    first, then any class; ties by (cost, truth index, prediction index)."""
    used_t: set[int] = set()
    used_p: set[int] = set()
    pairs: list[tuple[int, int, bool]] = []
    for same in (True, False):
        cand = []
        for i, t in enumerate(truth):
            if i in used_t:
                continue
            for j, p in enumerate(preds):
                if j in used_p or (p.cls == t.cls) != same:
                    continue
                c = cost(t, p)
                if c is not None:
                    cand.append((float(c), i, j))
        cand.sort()
        for _, i, j in cand:
            if i in used_t or j in used_p:
                continue
            used_t.add(i)
            used_p.add(j)
            pairs.append((i, j, same))
    return (pairs, [i for i in range(len(truth)) if i not in used_t],
            [j for j in range(len(preds)) if j not in used_p])


class Canon:
    """Detector and label names -> (class, side, type, form), cached: vision_replay.Names (the
    detector's crosswalk) plus the form, which Names leaves out."""

    def __init__(self, table: Any):
        self.table = table
        self.names = VR.Names(table)
        self._forms: dict[str, int] = {}

    def __call__(self, name: str) -> tuple[str, int | None, str, int]:
        name = str(name)
        cls, side, typ = self.names.canon(name)
        form = self._forms.get(name)
        if form is None:
            form = self._forms[name] = int(resolve_label(self.table, split_side(name)[0]).form or FORM_BASE)
        return cls, side, typ, form


class LabelFiles:
    """The YOLO label files under a folder by image stem (vision_replay.LabelIndex finds them),
    read as unit Objs with their form and world ground point."""

    def __init__(self, folder: str | Path, names: Mapping[int, str], canon: Canon):
        self.canon = canon
        self.index = VR.LabelIndex(folder, names, canon.table, types=None, canon=canon.names)
        self.bad_lines = 0
        self.unknown: Counter = Counter()           # label names the crosswalk cannot place
        self.used_by: dict[str, set[str]] = {}      # stem -> sessions it was scored in

    def __len__(self) -> int:
        return len(self.index)

    def has(self, stem: str | None) -> bool:
        return stem is not None and stem in self.index.files

    def objects(self, stem: str, size: Sequence[int], geometry: Any, session: str | None = None
                ) -> list[Obj] | None:
        """The unit labels of image `stem` on a frame of `size` (pixels), or None without a file."""
        f = self.index.files.get(stem)
        if f is None:
            return None
        self.index.used.add(stem)
        if session is not None:
            self.used_by.setdefault(stem, set()).add(str(session))
        w, h = int(size[0]), int(size[1])
        out = []
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                k = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except (ValueError, IndexError):            # a short or non-numeric line
                self.bad_lines += 1
                continue
            name = self.index.names.get(k, f"class_{k}")
            cls, side, typ, form = self.canon(name)
            if typ == TYPE_UNKNOWN:
                self.unknown[name] += 1
                continue
            if typ != TYPE_UNIT:
                continue
            if side is None and len(parts) >= 6:
                side = parse_side(parts[5])
            box = (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)
            fly = _flying(self.canon.table, cls, form)
            gx, gy = geometry.ground_point((box[0] * w, box[1] * h, box[2] * w, box[3] * h), (w, h),
                                           cls=cls, flying=fly)
            out.append(Obj(cls, side, form, float(gx), float(gy), box, fly))
        return out

    def warnings(self) -> list[str]:
        out = []
        multi = sorted(s for s, used in self.used_by.items() if len(used) > 1)
        if multi:
            out.append(f"{len(multi)} label files were scored in more than one session (e.g. {multi[0]}): "
                       "label files are matched by image stem, so give one recording per labels folder")
        if self.index.duplicates:
            out.append(f"{len(self.index.duplicates)} label files share a stem with another and were not "
                       f"read (e.g. {self.index.duplicates[0]})")
        if self.unknown:
            out.append(f"label names the crosswalk cannot place were skipped: {dict(self.unknown)}")
        if self.bad_lines:
            out.append(f"{self.bad_lines} unreadable label lines skipped")
        return out


def track_objects(tracks: Iterable[Any], canon: Canon, skipped: Counter | None = None) -> list[Obj]:
    """A frame's unit tracks as Objs, as tools/public_state.py builds the state from them: a
    name the crosswalk cannot place is left out (counted in `skipped`), an undecided team takes
    the side of the board half the track stands in (decided False)."""
    out = []
    for tr in tracks:
        cls, _, typ, _ = canon(tr.cls)
        if typ == TYPE_UNKNOWN:
            if skipped is not None:
                skipped[str(tr.cls)] += 1
            continue
        if typ != TYPE_UNIT:
            continue
        side = _team_side(int(tr.team))
        decided = side is not None
        if side is None:
            side = ALLY if float(tr.y_mt) > MID_Y_MT else ENEMY
        form = int(tr.form)
        out.append(Obj(cls, side, form, float(tr.x_mt), float(tr.y_mt), None,
                       _flying(canon.table, cls, form), float(tr.conf), decided))
    return out


def detection_objects(dets: Iterable[Any], size: Sequence[int], geometry: Any, canon: Canon,
                      skipped: Counter | None = None) -> list[Obj]:
    """A frame's unit detections (pixels of a frame of `size`) as Objs: normalised boxes, the
    detector's ground point (its ground_px, else the box's), side from side_prob (None at 0.5).
    Names the crosswalk cannot place are left out (counted in `skipped`)."""
    w, h = float(size[0]), float(size[1])
    out = []
    for d in dets:
        cls, _, typ, _ = canon(d.cls)
        if typ == TYPE_UNKNOWN:
            if skipped is not None:
                skipped[str(d.cls)] += 1
            continue
        if typ != TYPE_UNIT:
            continue
        form = int(d.form)
        fly = _flying(canon.table, cls, form)
        if d.ground_px is not None:
            gx, gy = geometry.px_to_world(d.ground_px[0], d.ground_px[1], size)
        else:
            gx, gy = geometry.ground_point(d.box, size, cls=cls, flying=fly)
        x1, y1, x2, y2 = (float(v) for v in d.box)
        out.append(Obj(cls, VR.side_of(float(d.side_prob)), form, float(gx), float(gy),
                       (x1 / w, y1 / h, x2 / w, y2 / h), fly, float(d.conf)))
    return out


class LabelErrors:
    """Perception against labelled frames at one level, summed over frames (module doc, (a)).

    add(truth, preds) scores one frame. size: the base profile's PN.Size (the drop buckets)."""

    def __init__(self, level: str, table: Any, size: Any = None, config: FitConfig | None = None):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, not {level!r}")
        self.level = level
        self.table = table
        self.size = size if size is not None else PN.Size()
        self.config = config or FitConfig()
        self.frames = 0
        self.truth: Counter = Counter()
        self.dropped: Counter = Counter()
        self.truth_bucket: Counter = Counter()
        self.dropped_bucket: Counter = Counter()
        self.matched: Counter = Counter()
        self.confused: dict[str, Counter] = {}      # truth class -> {perceived class: n}
        self.forms: Counter = Counter()             # evo, evo_base, hero, hero_base, unlabelled, ...
        self.side: Counter = Counter()              # decided, flipped, by_half, undecided
        self.ground: list[tuple[float, float]] = []
        self.flyer: list[tuple[float, float]] = []
        self.fp = 0
        self.fp_near: list[tuple[float, float]] = []
        self.fp_far: Counter = Counter()
        self.skipped: Counter = Counter()           # perceived names left out (track_objects, ...)

    def cost(self, t: Obj, p: Obj) -> float | None:
        if self.level == LEVEL_TRACKS:
            d = math.hypot(p.x_mt - t.x_mt, p.y_mt - t.y_mt)
            return d if d <= self.config.radius_mt else None
        if t.box is None or p.box is None:
            return None
        iou = box_iou(t.box, p.box)
        return 1.0 - iou if iou >= self.config.iou else None

    def bucket(self, t: Obj) -> str:
        """perception_noise's Drop bucket of a labelled unit: its class's hit points at level_hp."""
        c = self.table.get(t.cls)
        if c is None:
            return "medium"
        if c.form(t.form).building:
            return "building"
        hp = self.table.hitpoints(c.name, t.form, self.config.level_hp)
        if hp is None:
            return "medium"
        if hp < self.size.small_max_hp:
            return "small"
        if hp >= self.size.large_min_hp:
            return "large"
        return "medium"

    def _form(self, t: Obj, p: Obj) -> None:
        key = "evo" if t.form == FORM_EVO else "hero" if t.form == FORM_HERO else None
        c = self.table.get(t.cls)
        if key is None or c is None:
            return
        if t.form not in c.katacr:
            key = "unlabelled"                      # the detector has no label for this form
        self.forms[key] += 1
        if p.form == FORM_BASE:
            self.forms[key + "_base"] += 1

    def add(self, truth: Sequence[Obj], preds: Sequence[Obj]) -> dict[str, int]:
        """Score one frame; returns its counts."""
        pairs, fn, fp = match_objects(truth, preds, self.cost)
        self.frames += 1
        for t in truth:
            self.truth[t.cls] += 1
            self.truth_bucket[self.bucket(t)] += 1
        for i in fn:
            self.dropped[truth[i].cls] += 1
            self.dropped_bucket[self.bucket(truth[i])] += 1
        for i, j, same in pairs:
            t, p = truth[i], preds[j]
            self.matched[t.cls] += 1
            if not same:
                self.confused.setdefault(t.cls, Counter())[p.cls] += 1
            if t.side is not None:
                if p.side is None:
                    self.side["undecided"] += 1
                else:
                    self.side["decided"] += 1
                    self.side["flipped"] += int(p.side != t.side)
                    self.side["by_half"] += int(not p.decided)
            if same:
                (self.flyer if t.flying else self.ground).append((p.x_mt - t.x_mt, p.y_mt - t.y_mt))
                self._form(t, p)
        for j in fp:
            p = preds[j]
            self.fp += 1
            near = None
            if truth:
                near = min(truth, key=lambda t: math.hypot(p.x_mt - t.x_mt, p.y_mt - t.y_mt))
            if near is not None and math.hypot(p.x_mt - near.x_mt, p.y_mt - near.y_mt) <= self.config.near_radius_mt:
                self.fp_near.append((p.x_mt - near.x_mt, p.y_mt - near.y_mt))
            else:
                self.fp_far[p.cls] += 1
        return {"truth": len(truth), "matched": len(pairs), "dropped": len(fn), "fp": len(fp)}

    def summary(self) -> dict[str, Any]:
        def pos(pts):
            return {"x": axis_stats([d[0] for d in pts]), "y": axis_stats([d[1] for d in pts])}
        return {
            "level": self.level, "frames": self.frames, "units": sum(self.truth.values()),
            "buckets": {b: {"n": self.truth_bucket[b], "dropped": self.dropped_bucket[b],
                            "rate": None if not self.truth_bucket[b] else _p4(self.dropped_bucket[b] / self.truth_bucket[b])}
                        for b in PN.BUCKETS},
            "classes": {c: {"n": self.truth[c], "dropped": self.dropped[c], "matched": self.matched[c],
                            "confused": dict(self.confused.get(c, {}))}
                        for c in sorted(self.truth)},
            "forms": dict(self.forms), "side": dict(self.side),
            "position": {"ground": pos(self.ground), "flyer": pos(self.flyer)},
            "false_positives": {"n": self.fp, "near": len(self.fp_near), "far": dict(self.fp_far)},
            "skipped": dict(self.skipped),
        }


def _frame_stem(entry: Any) -> str | None:
    """The image a frame record shows: the replayed image's name, else its own saved image."""
    rep = (entry.data.get("extra") or {}).get("replay")
    if isinstance(rep, Mapping) and rep.get("image"):
        return Path(str(rep["image"])).stem
    img = entry.data.get("image")
    return Path(str(img)).stem if img else None


def _ran(entry: Any, level: str) -> bool:
    """Whether perception produced tracks (detections) on this frame. A replay says so in its
    extra.replay; a live frame tracked exactly when the detector ran, which left detections
    (a battle frame always shows the crown towers)."""
    rep = (entry.data.get("extra") or {}).get("replay")
    if isinstance(rep, Mapping):
        return rep.get("tracks" if level == LEVEL_TRACKS else "det") is not None
    vf = entry.frame
    return vf is not None and vf.screen == SCREEN_BATTLE and bool(vf.detections)


def score_session(path: str | Path, files: LabelFiles, geometry: Any, canon: Canon,
                  errors: Mapping[str, LabelErrors], config: FitConfig | None = None) -> dict[str, Any]:
    """Score one session's labelled frames into errors[level] (module doc, (a)); returns what
    was done (level, sparse, frames labelled / scored / not scored)."""
    cfg = config or FitConfig()
    reader = VisionLogReader(path)
    name = reader.session_dir.name
    times, labelled, tracked_labelled = [], 0, 0
    for e in reader.frames():
        if _ran(e, LEVEL_TRACKS):
            times.append(float(e.frame.t_capture))
        if files.has(_frame_stem(e)):
            labelled += 1
            tracked_labelled += int(_ran(e, LEVEL_TRACKS))
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    gap = statistics.median(gaps) if gaps else None
    sparse = gap is None or gap > cfg.sparse_gap_s
    level = cfg.level
    if level == LEVEL_AUTO:
        level = LEVEL_TRACKS if tracked_labelled and not sparse else LEVEL_DETECTIONS
    info: dict[str, Any] = {"session": str(reader.session_dir), "level": level, "sparse": sparse,
                            "tracked_frames": len(times), "median_gap_s": None if gap is None else round(gap, 4),
                            "labelled": labelled, "scored": 0, "not_scored": 0, "errors": 0}
    E = errors[level]
    for e in reader.frames():
        stem = _frame_stem(e)
        if not files.has(stem):
            continue
        if not _ran(e, level):
            info["not_scored"] += 1
            continue
        vf = e.frame
        size = vf.frame_size or geometry.frame_size
        try:
            truth = files.objects(stem, size, geometry, session=name)
            if level == LEVEL_TRACKS:
                preds = track_objects(vf.tracks, canon, E.skipped)
            else:
                preds = detection_objects(vf.detections, size, geometry, canon, E.skipped)
        except ValueError:                          # a frame size the geometry refuses
            info["errors"] += 1
            continue
        E.add(truth, preds)
        info["scored"] += 1
    info["bad_lines"] = reader.bad_lines
    return info


def pick_errors(errors: Mapping[str, LabelErrors], config: FitConfig | None = None) -> LabelErrors | None:
    """The level to fit from: tracks when it scored frames (at least min_samples, or at least
    as many as detections), else detections; None when no frame was scored."""
    cfg = config or FitConfig()
    t, d = errors.get(LEVEL_TRACKS), errors.get(LEVEL_DETECTIONS)
    tf, df = (t.frames if t is not None else 0), (d.frames if d is not None else 0)
    if tf and (tf >= cfg.min_samples or tf >= df):
        return t
    return d if df else None


# ---------------------------------------------------------------------------
# (b) own taps
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Report:
    """One frame's report of a track: the tracker's tick, where, which team."""
    tick: int
    x_mt: float
    y_mt: float
    team: int


@dataclass
class TrackHistory:
    """One track of one battle over every frame it was reported in, oldest first."""
    uid: int
    first_tick: int
    reports: list[Report] = field(default_factory=list)
    labels: Counter = field(default_factory=Counter)
    forms: Counter = field(default_factory=Counter)
    claimed: bool = False

    @property
    def label(self) -> str:
        return self.labels.most_common(1)[0][0]

    @property
    def form(self) -> int:
        return int(self.forms.most_common(1)[0][0])

    @property
    def own(self) -> bool:
        """Ours in more of its reports than the opponent's (latency_probe.Sighting.own)."""
        own = sum(1 for r in self.reports if r.team == OWN_TEAM)
        enemy = sum(1 for r in self.reports if r.team == ENEMY_TEAM)
        return own > enemy

    @property
    def first_xy(self) -> tuple[float, float]:
        return self.reports[0].x_mt, self.reports[0].y_mt


def frame_tick(entry: Any) -> int | None:
    """The tick the tracker ran on for a frame record: a replay's extra.replay.track_tick, else
    the frame's tick, else (the clock unread: an extrapolated tick) the newest last_seen_tick of
    its tracks; None without any."""
    rep = (entry.data.get("extra") or {}).get("replay")
    if isinstance(rep, Mapping) and rep.get("track_tick") is not None:
        return int(rep["track_tick"])
    vf = entry.frame
    if vf is None:
        return None
    if vf.tick is not None:
        return int(vf.tick)
    seen = [int(tr.last_seen_tick) for tr in vf.tracks]
    return max(seen) if seen else None


def read_track_histories(path: str | Path) -> dict[int, dict[tuple[int, int], TrackHistory]]:
    """battle -> (uid, first_seen_tick) -> TrackHistory, over a session's in-battle frames (the
    frames tools/latency_probe.load_session reads), each report at frame_tick."""
    out: dict[int, dict[tuple[int, int], TrackHistory]] = {}
    for e in VisionLogReader(path).frames():
        battle = (e.data.get("extra") or {}).get("battle")
        if battle is None or e.frame is None or not e.frame.tracks:
            continue
        tick = frame_tick(e)
        if tick is None:
            continue
        hist = out.setdefault(int(battle), {})
        for tr in e.frame.tracks:
            key = (int(tr.uid), int(tr.first_seen_tick))
            h = hist.get(key)
            if h is None:
                h = hist[key] = TrackHistory(key[0], key[1])
            h.reports.append(Report(int(tick), float(tr.x_mt), float(tr.y_mt), int(tr.team)))
            h.labels[str(tr.cls)] += 1
            h.forms[int(tr.form)] += 1
    return out


def match_tap(tracks: Mapping[Any, TrackHistory], tap: Any, labels: Iterable[str], kind: str | None,
              config: LP.ProbeConfig | None = None) -> list[TrackHistory]:
    """The tracks of `tap`'s play, first sighting first (module doc, (b)); [] when none: the
    rule of latency_probe.match_sighting, which also claims the play's other units."""
    cfg = config or LP.ProbeConfig()
    labels = frozenset(labels)
    if not labels:
        return []
    aim = tap.aim_mt
    radius = None if kind == LP.KIND_SPELL else cfg.radius_mt
    best, best_key = None, None
    for h in tracks.values():
        if h.claimed or not h.reports or not h.own or h.label not in labels:
            continue
        if not cfg.min_lag_ticks <= h.first_tick - tap.tick <= cfg.max_lag_ticks:
            continue
        d = 0.0 if aim is None else math.hypot(h.first_xy[0] - aim[0], h.first_xy[1] - aim[1])
        if radius is not None and d > radius:
            continue
        key = (h.first_tick, d, h.uid)
        if best_key is None or key < best_key:
            best, best_key = h, key
    if best is None:
        return []
    best.claimed = True
    group = [best]
    for h in sorted(tracks.values(), key=lambda t: (t.first_tick, t.uid)):
        if (not h.claimed and h.reports and h.own and h.label == best.label
                and 0 <= h.first_tick - best.first_tick < cfg.play_window_ticks
                and math.hypot(h.first_xy[0] - best.first_xy[0], h.first_xy[1] - best.first_xy[1])
                <= cfg.merge_radius_mt):
            h.claimed = True
            group.append(h)
    return group


def load_compositions(path: str | Path | None = None) -> dict[str, int]:
    """Card name -> units one play deploys, from data/vision/spawn_children.json (or `path`)."""
    doc = EV.load_spawn_children(path)
    out = {}
    for name, comp in (doc.get("compositions") or {}).items():
        if isinstance(comp, Mapping):
            out[str(name)] = sum(int(n) for n in comp.values())
        elif isinstance(comp, (list, tuple)):
            out[str(name)] = len(comp)
    return out


class TapErrors:
    """Own plays against what perception showed of them (module doc, (b))."""

    def __init__(self):
        self.sessions: list[str] = []
        self.taps: Counter = Counter()                  # by card kind
        self.seen = 0
        self.unseen = 0
        self.delays: list[int] = []                     # first report - first sighting, per unit
        self.position: list[tuple[float, float]] = []   # perceived - aimed, per deploying report
        self.position_tracks = 0
        self.position_outliers = 0
        self.plays: list[dict[str, Any]] = []

    def add(self, tap: Any, tracks: Mapping[Any, TrackHistory], classes: LP.CardClasses, table: Any,
            compositions: Mapping[str, int], config: FitConfig | None = None, battle: int | None = None) -> None:
        cfg = config or FitConfig()
        kind = classes.kind(tap.name)
        self.taps[kind or "unknown"] += 1
        rec: dict[str, Any] = {"session": tap.session, "battle": battle, "tick": tap.tick, "name": tap.name,
                               "kind": kind or "unknown", "uids": [], "delays": [], "position_samples": 0}
        if kind not in (KIND_TROOP, KIND_BUILDING):     # spells and unknown cards: not units
            rec["skipped"] = True
            self.plays.append(rec)
            return
        group = match_tap(tracks, tap, classes.labels(tap.name), kind, cfg.probe)
        if not group:
            self.unseen += 1
            self.plays.append(rec)
            return
        self.seen += 1
        rec.update(label=group[0].label, lag_ticks=group[0].first_tick - tap.tick)
        for h in group:
            delay = h.reports[0].tick - h.first_tick
            self.delays.append(delay)
            rec["uids"].append(h.uid)
            rec["delays"].append(delay)
        aim = tap.aim_mt
        if kind == KIND_TROOP and compositions.get(str(tap.name)) == 1 and len(group) == 1 and aim is not None:
            h = group[0]
            if math.hypot(h.first_xy[0] - aim[0], h.first_xy[1] - aim[1]) > cfg.position_max_mt:
                self.position_outliers += 1
                rec["position_outlier"] = True
            else:
                c = table.get(h.label)
                deploy = c.form(h.form).deploy_ticks if c is not None else PN.DEFAULT_DEPLOY_TICKS
                end = h.first_tick + deploy - cfg.deploy_margin_ticks
                errs = [(r.x_mt - aim[0], r.y_mt - aim[1]) for r in h.reports if r.tick < end]
                if errs:
                    self.position.extend(errs)
                    self.position_tracks += 1
                    rec["position_samples"] = len(errs)
        self.plays.append(rec)

    def summary(self) -> dict[str, Any]:
        return {"sessions": list(self.sessions), "taps": dict(self.taps), "seen": self.seen, "unseen": self.unseen,
                "delays": LP.describe(self.delays),
                "position": {"tracks": self.position_tracks, "samples": len(self.position),
                             "outliers": self.position_outliers,
                             "x": axis_stats([d[0] for d in self.position]),
                             "y": axis_stats([d[1] for d in self.position])},
                "plays": list(self.plays)}


def own_tap_errors(paths: Iterable[str | Path], classes: LP.CardClasses, table: Any,
                   compositions: Mapping[str, int], config: FitConfig | None = None) -> TapErrors:
    """Every tap of these sessions (tools/latency_probe.load_session: battles and taps) matched
    to the tracks of its battle, oldest tap first; a tap of no battle counts as unseen."""
    cfg = config or FitConfig()
    out = TapErrors()
    for path in paths:
        log = LP.load_session(path)
        hist = read_track_histories(path)
        out.sessions.append(str(log.path))
        for tap in log.unassigned:
            out.add(tap, {}, classes, table, compositions, cfg)
        for n in sorted(log.battles):
            tracks = hist.get(n, {})
            for tap in sorted(log.battles[n].taps, key=lambda tp: (tp.tick, tp.t if tp.t is not None else 0.0)):
                out.add(tap, tracks, classes, table, compositions, cfg, battle=n)
    return out


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------
def _groups(doc: Mapping[str, Any]) -> list[str]:
    return [k for k in doc if k not in ("schema", "name", "description") and not str(k).startswith("_")]


@dataclass
class FitResult:
    """The fitted profile, the base it started from, and where each value came from."""
    profile: PN.NoiseProfile
    base: PN.NoiseProfile
    notes: dict[str, dict[str, Any]]
    measured: list[str]
    warnings: list[str]
    meta: dict[str, Any]

    def document(self) -> dict[str, Any]:
        """The JSON written to --out: the profile, a "_fit" note per group and at the top."""
        doc = self.profile.to_dict()
        for g, note in self.notes.items():
            if g in doc:
                doc[g]["_fit"] = note
        doc["_fit"] = {**self.meta, "measured": list(self.measured), "warnings": list(self.warnings)}
        return doc

    def describe(self) -> list[str]:
        p, b = self.profile, self.base
        lines = [f"noise profile {p.name} [{p.digest()}]: {len(self.measured)} value(s) measured, "
                 f"the rest from {b.name} [{b.digest()}]"]
        old, new = b.to_dict(), p.to_dict()
        for key in self.measured:
            g, f = key.split(".", 1)
            note = self.notes[g]["measured"][f]
            was, now = old[g][f], new[g][f]
            if isinstance(now, dict):
                was, now = f"{len(was)} entries", f"{len(now)} entries"
            lines.append(f"  {key:30s} {was} -> {now}  ({note['from']}, n {note['n']})")
        kept = [g for g in _groups(new) if not self.notes[g]["measured"]]
        if kept:
            lines.append(f"  kept from {b.name}: {', '.join(kept)}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return lines


def fit(base: Any, labels: LabelErrors | None = None, taps: TapErrors | None = None,
        config: FitConfig | None = None, *, name: str = "measured", description: str | None = None) -> FitResult:
    """A profile from `base` (a NoiseProfile, its JSON, a path or "default") with every value the
    label errors and own-tap errors measure replaced (module doc)."""
    cfg = config or FitConfig()
    base = PN.as_profile(base)
    doc = base.to_dict()
    groups = _groups(doc)
    notes: dict[str, dict[str, Any]] = {g: {"measured": {}} for g in groups}
    measured: list[str] = []
    warnings: list[str] = []

    def put(group: str, fld: str, value: Any, src: str, n: int, **extra: Any) -> None:
        doc[group][fld] = value
        notes[group]["measured"][fld] = {"from": src, "n": int(n), **extra}
        measured.append(f"{group}.{fld}")

    E = labels if labels is not None and labels.frames else None
    T = taps if taps is not None and (taps.delays or taps.position_tracks) else None
    if E is not None:
        src = f"labels/{E.level}"
        if E.level == LEVEL_DETECTIONS:
            warnings.append("labelled frames were scored on raw detections (a sparse replay, or no tracks): "
                            "per-frame detector errors overstate the drops, phantoms and side flips the tracker "
                            "removes; record densely and replay for track-level figures")
        # drop
        for b in PN.BUCKETS:
            n = E.truth_bucket[b]
            if n >= cfg.min_samples:
                put("drop", b, _p4(E.dropped_bucket[b] / n), src, n, dropped=E.dropped_bucket[b])
        rates = {c: _p4(E.dropped[c] / E.truth[c]) for c in sorted(E.truth) if E.truth[c] >= cfg.min_class_samples}
        if rates:
            put("drop", "by_class", {**doc["drop"]["by_class"], **rates}, src, sum(E.truth[c] for c in rates),
                classes={c: E.truth[c] for c in rates})
        # forms
        for key, fld in (("evo", "evo_to_base_p"), ("hero", "hero_to_base_p")):
            n = E.forms[key]
            if n >= cfg.min_samples:
                put("forms", fld, _p4(E.forms[key + "_base"] / n), src, n)
        n = E.forms["unlabelled"]
        if n >= cfg.min_samples and doc["forms"]["unlabelled_to_base"] and E.forms["unlabelled_base"] * 2 < n:
            warnings.append(f"{n - E.forms['unlabelled_base']} of {n} labelled forms the KataCR table lacks were "
                            "read as their own form: this detector knows them, so forms.unlabelled_to_base "
                            "may be false")
        # confusion pairs (unknown_p is kept: KEPT_FIELD_WHY)
        rows = {}
        for c in sorted(E.matched):
            if E.matched[c] >= cfg.min_class_samples:
                row = {t: _floor4(k / E.matched[c]) for t, k in sorted(E.confused.get(c, {}).items())}
                rows[c] = {t: p for t, p in row.items() if p > 0}
        if rows:
            pairs = {c: dict(r) for c, r in doc["confusion"]["pairs"].items()}
            for c, row in rows.items():
                if row:
                    pairs[c] = row
                else:
                    pairs.pop(c, None)
            put("confusion", "pairs", pairs, src, sum(E.matched[c] for c in rows),
                classes={c: E.matched[c] for c in rows})
        # side flips
        n = E.side["decided"]
        if n >= cfg.min_samples:
            put("side_flip", "p", _p4(E.side["flipped"] / n), src, n, flipped=E.side["flipped"],
                by_half=E.side["by_half"], undecided=E.side["undecided"])
        # false positives
        if E.frames >= cfg.min_samples:
            put("false_positives", "rate", _p4(E.fp / E.frames), src, E.frames, phantoms=E.fp)
            near, far = len(E.fp_near), sum(E.fp_far.values())
            if E.fp >= cfg.min_samples:
                if far > near:                      # phantoms anywhere: their classes as weights
                    put("false_positives", "classes", {c: float(k) for c, k in sorted(E.fp_far.items())},
                        src, far, near=near)
                else:                               # double boxes: copies of a unit nearby
                    put("false_positives", "classes", {}, src, near, far=far)
                    if near >= cfg.min_samples:
                        sq = sum(dx * dx + dy * dy for dx, dy in E.fp_near)
                        put("false_positives", "near_sigma_mt", _mt(math.sqrt(sq / (2 * near))), src, near)
    # position
    if T is not None and T.position_tracks >= cfg.min_tracks:
        xs, ys = [d[0] for d in T.position], [d[1] for d in T.position]
        sx, sy = axis_stats(xs), axis_stats(ys)
        put("position", "sigma_x_mt", sx["rms_mt"], SRC_TAPS, T.position_tracks, samples=len(xs),
            bias_mt=sx["bias_mt"], std_mt=sx["std_mt"])
        put("position", "sigma_y_mt", sy["rms_mt"], SRC_TAPS, T.position_tracks, samples=len(ys),
            bias_mt=sy["bias_mt"], std_mt=sy["std_mt"])
        bias = math.hypot(statistics.fmean(xs), statistics.fmean(ys))
        if bias > cfg.bias_warn_mt:
            warnings.append(f"own units read {bias:.0f} mt off their aimed tiles on average (dx "
                            f"{sx['bias_mt']:+g}, dy {sy['bias_mt']:+g}): refit the geometry (the calibration "
                            "friendly); the next fit then gives a smaller sigma")
    elif E is not None and len(E.ground) >= cfg.min_samples:
        sx = axis_stats([d[0] for d in E.ground])
        sy = axis_stats([d[1] for d in E.ground])
        lower = "a lower bound: the label's ground point goes through the same geometry"
        put("position", "sigma_x_mt", sx["rms_mt"], f"labels/{E.level}", len(E.ground), bias_mt=sx["bias_mt"],
            std_mt=sx["std_mt"], note=lower)
        put("position", "sigma_y_mt", sy["rms_mt"], f"labels/{E.level}", len(E.ground), bias_mt=sy["bias_mt"],
            std_mt=sy["std_mt"], note=lower)
    # new units
    if T is not None and len(T.delays) >= cfg.min_tracks:
        st = LP.describe(T.delays)
        off = int(doc["deploy"]["age_offset_ticks"])
        delay, extra = off + LP.half_up(st["median"]), {}
        if delay < 0:                       # a profile's ticks are >= 0 (NoiseProfile refuses less)
            warnings.append(f"own units were first reported {-st['median']:g} ticks (median) before the "
                            "tracker first saw them: the frames' ticks and the tracker's disagree (check the "
                            "clock reads); new_units.delay_ticks set to 0")
            delay, extra = 0, {"clamped_from": delay}
        put("new_units", "delay_ticks", delay, SRC_TAPS, st["n"],
            confirm_median=st["median"], age_offset_ticks=off, **extra)
        put("new_units", "jitter_ticks", LP.half_up((st["p90"] - st["p10"]) / 2.0), SRC_TAPS, st["n"],
            p10=st["p10"], p90=st["p90"])
    # the notes for what was kept
    have = {"labels": E is not None, "taps": T is not None}

    def why_kept(g: str, f: str) -> str:
        if (g, f) in KEPT_FIELD_WHY:
            return KEPT_FIELD_WHY[(g, f)]
        if g in KEPT_WHY:
            return KEPT_WHY[g]
        need = ("taps",) if g == "new_units" else ("taps", "labels") if g == "position" else ("labels",)
        missing = [NO_TAPS if s == "taps" else NO_LABELS for s in need if not have[s]]
        if len(missing) == len(need):
            return "; ".join(missing)
        if g == "forms" and f == "hero_to_base_p" and E is not None and not E.forms["hero"]:
            return HERO_WHY
        return f"fewer samples than the minimum ({_minimum(g, cfg)})"

    for g in groups:
        kept = [f for f in doc[g] if f not in notes[g]["measured"]]
        if kept:
            notes[g].update({"kept": kept, "from": base.name,
                             "why": "; ".join(dict.fromkeys(why_kept(g, f) for f in kept))})
    doc["name"] = str(name)
    srcs = []
    if E is not None:
        srcs.append(f"{E.frames} labelled frames ({E.level})")
    if T is not None:
        srcs.append(f"{len(T.delays)} own-tap units")
    doc["description"] = description if description is not None else (
        "Fitted by tools/fit_noise_profile.py (V11) from " + (" and ".join(srcs) or "nothing")
        + f"; every value not measured is {base.name}'s.")
    profile = PN.NoiseProfile.from_dict(doc)          # every value checked as a profile file is
    meta = {"schema": FIT_SCHEMA, "tool": "tools/fit_noise_profile.py", "base": base.name,
            "base_digest": base.digest(), "config": asdict(cfg)}
    return FitResult(profile, base, notes, measured, warnings, meta)


def _minimum(group: str, cfg: FitConfig) -> str:
    if group == "new_units":
        return f"--min-tracks {cfg.min_tracks}"
    if group == "position":
        return f"--min-tracks {cfg.min_tracks} own-tap tracks, --min-samples {cfg.min_samples} label matches"
    if group in ("drop", "confusion"):
        return f"--min-samples {cfg.min_samples} / --min-class-samples {cfg.min_class_samples}"
    return f"--min-samples {cfg.min_samples}"


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tools/fit_noise_profile.py",
                                 description="Fit the perception-noise profile (E17) from labelled frames "
                                             "and the bot's own taps (V11).")
    ap.add_argument("sessions", nargs="*",
                    help="sessions to score against --labels: tools/vision_replay.py outputs, or recorded "
                         "sessions whose frames saved images (a session dir, its frames.jsonl, or a root)")
    ap.add_argument("--labels", default=None, help="a folder of YOLO label files named like the images")
    ap.add_argument("--label-names", default=None,
                    help="class names of the labels (detector_labels.json, a dataset yaml, or one per line); "
                         "default: a names file in the labels folder")
    ap.add_argument("--taps", action="append", default=[], metavar="PATH",
                    help="vision_play sessions run with --tap, or roots of them (repeatable)")
    ap.add_argument("--last", type=int, default=None, help="only the newest N sessions of each --taps root")
    ap.add_argument("--base", default="default",
                    help="the profile unmeasured values come from: a JSON, 'default' (the prior) or 'none'")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"the fitted profile (default {DEFAULT_OUT})")
    ap.add_argument("--report", default=None, help="also write the full fit report (JSON) here")
    ap.add_argument("--name", default="measured", help="the fitted profile's name")
    ap.add_argument("--level", choices=(LEVEL_AUTO,) + LEVELS, default=LEVEL_AUTO,
                    help="score labels against tracks or raw detections (auto: tracks unless sparse)")
    ap.add_argument("--iou", type=float, default=FitConfig.iou, help="detections: IoU a box needs to match")
    ap.add_argument("--radius-mt", type=float, default=FitConfig.radius_mt,
                    help="tracks: world distance a track needs to match a label")
    ap.add_argument("--min-samples", type=int, default=FitConfig.min_samples)
    ap.add_argument("--min-class-samples", type=int, default=FitConfig.min_class_samples)
    ap.add_argument("--min-tracks", type=int, default=FitConfig.min_tracks)
    ap.add_argument("--geometry", default=None, help="the ArenaGeometry JSON the replay used (default data/vision's)")
    ap.add_argument("--classes", default=None, help="classes.json (default data/vision/classes.json)")
    ap.add_argument("--spawn-children", default=None, help="spawn_children.json (default data/vision's)")
    ap.add_argument("--dry-run", action="store_true", help="print the fit, write nothing")
    return ap


def _write_json(path: str | Path, doc: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.sessions and not args.taps:
        print("nothing to fit: give labelled sessions with --labels FOLDER, and/or --taps sessions")
        return 2
    if bool(args.sessions) != bool(args.labels):
        print("labelled sessions and --labels FOLDER go together")
        return 2
    try:
        cfg = FitConfig(level=args.level, iou=args.iou, radius_mt=args.radius_mt, min_samples=args.min_samples,
                        min_class_samples=args.min_class_samples, min_tracks=args.min_tracks)
        base = PN.NoiseProfile.load(args.base)
    except (ValueError, FileNotFoundError) as exc:
        print(f"fit_noise_profile: {exc}")
        return 2
    table = load_class_table(args.classes)
    labels = taps = None
    sessions: list[dict[str, Any]] = []
    warnings: list[str] = []
    if args.sessions:
        names_path = Path(args.label_names) if args.label_names else VR.find_label_names(args.labels)
        if names_path is None:
            print("--labels: no class names (--label-names, or classes.txt / detector_labels.json / cr.yaml "
                  "/ data.yaml in the folder)")
            return 2
        paths = LP.find_sessions(args.sessions)
        if not paths:
            print(f"no sessions under {', '.join(args.sessions)}")
            return 2
        canon = Canon(table)
        files = LabelFiles(args.labels, VR.read_label_names(names_path), canon)
        geometry = load_geometry(args.geometry)
        errors = {lv: LabelErrors(lv, table, base.size, cfg) for lv in LEVELS}
        print(f"labels     {len(files)} label files under {args.labels}; geometry {geometry.name}")
        for p in paths:
            info = score_session(p, files, geometry, canon, errors, cfg)
            sessions.append(info)
            print(f"  {Path(info['session']).name}: {info['scored']} of {info['labelled']} labelled frames scored "
                  f"on {info['level']}" + (" (sparse)" if info["sparse"] else ""))
        labels = pick_errors(errors, cfg)
        other = [lv for lv in LEVELS if errors[lv].frames and errors[lv] is not labels]
        if other and labels is not None:
            warnings.append(f"{errors[other[0]].frames} frames scored on {other[0]} were left out (fit on "
                            f"{labels.level})")
        warnings += files.warnings()
    if args.taps:
        paths = LP.find_sessions(args.taps, args.last)
        if not paths:
            print(f"no tap sessions under {', '.join(args.taps)}")
            return 2
        try:
            comps = load_compositions(args.spawn_children)
        except (OSError, ValueError) as exc:
            comps = {}
            warnings.append(f"no card compositions ({exc}): own-tap positions not measured")
        taps = own_tap_errors(paths, LP.CardClasses(table=table), table, comps, cfg)
        print(f"taps       {len(paths)} session(s): {sum(taps.taps.values())} taps, {taps.seen} plays seen, "
              f"{len(taps.delays)} units, {taps.position_tracks} position tracks")
    result = fit(base, labels, taps, cfg, name=args.name)
    result.warnings[:0] = warnings
    result.meta.update(created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       inputs={"labels": args.labels, "sessions": [s["session"] for s in sessions],
                               "taps": [] if taps is None else list(taps.sessions)})
    for line in result.describe():
        print(line)
    if not result.measured:
        print(f"nothing measured: the profile would be {base.name}'s; nothing written")
        return 1
    if args.dry_run:
        return 0
    doc = result.document()
    out = _write_json(args.out, doc)
    print(f"profile -> {out}  (check: tools/perception_noise.py {out})")
    if args.report:
        rep = {"fit": doc["_fit"], "profile": doc, "sessions": sessions,
               "labels": None if labels is None else labels.summary(),
               "taps": None if taps is None else taps.summary()}
        print(f"report  -> {_write_json(args.report, rep)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
