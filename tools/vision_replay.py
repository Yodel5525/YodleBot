r"""Re-run perception offline on a recording: HUD, detections, tracks, and precision / recall (V11).

    .venvs\buildabot\Scripts\python.exe tools\vision_replay.py reports\vision_play\vision_play_20261030_201500
    .venvs\buildabot\Scripts\python.exe tools\vision_replay.py reports\vision\recordings\capture_20261029_190000 --labels D:\labels\friendly1
    ... --no-detector    the screen cues and the HUD only
    ... --recorded       score the detections the recording holds (a vision_play session) instead of
                         running the detector again

INPUT. A session written by scripts/vision/record.py's VisionRecorder (python -m
scripts.vision.capture --record ..., or scripts/vision_play.py): its frames WITH an image on
disk are replayed in order at their recorded times (scripts/vision/capture.FileReplaySource,
sequential); or a plain folder of images (t = index / --fps). For tracker and event figures
record densely (capture --image-every 1, vision_play --record-images 1): with an image only
every N frames the tracks expire between images, and the summary says "sparse".

WHAT RUNS. Per image, the live chain of scripts/vision_play.py (LivePerception), on the pixels:
screen cues -> HudReader -> Detector -> HP bars -> Tracker -> EventDeriver, battles started
and ended by vision_play's BattleState (tracker and deriver reset at a start, the HUD too at
an end). Every "tap" the recording logged reaches the deriver and the tracker before the
first frame captured after the frame it was decided on, as live, so own-team votes and
landings work offline. Ticks come from the HUD clock as live (no reading: extrapolated for
5 s, then not tracked), or with --ticks time (the default when no clock templates were
captured) from the capture time, 20 ticks a second from the first image. Card ids are the
engine's (data/rs_rows/spawn_primary.json "card_names"), else replay-only ids from
classes.json. Nothing here taps, captures or touches the emulator.

OUTPUT. A session under reports/vision/replays/<input>_replay_<time>/ (or --out) in the
recorder's format, so scripts/vision/record.py and tools/latency_probe.py read it: one
"frame" record per replayed image, its VisionFrame (HUD, tracks, events, detections) and
extra.replay: the screen cues, the HUD report (clock text and confidence, elixir digit and
bar, tower fractions, hand), detection counts by type, tracker counts, the frame's TP / FP /
FN and, for a vision_play recording, agreement with what was read live; the recording's taps
(card ids and ticks of the replay); a "replay_summary" record; and summary.json. The summary
printed: HUD read rates (clock, elixir, hand, next, towers) and agreement with the live
reads (sparse images read differently: the HUD's filters use the frames between); detections
per frame by type, the most frequent classes, off-board drops, suppressions and names the
class list cannot place; tracks per frame, lifetimes, teams, label switches, short tracks,
ticks to a team; opponent plays and own landings.

PRECISION / RECALL (--labels FOLDER). YOLO label files named like the images (00000012.txt
for images/00000012.jpg; found anywhere under FOLDER; lines "cls cx cy w h [side]",
normalised to the image). Class names: --label-names (a detector_labels.json, a cr.yaml /
data.yaml with "names", or one name per line), else detector_labels.json, classes.txt,
names.txt, cr.yaml or data.yaml in FOLDER, else the detector's own names. Every name goes
through the detector's crosswalk (scripts/vision/labels.resolve_label): "hog-rider",
"HogRider@enemy" -> class HogRider, side enemy; a KataCR side column (0 ally, 1 enemy) is read
when the name has no side. On each frame with a label file, predictions and labels of types
unit / tower / spell (--pr-all: every type) are matched per class, greedily by confidence, at
IoU >= --iou (0.5): per class and overall TP, FP, FN, precision, recall, and the side
accuracy of matched pairs (P(enemy) > 0.5 against the label's side). Forms are not scored
(Skeleton_EV1 counts as Skeleton). An empty label file is a frame with nothing on it; frames
without one are not scored.

Not here: fitting the perception noise profile (V11's fit_noise_profile, from these errors
and own-tap ground truth) needs E17's profile format, which is not built yet.

Imports numpy and scripts/vision; the detector needs ultralytics and torch (the GPU venv).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:          # latency_probe, vision_classes
    sys.path.insert(0, str(ROOT / "tools"))
if str(ROOT) not in sys.path:                    # scripts.vision, scripts.vision_play
    sys.path.append(str(ROOT))

import numpy as np  # noqa: E402

from latency_probe import describe  # noqa: E402
from scripts import vision_play as VP  # noqa: E402
from scripts.vision import hpbars  # noqa: E402
from scripts.vision.detector import BAR_CLASSES, Detector, box_iou  # noqa: E402
from scripts.vision.geometry import check_frame_size  # noqa: E402
from scripts.vision.labels import (TYPE_SPELL, TYPE_TOWER, TYPE_UNIT, parse_side,  # noqa: E402
                                   resolve_label, split_side)
from scripts.vision.record import FRAMES_FILE, VisionLogReader  # noqa: E402
from scripts.vision.tracker import observations_from_detections  # noqa: E402
from scripts.vision.types import (EMPTY_CARD, ENEMY_TEAM, HAND_SIZE, OWN_TEAM, PHASE_ENDED,  # noqa: E402
                                  SCREEN_BATTLE, SCREEN_END, SCREEN_MENU, Detection, Hud, VisionFrame)

SCHEMA = "hastycr.vision_replay/1"
OUT_ROOT = ROOT / "reports" / "vision" / "replays"
SPAWN_TABLE = ROOT / "data" / "rs_rows" / "spawn_primary.json"
TICKS_PER_S = 20
TICKS_CLOCK, TICKS_TIME = "clock", "time"
#: Detection types precision / recall scores by default (not HP bars, UI or effects).
PR_TYPES = (TYPE_UNIT, TYPE_TOWER, TYPE_SPELL)
#: Files in a labels folder that name the classes (and are not label files).
NAME_FILES = ("detector_labels.json", "classes.txt", "names.txt", "cr.yaml", "data.yaml")
SHORT_TRACK_TICKS = 10           # a track living shorter than this: a flicker or a fragment
CLOCK_TOL_TICKS = 2              # live and replayed clock reads agree within this
ELIXIR_TOL_MILLI = 300           # ... elixir within this


# ---------------------------------------------------------------------------
# class names and label files
# ---------------------------------------------------------------------------
def _unquote(text: str) -> str:
    s = text.strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        try:
            return str(json.loads(s))
        except ValueError:
            return s[1:-1]
    if len(s) >= 2 and s[0] == s[-1] == "'":
        return s[1:-1].replace("''", "'")
    return s


def parse_yaml_names(text: str) -> dict[int, str]:
    """The `names:` of an ultralytics dataset yaml, without a yaml dependency: an inline
    [a, b] list, a block of `- name` lines or a block of `index: name` lines (what
    scripts/prepare_yolo_dataset.py writes). {} when there is none."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^names\s*:\s*(.*?)\s*$", line)
        if not m:
            continue
        rest = m.group(1)
        if rest.startswith("["):
            try:
                vals = json.loads(rest)
            except ValueError:
                vals = [_unquote(x) for x in rest.strip("[]").split(",") if x.strip()]
            return {k: str(v) for k, v in enumerate(vals)}
        out: dict[int, str] = {}
        for item in lines[i + 1:]:
            if not item.strip() or item.lstrip().startswith("#"):
                continue
            if not item[:1].isspace():
                break
            s = item.strip()
            if s.startswith("- "):
                out[len(out)] = _unquote(s[2:])
                continue
            m2 = re.match(r"^(\d+)\s*:\s*(.+?)\s*$", s)
            if m2:
                out[int(m2.group(1))] = _unquote(m2.group(2))
        return out
    return {}


def read_label_names(path: str | Path) -> dict[int, str]:
    """Class index -> name from a detector_labels.json ("yolo_names") or any JSON list /
    {"names": ...}, a dataset yaml, or a text file with one name per line."""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    suffix = p.suffix.lower()
    if suffix == ".json":
        doc = json.loads(text)
        names = doc if isinstance(doc, list) else (doc.get("yolo_names") or doc.get("names") or [])
        if isinstance(names, Mapping):
            return {int(k): str(v) for k, v in names.items()}
        return {k: str(v) for k, v in enumerate(names)}
    if suffix in (".yaml", ".yml"):
        return parse_yaml_names(text)
    rows = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return dict(enumerate(rows))


def find_label_names(folder: str | Path) -> Path | None:
    """The first of NAME_FILES in `folder`, or None."""
    for name in NAME_FILES:
        p = Path(folder) / name
        if p.is_file():
            return p
    return None


class Names:
    """Detector names -> (class, side, type) through the class table, cached: the class the
    detector reports (units and towers their class, spells their card, others the name)."""

    def __init__(self, table: Any):
        self.table = table
        self._cache: dict[str, tuple[str, int | None, str]] = {}

    def canon(self, name: str) -> tuple[str, int | None, str]:
        name = str(name)
        hit = self._cache.get(name)
        if hit is None:
            base, side = split_side(name)
            lab = resolve_label(self.table, base)
            hit = self._cache[name] = (Detector._cls_name(base, lab), side, str(lab.type))
        return hit


@dataclass(frozen=True)
class Box:
    """A labelled or predicted object, box normalised to the image (x1, y1, x2, y2)."""
    cls: str
    box: tuple[float, float, float, float]
    side: int | None = None          # 0 ally, 1 enemy, None unknown
    conf: float = 1.0
    type: str = TYPE_UNIT


def side_of(p_enemy: float) -> int | None:
    """A detection's side: enemy above P(enemy) 0.5, ally below, unknown at 0.5."""
    return 1 if p_enemy > 0.5 else 0 if p_enemy < 0.5 else None


def pred_boxes(dets: Iterable[Detection], size: Sequence[int], names: Names,
               types: Iterable[str] | None = PR_TYPES) -> list[Box]:
    """Detections (frame pixels of an image of `size`) as normalised Boxes of `types`."""
    w, h = float(size[0]), float(size[1])
    keep = None if types is None else frozenset(types)
    out = []
    for d in dets:
        cls, _, typ = names.canon(d.cls)
        if keep is not None and typ not in keep:
            continue
        x1, y1, x2, y2 = (float(v) for v in d.box)
        out.append(Box(cls, (x1 / w, y1 / h, x2 / w, y2 / h), side_of(float(d.side_prob)),
                       float(d.conf), typ))
    return out


class LabelIndex:
    """The YOLO label files under a folder (any depth), by image stem.

    names: class index -> name (read_label_names). truth_for(image) -> the Boxes of the
    image's label file, of `types` (None: all), or None when it has no label file. Lines that
    do not parse are counted in bad_lines; a stem found twice keeps the first path (sorted)
    and lists the others in duplicates."""

    def __init__(self, folder: str | Path, names: Mapping[int, str], table: Any, *,
                 types: Iterable[str] | None = PR_TYPES, canon: Names | None = None):
        self.folder = Path(folder)
        self.names = {int(k): str(v) for k, v in names.items()}
        self.canon = canon if canon is not None else Names(table)
        self.types = None if types is None else frozenset(types)
        skip = {n.lower() for n in NAME_FILES}
        self.files: dict[str, Path] = {}
        self.duplicates: list[str] = []
        for f in sorted(self.folder.rglob("*.txt")):
            if f.name.lower() in skip:
                continue
            if f.stem in self.files:
                self.duplicates.append(str(f))
                continue
            self.files[f.stem] = f
        self.used: set[str] = set()
        self.bad_lines = 0
        self.unknown: Counter = Counter()       # label names the crosswalk cannot place

    def __len__(self) -> int:
        return len(self.files)

    def read(self, path: str | Path) -> list[Box]:
        out = []
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                k = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except (ValueError, IndexError):          # a short or non-numeric line
                self.bad_lines += 1
                continue
            name = self.names.get(k, f"class_{k}")
            cls, side, typ = self.canon.canon(name)
            if typ == "unknown":
                self.unknown[name] += 1
            if side is None and len(parts) >= 6:
                side = parse_side(parts[5])
            if self.types is not None and typ not in self.types:
                continue
            out.append(Box(cls, (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), side, 1.0, typ))
        return out

    def truth_for(self, image: str | Path | None) -> list[Box] | None:
        if image is None:
            return None
        f = self.files.get(Path(image).stem)
        if f is None:
            return None
        self.used.add(f.stem)
        return self.read(f)


def match_boxes(truth: Sequence[Box], preds: Sequence[Box], iou_min: float = 0.5
                ) -> tuple[list[tuple[Box, Box]], list[Box], list[Box]]:
    """(true positives as (label, prediction), false positives, false negatives): each
    prediction, strongest first, takes the unmatched label of its class it overlaps most,
    at IoU >= iou_min (ties: the first label)."""
    used = [False] * len(truth)
    tp: list[tuple[Box, Box]] = []
    fp: list[Box] = []
    for i in sorted(range(len(preds)), key=lambda k: (-preds[k].conf, k)):
        p = preds[i]
        best, best_iou = None, 0.0
        for j, g in enumerate(truth):
            if used[j] or g.cls != p.cls:
                continue
            iou = box_iou(p.box, g.box)
            if iou >= iou_min and (best is None or iou > best_iou):
                best, best_iou = j, iou
        if best is None:
            fp.append(p)
        else:
            used[best] = True
            tp.append((truth[best], p))
    fn = [g for j, g in enumerate(truth) if not used[j]]
    return tp, fp, fn


def _ratio(a: int, b: int) -> float | None:
    return None if b == 0 else a / b


class PRCounter:
    """TP / FP / FN per class over labelled frames, and side accuracy of matched pairs."""

    def __init__(self, iou_min: float = 0.5):
        self.iou_min = float(iou_min)
        self.frames = 0
        self.tp: Counter = Counter()
        self.fp: Counter = Counter()
        self.fn: Counter = Counter()
        self.side_n: Counter = Counter()
        self.side_ok: Counter = Counter()

    def add(self, truth: Sequence[Box], preds: Sequence[Box]) -> dict[str, int]:
        tp, fp, fn = match_boxes(truth, preds, self.iou_min)
        self.frames += 1
        for g, p in tp:
            self.tp[g.cls] += 1
            if g.side is not None and p.side is not None:
                self.side_n[g.cls] += 1
                self.side_ok[g.cls] += int(g.side == p.side)
        for p in fp:
            self.fp[p.cls] += 1
        for g in fn:
            self.fn[g.cls] += 1
        return {"tp": len(tp), "fp": len(fp), "fn": len(fn)}

    @staticmethod
    def _row(tp: int, fp: int, fn: int, side_ok: int, side_n: int) -> dict[str, Any]:
        return {"tp": tp, "fp": fp, "fn": fn, "precision": _ratio(tp, tp + fp),
                "recall": _ratio(tp, tp + fn), "side_n": side_n, "side_accuracy": _ratio(side_ok, side_n)}

    def per_class(self) -> dict[str, dict[str, Any]]:
        classes = sorted(set(self.tp) | set(self.fp) | set(self.fn))
        return {c: self._row(self.tp[c], self.fp[c], self.fn[c], self.side_ok[c], self.side_n[c])
                for c in classes}

    def overall(self) -> dict[str, Any]:
        return self._row(sum(self.tp.values()), sum(self.fp.values()), sum(self.fn.values()),
                         sum(self.side_ok.values()), sum(self.side_n.values()))


# ---------------------------------------------------------------------------
# perception
# ---------------------------------------------------------------------------
def scale_detections(dets: Iterable[Detection], from_size: Sequence[int] | None,
                     to_size: Sequence[int]) -> list[Detection]:
    """Detections of a frame of from_size moved onto an image of to_size (a recording's
    images may be saved smaller than the frames were)."""
    dets = list(dets)
    if from_size is None or (int(from_size[0]), int(from_size[1])) == (int(to_size[0]), int(to_size[1])):
        return dets
    sx, sy = float(to_size[0]) / float(from_size[0]), float(to_size[1]) / float(from_size[1])
    out = []
    for d in dets:
        x1, y1, x2, y2 = d.box
        g = None if d.ground_px is None else (d.ground_px[0] * sx, d.ground_px[1] * sy)
        out.append(Detection(d.cls, d.side_prob, d.conf, (x1 * sx, y1 * sy, x2 * sx, y2 * sy), g, d.form))
    return out


@dataclass
class Step:
    """What perception made of one image."""
    vf: VisionFrame
    cues: Any                                  # vision_play.ScreenCues
    track_tick: int | None = None
    tick_source: str | None = None             # "clock" | "extrapolated" | "time"
    hud_report: dict[str, Any] | None = None
    det: Any = None                            # the DetectorResult when the detector ran
    det_source: str | None = None              # "detector" | "recorded"

    @property
    def battle_vf(self) -> VisionFrame:
        """The frame BattleState follows: with time ticks, the tick is the capture time's."""
        if self.tick_source == TICKS_TIME and self.vf.tick is None:
            return dataclasses.replace(self.vf, tick=self.track_tick)
        return self.vf


class ReplayPerception:
    """vision_play.LivePerception's chain on recorded images, keeping what it saw (Step).

    hud_reader, tracker, deriver, geometry, detector: as LivePerception. names: a Names (the
    detection types of recorded detections). ticks: TICKS_CLOCK (as live) or TICKS_TIME.
    recorded=True takes the recording's detections (step(recorded=VisionFrame)) instead of
    running the detector. step(detect=True) runs the detector on a battle-screen image even
    without a tick (a labelled image is scored either way)."""

    def __init__(self, *, hud_reader: Any, tracker: Any, deriver: Any, geometry: Any,
                 detector: Any = None, names: Names | None = None, ticks: str = TICKS_CLOCK,
                 recorded: bool = False, max_extrapolate_s: float = 5.0, hud_reset_gap_s: float = 3.0):
        if ticks not in (TICKS_CLOCK, TICKS_TIME):
            raise ValueError(f"ticks must be {TICKS_CLOCK!r} or {TICKS_TIME!r}, not {ticks!r}")
        self.hud_reader = hud_reader
        self.tracker = tracker
        self.deriver = deriver
        self.geometry = geometry
        self.detector = detector
        self.names = names
        self.ticks = ticks
        self.recorded = bool(recorded)
        self.max_extrapolate_s = float(max_extrapolate_s)
        self.hud_reset_gap_s = float(hud_reset_gap_s)
        self._anchor: tuple[int, float] | None = None
        self._last_panel_t: float | None = None
        self._t_origin: float | None = None

    def new_battle(self) -> None:
        self.tracker.reset()
        self.deriver.reset()

    def end_battle(self) -> None:
        self.hud_reader.reset()
        self.tracker.reset()
        self.deriver.reset()
        self._anchor = None

    def start_clock(self, t: float) -> None:
        """Time ticks count from the first image (call before its taps are replayed)."""
        if self.ticks == TICKS_TIME and self._t_origin is None:
            self._t_origin = float(t)

    def _time_tick(self, t: float) -> int:
        if self._t_origin is None:
            self._t_origin = float(t)
        return int(round((float(t) - self._t_origin) * TICKS_PER_S))

    def tick_at(self, t: float, clock_tick: int | None = None) -> tuple[int | None, str | None]:
        """The tick to track on at capture time t: the capture time's (time ticks), the clock's,
        or the last clock reading extrapolated (at most max_extrapolate_s)."""
        if self.ticks == TICKS_TIME:
            return self._time_tick(t), TICKS_TIME
        if clock_tick is not None:
            self._anchor = (int(clock_tick), float(t))
            return int(clock_tick), TICKS_CLOCK
        if self._anchor is None or t - self._anchor[1] > self.max_extrapolate_s:
            return None, None
        return self._anchor[0] + int(round((t - self._anchor[1]) * TICKS_PER_S)), "extrapolated"

    def tap_tick(self, clock_tick: Any, t: float) -> int | None:
        """The tick a recorded tap is replayed at: its own (clock) tick, or with time ticks the
        tick of the capture time it was decided on."""
        if self.ticks == TICKS_TIME:
            return self._time_tick(t)
        return None if clock_tick is None else int(clock_tick)

    def note_own_tap(self, card: int, tick: int, *, slot: int, aim_mt: Sequence[float] | None,
                     t: float | None) -> None:
        aim = (None, None) if aim_mt is None else (float(aim_mt[0]), float(aim_mt[1]))
        self.deriver.note_own_tap(int(card), int(tick), slot=int(slot), aim_x_mt=aim[0], aim_y_mt=aim[1],
                                  t=t, tracker=self.tracker)

    def _split(self, dets: list[Detection], res: Any) -> tuple[list, list, list]:
        if res is not None:
            return list(res.units()), list(res.spells()), list(res.bars())
        units, spells, bars = [], [], []
        for d in dets:
            typ = self.names.canon(d.cls)[2] if self.names is not None else TYPE_UNIT
            if d.cls in BAR_CLASSES:
                bars.append(d)
            elif typ == TYPE_UNIT:
                units.append(d)
            elif typ == TYPE_SPELL:
                spells.append(d)
        return units, spells, bars

    def step(self, rgb: np.ndarray, t: float, stale_s: float = 0.0, *,
             recorded: VisionFrame | None = None, detect: bool = False) -> Step:
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError(f"expected an HxWx3 RGB image, got {img.shape}")
        w, h = check_frame_size(img.shape[1], img.shape[0])
        t = float(t)
        self.start_clock(t)
        vf = VisionFrame(t_capture=t, frame_size=(w, h), stale_s=float(stale_s))
        cues = VP.read_cues(img)
        step = Step(vf, cues)
        if cues.end_screen:
            vf.screen = SCREEN_END
            vf.hud = Hud(crowns=VP.end_screen_crowns(img), phase=PHASE_ENDED)
            return step
        if not cues.hud_present:
            vf.screen = SCREEN_MENU
            return step
        vf.screen = SCREEN_BATTLE
        if self._last_panel_t is not None and t - self._last_panel_t > self.hud_reset_gap_s:
            self.hud_reader.reset()                 # back on a battle screen after a while
        self._last_panel_t = t
        vf.hud = self.hud_reader.read(img, t)
        report = getattr(self.hud_reader, "report", None)
        step.hud_report = report.summary() if hasattr(report, "summary") else None
        vf.tick = vf.hud.clock_tick
        step.track_tick, step.tick_source = self.tick_at(t, vf.tick)
        tick = step.track_tick
        res, dets = None, None
        if self.recorded:
            if recorded is not None:
                dets = scale_detections(recorded.detections, recorded.frame_size, (w, h))
                step.det_source = "recorded"
        elif self.detector is not None and (tick is not None or detect):
            res = self.detector.detect(img)
            dets = list(res.detections)
            step.det, step.det_source = res, "detector"
        if dets is None:
            return step
        vf.detections = dets
        if tick is None or self.tracker is None:
            return step
        units, spells, bars = self._split(dets, res)
        reads = hpbars.read_unit_hp(img, [d.box for d in units], [b.box for b in bars])
        obs = observations_from_detections(units + spells, self.geometry, (w, h))
        hpbars.annotate_observations(obs[:len(units)], reads)
        vf.tracks = self.tracker.update(obs, tick)
        vf.events = self.deriver.update(vf.tracks, tick, t=t, tracker=self.tracker)
        return step


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _rate(n: int, d: int) -> float | None:
    return None if d == 0 else round(n / d, 4)


class HudStats:
    """Read rates over battle-screen frames, and agreement with a recording's own reads."""

    KEYS = ("clock", "elixir", "hand", "next")

    def __init__(self):
        self.frames = 0
        self.clock = self.elixir = self.hand_full = self.hand_slots = self.next = 0
        self.phase: Counter = Counter()
        self.towers: Counter = Counter()
        self.agree = {k: [0, 0] for k in self.KEYS}     # [agreeing, compared]

    def add(self, hud: Hud, recorded: VisionFrame | None = None,
            names: Callable[[int, bool], Any] | None = None) -> dict[str, bool] | None:
        """One battle-screen frame. names(card, recorded) -> a comparable card key (the two
        runs may number cards differently). Returns this frame's agreement, or None."""
        self.frames += 1
        self.clock += hud.clock_tick is not None
        self.elixir += hud.elixir_milli is not None
        self.phase[hud.phase] += 1
        known = sum(1 for c in hud.hand if c != EMPTY_CARD)
        self.hand_slots += known
        self.hand_full += known == HAND_SIZE
        self.next += hud.next != EMPTY_CARD
        for k, v in hud.towers.items():
            if v.hp is not None:
                self.towers[k] += 1
        if recorded is None or recorded.screen != SCREEN_BATTLE:
            return None
        key = names or (lambda card, rec: card)
        live = recorded.hud
        out: dict[str, bool] = {}
        if hud.clock_tick is not None and live.clock_tick is not None:
            out["clock"] = abs(hud.clock_tick - live.clock_tick) <= CLOCK_TOL_TICKS
        if hud.elixir_milli is not None and live.elixir_milli is not None:
            out["elixir"] = abs(hud.elixir_milli - live.elixir_milli) <= ELIXIR_TOL_MILLI
        if EMPTY_CARD not in hud.hand and EMPTY_CARD not in live.hand:
            out["hand"] = [key(c, False) for c in hud.hand] == [key(c, True) for c in live.hand]
        if hud.next != EMPTY_CARD and live.next != EMPTY_CARD:
            out["next"] = key(hud.next, False) == key(live.next, True)
        for k, ok in out.items():
            self.agree[k][0] += int(ok)
            self.agree[k][1] += 1
        return out or None

    def summary(self) -> dict[str, Any]:
        n = self.frames
        return {"frames": n, "clock_read": _rate(self.clock, n), "elixir_read": _rate(self.elixir, n),
                "hand_full": _rate(self.hand_full, n), "hand_slots_read": _rate(self.hand_slots, n * HAND_SIZE),
                "next_read": _rate(self.next, n), "phase": dict(self.phase),
                "towers_read": {k: _rate(v, n) for k, v in sorted(self.towers.items())},
                "agreement": {k: {"agree": a, "n": c, "rate": _rate(a, c)} for k, (a, c) in self.agree.items()}}


class DetStats:
    """Detections per frame, by type and class; off-board drops, suppressions, time."""

    def __init__(self, names: Names | None):
        self.names = names
        self.frames = 0
        self.per_frame: list[int] = []
        self.by_type: Counter = Counter()
        self.by_class: Counter = Counter()
        self.dropped = self.suppressed = self.raw = 0
        self.ms: list[float] = []

    def add(self, step: Step) -> dict[str, Any] | None:
        if step.det_source is None:
            return None
        dets = step.vf.detections
        types: Counter = Counter()
        for d in dets:
            typ = self.names.canon(d.cls)[2] if self.names is not None else "unknown"
            types[typ] += 1
            self.by_class[d.cls] += 1
        self.frames += 1
        self.per_frame.append(len(dets))
        self.by_type.update(types)
        res = step.det
        out: dict[str, Any] = {"source": step.det_source, "n": len(dets), "by_type": dict(types)}
        if res is not None:
            dropped = len(getattr(res, "dropped", ()) or ())
            suppressed = int(getattr(res, "suppressed", 0) or 0)
            raw = int(getattr(res, "raw_boxes", 0) or 0)
            ms = getattr(res, "ms", None)
            self.dropped += dropped
            self.suppressed += suppressed
            self.raw += raw
            if ms is not None:
                self.ms.append(float(ms))
            out.update(dropped=dropped, suppressed=suppressed, raw=raw, ms=ms)
        return out

    def summary(self, top: int = 25) -> dict[str, Any]:
        return {"frames": self.frames, "per_frame": describe(self.per_frame), "by_type": dict(self.by_type),
                "top_classes": dict(self.by_class.most_common(top)), "classes": len(self.by_class),
                "dropped": self.dropped, "suppressed": self.suppressed, "raw_boxes": self.raw,
                "ms": describe(self.ms)}


class TrackStats:
    """Tracks over a replay: per frame, lifetimes, teams, label switches, short tracks.
    A segment is the span between two tracker resets (uids restart there)."""

    def __init__(self):
        self._tracks: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.per_frame: list[int] = []

    def add(self, segment: int, tracks: Sequence[Any], tick: int | None) -> dict[str, int]:
        self.per_frame.append(len(tracks))
        teams: Counter = Counter()
        for tr in tracks:
            key = (int(segment), int(tr.uid), int(tr.first_seen_tick))
            r = self._tracks.get(key)
            if r is None:
                r = self._tracks[key] = {"first": int(tr.first_seen_tick), "last": int(tr.last_seen_tick),
                                         "frames": 0, "labels": set(), "team": int(tr.team), "decided": None}
            r["last"] = max(r["last"], int(tr.last_seen_tick))
            r["frames"] += 1
            r["labels"].add(str(tr.cls))
            r["team"] = int(tr.team)
            if r["decided"] is None and tr.team in (OWN_TEAM, ENEMY_TEAM) and tick is not None:
                r["decided"] = int(tick) - int(tr.first_seen_tick)
            teams["own" if tr.team == OWN_TEAM else "enemy" if tr.team == ENEMY_TEAM else "unknown"] += 1
        return {"n": len(tracks), "own": teams["own"], "enemy": teams["enemy"], "unknown": teams["unknown"]}

    def summary(self) -> dict[str, Any]:
        recs = list(self._tracks.values())
        life = [r["last"] - r["first"] for r in recs]
        teams = Counter("own" if r["team"] == OWN_TEAM else "enemy" if r["team"] == ENEMY_TEAM else "unknown"
                        for r in recs)
        return {"tracks": len(recs), "per_frame": describe(self.per_frame), "lifetime_ticks": describe(life),
                "frames_per_track": describe(r["frames"] for r in recs),
                "short": sum(1 for x in life if x < SHORT_TRACK_TICKS), "short_ticks": SHORT_TRACK_TICKS,
                "label_switched": sum(1 for r in recs if len(r["labels"]) > 1),
                "team": {k: teams[k] for k in ("own", "enemy", "unknown")},
                "ticks_to_team": describe(r["decided"] for r in recs)}


# ---------------------------------------------------------------------------
# the replay
# ---------------------------------------------------------------------------
def read_recording_records(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """(tap records, the vision_play "config" record's data) of a recorded session; ([], None)
    for an image folder."""
    p = Path(path)
    if not ((p.is_dir() and (p / FRAMES_FILE).is_file()) or (p.is_file() and p.suffix == ".jsonl")):
        return [], None
    taps, config = [], None
    for e in VisionLogReader(p):
        if e.kind == "tap":
            taps.append(e.data)
        elif e.kind == "config" and config is None:
            config = dict(e.data.get("data") or {})
    return taps, config


class Replay:
    """Drive a ReplayPerception over a FileReplaySource-like source (module doc).

    source: read() -> capture.Frame or None at the end; after each read `vision_frame` (the
    recorded VisionFrame or None) and `image_path`. battle: vision_play.BattleState. recorder:
    a VisionRecorder (its clock must be this object's log_clock) or None. taps: the
    recording's "tap" records. labels: a LabelIndex or None. card_ids: card name -> this
    run's card id (taps are mapped by name); card_names: this run's id -> name; live_names:
    the recording's id -> name (vision_play's config deck); limit: stop after N images."""

    def __init__(self, source: Any, perception: ReplayPerception, *, battle: Any = None,
                 recorder: Any = None, taps: Iterable[Mapping[str, Any]] = (), labels: LabelIndex | None = None,
                 names: Names | None = None, iou: float = 0.5, pr_types: Iterable[str] | None = PR_TYPES,
                 card_ids: Mapping[str, int] | None = None, card_names: Mapping[int, str] | None = None,
                 live_names: Mapping[int, str] | None = None, limit: int = 0,
                 out: Callable[[str], Any] = print, progress_every: int = 0, max_error_logs: int = 20):
        self.source = source
        self.perception = perception
        self.battle = battle if battle is not None else VP.BattleState()
        self.recorder = recorder
        self.labels = labels
        self.names = names if names is not None else (labels.canon if labels is not None else perception.names)
        self.pr = PRCounter(iou) if labels is not None else None
        self.pr_types = None if pr_types is None else tuple(pr_types)
        self.card_ids = dict(card_ids or {})
        self.card_names = dict(card_names or {})
        self.live_names = dict(live_names or {})
        self.limit = int(limit or 0)
        self.out = out
        self.progress_every = int(progress_every or 0)
        self.max_error_logs = int(max_error_logs)
        self.hud = HudStats()
        self.dets = DetStats(self.names)
        self.tracks = TrackStats()
        self.events: Counter = Counter()
        self.opp_cards: Counter = Counter()
        self.screens: Counter = Counter()
        self.tick_sources: Counter = Counter()
        self.errors: Counter = Counter()
        self.frames = 0
        self.battles = 0
        self.segment = 0
        self.taps_replayed = 0
        self.taps_skipped = 0
        self.unscored_labelled = 0
        self.intervals: list[float] = []
        self.log_t = 0.0
        self._taps = sorted(self._tap_items(taps), key=lambda x: x[0])

    def log_clock(self) -> float:
        """The clock a VisionRecorder writing this replay is given: the time being replayed."""
        return self.log_t

    @staticmethod
    def _tap_items(taps: Iterable[Mapping[str, Any]]) -> list[tuple[float, float, dict]]:
        """(time decided on, time logged, data) per tap record with a time."""
        out = []
        for rec in taps:
            data = dict(rec.get("data") or {})
            t = rec.get("t")
            if t is None:
                continue
            c2t = (data.get("ms") or {}).get("capture_to_tap")
            t_dec = float(t) - float(c2t) / 1000.0 if c2t is not None else float(t)
            out.append((t_dec, float(t), data))
        return out

    def _log(self, kind: str, **data: Any) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.log(kind, **data)
        except Exception:
            self.errors["log"] += 1

    def _send_taps(self, t_frame: float) -> None:
        while self._taps and self._taps[0][0] < t_frame:
            t_dec, t_log, data = self._taps.pop(0)
            name = data.get("name")
            card = self.card_ids.get(str(name)) if name is not None else None
            if card is None:
                card = data.get("card")
            tile = data.get("screen_tile")
            aim = None if not tile else ((float(tile[0]) + 0.5) * 1000.0, (float(tile[1]) + 0.5) * 1000.0)
            tick = self.perception.tap_tick(data.get("tick"), t_dec)
            if card is None or tick is None:
                self.taps_skipped += 1
                continue
            try:
                self.perception.note_own_tap(int(card), tick, slot=int(data.get("slot", 0)), aim_mt=aim, t=t_log)
            except Exception:
                self.taps_skipped += 1
                self.errors["tap_replay"] += 1
                continue
            self.taps_replayed += 1
            self.log_t = t_log
            self._log("tap", **{**data, "card": int(card), "tick": int(tick), "live_card": data.get("card"),
                                "live_tick": data.get("tick"),
                                "battle": self.battle.number if self.battle.in_battle else None,
                                "replayed": True})

    def _card_key(self, card: int, recorded: bool) -> Any:
        names = self.live_names if recorded else self.card_names
        return names.get(int(card), int(card)) if names else int(card)

    def _events(self, vf: VisionFrame) -> dict[str, int]:
        for p in vf.events.opp_plays:
            self.events["opp_plays"] += 1
            self.opp_cards[self.card_names.get(int(p.card), str(p.card))] += 1
        landed = missed = human = 0
        for x in vf.events.own_landings:
            if x.slot < 0:
                human += 1
            elif x.land_tick is None:
                missed += 1
            else:
                landed += 1
        self.events["own_landed"] += landed
        self.events["own_missed"] += missed
        self.events["own_unmatched"] += human
        return {"opp_plays": len(vf.events.opp_plays), "own_landings": landed, "own_missed": missed,
                "own_unmatched": human}

    def run(self) -> dict[str, Any]:
        last_t = None
        while not (self.limit and self.frames >= self.limit):
            frame = self.source.read()
            if frame is None:
                break
            recorded = getattr(self.source, "vision_frame", None)
            image = getattr(self.source, "image_path", None)
            self.frames += 1
            if last_t is not None and frame.t > last_t:
                self.intervals.append(frame.t - last_t)
            last_t = frame.t
            self.log_t = float(frame.t)
            self.perception.start_clock(float(frame.t))
            self._send_taps(float(frame.t))
            truth = self.labels.truth_for(image) if self.labels is not None else None
            try:
                step = self.perception.step(frame.rgb, frame.t, getattr(frame, "stale_s", 0.0),
                                            recorded=recorded, detect=truth is not None)
            except Exception as exc:
                self.errors[type(exc).__name__] += 1
                if sum(self.errors.values()) <= self.max_error_logs:
                    self._log("replay_error", i=self.frames, image=None if image is None else str(image),
                              error=f"{type(exc).__name__}: {exc}")
                continue
            self._frame(step, recorded, image, truth)
            if self.progress_every and self.frames % self.progress_every == 0:
                self.out(f"  {self.frames} images, {self.battles} battle(s), "
                         f"{self.tracks.summary()['tracks']} tracks")
        return self.summary()

    def _frame(self, step: Step, recorded: VisionFrame | None, image: Any, truth: list[Box] | None) -> None:
        vf = step.vf
        self.screens[vf.screen] += 1
        if step.tick_source:
            self.tick_sources[step.tick_source] += 1
        events = self.battle.update(step.battle_vf)
        if "end" in events:
            self.perception.end_battle()
            self.segment += 1
        if "start" in events:
            self.perception.new_battle()
            self.segment += 1
            self.battles += 1
        info: dict[str, Any] = {"i": self.frames, "image": None if image is None else Path(image).name,
                                "tick_source": step.tick_source, "track_tick": step.track_tick,
                                "cues": step.cues.to_dict(), "hud_report": step.hud_report}
        if vf.screen == SCREEN_BATTLE:
            info["hud_agree"] = self.hud.add(vf.hud, recorded, self._card_key)
        info["det"] = self.dets.add(step)
        if step.det_source is not None and step.track_tick is not None:
            info["tracks"] = self.tracks.add(self.segment, vf.tracks, step.track_tick)
        info["events"] = self._events(vf)
        if truth is not None:
            if step.det_source is None or self.pr is None:
                self.unscored_labelled += 1
                info["pr"] = None
            else:
                preds = pred_boxes(vf.detections, vf.frame_size, self.names, self.pr_types)
                info["pr"] = self.pr.add(truth, preds)
        if self.recorder is not None:
            try:
                self.recorder.write(vf, None, battle=self.battle.number if self.battle.in_battle else None,
                                    state=self.battle.state, replay=info)
            except Exception:
                self.errors["record"] += 1

    def summary(self) -> dict[str, Any]:
        gaps = describe(self.intervals)
        tracker = getattr(self.perception, "tracker", None)
        max_age = getattr(getattr(tracker, "config", None), "max_age_ticks", 20.0)
        sparse = gaps is not None and gaps["median"] > float(max_age) / TICKS_PER_S
        deriver = getattr(self.perception, "deriver", None)
        landing = deriver.landing_stats.summary() if hasattr(deriver, "landing_stats") else None
        pr = None
        if self.pr is not None:
            pr = {"frames": self.pr.frames, "iou": self.pr.iou_min,
                  "types": None if self.pr_types is None else list(self.pr_types),
                  "label_files": len(self.labels), "label_files_used": len(self.labels.used),
                  "labelled_not_scored": self.unscored_labelled, "bad_lines": self.labels.bad_lines,
                  "unknown_names": dict(self.labels.unknown), "duplicates": list(self.labels.duplicates),
                  "overall": self.pr.overall(), "per_class": self.pr.per_class()}
        return {"schema": SCHEMA, "frames": self.frames, "errors": dict(self.errors),
                "screens": dict(self.screens), "ticks": self.perception.ticks,
                "tick_sources": dict(self.tick_sources), "battles": self.battles,
                "image_interval_s": gaps, "sparse": sparse,
                "hud": self.hud.summary(), "detections": self.dets.summary(), "tracker": self.tracks.summary(),
                "events": {"opp_plays": self.events["opp_plays"], "opp_cards": dict(self.opp_cards),
                           "own_landed": self.events["own_landed"], "own_missed": self.events["own_missed"],
                           "own_unmatched": self.events["own_unmatched"], "landing_stats": landing},
                "taps": {"replayed": self.taps_replayed, "skipped": self.taps_skipped, "left": len(self._taps)},
                "precision_recall": pr}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100.0 * x:.1f}%"


def format_summary(s: Mapping[str, Any]) -> list[str]:
    """The printed summary of Replay.summary()."""
    lines = [f"replayed {s['frames']} images ({', '.join(f'{k} {v}' for k, v in s['screens'].items()) or '-'}), "
             f"{s['battles']} battle(s), ticks from the {s['ticks']}"
             + (f"; errors {s['errors']}" if s["errors"] else "")]
    gi = s["image_interval_s"]
    if gi:
        lines.append(f"  images every {gi['median'] * 1000:.0f} ms (median)"
                     + ("; SPARSE: tracks expire between images, record densely for tracker figures"
                        if s["sparse"] else ""))
    h = s["hud"]
    if h["frames"]:
        lines.append(f"HUD over {h['frames']} battle frames: clock {_pct(h['clock_read'])}, elixir "
                     f"{_pct(h['elixir_read'])}, full hand {_pct(h['hand_full'])}, next {_pct(h['next_read'])}; "
                     "towers " + (", ".join(f"{k} {_pct(v)}" for k, v in h["towers_read"].items()) or "-"))
        agree = [f"{k} {_pct(v['rate'])} of {v['n']}" for k, v in h["agreement"].items() if v["n"]]
        if agree:
            lines.append("  agreement with the live reads: " + ", ".join(agree))
    d = s["detections"]
    if d["frames"]:
        pf = d["per_frame"]
        lines.append(f"detections on {d['frames']} frames: {pf['mean']:.1f} per frame (median {pf['median']:g}, "
                     f"max {pf['max']:g}); by type {d['by_type']}; off the board {d['dropped']}, "
                     f"suppressed {d['suppressed']}")
        if d["top_classes"]:
            lines.append("  classes: " + ", ".join(f"{k} {v}" for k, v in list(d["top_classes"].items())[:12]))
    t = s["tracker"]
    if t["tracks"]:
        lt = t["lifetime_ticks"]
        lines.append(f"tracks: {t['tracks']} (own {t['team']['own']}, enemy {t['team']['enemy']}, undecided "
                     f"{t['team']['unknown']}); lifetime median {lt['median']:g} ticks; {t['short']} shorter than "
                     f"{t['short_ticks']} ticks; {t['label_switched']} changed label")
    e = s["events"]
    lines.append(f"events: {e['opp_plays']} opponent plays {e['opp_cards'] or ''}; own landed {e['own_landed']}, "
                 f"missed {e['own_missed']}, not ours to match {e['own_unmatched']}; taps replayed {s['taps']['replayed']}")
    pr = s["precision_recall"]
    if pr is not None:
        o = pr["overall"]
        lines.append(f"precision / recall on {pr['frames']} labelled frames (IoU {pr['iou']:g}): precision "
                     f"{_pct(o['precision'])}, recall {_pct(o['recall'])} (TP {o['tp']}, FP {o['fp']}, FN {o['fn']}); "
                     f"side {_pct(o['side_accuracy'])} of {o['side_n']}")
        worst = sorted(pr["per_class"].items(), key=lambda kv: -(kv[1]["fp"] + kv[1]["fn"]))[:10]
        for cls, r in worst:
            lines.append(f"  {cls:22s} P {_pct(r['precision']):>6s}  R {_pct(r['recall']):>6s}  "
                         f"TP {r['tp']:4d} FP {r['fp']:4d} FN {r['fn']:4d}")
        if pr["labelled_not_scored"]:
            lines.append(f"  {pr['labelled_not_scored']} labelled images not scored (no detections: off the "
                         "battle screen, or no detector)")
    return lines


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def engine_card_ids(path: str | Path | None, table: Any) -> tuple[dict | None, dict[str, int], str]:
    """(spawn table or None, card name -> id, where the ids come from): the engine's catalogue
    order from spawn_primary.json, else replay-only ids (classes.json's cards, sorted)."""
    p = Path(path) if path else SPAWN_TABLE
    try:
        spawn = json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        spawn = None
    names = list(spawn.get("card_names") or ()) if isinstance(spawn, dict) else []
    if names:
        return spawn, {str(n): i for i, n in enumerate(names)}, f"engine catalogue ({p})"
    return None, {n: i for i, n in enumerate(sorted(table.cards))}, "replay-only ids (classes.json)"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tools/vision_replay.py",
                                 description="Re-run perception offline on a recording (V11).")
    ap.add_argument("recording", help="a recorded session (capture --record, vision_play) or an image folder")
    ap.add_argument("--out", default=None, help=f"root for the replay session (default {OUT_ROOT})")
    ap.add_argument("--labels", default=None, help="a folder of YOLO label files: precision / recall")
    ap.add_argument("--label-names", default=None,
                    help="class names of the labels (detector_labels.json, a dataset yaml, or one per line)")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU a prediction needs to match a label")
    ap.add_argument("--pr-all", action="store_true", help="score every type, HP bars and UI included")
    ap.add_argument("--detector", default=None, help="detector weights (default: the pin)")
    ap.add_argument("--device", default=None, help="detector device (default: CUDA GPU 0, else cpu)")
    ap.add_argument("--no-detector", action="store_true", help="screen cues and HUD only")
    ap.add_argument("--recorded", action="store_true",
                    help="use the recording's detections (a vision_play session) instead of the detector")
    ap.add_argument("--geometry", default=None, help="an ArenaGeometry JSON (default: data/vision's)")
    ap.add_argument("--classes", default=None, help="classes.json (default data/vision/classes.json)")
    ap.add_argument("--templates", default=None, help="HUD template root (default: the TemplateStore's)")
    ap.add_argument("--deck", type=VP.deck_names, default=VP.deck_names(VP.DEFAULT_DECK),
                    help="our deck: 'Hog 2.6' (default) or 8 catalogue names, comma-separated")
    ap.add_argument("--spawn-table", default=None, help=f"card ids from this (default {SPAWN_TABLE})")
    ap.add_argument("--ticks", choices=("auto", TICKS_CLOCK, TICKS_TIME), default="auto",
                    help="track on the HUD clock or the capture time (auto: the clock when its templates exist)")
    ap.add_argument("--fps", type=float, default=20.0, help="an image folder's frame rate")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (0: all)")
    ap.add_argument("--no-taps", action="store_true", help="do not replay the recording's taps")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from scripts.vision import capture as CAP
    from scripts.vision import events as EV
    from scripts.vision import hud as HUD
    from scripts.vision import templates as TPL
    from scripts.vision.geometry import load_geometry
    from scripts.vision.labels import load_class_table
    from scripts.vision.record import VisionRecorder
    from scripts.vision.tracker import Tracker

    rec_path = Path(args.recording)
    if not rec_path.exists():
        print(f"nothing to replay at {rec_path}")
        return 2
    table = load_class_table(args.classes)
    names = Names(table)
    geometry = load_geometry(args.geometry)
    deck = list(args.deck)
    spawn, card_ids, ids_from = engine_card_ids(args.spawn_table, table)
    cmap, ignore = EV.unit_class_map(table.doc), EV.cardless_classes(table.doc)
    if spawn is not None:
        tables = EV.EventTables.for_engine(spawn, class_of=cmap, ignore=ignore)
    else:
        tables = EV.EventTables.from_names(EV.load_spawn_children(), card_ids, class_of=cmap, ignore=ignore)
    deck_ids = [card_ids[n] for n in deck if n in card_ids]
    store = TPL.TemplateStore(args.templates) if args.templates else TPL.TemplateStore()
    hud_reader = HUD.HudReader(store, {n: card_ids[n] for n in deck if n in card_ids},
                               deck=deck_ids if len(deck_ids) == 8 else None)
    ticks = args.ticks
    if ticks == "auto":
        ticks = TICKS_CLOCK if hud_reader.clock_reader.ready else TICKS_TIME
    detector, det_note = None, "off (--no-detector)" if args.no_detector else None
    if args.recorded:
        det_note = "the recording's detections (--recorded)"
    elif not args.no_detector:
        from scripts.vision.detector import Detector as _Det
        try:
            detector = _Det.load(args.detector, device=args.device, geometry=geometry, table=table)
            detector.warmup()
            det_note = f"{detector.weights} sha256 {str(detector.sha256)[:12]} on {detector.device_desc}"
        except Exception as exc:
            det_note = f"NONE ({type(exc).__name__}: {exc})"
    labels = None
    if args.labels:
        names_path = Path(args.label_names) if args.label_names else find_label_names(args.labels)
        if names_path is not None:
            label_names = read_label_names(names_path)
        elif detector is not None:
            raw = getattr(detector.model, "names", None) or {}
            label_names = {int(k): str(v) for k, v in (raw.items() if isinstance(raw, Mapping) else enumerate(raw))}
        else:
            print("--labels: no class names (--label-names, a names file in the folder, or the detector's)")
            return 2
        labels = LabelIndex(args.labels, label_names, table, types=None if args.pr_all else PR_TYPES, canon=names)
    taps, config = ([], None) if args.no_taps else read_recording_records(rec_path)
    live_names = {}
    if config and config.get("deck") and config.get("deck_ids"):
        live_names = {int(i): str(n) for n, i in zip(config["deck"], config["deck_ids"])}
    perception = ReplayPerception(hud_reader=hud_reader, tracker=Tracker(own_classes=table.deck_labels(deck)),
                                  deriver=EV.EventDeriver(tables, own_deck=deck_ids), geometry=geometry,
                                  detector=detector, names=names, ticks=ticks, recorded=args.recorded)
    source = CAP.FileReplaySource(rec_path, realtime=False, fps=args.fps)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    src_name = (rec_path.parent if rec_path.is_file() else rec_path).name
    holder: dict[str, Replay] = {}
    recorder = VisionRecorder(Path(args.out) if args.out else OUT_ROOT, session=f"{src_name}_replay_{stamp}",
                              image_every=0, clock=lambda: holder["r"].log_clock() if holder else 0.0)
    print(f"vision_replay  {rec_path} ({len(source)} images) -> {recorder.session_dir}")
    print(f"  detector   {det_note}")
    print(f"  hud        clock templates {'ok' if hud_reader.clock_reader.ready else 'MISSING'}; ticks from the {ticks}")
    print(f"  card ids   {ids_from}; deck {', '.join(deck)}")
    if labels is not None:
        print(f"  labels     {len(labels)} label files under {args.labels}")
    replay = Replay(source, perception, recorder=recorder, taps=taps, labels=labels, names=names, iou=args.iou,
                    pr_types=None if args.pr_all else PR_TYPES, card_ids=card_ids,
                    card_names={i: n for n, i in card_ids.items()}, live_names=live_names, limit=args.limit,
                    progress_every=1000)
    holder["r"] = replay
    try:
        summary = replay.run()
        summary.update(input=str(rec_path), output=str(recorder.session_dir), detector=det_note,
                       card_ids=ids_from, deck=deck)
        recorder.log("replay_summary", **summary)
    finally:
        source.close()
        recorder.close()
    (recorder.session_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    for line in format_summary(summary):
        print(line)
    print(f"summary -> {recorder.session_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
