r"""Unit harvest scan: two perception chains over a recorded friendly, merged into tracks (N09).

docs/NEW_CARDS_PLAN.md 2.3 step 2. The owner records friendlies where his main account plays
the new cards against Yodle2 (plan section 4); this scan turns one recording into per-frame
PROPOSALS (every place a unit or spell may be) and HARVEST TRACKS (proposals followed over
time, by position only), next to what tonight's live chain made of the same images. N10
(flags S1-S7, splits, clustering, the contact sheet, `label`) reads its output; nothing here
names anything.

    & $BB -m scripts.vision.unit_harvest scan reports\vision\autolabel\$SID\recording --out reports\vision\autolabel\$SID\scan
    & $BB -m scripts.vision.unit_harvest scan reports\vision\harvest\$SID            # a recording not ingested yet
    ... --force                                          scan again (replaces the earlier scan's files)
    ... --detector runs\detect\cr_m0\weights\best.pt    weights other than the pin
    ... --no-detector                                    blobs only (no GPU; no labels to read)
    ... --limit 400                                      a quick look at the first 400 images

Run it in the GPU venv with MuMu and the browser closed: two detector passes per image (a few
minutes per battle), then the background median and the blobs on the CPU. The first line scans
a friendly that tools/vision_autolabel.py `ingest` (N08) put in its store
(<store>/<sid>/recording); vision_autolabel's `scan <sid>` subcommand is meant to call
scan_session(<store>/<sid>/recording, <store>/<sid>/scan) the same way.

INPUT. A session written by scripts/vision/record.py (python -m scripts.vision.capture --record
ROOT: ROOT/<session>/frames.jsonl plus images/<8 digits>.jpg), its frames.jsonl, a ROOT holding
exactly one session, or a plain folder of images (20 a second). Only frames with an image on
disk are scanned (scripts/vision/capture.FileReplaySource), in capture order.

WHAT RUNS, per image (pass 1, GPU):
1. THE LIVE CHAIN, exactly as tools/vision_replay.py replays it: ReplayPerception (HUD, the
   detector at the live DetectorConfig, HP bars, Tracker, EventDeriver) driven by
   vision_replay.Replay, which starts and ends battles with vision_play's BattleState and
   replays the recording's taps. ScanPerception WRAPS that ReplayPerception (it does not
   subclass it, and tools/vision_replay.py is imported only when a scan runs): Replay calls it
   like the perception it wraps. Its tracker is a RecordingTracker (a Tracker subclass that
   keeps the observations of the last update), so Tracker.last_assignment (N05) gives the
   live uid of every live detection: each observation is matched back to the detection it was
   made from (the same observations_from_detections call on the same box, so the floats agree
   exactly). Every live track is stored with EventDeriver.how_of (N05) and its card.
2. THE HARVEST DETECTOR PASS (LowConfPass): the same weights (the same model object) on the
   same crop (scripts/vision/detector.crop_rect_for, N05) at conf 0.05 and up to 300 boxes,
   with NO side merge (an ally and an enemy box of one sprite stay two boxes, each with its
   own side) and NO cross-class suppression; boxes off the board are dropped as the live
   detector drops them.
Pass 1 is written to <out>/pass1.jsonl.tmp; frames outside the battle screen keep no boxes.

BACKGROUND (between the passes, CPU). Per battle (the battle number BattleState gave the frame;
battle-screen frames before a start count to the battle that starts next) and per EPOCH: an
epoch ends when a tower falls (the HUD reads a tower at 0 for `tower_fall_frames` frames after
it read it standing; plan risk 2: ruins are a new background) and no epoch is longer than
`epoch_s`. Up to `bg_samples` frames, evenly spread over the epoch, give a per-pixel median in
which pixels under any unit, spell, object, unknown or UI box (live, or harvest at
`bg_mask_conf` or more, grown by `bg_mask_pad_tiles`) do not count (masked_median: rows in
bands of `bg_band_rows`, so its memory stays small; a pixel masked in every sample takes the
plain median). Only the detector crop is modelled. Backgrounds are saved as
<out>/background/<epoch>.png (or .npy without an image codec); N11 masks with them.

PASS 2, per battle-screen image (CPU):
3. BLOBS: pixels on the board (and in the crop) whose largest channel differs from the epoch's
   background by more than `fg_threshold`, minus tower and UI boxes (HP bars, the stopwatch:
   plan risk 1, "subtract bar boxes"), counted on a grid of `cell_tiles`; cells at least
   `cell_min_frac` full, closed by one cell, are joined 8-connected (scipy.ndimage.label when
   scipy imports, else a breadth-first fill with the same result), and kept between
   `blob_min_tiles` and `blob_max_tiles` square tiles.
4. PROPOSAL MERGE (merge_proposals): harvest boxes of type unit, spell, object or unknown at
   `prop_conf` or more, strongest first, join the group whose first (strongest) box they
   overlap at IoU >= `group_iou`, whatever their class or side; each group is one proposal,
   its box the strongest box's, its candidates every (class, side) seen with their best score.
   Each live detection (not towers or UI) joins the group it overlaps most (IoU >=
   `live_iou`, one per group), else becomes a proposal of its own ("live"). A blob that
   overlaps a proposal (IoU >= `blob_iou`, or `blob_cover` of the blob or of the proposal
   inside the other) is attached to the one it overlaps most and adds "+blob" to its source;
   a blob nothing explains is a proposal of its own ("blob", no candidates).
5. THE HARVEST TRACKER (HarvestTracker): a Tracker with class_cost and team_cost 0 (position
   only: a label flicker never splits a track), min_hits 1, on the capture time (20 ticks a
   second from the battle's first image), restarted per battle; Tracker.last_assignment gives
   each proposal its harvest track id `h` (unique over the scan).

OUTPUT (<out>, default <session>/scan; reports/vision is gitignored and images never leave the
data drive; --force replaces a previous scan's files and nothing else):
    scan.json           the header and summary: recording, detector (weights, SHA-256, live
                        and harvest settings), config, the live replay's summary (what
                        tonight's model read: opp_cards, tracks, detections), harvest counts,
                        epochs (frames, samples, background file, why each began), errors.
    scan_frames.jsonl   one record per image (not frames.jsonl, so the scan folder is never
                        mistaken for a recorder session): n (scan order), i (the recorder's
                        frame number), image (relative to the session), t, screen, size, seg
                        (live tracker segment: live uids restart at each battle start and
                        end), battle, hk (harvest battle), tick / tick_src (the live chain's);
                        on battle-screen images also live (rows of LIVE_FIELDS: uid is the live
                        track the detection went to, null for towers, UI or an untracked frame),
                        tracks (the live chain's tracks with how and card), plays (its opponent
                        plays), towers (HUD hit points), raw (rows of RAW_FIELDS), epoch, blobs
                        (rows of BLOB_FIELDS), props (Proposal.to_dict: box, src, type, conf,
                        cands [[cls, side, conf]], blob, live [seg, uid], live_cls, x, y, h).
    tracks.json         "harvest": one HarvestTrack.to_dict per harvest track (frames as
                        [n, k] pairs: scan_frames record n, its proposal k; sources; summed
                        candidate scores; the live tracks it held and their classes; birth
                        and last positions); "live": per live track ("<seg>:<uid>") its
                        classes, team, how, card and time span.
    background/         one image per epoch.
load_scan() and iter_scan_frames() read it back.

Hard rules (CLAUDE.md): nothing here taps, captures, starts or touches the emulator, or reads
game memory; it reads image files. Imports numpy and scripts/vision; tools/vision_replay.py
(scripts/vision_play.py with it) only when a live chain is built, and the detector's
ultralytics / torch only through Detector.load in the command line.
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__" and not __package__:
    # Run as a file, sys.path[0] is scripts/vision, whose types.py would shadow
    # the standard library's `types`: put the repo root there instead (PEP 366).
    _here = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if sys.path and os.path.normcase(os.path.abspath(sys.path[0] or os.curdir)) == _here:
        sys.path[0] = _root
    else:
        sys.path.insert(0, _root)
    __import__("scripts.vision")
    __package__ = "scripts.vision"

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from .detector import BAR_CLASSES, DetectorConfig, box_iou, crop_rect_for  # noqa: E402
from .geometry import ArenaGeometry, HEIGHT_MT, WIDTH_MT, check_frame_size, in_arena  # noqa: E402
from .labels import (TYPE_OBJECT, TYPE_PAD, TYPE_SPELL, TYPE_TOWER, TYPE_UI, TYPE_UNIT,  # noqa: E402
                     TYPE_UNKNOWN, load_class_table, resolve_label, split_side)
from .record import (FRAMES_FILE, IMAGES_DIR, image_codec_available, list_sessions,  # noqa: E402
                     load_image, save_image)
from .tracker import Observation, Tracker, TrackerConfig, observations_from_detections  # noqa: E402
from .types import FORM_BASE, SCREEN_BATTLE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCAN_SCHEMA = "hastycr.vision.unit_harvest.scan/1"
SCAN_DIR = "scan"                          # default output folder inside the session
SCAN_JSON = "scan.json"
SCAN_FRAMES = "scan_frames.jsonl"
TRACKS_JSON = "tracks.json"
BACKGROUND_DIR = "background"
PASS1_TMP = "pass1.jsonl.tmp"
TICKS_PER_S = 20
BLOB_LABEL = "blob"                        # the harvest tracker's label of a proposal without a class
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".npy")

#: Field order of a harvest detector box row (scan_frames.jsonl "raw"). side: 0 ally, 1 enemy,
#: null when the model's names carry none; cls: the vision class (units, towers), the spell's
#: card, else the detector name (name: the model's name without its side).
RAW_FIELDS = ("cls", "name", "side", "conf", "x1", "y1", "x2", "y2", "type", "form", "x_mt", "y_mt")
#: Field order of a live detection row ("live"); p_enemy is the live detector's merged P(enemy).
LIVE_FIELDS = ("uid", "cls", "type", "conf", "p_enemy", "form", "x1", "y1", "x2", "y2")
#: Field order of a blob row ("blobs"): its box in frame pixels and its foreground pixels.
BLOB_FIELDS = ("x1", "y1", "x2", "y2", "area_px")

#: Harvest box types that can be a unit or a spell on the board (proposals).
PROPOSAL_TYPES = (TYPE_UNIT, TYPE_SPELL, TYPE_OBJECT, TYPE_UNKNOWN)
#: Box types whose pixels never count for the background median (things that move or pop up).
MASK_TYPES = (TYPE_UNIT, TYPE_SPELL, TYPE_OBJECT, TYPE_UNKNOWN, TYPE_UI)
#: Box types cut out of the foreground before blobs are found (towers animate; bars and the
#: stopwatch ride on units and would make every blob taller).
EXCLUDE_TYPES = (TYPE_TOWER, TYPE_UI, TYPE_PAD)

_SENTINEL = 32767                          # int16 stand-in for a masked sample (sorts last)


class ScanError(RuntimeError):
    """The scan cannot run: no recording, an earlier scan in the way, bad settings."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def harvest_tracker_config(**over: Any) -> TrackerConfig:
    """The harvest tracker's settings: the live Tracker's gates, but position only (class_cost
    and team_cost 0), every sighting reported (min_hits 1), and a little more patience for a
    proposal that drops out for a few images (max_age 30 ticks, 1.5 s)."""
    base: dict[str, Any] = dict(class_cost=0.0, team_cost=0.0, min_hits=1, tentative_ttl_ticks=10.0,
                                coast_ticks=10.0, max_age_ticks=30.0, label_switch_frames=1)
    base.update(over)
    return TrackerConfig(**base)


@dataclass(frozen=True)
class ScanConfig:
    """Harvest scan settings (module doc). Sizes in tiles are measured at the board's centre
    through the geometry, so they hold at any frame size."""
    # the harvest detector pass
    low_conf: float = 0.05
    low_max_det: int = 300
    low_iou: float | None = None            # NMS IoU; None: the live DetectorConfig's
    # proposals
    prop_conf: float = 0.05
    group_iou: float = 0.5
    live_iou: float = 0.5
    blob_iou: float = 0.3
    blob_cover: float = 0.6
    # background
    bg_samples: int = 40
    bg_band_rows: int = 64
    bg_mask_conf: float = 0.15
    bg_mask_pad_tiles: float = 0.15
    epoch_s: float = 120.0
    min_epoch_s: float = 5.0                # a tower fall this close to an epoch edge starts none
    tower_fall_frames: int = 3
    # blobs
    blobs: bool = True
    fg_threshold: int = 30
    cell_tiles: float = 0.12
    cell_min_frac: float = 0.35
    blob_min_tiles: float = 0.12
    blob_max_tiles: float = 32.0
    exclude_conf: float = 0.2
    board_margin_mt: float = 0.0
    # the harvest tracker
    tracker: TrackerConfig = field(default_factory=harvest_tracker_config)

    def __post_init__(self) -> None:
        for name in ("low_conf", "prop_conf", "bg_mask_conf", "exclude_conf"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"ScanConfig.{name} must be in [0, 1]")
        for name in ("group_iou", "live_iou", "blob_iou", "blob_cover", "cell_min_frac"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"ScanConfig.{name} must be in (0, 1]")
        if self.low_iou is not None and not 0.0 < self.low_iou <= 1.0:
            raise ValueError("ScanConfig.low_iou must be in (0, 1]")
        for name in ("low_max_det", "bg_samples", "bg_band_rows", "tower_fall_frames"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"ScanConfig.{name} must be at least 1")
        if not 1 <= self.fg_threshold <= 255:
            raise ValueError("ScanConfig.fg_threshold must be in [1, 255]")
        if not self.cell_tiles > 0 or self.bg_mask_pad_tiles < 0:
            raise ValueError("ScanConfig.cell_tiles must be > 0 and bg_mask_pad_tiles >= 0")
        if not 0.0 <= self.blob_min_tiles < self.blob_max_tiles:
            raise ValueError("ScanConfig needs 0 <= blob_min_tiles < blob_max_tiles")
        if self.epoch_s < 0 or self.min_epoch_s < 0:
            raise ValueError("ScanConfig.epoch_s and min_epoch_s must be >= 0")


# ---------------------------------------------------------------------------
# boxes, blobs, proposals
# ---------------------------------------------------------------------------

def _r1(v: float) -> float:
    return round(float(v), 1)


def _r3(v: float) -> float:
    return round(float(v), 3)


def side_of(p_enemy: float) -> int | None:
    """A merged live detection's side: enemy above P(enemy) 0.5, ally below, unknown at 0.5."""
    p = float(p_enemy)
    return 1 if p > 0.5 else 0 if p < 0.5 else None


def box_area(b: Sequence[float]) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def box_inter(a: Sequence[float], b: Sequence[float]) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return float(ix * iy) if ix > 0 and iy > 0 else 0.0


@dataclass(slots=True)
class RawBox:
    """One harvest detector box in frame pixels (RAW_FIELDS)."""
    cls: str
    name: str
    side: int | None
    conf: float
    box: tuple[float, float, float, float]
    type: str = TYPE_UNIT
    form: int = FORM_BASE
    x_mt: float = 0.0
    y_mt: float = 0.0

    def to_row(self) -> list[Any]:
        x1, y1, x2, y2 = self.box
        return [self.cls, self.name, self.side, _r3(self.conf), _r1(x1), _r1(y1), _r1(x2), _r1(y2),
                self.type, int(self.form), int(round(self.x_mt)), int(round(self.y_mt))]

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "RawBox":
        c, name, side, conf, x1, y1, x2, y2, typ, form, x, y = row
        return cls(str(c), str(name), None if side is None else int(side), float(conf),
                   (float(x1), float(y1), float(x2), float(y2)), str(typ), int(form), float(x), float(y))


@dataclass(slots=True)
class LiveDet:
    """One live detection (LIVE_FIELDS) and the live track it went to (uid; seg: the live
    tracker segment it belongs to)."""
    uid: int | None
    cls: str
    type: str
    conf: float
    p_enemy: float
    form: int
    box: tuple[float, float, float, float]
    seg: int = 0

    def to_row(self) -> list[Any]:
        x1, y1, x2, y2 = self.box
        return [self.uid, self.cls, self.type, _r3(self.conf), _r3(self.p_enemy), int(self.form),
                _r1(x1), _r1(y1), _r1(x2), _r1(y2)]

    @classmethod
    def from_row(cls, row: Sequence[Any], seg: int = 0) -> "LiveDet":
        uid, c, typ, conf, p, form, x1, y1, x2, y2 = row
        return cls(None if uid is None else int(uid), str(c), str(typ), float(conf), float(p), int(form),
                   (float(x1), float(y1), float(x2), float(y2)), int(seg))


@dataclass(slots=True)
class Blob:
    """A connected foreground region: its box in frame pixels and its foreground pixels."""
    box: tuple[float, float, float, float]
    area_px: int
    cells: int = 0

    def to_row(self) -> list[Any]:
        x1, y1, x2, y2 = self.box
        return [_r1(x1), _r1(y1), _r1(x2), _r1(y2), int(self.area_px)]

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "Blob":
        x1, y1, x2, y2, area = row
        return cls((float(x1), float(y1), float(x2), float(y2)), int(area))


@dataclass
class Proposal:
    """One place a unit or spell may be on one image (module doc, step 4).

    src: "det" (harvest boxes), "live" (a live detection no harvest box matched), "blob", with
    "+blob" when a blob was attached. cands: (class, side, best score) of every harvest box of
    the group (and the live detection's own read), strongest first. live_uid / live_seg: the
    live track of the live detection that joined; h: the harvest track (HarvestTracker)."""
    box: tuple[float, float, float, float]
    src: str
    type: str
    conf: float
    cands: list[tuple[str, int | None, float]] = field(default_factory=list)
    x_mt: float = 0.0
    y_mt: float = 0.0
    blob: int | None = None
    live_uid: int | None = None
    live_seg: int | None = None
    live_cls: str | None = None
    h: int | None = None

    @property
    def label(self) -> str | None:
        """The strongest candidate's class, else the live class, else None (a bare blob)."""
        if self.cands:
            return self.cands[0][0]
        return self.live_cls

    def to_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.box
        return {"box": [_r1(x1), _r1(y1), _r1(x2), _r1(y2)], "src": self.src, "type": self.type,
                "conf": _r3(self.conf), "cands": [[c, s, _r3(v)] for c, s, v in self.cands],
                "blob": self.blob,
                "live": None if self.live_uid is None else [self.live_seg, self.live_uid],
                "live_cls": self.live_cls, "x": int(round(self.x_mt)), "y": int(round(self.y_mt)),
                "h": self.h}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Proposal":
        live = d.get("live")
        return cls(tuple(float(v) for v in d["box"]), str(d["src"]), str(d.get("type", TYPE_UNIT)),
                   float(d.get("conf", 0.0)),
                   [(str(c), None if s is None else int(s), float(v)) for c, s, v in d.get("cands", [])],
                   float(d.get("x", 0.0)), float(d.get("y", 0.0)), d.get("blob"),
                   None if live is None else int(live[1]), None if live is None else int(live[0]),
                   d.get("live_cls"), d.get("h"))


@dataclass
class _Group:
    anchor_box: tuple[float, float, float, float]
    anchor_type: str
    anchor_conf: float
    x_mt: float
    y_mt: float
    src: str
    cands: dict[tuple[str, int | None], float] = field(default_factory=dict)
    live: LiveDet | None = None
    blob: int | None = None
    blob_iou: float = -1.0

    def add_cand(self, cls: str, side: int | None, conf: float) -> None:
        key = (str(cls), side)
        if conf > self.cands.get(key, -1.0):
            self.cands[key] = float(conf)


GroundFn = Callable[[Sequence[float], str, str | None], tuple[float, float]]


def pixel_ground(box: Sequence[float], typ: str = TYPE_UNIT, cls: str | None = None) -> tuple[float, float]:
    """A box's bottom centre in pixels: merge_proposals' stand-in for a geometry (tests)."""
    return (float(box[0]) + float(box[2])) / 2.0, float(box[3])


def merge_proposals(raw: Sequence[RawBox], blobs: Sequence[Blob] = (), live: Sequence[LiveDet] = (), *,
                    config: ScanConfig | None = None, ground: GroundFn | None = None) -> list[Proposal]:
    """One image's proposals from its harvest boxes, blobs and live detections (module doc,
    step 4). ground(box, type, cls) -> world (x, y) places the live-only and blob proposals
    (default pixel_ground: the box's bottom centre in pixels). Order: the harvest groups
    strongest first, then live-only proposals, then bare blobs."""
    cfg = config or ScanConfig()
    place = ground if ground is not None else pixel_ground
    boxes = [b for b in raw if b.type in PROPOSAL_TYPES and b.conf >= cfg.prop_conf]
    groups: list[_Group] = []
    for i in sorted(range(len(boxes)), key=lambda k: (-boxes[k].conf, k)):
        b = boxes[i]
        best, best_iou = None, 0.0
        for g in groups:
            iou = box_iou(b.box, g.anchor_box)
            if iou >= cfg.group_iou and iou > best_iou:
                best, best_iou = g, iou
        if best is None:
            best = _Group(tuple(b.box), b.type, b.conf, b.x_mt, b.y_mt, "det")
            groups.append(best)
        best.add_cand(b.cls, b.side, b.conf)

    # live detections: each to the group it overlaps most, one per group
    lives = [d for d in live if d.type not in EXCLUDE_TYPES and d.cls not in BAR_CLASSES]
    pairs = sorted(((box_iou(d.box, g.anchor_box), d.conf, -j, -gi) for j, d in enumerate(lives)
                    for gi, g in enumerate(groups)), reverse=True)     # best IoU, then score, then order
    taken_live: set[int] = set()
    for iou, _conf, neg_j, neg_gi in pairs:
        j, gi = -neg_j, -neg_gi
        if iou < cfg.live_iou:
            break
        if j in taken_live or groups[gi].live is not None:
            continue
        groups[gi].live = lives[j]
        taken_live.add(j)
    for j, d in enumerate(lives):
        if j in taken_live:
            continue
        x, y = place(d.box, d.type, d.cls)
        g = _Group(tuple(d.box), d.type, d.conf, float(x), float(y), "live", live=d)
        g.add_cand(d.cls, side_of(d.p_enemy), d.conf)
        groups.append(g)

    # blobs: attached to the proposal they overlap most, else proposals of their own
    loose: list[int] = []
    for j, bl in enumerate(blobs):
        a_blob = box_area(bl.box)
        best_g, best_iou, explained = None, -1.0, False
        for g in groups:
            inter = box_inter(bl.box, g.anchor_box)
            if inter <= 0.0:
                continue
            a_g = box_area(g.anchor_box)
            iou = inter / max(a_blob + a_g - inter, 1e-9)
            ok = (iou >= cfg.blob_iou or (a_blob > 0 and inter / a_blob >= cfg.blob_cover)
                  or (a_g > 0 and inter / a_g >= cfg.blob_cover))
            if ok:
                explained = True
                if iou > best_iou:
                    best_g, best_iou = g, iou
        if not explained:
            loose.append(j)
        elif best_g is not None and best_iou > best_g.blob_iou:
            best_g.blob, best_g.blob_iou = j, best_iou

    out: list[Proposal] = []
    for g in groups:
        cands = sorted(((c, s, v) for (c, s), v in g.cands.items()), key=lambda x: (-x[2], x[0], -1 if x[1] is None else x[1]))
        p = Proposal(g.anchor_box, g.src + ("+blob" if g.blob is not None else ""), g.anchor_type,
                     float(g.anchor_conf), cands, g.x_mt, g.y_mt, g.blob)
        if g.live is not None:
            p.live_uid, p.live_seg, p.live_cls = g.live.uid, g.live.seg, g.live.cls
        out.append(p)
    for j in loose:
        bl = blobs[j]
        x, y = place(bl.box, BLOB_LABEL, None)
        out.append(Proposal(tuple(bl.box), "blob", BLOB_LABEL, 0.0, [], float(x), float(y), j))
    return out


# ---------------------------------------------------------------------------
# the harvest detector pass
# ---------------------------------------------------------------------------

def _as_array(value: Any, tail: tuple[int, ...] = ()) -> np.ndarray:
    """A torch tensor, numpy array or list -> float64 numpy array of shape (-1, *tail)."""
    if value is None:
        return np.zeros((0,) + tail)
    for attr in ("detach", "cpu", "numpy"):
        if hasattr(value, attr):
            value = getattr(value, attr)()
    arr = np.asarray(value, dtype=np.float64)
    return arr.reshape((-1,) + tail) if arr.size else np.zeros((0,) + tail)


def _model_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None)
    if names is None:
        raise ScanError("the detector model has no class names (model.names)")
    if isinstance(names, Mapping):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def _cls_name(base: str, lab: Any) -> str:
    """What the live detector reports as Detection.cls (scripts/vision/detector.py): the class
    for units and towers, the card for spells, else the detector name."""
    if lab.type in (TYPE_UNIT, TYPE_TOWER) and lab.name:
        return str(lab.name)
    if lab.type == TYPE_SPELL and lab.card:
        return str(lab.card)
    return str(base)


class LowConfPass:
    """The detector's weights at a low score (module doc, step 2): every box of the crop the
    live detector sees (crop_rect_for), each with the side its own class name carries.

    model: anything with `.names` and ultralytics' `.predict(source=bgr, imgsz=, conf=, iou=,
    max_det=, device=, half=, verbose=)` (a YOLO, or a fake in tests). config: the live
    DetectorConfig whose crop, imgsz, NMS IoU and board margins are kept (conf, max_det and
    cross_class_iou are replaced). from_detector(det) shares a loaded Detector's model.
    ultralytics updates a cached predictor's arguments on every call, so the live pass and
    this one can alternate on one model."""

    def __init__(self, model: Any, *, geometry: ArenaGeometry | None = None, table: Any = None,
                 config: DetectorConfig | None = None, device: str = "cpu", half: bool | None = None,
                 conf: float = 0.05, max_det: int = 300, iou: float | None = None):
        base = config if config is not None else DetectorConfig()
        over: dict[str, Any] = {"conf": float(conf), "max_det": int(max_det), "cross_class_iou": None}
        if iou is not None:
            over["iou"] = float(iou)
        self.config = dataclasses.replace(base, **over)
        self.model = model
        self.geometry = geometry if geometry is not None else ArenaGeometry.default()
        self.table = load_class_table(table)
        self.device = str(device)
        if half is None:
            half = (self.device != "cpu") if self.config.half is None else bool(self.config.half)
        self.half = bool(half)
        self._names: dict[int, tuple[str, int | None, Any]] = {}
        for idx, name in sorted(_model_names(model).items()):
            base_name, side = split_side(name)
            self._names[idx] = (base_name, side, resolve_label(self.table, base_name))
        self.frames = 0
        self.boxes = 0
        self.dropped = 0
        self.last_ms = 0.0

    @classmethod
    def from_detector(cls, det: Any, *, conf: float = 0.05, max_det: int = 300,
                      iou: float | None = None) -> "LowConfPass":
        """The same model, geometry, class table, crop and device as a loaded Detector."""
        return cls(det.model, geometry=det.geometry, table=det.table, config=det.config,
                   device=getattr(det, "device", "cpu"), half=getattr(det, "half", None),
                   conf=conf, max_det=max_det, iou=iou)

    def crop_rect(self, size: Sequence[int]) -> tuple[int, int, int, int]:
        return crop_rect_for(self.geometry, self.config, size)

    def _flying(self, lab: Any) -> bool:
        try:
            return bool(self.table.classes[lab.class_id].form(lab.form).flying)
        except (IndexError, KeyError, TypeError, AttributeError):
            return False

    def detect(self, rgb: np.ndarray) -> list[RawBox]:
        """Every box of one HxWx3 RGB frame, strongest first, in frame pixels."""
        t0 = time.perf_counter()
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] < 3:
            raise ScanError(f"expected an HxWx3 RGB frame, got shape {img.shape}")
        size = check_frame_size(img.shape[1], img.shape[0])
        x0, y0, x1, y1 = self.crop_rect(size)
        crop = np.ascontiguousarray(img[y0:y1, x0:x1, 2::-1])         # RGB -> BGR, as the live detector
        cfg = self.config
        out = self.model.predict(source=crop, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, max_det=cfg.max_det,
                                 device=self.device, half=self.half, verbose=False)
        res = out[0] if isinstance(out, (list, tuple)) else out
        boxes = getattr(res, "boxes", None) if res is not None else None
        xyxy = _as_array(getattr(boxes, "xyxy", None), (4,))
        conf = _as_array(getattr(boxes, "conf", None))
        cls = _as_array(getattr(boxes, "cls", None))
        if not (len(xyxy) == len(conf) == len(cls)):
            raise ScanError(f"model returned {len(xyxy)} boxes, {len(conf)} scores, {len(cls)} classes")
        found: list[RawBox] = []
        for (bx1, by1, bx2, by2), c, ci in zip(xyxy, conf, cls):
            idx = int(round(float(ci)))
            if idx not in self._names:
                name = f"class_{idx}"
                self._names[idx] = (name, None, resolve_label(self.table, name))
            base, side, lab = self._names[idx]
            box = (float(bx1) + x0, float(by1) + y0, float(bx2) + x0, float(by2) + y0)
            name = _cls_name(base, lab)
            form = int(lab.form or FORM_BASE) if lab.type in (TYPE_UNIT, TYPE_TOWER, TYPE_SPELL) else FORM_BASE
            if lab.type in (TYPE_UNIT, TYPE_TOWER):
                wx, wy = self.geometry.ground_point(box, size, cls=name, flying=self._flying(lab))
            else:
                wx, wy = self.geometry.px_to_world((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, size)
            margin = cfg.ui_margin_mt if lab.type in (TYPE_UI, TYPE_PAD) else cfg.board_margin_mt
            if not in_arena(float(wx), float(wy), margin):
                self.dropped += 1
                continue
            found.append(RawBox(name, base, side, float(c), box, str(lab.type), form, float(wx), float(wy)))
        found.sort(key=lambda b: -b.conf)
        self.frames += 1
        self.boxes += len(found)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return found


# ---------------------------------------------------------------------------
# background median and blobs
# ---------------------------------------------------------------------------

def masked_median(stack: np.ndarray, masks: np.ndarray | None = None, band_rows: int = 64) -> np.ndarray:
    """Per-pixel median over N images (stack: N x H x W x C uint8) of the samples not masked
    (masks: N x H x W, True = do not count), computed band_rows rows at a time. An even count
    takes the mean of the two middle values, rounded half up; a pixel masked in every sample
    takes the plain median of all of them. Returns H x W x C uint8."""
    s = np.asarray(stack)
    if s.ndim != 4 or s.shape[0] < 1:
        raise ValueError(f"stack must be N x H x W x C with N >= 1, got {s.shape}")
    n, h, w, c = s.shape
    if masks is None:
        m_all = np.zeros((n, h, w), dtype=bool)
    else:
        m_all = np.asarray(masks, dtype=bool)
        if m_all.shape != (n, h, w):
            raise ValueError(f"masks must be {(n, h, w)}, got {m_all.shape}")
    out = np.empty((h, w, c), dtype=np.uint8)
    step = max(1, int(band_rows))
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        band = s[:, y0:y1].astype(np.int16)
        mb = m_all[:, y0:y1]
        valid = (~mb).sum(axis=0)                                  # rows x W
        vals = np.where(mb[..., None], np.int16(_SENTINEL), band)
        vals.sort(axis=0)
        lo = (np.maximum(valid, 1) - 1) // 2
        hi = np.maximum(valid, 1) // 2
        idx_lo = np.broadcast_to(lo[None, :, :, None], (1,) + vals.shape[1:])
        idx_hi = np.broadcast_to(hi[None, :, :, None], (1,) + vals.shape[1:])
        a = np.take_along_axis(vals, idx_lo, axis=0)[0].astype(np.int32)
        b = np.take_along_axis(vals, idx_hi, axis=0)[0].astype(np.int32)
        med = (a + b + 1) // 2
        none = valid == 0
        if none.any():
            plain = np.sort(band, axis=0)
            pa = plain[(n - 1) // 2].astype(np.int32)
            pb = plain[n // 2].astype(np.int32)
            med = np.where(none[..., None], (pa + pb + 1) // 2, med)
        out[y0:y1] = np.clip(med, 0, 255).astype(np.uint8)
    return out


def fit_background(images: Sequence[np.ndarray], masks: Sequence[np.ndarray] | None = None, *,
                   rect: Sequence[int] | None = None, band_rows: int = 64) -> np.ndarray:
    """A background image from sample frames (all one size): the masked median inside `rect`
    (x0, y0, x1, y1; default the whole frame), the first sample outside it."""
    if not images:
        raise ValueError("fit_background needs at least one image")
    first = np.asarray(images[0])[..., :3]
    h, w = first.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if rect is None else (int(v) for v in rect)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    out = np.ascontiguousarray(first.copy())
    if x1 <= x0 or y1 <= y0:
        return out
    stack = np.stack([np.asarray(im)[y0:y1, x0:x1, :3] for im in images])
    mstack = None if masks is None else np.stack([np.asarray(m, dtype=bool)[y0:y1, x0:x1] for m in masks])
    out[y0:y1, x0:x1] = masked_median(stack, mstack, band_rows)
    return out


def box_mask(shape: Sequence[int], boxes: Iterable[Sequence[float]], pad_px: float = 0.0) -> np.ndarray:
    """H x W bool, True inside any (x1, y1, x2, y2) box grown by pad_px (whole pixels touched)."""
    h, w = int(shape[0]), int(shape[1])
    m = np.zeros((h, w), dtype=bool)
    for b in boxes:
        x0 = max(0, int(math.floor(float(b[0]) - pad_px)))
        y0 = max(0, int(math.floor(float(b[1]) - pad_px)))
        x1 = min(w, int(math.ceil(float(b[2]) + pad_px)))
        y1 = min(h, int(math.ceil(float(b[3]) + pad_px)))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def foreground_mask(rgb: np.ndarray, background: np.ndarray, *, threshold: int = 30,
                    region: np.ndarray | None = None, exclude: Iterable[Sequence[float]] = ()) -> np.ndarray:
    """H x W bool: the largest channel differs from the background by more than `threshold`,
    inside `region` (H x W bool), outside the `exclude` boxes."""
    a = np.asarray(rgb)[..., :3].astype(np.int16)
    b = np.asarray(background)[..., :3].astype(np.int16)
    if a.shape != b.shape:
        raise ValueError(f"image {a.shape} and background {b.shape} differ in size")
    fg = np.abs(a - b).max(axis=2) > int(threshold)
    if region is not None:
        fg &= np.asarray(region, dtype=bool)
    ex = list(exclude)
    if ex:
        fg &= ~box_mask(fg.shape, ex)
    return fg


def _dilate(a: np.ndarray) -> np.ndarray:
    """3x3 (8-neighbour) binary dilation; outside the array counts as False."""
    out = a.copy()
    out[1:] |= a[:-1]
    out[:-1] |= a[1:]
    tmp = out.copy()
    out[:, 1:] |= tmp[:, :-1]
    out[:, :-1] |= tmp[:, 1:]
    return out


def binary_close(a: np.ndarray, steps: int = 1) -> np.ndarray:
    """Dilate `steps` times, then erode as often (outside the array counts as set for the
    erosion, so the border never eats a blob)."""
    out = np.asarray(a, dtype=bool)
    for _ in range(max(0, int(steps))):
        out = _dilate(out)
    for _ in range(max(0, int(steps))):
        out = ~_dilate(~out)
    return out


def _label_bfs(a: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = a.shape
    grid = a.tolist()
    labels = np.zeros((h, w), dtype=np.int32)
    seen = [[0] * w for _ in range(h)]
    n = 0
    ys, xs = np.nonzero(a)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y][x]:
            continue
        n += 1
        seen[y][x] = n
        stack = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            for ny in (cy - 1, cy, cy + 1):
                if ny < 0 or ny >= h:
                    continue
                row, srow = grid[ny], seen[ny]
                for nx in (cx - 1, cx, cx + 1):
                    if 0 <= nx < w and row[nx] and not srow[nx]:
                        srow[nx] = n
                        stack.append((ny, nx))
    if n:
        labels[:] = np.asarray(seen, dtype=np.int32)
    return labels, n


def label_components(a: np.ndarray, *, use_scipy: bool = True) -> tuple[np.ndarray, int]:
    """8-connected components of a 2-D bool array: (labels, count), labels 1..count in raster
    order of each component's first cell. scipy.ndimage.label when it imports (and use_scipy),
    else a breadth-first fill that numbers the same way."""
    a = np.asarray(a, dtype=bool)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got {a.shape}")
    if use_scipy:
        try:
            from scipy import ndimage
        except Exception:                           # ImportError, or a scipy for another numpy
            ndimage = None
        if ndimage is not None:
            labels, n = ndimage.label(a, structure=np.ones((3, 3), dtype=bool))
            return labels.astype(np.int32), int(n)
    return _label_bfs(a)


def find_blobs(mask: np.ndarray, *, cell: int = 4, min_frac: float = 0.35, min_area_px: float = 0.0,
               max_area_px: float = math.inf, close: int = 1) -> list[Blob]:
    """Foreground blobs of an H x W bool mask: cells of `cell` x `cell` pixels at least
    `min_frac` foreground, closed by `close` cells, joined 8-connected; a blob's box covers its
    cells (clipped to the mask) and its area counts the foreground pixels in them. Blobs whose
    area is outside [min_area_px, max_area_px] are dropped. Raster order."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got {m.shape}")
    h, w = m.shape
    c = max(1, int(cell))
    hc, wc = -(-h // c), -(-w // c)
    padded = np.zeros((hc * c, wc * c), dtype=bool)
    padded[:h, :w] = m
    counts = padded.reshape(hc, c, wc, c).sum(axis=(1, 3))
    occ = counts >= max(1.0, float(min_frac) * c * c)
    if close:
        occ = binary_close(occ, close)
    labels, n = label_components(occ)
    out: list[Blob] = []
    if not n:
        return out
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]
    starts = np.searchsorted(sorted_labels, np.arange(1, n + 1), side="left")
    ends = np.searchsorted(sorted_labels, np.arange(1, n + 1), side="right")
    cnt = counts.ravel()
    for k in range(n):
        cells = order[starts[k]:ends[k]]
        ys, xs = cells // wc, cells % wc
        area = int(cnt[cells].sum())
        if area < min_area_px or area > max_area_px:
            continue
        box = (float(xs.min() * c), float(ys.min() * c), float(min(w, (xs.max() + 1) * c)),
               float(min(h, (ys.max() + 1) * c)))
        out.append(Blob(box, area, int(len(cells))))
    return out


def tile_px(geometry: ArenaGeometry, size: Sequence[int]) -> tuple[float, float]:
    """(width, height) in pixels of one tile at the board's centre, for a frame of `size`."""
    cx, cy = WIDTH_MT / 2.0, HEIGHT_MT / 2.0
    x0, y0 = geometry.world_to_px(cx - 500.0, cy - 500.0, size)
    x1, _ = geometry.world_to_px(cx + 500.0, cy - 500.0, size)
    _, y1 = geometry.world_to_px(cx - 500.0, cy + 500.0, size)
    return abs(float(x1) - float(x0)), abs(float(y1) - float(y0))


def board_mask(geometry: ArenaGeometry, size: Sequence[int], margin_mt: float = 0.0) -> np.ndarray:
    """H x W bool: pixel centres on the 18x32 board grown by margin_mt."""
    w, h = check_frame_size(size[0], size[1])
    ys, xs = np.mgrid[0:h, 0:w]
    wx, wy = geometry.px_to_world(xs + 0.5, ys + 0.5, (w, h))
    m = float(margin_mt)
    return (wx >= -m) & (wx <= WIDTH_MT + m) & (wy >= -m) & (wy <= HEIGHT_MT + m)


@dataclass
class SizeInfo:
    """Per frame size: the detector crop, the blob region (board and crop), tile and cell size."""
    size: tuple[int, int]
    rect: tuple[int, int, int, int]
    region: np.ndarray
    tile_w: float
    tile_h: float
    cell: int

    @property
    def tile_area(self) -> float:
        return self.tile_w * self.tile_h


class SizeCache:
    """SizeInfo per frame size for one geometry, detector config and ScanConfig."""

    def __init__(self, geometry: ArenaGeometry, det_config: DetectorConfig | None, config: ScanConfig):
        self.geometry = geometry
        self.det_config = det_config
        self.config = config
        self._cache: dict[tuple[int, int], SizeInfo] = {}

    def get(self, size: Sequence[int]) -> SizeInfo:
        key = (int(size[0]), int(size[1]))
        info = self._cache.get(key)
        if info is None:
            rect = crop_rect_for(self.geometry, self.det_config, key)
            region = board_mask(self.geometry, key, self.config.board_margin_mt)
            inside = np.zeros_like(region)
            inside[rect[1]:rect[3], rect[0]:rect[2]] = True
            tw, th = tile_px(self.geometry, key)
            cell = max(1, int(round(self.config.cell_tiles * min(tw, th))))
            info = self._cache[key] = SizeInfo(key, rect, region & inside, tw, th, cell)
        return info


# ---------------------------------------------------------------------------
# background epochs
# ---------------------------------------------------------------------------

@dataclass
class FrameEntry:
    """What the background needs of one battle-screen image (kept in memory during a scan)."""
    n: int
    t: float
    hk: int
    path: Path
    size: tuple[int, int]
    mask_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    towers: dict[str, int | None] = field(default_factory=dict)


@dataclass
class Epoch:
    """A stretch of one battle with one background (module doc, BACKGROUND)."""
    id: str
    hk: int
    index: int
    t0: float
    t1: float
    reason: str
    members: list[int] = field(default_factory=list)
    samples: list[int] = field(default_factory=list)
    image: str | None = None
    fitted: int = 0                          # samples that went into the median

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "hk": self.hk, "index": self.index, "t0": round(self.t0, 4),
                "t1": None if math.isinf(self.t1) else round(self.t1, 4), "reason": self.reason,
                "frames": len(self.members), "first_n": self.members[0] if self.members else None,
                "last_n": self.members[-1] if self.members else None, "samples": list(self.samples),
                "fitted": self.fitted, "image": self.image}


def tower_falls(entries: Sequence[FrameEntry], frames_needed: int = 3) -> list[tuple[float, str]]:
    """(time of the first 0 reading, tower key) for each tower the HUD read standing and then
    at 0 hit points on `frames_needed` frames in a row (unread frames neither count nor break)."""
    state: dict[str, list[Any]] = {}            # key -> [alive, zeros, first_zero_t, done]
    out: list[tuple[float, str]] = []
    for e in entries:
        for key, hp in (e.towers or {}).items():
            st = state.setdefault(key, [False, 0, None, False])
            if st[3] or hp is None:
                continue
            if int(hp) > 0:
                st[0], st[1], st[2] = True, 0, None
            elif st[0]:
                if st[1] == 0:
                    st[2] = e.t
                st[1] += 1
                if st[1] >= max(1, int(frames_needed)):
                    st[3] = True
                    out.append((float(st[2]), str(key)))
    out.sort()
    return out


def _evenly(items: Sequence[int], k: int) -> list[int]:
    if len(items) <= k:
        return list(items)
    if k == 1:
        return [items[len(items) // 2]]
    picks = sorted({int(round(i * (len(items) - 1) / (k - 1))) for i in range(k)})
    return [items[i] for i in picks]


def plan_epochs(entries: Sequence[FrameEntry], config: ScanConfig | None = None) -> list[Epoch]:
    """The background epochs of a scan's battle-screen frames: per harvest battle, cut at each
    tower fall (not within min_epoch_s of an edge) and into equal parts no longer than epoch_s
    (0: no length limit); each with up to bg_samples evenly spread sample frames."""
    cfg = config or ScanConfig()
    by_hk: dict[int, list[FrameEntry]] = {}
    for e in sorted(entries, key=lambda x: x.n):
        by_hk.setdefault(int(e.hk), []).append(e)
    epochs: list[Epoch] = []
    for hk in sorted(by_hk):
        es = by_hk[hk]
        t_first, t_last = min(e.t for e in es), max(e.t for e in es)
        cuts: list[tuple[float, str]] = [(t_first, "start")]
        for t, key in tower_falls(es, cfg.tower_fall_frames):
            if t - cuts[-1][0] >= cfg.min_epoch_s and t_last - t >= cfg.min_epoch_s:
                cuts.append((t, f"tower {key}"))
        starts: list[tuple[float, str]] = []
        for j, (a, why) in enumerate(cuts):
            b = cuts[j + 1][0] if j + 1 < len(cuts) else t_last
            parts = max(1, int(math.ceil((b - a) / cfg.epoch_s - 1e-9))) if cfg.epoch_s > 0 and b > a else 1
            for part in range(parts):
                starts.append((a + (b - a) * part / parts, why if part == 0 else "length"))
        k = 0
        for j, (lo, why) in enumerate(starts):
            hi = starts[j + 1][0] if j + 1 < len(starts) else math.inf
            members = [e.n for e in es if lo <= e.t < hi]
            if not members:
                continue
            ep = Epoch(f"b{hk}e{k}", hk, k, lo, hi, why, members, _evenly(members, cfg.bg_samples))
            epochs.append(ep)
            k += 1
    return epochs


# ---------------------------------------------------------------------------
# the harvest tracker
# ---------------------------------------------------------------------------

def _side_name(side: int | None) -> str:
    return "ally" if side == 0 else "enemy" if side == 1 else "any"


@dataclass
class HarvestTrack:
    """One harvest track: its proposals as (scan record n, proposal k) and what they held."""
    h: int
    hk: int | None
    frames: list[tuple[int, int]] = field(default_factory=list)
    first_t: float | None = None
    last_t: float | None = None
    src: Counter = field(default_factory=Counter)
    cands: dict[str, list[float]] = field(default_factory=dict)       # "cls@side" -> [sum, max, frames]
    live: Counter = field(default_factory=Counter)                     # "seg:uid" -> frames
    live_cls: Counter = field(default_factory=Counter)
    birth: tuple[float, float] | None = None
    last: tuple[float, float] | None = None
    max_conf: float = 0.0

    def add(self, n: int, k: int, t: float, p: Proposal) -> None:
        self.frames.append((int(n), int(k)))
        if self.first_t is None:
            self.first_t, self.birth = float(t), (float(p.x_mt), float(p.y_mt))
        self.last_t, self.last = float(t), (float(p.x_mt), float(p.y_mt))
        self.src[p.src] += 1
        self.max_conf = max(self.max_conf, float(p.conf))
        for cls, side, conf in p.cands:
            key = f"{cls}@{_side_name(side)}"
            agg = self.cands.setdefault(key, [0.0, 0.0, 0])
            agg[0] += float(conf)
            agg[1] = max(agg[1], float(conf))
            agg[2] += 1
        if p.live_uid is not None:
            self.live[f"{p.live_seg}:{p.live_uid}"] += 1
            if p.live_cls:
                self.live_cls[p.live_cls] += 1

    @property
    def n(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict[str, Any]:
        cands = {k: {"sum": _r3(v[0]), "max": _r3(v[1]), "frames": int(v[2])}
                 for k, v in sorted(self.cands.items(), key=lambda kv: (-kv[1][0], kv[0]))}
        return {"h": self.h, "hk": self.hk, "n": self.n,
                "first_n": self.frames[0][0] if self.frames else None,
                "last_n": self.frames[-1][0] if self.frames else None,
                "first_t": None if self.first_t is None else round(self.first_t, 4),
                "last_t": None if self.last_t is None else round(self.last_t, 4),
                "frames": [list(f) for f in self.frames], "src": dict(sorted(self.src.items())),
                "max_conf": _r3(self.max_conf), "cands": cands, "live": dict(self.live.most_common()),
                "live_cls": dict(self.live_cls.most_common()),
                "birth": None if self.birth is None else [int(round(v)) for v in self.birth],
                "last": None if self.last is None else [int(round(v)) for v in self.last]}


class HarvestTracker:
    """Proposals followed by position only (module doc, step 5): a Tracker with
    harvest_tracker_config(); each update's Tracker.last_assignment gives every proposal its
    harvest track id h (unique over this object's life, across reset())."""

    def __init__(self, config: TrackerConfig | None = None, *, solver: str = "auto"):
        self.tracker = Tracker(config or harvest_tracker_config(), solver=solver)
        self.tracks: dict[int, HarvestTrack] = {}
        self._uid_to_h: dict[int, int] = {}
        self._next_h = 1
        self.hk: int | None = None
        self._t0: float | None = None

    def reset(self, hk: int | None = None) -> None:
        """A new battle: the tracker forgets every track; ids keep counting."""
        self.tracker.reset()
        self._uid_to_h = {}
        self.hk = hk
        self._t0 = None

    def tick_of(self, t: float) -> int:
        if self._t0 is None:
            self._t0 = float(t)
        return int(round((float(t) - self._t0) * TICKS_PER_S))

    def update(self, props: Sequence[Proposal], *, t: float, n: int) -> list[int]:
        """One image's proposals (scan record n, capture time t); sets and returns their h."""
        obs = [Observation(p.label or BLOB_LABEL, float(p.x_mt), float(p.y_mt), max(float(p.conf), 0.05))
               for p in props]
        self.tracker.update(obs, self.tick_of(t))
        out: list[int] = []
        for k, (p, uid) in enumerate(zip(props, self.tracker.last_assignment)):
            h = self._uid_to_h.get(uid)
            if h is None:
                h = self._uid_to_h[uid] = self._next_h
                self._next_h += 1
                self.tracks[h] = HarvestTrack(h, self.hk)
            p.h = h
            self.tracks[h].add(n, k, t, p)
            out.append(h)
        return out


# ---------------------------------------------------------------------------
# the live chain: last_assignment back to the detections
# ---------------------------------------------------------------------------

class RecordingTracker(Tracker):
    """The live Tracker, unchanged, that also keeps the observations of its last update
    (last_observations, in the order last_assignment lists their uids) and counts updates."""

    def reset(self) -> None:
        super().reset()
        self.last_observations: list[Observation] = []
        self.updates = getattr(self, "updates", 0)

    def update(self, observations: Iterable[Observation], tick: float):
        obs = list(observations)
        self.last_observations = obs
        self.updates = getattr(self, "updates", 0) + 1
        return super().update(obs, tick)


def _obs_key(o: Observation) -> tuple:
    return (str(o.cls), float(o.x_mt), float(o.y_mt), float(o.conf), float(o.side_prob), int(o.form))


def match_observations(detections: Sequence[Any], observations: Sequence[Observation], geometry: Any,
                       size: Sequence[int] | None) -> list[int | None]:
    """For each observation, the index of the detection it was made from (None when none
    matches): the observation observations_from_detections makes of each detection is compared
    field by field, so the match is exact; equal detections are taken in order."""
    keys: dict[tuple, list[int]] = {}
    for i, d in enumerate(detections):
        o = observations_from_detections([d], geometry, size)[0]
        keys.setdefault(_obs_key(o), []).append(i)
    out: list[int | None] = []
    for o in observations:
        lst = keys.get(_obs_key(o))
        out.append(lst.pop(0) if lst else None)
    return out


def _types_of(step: Any, dets: Sequence[Any], names: Any) -> list[str]:
    labels = getattr(getattr(step, "det", None), "labels", None)
    if labels is not None and len(labels) == len(dets):
        return [str(getattr(lab, "type", TYPE_UNIT)) for lab in labels]
    out = []
    for d in dets:
        if d.cls in BAR_CLASSES:
            out.append(TYPE_UI)
        elif names is not None:
            out.append(str(names.canon(d.cls)[2]))
        else:
            out.append(TYPE_UNIT)
    return out


def live_uids(step: Any, tracker: Any, geometry: Any, types: Sequence[str], *,
              updated: bool) -> tuple[list[int | None], str]:
    """(uid per detection of step.vf.detections, how it was found): the live track each
    tracked detection went to, through the tracker's last_assignment (N05). With a
    RecordingTracker the observations are matched back exactly ("observations"); else the
    detections are taken in the order ReplayPerception feeds them, units then spells
    ("order", only when the counts agree); "none" when the tracker did not run this frame."""
    dets = list(step.vf.detections)
    uids: list[int | None] = [None] * len(dets)
    if not updated or tracker is None:
        return uids, "none"
    assignment = list(getattr(tracker, "last_assignment", None) or [])
    obs = getattr(tracker, "last_observations", None)
    if obs is not None:
        idx = match_observations(dets, obs, geometry, step.vf.frame_size)
        for j, i in enumerate(idx):
            if i is not None and j < len(assignment):
                uids[i] = int(assignment[j])
        return uids, "observations"
    tracked = [i for i, ty in enumerate(types) if ty == TYPE_UNIT and dets[i].cls not in BAR_CLASSES]
    tracked += [i for i, ty in enumerate(types) if ty == TYPE_SPELL]
    if len(tracked) != len(assignment):
        return uids, "mismatch"
    for i, uid in zip(tracked, assignment):
        uids[i] = int(uid)
    return uids, "order"


# ---------------------------------------------------------------------------
# ScanPerception: the live chain plus the harvest detector pass on the same image
# ---------------------------------------------------------------------------

def _frame_number(image: Path | None, n: int) -> int:
    if image is not None and Path(image).stem.isdigit():
        return int(Path(image).stem)
    return int(n)


class ScanPerception:
    """Wraps a tools/vision_replay.ReplayPerception (`live`) for vision_replay.Replay: every
    call Replay makes reaches `live` unchanged (attributes it does not define, such as tracker,
    deriver, ticks and names, are read from `live`), and each step also makes one scan record
    (module doc, pass 1): the live detections with their live uids, the live tracks with
    EventDeriver.how_of and their card, and on battle-screen images the harvest boxes of
    `low_pass` (a LowConfPass, or None for blobs only). A record is finished at the next step
    (or finish()), when Replay has updated its BattleState for the frame, and handed to `sink`.
    `segment` counts tracker resets as Replay.segment does, so (seg, uid) names a live track."""

    def __init__(self, live: Any, *, low_pass: LowConfPass | None = None, card_names: Mapping[int, str] | None = None,
                 sink: Callable[[dict], Any] | None = None):
        self.live = live
        self.low_pass = low_pass
        self.card_names = {int(k): str(v) for k, v in (card_names or {}).items()}
        self.sink = sink
        self.battle: Any = None
        self.source: Any = None
        self.session_dir: Path | None = None
        self.segment = 0
        self.errors: Counter = Counter()
        self.counts: Counter = Counter()
        self.low_ms: list[float] = []
        self._pending: dict | None = None
        self._n = 0

    def __getattr__(self, name: str) -> Any:
        if name == "live" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.live, name)

    def attach(self, *, battle: Any = None, source: Any = None, session_dir: str | Path | None = None,
               sink: Callable[[dict], Any] | None = None) -> None:
        """The Replay's BattleState, its source (for image_path) and where records go."""
        self.battle = battle
        self.source = source
        self.session_dir = None if session_dir is None else Path(session_dir)
        if sink is not None:
            self.sink = sink

    # -- what vision_replay.Replay calls ----------------------------------------
    def start_clock(self, t: float) -> None:
        self.live.start_clock(t)

    def tap_tick(self, clock_tick: Any, t: float) -> int | None:
        return self.live.tap_tick(clock_tick, t)

    def note_own_tap(self, card: int, tick: int, **kw: Any) -> None:
        self.live.note_own_tap(card, tick, **kw)

    def new_battle(self) -> None:
        self.live.new_battle()
        self.segment += 1

    def end_battle(self) -> None:
        self.live.end_battle()
        self.segment += 1

    def step(self, rgb: np.ndarray, t: float, stale_s: float = 0.0, *, recorded: Any = None,
             detect: bool = False) -> Any:
        self.flush()
        tracker = getattr(self.live, "tracker", None)
        before = getattr(tracker, "updates", None)
        step = self.live.step(rgb, t, stale_s, recorded=recorded, detect=detect)
        if before is not None:
            updated = getattr(tracker, "updates", None) != before
        else:                                       # ReplayPerception updates the tracker exactly then
            updated = tracker is not None and step.det_source is not None and step.track_tick is not None
        image = getattr(self.source, "image_path", None) if self.source is not None else None
        rec = self._head(step, image)
        try:
            self._fill(rec, step, rgb, updated)
        except Exception as exc:                    # a bad harvest pass must not lose the live step
            self.errors[f"record:{type(exc).__name__}"] += 1
            rec["error"] = f"{type(exc).__name__}: {exc}"
        self._pending = rec
        return step

    def _rel(self, image: Path | None) -> str | None:
        if image is None:
            return None
        p = Path(image)
        if self.session_dir is not None:
            try:
                return Path(os.path.relpath(p, self.session_dir)).as_posix()
            except ValueError:                      # another drive (Windows)
                pass
        return p.as_posix()

    def _head(self, step: Any, image: Path | None) -> dict[str, Any]:
        vf = step.vf
        n = self._n
        self._n += 1
        return {"n": n, "i": _frame_number(image, n), "image": self._rel(image), "t": round(float(vf.t_capture), 4),
                "screen": vf.screen, "size": None if vf.frame_size is None else [int(v) for v in vf.frame_size],
                "seg": self.segment, "tick": step.track_tick, "tick_src": step.tick_source}

    def record_for(self, step: Any, rgb: np.ndarray, *, image: Path | None = None, updated: bool = True) -> dict:
        """The scan record of one step (before flush adds battle and hk)."""
        rec = self._head(step, image)
        self._fill(rec, step, rgb, updated)
        return rec

    def _fill(self, rec: dict[str, Any], step: Any, rgb: np.ndarray, updated: bool) -> None:
        vf = step.vf
        self.counts["frames"] += 1
        if vf.screen != SCREEN_BATTLE:
            return
        self.counts["battle_frames"] += 1
        tracker = getattr(self.live, "tracker", None)
        names = getattr(self.live, "names", None)
        dets = list(vf.detections)
        types = _types_of(step, dets, names)
        uids, how = live_uids(step, tracker, getattr(self.live, "geometry", None), types, updated=updated)
        self.counts[f"live_assign_{how}"] += 1
        rec["live"] = [LiveDet(uid, d.cls, ty, float(d.conf), float(d.side_prob), int(d.form),
                               tuple(float(v) for v in d.box), self.segment).to_row()
                       for d, ty, uid in zip(dets, types, uids)]
        deriver = getattr(self.live, "deriver", None)
        tracks = []
        for tr in vf.tracks:
            row = {"uid": int(tr.uid), "cls": tr.cls, "team": int(tr.team), "form": int(tr.form),
                   "conf": _r3(tr.conf), "x": int(round(tr.x_mt)), "y": int(round(tr.y_mt)),
                   "first": int(tr.first_seen_tick), "last": int(tr.last_seen_tick),
                   "hp": None if tr.hp_frac is None else _r3(tr.hp_frac), "how": None, "card": None}
            try:
                if hasattr(deriver, "how_of"):
                    row["how"] = deriver.how_of(tr.uid)
                card = deriver.card_for(tr) if hasattr(deriver, "card_for") else None
                if card is not None:
                    row["card"] = int(card)
                    row["card_name"] = self.card_names.get(int(card))
            except Exception as exc:
                self.errors[f"deriver:{type(exc).__name__}"] += 1
            tracks.append(row)
        rec["tracks"] = tracks
        plays = []
        for p in vf.events.opp_plays:
            d = p.to_dict()
            d["card_name"] = "UNKNOWN" if int(p.card) < 0 else self.card_names.get(int(p.card))
            plays.append(d)
        rec["plays"] = plays
        rec["towers"] = {str(k): v.hp for k, v in vf.hud.towers.items()}
        if self.low_pass is not None:
            raw = self.low_pass.detect(rgb)
            self.low_ms.append(self.low_pass.last_ms)
            self.counts["raw_boxes"] += len(raw)
            rec["raw"] = [b.to_row() for b in raw]

    def flush(self) -> None:
        """Finish the pending record: its battle (BattleState after its frame) and harvest
        battle hk (battle-screen frames only: the battle, or the next one before a start)."""
        rec, self._pending = self._pending, None
        if rec is None:
            return
        in_battle = bool(getattr(self.battle, "in_battle", False))
        number = int(getattr(self.battle, "number", 0) or 0)
        rec["battle"] = number if in_battle else None
        rec["hk"] = (number if in_battle else number + 1) if rec.get("screen") == SCREEN_BATTLE else None
        if self.sink is not None:
            self.sink(rec)

    def finish(self) -> None:
        self.flush()


# ---------------------------------------------------------------------------
# pass 1 sink, pass 2
# ---------------------------------------------------------------------------

class _Pass1:
    """Writes pass-1 records and keeps the FrameEntry index the background needs."""

    def __init__(self, path: Path, session_dir: Path, config: ScanConfig):
        self.path = path
        self.session_dir = session_dir
        self.config = config
        self.entries: list[FrameEntry] = []
        self.records = 0
        self._fh = open(path, "w", encoding="utf-8", newline="\n")

    def __call__(self, rec: dict) -> None:
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.records += 1
        if rec.get("hk") is None or not rec.get("image") or not rec.get("size"):
            return
        cfg = self.config
        boxes = [tuple(r[4:8]) for r in rec.get("raw", ()) if r[8] in MASK_TYPES and r[3] >= cfg.bg_mask_conf]
        boxes += [tuple(r[6:10]) for r in rec.get("live", ()) if r[2] in MASK_TYPES]
        self.entries.append(FrameEntry(int(rec["n"]), float(rec["t"]), int(rec["hk"]),
                                       self.session_dir / rec["image"], (int(rec["size"][0]), int(rec["size"][1])),
                                       boxes, dict(rec.get("towers") or {})))

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """The JSON objects of a JSONL file (a line cut short by a crash is skipped)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def _bg_format() -> str:
    return "png" if image_codec_available("png") else "npy"


def fit_epochs(epochs: Sequence[Epoch], entries: Sequence[FrameEntry], *, out_dir: Path, sizes: SizeCache,
               config: ScanConfig, load: Callable[[Path], np.ndarray] = load_image,
               log: Callable[[str], Any] = print) -> dict[str, np.ndarray]:
    """Fit and save each epoch's background (masked median of its samples); epoch id -> image."""
    by_n = {e.n: e for e in entries}
    out: dict[str, np.ndarray] = {}
    fmt = _bg_format()
    (out_dir / BACKGROUND_DIR).mkdir(parents=True, exist_ok=True)
    for ep in epochs:
        imgs, masks = [], []
        shape = None
        info = None
        for n in ep.samples:
            e = by_n.get(n)
            if e is None:
                continue
            try:
                rgb = np.asarray(load(e.path))[..., :3]
            except Exception:
                continue
            if shape is None:
                shape = rgb.shape
                info = sizes.get((rgb.shape[1], rgb.shape[0]))
            if rgb.shape != shape:
                continue
            pad = config.bg_mask_pad_tiles * min(info.tile_w, info.tile_h)
            imgs.append(rgb)
            masks.append(box_mask(rgb.shape[:2], e.mask_boxes, pad))
        if not imgs:
            log(f"  background {ep.id}: no sample image could be read; no blobs there")
            continue
        bg = fit_background(imgs, masks, rect=info.rect, band_rows=config.bg_band_rows)
        ep.fitted = len(imgs)
        rel = f"{BACKGROUND_DIR}/{ep.id}.{fmt}"
        save_image(out_dir / rel, bg, fmt)
        ep.image = rel
        out[ep.id] = bg
    return out


@dataclass
class _LiveTrack:
    first_t: float
    last_t: float
    frames: int = 0
    cls: Counter = field(default_factory=Counter)
    team: int | None = None
    how: str | None = None
    card: int | None = None
    card_name: str | None = None
    forms: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {"first_t": round(self.first_t, 4), "last_t": round(self.last_t, 4), "frames": self.frames,
                "cls": dict(self.cls.most_common()), "team": self.team, "how": self.how, "card": self.card,
                "card_name": self.card_name, "forms": {str(k): v for k, v in sorted(self.forms.items())}}


def harvest_pass(pass1: Path, frames_out: Path, *, session_dir: Path, epochs: Sequence[Epoch],
                 backgrounds: Mapping[str, np.ndarray], geometry: ArenaGeometry, sizes: SizeCache,
                 config: ScanConfig, load: Callable[[Path], np.ndarray] = load_image,
                 log: Callable[[str], Any] = print, progress_every: int = 1000
                 ) -> tuple[HarvestTracker, dict[str, _LiveTrack], Counter]:
    """Pass 2 (module doc): blobs, proposals and harvest tracks per battle-screen record of
    `pass1`, written with the pass-1 fields to `frames_out`."""
    cfg = config
    epoch_of = {n: ep.id for ep in epochs for n in ep.members}
    harvest = HarvestTracker(cfg.tracker)
    live_tracks: dict[str, _LiveTrack] = {}
    counts: Counter = Counter()
    current_hk: Any = object()
    with open(frames_out, "w", encoding="utf-8", newline="\n") as fh:
        for rec in iter_jsonl(pass1):
            counts["records"] += 1
            hk = rec.get("hk")
            if hk is None or not rec.get("size"):
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                continue
            size = (int(rec["size"][0]), int(rec["size"][1]))
            if hk != current_hk:
                harvest.reset(int(hk))
                current_hk = hk
            seg = int(rec.get("seg", 0))
            raw = [RawBox.from_row(r) for r in rec.get("raw", ())]
            live = [LiveDet.from_row(r, seg) for r in rec.get("live", ())]
            ep = epoch_of.get(int(rec["n"]))
            rec["epoch"] = ep
            blobs: list[Blob] = []
            bg = backgrounds.get(ep) if ep is not None else None
            if cfg.blobs and bg is not None and rec.get("image"):
                try:
                    rgb = np.asarray(load(session_dir / rec["image"]))[..., :3]
                except Exception as exc:
                    counts[f"image:{type(exc).__name__}"] += 1
                    rgb = None
                if rgb is not None and rgb.shape != bg.shape:
                    counts["size_mismatch"] += 1
                    rgb = None
                if rgb is not None:
                    info = sizes.get(size)
                    exclude = [b.box for b in raw if (b.type in EXCLUDE_TYPES or b.cls in BAR_CLASSES)
                               and b.conf >= cfg.exclude_conf]
                    exclude += [d.box for d in live if d.type in EXCLUDE_TYPES or d.cls in BAR_CLASSES]
                    fg = foreground_mask(rgb, bg, threshold=cfg.fg_threshold, region=info.region, exclude=exclude)
                    blobs = find_blobs(fg, cell=info.cell, min_frac=cfg.cell_min_frac,
                                       min_area_px=cfg.blob_min_tiles * info.tile_area,
                                       max_area_px=cfg.blob_max_tiles * info.tile_area)
                    counts["blob_frames"] += 1

            def ground(box, typ, cls, _size=size):
                if typ == TYPE_SPELL:
                    x, y = geometry.px_to_world((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, _size)
                else:
                    x, y = geometry.ground_point(box, _size, cls=cls)
                return float(x), float(y)

            props = merge_proposals(raw, blobs, live, config=cfg, ground=ground)
            harvest.update(props, t=float(rec["t"]), n=int(rec["n"]))
            counts["blobs"] += len(blobs)
            counts["proposals"] += len(props)
            for p in props:
                counts[f"src:{p.src}"] += 1
            rec["blobs"] = [b.to_row() for b in blobs]
            rec["props"] = [p.to_dict() for p in props]
            for tr in rec.get("tracks", ()):
                key = f"{seg}:{tr['uid']}"
                lt = live_tracks.get(key)
                if lt is None:
                    lt = live_tracks[key] = _LiveTrack(float(rec["t"]), float(rec["t"]))
                lt.last_t = float(rec["t"])
                lt.frames += 1
                lt.cls[tr.get("cls")] += 1
                lt.forms[int(tr.get("form", 0))] += 1
                lt.team, lt.how = tr.get("team"), tr.get("how")
                if tr.get("card") is not None:
                    lt.card, lt.card_name = tr.get("card"), tr.get("card_name")
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            if progress_every and counts["records"] % progress_every == 0:
                log(f"  pass 2: {counts['records']} records, {len(harvest.tracks)} harvest tracks")
    return harvest, live_tracks, counts


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------

def _vision_replay():
    """tools/vision_replay.py as the module `vision_replay` (the way the tools and their tests
    import it), with tools/ and the repo root on sys.path."""
    if str(ROOT / "tools") not in sys.path:          # latency_probe, vision_classes, vision_replay
        sys.path.insert(0, str(ROOT / "tools"))
    if str(ROOT) not in sys.path:                    # scripts.vision_play (vision_replay does the same)
        sys.path.append(str(ROOT))
    import vision_replay
    return vision_replay


def resolve_session(path: str | Path) -> Path:
    """The session folder of a recording given as the folder, its frames.jsonl, a root holding
    exactly one session, or a plain folder of images. ScanError otherwise."""
    p = Path(path)
    if p.is_file() and p.name == FRAMES_FILE:
        return p.parent
    if p.is_dir() and (p / FRAMES_FILE).is_file():
        return p
    sessions = list_sessions(p)
    if len(sessions) == 1:
        return sessions[0]
    if len(sessions) > 1:
        raise ScanError(f"{p} holds {len(sessions)} recorded sessions ({', '.join(s.name for s in sessions)}); "
                        "pass the one to scan")
    if p.is_dir() and any(f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES for f in p.iterdir()):
        return p
    raise ScanError(f"no recording at {p} (need {FRAMES_FILE} with {IMAGES_DIR}/, or a folder of images)")


SCAN_FILES = (SCAN_JSON, SCAN_FRAMES, TRACKS_JSON, PASS1_TMP)


def prepare_out_dir(out_dir: str | Path, session_dir: Path, *, overwrite: bool = False) -> Path:
    """Create the output folder. An earlier scan there is refused unless overwrite, which
    deletes that scan's own files (SCAN_FILES and background/<epoch> images) and nothing else."""
    out = Path(out_dir)
    if out.resolve() == (session_dir / IMAGES_DIR).resolve() or out.resolve() == session_dir.resolve():
        raise ScanError(f"--out {out} would mix the scan with the recording; use a folder of its own")
    old = [f for f in SCAN_FILES if (out / f).exists()]
    if old and not overwrite:
        raise ScanError(f"{out} already holds a scan ({', '.join(old)}); pass --force to scan again")
    out.mkdir(parents=True, exist_ok=True)
    for f in old:
        (out / f).unlink()
    bg = out / BACKGROUND_DIR
    if bg.is_dir():
        for f in bg.iterdir():
            if f.is_file() and f.suffix.lower() in (".png", ".npy") and f.stem.startswith("b"):
                f.unlink()
    return out


def _summary_of_live(s: Mapping[str, Any]) -> dict[str, Any]:
    keep = ("frames", "errors", "screens", "ticks", "tick_sources", "battles", "image_interval_s", "sparse",
            "detections", "tracker", "events", "taps")
    return {k: s.get(k) for k in keep if k in s}


def scan_recording(recording: str | Path, out_dir: str | Path | None = None, *, live: Any,
                   low_pass: LowConfPass | None = None, config: ScanConfig | None = None,
                   geometry: ArenaGeometry | None = None, det_config: DetectorConfig | None = None,
                   card_ids: Mapping[str, int] | None = None, card_names: Mapping[int, str] | None = None,
                   taps: Iterable[Mapping[str, Any]] | None = None, limit: int = 0, overwrite: bool = False,
                   meta: Mapping[str, Any] | None = None, load: Callable[[Path], np.ndarray] = load_image,
                   log: Callable[[str], Any] = print, progress_every: int = 1000) -> dict[str, Any]:
    """The scan (module doc) of one recording with a built live chain: `live` is a
    vision_replay.ReplayPerception (best with a RecordingTracker), `low_pass` the harvest
    detector pass or None. card_ids / card_names: the live chain's card ids (taps are mapped by
    name; plays and tracks are named). taps: the recording's tap records (None: read them from
    the recording). Returns the summary written to scan.json."""
    VR = _vision_replay()
    from .capture import FileReplaySource

    cfg = config or ScanConfig()
    session = resolve_session(recording)
    out = prepare_out_dir(out_dir if out_dir is not None else session / SCAN_DIR, session, overwrite=overwrite)
    geometry = geometry if geometry is not None else getattr(live, "geometry", None) or ArenaGeometry.default()
    if det_config is None:
        det_config = (low_pass.config if low_pass is not None
                      else getattr(getattr(live, "detector", None), "config", None))
    sizes = SizeCache(geometry, det_config, cfg)
    names_by_id = {int(k): str(v) for k, v in (card_names or {}).items()}
    if card_ids is not None and not names_by_id:
        names_by_id = {int(i): str(n) for n, i in card_ids.items()}
    if taps is None:
        taps, _config = VR.read_recording_records(session)
    t_start = time.time()
    source = FileReplaySource(session, realtime=False, fps=float(TICKS_PER_S))
    if len(source) == 0:
        source.close()
        raise ScanError(f"{session} has no image on disk to scan")
    log(f"unit_harvest scan  {session} ({len(source)} images) -> {out}")
    sink = _Pass1(out / PASS1_TMP, session, cfg)
    perception = ScanPerception(live, low_pass=low_pass, card_names=names_by_id)
    replay = VR.Replay(source, perception, taps=taps, names=getattr(live, "names", None),
                       card_ids=dict(card_ids or {}), card_names=names_by_id, limit=limit, out=log,
                       progress_every=progress_every)
    perception.attach(battle=replay.battle, source=source, session_dir=session, sink=sink)
    try:
        live_summary = replay.run()
        perception.finish()
    finally:
        source.close()
        sink.close()
    t_pass1 = time.time()
    log(f"  pass 1: {perception.counts['frames']} images, {perception.counts['battle_frames']} on the battle "
        f"screen, {perception.counts['raw_boxes']} harvest boxes, {replay.battles} battle(s)")

    epochs = plan_epochs(sink.entries, cfg)
    backgrounds = fit_epochs(epochs, sink.entries, out_dir=out, sizes=sizes, config=cfg, load=load, log=log) \
        if cfg.blobs else {}
    t_bg = time.time()
    log(f"  background: {len(epochs)} epoch(s), {len(backgrounds)} fitted")
    harvest, live_tracks, counts = harvest_pass(out / PASS1_TMP, out / SCAN_FRAMES, session_dir=session,
                                                epochs=epochs, backgrounds=backgrounds, geometry=geometry,
                                                sizes=sizes, config=cfg, load=load, log=log,
                                                progress_every=progress_every)
    (out / PASS1_TMP).unlink()
    tracks = [harvest.tracks[h].to_dict() for h in sorted(harvest.tracks)]
    linked = sum(1 for tr in harvest.tracks.values() if tr.live)
    bare = sum(1 for tr in harvest.tracks.values() if not tr.cands and not tr.live)
    tracks_doc = {"schema": SCAN_SCHEMA, "harvest": tracks,
                  "live": {k: v.to_dict() for k, v in sorted(live_tracks.items())}}
    (out / TRACKS_JSON).write_text(json.dumps(tracks_doc, separators=(",", ":")) + "\n", encoding="utf-8")
    low_ms = sorted(perception.low_ms)
    summary: dict[str, Any] = {
        "schema": SCAN_SCHEMA, "created": datetime.now().isoformat(timespec="seconds"),
        "recording": str(Path(recording)), "session_dir": str(session), "out": str(out),
        "images": len(source), "frames": perception.counts["frames"],
        "battle_frames": perception.counts["battle_frames"],
        "files": {"frames": SCAN_FRAMES, "tracks": TRACKS_JSON, "background": BACKGROUND_DIR},
        "config": dataclasses.asdict(cfg),
        "detector_config": None if det_config is None else dataclasses.asdict(det_config),
        "harvest_pass": None if low_pass is None else {
            "conf": low_pass.config.conf, "max_det": low_pass.config.max_det, "iou": low_pass.config.iou,
            "imgsz": low_pass.config.imgsz, "boxes": low_pass.boxes, "dropped_off_board": low_pass.dropped,
            "ms_median": low_ms[len(low_ms) // 2] if low_ms else None},
        "live": _summary_of_live(live_summary),
        "live_assignment": {k[len("live_assign_"):]: v for k, v in perception.counts.items()
                            if k.startswith("live_assign_")},
        "harvest": {"records": counts["records"], "blob_frames": counts["blob_frames"], "blobs": counts["blobs"],
                    "proposals": counts["proposals"],
                    "by_source": {k[4:]: v for k, v in sorted(counts.items()) if k.startswith("src:")},
                    "tracks": len(tracks), "tracks_with_live": linked, "tracks_blob_only": bare,
                    "live_tracks": len(live_tracks)},
        "epochs": [ep.to_dict() for ep in epochs],
        "errors": dict(perception.errors + Counter({k: v for k, v in counts.items()
                                                    if k.startswith("image:") or k == "size_mismatch"})),
        "seconds": {"pass1": round(t_pass1 - t_start, 1), "background": round(t_bg - t_pass1, 1),
                    "pass2": round(time.time() - t_bg, 1)},
    }
    if meta:
        summary["meta"] = dict(meta)
    (out / SCAN_JSON).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    h = summary["harvest"]
    log(f"  harvest: {h['proposals']} proposals in {h['records']} records, {h['tracks']} harvest tracks "
        f"({h['tracks_with_live']} with a live track, {h['tracks_blob_only']} blobs only); "
        f"live opponent plays {(summary['live'].get('events') or {}).get('opp_cards')}")
    log(f"scan -> {out / SCAN_JSON}")
    return summary


@dataclass
class ScanResult:
    """A finished scan read back (load_scan)."""
    out_dir: Path
    header: dict[str, Any]
    tracks: list[dict[str, Any]]
    live_tracks: dict[str, dict[str, Any]]

    def frames(self) -> Iterator[dict]:
        return iter_scan_frames(self.out_dir)

    def background(self, epoch_id: str) -> np.ndarray | None:
        for ep in self.header.get("epochs", ()):
            if ep.get("id") == epoch_id and ep.get("image"):
                return load_image(self.out_dir / ep["image"])
        return None


def iter_scan_frames(out_dir: str | Path) -> Iterator[dict]:
    """The records of <out_dir>/scan_frames.jsonl, in scan order."""
    return iter_jsonl(Path(out_dir) / SCAN_FRAMES)


def load_scan(out_dir: str | Path) -> ScanResult:
    """scan.json and tracks.json of a finished scan (ScanError when there is none)."""
    out = Path(out_dir)
    if not (out / SCAN_JSON).is_file() or not (out / TRACKS_JSON).is_file():
        raise ScanError(f"no finished scan in {out} (need {SCAN_JSON} and {TRACKS_JSON})")
    header = json.loads((out / SCAN_JSON).read_text(encoding="utf-8"))
    if header.get("schema") != SCAN_SCHEMA:
        raise ScanError(f"{out / SCAN_JSON} is {header.get('schema')!r}, not {SCAN_SCHEMA}")
    doc = json.loads((out / TRACKS_JSON).read_text(encoding="utf-8"))
    return ScanResult(out, header, list(doc.get("harvest") or []), dict(doc.get("live") or {}))


# ---------------------------------------------------------------------------
# building the chains (command line, vision_autolabel)
# ---------------------------------------------------------------------------

def build_live_chain(*, detector: Any, table: Any, geometry: ArenaGeometry, our_deck: Sequence[str] | None = None,
                     templates: str | Path | None = None, spawn_table: str | Path | None = None,
                     ticks: str = "auto", unknown_plays: bool = False,
                     unknown_cost: int | None = None) -> tuple[Any, dict[str, Any]]:
    """The live chain tools/vision_replay.py's main() builds (HUD reader from the template
    store, event tables for the engine's card ids, the deriver, our deck's classes for the
    tracker), with a RecordingTracker. Returns (ReplayPerception, info: card_ids, card_names,
    ids_from, deck, ticks)."""
    VR = _vision_replay()
    VP = VR.VP
    from . import events as EV
    from . import hud as HUD
    from . import templates as TPL

    deck = list(VP.deck_names(VP.DEFAULT_DECK) if our_deck is None else our_deck)
    spawn, card_ids, ids_from = VR.engine_card_ids(spawn_table, table)
    cmap, ignore = EV.unit_class_map(table.doc), EV.cardless_classes(table.doc)
    if spawn is not None:
        tables = EV.EventTables.for_engine(spawn, class_of=cmap, ignore=ignore)
    else:
        tables = EV.EventTables.from_names(EV.load_spawn_children(), card_ids, class_of=cmap, ignore=ignore)
    deck_ids = [card_ids[n] for n in deck if n in card_ids]
    store = TPL.TemplateStore(templates) if templates else TPL.TemplateStore()
    hud_reader = HUD.HudReader(store, {n: card_ids[n] for n in deck if n in card_ids},
                               deck=deck_ids if len(deck_ids) == 8 else None)
    if ticks == "auto":
        ticks = VR.TICKS_CLOCK if hud_reader.clock_reader.ready else VR.TICKS_TIME
    ev_config = None
    if unknown_plays:
        ev_config = EV.EventConfig(unknown_plays=True, **({} if unknown_cost is None else {"unknown_cost": unknown_cost}))
    perception = VR.ReplayPerception(hud_reader=hud_reader, tracker=RecordingTracker(own_classes=table.deck_labels(deck)),
                                     deriver=EV.EventDeriver(tables, ev_config, own_deck=deck_ids), geometry=geometry,
                                     detector=detector, names=VR.Names(table), ticks=ticks)
    info = {"card_ids": card_ids, "card_names": {i: n for n, i in card_ids.items()}, "ids_from": ids_from,
            "deck": deck, "ticks": ticks, "unknown_plays": bool(unknown_plays)}
    return perception, info


def scan_session(recording: str | Path, out_dir: str | Path | None = None, *, weights: str | Path | None = None,
                 device: str | None = None, use_detector: bool = True, geometry: str | Path | None = None,
                 classes: str | Path | None = None, templates: str | Path | None = None,
                 our_deck: Sequence[str] | None = None, spawn_table: str | Path | None = None,
                 ticks: str = "auto", imgsz: int | None = None, unknown_plays: bool = False,
                 config: ScanConfig | None = None, limit: int = 0, use_taps: bool = True,
                 overwrite: bool = False, meta: Mapping[str, Any] | None = None,
                 log: Callable[[str], Any] = print) -> dict[str, Any]:
    """Load the detector (weights, else the pin), build the live chain and the harvest pass,
    and scan one recording (scan_recording). use_detector=False: HUD, tracks of nothing and
    blobs only. imgsz replaces the input size of both passes. DetectorError and ScanError
    propagate."""
    from .geometry import load_geometry

    resolve_session(recording)                      # a wrong path fails before any model loads
    cfg = config or ScanConfig()
    VR = _vision_replay()
    table = load_class_table(classes)
    geo = load_geometry(geometry)
    detector, det_note = None, "off (--no-detector)"
    if use_detector:
        from .detector import Detector
        det_config = VR.detector_config(None, imgsz, None)
        detector = Detector.load(weights, device=device, geometry=geo, table=table, config=det_config)
        detector.warmup()
        det_note = f"{detector.weights} sha256 {str(detector.sha256)[:12]} on {detector.device_desc}"
    live, info = build_live_chain(detector=detector, table=table, geometry=geo, our_deck=our_deck,
                                  templates=templates, spawn_table=spawn_table, ticks=ticks,
                                  unknown_plays=unknown_plays)
    low_pass = None
    if detector is not None:
        low_pass = LowConfPass.from_detector(detector, conf=cfg.low_conf, max_det=cfg.low_max_det, iou=cfg.low_iou)
    log(f"  detector   {det_note}")
    log(f"  live chain ticks from the {info['ticks']}; card ids {info['ids_from']}; our deck {', '.join(info['deck'])}")
    if low_pass is not None:
        log(f"  harvest    conf {low_pass.config.conf:g}, max_det {low_pass.config.max_det}, no side merge, "
            "no cross-class suppression")
    head = {"detector": det_note, "weights": None if detector is None else str(detector.weights),
            "sha256": None if detector is None else detector.sha256,
            "classes_digest": getattr(table, "digest", None), "geometry": getattr(geo, "name", None),
            "our_deck": info["deck"], "ticks": info["ticks"], "card_ids_from": info["ids_from"],
            "unknown_plays": info["unknown_plays"]}
    head.update(meta or {})
    return scan_recording(recording, out_dir, live=live, low_pass=low_pass, config=cfg, geometry=geo,
                          det_config=None if detector is None else detector.config,
                          card_ids=info["card_ids"], card_names=info["card_names"],
                          taps=None if use_taps else [], limit=limit, overwrite=overwrite, meta=head, log=log)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m scripts.vision.unit_harvest",
                                 description="Unit harvest scan of a recorded friendly (N09).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan", help="live chain + harvest pass + background + blobs + harvest tracks")
    p.add_argument("recording", help="a recorded session (capture --record), its frames.jsonl, a root with one "
                                     "session, or a folder of images")
    p.add_argument("--out", default=None, help=f"output folder (default <session>/{SCAN_DIR})")
    p.add_argument("--force", action="store_true", help="replace an earlier scan in the output folder")
    p.add_argument("--detector", default=None, help="detector weights (default: the pin)")
    p.add_argument("--device", default=None, help="detector device (default: CUDA GPU 0, else cpu)")
    p.add_argument("--no-detector", action="store_true", help="no detector: HUD and blobs only")
    p.add_argument("--imgsz", type=int, default=None, help="detector input size for both passes (default: live 640)")
    p.add_argument("--low-conf", type=float, default=ScanConfig.low_conf, help="harvest pass score threshold")
    p.add_argument("--low-max-det", type=int, default=ScanConfig.low_max_det, help="harvest pass box cap")
    p.add_argument("--epoch-s", type=float, default=ScanConfig.epoch_s,
                   help="longest stretch with one background, seconds (0: one per battle and tower fall)")
    p.add_argument("--bg-samples", type=int, default=ScanConfig.bg_samples, help="frames per background median")
    p.add_argument("--fg-threshold", type=int, default=ScanConfig.fg_threshold,
                   help="foreground: largest channel difference from the background")
    p.add_argument("--no-blobs", action="store_true", help="skip the background and the blobs")
    p.add_argument("--our-deck", default=None, help="Yodle2's deck: 'Hog 2.6' (default) or 8 catalogue names")
    p.add_argument("--geometry", default=None, help="an ArenaGeometry JSON (default: data/vision's)")
    p.add_argument("--classes", default=None, help="classes.json (default data/vision/classes.json)")
    p.add_argument("--templates", default=None, help="HUD template root (default: the TemplateStore's)")
    p.add_argument("--spawn-table", default=None, help="card ids from this spawn_primary.json")
    p.add_argument("--ticks", choices=("auto", "clock", "time"), default="auto",
                   help="the live chain tracks on the HUD clock or the capture time (auto: clock when templated)")
    p.add_argument("--unknown-plays", action="store_true", help="the live deriver reports UNKNOWN plays (N02)")
    p.add_argument("--no-taps", action="store_true", help="do not replay the recording's taps")
    p.add_argument("--limit", type=int, default=0, help="stop after N images (0: all)")
    return ap


def _cmd_scan(args: argparse.Namespace) -> int:
    try:
        cfg = ScanConfig(low_conf=args.low_conf, low_max_det=args.low_max_det, epoch_s=args.epoch_s,
                         bg_samples=args.bg_samples, fg_threshold=args.fg_threshold, blobs=not args.no_blobs)
    except ValueError as exc:
        print(f"scan settings: {exc}")
        return 2
    deck = None
    if args.our_deck:
        VP = _vision_replay().VP
        try:
            deck = VP.deck_names(args.our_deck)
        except argparse.ArgumentTypeError as exc:
            print(str(exc))
            return 2
    from .detector import DetectorError
    try:
        scan_session(args.recording, args.out, weights=args.detector, device=args.device,
                     use_detector=not args.no_detector, geometry=args.geometry, classes=args.classes,
                     templates=args.templates, our_deck=deck, spawn_table=args.spawn_table, ticks=args.ticks,
                     imgsz=args.imgsz, unknown_plays=args.unknown_plays, config=cfg, limit=args.limit,
                     use_taps=not args.no_taps, overwrite=args.force)
    except (ScanError, DetectorError, ValueError) as exc:
        print(f"scan: {exc}")
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "scan":
        return _cmd_scan(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
