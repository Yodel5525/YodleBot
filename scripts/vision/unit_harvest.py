r"""Unit harvest scan (N09), and the flags, split, clustering and sheet that follow it (N10).

docs/NEW_CARDS_PLAN.md 2.3 step 2. The owner records friendlies where his main account plays
the new cards against Yodle2 (plan section 4); this scan turns one recording into per-frame
PROPOSALS (every place a unit or spell may be) and HARVEST TRACKS (proposals followed over
time, by position only), next to what tonight's live chain made of the same images. N10
(flags S1-S7, splits, clustering, the contact sheet; section N10 below) reads its output;
nothing here names anything: the owner does, through tools/vision_autolabel.py `label`.

    & $BB -m scripts.vision.unit_harvest scan reports\vision\autolabel\$SID\recording --out reports\vision\autolabel\$SID\scan
    & $BB -m scripts.vision.unit_harvest scan reports\vision\harvest\$SID            # a recording not ingested yet
    ... --force                                          scan again (replaces the earlier scan's files)
    ... --detector runs\detect\cr_m0\weights\best.pt    weights other than the pin
    ... --no-detector                                    blobs only (no GPU; no labels to read)
    ... --limit 400                                      a quick look at the first 400 images

Run it in the GPU venv with MuMu and the browser closed: two detector passes per image, then
the background median and the blobs on the CPU. Budget (estimates, not measured yet): 10-20
minutes per battle (HUD reads, 2 YOLO passes and 2 image decodes per image, then the merge and
the harvest tracker in Python), and 150-300 MB of scan files per battle of about 4,400 images
(scan_frames.jsonl keeps every harvest box at conf 0.05 or more, up to 300 per image;
pass1.jsonl.tmp is as large again while pass 2 runs), so about 10 GB for the 36 Tier 1 and 2
battles. The scan folder can be deleted once its labels are emitted and reviewed (N11, N12: a
re-emit after `mark` or a relabel needs it). The first line scans
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

N10 (analyze_scan; tools/vision_autolabel.py `sheet` and `label` drive it). Per finished scan:
- WHICH TRACKS. Harvest tracks of at least `min_images` images whose side is enemy or unknown
  (track_side: candidate scores, a class Yodle2's Hog 2.6 cannot show counting as enemy
  whatever its side, plus the images held by live tracks of each team; a bare blob is
  "unknown", as an enemy spell cast on our half is). Our tracks are left out: their labels
  come from tonight's model (plan risk 4). A split track's parts each get their own side
  (frames_side); an ally part is left out the same way.
- THE SPLIT (S7). Each track's look is a signature per image (appearance: a 64-bin colour
  histogram of the box's foreground pixels against its epoch background, and its log width
  and height in tiles), on every image of a track up to `sig_max_frames` images, else on that
  many evenly spread ones. find_splits cuts a track where the mean signatures before and after
  differ by `split_dist` or more and each side is tight; on a sparsely sampled track each cut
  is then moved to the exact image between its two neighbouring samples (refine_cut); each
  part needs `split_min_images` images. Parts are what is flagged, clustered and labelled (key
  "<h>" or "<h>.<part>").
- THE FLAGS (flag_part), on the live tracks a part held (tracks.json "live"): S1 no enemy (or
  undecided) live track; S2 one no rule explained (how_of unknown, unknown_play or pending,
  card -1, an unknown class); S3 a card or class (or form) the declared deck cannot show
  (DeckContext); S4 the live (class, form) flickers (the minority reads at least
  `s4_min_frames` images and `s4_frac` of them); S5 at least `s5_min_images` images on the
  water (geometry.on_water) read as a ground class, RIVER_JUMPERS exempt; S6 under an enemy
  stopwatch (KataCR "clock", enemy side) that the live chain did not match to a play
  (link_markers, match_markers); S7 split.
- AUTO-ACCEPT. A part with no flag, of the enemy side, at least `accept_min_images` images,
  live coverage `accept_cover`, mean live score `accept_conf` and one (class, form) on
  `accept_agree` of its live images carries that read in PartInfo.auto, unless one of
  read["no_auto"] holds (flag_part): a class the declared deck shows in an evolution or hero
  slot read in another form (DeckContext.form_open: the model may read both forms as the base,
  and the true form must be labelled for it ever to reach thin_tracks), an image held by one
  of our live tracks, another enemy live track with another read (a unit born where another
  died inherits the position-only track), the detector's side head under `accept_side` enemy
  (detector_side: our unit misread as a deck class is put on the enemy team by the tracker's
  deck rule), or a unit born on our half. vision_autolabel never auto-accepts in a gold session.
- CLUSTERING (cluster_parts): leader clustering of the parts' mean signatures within
  `cluster_dist`, then a merge pass; largest first. Parts whose confident live reads name two
  different legend labels are never put in one cluster (keys).
- THE SHEET: sheet_crop (birth, +1 s, mid-life per shown track; crop_frames), render_sheet
  (header, legend, one tile per cluster) in a 3x5 bitmap font, write_sheet_image.
save_analysis / load_analysis keep it in the store, keyed by the scan's identity
(scan_identity: SHA-1 of scan.json and tracks.json) and the settings.

N11: propagate and emit (tools/vision_autolabel.py `emit` drives it; EmitConfig holds the
settings). decide_parts reads what tracks.json says of each part of the sheet: a label, ? or
a bad mark (masked), x (an effect: no box, no mask), o (ours), or nothing yet (pending).
propagate_parts gives each decided part a box on every image of its harvest track between
its first and last image: the linked live detection's box (the 0.35 one, nearer KataCR's
convention), else the proposal's, sized by the median of `smooth_window` images that came
from a detection (smooth_boxes; the bottom centre stays, or the centre for a spell); up to
`max_gap` missing images between two of its images filled by interpolation (fill_gaps:
labels and masks only, never across a battle or over one of our units); an image one of our
live tracks held is left out (a position-only track can run through one of our units). A
spell with a radius (DarkMagic 2.5, Vines 2.5, GoblinCurse 3.0 tiles) is its circle's box
around the median centre on every image (circle_box, through the geometry). plan_frame
turns one image into the boxes to write (enemy labels; Yodle2's units, towers, UI and
objects as tonight's pseudo-labels) and the regions to mask (?, pending, bad marks, o parts
with no Yodle2 read, every live unit or spell no decided part explains, and every leftover:
a detector proposal at `leftover_conf` or more of a track no decided part holds there, such
as one of Yodle2's units tonight's model missed or an enemy track too short for the sheet,
where no box is kept or masked already), or the reason the image is dropped; feather_paste pastes the epoch background into the masks (grown by
`mask_pad_tiles`, feathered over `feather_px`); crop_image, to_yolo and from_yolo give the
live crop framing (detector.crop_rect_for); frame_yolo gives gold's full-frame rows. The
scan folder must stay until emit (and any emit --force after review) is done.

N12: review pages (tools/vision_autolabel.py `review` drives them). Page A is render_sheet over
one tile per part (sheet_crop of the raw image with the propagated box, outlined in DIM_COLOUR
for a filled image; text_tile where the raw is gone); page B is render_frames: whole frames
(emitted crops, or gold's raw frames) scaled to a grid, with their boxes (draw_box, the label
above each) and their masks or ignore regions outlined.

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
import bisect  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Container, Iterable, Iterator, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from .detector import BAR_CLASSES, DetectorConfig, box_iou, crop_rect_for, precision_kwargs  # noqa: E402
from .geometry import (ArenaGeometry, HEIGHT_MT, RIVER_Y_MT, WIDTH_MT, check_frame_size, in_arena,  # noqa: E402
                       on_water)
from .labels import (ENEMY_SIDE, TYPE_OBJECT, TYPE_PAD, TYPE_SPELL, TYPE_TOWER, TYPE_UI, TYPE_UNIT,  # noqa: E402
                     TYPE_UNKNOWN, load_class_table, resolve_label, split_side)
from .record import (FRAMES_FILE, IMAGE_FORMATS, IMAGES_DIR, image_codec_available, list_sessions,  # noqa: E402
                     load_image, save_image)
from .tracker import Observation, Tracker, TrackerConfig, observations_from_detections  # noqa: E402
from .types import ENEMY_TEAM, FORM_BASE, OWN_TEAM, SCREEN_BATTLE, UNKNOWN_TEAM  # noqa: E402

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


def _form_flies(table: Any, lab: Any) -> bool:
    """Whether a resolved label's class form flies (classes.json), False when unknown."""
    try:
        return bool(table.classes[lab.class_id].form(lab.form).flying)
    except (IndexError, KeyError, TypeError, AttributeError):
        return False


def flying_classes(table: Any) -> Callable[[str | None], bool]:
    """name -> whether its class form flies, cached: harvest_pass places a live-only proposal
    with it as LowConfPass places the harvest boxes, so one flyer's track is not 0.6 tile apart
    between its detector and live-only images. (The live tracker places no class as flying.)"""
    cache: dict[str, bool] = {}

    def fly(name: str | None) -> bool:
        if not name:
            return False
        if name not in cache:
            lab = resolve_label(table, str(name))
            cache[name] = lab.type == TYPE_UNIT and lab.class_id is not None and _form_flies(table, lab)
        return cache[name]
    return fly


class LowConfPass:
    """The detector's weights at a low score (module doc, step 2): every box of the crop the
    live detector sees (crop_rect_for), each with the side its own class name carries.

    model: anything with `.names` and ultralytics' `.predict(source=bgr, imgsz=, conf=, iou=,
    max_det=, device=, quantize=, verbose=)` (`half=` before ultralytics 8.4.80, as the live
    detector: detector.precision_kwargs; a YOLO, or a fake in tests). config: the live
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
        self.precision = precision_kwargs(self.half)   # predict()'s quantize= (half= before 8.4.80)
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
        return _form_flies(self.table, lab)

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
                                 device=self.device, verbose=False, **self.precision)
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

    def new_battle(self, **kw: Any) -> None:
        self.live.new_battle(**kw)
        self.segment += 1

    def end_battle(self, **kw: Any) -> None:
        # Replay passes keep_towers=True when a battle ends without its end screen
        # (vision_replay._frame, vision_play.keeps_towers): handed on as given.
        self.live.end_battle(**kw)
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
                 log: Callable[[str], Any] = print, progress_every: int = 1000,
                 flying: Callable[[str | None], bool] | None = None
                 ) -> tuple[HarvestTracker, dict[str, _LiveTrack], Counter]:
    """Pass 2 (module doc): blobs, proposals and harvest tracks per battle-screen record of
    `pass1`, written with the pass-1 fields to `frames_out`. flying(class name): whether a
    live-only proposal stands under the geometry's flyer offset, as LowConfPass places the
    harvest boxes (flying_classes; None: no class flies)."""
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
                    x, y = geometry.ground_point(box, _size, cls=cls, flying=bool(flying and flying(cls)))
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
    table = getattr(low_pass, "table", None) or getattr(live, "table", None)
    harvest, live_tracks, counts = harvest_pass(out / PASS1_TMP, out / SCAN_FRAMES, session_dir=session,
                                                epochs=epochs, backgrounds=backgrounds, geometry=geometry,
                                                sizes=sizes, config=cfg, load=load, log=log,
                                                progress_every=progress_every,
                                                flying=None if table is None else flying_classes(table))
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
# N10: flags S1-S7, the look split, auto-accept, clustering, the contact sheet
# ---------------------------------------------------------------------------

#: The flags of an enemy track the live chain gets wrong (plan 2.3 step 3).
FLAG_TEXT: dict[str, str] = {
    "S1": "no live track",
    "S2": "unknown",
    "S3": "not the declared deck",
    "S4": "flickering label",
    "S5": "ground class over the river",
    "S6": "under an unmatched enemy stopwatch",
    "S7": "look changes mid-track",
}
FLAGS = tuple(FLAG_TEXT)
SIDE_ENEMY, SIDE_ALLY, SIDE_UNKNOWN = "enemy", "ally", "unknown"
#: S5 leaves these classes out: the units with JumpEnabled in
#: data/royaleapi/cards_stats_characters.json (HogRider, RoyalHog, Ram, Prince, PrinceBuff,
#: Prince_Chess, DarkPrince, VoodooHog, Event_VoodooHog, SuperHogRider; plus the terry hog and
#: the hero Dark Prince's mount), the dashers and leapers that cross the water (Assassin is the
#: Bandit, BossBandit, GoldenKnight, MegaKnight, and the spirits, which leap onto a target) and
#: the units that travel underground (Miner, GoblinDrillDig). Vision class names.
RIVER_JUMPERS = frozenset({
    "HogRider", "RoyalHog", "Ram", "RamRider", "Prince", "PrinceBuff", "Prince_Chess", "DarkPrince",
    "DarkPrinceHero_Mount", "VoodooHog", "Event_VoodooHog", "SuperHogRider", "SuperHogRider_Terry",
    "Assassin", "BossBandit", "GoldenKnight", "MegaKnight",
    "FireSpirits", "IceSpirits", "ElectroSpirit", "HealSpirit",
    "Miner", "GoblinDrillDig"})
MARKER_CLS = "clock"                       # KataCR's deploy stopwatch (a UI label of tonight's set)
#: EventDeriver.how_of values of a live track that is (part of) an opponent play.
PLAY_HOWS = frozenset({"play", "unknown_play", "merged"})
#: how_of values of a track no rule named: S2 (with None, and a card of -1).
UNEXPLAINED_HOWS = frozenset({"unknown", "unknown_play", "pending"})
#: Yodle2's deck when the scan does not say (scan_session writes meta our_deck; the same cards
#: as scripts/vision_play.py DEFAULT_DECK, Hog 2.6, restated so this needs no vision_play).
DEFAULT_OUR_DECK = ("HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons", "Fireball", "Log")
UNKNOWN_CARD_NAME = "UNKNOWN"
ANALYSIS_SCHEMA = "hastycr.vision.unit_harvest.analysis/1"
ANALYSIS_JSON = "analysis.json"
ANALYSIS_CROPS = "crops.npz"
HIST_LEVELS = 4                            # per channel: a 64-bin colour histogram


@dataclass(frozen=True)
class FlagConfig:
    """N10 settings (module doc, N10). Counts are recorded images (about 19 a second at
    capture --image-every 3); tiles are measured at the board's centre."""
    min_images: int = 4                     # shorter harvest tracks are not looked at
    # S4: the live label flickers
    s4_min_frames: int = 3
    s4_frac: float = 0.2
    # S5: a ground class over the water
    s5_min_images: int = 6
    # S6: enemy stopwatches
    marker_conf: float = 0.3
    marker_gap_s: float = 0.5
    marker_link_tiles: float = 1.0
    marker_min_frames: int = 2
    marker_before_s: float = 0.5
    marker_after_s: float = 1.5
    marker_play_tiles: float = 3.0
    # appearance: signatures, the S7 split, clustering
    fg_threshold: int = 30
    sig_max_frames: int = 480               # a signature on every image up to this many, else evenly spread
    size_weight: float = 0.5
    split_dist: float = 0.35
    split_ratio: float = 0.6
    split_min_images: int = 6
    max_parts: int = 3
    cluster_dist: float = 0.3
    # auto-accept
    accept_min_images: int = 8
    accept_cover: float = 0.7
    accept_conf: float = 0.6
    accept_agree: float = 0.8
    accept_side: float = 0.8                # share of the detector's own side scores that are enemy
    # the contact sheet
    cell: int = 72
    crop_pad: float = 0.3

    def __post_init__(self) -> None:
        for name in ("min_images", "s4_min_frames", "s5_min_images", "marker_min_frames", "sig_max_frames",
                     "split_min_images", "max_parts", "accept_min_images", "cell"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"FlagConfig.{name} must be at least 1")
        for name in ("s4_frac", "marker_conf", "split_ratio", "accept_cover", "accept_conf", "accept_agree",
                     "accept_side"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"FlagConfig.{name} must be in [0, 1]")
        for name in ("marker_gap_s", "marker_link_tiles", "marker_before_s", "marker_after_s",
                     "marker_play_tiles", "size_weight", "split_dist", "cluster_dist", "crop_pad"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"FlagConfig.{name} must be >= 0")
        if not 1 <= int(self.fg_threshold) <= 255:
            raise ValueError("FlagConfig.fg_threshold must be in [1, 255]")


@dataclass(slots=True)
class TrackFrame:
    """One image of a harvest track: its proposal (scan record n, proposal k) and the live
    detection that joined it (live: "seg:uid")."""
    n: int
    k: int
    t: float
    image: str | None
    epoch: str | None
    size: tuple[int, int] | None
    box: tuple[float, float, float, float]
    x: float
    y: float
    src: str
    cands: list[tuple[str, int | None, float]] = field(default_factory=list)
    live: str | None = None
    live_cls: str | None = None
    live_form: int = FORM_BASE
    live_conf: float = 0.0
    live_type: str | None = None


@dataclass
class PartInfo:
    """One enemy (or side-unknown) harvest track, or one part of it after an S7 split: what is
    flagged, clustered, shown on the sheet and labelled. key: "<h>" or "<h>.<part>"."""
    h: int
    part: int
    parts: int
    hk: int | None
    side: str
    n: int
    first_n: int
    last_n: int
    first_t: float
    last_t: float
    flags: dict[str, str] = field(default_factory=dict)
    read: dict[str, Any] = field(default_factory=dict)
    cands: list[tuple[str, float]] = field(default_factory=list)            # harvest class, mean score
    live_classes: list[tuple[str, int, int]] = field(default_factory=list)  # (class, form, images)
    sig: list[float] | None = None
    auto: dict[str, Any] | None = None       # {"cls", "form", "type"} of an auto-accepted read
    birth: tuple[int, int] | None = None
    crop_n: list[int] = field(default_factory=list)

    @property
    def key(self) -> str:
        return part_key(self.h, self.part, self.parts)

    def to_dict(self, *, with_sig: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "key": self.key, "h": self.h, "part": self.part, "parts": self.parts, "hk": self.hk,
            "side": self.side, "n": self.n, "first_n": self.first_n, "last_n": self.last_n,
            "first_t": round(self.first_t, 4), "last_t": round(self.last_t, 4), "flags": dict(self.flags),
            "read": dict(self.read), "cands": [[c, s] for c, s in self.cands],
            "live_classes": [[c, f, k] for c, f, k in self.live_classes], "auto": self.auto,
            "birth": None if self.birth is None else list(self.birth), "crop_n": list(self.crop_n)}
        if with_sig:
            d["sig"] = None if self.sig is None else [round(float(v), 5) for v in self.sig]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PartInfo":
        return cls(int(d["h"]), int(d.get("part", 0)), int(d.get("parts", 1)),
                   None if d.get("hk") is None else int(d["hk"]), str(d.get("side", SIDE_UNKNOWN)),
                   int(d["n"]), int(d["first_n"]), int(d["last_n"]), float(d["first_t"]), float(d["last_t"]),
                   dict(d.get("flags") or {}), dict(d.get("read") or {}),
                   [(str(c), float(s)) for c, s in d.get("cands") or ()],
                   [(str(c), int(f), int(k)) for c, f, k in d.get("live_classes") or ()],
                   None if d.get("sig") is None else [float(v) for v in d["sig"]],
                   None if d.get("auto") is None else dict(d["auto"]),
                   None if d.get("birth") is None else (int(d["birth"][0]), int(d["birth"][1])),
                   [int(v) for v in d.get("crop_n") or ()])


def part_key(h: int, part: int = 0, parts: int = 1) -> str:
    """"12" for a whole harvest track, "12.1" for the second part of a split one."""
    return str(int(h)) if int(parts) <= 1 else f"{int(h)}.{int(part)}"


FORM_WORDS = ("base", "evo", "hero")


@dataclass
class DeckContext:
    """What the declared deck allows (S3, auto-accept): its cards (catalogue names), every
    label its cards can show (ClassTable.deck_labels) and, per class, the forms its slots show
    (the base form always; an evolution or hero slot also its form). slot_forms: the classes
    an evolution or hero slot shows in a form other than their base (form_open)."""
    table: Any
    cards: frozenset[str]
    labels: frozenset[str]
    forms: dict[str, frozenset[int]]
    slot_forms: dict[str, frozenset[int]] = field(default_factory=dict)

    @classmethod
    def build(cls, table: Any, deck: Sequence[tuple[str, int]]) -> "DeckContext":
        cards = [str(c) for c, _ in deck]
        forms: dict[str, set[int]] = {}
        slots: dict[str, set[int]] = {}
        for card, slot in deck:
            for name in table.deck_classes([str(card)]):
                vc = table.by_name[name]
                fs = forms.setdefault(name, set())
                fs.add(vc.base_form)
                if int(slot) in vc.forms:
                    fs.add(int(slot))
                    if int(slot) != vc.base_form:
                        slots.setdefault(name, set()).add(int(slot))
        return cls(table, frozenset(cards), frozenset(table.deck_labels(cards)),
                   {k: frozenset(v) for k, v in forms.items()}, {k: frozenset(v) for k, v in slots.items()})

    def form_open(self, cls: str | None, form: int = FORM_BASE) -> str | None:
        """Why a live read of `cls` in `form` does not tell the unit's form, or None. An
        evolution or hero slot shows its card in both forms within a match (the evolved cycle
        only, or after the ability), and a model without the form's label (tonight's has none
        for most of them, plan 1.3, 1.4) reads both as the base: such a read is never
        auto-accepted or suggested, so the true form gets labelled (plan 4.5)."""
        vc = self.table.by_name.get(cls) if cls else None
        if vc is None:
            return None
        slots = self.slot_forms.get(vc.name)
        if not slots or int(form) in slots:
            return None
        other = " or ".join(FORM_WORDS[f] if 0 <= f < len(FORM_WORDS) else str(f) for f in sorted(slots))
        here = FORM_WORDS[int(form)] if 0 <= int(form) < len(FORM_WORDS) else str(form)
        return f"base or {other}? (a {here} read of {vc.name} does not tell)"

    def problem(self, cls: str | None, form: int = FORM_BASE, card_name: str | None = None) -> str | None:
        """Why a live read (class, form, card) is not this deck's; None when it is."""
        if card_name and card_name != UNKNOWN_CARD_NAME and card_name not in self.cards:
            return f"card {card_name}"
        if cls is None:
            return None
        if cls not in self.labels:
            return f"class {cls}"
        vc = self.table.by_name.get(cls)
        if vc is not None and vc.name in self.forms and int(form) not in self.forms[vc.name]:
            return f"the {('base', 'evo', 'hero')[int(form)] if 0 <= int(form) <= 2 else form} form of {vc.name}"
        return None


def track_side(track: Mapping[str, Any], live_tracks: Mapping[str, Mapping[str, Any]],
               own_labels: Iterable[str]) -> tuple[str, float, float]:
    """(side, enemy evidence, ally evidence) of a harvest track (HarvestTrack.to_dict): its
    candidates' summed scores (a class Yodle2's deck cannot show counts as enemy whatever its
    side; one it can counts by its side) plus one per image held by a live track of either
    team. A bare blob (no evidence) is "unknown": an enemy spell cast on our half looks so."""
    own = set(own_labels)
    enemy = ally = 0.0
    for key, agg in (track.get("cands") or {}).items():
        cls, _, side = str(key).rpartition("@")
        s = float(agg.get("sum", 0.0)) if isinstance(agg, Mapping) else float(agg)
        if cls in own:
            if side == SIDE_ALLY:
                ally += s
            elif side == SIDE_ENEMY:
                enemy += s
        else:
            enemy += s
    for lk, frames in (track.get("live") or {}).items():
        team = (live_tracks.get(lk) or {}).get("team")
        if team == ENEMY_TEAM:
            enemy += float(frames)
        elif team == OWN_TEAM:
            ally += float(frames)
    side = SIDE_ENEMY if enemy > ally else SIDE_ALLY if ally > enemy else SIDE_UNKNOWN
    return side, round(enemy, 3), round(ally, 3)


def frames_side(frames: Sequence["TrackFrame"], live_tracks: Mapping[str, Mapping[str, Any]],
                own_labels: Iterable[str]) -> tuple[str, float, float]:
    """track_side over some images of a track (one part of it): the same evidence, so over all
    of a track's images both agree. A part is shown and labelled by its own side."""
    own = set(own_labels)
    enemy = ally = 0.0
    for f in frames:
        for cls, side, conf in f.cands:
            if cls in own:
                if side == 0:
                    ally += float(conf)
                elif side == ENEMY_SIDE:
                    enemy += float(conf)
            else:
                enemy += float(conf)
        if f.live:
            team = (live_tracks.get(f.live) or {}).get("team")
            if team == ENEMY_TEAM:
                enemy += 1.0
            elif team == OWN_TEAM:
                ally += 1.0
    side = SIDE_ENEMY if enemy > ally else SIDE_ALLY if ally > enemy else SIDE_UNKNOWN
    return side, round(enemy, 3), round(ally, 3)


def detector_side(frames: Sequence["TrackFrame"]) -> tuple[float, float]:
    """(enemy, ally): the detector's own side scores summed over the images, whatever the
    class. track_side counts a class Yodle2 cannot show as enemy, and the live tracker puts
    such a label on the enemy team (tracker.py team_of), so one of our units misread as a
    class of the declared deck looks enemy to both; auto-accept also asks the side head."""
    enemy = ally = 0.0
    for f in frames:
        for _cls, side, conf in f.cands:
            if side == ENEMY_SIDE:
                enemy += float(conf)
            elif side == 0:
                ally += float(conf)
    return round(enemy, 3), round(ally, 3)


@dataclass(slots=True)
class Clock:
    """One enemy stopwatch box on one image."""
    n: int
    t: float
    box: tuple[float, float, float, float]
    size: tuple[int, int] | None
    conf: float


@dataclass
class MarkerEvent:
    """Enemy stopwatch boxes linked over consecutive images: one deploy."""
    clocks: list[Clock] = field(default_factory=list)
    under: dict[int, set[int]] = field(default_factory=dict)       # harvest track -> images under it
    matched: bool = False
    why: str = ""

    @property
    def t0(self) -> float:
        return self.clocks[0].t

    @property
    def t1(self) -> float:
        return self.clocks[-1].t

    def to_dict(self) -> dict[str, Any]:
        b = self.clocks[0].box
        return {"t0": round(self.t0, 4), "t1": round(self.t1, 4), "first_n": self.clocks[0].n,
                "images": len(self.clocks), "box": [_r1(v) for v in b], "matched": self.matched,
                "why": self.why, "under": sorted(self.under)}


def collect_frames(records: Iterable[Mapping[str, Any]], wanted: Iterable[int], *,
                   marker_conf: float = 0.3) -> tuple[dict[int, list[TrackFrame]], list[Clock], list[dict]]:
    """One pass over scan_frames records: the images of the `wanted` harvest tracks (with the
    live detection that joined each: class, form, score), the enemy stopwatches (harvest boxes
    named "clock" on the enemy side, or live ones with P(enemy) > 0.5, at marker_conf or more)
    and the live chain's opponent plays (with the record's time and segment)."""
    want = {int(h) for h in wanted}
    frames: dict[int, list[TrackFrame]] = {h: [] for h in want}
    clocks: list[Clock] = []
    plays: list[dict] = []
    for rec in records:
        props = rec.get("props")
        n, t = int(rec.get("n", 0)), float(rec.get("t", 0.0))
        size = None if not rec.get("size") else (int(rec["size"][0]), int(rec["size"][1]))
        seg = int(rec.get("seg", 0))
        lives: dict[int, LiveDet] = {}
        live_clocks: list[LiveDet] = []
        for row in rec.get("live") or ():
            d = LiveDet.from_row(row, seg)
            if d.uid is not None:
                lives[int(d.uid)] = d
            if d.cls == MARKER_CLS and d.p_enemy > 0.5 and d.conf >= marker_conf:
                live_clocks.append(d)
        here: list[Clock] = []
        for row in rec.get("raw") or ():
            r = RawBox.from_row(row)
            if (r.cls == MARKER_CLS or r.name == MARKER_CLS) and r.side == ENEMY_SIDE and r.conf >= marker_conf:
                here.append(Clock(n, t, r.box, size, r.conf))
        for d in live_clocks:
            if not any(box_iou(d.box, c.box) >= 0.5 for c in here):
                here.append(Clock(n, t, d.box, size, d.conf))
        clocks.extend(here)
        for p in rec.get("plays") or ():
            plays.append({"t": t, "n": n, "seg": seg, "card": p.get("card"), "card_name": p.get("card_name"),
                          "x_mt": p.get("x_mt"), "y_mt": p.get("y_mt"), "kind": p.get("kind"),
                          "uids": list(p.get("uids") or ())})
        if not props:
            continue
        for k, p in enumerate(props):
            h = p.get("h")
            if h is None or int(h) not in want:
                continue
            live = p.get("live")
            key = None if live is None else f"{int(live[0])}:{int(live[1])}"
            d = None if live is None else lives.get(int(live[1]))
            frames[int(h)].append(TrackFrame(
                n, k, t, rec.get("image"), rec.get("epoch"), size, tuple(float(v) for v in p["box"]),
                float(p.get("x", 0.0)), float(p.get("y", 0.0)), str(p.get("src", "")),
                [(str(c), None if s is None else int(s), float(v)) for c, s, v in p.get("cands") or ()],
                key, p.get("live_cls") or (d.cls if d is not None else None),
                int(d.form) if d is not None else FORM_BASE, float(d.conf) if d is not None else 0.0,
                d.type if d is not None else None))
    for fs in frames.values():
        fs.sort(key=lambda f: (f.n, f.k))
    return frames, clocks, plays


# -- appearance --------------------------------------------------------------

def appearance(rgb: np.ndarray, box: Sequence[float], background: np.ndarray | None = None, *,
               threshold: int = 30, tile: Sequence[float] = (1.0, 1.0)) -> np.ndarray | None:
    """A look signature of the box on one image: a 64-bin colour histogram (4 levels per
    channel) of its foreground pixels (differing from `background` by more than `threshold` on
    the largest channel; all its pixels when there is no background or too few differ), then
    log(width) and log(height) in tiles. None for a box under 2 px."""
    img = np.asarray(rgb)
    h, w = img.shape[:2]
    x0, y0 = max(0, int(math.floor(float(box[0])))), max(0, int(math.floor(float(box[1]))))
    x1, y1 = min(w, int(math.ceil(float(box[2])))), min(h, int(math.ceil(float(box[3]))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = img[y0:y1, x0:x1, :3].astype(np.int16)
    fg = None
    if background is not None and np.asarray(background).shape[:2] == (h, w):
        bg = np.asarray(background)[y0:y1, x0:x1, :3].astype(np.int16)
        fg = np.abs(crop - bg).max(axis=2) > int(threshold)
        if int(fg.sum()) < max(4, int(0.08 * fg.size)):
            fg = None
    px = crop.reshape(-1, 3) if fg is None else crop[fg]
    q = np.clip(px // (256 // HIST_LEVELS), 0, HIST_LEVELS - 1)
    idx = (q[:, 0] * HIST_LEVELS + q[:, 1]) * HIST_LEVELS + q[:, 2]
    hist = np.bincount(idx, minlength=HIST_LEVELS ** 3).astype(np.float64)
    hist /= max(1.0, float(hist.sum()))
    tw, th = max(float(tile[0]), 1e-6), max(float(tile[1]), 1e-6)
    size = [math.log(max(float(box[2]) - float(box[0]), 1e-3) / tw),
            math.log(max(float(box[3]) - float(box[1]), 1e-3) / th)]
    return np.concatenate([hist, np.asarray(size, dtype=np.float64)])


def sig_distance(a: Sequence[float], b: Sequence[float], size_weight: float = 0.5) -> float:
    """Half the L1 distance of the histograms (0 same colours, 1 none shared) plus size_weight
    times the mean absolute difference of the log sizes."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(0.5 * np.abs(a[:-2] - b[:-2]).sum() + size_weight * 0.5 * np.abs(a[-2:] - b[-2:]).sum())


def mean_signature(sigs: Sequence[Sequence[float]]) -> np.ndarray:
    return np.mean(np.stack([np.asarray(s, dtype=np.float64) for s in sigs]), axis=0)


def _best_cut(S: np.ndarray, lo: int, hi: int, *, min_part: int, split_dist: float, split_ratio: float,
              size_weight: float) -> tuple[int, float] | None:
    m = hi - lo
    if m < 2 * min_part:
        return None
    csum = np.cumsum(S[lo:hi], axis=0)
    total = csum[-1]
    best_j, best_d = None, -1.0
    for j in range(min_part, m - min_part + 1):
        a = csum[j - 1] / j
        b = (total - csum[j - 1]) / (m - j)
        d = sig_distance(a, b, size_weight)
        if d > best_d + 1e-12:
            best_j, best_d = j, d
    if best_j is None or best_d < split_dist:
        return None
    a, b = S[lo:lo + best_j].mean(axis=0), S[lo + best_j:hi].mean(axis=0)
    spread = max(float(np.mean([sig_distance(s, a, size_weight) for s in S[lo:lo + best_j]])),
                 float(np.mean([sig_distance(s, b, size_weight) for s in S[lo + best_j:hi]])))
    if spread > split_ratio * best_d:
        return None
    return lo + best_j, best_d


def find_splits(sigs: Sequence[Sequence[float]], *, min_part: int = 3, split_dist: float = 0.35,
                split_ratio: float = 0.6, size_weight: float = 0.5, max_parts: int = 3) -> list[int]:
    """Where the look changes (S7): indices into `sigs` (a track's signatures in time order)
    at which a new part starts. A cut is the point that best separates the mean signatures
    before and after it; it is kept when that distance is at least split_dist and each side
    is tight (its mean distance to its own mean at most split_ratio times the cut's), each
    side holding min_part signatures or more. Cuts are added best first, up to max_parts parts."""
    if len(sigs) < 2 * max(1, min_part) or max_parts < 2:
        return []
    S = np.stack([np.asarray(s, dtype=np.float64) for s in sigs])
    segments = [(0, len(S))]
    cuts: list[int] = []
    while len(segments) < max_parts:
        best = None
        for lo, hi in segments:
            r = _best_cut(S, lo, hi, min_part=max(1, min_part), split_dist=split_dist,
                          split_ratio=split_ratio, size_weight=size_weight)
            if r is not None and (best is None or r[1] > best[1]):
                best = (r[0], r[1], lo, hi)
        if best is None:
            break
        k, _d, lo, hi = best
        cuts.append(k)
        segments.remove((lo, hi))
        segments += [(lo, k), (k, hi)]
    return sorted(cuts)


# -- the clustering ------------------------------------------------------------

@dataclass
class TrackCluster:
    """Parts that look alike (one tile of the sheet): keys most typical first, their
    distances to the centroid."""
    id: int
    keys: list[str]
    dists: list[float]
    centroid: np.ndarray | None = None
    images: int = 0


def cluster_parts(parts: Sequence[PartInfo], *, max_dist: float = 0.3, size_weight: float = 0.5,
                  keys: Mapping[str, Any] | None = None) -> list[TrackCluster]:
    """Leader clustering of the parts' mean signatures (most images first: a part joins the
    nearest centroid within max_dist, else starts a cluster), then clusters whose centroids
    are within max_dist are merged, closest first. A part without a signature is a cluster of
    its own. keys: part key -> a value (the legend number of a confident live read); parts of
    two different values never share a cluster (one map value labels a whole cluster), a
    part without one joins any. Clusters come largest first (tracks, then images), ids 0..
    in that order."""
    keyed = dict(keys or {})
    order = sorted(parts, key=lambda p: (-p.n, p.h, p.part))
    groups: list[dict[str, Any]] = []

    def clash(a: Any, b: Any) -> bool:
        return a is not None and b is not None and a != b

    for p in order:
        k = keyed.get(p.key)
        if p.sig is None:
            groups.append({"members": [p], "sum": None, "key": k})
            continue
        s = np.asarray(p.sig, dtype=np.float64)
        best, best_d = None, None
        for g in groups:
            if g["sum"] is None or clash(k, g["key"]):
                continue
            d = sig_distance(s, g["sum"] / len(g["members"]), size_weight)
            if d <= max_dist and (best_d is None or d < best_d):
                best, best_d = g, d
        if best is None:
            groups.append({"members": [p], "sum": s.copy(), "key": k})
        else:
            best["members"].append(p)
            best["sum"] = best["sum"] + s
            if best["key"] is None:
                best["key"] = k
    live = [g for g in groups if g["sum"] is not None]
    if len(live) > 1:
        cen = np.stack([g["sum"] / len(g["members"]) for g in live])

        def row(i: int) -> np.ndarray:
            r_ = (0.5 * np.abs(cen[:, :-2] - cen[i, :-2]).sum(axis=1)
                  + size_weight * 0.5 * np.abs(cen[:, -2:] - cen[i, -2:]).sum(axis=1))
            r_[[clash(live[i]["key"], g["key"]) for g in live]] = np.inf
            return r_

        dist = np.stack([row(i) for i in range(len(live))])
        np.fill_diagonal(dist, np.inf)
        alive = np.ones(len(live), dtype=bool)
        while True:
            i, j = divmod(int(np.argmin(dist)), len(live))
            if not dist[i, j] <= max_dist:
                break
            k, gone = min(i, j), max(i, j)              # the later group joins the earlier one
            live[k]["members"] += live[gone]["members"]
            live[k]["sum"] = live[k]["sum"] + live[gone]["sum"]
            if live[k]["key"] is None:
                live[k]["key"] = live[gone]["key"]
            live[gone]["members"] = []
            alive[gone] = False
            cen[k] = live[k]["sum"] / len(live[k]["members"])
            r = row(k)
            r[~alive] = np.inf
            r[k] = np.inf
            dist[k, :] = r
            dist[:, k] = r
            dist[gone, :] = np.inf
            dist[:, gone] = np.inf
        groups = [g for g in groups if g["members"]]
    out: list[TrackCluster] = []
    for g in groups:
        cen = None if g["sum"] is None else g["sum"] / len(g["members"])
        scored = [(0.0 if cen is None or m.sig is None else sig_distance(m.sig, cen, size_weight), m) for m in g["members"]]
        scored.sort(key=lambda x: (round(x[0], 9), -x[1].n, x[1].h, x[1].part))
        out.append(TrackCluster(0, [m.key for _, m in scored], [round(d, 4) for d, _ in scored], cen,
                                sum(m.n for m in g["members"])))
    out.sort(key=lambda c: (-len(c.keys), -c.images, c.keys[0]))
    for i, c in enumerate(out):
        c.id = i
    return out


def spread_members(keys: Sequence[str], k: int = 4) -> list[str]:
    """Up to k members of a cluster for its tile: the most typical, then spread over the rest
    to the least typical (keys most typical first)."""
    if len(keys) <= k:
        return list(keys)
    return _evenly(list(keys), k)


# -- the analysis ----------------------------------------------------------------

@dataclass
class Analysis:
    """What N10 made of one scan: the parts (flags, read, signature, auto-accepted read), the
    sheet crops per part key (birth, +1 s, mid-life), the stopwatch events, counts, and the
    identity of the scan and settings it was made from (load_analysis compares it)."""
    parts: list[PartInfo]
    crops: dict[str, list[np.ndarray]]
    markers: list[dict[str, Any]]
    stats: dict[str, Any]
    identity: dict[str, Any]

    def by_key(self) -> dict[str, PartInfo]:
        return {p.key: p for p in self.parts}


def _sha1_file(path: Path) -> str | None:
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def scan_identity(out_dir: str | Path) -> dict[str, Any]:
    """What names one finished scan: the SHA-1 of its scan.json and tracks.json (a scan run
    again gets new harvest track ids, so labels of the old one do not apply)."""
    out = Path(out_dir)
    created = None
    try:
        doc = json.loads((out / SCAN_JSON).read_text(encoding="utf-8"))
        created = doc.get("created") if isinstance(doc, Mapping) else None
    except (OSError, ValueError):
        pass
    return {"created": created, "scan_sha1": _sha1_file(out / SCAN_JSON), "tracks_sha1": _sha1_file(out / TRACKS_JSON)}


def analysis_identity(scan: ScanResult, *, deck: Sequence[tuple[str, int]], config: FlagConfig | None = None,
                      table: Any = None, our_deck: Sequence[str] | None = None) -> dict[str, Any]:
    """What an Analysis was made from: the scan (scan_identity), the settings, the declared
    deck, Yodle2's deck and the class table. load_analysis reuses a saved one only when equal."""
    meta = scan.header.get("meta") or {}
    ours = list(our_deck if our_deck is not None else (meta.get("our_deck") or DEFAULT_OUR_DECK))
    return {"schema": ANALYSIS_SCHEMA, "scan": scan_identity(scan.out_dir),
            "config": dataclasses.asdict(config or FlagConfig()),
            "deck": [[str(c), int(f)] for c, f in deck], "our_deck": [str(c) for c in ours],
            "classes_digest": getattr(load_class_table(table), "digest", None)}


def _ground_class(table: Any, cls: str | None, form: int) -> bool:
    if not cls:
        return False
    lab = resolve_label(table, cls)
    if lab.type != TYPE_UNIT or lab.class_id is None:
        return False
    try:
        return not table.classes[lab.class_id].form(int(form)).flying
    except (IndexError, KeyError, AttributeError):
        return False


def _main_cls(info: Mapping[str, Any]) -> str:
    cls = info.get("cls") or {}
    if isinstance(cls, Mapping) and cls:
        return str(max(cls.items(), key=lambda kv: (kv[1], kv[0]))[0])
    return "?"


def flag_part(frames: Sequence[TrackFrame], live_tracks: Mapping[str, Mapping[str, Any]], deck: DeckContext,
              config: FlagConfig | None = None, *, side: str = SIDE_ENEMY, stopwatch: str | None = None,
              split: str | None = None, born_y: float | None = None
              ) -> tuple[dict[str, str], dict[str, Any], list[tuple[str, float]],
                         list[tuple[str, int, int]], dict[str, Any] | None]:
    """(flags, read, harvest classes, live classes, auto-accepted read) of one part's images.
    S1-S5 come from the images and the live tracks they held (live_tracks: tracks.json "live",
    "seg:uid" -> team, how, card, card_name, cls); S6 and S7 are passed in (stopwatch: the
    S6 detail; split: the S7 detail). Live tracks of our team count as no live track (S1);
    the others (enemy, or a team not voted yet) are checked. born_y: where the harvest track
    began (world y, mt).

    Auto-accept also needs none of read["no_auto"] ("<code>: why"): form, the read cannot tell
    the form (DeckContext.form_open); ours, an image held by one of our live tracks; units,
    another enemy live track with another read (a unit born where another died inherits the
    position-only harvest track); side, the detector's side head under accept_side enemy;
    half, a unit born on our half. read["held"]: every live track the part held, as [key, team, class,
    form, images, first n, last n], so a later step can leave out the images of another unit."""
    cfg = config or FlagConfig()
    table = deck.table
    n = len(frames)
    held = Counter(f.live for f in frames if f.live)

    def team(k: str) -> int:
        return int((live_tracks.get(k) or {}).get("team", UNKNOWN_TEAM))

    theirs = {k for k in held if team(k) != OWN_TEAM}
    tf = [f for f in frames if f.live in theirs]
    flags: dict[str, str] = {}
    if not theirs:
        if held:
            ours = sorted({f.live_cls for f in frames if f.live and f.live_cls})
            flags["S1"] = "the live chain made it ours (" + ", ".join(ours or ["?"]) + ")"
        else:
            flags["S1"] = FLAG_TEXT["S1"]
    unexplained, wrong = [], []
    for k in sorted(theirs, key=lambda x: (-held[x], x)):
        info = live_tracks.get(k) or {}
        how, card = info.get("how"), info.get("card")
        kf = [f for f in tf if f.live == k and f.live_cls]
        cls, form = (Counter((f.live_cls, f.live_form) for f in kf).most_common(1)[0][0] if kf
                     else (_main_cls(info), FORM_BASE))
        if how is None or how in UNEXPLAINED_HOWS or (card is not None and int(card) < 0):
            unexplained.append(f"{cls} {how or 'unexplained'}")
        elif resolve_label(table, cls).type == TYPE_UNKNOWN:
            unexplained.append(f"{cls} unknown class")
        why = deck.problem(cls if cls != "?" else None, form, info.get("card_name"))
        if why:
            wrong.append(why)
    if unexplained:
        flags["S2"] = "; ".join(unexplained)
    if wrong:
        flags["S3"] = "; ".join(sorted(set(wrong)))
    reads = Counter((f.live_cls, f.live_form) for f in tf if f.live_cls)
    total = sum(reads.values())
    if reads:
        others = total - reads.most_common(1)[0][1]
        if others >= cfg.s4_min_frames and others / total >= cfg.s4_frac:
            flags["S4"] = " / ".join(f"{c}{'' if f == FORM_BASE else '#' + str(f)} {m}" for (c, f), m in reads.most_common(4))
    water = Counter(f.live_cls for f in tf if f.live_cls and f.live_cls not in RIVER_JUMPERS
                    and _ground_class(table, f.live_cls, f.live_form) and on_water(f.x, f.y))
    if sum(water.values()) >= cfg.s5_min_images:
        flags["S5"] = ", ".join(f"{m} images on the water as {c}" for c, m in water.most_common(3))
    if stopwatch:
        flags["S6"] = stopwatch
    if split:
        flags["S7"] = split
    flags = {k: flags[k] for k in FLAGS if k in flags}

    cand_sum: Counter = Counter()
    for f in frames:
        for c, _s, v in f.cands:
            cand_sum[c] += v
    cands = [(c, round(s / max(1, n), 3)) for c, s in sorted(cand_sum.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
    live_classes = [(c, int(f), int(m)) for (c, f), m in sorted(reads.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))]
    main = Counter(f.live for f in tf).most_common(1)
    info = (live_tracks.get(main[0][0]) or {}) if main else {}
    held_info: list[list[Any]] = []
    for k in sorted(held, key=lambda x: (min(f.n for f in frames if f.live == x), x)):
        kf = [f for f in frames if f.live == k]
        kr = Counter((f.live_cls, f.live_form) for f in kf if f.live_cls)
        kc, kform = kr.most_common(1)[0][0] if kr else (None, FORM_BASE)
        held_info.append([k, team(k), kc, int(kform), len(kf), kf[0].n, kf[-1].n])
    top = live_classes[0] if live_classes else None
    agree = top[2] / max(1, total) if top else 0.0
    e_det, a_det = detector_side(frames)
    read = {"live": top[0] if top else None,
            "form": top[1] if top else FORM_BASE,
            "card": info.get("card_name"), "how": info.get("how"),
            "uid": main[0][0] if main else None, "live_tracks": len(theirs),
            "harvest": cands[0][0] if cands else None, "harvest_score": cands[0][1] if cands else None,
            "cover": round(len(tf) / max(1, n), 3),
            "conf": round(sum(f.live_conf for f in tf) / len(tf), 3) if tf else None,
            "agree": round(agree, 3) if top else None,
            "ours": sorted(k for k in held if k not in theirs),
            "held": held_info, "side_det": [e_det, a_det],
            "form_open": deck.form_open(top[0], top[1]) if top else None}
    typ = resolve_label(table, top[0]).type if top else None
    no_auto: list[str] = []
    if read["form_open"]:
        no_auto.append("form: " + read["form_open"])
    mine = sum(hi[4] for hi in held_info if hi[1] == OWN_TEAM)
    if mine:
        no_auto.append(f"ours: {mine} image(s) held by our live track(s)")
    other = sorted({str(hi[2]) for hi in held_info if hi[1] != OWN_TEAM and top and (hi[2], hi[3]) != (top[0], top[1])})
    if other:
        no_auto.append("units: another enemy live track read " + ", ".join(other))
    if e_det + a_det <= 0.0 or e_det / (e_det + a_det) < cfg.accept_side:
        share = 0.0 if e_det + a_det <= 0.0 else e_det / (e_det + a_det)
        no_auto.append(f"side: the detector's side scores are {share:.0%} enemy")
    if typ == TYPE_UNIT and born_y is not None and float(born_y) >= RIVER_Y_MT[1]:
        no_auto.append("half: born on our half")
    read["no_auto"] = no_auto
    auto = None
    if (not flags and not no_auto and side == SIDE_ENEMY and n >= cfg.accept_min_images
            and read["cover"] >= cfg.accept_cover and (read["conf"] or 0.0) >= cfg.accept_conf and top
            and agree >= cfg.accept_agree and typ in (TYPE_UNIT, TYPE_SPELL)):
        auto = {"cls": top[0], "form": top[1], "type": typ}
    return flags, read, cands, live_classes, auto


def link_markers(clocks: Sequence[Clock], tile_of: Callable[[tuple[int, int] | None], tuple[float, float]], *,
                 gap_s: float = 0.5, link_tiles: float = 1.0, min_frames: int = 2) -> list[MarkerEvent]:
    """Enemy stopwatch boxes joined into deploy events: a box joins the open event whose last
    box is at most gap_s older and whose centre is within link_tiles; at most one box per
    image per event. Events with fewer than min_frames boxes are dropped."""
    events: list[MarkerEvent] = []
    open_: list[MarkerEvent] = []
    for c in sorted(clocks, key=lambda c: (c.n, -c.conf)):
        tw, th = tile_of(c.size)
        cx, cy = (c.box[0] + c.box[2]) / 2.0, (c.box[1] + c.box[3]) / 2.0
        open_ = [e for e in open_ if c.t - e.clocks[-1].t <= gap_s]
        best, best_d = None, None
        for e in open_:
            last = e.clocks[-1]
            if last.n == c.n:
                continue
            lx, ly = (last.box[0] + last.box[2]) / 2.0, (last.box[1] + last.box[3]) / 2.0
            d = math.hypot((cx - lx) / max(tw, 1e-6), (cy - ly) / max(th, 1e-6))
            if d <= link_tiles and (best_d is None or d < best_d):
                best, best_d = e, d
        if best is None:
            best = MarkerEvent()
            events.append(best)
            open_.append(best)
        best.clocks.append(c)
    return [e for e in events if len(e.clocks) >= max(1, min_frames)]


def _under(clock: Clock, box: Sequence[float], tw: float, th: float) -> bool:
    """A track box is under a stopwatch: its centre within the stopwatch's columns (grown by
    half a tile) and between the stopwatch's top and 2.5 tiles below its bottom."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return (clock.box[0] - 0.5 * tw <= cx <= clock.box[2] + 0.5 * tw
            and clock.box[1] <= cy <= clock.box[3] + 2.5 * th)


def match_markers(events: Sequence[MarkerEvent], frames: Mapping[int, Sequence[TrackFrame]],
                  live_tracks: Mapping[str, Mapping[str, Any]], plays: Sequence[Mapping[str, Any]],
                  tile_of: Callable[[tuple[int, int] | None], tuple[float, float]],
                  config: FlagConfig | None = None) -> None:
    """Fill each event's tracks under it and whether the live chain matched it: a track under
    it held an enemy (or undecided) live track that is part of a play (PLAY_HOWS), or the live
    chain reported an opponent play from marker_before_s before to marker_after_s after it,
    within marker_play_tiles of a track under it (any position when the play has none)."""
    cfg = config or FlagConfig()
    at: dict[int, list[tuple[int, TrackFrame]]] = {}
    for h, fs in frames.items():
        for f in fs:
            at.setdefault(f.n, []).append((int(h), f))
    for ev in events:
        for c in ev.clocks:
            tw, th = tile_of(c.size)
            for h, f in at.get(c.n, ()):
                if _under(c, f.box, tw, th):
                    ev.under.setdefault(h, set()).add(c.n)
        for h in sorted(ev.under):
            for f in frames.get(h, ()):
                info = live_tracks.get(f.live) if f.live else None
                if info and info.get("team") != OWN_TEAM and info.get("how") in PLAY_HOWS:
                    ev.matched, ev.why = True, f"live track {f.live} is a {info.get('how')}"
                    break
            if ev.matched:
                break
        if ev.matched:
            continue
        spots = [(f.x, f.y) for h in ev.under for f in frames.get(h, ()) if f.n in ev.under[h]]
        for p in plays:
            if not ev.t0 - cfg.marker_before_s <= float(p["t"]) <= ev.t1 + cfg.marker_after_s:
                continue
            if p.get("x_mt") is None or p.get("y_mt") is None:
                ev.matched, ev.why = True, f"play {p.get('card_name')} at {float(p['t']):.1f} s"
                break
            near = any(math.hypot(float(p["x_mt"]) - x, float(p["y_mt"]) - y) <= cfg.marker_play_tiles * 1000.0
                       for x, y in spots)
            if near:
                ev.matched, ev.why = True, f"play {p.get('card_name')} at {float(p['t']):.1f} s"
                break
        if not ev.matched:
            ev.why = "no live play" if ev.under else "no live play and no track under it"


def sheet_crop(rgb: np.ndarray, box: Sequence[float], cell: int = 72, pad_frac: float = 0.3,
               outline: bool = True, colour: Sequence[int] | None = None) -> np.ndarray:
    """A cell x cell RGB tile of the box on one image: the box grown by pad_frac of its larger
    side, scaled (nearest neighbour, aspect kept) and centred, the box then drawn in yellow
    (BOX_COLOUR, or `colour`; after scaling, so a 1-pixel line is never lost between samples)."""
    line = tuple(int(v) for v in (BOX_COLOUR if colour is None else colour))[:3]
    img = np.asarray(rgb)[..., :3]
    h, w = img.shape[:2]
    bw, bh = float(box[2]) - float(box[0]), float(box[3]) - float(box[1])
    pad = pad_frac * max(bw, bh) + (2.0 if outline else 0.0)
    x0, y0 = max(0, int(math.floor(float(box[0]) - pad))), max(0, int(math.floor(float(box[1]) - pad)))
    x1, y1 = min(w, int(math.ceil(float(box[2]) + pad))), min(h, int(math.ceil(float(box[3]) + pad)))
    out = np.empty((cell, cell, 3), dtype=np.uint8)
    out[:] = CELL_BG
    if x1 - x0 < 1 or y1 - y0 < 1:
        return out
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    s = min(cell / ch, cell / cw)
    th, tw = max(1, min(cell, int(round(ch * s)))), max(1, min(cell, int(round(cw * s))))
    ys = np.minimum(ch - 1, ((np.arange(th) + 0.5) * ch / th).astype(int))
    xs = np.minimum(cw - 1, ((np.arange(tw) + 0.5) * cw / tw).astype(int))
    oy, ox = (cell - th) // 2, (cell - tw) // 2
    out[oy:oy + th, ox:ox + tw] = crop[ys][:, xs]
    if outline:
        bx0 = ox + int(math.floor((float(box[0]) - x0) * tw / cw))
        bx1 = ox + int(math.ceil((float(box[2]) - x0) * tw / cw)) - 1
        by0 = oy + int(math.floor((float(box[1]) - y0) * th / ch))
        by1 = oy + int(math.ceil((float(box[3]) - y0) * th / ch)) - 1
        bx0, bx1 = max(ox, min(ox + tw - 1, bx0)), max(ox, min(ox + tw - 1, bx1))
        by0, by1 = max(oy, min(oy + th - 1, by0)), max(oy, min(oy + th - 1, by1))
        if bx0 <= bx1 and by0 <= by1:
            out[by0, bx0:bx1 + 1] = line
            out[by1, bx0:bx1 + 1] = line
            out[by0:by1 + 1, bx0] = line
            out[by0:by1 + 1, bx1] = line
    return out


def crop_frames(frames: Sequence[TrackFrame]) -> list[TrackFrame]:
    """The sheet's images of a part: at birth, the first at least 1 s later (else the last),
    and the one nearest mid-life."""
    if not frames:
        return []
    first = frames[0]
    plus = next((f for f in frames if f.t >= first.t + 1.0), frames[-1])
    mid_t = (first.t + frames[-1].t) / 2.0
    mid = min(frames, key=lambda f: (abs(f.t - mid_t), f.n))
    return [first, plus, mid]


def analyze_scan(scan: ScanResult, *, deck: Sequence[tuple[str, int]], table: Any = None,
                 geometry: ArenaGeometry | None = None, our_deck: Sequence[str] | None = None,
                 session_dir: str | Path | None = None, config: FlagConfig | None = None,
                 frames: Iterable[Mapping[str, Any]] | None = None,
                 backgrounds: Mapping[str, np.ndarray] | None = None,
                 load: Callable[[Path], np.ndarray] = load_image, log: Callable[[str], Any] = print) -> Analysis:
    """N10 over one finished scan (module doc, N10). deck: the declared deck as (catalogue
    card, slot form); our_deck: Yodle2's cards (default the scan's meta, else Hog 2.6);
    session_dir: where the scan's image paths start (default the scan's session_dir).
    frames / backgrounds / load stand in for scan_frames.jsonl, the epoch backgrounds and
    the image reader (tests)."""
    cfg = config or FlagConfig()
    table = load_class_table(table)
    geo = geometry if geometry is not None else ArenaGeometry.default()
    meta = scan.header.get("meta") or {}
    ours = list(our_deck if our_deck is not None else (meta.get("our_deck") or DEFAULT_OUR_DECK))
    session = Path(session_dir if session_dir is not None else scan.header.get("session_dir") or scan.out_dir)
    deck = [(str(c), int(f)) for c, f in deck]
    ctx = DeckContext.build(table, deck)
    own_labels = table.deck_labels(ours)
    live = scan.live_tracks
    stats: Counter = Counter()
    sides: dict[int, str] = {}
    meta_of: dict[int, Mapping[str, Any]] = {}
    for tr in scan.tracks:
        h = int(tr["h"])
        if int(tr.get("n", 0)) < cfg.min_images:
            stats["short"] += 1
            continue
        side, _e, _a = track_side(tr, live, own_labels)
        if side == SIDE_ALLY:
            stats["ally"] += 1
            continue
        sides[h], meta_of[h] = side, tr
        stats[f"side_{side}"] += 1
    by_h, clocks, plays = collect_frames(frames if frames is not None else scan.frames(), sides,
                                         marker_conf=cfg.marker_conf)
    tiles: dict[tuple[int, int] | None, tuple[float, float]] = {}

    def tile_of(size: tuple[int, int] | None) -> tuple[float, float]:
        if size not in tiles:
            try:
                tiles[size] = tile_px(geo, size if size is not None else geo.frame_size)
            except Exception:
                tiles[size] = (1.0, 1.0)
        return tiles[size]

    bgs: dict[str, np.ndarray | None] = dict(backgrounds or {})

    def background(epoch: str | None) -> np.ndarray | None:
        if epoch is None:
            return None
        if epoch not in bgs:
            try:
                bgs[epoch] = scan.background(epoch)
            except Exception:
                bgs[epoch] = None
        return bgs[epoch]

    # signatures of every image of a track, or of sig_max_frames evenly spread ones (each
    # image read once)
    picks: dict[int, list[int]] = {h: _evenly(list(range(len(fs))), cfg.sig_max_frames) for h, fs in by_h.items()}
    sigs: dict[int, dict[int, np.ndarray]] = {h: {} for h in by_h}

    def read_sigs(want_: Mapping[int, Iterable[int]]) -> None:
        need: dict[str, list[tuple[int, int]]] = {}
        for h_, idx_ in want_.items():
            for i_ in idx_:
                f_ = by_h[h_][i_]
                if f_.image and i_ not in sigs[h_]:
                    need.setdefault(f_.image, []).append((h_, i_))
        for image in sorted(need, key=lambda im: min(by_h[h_][i_].n for h_, i_ in need[im])):
            try:
                rgb = np.asarray(load(session / image))[..., :3]
            except Exception:
                stats["image_errors"] += 1
                continue
            stats["images_read"] += 1
            for h_, i_ in need[image]:
                f_ = by_h[h_][i_]
                sg = appearance(rgb, f_.box, background(f_.epoch), threshold=cfg.fg_threshold,
                                tile=tile_of((rgb.shape[1], rgb.shape[0])))
                if sg is not None:
                    sigs[h_][i_] = sg

    read_sigs(picks)

    # the S7 split: cuts on the sampled signatures, then (a long track is sampled sparsely)
    # each cut moved to the image between its two neighbouring samples that best separates
    # the two looks (refine_cut), from one more read of those images
    found: dict[int, list[tuple[int, int, np.ndarray, np.ndarray]]] = {}
    for h, fs in by_h.items():
        idx = [i for i in picks[h] if i in sigs[h]]
        if len(fs) < 2 * cfg.split_min_images or len(idx) < 2:
            continue
        min_part = max(3, int(math.ceil(cfg.split_min_images * len(idx) / max(1, len(fs)))))
        cuts = find_splits([sigs[h][i] for i in idx], min_part=min_part, split_dist=cfg.split_dist,
                           split_ratio=cfg.split_ratio, size_weight=cfg.size_weight, max_parts=cfg.max_parts)
        if not cuts:
            continue
        marks = [0] + list(cuts) + [len(idx)]
        found[h] = [(idx[c - 1], idx[c], mean_signature([sigs[h][idx[i]] for i in range(marks[j], c)]),
                     mean_signature([sigs[h][idx[i]] for i in range(c, marks[j + 2])]))
                    for j, c in enumerate(cuts)]
    read_sigs({h: [i for lo, hi, _a, _b in plan for i in range(lo + 1, hi)] for h, plan in found.items()})
    cuts_of: dict[int, list[int]] = {}
    for h, plan in found.items():
        fs = by_h[h]
        starts = [refine_cut(sigs[h], lo, hi, a, b, cfg.size_weight) for lo, hi, a, b in plan]
        bounds = [fs[i].n for i in starts]
        pieces = _cut(fs, bounds)
        if len(pieces) > 1 and all(len(p) >= cfg.split_min_images for p in pieces):
            cuts_of[h] = bounds
            stats["split_tracks"] += 1
            stats["split_cuts_moved"] += sum(1 for i, (_lo, hi, _a, _b) in zip(starts, plan) if i != hi)

    # the stopwatches (S6)
    events = link_markers(clocks, tile_of, gap_s=cfg.marker_gap_s, link_tiles=cfg.marker_link_tiles,
                          min_frames=cfg.marker_min_frames)
    match_markers(events, by_h, live, plays, tile_of, cfg)
    unmatched: dict[int, list[tuple[set[int], str]]] = {}
    for ev in events:
        stats["markers"] += 1
        if ev.matched:
            stats["markers_matched"] += 1
            continue
        if not ev.under:
            stats["markers_orphan"] += 1
        for h, ns in ev.under.items():
            unmatched.setdefault(h, []).append((ns, f"stopwatch at {ev.t0:.1f} s, {ev.why}"))

    # the parts, each on its own side (an ally part of a track is ours: left out, as ally tracks)
    parts: list[PartInfo] = []
    part_frames: dict[str, list[TrackFrame]] = {}
    for h in sorted(by_h):
        fs = by_h[h]
        if not fs:
            stats["no_frames"] += 1
            continue
        pieces = _cut(fs, cuts_of.get(h, []))
        for j, pf in enumerate(pieces):
            side = sides[h] if len(pieces) == 1 else frames_side(pf, live, own_labels)[0]
            if side == SIDE_ALLY:
                stats["ally_parts"] += 1
                continue
            ns = {f.n for f in pf}
            sw = "; ".join(why for marked, why in unmatched.get(h, ()) if marked & ns)
            split = None
            if len(pieces) > 1:
                split = f"part {j + 1} of {len(pieces)}: images {pf[0].n}-{pf[-1].n}"
            flags, read, cands, live_classes, auto = flag_part(pf, live, ctx, cfg, side=side, stopwatch=sw or None,
                                                               split=split, born_y=fs[0].y)
            idx = [i for i in sorted(sigs[h]) if pf[0].n <= fs[i].n <= pf[-1].n]
            sig = mean_signature([sigs[h][i] for i in idx]).tolist() if idx else None
            p = PartInfo(h, j, len(pieces), meta_of[h].get("hk"), side, len(pf), pf[0].n, pf[-1].n,
                         pf[0].t, pf[-1].t, flags, read, cands, live_classes, sig, auto,
                         (int(round(pf[0].x)), int(round(pf[0].y))), [f.n for f in crop_frames(pf)])
            parts.append(p)
            part_frames[p.key] = pf
            for k in flags:
                stats[f"flag_{k}"] += 1
            stats["auto" if auto else ("flagged" if flags else "weak")] += 1

    # the sheet crops (only the images they need)
    crops: dict[str, list[np.ndarray]] = {}
    want: dict[str, list[tuple[str, int, TrackFrame]]] = {}
    for p in parts:
        cf = crop_frames(part_frames[p.key])
        crops[p.key] = [np.full((cfg.cell, cfg.cell, 3), CELL_BG, np.uint8) for _ in cf]
        for j, f in enumerate(cf):
            if f.image:
                want.setdefault(f.image, []).append((p.key, j, f))
    for image in sorted(want):
        try:
            rgb = np.asarray(load(session / image))[..., :3]
        except Exception:
            stats["image_errors"] += 1
            continue
        for key, j, f in want[image]:
            crops[key][j] = sheet_crop(rgb, f.box, cfg.cell, cfg.crop_pad)
    stats["parts"] = len(parts)
    stats["tracks"] = len(by_h)
    identity = analysis_identity(scan, deck=deck, config=cfg, table=table, our_deck=ours)
    log(f"  analysis: {len(by_h)} enemy or unknown-side tracks ({stats['ally']} ours, {stats['short']} under "
        f"{cfg.min_images} images left out), {len(parts)} parts, {stats['auto']} auto-accepted, "
        f"{stats['flagged']} flagged, {stats['weak']} weak; stopwatches {stats['markers']} "
        f"({stats['markers_matched']} matched)")
    return Analysis(parts, crops, [ev.to_dict() for ev in events], dict(stats), identity)


def refine_cut(sigs: Mapping[int, Sequence[float]], lo: int, hi: int, left: Sequence[float], right: Sequence[float],
               size_weight: float = 0.5) -> int:
    """Where a part starts, between two sampled images lo < hi of a track (find_splits put the
    cut at hi): the image b in (lo, hi] that leaves the fewest images on the wrong side, an
    image being on the right's side when its signature is nearer `right`'s mean than
    `left`'s. Images without a signature do not count; ties go to the later b (hi itself when
    every image was sampled)."""
    window = [i for i in range(lo + 1, hi) if i in sigs]
    nearer_right = {i: sig_distance(sigs[i], right, size_weight) < sig_distance(sigs[i], left, size_weight)
                    for i in window}
    best, best_err = hi, None
    for b in range(hi, lo, -1):
        err = sum(1 for i in window if (i < b) == nearer_right[i])
        if best_err is None or err < best_err:
            best, best_err = b, err
    return best


def _cut(frames: Sequence[TrackFrame], bounds: Sequence[int]) -> list[list[TrackFrame]]:
    """Frames split where image n reaches each bound (a part starts at each bound)."""
    out: list[list[TrackFrame]] = [[]]
    bs = sorted(bounds)
    j = 0
    for f in frames:
        while j < len(bs) and f.n >= bs[j]:
            out.append([])
            j += 1
        out[-1].append(f)
    return [p for p in out if p]


def save_analysis(analysis: Analysis, out_dir: str | Path) -> Path:
    """analysis.json and crops.npz in out_dir (the store's sheet folder)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    parts = []
    for i, p in enumerate(analysis.parts):
        d = p.to_dict()
        d["crops"] = []
        for j, c in enumerate(analysis.crops.get(p.key, ())):
            name = f"p{i}_{j}"
            arrays[name] = np.ascontiguousarray(c, dtype=np.uint8)
            d["crops"].append(name)
        parts.append(d)
    with open(out / (ANALYSIS_CROPS + ".tmp"), "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(out / (ANALYSIS_CROPS + ".tmp"), out / ANALYSIS_CROPS)
    doc = {"schema": ANALYSIS_SCHEMA, "identity": analysis.identity, "stats": analysis.stats,
           "markers": analysis.markers, "parts": parts}
    tmp = out / (ANALYSIS_JSON + ".tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, out / ANALYSIS_JSON)
    return out / ANALYSIS_JSON


def load_analysis(out_dir: str | Path, identity: Mapping[str, Any] | None = None) -> Analysis | None:
    """The analysis saved in out_dir, or None when there is none, it cannot be read, or its
    identity differs from `identity` (another scan, deck or settings)."""
    out = Path(out_dir)
    try:
        doc = json.loads((out / ANALYSIS_JSON).read_text(encoding="utf-8"))
        if doc.get("schema") != ANALYSIS_SCHEMA:
            return None
        if identity is not None and json.loads(json.dumps(identity)) != doc.get("identity"):
            return None
        parts, crops = [], {}
        with np.load(out / ANALYSIS_CROPS, allow_pickle=False) as z:
            for d in doc.get("parts") or ():
                p = PartInfo.from_dict(d)
                parts.append(p)
                crops[p.key] = [np.asarray(z[name]) for name in d.get("crops") or ()]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return Analysis(parts, crops, list(doc.get("markers") or []), dict(doc.get("stats") or {}),
                    dict(doc.get("identity") or {}))


# -- N11: propagate and emit ----------------------------------------------------------
#
# tools/vision_autolabel.py `emit` drives these (module doc, N11). Each part of the sheet, as
# tracks.json decided it, gets a box on each of its images (propagate_parts); each battle
# image then gets a plan (plan_frame: the boxes to write, the regions to mask, or why it is
# dropped); the pixels (feather_paste, crop_image) and the YOLO rows (to_yolo, frame_yolo)
# follow from the plan.

DECISION_LABEL, DECISION_MASK, DECISION_EFFECT, DECISION_OURS, DECISION_PENDING = (
    "label", "mask", "effect", "ours", "pending")
DECISIONS = (DECISION_LABEL, DECISION_MASK, DECISION_EFFECT, DECISION_OURS, DECISION_PENDING)
#: tracks.json codes (tools/vision_autolabel.py CODES), restated so this needs no tool import.
CODE_EFFECT, CODE_OURS, CODE_UNSURE = "x", "o", "?"
#: Where a box's name comes from: the owner's map, an auto-accepted read, a bad mark (masked),
#: a part nobody decided yet (masked), tonight's live model (pseudo-labels, plan risk 4).
SRC_OWNER, SRC_AUTO, SRC_BAD, SRC_PENDING, SRC_LIVE = "owner", "auto", "bad", "pending", "live"
#: How a box was made: a smoothed proposal, a short gap filled, a spell circle, a live box.
HOW_SMOOTH, HOW_FILL, HOW_CIRCLE, HOW_LIVE = "smooth", "fill", "circle", "live"
ANCHOR_BOTTOM, ANCHOR_CENTRE = "bottom", "centre"
#: Proposal box sources whose size smoothing trusts (a blob's box may hold a shadow or a bar).
SIZE_TAGS = frozenset({"live", "det"})
DROP_MASKS, DROP_OVERLAP, DROP_NO_BACKGROUND = "masks", "overlap", "no_background"
#: Live detection types a decided part's box covers by overlap; an object or UI box on a part
#: is not the part (KataCR's "dirt", a unit's HP bar) and keeps its own pseudo-label.
COVER_TYPES = frozenset({TYPE_UNIT, TYPE_SPELL, TYPE_UNKNOWN})
#: Kept box types a mask must not paste over: they move, so the epoch background lacks them.
#: A tower is in the background, so a mask over it shows the tower again.
OVERLAP_TYPES = frozenset({TYPE_UNIT, TYPE_SPELL, TYPE_OBJECT, TYPE_UNKNOWN})
_BIRTH_EPS = 1e-6
MT_PER_TILE = 1000.0
Box4 = tuple[float, float, float, float]


@dataclass(frozen=True)
class EmitConfig:
    """N11 settings (tools/vision_autolabel.py emit). Images are counted by scan record n;
    tiles are measured at the board's centre."""
    stride: int = 3                 # images with an enemy label: keep 1 in `stride` (by scan record n)
    empty_stride: int = 15          # images with none: keep 1 in this many (Yodle2's units, towers, UI)
    birth_s: float = 1.0            # every image up to this long after a labelled part's birth is kept
    birth_repeat: int = 2           # copies of a birth-window image in a training mix (dev and gold: 1)
    smooth_window: int = 5          # box size: median over this many of the part's images (odd)
    max_gap: int = 3                # fill up to this many missing images between two of a part's images
    min_box_px: float = 3.0         # a clipped box narrower or shorter than this is dropped
    live_iou: float = 0.5           # a live detection this close to a decided part's box is that part's
    mask_pad_tiles: float = 0.15    # a mask grows by this much; the background is pasted inside it
    feather_px: int = 3             # ... and blends back to the image over this many pixels outside it
    max_masks: int = 3              # an image with more masks than this is dropped
    mask_overlap_frac: float = 0.05  # ... so is one whose grown mask covers more of a kept moving box
    mask_leftovers: bool = True     # mask detector proposals of tracks no decided part holds (plan 2.3 step 5)
    leftover_conf: float = 0.15     # ... at this score or more (ScanConfig.bg_mask_conf: what the background leaves out)
    image_format: str = "jpg"       # record.save_image formats; tests pass "npy"
    quality: int = 95
    spell_radius_tiles: tuple[tuple[str, float], ...] = (("DarkMagic", 2.5), ("Vines", 2.5), ("GoblinCurse", 3.0))

    def __post_init__(self) -> None:
        for name in ("stride", "empty_stride", "birth_repeat", "smooth_window"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"EmitConfig.{name} must be at least 1")
        for name in ("max_gap", "feather_px", "max_masks"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"EmitConfig.{name} must be >= 0")
        if int(self.smooth_window) % 2 == 0:
            raise ValueError("EmitConfig.smooth_window must be odd (a centred window)")
        for name in ("live_iou", "mask_overlap_frac", "leftover_conf"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"EmitConfig.{name} must be in [0, 1]")
        for name in ("birth_s", "min_box_px", "mask_pad_tiles"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"EmitConfig.{name} must be >= 0")
        if self.image_format not in IMAGE_FORMATS:
            raise ValueError(f"EmitConfig.image_format must be one of {', '.join(IMAGE_FORMATS)}")
        if not 1 <= int(self.quality) <= 100:
            raise ValueError("EmitConfig.quality must be in [1, 100]")
        radii = []
        for item in self.spell_radius_tiles:
            name, r = item
            if not str(name) or float(r) <= 0.0:
                raise ValueError(f"EmitConfig.spell_radius_tiles: {item!r} is not (spell label, radius > 0)")
            radii.append((str(name), float(r)))
        object.__setattr__(self, "spell_radius_tiles", tuple(radii))


def live_key(seg: Any, uid: Any) -> str:
    """"<seg>:<uid>", how tracks.json "live" names a live track."""
    return f"{int(seg)}:{int(uid)}"


def _box(b: Sequence[float]) -> Box4:
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def clip_box(box: Sequence[float], rect: Sequence[float]) -> Box4 | None:
    """`box` inside `rect` (both x1, y1, x2, y2); None when nothing of it is left."""
    x1, y1 = max(float(box[0]), float(rect[0])), max(float(box[1]), float(rect[1]))
    x2, y2 = min(float(box[2]), float(rect[2])), min(float(box[3]), float(rect[3]))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def grow_box(box: Sequence[float], pad: float) -> Box4:
    p = float(pad)
    return (float(box[0]) - p, float(box[1]) - p, float(box[2]) + p, float(box[3]) + p)


def cover_frac(region: Sequence[float], box: Sequence[float]) -> float:
    """The share of `box`'s area inside `region`."""
    a = box_area(box)
    return box_inter(region, box) / a if a > 0 else 0.0


@dataclass(slots=True)
class PropRef:
    """One proposal of a scan record (Proposal.to_dict, the fields emit needs): its harvest
    track h, box (frame px), source, type, the live detection that joined it ("seg:uid"),
    its candidates, its world position and its score."""
    h: int | None
    box: Box4
    src: str
    type: str
    live: str | None
    cands: list[tuple[str, int | None, float]] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    conf: float = 0.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PropRef":
        live = d.get("live")
        return cls(None if d.get("h") is None else int(d["h"]), _box(d["box"]), str(d.get("src", "")),
                   str(d.get("type", TYPE_UNIT)), None if live is None else live_key(live[0], live[1]),
                   [(str(c), None if s is None else int(s), float(v)) for c, s, v in d.get("cands") or ()],
                   float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("conf") or 0.0))


def is_leftover(src: Any, typ: Any, conf: Any, h: Any, min_conf: float) -> bool:
    """Whether a proposal is one plan_frame masks when no decided part holds its track there
    (EmitConfig.mask_leftovers): a tracked detector proposal ("det", "det+blob") of a moving
    type at min_conf or more. Bare blobs and live-only proposals are not (the live layer has
    those), and neither are objects."""
    return (h is not None and "det" in str(src).split("+") and str(typ) in COVER_TYPES
            and float(conf or 0.0) >= float(min_conf))


@dataclass(slots=True)
class FrameRec:
    """One battle-screen scan record (hk and size set): the live detections and the
    proposals emit needs (props: proposal k -> PropRef)."""
    n: int
    i: int
    image: str | None
    t: float
    hk: int
    seg: int
    epoch: str | None
    size: tuple[int, int]
    live: list[LiveDet] = field(default_factory=list)
    props: dict[int, PropRef] = field(default_factory=dict)

    def live_for(self, key: str | None) -> LiveDet | None:
        """The live detection of live track `key` ("seg:uid") on this image, or None."""
        if not key:
            return None
        seg, _, uid = str(key).partition(":")
        try:
            s, u = int(seg), int(uid)
        except ValueError:
            return None
        for d in self.live:
            if d.uid == u and d.seg == s:
                return d
        return None


def load_frame_index(records: Iterable[Mapping[str, Any]], wanted: Iterable[int] | None = None, *,
                     leftover_conf: float | None = None) -> dict[int, FrameRec]:
    """scan record n -> FrameRec for every battle-screen record, in one streaming pass. wanted:
    keep only the proposals of these harvest tracks (the decided parts'), so a long scan
    stays small in memory, plus, with leftover_conf, every other proposal plan_frame may mask
    as a leftover (is_leftover); the live detections are always kept."""
    want = None if wanted is None else {int(h) for h in wanted}
    out: dict[int, FrameRec] = {}
    for rec in records:
        if rec.get("hk") is None or not rec.get("size"):
            continue
        seg = int(rec.get("seg", 0))
        props: dict[int, PropRef] = {}
        for k, p in enumerate(rec.get("props") or ()):
            h = p.get("h")
            if want is not None and (h is None or int(h) not in want):
                if leftover_conf is None or not is_leftover(p.get("src"), p.get("type", TYPE_UNIT), p.get("conf"), h,
                                                            leftover_conf):
                    continue
            props[k] = PropRef.from_dict(p)
        n = int(rec["n"])
        out[n] = FrameRec(n, int(rec.get("i", n)), rec.get("image"), float(rec.get("t", 0.0)), int(rec["hk"]), seg,
                          rec.get("epoch"), (int(rec["size"][0]), int(rec["size"][1])),
                          [LiveDet.from_row(r, seg) for r in rec.get("live") or ()], props)
    return out


def _part_order(key: str) -> tuple[int, str]:
    return int(str(key).split(".")[0]), str(key)


@dataclass
class PartPlan:
    """One part of the sheet as emit writes it: its decision and its box per image (frame
    pixels, how it was made). side: 1 enemy (every label), 0 ours (o), None for the rest."""
    key: str
    h: int
    part: int
    decision: str
    label: str | None
    side: int | None
    src: str
    type: str
    first_n: int
    last_n: int
    first_t: float
    last_t: float
    hk: int | None
    uid: str | None
    boxes: dict[int, tuple[Box4, str]] = field(default_factory=dict)
    props: dict[int, PropRef] = field(default_factory=dict)
    birth_xy: tuple[float, float] | None = None
    last_xy: tuple[float, float] | None = None
    skipped: int = 0                         # images left out: held by one of our live tracks
    card: str | None = None                  # set by vision_autolabel (card_of_label: the deck's card)
    form: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """A parts.jsonl line (boxes in frame pixels at 1 decimal)."""
        return {"key": self.key, "h": self.h, "part": self.part, "decision": self.decision, "label": self.label,
                "side": self.side, "src": self.src, "type": self.type, "card": self.card, "form": self.form,
                "first_n": self.first_n, "last_n": self.last_n, "first_t": round(self.first_t, 4),
                "last_t": round(self.last_t, 4), "hk": self.hk, "uid": self.uid,
                "birth": None if self.birth_xy is None else [int(round(v)) for v in self.birth_xy],
                "last": None if self.last_xy is None else [int(round(v)) for v in self.last_xy],
                "skipped": self.skipped,
                "boxes": [[n, _r1(b[0]), _r1(b[1]), _r1(b[2]), _r1(b[3]), how]
                          for n, (b, how) in sorted(self.boxes.items())]}


def decide_parts(sheet_parts: Mapping[str, Mapping[str, Any]], tracks_doc: Mapping[str, Any] | None,
                 bad: Iterable[str] = ()) -> dict[str, tuple[str, str | None, str]]:
    """key -> (decision, label, src) for every part of the sheet (sheet.json "tracks", the
    analysis' universe): a tracks.json label -> (label, the label, its source); a key in
    `bad` (marked bad and not relabelled since) or code ? -> mask; x -> effect; o -> ours;
    on the sheet but not in tracks.json, or in its "split" list -> pending."""
    bad_keys = {str(k) for k in bad}
    entries = {str(t.get("key")): t for t in (tracks_doc or {}).get("tracks") or ()
               if isinstance(t, Mapping) and t.get("key") is not None}
    split = {str(k) for k in (tracks_doc or {}).get("split") or ()}
    out: dict[str, tuple[str, str | None, str]] = {}
    for key in sorted(sheet_parts, key=_part_order):
        e = entries.get(key)
        if e is None or key in split:
            out[key] = (DECISION_PENDING, None, SRC_PENDING)
        elif key in bad_keys:
            out[key] = (DECISION_MASK, None, SRC_BAD)
        elif isinstance(e.get("label"), str) and e["label"]:
            out[key] = (DECISION_LABEL, str(e["label"]), str(e.get("source") or SRC_OWNER))
        elif e.get("code") == CODE_UNSURE:
            out[key] = (DECISION_MASK, None, SRC_OWNER)
        elif e.get("code") == CODE_EFFECT:
            out[key] = (DECISION_EFFECT, None, SRC_OWNER)
        elif e.get("code") == CODE_OURS:
            out[key] = (DECISION_OURS, None, SRC_OWNER)
        else:
            out[key] = (DECISION_PENDING, None, SRC_PENDING)
    return out


def _params(box: Sequence[float], anchor: str) -> tuple[float, float, float, float]:
    """(centre x, anchor y, width, height): the anchor is the bottom edge (a unit stands on
    its feet: KataCR's convention) or the centre (a spell)."""
    x1, y1, x2, y2 = _box(box)
    ay = y2 if anchor == ANCHOR_BOTTOM else (y1 + y2) / 2.0
    return (x1 + x2) / 2.0, ay, x2 - x1, y2 - y1


def _from_params(cx: float, ay: float, w: float, h: float, anchor: str) -> Box4:
    if anchor == ANCHOR_BOTTOM:
        return (cx - w / 2.0, ay - h, cx + w / 2.0, ay)
    return (cx - w / 2.0, ay - h / 2.0, cx + w / 2.0, ay + h / 2.0)


def smooth_boxes(seq: Sequence[tuple[int, Sequence[float], str]], window: int = 5,
                 anchor: str = ANCHOR_BOTTOM) -> list[tuple[int, Box4]]:
    """A part's boxes, (n, box, source tag) in image order, with steadier sizes: width and
    height are the median over the centred `window` of the images whose box came from a live
    or harvest detection (SIZE_TAGS); with none in the window, over all such images of the
    part; with none at all, over the window's blob boxes. Each image keeps its own anchor
    (bottom centre, or the centre for a spell)."""
    items = [(int(n), _box(b), str(tag)) for n, b, tag in seq]
    half = max(1, int(window)) // 2
    trusted = [j for j, it in enumerate(items) if it[2] in SIZE_TAGS]
    out: list[tuple[int, Box4]] = []
    for i, (n, b, _tag) in enumerate(items):
        lo, hi = max(0, i - half), min(len(items), i + half + 1)
        pick = [j for j in range(lo, hi) if items[j][2] in SIZE_TAGS] or trusted or list(range(lo, hi))
        w = float(np.median([items[j][1][2] - items[j][1][0] for j in pick]))
        h = float(np.median([items[j][1][3] - items[j][1][1] for j in pick]))
        cx, ay, _w, _h = _params(b, anchor)
        out.append((n, _from_params(cx, ay, w, h, anchor)))
    return out


def fill_gaps(seq: Sequence[tuple[int, Sequence[float]]], record_ns: Sequence[int], max_gap: int = 3, *,
              anchor: str = ANCHOR_BOTTOM, hk_of: Mapping[int, Any] | None = None,
              skip: Iterable[int] = ()) -> list[tuple[int, Box4]]:
    """Boxes for the records missing between two consecutive images n_a < n_b of a part:
    when 1 to `max_gap` battle records (record_ns) lie strictly between them, their centre x,
    anchor y, width and height are interpolated linearly by record order. Never across a
    battle change (hk_of: record -> harvest battle), and never over a record in `skip` (an
    image of the part's track that one of our units held)."""
    rs = sorted(int(n) for n in record_ns)
    banned = {int(n) for n in skip}
    items = sorted((int(n), _box(b)) for n, b in seq)
    out: list[tuple[int, Box4]] = []
    for (na, ba), (nb, bb) in zip(items, items[1:]):
        between = rs[bisect.bisect_right(rs, na):bisect.bisect_left(rs, nb)]
        if not between or len(between) > int(max_gap) or banned.intersection(between):
            continue
        if hk_of is not None:
            hk = hk_of.get(na)
            if hk is None or hk_of.get(nb) != hk or any(hk_of.get(m) != hk for m in between):
                continue
        pa, pb = _params(ba, anchor), _params(bb, anchor)
        k = len(between) + 1
        for j, m in enumerate(between, 1):
            f = j / k
            out.append((m, _from_params(*(a + (b - a) * f for a, b in zip(pa, pb)), anchor)))
    return out


def circle_box(center_mt: Sequence[float], size: Sequence[int], geometry: ArenaGeometry,
               radius_tiles: float) -> Box4:
    """The pixel box of a spell's circle of `radius_tiles` around world `center_mt`: the
    extremes of world_to_px at x -/+ r and y -/+ r (the geometry is a homography: an
    ellipse on screen, whose box this is for an affine one)."""
    r = float(radius_tiles) * MT_PER_TILE
    cx, cy = float(center_mt[0]), float(center_mt[1])
    xs = np.array([cx - r, cx + r, cx, cx], dtype=np.float64)
    ys = np.array([cy, cy, cy - r, cy + r], dtype=np.float64)
    px, py = geometry.world_to_px(xs, ys, size)
    return (float(np.min(px)), float(np.min(py)), float(np.max(px)), float(np.max(py)))


def spell_circle(center_px: Sequence[float], size: Sequence[int], geometry: ArenaGeometry,
                 radius_tiles: float) -> Box4:
    """circle_box around the world point under pixel `center_px` of a frame of `size`."""
    wx, wy = geometry.px_to_world(float(center_px[0]), float(center_px[1]), size)
    return circle_box((float(wx), float(wy)), size, geometry, radius_tiles)


def propagate_parts(decisions: Mapping[str, tuple[str, str | None, str]], sheet_parts: Mapping[str, Mapping[str, Any]],
                    harvest_tracks: Mapping[int, Mapping[str, Any]], index: Mapping[int, FrameRec],
                    live_teams: Mapping[str, int | None], *, table: Any, geometry: ArenaGeometry,
                    config: EmitConfig | None = None) -> dict[str, PartPlan]:
    """Each decided part (decide_parts) over its images (module doc, N11, propagate): the
    harvest track's proposals with first_n <= n <= last_n; an image whose proposal one of our
    live tracks held is left out (except for an o part); the box is the linked live
    detection's, else the proposal's; sizes smoothed (smooth_boxes); short gaps filled for
    labels and masks (fill_gaps); a spell with a radius (EmitConfig.spell_radius_tiles) is
    its circle around the median centre on every image."""
    cfg = config or EmitConfig()
    table = load_class_table(table)
    radii = {str(k): float(v) for k, v in cfg.spell_radius_tiles}
    by_hk: dict[int, list[int]] = {}
    for n, rec in index.items():
        by_hk.setdefault(rec.hk, []).append(int(n))
    for ns in by_hk.values():
        ns.sort()
    hk_of = {int(n): rec.hk for n, rec in index.items()}
    out: dict[str, PartPlan] = {}
    for key in sorted(decisions, key=_part_order):
        decision, label, src = decisions[key]
        sp = sheet_parts[key]
        h, lo, hi = int(sp["h"]), int(sp["first_n"]), int(sp["last_n"])
        typ = TYPE_UNIT
        if label:
            typ = TYPE_SPELL if resolve_label(table, label).type == TYPE_SPELL else TYPE_UNIT
        anchor = ANCHOR_CENTRE if typ == TYPE_SPELL else ANCHOR_BOTTOM
        seq: list[tuple[int, Box4, str]] = []
        props: dict[int, PropRef] = {}
        skipped: set[int] = set()
        for n, k in (harvest_tracks.get(h) or {}).get("frames") or ():
            n, k = int(n), int(k)
            if not lo <= n <= hi or n in props or n in skipped:
                continue
            rec = index.get(n)
            prop = None if rec is None else rec.props.get(k)
            if prop is None:
                continue
            if decision != DECISION_OURS and prop.live and live_teams.get(prop.live) == OWN_TEAM:
                skipped.add(n)
                continue
            d = rec.live_for(prop.live)
            if d is not None:
                box, tag = _box(d.box), "live"
            else:
                box, tag = prop.box, prop.src.split("+")[0]
            seq.append((n, box, tag))
            props[n] = prop
        seq.sort(key=lambda s: s[0])
        smooth = smooth_boxes(seq, cfg.smooth_window, anchor)
        boxes: dict[int, tuple[Box4, str]] = {n: (b, HOW_SMOOTH) for n, b in smooth}
        if decision in (DECISION_LABEL, DECISION_MASK) and smooth and cfg.max_gap > 0:
            ns = by_hk.get(index[smooth[0][0]].hk, [])
            for n, b in fill_gaps(smooth, ns, cfg.max_gap, anchor=anchor, hk_of=hk_of, skip=skipped):
                boxes.setdefault(n, (b, HOW_FILL))
        if decision == DECISION_LABEL and label in radii and smooth:
            pts = [geometry.px_to_world((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, index[n].size) for n, b in smooth]
            centre = (float(np.median([float(p[0]) for p in pts])), float(np.median([float(p[1]) for p in pts])))
            boxes = {n: (circle_box(centre, index[n].size, geometry, radii[label]), HOW_CIRCLE) for n in boxes}
        first = props[seq[0][0]] if seq else None
        last = props[seq[-1][0]] if seq else None
        side = 1 if decision == DECISION_LABEL else 0 if decision == DECISION_OURS else None
        read = sp.get("read") if isinstance(sp.get("read"), Mapping) else {}
        out[key] = PartPlan(key, h, int(sp.get("part", 0)), decision, label, side, src, typ, lo, hi,
                            float(sp.get("first_t", 0.0)), float(sp.get("last_t", 0.0)),
                            None if sp.get("hk") is None else int(sp["hk"]), read.get("uid"), boxes, props,
                            None if first is None else (first.x, first.y), None if last is None else (last.x, last.y),
                            len(skipped))
    return out


def parts_at(plans: Mapping[str, PartPlan]) -> dict[int, list[tuple[PartPlan, PropRef | None, Box4, str]]]:
    """Scan record n -> the decided parts with a box there: (part, its proposal or None for a
    filled image, box, how), in part order."""
    out: dict[int, list[tuple[PartPlan, PropRef | None, Box4, str]]] = {}
    for p in plans.values():
        for n, (b, how) in p.boxes.items():
            out.setdefault(int(n), []).append((p, p.props.get(n), b, how))
    for v in out.values():
        v.sort(key=lambda e: _part_order(e[0].key))
    return out


def birth_images(plans: Mapping[str, PartPlan], index: Mapping[int, FrameRec], config: EmitConfig | None = None) -> set[int]:
    """The images in a birth window: up to birth_s after the birth of a labelled part that is
    a track's first part (later parts of a split track are look changes, not births), where
    that part has a box (plan 2.3: the first second decides the card)."""
    cfg = config or EmitConfig()
    out: set[int] = set()
    for p in plans.values():
        if p.decision != DECISION_LABEL or p.part != 0:
            continue
        for n in p.boxes:
            rec = index.get(n)
            if rec is not None and -_BIRTH_EPS <= rec.t - p.first_t <= cfg.birth_s + _BIRTH_EPS:
                out.add(int(n))
    return out


@dataclass(frozen=True)
class EmitBox:
    """One box an image is written with: a detector label name, side (1 enemy, 0 ally), frame
    pixels, the label's type, src (owner, auto, live) and how (smooth, fill, circle, live);
    key: the part it came from (None for a live pseudo-label)."""
    label: str
    side: int
    box: Box4
    type: str = TYPE_UNIT
    src: str = SRC_OWNER
    how: str = HOW_SMOOTH
    key: str | None = None


@dataclass
class FramePlan:
    """What one image becomes: the boxes kept (clipped), the regions masked with the
    background (frame px, before the pad), the reason it is dropped (training only), and for
    gold the ignore regions (masks grown by the pad, nothing pasted)."""
    n: int
    keep: list[EmitBox] = field(default_factory=list)
    masks: list[Box4] = field(default_factory=list)
    drop: str | None = None
    ignore: list[Box4] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)

    @property
    def enemy_labels(self) -> int:
        """Boxes of the declared opponent's labels (owner or auto) on this image."""
        return sum(1 for b in self.keep if b.side == 1 and b.src in (SRC_OWNER, SRC_AUTO))


def _spell_name(table: Any, card: str, known: Container[str]) -> str:
    """tools/vision_autolabel.spell_label, restated: a spell's KataCR name when the label set
    has it (Fireball -> "fireball"), else the card name (the registry's Vines)."""
    for name in (table.cards.get(card) or {}).get("katacr") or ():
        if name in known:
            return str(name)
    return str(card)


def canonical_name(table: Any, name: str, known: Container[str]) -> str | None:
    """A class, unit, spell or KataCR name as a label of `known`: units and towers as their
    detector key (in the name's own form), spells as _spell_name, other names as they are;
    None when the label set has no such label."""
    lab = resolve_label(table, str(name))
    if lab.type in (TYPE_UNIT, TYPE_TOWER) and lab.class_id is not None:
        out = table.detector_key(lab.class_id, lab.form)
    elif lab.type == TYPE_SPELL and lab.card:
        out = _spell_name(table, lab.card, known)
    elif lab.type in (TYPE_UI, TYPE_OBJECT):
        out = str(name)
    else:
        out = None
    return out if out is not None and out in known else None


def live_label(d: LiveDet, table: Any, known: Container[str]) -> str | None:
    """The label a live detection is written as: a unit or tower its class's detector key in
    the detection's form, a spell its card's label, a UI or object box its own name; None
    when it is none of these or `known` (the label set) lacks it."""
    if d.type in (TYPE_UNIT, TYPE_TOWER):
        lab = resolve_label(table, d.cls)
        if lab.class_id is None or lab.type not in (TYPE_UNIT, TYPE_TOWER):
            return None
        name = table.detector_key(lab.class_id, int(d.form))
    elif d.type == TYPE_SPELL:
        lab = resolve_label(table, d.cls)
        name = _spell_name(table, lab.card, known) if lab.type == TYPE_SPELL and lab.card else None
    elif d.type in (TYPE_UI, TYPE_OBJECT):
        name = str(d.cls)
    else:
        return None
    return name if name is not None and name in known else None


def ours_label(rec: FrameRec, prop: PropRef | None, table: Any, known: Container[str],
               own_labels: Container[str]) -> str | None:
    """The ally label of an o part's image: the linked live detection's label when it is one
    of Yodle2's, else the strongest harvest candidate whose class is; None (masked) else."""
    if prop is None:
        return None
    d = rec.live_for(prop.live)
    if d is not None:
        lab = live_label(d, table, known)
        if lab is not None and lab in own_labels:
            return lab
    for cls, _side, _conf in sorted(prop.cands, key=lambda c: -c[2]):
        if cls not in own_labels:
            continue
        lab = canonical_name(table, cls, known)
        if lab is not None and lab in own_labels:
            return lab
    return None


def _type_of(table: Any, label: str) -> str:
    typ = resolve_label(table, label).type
    return typ if typ in (TYPE_UNIT, TYPE_SPELL, TYPE_TOWER, TYPE_OBJECT, TYPE_UI) else TYPE_UNIT


def plan_frame(rec: FrameRec, at_n: Sequence[tuple[PartPlan, PropRef | None, Sequence[float], str]], *, table: Any,
               known: Container[str], own_labels: Container[str], live_teams: Mapping[str, int | None] | None = None,
               config: EmitConfig | None = None, gold: bool = False, crop: Sequence[float] | None = None,
               pad_px: float = 0.0, background: bool = True) -> FramePlan:
    """One image's plan (module doc, N11, plan_frame), in this order:
    1. the decided parts with a box here (at_n, parts_at): a label is kept on the enemy side;
       ? / a bad mark / a pending part is masked; x writes nothing; o is kept on our side as
       its live read or best candidate when that is one of Yodle2's labels, else masked.
       Their proposals and boxes are COVERED.
    2. the live layer: a detection linked to a covered proposal, or a unit, spell or unknown
       one overlapping a covered box by live_iou, is that part's and skipped (an object or UI
       box on an x part is kept: "dirt" is no unit). The rest: our side (the live track's
       team, else P(enemy)) with one of Yodle2's labels is kept as a pseudo-label; any other
       unit or spell is masked (an unvetted enemy, or a misread that cannot be ours); towers,
       UI and objects are kept as pseudo-labels (side at P 0.5: the board half).
    2b. leftovers (mask_leftovers): a proposal of a track no decided part holds on this image
       (is_leftover: a detector proposal at leftover_conf or more; one of Yodle2's units the
       live model missed, an enemy track too short for the sheet) is masked unless a kept,
       covered or masked box already has it (IoU >= live_iou, or half of it inside one).
    3. kept boxes are clipped to the crop (gold: to the frame); under min_box_px they go.
    4. training only, a UI box a grown mask covers (its unit's HP bar) is masked with it; the
       image is dropped for more than max_masks masks ("masks"), for a grown mask (pad +
       feather) over more than mask_overlap_frac of a kept moving box ("overlap": a tower is
       in the background), or for masks without a background ("no_background"). Gold drops
       nothing: its masks, grown by the pad, are ignore regions."""
    cfg = config or EmitConfig()
    teams = live_teams or {}
    width, height = int(rec.size[0]), int(rec.size[1])
    keep: list[EmitBox] = []
    masks: list[Box4] = []
    counts: Counter = Counter()
    covered_live: dict[str, str] = {}
    covered: list[Box4] = []
    for part, prop, box, how in at_n:
        b = _box(box)
        dec = part.decision
        if dec == DECISION_LABEL and part.label:
            keep.append(EmitBox(str(part.label), 1, b, part.type, part.src, how, part.key))
        elif dec in (DECISION_MASK, DECISION_PENDING):
            masks.append(b)
            counts["part_masks"] += 1
        elif dec == DECISION_OURS:
            label = ours_label(rec, prop, table, known, own_labels)
            if label is None:
                masks.append(b)
                counts["ours_masked"] += 1
            else:
                keep.append(EmitBox(label, 0, b, _type_of(table, label), SRC_LIVE, how, part.key))
        if prop is not None and prop.live:
            covered_live[prop.live] = dec
        covered.append(b)
    for d in rec.live:
        key = None if d.uid is None else live_key(d.seg, d.uid)
        dec = covered_live.get(key) if key is not None else None
        if dec is not None and not (dec == DECISION_EFFECT and d.type in (TYPE_OBJECT, TYPE_UI)):
            counts["live_covered"] += 1
            continue
        moving = d.type in COVER_TYPES
        if moving and any(box_iou(d.box, c) >= cfg.live_iou for c in covered):
            counts["live_covered"] += 1
            continue
        label = live_label(d, table, known)
        team = teams.get(key) if key is not None else None
        side = 0 if team == OWN_TEAM else 1 if team == ENEMY_TEAM else side_of(d.p_enemy)
        if moving:
            if side == 0 and label is not None and label in own_labels:
                keep.append(EmitBox(label, 0, _box(d.box), d.type, SRC_LIVE, HOW_LIVE, None))
            else:
                masks.append(_box(d.box))
                counts["live_masks"] += 1
                if label is None:
                    counts["no_label"] += 1
            continue
        if label is None:
            counts["no_label"] += 1
            continue
        if side is None:
            side = 1 if (float(d.box[1]) + float(d.box[3])) / 2.0 < height / 2.0 else 0
        keep.append(EmitBox(label, side, _box(d.box), d.type, SRC_LIVE, HOW_LIVE, None))
    if cfg.mask_leftovers and rec.props:
        held = {part.h for part, _prop, _box_, _how in at_n}
        taken = covered + [b.box for b in keep] + list(masks)
        for _k, prop in sorted(rec.props.items()):
            if prop.h in held or not is_leftover(prop.src, prop.type, prop.conf, prop.h, cfg.leftover_conf):
                continue
            if any(box_iou(prop.box, c) >= cfg.live_iou or cover_frac(c, prop.box) >= 0.5 for c in taken):
                continue
            masks.append(prop.box)
            taken.append(prop.box)
            counts["leftover_masks"] += 1
    lim = (0.0, 0.0, float(width), float(height)) if gold or crop is None else _box(crop)
    kept: list[EmitBox] = []
    for b in keep:
        c = clip_box(b.box, lim)
        if c is None or c[2] - c[0] < cfg.min_box_px or c[3] - c[1] < cfg.min_box_px:
            counts["clipped"] += 1
            continue
        kept.append(dataclasses.replace(b, box=c))
    ms = [c for c in (clip_box(m, lim) for m in masks) if c is not None]
    if gold:
        ignore = [c for c in (clip_box(grow_box(m, pad_px), lim) for m in ms) if c is not None]
        return FramePlan(rec.n, kept, ms, None, ignore, counts)
    reach = float(pad_px) + float(cfg.feather_px)
    grown = [grow_box(m, reach) for m in ms]
    absorbed: list[Box4] = []
    if grown:
        rest = []
        for b in kept:
            if b.type == TYPE_UI and any(cover_frac(g, b.box) > cfg.mask_overlap_frac for g in grown):
                absorbed.append(b.box)
                counts["ui_absorbed"] += 1
            else:
                rest.append(b)
        kept = rest
    paste = ms + absorbed
    every = grown + [grow_box(a, reach) for a in absorbed]
    drop = None
    if len(ms) > cfg.max_masks:
        drop = DROP_MASKS
    elif any(b.type in OVERLAP_TYPES and cover_frac(g, b.box) > cfg.mask_overlap_frac for b in kept for g in every):
        drop = DROP_OVERLAP
    elif paste and not background:
        drop = DROP_NO_BACKGROUND
    return FramePlan(rec.n, kept, paste, drop, [], counts)


def keep_frame(n: int, plan: FramePlan, birth: bool, config: EmitConfig | None = None) -> bool:
    """Whether a training image is written: every birth-window image, 1 in `stride` of the
    images with an enemy label box, 1 in `empty_stride` of all (by scan record n)."""
    cfg = config or EmitConfig()
    n = int(n)
    return bool(birth) or (plan.enemy_labels > 0 and n % cfg.stride == 0) or n % cfg.empty_stride == 0


def emit_framing(geometry: ArenaGeometry, det_config: DetectorConfig | None, size: Sequence[int],
                 config: EmitConfig | None = None) -> tuple[tuple[int, int, int, int], float]:
    """(the live crop, the mask pad in pixels) for a frame of `size`: detector.crop_rect_for,
    so the emitted framing is the one the live model sees."""
    cfg = config or EmitConfig()
    crop = crop_rect_for(geometry, det_config, size)
    tw, th = tile_px(geometry, size)
    return crop, float(cfg.mask_pad_tiles) * min(tw, th)


def feather_paste(rgb: np.ndarray, background: np.ndarray, boxes: Iterable[Sequence[float]], *,
                  pad_px: float = 0.0, feather_px: int = 3) -> np.ndarray:
    """A copy of `rgb` with each box grown by pad_px (whole pixels touched, as box_mask) set to
    the background, and blended back to the image over feather_px pixels outside it (alpha
    falls by 1/(feather_px+1) a pixel, Chebyshev distance), so a patch has no hard edge (plan
    risk 5). Boxes may reach past the image."""
    img = np.asarray(rgb)[..., :3]
    bg = np.asarray(background)[..., :3]
    if img.shape != bg.shape:
        raise ValueError(f"image {img.shape} and background {bg.shape} differ in size")
    h, w = img.shape[:2]
    f = max(0, int(feather_px))
    alpha = np.zeros((h, w), dtype=np.float64)
    for b in boxes:
        x0, y0 = int(math.floor(float(b[0]) - pad_px)), int(math.floor(float(b[1]) - pad_px))
        x1, y1 = int(math.ceil(float(b[2]) + pad_px)), int(math.ceil(float(b[3]) + pad_px))
        gx0, gy0, gx1, gy1 = max(0, x0 - f), max(0, y0 - f), min(w, x1 + f), min(h, y1 + f)
        if gx1 <= gx0 or gy1 <= gy0:
            continue
        xs, ys = np.arange(gx0, gx1), np.arange(gy0, gy1)
        dx = np.maximum(np.maximum(x0 - xs, xs - (x1 - 1)), 0)
        dy = np.maximum(np.maximum(y0 - ys, ys - (y1 - 1)), 0)
        a = np.clip(1.0 - np.maximum(dy[:, None], dx[None, :]) / (f + 1.0), 0.0, 1.0)
        alpha[gy0:gy1, gx0:gx1] = np.maximum(alpha[gy0:gy1, gx0:gx1], a)
    out = img.astype(np.float64)
    if alpha.any():
        out = out * (1.0 - alpha[..., None]) + bg.astype(np.float64) * alpha[..., None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def crop_image(rgb: np.ndarray, crop: Sequence[int]) -> np.ndarray:
    """rgb[y0:y1, x0:x1] (RGB, contiguous)."""
    x0, y0, x1, y1 = (int(v) for v in crop)
    return np.ascontiguousarray(np.asarray(rgb)[y0:y1, x0:x1, :3])


def image_size(rgb: np.ndarray) -> tuple[int, int]:
    """(width, height) of an image array."""
    a = np.asarray(rgb)
    return int(a.shape[1]), int(a.shape[0])


def to_yolo(box: Sequence[float], crop: Sequence[float], min_px: float = 3.0) -> tuple[float, float, float, float] | None:
    """A frame-pixel box as a YOLO row of the crop: clipped to the crop, then (cx, cy, w, h)
    relative to the crop's size; None when less than min_px of it is left either way."""
    x0, y0, x1, y1 = (float(v) for v in crop)
    c = clip_box(box, (x0, y0, x1, y1))
    if c is None or c[2] - c[0] < float(min_px) or c[3] - c[1] < float(min_px):
        return None
    cw, ch = x1 - x0, y1 - y0
    return ((c[0] + c[2]) / 2.0 - x0) / cw, ((c[1] + c[3]) / 2.0 - y0) / ch, (c[2] - c[0]) / cw, (c[3] - c[1]) / ch


def from_yolo(row: Sequence[float], crop: Sequence[float]) -> Box4:
    """to_yolo back: a YOLO row of the crop as a frame-pixel box."""
    x0, y0, x1, y1 = (float(v) for v in crop)
    cw, ch = x1 - x0, y1 - y0
    cx, cy, w, h = (float(v) for v in row[:4])
    return (x0 + (cx - w / 2.0) * cw, y0 + (cy - h / 2.0) * ch, x0 + (cx + w / 2.0) * cw, y0 + (cy + h / 2.0) * ch)


def frame_yolo(box: Sequence[float], size: Sequence[int]) -> tuple[float, float, float, float]:
    """A frame-pixel box as a YOLO row of the whole frame (gold labels: vision_replay reads
    them against the full image)."""
    w, h = float(size[0]), float(size[1])
    x1, y1, x2, y2 = _box(box)
    return (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h, (x2 - x1) / w, (y2 - y1) / h


def crop_masks(masks: Iterable[Sequence[float]], crop: Sequence[float], pad_px: float = 0.0) -> list[list[float]]:
    """The masked regions (grown by pad_px, the part pasted fully) in crop pixels, 1 decimal."""
    x0, y0 = float(crop[0]), float(crop[1])
    out = []
    for m in masks:
        c = clip_box(grow_box(m, pad_px), crop)
        if c is not None:
            out.append([_r1(c[0] - x0), _r1(c[1] - y0), _r1(c[2] - x0), _r1(c[3] - y0)])
    return out


# -- the contact sheet -------------------------------------------------------------

SHEET_BG = (32, 32, 32)
CELL_BG = (64, 64, 64)
TEXT_COLOUR = (255, 255, 255)
DIM_COLOUR = (190, 190, 190)
FLAG_COLOUR = (255, 140, 120)
BOX_COLOUR = (255, 220, 0)
#: A 3x5 bitmap font (rows top to bottom), upper case; anything else draws as '?'.
_GLYPHS = {
    "A": "010101111101101", "B": "110101110101110", "C": "011100100100011", "D": "110101101101110",
    "E": "111100110100111", "F": "111100110100100", "G": "011100101101011", "H": "101101111101101",
    "I": "111010010010111", "J": "001001001101010", "K": "101101110101101", "L": "100100100100111",
    "M": "101111111101101", "N": "110101101101101", "O": "010101101101010", "P": "110101110100100",
    "Q": "010101101110011", "R": "110101110101101", "S": "011100010001110", "T": "111010010010010",
    "U": "101101101101111", "V": "101101101101010", "W": "101101111111101", "X": "101101010101101",
    "Y": "101101010010010", "Z": "111001010100111",
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001001001",
    "8": "111101111101111", "9": "111101111001111",
    " ": "000000000000000", "#": "101111101111101", "=": "000111000111000", ":": "000010000010000",
    ".": "000000000000010", ",": "000000000010100", "-": "000000111000000", "_": "000000000000111",
    "?": "110001010000010", "/": "001001010100100", "+": "000010111010000", "(": "010100100100010",
    ")": "010001001001010", "%": "101001010100101", "@": "010101111100011", "*": "000101010101000",
    ">": "100010001010100", "<": "001010100010001", "!": "010010010000010", "'": "010010000000000",
    "[": "110100100100110", "]": "011001001001011", "|": "010010010010010", "\\": "100100010001001",
    '"': "101101000000000", "&": "010101010101011", "$": "011110010011110",
}


def text_mask(text: str, scale: int = 2) -> np.ndarray:
    """`text` (upper-cased) in the 3x5 font, `scale` pixels per dot, one dot between glyphs."""
    parts = []
    for ch in str(text).upper():
        pat = _GLYPHS.get(ch, _GLYPHS["?"])
        parts.append(np.array([c == "1" for c in pat], dtype=bool).reshape(5, 3))
        parts.append(np.zeros((5, 1), dtype=bool))
    m = np.concatenate(parts, axis=1) if parts else np.zeros((5, 1), dtype=bool)
    s = max(1, int(scale))
    return np.kron(m, np.ones((s, s), dtype=bool)).astype(bool)


def draw_text(img: np.ndarray, x: int, y: int, text: str, colour: Sequence[int] = TEXT_COLOUR, *,
              scale: int = 2, max_w: int | None = None) -> None:
    """Draw `text` with its top-left at (x, y), cut at max_w pixels and at the image's edge."""
    m = text_mask(text, scale)
    h, w = img.shape[:2]
    mh = min(m.shape[0], max(0, h - y))
    mw = min(m.shape[1], max(0, w - x), m.shape[1] if max_w is None else max(0, int(max_w)))
    if mh <= 0 or mw <= 0:
        return
    img[y:y + mh, x:x + mw][m[:mh, :mw]] = colour


@dataclass
class SheetTile:
    """One tile: a title ('#id n=K' and its flags), text lines (what tonight's model read, the
    suggestion), and one row of crops per shown track."""
    title: str
    lines: list[str] = field(default_factory=list)
    rows: list[list[np.ndarray]] = field(default_factory=list)


def render_sheet(tiles: Sequence[SheetTile], *, cell: int = 72, width: int = 1400, header: Sequence[str] = (),
                 legend: Sequence[str] = (), rows: int = 4, scale: int = 2) -> np.ndarray:
    """The contact sheet: header lines, the legend in columns, then the tiles in a grid
    (each: its title, its lines, then up to `rows` rows of 3 crops)."""
    pad = 4
    line_h = 5 * scale + 3
    char_w = 4 * scale
    tile_w = pad + 3 * (cell + pad)
    n_lines = 1 + max((len(t.lines) for t in tiles), default=0)
    tile_h = pad + n_lines * line_h + rows * (cell + pad)
    cols = max(1, int(width) // tile_w)
    W = cols * tile_w
    leg = [str(s) for s in legend]
    col_chars = min(40, max((len(s) for s in leg), default=1))
    leg_cols = max(1, (W - pad) // (col_chars * char_w + 3 * pad))
    leg_rows = -(-len(leg) // leg_cols) if leg else 0
    top = pad + len(header) * line_h + (pad + leg_rows * line_h if leg else 0) + pad
    grid_rows = -(-len(tiles) // cols) if tiles else 0
    H = top + grid_rows * tile_h + pad
    sheet = np.empty((H, W, 3), dtype=np.uint8)
    sheet[:] = SHEET_BG
    y = pad
    for s in header:
        draw_text(sheet, pad, y, s, TEXT_COLOUR, scale=scale, max_w=W - 2 * pad)
        y += line_h
    if leg:
        y += pad
        cw = (W - pad) // leg_cols
        for i, s in enumerate(leg):
            c, r = divmod(i, leg_rows)
            draw_text(sheet, pad + c * cw, y + r * line_h, s, DIM_COLOUR, scale=scale, max_w=cw - pad)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        x0, y0 = c * tile_w, top + r * tile_h
        sheet[y0 + 1:y0 + tile_h - 1, x0 + 1:x0 + tile_w - 1] = (44, 44, 44)
        draw_text(sheet, x0 + pad, y0 + pad, t.title, FLAG_COLOUR if " S" in t.title else TEXT_COLOUR,
                  scale=scale, max_w=tile_w - 2 * pad)
        for j, s in enumerate(t.lines):
            draw_text(sheet, x0 + pad, y0 + pad + (j + 1) * line_h, s, DIM_COLOUR, scale=scale, max_w=tile_w - 2 * pad)
        gy = y0 + pad + n_lines * line_h
        for rr, crops in enumerate(t.rows[:rows]):
            for cc, crop in enumerate(list(crops)[:3]):
                xx, yy = x0 + pad + cc * (cell + pad), gy + rr * (cell + pad)
                a = np.asarray(crop, dtype=np.uint8)
                if a.shape[:2] != (cell, cell):
                    a = sheet_crop(a, (0, 0, a.shape[1], a.shape[0]), cell, 0.0, outline=False)
                sheet[yy:yy + cell, xx:xx + cell] = a[..., :3]
    return sheet


def write_sheet_image(path: str | Path, sheet: np.ndarray) -> Path:
    """The sheet as PNG, or as .npy when no image codec is installed; returns the path written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if image_codec_available("png"):
        try:
            save_image(p.with_suffix(".png"), sheet, "png")
            return p.with_suffix(".png")
        except (RuntimeError, OSError, ValueError):
            pass
    save_image(p.with_suffix(".npy"), sheet, "npy")
    return p.with_suffix(".npy")


# -- N12: review pages ----------------------------------------------------------------
#
# tools/vision_autolabel.py `review` draws its page A (one tile per part: render_sheet with
# sheet_crop tiles, text_tile where the raw image is gone) and page B (whole frames with their
# boxes: render_frames).

PSEUDO_COLOUR = (120, 200, 255)          # page B: tonight's pseudo-labels (Yodle2's units, towers, UI)


def text_tile(text: str, cell: int = 72, colour: Sequence[int] = DIM_COLOUR) -> np.ndarray:
    """A cell x cell CELL_BG tile with a short text across its middle ("RAW GONE")."""
    out = np.empty((int(cell), int(cell), 3), dtype=np.uint8)
    out[:] = CELL_BG
    scale = 2 if len(str(text)) * 8 <= int(cell) - 4 else 1
    draw_text(out, 2, max(0, int(cell) // 2 - 5), str(text), colour, scale=scale, max_w=int(cell) - 4)
    return out


def draw_box(img: np.ndarray, box: Sequence[float], colour: Sequence[int], thickness: int = 2) -> None:
    """Outline `box` (x1, y1, x2, y2 in pixels) on `img` in place, `thickness` pixels inward;
    the parts of the outline outside the image are left out."""
    h, w = img.shape[:2]
    x0, y0 = int(math.floor(float(box[0]))), int(math.floor(float(box[1])))
    x1, y1 = int(math.ceil(float(box[2]))) - 1, int(math.ceil(float(box[3]))) - 1
    if x1 < x0 or y1 < y0 or x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
        return
    col = tuple(int(v) for v in colour)[:3]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)
    for k in range(max(1, int(thickness))):
        if 0 <= y0 + k < h:
            img[y0 + k, cx0:cx1 + 1] = col
        if 0 <= y1 - k < h:
            img[y1 - k, cx0:cx1 + 1] = col
        if 0 <= x0 + k < w:
            img[cy0:cy1 + 1, x0 + k] = col
        if 0 <= x1 - k < w:
            img[cy0:cy1 + 1, x1 - k] = col


def resize_nearest(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """`img` scaled to width x height by nearest neighbour (no image library)."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    width, height = max(1, int(width)), max(1, int(height))
    ys = np.minimum(h - 1, ((np.arange(height) + 0.5) * h / height).astype(int))
    xs = np.minimum(w - 1, ((np.arange(width) + 0.5) * w / width).astype(int))
    return np.ascontiguousarray(a[ys][:, xs])


@dataclass
class FrameTile:
    """One frame of page B: a title, text lines, the image (None: the raw is gone), its boxes
    ((x1, y1, x2, y2) in the image's pixels, text, colour) and outlined regions (masks or
    ignore regions: (box, colour))."""
    title: str
    image: np.ndarray | None = None
    boxes: list[tuple[Sequence[float], str, Sequence[int]]] = field(default_factory=list)
    regions: list[tuple[Sequence[float], Sequence[int]]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    note: str = ""


def render_frames(tiles: Sequence[FrameTile], *, width: int = 1400, cols: int = 4, header: Sequence[str] = (),
                  scale: int = 2, label_scale: int = 1) -> np.ndarray:
    """Page B of a review: header lines, then the frames in a grid of `cols`, each scaled to the
    column width (nearest neighbour) under its title and lines. Boxes and regions are drawn on
    the scaled frame (crisp lines at any scale), each box with its text above it."""
    pad = 4
    line_h = 5 * scale + 3
    cols = max(1, int(cols))
    thumb_w = max(32, (int(width) - pad) // cols - pad)
    W = pad + cols * (thumb_w + pad)
    thumbs: list[np.ndarray] = []
    for t in tiles:
        if t.image is None:
            th = int(round(thumb_w * 16 / 9))
            img = np.empty((th, thumb_w, 3), dtype=np.uint8)
            img[:] = CELL_BG
            draw_text(img, pad, th // 2, t.note or "raw deleted", DIM_COLOUR, scale=scale, max_w=thumb_w - 2 * pad)
            thumbs.append(img)
            continue
        src = np.asarray(t.image)[..., :3]
        h, w = src.shape[:2]
        sx = thumb_w / float(w)
        th = max(1, int(round(h * sx)))
        sy = th / float(h)
        img = resize_nearest(src, thumb_w, th).copy()

        def scaled(b: Sequence[float]) -> Box4:
            return (float(b[0]) * sx, float(b[1]) * sy, float(b[2]) * sx, float(b[3]) * sy)

        for box, colour in t.regions:
            draw_box(img, scaled(box), colour, 1)
        for box, text, colour in t.boxes:
            sb = scaled(box)
            draw_box(img, sb, colour, 2)
            if text:
                tx = min(max(0, int(sb[0])), thumb_w - 1)
                ty = max(0, int(sb[1]) - 5 * label_scale - 2)
                draw_text(img, tx, ty, text, colour, scale=label_scale, max_w=thumb_w - tx)
        thumbs.append(img)
    n_lines = 1 + max((len(t.lines) for t in tiles), default=0)
    rows = [list(range(i, min(i + cols, len(tiles)))) for i in range(0, len(tiles), cols)]
    row_h = [pad + n_lines * line_h + max(thumbs[j].shape[0] for j in r) + pad for r in rows]
    top = pad + len(header) * line_h + pad
    H = top + sum(row_h) + pad
    page = np.empty((H, W, 3), dtype=np.uint8)
    page[:] = SHEET_BG
    y = pad
    for s in header:
        draw_text(page, pad, y, s, TEXT_COLOUR, scale=scale, max_w=W - 2 * pad)
        y += line_h
    y0 = top
    for r, hgt in zip(rows, row_h):
        for c, j in enumerate(r):
            t, img = tiles[j], thumbs[j]
            x0 = pad + c * (thumb_w + pad)
            draw_text(page, x0, y0 + pad, t.title, TEXT_COLOUR, scale=scale, max_w=thumb_w)
            for k, s in enumerate(t.lines):
                draw_text(page, x0, y0 + pad + (k + 1) * line_h, s, DIM_COLOUR, scale=scale, max_w=thumb_w)
            iy = y0 + pad + n_lines * line_h
            page[iy:iy + img.shape[0], x0:x0 + img.shape[1]] = img
        y0 += hgt
    return page


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
