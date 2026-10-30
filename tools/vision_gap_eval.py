r"""The gate before a fine-tuned detector is pinned: gold battles through the live chain (N14).

docs/NEW_CARDS_PLAN.md 2.4, 6 and N14. A candidate detector (a fine-tune on a `mix`, N13), the
baseline (the current pin; on Round 0 that is tonight's M0) and, once a fine-tune is pinned,
M0 itself (the play-ID gain is plan 6's gain over tonight's model) are each run through the exact
live chain on every gold battle (tools/vision_replay.py, in process: screen cues, HUD, the
detector at the live DetectorConfig, HP bars, tracker, EventDeriver with events.LIVE_GUARDS
and, by default, the UNKNOWN safety net), and what they read is scored against the owner's
gold labels (tools/vision_autolabel.py emit, review, mark and plays: N11, N12):

  detection   per gold image (gold.jsonl, bad frames of marks.json left out), the unit and
              spell boxes matched per class at IoU 0.5 (vision_replay.match_boxes), card-level
              (the class: MinionHorde_EV1 counts as Minion) and form-aware (the detector key);
              a prediction inside an ignore region (an unvetted unit, a ?, a pending or
              bad-marked track: plan_frame) is no false positive. Per new label (the
              registry's, and any gold label tonight's set lacks) recall and precision with
              95% bootstrap intervals over harvest tracks (recall) and 1-s image blocks
              (precision); side accuracy of the matched pairs on owner-vetted boxes (a part's
              box: the owner's enemy labels and our o tracks; tonight's own pseudo-labels are
              tonight's reads, so their agreement is info only). Every other interval (side,
              birth, plays, UNKNOWN, our boxes) is the Wilson interval of N12 (vision_autolabel.wilson).
  birth       the share of enemy gold tracks (their first part) whose class is the top box (the
              strongest prediction at IoU 0.5 on the gold box) on at least 2 of their first 4
              images: these frames decide the card. The seconds the 4 images span are reported
              (Round 0 has about 11 images a second, harvest friendlies about 19).
  plays       the replay's OppPlays (t = the replay's time origin + tick / 20) against the
              confirmed gold plays (plays.json): within 1 s and 3 tiles (time only when either
              has no position), same-card pairs first, then any card, each stage the most
              pairs at the least |dt| then distance (vision_autolabel.assign_pairs, the Hungarian
              method). Per new card recall and card precision; per evolution or hero form the
              card accuracy (and the form accuracy, reported); pooled play-ID accuracy, its gain
              over M0, phantom plays per minute, and the elixir error per minute: the
              cost of missed and phantom plays plus the cost difference of wrong-card plays (an
              UNKNOWN play costs what the deriver charged it). The opponent elixir counted from
              each list (spent per minute) is reported beside it.
  safety net  the share of UNKNOWN plays (card -1, --unknown-plays) that match a real play,
              and their share of all plays.
  regression  KataCR val mAP50 over tonight's 310 classes (ultralytics val on tmp/yolo/cr.yaml,
              GPU; cached per weights SHA-256, yaml and val split), each of Yodle2's labels'
              recall on gold on the boxes of our o tracks (owner-vetted; the rest of our boxes
              are tonight's pseudo-labels: their agreement, reg.own_live, is info), new labels
              on our own boxes (a box that would become a phantom enemy play), and the
              own-landing miss rate. Against the baseline (the current pin).
  leak        the candidate's run folder (weights/../..) holds detector_manifest.json, whose
              "data" is the mix's cr_mix.yaml; mix.json beside it must hold no session that is
              gold, shares a gold recording (frames_sha1, or name and created) or lies under a
              gold session's store. No manifest or no mix.json: "insufficient", so a candidate
              trained outside `mix` cannot PASS. The mix.json must be the one the run trained
              on (the manifest's mix_sha1, which train_detector records as the run starts; an
              older manifest: not newer than the weights), else, with no leak found in it,
              "insufficient" too: a mix built again after training cannot hide a gold session.

Thresholds (plan 6; GateConfig). Gated on the point estimate; the intervals are reported:
  det.recall.<label> / det.precision.<label>  >= 0.85 / >= 0.85, card-level, per new label
                                               with at least 15 gold tracks
  det.side         side accuracy >= 0.98                       required
  det.birth        birth-window recall >= 0.85                 required
  play.recall.<card> / play.precision.<card>   >= 0.85 / >= 0.90, per new card with >= 4 plays
  play.form.<card>.<form>  card accuracy >= 0.90 per evolution or hero with >= 4 plays (form
                   accuracy reported, target 0.8)
  play.id          pooled play-ID accuracy >= 0.85             required
  play.id_gain     >= +0.15 over M0 (tonight's model)          required
  play.phantom     <= 0.5 phantom plays per minute             required
  play.elixir      <= 1.5 elixir error per minute              required
  play.unknown     >= 0.90 of UNKNOWN plays match a real play
  reg.katacr       KataCR val mAP50 >= baseline - 0.01         required
  reg.own.<label>  recall on our o tracks' boxes >= baseline - 0.01, per Yodle2 label
  reg.newlabel_on_own  <= 0.01 of our boxes read as a new label  required
  reg.landing      own-landing miss rate <= the baseline's
  leak             no gold recording in the candidate's mix    required
Numbers the plan does not give, chosen here (GateConfig fields, and the summary):
  evidence minimums: det.side 50 matched pairs (side_min_pairs); det.birth 15 tracks
  (birth_min_tracks, as the plan's 15 for detection); the pooled play checks 20 gold plays
  (pooled_min_plays); play.unknown 5 UNKNOWN plays (unknown_min); reg.own and
  reg.newlabel_on_own 50 boxes (own_min_boxes); reg.landing 10 taps (landing_min_taps);
  a prediction is inside an ignore region at IoA >= 0.5 (ignore_ioa); bootstrap 1,000
  resamples, seed 0; precision blocks of 1 s; a birth-window track needs at least 2 images
  (shorter ones are left out and counted as "short"); play.precision.<card> takes its evidence
  from the card's gold plays (>= 4), like its recall.
Each check is "pass", "fail", "insufficient" (below its evidence minimum, or not measured) or
"info". The verdict (plan 6.4, D22): FAIL when any check with enough evidence fails (a
failing leak always does); else INSUFFICIENT when a required check lacks evidence; else PASS.
Thin items do not gate: an optional check below its minimum is only reported.

Owner commands ($BB = .venvs\buildabot\Scripts\python.exe; the GPU venv, MuMu and the browser
closed: each gold battle is replayed once per model unless cached):

    # the M0 baseline on the gold set (Round 0): report.md with M0's numbers, what M0 reads
    # each gold label as, and M0's plays per gold card; verdict BASELINE
    & $BB tools\vision_gap_eval.py baseline --gold 20261029_R0_1
    # the gate: PASS / FAIL / INSUFFICIENT; never pins
    & $BB tools\vision_gap_eval.py gate --candidate tmp\yolo\runs\ft1\weights\best.pt
    # print a report again from its gate.json
    & $BB tools\vision_gap_eval.py report reports\vision\gate\ft1_20261115_093000

    --gold defaults to every gold session with a fresh gold emit and confirmed plays (the
    others are listed as skipped, with why); --baseline / --weights default to the pin (its
    SHA-256 checked); the first `baseline` run records its weights as M0 in
    reports/vision/gate/m0.json, which play.id_gain is measured over from then on (--m0 WEIGHTS
    for another; delete m0.json to record M0 again);
    --no-unknown-plays replays without the safety net (an UNKNOWN play matched to a gold play
    is a wrong card, unmatched a phantom); --no-katacr-val skips the GPU val (reg.katacr is
    then insufficient: no PASS); --no-reuse replays again although the cache holds the run.

Output: reports/vision/gate/<name>/gate.json and report.md (default name <run>_<time>, or
M0_<time> for baseline); replays in reports/vision/gate/cache/<sha12>/<sid>_<flags8>/ with
key.json (weights SHA-256, the recording's frames.jsonl SHA-1, the replay flags, digests of the
live chain's source files and of the data it reads: classes.json, the event tables, the card
maps, the HUD regions and templates, the spawn table (or its absence) and the card ids), reused
while the key holds; a replay whose card ids came from elsewhere is refused; KataCR val numbers in
cache/<sha12>/katacr_val.json. All of reports/vision/ is gitignored. A gold session must be
`is_gold` (else exit 2), its gold emit fresh (emit_fresh) and its play list confirmed and fresh
(plays_fresh), else the gate refuses it (exit 1). Gold is only read here: nothing is written
into a store, a train or val folder or the pin.

Exit codes: 0 PASS, or a baseline written; 1 FAIL, INSUFFICIENT, or a refused gold set; 2 bad
input. Imports numpy (vision_replay); the replays need the detector's ultralytics and torch,
the KataCR val a CUDA GPU. Nothing here talks to the emulator, reads game memory or taps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:          # vision_autolabel, vision_replay
    sys.path.insert(0, str(ROOT / "tools"))
if str(ROOT) not in sys.path:                    # scripts.vision
    sys.path.append(str(ROOT))

import vision_autolabel as VA  # noqa: E402
import vision_replay as VR  # noqa: E402
from scripts.vision.detector import DetectorError, box_iou, file_sha256, resolve_weights  # noqa: E402
from scripts.vision.labels import TYPE_SPELL, TYPE_TOWER, TYPE_UNIT, resolve_label, split_side  # noqa: E402
from scripts.vision.record import VisionLogReader, list_sessions  # noqa: E402

GATE_SCHEMA = "hastycr.vision.gap_gate/1"
KEY_SCHEMA = "hastycr.vision.gap_gate_replay/1"
KATACR_SCHEMA = "hastycr.vision.gap_gate_katacr/1"
GATE_ROOT = ROOT / "reports" / "vision" / "gate"
CACHE_DIR = "cache"
KATACR_DATA = VA.KATACR_ROOT / "cr.yaml"
GATE_FILE, REPORT_FILE, KEY_FILE, KATACR_FILE = "gate.json", "report.md", "key.json", "katacr_val.json"
M0_FILE = "m0.json"                              # <gate root>/m0.json: tonight's model, recorded by the first baseline
M0_SCHEMA = "hastycr.vision.gap_gate_m0/1"
MANIFEST_FILE, MIX_FILE, SUMMARY_FILE = "detector_manifest.json", "mix.json", "summary.json"
#: The live chain's source files: a replay is redone when one of them changed (the cache key).
CHAIN_FILES = ("tools/vision_replay.py", "scripts/vision_play.py", "scripts/vision/events.py",
               "scripts/vision/tracker.py", "scripts/vision/detector.py", "scripts/vision/hud.py",
               "scripts/vision/hpbars.py", "scripts/vision/geometry.py", "scripts/vision/types.py",
               "scripts/vision/labels.py", "tools/vision_classes.py", "scripts/vision/capture.py",
               "scripts/vision/record.py", "scripts/vision/templates.py")
#: The data the live chain reads (the cache key, data_digest): the class table (its sorted cards are
#: the replay-only card ids), the event tables, the card maps, the default geometry, the HUD regions
#: (the PC's first) and the HUD templates; data_digest adds the spawn table vision_replay resolves
#: (the engine's card ids when it is there) and a --geometry file.
DATA_FILES = ("data/vision/classes.json", "data/vision/spawn_children.json",
              "data/vision/geometry_default_1080x1920.json", "data/vision/hud_regions_1080x1920.json",
              "data/vision/hud_regions_pc.json")
DATA_GLOBS = ("data/cards/*.json", "data/vision/templates/**/*")
TICKS_PER_S = VR.TICKS_PER_S
TICKS_TIME = VR.TICKS_TIME
MT_PER_TILE = 1000.0
UNKNOWN = VR.UNKNOWN_NAME                       # an UNKNOWN play's card (card -1)
SCORED_TYPES = (TYPE_UNIT, TYPE_SPELL)           # detection scores units and spells
LABEL_SOURCES = ("owner", "auto")                # gold boxes of the owner's labels (enemy)
SRC_LIVE = "live"                                # gold boxes of tonight's pseudo-labels (ours, towers, UI)
MARKER_CLASS = "clock"                           # KataCR's deploy stopwatch (N03, not built: report only)
FORM_SUFFIX = {1: "_EV1", 2: "Hero"}             # a spell's form-aware key (units: ClassTable.detector_key)
PASS, FAIL, INSUFFICIENT, BASELINE = "PASS", "FAIL", "INSUFFICIENT", "BASELINE"
ST_PASS, ST_FAIL, ST_INSUFFICIENT, ST_INFO = "pass", "fail", "insufficient", "info"
MODE_GATE, MODE_BASELINE = "gate", "baseline"
UNIT_SHARE, UNIT_RATE, UNIT_MAP, UNIT_COUNT, UNIT_GAIN = "share", "per_min", "map50", "count", "gain"
EXIT_OK, EXIT_CHECK, EXIT_INPUT = VA.EXIT_OK, VA.EXIT_CHECK, VA.EXIT_INPUT
_EPS = 1e-9


class GateError(ValueError):
    """The gate cannot run (code 1: a gold session is not ready, a replay failed; code 2: bad input)."""

    def __init__(self, message: str, code: int = EXIT_CHECK):
        self.code = int(code)
        super().__init__(message)


@dataclass(frozen=True)
class GateConfig:
    """The gate's thresholds (plan 6) and the evidence each needs (module doc)."""
    match_iou: float = 0.5
    ignore_ioa: float = 0.5
    det_min_tracks: int = 15
    det_recall: float = 0.85
    det_precision: float = 0.85
    side_acc: float = 0.98
    side_min_pairs: int = 50
    birth_images: int = 4
    birth_hits: int = 2
    birth_recall: float = 0.85
    birth_min_tracks: int = 15
    play_dt_s: float = 1.0
    play_tiles: float = 3.0
    play_min: int = 4
    play_recall: float = 0.85
    card_precision: float = 0.90
    form_card_acc: float = 0.90
    form_acc_target: float = 0.8
    pooled_min_plays: int = 20
    pooled_play_id: float = 0.85
    pooled_gain: float = 0.15
    phantom_per_min: float = 0.5
    elixir_per_min: float = 1.5
    unknown_min: int = 5
    unknown_match: float = 0.90
    katacr_drop: float = 0.01
    own_recall_drop: float = 0.01
    own_min_boxes: int = 50
    newlabel_on_own: float = 0.01
    landing_min_taps: int = 10
    boot: int = 1000
    seed: int = 0
    block_s: float = 1.0

    def __post_init__(self) -> None:
        for name in ("match_iou", "ignore_ioa", "det_recall", "det_precision", "side_acc", "birth_recall",
                     "play_recall", "card_precision", "form_card_acc", "form_acc_target", "pooled_play_id",
                     "unknown_match", "newlabel_on_own"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"GateConfig.{name} must be in [0, 1]")
        for name in ("birth_images", "birth_hits", "boot"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"GateConfig.{name} must be at least 1")
        if self.birth_hits > self.birth_images:
            raise ValueError("GateConfig.birth_hits cannot exceed birth_images")
        for name in ("det_min_tracks", "side_min_pairs", "birth_min_tracks", "play_min", "pooled_min_plays",
                     "unknown_min", "own_min_boxes", "landing_min_taps"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"GateConfig.{name} must be >= 0")
        for name in ("play_dt_s", "play_tiles", "pooled_gain", "phantom_per_min", "elixir_per_min", "katacr_drop",
                     "own_recall_drop"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"GateConfig.{name} must be >= 0")
        if float(self.block_s) <= 0.0:
            raise ValueError("GateConfig.block_s must be > 0")


@dataclass(frozen=True)
class ReplayOptions:
    """How vision_replay runs the live chain (module doc): always --ticks time."""
    unknown_plays: bool = True
    unknown_cost: int | None = None
    device: str | None = None
    spawn_table: str | None = None
    geometry: str | None = None

    def flags(self) -> dict[str, Any]:
        """What the replay's result depends on (the device does not: it is left out of the cache key)."""
        return {"ticks": TICKS_TIME, "unknown_plays": bool(self.unknown_plays),
                "unknown_cost": None if self.unknown_cost is None else int(self.unknown_cost),
                "spawn_table": None if not self.spawn_table else str(self.spawn_table),
                "geometry": None if not self.geometry else str(self.geometry)}


@dataclass
class GoldImage:
    """One gold battle image (a gold.jsonl line): boxes and ignore regions in frame pixels."""
    sid: str
    stem: str
    n: int
    i: int | None
    t: float
    hk: int | None
    size: tuple[int, int]
    boxes: list[dict[str, Any]]
    ignore: list[tuple[float, float, float, float]]


@dataclass
class GoldTrack:
    """One labelled part (parts.jsonl, decision label): its gold images in time order."""
    sid: str
    key: str
    h: int
    part: int
    label: str
    cls: str                       # card-level class (the detector's class; spells their card)
    form_label: str                # form-aware key
    form: int
    side: int
    card: str | None
    first_t: float
    stems: list[str]


@dataclass
class GoldSession:
    sid: str
    session: dict[str, Any]
    emit: dict[str, Any]
    images: dict[str, GoldImage]
    tracks: dict[str, GoldTrack]
    plays: list[dict[str, Any]]
    minutes: float
    bad_frames: set[str]
    recording: Path
    plays_doc: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def frames_sha1(self) -> str | None:
        return (self.session.get("recording") or {}).get("frames_sha1")

    def summary(self) -> dict[str, Any]:
        return {"sid": self.sid, "capture": self.session.get("capture"), "role": self.session.get("role"),
                "images": len(self.images), "bad_frames": len(self.bad_frames), "tracks": len(self.tracks),
                "plays": len(self.plays), "minutes": round(self.minutes, 3), "frames_sha1": self.frames_sha1,
                "emit_created": self.emit.get("created"), "plays_list": self.plays_doc.get("list"),
                "pending": len(self.emit.get("pending") or ())}


@dataclass(frozen=True)
class Pred:
    """One replayed detection of a scored type, box normalised to its frame (x1, y1, x2, y2)."""
    cls: str
    label: str
    side: int | None
    conf: float
    box: tuple[float, float, float, float]
    type: str = TYPE_UNIT
    form: int = 0


@dataclass(frozen=True)
class PredPlay:
    """One replayed OppPlay: t on the capture clock (t_origin + tick / 20), card a name or UNKNOWN."""
    card: str
    card_id: int
    t: float
    x: float | None
    y: float | None
    kind: str = "unit"
    cost: int | None = None
    uids: tuple[int, ...] = ()
    form: int | None = None
    tick: int | None = None
    sid: str | None = None


@dataclass
class ReplayRun:
    images: dict[str, list[Pred]]                      # recorder image stem -> predictions
    plays: list[PredPlay]
    t_origin: float | None
    summary: dict[str, Any]
    markers: list[tuple[float, float | None, float | None]] = field(default_factory=list)
    frames: int = 0


@dataclass
class Check:
    id: str
    group: str
    value: float | None
    threshold: float | None
    op: str
    n: int
    min_n: int
    status: str
    required: bool = False
    ci: tuple[float, float] | None = None
    baseline: float | None = None
    note: str = ""
    unit: str = UNIT_SHARE


@dataclass
class GateResult:
    name: str
    out_dir: Path
    doc: dict[str, Any]
    verdict: str
    checks: list[Check]
    report: str


# ---------------------------------------------------------------------------
# names: card-level and form-aware
# ---------------------------------------------------------------------------

class Keys:
    """A detector or gold name -> (card-level class, form-aware key, type), cached. The class is
    what vision_replay scores (Detector._cls_name: a unit's class, a spell's card, any other its
    name); the form-aware key is the class's detector key in the form (MinionHorde_EV1)."""

    def __init__(self, table: Any):
        self.table = table
        self._cache: dict[tuple[str, int | None], tuple[str, str, str]] = {}

    def of(self, name: Any, form: int | None = None) -> tuple[str, str, str]:
        k = (str(name), None if form is None else int(form))
        hit = self._cache.get(k)
        if hit is None:
            base, _side = split_side(str(name))
            lab = resolve_label(self.table, base)
            typ = str(lab.type)
            f = int(lab.form or 0) if form is None else int(form)
            if typ in (TYPE_UNIT, TYPE_TOWER) and lab.name:
                cls = str(lab.name)
                label = self.table.detector_key(lab.class_id, f) or cls
            elif typ == TYPE_SPELL and lab.card:
                cls = str(lab.card)
                label = cls if f == 0 else f"{cls}{FORM_SUFFIX.get(f, f'_F{f}')}"
            else:
                cls = label = base
            hit = self._cache[k] = (cls, label, typ)
        return hit


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _ratio(a: float, b: float) -> float | None:
    return None if not b else a / b


def _wilson(k: int, n: int) -> list[float] | None:
    return None if n <= 0 else [round(v, 6) for v in VA.wilson(int(k), int(n))]


def _side(v: Any) -> int | None:
    try:
        s = int(v)
    except (TypeError, ValueError):
        return None
    return s if s in (0, 1) else None


def _norm_box(box: Sequence[float], size: Sequence[int]) -> tuple[float, float, float, float]:
    w, h = float(size[0]), float(size[1])
    return (float(box[0]) / w, float(box[1]) / h, float(box[2]) / w, float(box[3]) / h)


def ioa(region: Sequence[float], box: Sequence[float]) -> float:
    """The share of `box` inside `region` (intersection over the box's area)."""
    ix = max(0.0, min(float(region[2]), float(box[2])) - max(float(region[0]), float(box[0])))
    iy = max(0.0, min(float(region[3]), float(box[3])) - max(float(region[1]), float(box[1])))
    area = max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
    return 0.0 if area <= 0.0 else ix * iy / area


def _dist(ax: Any, ay: Any, bx: Any, by: Any) -> float | None:
    if None in (ax, ay, bx, by):
        return None
    return math.hypot(float(ax) - float(bx), float(ay) - float(by))


def _cost(v: Any) -> int:
    return 0 if v is None else int(v)


def _now(clock: Callable[[], float]) -> str:
    return datetime.fromtimestamp(clock()).isoformat(timespec="seconds")


def _shown(path: Any) -> str | None:
    if path is None:
        return None
    return VA.shown_path(path) or str(path)


def _inside(child: Path, parent: Path) -> bool:
    try:
        c, p = Path(child).resolve(), Path(parent).resolve()
    except OSError:
        return False
    return c == p or p in c.parents


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def data_digest(root: Path = ROOT, *, spawn_table: Any = None, geometry: Any = None) -> dict[str, str]:
    """SHA-1s of what the live chain reads besides its code (DATA_FILES, DATA_GLOBS: "data"), the
    spawn table vision_replay resolves (engine_card_ids; "absent" without one: the card ids are
    then classes.json's) and a --geometry file: a replay is redone when one of them changed."""
    h = hashlib.sha1()
    files = [root / rel for rel in DATA_FILES]
    for pattern in DATA_GLOBS:
        files += sorted(p for p in root.glob(pattern) if p.is_file())
    for p in files:
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = p.as_posix()
        h.update(rel.encode("utf-8") + b"\0")
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<missing>")
    out = {"data": h.hexdigest(),
           "spawn_table": _sha1_file(Path(spawn_table) if spawn_table else VR.SPAWN_TABLE) or "absent"}
    if geometry:
        out["geometry"] = _sha1_file(Path(geometry)) or "absent"
    return out


def card_ids_digest(ids: Mapping[str, int]) -> str:
    """SHA-1 of a card name -> id map: the ids a replay's OppPlays carry and read_replay decodes."""
    return hashlib.sha1(json.dumps(sorted((str(k), int(v)) for k, v in ids.items())).encode("utf-8")).hexdigest()


def chain_digest(root: Path = ROOT) -> str:
    """SHA-1 over the live chain's source files (CHAIN_FILES): a changed chain is replayed again."""
    h = hashlib.sha1()
    for rel in CHAIN_FILES:
        p = root / rel
        h.update(rel.encode("utf-8") + b"\0")
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# gold input
# ---------------------------------------------------------------------------

def load_gold(sid: str, *, labels_root: Any = None, store: Any = None, table: Any = None,
              keys: Keys | None = None) -> GoldSession:
    """A gold session ready to score (module doc), or GateError: not gold (2), its gold emit stale or
    its play list not confirmed or stale (1), its raws gone (1)."""
    try:
        paths = VA.session_paths(sid, labels_root, store)
        session = VA.load_session(sid, labels_root)
    except (OSError, ValueError) as exc:
        raise GateError(f"no ingested session {sid} ({exc})", EXIT_INPUT) from None
    if not VA.is_gold(session):
        raise GateError(f"{sid} is not a gold session (role {session.get('role')}, capture {session.get('capture')}): "
                        "the gate scores held-out gold battles only", EXIT_INPUT)
    stale = VA.emit_fresh(paths, session)
    if stale:
        raise GateError(f"{sid}: the gold emit cannot be scored ({'; '.join(stale)}): run "
                        f"vision_autolabel.py emit {sid} --force, then review, mark and plays again")
    why = VA.plays_fresh(paths)
    if why:
        raise GateError(f"{sid}: the gold play list cannot be scored ({'; '.join(why)}): run vision_autolabel.py "
                        f"plays {sid}, check it, then plays {sid} --list K --confirm")
    if not paths.frames.is_file():
        raise GateError(f"{sid}: the gold raws are gone ({paths.frames} is missing): the live chain needs them "
                        "(plan 4.7: keep gold raws)")
    table = VA.load_table() if table is None else table
    keys = keys or Keys(table)
    out = paths.gold_dir
    try:
        emit = dict(_read_json(out / VA.EMIT_JSON))
        rows = VA.read_jsonl(out / VA.GOLD_FILE)
        parts = VA.read_jsonl(out / VA.PARTS_FILE)
        plays_doc = VA.read_plays(paths) or {}
        marks = VA.read_marks(paths)
    except (OSError, ValueError, VA.LabelError) as exc:
        raise GateError(f"{sid}: the gold files cannot be read ({exc}): run vision_autolabel.py emit {sid} --force") from None
    bad_i = VA.bad_frame_numbers(marks)
    images: dict[str, GoldImage] = {}
    bad: set[str] = set()
    for r in rows:
        stem = str(r.get("stem") or Path(str(r.get("image"))).stem)
        i = None if r.get("i") is None else int(r["i"])
        if i is not None and i in bad_i:
            bad.add(stem)
            continue
        size = r.get("size") or (0, 0)
        if not (size and len(size) == 2 and int(size[0]) > 0 and int(size[1]) > 0):
            continue                                   # no frame size: nothing to normalise its boxes by
        images[stem] = GoldImage(sid, stem, int(r.get("n") or 0), i, float(r.get("t") or 0.0),
                                 None if r.get("hk") is None else int(r["hk"]), (int(size[0]), int(size[1])),
                                 [dict(b) for b in r.get("boxes") or ()],
                                 [tuple(float(v) for v in g) for g in r.get("ignore") or ()])
    order = sorted(images.values(), key=lambda g: (g.t, g.n))
    stems_of: dict[str, list[str]] = defaultdict(list)
    for img in order:
        for b in img.boxes:
            if b.get("key") is not None and b.get("src") in LABEL_SOURCES:
                stems_of[str(b["key"])].append(img.stem)
    tracks: dict[str, GoldTrack] = {}
    for p in parts:
        if p.get("decision") != "label" or not p.get("label"):
            continue
        key = str(p.get("key"))
        cls, flabel, _typ = keys.of(p["label"])
        h = p.get("h")
        side = _side(p.get("side"))
        tracks[key] = GoldTrack(sid, key, int(h) if h is not None else int(key.split(".")[0]), int(p.get("part") or 0),
                                str(p["label"]), cls, flabel, int(p.get("form") or 0), 1 if side is None else side,
                                p.get("card"), float(p.get("first_t") or 0.0), list(dict.fromkeys(stems_of.get(key, ()))))
    plays = [dict(p, sid=sid) for p in plays_doc.get("plays") or () if isinstance(p, Mapping)]
    minutes = sum(float(v) for v in (emit.get("battle_seconds") or {}).values()) / 60.0
    warnings = []
    if emit.get("pending"):
        warnings.append(f"{sid}: {len(emit['pending'])} track(s) were not labelled (emit --allow-pending): ignore "
                        "regions, scored as nothing")
    if minutes <= 0.0:
        warnings.append(f"{sid}: emit.json has no battle_seconds: the per-minute rates leave it out")
    return GoldSession(sid, session, emit, images, tracks, plays, minutes, bad, paths.recording, dict(plays_doc), warnings)


def gold_ready(labels_root: Any = None, store: Any = None) -> tuple[list[str], list[tuple[str, str]]]:
    """(gold sessions ready to score, [(sid, why not)] for the other gold sessions)."""
    ready: list[str] = []
    skipped: list[tuple[str, str]] = []
    for sid in VA.session_ids(labels_root):
        try:
            session = VA.load_session(sid, labels_root)
        except (OSError, ValueError) as exc:
            skipped.append((sid, f"session.json does not load ({exc})"))
            continue
        if not VA.is_gold(session):
            continue
        paths = VA.session_paths(sid, labels_root, store)
        why = VA.emit_fresh(paths, session)
        if why:
            skipped.append((sid, "gold emit: " + "; ".join(why)))
            continue
        why = VA.plays_fresh(paths)
        if why:
            skipped.append((sid, "plays: " + "; ".join(why)))
            continue
        ready.append(sid)
    return ready, skipped


# ---------------------------------------------------------------------------
# running the live chain
# ---------------------------------------------------------------------------

def replay_argv(recording: Path, weights: Path, out_dir: Path, options: ReplayOptions | None = None) -> list[str]:
    """vision_replay's command line for one gold recording (time ticks, the live DetectorConfig)."""
    o = options or ReplayOptions()
    argv = [str(recording), "--detector", str(weights), "--ticks", TICKS_TIME, "--out", str(out_dir)]
    if o.unknown_plays:
        argv.append("--unknown-plays")
        if o.unknown_cost is not None:
            argv += ["--unknown-cost", str(int(o.unknown_cost))]
    if o.device:
        argv += ["--device", str(o.device)]
    if o.spawn_table:
        argv += ["--spawn-table", str(o.spawn_table)]
    if o.geometry:
        argv += ["--geometry", str(o.geometry)]
    return argv


def run_replay(recording: Path, weights: Path, out_dir: Path, *, options: ReplayOptions | None = None,
               runner: Callable[[list[str]], int] | None = None) -> Path:
    """Replay one recording into out_dir (runner: vision_replay.main by default, in process: the exact
    live chain) and return the one session folder it wrote; GateError otherwise."""
    run = runner or VR.main
    argv = replay_argv(recording, weights, out_dir, options)
    try:
        rc = run(argv)
    except SystemExit as exc:                 # argparse inside the runner
        rc = exc.code if isinstance(exc.code, int) else 2
    except Exception as exc:
        raise GateError(f"the replay of {recording} failed: {type(exc).__name__}: {exc}") from None
    if rc not in (0, None):
        raise GateError(f"vision_replay exited {rc} on {recording}")
    sessions = list_sessions(out_dir)
    if len(sessions) != 1:
        raise GateError(f"the replay of {recording} left {len(sessions)} session folders in {out_dir}, not 1")
    return sessions[0]


def cached_replay(gold: GoldSession, weights: Path, sha256: str, cache_root: Path, options: ReplayOptions, *,
                  reuse: bool = True, runner: Callable[[list[str]], int] | None = None,
                  log: Callable[[str], Any] = print, card_ids: str | None = None) -> tuple[Path, bool]:
    """(the replay session of `weights` on `gold`, reused?): reports/vision/gate/cache/<sha12>/<sid>_<flags8>,
    reused while key.json holds the same weights, recording, flags, live-chain sources, the data
    they read (data_digest) and card ids (card_ids: card_ids_digest of the ids the gate decodes)."""
    flags = options.flags()
    flags8 = hashlib.sha1(json.dumps(flags, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    out = Path(cache_root) / sha256[:12] / f"{gold.sid}_{flags8}"
    key = {"schema": KEY_SCHEMA, "weights_sha256": sha256, "sid": gold.sid, "frames_sha1": gold.frames_sha1,
           "flags": flags, "chain": chain_digest(),
           **data_digest(spawn_table=options.spawn_table, geometry=options.geometry), "card_ids": card_ids}
    kp = out / KEY_FILE
    if reuse and kp.is_file():
        try:
            old = _read_json(kp)
        except (OSError, ValueError):
            old = None
        if isinstance(old, Mapping) and all(old.get(k) == v for k, v in key.items()):
            sessions = list_sessions(out)
            if len(sessions) == 1 and (sessions[0] / SUMMARY_FILE).is_file():
                return sessions[0], True
    if out.exists():
        for d in list_sessions(out):          # only replay sessions (folders holding frames.jsonl) and the key
            shutil.rmtree(d)
        kp.unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    log(f"  replaying {gold.sid} with {Path(weights).name} ({sha256[:12]}) ...")
    session = run_replay(gold.recording, Path(weights), out, options=options, runner=runner)
    VA.write_json(kp, {**key, "argv": replay_argv(gold.recording, Path(weights), out, options),
                       "session": session.name, "created": datetime.now().isoformat(timespec="seconds")})
    return session, False


def _majority(values: Iterable[int]) -> int | None:
    c = Counter(int(v) for v in values)
    if not c:
        return None
    return max(c.items(), key=lambda kv: (kv[1], kv[0]))[0]


def read_replay(session_dir: Path, table: Any, card_names: Mapping[int, str], *, geometry: Any = None,
                keys: Keys | None = None, sha256: str | None = None, ids_from: str | None = None) -> ReplayRun:
    """What a vision_replay session read (module doc): predictions per image stem (units and
    spells), the OppPlays on the capture clock, enemy stopwatches, and the summary. GateError
    when the replay did not run on time ticks or without its detector (or with other weights
    than sha256), or took its card ids from elsewhere than ids_from (vision_replay's summary
    "card_ids": its OppPlays' ids would decode to other cards)."""
    keys = keys or Keys(table)
    images: dict[str, list[Pred]] = {}
    raw_plays: list[tuple[Any, list[int]]] = []
    marker_ticks: list[tuple[float, float | None, float | None]] = []
    origin = first_t = None
    summary: dict[str, Any] | None = None
    frames = 0
    for e in VisionLogReader(session_dir):
        if e.kind == "replay_summary" and summary is None:
            summary = dict(e.data.get("data") or {})
            continue
        if e.kind != "frame" or e.frame is None:
            continue
        vf = e.frame
        frames += 1
        info = (e.data.get("extra") or {}).get("replay") or {}
        if first_t is None:
            first_t = float(vf.t_capture)
        if origin is None and info.get("tick_source") == TICKS_TIME and info.get("track_tick") is not None:
            origin = float(vf.t_capture) - int(info["track_tick"]) / TICKS_PER_S
        size = vf.frame_size
        name = info.get("image")
        if name and size and size[0] and size[1]:
            w, h = float(size[0]), float(size[1])
            preds = []
            for d in vf.detections:
                cls, label, typ = keys.of(d.cls, d.form)
                if cls == MARKER_CLASS and VR.side_of(float(d.side_prob)) == 1:
                    x = y = None
                    if geometry is not None:
                        try:
                            x, y = geometry.px_to_world((d.box[0] + d.box[2]) / 2.0, (d.box[1] + d.box[3]) / 2.0,
                                                        (int(w), int(h)))
                            x, y = float(x), float(y)
                        except Exception:
                            x = y = None
                    marker_ticks.append((float(vf.t_capture), x, y))
                    continue
                if typ not in SCORED_TYPES:
                    continue
                x1, y1, x2, y2 = (float(v) for v in d.box)
                preds.append(Pred(cls, label, VR.side_of(float(d.side_prob)), float(d.conf),
                                  (x1 / w, y1 / h, x2 / w, y2 / h), typ, int(d.form)))
            images.setdefault(Path(str(name)).stem, preds)
        forms = {int(t.uid): int(t.form) for t in vf.tracks}
        for p in vf.events.opp_plays:
            raw_plays.append((p, [forms[u] for u in p.uids if u in forms]))
    path = Path(session_dir) / SUMMARY_FILE
    if path.is_file():
        try:
            summary = dict(_read_json(path))
        except (OSError, ValueError):
            pass
    summary = summary or {}
    if summary.get("ticks") not in (None, TICKS_TIME):
        raise GateError(f"{session_dir}: the replay ran on {summary.get('ticks')} ticks, not time ticks")
    det = str(summary.get("detector") or "")
    if det.startswith("NONE") or det.startswith("off"):
        raise GateError(f"{session_dir}: the replay ran without its detector ({det})")
    m = re.search(r"sha256 ([0-9a-f]{6,64})", det)
    if sha256 and m and not sha256.lower().startswith(m.group(1).lower()):
        raise GateError(f"{session_dir}: the replay ran other weights (sha256 {m.group(1)}, not {sha256[:12]})")
    if ids_from and summary.get("card_ids") and str(summary["card_ids"]) != str(ids_from):
        raise GateError(f"{session_dir}: the replay took its card ids from {summary['card_ids']}, the gate from "
                        f"{ids_from}: its plays would decode to other cards (run the gate with --no-reuse)")
    t0 = origin if origin is not None else first_t
    plays = []
    for p, pforms in raw_plays:
        cid = int(p.card)
        name = UNKNOWN if cid < 0 else str(card_names.get(cid, cid))
        cost = p.cost
        if cost is None and cid >= 0:
            e = (table.cards.get(name) or {}).get("elixir")
            cost = None if e is None else int(e)
        plays.append(PredPlay(name, cid, (t0 or 0.0) + int(p.tick) / TICKS_PER_S, p.x_mt, p.y_mt, str(p.kind),
                              None if cost is None else int(cost), tuple(int(u) for u in p.uids), _majority(pforms),
                              int(p.tick)))
    return ReplayRun(images, plays, t0, summary, marker_ticks, frames)


# ---------------------------------------------------------------------------
# scoring (pure Python on the read structures)
# ---------------------------------------------------------------------------

def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap_ratio(units: Sequence[tuple[float, float]], *, boot: int, seed: Any) -> tuple[float, float]:
    """The 95% percentile bootstrap interval of sum(a) / sum(b) over resampled units (a, b): a
    track's (hits, boxes) for recall, a 1-s block's (true, all predictions) for precision.
    (0.0, 1.0) when there is nothing to resample. The same seed gives the same interval."""
    units = [(float(a), float(b)) for a, b in units]
    if not units or sum(b for _a, b in units) <= 0.0:
        return (0.0, 1.0)
    rng = random.Random(seed)
    m = len(units)
    vals = []
    for _ in range(int(boot)):
        sample = rng.choices(units, k=m)
        den = sum(b for _a, b in sample)
        if den > 0.0:
            vals.append(sum(a for a, _b in sample) / den)
    if not vals:
        return (0.0, 1.0)
    vals.sort()
    return (round(_quantile(vals, 0.025), 6), round(_quantile(vals, 0.975), 6))


def _row(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {"tp": tp, "fp": fp, "fn": fn, "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn)}


def _best(preds: Iterable[Pred], box: Sequence[float], iou_min: float) -> Pred | None:
    """The prediction overlapping `box` most, at IoU >= iou_min (ties: the stronger)."""
    best, best_key = None, None
    for p in preds:
        iou = box_iou(p.box, box)
        if iou < iou_min:
            continue
        k = (iou, p.conf)
        if best_key is None or k > best_key:
            best, best_key = p, k
    return best


def score_detection(gold: Mapping[str, GoldImage], preds: Mapping[str, Sequence[Pred]], table: Any, *,
                    new_labels: Iterable[str], own_labels: Iterable[str], config: GateConfig | None = None,
                    keys: Keys | None = None, new_classes: Iterable[str] | None = None, boot: bool = True) -> dict[str, Any]:
    """Detection on gold images (module doc). gold and preds share keys (here "<sid>/<stem>"); an
    image with no predictions entry was not replayed (counted, scored as nothing detected).
    Labels rows: per gold label, card-level (its class) and form-aware (its key): recall over its
    boxes (bootstrap units: its harvest tracks), precision of its class / key over all
    predictions (units: 1-s image blocks); intervals by bootstrap for new labels, Wilson
    otherwise. Side accuracy and our labels' recall ("own") count only boxes the owner vetted
    (a part's box: an enemy label, or one of our o tracks); tonight's own pseudo-labels (src
    live, no part) are tonight's model's own reads, so its replay agrees with them by
    construction: their agreement is reported apart ("side_live", "own_live"), as info."""
    cfg = config or GateConfig()
    keys = keys or Keys(table)
    new_labels = set(new_labels)
    new_classes = set(new_classes) if new_classes is not None else {keys.of(x)[0] for x in new_labels}
    own_labels = set(own_labels)
    levels = ("card", "form")
    count = {lv: {"tp": Counter(), "fp": Counter(), "fn": Counter()} for lv in levels}
    ignored: Counter = Counter()
    rec_units = {lv: defaultdict(lambda: defaultdict(lambda: [0, 0])) for lv in levels}
    prec_units = {lv: defaultdict(lambda: defaultdict(lambda: [0, 0])) for lv in levels}
    label_tracks: dict[str, set[str]] = defaultdict(set)
    label_cls: dict[str, tuple[str, str]] = {}
    side_n = side_ok = 0
    live_n = live_ok = 0
    own: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    own_live: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    on_own: Counter = Counter()
    on_own_n = 0
    unreplayed = 0
    for k in sorted(gold):
        img = gold[k]
        ps = preds.get(k)
        if ps is None:
            unreplayed += 1
            ps = ()
        block = f"{img.sid}:{int(math.floor(img.t / cfg.block_s))}"
        boxes = [b for b in img.boxes if b.get("type") in SCORED_TYPES and b.get("label")]
        ign = [_norm_box(r, img.size) for r in img.ignore]
        for lv in levels:
            truth, meta = [], []
            for b in boxes:
                cls, flab, typ = keys.of(b["label"])
                truth.append(VR.Box(cls if lv == "card" else flab, _norm_box(b["box"], img.size), _side(b.get("side")),
                                    1.0, typ))
                meta.append(b)
                label_cls[str(b["label"])] = (cls, flab)
            pb = [VR.Box(p.cls if lv == "card" else p.label, p.box, p.side, p.conf, p.type) for p in ps]
            tp, fp, fn = VR.match_boxes(truth, pb, cfg.match_iou)
            hit = {id(g) for g, _p in tp}
            for g, p in tp:
                count[lv]["tp"][g.cls] += 1
                u = prec_units[lv][p.cls][block]
                u[0] += 1
                u[1] += 1
            for g in fn:
                count[lv]["fn"][g.cls] += 1
            for p in fp:
                if any(ioa(r, p.box) >= cfg.ignore_ioa for r in ign):
                    ignored[lv] += 1
                    continue
                count[lv]["fp"][p.cls] += 1
                prec_units[lv][p.cls][block][1] += 1
            for g, b in zip(truth, meta):
                # a harvest track is the bootstrap unit (40.0 and 40.1 are one track), as its evidence n
                unit = f"{img.sid}:{str(b['key']).split('.')[0]}" if b.get("key") is not None else block
                u = rec_units[lv][str(b["label"])][unit]
                u[0] += int(id(g) in hit)
                u[1] += 1
            if lv != "card":
                continue
            vetted = {id(g) for g, b in zip(truth, meta) if b.get("key") is not None}
            for g, p in tp:
                if g.side is not None and p.side is not None:
                    if id(g) in vetted:
                        side_n += 1
                        side_ok += int(g.side == p.side)
                    else:
                        live_n += 1
                        live_ok += int(g.side == p.side)
            for g, b in zip(truth, meta):
                lab = str(b["label"])
                if b.get("key") is not None and b.get("src") in LABEL_SOURCES:
                    label_tracks[lab].add(f"{img.sid}:{str(b['key']).split('.')[0]}")
                if _side(b.get("side")) == 0 and b.get("src") == SRC_LIVE and lab in own_labels:
                    o = own[lab] if b.get("key") is not None else own_live[lab]
                    o[0] += int(id(g) in hit)
                    o[1] += 1
        for b in boxes:
            if _side(b.get("side")) != 0 or b.get("src") != SRC_LIVE:
                continue
            on_own_n += 1
            best = _best(ps, _norm_box(b["box"], img.size), cfg.match_iou)
            if best is not None and (best.label in new_labels or best.cls in new_classes):
                on_own[best.label] += 1
    per_class = {}
    for lv in levels:
        c = count[lv]
        per_class[lv] = {name: _row(c["tp"][name], c["fp"][name], c["fn"][name])
                         for name in sorted(set(c["tp"]) | set(c["fp"]) | set(c["fn"]))}
    labels: dict[str, Any] = {}
    names = set(rec_units["card"]) | {x for x in new_labels if x in prec_units["form"]}
    for lab in sorted(names):
        cls, flab = label_cls.get(lab) or keys.of(lab)[:2]
        new = lab in new_labels
        row: dict[str, Any] = {"class": cls, "form_label": flab, "new": new, "tracks": len(label_tracks.get(lab, ())),
                               "boxes": sum(u[1] for u in rec_units["card"].get(lab, {}).values())}
        for lv, key in (("card", cls), ("form", flab)):
            r_units = [tuple(u) for u in rec_units[lv].get(lab, {}).values()]
            p_units = [tuple(u) for u in prec_units[lv].get(key, {}).values()]
            rtp, rn = sum(u[0] for u in r_units), sum(u[1] for u in r_units)
            ptp, pn = sum(u[0] for u in p_units), sum(u[1] for u in p_units)
            d = {"tp": rtp, "n": rn, "recall": _ratio(rtp, rn), "pred_tp": ptp, "pred_n": pn,
                 "precision": _ratio(ptp, pn)}
            if new and boot:
                d["recall_ci"] = list(bootstrap_ratio(r_units, boot=cfg.boot, seed=f"{cfg.seed}:{lab}:{lv}:recall")) if rn else None
                d["precision_ci"] = (list(bootstrap_ratio(p_units, boot=cfg.boot, seed=f"{cfg.seed}:{lab}:{lv}:precision"))
                                     if pn else None)
                d["ci"] = "bootstrap"
            else:
                d["recall_ci"], d["precision_ci"], d["ci"] = _wilson(rtp, rn), _wilson(ptp, pn), "wilson"
            row[lv] = d
        labels[lab] = row
    return {"images": len(gold), "unreplayed": unreplayed, "ignored_fp": dict(ignored), "per_class": per_class,
            "labels": labels,
            "side": {"n": side_n, "ok": side_ok, "accuracy": _ratio(side_ok, side_n), "ci": _wilson(side_ok, side_n)},
            "side_live": {"n": live_n, "ok": live_ok, "accuracy": _ratio(live_ok, live_n)},
            "own": {lab: {"tp": v[0], "n": v[1], "recall": _ratio(v[0], v[1])} for lab, v in sorted(own.items())},
            "own_live": {lab: {"tp": v[0], "n": v[1], "recall": _ratio(v[0], v[1])} for lab, v in sorted(own_live.items())},
            "newlabel_on_own": {"n": on_own_n, "hits": sum(on_own.values()), "share": _ratio(sum(on_own.values()), on_own_n),
                                "ci": _wilson(sum(on_own.values()), on_own_n), "by_label": dict(on_own.most_common())}}


def read_table(gold: Mapping[str, GoldImage], preds: Mapping[str, Sequence[Pred]], *,
               config: GateConfig | None = None) -> dict[str, dict[str, int]]:
    """Gold label -> what the model read on its boxes (the owner's labels only): the form-aware key
    of the prediction overlapping each gold box most (IoU >= match_iou, any class), else "none".
    Plan 4.3: what tonight's model reads each new card as."""
    cfg = config or GateConfig()
    out: dict[str, Counter] = defaultdict(Counter)
    for k in sorted(gold):
        img = gold[k]
        ps = preds.get(k) or ()
        for b in img.boxes:
            if b.get("src") not in LABEL_SOURCES or b.get("type") not in SCORED_TYPES:
                continue
            best = _best(ps, _norm_box(b["box"], img.size), cfg.match_iou)
            out[str(b["label"])]["none" if best is None else best.label] += 1
    return {lab: dict(c.most_common()) for lab, c in sorted(out.items())}


def birth_window_recall(tracks: Mapping[str, GoldTrack], gold: Mapping[str, GoldImage],
                        preds: Mapping[str, Sequence[Pred]], config: GateConfig | None = None) -> dict[str, Any]:
    """Plan 6.1's birth window (module doc): per enemy gold track's first part, its first
    birth_images scored images; a hit is an image whose top prediction on the track's gold box
    (the strongest at IoU >= match_iou) has the track's class; the track counts with
    >= birth_hits hits. Tracks with fewer images than birth_hits are left out ("short")."""
    cfg = config or GateConfig()
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    spans: list[float] = []
    n = ok = short = 0
    for tk in sorted(tracks):
        tr = tracks[tk]
        if tr.part != 0 or tr.side != 1:
            continue
        stems = [s for s in tr.stems if f"{tr.sid}/{s}" in gold][:cfg.birth_images]
        if len(stems) < cfg.birth_hits:
            short += 1
            continue
        hits = 0
        for s in stems:
            k = f"{tr.sid}/{s}"
            img = gold[k]
            box = next((b for b in img.boxes if str(b.get("key")) == tr.key and b.get("src") in LABEL_SOURCES), None)
            if box is None:
                continue
            gb = _norm_box(box["box"], img.size)
            top = None
            for p in preds.get(k) or ():
                if box_iou(p.box, gb) >= cfg.match_iou and (top is None or p.conf > top.conf):
                    top = p
            if top is not None and top.cls == tr.cls:
                hits += 1
        good = hits >= cfg.birth_hits
        n += 1
        ok += int(good)
        r = per[tr.label]
        r[0] += int(good)
        r[1] += 1
        spans.append(gold[f"{tr.sid}/{stems[-1]}"].t - gold[f"{tr.sid}/{stems[0]}"].t)
    return {"tracks": n, "hits": ok, "recall": _ratio(ok, n), "ci": _wilson(ok, n), "short": short,
            "images": cfg.birth_images, "need": cfg.birth_hits,
            "span_s": None if not spans else round(statistics.median(spans), 3),
            "per_label": {lab: {"hits": v[0], "tracks": v[1], "recall": _ratio(v[0], v[1])}
                          for lab, v in sorted(per.items())}}


def match_plays(gold: Sequence[Mapping[str, Any]], pred: Sequence[PredPlay], *, dt_s: float, tiles: float
                ) -> tuple[list[tuple[int, int, bool]], list[int], list[int]]:
    """(pairs (gold index, pred index, same card), missed gold indices, phantom pred indices).
    Candidates: the same session, |dt| <= dt_s and, when both have a position, within tiles
    tiles. Two stages: same-card pairs first, then any card; each stage takes the most pairs at
    the least total |dt| (then distance) (vision_autolabel.assign_pairs), so a right read is not
    taken by a wrong one nearby and a burst of plays keeps every pair it has."""
    edges = []
    for gi, g in enumerate(gold):
        gt = float(g["t"])
        for pj, p in enumerate(pred):
            if p.sid != g.get("sid"):
                continue
            dt = abs(float(p.t) - gt)
            if dt > dt_s + _EPS:
                continue
            d = _dist(g.get("x_mt"), g.get("y_mt"), p.x, p.y)
            if d is not None and d > tiles * MT_PER_TILE + _EPS:
                continue
            same = str(p.card) == str(g.get("card"))
            edges.append((0 if same else 1, dt + (d or 0.0) * 1e-7, gi, pj))
    hit = VA.assign_pairs(edges)
    pairs = sorted((gi, pj, stage == 0) for gi, (pj, stage) in hit.items())
    used_p = {pj for _gi, pj, _same in pairs}
    return pairs, [i for i in range(len(gold)) if i not in hit], [j for j in range(len(pred)) if j not in used_p]


def score_plays(gold_plays: Sequence[Mapping[str, Any]], pred_plays: Sequence[PredPlay], *, minutes: float,
                new_cards: Iterable[str] = (), config: GateConfig | None = None) -> dict[str, Any]:
    """Plays (module doc): per card, per evolution or hero form, pooled, elixir, the safety net."""
    cfg = config or GateConfig()
    gold, pred = list(gold_plays), list(pred_plays)
    new_cards = set(new_cards)
    pairs, missed, phantom = match_plays(gold, pred, dt_s=cfg.play_dt_s, tiles=cfg.play_tiles)
    g2p = {gi: (pj, same) for gi, pj, same in pairs}
    p2g = {pj: gi for gi, pj, _same in pairs}
    per_card: dict[str, Any] = {}
    for c in sorted({str(g.get("card")) for g in gold} | {str(p.card) for p in pred}):
        gi_c = [i for i, g in enumerate(gold) if str(g.get("card")) == c]
        right = sum(1 for i in gi_c if i in g2p and g2p[i][1])
        pj_c = [j for j, p in enumerate(pred) if str(p.card) == c]
        pright = sum(1 for j in pj_c if j in p2g and str(gold[p2g[j]].get("card")) == c)
        read_as = Counter(str(pred[g2p[i][0]].card) if i in g2p else "missed" for i in gi_c)
        per_card[c] = {"gold": len(gi_c), "right": right, "recall": _ratio(right, len(gi_c)),
                       "recall_ci": _wilson(right, len(gi_c)), "pred": len(pj_c), "pred_right": pright,
                       "precision": _ratio(pright, len(pj_c)), "precision_ci": _wilson(pright, len(pj_c)),
                       "new": c in new_cards, "read_as": dict(read_as.most_common())}
    forms: dict[str, Any] = {}
    for i, g in enumerate(gold):
        f = int(g.get("form") or 0)
        if f <= 0:
            continue
        r = forms.setdefault(f"{g.get('card')}.{f}", {"card": g.get("card"), "form": f, "gold": 0, "right": 0,
                                                      "form_right": 0})
        r["gold"] += 1
        if i in g2p and g2p[i][1]:
            r["right"] += 1
            pf = pred[g2p[i][0]].form
            r["form_right"] += int(pf is not None and int(pf) == f)
    for r in forms.values():
        r["card_accuracy"] = _ratio(r["right"], r["gold"])
        r["card_ci"] = _wilson(r["right"], r["gold"])
        r["form_accuracy"] = _ratio(r["form_right"], r["right"])
    right = sum(1 for _gi, _pj, same in pairs if same)
    missed_cost = sum(_cost(gold[i].get("cost")) for i in missed)
    phantom_cost = sum(_cost(pred[j].cost) for j in phantom)
    wrong_cost = sum(abs(_cost(pred[pj].cost) - _cost(gold[gi].get("cost"))) for gi, pj, same in pairs if not same)
    no_cost = sum(1 for g in gold if g.get("cost") is None) + sum(1 for p in pred if p.cost is None)
    elixir = missed_cost + phantom_cost + wrong_cost
    unknown = [j for j, p in enumerate(pred) if p.card == UNKNOWN]
    unk_matched = sum(1 for j in unknown if j in p2g)
    per_min = (lambda v: None) if minutes <= 0 else (lambda v: v / minutes)
    gold_spent = sum(_cost(g.get("cost")) for g in gold)
    pred_spent = sum(_cost(p.cost) for p in pred)
    return {"gold": len(gold), "pred": len(pred), "matched": len(pairs), "right": right, "wrong": len(pairs) - right,
            "missed": len(missed), "phantoms": len(phantom), "minutes": round(float(minutes), 4),
            "play_id": _ratio(right, len(gold)), "play_id_ci": _wilson(right, len(gold)),
            "phantom_per_min": per_min(len(phantom)),
            "elixir": {"error": elixir, "missed": missed_cost, "phantom": phantom_cost, "wrong_card": wrong_cost,
                       "per_min": per_min(elixir), "no_cost": no_cost, "gold_spent": gold_spent, "pred_spent": pred_spent,
                       "gold_spent_per_min": per_min(gold_spent), "pred_spent_per_min": per_min(pred_spent)},
            "unknown": {"pred": len(unknown), "matched": unk_matched, "matched_share": _ratio(unk_matched, len(unknown)),
                        "ci": _wilson(unk_matched, len(unknown)), "share": _ratio(len(unknown), len(pred))},
            "per_card": per_card, "forms": forms,
            "pairs": [[gi, pj, same] for gi, pj, same in pairs], "missed_ids": [gold[i].get("id") for i in missed]}


def marker_recall(markers: Sequence[tuple[str | None, float, float | None, float | None]],
                  gold_plays: Sequence[Mapping[str, Any]], config: GateConfig | None = None) -> dict[str, Any]:
    """Plan 6.2's marker (N03 is not built: report only): the share of gold unit plays with an
    enemy stopwatch (KataCR clock) seen within play_dt_s and play_tiles of the play."""
    cfg = config or GateConfig()
    units = [g for g in gold_plays if g.get("kind") != "spell"]
    hits = 0
    for g in units:
        for sid, t, x, y in markers:
            if sid != g.get("sid") or abs(float(t) - float(g["t"])) > cfg.play_dt_s + _EPS:
                continue
            d = _dist(g.get("x_mt"), g.get("y_mt"), x, y)
            if d is not None and d > cfg.play_tiles * MT_PER_TILE + _EPS:
                continue
            hits += 1
            break
    return {"plays": len(units), "hits": hits, "recall": _ratio(hits, len(units)), "ci": _wilson(hits, len(units)),
            "markers": len(markers), "note": "N03 is not built: report only"}


# ---------------------------------------------------------------------------
# regression on KataCR val (GPU) and the leak check
# ---------------------------------------------------------------------------

def _yolo(path: str) -> Any:
    from ultralytics import YOLO
    return YOLO(path)


def _sha1_file(path: Path) -> str | None:
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _val_dirs(data_yaml: Path) -> list[Path]:
    """The val folders a dataset yaml names (absolute, or under its path:, else its own folder),
    each with its label folder (images -> labels, ultralytics' rule)."""
    text = Path(data_yaml).read_text(encoding="utf-8-sig")
    m = re.search(r"^path\s*:\s*(.*?)\s*$", text, re.M)
    base = Path(_unquote(m.group(1))) if m and m.group(1) else Path(data_yaml).parent
    if not base.is_absolute():
        base = Path(data_yaml).parent / base
    lines = text.splitlines()
    vals: list[str] = []
    for i, line in enumerate(lines):
        if re.match(r"^val\s*:", line):
            block = [line]
            for item in lines[i + 1:]:
                if not item.strip().startswith("- "):
                    break
                block.append(item)
            vals = _yaml_paths("\n".join(block))
            break
    out: list[Path] = []
    for v in vals:
        root = Path(v) if Path(v).is_absolute() else base / v
        out.append(root)
        parts = list(root.parts)
        if "images" in parts:
            k = len(parts) - 1 - parts[::-1].index("images")
            out.append(Path(*parts[:k], "labels", *parts[k + 1:]))
    return out


def val_signature(data_yaml: Path) -> str:
    """SHA-1 over the val split a dataset yaml names (_val_dirs): every file by relative name,
    size and mtime. cr.yaml holds only paths and names, so a prepare_yolo_dataset run again with
    another split keeps its text; this changes with the split."""
    try:
        dirs = _val_dirs(data_yaml)
    except OSError:
        return "unreadable"
    h = hashlib.sha1()
    for d in dirs:
        h.update(f"{d.as_posix()}\0".encode("utf-8"))
        if not d.exists():
            h.update(b"<missing>")
            continue
        entries = []
        for p in (sorted(d.rglob("*")) if d.is_dir() else [d]):
            if p.is_file():
                st = p.stat()
                rel = p.relative_to(d).as_posix() if d.is_dir() else p.name
                entries.append(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}")
        for e in entries:
            h.update(e.encode("utf-8") + b"\n")
    return h.hexdigest()


def katacr_val(weights: Path, data_yaml: Path, *, device: str | None = None, yolo_factory: Callable[[str], Any] | None = None,
               cache: Path | None = None, imgsz: int = 640, log: Callable[[str], Any] = print) -> dict[str, Any]:
    """ultralytics val of `weights` on the KataCR dataset yaml (split val): mAP50 and AP50 per class
    name, cached in <cache>/<sha12>/katacr_val.json for the same weights, yaml and imgsz. A model
    with more classes than the yaml names is scored over the classes the labels hold (the
    validator takes its names from the model, ultralytics models/yolo/detect/val.py)."""
    sha = file_sha256(weights)
    data_sha = _sha1_file(Path(data_yaml))
    val_sig = val_signature(Path(data_yaml))
    path = None if cache is None else Path(cache) / sha[:12] / KATACR_FILE
    if path is not None and path.is_file():
        try:
            doc = _read_json(path)
            if (doc.get("weights_sha256") == sha and doc.get("data_sha1") == data_sha and doc.get("imgsz") == int(imgsz)
                    and doc.get("val_signature") == val_sig):
                return dict(doc)
        except (OSError, ValueError, AttributeError):
            pass
    log(f"  KataCR val of {Path(weights).name} ({sha[:12]}) on {Path(data_yaml).name} ...")
    model = (yolo_factory or _yolo)(str(weights))
    kw: dict[str, Any] = {"data": str(data_yaml), "split": "val", "imgsz": int(imgsz), "device": device,
                          "plots": False, "verbose": False, "workers": 2}
    if cache is not None:
        kw.update(project=str(Path(cache) / sha[:12] / "val"), name="katacr", exist_ok=True)
    m = model.val(**kw)
    names = m.names if isinstance(getattr(m, "names", None), Mapping) else dict(enumerate(getattr(m, "names", None) or ()))
    box = m.box
    ap = {str(names.get(int(c), int(c))): round(float(a), 6) for c, a in zip(list(box.ap_class_index), list(box.ap50))}
    doc = {"schema": KATACR_SCHEMA, "weights": _shown(weights), "weights_sha256": sha, "data": _shown(data_yaml),
           "data_sha1": data_sha, "val_signature": val_sig, "imgsz": int(imgsz), "map50": round(float(box.map50), 6),
           "classes": len(ap),
           "ap50": ap, "created": datetime.now().isoformat(timespec="seconds")}
    if path is not None:
        VA.write_json(path, doc)
    return doc


def katacr_regression(candidate: Path, baseline: Path, data_yaml: Path, *, device: str | None = None,
                      yolo_factory: Callable[[str], Any] | None = None, cache: Path | None = None,
                      log: Callable[[str], Any] = print) -> dict[str, Any]:
    """Both models' KataCR val (katacr_val) and the drop of the candidate's mAP50 (baseline minus
    candidate; reg.katacr passes while it is <= GateConfig.katacr_drop), with the classes whose
    AP50 fell most."""
    c = katacr_val(candidate, data_yaml, device=device, yolo_factory=yolo_factory, cache=cache, log=log)
    b = katacr_val(baseline, data_yaml, device=device, yolo_factory=yolo_factory, cache=cache, log=log)
    shared = sorted(set(c["ap50"]) & set(b["ap50"]))
    worst = sorted(((round(b["ap50"][n] - c["ap50"][n], 6), n) for n in shared), reverse=True)[:10]
    return {"candidate": c, "baseline": b, "drop": round(float(b["map50"]) - float(c["map50"]), 6),
            "worst": [{"class": n, "drop": d, "candidate": c["ap50"][n], "baseline": b["ap50"][n]} for d, n in worst],
            "only_candidate": len(set(c["ap50"]) - set(b["ap50"])), "only_baseline": len(set(b["ap50"]) - set(c["ap50"]))}


def _yaml_paths(text: str) -> list[str]:
    """The train / val / test paths of a dataset yaml (an inline JSON list, a quoted or bare
    scalar, or a block of "- path" lines), without a yaml dependency."""
    out: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(train|val|test)\s*:\s*(.*?)\s*$", line)
        if not m:
            continue
        rest = m.group(2).split(" #")[0].strip()
        if rest.startswith("["):
            try:
                out += [str(x) for x in json.loads(rest)]
            except ValueError:
                out += [_unquote(x) for x in rest.strip("[]").split(",") if x.strip()]
        elif rest:
            out.append(_unquote(rest))
        else:
            for item in lines[i + 1:]:
                s = item.strip()
                if not s.startswith("- "):
                    break
                out.append(_unquote(s[2:]))
    return out


def _unquote(text: str) -> str:
    s = text.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def _gold_identities(golds: Sequence[GoldSession], labels_root: Any, store: Any) -> list[dict[str, Any]]:
    """Every gold session the owner has (the scored ones and any other is_gold one), as the leak
    check compares them: sid, frames_sha1, recording name and created, and its folders."""
    seen: dict[str, dict[str, Any]] = {}
    sessions = {g.sid: g.session for g in golds}
    for sid in VA.session_ids(labels_root):
        if sid in sessions:
            continue
        try:
            s = VA.load_session(sid, labels_root)
        except (OSError, ValueError):
            continue
        if VA.is_gold(s):
            sessions[sid] = s
    for sid, s in sorted(sessions.items()):
        rec = s.get("recording") or {}
        paths = VA.session_paths(sid, labels_root, store)
        dirs = [paths.store_dir]
        if rec.get("source"):
            dirs.append(ROOT / str(rec["source"]))
        seen[sid] = {"sid": sid, "frames_sha1": rec.get("frames_sha1"), "name": rec.get("name"),
                     "created": rec.get("created"), "dirs": dirs}
    return list(seen.values())


def leak_check(candidate: Path, golds: Sequence[GoldSession], *, labels_root: Any = None,
               store: Any = None) -> dict[str, Any]:
    """Plan rule "gold never trains" checked on the candidate (module doc, leak): measured is False
    (the check is then insufficient) without a manifest or mix.json."""
    run_dir = Path(candidate).resolve().parent.parent
    manifest = run_dir / MANIFEST_FILE
    out: dict[str, Any] = {"measured": False, "leaks": [], "manifest": _shown(manifest)}
    if not manifest.is_file():
        out["note"] = f"no {MANIFEST_FILE} in {run_dir.name}: leak not checked (train it from a `mix`)"
        return out
    try:
        man = _read_json(manifest)
        data = Path(str(man["data"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        out["note"] = f"{MANIFEST_FILE} cannot be read ({exc}): leak not checked"
        return out
    out["label_set_digest"] = man.get("label_set_digest")
    mix_path = data.parent / MIX_FILE
    out["data"], out["mix"] = _shown(data), _shown(mix_path)
    try:
        mix = _read_json(mix_path)
    except (OSError, ValueError) as exc:
        out["note"] = f"no readable {MIX_FILE} beside {data.name} ({type(exc).__name__}): leak not checked"
        return out
    # the mix.json read now must be the one the run trained on: train_detector records its SHA-1 as
    # the run starts; with an older manifest, mix.json must not be newer than the weights
    changed = None
    if man.get("mix_sha1"):
        if _sha1_file(mix_path) != man["mix_sha1"]:
            changed = f"{MIX_FILE} changed since this run was trained (built again?)"
        elif man.get("data_sha1") and _sha1_file(data) != man["data_sha1"]:
            changed = f"{data.name} changed since this run was trained"
    else:
        try:
            if mix_path.stat().st_mtime > Path(candidate).stat().st_mtime + 1.0:
                changed = f"{MIX_FILE} is newer than the weights (built again after this run?)"
        except OSError:
            changed = f"{MIX_FILE} or the weights cannot be dated"
    sessions = mix.get("sessions") if isinstance(mix, Mapping) else None
    if not isinstance(sessions, list):
        out["note"] = f"{MIX_FILE} has no sessions list: leak not checked"
        return out
    out["measured"] = True
    out["sessions"] = len(sessions)
    golden = _gold_identities(golds, labels_root, store)
    leaks: list[str] = []
    for s in sessions:
        if not isinstance(s, Mapping):
            continue
        rec = s.get("recording")
        name = rec.get("name") if isinstance(rec, Mapping) else rec
        created = rec.get("created") if isinstance(rec, Mapping) else s.get("created")
        fsha = s.get("frames_sha1") or (rec.get("frames_sha1") if isinstance(rec, Mapping) else None)
        for g in golden:
            if s.get("sid") == g["sid"]:
                leaks.append(f"the mix holds gold session {g['sid']}")
            elif fsha and fsha == g["frames_sha1"]:
                leaks.append(f"mix session {s.get('sid')} is the recording of gold session {g['sid']} (frames_sha1)")
            elif name and created and (name, created) == (g["name"], g["created"]):
                leaks.append(f"mix session {s.get('sid')} is the recording of gold session {g['sid']} (name and created)")
    try:
        paths = _yaml_paths(data.read_text(encoding="utf-8-sig"))
    except OSError:
        paths = []
        out["note"] = f"{data.name} cannot be read: its folders were not checked"
    for p in paths:
        for g in golden:
            if any(_inside(Path(p), d) for d in g["dirs"]):
                leaks.append(f"{data.name} trains or validates on {p}, under gold session {g['sid']}")
    out["leaks"] = sorted(set(leaks))
    if changed and not out["leaks"]:
        # nothing found, but not in the mix the weights saw: not checked (a leak found is a leak all the same)
        out["measured"] = False
        out["note"] = f"{changed}: leak not checked; train again from the current mix, or restore its mix.json"
    elif changed:
        out["note"] = changed
    return out


# ---------------------------------------------------------------------------
# checks and the verdict
# ---------------------------------------------------------------------------

def _status(value: float | None, threshold: float | None, op: str, n: int, min_n: int) -> str:
    if value is None or threshold is None or int(n) < int(min_n):
        return ST_INSUFFICIENT
    ok = value >= threshold - _EPS if op == ">=" else value <= threshold + _EPS
    return ST_PASS if ok else ST_FAIL


def _check(cid: str, group: str, value: float | None, threshold: float | None, op: str, n: int, min_n: int, *,
           required: bool = False, ci: Sequence[float] | None = None, baseline: float | None = None, note: str = "",
           info: bool = False, unit: str = UNIT_SHARE) -> Check:
    status = ST_INFO if info else _status(value, threshold, op, n, min_n)
    return Check(cid, group, None if value is None else round(float(value), 6),
                 None if threshold is None else round(float(threshold), 6), op, int(n), int(min_n), status,
                 bool(required) and not info, None if ci is None else (float(ci[0]), float(ci[1])),
                 None if baseline is None else round(float(baseline), 6), note, unit)


def build_checks(cand: Mapping[str, Any], base: Mapping[str, Any] | None, config: GateConfig | None = None, *,
                 m0: Mapping[str, Any] | None = None) -> list[Check]:
    """The checks of plan 6 (module doc) for one model's scores, against the baseline's (None: the
    checks that need a baseline are insufficient). m0: tonight's model's scores, which play.id_gain
    is measured over (plan 6: "at least 0.15 above the baseline", tonight's model); None: the
    baseline's. With an m0 other than the baseline, the gain over the baseline is info."""
    cfg = config or GateConfig()
    base = base or {}
    gain_ref = m0 if m0 is not None else base
    out: list[Check] = []
    det, bdet = cand.get("detection") or {}, base.get("detection") or {}
    for lab, row in sorted((det.get("labels") or {}).items()):
        if not row.get("new"):
            continue
        c = row.get("card") or {}
        bc = ((bdet.get("labels") or {}).get(lab) or {}).get("card") or {}
        out.append(_check(f"det.recall.{lab}", "detection", c.get("recall"), cfg.det_recall, ">=", row.get("tracks", 0),
                          cfg.det_min_tracks, ci=c.get("recall_ci"), baseline=bc.get("recall"),
                          note=f"card-level ({row.get('class')}); n = gold tracks"))
        out.append(_check(f"det.precision.{lab}", "detection", c.get("precision"), cfg.det_precision, ">=",
                          row.get("tracks", 0), cfg.det_min_tracks, ci=c.get("precision_ci"), baseline=bc.get("precision"),
                          note=f"card-level ({row.get('class')}); n = gold tracks"))
    side, bside = det.get("side") or {}, bdet.get("side") or {}
    out.append(_check("det.side", "detection", side.get("accuracy"), cfg.side_acc, ">=", side.get("n", 0),
                      cfg.side_min_pairs, required=True, ci=side.get("ci"), baseline=bside.get("accuracy"),
                      note="n = matched pairs on owner-vetted boxes with both sides known"))
    sl, bsl = det.get("side_live") or {}, bdet.get("side_live") or {}
    if sl.get("n"):
        out.append(_check("det.side_live", "detection", sl.get("accuracy"), None, "", sl.get("n", 0), 0,
                          baseline=bsl.get("accuracy"), info=True,
                          note="agreement with tonight's pseudo-labels (our units, unvetted): not a measure"))
    birth, bbirth = cand.get("birth") or {}, base.get("birth") or {}
    out.append(_check("det.birth", "birth", birth.get("recall"), cfg.birth_recall, ">=", birth.get("tracks", 0),
                      cfg.birth_min_tracks, required=True, ci=birth.get("ci"), baseline=bbirth.get("recall"),
                      note=f"top box on >= {cfg.birth_hits} of the first {cfg.birth_images} images; n = tracks"))
    plays, bplays = cand.get("plays") or {}, base.get("plays") or {}
    bcards = bplays.get("per_card") or {}
    for card, row in sorted((plays.get("per_card") or {}).items()):
        if not row.get("new") or card == UNKNOWN:
            continue
        b = bcards.get(card) or {}
        out.append(_check(f"play.recall.{card}", "plays", row.get("recall"), cfg.play_recall, ">=", row.get("gold", 0),
                          cfg.play_min, ci=row.get("recall_ci"), baseline=b.get("recall"), note="n = gold plays"))
        out.append(_check(f"play.precision.{card}", "plays", row.get("precision"), cfg.card_precision, ">=",
                          row.get("gold", 0), cfg.play_min, ci=row.get("precision_ci"), baseline=b.get("precision"),
                          note=f"{row.get('pred_right', 0)} of {row.get('pred', 0)} plays read as {card}; n = gold plays"))
    bforms = bplays.get("forms") or {}
    for key, row in sorted((plays.get("forms") or {}).items()):
        b = bforms.get(key) or {}
        out.append(_check(f"play.form.{key}", "plays", row.get("card_accuracy"), cfg.form_card_acc, ">=",
                          row.get("gold", 0), cfg.play_min, ci=row.get("card_ci"), baseline=b.get("card_accuracy"),
                          note="the right card for the form's gold plays"))
        out.append(_check(f"play.form_acc.{key}", "plays", row.get("form_accuracy"), cfg.form_acc_target, ">=",
                          row.get("right", 0), 0, baseline=b.get("form_accuracy"), info=True,
                          note="the form read right among the right-card plays (target, reported)"))
    n_plays = int(plays.get("gold", 0))
    out.append(_check("play.id", "plays", plays.get("play_id"), cfg.pooled_play_id, ">=", n_plays, cfg.pooled_min_plays,
                      required=True, ci=plays.get("play_id_ci"), baseline=bplays.get("play_id")))
    gplays = (gain_ref or {}).get("plays") or {}
    gain = None
    if plays.get("play_id") is not None and gplays.get("play_id") is not None:
        gain = float(plays["play_id"]) - float(gplays["play_id"])
    same_ref = m0 is None or m0.get("sha256") == base.get("sha256")
    out.append(_check("play.id_gain", "plays", gain, cfg.pooled_gain, ">=", n_plays, cfg.pooled_min_plays, required=True,
                      baseline=gplays.get("play_id"), unit=UNIT_GAIN,
                      note=("over M0 (tonight's model)" if m0 is not None else "over the baseline")
                      + ("" if gain_ref else ": no scores, the gain cannot be measured")))
    if not same_ref and plays.get("play_id") is not None and bplays.get("play_id") is not None:
        out.append(_check("play.id_gain_baseline", "plays", float(plays["play_id"]) - float(bplays["play_id"]), None, "",
                          n_plays, 0, baseline=bplays.get("play_id"), unit=UNIT_GAIN, info=True,
                          note="over the baseline (the current pin), reported"))
    out.append(_check("play.phantom", "plays", plays.get("phantom_per_min"), cfg.phantom_per_min, "<=", n_plays,
                      cfg.pooled_min_plays, required=True, baseline=bplays.get("phantom_per_min"), unit=UNIT_RATE,
                      note=f"{plays.get('phantoms', 0)} phantoms over {plays.get('minutes', 0):g} min"))
    el, bel = plays.get("elixir") or {}, bplays.get("elixir") or {}
    out.append(_check("play.elixir", "plays", el.get("per_min"), cfg.elixir_per_min, "<=", n_plays, cfg.pooled_min_plays,
                      required=True, baseline=bel.get("per_min"), unit=UNIT_RATE,
                      note=f"missed {el.get('missed', 0)} + phantom {el.get('phantom', 0)} + wrong card "
                           f"{el.get('wrong_card', 0)} elixir"))
    unk, bunk = plays.get("unknown") or {}, bplays.get("unknown") or {}
    out.append(_check("play.unknown", "safety", unk.get("matched_share"), cfg.unknown_match, ">=", unk.get("pred", 0),
                      cfg.unknown_min, ci=unk.get("ci"), baseline=bunk.get("matched_share"), note="n = UNKNOWN plays"))
    out.append(_check("play.unknown_share", "safety", unk.get("share"), None, "", plays.get("pred", 0), 0,
                      baseline=bunk.get("share"), info=True, note="UNKNOWN plays among all plays read"))
    kat = cand.get("katacr")
    if kat:
        c_map, b_map = float(kat["candidate"]["map50"]), float(kat["baseline"]["map50"])
        out.append(_check("reg.katacr", "regression", c_map, b_map - cfg.katacr_drop, ">=", 1, 1, required=True,
                          baseline=b_map, unit=UNIT_MAP, note=f"KataCR val mAP50, drop {b_map - c_map:+.4f}"))
    elif cand.get("katacr_val"):                  # baseline mode: the model's own KataCR val, reported
        kv = cand["katacr_val"]
        out.append(_check("reg.katacr", "regression", kv.get("map50"), None, "", 1, 0, unit=UNIT_MAP, info=True,
                          note=f"KataCR val mAP50 over {kv.get('classes')} classes (the gate compares a candidate "
                               "with it)"))
    else:
        out.append(_check("reg.katacr", "regression", None, None, ">=", 0, 1, required=True, unit=UNIT_MAP,
                          note=cand.get("katacr_note") or "KataCR val not run"))
    bown = bdet.get("own") or {}
    for lab, row in sorted((det.get("own") or {}).items()):
        b = bown.get(lab) or {}
        thr = None if b.get("recall") is None else float(b["recall"]) - cfg.own_recall_drop
        out.append(_check(f"reg.own.{lab}", "regression", row.get("recall"), thr, ">=", row.get("n", 0), cfg.own_min_boxes,
                          baseline=b.get("recall"),
                          note="our o tracks' boxes (owner-vetted)" + ("" if base else "; no baseline scores")))
    ol, bol = det.get("own_live") or {}, bdet.get("own_live") or {}
    ol_tp, ol_n = sum(int(v.get("tp", 0)) for v in ol.values()), sum(int(v.get("n", 0)) for v in ol.values())
    b_tp, b_n = sum(int(v.get("tp", 0)) for v in bol.values()), sum(int(v.get("n", 0)) for v in bol.values())
    if ol_n:
        out.append(_check("reg.own_live", "regression", _ratio(ol_tp, ol_n), None, "", ol_n, 0, baseline=_ratio(b_tp, b_n),
                          info=True, note="agreement with tonight's pseudo-labels of our units (unvetted: tonight's "
                                          "own reads, so a measure of sameness, not recall)"))
    on_own = det.get("newlabel_on_own") or {}
    out.append(_check("reg.newlabel_on_own", "regression", on_own.get("share"), cfg.newlabel_on_own, "<=",
                      on_own.get("n", 0), cfg.own_min_boxes, required=True, ci=on_own.get("ci"),
                      baseline=(bdet.get("newlabel_on_own") or {}).get("share"), note="n = our own gold boxes"))
    land, bland = cand.get("landing") or {}, base.get("landing") or {}
    out.append(_check("reg.landing", "regression", land.get("miss_rate"), bland.get("miss_rate"), "<=", land.get("taps", 0),
                      cfg.landing_min_taps, baseline=bland.get("miss_rate"), note="n = own taps the replays matched"))
    leak = cand.get("leak")
    if leak is not None:
        if leak.get("measured"):
            out.append(_check("leak", "leak", len(leak.get("leaks") or ()), 0, "<=", 1, 1, required=True, unit=UNIT_COUNT,
                              note="; ".join(leak.get("leaks") or ()) or f"{leak.get('sessions', 0)} mix sessions, none gold"))
        else:
            out.append(_check("leak", "leak", None, 0, "<=", 0, 1, required=True, unit=UNIT_COUNT,
                              note=leak.get("note") or "leak not checked"))
    marker = cand.get("marker")
    if marker:
        out.append(_check("marker.recall", "marker", marker.get("recall"), None, "", marker.get("plays", 0), 0,
                          ci=marker.get("ci"), info=True, note=marker.get("note", "")))
    return out


def verdict(checks: Sequence[Check]) -> str:
    """FAIL when any check with enough evidence fails (a failing leak always does); else
    INSUFFICIENT when a required check lacks evidence; else PASS (plan 6.4, D22)."""
    if any(c.status == ST_FAIL for c in checks):
        return FAIL
    if any(c.required and c.status == ST_INSUFFICIENT for c in checks):
        return INSUFFICIENT
    return PASS


def verdict_reason(checks: Sequence[Check], v: str) -> str:
    if v == FAIL:
        return "failed: " + ", ".join(c.id for c in checks if c.status == ST_FAIL)
    if v == INSUFFICIENT:
        return "not enough evidence for: " + ", ".join(c.id for c in checks if c.required and c.status == ST_INSUFFICIENT)
    if v == PASS:
        return "every required check passed and no measured check failed"
    return "the baseline's numbers on the gold set (no verdict)"


# ---------------------------------------------------------------------------
# one model on the gold set, and the gate
# ---------------------------------------------------------------------------

@dataclass
class _Context:
    table: Any
    keys: Keys
    card_names: dict[int, str]
    card_ids_from: str
    geometry: Any
    new_labels: set[str]
    new_classes: set[str]
    new_cards: set[str]
    own_labels: set[str]
    config: GateConfig
    options: ReplayOptions
    cache_root: Path
    reuse: bool
    runner: Callable[[list[str]], int] | None
    log: Callable[[str], Any]


def label_sets(golds: Sequence[GoldSession], table: Any, keys: Keys, registry: Any,
               base_labels: Any = VA.BASE_LABELS) -> dict[str, set[str]]:
    """new_labels: the registry's labels and every gold label tonight's set lacks (form-aware);
    new_classes: their classes the base set has no label of; new_cards: gold (and registry) cards
    whose units or spell label tonight's set lacks (BossBandit, Vines; not the Minion Horde, whose
    evolution is a new form of a known class); own_labels: Yodle2's deck labels."""
    base = VA.base_labels(table, base_labels)
    have = set(base.labels)
    base_cls = {keys.of(x)[0] for x in have}
    gold_labels = {t.label for g in golds for t in g.tracks.values()}
    reg_labels = set(registry.labels) if registry is not None else set()
    new_labels = reg_labels | {x for x in gold_labels if x not in have}
    new_classes = {keys.of(x)[0] for x in new_labels} - base_cls

    def card_is_new(card: str) -> bool:
        info = table.cards.get(card) or {}
        classes = list(info.get("classes") or ())
        if classes:
            return any(c not in base_cls for c in classes)
        if info.get("kind") == "spell":
            return VA.spell_label(table, card, have) not in have
        return False

    cards = {str(p.get("card")) for g in golds for p in g.plays if p.get("card")}
    cards |= {str(getattr(e, "card", "")) for e in (registry or ()) if getattr(e, "card", None)}
    return {"new_labels": new_labels, "new_classes": new_classes, "new_cards": {c for c in cards if card_is_new(c)},
            "own_labels": set(table.deck_labels(list(VA.OUR_DECK))), "base_labels": have}


def evaluate_model(weights: Path, golds: Sequence[GoldSession], ctx: _Context) -> dict[str, Any]:
    """One model's scores on the gold set: every gold battle replayed (cached), read and scored."""
    sha = file_sha256(weights)
    images: dict[str, GoldImage] = {}
    preds: dict[str, list[Pred]] = {}
    tracks: dict[str, GoldTrack] = {}
    gold_plays: list[dict[str, Any]] = []
    pred_plays: list[PredPlay] = []
    markers: list[tuple[str | None, float, float | None, float | None]] = []
    landed = missed = 0
    minutes = 0.0
    replays, warnings = [], []
    for g in golds:
        session_dir, reused = cached_replay(g, weights, sha, ctx.cache_root, ctx.options, reuse=ctx.reuse,
                                            runner=ctx.runner, log=ctx.log,
                                            card_ids=card_ids_digest({n: i for i, n in ctx.card_names.items()}))
        rr = read_replay(session_dir, ctx.table, ctx.card_names, geometry=ctx.geometry, keys=ctx.keys, sha256=sha,
                         ids_from=ctx.card_ids_from)
        for stem, img in g.images.items():
            images[f"{g.sid}/{stem}"] = img
            if stem in rr.images:
                preds[f"{g.sid}/{stem}"] = rr.images[stem]
        gone = sum(1 for stem in g.images if stem not in rr.images)
        if gone:
            warnings.append(f"{g.sid}: {gone} of {len(g.images)} gold images have no replayed frame (scored as nothing "
                            "detected): the recording and the gold emit disagree, or the replay skipped them")
        tracks.update({f"{g.sid}/{k}": t for k, t in g.tracks.items()})
        gold_plays += g.plays
        pred_plays += [replace(p, sid=g.sid) for p in rr.plays]
        markers += [(g.sid, t, x, y) for t, x, y in rr.markers]
        ev = rr.summary.get("events") or {}
        landed += int(ev.get("own_landed") or 0)
        missed += int(ev.get("own_missed") or 0)
        minutes += g.minutes
        replays.append({"sid": g.sid, "session": _shown(session_dir), "reused": reused, "frames": rr.frames,
                        "plays": len(rr.plays), "t_origin": rr.t_origin, "detector": rr.summary.get("detector"),
                        "unknown_plays": rr.summary.get("unknown_plays"), "card_ids": rr.summary.get("card_ids")})
    det = score_detection(images, preds, ctx.table, new_labels=ctx.new_labels, own_labels=ctx.own_labels,
                          config=ctx.config, keys=ctx.keys, new_classes=ctx.new_classes)
    return {"weights": _shown(weights), "sha256": sha, "detection": det,
            "birth": birth_window_recall(tracks, images, preds, ctx.config),
            "plays": score_plays(gold_plays, pred_plays, minutes=minutes, new_cards=ctx.new_cards, config=ctx.config),
            "reads": read_table(images, preds, config=ctx.config),
            "marker": marker_recall(markers, gold_plays, ctx.config),
            "landing": {"landed": landed, "missed": missed, "taps": landed + missed,
                        "miss_rate": _ratio(missed, landed + missed)},
            "replays": replays, "warnings": warnings}


def _run_name(candidate: Path) -> str:
    p = Path(candidate)
    name = p.parent.parent.name if p.parent.name == "weights" else p.stem
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-") or "candidate"
    return name[:60]


def _unique(root: Path, name: str) -> Path:
    path, k = root / name, 2
    while path.exists():
        path, k = root / f"{name}_{k}", k + 1
    return path


def read_m0(out_root: Path) -> dict[str, Any] | None:
    """<gate root>/m0.json (the weights and SHA-256 of tonight's model, M0), or None."""
    try:
        doc = _read_json(Path(out_root) / M0_FILE)
    except (OSError, ValueError):
        return None
    return dict(doc) if isinstance(doc, Mapping) and doc.get("schema") == M0_SCHEMA and doc.get("sha256") else None


def resolve_m0(m0: Any, out_root: Path, baseline: Path) -> tuple[Path, str, str]:
    """(M0's weights, SHA-256, where it came from): --m0, else m0.json (the first baseline run's
    weights), else the baseline itself. GateError when the weights are gone or not M0's SHA-256."""
    if m0 is not None:
        p = Path(m0)
        if not p.is_file():
            raise GateError(f"--m0 weights {p} do not exist", EXIT_INPUT)
        return p, file_sha256(p), "--m0"
    doc = read_m0(out_root)
    if doc is None:
        return baseline, file_sha256(baseline), "the baseline (no m0.json yet: run baseline first)"
    p = Path(str(doc.get("weights") or ""))
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        raise GateError(f"M0's weights {p} (m0.json) are gone: restore them, or pass --m0 WEIGHTS", EXIT_CHECK)
    sha = file_sha256(p)
    if sha != str(doc["sha256"]).lower():
        raise GateError(f"{p} is not M0 any more (SHA-256 {sha[:12]}, m0.json says {str(doc['sha256'])[:12]}): "
                        "restore it, or pass --m0 WEIGHTS", EXIT_CHECK)
    return p, sha, M0_FILE


def run_gate(candidate: Any = None, *, baseline: Any = None, mode: str = MODE_GATE, gold: Sequence[str] | None = None,
             labels_root: Any = None, store: Any = None, out_root: Any = None, name: str | None = None,
             config: GateConfig | None = None, options: ReplayOptions | None = None, katacr_data: Any = KATACR_DATA,
             katacr: bool = True, reuse: bool = True, runner: Callable[[list[str]], int] | None = None,
             yolo_factory: Callable[[str], Any] | None = None, table: Any = None, registry: Any = None,
             base_labels: Any = VA.BASE_LABELS, pin: Any = None, m0: Any = None, clock: Callable[[], float] = time.time,
             log: Callable[[str], Any] = print) -> GateResult:
    """The gate (mode "gate": candidate against baseline) or the baseline report (mode "baseline":
    the baseline alone, verdict BASELINE): writes <out_root>/<name>/gate.json and report.md. The
    baseline defaults to the pin (its SHA-256 checked against the pin's); the regression checks
    compare with it. play.id_gain is measured over M0 (resolve_m0: --m0, else the weights the
    first baseline run recorded in <out_root>/m0.json, else the baseline). GateError on a refused
    gold set or bad input."""
    if mode not in (MODE_GATE, MODE_BASELINE):
        raise GateError(f"mode must be {MODE_GATE} or {MODE_BASELINE}", EXIT_INPUT)
    cfg = config or GateConfig()
    opts = options or ReplayOptions()
    if opts.unknown_cost is not None and (not opts.unknown_plays or int(opts.unknown_cost) < 0):
        raise GateError("--unknown-cost needs the UNKNOWN plays on and must be >= 0", EXIT_INPUT)
    if mode == MODE_GATE:
        if candidate is None:
            raise GateError("the gate needs --candidate WEIGHTS", EXIT_INPUT)
        candidate = Path(candidate)
        if not candidate.is_file():
            raise GateError(f"candidate weights {candidate} do not exist", EXIT_INPUT)
    pinned = None
    if baseline is None:
        try:
            baseline, pinned = resolve_weights(None, pin)
        except DetectorError as exc:
            raise GateError(f"no baseline: {exc}; pass the baseline weights", EXIT_INPUT) from None
    baseline = Path(baseline)
    if not baseline.is_file():
        raise GateError(f"baseline weights {baseline} do not exist", EXIT_INPUT)
    if pinned and file_sha256(baseline) != pinned:
        raise GateError(f"the pinned baseline {baseline} is not the pinned file (SHA-256 {file_sha256(baseline)[:12]}, "
                        f"the pin says {pinned[:12]}): restore it, pin again, or pass --baseline", EXIT_CHECK)
    if name is not None and not VA.SID_RE.match(str(name)):
        raise GateError(f"--name {name!r}: use letters, digits, '_', '-' and '.'", EXIT_INPUT)
    table = VA.load_table() if table is None else table
    keys = Keys(table)
    warnings: list[str] = []
    if registry is None:
        try:
            registry = VA.Registry.load()
        except VA.RegistryError as exc:
            registry = None
            warnings.append(f"the label registry does not load ({exc}): new labels are the gold labels tonight's set lacks")
    if gold:
        sids, skipped = list(dict.fromkeys(str(s) for s in gold)), []
    else:
        sids, skipped = gold_ready(labels_root, store)
        if not sids:
            why = "; ".join(f"{s}: {w}" for s, w in skipped) or "no gold session is ingested"
            raise GateError(f"no gold session is ready to score ({why})")
    golds = [load_gold(s, labels_root=labels_root, store=store, table=table, keys=keys) for s in sids]
    for g in golds:
        warnings += g.warnings
    sets = label_sets(golds, table, keys, registry, base_labels)
    _spawn, ids, ids_from = VR.engine_card_ids(opts.spawn_table, table)
    geometry = None
    try:
        from scripts.vision.geometry import load_geometry
        geometry = load_geometry(opts.geometry)
    except Exception as exc:                      # markers then have no position (report only)
        warnings.append(f"geometry does not load ({type(exc).__name__}: {exc}): stopwatch markers matched by time only")
    out_root = Path(out_root) if out_root is not None else GATE_ROOT
    m0_path, m0_sha, m0_from = (None, None, None) if mode == MODE_BASELINE else resolve_m0(m0, out_root, baseline)
    ctx = _Context(table, keys, {i: n for n, i in ids.items()}, ids_from, geometry, sets["new_labels"],
                   sets["new_classes"], sets["new_cards"], sets["own_labels"], cfg, opts, out_root / CACHE_DIR,
                   bool(reuse), runner, log)
    stamp = datetime.fromtimestamp(clock()).strftime("%Y%m%d_%H%M%S")
    log(f"gold: {', '.join(sids)}" + (f" (skipped {', '.join(s for s, _ in skipped)})" if skipped else ""))
    base_scores = evaluate_model(baseline, golds, ctx)
    warnings += base_scores.pop("warnings")
    cand_scores = None
    checks: list[Check] = []
    base_checks: list[Check] = []
    katacr_note = None
    m0_scores = None
    if mode == MODE_GATE:
        if file_sha256(candidate) == base_scores["sha256"]:
            warnings.append("the candidate is the baseline (the same SHA-256): the gain is 0")
        if m0_sha == base_scores["sha256"]:
            m0_scores = base_scores
        else:
            log(f"M0 ({m0_from}): {_shown(m0_path)}")
            m0_scores = evaluate_model(m0_path, golds, ctx)
            warnings += m0_scores.pop("warnings")
        cand_scores = evaluate_model(candidate, golds, ctx)
        warnings += cand_scores.pop("warnings")
        regression = None
        if not katacr:
            katacr_note = "--no-katacr-val: KataCR val not run"
        elif not Path(katacr_data).is_file():
            katacr_note = f"no KataCR dataset yaml at {_shown(katacr_data)} (prepare_yolo_dataset writes tmp/yolo/cr.yaml)"
        else:
            try:
                regression = katacr_regression(candidate, baseline, Path(katacr_data), device=opts.device,
                                               yolo_factory=yolo_factory, cache=ctx.cache_root, log=log)
            except Exception as exc:              # no ultralytics, no CUDA, a broken dataset
                katacr_note = f"KataCR val failed ({type(exc).__name__}: {exc})"
        cand_scores["katacr"], cand_scores["katacr_note"] = regression, katacr_note
        cand_scores["leak"] = leak_check(candidate, golds, labels_root=labels_root, store=store)
        checks = build_checks(cand_scores, base_scores, cfg, m0=m0_scores)
        v = verdict(checks)
    else:
        if katacr and Path(katacr_data).is_file():
            try:
                base_scores["katacr_val"] = katacr_val(baseline, Path(katacr_data), device=opts.device,
                                                       yolo_factory=yolo_factory, cache=ctx.cache_root, log=log)
            except Exception as exc:
                warnings.append(f"KataCR val of the baseline failed ({type(exc).__name__}: {exc})")
        base_checks = build_checks(base_scores, None, cfg)
        v = BASELINE
        if read_m0(out_root) is None:          # the first baseline run records tonight's model as M0
            out_root.mkdir(parents=True, exist_ok=True)
            VA.write_json(out_root / M0_FILE, {"schema": M0_SCHEMA, "weights": Path(baseline).resolve().as_posix(),
                                               "sha256": base_scores["sha256"], "created": _now(clock),
                                               "tool": "tools/vision_gap_eval.py baseline"})
            log(f"M0 recorded: {_shown(baseline)} ({base_scores['sha256'][:12]}) -> {out_root / M0_FILE}")
    default = f"M0_{stamp}" if mode == MODE_BASELINE else f"{_run_name(candidate)}_{stamp}"
    out_dir = _unique(out_root, name or default)
    doc = {
        "schema": GATE_SCHEMA, "created": _now(clock), "mode": mode, "verdict": v, "reason": verdict_reason(checks, v),
        "candidate": None if cand_scores is None else {
            "weights": _shown(candidate), "sha256": cand_scores["sha256"],
            "label_set_digest": (cand_scores.get("leak") or {}).get("label_set_digest"),
            "mix": (cand_scores.get("leak") or {}).get("mix")},
        "baseline": {"weights": _shown(baseline), "sha256": base_scores["sha256"]},
        "m0": None if m0_scores is None else {"weights": _shown(m0_path), "sha256": m0_sha, "from": m0_from,
                                               "play_id": (m0_scores.get("plays") or {}).get("play_id")},
        "gold": [g.summary() for g in golds], "skipped": [{"sid": s, "why": w} for s, w in skipped],
        "config": asdict(cfg),
        "replay": {**opts.flags(), "device": opts.device, "card_ids": ids_from,
                   "unknown_cost_used": next(((r.get("unknown_plays") or {}).get("cost") for r in base_scores["replays"]
                                              if isinstance(r.get("unknown_plays"), Mapping)), None)},
        "labels": {k: sorted(vals) for k, vals in sets.items() if k != "base_labels"},
        "checks": [asdict(c) for c in checks], "baseline_checks": [asdict(c) for c in base_checks],
        "candidate_scores": cand_scores, "baseline_scores": base_scores,
        "reads": base_scores["reads"],
        "baseline_plays": {c: r["read_as"] for c, r in (base_scores["plays"].get("per_card") or {}).items() if r["gold"]},
        "warnings": [VA.scrub_paths(w) for w in warnings], "tool": "tools/vision_gap_eval.py"}
    report = render_report(doc)
    out_dir.mkdir(parents=True, exist_ok=True)
    VA.write_json(out_dir / GATE_FILE, doc)
    (out_dir / REPORT_FILE).write_text(report, encoding="utf-8")
    return GateResult(out_dir.name, out_dir, doc, v, checks, report)


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------

def _pct(v: Any) -> str:
    return "-" if v is None else f"{100.0 * float(v):.1f}%"


def _ci_text(ci: Any) -> str:
    return "" if not ci else f" [{_pct(ci[0])}-{_pct(ci[1])}]"


def _fmt(v: Any, unit: str) -> str:
    if v is None:
        return "-"
    v = float(v)
    if unit == UNIT_RATE:
        return f"{v:.2f}/min"
    if unit == UNIT_MAP:
        return f"{v:.4f}"
    if unit == UNIT_COUNT:
        return str(int(round(v)))
    if unit == UNIT_GAIN:
        return f"{100.0 * v:+.1f} pts"
    return _pct(v)


def _table(head: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    return out


def _top(counter: Mapping[str, int], k: int = 4) -> str:
    total = sum(counter.values())
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return ", ".join(f"{name} {n} ({_pct(n / total) if total else '-'})" for name, n in items) or "-"


def _checks_table(checks: Sequence[Mapping[str, Any]]) -> list[str]:
    rows = []
    for c in checks:
        unit = c.get("unit", UNIT_SHARE)
        thr = "-" if c.get("threshold") is None else f"{c.get('op')} {_fmt(c['threshold'], unit)}"
        rows.append([f"`{c['id']}`", _fmt(c.get("value"), unit) + (_ci_text(c.get("ci")) if unit == UNIT_SHARE else ""),
                     thr, f"{c.get('n')} ({c.get('min_n')})", c.get("status", "").upper() if c.get("status") == ST_FAIL
                     else c.get("status"), _fmt(c.get("baseline"), UNIT_SHARE if unit == UNIT_GAIN else unit),
                     "yes" if c.get("required") else "", c.get("note", "")])
    return _table(["check", "value [95% CI]", "threshold", "n (min)", "status", "baseline", "required", "note"], rows)


def _score_sections(s: Mapping[str, Any], b: Mapping[str, Any] | None, who: str) -> list[str]:
    out: list[str] = []
    det = s.get("detection") or {}
    bdet = (b or {}).get("detection") or {}
    blabels = bdet.get("labels") or {}
    labels = det.get("labels") or {}
    out += [f"## Detection per gold label ({who}; card-level: the class, form ignored)", ""]
    rows = []
    for lab, r in sorted(labels.items(), key=lambda kv: (not kv[1].get("new"), kv[0])):
        c, br = r.get("card") or {}, (blabels.get(lab) or {}).get("card") or {}
        rows.append([lab + (" (new)" if r.get("new") else ""), r.get("class"), r.get("tracks"), r.get("boxes"),
                     _pct(c.get("recall")) + _ci_text(c.get("recall_ci")),
                     _pct(c.get("precision")) + _ci_text(c.get("precision_ci")),
                     _pct(br.get("recall")) if b else "-", _pct(br.get("precision")) if b else "-"])
    out += _table(["gold label", "class", "tracks", "boxes", "recall [CI]", "precision [CI]", "baseline recall",
                   "baseline precision"], rows) + [""]
    out += ["Form-aware (the detector key: an evolution read as its base form is a miss):", ""]
    rows = []
    for lab, r in sorted(labels.items(), key=lambda kv: (not kv[1].get("new"), kv[0])):
        f, bf = r.get("form") or {}, (blabels.get(lab) or {}).get("form") or {}
        rows.append([lab, r.get("form_label"), _pct(f.get("recall")) + _ci_text(f.get("recall_ci")),
                     _pct(f.get("precision")) + _ci_text(f.get("precision_ci")),
                     _pct(bf.get("recall")) if b else "-", _pct(bf.get("precision")) if b else "-"])
    out += _table(["gold label", "key", "recall [CI]", "precision [CI]", "baseline recall", "baseline precision"], rows)
    side = det.get("side") or {}
    out += ["", f"Side accuracy {_pct(side.get('accuracy'))}{_ci_text(side.get('ci'))} over {side.get('n', 0)} matched "
            f"pairs. {det.get('images', 0)} gold images scored, {det.get('unreplayed', 0)} without a replayed frame; "
            f"{(det.get('ignored_fp') or {}).get('card', 0)} predictions inside ignore regions left out.", ""]
    birth = s.get("birth") or {}
    out += ["## Birth window", "",
            f"{birth.get('hits', 0)} of {birth.get('tracks', 0)} enemy tracks: {_pct(birth.get('recall'))}"
            f"{_ci_text(birth.get('ci'))} have their class as the top box on >= {birth.get('need')} of their first "
            f"{birth.get('images')} images; those images spanned a median {birth.get('span_s')} s. "
            f"{birth.get('short', 0)} tracks with fewer images were left out.", ""]
    out += _table(["gold label", "tracks", "birth recall"],
                  [[lab, r["tracks"], _pct(r["recall"])] for lab, r in (birth.get("per_label") or {}).items()]) + [""]
    plays = s.get("plays") or {}
    out += ["## Plays per card", ""]
    rows = []
    for card, r in sorted((plays.get("per_card") or {}).items(), key=lambda kv: (-kv[1]["gold"], kv[0])):
        rows.append([card + (" (new)" if r.get("new") else ""), r["gold"], _pct(r["recall"]) + _ci_text(r["recall_ci"]),
                     r["pred"], _pct(r["precision"]) + _ci_text(r["precision_ci"]), _top(r.get("read_as") or {})])
    out += _table(["card", "gold plays", "recall [CI]", "plays read", "card precision [CI]", "gold plays read as"], rows)
    if plays.get("forms"):
        out += ["", "Evolution and hero forms:", ""]
        out += _table(["card.form", "gold plays", "card accuracy", "form accuracy (target 80%)"],
                      [[k, r["gold"], _pct(r["card_accuracy"]) + _ci_text(r.get("card_ci")), _pct(r["form_accuracy"])]
                       for k, r in sorted(plays["forms"].items())])
    el = plays.get("elixir") or {}
    unk = plays.get("unknown") or {}
    out += ["", "## Pooled plays, elixir and the safety net", "",
            f"- play-ID accuracy {_pct(plays.get('play_id'))}{_ci_text(plays.get('play_id_ci'))}: {plays.get('right', 0)} "
            f"right of {plays.get('gold', 0)} gold plays ({plays.get('wrong', 0)} wrong card, {plays.get('missed', 0)} "
            f"missed) over {plays.get('minutes', 0):g} battle minutes",
            f"- phantom plays {plays.get('phantoms', 0)}: {_fmt(plays.get('phantom_per_min'), UNIT_RATE)}",
            f"- elixir error {el.get('error', 0)}: {_fmt(el.get('per_min'), UNIT_RATE)} (missed {el.get('missed', 0)}, "
            f"phantom {el.get('phantom', 0)}, wrong card {el.get('wrong_card', 0)}; {el.get('no_cost', 0)} plays without "
            "a cost counted as 0)",
            f"- opponent elixir counted from the plays: gold {_fmt(el.get('gold_spent_per_min'), UNIT_RATE)}, read "
            f"{_fmt(el.get('pred_spent_per_min'), UNIT_RATE)}",
            f"- UNKNOWN plays {unk.get('pred', 0)} ({_pct(unk.get('share'))} of {plays.get('pred', 0)} read): "
            f"{_pct(unk.get('matched_share'))}{_ci_text(unk.get('ci'))} match a real play", ""]
    out += ["## Regression", ""]
    kat = s.get("katacr")
    if kat:
        out += [f"KataCR val mAP50: candidate {kat['candidate']['map50']:.4f}, baseline {kat['baseline']['map50']:.4f} "
                f"(drop {kat['drop']:+.4f}). Classes that fell most:", ""]
        out += _table(["class", "AP50 drop", "candidate", "baseline"],
                      [[w["class"], f"{w['drop']:+.3f}", f"{w['candidate']:.3f}", f"{w['baseline']:.3f}"]
                       for w in kat.get("worst") or ()]) + [""]
    elif s.get("katacr_note"):
        out += [f"KataCR val: {s['katacr_note']}.", ""]
    elif s.get("katacr_val"):
        out += [f"KataCR val mAP50 of this model: {s['katacr_val']['map50']:.4f}.", ""]
    own, bown = det.get("own") or {}, bdet.get("own") or {}
    out += _table(["our label", "boxes", "recall", "baseline recall"],
                  [[lab, r["n"], _pct(r["recall"]), _pct((bown.get(lab) or {}).get("recall")) if b else "-"]
                   for lab, r in own.items()])
    on = det.get("newlabel_on_own") or {}
    land = s.get("landing") or {}
    out += ["", f"New labels on our own boxes: {on.get('hits', 0)} of {on.get('n', 0)} ({_pct(on.get('share'))}"
            f"{_ci_text(on.get('ci'))}){': ' + _top(on.get('by_label') or {}) if on.get('hits') else ''}.",
            f"Own landings: {land.get('missed', 0)} missed of {land.get('taps', 0)} taps ({_pct(land.get('miss_rate'))}).", ""]
    marker = s.get("marker") or {}
    if marker:
        out += ["## Stopwatch marker (N03, report only)", "",
                f"{marker.get('hits', 0)} of {marker.get('plays', 0)} gold unit plays had an enemy stopwatch within the "
                f"play window: {_pct(marker.get('recall'))}{_ci_text(marker.get('ci'))} ({marker.get('markers', 0)} "
                "stopwatch sightings).", ""]
    return out


def render_report(doc: Mapping[str, Any]) -> str:
    """report.md from a gate.json document (module doc)."""
    mode, v = doc.get("mode"), doc.get("verdict")
    cand, base = doc.get("candidate_scores"), doc.get("baseline_scores") or {}
    lines: list[str] = []
    if mode == MODE_BASELINE:
        lines += [f"# Gap baseline: {(doc.get('baseline') or {}).get('weights')}", "",
                  "The baseline's numbers on the gold set. No verdict: the gate (`gate --candidate`) compares a "
                  "candidate with these.", ""]
    else:
        lines += [f"# Gap gate: {v}", "", f"{doc.get('reason')}.", ""]
    lines += ["## Models and the gold set", ""]
    rows = []
    if doc.get("candidate"):
        c = doc["candidate"]
        rows.append(["candidate", c.get("weights"), str(c.get("sha256"))[:12], c.get("mix") or "-"])
    b = doc.get("baseline") or {}
    rows.append(["baseline", b.get("weights"), str(b.get("sha256"))[:12], "-"])
    m0 = doc.get("m0") or {}
    if m0 and m0.get("sha256") != b.get("sha256"):
        rows.append(["M0 (play.id_gain)", m0.get("weights"), str(m0.get("sha256"))[:12], m0.get("from") or "-"])
    lines += _table(["model", "weights", "sha256", "mix"], rows) + [""]
    lines += _table(["gold session", "capture", "images", "bad frames", "tracks", "plays", "minutes", "plays list", "emit"],
                    [[g["sid"], g.get("capture"), g["images"], g.get("bad_frames", 0), g["tracks"], g["plays"],
                      g["minutes"], g.get("plays_list"), g.get("emit_created")] for g in doc.get("gold") or ()])
    for s in doc.get("skipped") or ():
        lines.append(f"- skipped {s['sid']}: {s['why']}")
    r = doc.get("replay") or {}
    lines += ["", f"Replays: vision_replay --ticks time, UNKNOWN plays {'on' if r.get('unknown_plays') else 'off'}"
              f"{'' if r.get('unknown_cost') is None else ' at ' + str(r['unknown_cost']) + ' elixir'}; card ids: "
              f"{r.get('card_ids')}. New labels: {', '.join((doc.get('labels') or {}).get('new_labels') or ()) or '-'}; "
              f"new cards: {', '.join((doc.get('labels') or {}).get('new_cards') or ()) or '-'}.", ""]
    if mode == MODE_BASELINE:
        lines += ["## Where the baseline stands against the thresholds (decides nothing)", ""]
        lines += _checks_table(doc.get("baseline_checks") or ()) + [""]
        lines += _score_sections(base, None, "baseline")
    else:
        lines += ["## Checks", ""] + _checks_table(doc.get("checks") or ()) + [""]
        lines += _score_sections(cand or {}, base, "candidate")
    lines += ["## What the baseline reads each gold label as", ""]
    lines += _table(["gold label", "boxes", "baseline reads (top 4)"],
                    [[lab, sum(c.values()), _top(c)] for lab, c in (doc.get("reads") or {}).items()]) + [""]
    lines += ["## Baseline plays per gold card", ""]
    lines += _table(["gold card", "gold plays", "baseline read them as"],
                    [[card, sum(c.values()), _top(c, 6)] for card, c in (doc.get("baseline_plays") or {}).items()]) + [""]
    if doc.get("warnings"):
        lines += ["## Warnings", ""] + [f"- {w}" for w in doc["warnings"]] + [""]
    lines += ["## Next", ""]
    if v == PASS:
        weights = (doc.get("candidate") or {}).get("weights")
        lines += ["PASS. Before pinning (plan 6.5): run one dry-run friendly (runbook 17) against a clanmate deck with "
                  "post-2024 cards and compare the OppPlay log with what happened; that recording becomes field gold. "
                  "Then pin:", "", f"    & $BB -m scripts.vision.detector pin {weights}", ""]
    elif v == FAIL:
        lines += ["FAIL: do not pin. Look at the failing checks above; label more of what fails (or fix the mix), "
                  "fine-tune again and gate again.", ""]
    elif v == INSUFFICIENT:
        lines += ["INSUFFICIENT: no failure was measured, but a required check lacks evidence. Record and label more "
                  "gold (status lists what to record next), run the KataCR val, or train from a `mix`; then gate "
                  "again.", ""]
    else:
        lines += ["Baseline written. Fine-tune on a mix (N13), then: `& $BB tools\\vision_gap_eval.py gate --candidate "
                  "<best.pt>`.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gold", nargs="+", default=None, metavar="SID",
                   help="gold sessions to score (default: every gold session ready: fresh gold emit, confirmed plays)")
    p.add_argument("--name", default=None, help="the gate folder's name (default <run>_<time>, or M0_<time>)")
    p.add_argument("--out", default=None, help=f"the gate root (default {GATE_ROOT})")
    p.add_argument("--device", default=None, help="detector device (default: CUDA GPU 0, else cpu)")
    p.add_argument("--no-unknown-plays", action="store_true",
                   help="replay without the UNKNOWN safety net (default: on, plan 6.2)")
    p.add_argument("--unknown-cost", type=int, default=None, metavar="ELIXIR",
                   help="the elixir each UNKNOWN play is charged (default: the deriver's, 4)")
    p.add_argument("--spawn-table", default=None, help="card ids from this spawn table (vision_replay's default)")
    p.add_argument("--geometry", default=None, help="an ArenaGeometry JSON (default: data/vision's)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--katacr-data", default=str(KATACR_DATA), help="tonight's KataCR dataset yaml (tmp/yolo/cr.yaml)")
    g.add_argument("--no-katacr-val", action="store_true", help="skip the KataCR val (reg.katacr is then insufficient)")
    p.add_argument("--no-reuse", action="store_true", help="replay again although the cache holds the run")
    p.add_argument("--boot", type=int, default=GateConfig.boot, help="bootstrap resamples for the label intervals")
    p.add_argument("--seed", type=int, default=GateConfig.seed, help="bootstrap seed")
    p.add_argument("--labels-root", default=None, help=f"owner labels root (default {VA.LABELS_ROOT})")
    p.add_argument("--store", default=None, help=f"session store root (default {VA.STORE_ROOT})")
    p.add_argument("--classes", default=None, help="classes.json (default data/vision/classes.json)")
    p.add_argument("--registry", default=None, help=f"the label registry (default {VA.REGISTRY_PATH})")
    p.add_argument("--base", default=str(VA.BASE_LABELS), help="tonight's label set (tmp/yolo/detector_labels.json)")
    p.add_argument("--pin", default=None, help="the detector pin file (default: the machine's)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tools/vision_gap_eval.py",
                                 description="The gate before a fine-tuned detector is pinned (N14).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline", help="the baseline's numbers on the gold set (verdict BASELINE)")
    b.add_argument("--weights", default=None, help="the baseline weights (default: the pin)")
    _add_common(b)
    g = sub.add_parser("gate", help="a candidate against the baseline: PASS, FAIL or INSUFFICIENT; never pins")
    g.add_argument("--candidate", required=True, help="the candidate weights (tmp/yolo/runs/<name>/weights/best.pt)")
    g.add_argument("--baseline", default=None, help="the baseline weights (default: the pin)")
    g.add_argument("--m0", default=None, help="tonight's model, which play.id_gain is measured over (default: the "
                                              "weights the first baseline run recorded in <out>/m0.json)")
    _add_common(g)
    r = sub.add_parser("report", help="print report.md again from a gate.json")
    r.add_argument("path", help="a gate folder or its gate.json")
    return ap


def _cmd_report(path: str) -> int:
    p = Path(path)
    if p.is_dir():
        p = p / GATE_FILE
    try:
        doc = _read_json(p)
    except (OSError, ValueError) as exc:
        print(f"cannot read {p}: {exc}")
        return EXIT_INPUT
    if not isinstance(doc, Mapping) or doc.get("schema") != GATE_SCHEMA:
        print(f"{p} is not a {GATE_SCHEMA} document")
        return EXIT_INPUT
    print(render_report(doc))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "report":
        return _cmd_report(args.path)
    if args.unknown_cost is not None and (args.no_unknown_plays or args.unknown_cost < 0):
        print("--unknown-cost needs the UNKNOWN plays on (no --no-unknown-plays) and must be >= 0")
        return EXIT_INPUT
    try:
        config = GateConfig(boot=args.boot, seed=args.seed)
    except ValueError as exc:
        print(f"settings: {exc}")
        return EXIT_INPUT
    options = ReplayOptions(unknown_plays=not args.no_unknown_plays, unknown_cost=args.unknown_cost, device=args.device,
                            spawn_table=args.spawn_table, geometry=args.geometry)
    try:
        table = VA.load_table(args.classes)
        registry = VA.Registry.load(args.registry) if args.registry else None
    except (OSError, ValueError) as exc:
        print(f"cannot load the class table or the registry: {exc}")
        return EXIT_INPUT
    mode = MODE_GATE if args.cmd == "gate" else MODE_BASELINE
    try:
        res = run_gate(getattr(args, "candidate", None),
                       baseline=args.baseline if mode == MODE_GATE else args.weights, mode=mode, gold=args.gold,
                       labels_root=args.labels_root, store=args.store, out_root=args.out, name=args.name,
                       config=config, options=options, katacr_data=args.katacr_data, katacr=not args.no_katacr_val,
                       reuse=not args.no_reuse, table=table, registry=registry, base_labels=args.base, pin=args.pin,
                       m0=getattr(args, "m0", None))
    except GateError as exc:
        print(f"refused: {exc}")
        return exc.code
    d = res.doc
    print(f"{'baseline' if mode == MODE_BASELINE else 'gate'} {res.verdict}: {d['reason']}")
    for g in d["gold"]:
        print(f"  gold {g['sid']}: {g['images']} images, {g['tracks']} tracks, {g['plays']} plays, {g['minutes']:g} min")
    for c in res.checks:
        if c.status in (ST_FAIL, ST_INSUFFICIENT) and (c.required or c.status == ST_FAIL):
            print(f"  {c.status:12s} {c.id}: {_fmt(c.value, c.unit)} (threshold {c.op} {_fmt(c.threshold, c.unit)}, "
                  f"n {c.n} of {c.min_n}){' - ' + c.note if c.note else ''}")
    for w in d["warnings"]:
        print(f"  warning: {w}")
    print(f"gate.json and report.md -> {res.out_dir}")
    if res.verdict == PASS:
        print("PASS: run one dry-run friendly first (runbook 17), then pin with: "
              f"& $BB -m scripts.vision.detector pin {d['candidate']['weights']}")
    return EXIT_OK if res.verdict in (PASS, BASELINE) else EXIT_CHECK


if __name__ == "__main__":
    sys.exit(main())
