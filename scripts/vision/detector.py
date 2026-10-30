"""YOLO unit detector on the arena crop, mapped through the vision class list (V09).

The old wrapper (scripts/brain/vision.py) ran YOLO on the whole 1080x1920
frame, so the card bar produced phantom units, picked the newest weights by
modification time, stripped evolved forms and had no side (vision-01/04/05).
This one:

1. CROPS the frame to the board plus a margin (ArenaGeometry, resolution-free:
   the PC's 747x1328 PrintWindow frames work as well as 1080x1920) and runs
   the model on that crop only, RGB -> BGR as ultralytics expects of arrays
   (crop_rect_for(geometry, config, size) gives the same crop without a model);
2. maps every box back to frame pixels, reads its name through the E16
   crosswalk (scripts/vision/labels.py -> tools/vision_classes.ClassTable):
   class x side names ("Skeleton_EV1@enemy", a model from
   scripts/prepare_yolo_dataset.py) and plain KataCR names ("skeleton-
   evolution", the author's 201-class model, side unknown) both work;
3. merges the ally / enemy boxes of one unit into one Detection whose
   side_prob is P(enemy) = conf_enemy / (conf_enemy + conf_ally); a box seen on
   one side only gets `single_side_prob` (an uncalibrated prior; the tracker
   clips each frame's side evidence anyway, scripts/vision/tracker.py);
4. places it: units and towers at the geometry's ground point (flyers with the
   flyer offset, from the class table's flying flag), spells, objects, UI and
   unknown names at the box centre; DROPS what lies outside the 18x32 board
   (grown by `board_margin_mt`; UI by `ui_margin_mt`);
5. suppresses a unit box that overlaps a stronger unit box of another label
   almost completely (`cross_class_iou`): one sprite, two labels, would
   otherwise be two tracks and a phantom play.

Detection.cls (scripts/vision/types.py) is the vision CLASS name for units and
towers ("Skeleton", form in Detection.form), the spell CARD for spells
("Fireball"), and the detector name for objects, UI ("bar", "bar-level": what
scripts/vision/hpbars.py reads) and unknown names. Names the crosswalk cannot
place are kept, counted in Detector.unresolved and logged once: a class the
bot is blind to is a fault to see, not to drop silently.

Card-level identity (what the legacy --live-compat rows use as the unit id):
DetectorResult.cards(table, deck) gives each detection's creating card, a
spell's own card and a unit's card through ClassTable.card_for (the deck's
card first: a Witch's skeleton is the Witch's); scripts/vision/events.py does
the same per track over time.

Weights are never picked by date. Detector.load takes an explicit path or the
pin file <data root>/vision/detector_pin.json (src/hastycr/machine.py data
root; scripts/train_detector.py writes it), logs the path and its SHA-256, and
refuses a file whose SHA-256 differs from the pinned or expected one. The
device comes from torch.cuda (first GPU, half precision) unless given.
ultralytics and torch are imported lazily, inside Detector.load / pick_device.

    from scripts.vision.detector import Detector
    det = Detector.load()                                  # pinned weights
    res = det.detect(frame.rgb)                            # frame: capture.Frame
    obs = observations_from_detections(res.for_tracker(), geometry, res.frame_size)

Command line (GPU venv):

    python -m scripts.vision.detector info  [--weights W]
    python -m scripts.vision.detector pin   WEIGHTS
    python -m scripts.vision.detector image FRAME.png [--weights W] [--conf 0.35]
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
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Iterable, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from .geometry import ArenaGeometry, HEIGHT_MT, WIDTH_MT, check_frame_size, in_arena  # noqa: E402
from .labels import (ENEMY_SIDE, TYPE_OBJECT, TYPE_PAD, TYPE_SPELL, TYPE_TOWER, TYPE_UI,  # noqa: E402
                     TYPE_UNIT, TYPE_UNKNOWN, load_class_table, resolve_label, split_side)
from .types import FORM_BASE, Detection  # noqa: E402

log = logging.getLogger("hastycr.vision.detector")

ROOT = Path(__file__).resolve().parents[2]
PIN_NAME = "detector_pin.json"
PIN_SCHEMA = "hastycr.vision.detector_pin/1"

#: KataCR HP-bar classes; the same as scripts/vision/hpbars.BAR_CLASSES.
BAR_CLASSES = ("bar", "bar-level")
#: Result groups: Detection types that go to the tracker by default.
TRACKED_TYPES = (TYPE_UNIT, TYPE_SPELL)

_EPS_PX = 1e-6


class DetectorError(RuntimeError):
    """The detector cannot be loaded or run (no weights, wrong SHA-256, predict failed)."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectorConfig:
    """Inference and filtering parameters. Distances in world millitiles.

    crop_margin_mt: (left, top, right, bottom) margins around the 18x32 board for
    the crop the model sees; the top margin keeps the enemy king tower's sprite,
    the bottom one stops above the card bar. crop_norm: (u0, v0, u1, v1) of the
    frame instead (normalised), e.g. to match another training framing.
    """
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.6
    max_det: int = 150
    crop_margin_mt: tuple[float, float, float, float] = (1000.0, 2000.0, 1000.0, 1000.0)
    crop_norm: tuple[float, float, float, float] | None = None
    board_margin_mt: float = 500.0          # units, towers, spells, objects, unknown
    ui_margin_mt: float = 1000.0            # HP bars and other UI boxes (box centre)
    side_merge_iou: float = 0.6             # an ally and an enemy box of one label: one unit
    single_side_prob: float = 0.9           # P(that side) for a box seen on one side only
    cross_class_iou: float | None = 0.85    # unit boxes of other labels overlapping this much
    half: bool | None = None                # fp16 inference; None = on CUDA only

    def __post_init__(self) -> None:
        if not 0.0 <= self.conf <= 1.0 or not 0.0 < self.iou <= 1.0:
            raise ValueError("conf must be in [0, 1] and iou in (0, 1]")
        if not 0.5 <= self.single_side_prob <= 1.0:
            raise ValueError("single_side_prob must be in [0.5, 1]")
        if len(self.crop_margin_mt) != 4:
            raise ValueError("crop_margin_mt is (left, top, right, bottom)")
        if self.crop_norm is not None:
            u0, v0, u1, v1 = self.crop_norm
            if not (0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0):
                raise ValueError(f"crop_norm {self.crop_norm} is not a box inside the frame")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "DetectorConfig":
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown detector settings: {unknown}")
        for key in ("crop_margin_mt", "crop_norm"):
            if d.get(key) is not None:
                d[key] = tuple(float(v) for v in d[key])
        return cls(**d)


# ---------------------------------------------------------------------------
# weights, pin file, device
# ---------------------------------------------------------------------------

def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, hex, read in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _machine_data_root() -> Path:
    """hastycr.machine's data root (HASTYCR_DATA / machine.json), else <repo>/data."""
    try:
        from hastycr import machine
    except ImportError:
        src = ROOT / "src"
        if not (src / "hastycr" / "machine.py").is_file():
            return ROOT / "data"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            from hastycr import machine
        except ImportError:
            return ROOT / "data"
    try:
        return Path(machine.get().data)
    except Exception as exc:                 # a broken machine.json must not hide the detector
        log.warning("machine settings unreadable (%s); using %s", exc, ROOT / "data")
        return ROOT / "data"


def default_pin_path() -> Path:
    """<data root>/vision/detector_pin.json."""
    return _machine_data_root() / "vision" / PIN_NAME


def _stored_path(path: Path) -> str:
    """A path as the pin stores it: relative to the repo when inside it, posix slashes."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _resolve_stored(text: str) -> Path:
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def write_pin(weights: str | os.PathLike, *, sha256: str | None = None,
              path: str | os.PathLike | None = None, meta: Mapping[str, Any] | None = None) -> Path:
    """Pin `weights` (path + SHA-256) for Detector.load(); returns the pin file."""
    weights = Path(weights)
    if not weights.is_file():
        raise DetectorError(f"cannot pin {weights}: no such file")
    digest = (sha256 or file_sha256(weights)).lower()
    doc = {"schema": PIN_SCHEMA, "weights": _stored_path(weights), "sha256": digest,
           "pinned_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if meta:
        doc["meta"] = dict(meta)
    out = Path(path) if path is not None else default_pin_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return out


def read_pin(path: str | os.PathLike | None = None) -> dict[str, Any] | None:
    """The pin file as a dict, None when it does not exist."""
    p = Path(path) if path is not None else default_pin_path()
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectorError(f"cannot read detector pin {p}: {exc}") from exc
    if not isinstance(doc, dict) or not doc.get("weights"):
        raise DetectorError(f"detector pin {p} has no weights")
    return doc


def resolve_weights(weights: str | os.PathLike | None = None,
                    pin: str | os.PathLike | None = None) -> tuple[Path, str | None]:
    """(weights path, pinned SHA-256 or None): the explicit path, else the pin file."""
    if weights is not None:
        p = Path(weights)
        if not p.is_file():
            raise DetectorError(f"detector weights {p} do not exist")
        return p, None
    doc = read_pin(pin)
    if doc is None:
        where = Path(pin) if pin is not None else default_pin_path()
        raise DetectorError(
            f"no detector weights: pass --weights, or train with scripts/train_detector.py "
            f"(it pins best.pt to {where}), or pin a file with "
            f"`python -m scripts.vision.detector pin WEIGHTS`")
    p = _resolve_stored(str(doc["weights"]))
    if not p.is_file():
        raise DetectorError(f"pinned detector weights {p} do not exist (pin: {pin or default_pin_path()})")
    return p, (str(doc["sha256"]).lower() if doc.get("sha256") else None)


def pick_device(requested: str | None = None, *, torch_module: Any = None) -> tuple[str, str, float | None]:
    """(device for ultralytics, description, GPU memory in GB or None).

    requested None / "" / "auto": the first CUDA GPU ("0") when torch.cuda has one,
    else "cpu". Anything else is passed through. torch_module stands in for
    `import torch` (tests); without torch the answer is "cpu". torch is imported only
    when the device is not given (importing it takes seconds)."""
    req = None if requested is None else str(requested).strip()
    if req and req.lower() != "auto":
        return req, f"{req} (requested)", None
    torch = torch_module
    if torch is None:
        try:
            import torch  # noqa: F811
        except Exception:
            torch = None
    try:
        cuda_ok = torch is not None and bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if not cuda_ok:
        return "cpu", "cpu (CUDA not available)", None
    name, vram = "CUDA GPU 0", None
    try:
        name = str(torch.cuda.get_device_name(0))
        vram = float(torch.cuda.get_device_properties(0).total_memory) / 1024 ** 3
    except Exception:
        pass
    desc = f"cuda:0 {name}" + (f" ({vram:.1f} GB)" if vram else "")
    return "0", desc, vram


def _yolo_factory(path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectorError("ultralytics does not import here; run the detector in the GPU venv "
                            "(.venvs/buildabot, requirements/gpu.txt)") from exc
    return YOLO(path)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class DetectorResult:
    """One frame's detections, strongest first.

    detections, labels (vision_classes.Label) and world (x_mt, y_mt of the ground
    point or box centre) are parallel lists. dropped: detections outside the board;
    suppressed: unit boxes removed by cross_class_iou; raw_boxes: boxes the model
    returned. crop: (x0, y0, x1, y1) of the frame the model saw; ms: wall time.
    """
    detections: list[Detection] = field(default_factory=list)
    labels: list[Any] = field(default_factory=list)
    world: list[tuple[float, float]] = field(default_factory=list)
    frame_size: tuple[int, int] = (0, 0)
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)
    dropped: list[Detection] = field(default_factory=list)
    suppressed: int = 0
    raw_boxes: int = 0
    ms: float = 0.0

    def of_type(self, *types: str) -> list[Detection]:
        return [d for d, lab in zip(self.detections, self.labels) if lab.type in types]

    def units(self) -> list[Detection]:
        """Troops and buildings (not crown towers): the tracker's units."""
        return self.of_type(TYPE_UNIT)

    def towers(self) -> list[Detection]:
        return self.of_type(TYPE_TOWER)

    def spells(self) -> list[Detection]:
        return self.of_type(TYPE_SPELL)

    def objects(self) -> list[Detection]:
        return self.of_type(TYPE_OBJECT)

    def ui(self) -> list[Detection]:
        return self.of_type(TYPE_UI, TYPE_PAD)

    def bars(self, bar_classes: Sequence[str] = BAR_CLASSES) -> list[Detection]:
        """HP-bar boxes (scripts/vision/hpbars.py attaches them to units)."""
        return [d for d in self.ui() if d.cls in bar_classes]

    def unknown(self) -> list[Detection]:
        """Boxes whose name the crosswalk cannot place (also in Detector.unresolved)."""
        return self.of_type(TYPE_UNKNOWN)

    def for_tracker(self, *, spells: bool = True, objects: bool = False,
                    unknown: bool = False) -> list[Detection]:
        """What the tracker follows: units, and by default spells (the event deriver
        counts a spell's first sighting as a play)."""
        types = {TYPE_UNIT} | ({TYPE_SPELL} if spells else set()) \
            | ({TYPE_OBJECT} if objects else set()) | ({TYPE_UNKNOWN} if unknown else set())
        return self.of_type(*types)

    def cards(self, table: Any, deck: Iterable[str] | None = None) -> list[str | None]:
        """Card-level identity per detection (the --live-compat unit id): a spell's
        card; a unit's creating card, the deck's own card first
        (ClassTable.card_for); None for towers, UI, objects and unknown names."""
        deck = None if deck is None else list(deck)
        out: list[str | None] = []
        for lab in self.labels:
            if lab.type == TYPE_SPELL:
                out.append(lab.card)
            elif lab.type == TYPE_UNIT and lab.class_id is not None:
                out.append(table.card_for(lab.class_id, deck))
            else:
                out.append(None)
        return out


@dataclass
class _Box:
    base: str            # detector name without its side suffix
    side: int | None     # 0 ally, 1 enemy, None unknown
    conf: float
    box: tuple[float, float, float, float]
    label: Any           # vision_classes.Label
    order: int


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop_rect_for(geometry: ArenaGeometry, config: DetectorConfig | None,
                  size: Sequence[int]) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) frame pixels a detector with `geometry` and `config` (None: the live
    DetectorConfig) shows the model, for a frame of `size`: config.crop_norm of the frame, else
    the 18x32 board grown by config.crop_margin_mt, mapped through the geometry, floored /
    ceiled to whole pixels and clipped to the frame. Detector.crop_rect is this; the harvest
    and label emitters (N05, N11) use it to write labels in the framing the live model sees
    without loading a model. DetectorError when the crop is empty (under 8 px a side)."""
    w, h = check_frame_size(size[0], size[1])
    cfg = config if config is not None else DetectorConfig()
    if cfg.crop_norm is not None:
        u0, v0, u1, v1 = cfg.crop_norm
        xs, ys = np.array([u0 * w, u1 * w]), np.array([v0 * h, v1 * h])
    else:
        left, top, right, bottom = cfg.crop_margin_mt
        wx = np.array([-left, WIDTH_MT + right, -left, WIDTH_MT + right], dtype=np.float64)
        wy = np.array([-top, -top, HEIGHT_MT + bottom, HEIGHT_MT + bottom], dtype=np.float64)
        xs, ys = geometry.world_to_px(wx, wy, (w, h))
    x0 = max(0, int(math.floor(float(np.min(xs)) + _EPS_PX)))
    y0 = max(0, int(math.floor(float(np.min(ys)) + _EPS_PX)))
    x1 = min(w, int(math.ceil(float(np.max(xs)) - _EPS_PX)))
    y1 = min(h, int(math.ceil(float(np.max(ys)) - _EPS_PX)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise DetectorError(f"crop {(x0, y0, x1, y1)} of a {w}x{h} frame is empty; check the geometry")
    return x0, y0, x1, y1


def _array(value: Any, shape_tail: tuple[int, ...] = ()) -> np.ndarray:
    """A torch tensor, numpy array or list -> float64 numpy array."""
    if value is None:
        return np.zeros((0,) + shape_tail)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    arr = np.asarray(value, dtype=np.float64)
    return arr.reshape((-1,) + shape_tail) if arr.size else np.zeros((0,) + shape_tail)


def _model_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None)
    if names is None:
        raise DetectorError("the model has no class names (model.names)")
    if isinstance(names, Mapping):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------

class Detector:
    """A YOLO model on the arena crop, read through the vision class list.

    model: anything with `.names` ({index: name}) and
    `.predict(source=bgr, imgsz=, conf=, iou=, max_det=, device=, half=, verbose=)`
    returning a list whose first item has `.boxes` with `.xyxy`, `.conf` and
    `.cls` (an ultralytics YOLO, or a fake in tests).
    """

    def __init__(self, model: Any, *, geometry: ArenaGeometry | None = None, table: Any = None,
                 config: DetectorConfig | None = None, device: str = "cpu",
                 device_desc: str | None = None, weights: str | os.PathLike | None = None,
                 sha256: str | None = None):
        self.model = model
        self.geometry = geometry if geometry is not None else ArenaGeometry.default()
        self.table = load_class_table(table)
        self.config = config if config is not None else DetectorConfig()
        self.device = str(device)
        self.device_desc = device_desc or self.device
        self.half = (self.device != "cpu") if self.config.half is None else bool(self.config.half)
        self.weights = None if weights is None else Path(weights)
        self.sha256 = sha256
        self.unresolved: Counter = Counter()    # unknown detector name -> boxes seen
        self.frames = 0
        self.last_ms = 0.0
        self._names: dict[int, tuple[str, int | None, Any]] = {}
        for idx, name in sorted(_model_names(model).items()):
            base, side = split_side(name)
            self._names[idx] = (base, side, resolve_label(self.table, base))
        sides = {side for _, side, _ in self._names.values()}
        self.side_mode = "joint" if sides <= {0, 1} and sides else ("none" if sides == {None} else "mixed")
        self.unresolved_names = sorted({base for base, _, lab in self._names.values()
                                        if lab.type == TYPE_UNKNOWN})
        if self.unresolved_names:
            log.warning("detector: %d model class(es) the class list cannot place (kept as unknown): %s",
                        len(self.unresolved_names), ", ".join(self.unresolved_names))

    # -- loading -------------------------------------------------------------
    @classmethod
    def load(cls, weights: str | os.PathLike | None = None, *, pin: str | os.PathLike | None = None,
             expected_sha256: str | None = None, device: str | None = None,
             geometry: ArenaGeometry | None = None, table: Any = None,
             config: DetectorConfig | None = None,
             model_factory: Callable[[str], Any] | None = None,
             torch_module: Any = None) -> "Detector":
        """Load weights (explicit path, else the pin file), check and log their SHA-256,
        choose the device from torch.cuda, and build the detector."""
        path, pinned = resolve_weights(weights, pin)
        digest = file_sha256(path)
        want = (expected_sha256 or pinned or "").lower() or None
        if want is not None and digest != want:
            raise DetectorError(f"detector weights {path} have SHA-256 {digest}, expected {want}; "
                                "re-pin them if they were retrained on purpose")
        dev, desc, _vram = pick_device(device, torch_module=torch_module)
        model = (model_factory or _yolo_factory)(str(path))
        det = cls(model, geometry=geometry, table=table, config=config, device=dev,
                  device_desc=desc, weights=path, sha256=digest)
        log.info("detector weights %s sha256 %s device %s (%d model classes, side %s, half %s)",
                 path, digest, desc, len(det._names), det.side_mode, det.half)
        return det

    def info(self) -> dict[str, Any]:
        cfg = asdict(self.config)
        return {"weights": None if self.weights is None else str(self.weights), "sha256": self.sha256,
                "device": self.device, "device_desc": self.device_desc, "half": self.half,
                "model_classes": len(self._names), "side_mode": self.side_mode,
                "unresolved_names": list(self.unresolved_names),
                "classes_digest": getattr(self.table, "digest", None),
                "geometry": getattr(self.geometry, "name", None), "config": cfg}

    def name_map(self) -> dict[int, dict[str, Any]]:
        """Model class index -> {name, base, side, type, cls, form, card}: how each model
        class is read (for the `info` command and debugging)."""
        out = {}
        for idx, (base, side, lab) in sorted(self._names.items()):
            out[idx] = {"base": base, "side": side, "type": lab.type, "cls": self._cls_name(base, lab),
                        "form": int(lab.form or 0), "card": lab.card}
        return out

    # -- geometry ------------------------------------------------------------
    def crop_rect(self, size: Sequence[int]) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) frame pixels the model sees, for a frame of `size`
        (crop_rect_for of this detector's geometry and config)."""
        return crop_rect_for(self.geometry, self.config, size)

    def _flying(self, lab: Any) -> bool:
        try:
            return bool(self.table.classes[lab.class_id].form(lab.form).flying)
        except (IndexError, KeyError, TypeError, AttributeError):
            return False

    @staticmethod
    def _cls_name(base: str, lab: Any) -> str:
        if lab.type in (TYPE_UNIT, TYPE_TOWER) and lab.name:
            return str(lab.name)
        if lab.type == TYPE_SPELL and lab.card:
            return str(lab.card)
        return base

    # -- inference -----------------------------------------------------------
    def _predict(self, crop_bgr: np.ndarray):
        cfg = self.config
        try:
            out = self.model.predict(source=crop_bgr, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
                                     max_det=cfg.max_det, device=self.device, half=self.half,
                                     verbose=False)
        except Exception as exc:
            raise DetectorError(f"predict failed: {type(exc).__name__}: {exc}") from exc
        res = out[0] if isinstance(out, (list, tuple)) else out
        boxes = getattr(res, "boxes", None) if res is not None else None
        if boxes is None:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
        xyxy = _array(getattr(boxes, "xyxy", None), (4,))
        conf = _array(getattr(boxes, "conf", None))
        cls = _array(getattr(boxes, "cls", None))
        if not (len(xyxy) == len(conf) == len(cls)):
            raise DetectorError(f"model returned {len(xyxy)} boxes, {len(conf)} scores, {len(cls)} classes")
        return xyxy, conf, cls

    def _merge_sides(self, raw: list[_Box]) -> list[tuple[_Box, float]]:
        """(box, side_prob) per unit: an ally and an enemy box of one label that overlap
        by side_merge_iou are one unit, P(enemy) = c_enemy / (c_enemy + c_ally)."""
        cfg = self.config
        order = sorted(raw, key=lambda b: (-b.conf, b.order))
        used = [False] * len(order)
        out: list[tuple[_Box, float]] = []
        for i, b in enumerate(order):
            if used[i]:
                continue
            used[i] = True
            if b.side is None:
                out.append((b, 0.5))
                continue
            partner = None
            for j in range(i + 1, len(order)):
                o = order[j]
                if (not used[j] and o.base == b.base and o.side is not None and o.side != b.side
                        and box_iou(b.box, o.box) >= cfg.side_merge_iou):
                    partner = o
                    used[j] = True
                    break
            if partner is not None:
                c_enemy = b.conf if b.side == ENEMY_SIDE else partner.conf
                total = b.conf + partner.conf
                p_enemy = c_enemy / total if total > 0 else 0.5
            else:
                p = cfg.single_side_prob
                p_enemy = p if b.side == ENEMY_SIDE else 1.0 - p
            out.append((b, float(p_enemy)))
        return out

    def detect(self, rgb: np.ndarray) -> DetectorResult:
        """Detections of one HxWx3 RGB frame (any 9:16 portrait size)."""
        t0 = time.perf_counter()
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] < 3:
            raise DetectorError(f"expected an HxWx3 RGB frame, got shape {img.shape}")
        size = check_frame_size(img.shape[1], img.shape[0])
        x0, y0, x1, y1 = self.crop_rect(size)
        crop = np.ascontiguousarray(img[y0:y1, x0:x1, 2::-1])        # RGB -> BGR
        xyxy, conf, cls = self._predict(crop)

        raw: list[_Box] = []
        for k, ((bx1, by1, bx2, by2), c, ci) in enumerate(zip(xyxy, conf, cls)):
            idx = int(round(float(ci)))
            if idx not in self._names:
                name = f"class_{idx}"
                self._names[idx] = (name, None, resolve_label(self.table, name))
            base, side, lab = self._names[idx]
            if lab.type == TYPE_UNKNOWN:
                if base not in self.unresolved:
                    log.warning("detector: class %r is not in the vision class list; kept as unknown", base)
                self.unresolved[base] += 1
            box = (float(bx1) + x0, float(by1) + y0, float(bx2) + x0, float(by2) + y0)
            raw.append(_Box(base, side, float(c), box, lab, k))

        cfg = self.config
        kept: list[tuple[Detection, Any, tuple[float, float]]] = []
        dropped: list[Detection] = []
        for b, p_enemy in self._merge_sides(raw):
            lab = b.label
            cls_name = self._cls_name(b.base, lab)
            form = int(lab.form or FORM_BASE) if lab.type in (TYPE_UNIT, TYPE_TOWER, TYPE_SPELL) else FORM_BASE
            if lab.type in (TYPE_UNIT, TYPE_TOWER):
                wx, wy = self.geometry.ground_point(b.box, size, cls=cls_name, flying=self._flying(lab))
            else:
                wx, wy = self.geometry.px_to_world((b.box[0] + b.box[2]) / 2.0, (b.box[1] + b.box[3]) / 2.0, size)
            wx, wy = float(wx), float(wy)
            gx, gy = self.geometry.world_to_px(wx, wy, size)
            det = Detection(cls_name, p_enemy, b.conf, b.box, (float(gx), float(gy)), form)
            margin = cfg.ui_margin_mt if lab.type in (TYPE_UI, TYPE_PAD) else cfg.board_margin_mt
            if not in_arena(wx, wy, margin):
                dropped.append(det)
                continue
            kept.append((det, lab, (wx, wy)))

        suppressed = 0
        if cfg.cross_class_iou is not None:
            final: list[tuple[Detection, Any, tuple[float, float]]] = []
            unit_boxes: list[tuple[float, float, float, float]] = []
            for item in kept:                       # already strongest first
                det, lab, _ = item
                if lab.type == TYPE_UNIT:
                    if any(box_iou(det.box, u) >= cfg.cross_class_iou for u in unit_boxes):
                        suppressed += 1
                        continue
                    unit_boxes.append(det.box)
                final.append(item)
            kept = final

        self.frames += 1
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return DetectorResult([d for d, _, _ in kept], [lab for _, lab, _ in kept],
                              [w for _, _, w in kept], size, (x0, y0, x1, y1), dropped,
                              suppressed, len(raw), self.last_ms)

    def warmup(self, size: Sequence[int] = (1080, 1920)) -> float:
        """One prediction on a blank frame (CUDA kernels, fp16 weights); returns ms."""
        w, h = check_frame_size(size[0], size[1])
        t0 = time.perf_counter()
        self.detect(np.zeros((h, w, 3), dtype=np.uint8))
        return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _load_from_args(args) -> Detector:
    geometry = ArenaGeometry.load(args.geometry) if args.geometry else None
    config = DetectorConfig(conf=args.conf, imgsz=args.imgsz) if hasattr(args, "conf") else None
    return Detector.load(args.weights, pin=args.pin_file, device=args.device, geometry=geometry,
                         table=args.classes, config=config)


def _cmd_info(args) -> int:
    det = _load_from_args(args)
    info = det.info()
    print(json.dumps({k: v for k, v in info.items() if k != "config"}, indent=1))
    groups: Counter = Counter(v["type"] for v in det.name_map().values())
    print("model classes by type: " + ", ".join(f"{k} {v}" for k, v in sorted(groups.items())))
    if det.unresolved_names:
        print("NOT in the vision class list: " + ", ".join(det.unresolved_names))
    return 0


def _cmd_pin(args) -> int:
    out = write_pin(args.weights_file, path=args.pin_file)
    doc = read_pin(out)
    print(f"pinned {doc['weights']}\n  sha256 {doc['sha256']}\n  -> {out}")
    return 0


def _cmd_image(args) -> int:
    from .record import load_image
    det = _load_from_args(args)
    rgb = load_image(args.image)
    res = det.detect(rgb)
    print(f"{args.image}: {rgb.shape[1]}x{rgb.shape[0]}, crop {res.crop}, {res.raw_boxes} boxes, "
          f"{len(res.detections)} kept, {len(res.dropped)} off the board, {res.suppressed} suppressed, "
          f"{res.ms:.1f} ms")
    for d, lab, (wx, wy) in zip(res.detections, res.labels, res.world):
        print(f"  {lab.type:7s} {d.cls:22s} form {d.form} conf {d.conf:.2f} P(enemy) {d.side_prob:.2f} "
              f"tile ({wx / 1000:.1f}, {wy / 1000:.1f})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="YOLO unit detector on the arena crop (V09)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, conf: bool = False) -> None:
        p.add_argument("--weights", default=None, help="weights file (default: the pinned one)")
        p.add_argument("--pin-file", default=None, help=f"pin file (default: <data root>/vision/{PIN_NAME})")
        p.add_argument("--device", default=None, help="'0', 'cpu', ... (default: CUDA when available)")
        p.add_argument("--geometry", default=None, help="geometry JSON (default: data/vision default)")
        p.add_argument("--classes", default=None, help="classes.json (default: data/vision/classes.json)")
        if conf:
            p.add_argument("--conf", type=float, default=DetectorConfig.conf)
            p.add_argument("--imgsz", type=int, default=DetectorConfig.imgsz)

    p = sub.add_parser("info", help="load the weights; print SHA-256, device and class mapping")
    common(p)
    p.set_defaults(func=_cmd_info)
    p = sub.add_parser("pin", help="pin a weights file (path + SHA-256) for Detector.load")
    p.add_argument("weights_file")
    p.add_argument("--pin-file", default=None)
    p.set_defaults(func=_cmd_pin)
    p = sub.add_parser("image", help="run on one saved frame and list the detections")
    p.add_argument("image")
    common(p, conf=True)
    p.set_defaults(func=_cmd_image)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return int(args.func(args))
    except DetectorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
