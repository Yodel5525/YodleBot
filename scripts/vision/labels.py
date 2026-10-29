"""Detector labels: the vision class list (E16) crossed with the side (V09).

The detector and the policy share one vocabulary, data/vision/classes.json
(tools/vision_classes.py). This module turns it into the label set a YOLO model
is trained on and reads its output back:

* LABELS. Every KataCR dataset class is mapped through the classes.json
  crosswalk (vision_classes.ClassTable.katacr):
    - unit and tower classes become this table's detector key, the canonical
      unit name of (class, form): "hog-rider" -> "HogRider",
      "skeleton-evolution" -> "Skeleton_EV1", "ram-rider" -> "Ram". Evolved
      forms stay labels of their own (vision-01);
    - spells, objects and UI keep their KataCR name ("fireball", "dirt",
      "bar"), which ClassTable.from_detector already reads (spell -> its card)
      and scripts/vision/hpbars.py expects for HP bars;
    - names the crosswalk does not know are KEPT under their own name and
      reported as unmapped, never dropped;
    - the 46 padding classes (pad_0..pad_44, pad_belong) are left out.
  `extra` labels (a class name, a form's unit name such as "MusketeerHero", a
  spell card) append classes KataCR never had, for frames labelled on the PC.

* SIDE. KataCR labels carry a side column (column 6: 0 = the bottom player,
  1 = the top). On the live screen the bot is the bottom (blue) player, so
  side 0 is ALLY and 1 is ENEMY. With a joint side (the default) YOLO class
  index = label index * 2 + side and its name is "<label>@ally" /
  "<label>@enemy" (with_side / split_side). A model whose names carry no side
  (the author's 201-class KataCR model) is read with side unknown.

* LabelConverter rewrites dataset label lines (scripts/prepare_yolo_dataset.py):
  KataCR class id + side column -> YOLO index, with a report of unmapped names,
  padding boxes and boxes whose side had to be guessed from the board half.

resolve_label() reads ANY detector name back into a vision_classes.Label
(type unit / tower / spell / object / ui / pad / unknown); the detector uses it
on the weights' own class names, so a model trained on another label set still
maps through the same crosswalk.

Imports only the standard library; tools/vision_classes.py is loaded by path
(as the module "vision_classes", the way the tools and their tests load it).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VISION_CLASSES_PY = ROOT / "tools" / "vision_classes.py"
SCHEMA = "hastycr.vision.detector_labels/1"

ALLY_SIDE, ENEMY_SIDE = 0, 1                # KataCR side column: 0 bottom (us), 1 top
SIDE_NAMES = ("ally", "enemy")
SIDE_SEP = "@"

TYPE_UNIT, TYPE_TOWER, TYPE_SPELL = "unit", "tower", "spell"
TYPE_OBJECT, TYPE_UI, TYPE_PAD, TYPE_UNKNOWN = "object", "ui", "pad", "unknown"
KNOWN_TYPES = frozenset({TYPE_UNIT, TYPE_TOWER, TYPE_SPELL, TYPE_OBJECT, TYPE_UI, TYPE_PAD})

MODE_VISION, MODE_KATACR = "vision", "katacr"
#: Label status in a LabelSet: crosswalked to a detector key, kept under its KataCR
#: name (spell / object / UI), unknown to the crosswalk, or appended by `extra`.
STATUS_MAPPED, STATUS_KEPT, STATUS_UNMAPPED, STATUS_EXTRA = "mapped", "kept", "unmapped", "extra"


# ---------------------------------------------------------------------------
# side suffixes
# ---------------------------------------------------------------------------

def with_side(label: str, side: int | None) -> str:
    """"HogRider", 1 -> "HogRider@enemy"; side None -> the label unchanged."""
    if side is None:
        return str(label)
    if side not in (ALLY_SIDE, ENEMY_SIDE):
        raise ValueError(f"side must be {ALLY_SIDE} (ally) or {ENEMY_SIDE} (enemy), got {side!r}")
    return f"{label}{SIDE_SEP}{SIDE_NAMES[side]}"


def split_side(name: str) -> tuple[str, int | None]:
    """"HogRider@enemy" -> ("HogRider", 1); a name without a side suffix -> (name, None)."""
    name = str(name)
    base, sep, suffix = name.rpartition(SIDE_SEP)
    if sep and base and suffix in SIDE_NAMES:
        return base, SIDE_NAMES.index(suffix)
    return name, None


def parse_side(text: Any) -> int | None:
    """A dataset side column ("0", "1", "1.0") -> 0 / 1; anything else -> None."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if value in (0.0, 1.0):
        return int(value)
    return None


# ---------------------------------------------------------------------------
# the class table (tools/vision_classes.py)
# ---------------------------------------------------------------------------

def vision_classes_module():
    """tools/vision_classes.py as the module "vision_classes" (loaded once)."""
    mod = sys.modules.get("vision_classes")
    if mod is None:
        spec = importlib.util.spec_from_file_location("vision_classes", VISION_CLASSES_PY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["vision_classes"] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop("vision_classes", None)
            raise
    return mod


def load_class_table(source: Any = None):
    """A vision_classes.ClassTable from None (data/vision/classes.json), a path, a
    classes.json document (dict) or a table (returned as is)."""
    vc = vision_classes_module()
    if source is None or isinstance(source, (str, os.PathLike)):
        return vc.ClassTable.load(source)
    if isinstance(source, Mapping):
        return vc.ClassTable(source)
    if hasattr(source, "from_detector") and hasattr(source, "katacr"):
        return source
    raise TypeError(f"not a class table, classes.json path or document: {type(source).__name__}")


def _label(type_: str, class_id: int | None = None, form: int = 0, card: str | None = None,
           name: str | None = None):
    return vision_classes_module().Label(type_, class_id, form, card, name)


def resolve_label(table: Any, name: str):
    """Any detector name (side suffix removed) -> vision_classes.Label. Names the table
    cannot place (and the table's "unmapped" KataCR entries) come back as type
    "unknown" with `name` set, so callers can report them."""
    lab = table.from_detector(str(name))
    if lab.type not in KNOWN_TYPES:
        return _label(TYPE_UNKNOWN, name=str(name))
    return lab


def katacr_label(table: Any, name: str, mode: str = MODE_VISION) -> tuple[str | None, str]:
    """(label, status) a KataCR class trains as; label None for a padding class.

    mode "vision": units and towers -> the detector key of their (class, form);
    spells, objects and UI keep the KataCR name; names the crosswalk does not know
    are kept and marked unmapped. mode "katacr": every name kept as it is (pads too,
    so the class ids stay the dataset's)."""
    if mode == MODE_KATACR:
        return str(name), STATUS_KEPT
    if mode != MODE_VISION:
        raise ValueError(f"mode must be {MODE_VISION!r} or {MODE_KATACR!r}, not {mode!r}")
    entry = table.katacr.get(str(name))
    if entry is None:
        return str(name), STATUS_UNMAPPED
    typ = entry.get("type")
    if typ == TYPE_PAD:
        return None, TYPE_PAD
    if typ in (TYPE_UNIT, TYPE_TOWER):
        key = table.detector_key(entry.get("class"), int(entry.get("form", 0)))
        return (str(name), STATUS_KEPT) if key is None else (key, STATUS_MAPPED)
    if typ in (TYPE_SPELL, TYPE_OBJECT, TYPE_UI):
        return str(name), STATUS_KEPT
    return str(name), STATUS_UNMAPPED


# ---------------------------------------------------------------------------
# the label set
# ---------------------------------------------------------------------------

@dataclass
class LabelSet:
    """Detector labels in index order, and how they become YOLO classes.

    joint_side: YOLO class = index * 2 + side, named "<label>@ally|enemy";
    otherwise YOLO class = index (the side kept apart, as sides.json).
    status[label] is mapped / kept / unmapped / extra; sources[label] lists the
    KataCR names trained as it.
    """
    labels: list[str] = field(default_factory=list)
    joint_side: bool = True
    status: dict[str, str] = field(default_factory=dict)
    sources: dict[str, list[str]] = field(default_factory=dict)
    mode: str = MODE_VISION
    classes_digest: str | None = None

    def __post_init__(self) -> None:
        self.labels = [str(x) for x in self.labels]
        self._index = {}
        for i, label in enumerate(self.labels):
            if label in self._index:
                raise ValueError(f"duplicate detector label {label!r}")
            if SIDE_SEP in label:
                raise ValueError(f"detector label {label!r} must not contain {SIDE_SEP!r}")
            self._index[label] = i
            self.status.setdefault(label, STATUS_KEPT)
            self.sources.setdefault(label, [])

    def __len__(self) -> int:
        return len(self.labels)

    def __contains__(self, label: object) -> bool:
        return label in self._index

    def add(self, label: str, status: str = STATUS_EXTRA) -> int:
        """Append `label` (if new) and return its index."""
        label = str(label)
        if label in self._index:
            return self._index[label]
        if SIDE_SEP in label:
            raise ValueError(f"detector label {label!r} must not contain {SIDE_SEP!r}")
        self._index[label] = len(self.labels)
        self.labels.append(label)
        self.status[label] = status
        self.sources.setdefault(label, [])
        return self._index[label]

    def index(self, label: str) -> int:
        return self._index[str(label)]

    def yolo_index(self, label: str, side: int | None = None) -> int:
        i = self.index(label)
        if not self.joint_side:
            return i
        if side not in (ALLY_SIDE, ENEMY_SIDE):
            raise ValueError(f"a joint-side label set needs side 0 or 1 for {label!r}, got {side!r}")
        return 2 * i + side

    def yolo_names(self) -> list[str]:
        """The class names of the YOLO dataset (cr.yaml), in class index order."""
        if not self.joint_side:
            return list(self.labels)
        return [with_side(label, side) for label in self.labels for side in (ALLY_SIDE, ENEMY_SIDE)]

    def decode(self, yolo_index: int) -> tuple[str, int | None]:
        """YOLO class index -> (label, side or None)."""
        k = int(yolo_index)
        if self.joint_side:
            return self.labels[k // 2], k % 2
        return self.labels[k], None

    @property
    def unmapped(self) -> list[str]:
        return [x for x in self.labels if self.status.get(x) == STATUS_UNMAPPED]

    @property
    def digest(self) -> str:
        """SHA-1 of the YOLO names, one per line: equal means the same class indices."""
        return hashlib.sha1("\n".join(self.yolo_names()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "mode": self.mode, "joint_side": self.joint_side,
                "side_names": list(SIDE_NAMES), "classes_digest": self.classes_digest,
                "digest": self.digest, "labels": list(self.labels),
                "status": {x: self.status[x] for x in self.labels},
                "sources": {x: list(self.sources[x]) for x in self.labels if self.sources[x]},
                "yolo_names": self.yolo_names()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LabelSet":
        return cls(list(d.get("labels") or []), bool(d.get("joint_side", True)),
                   dict(d.get("status") or {}), {k: list(v) for k, v in (d.get("sources") or {}).items()},
                   str(d.get("mode", MODE_VISION)), d.get("classes_digest"))


def build_label_set(table: Any, katacr_names: Sequence[str], *, mode: str = MODE_VISION,
                    joint_side: bool = True, extra: Iterable[str] = ()) -> tuple[LabelSet, dict[str, str | None]]:
    """(LabelSet, KataCR name -> label or None) for a dataset's class names in id order.

    Labels come in the order of their first KataCR name. `extra` labels are
    appended in vision mode: a class or unit name becomes its detector key
    (MusketeerHero), a spell card or KataCR name is kept; a name the table cannot
    place raises ValueError."""
    labels = LabelSet([], joint_side=joint_side, mode=mode,
                      classes_digest=getattr(table, "digest", None))
    mapping: dict[str, str | None] = {}
    for name in katacr_names:
        label, status = katacr_label(table, name, mode)
        mapping[str(name)] = label
        if label is None:
            continue
        labels.add(label, status)
        labels.sources[label].append(str(name))
    for name in extra:
        lab = resolve_label(table, name)
        if lab.type in (TYPE_UNKNOWN, TYPE_PAD):
            raise ValueError(f"extra label {name!r} is not a class, unit form, spell or KataCR name "
                             "of the class table")
        label = str(name)
        if mode == MODE_VISION and lab.type in (TYPE_UNIT, TYPE_TOWER):
            label = table.detector_key(lab.class_id, lab.form) or label
        labels.add(label, STATUS_EXTRA)
    return labels, mapping


# ---------------------------------------------------------------------------
# dataset label conversion
# ---------------------------------------------------------------------------

@dataclass
class ConversionReport:
    boxes: int = 0                                  # box lines read (valid coordinates)
    written: int = 0                                # box lines written
    pad_boxes: Counter = field(default_factory=Counter)        # padding class -> boxes dropped
    unmapped: Counter = field(default_factory=Counter)         # dataset name -> boxes kept unmapped
    unknown_ids: Counter = field(default_factory=Counter)      # class id not in the yaml -> boxes
    side_guessed: int = 0                           # side column missing / not 0-1: board half used
    sides: Counter = field(default_factory=Counter)             # side -> boxes written

    def to_dict(self) -> dict[str, Any]:
        return {"boxes": self.boxes, "written": self.written, "pad_boxes": dict(self.pad_boxes),
                "unmapped": dict(self.unmapped), "unknown_ids": {str(k): v for k, v in self.unknown_ids.items()},
                "side_guessed": self.side_guessed,
                "sides": {SIDE_NAMES[k] if k in (0, 1) else str(k): v for k, v in sorted(self.sides.items())}}


class LabelConverter:
    """Dataset class id + side column -> YOLO class index under a LabelSet.

    id_to_name: the dataset yaml's id -> name; mapping: build_label_set's name ->
    label. A box of a padding class is dropped (counted); an id the yaml lacks, or
    a name the mapping lacks, is kept under its own name ("class_<id>") and counted
    as unmapped. With a joint side, a side column that is missing or not 0 / 1 is
    replaced by the board half of the box (y centre < 0.5: the top player, enemy)
    and counted."""

    def __init__(self, labels: LabelSet, id_to_name: Mapping[int, str],
                 mapping: Mapping[str, str | None]):
        self.labels = labels
        self.id_to_name = {int(k): str(v) for k, v in id_to_name.items()}
        self.mapping = dict(mapping)
        self.report = ConversionReport()

    def convert(self, class_id: int, side_text: Any = None, y_center: float = 0.5) -> int | None:
        self.report.boxes += 1
        name = self.id_to_name.get(int(class_id))
        if name is None:
            name = f"class_{int(class_id)}"
            self.report.unknown_ids[int(class_id)] += 1
        if name in self.mapping:
            label = self.mapping[name]
        else:
            label = name
            self.mapping[name] = name
        if label is None:
            self.report.pad_boxes[name] += 1
            return None
        if label not in self.labels:
            self.labels.add(label, STATUS_UNMAPPED)
            self.labels.sources[label].append(name)
        if self.labels.status.get(label) == STATUS_UNMAPPED:
            self.report.unmapped[name] += 1
        side = parse_side(side_text)
        if self.labels.joint_side and side is None:
            side = ENEMY_SIDE if float(y_center) < 0.5 else ALLY_SIDE
            self.report.side_guessed += 1
        if side is not None:
            self.report.sides[side] += 1
        self.report.written += 1
        return self.labels.yolo_index(label, side)
