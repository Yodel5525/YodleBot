"""Convert the published Clash Royale dataset into a standard YOLO layout.

Two mismatches have to be fixed, and both are silent failures if missed.

**Label width.** The dataset's label lines carry twelve columns:

    0 0.500000 0.130580 0.187374 0.153061 1 0 0 0 0 0 0
    ^ class    ^x       ^y       ^w       ^h ^side ^six unused

Ultralytics expects exactly five (`class x y w h`). Column six is the **side**
flag - 0 for the bottom player, 1 for the top - which is genuinely useful to the
bot later, since knowing ally from enemy is half of reading the board. It is
recorded in `sides.json` here rather than thrown away. The first training run
kept it out of the class index (folding it in doubles the classes); that output
is still available as `--labels katacr --side drop`. The default now folds it
in, see "Class list and side" below.

**Directory layout.** The dataset keeps each label beside its image
(`00810.jpg` next to `00810.txt`). Ultralytics finds labels by swapping
`/images/` for `/labels/` in the path, so the upstream author patched
ultralytics itself to cope. Restructuring the data instead leaves the library
untouched, which is the cheaper thing to maintain.

Images are hard-linked, not copied, so the 640MB dataset is not duplicated.

**Class list and side (V09).** By default the classes are no longer KataCR's
201 names. Each name goes through the vision class list's crosswalk
(data/vision/classes.json, tools/vision_classes.py; scripts/vision/labels.py):
units and towers become the class list's detector keys, with evolved forms kept
as labels of their own ("skeleton-evolution" -> "Skeleton_EV1"), spells,
objects and UI keep their KataCR names, padding classes are left out, and
names the crosswalk does not know are kept and reported, never dropped. The
side column is now kept in the class index: YOLO class = label * 2 + side,
named "<label>@ally" / "<label>@enemy" (side 0, the bottom player, is the bot's
side on the live screen). A box whose side column is missing gets the board
half and is counted. `detector_labels.json` in the output records the label
set and the conversion report; sides.json is still written.

    python scripts/prepare_yolo_dataset.py                       class list x side
    python scripts/prepare_yolo_dataset.py --source D:/datasets/CR-Detection-Dataset
    python scripts/prepare_yolo_dataset.py --labels katacr --side drop    the old output

The dataset is looked for at --source, else <data root>/CR-Detection-Dataset
(HASTYCR_DATA / machine.json, src/hastycr/machine.py), else
vendor/CR-Detection-Dataset; each may be the dataset root or its images/part2.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))           # scripts.vision.labels, also when run as a file
VENDOR_DATASET = ROOT / "vendor" / "CR-Detection-Dataset"
SOURCE = VENDOR_DATASET / "images" / "part2"
OUT = ROOT / "tmp" / "yolo"
YAML_NAME = "ClashRoyale_detection.yaml"
LABELS_FILE = "detector_labels.json"


def read_class_names(yaml_path: Path) -> Dict[int, str]:
    """Pull `id: name` out of the dataset's yaml without a yaml dependency."""
    names: Dict[int, str] = {}
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            match = re.match(r"\s+(\d+):\s*(.+?)\s*$", line)
            if match:
                names[int(match.group(1))] = match.group(2)
            elif line.strip() and not line.startswith(" "):
                break
    return names


def flat_name(relative: str) -> str:
    """`./WTY_20240305/1/00810.jpg` -> `WTY_20240305_1_00810`.

    Flattened because every episode folder restarts numbering at 00000, so a
    flat directory would silently overwrite thousands of frames.
    """
    stem = relative.lstrip("./").rsplit(".", 1)[0]
    return stem.replace("/", "_").replace("\\", "_")


def _machine_data_root() -> Optional[Path]:
    """hastycr.machine's data root (HASTYCR_DATA / machine.json), None if unreadable."""
    src = ROOT / "src"
    if (src / "hastycr" / "machine.py").is_file() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from hastycr import machine
        return Path(machine.get().data)
    except Exception:
        return None


def resolve_source(source: Optional[Path] = None) -> Path:
    """The dataset folder that holds ClashRoyale_detection.yaml.

    --source if given, else <data root>/CR-Detection-Dataset, else
    vendor/CR-Detection-Dataset; each may be the dataset root or its
    images/part2. When none has the yaml, --source's images/part2 (or the
    vendor default) comes back, so the error names it.
    """
    if source is not None:
        candidates = [Path(source)]
    else:
        data = _machine_data_root()
        candidates = ([data / "CR-Detection-Dataset"] if data is not None else []) + [VENDOR_DATASET]
    for root in candidates:
        for folder in (root, root / "images" / "part2"):
            if (folder / YAML_NAME).is_file():
                return folder
    if source is None:
        return SOURCE
    return Path(source) if Path(source).name == "part2" else Path(source) / "images" / "part2"


def convert_split(name: str, listing: Path, sides: Dict[str, List[int]], *,
                  source: Optional[Path] = None, out: Optional[Path] = None,
                  converter=None) -> Tuple[int, int]:
    """Link one split's images and write its labels.

    converter: a scripts.vision.labels.LabelConverter (class list x side); None
    keeps the dataset's class ids (the original behaviour). sides[stem] lists
    each written box's raw side column (-1 when missing).
    """
    source = SOURCE if source is None else Path(source)
    out = OUT if out is None else Path(out)
    image_dir = out / "images" / name
    label_dir = out / "labels" / name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    frames = boxes = 0
    for line in listing.read_text(encoding="utf-8", errors="replace").splitlines():
        relative = line.strip().split()[0] if line.strip() else ""
        if not relative:
            continue
        source_image = (source / relative.lstrip("./")).resolve()
        source_label = source_image.with_suffix(".txt")
        if not source_image.exists() or not source_label.exists():
            continue

        stem = flat_name(relative)
        target_image = image_dir / f"{stem}{source_image.suffix}"
        if not target_image.exists():
            try:
                os.link(source_image, target_image)      # hard link, no copy
            except OSError:
                shutil.copyfile(source_image, target_image)

        rows, frame_sides = [], []
        for raw in source_label.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = raw.split()
            if len(parts) < 5:
                continue
            cls, x, y, w, h = parts[:5]
            # Guard against a stray out-of-range coordinate: ultralytics rejects
            # the whole label file rather than the offending line.
            try:
                values = [float(v) for v in (x, y, w, h)]
            except ValueError:
                continue
            if not all(0.0 <= v <= 1.0 for v in values):
                continue
            side_text = parts[5] if len(parts) > 5 else None
            if converter is None:
                index = int(cls)
            else:
                try:
                    class_id = int(float(cls))
                except ValueError:
                    continue
                index = converter.convert(class_id, side_text, values[1])
                if index is None:                       # a padding class
                    continue
            rows.append(f"{index} {x} {y} {w} {h}")
            frame_sides.append(int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else -1)
        (label_dir / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        sides[stem] = frame_sides
        frames += 1
        boxes += len(rows)
    return frames, boxes


def _yaml_name(name: str) -> str:
    """A class name as a YAML scalar: plain when safe, else double-quoted."""
    return name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.@-]*", name) else json.dumps(name)


def _print_report(report: dict, labels) -> None:
    unmapped = report.get("unmapped") or {}
    if unmapped:
        print(f"\nkept but NOT in the vision class list ({len(unmapped)} names, add them to "
              f"tools/vision_classes.py's crosswalk):")
        for name, count in sorted(unmapped.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {name:28s} {count:6d} boxes")
    pads = report.get("pad_boxes") or {}
    if pads:
        print(f"padding-class boxes dropped: {sum(pads.values())} ({', '.join(sorted(pads))})")
    if report.get("unknown_ids"):
        print(f"class ids missing from the yaml (kept as class_<id>): {report['unknown_ids']}")
    if report.get("side_guessed"):
        print(f"boxes without a 0/1 side column (board half used): {report['side_guessed']}")
    print(f"sides written: {report.get('sides')}")
    if labels.unmapped:
        print("unmapped labels: " + ", ".join(labels.unmapped))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the YOLO dataset")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source", type=Path, default=None,
                        help="the KataCR dataset (its root or images/part2); default: "
                             "<data root>/CR-Detection-Dataset, else vendor/CR-Detection-Dataset")
    parser.add_argument("--labels", choices=("vision", "katacr"), default="vision",
                        help="vision: names through the class list's crosswalk (default); "
                             "katacr: the dataset's own names and ids")
    parser.add_argument("--side", choices=("joint", "drop"), default="joint",
                        help="joint: class = label * 2 + side (default); drop: class only, "
                             "side in sides.json (the original output)")
    parser.add_argument("--classes", type=Path, default=None,
                        help="the class list (default data/vision/classes.json)")
    parser.add_argument("--extra-label", action="append", default=[], metavar="NAME",
                        help="append a label KataCR lacks (a class, a form such as MusketeerHero, "
                             "a spell card), for frames labelled later")
    args = parser.parse_args(argv)

    source = resolve_source(args.source)
    yaml_path = source / YAML_NAME
    names = read_class_names(yaml_path) if yaml_path.is_file() else {}
    if not names:
        print(f"could not read class names from the dataset yaml {yaml_path}")
        return 1
    highest = max(names)
    dataset_names = [names.get(index, f"class_{index}") for index in range(highest + 1)]

    converter = labels = None
    if not (args.labels == "katacr" and args.side == "drop"):
        from scripts.vision import labels as detector_labels
        table = detector_labels.load_class_table(args.classes)
        try:
            labels, mapping = detector_labels.build_label_set(
                table, dataset_names, mode=args.labels, joint_side=args.side == "joint",
                extra=args.extra_label)
        except ValueError as exc:
            print(f"error: {exc}")
            return 1
        converter = detector_labels.LabelConverter(labels, dict(enumerate(dataset_names)), mapping)
        known = list((getattr(table, "doc", None) or {}).get("katacr_names") or [])
        if known and known != dataset_names:
            print(f"note: the dataset yaml's {len(dataset_names)} names differ from the "
                  f"{len(known)} the class list was built from; names are matched by name")

    print(f"dataset {source}")
    sides: Dict[str, List[int]] = {}
    total = {}
    for split, listing in (("train", source / "train_annotation.txt"),
                           ("val", source / "val_annotation.txt")):
        if not listing.exists():
            print(f"missing {listing}")
            return 1
        frames, boxes = convert_split(split, listing, sides, source=source, out=args.out,
                                      converter=converter)
        total[split] = (frames, boxes)
        print(f"{split:5s} {frames:5d} frames  {boxes:6d} boxes")

    (args.out / "sides.json").write_text(json.dumps(sides), encoding="utf-8")

    yolo_names = dataset_names if labels is None else labels.yolo_names()
    lines = [f"path: {args.out.resolve().as_posix()}",
             "train: images/train", "val: images/val", "names:"]
    lines += [f"  {index}: {_yaml_name(name)}" for index, name in enumerate(yolo_names)]
    (args.out / "cr.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if labels is not None:
        doc = labels.to_dict()
        doc["dataset"] = {"source": source.resolve().as_posix(), "yaml_names": len(dataset_names),
                          "splits": {k: {"frames": f, "boxes": b} for k, (f, b) in total.items()}}
        doc["report"] = converter.report.to_dict()
        (args.out / LABELS_FILE).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"\n{len(labels)} labels x {'2 sides' if labels.joint_side else 'no side'} = "
              f"{len(yolo_names)} classes -> {args.out / 'cr.yaml'}")
        print(f"label set and report -> {args.out / LABELS_FILE}")
        _print_report(doc["report"], labels)
    else:
        print(f"\n{len(names)} named classes, {highest + 1} slots -> {args.out / 'cr.yaml'}")
    print(f"side flags for {len(sides)} frames -> {args.out / 'sides.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
