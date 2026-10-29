"""Train a Clash Royale unit detector on the published dataset.

Why one model with 201 classes rather than one per card: YOLO is multi-class by
construction, and the confusions are the point - a per-class model never sees a
Wizard, so it cannot learn that a Musketeer is not one. The existing detector
mislabelling spell particles as Bats is what that failure looks like.

One refinement worth knowing, from the author of this dataset: with ~150 classes
he ended up using **two** detectors split by object *scale*, because a Skeleton
is perhaps 20px and a King Tower 200px and a single anchor set struggles across
that range. This trains a single model first, to get a measured baseline before
adding that complexity.

Frames are 568x896, which is small, so this is an overnight job rather than a
week: roughly 3 hours for the `n` model and 6 for `s` at 640px on a 4070 Ti
SUPER. mAP starts being meaningful around epoch 10, so a run that is going
nowhere is obvious quickly.

V09 additions:

* The device comes from torch.cuda: the first GPU when CUDA is available.
  Without CUDA the run is REFUSED unless `--device cpu` is given (a CPU torch
  in the GPU venv would otherwise train for days without saying so).
* `--batch` defaults to a size for a 12 GB RTX 4070 SUPER at 640 px with AMP
  (n 64, s 32, m 16, l 8, x 4; scaled by (640 / imgsz)^2 and by the GPU's
  memory / 12 GB, in multiples of 8). `--batch -1` hands it to ultralytics'
  AutoBatch. `--epochs` / `--imgsz` / `--patience` / `--workers` as before;
  keep workers modest on a 16 GB-RAM PC, and close MuMu and the browser.
* When training ends, the SHA-256 of best.pt (and last.pt) is printed and
  written next to it (best.pt.sha256), with detector_manifest.json in the run
  folder, and best.pt is pinned for scripts/vision/detector.py
  (<data root>/vision/detector_pin.json; `--no-pin` keeps the old pin).

    python scripts/prepare_yolo_dataset.py           class list x side (V09)
    python scripts/train_detector.py                 yolov8s, 640 px, batch 32, 150 epochs
    python scripts/train_detector.py --epochs 60     a quicker first model
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))           # scripts.vision.detector, also when run as a file
DATA = ROOT / "tmp" / "yolo" / "cr.yaml"
RUNS = ROOT / "tmp" / "yolo" / "runs"

#: Batch per model size that fits a 12 GB GPU at 640 px with AMP (yolov8 / yolo11).
BATCH_12GB_640 = {"n": 64, "s": 32, "m": 16, "l": 8, "x": 4}
REFERENCE_VRAM_GB = 12.0
_SIZE = re.compile(r"yolo(?:v\d+|\d+)([nsmlx])", re.I)


def model_size(model: str) -> Optional[str]:
    """yolov8s.pt -> "s", yolo11m.pt -> "m"; None for other names (best.pt)."""
    m = _SIZE.search(Path(str(model)).name)
    return m.group(1).lower() if m else None


def default_batch(model: str, imgsz: int = 640, vram_gb: Optional[float] = None) -> int:
    """A batch size for `model` at `imgsz` on a GPU with `vram_gb` (default 12 GB).
    The memory is rounded to whole GB first: a 12 GB card reports ~11.99."""
    base = BATCH_12GB_640[model_size(model) or "s"]
    vram = REFERENCE_VRAM_GB if not vram_gb else float(round(vram_gb))
    scaled = base * (640.0 / float(imgsz)) ** 2 * (vram / REFERENCE_VRAM_GB)
    if scaled >= 8:
        return int(scaled // 8 * 8)
    return max(1, int(scaled))


def _import_torch():
    """torch, or None when it does not import (tests replace this)."""
    try:
        import torch
        return torch
    except Exception:
        return None


def _import_yolo():
    """ultralytics.YOLO (tests replace this)."""
    from ultralytics import YOLO
    return YOLO


def _versions(torch_module: Any) -> dict:
    """torch and ultralytics versions (ultralytics only if training imported it)."""
    ultralytics = sys.modules.get("ultralytics")
    return {"torch": getattr(torch_module, "__version__", None),
            "ultralytics": getattr(ultralytics, "__version__", None)}


def write_sha256_sidecar(weights: Path, digest: str) -> Path:
    """<weights>.sha256 in `sha256sum` format."""
    side = weights.with_name(weights.name + ".sha256")
    side.write_text(f"{digest} *{weights.name}\n", encoding="utf-8")
    return side


def _label_set_digest(data_yaml: Path) -> Optional[str]:
    """The label-set digest prepare_yolo_dataset.py wrote next to the data yaml."""
    path = Path(data_yaml).parent / "detector_labels.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("digest")
    except (OSError, ValueError):
        return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the CR unit detector")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="starting weights; yolov8n.pt is ~2x faster")
    parser.add_argument("--data", type=Path, default=DATA,
                        help="dataset yaml from scripts/prepare_yolo_dataset.py")
    parser.add_argument("--project", type=Path, default=RUNS, help="where runs are written")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None,
                        help="default: sized for a 12 GB RTX 4070 SUPER (s: 32 at 640 px); "
                             "-1 = ultralytics AutoBatch")
    parser.add_argument("--device", default="auto",
                        help="auto (first CUDA GPU; refuses to run without one), 0, cpu, ...")
    parser.add_argument("--name", default="cr_detector")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pin", action=argparse.BooleanOptionalAction, default=True,
                        help="pin best.pt for scripts/vision/detector.py when training ends")
    parser.add_argument("--pin-file", type=Path, default=None,
                        help="the pin file (default <data root>/vision/detector_pin.json)")
    args = parser.parse_args(argv)

    from scripts.vision import detector as vision_detector

    torch_module = _import_torch()
    device, device_desc, vram_gb = vision_detector.pick_device(args.device, torch_module=torch_module)
    if device == "cpu" and str(args.device).lower() != "cpu":
        print("CUDA is not available to this Python "
              f"(torch {getattr(torch_module, '__version__', 'missing')}). Training on the CPU takes "
              "days: install the CUDA torch in .venvs/buildabot (requirements/gpu.txt), or pass "
              "--device cpu to train on the CPU anyway.")
        return 2
    batch = args.batch if args.batch is not None else default_batch(args.model, args.imgsz, vram_gb)

    if not args.data.exists():
        print(f"missing {args.data}; run scripts/prepare_yolo_dataset.py first")
        return 1
    print(f"device {device_desc}; model {args.model}, imgsz {args.imgsz}, batch {batch}, "
          f"epochs {args.epochs}, workers {args.workers}; data {args.data}")

    YOLO = _import_yolo()
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        patience=args.patience,
        # Left on: these frames are all the same camera and scale, so the
        # geometric augmentations that help general detection matter less here
        # than colour, which varies with arena and with the red/blue side tint.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, shear=0.0, perspective=0.0,
        # A unit near our tower is never mirrored into the enemy half in this
        # game, but left-right is genuinely symmetric, so keep flips horizontal
        # only and never vertical.
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, close_mosaic=10,
        # Images stay on disk: 5551 frames decoded would not fit comfortably in
        # RAM alongside the emulator and the model.
        cache=False,
        plots=True,
        val=True,
    )
    run_dir = Path(args.project) / args.name
    print(f"\nweights and curves under {run_dir}")

    trainer_best = getattr(getattr(model, "trainer", None), "best", None)
    best = Path(trainer_best) if trainer_best and Path(trainer_best).is_file() else run_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"no best weights at {best}; nothing to hash or pin")
        return 1
    digest = vision_detector.file_sha256(best)
    write_sha256_sidecar(best, digest)
    last = best.with_name("last.pt")
    last_digest = vision_detector.file_sha256(last) if last.is_file() else None
    if last_digest:
        write_sha256_sidecar(last, last_digest)
    manifest = {
        "weights": best.resolve().as_posix(), "sha256": digest,
        "last": last.resolve().as_posix() if last_digest else None, "last_sha256": last_digest,
        "data": Path(args.data).resolve().as_posix(), "label_set_digest": _label_set_digest(args.data),
        "model": args.model, "epochs": args.epochs, "imgsz": args.imgsz, "batch": batch,
        "device": device_desc, "versions": _versions(torch_module),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "detector_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"best weights {best}\n  sha256 {digest}")
    if args.pin:
        pin = vision_detector.write_pin(best, sha256=digest, path=args.pin_file,
                                        meta={k: manifest[k] for k in ("data", "label_set_digest", "model",
                                                                       "epochs", "imgsz", "batch")})
        print(f"  pinned -> {pin} (scripts/vision/detector.py loads it)")
    else:
        print(f"  not pinned; pin it with: python -m scripts.vision.detector pin {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
