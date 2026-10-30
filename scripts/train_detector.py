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

N06, fine-tunes (docs/NEW_CARDS_PLAN.md 2.3, 8 and risk 9): `--optimizer`,
`--lr0`, `--cls-pw`, `--freeze` and `--warmup-bias-lr` reach ultralytics'
train() as optimizer, lr0, cls_pw, freeze and warmup_bias_lr; left out, they
are not passed and ultralytics' own defaults apply, as before. With
optimizer=auto ultralytics picks the optimizer and rate from the iteration
count (AdamW at about 0.01 / (4 + classes) up to 10,000 iterations, MuSGD at
0.01 above), ignores lr0 and starts the bias warmup at 0, so a short fine-tune
needs both --optimizer and --lr0. An explicit optimizer keeps ultralytics'
warmup_bias_lr of 0.1 (engine/trainer.py build_optimizer resets it only under
auto): every bias, the class-head priors included, would start the warmup at
0.1, 100x an AdamW lr0 of 0.001. So with --optimizer (not auto) this passes
warmup_bias_lr 0.0, as auto does, unless --warmup-bias-lr says otherwise.
--resume continues the checkpoint's own settings (ultralytics' check_resume
drops a new optimizer, lr0, cls_pw or warmup_bias_lr), so those flags are
refused with it; --freeze is kept.

A WARNING is printed, and training still runs, when a run would replace
weights or move the pin unasked: a start from trained weights (best.pt, not a
stock yolov8s.pt / yolo11m.pt / *.yaml) with the default --name; a run folder
(--project/--name, reused: exist_ok) that already holds best.pt or last.pt,
which then are replaced (a rebase from yolov8s.pt over tonight's M0, a
fine-tune re-run under its own name), with a line when the pinned weights are
among them; and --pin with trained starting weights (fine-tunes pass --no-pin
until the gate passes, docs/NEW_CARDS_PLAN.md 6.5).

    python scripts/prepare_yolo_dataset.py           class list x side (V09)
    python scripts/train_detector.py                 yolov8s, 640 px, batch 32, 150 epochs
    python scripts/train_detector.py --model tmp/yolo/runs/cr_detector/weights/best.pt
        --name ft_20261101 --optimizer AdamW --lr0 0.001 --epochs 40 --no-pin
                                                     a fine-tune, on one line (0.001 is
                                                     ultralytics' AdamW default, not a tuned rate)
    python scripts/train_detector.py ... --freeze "0,1,22.cv2"
                                                     quote a comma list in PowerShell (or
                                                     space-separate it: --freeze 0 1 22.cv2)

Fewer epochs is not a quicker equivalent: below 115 epochs (10,000 iterations of
ceil(5551 / 64) at nbs 64) optimizer=auto switches from MuSGD at 0.01 to AdamW at
about 3.2e-5 for 310 classes (docs/NEW_CARDS_PLAN.md 8); pass --optimizer and --lr0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))           # scripts.vision.detector, also when run as a file
DATA = ROOT / "tmp" / "yolo" / "cr.yaml"
RUNS = ROOT / "tmp" / "yolo" / "runs"

DEFAULT_NAME = "cr_detector"
#: ultralytics 8.4's optimizers (cfg/default.yaml `optimizer`); --optimizer matches them in any case.
OPTIMIZERS = ("auto", "SGD", "MuSGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSprop")

#: Batch per model size that fits a 12 GB GPU at 640 px with AMP (yolov8 / yolo11).
BATCH_12GB_640 = {"n": 64, "s": 32, "m": 16, "l": 8, "x": 4}
REFERENCE_VRAM_GB = 12.0
_SIZE = re.compile(r"yolo(?:v\d+|\d+)([nsmlx])", re.I)
_STOCK = re.compile(r"yolo(?:v\d+|\d+)[nsmlx]\.(?:pt|ya?ml)", re.I)
#: The warmup_bias_lr ultralytics' optimizer=auto uses (engine/trainer.py build_optimizer);
#: passed with an explicit --optimizer unless --warmup-bias-lr is given.
AUTO_WARMUP_BIAS_LR = 0.0
#: train() keywords ultralytics' check_resume drops on --resume (it keeps freeze).
RESUME_DROPS = ("optimizer", "lr0", "cls_pw", "warmup_bias_lr")


def model_size(model: str) -> Optional[str]:
    """yolov8s.pt -> "s", yolo11m.pt -> "m"; None for other names (best.pt)."""
    m = _SIZE.search(Path(str(model)).name)
    return m.group(1).lower() if m else None


def is_stock(model: str) -> bool:
    """Stock starting weights or architecture, by file name: yolov8s.pt, yolo11m.pt,
    yolov8s.yaml. Not best.pt, last.pt, or a trained file renamed yolov8s-m0.pt."""
    return _STOCK.fullmatch(Path(str(model)).name) is not None


def default_batch(model: str, imgsz: int = 640, vram_gb: Optional[float] = None) -> int:
    """A batch size for `model` at `imgsz` on a GPU with `vram_gb` (default 12 GB).
    The memory is rounded to whole GB first: a 12 GB card reports ~11.99."""
    base = BATCH_12GB_640[model_size(model) or "s"]
    vram = REFERENCE_VRAM_GB if not vram_gb else float(round(vram_gb))
    scaled = base * (640.0 / float(imgsz)) ** 2 * (vram / REFERENCE_VRAM_GB)
    if scaled >= 8:
        return int(scaled // 8 * 8)
    return max(1, int(scaled))


def optimizer_name(text: str) -> str:
    """--optimizer: one of OPTIMIZERS in any case (adamw -> AdamW)."""
    name = {o.lower(): o for o in OPTIMIZERS}.get(str(text).strip().lower())
    if name is None:
        raise argparse.ArgumentTypeError(f"unknown optimizer {text!r}; use one of {', '.join(OPTIMIZERS)}")
    return name


def _fraction(text: str, *, zero_ok: bool) -> float:
    """A number in [0, 1] ((0, 1] unless zero_ok): ultralytics' range for lr0, cls_pw and
    warmup_bias_lr."""
    try:
        x = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not ((0.0 <= x if zero_ok else 0.0 < x) and x <= 1.0):
        raise argparse.ArgumentTypeError(f"{text} is outside {'[0' if zero_ok else '(0'}, 1]")
    return x


def learning_rate(text: str) -> float:
    """--lr0: in (0, 1]."""
    return _fraction(text, zero_ok=False)


def class_weight_power(text: str) -> float:
    """--cls-pw: in [0, 1] (0 = no class weights, 1 = full inverse frequency)."""
    return _fraction(text, zero_ok=True)


def bias_warmup_rate(text: str) -> float:
    """--warmup-bias-lr: in [0, 1] (a fraction key of ultralytics' cfg, as lr0 and cls_pw)."""
    return _fraction(text, zero_ok=True)


def freeze_spec(text: str) -> int | list:
    """--freeze: N (the first N layers, an int) or a comma list of layer indices and module
    names (0,1,2 or 22.cv2) -> a list, as ultralytics' freeze takes them."""
    raw = str(text).strip()
    if raw.isdigit():
        return int(raw)
    parts = [p.strip() for p in raw.split(",")]
    if not raw or any(not p or p.startswith("-") for p in parts):
        raise argparse.ArgumentTypeError(f"takes N or a comma list of layer indices / names, not {text!r}")
    return [int(p) if p.isdigit() else p for p in parts]


def train_overrides(args: argparse.Namespace) -> dict:
    """The fine-tune settings given on the command line, as train() keywords (none: {}); with
    an explicit --optimizer other than auto, warmup_bias_lr AUTO_WARMUP_BIAS_LR unless
    --warmup-bias-lr is given (module doc)."""
    warmup = getattr(args, "warmup_bias_lr", None)
    if warmup is None and args.optimizer not in (None, "auto"):
        warmup = AUTO_WARMUP_BIAS_LR
    pairs = (("optimizer", args.optimizer), ("lr0", args.lr0), ("cls_pw", args.cls_pw), ("freeze", args.freeze),
             ("warmup_bias_lr", warmup))
    return {k: v for k, v in pairs if v is not None}


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except (OSError, ValueError):
        return False


def pinned_weights(pin_file: Optional[Path] = None) -> Optional[Path]:
    """The weights the pin file names, or None (no pin, or one that does not read)."""
    from scripts.vision import detector as vision_detector
    try:
        doc = vision_detector.read_pin(pin_file)
    except Exception:
        return None
    if not doc or not doc.get("weights"):
        return None
    p = Path(str(doc["weights"]))
    return p if p.is_absolute() else ROOT / p          # stored relative to the repo (detector.write_pin)


def fine_tune_warnings(args: argparse.Namespace,
                       read_pin: Callable[[Optional[Path]], Optional[Path]] = pinned_weights) -> List[str]:
    """WARNING lines for a run that would replace weights or move the pin unasked (module doc):
    a start from trained weights (not is_stock: best.pt, last.pt, ...) with the default --name;
    a run folder (--project/--name, which ultralytics reuses: exist_ok=True) already holding
    weights/best.pt or last.pt, which are replaced; and --pin with trained starting weights.
    --resume continues a run in its own folder on purpose and is never warned about. The pin
    (read_pin(args.pin_file) -> its weights) is read only when weights would be replaced."""
    if args.resume:
        return []
    run_dir = Path(args.project) / args.name
    trained = not is_stock(args.model)
    default_name = trained and args.name == DEFAULT_NAME
    held = [p.name for p in (run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt") if p.is_file()]
    replacing = default_name or bool(held)
    lines: List[str] = []
    if replacing:
        if default_name:
            lines.append(f"WARNING: fine-tuning from {args.model} with the default --name {args.name}: "
                         f"ultralytics reuses {run_dir} (exist_ok), so the weights/best.pt and last.pt "
                         "there are replaced.")
        else:
            lines.append(f"WARNING: {run_dir} already holds weights/{' and '.join(held)}: ultralytics "
                         "reuses the folder (exist_ok), so they are replaced.")
        if trained and _inside(Path(args.model), run_dir):
            lines.append("  The starting weights are in that folder: they are overwritten while training.")
        pinned = read_pin(args.pin_file)
        if pinned is not None and _inside(Path(pinned), run_dir):
            lines.append(f"  The pinned detector {pinned} is in that folder: once overwritten, its SHA-256 "
                         "no longer matches the pin and the live detector refuses to load"
                         + (" until the pin moves." if args.pin else " (--no-pin: the pin stays)."))
    if args.pin and (replacing or trained):
        if not lines:
            lines.append(f"WARNING: training from {args.model}, not stock weights, with the pin on.")
        lines.append("  When training ends the pin moves to the new best.pt (pass --no-pin until it passes the gate).")
    if replacing:
        lines.append("  Pass a new --name (e.g. --name ft_YYYYMMDD) to keep the old run.")
    return lines


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
    parser.add_argument("--name", default=DEFAULT_NAME,
                        help="the run folder under --project; ultralytics reuses it when it exists, so a "
                             "fine-tune passes a new name")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pin", action=argparse.BooleanOptionalAction, default=True,
                        help="pin best.pt for scripts/vision/detector.py when training ends")
    parser.add_argument("--pin-file", type=Path, default=None,
                        help="the pin file (default <data root>/vision/detector_pin.json)")
    parser.add_argument("--optimizer", type=optimizer_name, default=None,
                        help="ultralytics optimizer: " + ", ".join(OPTIMIZERS) + " (default: not passed, "
                             "ultralytics' auto, which ignores --lr0)")
    parser.add_argument("--lr0", type=learning_rate, default=None,
                        help="initial learning rate in (0, 1] (default: not passed, ultralytics' 0.01)")
    parser.add_argument("--cls-pw", type=class_weight_power, default=None,
                        help="class weight power in [0, 1]: 0 none, 1 full inverse frequency "
                             "(default: not passed, ultralytics' 0.0)")
    parser.add_argument("--freeze", nargs="+", default=None, metavar="N|LIST",
                        help="freeze the first N layers, or a list of layer indices / module names such as "
                             "22.cv2, comma- or space-separated (quote a comma list in PowerShell: "
                             "--freeze \"0,1,22.cv2\"; default: not passed, nothing frozen)")
    parser.add_argument("--warmup-bias-lr", type=bias_warmup_rate, default=None,
                        help="bias learning rate the warmup starts from, in [0, 1] (default: not passed; "
                             f"with --optimizer other than auto, {AUTO_WARMUP_BIAS_LR:g} as auto uses, since "
                             "ultralytics' 0.1 would start every bias at 0.1)")
    args = parser.parse_args(argv)
    if args.freeze is not None:
        try:                                # PowerShell splits an unquoted 0,1,22.cv2 into 3 arguments
            args.freeze = freeze_spec(",".join(args.freeze))
        except argparse.ArgumentTypeError as exc:
            parser.error(f"argument --freeze: {exc}")
    if args.resume:
        dropped = [f"--{k.replace('_', '-')}" for k in RESUME_DROPS if getattr(args, k, None) is not None]
        if dropped:
            parser.error(f"--resume continues the checkpoint's own settings: ultralytics ignores "
                         f"{', '.join(dropped)} then (start a new run to change them; --freeze is kept)")
    overrides = train_overrides(args)

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
    if overrides:
        print("train settings: " + ", ".join(f"{k} {v}" for k, v in overrides.items()))
    if args.lr0 is not None and overrides.get("optimizer", "auto") == "auto":
        print("note: ultralytics ignores --lr0 with optimizer auto; pass --optimizer too (e.g. AdamW)")
    if args.warmup_bias_lr is None and "warmup_bias_lr" in overrides:
        print(f"note: warmup_bias_lr {AUTO_WARMUP_BIAS_LR:g}, as optimizer auto starts it (ultralytics' 0.1 "
              "would start every bias at 0.1); --warmup-bias-lr sets another")
    elif args.warmup_bias_lr is not None and overrides.get("optimizer", "auto") == "auto":
        print("note: ultralytics resets warmup_bias_lr to 0 with optimizer auto; pass --optimizer too")
    for line in fine_tune_warnings(args):
        print(line)

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
        **overrides,
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
    if overrides:
        manifest["train_overrides"] = overrides
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
