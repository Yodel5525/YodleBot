"""Record VisionFrames per session, with images on a disk budget; read them back.

A recording root holds one directory per session (one friendly, say):

    <root>/<session>/frames.jsonl     one JSON object per line
    <root>/<session>/images/00000012.jpg

Lines are {"kind": ...}:
    "session"  the header: schema, creation time, image settings
    "frame"    {"i": frame index, "t_wall": time.time(), "image": relative
               path or null, "frame": VisionFrame.to_dict(), "extra": {...}}
    any other  VisionRecorder.log(kind, **data): {"kind", "t", "t_wall", "data"}
               (decisions, taps, battle results)

An image is saved with every `image_every`-th frame. After each image the
recorder prunes the OLDEST images under the whole root until they fit
`budget_mb`, always keeping the newest `keep_last_images`, like
scripts/capture_daemon.py does for its JPEGs. Only files named like
00000012.jpg/.png/.npy inside <session>/images of a directory holding
frames.jsonl are ever deleted; the JSONL logs are never pruned.

VisionLogReader reads a session back (it tolerates a truncated last line from
a crash). Image helpers (save_image, load_image, resize_rgb) import OpenCV or
Pillow lazily; the "npy" format needs neither.
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__" and not __package__:
    # Run as a file, sys.path[0] is scripts/vision, whose types.py would shadow
    # the standard library's `types`: put the repo root there instead and
    # import as part of the package (PEP 366), so the relative imports work.
    _here = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if sys.path and os.path.normcase(os.path.abspath(sys.path[0] or os.curdir)) == _here:
        sys.path[0] = _root
    else:
        sys.path.insert(0, _root)
    __import__("scripts.vision")
    __package__ = "scripts.vision"

import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Iterator, Mapping  # noqa: E402

import numpy as np  # noqa: E402

from .types import SCHEMA_VERSION, VisionFrame, now  # noqa: E402

RECORD_SCHEMA = "hastycr.vision.record/1"
FRAMES_FILE = "frames.jsonl"
IMAGES_DIR = "images"
IMAGE_FORMATS = ("jpg", "png", "npy")
IMAGE_NAME_RE = re.compile(r"^\d{8}\.(jpg|png|npy)$")
_MB = 1024 * 1024


# ---------------------------------------------------------------------------
# Image helpers (lazy OpenCV / Pillow)
# ---------------------------------------------------------------------------
def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        return None


def image_codec_available(fmt: str) -> bool:
    """True if `fmt` can be written and read in this environment."""
    if fmt == "npy":
        return True
    return _cv2() is not None or _pil() is not None


def _as_rgb(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 uint8 RGB image, got {a.shape} {a.dtype}")
    return a


def save_image(path: str | os.PathLike, rgb: np.ndarray, fmt: str | None = None,
               quality: int = 85) -> int:
    """Write an RGB image atomically (temp file + rename); returns its size."""
    p = Path(path)
    fmt = (fmt or p.suffix.lstrip(".")).lower()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in IMAGE_FORMATS:
        raise ValueError(f"image format must be one of {IMAGE_FORMATS}, not {fmt!r}")
    rgb = _as_rgb(rgb)
    tmp = p.with_name(p.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            if fmt == "npy":
                np.save(fh, np.ascontiguousarray(rgb), allow_pickle=False)
            else:
                cv2 = _cv2()
                if cv2 is not None:
                    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if fmt == "jpg" else []
                    ok, buf = cv2.imencode("." + fmt, np.ascontiguousarray(rgb[:, :, ::-1]), params)
                    if not ok:
                        raise OSError(f"OpenCV could not encode {p}")
                    fh.write(buf.tobytes())
                else:
                    Image = _pil()
                    if Image is None:
                        raise RuntimeError(f"saving .{fmt} needs opencv-python or pillow; use fmt='npy'")
                    kw = {"quality": int(quality)} if fmt == "jpg" else {}
                    Image.fromarray(np.ascontiguousarray(rgb)).save(
                        fh, format="JPEG" if fmt == "jpg" else "PNG", **kw)
        os.replace(tmp, p)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)   # no orphan .tmp (the budget never prunes those)
        except OSError:
            pass
        raise
    return p.stat().st_size


def load_image(path: str | os.PathLike) -> np.ndarray:
    """Read an image written by save_image (or any jpg/png) as HxWx3 uint8 RGB."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return _as_rgb(np.load(p, allow_pickle=False))
    cv2 = _cv2()
    if cv2 is not None:
        # imdecode(fromfile) instead of imread: imread cannot open non-ASCII paths on Windows.
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"cannot decode image {p}")
        return np.ascontiguousarray(bgr[:, :, ::-1])
    Image = _pil()
    if Image is None:
        raise RuntimeError(f"reading {p.suffix} needs opencv-python or pillow")
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB")).copy()


def resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to (width, height): OpenCV, else Pillow, else nearest neighbour."""
    w, h = int(size[0]), int(size[1])
    if rgb.shape[1] == w and rgb.shape[0] == h:
        return rgb
    cv2 = _cv2()
    if cv2 is not None:
        shrink = w < rgb.shape[1]
        return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    Image = _pil()
    if Image is not None:
        bilinear = getattr(Image, "Resampling", Image).BILINEAR     # Pillow >= 9.1 and older
        return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize((w, h), bilinear))
    ys = (np.arange(h) * rgb.shape[0] // h).clip(0, rgb.shape[0] - 1)
    xs = (np.arange(w) * rgb.shape[1] // w).clip(0, rgb.shape[1] - 1)
    return np.ascontiguousarray(rgb[ys][:, xs])


def round_floats(obj: Any, digits: int | None) -> Any:
    """Copy of a JSON-like value with every float rounded to `digits`."""
    if digits is None:
        return obj
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, digits) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Disk budget
# ---------------------------------------------------------------------------
def _session_images(root: Path) -> Iterator[Path]:
    """Recorder images under `root`: <root>/<session>/images/<8 digits>.<ext>,
    only in sessions that hold a frames.jsonl."""
    if not root.is_dir():
        return
    for session in root.iterdir():
        if not (session.is_dir() and (session / FRAMES_FILE).is_file()):
            continue
        images = session / IMAGES_DIR
        if not images.is_dir():
            continue
        for f in images.iterdir():
            if IMAGE_NAME_RE.match(f.name) and f.is_file():
                yield f


class DiskBudget:
    """Keeps recorder images under one root within a byte budget, oldest first."""

    def __init__(self, root: str | os.PathLike, budget_bytes: int, keep_last: int = 50):
        self.root = Path(root)
        self.budget_bytes = int(budget_bytes)
        self.keep_last = max(0, int(keep_last))
        self.removed = 0
        files = []
        for f in _session_images(self.root):
            try:
                st = f.stat()
            except OSError:
                continue
            files.append((st.st_mtime_ns, str(f), st.st_size))
        files.sort()
        self._files: deque[tuple[Path, int]] = deque((Path(p), size) for _, p, size in files)
        self.total_bytes = sum(size for _, size in self._files)

    def __len__(self) -> int:
        return len(self._files)

    def add(self, path: str | os.PathLike, size: int) -> None:
        """Register a newly written image as the newest."""
        self._files.append((Path(path), int(size)))
        self.total_bytes += int(size)

    def prune(self) -> int:
        """Delete oldest images until within budget (keeping keep_last); returns count."""
        n = 0
        while self.total_bytes > self.budget_bytes and len(self._files) > self.keep_last:
            path, size = self._files.popleft()
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass                      # locked (a viewer has it open); forget it anyway
            self.total_bytes -= size
            n += 1
        self.removed += n
        return n


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
def new_session_name(prefix: str = "session", when: datetime | None = None) -> str:
    """'<prefix>_YYYYmmdd_HHMMSS' (sorts chronologically for one prefix)."""
    return f"{prefix}_{(when or datetime.now()):%Y%m%d_%H%M%S}"


def _unique_dir(root: Path, name: str) -> Path:
    path = root / name
    k = 2
    while path.exists():
        path = root / f"{name}_{k}"
        k += 1
    return path


class VisionRecorder:
    """Append VisionFrames (and every Nth image) to <root>/<session>/.

    root: recording root shared by all sessions (the disk budget covers all
        of it). session: directory name; default new_session_name(prefix).
        An existing session is never overwritten: a suffix _2, _3 ... is added.
    image_every: save the frame's image every N frames (0 = never);
        `write(..., force_image=True)` saves one regardless.
    image_format: "jpg" (default), "png" or "npy" (lossless, no codec needed).
    image_width: downscale saved images to this width (keeps the aspect).
    budget_mb / keep_last_images: the disk budget, see DiskBudget.
    float_digits: floats in the JSONL are rounded to this many decimals
        (None = exact); 4 keeps t_capture to 0.1 ms.
    clock: the clock log() stamps records with (default: the vision clock,
        scripts.vision.types.now, the same one frames are captured on).
    Not thread-safe: call it from one thread.
    """

    def __init__(self, root: str | os.PathLike, session: str | None = None, *,
                 prefix: str = "session", image_every: int = 10, image_format: str = "jpg",
                 jpeg_quality: int = 85, image_width: int | None = None,
                 budget_mb: float = 4000.0, keep_last_images: int = 50,
                 float_digits: int | None = 4,
                 clock: Callable[[], float] = now,
                 wall_clock: Callable[[], float] = time.time):
        fmt = image_format.lower().replace("jpeg", "jpg")
        if fmt not in IMAGE_FORMATS:
            raise ValueError(f"image_format must be one of {IMAGE_FORMATS}, not {image_format!r}")
        if image_every and not image_codec_available(fmt):
            raise RuntimeError(f"saving .{fmt} images needs opencv-python or pillow "
                               "(or use image_format='npy', or image_every=0)")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_dir = _unique_dir(self.root, session or new_session_name(prefix))
        self.images_dir = self.session_dir / IMAGES_DIR
        self.images_dir.mkdir(parents=True)
        self.frames_path = self.session_dir / FRAMES_FILE
        self.image_every = max(0, int(image_every))
        self.image_format = fmt
        self.jpeg_quality = int(jpeg_quality)
        self.image_width = image_width
        self.float_digits = float_digits
        self._clock = clock
        self._wall = wall_clock
        self.frames_written = 0
        self.images_written = 0
        self._fh = open(self.frames_path, "a", encoding="utf-8", newline="\n")
        try:
            created = datetime.fromtimestamp(self._wall()).isoformat(timespec="seconds")
        except (OSError, OverflowError, ValueError):
            created = None
        # The header makes this a session directory before DiskBudget scans the root.
        self._write_line({"kind": "session", "schema": RECORD_SCHEMA, "created": created,
                          "image_every": self.image_every, "image_format": fmt,
                          "image_width": image_width, "frame_schema": SCHEMA_VERSION})
        self.budget = DiskBudget(self.root, int(budget_mb * _MB), keep_last_images)

    @property
    def images_pruned(self) -> int:
        return self.budget.removed

    def _write_line(self, rec: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(round_floats(dict(rec), self.float_digits),
                                  separators=(",", ":")) + "\n")
        self._fh.flush()

    def write(self, vf: VisionFrame, rgb: np.ndarray | None = None, *,
              force_image: bool = False, **extra: Any) -> dict[str, Any]:
        """Append one frame; save `rgb` if this frame is due an image. Returns the line."""
        i = self.frames_written
        image_rel = None
        due = force_image or (self.image_every > 0 and i % self.image_every == 0)
        if rgb is not None and due:
            img = _as_rgb(rgb)
            if self.image_width and img.shape[1] != self.image_width:
                img = resize_rgb(img, (int(self.image_width),
                                       max(1, round(img.shape[0] * self.image_width / img.shape[1]))))
            name = f"{i:08d}.{self.image_format}"
            path = self.images_dir / name
            size = save_image(path, img, self.image_format, self.jpeg_quality)
            self.images_written += 1
            self.budget.add(path, size)
            self.budget.prune()
            image_rel = f"{IMAGES_DIR}/{name}"
        rec: dict[str, Any] = {"kind": "frame", "i": i, "t_wall": self._wall(), "image": image_rel,
                               "frame": vf.to_dict()}
        if extra:
            rec["extra"] = extra
        self._write_line(rec)
        self.frames_written += 1
        return rec

    def log(self, kind: str, **data: Any) -> dict[str, Any]:
        """Append a non-frame record (decision, tap, result, ...)."""
        if kind in ("frame", "session"):
            raise ValueError(f"kind {kind!r} is reserved")
        rec = {"kind": str(kind), "t": self._clock(), "t_wall": self._wall(), "data": data}
        self._write_line(rec)
        return rec

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "VisionRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
@dataclass
class RecordEntry:
    """One line of a session log."""
    kind: str
    index: int | None
    data: dict[str, Any]
    frame: VisionFrame | None = None
    image_path: Path | None = None

    @property
    def has_image(self) -> bool:
        return self.image_path is not None and self.image_path.is_file()


class VisionLogReader:
    """Iterate a session written by VisionRecorder (a session dir or its frames.jsonl)."""

    def __init__(self, path: str | os.PathLike):
        p = Path(path)
        self.session_dir = p.parent if p.is_file() else p
        self.frames_path = p if p.is_file() else p / FRAMES_FILE
        if not self.frames_path.is_file():
            raise FileNotFoundError(f"no {FRAMES_FILE} in {self.session_dir}")
        self.header: dict[str, Any] | None = None
        self.bad_lines = 0

    def __iter__(self) -> Iterator[RecordEntry]:
        self.bad_lines = 0
        with open(self.frames_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    self.bad_lines += 1          # a line cut short by a crash
                    continue
                kind = str(rec.get("kind", ""))
                if kind == "session":
                    self.header = rec
                    yield RecordEntry(kind, None, rec)
                elif kind == "frame":
                    image = rec.get("image")
                    yield RecordEntry(kind, rec.get("i"), rec, VisionFrame.from_dict(rec["frame"]),
                                      None if not image else self.session_dir / image)
                else:
                    yield RecordEntry(kind, None, rec)

    def frames(self, with_images_only: bool = False) -> Iterator[RecordEntry]:
        """Frame entries in file (= capture) order."""
        for e in self:
            if e.kind == "frame" and (not with_images_only or e.has_image):
                yield e

    def entries(self, kind: str) -> Iterator[RecordEntry]:
        for e in self:
            if e.kind == kind:
                yield e


def list_sessions(root: str | os.PathLike) -> list[Path]:
    """Session directories under `root`, oldest first (by name)."""
    r = Path(root)
    if not r.is_dir():
        return []
    return sorted(d for d in r.iterdir() if d.is_dir() and (d / FRAMES_FILE).is_file())


def main(argv: list[str] | None = None) -> int:
    """`python -m scripts.vision.record <session or root>`: summarise recordings."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m scripts.vision.record",
                                 description="Summarise recorded vision sessions.")
    ap.add_argument("path", help="a session directory, or a root holding sessions")
    args = ap.parse_args(argv)
    p = Path(args.path)
    sessions = [p] if (p / FRAMES_FILE).is_file() or p.is_file() else list_sessions(p)
    if not sessions:
        print(f"no sessions under {p}")
        return 1
    for s in sessions:
        reader = VisionLogReader(s)
        frames = images = 0
        kinds: dict[str, int] = {}
        t0 = t1 = None
        for e in reader:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
            if e.frame is not None:
                frames += 1
                images += e.has_image
                t0 = e.frame.t_capture if t0 is None else t0
                t1 = e.frame.t_capture
        span = 0.0 if t0 is None else t1 - t0
        print(f"{reader.session_dir.name}: {frames} frames over {span:.1f} s, {images} images on disk, "
              f"records {kinds}, bad lines {reader.bad_lines}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
