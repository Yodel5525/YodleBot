"""Frame sources for live vision: threaded PrintWindow, adb screencap, file replay.

Every source implements FrameSource:

    frame = source.read()      # the newest Frame, or None (nothing yet / replay over)
    t, rgb, stale_s = frame    # Frame is a NamedTuple
    source.close()

t        seconds on the vision clock (scripts.vision.types.now, perf_counter),
         taken just before the grab that FIRST returned these pixels. A frozen
         window keeps its old t, so an unchanged t means an unchanged frame.
rgb      HxWx3 uint8 RGB. It is never modified after read() returns it.
stale_s  seconds since t when read() was called: how old the information is.
         In a battle the screen changes every frame (clock, animations), so a
         stale_s above ~0.25 s means an occluded/frozen window or a stalled adb.

Sources (vision-12: the default capture paths are 218-406 ms or stale):
* PrintWindowSource  MuMu's render window via PrintWindow (scripts/studio/
  capture.py SurfaceGrabber), ~4 ms a grab on a worker thread, but at the
  WINDOW's size (560x996 on the author's PC: make the window as large as the
  screen allows) and silently frozen while the window is fully occluded.
  Windows only; constructing it elsewhere raises RuntimeError.
* AdbSource          `adb exec-out screencap` raw framebuffer at device
  resolution, 120-400 ms a grab, on a worker thread. A fallback.
* FallbackSource     the primary's frame, or the fallback's while the primary
  is staler than max_stale_s (the fallback can run on demand only).
* FileReplaySource   a session recorded by scripts/vision/record.py, or a
  folder of images; sequential (every frame once) or realtime (newest frame
  by playback clock, dropping frames a slow consumer misses).

Capture only reads pixels: nothing here taps or navigates.

    python -m scripts.vision.capture --source auto --seconds 10
    python -m scripts.vision.capture --source auto --seconds 400 --record reports/vision/recordings

The bench prints the frame rate, distinct frames and staleness; --record keeps
the frames (VisionFrames with empty perception plus every Nth image), e.g. to
label a friendly later. The recorder prunes the OLDEST images under the whole
--record root once they pass --budget-mb (default 4000 MB, record.py's
VisionRecorder): give each session its own root, or raise the budget, to keep
every image of several battles. A --budget-mb no larger than the images other
sessions already hold under the root is refused (it would delete them at the
first image); a larger one prints how much of it they take.

    python -m scripts.vision.capture --source auto --seconds 420 --record reports/vision/harvest/S1 --budget-mb 8000
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

import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, NamedTuple, Protocol, runtime_checkable  # noqa: E402

import numpy as np  # noqa: E402

from .geometry import FrameSizeError, check_frame_size  # noqa: E402
from .record import FRAMES_FILE, VisionLogReader, load_image, resize_rgb  # noqa: E402
from .types import now  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".npy")
DEFAULT_SERIAL = "127.0.0.1:16384"          # MuMu instance 0 (16384 + 32 * index)
DEFAULT_INSTANCE = "Android Device"         # MuMu instance 0's window title


class CaptureError(RuntimeError):
    """A grab failed (adb error, unreadable framebuffer)."""


class Frame(NamedTuple):
    t: float
    rgb: np.ndarray
    stale_s: float


@runtime_checkable
class FrameSource(Protocol):
    def read(self) -> Frame | None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Machine settings (src/hastycr/machine.py when available)
# ---------------------------------------------------------------------------
def _import_machine():
    try:
        from hastycr import machine
        return machine
    except ImportError:
        src = ROOT / "src"
        if (src / "hastycr" / "machine.py").is_file() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
            try:
                from hastycr import machine
                return machine
            except ImportError:
                return None
        return None


def machine_settings() -> tuple[str, str, str]:
    """(adb, serial, window title) from hastycr.machine (environment variables,
    machine.json), or from HASTYCR_ADB / HASTYCR_SERIAL / HASTYCR_INSTANCE and
    defaults (adb on PATH, 127.0.0.1:16384, "Android Device") without it."""
    m = _import_machine()
    if m is not None:
        cfg = m.get()
        return str(cfg.adb), str(cfg.serial), str(cfg.instance)
    adb = os.environ.get("HASTYCR_ADB") or shutil.which("adb") or "adb"
    return (adb, os.environ.get("HASTYCR_SERIAL") or DEFAULT_SERIAL,
            os.environ.get("HASTYCR_INSTANCE") or DEFAULT_INSTANCE)


def _no_window() -> dict:
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)          # Windows only
    return {"creationflags": flag} if flag else {}


# ---------------------------------------------------------------------------
# The latest-frame worker
# ---------------------------------------------------------------------------
def _checksum(rgb: np.ndarray, stride: int) -> tuple:
    flat = rgb.reshape(-1)
    return rgb.shape, zlib.crc32(np.ascontiguousarray(flat[::stride]).tobytes())


class LatestFrameSource:
    """Grab frames on a worker thread and keep only the newest.

    grab: () -> HxWx3 uint8 RGB array, or None; it may raise (counted as a
        failure, the message kept in `.error`). Called from one thread.
    fps: grab rate cap. clock: the timestamp clock (default the vision clock).
    require_portrait: drop frames that are not 9:16 portrait (geometry
        check_frame_size) and say why in `.error`.
    resize_to: (width, height) every published frame is resized to.
    checksum_stride: a frame counts as new when a CRC of every Nth byte
        changes (a frozen window returns identical pixels).
    on_demand_s: if set, the worker grabs only while read() was called in the
        last on_demand_s seconds (an adb fallback that idles otherwise).
    start: start the thread now; with start=False drive it with poll_once().
    """

    def __init__(self, grab: Callable[[], np.ndarray | None], *, fps: float = 60.0,
                 clock: Callable[[], float] = now, name: str = "capture",
                 require_portrait: bool = True, resize_to: tuple[int, int] | None = None,
                 checksum_stride: int = 211, on_demand_s: float | None = None,
                 teardown: Callable[[], None] | None = None, start: bool = True):
        self._grab_fn = grab
        self.fps = float(fps)
        self._clock = clock
        self.name = name
        self.require_portrait = bool(require_portrait)
        self.resize_to = resize_to
        self._stride = max(1, int(checksum_stride))
        self._on_demand_s = on_demand_s
        self._teardown = teardown
        self._torn_down = False
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._t: float | None = None            # first capture time of the current content
        self._checksum: tuple | None = None
        self._last_read: float | None = None
        self.t_last_grab: float | None = None   # time of the newest successful grab
        self.frames = 0                         # distinct frames
        self.grabs = 0                          # successful grabs
        self.failures = 0
        self.error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if start:
            self.start()

    # -- worker ------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        if self._stop.is_set():
            raise RuntimeError(f"{self.name}: source is closed")
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def _wanted(self) -> bool:
        if self._on_demand_s is None:
            return True
        last = self._last_read
        return last is not None and self._clock() - last <= self._on_demand_s

    def _run(self) -> None:
        interval = 1.0 / max(0.1, self.fps)
        try:
            while not self._stop.is_set():
                if not self._wanted():
                    self._stop.wait(min(0.05, interval))
                    continue
                started = time.perf_counter()
                self.poll_once()
                self._stop.wait(max(0.0, interval - (time.perf_counter() - started)))
        except Exception as exc:                # never let the thread die silently
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._teardown_once()

    def _teardown_once(self) -> None:
        with self._lock:
            if self._torn_down:
                return
            self._torn_down = True
        if self._teardown is not None:
            try:
                self._teardown()
            except Exception as exc:
                self.error = f"teardown {type(exc).__name__}: {exc}"

    def poll_once(self) -> bool:
        """One grab; publish it if it is new. True on a successful grab."""
        t0 = self._clock()
        try:
            rgb = self._grab_fn()
        except Exception as exc:
            self.failures += 1
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        if rgb is None:
            self.failures += 1
            return False
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            self.failures += 1
            self.error = f"grab returned {rgb.shape} {rgb.dtype}, not HxWx3 uint8"
            return False
        if self.require_portrait:
            try:
                check_frame_size(rgb.shape[1], rgb.shape[0])
            except FrameSizeError as exc:
                self.failures += 1
                self.error = str(exc)
                return False
        cs = _checksum(rgb, self._stride)
        changed = cs != self._checksum
        if changed and self.resize_to is not None:
            rgb = resize_rgb(rgb, self.resize_to)
        with self._lock:
            self.grabs += 1
            self.t_last_grab = t0
            if changed:
                self._checksum = cs
                self._latest = rgb
                self._t = t0
                self.frames += 1
            self.error = ""
        return True

    # -- consumer side ------------------------------------------------------------
    def read(self) -> Frame | None:
        t_now = self._clock()
        with self._lock:
            self._last_read = t_now
            if self._latest is None or self._t is None:
                return None
            return Frame(self._t, self._latest, max(0.0, t_now - self._t))

    @property
    def stale_s(self) -> float:
        """Seconds since the current content first appeared (inf before any frame)."""
        t = self._t
        return float("inf") if t is None else max(0.0, self._clock() - t)

    def is_stale(self, max_stale_s: float) -> bool:
        return self.stale_s > max_stale_s

    def close(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        th = self._thread
        if th is not None and th is not threading.current_thread():
            th.join(timeout_s)
        if th is None or not th.is_alive():
            self._teardown_once()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# PrintWindow (Windows)
# ---------------------------------------------------------------------------
def printwindow_factory(instance: str | None = None, hwnd: int | None = None) -> Callable[[], object]:
    """A factory of scripts/studio/capture.py SurfaceGrabbers for one MuMu window.

    The window is chosen by its title (`instance`, default the machine's
    instance setting) or `hwnd`; it is looked up now, so a missing window fails
    here. The grabber itself is created by the worker thread that uses it.
    """
    if sys.platform != "win32":
        raise RuntimeError("PrintWindowSource captures the MuMu window and needs Windows; "
                           "use AdbSource or FileReplaySource here")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.studio.capture import SurfaceGrabber, find_surface   # Windows-only module
    title = instance if instance is not None else machine_settings()[2]
    surface = find_surface(hwnd=hwnd, instance=title)
    return lambda: SurfaceGrabber(surface)


class PrintWindowSource(LatestFrameSource):
    """MuMu's render surface via PrintWindow on a worker thread (Windows).

    factory: () -> an object with grab() -> HxWx4 BGRA array or None, and
        close(); default printwindow_factory(instance, hwnd). Tests pass fakes.
    The frame has the window's size (window_size); pass resize_to=(1080, 1920)
    if downstream readers need the device resolution.
    """

    def __init__(self, instance: str | None = None, hwnd: int | None = None, *,
                 factory: Callable[[], object] | None = None, fps: float = 60.0, **kw):
        self._factory = factory if factory is not None else printwindow_factory(instance, hwnd)
        self._grabber = None
        self.window_size: tuple[int, int] | None = None
        kw.setdefault("name", "printwindow")
        super().__init__(self._grab_rgb, fps=fps, teardown=self._close_grabber, **kw)

    def _grab_rgb(self) -> np.ndarray | None:
        if self._grabber is None:
            self._grabber = self._factory()
        bgra = self._grabber.grab()
        if bgra is None:
            return None
        self.window_size = (int(bgra.shape[1]), int(bgra.shape[0]))
        # BGRA -> RGB, copied out of the grabber's reused DIB buffer.
        return np.ascontiguousarray(bgra[:, :, 2::-1])

    def _close_grabber(self) -> None:
        g, self._grabber = self._grabber, None
        if g is not None:
            g.close()


# ---------------------------------------------------------------------------
# adb screencap
# ---------------------------------------------------------------------------
def parse_screencap_raw(raw: bytes) -> np.ndarray:
    """Raw `screencap` output (12- or 16-byte header, then RGBA/RGBX/BGRA) -> RGB."""
    if len(raw) < 12:
        raise CaptureError(f"screencap returned {len(raw)} bytes")
    w = int.from_bytes(raw[0:4], "little")
    h = int.from_bytes(raw[4:8], "little")
    fmt = int.from_bytes(raw[8:12], "little")
    if not (0 < w <= 16384 and 0 < h <= 16384):
        raise CaptureError(f"screencap header says {w}x{h}")
    n = w * h * 4
    header = len(raw) - n
    if header not in (12, 16):
        raise CaptureError(f"screencap: {len(raw)} bytes do not hold a {w}x{h} 4-byte frame")
    px = np.frombuffer(raw, dtype=np.uint8, count=n, offset=header).reshape(h, w, 4)
    if fmt in (1, 2):                     # RGBA_8888, RGBX_8888
        rgb = px[:, :, :3]
    elif fmt == 5:                        # BGRA_8888
        rgb = px[:, :, 2::-1]
    else:
        raise CaptureError(f"screencap pixel format {fmt} is not supported")
    return np.ascontiguousarray(rgb)


def adb_screencap(adb: str, serial: str, *, timeout_s: float = 3.0,
                  run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> np.ndarray:
    """One raw framebuffer grab over adb as RGB; raises CaptureError."""
    cmd = [str(adb), "-s", str(serial), "exec-out", "screencap"]
    try:
        p = run(cmd, capture_output=True, timeout=timeout_s, check=False, **_no_window())
    except (OSError, subprocess.SubprocessError) as exc:
        raise CaptureError(f"adb screencap failed: {type(exc).__name__}: {exc}") from exc
    if p.returncode != 0 or not p.stdout:
        err = (p.stderr or b"").decode("utf-8", "replace").strip()[:200]
        raise CaptureError(f"adb screencap exit {p.returncode}: {err or 'no output'}")
    return parse_screencap_raw(p.stdout)


class AdbSource(LatestFrameSource):
    """`adb exec-out screencap` on a worker thread: device resolution, slow.

    adb/serial default to machine_settings(). `run` replaces subprocess.run
    (tests). A screencap shares the adb server with the tap shell, so keep
    the rate low (fps), or use on_demand_s as a fallback.
    """

    def __init__(self, adb: str | None = None, serial: str | None = None, *, fps: float = 8.0,
                 timeout_s: float = 3.0, run: Callable[..., subprocess.CompletedProcess] | None = None,
                 **kw):
        if adb is None or serial is None:
            m_adb, m_serial, _ = machine_settings()
            adb = adb or m_adb
            serial = serial or m_serial
        self.adb, self.serial = str(adb), str(serial)
        self.timeout_s = float(timeout_s)
        self._run_cmd = run if run is not None else subprocess.run
        kw.setdefault("name", "adb-screencap")
        super().__init__(self._grab_rgb, fps=fps, **kw)

    def _grab_rgb(self) -> np.ndarray:
        return adb_screencap(self.adb, self.serial, timeout_s=self.timeout_s, run=self._run_cmd)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------
class FallbackSource:
    """The primary's frame unless it is staler than max_stale_s; then the
    fallback's, if that one is newer. `using` says which one answered."""

    def __init__(self, primary: FrameSource, fallback: FrameSource, *, max_stale_s: float = 0.5):
        self.primary, self.fallback = primary, fallback
        self.max_stale_s = float(max_stale_s)
        self.using: str | None = None

    def read(self) -> Frame | None:
        f = self.primary.read()
        if f is not None and f.stale_s <= self.max_stale_s:
            self.using = "primary"
            return f
        g = self.fallback.read()
        if g is not None and (f is None or g.t > f.t):
            self.using = "fallback"
            return g
        self.using = "primary" if f is not None else None
        return f

    def close(self) -> None:
        for s in (self.primary, self.fallback):
            try:
                s.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# File replay
# ---------------------------------------------------------------------------
class _Item(NamedTuple):
    t: float
    stale_s: float
    path: Path
    frame: object              # the recorded VisionFrame, or None for a plain image folder


class FileReplaySource:
    """Replay recorded frames as a FrameSource.

    path: a session directory (or its frames.jsonl) written by VisionRecorder,
        whose frames WITH an image on disk are replayed in file order; or a
        folder of .jpg/.png/.npy images, replayed in name order at `fps`
        (t = index / fps).
    realtime=False: every frame once, in order; read() returns None at the end.
    realtime=True: read() returns the newest frame whose recorded time has
        passed on `clock` since the first read (frames a slow consumer misses
        are dropped, as with a live source); None once the last frame is
        older than end_grace_s.
    timestamps="recorded": t is the recorded capture time; "rebased": t is
        moved onto `clock`, the first frame at the first read().
    After each read, `vision_frame` is the recorded VisionFrame (None for a
    plain folder) and `image_path` the image file.
    """

    def __init__(self, path: str | os.PathLike, *, realtime: bool = False,
                 timestamps: str = "recorded", fps: float = 20.0,
                 clock: Callable[[], float] = now, end_grace_s: float = 1.0,
                 resize_to: tuple[int, int] | None = None):
        if timestamps not in ("recorded", "rebased"):
            raise ValueError(f"timestamps must be 'recorded' or 'rebased', not {timestamps!r}")
        p = Path(path)
        if (p.is_dir() and (p / FRAMES_FILE).is_file()) or (p.is_file() and p.suffix == ".jsonl"):
            reader = VisionLogReader(p)
            items = [_Item(e.frame.t_capture, e.frame.stale_s, e.image_path, e.frame)
                     for e in reader.frames(with_images_only=True)]
        elif p.is_dir():
            files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)
            items = [_Item(i / float(fps), 0.0, f, None) for i, f in enumerate(files)]
        else:
            raise FileNotFoundError(f"nothing to replay at {p}")
        if realtime and any(b.t < a.t for a, b in zip(items, items[1:])):
            items.sort(key=lambda it: it.t)              # stable: ties keep file order
        self._items = items
        self.realtime = bool(realtime)
        self.timestamps = timestamps
        self._clock = clock
        self.end_grace_s = float(end_grace_s)
        self.resize_to = resize_to
        self._next = 0                 # sequential: index of the next frame
        self._cur = -1                 # realtime: index of the frame last shown
        self._t0_clock: float | None = None
        self._cache: tuple[int, np.ndarray] | None = None
        self._done = False
        self.vision_frame = None
        self.image_path: Path | None = None

    def __len__(self) -> int:
        return len(self._items)

    @property
    def exhausted(self) -> bool:
        return self._done

    def _image(self, i: int) -> np.ndarray:
        if self._cache is not None and self._cache[0] == i:
            return self._cache[1]
        rgb = load_image(self._items[i].path)
        if self.resize_to is not None:
            rgb = resize_rgb(rgb, self.resize_to)
        self._cache = (i, rgb)
        return rgb

    def _emit(self, i: int, stale_extra: float) -> Frame:
        item = self._items[i]
        t = item.t
        if self.timestamps == "rebased":
            t = self._t0_clock + (item.t - self._items[0].t)
        self.vision_frame = item.frame
        self.image_path = item.path
        return Frame(t, self._image(i), item.stale_s + max(0.0, stale_extra))

    def read(self) -> Frame | None:
        if self._done or not self._items:
            self._done = True
            return None
        if self._t0_clock is None:
            self._t0_clock = self._clock()
        if not self.realtime:
            if self._next >= len(self._items):
                self._done = True
                return None
            i = self._next
            self._next += 1
            return self._emit(i, 0.0)
        target = self._items[0].t + (self._clock() - self._t0_clock)
        if target > self._items[-1].t + self.end_grace_s:
            self._done = True
            return None
        j = max(self._cur, 0)
        while j + 1 < len(self._items) and self._items[j + 1].t <= target:
            j += 1
        self._cur = j
        return self._emit(j, target - self._items[j].t)

    def close(self) -> None:
        self._done = True
        self._cache = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def next_frame(source: FrameSource, after_t: float | None = None, *, timeout_s: float = 1.0,
               poll_s: float = 0.002, clock: Callable[[], float] = now,
               sleep: Callable[[float], None] = time.sleep) -> Frame | None:
    """Wait for a frame newer than `after_t` (any frame if None); None on timeout
    or when a replay is exhausted."""
    deadline = clock() + timeout_s
    while True:
        f = source.read()
        if f is not None and (after_t is None or f.t > after_t):
            return f
        if getattr(source, "exhausted", False) or clock() >= deadline:
            return None
        sleep(poll_s)


def open_source(kind: str = "auto", *, instance: str | None = None, adb: str | None = None,
                serial: str | None = None, max_stale_s: float = 0.5,
                resize_to: tuple[int, int] | None = None, realtime: bool = True) -> FrameSource:
    """A source by name: "printwindow", "adb", "auto" (PrintWindow with an
    on-demand adb fallback on Windows, adb elsewhere), or a path to replay."""
    if kind == "printwindow":
        return PrintWindowSource(instance, resize_to=resize_to)
    if kind == "adb":
        return AdbSource(adb, serial, resize_to=resize_to)
    if kind == "auto":
        if sys.platform == "win32":
            primary = PrintWindowSource(instance, resize_to=resize_to)
            fallback = AdbSource(adb, serial, resize_to=resize_to, on_demand_s=2.0)
            return FallbackSource(primary, fallback, max_stale_s=max_stale_s)
        return AdbSource(adb, serial, resize_to=resize_to)
    p = Path(kind)
    if p.exists():
        return FileReplaySource(p, realtime=realtime, resize_to=resize_to)
    raise ValueError(f"unknown source {kind!r}: use auto, printwindow, adb or a recording path")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m scripts.vision.capture",
                                 description="Benchmark a frame source and optionally record it.")
    ap.add_argument("--source", default="auto",
                    help="auto | printwindow | adb | a recording/image folder to replay (default auto)")
    ap.add_argument("--instance", help="MuMu window title (default: the machine's instance setting)")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to run")
    ap.add_argument("--max-stale", type=float, default=0.5,
                    help="auto: switch to adb while the window is staler than this (s)")
    ap.add_argument("--resize", default=None, help="publish frames at WIDTHxHEIGHT, e.g. 1080x1920")
    ap.add_argument("--record", default=None, help="record frames under this root (VisionRecorder)")
    ap.add_argument("--image-every", type=int, default=5, help="with --record: an image every N frames")
    ap.add_argument("--budget-mb", type=float, default=None,
                    help="with --record: the image budget of the WHOLE --record root in MB; above it the "
                         "oldest images under the root are deleted (default: the recorder's 4000)")
    args = ap.parse_args(argv)
    held_mb = 0.0
    if args.budget_mb is not None:
        if not args.record:
            ap.error("--budget-mb applies to --record")
        if not (math.isfinite(args.budget_mb) and args.budget_mb > 0):
            ap.error(f"--budget-mb must be a positive number of MB, not {args.budget_mb}")
        from .record import _MB, DiskBudget
        held_mb = DiskBudget(args.record, 0).total_bytes / _MB     # scans only; deletes nothing
        if held_mb and held_mb >= args.budget_mb:
            ap.error(f"--budget-mb {args.budget_mb:g}: {args.record} already holds {held_mb:.1f} MB of "
                     "other sessions' images, so the recorder would delete the oldest of them at its "
                     "first image. Record to a root of its own (one per session) or pass a larger budget.")
    from .geometry import parse_size
    resize = parse_size(args.resize) if args.resize else None
    source = open_source(args.source, instance=args.instance, max_stale_s=args.max_stale, resize_to=resize)
    recorder = None
    if args.record:
        from .record import VisionRecorder
        from .types import VisionFrame
        budget = {} if args.budget_mb is None else {"budget_mb": args.budget_mb}
        recorder = VisionRecorder(args.record, prefix="capture", image_every=args.image_every, **budget)
        print(f"recording to {recorder.session_dir}")
        if budget:
            print(f"  image budget {args.budget_mb:g} MB for everything under {recorder.root}")
            if held_mb > 0:
                print(f"  {held_mb:.1f} MB of it is other sessions' images: once the total passes the "
                      "budget the oldest go first")
    seen_t: list[float] = []
    stale: list[float] = []
    size = None
    t_end = now() + args.seconds
    last_t = None
    try:
        while now() < t_end:
            f = next_frame(source, last_t, timeout_s=1.0)
            if f is None:
                if getattr(source, "exhausted", False):
                    break
                continue
            last_t = f.t
            seen_t.append(f.t)
            stale.append(f.stale_s)
            size = (f.rgb.shape[1], f.rgb.shape[0])
            if recorder is not None:
                recorder.write(VisionFrame(t_capture=f.t, frame_size=size, stale_s=f.stale_s), f.rgb)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        if recorder is not None:
            recorder.close()
    gaps = [b - a for a, b in zip(seen_t, seen_t[1:])]
    print(f"source {args.source}: {len(seen_t)} distinct frames, size {size}")
    if gaps:
        print(f"  frame interval median {1000 * _percentile(gaps, 0.5):.1f} ms, "
              f"p95 {1000 * _percentile(gaps, 0.95):.1f} ms ({1 / max(1e-9, _percentile(gaps, 0.5)):.0f} fps)")
    if stale:
        print(f"  stale_s at read median {1000 * _percentile(stale, 0.5):.1f} ms, max {1000 * max(stale):.0f} ms")
    for s in (source, getattr(source, "primary", None), getattr(source, "fallback", None)):
        if s is not None and getattr(s, "error", ""):
            print(f"  {getattr(s, 'name', type(s).__name__)} error: {s.error}")
    return 0 if seen_t else 1


if __name__ == "__main__":
    sys.exit(main())
