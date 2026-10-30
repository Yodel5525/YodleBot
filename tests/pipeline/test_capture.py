"""V08: frame sources (scripts/vision/capture.py).

No window, device or emulator is used: PrintWindow and adb are driven through
injected fakes, staleness through a fake clock, replay through recordings
written into tmp_path.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import capture as C  # noqa: E402
from scripts.vision import record as R  # noqa: E402
from scripts.vision.types import VisionFrame  # noqa: E402


class FakeClock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def rgb(value: int, w: int = 9, h: int = 16) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def screencap_bytes(px: np.ndarray, fmt: int = 1, header: int = 16) -> bytes:
    h, w = px.shape[:2]
    head = w.to_bytes(4, "little") + h.to_bytes(4, "little") + fmt.to_bytes(4, "little")
    if header == 16:
        head += (0).to_bytes(4, "little")             # colour space
    return head + px.astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# staleness and the latest-frame worker
# ---------------------------------------------------------------------------
def test_staleness_with_a_fake_clock():
    clock = FakeClock(100.0)
    grabs = iter([rgb(1), rgb(1), rgb(2), None])
    src = C.LatestFrameSource(lambda: next(grabs), clock=clock, require_portrait=False, start=False)
    assert src.read() is None and src.stale_s == float("inf")
    assert src.poll_once()                                   # t=100.0: frame A
    clock.advance(0.1)
    f = src.read()
    assert f.t == 100.0 and f.stale_s == pytest.approx(0.1) and (f.rgb == 1).all()
    assert src.poll_once()                                   # t=100.1: same pixels (frozen window)
    clock.advance(0.3)
    f = src.read()
    assert f.t == 100.0 and f.stale_s == pytest.approx(0.4)  # keeps the FIRST capture time
    assert src.is_stale(0.25) and src.frames == 1 and src.grabs == 2
    assert src.poll_once()                                   # t=100.4: new pixels
    f = src.read()
    assert f.t == pytest.approx(100.4) and f.stale_s == 0.0 and (f.rgb == 2).all()
    assert not src.is_stale(0.25) and src.frames == 2
    assert not src.poll_once()                               # grab returned None
    clock.advance(1.0)
    f = src.read()
    assert f.t == pytest.approx(100.4) and f.stale_s == pytest.approx(1.0) and src.failures == 1
    assert isinstance(f, tuple) and len(f) == 3              # (t, rgb, stale_s)
    t, frame, stale = f
    assert frame is f.rgb and stale == f.stale_s
    src.close()


def test_grab_errors_are_counted_and_reported():
    clock = FakeClock()
    calls = {"n": 0}

    def grab():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("window gone")
        return rgb(3)

    src = C.LatestFrameSource(grab, clock=clock, require_portrait=False, start=False)
    assert not src.poll_once() and src.failures == 1 and "window gone" in src.error
    assert src.read() is None
    assert src.poll_once() and src.error == "" and src.read() is not None
    src.close()


def test_frames_that_are_not_9_16_portrait_are_refused():
    clock = FakeClock()
    frames = iter([rgb(1, 270, 600), rgb(2, 270, 480)])
    src = C.LatestFrameSource(lambda: next(frames), clock=clock, start=False)
    assert not src.poll_once()
    assert "9:16" in src.error and src.read() is None
    assert src.poll_once()
    assert src.read().rgb.shape == (480, 270, 3)
    src.close()


def test_resize_to_publishes_the_requested_size():
    clock = FakeClock()
    src = C.LatestFrameSource(lambda: rgb(5, 270, 480), clock=clock, resize_to=(540, 960), start=False)
    assert src.poll_once()
    f = src.read()
    assert f.rgb.shape == (960, 540, 3) and (f.rgb == 5).all()
    src.close()


def test_worker_thread_publishes_frames_and_tears_down():
    counter = {"n": 0}
    torn_down = threading.Event()

    def grab():
        counter["n"] += 1
        return rgb(counter["n"] % 250)

    src = C.LatestFrameSource(grab, fps=200, require_portrait=False, teardown=torn_down.set)
    try:
        deadline = time.monotonic() + 3.0
        while src.frames < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert src.frames >= 3
        f = src.read()
        assert f is not None and f.stale_s < 3.0
    finally:
        src.close()
    assert torn_down.is_set()
    assert not src._thread.is_alive()


def test_on_demand_worker_idles_until_read():
    counter = {"n": 0}

    def grab():
        counter["n"] += 1
        return rgb(counter["n"] % 250)

    src = C.LatestFrameSource(grab, fps=200, require_portrait=False, on_demand_s=0.5)
    try:
        time.sleep(0.1)
        assert counter["n"] == 0                       # nobody has asked for a frame yet
        assert src.read() is None                      # asking wakes it
        deadline = time.monotonic() + 3.0
        while src.frames < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert src.frames >= 1 and src.read() is not None
    finally:
        src.close()


# ---------------------------------------------------------------------------
# PrintWindow
# ---------------------------------------------------------------------------
class FakeGrabber:
    """Stands in for scripts/studio/capture.py SurfaceGrabber: BGRA from a reused buffer."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False

    def grab(self):
        return self.frames.pop(0) if self.frames else None

    def close(self):
        self.closed = True


def test_printwindow_source_converts_bgra_and_flags_a_frozen_window():
    clock = FakeClock()
    bgra = np.zeros((16, 9, 4), np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2], bgra[..., 3] = 10, 20, 30, 255
    grabber = FakeGrabber([bgra, bgra.copy()])
    made = []

    def factory():
        made.append(1)
        return grabber

    src = C.PrintWindowSource(factory=factory, clock=clock, require_portrait=False, start=False)
    assert made == []                                   # created by the thread that grabs
    assert src.poll_once() and made == [1]
    f = src.read()
    assert f.rgb[0, 0].tolist() == [30, 20, 10] and f.rgb.shape == (16, 9, 3)
    assert src.window_size == (9, 16)
    bgra[...] = 0                                       # the grabber reuses its buffer
    assert f.rgb[0, 0].tolist() == [30, 20, 10]         # the published frame is a copy
    clock.advance(0.5)
    assert src.poll_once()                              # identical pixels: occluded window
    f2 = src.read()
    assert f2.t == f.t and f2.stale_s == pytest.approx(0.5)
    src.close()
    assert grabber.closed


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows guard")
def test_printwindow_needs_windows_and_is_not_imported_elsewhere():
    with pytest.raises(RuntimeError, match="Windows"):
        C.PrintWindowSource(start=False)
    assert "scripts.studio.capture" not in sys.modules


# ---------------------------------------------------------------------------
# adb
# ---------------------------------------------------------------------------
def test_parse_screencap_raw_headers_and_formats():
    px = np.zeros((2, 3, 4), np.uint8)
    px[..., 0], px[..., 1], px[..., 2], px[..., 3] = 1, 2, 3, 255
    for header in (16, 12):
        out = C.parse_screencap_raw(screencap_bytes(px, 1, header))
        assert out.shape == (2, 3, 3) and out[1, 2].tolist() == [1, 2, 3]
    assert C.parse_screencap_raw(screencap_bytes(px, 5)).reshape(-1, 3)[0].tolist() == [3, 2, 1]   # BGRA
    with pytest.raises(C.CaptureError):
        C.parse_screencap_raw(screencap_bytes(px)[:-1])                 # truncated
    with pytest.raises(C.CaptureError):
        C.parse_screencap_raw(screencap_bytes(px, fmt=4))               # RGB_565
    with pytest.raises(C.CaptureError):
        C.parse_screencap_raw(b"\x00" * 8)


def test_adb_source_with_a_fake_runner():
    clock = FakeClock(50.0)
    px = np.zeros((16, 9, 4), np.uint8)
    px[..., 0] = 200
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        clock.advance(0.2)                              # a slow screencap
        return subprocess.CompletedProcess(cmd, 0, stdout=screencap_bytes(px), stderr=b"")

    src = C.AdbSource("adb.exe", "127.0.0.1:16384", run=run, clock=clock, require_portrait=False,
                      start=False)
    assert src.poll_once()
    cmd, kw = calls[0]
    assert cmd == ["adb.exe", "-s", "127.0.0.1:16384", "exec-out", "screencap"]
    assert kw["capture_output"] is True and kw["timeout"] == 3.0
    f = src.read()
    assert f.t == 50.0                                  # stamped BEFORE the call
    assert f.stale_s == pytest.approx(0.2) and f.rgb[0, 0].tolist() == [200, 0, 0]
    src.close()

    def failing(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"error: device offline")

    def hanging(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 3.0)

    for runner, text in ((failing, "device offline"), (hanging, "TimeoutExpired")):
        bad = C.AdbSource("adb", "s", run=runner, clock=clock, require_portrait=False, start=False)
        assert not bad.poll_once() and text in bad.error and bad.read() is None
        bad.close()


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------
class Scripted:
    def __init__(self, frame):
        self.frame = frame
        self.reads = 0
        self.closed = False

    def read(self):
        self.reads += 1
        return self.frame

    def close(self):
        self.closed = True


def test_fallback_source_switches_on_staleness():
    primary = Scripted(C.Frame(10.0, rgb(1), 0.1))
    fallback = Scripted(C.Frame(10.2, rgb(2), 0.0))
    src = C.FallbackSource(primary, fallback, max_stale_s=0.5)
    assert isinstance(src, C.FrameSource) and isinstance(primary, C.FrameSource)
    assert src.read().t == 10.0 and src.using == "primary" and fallback.reads == 0
    primary.frame = C.Frame(10.0, rgb(1), 0.9)                 # the window froze
    assert src.read().t == 10.2 and src.using == "fallback"
    fallback.frame = None                                      # fallback still warming up
    assert src.read().t == 10.0 and src.using == "primary"
    fallback.frame = C.Frame(9.0, rgb(2), 0.0)                 # older than the stale primary
    assert src.read().t == 10.0
    primary.frame = None
    assert src.read().t == 9.0 and src.using == "fallback"
    fallback.frame = None
    assert src.read() is None and src.using is None
    src.close()
    assert primary.closed and fallback.closed


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
def _record(root: Path, n: int, image_every: int) -> Path:
    with R.VisionRecorder(root, "s", image_every=image_every, image_format="npy") as rec:
        for i in range(n):
            rec.write(VisionFrame(t_capture=50.0 + 0.25 * i, tick=i, stale_s=0.0625 if i == 4 else 0.0),
                      rgb(i))
    return rec.session_dir


def test_replay_is_ordered_and_keeps_recorded_timestamps(tmp_path):
    session = _record(tmp_path, 6, image_every=2)              # images on frames 0, 2, 4
    src = C.FileReplaySource(session)
    assert len(src) == 3
    out = []
    while (f := src.read()) is not None:
        out.append(f)
    assert [f.t for f in out] == [50.0, 50.5, 51.0]
    assert [int(f.rgb[0, 0, 0]) for f in out] == [0, 2, 4]
    assert [f.stale_s for f in out] == [0.0, 0.0, 0.0625]
    assert src.exhausted and src.read() is None
    assert src.vision_frame.tick == 4 and src.image_path.name == "00000004.npy"
    (session / "images" / "00000002.npy").unlink()             # pruned by the disk budget
    assert [f.t for f in iter(C.FileReplaySource(session).read, None)] == [50.0, 51.0]


def test_replay_rebased_timestamps(tmp_path):
    session = _record(tmp_path, 3, image_every=1)
    clock = FakeClock(1000.0)
    src = C.FileReplaySource(session, timestamps="rebased", clock=clock)
    first = src.read()
    clock.advance(5.0)                                         # sequential: the clock only anchors
    rest = [src.read(), src.read()]
    assert [first.t] + [f.t for f in rest] == [1000.0, 1000.25, 1000.5]
    with pytest.raises(ValueError):
        C.FileReplaySource(session, timestamps="wall")


def test_realtime_replay_returns_the_newest_frame_and_drops_missed_ones(tmp_path):
    session = _record(tmp_path, 6, image_every=1)              # t = 50.0, 50.25, ... 51.25
    clock = FakeClock(0.0)
    src = C.FileReplaySource(session, realtime=True, clock=clock, end_grace_s=0.5)
    f = src.read()
    assert f.t == 50.0 and f.stale_s == 0.0 and int(f.rgb[0, 0, 0]) == 0
    clock.advance(0.1)
    f = src.read()
    assert f.t == 50.0 and f.stale_s == pytest.approx(0.1)    # same frame, older
    clock.advance(0.5)                                         # playback at 50.6: frame 1 was missed
    f = src.read()
    assert f.t == 50.5 and int(f.rgb[0, 0, 0]) == 2 and f.stale_s == pytest.approx(0.1)
    clock.advance(10.0)
    assert src.read() is None and src.exhausted


def test_replay_of_an_image_folder(tmp_path):
    for name, value in (("b.npy", 2), ("a.npy", 1), ("c.npy", 3)):
        R.save_image(tmp_path / name, rgb(value), "npy")
    (tmp_path / "readme.txt").write_text("not an image", encoding="utf-8")
    src = C.FileReplaySource(tmp_path, fps=10.0)
    out = list(iter(src.read, None))
    assert [int(f.rgb[0, 0, 0]) for f in out] == [1, 2, 3]
    assert [f.t for f in out] == pytest.approx([0.0, 0.1, 0.2])
    assert src.vision_frame is None
    with pytest.raises(FileNotFoundError):
        C.FileReplaySource(tmp_path / "missing")


def test_next_frame_waits_for_a_newer_frame():
    clock = FakeClock(0.0)
    grabs = iter([rgb(1), rgb(2)])
    src = C.LatestFrameSource(lambda: next(grabs), clock=clock, require_portrait=False, start=False)
    src.poll_once()
    first = C.next_frame(src, None, clock=clock, sleep=lambda s: None)
    assert first.t == 0.0

    def sleep_and_poll(_s):
        clock.advance(0.01)
        src.poll_once()                         # the second grab, then StopIteration failures

    second = C.next_frame(src, first.t, clock=clock, sleep=sleep_and_poll)
    assert second.t == pytest.approx(0.01) and int(second.rgb[0, 0, 0]) == 2
    assert C.next_frame(src, second.t, timeout_s=0.05, clock=clock, sleep=sleep_and_poll) is None
    src.close()


# ---------------------------------------------------------------------------
# the command line: --record and --budget-mb (N04)
# ---------------------------------------------------------------------------
REAL_RECORDER = R.VisionRecorder


def _capture_main(monkeypatch, tmp_path, root: str, *extra: str):
    """capture.main over a 3-image folder replayed in order, recording under tmp_path/root
    without images; returns (exit code, [(VisionRecorder keyword arguments, recorder)])."""
    frames = tmp_path / "frames"
    if not frames.is_dir():
        frames.mkdir()
        for i in range(3):
            R.save_image(frames / f"{i:03d}.npy", rgb(10 + i), "npy")
    monkeypatch.setattr(C, "open_source", lambda kind, **kw: C.FileReplaySource(kind))
    made = []

    class SpyRecorder(REAL_RECORDER):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append((dict(kw), self))

    monkeypatch.setattr(R, "VisionRecorder", SpyRecorder)    # main imports it from record when called
    rc = C.main(["--source", str(frames), "--seconds", "30", "--record", str(tmp_path / root),
                 "--image-every", "0", *extra])
    return rc, made


def test_capture_records_with_the_recorders_budget_by_default(tmp_path, monkeypatch, capsys):
    rc, made = _capture_main(monkeypatch, tmp_path, "rec")
    assert rc == 0 and len(made) == 1
    kw, rec = made[0]
    assert kw == {"prefix": "capture", "image_every": 0}          # the call as before --budget-mb
    assert rec.budget.budget_bytes == int(4000.0 * 1024 * 1024)
    assert rec.frames_written == 3 and rec.session_dir.parent == tmp_path / "rec"
    out = capsys.readouterr().out
    assert f"recording to {rec.session_dir}" in out and "image budget" not in out


def test_capture_budget_mb_sets_the_roots_prune_budget(tmp_path, monkeypatch, capsys):
    rc, made = _capture_main(monkeypatch, tmp_path, "harvest", "--budget-mb", "12000")
    assert rc == 0 and len(made) == 1
    kw, rec = made[0]
    assert kw == {"prefix": "capture", "image_every": 0, "budget_mb": 12000.0}
    assert rec.budget.budget_bytes == int(12000.0 * 1024 * 1024)
    assert "image budget 12000 MB for everything under" in capsys.readouterr().out
    rc, made = _capture_main(monkeypatch, tmp_path, "small", "--budget-mb", "0.5")
    assert rc == 0 and made[0][1].budget.budget_bytes == 512 * 1024


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf"])
def test_capture_refuses_a_budget_that_is_not_a_positive_size(tmp_path, monkeypatch, value):
    with pytest.raises(SystemExit) as exc:
        _capture_main(monkeypatch, tmp_path, "rec", "--budget-mb", value)
    assert exc.value.code == 2
    assert not (tmp_path / "rec").exists()                       # refused before recording


def test_capture_budget_mb_needs_record(monkeypatch):
    monkeypatch.setattr(C, "open_source", lambda kind, **kw: pytest.fail("opened a source"))
    with pytest.raises(SystemExit) as exc:
        C.main(["--source", "adb", "--budget-mb", "8000"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# import weight
# ---------------------------------------------------------------------------
def test_vision_modules_import_without_torch_cv2_or_the_memory_reader():
    code = ("import sys\n"
            "import scripts.vision, scripts.vision.geometry, scripts.vision.types\n"
            "import scripts.vision.record, scripts.vision.capture\n"
            "heavy = ('torch', 'cv2', 'PIL', 'ultralytics', 'scripts.studio.capture',\n"
            "         'scripts.gamestate.memread', 'gamestate.memread', 'royalesim')\n"
            "print(','.join(m for m in heavy if m in sys.modules))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""


@pytest.mark.parametrize("module", ["capture", "record"])
def test_modules_run_as_files(module):
    """Run directly, scripts/vision/types.py must not shadow the stdlib."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "vision" / f"{module}.py"), "--help"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout.lower()
