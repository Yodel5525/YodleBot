"""V02: the VisionFrame contract and the recorder (scripts/vision/types.py, record.py)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import record as R  # noqa: E402
from scripts.vision import types as VT  # noqa: E402

WALL = 1_761_000_000.0          # a fixed wall clock (2025-10-20), valid on every platform


def make_frame(t: float = 12.375) -> VT.VisionFrame:
    """A frame using every field. All floats are exact in binary with <= 4
    decimals, so the recorder's 4-digit rounding keeps them unchanged."""
    return VT.VisionFrame(
        t_capture=t,
        tick=1234,
        tracks=[
            VT.Track(uid=7, cls="HogRider", team=VT.ENEMY_TEAM, conf=0.875, x_mt=3500.5, y_mt=14000.25,
                     hp_frac=None, form=VT.FORM_BASE, first_seen_tick=1200, last_seen_tick=1234,
                     vx=0.5, vy=-12.25),
            VT.Track(uid=8, cls="Skeleton", team=VT.OWN_TEAM, conf=0.5, x_mt=9000.0, y_mt=25000.0,
                     hp_frac=0.25, form=VT.FORM_EVO, first_seen_tick=1230, last_seen_tick=1234),
        ],
        hud=VT.Hud(hand=[3, 17, VT.EMPTY_CARD, 42], next=5, elixir_milli=6250, clock_tick=1236,
                   phase=VT.PHASE_REGULATION,
                   towers={"own_king": VT.TowerHp(4824, 4824), "enemy_left": VT.TowerHp(1203, 3052),
                           "enemy_right": VT.TowerHp(None, None)},
                   crowns=(1, 0)),
        events=VT.Events(
            opp_plays=[VT.OppPlay(card=17, tick=1200, exec_tick=1198, x_mt=3500.0, y_mt=13000.0,
                                  kind="unit", uids=[7])],
            own_landings=[VT.OwnLanding(card=3, slot=0, tap_tick=1190, land_tick=1214, tap_t=11.5,
                                        aim_x_mt=9500.0, aim_y_mt=24500.0, x_mt=9000.0, y_mt=25000.0, uid=8),
                          VT.OwnLanding(card=42, slot=3, tap_tick=1220)]),
        screen=VT.SCREEN_BATTLE,
        detections=[VT.Detection("HogRider", side_prob=0.9375, conf=0.875, box=(100.0, 200.0, 150.0, 260.5),
                                 ground_px=(125.0, 260.5)),
                    VT.Detection("Skeleton", 0.0625, 0.5, (500.0, 900.0, 520.0, 930.0), None, VT.FORM_EVO)],
        frame_size=(1080, 1920),
        stale_s=0.0625,
    )


def rgb(value: int, w: int = 9, h: int = 16) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# types
# ---------------------------------------------------------------------------
def test_vision_frame_round_trips_exactly_through_json():
    vf = make_frame()
    text = vf.to_json()
    assert VT.VisionFrame.from_json(text) == vf
    plain = json.loads(text)                       # plain JSON values only
    assert json.loads(json.dumps(plain)) == plain
    assert plain["v"] == VT.SCHEMA_VERSION
    assert plain["detections"][0]["box"] == [100.0, 200.0, 150.0, 260.5]
    back = VT.VisionFrame.from_dict(plain)
    assert back.detections[0].box == (100.0, 200.0, 150.0, 260.5)        # tuples restored
    assert back.hud.crowns == (1, 0) and back.frame_size == (1080, 1920)
    assert back.tracks[0].hp_frac is None and back.hud.towers["enemy_right"].hp is None


def test_from_dict_fills_defaults_and_ignores_unknown_keys():
    vf = VT.VisionFrame.from_dict({"t_capture": 1.5, "a_future_field": 3})
    assert vf.tick is None and vf.tracks == [] and vf.detections == []
    assert vf.hud.hand == [VT.EMPTY_CARD] * 4 and vf.hud.next == VT.EMPTY_CARD
    assert vf.hud.phase == VT.PHASE_UNKNOWN and vf.hud.elixir_milli is None
    assert vf.screen == VT.SCREEN_UNKNOWN and vf.frame_size is None and vf.stale_s == 0.0
    tr = VT.Track.from_dict({"uid": 1, "cls": "Knight", "x_mt": 1, "y_mt": 2, "colour": "red"})
    assert tr.team == VT.UNKNOWN_TEAM and tr.hp_frac is None and tr.form == VT.FORM_BASE
    with pytest.raises(ValueError, match="newer"):
        VT.VisionFrame.from_dict({"v": VT.SCHEMA_VERSION + 1, "t_capture": 0.0})


def test_small_helpers():
    vf = make_frame()
    assert vf.in_battle
    assert [t.uid for t in vf.own_tracks()] == [8] and [t.uid for t in vf.enemy_tracks()] == [7]
    assert vf.hud.towers["enemy_left"].frac == pytest.approx(1203 / 3052)
    assert vf.hud.towers["enemy_right"].frac is None and VT.TowerHp(5, 0).frac is None
    assert vf.events.own_landings[0].lag_ticks == 24 and vf.events.own_landings[1].lag_ticks is None
    assert set(VT.TOWER_KEYS) >= set(vf.hud.towers)
    assert callable(VT.now) and VT.now() <= VT.now()


# ---------------------------------------------------------------------------
# recorder and reader
# ---------------------------------------------------------------------------
def test_recorder_writes_frames_images_and_logs(tmp_path):
    root = tmp_path / "rec"
    with R.VisionRecorder(root, "s1", image_every=2, image_format="npy", wall_clock=lambda: WALL) as rec:
        for i in range(5):
            rec.write(make_frame(t=10.0 + 0.0625 * i), rgb(i), battle=i)
        rec.write(make_frame(t=11.0), None)                    # no pixels: no image even if due
        rec.log("tap", slot=1, px=[10, 20])
    assert rec.frames_written == 6 and rec.images_written == 3
    reader = R.VisionLogReader(root / "s1")
    entries = list(reader)
    assert [e.kind for e in entries] == ["session"] + ["frame"] * 6 + ["tap"]
    assert reader.header["image_every"] == 2 and reader.header["image_format"] == "npy"
    frames = list(reader.frames())
    assert [e.index for e in frames] == [0, 1, 2, 3, 4, 5]
    assert [e.frame.t_capture for e in frames] == [10.0, 10.0625, 10.125, 10.1875, 10.25, 11.0]
    assert [e.has_image for e in frames] == [True, False, True, False, True, False]
    assert frames[0].frame == make_frame(t=10.0)
    assert frames[1].data["extra"] == {"battle": 1}
    img = R.load_image(frames[2].image_path)
    assert img.shape == (16, 9, 3) and img.dtype == np.uint8 and (img == 2).all()
    assert frames[2].image_path == root / "s1" / "images" / "00000002.npy"
    tap = entries[-1]
    assert tap.data["data"] == {"slot": 1, "px": [10, 20]} and "t" in tap.data
    assert [e.index for e in reader.frames(with_images_only=True)] == [0, 2, 4]
    assert [e.kind for e in reader.entries("tap")] == ["tap"]
    # the same reader also accepts the jsonl path
    assert len(list(R.VisionLogReader(root / "s1" / R.FRAMES_FILE).frames())) == 6


def test_force_image_and_downscale(tmp_path):
    with R.VisionRecorder(tmp_path, "s", image_every=0, image_format="npy", image_width=4) as rec:
        rec.write(make_frame(), rgb(7, w=8, h=16))
        rec.write(make_frame(), rgb(9, w=8, h=16), force_image=True)
    frames = list(R.VisionLogReader(tmp_path / "s").frames())
    assert [e.has_image for e in frames] == [False, True]
    img = R.load_image(frames[1].image_path)
    assert img.shape == (8, 4, 3) and (img == 9).all()


def _image_size(tmp_path: Path) -> int:
    return R.save_image(tmp_path / "probe.npy", rgb(0), "npy")


def test_budget_prunes_oldest_images_first(tmp_path):
    size = _image_size(tmp_path)
    root = tmp_path / "rec"
    budget_mb = 3.5 * size / (1024 * 1024)                     # room for 3 images
    with R.VisionRecorder(root, "s", image_every=1, image_format="npy", budget_mb=budget_mb,
                          keep_last_images=2) as rec:
        for i in range(10):
            rec.write(make_frame(t=float(i)), rgb(i))
    left = sorted(p.name for p in (root / "s" / "images").iterdir())
    assert left == ["00000007.npy", "00000008.npy", "00000009.npy"]
    assert rec.images_pruned == 7
    assert sum((root / "s" / "images" / n).stat().st_size for n in left) <= 3.5 * size
    frames = list(R.VisionLogReader(root / "s").frames())
    assert len(frames) == 10                                   # the log itself is never pruned
    assert [e.has_image for e in frames] == [False] * 7 + [True] * 3


def test_budget_always_keeps_the_newest_images(tmp_path):
    with R.VisionRecorder(tmp_path, "s", image_every=1, image_format="npy", budget_mb=0.0,
                          keep_last_images=2) as rec:
        for i in range(5):
            rec.write(make_frame(), rgb(i))
    assert sorted(p.name for p in (tmp_path / "s" / "images").iterdir()) == ["00000003.npy", "00000004.npy"]


def test_budget_spans_sessions_and_spares_foreign_files(tmp_path):
    size = _image_size(tmp_path)
    root = tmp_path / "rec"
    with R.VisionRecorder(root, "a_old", image_every=1, image_format="npy") as old:
        for i in range(3):
            old.write(make_frame(), rgb(i))
    old_images = root / "a_old" / "images"
    for f in old_images.iterdir():
        os.utime(f, ns=(1_000_000_000, 1_000_000_000))       # older than anything written next
    foreign = [root / "notes.npy", old_images / "keep_me.npy",
               root / "not_a_session" / "images" / "00000000.npy"]
    foreign[2].parent.mkdir(parents=True)
    for f in foreign:
        f.write_bytes(b"x" * (10 * size))
    with R.VisionRecorder(root, "b_new", image_every=1, image_format="npy",
                          budget_mb=3.5 * size / (1024 * 1024), keep_last_images=1) as new:
        for i in range(2):
            new.write(make_frame(), rgb(i))
    assert sorted(p.name for p in old_images.iterdir()) == ["00000002.npy", "keep_me.npy"]
    assert sorted(p.name for p in (root / "b_new" / "images").iterdir()) == ["00000000.npy", "00000001.npy"]
    assert all(f.exists() for f in foreign)
    assert new.images_pruned == 2
    assert [p.name for p in R.list_sessions(root)] == ["a_old", "b_new"]


def test_reader_skips_a_truncated_last_line(tmp_path):
    with R.VisionRecorder(tmp_path, "s", image_every=0) as rec:
        rec.write(make_frame(t=1.0))
        rec.write(make_frame(t=2.0))
    with open(tmp_path / "s" / R.FRAMES_FILE, "a", encoding="utf-8") as fh:
        fh.write('{"kind":"frame","i":2,"fra')                  # a crash mid-write
    reader = R.VisionLogReader(tmp_path / "s")
    assert [e.frame.t_capture for e in reader.frames()] == [1.0, 2.0]
    assert reader.bad_lines == 1


def test_sessions_are_never_overwritten(tmp_path):
    with R.VisionRecorder(tmp_path, "same", image_every=0) as a:
        a.write(make_frame(t=1.0))
    with R.VisionRecorder(tmp_path, "same", image_every=0) as b:
        b.write(make_frame(t=2.0))
    assert a.session_dir.name == "same" and b.session_dir.name == "same_2"
    assert [e.frame.t_capture for e in R.VisionLogReader(a.session_dir).frames()] == [1.0]
    assert R.new_session_name("friendly").startswith("friendly_")


def test_recorder_rounds_floats_and_rejects_bad_input(tmp_path):
    with R.VisionRecorder(tmp_path, "s", image_every=1, image_format="npy") as rec:
        line = rec.write(make_frame(t=1.23456789), rgb(1))
        with pytest.raises(ValueError):
            rec.write(make_frame(), np.zeros((4, 4), np.uint8))        # not HxWx3
        with pytest.raises(ValueError):
            rec.log("frame", x=1)
    assert line["frame"]["t_capture"] == 1.23456789                    # the returned dict is unrounded
    stored = next(R.VisionLogReader(tmp_path / "s").frames()).frame
    assert stored.t_capture == 1.2346
    assert R.round_floats({"a": [1.23456, 2], "b": {"c": 0.5}, "d": True}, 2) == {"a": [1.23, 2], "b": {"c": 0.5}, "d": True}
    with pytest.raises(ValueError):
        R.VisionRecorder(tmp_path, "t", image_format="gif")


def test_without_codecs_jpg_is_refused_cleanly(tmp_path, monkeypatch):
    """A bare venv (no OpenCV, no Pillow): jpg fails without leaving a .tmp,
    npy still works and resizing falls back to nearest neighbour."""
    monkeypatch.setattr(R, "_cv2", lambda: None)
    monkeypatch.setattr(R, "_pil", lambda: None)
    assert not R.image_codec_available("jpg") and R.image_codec_available("npy")
    with pytest.raises(RuntimeError):
        R.save_image(tmp_path / "x.jpg", rgb(1))
    assert list(tmp_path.iterdir()) == []                    # no orphan x.jpg.tmp
    with pytest.raises(RuntimeError, match="npy"):
        R.VisionRecorder(tmp_path / "rec", "s", image_every=1, image_format="jpg")
    small = R.resize_rgb(rgb(5, w=8, h=16), (4, 8))
    assert small.shape == (8, 4, 3) and (small == 5).all()


def test_jpeg_round_trip(tmp_path):
    if not R.image_codec_available("jpg"):
        pytest.skip("neither opencv-python nor pillow is installed")
    colour = np.zeros((32, 18, 3), np.uint8)
    colour[...] = (120, 60, 200)
    R.save_image(tmp_path / "x.jpg", colour, quality=95)
    back = R.load_image(tmp_path / "x.jpg")
    assert back.shape == (32, 18, 3) and back.dtype == np.uint8
    assert np.abs(back.astype(int) - colour.astype(int)).max() <= 8      # RGB order kept, lossy codec
    small = R.resize_rgb(colour, (9, 16))
    assert small.shape == (16, 9, 3)
