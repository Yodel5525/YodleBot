"""V11: tools/latency_probe.py (D and the pipeline latency) and tools/vision_replay.py (offline perception).

No device, detector, engine or emulator. The probe reads a vision_play-style session written with
the real VisionRecorder: two battles at 10 frames a second (2 ticks a frame), 8 taps whose units
are first seen 20, 26, 22, 40, 24 and 25 ticks later (D = the median 24.5, rounded half up: 25),
a spell seen 10 ticks after its tap, a tap never seen, and decoys (an enemy unit of the tapped
card, an own one far from the aim, tracks outside a battle). The replay runs the real tracker
and event deriver behind a scripted HUD reader and detector on a recorded npy session, with a
tap replayed from the recording and a labels folder, then the probe reads the replay's output.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "tools"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import latency_probe as LP  # noqa: E402
import vision_replay as VR  # noqa: E402
from scripts.vision import events as EV  # noqa: E402
from scripts.vision.capture import FileReplaySource  # noqa: E402
from scripts.vision.geometry import ArenaGeometry  # noqa: E402
from scripts.vision.labels import load_class_table  # noqa: E402
from scripts.vision.record import VisionLogReader, VisionRecorder  # noqa: E402
from scripts.vision.tracker import Tracker  # noqa: E402
from scripts.vision.types import (ENEMY_TEAM, OWN_TEAM, PHASE_REGULATION, SCREEN_BATTLE,  # noqa: E402
                                  SCREEN_END, SCREEN_MENU, UNKNOWN_TEAM, Detection, Events, Hud,
                                  OwnLanding, Track, VisionFrame)

#: conftest's DEFAULT_CATALOGUE order: Hog 2.6 is 0-7.
NAMES = {0: "Cannon", 1: "Fireball", 2: "HogRider", 3: "IceGolemite", 4: "IceSpirits", 5: "Musketeer",
         6: "Skeletons", 7: "Log", 8: "Knight"}
GEOMETRY = ArenaGeometry.builtin()


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


# ---------------------------------------------------------------------------
# the latency probe
# ---------------------------------------------------------------------------

def t1(tick: int) -> float:
    """Battle 1's capture time of a tick: 10 frames a second, 2 ticks a frame."""
    return 10.0 + (tick - 100) / 20.0


def t2(tick: int) -> float:
    return 30.0 + (tick - 280) / 20.0


#: (uid, label, first_seen_tick, x_mt, y_mt, team): reported on the even ticks first + 2 .. first + 8.
B1_TRACKS = [
    (9, "HogRider", 126, 3500.0, 28000.0, OWN_TEAM),      # ours, but 12.5 tiles from tap A's aim
    (8, "HogRider", 128, 13500.0, 20000.0, ENEMY_TEAM),   # the opponent's Hog, at the aim
    (1, "HogRider", 130, 13500.0, 21000.0, OWN_TEAM),     # tap A: 20 ticks
    (2, "Musketeer", 146, 8500.0, 26000.0, OWN_TEAM),     # tap B: 26 (undecided at first, one "Wizard" frame)
    (3, "Cannon", 152, 9500.0, 24500.0, OWN_TEAM),        # tap C: 22
    (10, "Fireball", 170, 9500.0, 8000.0, OWN_TEAM),      # tap F, a spell: 10
    (5, "Skeleton", 174, 12500.0, 22500.0, OWN_TEAM),     # tap E: 24 ...
    (6, "Skeleton", 176, 12800.0, 22500.0, OWN_TEAM),     # ... the rest of the same play
    (7, "Skeleton", 178, 12200.0, 22500.0, OWN_TEAM),
    (4, "IceGolemite", 180, 4600.0, 18600.0, OWN_TEAM),   # tap D: 40
]
B2_TRACKS = [(1, "HogRider", 325, 13500.0, 21000.0, OWN_TEAM)]   # tap H (uids restart): 25
#: (tick, card, slot, screen tile). The Log at 190 is never seen.
B1_TAPS = [(110, 2, 1, (13, 20)), (120, 5, 0, (8, 26)), (130, 0, 2, (9, 24)), (140, 3, 3, (4, 18)),
           (150, 6, 2, (12, 22)), (160, 1, 1, (9, 6)), (190, 7, 0, (9, 10))]
#: The event deriver's view, in the frame of that tick: E missed, one human play.
B1_LANDINGS = {140: OwnLanding(2, 1, 110, 130), 156: OwnLanding(5, 0, 120, 146), 160: OwnLanding(0, 2, 130, 152),
               188: OwnLanding(3, 3, 140, 180), 200: OwnLanding(6, 2, 150, None), 172: OwnLanding(4, -1, 170, 170)}

CLASSES = LP.CardClasses(
    {"HogRider": {"HogRider"}, "Musketeer": {"Musketeer"}, "Cannon": {"Cannon"}, "IceGolemite": {"IceGolemite"},
     "Skeletons": {"Skeleton"}, "Fireball": {"Fireball", "fireball"}, "Log": {"Log", "the-log"}},
    {"Fireball": "spell", "Log": "spell", "Cannon": "building", "HogRider": "troop", "Musketeer": "troop",
     "IceGolemite": "troop", "Skeletons": "troop"})


def tracks_at(spec, tick):
    out = []
    for uid, label, first, x, y, team in spec:
        if not first + 2 <= tick <= first + 8:
            continue
        if label == "Musketeer" and tick == first + 2:
            team = UNKNOWN_TEAM
        if label == "Musketeer" and tick == first + 4:
            label = "Wizard"
        out.append(Track(uid, label, team, 0.9, x, y, first_seen_tick=first, last_seen_tick=tick))
    return out


def write_vision_play_session(root, name="vision_play_20261030_120000"):
    """What scripts/vision_play.py writes: frames with extra battle / ms / decision, tap records
    (logged 40 ms after the frame they were decided on), control records."""
    clock = FakeClock()
    rec = VisionRecorder(root, session=name, image_every=0, clock=clock, wall_clock=lambda: 0.0)

    def frame(t, tick, battle, tracks=(), landings=(), decided=False, tapped=None, screen=SCREEN_BATTLE):
        ms = {"stale": 10.0, "perceive": 20.0}
        extra = {"battle": battle, "state": "battle" if battle else "idle", "ms": ms}
        if decided:
            ms["decide"] = 5.0
            extra["decision"] = {"tick": tick}
        if tapped is not None:
            card, slot, tile = tapped
            ms.update(tap=1.0, capture_to_tap=40.0)
            clock.t = t + 0.04
            rec.log("tap", battle=battle, tick=tick, card=card, name=NAMES[card], slot=slot,
                    model_tile=[17 - tile[0], tile[1]], screen_tile=list(tile), ms=dict(ms))
        vf = VisionFrame(t_capture=t, tick=tick, tracks=list(tracks), events=Events([], list(landings)),
                         screen=screen, frame_size=(747, 1328))
        rec.write(vf, None, **extra)

    taps = {tick: (card, slot, tile) for tick, card, slot, tile in B1_TAPS}
    for tick in range(100, 201, 2):
        frame(t1(tick), tick, 1, tracks_at(B1_TRACKS, tick), [B1_LANDINGS[tick]] if tick in B1_LANDINGS else [],
              decided=tick in (100, 104) or tick in taps, tapped=taps.get(tick))
    stray = [Track(1, "HogRider", OWN_TEAM, 0.9, 13500.0, 21000.0, first_seen_tick=305, last_seen_tick=307)]
    frame(20.0, None, None, stray, screen=SCREEN_MENU)          # between battles: not counted
    frame(20.1, None, None, stray, screen=SCREEN_MENU)
    for tick in range(280, 341, 2):
        frame(t2(tick), tick, 2, tracks_at(B2_TRACKS, tick),
              [OwnLanding(2, 1, 300, 325)] if tick == 334 else [], decided=tick == 300,
              tapped=(2, 1, (13, 20)) if tick == 300 else None)
    clock.t = 40.0
    rec.log("bot_tap_unseen", card=7, name="Log", slot=0, tap_tick=190, tick=252)
    rec.log("human_play", who="human", card=4, tick=170)
    rec.close()
    return rec.session_dir


def test_describe_and_half_up():
    assert LP.describe([26, 20, None, 40, 22, 24]) == {"n": 5, "mean": 26.4, "median": 24.0, "p10": 20.0,
                                                       "p90": 40.0, "min": 20.0, "max": 40.0}
    ten = LP.describe(range(1, 11))
    assert (ten["p10"], ten["median"], ten["p90"]) == (1.0, 5.5, 9.0)
    assert LP.describe([]) is None and LP.describe([None]) is None
    assert [LP.half_up(x) for x in (24.5, 24.49, 25.5, 10.0)] == [25, 24, 26, 10]


def test_latency_probe_finds_d_and_the_pipeline_latency(tmp_path):
    log = LP.load_session(write_vision_play_session(tmp_path))
    assert sorted(log.battles) == [1, 2] and len(log.taps) == 8 and not log.unassigned
    rep = LP.analyse([log], CLASSES)
    d = rep["D"]
    assert (d["ticks"], d["source"], d["n"], d["enough"]) == (25, "tracks", 6, True)
    assert (d["median"], d["p10"], d["p90"], d["legacy_lead_ticks"]) == (24.5, 20.0, 40.0, 26)

    plays = {(p["battle"], p["tick"]): p for p in rep["plays"]}
    assert [plays[(1, k)]["lag_ticks"] for k in (110, 120, 130, 140, 150, 160, 190)] == [20, 26, 22, 40, 24, 10, None]
    assert plays[(2, 300)]["lag_ticks"] == 25
    assert [plays[(1, k)]["uid"] for k in (110, 120, 150)] == [1, 2, 5]      # not the enemy's or the far one
    assert plays[(1, 120)]["label"] == "Musketeer" and plays[(1, 160)]["kind"] == "spell"
    assert [plays[(1, k)]["deriver"] for k in (110, 150, 160)] == ["landed", "missed", None]
    assert rep["counts"] == {"taps": 8, "seen": 7, "unseen": 1, "deriver_landed": 5, "deriver_missed": 1,
                             "human_plays": 1, "human_play_records": 1, "control_unseen": 1, "unassigned_taps": 0}

    lag = rep["lag_ticks"]
    assert (lag["deriver"]["median"], lag["deriver"]["n"]) == (25.0, 5)
    assert (lag["spells_tracks"]["median"], lag["spells_tracks"]["n"]) == (10.0, 1)
    assert lag["by_card"]["HogRider"]["median"] == 22.5 and lag["by_card"]["IceGolemite"]["n"] == 1
    # wall clock: the sighting's frame time minus the tap (logged 40 ms after its frame's capture)
    assert rep["wall_ms"]["tap_to_sight"]["median"] == pytest.approx(1185.0, abs=1e-6)
    assert rep["wall_ms"]["capture_to_sight"]["median"] == pytest.approx(1225.0, abs=1e-6)
    assert rep["capture_to_sight_ticks"]["median"] == pytest.approx(24.5, abs=1e-6)

    pm = rep["pipeline_ms"]
    assert (pm["stale"]["n"], pm["stale"]["median"], pm["perceive"]["median"]) == (82, 10.0, 20.0)
    assert (pm["capture_to_decision"]["n"], pm["capture_to_decision"]["median"]) == (10, 35.0)
    assert (pm["decision_to_tap"]["n"], pm["decision_to_tap"]["median"]) == (8, 5.0)
    assert (pm["capture_to_tap"]["n"], pm["capture_to_tap"]["max"], pm["tap"]["n"]) == (8, 40.0, 8)
    assert rep["frame_interval_ms"]["n"] == 80
    assert rep["frame_interval_ms"]["median"] == pytest.approx(100.0, abs=1e-6)
    assert (rep["decision_interval_ticks"]["n"], rep["decision_interval_ticks"]["median"]) == (8, 10.0)

    assert LP.analyse([log], CLASSES)["D"] == d               # claims are reset: the same answer again
    text = "\n".join(LP.format_report(rep))
    assert "D = 25 ticks" in text and "--delay 25" in text
    json.dumps(rep)                                            # the --json report is plain JSON


def test_latency_probe_window_and_radius(tmp_path):
    """A tighter window loses tap D (40 ticks); no radius lets the far own Hog (16 ticks) take tap A."""
    log = LP.load_session(write_vision_play_session(tmp_path))
    rep = LP.analyse([log], CLASSES, LP.ProbeConfig(max_lag_ticks=30))
    assert rep["D"]["n"] == 5 and rep["D"]["median"] == 24.0 and rep["D"]["ticks"] == 24
    rep = LP.analyse([log], CLASSES, LP.ProbeConfig(radius_mt=None))
    first = [p for p in rep["plays"] if (p["battle"], p["tick"]) == (1, 110)][0]
    assert (first["uid"], first["lag_ticks"]) == (9, 16)
    with pytest.raises(ValueError):
        LP.ProbeConfig(min_lag_ticks=5, max_lag_ticks=0)


def test_card_classes_from_the_class_list():
    cc = LP.CardClasses.load()
    assert "HogRider" in cc.labels("HogRider") and "Skeleton" in cc.labels("Skeletons")
    assert {"Fireball", "fireball"} <= cc.labels("Fireball") and cc.kind("Fireball") == "spell"
    assert cc.kind("Cannon") == "building" and cc.kind("HogRider") == "troop"
    assert cc.labels("NoSuchCard") == frozenset() and cc.kind("NoSuchCard") is None and cc.labels(None) == frozenset()


def test_latency_probe_command_line(tmp_path, capsys):
    root = tmp_path / "vision_play"
    VisionRecorder(root, session="vision_play_20261029_120000", image_every=0).close()   # an older dry run
    write_vision_play_session(root)
    out = tmp_path / "latency.json"
    assert LP.main([str(root), "--last", "1", "--json", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["D"]["ticks"] == 25 and len(rep["sessions"]) == 1
    assert Path(rep["sessions"][0]).name == "vision_play_20261030_120000"
    assert "D = 25 ticks" in capsys.readouterr().out
    assert LP.main([str(tmp_path / "nothing")]) == 1


# ---------------------------------------------------------------------------
# replay: names, labels, matching
# ---------------------------------------------------------------------------

def test_label_names_files(tmp_path):
    assert VR.parse_yaml_names('path: x\nnames:\n  0: HogRider@ally\n  1: "odd name"\ntrain: a\n') == {
        0: "HogRider@ally", 1: "odd name"}
    assert VR.parse_yaml_names("names: ['a', 'b']\n") == {0: "a", 1: "b"}
    assert VR.parse_yaml_names("names:\n  - a\n  - 'b c'\nnc: 2\n") == {0: "a", 1: "b c"}
    assert VR.parse_yaml_names("nc: 2\n") == {}
    (tmp_path / "detector_labels.json").write_text(json.dumps({"yolo_names": ["HogRider@ally", "HogRider@enemy"]}),
                                                   encoding="utf-8")
    (tmp_path / "classes.txt").write_text("knight\n# a comment\n\nfireball\n", encoding="utf-8")
    assert VR.find_label_names(tmp_path) == tmp_path / "detector_labels.json"
    assert VR.read_label_names(tmp_path / "detector_labels.json") == {0: "HogRider@ally", 1: "HogRider@enemy"}
    assert VR.read_label_names(tmp_path / "classes.txt") == {0: "knight", 1: "fireball"}
    assert VR.find_label_names(tmp_path / "missing") is None


def box(cls, b, side=None, conf=1.0):
    return VR.Box(cls, b, side, conf)


def test_match_boxes_and_precision_recall():
    truth = [box("HogRider", (0.1, 0.1, 0.2, 0.2), 1), box("HogRider", (0.7, 0.7, 0.8, 0.8), 0),
             box("Musketeer", (0.5, 0.5, 0.6, 0.6), 0)]
    preds = [box("HogRider", (0.1, 0.11, 0.2, 0.2), 1, 0.8),     # the same Hog again: a duplicate
             box("HogRider", (0.1, 0.1, 0.2, 0.19), 1, 0.9),     # the stronger box takes the label
             box("Musketeer", (0.5, 0.5, 0.6, 0.575), 1, 0.7),   # IoU 0.75, the wrong side
             box("Skeleton", (0.3, 0.3, 0.35, 0.35), 0, 0.6)]    # nothing there
    tp, fp, fn = VR.match_boxes(truth, preds)
    assert [(g.box, p.conf) for g, p in tp] == [((0.1, 0.1, 0.2, 0.2), 0.9), ((0.5, 0.5, 0.6, 0.6), 0.7)]
    assert [p.conf for p in fp] == [0.8, 0.6] and [g.box for g in fn] == [(0.7, 0.7, 0.8, 0.8)]
    assert VR.match_boxes(truth, preds, iou_min=0.95)[0] == []

    pr = VR.PRCounter(0.5)
    assert pr.add(truth, preds) == {"tp": 2, "fp": 2, "fn": 1}
    per = pr.per_class()
    assert per["HogRider"] == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "side_n": 1,
                               "side_accuracy": 1.0}
    assert (per["Musketeer"]["recall"], per["Musketeer"]["side_accuracy"]) == (1.0, 0.0)
    assert (per["Skeleton"]["precision"], per["Skeleton"]["recall"]) == (0.0, None)
    o = pr.overall()
    assert (o["tp"], o["fp"], o["fn"], o["precision"], o["side_n"], o["side_accuracy"]) == (2, 2, 1, 0.5, 2, 0.5)
    assert o["recall"] == pytest.approx(2 / 3)
    assert [VR.side_of(p) for p in (0.9, 0.1, 0.5)] == [1, 0, None]


def test_label_files_through_the_crosswalk(tmp_path):
    table = load_class_table()
    names = {0: "hog-rider", 1: "HogRider@enemy", 2: "Musketeer@ally", 3: "bar", 4: "fireball", 5: "no-such-thing"}
    sub = tmp_path / "labels" / "part1"
    sub.mkdir(parents=True)
    (sub / "00000005.txt").write_text(
        "0 0.15 0.15 0.1 0.1 1\n1 0.55 0.55 0.1 0.1\n2 0.5 0.5 0.2 0.1 1\n3 0.5 0.5 0.05 0.01\n"
        "4 0.3 0.3 0.1 0.1\n5 0.9 0.9 0.1 0.1\ngarbage\n0 0.1\n", encoding="utf-8")
    (tmp_path / "labels" / "classes.txt").write_text("x\n", encoding="utf-8")    # a names file, not labels
    idx = VR.LabelIndex(tmp_path / "labels", names, table)
    assert len(idx) == 1 and idx.truth_for(None) is None and idx.truth_for("images/00000006.npy") is None
    got = idx.truth_for("images/00000005.npy")
    # KataCR side column 1 = enemy; a side in the name wins over the column; HP bars and names
    # the class list cannot place are not scored by default
    assert [(b.cls, b.side, b.type) for b in got] == [("HogRider", 1, "unit"), ("HogRider", 1, "unit"),
                                                      ("Musketeer", 0, "unit"), ("Fireball", None, "spell")]
    assert got[0].box == pytest.approx((0.1, 0.1, 0.2, 0.2))
    assert idx.bad_lines == 2 and dict(idx.unknown) == {"no-such-thing": 1} and idx.used == {"00000005"}
    every = VR.LabelIndex(tmp_path / "labels", names, table, types=None)
    assert [b.type for b in every.truth_for("00000005.jpg")] == ["unit", "unit", "unit", "ui", "spell", "unknown"]


# ---------------------------------------------------------------------------
# replay: perception on a recording
# ---------------------------------------------------------------------------

SIZE = (225, 400)


def _fill(img, b, rgb):
    h, w = img.shape[:2]
    img[int(b[1] * h):int(math.ceil(b[3] * h)), int(b[0] * w):int(math.ceil(b[2] * w))] = rgb


def battle_image(k):
    """The bottom panel's blue top edge (the owner's colour); pixel (0, 0) carries the frame
    number k for the scripted HUD reader and detector (no screen cue reads it)."""
    img = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    _fill(img, (0.0, 0.805, 1.0, 0.83), (30, 144, 230))
    img[0, 0, 0] = k
    return img


def end_image():
    """The end screen's red and blue banners and its OK button (the owner's colours)."""
    img = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    _fill(img, (0.08, 0.24, 0.92, 0.34), (157, 43, 68))
    _fill(img, (0.08, 0.54, 0.92, 0.63), (64, 110, 169))
    _fill(img, (0.37, 0.875, 0.63, 0.94), (100, 178, 241))
    return img


def det_at(cls, x, y, p_enemy, size=SIZE, half_w=10.0, height=30.0):
    gx, gy = (float(v) for v in GEOMETRY.world_to_px(x, y, size))
    return Detection(cls, p_enemy, 0.9, (gx - half_w, gy - height, gx + half_w, gy), (gx, gy))


KNIGHT = det_at("Knight", 9000.0, 10000.0, 0.9)     # the opponent's half; not a Hog 2.6 class
HOG = det_at("HogRider", 13500.0, 22500.0, 0.1)     # at the replayed tap's aim, screen tile (13, 22)


class ReplayHud:
    """Battle frame k reads tick 100 + 2k (frame 25 misreads 10 ticks low), elixir 5000 (6000 on
    frame 26), the hand Log / Hog / Skeletons / Ice Spirits, next Cannon."""

    def __init__(self):
        self.reads = []
        self.resets = 0

    def read(self, rgb, t):
        k = int(rgb[0, 0, 0])
        self.reads.append(k)
        return Hud(hand=[7, 2, 6, 4], next=0, elixir_milli=6000 if k == 26 else 5000,
                   clock_tick=100 + 2 * k - (10 if k == 25 else 0), phase=PHASE_REGULATION)

    def reset(self):
        self.resets += 1


class ReplayDetector:
    """The Knight from battle frame 12 on, our Hog from frame 20 on."""

    def __init__(self):
        self.calls = 0

    def detect(self, rgb):
        self.calls += 1
        k = int(rgb[0, 0, 0])
        dets = ([KNIGHT] if k >= 12 else []) + ([HOG] if k >= 20 else [])
        return SimpleNamespace(detections=dets, frame_size=SIZE, units=lambda: list(dets), spells=lambda: [],
                               bars=lambda: [], dropped=[], suppressed=0, raw_boxes=len(dets), ms=1.0)


def write_recording(root):
    """A vision_play-style recording with an npy image per frame: 2 menu frames, 30 battle frames
    (k = 0..29 at t 2.0 + 0.1k, live ticks 100 + 2k), 4 end-screen frames; a tap of the Hog at
    tick 130 (decided on frame 15, logged at 3.56 with capture_to_tap 50 ms). Images are named by
    the recorder's frame index: battle frame k is images/<2 + k>.npy."""
    clock = FakeClock()
    rec = VisionRecorder(root, session="vision_play_s", image_every=1, image_format="npy", clock=clock,
                         wall_clock=lambda: 0.0)
    rec.log("config", deck=[NAMES[i] for i in range(8)], deck_ids=list(range(8)))
    for t in (1.0, 1.1):
        rec.write(VisionFrame(t_capture=t, screen=SCREEN_MENU, frame_size=SIZE), np.zeros((400, 225, 3), np.uint8))
    for k in range(30):
        hud = Hud(hand=[7, 2, 6, 4], next=0, elixir_milli=5000, clock_tick=100 + 2 * k, phase=PHASE_REGULATION)
        rec.write(VisionFrame(t_capture=round(2.0 + 0.1 * k, 4), tick=100 + 2 * k, hud=hud, screen=SCREEN_BATTLE,
                              frame_size=SIZE), battle_image(k))
        if k == 15:
            clock.t = 3.56
            rec.log("tap", battle=1, tick=130, card=2, name="HogRider", slot=1, model_tile=[4, 22],
                    screen_tile=[13, 22], ms={"capture_to_tap": 50.0})
    for j in range(4):
        rec.write(VisionFrame(t_capture=round(5.0 + 0.1 * j, 4), screen=SCREEN_END, frame_size=SIZE), end_image())
    rec.close()
    return rec.session_dir


def yolo_line(k, det, side=None):
    x1, y1, x2, y2 = det.box
    cx, cy, w, h = (x1 + x2) / 2 / SIZE[0], (y1 + y2) / 2 / SIZE[1], (x2 - x1) / SIZE[0], (y2 - y1) / SIZE[1]
    return f"{k} {cx!r} {cy!r} {w!r} {h!r}" + ("" if side is None else f" {side}")


def write_labels(folder):
    folder.mkdir()
    (folder / "classes.txt").write_text("HogRider@ally\nknight\nMusketeer@enemy\n", encoding="utf-8")
    (folder / "00000023.txt").write_text(            # battle frame 21: the Hog, the Knight, a missed Musketeer
        yolo_line(0, HOG) + "\n" + yolo_line(1, KNIGHT, 1) + "\n2 0.5 0.3 0.05 0.05\n", encoding="utf-8")
    (folder / "00000005.txt").write_text("", encoding="utf-8")                       # frame 3: nothing there
    (folder / "00000000.txt").write_text("1 0.5 0.5 0.1 0.1 1\n", encoding="utf-8")   # a menu frame: not scored
    return folder


def test_vision_replay_reruns_perception_and_feeds_the_latency_probe(tmp_path):
    session = write_recording(tmp_path / "rec")
    table = load_class_table()
    names = VR.Names(table)
    tables = EV.EventTables({2: {"HogRider": 1}, 8: {"Knight": 1}})
    hud, detector = ReplayHud(), ReplayDetector()
    perception = VR.ReplayPerception(hud_reader=hud, tracker=Tracker(own_classes={"HogRider"}),
                                     deriver=EV.EventDeriver(tables, own_deck=range(8)), geometry=GEOMETRY,
                                     detector=detector, names=names)
    taps, config = VR.read_recording_records(session)
    assert len(taps) == 1 and config["deck_ids"] == list(range(8))
    label_dir = write_labels(tmp_path / "labels")
    labels = VR.LabelIndex(label_dir, VR.read_label_names(VR.find_label_names(label_dir)), table, canon=names)
    replay = VR.Replay(FileReplaySource(session), perception, taps=taps, labels=labels, names=names,
                       card_ids={n: i for i, n in NAMES.items()}, card_names=NAMES, live_names=NAMES)
    replay.recorder = VisionRecorder(tmp_path / "out", session="r", image_every=0, clock=replay.log_clock,
                                     wall_clock=lambda: 0.0)
    s = replay.run()
    replay.recorder.close()

    # the battle starts on frame 10 (the clock ran a second) and ends on the third end-screen frame
    assert (s["frames"], s["screens"], s["battles"], s["errors"]) == (36, {"menu": 2, "battle": 30, "end": 4}, 1, {})
    assert (s["ticks"], s["tick_sources"], s["sparse"]) == ("clock", {"clock": 30}, False)
    assert s["image_interval_s"]["median"] == pytest.approx(0.1)
    assert hud.reads == list(range(30)) and hud.resets == 1 and detector.calls == 30

    h = s["hud"]
    assert (h["frames"], h["clock_read"], h["elixir_read"], h["hand_full"], h["next_read"]) == (30, 1.0, 1.0, 1.0, 1.0)
    assert h["agreement"] == {"clock": {"agree": 29, "n": 30, "rate": 0.9667},
                              "elixir": {"agree": 29, "n": 30, "rate": 0.9667},
                              "hand": {"agree": 30, "n": 30, "rate": 1.0}, "next": {"agree": 30, "n": 30, "rate": 1.0}}
    d = s["detections"]
    assert (d["frames"], d["by_type"], d["top_classes"], d["raw_boxes"]) == (
        30, {"unit": 28}, {"Knight": 18, "HogRider": 10}, 28)
    assert (d["per_frame"]["max"], d["ms"]["n"], d["dropped"], d["suppressed"]) == (2, 30, 0, 0)

    tr = s["tracker"]
    assert (tr["tracks"], tr["team"], tr["short"], tr["label_switched"]) == (
        2, {"own": 1, "enemy": 1, "unknown": 0}, 0, 0)
    assert (tr["lifetime_ticks"]["min"], tr["lifetime_ticks"]["max"]) == (18.0, 34.0)   # Hog 140-158, Knight 124-158
    assert (tr["per_frame"]["n"], tr["per_frame"]["median"], tr["ticks_to_team"]["median"]) == (30, 1.0, 2.0)

    e = s["events"]
    assert (e["opp_plays"], e["opp_cards"], e["own_landed"], e["own_missed"], e["own_unmatched"]) == (
        1, {"Knight": 1}, 1, 0, 0)
    assert e["landing_stats"]["landed"] == 1 and e["landing_stats"]["lag_ticks"]["median"] == 10
    assert s["taps"] == {"replayed": 1, "skipped": 0, "left": 0}

    pr = s["precision_recall"]
    assert (pr["frames"], pr["labelled_not_scored"], pr["label_files"], pr["label_files_used"]) == (2, 1, 3, 3)
    o = pr["overall"]
    assert (o["tp"], o["fp"], o["fn"], o["precision"], o["side_n"], o["side_accuracy"]) == (2, 0, 1, 1.0, 2, 1.0)
    assert o["recall"] == pytest.approx(2 / 3)
    assert sorted(pr["per_class"]) == ["HogRider", "Knight", "Musketeer"] and pr["per_class"]["Musketeer"]["fn"] == 1
    assert any(line.startswith("precision / recall on 2 labelled frames") for line in VR.format_summary(s))
    json.dumps(s)

    # the replay's own session: every image, the replayed tap, readable by the latency probe
    out = tmp_path / "out" / "r"
    frames = list(VisionLogReader(out).frames())
    assert len(frames) == 36
    by_image = {f.data["extra"]["replay"]["image"]: f.data["extra"] for f in frames}
    assert by_image["00000023.npy"]["replay"]["pr"] == {"tp": 2, "fp": 0, "fn": 1}
    assert by_image["00000023.npy"]["battle"] == 1 and by_image["00000001.npy"]["battle"] is None
    tap = [x.data for x in VisionLogReader(out).entries("tap")]
    assert len(tap) == 1 and tap[0]["t"] == 3.56
    assert (tap[0]["data"]["card"], tap[0]["data"]["tick"], tap[0]["data"]["battle"], tap[0]["data"]["replayed"]) == (
        2, 130, 1, True)
    rep = LP.analyse([LP.load_session(out)], LP.CardClasses.load())
    assert (rep["D"]["ticks"], rep["D"]["n"], rep["D"]["enough"], rep["D"]["source"]) == (10, 1, False, "tracks")
    assert rep["counts"]["deriver_landed"] == 1
    assert rep["wall_ms"]["tap_to_sight"]["median"] == pytest.approx(440.0, abs=1e-6)   # frame 20 at 4.0 s


def test_replay_ticks_and_scaled_detections():
    p = VR.ReplayPerception(hud_reader=None, tracker=None, deriver=None, geometry=GEOMETRY, ticks=VR.TICKS_TIME)
    p.start_clock(1.0)
    assert p.tick_at(1.5) == (10, "time") and p.tap_tick(999, 2.0) == 20
    c = VR.ReplayPerception(hud_reader=None, tracker=None, deriver=None, geometry=GEOMETRY)
    assert c.tick_at(1.0) == (None, None)
    assert c.tick_at(1.0, 100) == (100, "clock") and c.tick_at(2.0) == (120, "extrapolated")
    assert c.tick_at(7.5) == (None, None)
    assert c.tap_tick(130, 9.0) == 130 and c.tap_tick(None, 9.0) is None
    with pytest.raises(ValueError):
        VR.ReplayPerception(hud_reader=None, tracker=None, deriver=None, geometry=GEOMETRY, ticks="wall")
    d = Detection("Knight", 0.9, 0.8, (100.0, 200.0, 140.0, 260.0), (120.0, 260.0))
    half = VR.scale_detections([d], (450, 800), SIZE)[0]
    assert (half.cls, half.box, half.ground_px, half.side_prob) == ("Knight", (50.0, 100.0, 70.0, 130.0),
                                                                    (60.0, 130.0), 0.9)
    assert VR.scale_detections([d], SIZE, SIZE) == [d] and VR.scale_detections([d], None, SIZE) == [d]


def test_replay_perception_can_track_the_recorded_detections():
    """--recorded: the recording's detections (made on 450x800 frames) are moved onto the
    225x400 images and tracked; an image without its recorded frame has nothing to score."""
    knight = det_at("Knight", 9000.0, 10000.0, 0.9, size=(450, 800), half_w=20.0, height=60.0)
    recorded = VisionFrame(t_capture=2.0, tick=100, detections=[knight], frame_size=(450, 800), screen=SCREEN_BATTLE)
    p = VR.ReplayPerception(hud_reader=ReplayHud(), tracker=Tracker(own_classes={"HogRider"}),
                            deriver=EV.EventDeriver(EV.EventTables({8: {"Knight": 1}})), geometry=GEOMETRY,
                            names=VR.Names(load_class_table()), recorded=True)
    a = p.step(battle_image(0), 2.0, recorded=recorded)
    b = p.step(battle_image(1), 2.1, recorded=recorded)
    gx, gy = knight.ground_px
    assert a.det_source == "recorded" and a.det is None
    assert a.vf.detections[0].box == pytest.approx((gx / 2 - 10.0, gy / 2 - 30.0, gx / 2 + 10.0, gy / 2))
    assert [(t.cls, t.team, t.first_seen_tick) for t in b.vf.tracks] == [("Knight", ENEMY_TEAM, 100)]
    assert abs(b.vf.tracks[0].x_mt - 9000.0) < 1.0 and abs(b.vf.tracks[0].y_mt - 10000.0) < 1.0
    assert p.step(battle_image(2), 2.2).det_source is None


def test_vision_replay_command_line_refuses_a_missing_recording(tmp_path, capsys):
    assert VR.main([str(tmp_path / "nothing")]) == 2
    assert "nothing to replay" in capsys.readouterr().out
