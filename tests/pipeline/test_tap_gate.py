"""The tap gate: no AUTO tap until this battle's towers are calibrated and its mode is classic
(scripts/vision/tap_gate.py, skin audit fix list P0-4 and P1-12).

Before the gate, nothing stopped the bot from tapping in a battle it could not read: the refused-tap
guard reacts only after 3 taps that did nothing. Now every battle's AUTO taps wait until all four
princess labels are calibrated (tower_calib), the detector boxes a tower body at each of the six
tower positions on half the calibration frames, and the mode check (mode.py) says classic. When a
check fails the runner prints one line and switches AUTO to MANUAL (mode source "towers"); the
owner's F8 overrides that for the rest of the battle; a dry run only says what the gate would do.

The runner runs on test_vision_play's fakes (a scripted perception, make_decider's accept-all
decider, a FakeTapper, a real ControlState and VisionRecorder). Two battles run through the real
HudReader on tests/pipeline/skin_synth.py's synthetic frames (no real frame is used): one with the
enemy labels outside the calibration window, one with the chicken skin's labels 36 px up.
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import vision_play as VP  # noqa: E402
from scripts.vision import control as CT  # noqa: E402
from scripts.vision import hud as H  # noqa: E402
from scripts.vision import mode as M  # noqa: E402
from scripts.vision import tap_gate as TG  # noqa: E402
from scripts.vision import tower_labels as TL  # noqa: E402
from scripts.vision.capture import Frame  # noqa: E402
from scripts.vision.record import VisionRecorder  # noqa: E402
from scripts.vision.types import PHASE_REGULATION, SCREEN_BATTLE, Detection, Events, Hud  # noqa: E402
from tests.pipeline import skin_synth as S  # noqa: E402
from tests.pipeline import test_vision_play as TVP  # noqa: E402

SIZE = S.OWNER_SIZE
PRINCESS = TL.PRINCESS_KEYS
t_of = TVP.t_of
CLOSED_LINE = "tapping off this battle; play yourself"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def body(key: str, *, cls: str | None = None, size=SIZE, dx: float = 0.0, dy: float = 0.0,
         top: float = 0.0) -> Detection:
    """A detector body box at `key`'s reference body (px at 1080), moved by (dx, dy) and with its
    top `top` px higher (a taller tower skin), in pixels of `size`."""
    x0, y0, x1, y1 = TL.TOWER_BODY_REF_PX[key]
    cls = cls or ("KingTower" if key.endswith("king") else "PrincessTower")
    sx, sy = size[0] / 1080.0, size[1] / 1920.0
    return Detection(cls, 0.5, 0.9, ((x0 + dx) * sx, (y0 + dy - top) * sy, (x1 + dx) * sx, (y1 + dy) * sy))


def bodies(keys=TG.BODY_KEYS, **kw) -> list[Detection]:
    return [body(k, **kw) for k in keys]


def elixir_at(tick: int, period: float = M.REGEN_S_1X, e0: float = 5.0) -> float:
    """Our elixir at `tick` of a battle that started with e0 and fills one per `period` s."""
    return min(10.0, e0 + tick / 20.0 / period)


def locked(*, expired=(), searching=()) -> dict[str, tuple[str, bool]]:
    out = {k: ("locked", True) for k in PRINCESS}
    out.update({k: ("expired", False) for k in expired})
    out.update({k: ("searching", False) for k in searching})
    return out


def gate_script(last_tick: int = 300, *, first_tick: int = 0, battles: int = 1, ticks=None):
    """Battle frames from `first_tick` to `last_tick` (2 ticks a frame, 0.1 s apart: the battle
    clock's 20 ticks a second), then the end screen and the menu, `battles` times. ticks: the
    battle's ticks instead (one battle)."""
    items = [TVP.menu_vf(t_of(0)), TVP.menu_vf(t_of(1))]
    for _ in range(battles):
        for tick in (ticks if ticks is not None else range(first_tick, last_tick + 1, 2)):
            items.append(TVP.battle_vf(t_of(len(items)), tick))
        items += [TVP.end_vf(t_of(len(items)), tick=last_tick + 2) for _ in range(5)]
        items += [TVP.menu_vf(t_of(len(items))) for _ in range(3)]
    return [(t_of(i), item) for i, item in enumerate(items)]


def frame_at(script, tick: int, battle: int = 1) -> int:
    """The index of the `battle`-th frame with `tick`."""
    seen = 0
    for i, (_, vf) in enumerate(script):
        if vf.screen == SCREEN_BATTLE and vf.tick == tick:
            seen += 1
            if seen == battle:
                return i
    raise KeyError(tick)


class GateHud:
    """The HUD reader as the gate sees it: calibrate, calib.towers[key].status / .lock, and a
    report (t, elixir_read: the count digit and the bar) for the frame. The princess labels lock
    `lock_at` s after the battle screen came (None: never); `expired` ones expire at `expire_at`."""

    calibrate = True

    def __init__(self, *, lock_at: float | None = 0.5, expired=(), expire_at: float = 8.0,
                 period: float = M.REGEN_S_1X):
        self.lock_at, self.expired, self.expire_at, self.period = lock_at, tuple(expired), expire_at, period
        self.reset()

    def reset(self) -> None:
        self.calib = SimpleNamespace(towers={k: SimpleNamespace(status="searching", lock=None) for k in PRINCESS})
        self.report = None

    def see(self, t: float, tick: int | None, since: float) -> None:
        for k, st in self.calib.towers.items():
            if k in self.expired:
                if since >= self.expire_at:
                    st.status = "expired"
            elif self.lock_at is not None and since >= self.lock_at and st.lock is None:
                st.status, st.lock = "locked", object()
        e = elixir_at(tick or 0, self.period)
        self.report = SimpleNamespace(t=t, elixir_read=SimpleNamespace(digit=int(math.floor(e)), bar_frac=e / 10.0))


class GatePerception(TVP.FakePerception):
    """FakePerception with a GateHud, a detector (boxes(tick) -> its boxes on battle frames, or
    None: no detector), and LivePerception's kept-towers / joined-battle attributes."""

    def __init__(self, script, hud: GateHud | None = None, *, boxes=lambda tick: bodies(), detector: bool = True):
        super().__init__(script)
        self.hud_reader = hud if hud is not None else GateHud()
        self.detector = object() if detector else None
        self.boxes = boxes
        self.towers_kept = False
        self.primed_until = None
        self.last_tick = None
        self._t_first = None

    def perceive(self, frame):
        vf = super().perceive(frame)
        if vf.screen == SCREEN_BATTLE:
            if self._t_first is None:
                self._t_first = vf.t_capture
            self.hud_reader.see(vf.t_capture, vf.tick, vf.t_capture - self._t_first)
            vf.frame_size = SIZE                       # the boxes' pixels
            if self.detector is not None:
                vf.detections = list(self.boxes(vf.tick))
            self.last_tick = vf.tick
        return vf

    def new_battle(self):
        super().new_battle()
        joined = self.last_tick is not None and self.last_tick > VP.CLEAN_START_TICK
        self.primed_until = self.last_tick + VP.PRIME_TICKS if joined else None
        if self.towers_kept and not joined:
            self.hud_reader.reset()
            self._t_first = None
        self.towers_kept = False

    def end_battle(self, *, keep_towers=False):
        super().end_battle(keep_towers=keep_towers)
        self.towers_kept = bool(keep_towers)
        if not keep_towers:
            self.hud_reader.reset()
            self._t_first = None


def run_gate(tmp_path, script, *, perception=None, gate=True, mode=CT.AUTO, taps=True, play_ticks=(100,),
             on_read=None, control=None, size=(225, 400)):
    """test_vision_play.run_script with a TapGate (gate=True: a new one; a TapGate; None: none)."""
    clock = TVP.FakeClock()
    rgb = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frames = [Frame(t, rgb, 0.0) for t, _ in script]
    perception = GatePerception(script) if perception is None else perception
    ctl = CT.ControlState(mode, taps_enabled=taps, stop_file=tmp_path / "STOP", control_file=None,
                          names=TVP.NAMES, clock=clock)
    if control is not None:
        control["state"] = ctl
    recorder = VisionRecorder(tmp_path / "rec", session="s", image_every=0, image_format="npy",
                              clock=clock, wall_clock=lambda: 0.0)
    tapper = TVP.FakeTapper()
    lines: list[str] = []
    tap_gate = TG.TapGate() if gate is True else (gate or None)
    runner = VP.VisionPlay(source=TVP.FakeSource(frames, clock, on_read), perception=perception,
                           decider=TVP.make_decider(TVP.FakePolicy(play_ticks)), control=ctl, recorder=recorder,
                           geometry=TVP.GEOMETRY, tapper=tapper, card_names=TVP.NAMES, clock=clock,
                           sleep=lambda s: None, out=lines.append, tap_gate=tap_gate)
    rc = runner.run()
    recorder.close()
    return SimpleNamespace(rc=rc, runner=runner, tapper=tapper, control=ctl, lines=lines, gate=tap_gate,
                           records=TVP.read_records(tmp_path / "rec" / "s"), perception=perception)


def gate_lines(res) -> list[str]:
    return [x for x in res.lines if CLOSED_LINE in x or "tower gate" in x]


def play_actions(res) -> list[tuple[int, str, str]]:
    """(tick, action, why) of every play decision on the frame lines."""
    out = []
    for f in res.records["frame"]:
        e = f["extra"]
        d = e.get("decision") or {}
        if d.get("kind") == VP.RD.PLAY:
            out.append((d["tick"], e["action"], ""))
    whys = {d["tick"]: d.get("why", "") for d in TVP.decisions(res.records) if d["card"] >= 0}
    return [(tick, action, whys.get(tick, "")) for tick, action, _ in out]


# ---------------------------------------------------------------------------
# where the towers are
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", S.SIZES)
def test_a_body_box_counts_within_half_a_tile_of_its_tower(size):
    """The six reference bodies are found; a box 0.4 tile off is still its tower's, one 0.6 tile
    off is not; a taller tower skin (the box's top 40 px higher) stands on the same ground; a unit
    class at the tower is no body."""
    assert TG.body_positions(bodies(size=size), size) == set(TG.BODY_KEYS)
    tile_x, tile_y = TG.TILE_PX
    assert TG.body_positions([body("enemy_left", dx=0.4 * tile_x, size=size)], size) == {"enemy_left"}
    assert TG.body_positions([body("enemy_left", dx=0.6 * tile_x, size=size)], size) == set()
    assert TG.body_positions([body("own_king", dy=0.6 * tile_y, size=size)], size) == set()
    assert TG.body_positions([body("enemy_right", top=40.0, size=size, cls="Cannoneer")], size) == {"enemy_right"}
    assert TG.body_positions([body("own_left", cls="Knight", size=size)], size) == set()
    assert TG.body_positions(bodies(), None) == set()


def test_the_calibration_states_come_from_the_hud_reader():
    hud = GateHud(lock_at=None)
    assert TG.calib_status(hud) == {k: ("searching", False) for k in PRINCESS}
    hud.see(1.0, 20, 1.0)
    assert TG.calib_status(SimpleNamespace(calibrate=False, calib=hud.calib)) is None
    assert TG.calib_status(None) is None
    real = H.HudReader(S.digit_store(SIZE), {}, regions=S.REGIONS)
    assert TG.calib_status(real) == {k: ("waiting", False) for k in PRINCESS}


# ---------------------------------------------------------------------------
# the towers check
# ---------------------------------------------------------------------------
def test_the_towers_pass_once_calibrated_and_boxed_on_enough_frames():
    tc = TG.TowerCheck()
    tc.start(0.0)
    assert tc.note(0.1, locked(searching=["enemy_left"]), detector=True, detections=bodies(), frame_size=SIZE) is None
    assert tc.note(0.2, locked(), detector=True, detections=bodies(), frame_size=SIZE) is None   # 2 frames
    assert tc.note(0.3, locked(), detector=True, detections=bodies(), frame_size=SIZE) == (True, [])
    assert tc.summary()["bodies"] == {k: 1.0 for k in TG.BODY_KEYS} and tc.summary()["det_frames"] == 3


def test_a_tower_whose_calibration_expired_fails():
    tc = TG.TowerCheck()
    tc.start(0.0)
    assert tc.note(8.1, locked(expired=["enemy_right", "enemy_left"]), detector=False) == (
        False, ["towers not calibrated (enemy_left, enemy_right)"])


def test_a_tower_the_detector_boxes_on_too_few_frames_fails():
    """The enemy king boxed on 1 of the first 3 calibration frames: the board is not as expected."""
    tc = TG.TowerCheck()
    tc.start(0.0)
    tc.note(0.1, locked(), detector=True, detections=bodies(), frame_size=SIZE)
    tc.note(0.2, locked(), detector=True, detections=bodies([k for k in TG.BODY_KEYS if k != "enemy_king"]),
            frame_size=SIZE)
    assert tc.note(0.3, locked(), detector=True, detections=bodies([k for k in TG.BODY_KEYS if k != "enemy_king"]),
                   frame_size=SIZE) == (False, ["towers not boxed by the detector (enemy_king 33% of 3 frames)"])


def test_without_a_detector_only_the_labels_count():
    tc = TG.TowerCheck()
    tc.start(0.0)
    assert tc.note(0.1, locked(), detector=False) == (True, [])


def test_labels_still_searching_at_the_deadline_fail_and_so_does_a_silent_detector():
    tc = TG.TowerCheck()
    tc.start(0.0)
    assert tc.note(19.9, locked(searching=["own_left"]), detector=True, detections=None) is None
    assert tc.note(20.0, locked(searching=["own_left"]), detector=True, detections=None) == (
        False, ["towers not calibrated (own_left)"])
    tc = TG.TowerCheck()
    tc.start(0.0)
    assert tc.note(1.0, locked(), detector=True, detections=None) is None
    assert tc.note(20.0, locked(), detector=True, detections=None) == (False, ["the detector boxed no tower"])
    tc = TG.TowerCheck()
    tc.start(0.0)
    assert tc.note(0.1, None) == (False, ["towers not calibrated (no label calibration)"])


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def feed(gate, *, seconds=4.0, first_tick=30, calib=None, period=M.REGEN_S_1X, detector=True, boxes=None):
    """Battle frames at 10 a second into a fresh gate; returns every record it gave."""
    gate.new_battle(1, first_tick, 0.0)
    out = []
    for i in range(int(seconds * 10) + 1):
        t = i / 10.0
        tick = first_tick + 2 * i
        e = elixir_at(tick, period)
        out += gate.observe(t, tick, calib=locked() if calib is None else calib(t), detector=detector,
                            detections=bodies() if boxes is None else boxes, frame_size=SIZE,
                            elixir=(int(e), e / 10.0))
    return out


def test_the_gate_opens_only_when_the_towers_and_the_mode_pass():
    g = TG.TapGate()
    g.new_battle(1, 30, 0.0)
    assert g.state == TG.PENDING and not g.allows()
    assert g.hold_reason() == "tap gate: towers and mode not checked yet" and g.status() == "gate wait"
    recs = feed(g)
    kinds = [k for k, _ in recs]
    assert kinds == ["game_mode", "tap_gate"] or kinds == ["tap_gate", "game_mode"][::-1]
    mode = dict(recs)["game_mode"]
    assert mode["mode"] == "classic" and mode["first_clock"] == "2:59"
    rec = dict(recs)["tap_gate"]
    assert rec["state"] == TG.OPEN and rec["reasons"] == [] and rec["towers"]["ok"] is True
    assert g.allows() and g.hold_reason() is None and g.status() is None
    assert 1.5 <= rec["decided_s"] <= 3.0                        # the mode check: the first count change


def test_the_gate_closes_on_any_failed_check_and_keeps_what_failed_later():
    g = TG.TapGate()
    recs = feed(g, calib=lambda t: locked(expired=["enemy_left"]) if t >= 1.0 else locked(searching=["enemy_left"]),
                period=M.REGEN_S_2X, seconds=6.0)          # 2x: "not classic" holds 1.5 s first
    gate = dict(recs)["tap_gate"]
    assert gate["state"] == TG.CLOSED
    assert g.reasons[0] in ("towers not calibrated (enemy_left)", "mode unknown (regen 1.4 s)")
    assert set(g.reasons) == {"towers not calibrated (enemy_left)", "mode unknown (regen 1.4 s)"}
    assert g.source() == TG.SOURCE_TOWERS and g.status() == "gate OFF" and not g.allows()
    g = TG.TapGate()
    feed(g, period=M.REGEN_S_2X, seconds=6.0)
    assert g.reasons == ["mode unknown (regen 1.4 s)"] and g.source() == TG.SOURCE_MODE
    g = TG.TapGate()
    recs = feed(g, first_tick=1210, seconds=0.0)
    assert [k for k, _ in recs] == ["game_mode", "tap_gate"]
    assert g.reasons == ["mode unknown (first clock 2:00 (tick 1210), not 2:5x-3:00)"]


def test_an_override_lasts_for_the_battle_and_a_resumed_battle_keeps_the_verdict():
    g = TG.TapGate()
    feed(g, period=M.REGEN_S_3X)
    assert g.state == TG.CLOSED and not g.allows()
    g.override("hotkey", 400, 20.0)
    assert g.allows() and g.status() == "gate overridden"
    assert g.summary()["overridden"] == {"source": "hotkey", "tick": 400, "t": 20.0}
    g.resume(2)                                                   # the same battle, followed again
    assert g.allows() and g.number == 2 and g.summary()["resumed"] == 1
    g.new_battle(3, 30, 100.0)                                    # a new battle: checked again
    assert g.state == TG.PENDING and g.overridden is None and not g.allows()


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def synth_base(enemy_dv: float) -> np.ndarray:
    labels = {k: S.Label(dv=enemy_dv if k in S.ENEMY else 0.0) for k in PRINCESS}
    img = S.battle_frame("sand", labels=labels, size=SIZE)
    img.setflags(write=False)
    return img


class SynthPerception(TVP.FakePerception):
    """The real HudReader on skin_synth frames: the towers read from the drawn labels (the
    detector's boxes: the six bodies), our elixir from the drawn bar filling at 1x; the clock,
    the hand and the screen are scripted."""

    def __init__(self, script, enemy_dv: float):
        super().__init__(script)
        self.hud_reader = H.HudReader(S.digit_store(SIZE), {}, regions=S.REGIONS, towers_wait_for_clock=False)
        self.detector = object()
        self.base = synth_base(enemy_dv)
        self.boxes = bodies(size=SIZE)
        self._imgs: dict[float, np.ndarray] = {}

    def perceive(self, frame):
        vf = super().perceive(frame)
        if vf.screen != SCREEN_BATTLE:
            return vf
        e = round(elixir_at(vf.tick) * 20.0) / 20.0
        img = self._imgs.get(e)
        if img is None:
            img = S.draw_hud(self.base.copy(), e)
            self._imgs[e] = img
        hud = self.hud_reader.read(img, frame.t, towers=False)
        hud = self.hud_reader.read_towers(img, frame.t, hud, self.hud_reader.tower_evidence(self.boxes, SIZE))
        # The decider keeps seeing 7.0 elixir (battle_vf's): a full bar would bring in the cap rule.
        vf.hud = Hud(hand=list(TVP.HAND), next=0, elixir_milli=7000, clock_tick=vf.tick,
                     phase=PHASE_REGULATION, towers=hud.towers, crowns=hud.crowns,
                     towers_destroyed=hud.towers_destroyed)
        vf.frame_size, vf.detections, vf.events = SIZE, list(self.boxes), Events()
        return vf


def test_a_battle_whose_labels_lie_outside_the_window_never_taps(tmp_path):
    """The enemy labels drawn 200 px above their references (outside the calibration window): the
    enemy towers never lock, the gate closes when their window ends (8 s), with one console line,
    and AUTO switches to MANUAL (source "towers"). The plays the bot wanted before that were held;
    nothing was ever tapped."""
    script = gate_script(300)
    perception = SynthPerception(script, -200.0)
    res = run_gate(tmp_path, script, perception=perception, play_ticks=(90, 120, 200, 260))
    assert res.tapper.calls == [] and "tap" not in res.records
    cal = perception.hud_reader.calib
    assert {k: cal.towers[k].status for k in S.ENEMY} == {"enemy_left": "expired", "enemy_right": "expired"}
    assert all(cal.calibrated(k) for k in S.OWN)
    [line] = gate_lines(res)
    assert line.startswith("towers not calibrated (enemy_left, enemy_right): tapping off this battle; play "
                           "yourself (switched to MANUAL; F8 hands back to the bot anyway, for this battle only)")
    [mode] = res.records["mode"]
    assert (mode["mode"], mode["prev"], mode["source"]) == (CT.MANUAL, CT.AUTO, "towers")
    assert res.control.mode == CT.MANUAL
    [gate] = res.records["tap_gate"]
    assert gate["state"] == TG.CLOSED and gate["switched"] is True and gate["enforced"] is True
    assert gate["reasons"] == ["towers not calibrated (enemy_left, enemy_right)"]
    assert 150 <= gate["tick"] <= 170                             # the calibration window: 8 s of clock
    actions = play_actions(res)
    assert [(t, a) for t, a, _ in actions] == [(90, CT.ACTION_SKIPPED), (120, CT.ACTION_SKIPPED),
                                               (200, CT.ACTION_MANUAL), (260, CT.ACTION_MANUAL)]
    assert actions[0][2] == "not tapped: tap gate: towers and mode not checked yet"
    assert any(x.startswith("battle 1: mode classic (first clock 3:00, elixir every ") for x in res.lines)
    [end] = res.records["battle_end"]
    assert end["tap_gate"]["state"] == TG.CLOSED and end["tap_gate"]["held"] == 2
    assert end["tap_gate"]["towers"]["labels"]["enemy_left"] == "expired"


def test_a_calibrated_battle_taps_from_tick_90(tmp_path):
    """The chicken skin: the enemy labels 36 px up. They lock within the first second, the
    detector boxes all six towers, the elixir fills at 1x: the gate is open before tick 90
    (MIN_DECIDE_TICK), so the first play is tapped at 90. No gate line; "mode classic"."""
    script = gate_script(200)
    perception = SynthPerception(script, S.SKIN_DV)
    res = run_gate(tmp_path, script, perception=perception, play_ticks=(90, 150))
    assert [t["tick"] for t in res.records["tap"]] == [90, 150] and len(res.tapper.calls) == 2
    assert all(perception.hud_reader.calib.calibrated(k) for k in PRINCESS)
    assert gate_lines(res) == []
    [gate] = res.records["tap_gate"]
    assert gate["state"] == TG.OPEN and gate["tick"] < VP.MIN_DECIDE_TICK
    assert gate["towers"]["bodies"] == {k: 1.0 for k in TG.BODY_KEYS}
    [mode] = res.records["game_mode"]
    assert mode["mode"] == "classic" and mode["first_clock"] == "3:00"
    assert any(x.startswith("battle 1: mode classic (first clock 3:00, elixir every ") for x in res.lines)
    assert res.control.mode == CT.AUTO and "mode" not in res.records


def test_the_owners_f8_overrides_the_gate_for_that_battle_only(tmp_path):
    """Battle 1: the enemy labels expire at 4 s, the gate closes (MANUAL); the owner presses F8 at
    tick 160: tap_gate_override (source "hotkey"), and the plays from then on are tapped. Battle 2
    starts in AUTO (the mode carries over), its gate closes again and switches to MANUAL again."""
    script = gate_script(260, battles=2)
    hud = GateHud(expired=S.ENEMY, expire_at=4.0)
    holder: dict = {}

    def on_read(i):
        if i == frame_at(script, 160):
            holder["state"].toggle("hotkey")

    res = run_gate(tmp_path, script, perception=GatePerception(script, hud), play_ticks=(100, 200, 240),
                   on_read=on_read, control=holder)
    [over] = res.records["tap_gate_override"]
    assert (over["battle"], over["tick"], over["source"], over["dry"]) == (1, 160, "hotkey", False)
    assert over["reasons"] == ["towers not calibrated (enemy_left, enemy_right)"]
    assert [(t["battle"], t["tick"]) for t in res.records["tap"]] == [(1, 200), (1, 240)]
    assert [(m["mode"], m["source"]) for m in res.records["mode"]] == [
        (CT.MANUAL, "towers"), (CT.AUTO, "hotkey"), (CT.MANUAL, "towers")]
    assert "tower gate overridden (hotkey): AUTO taps for the rest of battle 1" in res.lines
    assert len([x for x in res.lines if CLOSED_LINE in x]) == 2
    ends = res.records["battle_end"]
    assert ends[0]["tap_gate"]["overridden"]["source"] == "hotkey" and ends[1]["tap_gate"]["overridden"] is None
    assert res.control.mode == CT.MANUAL


def test_a_dry_run_only_says_what_the_gate_would_do(tmp_path):
    """No --tap: the gate closes, prints "dry run: the tower gate would keep tapping off ...", never
    switches, and the decisions are exactly those of a run without the gate."""
    script = gate_script(260)
    hud = GateHud(expired=S.ENEMY, expire_at=4.0)
    res = run_gate(tmp_path / "gate", script, perception=GatePerception(script, hud), taps=False,
                   play_ticks=(100, 200, 240))
    plain = run_gate(tmp_path / "plain", script, perception=GatePerception(script, GateHud(expired=S.ENEMY,
                                                                                           expire_at=4.0)),
                     taps=False, gate=None, play_ticks=(100, 200, 240))
    assert TVP.played(res) == TVP.played(plain)
    assert all(p[1] == CT.ACTION_DRY for p in TVP.played(res))
    [line] = gate_lines(res)
    assert line == ("dry run: the tower gate would keep tapping off this battle "
                    "(towers not calibrated (enemy_left, enemy_right))")
    assert "mode" not in res.records and res.control.mode == CT.AUTO
    [gate] = res.records["tap_gate"]
    assert (gate["state"], gate["enforced"], gate["switched"]) == (TG.CLOSED, False, False)
    assert gate_lines(plain) == [] and "tap_gate" not in plain.records


def test_double_elixir_turns_tapping_off(tmp_path):
    """The elixir fills twice as fast (1.4 s each): mode unknown, MANUAL with source "game mode",
    nothing tapped."""
    script = gate_script(260)
    res = run_gate(tmp_path, script, perception=GatePerception(script, GateHud(period=M.REGEN_S_2X)),
                   play_ticks=(100, 200))
    assert res.tapper.calls == []
    [line] = gate_lines(res)
    assert line.startswith("mode unknown (regen 1.4 s): tapping off this battle; play yourself (switched to MANUAL")
    [mode] = res.records["mode"]
    assert (mode["mode"], mode["source"]) == (CT.MANUAL, "game mode")
    [gm] = res.records["game_mode"]
    assert gm["mode"] == "unknown" and gm["why"] == "regen 1.4 s"
    assert not any("mode classic" in x for x in res.lines)


def test_a_clock_that_starts_at_2_00_turns_tapping_off_at_once(tmp_path):
    script = gate_script(1500, first_tick=1210)
    res = run_gate(tmp_path, script, play_ticks=(1300,))
    assert res.tapper.calls == []
    [line] = gate_lines(res)
    assert line.startswith("mode unknown (first clock 2:00 (tick 1210), not 2:5x-3:00): tapping off this battle")
    [gate] = res.records["tap_gate"]
    assert gate["tick"] == 1230                                   # the battle's start frame


def test_towers_the_detector_does_not_box_turn_tapping_off(tmp_path):
    """A king skin the detector misses (no box at the enemy king): the board check fails."""
    script = gate_script(260)
    perception = GatePerception(script, boxes=lambda tick: bodies([k for k in TG.BODY_KEYS if k != "enemy_king"]))
    res = run_gate(tmp_path, script, perception=perception, play_ticks=(100, 200))
    assert res.tapper.calls == []
    [line] = gate_lines(res)
    assert line.startswith("towers not boxed by the detector (enemy_king 0% of 3 frames): tapping off this battle")
    assert res.records["mode"][0]["source"] == "towers"


def test_taps_wait_while_the_labels_calibrate(tmp_path):
    """The labels lock only 6 s into the battle (tick ~120): the play at 100 is held (skipped, why
    the gate), the one at 200 is tapped. No gate line: the gate opened."""
    script = gate_script(260)
    res = run_gate(tmp_path, script, perception=GatePerception(script, GateHud(lock_at=6.0)), play_ticks=(100, 200))
    assert [t["tick"] for t in res.records["tap"]] == [200]
    acts = play_actions(res)
    assert acts[0] == (100, CT.ACTION_SKIPPED, "not tapped: tap gate: towers and mode not checked yet")
    assert gate_lines(res) == [] and res.records["tap_gate"][0]["state"] == TG.OPEN
    assert any("| AUTO(tap) | gate wait | last" in x for x in res.lines)


def test_a_gate_that_closes_in_manual_turns_the_routine_f8_back_and_the_second_overrides(tmp_path):
    """--tap in MANUAL, as the runbook's first tap friendly starts: the gate closes without a switch
    ("MANUAL already"). The owner's routine F8 to hand over must not lift it (the review: the gate
    was then only advisory): the runner switches back to MANUAL with the line again
    (tap_gate_handover), nothing is tapped; only a second F8 overrides (tap_gate_override), and
    the plays after it are tapped. The control window shows the gate all along."""
    script = gate_script(260)
    holder: dict = {}
    seen: dict = {}

    def on_read(i):
        if i == frame_at(script, 140):
            seen["closed"] = CT.window_lines(holder["state"].snapshot())["mode"]
        if i in (frame_at(script, 160), frame_at(script, 190)):
            holder["state"].toggle("hotkey")
        if i == frame_at(script, 180):
            seen["refused"] = CT.window_lines(holder["state"].snapshot())["mode"]
        if i == frame_at(script, 220):
            seen["overridden"] = CT.window_lines(holder["state"].snapshot())["mode"]

    res = run_gate(tmp_path, script, perception=GatePerception(script, GateHud(expired=S.ENEMY, expire_at=4.0)),
                   mode=CT.MANUAL, play_ticks=(176, 200), on_read=on_read, control=holder)
    lines = gate_lines(res)
    assert lines[0].endswith("(MANUAL already; your next F8 keeps it MANUAL, the one after hands back to the bot "
                             "anyway, for this battle only)")
    assert lines[1] == ("towers not calibrated (enemy_left, enemy_right): tapping off this battle; play yourself "
                        "(switched back to MANUAL; F8 again hands back to the bot anyway, for this battle only)")
    [gate] = res.records["tap_gate"]
    assert (gate["mode_before"], gate["switched"]) == (CT.MANUAL, False)
    [hand] = res.records["tap_gate_handover"]
    assert (hand["tick"], hand["source"], hand["switched"]) == (160, "hotkey", True)
    [over] = res.records["tap_gate_override"]
    assert over["tick"] == 190
    assert [(m["mode"], m["source"]) for m in res.records["mode"]] == [
        (CT.AUTO, "hotkey"), (CT.MANUAL, "towers"), (CT.AUTO, "hotkey")]
    assert [t["tick"] for t in res.records["tap"]] == [200]                 # not the play at 176
    assert seen["closed"] == "MANUAL: you play  |  tap gate OFF this battle (towers not calibrated (enemy_left, enemy_right))"
    assert seen["refused"].startswith("MANUAL: you play  |  tap gate OFF")
    assert seen["overridden"] == "AUTO: the bot plays  |  tap gate overridden this battle (hotkey)"
    assert CT.window_lines(res.control.snapshot())["mode"] == "AUTO: the bot plays"   # the battle ended


def test_a_dry_run_in_manual_says_what_the_routine_f8_would_do(tmp_path):
    """No --tap, MANUAL: the first F8 after the gate closed only says that a --tap run would switch
    back to MANUAL now; nothing switches and no override is logged for it."""
    script = gate_script(260)
    holder: dict = {}

    def on_read(i):
        if i == frame_at(script, 160):
            holder["state"].toggle("hotkey")

    res = run_gate(tmp_path, script, perception=GatePerception(script, GateHud(expired=S.ENEMY, expire_at=4.0)),
                   mode=CT.MANUAL, taps=False, play_ticks=(200,), on_read=on_read, control=holder)
    assert ("dry run: the tower gate would switch back to MANUAL now (towers not calibrated (enemy_left, "
            "enemy_right)); with --tap, F8 a second time would hand back anyway") in res.lines
    assert [(m["mode"], m["source"]) for m in res.records["mode"]] == [(CT.AUTO, "hotkey")]
    assert "tap_gate_override" not in res.records and res.records["tap_gate_handover"][0]["enforced"] is False


class MisreadHud(GateHud):
    """A GateHud whose count digit reads one too high on the frame of tick `at`."""

    def __init__(self, at, **kw):
        super().__init__(**kw)
        self.at = at

    def see(self, t, tick, since):
        super().see(t, tick, since)
        if tick == self.at:
            er = self.report.elixir_read
            self.report = SimpleNamespace(t=t, elixir_read=SimpleNamespace(digit=er.digit + 1, bar_frac=er.bar_frac))


@pytest.mark.parametrize("at", [24, 26, 30, 34])
def test_one_misread_elixir_digit_early_in_a_classic_battle_keeps_tapping(tmp_path, at):
    """The review's end-to-end case: a normal 1x battle with ONE count digit read +1 in its first
    1.5 s used to close the gate ("mode unknown (... do not fit one regeneration rate)" or
    "(regen 0.8 s)"): no tap, MANUAL. Now the frame's digit disagrees with its bar and gives no
    reading: classic, and the taps go out as without the misread."""
    script = gate_script(300)
    res = run_gate(tmp_path, script, perception=GatePerception(script, MisreadHud(at)), play_ticks=(100, 160, 220))
    assert [t["tick"] for t in res.records.get("tap", [])] == [100, 160, 220]
    assert res.control.mode == CT.AUTO and gate_lines(res) == []
    assert any(line.startswith("battle 1: mode classic") for line in res.lines)


def test_the_same_battle_followed_again_keeps_its_gate(tmp_path):
    """A misread clock (tick 40 after 300) ends battle 1 ("clock reset"); the runner follows the
    same battle again as battle 2, joined in the middle with its towers kept: its open gate stays
    open (its first clock reading, 3:00 - 15 s, would fail the mode check), and the play at 360
    is tapped. A new battle in that state would be checked again and closed."""
    ticks = list(range(0, 302, 2)) + [40] + list(range(304, 402, 2))
    script = gate_script(400, ticks=ticks)
    res = run_gate(tmp_path, script, play_ticks=(100, 360))
    starts = res.records["battle_start"]
    assert [s["battle"] for s in starts] == [1, 2] and starts[1].get("joined") is True
    assert res.records["battle_end"][0]["reason"] == "clock reset"
    assert [(t["battle"], t["tick"]) for t in res.records["tap"]] == [(1, 100), (2, 360)]
    assert [g["state"] for g in res.records["tap_gate"]] == [TG.OPEN]
    assert res.records["battle_end"][-1]["tap_gate"]["resumed"] == 1
    assert gate_lines(res) == []


def test_status_line_and_battle_end_show_the_gate(tmp_path):
    script = gate_script(260)
    res = run_gate(tmp_path, script, perception=GatePerception(script, GateHud(expired=S.ENEMY, expire_at=4.0)),
                   play_ticks=())
    assert any("| MANUAL | gate OFF | last" in x for x in res.lines)
    [end] = res.records["battle_end"]
    summary = end["tap_gate"]
    assert summary["state"] == TG.CLOSED and summary["mode"]["mode"] == "classic"
    assert summary["towers"]["detector"] is True and summary["towers"]["det_frames"] >= 1
    assert summary["held"] == 0 and summary["overridden"] is None
