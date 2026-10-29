"""V12: human takeover, AUTO / MANUAL (scripts/vision/control.py).

No window, hotkey or device is used: the Win32 hotkey is driven through a
fake backend, the window's availability through injected platform / env
values, and the runner's tap path through a FakeTapper. Clocks are fakes
except in the two thread tests, which wait at most a few seconds.
"""
from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import control as C  # noqa: E402
from scripts.vision.types import ENEMY_TEAM, OWN_TEAM, OwnLanding, Track  # noqa: E402

NAMES = {0: "Cannon", 1: "Fireball", 2: "Hog Rider", 3: "Ice Golem", 4: "Ice Spirit",
         5: "Musketeer", 6: "Skeletons", 7: "The Log"}
CLASS_CARD = {"skeleton": 6, "hog-rider": 2, "cannon": 0, "the-log": 7}


class FakeClock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeTapper:
    def __init__(self):
        self.calls = []

    def tap_pair(self, a, b):
        self.calls.append((a, b))


class FakeRecorder:
    def __init__(self):
        self.lines = []

    def log(self, kind, **data):
        self.lines.append((kind, data))


class FakeHotkeyBackend:
    """Stands in for Win32: press() delivers WM_HOTKEY and returns once it was handled."""

    def __init__(self, error: str | None = None):
        self.error = error
        self.q: queue.Queue = queue.Queue()
        self.registered, self.unregistered = [], []
        self.entered = 0                    # wait() calls so far
        self.sent = 0
        self.cv = threading.Condition()

    def register(self, hotkey_id, modifiers, vk):
        self.registered.append((hotkey_id, modifiers, vk))
        return self.error

    def wait(self):
        with self.cv:
            self.entered += 1
            self.cv.notify_all()
        return self.q.get()

    def wake(self):
        self.q.put(None)

    def unregister(self, hotkey_id):
        self.unregistered.append(hotkey_id)

    def press(self, hotkey_id: int = C.HOTKEY_ID, timeout: float = 5.0) -> None:
        # Press n is handled once the listener has entered wait() n + 2 times.
        n, self.sent = self.sent, self.sent + 1
        self.q.put(hotkey_id)
        with self.cv:
            assert self.cv.wait_for(lambda: self.entered >= n + 2, timeout)


def make_ctl(mode: str = C.AUTO, *, taps: bool = True, **kw) -> C.ControlState:
    kw.setdefault("clock", FakeClock())
    return C.ControlState(mode, taps_enabled=taps, stop_file=None, control_file=None, **kw)


def wait_until(pred, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def thread_names() -> set[str]:
    return {t.name for t in threading.enumerate()}


def tr(uid: int, cls: str, x: float, y: float, team: int = OWN_TEAM) -> Track:
    return Track(uid, cls, team, 0.9, float(x), float(y))


# ---------------------------------------------------------------------------
# parsing and mode switches
# ---------------------------------------------------------------------------
def test_parse_mode_and_hotkey():
    assert C.parse_mode(" Manual ") == C.MANUAL and C.parse_mode("AUTO") == C.AUTO
    with pytest.raises(ValueError):
        C.parse_mode("bot")
    assert C.parse_hotkey("F8") == (0, 0x77)
    assert C.parse_hotkey("ctrl+shift+f9") == (C.MOD_CONTROL | C.MOD_SHIFT, 0x78)
    assert C.parse_hotkey("Alt + Q") == (C.MOD_ALT, ord("Q"))
    assert C.parse_hotkey("pause") == (0, 0x13)
    assert C.parse_hotkey("F24") == (0, 0x87)
    for off in (None, "", "none", "OFF"):
        assert C.parse_hotkey(off) is None
    for bad in ("ctrl+", "f25", "hyper+f8", "f8x"):
        with pytest.raises(ValueError):
            C.parse_hotkey(bad)


def test_default_start_is_manual_and_every_switch_is_logged():
    ctl = C.ControlState(stop_file=None, control_file=None)
    assert ctl.mode == C.MANUAL and ctl.generation == 0 and not ctl.taps_enabled
    assert ctl.toggle("button") == C.AUTO and ctl.generation == 1
    assert ctl.set_mode("auto", "api") is False and ctl.generation == 1
    assert ctl.set_mode(C.MANUAL, "hotkey") is True and ctl.generation == 2
    recs = ctl.drain_log()
    assert [(k, d["prev"], d["mode"], d["source"], d["gen"]) for k, d in recs] == [
        ("mode", "manual", "auto", "button", 1), ("mode", "auto", "manual", "hotkey", 2)]
    assert ctl.drain_log() == []
    assert ctl.counts["switches"] == 2


def test_toggles_from_many_threads_are_all_counted():
    ctl = make_ctl(C.MANUAL)
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        for _ in range(125):
            ctl.toggle("thread")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert ctl.generation == 1000 and ctl.counts["switches"] == 1000 and ctl.mode == C.MANUAL
    recs = ctl.drain_log()
    assert [d["gen"] for _, d in recs] == list(range(1, 1001))
    assert all(d["mode"] == (C.AUTO if d["gen"] % 2 else C.MANUAL) for _, d in recs)


# ---------------------------------------------------------------------------
# inputs: hotkey, button, MANUAL file
# ---------------------------------------------------------------------------
def test_hotkey_press_toggles_through_a_fake_backend():
    ctl = make_ctl(C.MANUAL)
    fb = FakeHotkeyBackend()
    status = ctl.start(hotkey="F8", window=False, control_file=False, hotkey_backend=fb)
    try:
        assert status == {"control_file": "off: disabled", "hotkey": "on: F8",
                          "window": "off: disabled"}
        assert fb.registered == [(C.HOTKEY_ID, 0, 0x77)]
        fb.press()
        assert ctl.mode == C.AUTO
        fb.press(C.HOTKEY_ID + 1)                  # someone else's hotkey id: ignored
        assert ctl.mode == C.AUTO
    finally:
        ctl.stop()
    assert fb.unregistered == [C.HOTKEY_ID]
    assert "control-hotkey" not in thread_names()
    recs = ctl.drain_log()
    assert [(d["mode"], d["source"]) for k, d in recs if k == "mode"] == [(C.AUTO, "hotkey")]
    assert [d["inputs"] for k, d in recs if k == "control_start"] == [status]


def test_hotkey_debounce_and_failures():
    clock = FakeClock(10.0)
    presses = []
    fb = FakeHotkeyBackend()
    hk = C.HotkeyListener(lambda: presses.append(clock.t), "ctrl+F9", backend=fb,
                          platform="linux", debounce_s=0.3, clock=clock)
    assert hk.start() and hk.running
    try:
        assert fb.registered == [(C.HOTKEY_ID, C.MOD_CONTROL, 0x78)]
        fb.press()
        clock.t = 10.1
        fb.press()                                 # 0.1 s after the last: a bounce
        clock.t = 10.5
        fb.press()
    finally:
        hk.stop()
    assert presses == [10.0, 10.5] and hk.presses == 2

    taken = FakeHotkeyBackend(error="RegisterHotKey failed (error 1409: taken)")
    hk2 = C.HotkeyListener(lambda: None, "F8", backend=taken)
    assert hk2.start() is False and hk2.error == "RegisterHotKey failed (error 1409: taken)"
    hk2.stop()
    assert taken.unregistered == []                # nothing was registered

    bad = C.HotkeyListener(lambda: None, "hyper+F8", backend=FakeHotkeyBackend())
    assert bad.start() is False and "unknown modifier" in bad.error
    off = C.HotkeyListener(lambda: None, "none", backend=FakeHotkeyBackend())
    assert off.start() is False and off.error == "disabled"


def test_hotkey_and_window_are_skipped_off_windows_and_headless():
    ctl = make_ctl(C.MANUAL)
    status = ctl.start(hotkey="F8", window=True, control_file=False, platform="linux", env={})
    try:
        assert status["hotkey"] == ("off: global hotkeys need Windows (RegisterHotKey); "
                                    "this is linux")
        assert status["window"] == "off: no display (headless)"
        assert not thread_names() & {"control-hotkey", "control-window", "control-file"}
    finally:
        ctl.stop()
    assert C.window_available({"HASTYCR_HEADLESS": "1"}, "win32") == (False, "HASTYCR_HEADLESS=1")
    assert C.window_available({}, "darwin")[0] is False
    assert C.window_available({}, "linux") == (False, "no display (headless)")
    win = C.ControlWindow(ctl, env={"HASTYCR_HEADLESS": "1"}, platform="win32")
    assert win.start() is False and win.error == "HASTYCR_HEADLESS=1" and not win.running
    win.stop()


def test_the_window_button_toggles_without_opening_a_window():
    ctl = make_ctl(C.MANUAL)
    win = C.ControlWindow(ctl, platform="linux", env={})
    assert win.press_button() == C.AUTO
    assert win.press_button() == C.MANUAL
    assert [(d["mode"], d["source"]) for k, d in ctl.drain_log() if k == "mode"] == [
        (C.AUTO, "button"), (C.MANUAL, "button")]


def test_control_file_appearing_and_disappearing(tmp_path):
    manual = tmp_path / "MANUAL"
    ctl = C.ControlState(C.AUTO, taps_enabled=True, stop_file=None, control_file=manual,
                         clock=FakeClock())
    status = ctl.start(hotkey=None, window=False, poll_s=None)
    try:
        assert status["control_file"] == f"on: {manual} (polled by the runner)"
        assert ctl.poll() == C.AUTO
        manual.write_text("")
        assert ctl.poll() == C.MANUAL and not ctl.should_tap()
        assert ctl.poll() == C.MANUAL              # no change, no switch
        ctl.toggle("hotkey")                       # the hotkey still works while the file exists
        assert ctl.poll() == C.AUTO
        manual.unlink()
        assert ctl.poll() == C.AUTO                # disappearing means AUTO: already
        manual.write_text("")
        assert ctl.poll() == C.MANUAL
        manual.unlink()
        assert ctl.poll() == C.AUTO
    finally:
        ctl.stop()
    assert [(d["mode"], d["source"]) for k, d in ctl.drain_log() if k == "mode"] == [
        (C.MANUAL, "control file"), (C.AUTO, "hotkey"),
        (C.MANUAL, "control file"), (C.AUTO, "control file")]


def test_a_manual_file_present_at_start_wins_over_the_start_mode(tmp_path):
    manual = tmp_path / "MANUAL"
    manual.write_text("")
    ctl = C.ControlState(C.AUTO, taps_enabled=True, stop_file=None, control_file=manual)
    ctl.start(hotkey=None, window=False, poll_s=None)
    try:
        assert ctl.mode == C.MANUAL and not ctl.should_tap()
    finally:
        ctl.stop()


def test_control_file_watcher_thread(tmp_path):
    manual = tmp_path / "MANUAL"
    ctl = C.ControlState(C.AUTO, taps_enabled=True, stop_file=None, control_file=manual)
    status = ctl.start(hotkey=None, window=False, poll_s=0.01)
    try:
        assert status["control_file"] == f"on: {manual}"
        manual.write_text("")
        assert wait_until(lambda: ctl.mode == C.MANUAL)
        manual.unlink()
        assert wait_until(lambda: ctl.mode == C.AUTO)
    finally:
        ctl.stop()
    assert "control-file" not in thread_names()


# ---------------------------------------------------------------------------
# suggestions, taps and decisions
# ---------------------------------------------------------------------------
def test_manual_mode_never_taps_but_keeps_deciding():
    ctl = make_ctl(C.MANUAL, names=NAMES)
    tapper, decided = FakeTapper(), []

    def step(tick):
        decided.append(tick)                       # the runner's decide()
        card, slot, tile = 2, 0, (3, 15)
        s = ctl.suggest(card, slot=slot, tile=tile, tick=tick, p_wait=0.25)
        tapped = False
        if ctl.should_tap(s):
            tapper.tap_pair(("slot", slot), tile)
            tapped = True
        return ctl.note_decision(s, tapped=tapped, ms=7)

    recs = [step(t) for t in range(100, 110)]
    assert tapper.calls == [] and decided == list(range(100, 110))
    assert [s.tick for s in ctl.suggestions()] == list(range(100, 110))
    assert recs[0] == {"mode": "manual", "action": "manual", "tick": 100, "card": 2,
                       "name": "Hog Rider", "slot": 0, "tile": [3, 15], "p_wait": 0.25,
                       "reason": "", "gen": 0, "seq": 1, "mode_now": "manual",
                       "taps_enabled": True, "ms": 7}
    assert all(r["action"] == C.ACTION_MANUAL and r["mode"] == C.MANUAL for r in recs)
    # ten identical suggestions: logged once per REPEAT_TICKS
    assert [d["tick"] for k, d in ctl.drain_log() if k == "decision"] == [100]

    ctl.toggle("hotkey")                           # hand back: the next decision taps
    r = step(110)
    assert r["action"] == C.ACTION_TAP and r["mode"] == C.AUTO
    assert tapper.calls == [(("slot", 0), (3, 15))] and ctl.counts["taps"] == 1
    assert [d["action"] for k, d in ctl.drain_log() if k == "decision"] == ["tap"]


def test_dry_run_skip_and_wait_actions():
    dry = make_ctl(C.AUTO, taps=False)
    s = dry.suggest(2, slot=0, tile=(3, 15), tick=10)
    assert not dry.should_tap() and not dry.should_tap(s)
    assert dry.note_decision(s, tapped=False)["action"] == C.ACTION_DRY

    ctl = make_ctl(C.AUTO, log_waits=True)
    s = ctl.suggest(2, slot=0, tile=(3, 15), tick=10)
    assert ctl.should_tap(s)
    rec = ctl.note_decision(s, tapped=False, why="no legal tile")
    assert rec["action"] == C.ACTION_SKIPPED and rec["why"] == "no legal tile"
    w = ctl.suggest(tick=11, reason="no elixir")
    assert not ctl.should_tap(w)
    assert ctl.note_decision(w, tapped=False)["action"] == C.ACTION_WAIT
    assert ctl.last_decision["action"] == C.ACTION_SKIPPED       # a wait is not a "decision"
    assert [d["action"] for k, d in ctl.drain_log() if k == "decision"] == ["skipped", "wait"]


def test_a_decided_tap_is_dropped_when_the_mode_switches_before_sending():
    ctl = make_ctl(C.AUTO, names=NAMES)
    s = ctl.suggest(5, slot=1, tile=(8, 26), tick=200)
    assert ctl.should_tap(s)
    ctl.toggle("hotkey")                           # MANUAL before the tap went out
    assert not ctl.should_tap(s)
    rec = ctl.note_decision(s, tapped=False)
    assert (rec["action"], rec["mode"], rec["mode_now"]) == (C.ACTION_DROPPED, C.AUTO, C.MANUAL)

    ctl.toggle("hotkey")                           # back to AUTO: s is still stale
    assert ctl.mode == C.AUTO and ctl.should_tap() and not ctl.should_tap(s)
    s2 = ctl.suggest(5, slot=1, tile=(8, 26), tick=201)
    ctl.toggle("button")
    ctl.toggle("button")                           # away and back before sending: dropped
    assert ctl.mode == C.AUTO and not ctl.should_tap(s2)
    assert ctl.note_decision(s2, tapped=False)["action"] == C.ACTION_DROPPED
    s3 = ctl.suggest(5, slot=1, tile=(8, 26), tick=202)
    assert ctl.should_tap(s3)                      # the next decision may tap
    assert not ctl.should_tap(ctl.suggest(tick=203))               # a wait never taps
    assert ctl.counts["dropped"] == 2
    assert [d["action"] for k, d in ctl.drain_log() if k == "decision"] == ["dropped", "dropped"]


def test_stop_works_in_both_modes(tmp_path):
    stop = tmp_path / "STOP"
    for mode in (C.AUTO, C.MANUAL):
        stop.unlink(missing_ok=True)
        ctl = C.ControlState(mode, taps_enabled=True, stop_file=stop, control_file=None)
        assert not ctl.stop_requested and ctl.stop_reason is None
        assert ctl.should_tap() is (mode == C.AUTO)
        s = ctl.suggest(2, slot=0, tile=(3, 15), tick=50)
        stop.write_text("")
        assert ctl.stop_requested and ctl.stop_reason == "STOP file"
        assert not ctl.should_tap() and not ctl.should_tap(s)
        assert ctl.note_decision(s, tapped=False)["action"] == C.ACTION_STOPPED
        stop.unlink()
        assert ctl.stop_requested                  # latched: the runner ends
        ctl.toggle("hotkey")
        assert not ctl.should_tap()
        stops = [d for k, d in ctl.drain_log() if k == "control_stop"]
        assert len(stops) == 1 and stops[0]["reason"] == "STOP file" and stops[0]["tick"] == 50
    other = make_ctl(C.AUTO)
    other.request_stop("owner")
    assert other.stop_requested and other.stop_reason == "owner" and not other.should_tap()


def test_suggestions_normalise_waits_and_records_keep_their_core():
    ctl = make_ctl(C.AUTO, names=["Cannon", "Fireball"])
    w = ctl.suggest(-1, slot=2, tile=(1, 1), tick=5, reason="nothing affordable")
    assert w.is_wait and w.slot == -1 and w.tile is None and w.name == "wait"
    assert w.label() == "wait (nothing affordable)"
    s = ctl.suggest(3, slot=1, tile=(9.7, 20.2), tick=6, origin="test")
    assert s.tile == (9, 20) and s.name == "card 3" and s.label() == "card 3 -> tile (9, 20)"
    assert ctl.card_name(1) == "Fireball"
    assert C.ControlState(stop_file=None, control_file=None, names=lambda c: f"#{c}").card_name(4) == "#4"
    rec = C.decision_record(s, C.ACTION_SKIPPED, mode="x", action="y", why="no legal tile")
    assert (rec["mode"], rec["action"], rec["origin"], rec["why"]) == (
        C.AUTO, C.ACTION_SKIPPED, "test", "no legal tile")
    assert ctl.current_suggestion is s


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------
def test_unmatched_own_plays_are_attributed_to_the_human():
    ctl = make_ctl(C.MANUAL, names=NAMES)
    ctl.suggest(0, slot=0, tile=(9, 25), tick=100)
    ctl.suggest(tick=110, reason="wait")
    ctl.suggest(5, slot=2, tile=(8, 26), tick=120)
    plays = [C.OwnPlay(5, 150, 8400.0, 26700.0, uid=7),          # est. tap 124: the Musketeer at 120
             {"card": 6, "tick": 136},                          # est. 110: the bot was waiting
             OwnLanding(card=1, slot=-1, tap_tick=0, land_tick=110),   # est. 84: no suggestion yet
             OwnLanding(card=2, slot=0, tap_tick=90)]            # a missed tap report: not a play
    out = ctl.attribute(plays, tick=150)
    assert [a.who for a in out] == [C.HUMAN] * 3
    a = out[0]
    assert (a.tap_tick, a.lag_ticks, a.suggestion.tick, a.same_card, a.tile) == (124, 26, 120, True, (8, 26))
    assert a.tile_dist == pytest.approx(math.sqrt(0.05))        # (8.4, 26.7) vs centre (8.5, 26.5)
    assert out[1].suggestion.is_wait and out[1].same_card is False and out[1].tile_dist is None
    assert out[2].suggestion is None and out[2].same_card is None
    human = [d for k, d in ctl.drain_log() if k == "human_play"]
    assert human[0] == {
        "who": "human", "mode": "manual", "card": 5, "name": "Musketeer", "tick": 150,
        "tap_tick": 124, "estimated": True, "lag_ticks": 26, "x_mt": 8400.0, "y_mt": 26700.0,
        "tile": [8, 26], "uid": 7, "tapped_card": None,
        "bot": {"tick": 120, "card": 5, "name": "Musketeer", "slot": 2, "tile": [8, 26],
                "p_wait": None, "reason": "", "mode": "manual", "gen": 0, "seq": 3},
        "bot_age_ticks": 4, "same_card": True, "tile_dist": 0.22, "gen": 0}
    assert [d["bot"] is None for d in human] == [False, False, True]
    assert ctl.counts["human_plays"] == 3 and ctl.last_human["card"] == 1


def test_bot_plays_are_matched_and_the_lag_is_learned():
    ctl = make_ctl(C.AUTO, names=NAMES)

    def play(card, slot, tile, tick):
        s = ctl.suggest(card, slot=slot, tile=tile, tick=tick)
        assert ctl.should_tap(s)
        return ctl.note_decision(s, tapped=True)

    play(2, 1, (3, 15), 100)
    [a] = ctl.attribute([C.OwnPlay(2, 122, 3500.0, 15400.0, uid=21)], tick=122)
    assert (a.who, a.tap_tick, a.lag_ticks, a.tapped_card, a.suggestion.tick) == (C.BOT, 100, 22, 2, 100)
    play(0, 0, (9, 24), 200)
    # perception read the Cannon as Skeletons: matched by the aim, not the card
    [a] = ctl.attribute([C.OwnPlay(6, 224, 9400.0, 24600.0, uid=22)], tick=224)
    assert (a.who, a.card, a.tapped_card, a.lag_ticks) == (C.BOT, 6, 0, 24)
    play(4, 2, (9, 22), 300)                       # the event deriver matched this one itself
    [a] = ctl.attribute([OwnLanding(card=4, slot=2, tap_tick=300, land_tick=330, x_mt=9500.0,
                                    y_mt=22500.0, uid=23)], tick=330)
    assert (a.who, a.lag_ticks, a.tapped_card) == (C.BOT, 30, 4)
    assert ctl.attributor.lag_estimate() == 24     # median of 22, 24, 30
    assert ctl.attributor.pending == []

    ctl.suggest(tick=390, reason="wait")
    [h] = ctl.attribute([C.OwnPlay(5, 420, 3000.0, 27000.0, uid=24)], tick=420)   # you, in AUTO
    assert (h.who, h.mode, h.tap_tick, h.lag_ticks, h.suggestion.tick, h.same_card) == (
        C.HUMAN, C.AUTO, 396, 24, 390, False)

    play(6, 3, (8, 27), 430)                       # never seen
    assert ctl.attribute([], tick=490) == [] and ctl.counts["bot_unseen"] == 0
    ctl.attribute([], tick=491)
    unseen = [d for k, d in ctl.drain_log() if k == "bot_tap_unseen"]
    assert unseen == [{"card": 6, "name": "Skeletons", "slot": 3, "tap_tick": 430, "tick": 491,
                       "mode": "auto", "gen": 0}]
    assert ctl.counts == {"switches": 0, "suggestions": 5, "taps": 4, "dropped": 0,
                          "bot_plays": 3, "human_plays": 1, "bot_unseen": 1}


def test_note_human_play_pairs_with_a_recent_suggestion_only():
    ctl = make_ctl(C.MANUAL, names=NAMES)
    ctl.suggest(2, slot=0, tile=(3, 15), tick=100)
    a = ctl.note_human_play(2, 126, x_mt=3500.0, y_mt=15500.0)
    assert (a.who, a.tap_tick, a.suggestion.tick, a.same_card, a.tile_dist) == (C.HUMAN, 100, 100, True, 0.0)
    b = ctl.note_human_play(5, 300)                # est. 274: the suggestion at 100 is too old
    assert b.suggestion is None and ctl.suggestion_at(274) is None and ctl.suggestion_at(160) is not None
    with pytest.raises(TypeError):
        C.as_own_play(("not", "a", "play"))


def test_new_battle_forgets_suggestions_and_pending_taps():
    ctl = make_ctl(C.AUTO)
    s = ctl.suggest(2, slot=0, tile=(3, 15), tick=3000)
    ctl.note_decision(s, tapped=True)
    ctl.new_battle()
    assert ctl.suggestions() == [] and ctl.attributor.pending == [] and ctl.tick is None
    [a] = ctl.attribute([C.OwnPlay(2, 30, 3500.0, 15500.0)], tick=30)
    assert a.who == C.HUMAN and a.suggestion is None


def test_own_play_finder_counts_each_play_once():
    f = C.OwnPlayFinder(lambda t: CLASS_CARD.get(t.cls))
    assert f.update([], 0) == []
    skel = [tr(1, "skeleton", 9000, 26000), tr(2, "skeleton", 9400, 26200), tr(3, "skeleton", 8700, 26300)]
    assert f.update(skel, 10) == [C.OwnPlay(6, 10, 9000.0, 26000.0, 1)]
    moved = [tr(1, "skeleton", 9000, 25800), tr(2, "skeleton", 9400, 26000), tr(3, "skeleton", 8700, 26100)]
    assert f.update(moved + [tr(9, "hog-rider", 3500, 14000, ENEMY_TEAM)], 11) == []
    assert f.update(moved + [tr(4, "skeleton", 9100, 25900)], 12) == []    # a split track
    assert f.update([tr(5, "hog-rider", 14500, 17500), tr(8, "king-tower", 9000, 29000)], 30) == [
        C.OwnPlay(2, 30, 14500.0, 17500.0, 5)]
    assert f.update([tr(6, "hog-rider", 14500, 16000)], 38) == []          # re-acquired, new uid
    assert f.update([tr(10, "the-log", 9000, 20000)], 50) == [C.OwnPlay(7, 50, 9000.0, 20000.0, 10)]
    assert f.update([tr(11, "the-log", 9000, 15000)], 60) == []            # rolled 5 tiles, re-found
    assert f.update([tr(7, "hog-rider", 3500, 17500)], 200) == [C.OwnPlay(2, 200, 3500.0, 17500.0, 7)]


def test_own_play_finder_primes_on_the_first_frame_and_feeds_attribution():
    on_board = [tr(1, "cannon", 9000, 24500)]
    f = C.OwnPlayFinder(lambda t: CLASS_CARD.get(t.cls))
    assert f.update(on_board, 900) == []           # started mid-battle: not a play
    plays = f.update(on_board + [tr(2, "hog-rider", 14500, 17500)], 901)
    assert plays == [C.OwnPlay(2, 901, 14500.0, 17500.0, 2)]
    ctl = make_ctl(C.MANUAL)
    assert [a.who for a in ctl.attribute(plays, tick=901)] == [C.HUMAN]
    f.reset()
    assert f.update([tr(3, "cannon", 9000, 24500)], 0) == []
    unprimed = C.OwnPlayFinder(lambda t: CLASS_CARD.get(t.cls), prime=False)
    assert unprimed.update(on_board, 900) == [C.OwnPlay(0, 900, 9000.0, 24500.0, 1)]


# ---------------------------------------------------------------------------
# the window's texts, logging and options
# ---------------------------------------------------------------------------
def test_window_lines():
    ctl = make_ctl(C.MANUAL, names=NAMES)
    lines = C.window_lines(ctl.snapshot())
    assert (lines["mode"], lines["color"]) == ("MANUAL: you play", C.COLORS["manual"])
    assert (lines["button"], lines["button_state"]) == ("Hand back to the bot  [F8]", "normal")
    assert (lines["suggestion"], lines["decision"], lines["human"]) == (
        "Bot suggests: -", "Last decision: -", "Your last play: -")
    s = ctl.suggest(2, slot=0, tile=(3, 15), tick=120)
    ctl.note_decision(s, tapped=False)
    ctl.note_decision(ctl.suggest(tick=121, reason="no elixir"), tapped=False)
    ctl.note_human_play(5, 150, x_mt=8400.0, y_mt=26700.0)         # est. 124: the wait at 121
    lines = C.window_lines(ctl.snapshot(), hotkey=None)
    assert lines["suggestion"] == "Bot suggests: wait (no elixir)   (t=121)"
    assert lines["decision"] == "Last decision: t=120 manual, would play Hog Rider (3, 15)"
    assert lines["human"] == "Your last play: t=150 Musketeer (8, 26); the bot had wait"
    assert lines["button"] == "Hand back to the bot"
    assert lines["status"] == "taps 0   dropped 0   your plays 1   switches 0"
    ctl.toggle("button")
    assert C.window_lines(ctl.snapshot())["mode"] == "AUTO: the bot plays"

    dry = make_ctl(C.AUTO, taps=False)
    lines = C.window_lines(dry.snapshot())
    assert (lines["mode"], lines["button"], lines["color"]) == (
        "AUTO (dry run: no --tap)", "Take over  [F8]", C.COLORS["dry"])
    dry.request_stop("STOP file")
    lines = C.window_lines(dry.snapshot())
    assert (lines["mode"], lines["button_state"]) == ("STOPPED (STOP file)", "disabled")

    tap = {"action": "tap", "tick": 5, "name": "Hog Rider", "tile": [3, 15]}
    assert C.describe_decision(tap) == "t=5 tapped Hog Rider (3, 15)"
    assert C.describe_decision({**tap, "action": "dropped"}) == (
        "t=5 dropped Hog Rider (3, 15) (mode switched before the tap)")
    assert C.describe_decision({"action": "skipped", "card": 4, "reason": "no legal tile"}) == (
        "not played card 4 (no legal tile)")


def test_records_reach_a_recorder():
    ctl = make_ctl(C.MANUAL, names=NAMES)
    ctl.toggle("hotkey")
    s = ctl.suggest(2, slot=0, tile=(3, 15), tick=100)
    rec = ctl.note_decision(s, tapped=True, kind="odd", px=[256, 783])
    assert rec["kind"] == "odd"                    # returned as given ...
    fake = FakeRecorder()
    assert ctl.flush_to(fake) == 2 and ctl.drain_log() == []
    assert [k for k, _ in fake.lines] == ["mode", "decision"]
    dec = fake.lines[1][1]
    assert dec["data_kind"] == "odd" and "kind" not in dec          # ... renamed for the log
    assert (dec["mode"], dec["action"], dec["px"]) == (C.AUTO, C.ACTION_TAP, [256, 783])
    assert ctl.record("note", mode="x") == ("note", {"mode": "x", "gen": 1, "tick": 100})


def test_records_through_a_vision_recorder(tmp_path):
    R = pytest.importorskip("scripts.vision.record")     # needs numpy
    ctl = make_ctl(C.MANUAL, names=NAMES)
    ctl.toggle("hotkey")
    ctl.note_decision(ctl.suggest(2, slot=0, tile=(3, 15), tick=100), tapped=True)
    with R.VisionRecorder(tmp_path, "s1", image_every=0) as recorder:
        assert ctl.flush_to(recorder) == 2
    reader = R.VisionLogReader(tmp_path / "s1")
    assert [e.kind for e in reader if e.kind != "session"] == ["mode", "decision"]
    dec = next(reader.entries("decision")).data["data"]
    assert (dec["mode"], dec["action"], dec["name"], dec["tile"]) == ("auto", "tap", "Hog Rider", [3, 15])


def test_command_line_options():
    ap = argparse.ArgumentParser()
    C.add_control_args(ap)
    args = ap.parse_args([])
    assert (args.start_mode, args.hotkey, args.no_control_window, args.manual_file) == (
        "manual", "F8", False, str(C.DEFAULT_CONTROL_FILE))
    assert C.start_kwargs(args) == {"hotkey": "F8", "window": True}
    ctl = C.ControlState.from_args(args, taps_enabled=False, stop_file=None)
    assert ctl.mode == C.MANUAL and ctl.control_file == C.DEFAULT_CONTROL_FILE
    args = ap.parse_args(["--start-mode", "auto", "--hotkey", "ctrl+F9", "--no-control-window",
                          "--manual-file", "none"])
    ctl = C.ControlState.from_args(args, taps_enabled=True, stop_file=None)
    assert ctl.mode == C.AUTO and ctl.control_file is None and ctl.taps_enabled
    assert C.start_kwargs(args) == {"hotkey": "ctrl+F9", "window": False}
    for bad in (["--hotkey", "hyper+F8"], ["--start-mode", "bot"]):
        with pytest.raises(SystemExit):
            ap.parse_args(bad)


def test_the_demo_never_taps(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(C, "DEFAULT_STOP_FILE", tmp_path / "STOP")
    rc = C.main(["--seconds", "0.01", "--no-control-window", "--hotkey", "none",
                 "--manual-file", "none"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "control_file  off: disabled" in out and "hotkey        off: disabled" in out
    assert "done; mode MANUAL, 0 switches" in out
    (tmp_path / "STOP").write_text("")
    assert C.main(["--no-control-window", "--hotkey", "none", "--manual-file", "none"]) == 2
