"""V10: scripts/vision_play.py, the friendly-only live runner (with V12's takeover wired in).

No device, model, detector or emulator. The loop runs on a scripted FrameSource, a scripted
perception (the VisionFrame of each frame is given), a fake policy speaking rs_decide's
batch / gate / tiles (it plays slot 1 on model tile (4, 22) at chosen ticks and waits
otherwise), a legality that accepts every tile, a FakeTapper, and a real ControlState and
VisionRecorder in tmp_path. Then the parts: screen cues on synthetic frames, LivePerception
with the real tracker and event deriver, BattleState, ResultBook, AdbTapper with a fake adb,
the instance lock, the command line, and the import rule (scripts/gamestate is never loaded).

The standard script, 10 frames a second, 2 ticks a frame:
    frames 0-1    menu
    frames 2-27   battle screen, ticks 70..120; the battle starts at tick 90 (the clock
                  has run one displayed second), decisions from tick 90 on
    frames 28-32  the end screen (crowns 1-0), its clock still showing (tick 122)
    frames 33-35  menu
"""
from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import vision_play as VP  # noqa: E402
from scripts.vision import control as CT  # noqa: E402
from scripts.vision import events as EV  # noqa: E402
from scripts.vision.capture import Frame  # noqa: E402
from scripts.vision.geometry import ArenaGeometry  # noqa: E402
from scripts.vision.record import VisionLogReader, VisionRecorder  # noqa: E402
from scripts.vision.types import (ENEMY_TEAM, PHASE_ENDED, PHASE_REGULATION, SCREEN_BATTLE,  # noqa: E402
                                  SCREEN_END, SCREEN_MENU, Detection, Events, Hud, OppPlay,
                                  OwnLanding, VisionFrame)

RD = VP.RD

#: conftest's DEFAULT_CATALOGUE order: Hog 2.6 is 0-7.
NAMES = ["Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons",
         "Log", "Knight", "Archer", "Giant", "Arrows", "Valkyrie", "MiniPekka", "BabyDragon", "Zap"]
COST = {0: 3, 1: 4, 2: 4, 3: 2, 4: 1, 5: 4, 6: 1, 7: 2}
HAND = [7, 2, 6, 4]                  # Log, HogRider, Skeletons, IceSpirits: slot 1 is the Hog
PLAY_SLOT, PLAY_TILE = 1, (4, 22)     # model (own-frame) tile; screen tile (17 - 4, 22) = (13, 22)
GEOMETRY = ArenaGeometry.builtin()    # the author's fit: px = 78 + 0.051 x, 151.2 + 0.0414 y at 1080x1920
DEVICE = (1080, 1920)
T0 = 10.0


def t_of(i: int) -> float:
    return round(T0 + 0.1 * i, 3)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, t: float = T0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeSource:
    """Frames in order, each once; the clock jumps to the frame's time when it is read."""

    def __init__(self, frames, clock, on_read=None):
        self.frames = list(frames)
        self.i = 0
        self.clock = clock
        self.on_read = on_read
        self.closed = False

    @property
    def exhausted(self) -> bool:
        return self.i >= len(self.frames)

    def read(self):
        if self.i >= len(self.frames):
            return None
        f = self.frames[self.i]
        self.i += 1
        self.clock.t = f.t
        if self.on_read is not None:
            self.on_read(self.i - 1)
        return f

    def close(self):
        self.closed = True


class FakePerception:
    """Returns the scripted VisionFrame of each frame (by its time), or raises the scripted error."""

    def __init__(self, script):
        self.script = dict(script)
        self.seen = []
        self.new_battles = 0
        self.end_battles = 0
        self.own_taps = []

    def perceive(self, frame):
        self.seen.append(frame.t)
        item = self.script[frame.t]
        if isinstance(item, BaseException):
            raise item
        return item

    def new_battle(self):
        self.new_battles += 1

    def end_battle(self):
        self.end_battles += 1

    def note_own_tap(self, card, tick, *, slot, aim_mt, t):
        self.own_taps.append((card, tick, slot, tuple(aim_mt)))

    def track_cards(self, tracks):
        return {}


class FakePolicy:
    """rs_decide's policy protocol. At a tick in play_ticks the gate is all for slot PLAY_SLOT
    (wait logit -1e4: p_wait is exactly 0); otherwise all for waiting (p_wait exactly 1). The
    tile logits put PLAY_TILE first."""

    def __init__(self, play_ticks=(100,)):
        self.play_ticks = set(play_ticks)

    def batch(self, row):
        return row

    def gate(self, b):
        g = np.full(5, -60.0, dtype=np.float32)
        if b["tick"] in self.play_ticks:
            g[0] = -1e4
            g[1 + PLAY_SLOT] = 50.0
        else:
            g[0] = 50.0
        return g

    def tiles(self, b, slot):
        t = np.zeros(RD.TILES, dtype=np.float32)
        t[PLAY_TILE[1] * RD.GRID_W + PLAY_TILE[0]] = 10.0
        return t


class FakeTapper:
    def __init__(self, fail: bool = False):
        self.calls = []
        self.fail = fail

    def tap_pair(self, a, b):
        if self.fail:
            raise OSError("pipe gone")
        self.calls.append((tuple(a), tuple(b)))


def make_decider(policy, *, captured=None, fail_at=None):
    """A Decider on a minimal state built from the HUD; row_fn records (tick, view, opp cards)."""

    def state_fn(vf, ctx):
        if fail_at is not None and vf.tick == fail_at:
            raise RuntimeError("state boom")
        hand = list(vf.hud.hand)
        me = {"hand": hand, "hand_costs": [COST.get(c, 0) for c in hand],
              "elixir_milli": vf.hud.elixir_milli or 0, "next_card": vf.hud.next}
        them = {"hand": [], "hand_costs": [], "elixir_milli": 0}
        if captured is not None:
            captured.setdefault("opp_cards", []).append((vf.tick, list(ctx.opp_cards)))
        return {"tick": vf.tick, "entities": [], "players": [them, me]}

    def row_fn(state, view, tick):
        if captured is not None:
            captured.setdefault("views", []).append((tick, view))
        return {"tick": tick}

    return VP.Decider(policy=policy, legality=lambda slot, card, x, y: True,
                      config=RD.DecideConfig.live(0.0), state_fn=state_fn, row_fn=row_fn,
                      rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# scripted frames
# ---------------------------------------------------------------------------

def battle_vf(t, tick, *, opp=(), own=()):
    hud = Hud(hand=list(HAND), next=0, elixir_milli=7000, clock_tick=tick, phase=PHASE_REGULATION)
    return VisionFrame(t_capture=t, tick=tick, hud=hud, events=Events(list(opp), list(own)),
                       screen=SCREEN_BATTLE, frame_size=(225, 400))


def end_vf(t, tick=122, crowns=(1, 0)):
    return VisionFrame(t_capture=t, tick=tick, hud=Hud(crowns=crowns, phase=PHASE_ENDED), screen=SCREEN_END)


def menu_vf(t):
    return VisionFrame(t_capture=t, screen=SCREEN_MENU)


def standard_script(*, opp_at=None, own_at=None, error_at=None):
    """[(t, VisionFrame or exception)] of the module doc's script. opp_at / own_at: {tick:
    [OppPlay] / [OwnLanding]} for battle frames; error_at: {tick: exception} replaces a frame."""
    items = [menu_vf(t_of(0)), menu_vf(t_of(1))]
    for k in range(26):
        i, tick = 2 + k, 70 + 2 * k
        if error_at and tick in error_at:
            items.append(error_at[tick])
            continue
        items.append(battle_vf(t_of(i), tick, opp=(opp_at or {}).get(tick, ()),
                               own=(own_at or {}).get(tick, ())))
    items += [end_vf(t_of(28 + k)) for k in range(5)]
    items += [menu_vf(t_of(33 + k)) for k in range(3)]
    return [(t_of(i), item) for i, item in enumerate(items)]


FRAME_OF_TICK = {70 + 2 * k: 2 + k for k in range(26)}     # tick 100 -> frame 17


def read_records(session_dir):
    out = {}
    for e in VisionLogReader(session_dir):
        out.setdefault(e.kind, []).append(e.data if e.kind in ("frame", "session") else e.data["data"])
    return out


def run_script(tmp_path, script, *, mode=CT.AUTO, taps=True, play_ticks=(100,), tapper=None,
               size=(225, 400), on_read=None, decider=None, image_every=0, tap_guard=None):
    clock = FakeClock()
    rgb = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frames = [Frame(t, rgb, 0.0) for t, _ in script]
    perception = FakePerception(script)
    control = CT.ControlState(mode, taps_enabled=taps, stop_file=tmp_path / "STOP", control_file=None,
                              names=NAMES, clock=clock)
    recorder = VisionRecorder(tmp_path / "rec", session="s", image_every=image_every, image_format="npy",
                              clock=clock, wall_clock=lambda: 0.0)
    tapper = FakeTapper() if tapper is None else tapper
    lines = []
    runner = VP.VisionPlay(source=FakeSource(frames, clock, on_read), perception=perception,
                           decider=decider or make_decider(FakePolicy(play_ticks)), control=control,
                           recorder=recorder, geometry=GEOMETRY, tapper=tapper, card_names=NAMES,
                           clock=clock, sleep=lambda s: None, out=lines.append, tap_guard=tap_guard)
    rc = runner.run()
    recorder.close()
    return SimpleNamespace(rc=rc, runner=runner, tapper=tapper, perception=perception, control=control,
                           lines=lines, records=read_records(tmp_path / "rec" / "s"),
                           session=tmp_path / "rec" / "s")


def decisions(records, action=None):
    return [d for d in records.get("decision", []) if action is None or d["action"] == action]


def expected_px():
    """The device pixels of the Hog's slot and of model tile (4, 22), worked out by hand."""
    card = (538, 1711)                        # DEFAULT_CARD_SLOTS_PX[1]
    # model tile (4, 22) -> world (18000 - 4500, 22500) = (13500, 22500), a mirror in x:
    # px = 78 + 0.051 * 13500 = 766.5 (a tie; floating point picks 766 or 767), py = 151.2 +
    # 0.0414 * 22500 = 1082.7 -> 1083
    return card, (766.5, 1083)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(225, 400), (747, 1328)])
def test_tap_and_auto_tap_only_in_battle_at_device_pixels(tmp_path, size):
    """--tap and AUTO: one tap, at tick 100, the Hog's slot then model tile (4, 22), in 1080x1920
    device pixels whatever the frame size. The policy also wants to play at tick 80 (battle
    screen, battle not started yet: no decision at all) and 122 (the end screen's clock): no tap."""
    res = run_script(tmp_path, standard_script(), play_ticks=(80, 100, 122), size=size)
    assert res.rc == 0
    card_px, (tile_x, tile_y) = expected_px()
    tile_px = GEOMETRY.tap_px(*PLAY_TILE, size=DEVICE)
    assert abs(tile_px[0] - tile_x) <= 0.5 and tile_px[1] == tile_y
    assert res.tapper.calls == [(card_px, tile_px)]
    assert res.runner.taps == 1
    # the tap told perception where it aimed (world mt) for the landing match
    assert res.perception.own_taps == [(2, 100, PLAY_SLOT, (13500, 22500))]
    recs = res.records
    [start] = recs["battle_start"]
    assert start["battle"] == 1 and start["tick"] == 90
    [tap] = recs["tap"]
    assert (tap["tick"], tap["card"], tap["name"], tap["slot"]) == (100, 2, "HogRider", PLAY_SLOT)
    assert tap["model_tile"] == [4, 22] and tap["screen_tile"] == [13, 22]
    assert tap["card_px"] == list(card_px) and tap["tile_px"] == list(tile_px)
    [d] = decisions(recs, CT.ACTION_TAP)
    assert d["tick"] == 100 and d["mode"] == CT.AUTO and d["tile"] == [13, 22]
    assert d["px"] == {"card": list(card_px), "tile": list(tile_px)}
    assert decisions(recs) == [d]                  # waits are not logged as decisions
    # every decision of the battle is on its frame line, the tap with its pixels
    decided = [f["extra"] for f in recs["frame"] if "decision" in f.get("extra", {})]
    assert [e["decision"]["tick"] for e in decided] == list(range(90, 121, 2))
    [tapped] = [e for e in decided if e["action"] == CT.ACTION_TAP]
    assert tapped["decision"]["kind"] == RD.PLAY and tapped["px"]["tile"] == list(tile_px)
    assert {e["action"] for e in decided} == {CT.ACTION_TAP, CT.ACTION_WAIT}
    # the end screen ended the battle; the result came from its crowns
    [end] = recs["battle_end"]
    assert end["reason"] == "end screen" and end["taps"] == 1
    [result] = recs["result"]
    assert result["outcome"] == "win" and result["crowns"] == [1, 0] and result["end_crowns"] == [1, 0]
    assert res.perception.new_battles == 1 and res.perception.end_battles == 1
    assert recs["session_end"][0]["taps"] == 1
    assert any(line.startswith("battle 1: WIN 1-0") for line in res.lines)


def test_dry_run_logs_decisions_and_never_taps(tmp_path):
    """No --tap (the default): AUTO decides the same play, logs it as "dry", taps nothing."""
    res = run_script(tmp_path, standard_script(), taps=False, play_ticks=(100,))
    assert res.tapper.calls == [] and res.runner.taps == 0
    [d] = decisions(res.records)
    assert d["action"] == CT.ACTION_DRY and d["tick"] == 100 and d["card"] == 2
    assert d["taps_enabled"] is False
    card_px, _ = expected_px()
    assert d["px"]["card"] == list(card_px)        # where it would have tapped
    assert "tap" not in res.records and res.perception.own_taps == []
    assert any("AUTO(dry)" in line for line in res.lines)


def test_manual_never_taps_and_logs_suggestions_and_human_plays(tmp_path):
    """MANUAL with --tap: decide() still runs; the play is a suggestion; our Musketeer seen at
    tick 120 matches no bot tap, so it is the human's."""
    musk = OwnLanding(card=5, slot=-1, tap_tick=120, land_tick=120, x_mt=9000.0, y_mt=25000.0, uid=7)
    res = run_script(tmp_path, standard_script(own_at={120: [musk]}), mode=CT.MANUAL, taps=True,
                     play_ticks=(100,))
    assert res.tapper.calls == [] and res.runner.taps == 0
    [d] = decisions(res.records)
    assert d["action"] == CT.ACTION_MANUAL and d["mode"] == CT.MANUAL and d["tile"] == [13, 22]
    plays = [s for s in res.control.suggestions() if not s.is_wait]
    assert [(s.tick, s.card, s.slot, s.tile, s.mode) for s in plays] == [(100, 2, 1, (13, 22), CT.MANUAL)]
    [human] = res.records["human_play"]
    assert (human["who"], human["card"], human["name"], human["mode"]) == ("human", 5, "Musketeer", CT.MANUAL)
    assert res.control.counts["human_plays"] == 1
    assert any("| MANUAL |" in line for line in res.lines)


@pytest.mark.parametrize("mode", [CT.AUTO, CT.MANUAL])
def test_stop_file_ends_the_runner_without_tapping(tmp_path, mode):
    """STOP appears while the tick-100 frame (a play) is read: that frame taps nothing and the
    runner ends before the next one, in either mode."""
    stop_at = FRAME_OF_TICK[100]

    def on_read(i):
        if i == stop_at:
            (tmp_path / "STOP").write_text("", encoding="utf-8")

    res = run_script(tmp_path, standard_script(), mode=mode, play_ticks=(100, 102), on_read=on_read)
    assert res.tapper.calls == []
    assert len(res.perception.seen) == stop_at + 1
    [d] = decisions(res.records)
    assert d["tick"] == 100 and d["action"] == CT.ACTION_STOPPED
    assert res.records["control_stop"][0]["reason"] == "STOP file"
    assert res.records["session_end"][0]["reason"].startswith("STOP")
    assert res.runner.stop_reason.startswith("STOP")


def test_a_perception_error_never_taps_that_frame(tmp_path):
    """The tick-100 frame raises in perception: logged, nothing decided or tapped there; the next
    frame (tick 102) plays and taps."""
    script = standard_script(error_at={100: RuntimeError("perception boom")})
    res = run_script(tmp_path, script, play_ticks=(100, 102))
    [err] = res.records["frame_error"]
    assert err["error"] == "RuntimeError: perception boom" and err["count"] == 1
    [tap] = res.records["tap"]
    assert tap["tick"] == 102 and len(res.tapper.calls) == 1
    assert res.runner.errors["frame_error"] == 1 and "decide_error" not in res.records


def test_a_decision_error_never_taps_that_frame(tmp_path):
    res = run_script(tmp_path, standard_script(),
                     decider=make_decider(FakePolicy((100, 102)), fail_at=100))
    [err] = res.records["decide_error"]
    assert err["error"] == "RuntimeError: state boom" and err["tick"] == 100
    assert [t["tick"] for t in res.records["tap"]] == [102] and len(res.tapper.calls) == 1


def test_stale_pixels_are_not_decided_on(tmp_path):
    """Pixels 0.8 s old (an occluded window, a stalled adb) at tick 100: no decision there."""
    script = standard_script()
    script[FRAME_OF_TICK[100]][1].stale_s = 0.8
    res = run_script(tmp_path, script, play_ticks=(100, 102))
    assert [t["tick"] for t in res.records["tap"]] == [102]
    decided = [f["extra"]["decision"]["tick"] for f in res.records["frame"] if "decision" in f["extra"]]
    assert 100 not in decided and 102 in decided


def test_a_failed_tap_is_logged_and_not_counted(tmp_path):
    res = run_script(tmp_path, standard_script(), tapper=FakeTapper(fail=True), play_ticks=(100,))
    assert res.runner.taps == 0 and "tap" not in res.records
    assert res.records["tap_error"][0]["error"] == "OSError: pipe gone"
    [d] = decisions(res.records)
    assert d["action"] == CT.ACTION_SKIPPED and d["why"].startswith("tap failed")


def test_a_changed_screen_vetoes_the_tap(tmp_path):
    """The last look before the tap of tick 100 finds the end screen (the battle ended between
    capture and tap): nothing is tapped there, the decision is "skipped" with the reason and a
    tap_vetoed record; the play at tick 102, whose screen is fine, taps."""
    vetoed_t = t_of(FRAME_OF_TICK[100])
    looked = []

    def guard(frame):
        looked.append(frame.t)
        return "the end screen is up" if frame.t == vetoed_t else None

    res = run_script(tmp_path, standard_script(), play_ticks=(100, 102), tap_guard=guard)
    assert looked == [vetoed_t, t_of(FRAME_OF_TICK[102])]       # asked only before a tap
    assert [t["tick"] for t in res.records["tap"]] == [102] and len(res.tapper.calls) == 1
    first, second = decisions(res.records)
    assert (first["tick"], first["action"], first["why"]) == (100, CT.ACTION_SKIPPED,
                                                              "not tapped: the end screen is up")
    assert (second["tick"], second["action"]) == (102, CT.ACTION_TAP)
    [veto] = res.records["tap_vetoed"]
    assert (veto["tick"], veto["name"], veto["reason"]) == (100, "HogRider", "the end screen is up")
    assert res.runner.vetoed == 1 and res.records["session_end"][0]["vetoed"] == 1


def test_a_guard_that_fails_vetoes_and_a_dry_run_never_asks_it(tmp_path):
    def broken(frame):
        raise RuntimeError("grab failed")

    res = run_script(tmp_path, standard_script(), play_ticks=(100,), tap_guard=broken)
    assert res.tapper.calls == [] and "tap" not in res.records
    [d] = decisions(res.records)
    assert d["action"] == CT.ACTION_SKIPPED and d["why"].startswith("not tapped: screen check failed")
    res = run_script(tmp_path / "dry", standard_script(), taps=False, play_ticks=(100,), tap_guard=broken)
    assert "tap_vetoed" not in res.records and decisions(res.records)[0]["action"] == CT.ACTION_DRY


class LatestSource:
    """capture.LatestFrameSource's read(): the newest frame, not consumed."""

    def __init__(self, frame=None, fail=False):
        self.frame, self.fail, self.reads = frame, fail, 0

    def read(self):
        self.reads += 1
        if self.fail:
            raise RuntimeError("grab failed")
        return self.frame


def test_screen_guard_looks_at_the_newest_frame():
    decided = Frame(5.0, battle_image(), 0.0)
    black = np.zeros((400, 225, 3), dtype=np.uint8)
    assert VP.ScreenGuard(LatestSource(None))(decided) is None                         # nothing yet
    assert VP.ScreenGuard(LatestSource(Frame(5.0, end_image(), 0.0)))(decided) is None  # not newer
    assert VP.ScreenGuard(LatestSource(Frame(5.02, battle_image(), 0.0)))(decided) is None
    big = Frame(5.02, battle_image((1080, 1920)), 0.0)                                   # the adb fallback's size
    assert VP.ScreenGuard(LatestSource(big))(decided) is None
    guard = VP.ScreenGuard(LatestSource(Frame(5.02, end_image(), 0.0)))
    assert guard(decided) == "the end screen is up" and guard.vetoes == 1
    assert VP.ScreenGuard(LatestSource(Frame(5.02, black, 0.0)))(decided) == "the battle screen is gone"
    assert VP.ScreenGuard(LatestSource(fail=True))(decided) == "no newest frame (RuntimeError: grab failed)"


def test_opponent_plays_reach_the_memory_at_the_tick_they_are_reported(tmp_path):
    """A Giant reported at tick 104 is in view(now, 0) from the tick-104 decision on, and its
    card is passed on as a known opponent card; the decision at 102 did not see it."""
    giant = OppPlay(card=10, tick=104, exec_tick=104, x_mt=4000.0, y_mt=8000.0)
    captured = {}
    run_script(tmp_path, standard_script(opp_at={104: [giant]}),
               decider=make_decider(FakePolicy(()), captured=captured))
    views = dict(captured["views"])
    assert views[102]["last"][0] == [] and views[102]["seen"][0] == set()
    assert views[104]["last"][0] == [(12, 104)] and views[104]["seen"][0] == {12}
    assert views[120]["last"][0] == [(12, 104)]
    known = dict(captured["opp_cards"])
    assert known[102] == [] and known[104] == [10]


def test_status_line_every_second(tmp_path):
    res = run_script(tmp_path, standard_script(), play_ticks=(100,))
    status = [line for line in res.lines if line.startswith("[")]
    assert len(status) == 3                         # at 11.0, 12.0 and 13.0 s
    assert status[0].startswith("[battle] tick 86 | elixir 7.0 | ")
    assert status[1].startswith(
        "[battle 1] tick 106 | elixir 7.0 | hand Log/HogRider/Skeletons/IceSpirits next Cannon | "
        "AUTO(tap) | last HogRider -> (13, 22) [tap] @100 | 10 fps, 1 taps")
    assert status[2].startswith("[end] tick 122")


def test_frames_and_images_are_recorded(tmp_path):
    """Battle and end-screen frames are written (31), menu frames only every 20th; an image every
    5 written frames."""
    res = run_script(tmp_path, standard_script(), image_every=5)
    assert len(res.records["frame"]) == 31
    assert len(list((res.session / "images").glob("*.npy"))) == 7
    first = res.records["frame"][0]
    assert first["frame"]["screen"] == SCREEN_BATTLE and first["extra"]["state"] == VP.IDLE


# ---------------------------------------------------------------------------
# screen cues and perception
# ---------------------------------------------------------------------------

def _fill(img, box, rgb):
    h, w = img.shape[:2]
    img[int(box[1] * h):int(math.ceil(box[3] * h)), int(box[0] * w):int(math.ceil(box[2] * w))] = rgb


def battle_image(size=(225, 400)):
    """Black, with the bottom panel's blue top edge (30, 144, 230), the owner's colour."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    _fill(img, (0.0, 0.805, 1.0, 0.83), (30, 144, 230))
    return img


def end_image(size=(225, 400)):
    """The end screen's red and blue banners, the OK button and one gold crown on our side
    (the owner's colours: (157, 43, 68), (64, 110, 169), (100, 178, 241))."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    _fill(img, (0.08, 0.24, 0.92, 0.34), (157, 43, 68))
    _fill(img, (0.08, 0.54, 0.92, 0.63), (64, 110, 169))
    _fill(img, (0.37, 0.875, 0.63, 0.94), (100, 178, 241))
    _fill(img, (0.17, 0.45, 0.35, 0.52), (230, 170, 40))
    return img


@pytest.mark.parametrize("size", [(225, 400), (450, 800)])
def test_screen_cues(size):
    black = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    c = VP.read_cues(black)
    assert not c.hud_present and not c.end_screen
    c = VP.read_cues(battle_image(size))
    assert c.panel == 1.0 and c.hud_present and not c.end_screen
    c = VP.read_cues(end_image(size))
    assert c.end_screen and not c.hud_present and c.panel == 0.0
    assert VP.end_screen_crowns(end_image(size)) == (1, 0)
    assert VP.end_screen_crowns(battle_image(size)) == (0, 0)


def test_battle_outcome():
    assert [VP.battle_outcome(*c) for c in ((1, 0), (0, 2), (1, 1), (None, 0))] == [
        "win", "loss", "draw", "unknown"]


class FakeHudReader:
    def __init__(self, ticks):
        self.ticks = iter(ticks)
        self.reads = 0
        self.resets = 0

    def read(self, rgb, t):
        self.reads += 1
        return Hud(hand=list(HAND), elixir_milli=5000, clock_tick=next(self.ticks), phase=PHASE_REGULATION)

    def reset(self):
        self.resets += 1


class FakeDetector:
    def __init__(self, dets, size):
        self.dets = list(dets)
        self.size = size
        self.calls = 0

    def detect(self, rgb):
        self.calls += 1
        dets = list(self.dets)
        return SimpleNamespace(detections=dets, frame_size=self.size, units=lambda: list(dets),
                               spells=lambda: [], bars=lambda: [])


def test_live_perception_reads_the_screen_and_tracks_units():
    """Menu and end-screen frames read no HUD and run no detector; battle frames do. A Knight
    (not a Hog 2.6 card: an enemy by the deck rule) seen twice becomes a track at its ground
    point and an opponent play of card 8, first seen at tick 100."""
    from scripts.vision.tracker import Tracker
    size = (225, 400)
    tables = EV.EventTables({8: {"Knight": 1}})
    deriver = EV.EventDeriver(tables, own_deck=range(8))
    tracker = Tracker(own_classes={"HogRider"})
    hud = FakeHudReader([100, 102])
    gx, gy = GEOMETRY.world_to_px(9000.0, 10000.0, size)
    knight = Detection("Knight", 0.9, 0.9, (gx - 10.0, gy - 30.0, gx + 10.0, gy), (float(gx), float(gy)))
    detector = FakeDetector([knight], size)
    p = VP.LivePerception(hud_reader=hud, tracker=tracker, deriver=deriver, geometry=GEOMETRY,
                          detector=detector)
    menu = p.perceive(Frame(1.0, np.zeros((400, 225, 3), dtype=np.uint8), 0.0))
    assert menu.screen == SCREEN_MENU and menu.tick is None and menu.frame_size == size
    end = p.perceive(Frame(1.1, end_image(size), 0.0))
    assert end.screen == SCREEN_END and end.hud.crowns == (1, 0) and end.hud.phase == PHASE_ENDED
    assert hud.reads == 0 and detector.calls == 0
    a = p.perceive(Frame(2.0, battle_image(size), 0.0))
    assert a.screen == SCREEN_BATTLE and a.tick == 100 and a.tracks == [] and detector.calls == 1
    b = p.perceive(Frame(2.1, battle_image(size), 0.0))
    assert [(tr.uid, tr.cls, tr.team, tr.first_seen_tick) for tr in b.tracks] == [(1, "Knight", ENEMY_TEAM, 100)]
    assert abs(b.tracks[0].x_mt - 9000.0) < 1.0 and abs(b.tracks[0].y_mt - 10000.0) < 1.0
    assert [(op.card, op.tick, op.exec_tick) for op in b.events.opp_plays] == [(8, 100, 100)]
    assert b.detections == [knight] and p.track_cards(b.tracks) == {1: 8}
    p.note_own_tap(2, 104, slot=1, aim_mt=(13500, 22500), t=2.2)
    assert [(tp.card, tp.tick, tp.slot, tp.aim) for tp in deriver.pending_taps] == [(2, 104, 1, (13500.0, 22500.0))]
    p.end_battle()
    assert hud.resets == 1 and tracker.tracks() == [] and deriver.pending_taps == []


def test_live_perception_without_detector_reads_the_hud_only():
    hud = FakeHudReader([100])
    p = VP.LivePerception(hud_reader=hud, tracker=None, deriver=None, geometry=GEOMETRY, detector=None)
    vf = p.perceive(Frame(1.0, battle_image(), 0.0))
    assert vf.screen == SCREEN_BATTLE and vf.tick == 100 and vf.tracks == [] and vf.hud.hand == HAND


# ---------------------------------------------------------------------------
# battle state and results
# ---------------------------------------------------------------------------

def vf_at(t, tick, screen=SCREEN_BATTLE, phase=PHASE_REGULATION):
    return VisionFrame(t_capture=t, tick=tick, hud=Hud(clock_tick=tick, phase=phase), screen=screen)


def started(b, t=20.0, first=100):
    """Feed one running second (ticks first..first+20); returns the time of the last frame."""
    events = [b.update(vf_at(t + 0.05 * i, first + i)) for i in range(21)]
    assert events[-1] == ["start"] and b.in_battle
    return t + 0.05 * 20


def test_battle_state_starts_only_on_a_running_clock():
    b = VP.BattleState()
    # a clock frozen at the end of its second (the 3:00 intro, or an end screen behind a panel)
    assert all(b.update(vf_at(10.0 + 0.05 * i, 19)) == [] for i in range(60))
    assert b.update(vf_at(13.0, None, SCREEN_MENU)) == [] and not b.in_battle
    events = [b.update(vf_at(20.0 + 0.05 * i, 30 + i)) for i in range(25)]
    assert events.index(["start"]) == 20 and events.count(["start"]) == 1   # tick 50 = 30 + 20
    assert b.number == 1 and b.last_tick == 54 and b.running(20.0 + 0.05 * 24)


def test_battle_state_start_list_stays_bounded_on_a_frozen_clock():
    """20 s of a clock frozen on a battle screen at 50 fps keep the start list bounded; once the
    clock runs the battle starts exactly as before (20 ticks past the frozen reading)."""
    b = VP.BattleState()
    assert all(b.update(vf_at(10.0 + 0.02 * i, 19)) == [] for i in range(1000))
    assert len(b._cand) <= 256
    events = [b.update(vf_at(30.0 + 0.05 * i, 20 + i)) for i in range(25)]
    assert events.index(["start"]) == 19 and events.count(["start"]) == 1 and b.number == 1


def test_battle_state_end_rules():
    # a clock that is read but stops: over after 2.5 s; running() goes false after 1.2 s
    b = VP.BattleState()
    t = started(b)
    assert b.running(t + 1.0) and not b.running(t + 1.5)
    assert b.update(vf_at(t + 2.0, 120)) == []
    assert b.update(vf_at(t + 3.0, 120)) == ["end"] and b.ended_reason == "clock stopped"
    # an unread clock ends nothing; 8 s off the battle screen does
    b = VP.BattleState()
    t = started(b)
    assert all(b.update(vf_at(t + 0.5 * k, None)) == [] for k in range(1, 11))     # 5 s unread
    t += 5.0
    assert b.update(vf_at(t + 7.5, None, SCREEN_MENU)) == []
    assert b.update(vf_at(t + 8.5, None, SCREEN_MENU)) == ["end"]
    assert b.ended_reason == "left the battle screen"
    # a clock that jumps back: a new battle; it can start right after
    b = VP.BattleState()
    t = started(b, first=1000)
    assert b.update(vf_at(t + 0.05, 30)) == ["end"] and b.ended_reason == "clock reset"
    assert started(b, t=t + 0.1, first=31) and b.number == 2
    # the end screen, three frames; overtime 0:00 at once
    b = VP.BattleState()
    t = started(b)
    assert [b.update(vf_at(t + 0.05 * k, None, SCREEN_END)) for k in (1, 2, 3)] == [[], [], ["end"]]
    assert b.ended_reason == "end screen"
    b = VP.BattleState()
    t = started(b)
    assert b.update(vf_at(t + 0.05, 121, phase=PHASE_ENDED)) == ["end"] and b.ended_reason == "overtime over"


def test_result_book():
    book = VP.ResultBook()
    book.open(1, "end screen", 50.0, (1, 0), towers={})
    book.end_frame((0, 0), 50.1)
    book.end_frame((1, 0), 50.2)
    assert not book.due(SCREEN_END, 52.0) and book.due(SCREEN_END, 53.2) and book.due(SCREEN_MENU, 50.3)
    rec = book.finish()
    assert (rec["outcome"], rec["crowns"], rec["end_frames"], rec["source"]) == ("win", [1, 0], 2, "end screen + hud")
    book.open(2, "clock stopped", 100.0, (0, 1))
    assert not book.due(SCREEN_MENU, 119.0) and book.due(SCREEN_MENU, 120.0)
    rec = book.finish()
    assert (rec["outcome"], rec["source"], rec["record"]) == ("loss", "hud", {"win": 1, "loss": 1})
    assert book.done == 2 and book.finish() is None


# ---------------------------------------------------------------------------
# taps, the lock, the command line
# ---------------------------------------------------------------------------

class FakeStdin:
    def __init__(self):
        self.data = b""
        self.closed = False

    def write(self, b):
        if self.closed:
            raise ValueError("closed")
        self.data += b

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeAdb:
    """Stands in for subprocess.run (wm size) and subprocess.Popen (the shell); logs the order."""

    def __init__(self, wm="Physical size: 1080x1920\n"):
        self.wm = wm
        self.log = []
        self.procs = []
        self.refuse = False

    def verify(self, serial, adb):
        self.log.append(("verify", serial, adb))
        if self.refuse:
            raise SystemExit("refusing to run: com.supercell.clashroyale is not installed")

    def run(self, cmd, **kw):
        self.log.append(("run", list(cmd)))
        return SimpleNamespace(stdout=self.wm, returncode=0)

    def popen(self, cmd, **kw):
        self.log.append(("popen", list(cmd)))
        proc = SimpleNamespace(cmd=list(cmd), kw=kw, stdin=FakeStdin(), poll=lambda: None,
                               terminate=lambda: None, wait=lambda timeout=None: 0)
        self.procs.append(proc)
        return proc


def test_adb_tapper_verifies_first_and_writes_tap_lines():
    adb = FakeAdb()
    tapper = VP.AdbTapper("adb.exe", "fake-serial", verify=adb.verify, popen=adb.popen, run=adb.run)
    assert [entry[0] for entry in adb.log] == ["verify", "run", "popen"]
    assert adb.log[0] == ("verify", "fake-serial", "adb.exe")
    assert adb.log[1][1] == ["adb.exe", "-s", "fake-serial", "shell", "wm", "size"]
    [proc] = adb.procs
    assert proc.cmd == ["adb.exe", "-s", "fake-serial", "shell"]
    assert proc.kw["stdin"] == subprocess.PIPE and "text" not in proc.kw and "encoding" not in proc.kw
    tapper.tap_pair((538, 1711), (766, 1083))
    assert proc.stdin.data == b"input tap 538 1711\ninput tap 766 1083\n" and tapper.sent == 1
    with pytest.raises(VP.TapError):
        tapper.tap_pair((538, 1711), (1080, 500))           # off the 1080x1920 screen
    assert proc.stdin.data == b"input tap 538 1711\ninput tap 766 1083\n"
    tapper.close()
    assert proc.stdin.closed


def test_adb_tapper_refuses_without_verify_or_on_the_wrong_device():
    adb = FakeAdb()
    with pytest.raises(VP.TapError):
        VP.AdbTapper("adb", "s", verify=None, popen=adb.popen, run=adb.run)

    def refuse(serial, adb_path):
        raise SystemExit("refusing to run: not the Clash Royale instance")

    with pytest.raises(SystemExit):
        VP.AdbTapper("adb", "s", verify=refuse, popen=adb.popen, run=adb.run)
    small = FakeAdb(wm="Physical size: 1080x1920\nOverride size: 720x1280\n")
    with pytest.raises(VP.TapError, match="720x1280"):
        VP.AdbTapper("adb", "s", verify=small.verify, popen=small.popen, run=small.run)
    assert adb.procs == [] and small.procs == []


def test_adb_tapper_verifies_again_before_reopening_a_dead_shell():
    """The shell died (adb server restart): verify, then a new shell, then the tap. When verify
    refuses at a reopen, no shell is opened and no tap is ever sent again."""
    adb = FakeAdb()
    tapper = VP.AdbTapper("adb", "s", verify=adb.verify, popen=adb.popen, run=adb.run)
    first = adb.procs[0]
    first.poll = lambda: 1                                  # exited
    tapper.tap_pair((538, 1711), (766, 1083))
    assert [entry[0] for entry in adb.log] == ["verify", "run", "popen", "verify", "popen"]
    second = adb.procs[1]
    assert first.stdin.closed and first.stdin.data == b""
    assert second.stdin.data == b"input tap 538 1711\ninput tap 766 1083\n" and tapper.opened == 2
    second.poll = lambda: 1
    adb.refuse = True
    with pytest.raises(VP.TapError, match="no more taps"):
        tapper.tap_pair((538, 1711), (766, 1083))
    assert tapper.dead and len(adb.procs) == 2 and second.stdin.data.count(b"input tap") == 2
    adb.refuse = False
    with pytest.raises(VP.TapError):                        # dead for good: not even verify again
        tapper.tap_pair((538, 1711), (766, 1083))
    assert [entry[0] for entry in adb.log].count("verify") == 3 and len(adb.procs) == 2
    tapper.close()


def test_parse_wm_size():
    assert VP.parse_wm_size("Physical size: 1080x1920\r\n") == (1080, 1920)
    assert VP.parse_wm_size("Physical size: 1080x1920\nOverride size: 720x1280") == (720, 1280)
    assert VP.parse_wm_size("") is None and VP.parse_wm_size("error: device offline") is None


def test_instance_lock_is_exclusive(tmp_path):
    path = tmp_path / "vision_play.lock"
    first, second = VP.InstanceLock(path), VP.InstanceLock(path)
    assert first.acquire() and first.held
    assert "pid" in first.holder()
    assert not second.acquire() and not second.held
    first.release()
    assert not first.held
    assert second.acquire()
    second.release()


def test_command_line_defaults_are_safe():
    args = VP.build_parser().parse_args([])
    assert args.tap is False and args.start_mode == CT.MANUAL
    assert args.deck == VP.HOG26 and set(args.deck) == {"Cannon", "Fireball", "HogRider", "IceGolemite",
                                                        "IceSpirits", "Musketeer", "Skeletons", "Log"}
    assert (args.ckpt, args.detector, args.source) == (None, None, "auto")
    assert (args.seconds, args.battles, args.record_images) == (0.0, 0, 20)
    assert args.no_control_window is False and args.hotkey == "F8"
    assert VP.build_parser().parse_args(["--no-window"]).no_control_window is True
    assert VP.build_parser().parse_args(["--no-control-window"]).no_control_window is True
    assert VP.build_parser().parse_args(["--deck", "hog 2.6"]).deck == VP.HOG26
    with pytest.raises(SystemExit):
        VP.build_parser().parse_args(["--deck", "HogRider,Log"])
    assert VP.default_ckpt().as_posix().endswith("tokens/live_big_hog/best.pt")


def test_main_refuses_while_stop_exists_or_another_runner_holds_the_lock(tmp_path, monkeypatch, capsys):
    stop, lock = tmp_path / "STOP", tmp_path / "vision_play.lock"
    monkeypatch.setattr(VP, "STOP_FILE", stop)
    monkeypatch.setattr(VP, "LOCK_FILE", lock)
    stop.write_text("", encoding="utf-8")
    assert VP.main([]) == 2
    assert "refusing to start" in capsys.readouterr().out and not lock.exists()
    stop.unlink()
    holder = VP.InstanceLock(lock)
    assert holder.acquire()
    try:
        assert VP.main([]) == 3
        assert "another vision_play is running" in capsys.readouterr().out
    finally:
        holder.release()


# ---------------------------------------------------------------------------
# the memory reader is never loaded
# ---------------------------------------------------------------------------

FORBIDDEN = ("gamestate", "memread", "cr_bot", "live_bridge", "live_play", "hcr_reader")


def test_vision_play_imports_no_memory_reader():
    tree = ast.parse((ROOT / "scripts" / "vision_play.py").read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            names += [f"{node.module}.{a.name}" for a in node.names]
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and node.value.startswith(("scripts.", "gamestate"))):
            names.append(node.value)               # import_live_modules' names
    assert names and not [n for n in names if any(bad in n for bad in FORBIDDEN)]
    assert "scripts.vision.detector" in names and "scripts.emulator" in names


def test_running_vision_play_never_loads_scripts_gamestate():
    """A fresh interpreter imports the runner and every module main() uses, and builds the
    parser; no module of scripts/gamestate (or the native reader) is loaded."""
    code = "\n".join([
        "import json, sys",
        f"sys.path.insert(0, {str(ROOT)!r})",
        "import scripts.vision_play as V",
        "mods = V.import_live_modules()",
        "V.build_parser().parse_args([])",
        f"bad = sorted(m for m in sys.modules if any(b in m for b in {FORBIDDEN!r}))",
        "print(json.dumps({'bad': bad, 'mods': len(mods), 'torch': 'torch' in sys.modules}))",
    ])
    res = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                         text=True, timeout=180)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout.strip().splitlines()[-1]) == {"bad": [], "mods": 15, "torch": False}
