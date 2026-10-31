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
        self.kept_towers = []
        self.own_taps = []

    def perceive(self, frame):
        self.seen.append(frame.t)
        item = self.script[frame.t]
        if isinstance(item, BaseException):
            raise item
        return item

    def new_battle(self):
        self.new_battles += 1

    def end_battle(self, *, keep_towers=False):
        self.end_battles += 1
        self.kept_towers.append(keep_towers)

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


def make_decider(policy, *, captured=None, fail_at=None, refusals=None):
    """A Decider on a minimal state built from the HUD; row_fn records (tick, view, opp cards).
    With `refusals` (a RefusedTapGuard) its legality is RefusalLegality over the accept-all one."""

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

    legality = lambda slot, card, x, y: True  # noqa: E731
    if refusals is not None:
        legality = VP.RefusalLegality(legality, refusals)
    return VP.Decider(policy=policy, legality=legality,
                      config=RD.DecideConfig.live(0.0), state_fn=state_fn, row_fn=row_fn,
                      rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# scripted frames
# ---------------------------------------------------------------------------

def battle_vf(t, tick, *, opp=(), own=(), hand=None, elixir=7000):
    hud = Hud(hand=list(HAND if hand is None else hand), next=0, elixir_milli=elixir, clock_tick=tick,
              phase=PHASE_REGULATION)
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
               size=(225, 400), on_read=None, decider=None, image_every=0, tap_guard=None,
               perception=None, refusals=None, control=None):
    """Runs `script` (see the module doc). refusals: a RefusedTapGuard for the runner and the
    default decider's legality (None: no guard, as before). control: a dict that receives the
    ControlState as "state" before the run (for on_read hooks that press F8)."""
    clock = FakeClock()
    rgb = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frames = [Frame(t, rgb, 0.0) for t, _ in script]
    perception = FakePerception(script) if perception is None else perception
    ctl = CT.ControlState(mode, taps_enabled=taps, stop_file=tmp_path / "STOP", control_file=None,
                          names=NAMES, clock=clock)
    if control is not None:
        control["state"] = ctl
    control = ctl
    recorder = VisionRecorder(tmp_path / "rec", session="s", image_every=image_every, image_format="npy",
                              clock=clock, wall_clock=lambda: 0.0)
    tapper = FakeTapper() if tapper is None else tapper
    lines = []
    runner = VP.VisionPlay(source=FakeSource(frames, clock, on_read), perception=perception,
                           decider=decider or make_decider(FakePolicy(play_ticks), refusals=refusals),
                           control=control, recorder=recorder, geometry=GEOMETRY, tapper=tapper,
                           card_names=NAMES, clock=clock, sleep=lambda s: None, out=lines.append,
                           tap_guard=tap_guard, refusals=refusals)
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
    # the bot taps: no second look on its waits (that is for MANUAL and dry runs)
    assert not any("would" in f.get("extra", {}) for f in recs["frame"])
    assert recs["frame_size"][0]["size"] == [225, 400]          # the scripted frames' own size
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


def test_status_line_shows_the_towers_the_bot_plays_on(tmp_path):
    """The runbook has the owner hand back to AUTO only once the status line's towers look right
    (a misread 0 opens a pocket: the 2026-10-31 failure): the princess towers' HP as read, "?"
    unread, and the crowns; nothing when no tower is read (test_status_line_every_second)."""
    from scripts.vision.types import TowerHp
    script = standard_script()
    for _, vf in script:
        if isinstance(vf, VisionFrame) and vf.screen == VP.SCREEN_BATTLE:
            vf.hud.towers = {"enemy_left": TowerHp(3052, 3052), "enemy_right": TowerHp(0, 3052),
                             "own_left": TowerHp(2101, 3052), "own_right": TowerHp(None, None),
                             "enemy_king": TowerHp(4824, 4824), "own_king": TowerHp(4824, 4824)}
            vf.hud.crowns = (1, 0)
    res = run_script(tmp_path, script, play_ticks=(100,))
    status = [line for line in res.lines if line.startswith("[battle 1]")]
    assert status and status[0].startswith(
        "[battle 1] tick 106 | elixir 7.0 | hand Log/HogRider/Skeletons/IceSpirits next Cannon | "
        "towers them 3052 0 us 2101 ? crowns 1-0 | AUTO(tap) | last HogRider")
    assert VP.towers_text(Hud()) == ""
    assert VP.towers_text(Hud(towers={"own_left": TowerHp(3052, 3052)})) == (
        " | towers them ? ? us 3052 ? crowns 0-0")


def test_human_plays_are_compared_with_what_the_bot_would_play(tmp_path):
    """MANUAL: nearly every decision is a wait, so on a wait (at most every 10 ticks) the bot
    also works out what it would play now: the gate's best playable card at its best tile, on
    its own rng. It is never tapped or logged as a decision; the human play is compared with
    it, and the frames and the status line show it."""
    musk = OwnLanding(card=5, slot=-1, tap_tick=120, land_tick=120, x_mt=9000.0, y_mt=25000.0, uid=7)
    res = run_script(tmp_path, standard_script(own_at={120: [musk]}), mode=CT.MANUAL, taps=True,
                     play_ticks=(100,))
    assert res.tapper.calls == [] and [d["tick"] for d in decisions(res.records)] == [100]
    would = [(f["extra"]["decision"]["tick"], f["extra"]["would"]) for f in res.records["frame"]
             if "would" in f.get("extra", {})]
    assert [t for t, _ in would] == [90, 102, 112]
    w = would[0][1]
    assert (w["card"], w["name"], w["slot"], w["tile"], w["reason"], w["gen"]) == (
        7, "Log", 0, [13, 22], "would", -1)
    [human] = res.records["human_play"]
    assert human["bot"]["card"] == -1 and human["same_card"] is False      # the wait at tick 94
    assert (human["would"]["card"], human["would"]["tick"], human["would_age_ticks"]) == (7, 90, 4)
    assert human["would_same_card"] is False and human["would_tile_dist"] == 5.15
    assert any("| would Log -> (13, 22) @" in line for line in res.lines)


def drag_script(land_tick=126):
    """standard_script's battle stretched to tick 140: from tick 98 the Log's slot (0) reads
    empty, the human dragging it, and the Log lands at `land_tick` (est. tap 26 ticks before)."""
    log = OwnLanding(card=7, slot=-1, tap_tick=land_tick, land_tick=land_tick, x_mt=4500.0,
                     y_mt=25000.0, uid=9)
    items = [menu_vf(t_of(0)), menu_vf(t_of(1))]
    for k in range(36):
        i, tick = 2 + k, 70 + 2 * k
        vf = battle_vf(t_of(i), tick, own=[log] if tick == land_tick else ())
        if tick >= 98:
            vf.hud.hand = [VP.EMPTY_CARD, 2, 6, 4]
        items.append(vf)
    items += [end_vf(t_of(38 + k), tick=142) for k in range(5)]
    items += [menu_vf(t_of(43 + k)) for k in range(3)]
    return [(t_of(i), item) for i, item in enumerate(items)]


def test_a_human_play_is_compared_with_a_would_play_that_could_choose_its_card(tmp_path):
    """The HUD slot empties when the drag starts, before the card is placed: the would-play
    made during the drag (tick 100, the Log's slot masked: it would play the Hog) never had
    the Log to choose. The human Log is compared with the newest would-play whose hand still
    held it (tick 90: the Log), which each would-play keeps in its extra."""
    res = run_script(tmp_path, drag_script(), mode=CT.MANUAL, taps=True, play_ticks=())
    would = {f["extra"]["decision"]["tick"]: f["extra"]["would"] for f in res.records["frame"]
             if "would" in f.get("extra", {})}
    assert sorted(would)[:3] == [90, 100, 110]
    assert (would[90]["card"], would[90]["extra"]["hand"]) == (7, [7, 2, 6, 4])
    assert (would[100]["card"], would[100]["extra"]["hand"]) == (2, [VP.EMPTY_CARD, 2, 6, 4])
    [human] = res.records["human_play"]
    assert (human["card"], human["tap_tick"], human["bot"]["card"]) == (7, 100, -1)
    assert (human["would"]["tick"], human["would"]["card"], human["would_age_ticks"]) == (90, 7, 10)
    assert human["would_same_card"] is True


def test_the_start_frames_events_are_not_recorded_twice(tmp_path):
    """The frame a battle starts on carries what the deriver reported BEFORE the start reset
    it; the fresh deriver reports what is still on the board again, so that frame's events are
    dropped."""
    giant = OppPlay(card=10, tick=90, exec_tick=90, x_mt=4000.0, y_mt=8000.0)
    captured = {}
    res = run_script(tmp_path, standard_script(opp_at={90: [giant]}),
                     decider=make_decider(FakePolicy(()), captured=captured))
    views = dict(captured["views"])
    assert views[90]["last"][0] == [] and views[120]["last"][0] == []
    assert res.records["battle_end"][0]["opp_plays"] == 0


class FlakyDetectorPerception(FakePerception):
    """FakePerception whose detector fails on the frames of `fail_ticks`, as LivePerception
    counts it (detect_errors, last_detect_error)."""

    def __init__(self, script, fail_ticks):
        super().__init__(script)
        self.fail_ticks = set(fail_ticks)
        self.detect_errors = 0
        self.last_detect_error = None

    def perceive(self, frame):
        vf = super().perceive(frame)
        if vf.tick in self.fail_ticks:
            self.detect_errors += 1
            self.last_detect_error = RuntimeError("CUDA out of memory")
        return vf


def test_a_failing_detector_is_logged_and_the_frame_still_decides(tmp_path):
    script = standard_script()
    res = run_script(tmp_path, script, play_ticks=(100,),
                     perception=FlakyDetectorPerception(script, {100}))
    [err] = res.records["detect_error"]
    assert err["error"] == "RuntimeError: CUDA out of memory" and err["tick"] == 100
    assert [t["tick"] for t in res.records["tap"]] == [100]


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


def test_live_perception_resets_the_hud_only_outside_a_battle():
    """A destroyed tower reads as rubble (no digits, no bar), which a fresh tower filter takes
    for a full tower: seconds without the battle screen reset the HUD only outside a battle."""
    hud = FakeHudReader([100, 200, 300])
    p = VP.LivePerception(hud_reader=hud, tracker=None, deriver=None, geometry=GEOMETRY, detector=None)
    p.perceive(Frame(1.0, battle_image(), 0.0))
    p.new_battle()
    p.perceive(Frame(10.0, battle_image(), 0.0))         # 9 s without the battle screen, in a battle
    assert hud.resets == 0 and p.primed_until is None
    p.end_battle()
    assert hud.resets == 1
    p.perceive(Frame(20.0, battle_image(), 0.0))         # outside a battle: back after 10 s
    assert hud.resets == 2


def test_live_perception_primes_the_board_of_a_battle_joined_in_the_middle():
    """A battle seen starting at tick 5000 was joined in the middle: the Knight on the board then
    is not an opponent play. At a clean start (tick 100) the same Knight is one."""
    from scripts.vision.tracker import Tracker
    size = (225, 400)
    gx, gy = GEOMETRY.world_to_px(9000.0, 10000.0, size)
    knight = Detection("Knight", 0.9, 0.9, (gx - 10.0, gy - 30.0, gx + 10.0, gy), (float(gx), float(gy)))

    def run(first):
        p = VP.LivePerception(hud_reader=FakeHudReader([first, first + 2, first + 4]),
                              tracker=Tracker(own_classes={"HogRider"}),
                              deriver=EV.EventDeriver(EV.EventTables({8: {"Knight": 1}}), own_deck=range(8)),
                              geometry=GEOMETRY, detector=FakeDetector([knight], size))
        p.perceive(Frame(1.0, battle_image(size), 0.0))
        p.new_battle()                                   # the battle starts on this frame
        plays = [op for i in (1, 2)
                 for op in p.perceive(Frame(1.0 + 0.1 * i, battle_image(size), 0.0)).events.opp_plays]
        return p, plays

    joined, plays = run(5000)
    assert joined.primed_until == 5000 + VP.PRIME_TICKS and plays == []
    clean, plays = run(100)
    assert clean.primed_until is None and [(op.card, op.tick) for op in plays] == [(8, 102)]


def test_live_perception_survives_a_failing_detector():
    from scripts.vision.tracker import Tracker

    class Broken:
        def detect(self, rgb):
            raise RuntimeError("CUDA out of memory")

    p = VP.LivePerception(hud_reader=FakeHudReader([100]), tracker=Tracker(own_classes={"HogRider"}),
                          deriver=EV.EventDeriver(EV.EventTables({}), own_deck=range(8)),
                          geometry=GEOMETRY, detector=Broken())
    vf = p.perceive(Frame(1.0, battle_image(), 0.0))
    assert vf.screen == SCREEN_BATTLE and vf.tick == 100 and vf.hud.hand == HAND and vf.tracks == []
    assert p.detect_errors == 1 and str(p.last_detect_error) == "CUDA out of memory"


class TwoPhaseHud(FakeHudReader):
    """FakeHudReader read in two phases (HudReader's surface): read(towers=False), then
    read_towers with the detector's tower evidence, which it records."""

    def __init__(self, ticks):
        super().__init__(ticks)
        self.evidence = []
        self.towers_flags = []
        self.report = SimpleNamespace(towers_revived=[])

    def read(self, rgb, t, towers=True):
        self.towers_flags.append(towers)
        return super().read(rgb, t)

    def read_towers(self, rgb, t, hud, evidence=None):
        self.evidence.append(evidence)
        return hud


def test_live_perception_reads_the_towers_with_the_detectors_boxes():
    """FIX A/B: the HUD's towers are read after the detector, with the frame's tower boxes (a
    tower skin's label is read where the detector puts it; a tower it sees is never read
    destroyed). A failing detector or none: read_towers still runs, with None. A HUD reader
    without read_towers is read in one call, as before."""
    from scripts.vision.tower_labels import LABEL_REF_PX, TOWER_BODY_REF_PX
    from scripts.vision.tracker import Tracker

    size = (225, 400)
    sx, sy = size[0] / 1080.0, size[1] / 1920.0

    def scaled(box):
        return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)

    towers = [Detection("PrincessTower", 0.9, 0.9, scaled(TOWER_BODY_REF_PX["enemy_right"])),
              Detection("tower-bar", 0.9, 0.9, scaled(LABEL_REF_PX["enemy_left"]))]

    def perception(hud, detector):
        return VP.LivePerception(hud_reader=hud, tracker=Tracker(own_classes={"HogRider"}),
                                 deriver=EV.EventDeriver(EV.EventTables({}), own_deck=range(8)),
                                 geometry=GEOMETRY, detector=detector)

    hud = TwoPhaseHud([100])
    p = perception(hud, FakeDetector(towers, size))
    vf = p.perceive(Frame(1.0, battle_image(size), 0.0))
    [ev] = hud.evidence
    assert hud.towers_flags == [False] and vf.tick == 100 and p.detect_errors == 0
    assert ev.seen("enemy_right") and ev.seen("enemy_left") and not ev.seen("own_left")
    assert set(ev.placement) == {"enemy_left"}

    class Broken:
        def detect(self, rgb):
            raise RuntimeError("CUDA out of memory")

    hud = TwoPhaseHud([100])
    p = perception(hud, Broken())
    vf = p.perceive(Frame(1.0, battle_image(size), 0.0))
    assert hud.evidence == [None] and p.detect_errors == 1 and vf.hud.hand == HAND
    assert str(p.last_detect_error) == "CUDA out of memory"
    hud = TwoPhaseHud([100])
    p = perception(hud, None)
    p.perceive(Frame(1.0, battle_image(size), 0.0))
    assert hud.evidence == [None] and hud.towers_flags == [False]
    hud = TwoPhaseHud([None])                              # no tick: the detector does not run
    det = FakeDetector(towers, size)
    perception(hud, det).perceive(Frame(1.0, battle_image(size), 0.0))
    assert hud.evidence == [None] and det.calls == 0
    plain = FakeHudReader([100])                            # read(rgb, t) only: one call, as before
    p = perception(plain, FakeDetector(towers, size))
    vf = p.perceive(Frame(1.0, battle_image(size), 0.0))
    assert plain.reads == 1 and vf.tick == 100 and p.detect_errors == 0


def test_live_perception_takes_revived_towers_out_of_the_fallen_ones():
    from scripts.vision.tracker import Tracker

    hud = TwoPhaseHud([3610])
    hud.report.towers_revived = ["enemy_left"]
    d = EV.EventDeriver(EV.EventTables({}), own_deck=range(8))
    d.fallen_towers.update({"enemy_left", "own_right"})
    p = VP.LivePerception(hud_reader=hud, tracker=Tracker(own_classes={"HogRider"}), deriver=d,
                          geometry=GEOMETRY, detector=FakeDetector([], (225, 400)))
    p.perceive(Frame(1.0, battle_image(), 0.0))
    assert d.fallen_towers == {"own_right"}


class TowerReportPerception(FakePerception):
    """FakePerception with a HUD reader whose report moves both enemy labels 36 px up from tick
    80 (the detector) and revives two towers at tick 96 (the overtime check)."""

    def __init__(self, script):
        from scripts.vision import hud as H
        super().__init__(script)
        self.H = H
        info = {"shift_px": {"enemy_left": [0.0, -36.0]}, "source": {"enemy_left": "detector"},
                "holds": {"enemy_left": 12}, "revived": ["enemy_left", "enemy_right"]}
        self.hud_reader = SimpleNamespace(report=None, towers_info=lambda: info)

    def perceive(self, frame):
        vf = super().perceive(frame)
        if vf.screen == SCREEN_BATTLE:
            rep = self.H.HudReport(t=vf.t_capture)
            rep.tower_frac = {"enemy_left": 0.222}
            if vf.tick >= 80:
                rep.tower_shift = {"enemy_left": [0.0, -36.0, "detector"], "enemy_right": [1.4, -36.3, "detector"],
                                   "own_left": [0.0, 0.0, "fixed"]}
            if vf.tick == 96:
                rep.towers_revived = ["enemy_left", "enemy_right"]
                rep.towers_misread = {"crowns_before": [2, 0], "revived": ["enemy_left", "enemy_right"],
                                      "crowns_after": [0, 0]}
            self.hud_reader.report = rep
        return vf


def test_tower_labels_misreads_and_raw_reads_are_logged(tmp_path):
    res = run_script(tmp_path, standard_script(), perception=TowerReportPerception(standard_script()))
    recs = res.records
    labels = recs["tower_label"]
    assert [(r["key"], r["shift_px"], r["source"], r["battle"]) for r in labels] == [
        ("enemy_left", [0.0, -36.0], "detector", 1), ("enemy_right", [1.4, -36.3], "detector", 1)]
    assert labels[0]["tick"] == 90                          # the battle's first frame (from tick 90)
    assert [ln for ln in res.lines if "calibration" in ln] == [
        "enemy labels 36 px above calibration (tower skin; from the detector): HP read there"]
    [mis] = recs["towers_misread"]
    assert (mis["battle"], mis["tick"], mis["crowns_before"], mis["crowns_after"]) == (1, 96, [2, 0], [0, 0])
    assert any(ln.startswith("towers misread: overtime began at 2-0") for ln in res.lines)
    [end] = recs["battle_end"]
    assert end["towers_info"]["holds"] == {"enemy_left": 12} and end["towers_info"]["revived"] == [
        "enemy_left", "enemy_right"]
    raw = [f["extra"].get("towers_raw") for f in recs["frame"] if f["frame"]["screen"] == SCREEN_BATTLE]
    assert raw and all(r is not None and r["frac"] == {"enemy_left": 0.222} for r in raw)
    assert raw[-1]["shift"] == {"enemy_left": [0.0, -36.0, "detector"], "enemy_right": [1.4, -36.3, "detector"]}
    assert all("towers_raw" not in f["extra"] for f in recs["frame"] if f["frame"]["screen"] != SCREEN_BATTLE)


def test_live_perception_survives_a_failing_board_chain():
    """Not only detect(): the HP bars, the ground points, the tracker and the events run on
    the detector's output and have never run on the pinned detector's real frames. A failure
    there costs the frame its units and events; its HUD and tick still count."""

    class BrokenTracker:
        def update(self, obs, tick):
            raise ValueError("bad box")

        def tracks(self):
            return []

        def reset(self):
            pass

    size = (225, 400)
    gx, gy = GEOMETRY.world_to_px(9000.0, 10000.0, size)
    knight = Detection("Knight", 0.9, 0.9, (gx - 10.0, gy - 30.0, gx + 10.0, gy), (float(gx), float(gy)))
    p = VP.LivePerception(hud_reader=FakeHudReader([100]), tracker=BrokenTracker(),
                          deriver=EV.EventDeriver(EV.EventTables({}), own_deck=range(8)),
                          geometry=GEOMETRY, detector=FakeDetector([knight], size))
    vf = p.perceive(Frame(1.0, battle_image(size), 0.0))
    assert vf.screen == SCREEN_BATTLE and vf.tick == 100 and vf.hud.hand == HAND
    assert vf.tracks == [] and vf.events.opp_plays == [] and vf.events.own_landings == []
    assert (p.detect_errors, str(p.last_detect_error), p.detect_frames) == (1, "bad box", 0)


class TowerKeepingHud(FakeHudReader):
    """FakeHudReader with HudReader's reset(keep_towers=) and reset_towers()."""

    def __init__(self, ticks):
        super().__init__(ticks)
        self.calls = []

    def reset(self, *, keep_towers=False):
        self.resets += 1
        self.calls.append("keep" if keep_towers else "all")

    def reset_towers(self):
        self.calls.append("towers")


def test_a_battle_lost_without_its_end_screen_keeps_its_towers_until_a_new_one_starts():
    """A battle that ended on a misread clock, a stopped clock or 8 s off the battle screen
    may come back: the HUD is reset but for the towers (a destroyed tower would read full
    again), also by the gap reset; a battle joined in the middle is that battle and keeps them,
    a clean start is a new battle and starts them again. The end screen resets everything."""
    hud = TowerKeepingHud([2000, 2100, 100])
    p = VP.LivePerception(hud_reader=hud, tracker=None, deriver=None, geometry=GEOMETRY, detector=None)
    p.perceive(Frame(1.0, battle_image(), 0.0))            # tick 2000
    p.new_battle()
    p.end_battle(keep_towers=True)                         # e.g. a misread clock jumped back
    assert hud.calls == ["keep"] and p.towers_kept
    p.perceive(Frame(10.0, battle_image(), 0.0))           # back after 9 s (tick 2100): the gap reset
    assert hud.calls == ["keep", "keep"]
    p.new_battle()                                         # joined at 2100: the same battle
    assert hud.calls == ["keep", "keep"] and not p.towers_kept and p.primed_until == 2110
    p.end_battle(keep_towers=True)
    p.perceive(Frame(11.0, battle_image(), 0.0))           # tick 100: a clean start, a new battle
    p.new_battle()
    assert hud.calls == ["keep", "keep", "keep", "towers"] and p.primed_until is None
    p.end_battle()                                         # the end screen: everything
    assert hud.calls[-1] == "all" and not p.towers_kept
    plain = FakeHudReader([100])                           # no reset_towers: reset whole
    q = VP.LivePerception(hud_reader=plain, tracker=None, deriver=None, geometry=GEOMETRY, detector=None)
    q.end_battle(keep_towers=True)
    assert plain.resets == 1 and not q.towers_kept
    reasons = ("end screen", "overtime over", "clock reset", "clock stopped", "left the battle screen")
    assert [VP.keeps_towers(r) for r in reasons] == [False, False, True, True, True]


def test_the_runner_keeps_the_towers_only_when_the_end_screen_was_not_seen(tmp_path):
    res = run_script(tmp_path / "a", standard_script())
    assert res.records["battle_end"][0]["reason"] == "end screen" and res.perception.kept_towers == [False]
    items = [vf for _, vf in standard_script()[:28]] + [menu_vf(t_of(28 + k)) for k in range(95)]
    res = run_script(tmp_path / "b", [(t_of(i), vf) for i, vf in enumerate(items)])
    assert res.records["battle_end"][0]["reason"] == "left the battle screen"
    assert res.perception.kept_towers == [True] and "perception_error" not in res.records


def test_the_rows_hand_shows_the_card_in_flight_and_never_token_1():
    """decide() reads the HUD's hand (-1: masked); the row shows a played slot's card in flight
    (as RoyaleSim's state does until the play runs) and an unread slot as a pad, token 0:
    obs_row's token is card + 2, and no training row ever held token 1."""
    hud = FakeHudReader([100])
    hud.hand = SimpleNamespace(left={1: 2})               # the Hog left slot 1
    p = VP.LivePerception(hud_reader=hud, tracker=None, deriver=None, geometry=GEOMETRY)
    assert p.row_hand([7, -1, 6, -1]) == [7, 2, 6, VP.ROW_PAD_CARD]
    assert VP.LivePerception(hud_reader=FakeHudReader([]), tracker=None, deriver=None,
                             geometry=GEOMETRY).row_hand([7, -1]) == [7, VP.ROW_PAD_CARD]
    plan = VP.row_plan({"live_compat": True})
    builder = SimpleNamespace(state=lambda vf, **kw: {"players": [{}, {"hand": [7, -1, 6, -1]}]})
    perception = SimpleNamespace(track_cards=lambda tracks: {}, row_hand=p.row_hand)
    state = VP.make_state_fn(builder, perception, plan)(battle_vf(1.0, 100), VP.BattleContext(1, 0.0, 90))
    assert state["players"][1]["hand"] == [7, -1, 6, -1]           # what decide() masks by
    seen = []
    row_fn = VP.make_row_fn(lambda st, *a, **kw: seen.append(st["players"][1]["hand"]) or {}, 18, plan,
                            live_compat=True, units="types")
    row_fn(state, {}, 100)
    assert seen == [[7, 2, 6, -2]] and [c + 2 for c in seen[0]] == [9, 4, 8, 0]
    assert state["players"][1]["hand"] == [7, -1, 6, -1]           # the state itself is untouched
    assert VP.row_state({"tick": 1}) == {"tick": 1}


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
    assert b.first_tick == 30                    # the first clock reading of the run that started it


def test_battle_state_start_list_stays_bounded_on_a_frozen_clock():
    """20 s of a clock frozen on a battle screen at 50 fps keep the start list bounded; once the
    clock runs the battle starts exactly as before (20 ticks past the frozen reading)."""
    b = VP.BattleState()
    assert all(b.update(vf_at(10.0 + 0.02 * i, 19)) == [] for i in range(1000))
    assert len(b._cand) <= 256
    events = [b.update(vf_at(30.0 + 0.05 * i, 20 + i)) for i in range(25)]
    assert events.index(["start"]) == 19 and events.count(["start"]) == 1 and b.number == 1


def test_battle_state_end_rules():
    # a clock that is read but stops: running() goes false after 1.2 s (no decisions); a few
    # seconds are a pause (a lag spike with the HUD up), 10 s the end
    b = VP.BattleState()
    t = started(b)
    assert b.running(t + 1.0) and not b.running(t + 1.5)
    assert b.update(vf_at(t + 2.0, 120)) == [] and b.update(vf_at(t + 3.0, 120)) == []
    assert b.update(vf_at(t + 9.5, 120)) == [] and b.in_battle
    assert b.update(vf_at(t + 10.5, 120)) == ["end"] and b.ended_reason == "clock stopped"
    assert VP.BattleState(frozen_s=2.5).frozen_s == 2.5
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
    # the end screen's crowns stand when read (FIX D; it was the larger of the two, "end screen +
    # hud", which kept the HUD's misread 2-0 of the 2026-10-31 friendly); equal crowns: no note
    assert (rec["outcome"], rec["crowns"], rec["end_frames"], rec["source"]) == ("win", [1, 0], 2, "end screen")
    assert "crowns_disagree" not in rec
    book.open(2, "clock stopped", 100.0, (0, 1))
    assert not book.due(SCREEN_MENU, 119.0) and book.due(SCREEN_MENU, 120.0)
    rec = book.finish()
    assert (rec["outcome"], rec["source"], rec["record"]) == ("loss", "hud", {"win": 1, "loss": 1})
    assert book.done == 2 and book.finish() is None


def test_result_book_the_end_screen_beats_the_hud_crowns():
    """The 2026-10-31 friendly: the HUD read both enemy towers destroyed (2-0, a tower skin), the
    end screen showed 1-0. The end screen's crowns stand and the HUD's go in crowns_disagree; an
    end screen read 0-0 before the clock ran out keeps the HUD's (no battle ends 0-0 then, and
    the loss layout is unverified), but not at the end of overtime."""
    book = VP.ResultBook()
    book.open(1, "end screen", 50.0, (2, 0), last_tick=4003)
    book.end_frame((1, 0), 50.1)
    rec = book.finish()
    assert (rec["crowns"], rec["source"], rec["outcome"]) == ([1, 0], "end screen", "win")
    assert rec["crowns_disagree"] == {"hud": [2, 0], "end": [1, 0]}
    book.open(2, "end screen", 90.0, (0, 1), last_tick=2500)
    book.end_frame((0, 0), 90.1)
    rec = book.finish()
    assert (rec["crowns"], rec["source"], rec["outcome"]) == ([0, 1], "hud (end screen read 0-0)", "loss")
    assert rec["crowns_disagree"] == {"hud": [0, 1], "end": [0, 0]}
    book.open(3, "end screen", 130.0, (0, 1), last_tick=5990)
    book.end_frame((0, 0), 130.1)
    rec = book.finish()
    assert (rec["crowns"], rec["source"], rec["outcome"]) == ([0, 0], "end screen", "draw")
    book.open(4, "end screen", 170.0, (0, 0), last_tick=1000)
    book.end_frame((0, 0), 170.1)
    rec = book.finish()
    assert (rec["crowns"], rec["source"]) == ([0, 0], "end screen") and "crowns_disagree" not in rec
    assert book.tally == {"win": 1, "loss": 1, "draw": 2}


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
    assert (args.decide, args.would_every, args.replay_fast, args.torch_threads) == ("live", 10, False, 0)
    assert VP.build_parser().parse_args(["--decide", "legacy"]).decide == "legacy"
    with pytest.raises(SystemExit):
        VP.build_parser().parse_args(["--decide", "sample"])           # RL's rule never plays live
    assert [VP.is_replay(s) for s in ("auto", "printwindow", "adb", "reports/x")] == [False] * 3 + [True]
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
# _main end to end on a recording: the offline rehearsal (runbook 17.1)
# ---------------------------------------------------------------------------

class ScriptedHud:
    """HudReader's surface for _main: the clock runs 2 ticks a battle frame from tick 70; slot 3
    reads empty while the hand tracker knows the Ice Spirits (card 4) left it."""

    def __init__(self, store, card_ids, *, deck=None, regions=None):
        self.card_ids, self.deck = dict(card_ids), list(deck or ())
        self.clock_reader = SimpleNamespace(ready=True)
        self.regions = SimpleNamespace(name="scripted")
        self.hand = SimpleNamespace(left={3: 4})
        self.reads = self.resets = 0

    def read(self, rgb, t):
        tick = 70 + 2 * self.reads
        self.reads += 1
        return Hud(hand=[7, 2, 6, -1], next=0, elixir_milli=7000, clock_tick=tick, phase=PHASE_REGULATION)

    def reset(self):
        self.resets += 1


def write_session(root):
    """A capture --record style session: 2 menu, 26 battle, 5 end-screen and 3 menu frames,
    each with its image (npy), 0.1 s apart."""
    rec = VisionRecorder(root, session="capture_test", image_every=1, image_format="npy",
                         wall_clock=lambda: 0.0)
    black = np.zeros((400, 225, 3), dtype=np.uint8)
    images = [black] * 2 + [battle_image()] * 26 + [end_image()] * 5 + [black] * 3
    for i, img in enumerate(images):
        rec.write(VisionFrame(t_capture=round(500.0 + 0.1 * i, 4), frame_size=(225, 400)), img)
    rec.close()
    return rec.session_dir


def fake_live_modules(calls):
    """import_live_modules() for _main with torch, royalesim, the checkpoint, the HUD templates
    and the detector replaced; capture (the replay source), geometry, record and the tracker are
    the real ones. `calls` collects what reached the fakes."""
    from scripts.vision import capture, geometry, record, tracker

    class EngineError(RuntimeError):
        pass

    def obs_row(state, team, sub, view, tick, **kw):
        calls.setdefault("row_hands", []).append(list(state["players"][team]["hand"]))
        return {"tick": tick}

    class Builder:
        def __init__(self, *args, **kw):
            calls["builder"] = sorted(kw)

        def state(self, vf, **kw):
            hand = list(vf.hud.hand)
            me = {"hand": hand, "hand_costs": [COST.get(c, 0) if c >= 0 else 0 for c in hand],
                  "elixir_milli": vf.hud.elixir_milli or 0, "next_card": vf.hud.next}
            st = {"tick": vf.tick, "entities": [],
                  "players": [{"hand": [], "hand_costs": [], "elixir_milli": 0}, me]}
            calls.setdefault("states", []).append(st)
            return st

    def load_detector(*args, **kw):
        raise RuntimeError("no detector weights (test)")

    def verify(serial, adb):
        calls["verify"] = (serial, adb)
        raise AssertionError("a dry run never calls emulator.verify")

    royalesim = SimpleNamespace(SUBTILE_PER_MILLITILE=18, CATALOGUE_FIELDS=())
    ids = {n: i for i, n in enumerate(NAMES)}
    cat = [[n] for n in NAMES]
    table = SimpleNamespace(doc={"classes": [], "units": {}}, deck_labels=lambda deck: set(),
                            spell_labels=lambda: frozenset(), digest="dc897910e1e8")
    return {
        "public_state": SimpleNamespace(PublicStateBuilder=Builder, LiveObserver=None,
                                        row_inputs=lambda state: {}),
        "rs_rows": SimpleNamespace(UNITS_TYPES="types", UNITS_VISION="vision", UNITS="types", F={},
                                   SF=None, obs_row=obs_row, elixir_law=lambda eng: None,
                                   _setup=lambda eng: (royalesim, list(NAMES), ids, cat)),
        "rs_checkpoint": SimpleNamespace(
            load_policy=lambda blob, eng: SimpleNamespace(model="the model", drift=())),
        "rs_engine": SimpleNamespace(
            EngineError=EngineError,
            config_from_args=lambda args: SimpleNamespace(to_json=lambda: {"engine": "test"})),
        "rs_spawn_table": SimpleNamespace(DEFAULT_PATH="spawn_primary.json",
                                          ensure=lambda engine=None, rs=None: {"card_names": []}),
        "vision_classes": SimpleNamespace(load=lambda: table),
        "scripts.vision.capture": capture,
        "scripts.vision.detector": SimpleNamespace(Detector=SimpleNamespace(load=load_detector),
                                                   read_pin=lambda: None),
        "scripts.vision.geometry": geometry,
        "scripts.vision.hpbars": SimpleNamespace(),
        "scripts.vision.hud": SimpleNamespace(HudReader=ScriptedHud, load_regions=lambda geometry=None: None),
        "scripts.vision.record": record,
        "scripts.vision.templates": SimpleNamespace(TemplateStore=lambda: SimpleNamespace(root="templates")),
        "scripts.vision.tracker": tracker,
        "scripts.emulator": SimpleNamespace(verify=verify),
    }


def rehearsal_env(tmp_path, monkeypatch, calls):
    """main()'s files under tmp_path, the fake modules, a fake torch and a checkpoint with its
    args.json (wait_keep 0.25, as live_big_hog's)."""
    monkeypatch.setattr(VP, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(VP, "LOCK_FILE", tmp_path / "vision_play.lock")
    monkeypatch.setattr(VP, "REPORT_ROOT", tmp_path / "reports")
    monkeypatch.setattr(VP, "REPLAY_ROOT", tmp_path / "reports" / "replays")
    monkeypatch.setattr(VP, "import_live_modules", lambda: fake_live_modules(calls))
    monkeypatch.setattr(VP.RD, "TokenPolicyAdapter", lambda model: FakePolicy((100,)))
    monkeypatch.setattr(VP.RD, "TowersOnlyLegality", lambda *a, **kw: (lambda slot, card, x, y: True))
    torch = SimpleNamespace(load=lambda path, map_location=None: {"meta": {"live_compat": True}},
                            set_num_threads=lambda n: calls.__setitem__("threads", n),
                            get_num_threads=lambda: 4)
    monkeypatch.setitem(sys.modules, "torch", torch)
    ckpt = tmp_path / "runs" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"not a real checkpoint")
    (ckpt.parent / "args.json").write_text(json.dumps({"wait_keep": 0.25}), encoding="utf-8")
    return ckpt


def test_main_rehearses_the_whole_loop_on_a_recording(tmp_path, monkeypatch, capsys):
    """vision_play's _main end to end on a recorded session (--source <folder> --replay-fast):
    the wiring, the setup lines, the config record, the replay source, the runner, a battle and
    its result, the row's hand, a session under reports/vision_play/replays. No adb, no
    emulator, no tap; the heavy modules are fakes (fake_live_modules)."""
    calls = {}
    ckpt = rehearsal_env(tmp_path, monkeypatch, calls)
    session = write_session(tmp_path / "recordings")
    rc = VP.main(["--source", str(session), "--replay-fast", "--no-window", "--hotkey", "none",
                  "--manual-file", "none", "--ckpt", str(ckpt), "--record-images", "0", "--seed", "1",
                  "--torch-threads", "2"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "source     replay of" in out and "(36 images, as fast as the chain runs); no device, no adb" in out
    assert "taps       off (the source is a recording)" in out
    assert "refusals   refused taps: (dry run: logs only)" in out
    assert "detector   NONE (RuntimeError: no detector weights (test))" in out
    assert "decide     live; wait_keep 0.25 (wait bias 1.386); would-play every 10 ticks" in out
    assert "CPU, 4 torch threads" in out and calls["threads"] == 2
    assert "battle 1 started (tick 90, mode MANUAL)" in out
    assert "battle 1: WIN 1-0" in out
    assert "stopped: the frame source ended; 36 frames, 0 taps" in out and "errors -" in out
    assert "verify" not in calls
    # the session: under the replays root, not where latency_probe --last looks
    assert sorted(p.name for p in (tmp_path / "reports").iterdir()) == ["replays"]
    [replay] = list((tmp_path / "reports" / "replays").iterdir())
    assert replay.name.startswith("replay_")
    recs = read_records(replay)
    [cfg] = recs["config"]
    assert (cfg["source_kind"], cfg["replay"]["images"], cfg["replay"]["realtime"]) == ("replay", 36, False)
    assert (cfg["taps"], cfg["tap_refusal"], cfg["decide"], cfg["wait_keep"], cfg["args_json"]) == (
        False, "the source is a recording", "live", 0.25, True)
    assert cfg["tap_guard"]["enforce"] is False and cfg["tap_guard"]["k"] == 3
    assert cfg["tap_guard"]["refuse_after_ticks"] == 30 and cfg["tap_guard"]["final_after_ticks"] == 60
    # MANUAL: no tap and no dry play, so nothing was watched
    assert recs["session_end"][0]["tap_guard"]["watched"] == 0
    assert recs["battle_end"][0]["tap_guard"]["watched"] == 0
    assert recs["battle_start"][0]["tick"] == 90 and recs["result"][0]["outcome"] == "win"
    [d] = decisions(recs)
    assert (d["tick"], d["card"], d["action"]) == (100, 2, CT.ACTION_MANUAL)
    # the row's hand shows the Ice Spirits in flight; the state decide() masks by keeps -1
    assert len(calls["row_hands"]) == 16 and all(h == [7, 2, 6, 4] for h in calls["row_hands"])
    assert all(st["players"][1]["hand"] == [7, 2, 6, -1] for st in calls["states"])
    assert not lock_left(tmp_path)


def lock_left(tmp_path):
    return (tmp_path / "vision_play.lock").exists()


def test_main_refuses_a_bad_combination_before_anything_starts(tmp_path, monkeypatch, capsys):
    calls = {}
    rehearsal_env(tmp_path, monkeypatch, calls)
    assert VP.main(["--replay-fast"]) == 2
    assert "--replay-fast replays a recording" in capsys.readouterr().out
    assert VP.main(["--source", str(tmp_path / "missing")]) == 2
    assert "no recording at" in capsys.readouterr().out
    assert VP.main(["--tap", "--decide", "legacy"]) == 2
    assert "--decide legacy is for dry runs" in capsys.readouterr().out
    assert calls == {} and not (tmp_path / "reports").exists()


def test_a_dry_run_over_adb_verifies_the_device_before_its_first_screencap(tmp_path, monkeypatch, capsys):
    """--source adb without --tap reads the device by screencap: emulator.verify runs first
    (no other MuMu instance is touched), and its refusal stops the runner before any frame
    source is opened."""
    calls = {}
    ckpt = rehearsal_env(tmp_path, monkeypatch, calls)
    mods = fake_live_modules(calls)

    def refuse(serial, adb):
        calls["verify"] = (serial, adb)
        raise SystemExit(f"refusing to run: {serial} is the CLASH OF CLANS instance.")

    mods["scripts.emulator"] = SimpleNamespace(verify=refuse)
    monkeypatch.setattr(VP, "import_live_modules", lambda: mods)
    opened = []
    monkeypatch.setattr(mods["scripts.vision.capture"], "open_source",
                        lambda *a, **kw: opened.append((a, kw)))
    rc = VP.main(["--source", "adb", "--no-window", "--hotkey", "none", "--manual-file", "none",
                  "--ckpt", str(ckpt), "--record-images", "0"])
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "--source adb reads the device over adb, and emulator.verify refused it" in out
    assert "CLASH OF CLANS" in out and "verify" in calls and opened == []
    assert not lock_left(tmp_path)


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


# ---------------------------------------------------------------------------
# FIX E: the refused-tap guard (scripts/vision/tap_refusal.py) in the loop
# ---------------------------------------------------------------------------

from scripts.vision import tap_refusal as TR  # noqa: E402


def long_script(last_tick=320, *, hand_at=None, elixir_at=None, battles=1):
    """standard_script's battle stretched to ticks 70..last_tick (2 ticks a frame, 0.1 s apart),
    the end screen after each battle, `battles` times. hand_at(tick) / elixir_at(tick): the
    HUD's hand and elixir (default: HAND and 7.0)."""
    items = [menu_vf(t_of(0)), menu_vf(t_of(1))]
    for _ in range(battles):
        for tick in range(70, last_tick + 1, 2):
            items.append(battle_vf(t_of(len(items)), tick, hand=None if hand_at is None else hand_at(tick),
                                   elixir=7000 if elixir_at is None else elixir_at(tick)))
        items += [end_vf(t_of(len(items)), tick=last_tick + 2) for _ in range(5)]
        items += [menu_vf(t_of(len(items))) for _ in range(3)]
    return [(t_of(i), item) for i, item in enumerate(items)]


def frame_of(tick, battle=1, last_tick=320):
    """The frame index of `tick` in long_script (battle 1, 2, ...)."""
    per = (last_tick - 70) // 2 + 1 + 8
    return 2 + (battle - 1) * per + (tick - 70) // 2


def played(res):
    """(tick, action, model tile, legality) of each play decision on the frame lines."""
    return [(e["decision"]["tick"], e["action"], (e["decision"]["tx"], e["decision"]["ty"]),
             e["decision"]["legality"])
            for e in (f["extra"] for f in res.records["frame"]) if e.get("decision", {}).get("kind") == RD.PLAY]


def in_box(tile, centre=PLAY_TILE, r=2):
    return abs(tile[0] - centre[0]) <= r and abs(tile[1] - centre[1]) <= r


def test_refusal_guard_a_landed_tap_is_never_refused(tmp_path):
    """The Hog tapped at 100: its slot empties at 110 and refills with the Musketeer at 130, the
    elixir falls by 4 at 110. tap_landed, no tap_refused, AUTO all along."""
    g = TR.RefusedTapGuard(enforce=True)
    hand = lambda tk: HAND if tk < 110 else ([7, VP.EMPTY_CARD, 6, 4] if tk < 130 else [7, 5, 6, 4])  # noqa: E731
    elixir = lambda tk: 7000 if tk < 110 else 3000  # noqa: E731
    res = run_script(tmp_path, long_script(200, hand_at=hand, elixir_at=elixir), play_ticks=(100,),
                     refusals=g)
    [landed] = res.records["tap_landed"]
    assert (landed["battle"], landed["tap_tick"], landed["card"], landed["name"], landed["slot"]) == (
        1, 100, 2, "HogRider", PLAY_SLOT)
    assert landed["evidence"] in ("elixir", "hand") and landed["tick"] <= 116 and landed["dry"] is False
    assert "tap_refused" not in res.records and "tap_guard_manual" not in res.records
    assert res.control.mode == CT.AUTO and len(res.tapper.calls) == 1
    [end] = res.records["battle_end"]
    assert (end["tap_guard"]["watched"], end["tap_guard"]["landed"], end["tap_guard"]["refused"]) == (1, 1, 0)
    assert res.records["session_end"][0]["tap_guard"]["landed"] == 1
    assert not any("tap did nothing" in line for line in res.lines)


def test_refusal_guard_a_tap_that_did_nothing_is_refused_and_masked(tmp_path):
    """The hand never changes: the Hog tapped at 100 is refused at 130 and kept off the box
    around (4, 22); the play at 162 goes elsewhere through the "+refusals" legality."""
    g = TR.RefusedTapGuard(enforce=True)
    res = run_script(tmp_path, long_script(200), play_ticks=(100, 162), refusals=g)
    [refused, *_] = res.records["tap_refused"]
    assert (refused["battle"], refused["tick"], refused["tap_tick"], refused["age"]) == (1, 130, 100, 30)
    assert (refused["card"], refused["name"], refused["slot"], refused["cost"]) == (2, "HogRider", 1, 4)
    assert refused["model_tile"] == [4, 22] and refused["screen_tile"] == [13, 22] and refused["zone"] == "own"
    assert refused["mask"] == {"kind": "box", "x0": 2, "y0": 20, "x1": 6, "y1": 24, "until_tick": 730}
    assert (refused["dry"], refused["applied"], refused["refusals"]) == (False, True, 1)
    assert refused["elixir_at_tap"] == 7000 and refused["present_ticks"] == 15
    plays = played(res)
    assert plays[0] == (100, CT.ACTION_TAP, PLAY_TILE, "callable")      # nothing masked yet: as before
    tick, action, tile, legality = plays[1]
    assert (tick, action) == (162, CT.ACTION_TAP) and not in_box(tile) and legality == "callable+refusals"
    assert [t["tick"] for t in res.records["tap"]] == [100, 162]
    assert any(line.startswith("tap did nothing: HogRider -> (13, 22) at tick 100 (still in its slot 30 ticks")
               for line in res.lines)
    assert any("| AUTO(tap) | refused 1/3 | last" in line for line in res.lines)
    assert res.control.mode == CT.AUTO


def test_refusal_guard_three_refusals_switch_to_manual(tmp_path):
    """Taps at 100, 162 (the slot of the last refusal: toggle_suspect) and 224, finals at 160,
    222 and 284: MANUAL from the decision at 284 on; the play at 300 is a suggestion only."""
    g = TR.RefusedTapGuard(enforce=True)
    res = run_script(tmp_path, long_script(320), play_ticks=(100, 162, 224, 300), refusals=g)
    assert len(res.tapper.calls) == 3 and [t["tick"] for t in res.records["tap"]] == [100, 162, 224]
    refused = res.records["tap_refused"]
    assert [(r["tick"], r["toggle_suspect"], r["mask"] is None) for r in refused] == [
        (130, False, False), (192, True, True), (254, False, False)]
    assert [r["tick"] for r in res.records["tap_refusal_final"]] == [160, 222, 284]
    [manual] = res.records["tap_guard_manual"]
    assert (manual["battle"], manual["tick"], manual["k"], manual["mode_before"], manual["switched"],
            manual["dry"]) == (1, 284, 3, CT.AUTO, True, False)
    assert [f["tap_tick"] for f in manual["finals"]] == [100, 162, 224]
    [mode] = res.records["mode"]
    assert (mode["mode"], mode["prev"], mode["source"], mode["tick"]) == (CT.MANUAL, CT.AUTO, "refused taps", 284)
    assert res.control.mode == CT.MANUAL
    [line] = [x for x in res.lines if x.startswith("REFUSED TAPS:")]
    assert line.startswith("REFUSED TAPS: 3 taps this battle did nothing in the game (last HogRider -> (")
    assert line.endswith("at tick 224). Switched to MANUAL: play yourself, or press F8 for AUTO once the "
                         "status line looks right.")
    late = [p for p in played(res) if p[0] == 300]
    assert late and late[0][1] == CT.ACTION_MANUAL
    [end] = res.records["battle_end"]
    assert (end["tap_guard"]["refused"], end["tap_guard"]["final"], end["tap_guard"]["toggle"],
            end["tap_guard"]["switched"]) == (3, 3, 1, 1)


def test_refusal_guard_a_dry_run_only_logs_and_decides_as_without_it(tmp_path):
    """No --tap: the AUTO(dry) plays get virtual watches, refused because nothing was tapped;
    the decisions are those of a run without the guard, nothing switches, one dry line."""
    g = TR.RefusedTapGuard(enforce=False)
    res = run_script(tmp_path / "guard", long_script(320), taps=False, play_ticks=(100, 162, 224, 300),
                     refusals=g)
    plain = run_script(tmp_path / "plain", long_script(320), taps=False, play_ticks=(100, 162, 224, 300))
    assert played(res) == played(plain)
    assert all(p[3] == "callable" and p[1] == CT.ACTION_DRY for p in played(res))
    refused = res.records["tap_refused"]
    assert [(r["tick"], r["toggle_suspect"]) for r in refused] == [(130, False), (192, True), (254, False)]
    assert all(r["dry"] is True and r["applied"] is False for r in refused)
    [manual] = res.records["tap_guard_manual"]
    assert (manual["tick"], manual["switched"], manual["dry"], manual["mode_before"]) == (284, False, True, CT.AUTO)
    assert res.control.mode == CT.AUTO and "mode" not in res.records
    assert res.tapper.calls == [] and "tap" not in res.records
    dry = [x for x in res.lines if x.startswith("dry run: the refused-tap guard would switch")]
    assert len(dry) == 1 and "no tap was sent, so the plays were watched as if tapped" in dry[0]
    assert not any("tap did nothing" in x or x.startswith("REFUSED TAPS") for x in res.lines)


def test_refusal_guard_watches_only_taps_that_were_sent(tmp_path):
    """MANUAL with --tap, a vetoed tap and a failed tap: no watch at all."""
    g = TR.RefusedTapGuard(enforce=True)
    res = run_script(tmp_path / "manual", long_script(200), mode=CT.MANUAL, play_ticks=(100,), refusals=g)
    assert g.summary(session=True)["watched"] == 0 and "tap_refused" not in res.records
    g = TR.RefusedTapGuard(enforce=True)
    run_script(tmp_path / "veto", long_script(200), play_ticks=(100,), refusals=g,
               tap_guard=lambda frame: "the end screen is up")
    assert g.summary(session=True)["watched"] == 0
    g = TR.RefusedTapGuard(enforce=True)
    run_script(tmp_path / "fail", long_script(200), play_ticks=(100,), refusals=g, tapper=FakeTapper(fail=True))
    assert g.summary(session=True)["watched"] == 0


def test_refusal_guard_rearms_when_the_owner_hands_back(tmp_path):
    """After the switch at 284 the owner presses F8 (tick 290): tap_guard_rearmed, the masks
    stay, and the next single refusal (the tap at 300, final at 360) does not switch again."""
    g = TR.RefusedTapGuard(enforce=True)
    holder = {}

    def on_read(i):
        if i == frame_of(290, last_tick=380):
            holder["state"].toggle("hotkey")

    res = run_script(tmp_path, long_script(380), play_ticks=(100, 162, 224, 300), refusals=g,
                     on_read=on_read, control=holder)
    [rearmed] = res.records["tap_guard_rearmed"]
    assert (rearmed["battle"], rearmed["tick"]) == (1, 290)
    assert [m["source"] for m in res.records["mode"]] == ["refused taps", "hotkey"]
    assert [t["tick"] for t in res.records["tap"]] == [100, 162, 224, 300]
    assert len(res.records["tap_guard_manual"]) == 1 and res.control.mode == CT.AUTO
    assert [r["tick"] for r in res.records["tap_refusal_final"]][-1] == 360
    tile = [p for p in played(res) if p[0] == 300][0][2]
    assert not in_box(tile)                                   # the first refusal's mask stayed


def test_refusal_guard_rearms_when_the_owner_took_over_before_its_switch(tmp_path):
    """The owner presses F8 (MANUAL) at tick 250, on the "tap did nothing" lines, before the 3rd
    refusal is final (284): the guard cannot switch, says so, and disarms. His F8 back to AUTO
    at 290 rearms it, so three more refusals (finals 360, 422, 484) switch to MANUAL again;
    before the fix it stayed disarmed and the bot kept tapping ("refused 7/3")."""
    g = TR.RefusedTapGuard(enforce=True)
    holder = {}
    last = 560

    def on_read(i):
        if i in (frame_of(250, last_tick=last), frame_of(290, last_tick=last)):
            holder["state"].toggle("hotkey")

    res = run_script(tmp_path, long_script(last), play_ticks=(100, 162, 224, 300, 362, 424, 486),
                     refusals=g, on_read=on_read, control=holder)
    assert [r["tick"] for r in res.records["tap_refusal_final"]] == [160, 222, 284, 360, 422, 484]
    first, second = res.records["tap_guard_manual"]
    assert (first["tick"], first["mode_before"], first["switched"], first["dry"]) == (284, CT.MANUAL, False, False)
    assert (second["tick"], second["mode_before"], second["switched"]) == (484, CT.AUTO, True)
    [rearmed] = res.records["tap_guard_rearmed"]
    assert (rearmed["battle"], rearmed["tick"]) == (1, 290)
    assert [(m["mode"], m["source"]) for m in res.records["mode"]] == [
        (CT.MANUAL, "hotkey"), (CT.AUTO, "hotkey"), (CT.MANUAL, "refused taps")]
    assert res.control.mode == CT.MANUAL
    assert [t["tick"] for t in res.records["tap"]] == [100, 162, 224, 300, 362, 424]
    said = [x for x in res.lines if x.startswith("REFUSED TAPS:")]
    assert len(said) == 2
    assert said[0].endswith("Already MANUAL: play yourself; press F8 for AUTO once the status line looks "
                            "right (the count then starts again).")
    assert "Switched to MANUAL" in said[1]
    assert not any("refused 4/3" in x for x in res.lines)
    [end] = res.records["battle_end"]
    assert (end["tap_guard"]["final"], end["tap_guard"]["switched"]) == (6, 1)


def test_refusal_guard_a_new_battle_clears_masks_and_count(tmp_path):
    """Battle 1: the Hog at 100 refused at 130 (masked around (4, 22)) and final at 160.
    Battle 2: the Hog at 100 taps (4, 22) again, and the count starts at 0."""
    g = TR.RefusedTapGuard(enforce=True)
    res = run_script(tmp_path, long_script(170, battles=2), play_ticks=(100, 162), refusals=g)
    taps = [(t["battle"], t["tick"], tuple(t["model_tile"])) for t in res.records["tap"]]
    assert taps[0] == (1, 100, PLAY_TILE) and taps[1][:2] == (1, 162) and not in_box(taps[1][2])
    assert taps[2] == (2, 100, PLAY_TILE) and taps[3][:2] == (2, 162)
    ends = res.records["battle_end"]
    assert [e["tap_guard"]["final"] for e in ends] == [1, 1]
    assert [e["tap_guard"]["watched"] for e in ends] == [2, 2]
    assert res.records["session_end"][0]["tap_guard"]["watched"] == 4
    assert [r["refusals"] for r in res.records["tap_refused"]] == [1, 1]
