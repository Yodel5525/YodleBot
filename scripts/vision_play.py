r"""Friendly-only live runner: screen vision -> token model -> taps (V10, with V12's takeover).

    .venvs\buildabot\Scripts\python.exe scripts\vision_play.py                  dry run (the default)
    .venvs\buildabot\Scripts\python.exe scripts\vision_play.py --tap --start-mode auto
                                                                        taps, only while you watch

FRIENDLY BATTLES ONLY. A person starts every battle (clan chat or a friend's profile). This
runner never navigates: it does not press Battle, OK, or anything outside the arena, and it has
no option that would (hastycr.machine.FRIENDLY_ONLY). It taps only while a battle is on screen:
the battle clock is read and running, the bottom HUD panel is there and the end screen is not.
It never reads the game's memory (scripts/gamestate is not imported) and it never starts or
restarts the emulator.

DRY RUN BY DEFAULT. Decisions are made, shown and logged, nothing is tapped. --tap enables taps;
use it only while you watch. Before the first tap scripts/emulator.verify() must pass (the serial
is the Clash Royale instance) and `adb shell wm size` must report 1080x1920. A file named STOP in
the repo root ends the runner in any mode (it refuses to start while STOP exists). One runner at a
time: reports/vision_play/vision_play.lock is held by the running process (the OS releases it if
the process dies).

TAKEOVER (scripts/vision/control.py). The runner starts in MANUAL unless --start-mode auto: the
bot keeps perceiving and deciding, and shows its suggestions (status line, control window) but does
not tap. F8, the control window's button or a MANUAL file in the repo root switch to AUTO and
back; a switch drops a tap decided before it. Own plays the bot did not tap are logged as the
human's, next to what the bot would have played.

UNKNOWN PLAYS (--unknown-plays, off by default; docs/NEW_CARDS_PLAN.md 2.2). An enemy unit no
card of the event tables explains (a card the detector names and the tables lack, a hero's extra
unit, a misread) is reported as an UNKNOWN play, card -1: the model reads it as row token 1 in
the opponent's memory (opp_seen / opp_last, legacy and v2 rows), a v2 row's opponent elixir is
charged events.UNKNOWN_PLAY_COST for it and is no longer exact, it is not added to the
opponent's known cards, and an enemy label the class table cannot place stays on the board as
an unknown unit. Without the flag such units are dropped, as before. Each battle's end record
then counts the UNKNOWN plays (opp_unknown).

THE LOOP, per new frame (the newest one; frames that arrived meanwhile are skipped):

  FrameSource (scripts/vision/capture.py open_source, "auto": PrintWindow at the window size,
      747x1328 on the owner's PC, with an adb fallback)
  -> screen cues (pixels, resolution-free): the HUD panel's top edge (blue band), the end screen
     (red and blue banners and the OK button), the end screen's crowns
  -> HudReader (clock -> tick, elixir, hand, towers; templates captured on the PC)
  -> Detector (scripts/vision/detector.py; weights from --detector or the pin file) -> HP bars
     -> Tracker (own_classes = the deck's labels, vision_classes.ClassTable.deck_labels)
     -> EventDeriver (opponent plays, own landings; EventTables from data/rs_rows/spawn_primary.json)
  -> BattleState: a battle starts once the clock runs for a second with the panel on screen, and
     ends on the end screen, a stopped clock (2.5 s), a clock reset or 8 s off the battle screen
  -> PublicMemory (an opponent play recorded at the tick perception first saw it; view(now, 0))
  -> tools/public_state (the state obs_row reads) -> tools/rs_rows.obs_row with the checkpoint's
     row schema, picked by its meta "schema" (row_plan): rs_rows/legacy (--live-compat rows for
     live_big_hog) exactly as before; rs_rows/v2 (E14) also gets a public_state.LiveObserver
     per battle, fed every frame's opponent plays and our landings and each tap sent, whose
     inputs (the opponent's elixir counted from the plays seen, our taps still in flight
     until they land, at the checkpoint's --obs-delay) ride in the state to obs_row, and the
     spell tracks as spells in flight
  -> tools/rs_decide.decide with DecideConfig.live() (pending taps, fallback, cap rule, lane guard;
     a v2 model's ability buttons stay masked: our deck has none and the HUD reads none)
  -> ControlState.suggest / should_tap -> ScreenGuard (the NEWEST captured frame, taken after the
     one decided on, must still show the battle screen: no tap lands on an end screen whose OK
     button covers hand slot 1) -> AdbTapper: ONE persistent `adb -s SERIAL shell`, a play is
     two `input tap X Y` lines (card slot, then arena tile), both in DEVICE pixels 1080x1920
     from ArenaGeometry, whatever size the captured frames have; a shell that died is reopened
     only after emulator.verify passes again
  -> VisionRecorder: reports/vision_play/<session>/frames.jsonl, one line per frame (VisionFrame,
     decision, latencies in ms), plus records for decisions, taps, mode switches, human plays,
     battle start / end and results; an image every --record-images frames.

A status line every second shows the battle, tick, elixir, hand, mode and the last decision.
Per-frame errors are logged and never lead to a tap in that frame. Without detector weights the
runner still reads the HUD and decides (no units on the board), logs "no detector" and refuses
to tap.

Imports numpy and the stdlib at module level; torch, ultralytics and royalesim only in main().
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__" and not __package__:
    # Run as a file, sys.path[0] is scripts/, whose modules would shadow others: put the repo
    # root there instead, so `scripts.vision...` imports as a package.
    _here = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sys.path and os.path.normcase(os.path.abspath(sys.path[0] or os.curdir)) == _here:
        sys.path[0] = _root
    else:
        sys.path.insert(0, _root)

import argparse  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Iterable, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "tools", ROOT):          # hastycr, the tools, scripts.vision
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rs_decide as RD  # noqa: E402
import rs_memory  # noqa: E402
import rs_schema  # noqa: E402  (standard library only: the row schema names)
from scripts.vision import control as CT  # noqa: E402
from scripts.vision import events as EV  # noqa: E402
from scripts.vision import hpbars  # noqa: E402
from scripts.vision.geometry import check_frame_size  # noqa: E402
from scripts.vision.tracker import observations_from_detections  # noqa: E402
from scripts.vision.types import (  # noqa: E402
    EMPTY_CARD, OWN_TEAM, PHASE_ENDED, PHASE_OVERTIME, SCREEN_BATTLE, SCREEN_END, SCREEN_MENU,
    Hud, VisionFrame, now)

STOP_FILE = ROOT / "STOP"
REPORT_ROOT = ROOT / "reports" / "vision_play"
LOCK_FILE = REPORT_ROOT / "vision_play.lock"
#: Taps go to the device framebuffer (adb `input tap`), not to the captured frame.
DEVICE_SIZE = (1080, 1920)
#: Hog 2.6, all base forms (the bot account has no evolutions or heroes).
HOG26 = ("HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons",
         "Fireball", "Log")
DECK_ALIASES = {"hog 2.6": HOG26, "hog2.6": HOG26, "hog26": HOG26, "hog": HOG26}
DEFAULT_DECK = "Hog 2.6"
CKPT_IN_RUNS = Path("tokens") / "live_big_hog" / "best.pt"
FRIENDLY_LEVEL = 11

#: RoyaleSim takes no play before tick 90 (match.DEPLOY_LOCKOUT_TICKS); neither does the bot.
MIN_DECIDE_TICK = 90
#: Pixels older than this when read (an occluded window, a stalled adb) are not decided on.
MAX_STALE_S = 0.5

# ---------------------------------------------------------------------------
# screen cues (normalised boxes; measured on the owner's 747x1328 frames, 2026-10-29)
# ---------------------------------------------------------------------------
#: (u0, v0, u1, v1) of the bottom HUD panel's top edge: 100% "panel blue" in the three
#: battle frames (regulation, a fight, overtime), 0% on the end screen (forest).
PANEL_BAND = (0.18, 0.812, 0.98, 0.824)
#: The end screen: the opponent's red banner (75% red there, 0-3% in battle), our blue
#: banner (74% blue, 3-4%) and the OK button (85% blue, 12-13% in battle: card art).
END_RED_BANNER = (0.15, 0.255, 0.85, 0.325)
END_BLUE_BANNER = (0.15, 0.55, 0.85, 0.62)
END_OK_BUTTON = (0.40, 0.885, 0.60, 0.93)
PANEL_MIN = 0.6
END_MIN = 0.45
#: The crown cushions above each banner: three slots across, gold where a crown was won
#: (our won crown read 0.27 gold, empty cushions 0.00). Read on a WIN; the layout on a loss
#: is assumed to be the same (unverified), and the HUD's crowns are kept next to it.
CROWN_SLOTS_U = (0.26, 0.50, 0.735)
CROWN_HALF_U = 0.09
CROWN_ROWS_V = {"enemy": (0.17, 0.26), "own": (0.44, 0.535)}
CROWN_MIN = 0.10


def _box(rgb: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """The pixels of normalised box (u0, v0, u1, v1), at least one pixel."""
    h, w = rgb.shape[:2]
    x0, x1 = int(box[0] * w), max(int(box[0] * w) + 1, int(box[2] * w))
    y0, y1 = int(box[1] * h), max(int(box[1] * h) + 1, int(box[3] * h))
    return rgb[y0:y1, x0:x1, :3].astype(np.int16)


def blue_mask(px: np.ndarray) -> np.ndarray:
    """HUD panel / blue banner / OK button blue: B > 120, B > R + 60, B > G + 25."""
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    return (b > 120) & (b > r + 60) & (b > g + 25)


def red_mask(px: np.ndarray) -> np.ndarray:
    """The opponent's banner red: R > 120, R > G + 60, R > B + 30."""
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    return (r > 120) & (r > g + 60) & (r > b + 30)


def gold_mask(px: np.ndarray) -> np.ndarray:
    """A won crown's gold: R > 190, G > 130, B < 110, R - B > 110."""
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    return (r > 190) & (g > 130) & (b < 110) & (r - b > 110)


@dataclass(frozen=True)
class ScreenCues:
    """Colour fractions of the cue boxes of one frame."""
    panel: float
    red_banner: float
    blue_banner: float
    ok_button: float

    @property
    def end_screen(self) -> bool:
        return min(self.red_banner, self.blue_banner, self.ok_button) >= END_MIN

    @property
    def hud_present(self) -> bool:
        return self.panel >= PANEL_MIN and not self.end_screen

    def to_dict(self) -> dict[str, float]:
        return {"panel": round(self.panel, 3), "red_banner": round(self.red_banner, 3),
                "blue_banner": round(self.blue_banner, 3), "ok_button": round(self.ok_button, 3)}


def read_cues(rgb: np.ndarray) -> ScreenCues:
    """The screen cues of an HxWx3 RGB frame of any 9:16 size."""
    return ScreenCues(float(blue_mask(_box(rgb, PANEL_BAND)).mean()),
                      float(red_mask(_box(rgb, END_RED_BANNER)).mean()),
                      float(blue_mask(_box(rgb, END_BLUE_BANNER)).mean()),
                      float(blue_mask(_box(rgb, END_OK_BUTTON)).mean()))


def end_screen_crowns(rgb: np.ndarray) -> tuple[int, int]:
    """(own, enemy) crowns shown on the end screen: gold cushion slots per banner."""
    out = {}
    for side, (v0, v1) in CROWN_ROWS_V.items():
        out[side] = sum(float(gold_mask(_box(rgb, (u - CROWN_HALF_U, v0, u + CROWN_HALF_U, v1))).mean())
                        >= CROWN_MIN for u in CROWN_SLOTS_U)
    return int(out["own"]), int(out["enemy"])


def battle_outcome(own: int | None, enemy: int | None) -> str:
    """"win" / "loss" / "draw" from the crowns, "unknown" without them."""
    if own is None or enemy is None:
        return "unknown"
    return "win" if own > enemy else "loss" if own < enemy else "draw"


# ---------------------------------------------------------------------------
# battle state
# ---------------------------------------------------------------------------
IDLE, IN_BATTLE = "idle", "battle"


class BattleState:
    """When a battle is on, from the VisionFrames (screen, tick, t_capture).

    Start: `start_frames` battle-screen frames with the clock read, uninterrupted by another
    screen, over which the tick advanced by `start_advance_ticks` (one displayed second): a
    clock frozen on the end screen or a paused game never starts a battle. Frames whose clock
    is unread neither count nor interrupt.
    End: the end screen for `end_frames` frames; the clock read but not advancing for
    `frozen_s` (the game is over); a clock that jumps back more than `reset_ticks` (a new
    battle the end screen of which was missed); overtime 0:00; or `lost_s` without a
    battle-screen frame. An unread clock alone ends nothing (the bot only stops deciding).
    running(t): the clock advanced within `running_s` of t (decisions need it).
    """

    def __init__(self, *, start_frames: int = 3, start_advance_ticks: int = 20, end_frames: int = 3,
                 frozen_s: float = 2.5, lost_s: float = 8.0, running_s: float = 1.2,
                 reset_ticks: int = 200):
        self.start_frames = max(1, int(start_frames))
        self.start_advance_ticks = int(start_advance_ticks)
        self.end_frames = max(1, int(end_frames))
        self.frozen_s = float(frozen_s)
        self.lost_s = float(lost_s)
        self.running_s = float(running_s)
        self.reset_ticks = int(reset_ticks)
        self.state = IDLE
        self.number = 0
        self.ended_reason: str | None = None
        self.last_tick: int | None = None
        self.last_advance_t: float | None = None
        self._last_battle_t: float | None = None
        self._end_count = 0
        self._cand: list[tuple[float, int]] = []

    @property
    def in_battle(self) -> bool:
        return self.state == IN_BATTLE

    def running(self, t: float) -> bool:
        return (self.in_battle and self.last_advance_t is not None
                and t - self.last_advance_t <= self.running_s)

    def update(self, vf: VisionFrame) -> list[str]:
        """Advance on one frame; returns the events of this frame: "end", "start"."""
        t, tick, screen = float(vf.t_capture), vf.tick, vf.screen
        if self.state == IN_BATTLE:
            reason = None
            if screen == SCREEN_END:
                self._end_count += 1
                if self._end_count >= self.end_frames:
                    reason = "end screen"
            else:
                self._end_count = 0
            if screen == SCREEN_BATTLE:
                self._last_battle_t = t
            if reason is None and tick is not None:
                if tick > self.last_tick:
                    self.last_tick, self.last_advance_t = int(tick), t
                elif tick < self.last_tick - self.reset_ticks:
                    reason = "clock reset"
            if reason is None and screen == SCREEN_BATTLE and vf.hud.phase == PHASE_ENDED:
                reason = "overtime over"
            if reason is None and tick is not None and t - self.last_advance_t > self.frozen_s:
                reason = "clock stopped"         # read, and not moving: the game is over
            if reason is None and t - self._last_battle_t > self.lost_s:
                reason = "left the battle screen"
            if reason is None:
                return []
            self.state, self.ended_reason = IDLE, reason
            self._cand.clear()
            return ["end"]
        if screen != SCREEN_BATTLE:
            self._cand.clear()
            return []
        if tick is None:
            return []
        if self._cand and tick < self._cand[-1][1]:
            self._cand.clear()
        self._cand.append((t, int(tick)))
        if len(self._cand) > 256:
            # A clock frozen on a battle screen (a pause, an intro) must not grow this without
            # bound. Ticks here never decrease, so the first and the newest entries are all the
            # start rule reads: drop the middle.
            del self._cand[1:-64]
        if (len(self._cand) >= self.start_frames
                and self._cand[-1][1] - self._cand[0][1] >= self.start_advance_ticks):
            self.state, self.number, self.ended_reason = IN_BATTLE, self.number + 1, None
            self.last_tick, self.last_advance_t = int(tick), t
            self._last_battle_t, self._end_count = t, 0
            self._cand.clear()
            return ["start"]
        return []


@dataclass
class BattleContext:
    """What the runner keeps for one battle."""
    number: int
    started_t: float
    start_tick: int | None
    memory: Any = field(default_factory=rs_memory.PublicMemory)
    pending: list = field(default_factory=list)          # rs_decide.Pending, our taps in flight
    last_decision_tick: int | None = None
    opp_cards: list[int] = field(default_factory=list)   # opponent cards seen, in order
    decisions: int = 0
    taps: int = 0
    opp_plays: int = 0                # UNKNOWN plays (card -1) included
    opp_unknown: int = 0              # UNKNOWN plays (--unknown-plays)
    own_plays: int = 0
    last_hud: Hud | None = None
    observer: Any = None              # v2 rows: the battle's public_state.LiveObserver


class ResultBook:
    """The result of each battle: the HUD's crowns at the end, then the end screen's."""

    SETTLE_S = 3.0       # end-screen frames read this long (the crowns animate in)
    WAIT_S = 20.0        # no end screen this long after the end: the HUD's crowns stand

    def __init__(self):
        self.open_rec: dict[str, Any] | None = None
        self.tally = Counter()
        self.done = 0

    @property
    def pending(self) -> bool:
        return self.open_rec is not None

    def open(self, battle: int, reason: str | None, t: float, hud_crowns: Sequence[int],
             **extra: Any) -> None:
        self.open_rec = {"battle": battle, "reason": reason, "t_end": t,
                         "hud_crowns": [int(c) for c in hud_crowns], "end_crowns": None,
                         "end_frames": 0, "end_first_t": None, **extra}

    def end_frame(self, crowns: Sequence[int], t: float) -> None:
        rec = self.open_rec
        if rec is None:
            return
        rec["end_frames"] += 1
        if rec["end_first_t"] is None:
            rec["end_first_t"] = t
        old = rec["end_crowns"] or [0, 0]
        rec["end_crowns"] = [max(int(old[0]), int(crowns[0])), max(int(old[1]), int(crowns[1]))]

    def due(self, screen: str, t: float) -> bool:
        rec = self.open_rec
        if rec is None:
            return False
        if rec["end_frames"]:
            return screen != SCREEN_END or t - rec["end_first_t"] >= self.SETTLE_S
        return t - rec["t_end"] >= self.WAIT_S

    def finish(self) -> dict[str, Any] | None:
        """Close the open result; crowns are the larger of the two readings per side
        (perception misses crowns rather than inventing them)."""
        rec, self.open_rec = self.open_rec, None
        if rec is None:
            return None
        hud, end = rec["hud_crowns"], rec["end_crowns"]
        crowns = hud if end is None else [max(hud[0], end[0]), max(hud[1], end[1])]
        rec["crowns"] = crowns
        rec["source"] = "hud" if end is None else "end screen + hud"
        rec["outcome"] = battle_outcome(crowns[0], crowns[1])
        self.tally[rec["outcome"]] += 1
        self.done += 1
        rec["record"] = dict(self.tally)
        rec.pop("end_first_t", None)
        return rec


# ---------------------------------------------------------------------------
# taps
# ---------------------------------------------------------------------------
class TapError(RuntimeError):
    """A tap that cannot be sent (off the screen, adb shell gone, wrong device size)."""


def _no_window() -> dict:
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)          # Windows only
    return {"creationflags": flag} if flag else {}


_WM_SIZE = re.compile(r"^\s*(Physical|Override) size:\s*(\d+)\s*x\s*(\d+)", re.I)


def parse_wm_size(text: str) -> tuple[int, int] | None:
    """`adb shell wm size` output -> (width, height): the override size if set, else the
    physical one; None when neither is there."""
    found: dict[str, tuple[int, int]] = {}
    for line in (text or "").splitlines():
        m = _WM_SIZE.match(line)
        if m:
            found[m.group(1).lower()] = (int(m.group(2)), int(m.group(3)))
    return found.get("override") or found.get("physical")


def device_size(adb: str, serial: str, *, run: Callable = subprocess.run) -> tuple[int, int] | None:
    """The device's screen size by `adb shell wm size`, None when it cannot be read."""
    try:
        out = run([str(adb), "-s", str(serial), "shell", "wm", "size"], capture_output=True,
                  text=True, encoding="utf-8", errors="replace", timeout=15, **_no_window())
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_wm_size(getattr(out, "stdout", "") or "")


class AdbTapper:
    """One `adb -s SERIAL shell` kept open; a play is two `input tap X Y` lines written to it.

    A fresh `adb shell` per play cost ~40 ms each time (live_play.py, measured 2026-10-10).
    verify(serial, adb) runs BEFORE the shell is opened (scripts/emulator.verify: raises
    SystemExit unless the serial is the Clash Royale instance), then the device's screen size
    must be `size` (DEVICE_SIZE) when `adb shell wm size` answers. Lines are written as bytes
    with a bare "\n" (no Windows "\r\n" reaches the device shell). Every tap is checked to lie
    on the screen. A shell that died (adb server restarted, device reconnected) is reopened
    only after verify() passes again; a refusal then (SystemExit) leaves the tapper `dead`:
    every later tap_pair raises TapError and nothing reaches the device."""

    def __init__(self, adb: str, serial: str, *, verify: Callable[[str, str], Any] | None,
                 size: Sequence[int] = DEVICE_SIZE, popen: Callable = subprocess.Popen,
                 run: Callable = subprocess.run):
        self.adb, self.serial = str(adb), str(serial)
        self.size = (int(size[0]), int(size[1]))
        if verify is None:
            raise TapError("taps need scripts/emulator.verify; refusing to open the adb shell")
        self._verify = verify
        self._popen = popen
        self.p = None
        self.sent = 0
        self.opened = 0
        self.dead: str | None = None
        verify(self.serial, self.adb)          # SystemExit unless this is the Clash Royale instance
        found = device_size(self.adb, self.serial, run=run)
        if found is not None and tuple(found) != self.size:
            raise TapError(f"{self.serial} reports a {found[0]}x{found[1]} screen; taps are computed "
                           f"for {self.size[0]}x{self.size[1]}. Set MuMu's resolution to "
                           f"{self.size[0]}x{self.size[1]} (do not let the bot change it).")
        self.device_size_seen = found
        self._open()

    def _open(self) -> None:
        self.p = self._popen([self.adb, "-s", self.serial, "shell"], stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **_no_window())
        self.opened += 1

    def _reopen(self) -> None:
        """A new shell in place of one that died, after emulator.verify passes again (the
        emulator may have been restarted since: never by this program)."""
        self._kill()
        try:
            self._verify(self.serial, self.adb)
        except (SystemExit, Exception) as exc:   # verify refuses with SystemExit
            self.dead = (f"emulator.verify refused {self.serial} before reopening the adb shell "
                         f"({type(exc).__name__}: {exc}); no more taps this run")
            raise TapError(self.dead) from None
        self._open()

    def _check(self, xy: Sequence[Any]) -> tuple[int, int]:
        x, y = int(xy[0]), int(xy[1])
        if not (0 <= x < self.size[0] and 0 <= y < self.size[1]):
            raise TapError(f"tap ({x}, {y}) is off the {self.size[0]}x{self.size[1]} screen")
        return x, y

    def tap_pair(self, a: Sequence[Any], b: Sequence[Any]) -> None:
        """Tap `a` (the card slot) then `b` (the arena tile), device pixels."""
        if self.dead:
            raise TapError(self.dead)
        (ax, ay), (bx, by) = self._check(a), self._check(b)
        line = f"input tap {ax} {ay}\ninput tap {bx} {by}\n".encode("ascii")
        last: Exception | None = None
        for _ in range(2):
            try:
                if self.p is None or self.p.poll() is not None:
                    self._reopen()               # TapError (not caught here) when verify refuses
                self.p.stdin.write(line)
                self.p.stdin.flush()
                self.sent += 1
                return
            except (OSError, ValueError, AttributeError) as exc:
                last = exc
                self._kill()
        raise TapError(f"the adb shell does not take taps ({type(last).__name__}: {last})")

    def _kill(self) -> None:
        p, self.p = self.p, None
        if p is None:
            return
        try:
            p.stdin.close()
        except Exception:
            pass
        try:
            p.terminate()
        except Exception:
            pass

    def close(self) -> None:
        p = self.p
        self._kill()
        if p is not None:
            try:
                p.wait(timeout=3)
            except Exception:
                pass


class ScreenGuard:
    """The last look before a tap: the newest frame the source holds, when it is newer than
    the frame the decision was made on, must still show the battle screen (the HUD panel's
    blue edge, no end screen). A battle that ends between capture and tap therefore gets no
    tap: on the end screen the OK button covers hand slot 1 (u 0.37-0.63, v 0.88-0.94 against
    the slot's 0.50, 0.89), and pressing it would navigate. A source that cannot be read
    vetoes too. For live sources only (read() returns the newest frame without consuming
    it: capture.LatestFrameSource, FallbackSource); a replay never taps."""

    def __init__(self, source: Any):
        self.source = source
        self.vetoes = 0

    def _veto(self, why: str) -> str:
        self.vetoes += 1
        return why

    def __call__(self, frame: Any) -> str | None:
        """None when the tap may go; else why not."""
        try:
            newest = self.source.read()
        except Exception as exc:
            return self._veto(f"no newest frame ({_short(exc)})")
        if newest is None or float(newest.t) <= float(frame.t):
            return None                          # nothing newer than what was decided on
        cues = read_cues(np.asarray(newest.rgb))
        if cues.end_screen:
            return self._veto("the end screen is up")
        if not cues.hud_present:
            return self._veto("the battle screen is gone")
        return None


# ---------------------------------------------------------------------------
# one runner at a time
# ---------------------------------------------------------------------------
_LOCK_OFFSET = 4096        # Windows locks a byte past the text, so others can still read it


class InstanceLock:
    """An exclusive OS lock on a file, held while the runner lives (released by the OS when
    the process dies, so a crash leaves no stale lock). msvcrt on Windows, fcntl elsewhere."""

    def __init__(self, path: str | os.PathLike = LOCK_FILE):
        self.path = Path(path)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> bool:
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, f"pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
        except OSError:
            pass
        return True

    def holder(self) -> str:
        """What the lock file says about its holder (best effort)."""
        try:
            return self.path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)
        try:
            self.path.unlink()
        except OSError:
            pass

    def __enter__(self) -> "InstanceLock":
        if not self.acquire():
            raise RuntimeError(f"another vision_play holds {self.path} ({self.holder()})")
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# ---------------------------------------------------------------------------
# perception
# ---------------------------------------------------------------------------
class LivePerception:
    """Frame -> VisionFrame: screen cues, HUD, detector, HP bars, tracker and events.

    hud_reader: scripts/vision/hud.HudReader (or anything with read(rgb, t) -> Hud and
    reset()). detector: scripts/vision/detector.Detector, or None (no units: the HUD is still
    read). tracker / deriver: scripts/vision/tracker.Tracker and events.EventDeriver. The HUD
    is read only on battle-screen frames, and the detector only runs when there is a tick to
    track on: the HUD clock's, or, while the clock is briefly unread, one extrapolated from
    the last reading (at most `max_extrapolate_s`).
    """

    def __init__(self, *, hud_reader: Any, tracker: Any, deriver: Any, geometry: Any,
                 detector: Any = None, max_extrapolate_s: float = 5.0, hud_reset_gap_s: float = 3.0):
        self.hud_reader = hud_reader
        self.tracker = tracker
        self.deriver = deriver
        self.geometry = geometry
        self.detector = detector
        self.max_extrapolate_s = float(max_extrapolate_s)
        self.hud_reset_gap_s = float(hud_reset_gap_s)
        self._anchor: tuple[int, float] | None = None
        self._last_panel_t: float | None = None
        self.detect_frames = 0

    def new_battle(self) -> None:
        self.tracker.reset()
        self.deriver.reset()

    def end_battle(self) -> None:
        self.hud_reader.reset()
        self.tracker.reset()
        self.deriver.reset()
        self._anchor = None

    def note_own_tap(self, card: int, tick: int, *, slot: int, aim_mt: Sequence[float] | None,
                     t: float | None) -> None:
        aim = (None, None) if aim_mt is None else (float(aim_mt[0]), float(aim_mt[1]))
        self.deriver.note_own_tap(int(card), int(tick), slot=int(slot), aim_x_mt=aim[0],
                                  aim_y_mt=aim[1], t=t, tracker=self.tracker)

    def track_cards(self, tracks: Iterable[Any]) -> dict[int, int]:
        return self.deriver.track_cards(tracks)

    def _track_tick(self, tick: int | None, t: float) -> int | None:
        if tick is not None:
            self._anchor = (int(tick), float(t))
            return int(tick)
        if self._anchor is None or t - self._anchor[1] > self.max_extrapolate_s:
            return None
        return self._anchor[0] + int(round((t - self._anchor[1]) * 20))

    def perceive(self, frame: Any) -> VisionFrame:
        rgb = np.asarray(frame.rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"expected an HxWx3 RGB frame, got {rgb.shape}")
        w, h = check_frame_size(rgb.shape[1], rgb.shape[0])
        t = float(frame.t)
        vf = VisionFrame(t_capture=t, frame_size=(w, h), stale_s=float(getattr(frame, "stale_s", 0.0)))
        cues = read_cues(rgb)
        if cues.end_screen:
            vf.screen = SCREEN_END
            vf.hud = Hud(crowns=end_screen_crowns(rgb), phase=PHASE_ENDED)
            return vf
        if not cues.hud_present:
            vf.screen = SCREEN_MENU
            return vf
        vf.screen = SCREEN_BATTLE
        if self._last_panel_t is not None and t - self._last_panel_t > self.hud_reset_gap_s:
            self.hud_reader.reset()                  # back on a battle screen after a while
        self._last_panel_t = t
        vf.hud = self.hud_reader.read(rgb, t)
        vf.tick = vf.hud.clock_tick
        tick = self._track_tick(vf.tick, t)
        if tick is None or self.detector is None:
            return vf
        res = self.detector.detect(rgb)
        units, spells, bars = res.units(), res.spells(), res.bars()
        reads = hpbars.read_unit_hp(rgb, [d.box for d in units], [b.box for b in bars])
        obs = observations_from_detections(list(units) + list(spells), self.geometry, res.frame_size)
        hpbars.annotate_observations(obs[:len(units)], reads)
        vf.tracks = self.tracker.update(obs, tick)
        vf.detections = list(res.detections)
        vf.events = self.deriver.update(vf.tracks, tick, t=t, tracker=self.tracker)
        self.detect_frames += 1
        return vf


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------
class Decider:
    """VisionFrame -> rs_decide.Decision through the public state and the model's row.

    state_fn(vf, ctx) -> the state dict (tools/public_state); row_fn(state, mem_view, tick)
    -> the obs_row (tools/rs_rows); policy / legality / config / rng as rs_decide.decide
    takes them. Our pending taps are pruned against the HUD's hand first (rs_decide).
    observer(ctx) -> a new battle's public_state.LiveObserver (v2 rows; make_observer), None
    for legacy rows: the runner keeps it in ctx.observer (observer_for)."""

    def __init__(self, *, policy: Any, legality: Any, config: Any,
                 state_fn: Callable[[VisionFrame, BattleContext], dict],
                 row_fn: Callable[[dict, dict, int], Mapping[str, Any]], rng: Any = None,
                 team: int = OWN_TEAM, sub: int = RD.SUBTILE_PER_MILLITILE,
                 observer: Callable[[BattleContext], Any] | None = None):
        self.policy = RD.as_policy(policy)
        self.legality = legality
        self.config = config
        self.state_fn = state_fn
        self.row_fn = row_fn
        self.rng = rng if rng is not None else np.random.default_rng()
        self.team = int(team)
        self.sub = int(sub)
        self.observer = observer

    def observer_for(self, ctx: BattleContext) -> Any:
        """The battle's LiveObserver (v2 rows), None for legacy rows."""
        return None if self.observer is None else self.observer(ctx)

    def decide(self, vf: VisionFrame, ctx: BattleContext) -> tuple[Any, dict]:
        now_tick = int(vf.tick)
        ctx.pending = RD.prune_pending(ctx.pending, list(vf.hud.hand), now_tick)
        state = self.state_fn(vf, ctx)
        row = self.row_fn(state, ctx.memory.view(now_tick, 0), now_tick)
        d = RD.decide(self.policy, row, state, self.team, self.legality, self.config, rng=self.rng,
                      pending=ctx.pending, tick=now_tick,
                      dt=RD.cadence_dt(now_tick, ctx.last_decision_tick), sub=self.sub)
        ctx.last_decision_tick = now_tick
        ctx.decisions += 1
        return d, state


# ---------------------------------------------------------------------------
# the checkpoint's rows: legacy or v2
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RowPlan:
    """Which rows the runner builds for a checkpoint (row_plan): its meta "schema" (legacy:
    rs_rows/legacy or absent; v2: rs_rows/v2) and, for v2, the tap-to-run delay its rows
    were built with (rs_decide.obs_delay_ticks: own_pending's clock; 0 shows no pending)."""
    schema: str
    v2: bool = False
    obs_delay: int = 0

    def to_json(self) -> dict[str, Any]:
        return {"schema": self.schema, "v2": self.v2, "obs_delay": self.obs_delay}


def row_plan(meta: Mapping[str, Any] | None) -> RowPlan:
    """The RowPlan of a checkpoint meta. No "schema" (every checkpoint before E14) or
    rs_rows/legacy: the legacy rows, built exactly as before. rs_rows/v2 (or "v2",
    rs_decide.is_v2_meta): v2 rows. ValueError for any other schema."""
    meta = meta or {}
    schema = meta.get("schema", rs_schema.SCHEMA_LEGACY)
    if RD.is_v2_meta(meta):
        return RowPlan(rs_schema.SCHEMA_V2, True, RD.obs_delay_ticks(meta))
    if schema != rs_schema.SCHEMA_LEGACY:
        raise ValueError(f"the checkpoint wants rows of schema {schema!r}; this runner builds "
                         f"{rs_schema.SCHEMA_LEGACY!r} or {rs_schema.SCHEMA_V2!r}")
    return RowPlan(rs_schema.SCHEMA_LEGACY)


def make_state_fn(builder: Any, perception: Any, plan: RowPlan
                  ) -> Callable[[VisionFrame, BattleContext], dict]:
    """Decider's state_fn: builder.state (public_state.PublicStateBuilder) of the frame with
    the opponent cards seen, the creating card of each track and the elixir of our taps in
    flight; for v2 rows also the battle's observer (ctx.observer)."""
    def state_fn(vf: VisionFrame, ctx: BattleContext) -> dict:
        kw = {"observer": ctx.observer} if plan.v2 else {}
        return builder.state(vf, opp_deck=ctx.opp_cards, track_cards=perception.track_cards(vf.tracks),
                             pending_cost=sum(max(0, int(p.cost)) for p in ctx.pending), **kw)
    return state_fn


def make_row_fn(obs_row: Callable[..., Mapping[str, Any]], sub: int, plan: RowPlan, *,
                live_compat: bool, units: str,
                row_inputs: Callable[[Mapping[str, Any]], dict] | None = None
                ) -> Callable[[dict, dict, int], Mapping[str, Any]]:
    """Decider's row_fn: obs_row (tools/rs_rows.obs_row) of our side (OWN_TEAM) with the
    checkpoint's unit tokens; for v2 rows also row_inputs(state) (public_state.row_inputs:
    schema, opp_elixir and pending from the state's observer block)."""
    if not plan.v2:
        def row_fn(state: dict, view: dict, tick: int) -> Mapping[str, Any]:
            return obs_row(state, OWN_TEAM, sub, view, tick, live=live_compat,
                           units=None if live_compat else units)
        return row_fn
    if row_inputs is None:
        import public_state
        row_inputs = public_state.row_inputs

    def row_fn_v2(state: dict, view: dict, tick: int) -> Mapping[str, Any]:
        return obs_row(state, OWN_TEAM, sub, view, tick, live=live_compat,
                       units=None if live_compat else units, **row_inputs(state))
    return row_fn_v2


def make_observer(catalogue: Sequence[Sequence[Any]], plan: RowPlan, *,
                  fields: Sequence[str] | None = None, law: Any = None, blind: bool = False,
                  observer_cls: Any = None) -> Callable[[BattleContext], Any] | None:
    """Decider's observer: for v2 rows, ctx -> a public_state.LiveObserver for the battle
    (delay plan.obs_delay, watching from the battle's start tick; blind: no detector, so no
    play is seen and the count is never exact); None for legacy rows."""
    if not plan.v2:
        return None
    if observer_cls is None:
        import public_state
        observer_cls = public_state.LiveObserver

    def observer(ctx: BattleContext) -> Any:
        return observer_cls(catalogue, delay=plan.obs_delay, fields=fields, law=law,
                            start_tick=ctx.start_tick, blind=blind)
    return observer


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------
def _short(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class VisionPlay:
    """The live loop (module doc). Every part is injected, so tests run it with fakes.

    source      a FrameSource: read() -> capture.Frame (t, rgb, stale_s) or None; a replay
                sets `exhausted` when it is done
    perception  LivePerception (perceive, new_battle, end_battle, note_own_tap, track_cards)
    decider     Decider
    control     scripts/vision/control.ControlState (mode, taps_enabled, STOP)
    recorder    scripts/vision/record.VisionRecorder (write, log)
    geometry    ArenaGeometry: card slot and tile -> DEVICE pixels (device_size)
    tapper      AdbTapper (tap_pair) or None (no taps possible: a dry run)
    tap_guard   ScreenGuard, or any guard(frame) -> None (tap) / why not; asked right before
                each tap (None: no last look, as in the tests' scripted sources)
    seconds / battles   stop after this long / after this many battle results (0: never)
    clock       the vision clock (types.now); frame times are on it
    out         where the status lines go (print)
    unknown_plays  the deriver reports UNKNOWN plays (--unknown-plays, N02): each battle's end
                record then also counts them ("opp_unknown")
    """

    def __init__(self, *, source: Any, perception: Any, decider: Decider, control: Any,
                 recorder: Any, geometry: Any, tapper: Any = None, battle: BattleState | None = None,
                 card_names: Sequence[str] | Mapping[int, str] | None = None,
                 seconds: float = 0.0, battles: int = 0, device_size: Sequence[int] = DEVICE_SIZE,
                 idle_record_every: int = 20, status_every_s: float = 1.0,
                 clock: Callable[[], float] = now, sleep: Callable[[float], None] = time.sleep,
                 out: Callable[[str], Any] = print, poll_s: float = 0.002, max_error_logs: int = 50,
                 max_stale_s: float = MAX_STALE_S,
                 tap_guard: Callable[[Any], str | None] | None = None,
                 unknown_plays: bool = False):
        self.unknown_plays = bool(unknown_plays)
        self.source = source
        self.perception = perception
        self.decider = decider
        self.control = control
        self.recorder = recorder
        self.geometry = geometry
        self.tapper = tapper
        self.tap_guard = tap_guard
        self.vetoed = 0
        self.battle = battle if battle is not None else BattleState()
        self.card_names = card_names
        self.seconds = float(seconds or 0.0)
        self.battles = int(battles or 0)
        self.device_size = (int(device_size[0]), int(device_size[1]))
        self.idle_record_every = max(1, int(idle_record_every))
        self.status_every_s = float(status_every_s)
        self.clock = clock
        self.sleep = sleep
        self.out = out
        self.poll_s = float(poll_s)
        self.max_error_logs = int(max_error_logs)
        self.max_stale_s = float(max_stale_s)
        self.results = ResultBook()
        self.ctx: BattleContext | None = None
        self.last_vf: VisionFrame | None = None
        self.last_decision: dict[str, Any] | None = None
        self.frames = 0
        self.taps = 0
        self.errors = Counter()
        self.stop_reason: str | None = None
        self._last_t: float | None = None
        self._last_status: float | None = None
        self._fps_frames = 0
        self._fps = 0.0

    # -- names ----------------------------------------------------------------
    def name(self, card: int | None) -> str:
        if card is None or int(card) < 0:
            return "-"
        names = self.card_names
        try:
            if isinstance(names, Mapping):
                return str(names.get(int(card), card))
            if names is not None and 0 <= int(card) < len(names):
                return str(names[int(card)])
        except Exception:
            pass
        return str(card)

    # -- logging ----------------------------------------------------------------
    def _log(self, kind: str, **data: Any) -> None:
        try:
            self.recorder.log(kind, **data)
        except Exception as exc:                      # a full disk must not stop the battle
            self.errors["log"] += 1
            if self.errors["log"] <= 3:
                self.out(f"log failed ({_short(exc)})")

    def _error(self, kind: str, exc: BaseException, **data: Any) -> None:
        self.errors[kind] += 1
        n = self.errors[kind]
        if n <= self.max_error_logs or n % 100 == 0:
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            self._log(kind, error=_short(exc), count=n, trace="".join(tb[-3:]), **data)
        if n <= 3:
            self.out(f"{kind}: {_short(exc)}")

    # -- the loop -----------------------------------------------------------------
    def run(self) -> int:
        start = self.clock()
        self._last_status = start
        self._log("session_start", mode=self.control.mode, taps_enabled=bool(self.control.taps_enabled),
                  tapper=self.tapper is not None, seconds=self.seconds, battles=self.battles,
                  device_size=list(self.device_size))
        try:
            while True:
                if self.control.stop_requested:
                    self.stop_reason = f"STOP ({self.control.stop_reason})"
                    break
                if self.seconds and self.clock() - start >= self.seconds:
                    self.stop_reason = f"{self.seconds:g} s"
                    break
                if self.battles and self.results.done >= self.battles:
                    self.stop_reason = f"{self.battles} battle(s)"
                    break
                try:
                    frame = self.source.read()
                except Exception as exc:              # a bad grab or a broken replay image
                    self._error("source_error", exc)
                    self.sleep(0.05)
                    continue
                if frame is None or (self._last_t is not None and frame.t <= self._last_t):
                    if getattr(self.source, "exhausted", False):
                        self.stop_reason = "the frame source ended"
                        break
                    self._status()
                    self.sleep(self.poll_s)
                    continue
                self._last_t = frame.t
                self.step(frame)
                self._status()
        except KeyboardInterrupt:
            self.stop_reason = "Ctrl+C"
        finally:
            if self.results.pending:
                self._finish_result()
            self._log("session_end", reason=self.stop_reason, frames=self.frames, taps=self.taps,
                      vetoed=self.vetoed, errors=dict(self.errors), results=dict(self.results.tally),
                      control=dict(getattr(self.control, "counts", {}) or {}))
            self._flush_control()
        self.out(f"stopped: {self.stop_reason}; {self.frames} frames, {self.taps} taps, "
                 f"results {dict(self.results.tally) or '-'}, errors {dict(self.errors) or '-'}")
        return 0

    def _flush_control(self) -> None:
        try:
            self.control.flush_to(self.recorder)
        except Exception as exc:
            self.errors["log"] += 1
            if self.errors["log"] <= 3:
                self.out(f"log failed ({_short(exc)})")

    def step(self, frame: Any) -> None:
        """One frame: perceive, follow the battle, maybe decide and tap, record."""
        self.frames += 1
        self._fps_frames += 1
        t0 = self.clock()
        try:
            vf = self.perception.perceive(frame)
        except Exception as exc:                       # no VisionFrame: nothing is decided
            self._error("frame_error", exc, t=float(frame.t))
            return
        ms = {"stale": round(float(getattr(frame, "stale_s", 0.0)) * 1000.0, 2),
              "perceive": round((self.clock() - t0) * 1000.0, 2)}
        self.last_vf = vf
        out = None
        try:
            out = self._after_perception(vf, frame, ms)
        except Exception as exc:                       # before the tap: no tap this frame
            self._error("decide_error", exc, tick=vf.tick)
        self._record(vf, frame, out, ms)
        self._flush_control()

    def _after_perception(self, vf: VisionFrame, frame: Any, ms: dict) -> dict | None:
        events = self.battle.update(vf)
        if "end" in events:
            self._end_battle(vf)
        if "start" in events:
            self._start_battle(vf)
        if self.results.pending:
            if vf.screen == SCREEN_END:
                self.results.end_frame(vf.hud.crowns, vf.t_capture)
            if self.results.due(vf.screen, vf.t_capture):
                self._finish_result()
        ctx = self.ctx
        if ctx is None or not self.battle.in_battle:
            return None
        if vf.screen == SCREEN_BATTLE:
            ctx.last_hud = vf.hud
        if vf.events.opp_plays:
            EV.record_plays(ctx.memory, vf.events.opp_plays)     # an UNKNOWN play: token 1
            for p in vf.events.opp_plays:
                ctx.opp_plays += 1
                if int(p.card) < 0:                    # UNKNOWN: no card to add to their deck
                    ctx.opp_unknown += 1
                elif int(p.card) not in ctx.opp_cards:
                    ctx.opp_cards.append(int(p.card))
        if vf.events.own_landings:
            ctx.own_plays += sum(1 for x in vf.events.own_landings if x.land_tick is not None)
        if ctx.observer is not None:                   # v2 rows: the count and our taps in flight
            ctx.observer.events(vf.events)
        if vf.tick is not None or vf.events.own_landings:
            self.control.attribute(vf.events.own_landings, tick=vf.tick)
        if not self.can_decide(vf):
            return None
        return self._decide(vf, frame, ctx, ms)

    def _tap_veto(self, frame: Any) -> str | None:
        """The tap guard's answer for the frame decided on; a guard that fails vetoes."""
        guard = self.tap_guard
        if guard is None:
            return None
        try:
            return guard(frame)
        except Exception as exc:
            return f"screen check failed ({_short(exc)})"

    def can_decide(self, vf: VisionFrame) -> bool:
        """In a battle, on the battle screen, the clock read and running, past the deploy
        lockout, fresh pixels, and a tick we have not decided on yet."""
        ctx = self.ctx
        return (ctx is not None and self.battle.in_battle and vf.screen == SCREEN_BATTLE
                and vf.tick is not None and vf.tick >= MIN_DECIDE_TICK
                and float(vf.stale_s) <= self.max_stale_s
                and vf.hud.phase != PHASE_ENDED and self.battle.running(vf.t_capture)
                and (ctx.last_decision_tick is None or vf.tick > ctx.last_decision_tick))

    def _decide(self, vf: VisionFrame, frame: Any, ctx: BattleContext, ms: dict) -> dict:
        t0 = self.clock()
        d, state = self.decider.decide(vf, ctx)
        ms["decide"] = round((self.clock() - t0) * 1000.0, 2)
        s = self.control.suggest(**d.suggest_kwargs())
        px = None
        if d.is_play:
            px = {"card": list(self.geometry.card_slot_px(d.slot, size=self.device_size)),
                  "tile": list(self.geometry.tap_px(d.tx, d.ty, size=self.device_size))}
        tapped, why = False, ""
        if d.is_play and self.control.should_tap(s):
            veto = None if self.tapper is None else self._tap_veto(frame)
            if self.tapper is None:
                why = "no tapper"
            elif veto:
                why = f"not tapped: {veto}"
                self.vetoed += 1
                self._log("tap_vetoed", battle=ctx.number, tick=int(vf.tick), card=d.card,
                          name=self.name(d.card), reason=veto)
            else:
                t1 = self.clock()
                try:
                    self.tapper.tap_pair(tuple(px["card"]), tuple(px["tile"]))
                    tapped = True
                except Exception as exc:
                    why = f"tap failed: {_short(exc)}"
                    self._error("tap_error", exc, tick=vf.tick)
                ms["tap"] = round((self.clock() - t1) * 1000.0, 2)
        if tapped:
            self.taps += 1
            ctx.taps += 1
            ctx.pending.append(RD.Pending(d.card, d.cost, d.slot, int(vf.tick)))
            ms["capture_to_tap"] = round((self.clock() - float(frame.t)) * 1000.0, 2)
            self._log("tap", battle=ctx.number, tick=int(vf.tick), card=d.card, name=self.name(d.card),
                      slot=d.slot, model_tile=[d.tx, d.ty], screen_tile=list(d.world_tile),
                      card_px=px["card"], tile_px=px["tile"], ms=dict(ms))
            try:
                self.perception.note_own_tap(d.card, int(vf.tick), slot=d.slot, aim_mt=d.world_mt,
                                             t=self.clock())
            except Exception as exc:
                self._error("note_tap_error", exc, tick=vf.tick)
            if ctx.observer is not None:               # v2 rows: in own_pending until it lands
                try:
                    ctx.observer.sent(d.card, int(vf.tick), slot=d.slot, cost=d.cost)
                except Exception as exc:
                    self._error("observer_error", exc, tick=vf.tick)
        extra: dict[str, Any] = {"battle": ctx.number, "px": px, "ms": dict(ms),
                                 "elixir": state["players"][OWN_TEAM].get("elixir_milli")}
        seen = state.get("observer")
        if seen is not None:                           # v2 rows: what the row was given
            extra["opp_elixir"] = list(seen.get("opp_elixir") or ())
            extra["own_pending"] = [list(p) for p in seen.get("own_pending") or ()]
        if why:
            extra["why"] = why
        rec = self.control.note_decision(s, tapped=tapped, **extra)
        if d.is_play or self.last_decision is None:   # the status line keeps the last play
            self.last_decision = rec
        return {"decision": d.to_record(), "action": rec.get("action"), "px": px}

    def _start_battle(self, vf: VisionFrame) -> None:
        if self.results.pending:
            self._finish_result()
        n = self.battle.number
        self.ctx = BattleContext(n, float(vf.t_capture), vf.tick)
        self.perception.new_battle()
        self.control.new_battle()
        self.last_decision = None
        self._log("battle_start", battle=n, tick=vf.tick, mode=self.control.mode)
        self.out(f"battle {n} started (tick {vf.tick}, mode {self.control.mode.upper()})")
        make = getattr(self.decider, "observer_for", None)
        if make is not None:
            try:
                self.ctx.observer = make(self.ctx)
            except Exception as exc:                   # v2 rows then fail: no decision, no tap
                self._error("observer_error", exc, tick=vf.tick)

    def _end_battle(self, vf: VisionFrame) -> None:
        ctx = self.ctx
        hud = ctx.last_hud if ctx is not None and ctx.last_hud is not None else Hud()
        towers = {k: v.to_dict() for k, v in (hud.towers or {}).items()}
        n = ctx.number if ctx is not None else self.battle.number
        stats = {} if ctx is None else {"decisions": ctx.decisions, "taps": ctx.taps,
                                        "opp_plays": ctx.opp_plays, "own_plays": ctx.own_plays,
                                        "opp_cards": [self.name(c) for c in ctx.opp_cards]}
        if ctx is not None and self.unknown_plays:
            stats["opp_unknown"] = ctx.opp_unknown
        self.results.open(n, self.battle.ended_reason, float(vf.t_capture), hud.crowns,
                          last_tick=self.battle.last_tick, towers=towers, **stats)
        self._log("battle_end", battle=n, reason=self.battle.ended_reason, tick=self.battle.last_tick,
                  hud_crowns=list(hud.crowns), towers=towers, **stats)
        self.out(f"battle {n} over ({self.battle.ended_reason}) at tick {self.battle.last_tick}")
        try:
            self.perception.end_battle()
        except Exception as exc:
            self._error("perception_error", exc)
        self.ctx = None

    def _finish_result(self) -> None:
        rec = self.results.finish()
        if rec is None:
            return
        self._log("result", **rec)
        own, enemy = rec["crowns"]
        tally = self.results.tally
        self.out(f"battle {rec['battle']}: {rec['outcome'].upper()} {own}-{enemy} ({rec['source']}); "
                 f"record {tally['win']}-{tally['loss']}-{tally['draw']} (W-L-D)")

    def _record(self, vf: VisionFrame, frame: Any, out: dict | None, ms: dict) -> None:
        active = self.battle.in_battle or vf.screen in (SCREEN_BATTLE, SCREEN_END)
        if not active and self.frames % self.idle_record_every:
            return
        extra: dict[str, Any] = {"battle": self.battle.number if self.battle.in_battle else None,
                                 "state": self.battle.state, "ms": ms}
        if out is not None:
            extra.update(out)
        try:
            self.recorder.write(vf, frame.rgb, **extra)
        except Exception as exc:
            self._error("record_error", exc)

    # -- status line --------------------------------------------------------------
    def _status(self) -> None:
        t = self.clock()
        if self._last_status is None:
            self._last_status = t
            return
        dt = t - self._last_status
        if dt < self.status_every_s:
            return
        self._fps = self._fps_frames / dt if dt > 0 else 0.0
        self._fps_frames = 0
        self._last_status = t
        self.out(self.status_line())

    def mode_label(self) -> str:
        ctl = self.control
        if ctl.stop_requested:
            return "STOPPED"
        if ctl.mode == CT.AUTO:
            if not ctl.taps_enabled or self.tapper is None:
                return "AUTO(dry)"
            return "AUTO(adb shell dead)" if getattr(self.tapper, "dead", None) else "AUTO(tap)"
        return "MANUAL"

    def status_line(self) -> str:
        vf = self.last_vf
        hud = vf.hud if vf is not None else Hud()
        where = f"battle {self.battle.number}" if self.battle.in_battle else (
            vf.screen if vf is not None else "no frame")
        tick = "-" if vf is None or vf.tick is None else str(vf.tick)
        if vf is not None and vf.tick is not None and hud.phase == PHASE_OVERTIME:
            tick += " OT"
        elixir = "-" if hud.elixir_milli is None else f"{hud.elixir_milli / 1000:.1f}"
        hand = "/".join(self.name(c) if c != EMPTY_CARD else "?" for c in hud.hand)
        last = "-"
        rec = self.last_decision
        if rec is not None:
            tile = rec.get("tile")
            what = rec.get("name") or self.name(rec.get("card"))
            last = (f"{what} -> {tuple(tile)}" if tile else str(what)) + \
                f" [{rec.get('action')}] @{rec.get('tick')}"
        errs = sum(self.errors.values())
        return (f"[{where}] tick {tick} | elixir {elixir} | hand {hand} next {self.name(hud.next)} | "
                f"{self.mode_label()} | last {last} | {self._fps:.0f} fps, {self.taps} taps"
                + (f", {errs} errors" if errs else ""))


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def deck_names(text: str | None) -> tuple[str, ...]:
    """--deck: "Hog 2.6" (the default) or 8 catalogue card names separated by commas."""
    raw = (text or DEFAULT_DECK).strip()
    alias = DECK_ALIASES.get(raw.lower())
    if alias is not None:
        return tuple(alias)
    names = tuple(n.strip() for n in raw.split(",") if n.strip())
    if len(names) != 8 or len(set(names)) != 8:
        raise argparse.ArgumentTypeError(
            f"--deck: 'Hog 2.6' or 8 distinct catalogue names separated by commas, got {raw!r}")
    return names


def default_ckpt() -> Path:
    """runs/tokens/live_big_hog/best.pt under the machine's runs root (hastycr.machine)."""
    try:
        from hastycr import machine
        return Path(machine.get().runs) / CKPT_IN_RUNS
    except Exception:
        return ROOT / "runs" / CKPT_IN_RUNS


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="scripts/vision_play.py",
        description="Play the friendly battle on screen with the token model, from screen vision. "
                    "Friendly battles only: start each battle yourself; the runner never navigates. "
                    "Dry run unless --tap.")
    ap.add_argument("--tap", action="store_true",
                    help="send taps (only while you watch). Without it nothing is tapped.")
    ap.add_argument("--ckpt", default=None,
                    help=f"token-model checkpoint (default: <runs>/{CKPT_IN_RUNS.as_posix()})")
    ap.add_argument("--detector", default=None,
                    help="detector weights (default: the pin, <data>/vision/detector_pin.json)")
    ap.add_argument("--device", default=None, help="detector device (default: CUDA GPU 0, else cpu)")
    ap.add_argument("--deck", type=deck_names, default=deck_names(DEFAULT_DECK),
                    help="our deck: 'Hog 2.6' (default) or 8 catalogue names, comma-separated")
    ap.add_argument("--level", type=int, default=FRIENDLY_LEVEL, help="card level (friendlies: 11)")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0: never)")
    ap.add_argument("--battles", type=int, default=0, help="stop after this many battles (0: never)")
    ap.add_argument("--record-images", type=int, default=20, metavar="N",
                    help="save the frame image every N recorded frames (0: never)")
    ap.add_argument("--source", default="auto",
                    help="auto | printwindow | adb | a recorded session to replay (taps refused)")
    ap.add_argument("--geometry", default=None,
                    help="an ArenaGeometry JSON (default data/vision/geometry_default_1080x1920.json)")
    ap.add_argument("--no-window", dest="no_control_window", action="store_true",
                    help="no control window (the same as --no-control-window)")
    ap.add_argument("--seed", type=int, default=None, help="seed of the decision rng")
    ap.add_argument("--no-lane-guard", action="store_true", help="turn off the Cannon lane guard")
    ap.add_argument("--no-cap-rule", action="store_true",
                    help="allow waiting at full elixir with a playable card")
    ap.add_argument("--unknown-plays", action="store_true",
                    help="an enemy unit no card explains is an UNKNOWN play (row token 1, charged "
                         f"{EV.UNKNOWN_PLAY_COST} elixir; the count is then not exact) and an "
                         "unplaceable enemy label stays on the board; default: dropped, as before")
    CT.add_control_args(ap)
    import rs_engine
    rs_engine.add_engine_args(ap)
    return ap


def import_live_modules() -> dict[str, Any]:
    """Every module main() uses, imported (torch, ultralytics and royalesim stay lazy inside
    them). The import-rule test calls this in a fresh interpreter."""
    import importlib
    names = ("public_state", "rs_rows", "rs_checkpoint", "rs_engine", "rs_spawn_table",
             "vision_classes", "scripts.vision.capture", "scripts.vision.detector",
             "scripts.vision.geometry", "scripts.vision.hpbars", "scripts.vision.hud",
             "scripts.vision.record", "scripts.vision.templates", "scripts.vision.tracker",
             "scripts.emulator")
    return {n: importlib.import_module(n) for n in names}


def _say(lines: Iterable[str]) -> None:
    for line in lines:
        print(line, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if STOP_FILE.exists():
        print(f"refusing to start: {STOP_FILE} exists. Delete it to run.")
        return 2
    lock = InstanceLock(LOCK_FILE)
    if not lock.acquire():
        print(f"another vision_play is running ({lock.holder() or LOCK_FILE}); one at a time.")
        return 3
    try:
        return _main(args)
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    finally:
        lock.release()


def _main(args: argparse.Namespace) -> int:
    mods = import_live_modules()
    PS, RR, CK, RE = mods["public_state"], mods["rs_rows"], mods["rs_checkpoint"], mods["rs_engine"]
    ST, VC, CAP = mods["rs_spawn_table"], mods["vision_classes"], mods["scripts.vision.capture"]
    DET, GEO, HUD = mods["scripts.vision.detector"], mods["scripts.vision.geometry"], mods["scripts.vision.hud"]
    REC, TPL, TRK = mods["scripts.vision.record"], mods["scripts.vision.templates"], mods["scripts.vision.tracker"]

    # --- the engine and the model ------------------------------------------------
    eng = RE.config_from_args(args)
    ckpt = Path(args.ckpt) if args.ckpt else default_ckpt()
    if not ckpt.is_file():
        print(f"no checkpoint at {ckpt} (--ckpt)")
        return 2
    import torch
    blob = torch.load(str(ckpt), map_location="cpu")
    meta = blob["meta"]
    try:
        plan = row_plan(meta)                           # legacy rows, or v2 (E14) rows
    except ValueError as exc:
        print(f"{ckpt}: {exc}")
        return 2
    schema = plan.schema
    live_compat = bool(meta.get("live_compat"))
    units = meta.get("units_mode") or RR.UNITS_TYPES
    if units == RR.UNITS_VISION and not live_compat:
        RR.UNITS = RR.UNITS_VISION                      # _setup fills the class lookup
    royalesim, _names, ids, cat = RR._setup(eng)
    sub = int(royalesim.SUBTILE_PER_MILLITILE)
    loaded = CK.load_policy(blob, eng)                  # refuses another engine's vocabulary
    policy = RD.TokenPolicyAdapter(loaded.model)
    names = [str(row[0]) for row in cat]
    missing = [n for n in args.deck if n not in ids]
    if missing:
        print(f"--deck: not in the engine's catalogue: {missing}")
        return 2
    deck_ids = [int(ids[n]) for n in args.deck]
    guard = () if args.no_lane_guard else RD.card_ids(cat, RD.GUARD_CARD_NAMES)
    dcfg = RD.DecideConfig.live(RD.wait_bias_for(str(ckpt)), guard_cards=guard)
    if args.no_cap_rule:
        dcfg = dcfg.replace(cap_milli=None)
    legality = RD.TowersOnlyLegality(eng, royalesim, catalogue=cat)

    # --- vision -------------------------------------------------------------------
    table = VC.load()
    print(f"spawn table {ST.DEFAULT_PATH} (made under this engine on first use)", flush=True)
    spawn = ST.ensure(engine=eng, rs=royalesim)
    tables = EV.EventTables.for_engine(spawn, class_of=EV.unit_class_map(table.doc),
                                       ignore=EV.cardless_classes(table.doc))
    geometry = GEO.load_geometry(args.geometry)
    recorder = REC.VisionRecorder(REPORT_ROOT, prefix="vision_play", image_every=max(0, args.record_images))
    detector, det_error = None, None
    try:
        detector = DET.Detector.load(args.detector, device=args.device, geometry=geometry, table=table)
        detector.warmup()
    except Exception as exc:                           # no weights, no ultralytics, bad pin
        detector, det_error = None, _short(exc)
    store = TPL.TemplateStore()
    hud_reader = HUD.HudReader(store, {n: ids[n] for n in args.deck}, deck=deck_ids)
    tracker = TRK.Tracker(own_classes=table.deck_labels(args.deck))
    # N02 (--unknown-plays): UNKNOWN plays and unknown units; without it, the calls as before
    unknown = bool(args.unknown_plays)
    deriver = EV.EventDeriver(tables, EV.EventConfig(unknown_plays=True) if unknown else None,
                              own_deck=deck_ids)
    perception = LivePerception(hud_reader=hud_reader, tracker=tracker, deriver=deriver,
                                geometry=geometry, detector=detector)
    # v2 rows (E14) also show the spells in flight: spell tracks become "spells" rows in the
    # engine's SPELL_FIELDS layout (rs_rows.SF, filled by _setup). Legacy: as before.
    spell_kw = {"spell_fields": RR.SF} if plan.v2 else {}
    if unknown:
        spell_kw["unknown_units"] = True
    builder = PS.PublicStateBuilder(table, spawn, deck=deck_ids, levels=args.level, catalogue=cat,
                                    fields=RR.F, sub=sub, **spell_kw)
    observer = make_observer(cat, plan, fields=list(getattr(royalesim, "CATALOGUE_FIELDS", ()) or ())
                             or None, law=RR.elixir_law(eng) if plan.v2 else None,
                             blind=detector is None, observer_cls=PS.LiveObserver)
    decider = Decider(policy=policy, legality=legality, config=dcfg,
                      state_fn=make_state_fn(builder, perception, plan),
                      row_fn=make_row_fn(RR.obs_row, sub, plan, live_compat=live_compat, units=units,
                                         row_inputs=PS.row_inputs),
                      rng=np.random.default_rng(args.seed), sub=sub, observer=observer)

    # --- taps: only with --tap, a live source, a detector, and a verified emulator -------------
    replay = args.source not in ("auto", "printwindow", "adb")
    taps = bool(args.tap)
    refusal = None
    if taps and replay:
        taps, refusal = False, "the source is a recording"
    if taps and detector is None:
        taps, refusal = False, "no detector"
    adb, serial, instance = CAP.machine_settings()
    attributor = CT.PlayAttributor(max_lag_ticks=CT.MAX_LAG_TICKS + EV.EventConfig().team_wait_ticks)
    ctl = CT.ControlState.from_args(args, taps_enabled=taps, stop_file=STOP_FILE, names=names,
                                    attributor=attributor)
    tapper = source = None
    try:
        if taps:
            emulator = mods["scripts.emulator"]
            try:
                tapper = AdbTapper(adb, serial, verify=emulator.verify)
            except SystemExit as exc:                  # emulator.verify refused the device
                print(f"refusing to tap: {exc}")
                return 2
            except (TapError, OSError) as exc:          # OSError: adb itself would not start
                print(f"refusing to tap: {_short(exc)}")
                return 2
        try:
            source = CAP.open_source(args.source, instance=instance, adb=adb, serial=serial)
        except Exception as exc:
            print(f"cannot open the frame source {args.source!r}: {_short(exc)}")
            return 2
        lines = [f"vision_play  session {recorder.session_dir}",
                 f"  model      {ckpt} (schema {schema}, live_compat {live_compat}, units {units})",
                 f"  deck       {', '.join(args.deck)} (level {args.level})",
                 "  detector   " + (f"{detector.weights} sha256 {str(detector.sha256)[:12]} on "
                                     f"{detector.device_desc}" if detector is not None
                                     else f"NONE ({det_error})"),
                 f"  hud        clock templates {'ok' if hud_reader.clock_reader.ready else 'MISSING'}"
                 f" ({store.root})",
                 f"  geometry   {geometry.name}; taps in device pixels {DEVICE_SIZE[0]}x{DEVICE_SIZE[1]}",
                 f"  source     {args.source}; device {serial} ({instance})",
                 "  taps       " + ("ON (verified Clash Royale instance)" if tapper is not None else
                                     f"off ({refusal or 'dry run: no --tap'})")]
        if plan.v2:
            lines.append(f"  v2 rows    opponent elixir counted from the plays seen; our taps in flight "
                         f"until they land (--obs-delay {plan.obs_delay} ticks); spells in flight")
        if unknown:
            lines.append(f"  unknown    ON: an enemy unit no card explains is an UNKNOWN play (token 1, "
                         f"{deriver.config.unknown_cost} elixir, count not exact); "
                         "unplaceable enemy labels stay on the board")
        if detector is None:
            lines.append("  no detector: the board stays empty for the model and taps are refused")
        if not hud_reader.clock_reader.ready:
            lines.append("  no clock templates: the clock cannot be read, so the bot never decides "
                         "(python -m scripts.vision.templates digits --font clock ...)")
        _say(lines)
        recorder.log("config", ckpt=str(ckpt), schema=schema, live_compat=live_compat, units=units,
                     deck=list(args.deck), deck_ids=deck_ids, level=args.level,
                     detector=None if detector is None else detector.info(), detector_error=det_error,
                     geometry=geometry.name, source=args.source, serial=serial, taps=tapper is not None,
                     tap_refusal=refusal, start_mode=args.start_mode, decide=dcfg.name,
                     engine=eng.to_json(), vocab_drift=list(loaded.drift),
                     clock_templates=bool(hud_reader.clock_reader.ready),
                     **({"rows": plan.to_json()} if plan.v2 else {}),
                     **({"unknown_plays": {"cost": deriver.config.unknown_cost}} if unknown else {}))
        if detector is None:
            recorder.log("no_detector", error=det_error)
        for name, state in ctl.start(**CT.start_kwargs(args)).items():
            print(f"  {name:10s} {state}", flush=True)
        print(f"mode {ctl.mode.upper()}; switch with {args.hotkey}, the window or {ctl.control_file}. "
              f"Stop: Ctrl+C or create {STOP_FILE}.", flush=True)
        runner = VisionPlay(source=source, perception=perception, decider=decider, control=ctl,
                            recorder=recorder, geometry=geometry, tapper=tapper, card_names=names,
                            seconds=args.seconds, battles=args.battles,
                            out=lambda s: print(s, flush=True),
                            tap_guard=None if tapper is None else ScreenGuard(source),
                            unknown_plays=unknown)
        return runner.run()
    finally:
        if ctl is not None:
            ctl.stop()
        if tapper is not None:
            tapper.close()
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        recorder.close()


if __name__ == "__main__":
    sys.exit(main())
