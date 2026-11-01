r"""Friendly-only live runner: screen vision -> token model -> taps (V10, with V12's takeover).

    .venvs\buildabot\Scripts\python.exe scripts\vision_play.py                  dry run (the default)
    .venvs\buildabot\Scripts\python.exe scripts\vision_play.py --tap --start-mode auto
                                                                        taps, only while you watch
    .venvs\buildabot\Scripts\python.exe scripts\vision_play.py --source <recorded session> --no-window
                                                                        the whole loop offline (rehearsal)

FRIENDLY BATTLES ONLY. A person starts every battle (clan chat or a friend's profile). This
runner never navigates: it does not press Battle, OK, or anything outside the arena, and it has
no option that would (hastycr.machine.FRIENDLY_ONLY). It taps only while a battle is on screen:
the battle clock is read and running, the bottom HUD panel is there and the end screen is not.
It never reads the game's memory (scripts/gamestate is not imported) and it never starts or
restarts the emulator.

DRY RUN BY DEFAULT. Decisions are made, shown and logged, nothing is tapped, and with --source
auto (the default) or printwindow no adb is opened (auto is PrintWindow alone then; its adb
fallback comes only with --tap, after emulator.verify; an explicit --source adb runs
emulator.verify before its first screencap). --tap enables taps; use it only while you watch. Before the first tap scripts/emulator.verify() must pass (the serial
is the Clash Royale instance) and `adb shell wm size` must report 1080x1920. A file named STOP in
the repo root ends the runner in any mode (it refuses to start while STOP exists). One runner at a
time: reports/vision_play/vision_play.lock is held by the running process (the OS releases it if
the process dies).

TAKEOVER (scripts/vision/control.py). The runner starts in MANUAL unless --start-mode auto: the
bot keeps perceiving and deciding, and shows its suggestions (status line, control window) but does
not tap. F8, the control window's button or a MANUAL file in the repo root switch to AUTO and
back; a switch drops a tap decided before it. Own plays the bot did not tap are logged as the
human's, next to what the bot would have played: the runner decides on nearly every tick, so
almost every decision is a wait; while it does not tap, a wait is followed (at most every
--would-every ticks) by a second look that plays the gate's best card at its best tile
(Decider.would, never tapped), and each human play is compared with the newest of those.

REFUSED TAPS (scripts/vision/tap_refusal.py, FIX E). A tap on a tile where the card may not go
does nothing in the game. Each tap sent is watched from the HUD: the card leaving its slot (or
the elixir falling by its cost, or the deriver's landing) is a landing; the card still in its
slot 30 ticks later is a refusal (tap_refused, a console line). The refused card is then kept
off that spot through the legality (RefusalLegality: the enemy half of that lane for the
battle, else 2 tiles around it for 30 s), and after 3 final refusals in a battle the runner
switches AUTO to MANUAL (tap_guard_manual, mode source "refused taps"); F8 back to AUTO
rearms it, also when the owner was in MANUAL already as the 3rd one came. The guard never
taps and never sets AUTO. In a dry run it only logs: the AUTO(dry) plays are watched as if
tapped, the legality and the decisions stay exactly as without it.

OUR ELIXIR AFTER OUR TAPS (tools/rs_decide.ElixirCap, one per battle in BattleContext). The
card leaves its slot 2-4 ticks after a tap, but the HUD's elixir reading can lag the payment
and reads high while the counter animates (first --tap friendly, 2026-11-01: 3.00 for 5 ticks
after a Cannon paid from 4.26; the Log tapped on it was refused). Once a tap of ours is paid
(its card left the slot), the decision spends at most min(HUD, cap), cap = the decision's
spendable at the tap (with the costs it held for the taps then in flight added back, each
taken off once that tap is paid) - the cost + the most regen possible since (1x / 2x / 3x by
the tick), until the HUD reads at most 0.05 elixir above the cap on two frames in a row or 30
ticks pass; taps in a row stack; the card read back in its slot within 20 ticks of leaving it
was a misread (the tap is unpaid again). The frame and decision records carry "elixir_cap"
when it bites (the cap below the HUD reading), battle_end its counts. Only real taps open a
cap: a dry run decides exactly as before.

OUR TAPS IN FLIGHT, V2 ROWS (tools/public_state.LiveObserver, OwnTaps; the public_state module
doc, V2 ROWS). The decision-time rows list a play in own_pending from its tap until it runs
(24 +- 3 ticks), with the card in hand, the elixir unpaid and no unit; live the play runs a
median 25 ticks after the decided frame, but the client shows a placement preview from the tap
+ 3 that the detector boxes as our unit, and the HUD pays the bar from +3..10. Each battle frame
the observer reads the HUD hand and next card and the tracks (watch): a tap stays in
own_pending until it is seen to run (its next-card change, its slot refilled, its real unit
dropping in, each 20+ ticks after the tap), tap_refused, or the --obs-delay + 3 ticks; the row
(make_state_fn, row_state) leaves the preview's tracks out until the play runs and dates their
deploy timer from the run, and shows its own elixir as min(HUD, cap) + the costs the HUD has
paid for taps not run yet (frame and decision records: "row_elixir"; battle_end: "own_flight").
The decision itself still spends the HUD's elixir (and the cap). No tap: the rows are as before.

THE TAP GATE (scripts/vision/tap_gate.py and mode.py, skin audit P0-4 and P1-12). Every battle starts
with no AUTO tap allowed until its gate opens: all four princess labels calibrated (tower skins),
with a detector, a tower body boxed at each of the six towers (its box centred there, so a skin
that moves the box's top or bottom still counts) on half the detector frames from the battle's
start until the labels and the mode are checked (at least 1.5 s and 10 frames), and classic 1v1
(the first clock reading 2:5x-3:00, our elixir filling at one per 2.8 s).
An AUTO tap decided before that is held ("skipped", why "not tapped: tap gate: ..."). When a check
fails the gate closes for the battle: one console line ("towers not calibrated (enemy_left, ...):
tapping off this battle; play yourself ...") and AUTO switches to MANUAL (mode source "towers",
or "game mode"); the owner's own switch back to AUTO (F8) after that overrides it for the rest of
that battle (tap_gate_override). When it closed while MANUAL already, the owner's next F8 (the
routine hand-over) is switched back to MANUAL with the line again (tap_gate_handover) and only
the F8 after it overrides. The control window shows the gate after the mode. A dry run only
prints "dry run: the tower gate would keep tapping off this battle (...)". Records: game_mode
("mode classic (...)" is also printed), tap_gate, tap_gate_handover, tap_gate_override, and the
battle_end record's tap_gate.

REPLAY (--source <a session folder written by VisionRecorder>, e.g. capture --record's or a
vision_play session's): the whole loop on the recording's images, on this process's clock, in
real time or with --replay-fast as fast as it runs; taps are refused, no emulator or adb is
touched, and the session goes to reports/vision_play/replays/replay_<time> (outside
latency_probe's --last and the live sessions' image budget).

OPPONENT-PLAY GUARDS (on by default; --legacy-opp-plays turns them off). The event deriver runs
with scripts/vision/events.LIVE_GUARDS (EventConfig.live(); the events module doc, GUARDS): an
enemy unit is a play only once it was seen for 10 ticks, over the short tracks of the same unit
(named by the label it has then, reported about 0.5 s later, still dated at its first sighting),
a lost enemy unit may come back under another label while both keep moving the same way, and a
unit first seen past the river (no princess tower of the other side down in its lane: the HUD's
tower reads) where a unit of a reported play could have walked is not a play. On the owner's
2026-10-29 friendly: 103 opponent plays of 27 cards -> 46 of 18 (375 elixir -> 155).

UNKNOWN PLAYS (--unknown-plays, off by default; docs/NEW_CARDS_PLAN.md 2.2). An enemy unit no
card of the event tables explains (a card the detector names and the tables lack, a hero's extra
unit, a misread) is reported as an UNKNOWN play, card -1: the model reads it as row token 1 in
the opponent's memory (opp_seen / opp_last, legacy and v2 rows), a v2 row's opponent elixir is
charged events.UNKNOWN_PLAY_COST for it (--unknown-cost sets another) and is no longer exact,
and it is not added to the opponent's known cards. Without the flag such a unit is only counted
(OppStats.unknown), as before. Either way the unit stays on the board, named by its class's card
in the class table: the board's card -1 rows (PublicStateBuilder unknown_units, which the flag also turns on) are for
labels the class table cannot place, and the live chain never tracks those (the tracker gets the
detector's units and spells only). Each battle's end record then counts the UNKNOWN plays
(opp_unknown).

THE LOOP, per new frame (the newest one; frames that arrived meanwhile are skipped):

  FrameSource (scripts/vision/capture.py open_source, "auto": PrintWindow at the window size,
      747x1328 on the owner's PC, with an adb fallback)
  -> screen cues (pixels, resolution-free): the HUD panel's top edge (blue band), the end screen
     (red and blue banners and the OK button), the end screen's crowns
  -> HudReader (clock -> tick, elixir, hand, towers; templates captured on the PC)
  -> Detector (scripts/vision/detector.py; weights from --detector or the pin file) -> HP bars
     -> Tracker (own_classes = the deck's labels, vision_classes.ClassTable.deck_labels;
        spell_classes = ClassTable.spell_labels: a spell's boxes and a unit's never share a track)
     -> EventDeriver (opponent plays, own landings; EventTables from data/rs_rows/spawn_primary.json)
  -> BattleState: a battle starts once the clock runs for a second with the panel on screen, and
     ends on the end screen, a stopped clock (10 s), a clock reset or 8 s off the battle screen
  -> PublicMemory (an opponent play recorded at the tick perception first saw it; view(now, 0))
  -> tools/public_state (the state obs_row reads; the row's hand shows a played slot's card in
     flight and a pad for an unread slot, never hand token 1, which no training row held, while
     decide still masks both) -> tools/rs_rows.obs_row with the checkpoint's
     row schema, picked by its meta "schema" (row_plan): rs_rows/legacy (--live-compat rows for
     live_big_hog) exactly as before; rs_rows/v2 (E14) also gets a public_state.LiveObserver
     per battle, fed every frame's opponent plays, HUD hand / next card and tracks and each
     tap sent, whose inputs (the opponent's elixir counted from the plays seen, our taps still
     in flight until they run, at the checkpoint's --obs-delay) ride in the state to obs_row,
     and the spell tracks as spells in flight (OUR TAPS IN FLIGHT above)
  -> tools/rs_decide.decide with DecideConfig.live() (pending taps, fallback, cap rule, lane guard;
     a v2 model's ability buttons stay masked: our deck has none and the HUD reads none), or
     with --decide legacy the rule rs_eval measured (dry runs only). Live, legacy is not quite
     that rule while a card is in flight: under rs_eval --delay the card stays in its slot and
     an argmax on it finds no legal tile (a wait); live, its HUD slot is empty from the drag to
     the refill (about 1.5 s) and masked, so legacy plays the next-best card instead
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
import dataclasses  # noqa: E402
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
from scripts.vision import overlay as OV  # noqa: E402
from scripts.vision import tap_gate as TG  # noqa: E402
from scripts.vision import tap_refusal as TR  # noqa: E402
from scripts.vision import tower_calib as TC  # noqa: E402
from scripts.vision import tower_labels as TL  # noqa: E402
from scripts.vision.geometry import check_frame_size  # noqa: E402
from scripts.vision.tracker import observations_from_detections  # noqa: E402
from scripts.vision.types import (  # noqa: E402
    EMPTY_CARD, OWN_TEAM, PHASE_ENDED, PHASE_OVERTIME, SCREEN_BATTLE, SCREEN_END, SCREEN_MENU,
    Events, Hud, VisionFrame, now)

STOP_FILE = ROOT / "STOP"
REPORT_ROOT = ROOT / "reports" / "vision_play"
#: Replays of recordings (--source <session>): their own root, outside latency_probe's --last
#: and the live sessions' image budget (both read REPORT_ROOT's own session folders only).
REPLAY_ROOT = REPORT_ROOT / "replays"
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
#: The frame size the HUD templates and regions were measured at (the owner's PrintWindow).
CALIBRATED_FRAME = (747, 1328)

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
    """Colour fractions of the cue boxes of one frame. overlay: the share of the enemy half's
    floor patches tinted by the deploy overlay (overlay.mirror_tint), measured only when the
    three end-screen cues pass (None: not measured): the overlay passes the red-banner cue
    (0.65-1.0 in battle, skin audit P1-2), so such a frame is no end screen."""
    panel: float
    red_banner: float
    blue_banner: float
    ok_button: float
    overlay: float | None = None

    @property
    def end_screen(self) -> bool:
        return (min(self.red_banner, self.blue_banner, self.ok_button) >= END_MIN
                and (self.overlay is None or self.overlay < OV.OVERLAY_SHARE))

    @property
    def hud_present(self) -> bool:
        return self.panel >= PANEL_MIN and not self.end_screen

    def to_dict(self) -> dict[str, float]:
        return {"panel": round(self.panel, 3), "red_banner": round(self.red_banner, 3),
                "blue_banner": round(self.blue_banner, 3), "ok_button": round(self.ok_button, 3),
                **({} if self.overlay is None else {"overlay": round(self.overlay, 3)})}


def read_cues(rgb: np.ndarray) -> ScreenCues:
    """The screen cues of an HxWx3 RGB frame of any 9:16 size."""
    red = float(red_mask(_box(rgb, END_RED_BANNER)).mean())
    blue = float(blue_mask(_box(rgb, END_BLUE_BANNER)).mean())
    ok = float(blue_mask(_box(rgb, END_OK_BUTTON)).mean())
    overlay = OV.mirror_tint(rgb) if min(red, blue, ok) >= END_MIN else None
    return ScreenCues(float(blue_mask(_box(rgb, PANEL_BAND)).mean()), red, blue, ok, overlay)


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
    is unread neither count nor interrupt. first_tick: the tick of the first of those frames
    (the battle's first clock reading when the runner saw the battle begin).
    End: the end screen for `end_frames` frames; the clock read but not advancing for
    `frozen_s` (10 s: a game that froze for a few seconds, a lag spike with the HUD still up,
    is a pause, and running() already stops the decisions after `running_s`; the recorded
    friendly ended on its end screen 6 s after the last clock frame); a clock that jumps back
    more than `reset_ticks` (a new battle the end screen of which was missed); overtime 0:00;
    or `lost_s` without a battle-screen frame. An unread clock alone ends nothing (the bot only
    stops deciding). running(t): the clock advanced within `running_s` of t (decisions need it).
    """

    def __init__(self, *, start_frames: int = 3, start_advance_ticks: int = 20, end_frames: int = 3,
                 frozen_s: float = 10.0, lost_s: float = 8.0, running_s: float = 1.2,
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
        self.first_tick: int | None = None
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
            self.first_tick = int(self._cand[0][1])
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
    elixir_cap: Any = None            # rs_decide.ElixirCap (Decider.elixir_cap_for), None: off


class ResultBook:
    """The result of each battle: the HUD's crowns at the end, then the end screen's, which
    win when read (the HUD's tower reading can be wrong: a tower skin read both enemy towers
    destroyed on 2026-10-31, the end screen showed 1-0). The end screen's 0-0 before the clock
    ran out is the exception: no battle ends 0-0 then, and the loss layout of the end screen
    is unverified, so the HUD's crowns stand."""

    SETTLE_S = 3.0       # end-screen frames read this long (the crowns animate in)
    WAIT_S = 20.0        # no end screen this long after the end: the HUD's crowns stand
    #: An end screen's 0-0 counts only from this tick (overtime ends at 6000).
    ZERO_ZERO_FROM_TICK = 5900

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
        """Close the open result: the end screen's crowns when it was read ("end screen"; a
        HUD reading that differs goes in crowns_disagree, unless the battle's end explains it:
        crowns_explained, tower_calib.end_crowns_explained with open()'s could_fall: the tower
        whose fall ended the battle is never counted by the HUD), else the HUD's ("hud"); an end
        screen read 0-0 before tick ZERO_ZERO_FROM_TICK while the HUD has crowns keeps the HUD's
        ("hud (end screen read 0-0)")."""
        rec, self.open_rec = self.open_rec, None
        if rec is None:
            return None
        hud, end = [int(c) for c in rec["hud_crowns"]], rec["end_crowns"]
        last = rec.get("last_tick")
        if end is None:
            crowns, source = hud, "hud"
        elif (end == [0, 0] and hud != [0, 0]
              and (last is None or int(last) < self.ZERO_ZERO_FROM_TICK)):
            crowns, source = hud, "hud (end screen read 0-0)"
        else:
            crowns, source = [int(end[0]), int(end[1])], "end screen"
        if end is not None and hud != end:
            ok, why = TC.end_crowns_explained(hud, end, rec.get("could_fall") or {})
            if ok:
                rec["crowns_explained"] = why
            else:
                rec["crowns_disagree"] = {"hud": hud, "end": [int(end[0]), int(end[1])]}
                rec["crowns_why"] = why
        rec["crowns"] = crowns
        rec["source"] = source
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
#: A battle the runner sees start later than this tick was joined in the middle (the runner
#: started, or BattleState restarted, mid-battle): what is on the board then is not a play.
CLEAN_START_TICK = 300
#: ... tracks first seen within this many ticks of such a start are primed (events.py).
PRIME_TICKS = 10
#: BattleState end reasons after which the game is over: the next battle is a new one.
FINAL_END_REASONS = frozenset({"end screen", "overtime over"})


def keeps_towers(reason: str | None) -> bool:
    """Whether a battle that ended for `reason` keeps its tower readings until the next start
    (LivePerception.end_battle): any end but the end screen or overtime's 0:00 (a misread
    clock jumping back, a stopped clock, 8 s off the battle screen) may be the same battle
    coming back, where a destroyed tower must stay destroyed."""
    return reason not in FINAL_END_REASONS


#: obs_row's hand token is card + 2: a slot written as -2 is token 0 (pad, left out of the
#: model's attention), never token 1 ("unknown card"), which no training hand ever held.
ROW_PAD_CARD = -2


class LivePerception:
    """Frame -> VisionFrame: screen cues, HUD, detector, HP bars, tracker and events.

    hud_reader: scripts/vision/hud.HudReader (or anything with read(rgb, t) -> Hud and
    reset()). detector: scripts/vision/detector.Detector, or None (no units: the HUD is still
    read). tracker / deriver: scripts/vision/tracker.Tracker and events.EventDeriver. The HUD
    is read only on battle-screen frames, and the detector only runs when there is a tick to
    track on: the HUD clock's, or, while the clock is briefly unread, one extrapolated from
    the last reading (at most `max_extrapolate_s`).

    The HUD's filters are reset at the end of a battle, and outside a battle when the battle
    screen comes back after `hud_reset_gap_s` without it; never in a battle (new_battle ..
    end_battle): a destroyed tower reads as rubble, no digits and no bar, which a fresh tower
    filter takes for occlusion of a full tower, so a reset there would bring the tower back
    for the rest of the battle (the clock and elixir filters bridge a gap by themselves).
    end_battle(keep_towers=True) (a battle that ended without its end screen or overtime's
    0:00, keeps_towers: a misread clock jumping back, a stopped clock, 8 s off the battle
    screen) keeps the tower tracker through those resets until the next battle starts: a
    battle joined in the middle is the same battle coming back and keeps it; a clean start is
    a new battle and starts the towers again (a HUD reader without reset_towers is reset
    whole, as before).
    new_battle resets the tracker and the deriver; a battle joined in the middle (its first
    tick above CLEAN_START_TICK) primes what is on the board (the deriver's prime_until_tick).
    A failing board chain (the detector: DetectorError, CUDA out of memory; or the HP bars,
    the tracker or the events on its output) costs the frame its units and events only: the
    HUD, the tick and the coasting tracks are still returned, `detect_errors` counts the
    failures and `last_detect_error` keeps the last one.

    The towers are read after the detector (a HUD reader with read_towers, HudReader's two
    phases): read(rgb, t, towers=False), the tick, detect(), then always read_towers(rgb, t,
    hud, evidence) with the frame's tower boxes (tower_evidence; None when the detector did not
    run or failed): a tower skin's label is read where the detector puts it, and a tower the
    detector still sees is never read destroyed. Towers the HUD's overtime check revived
    (report.towers_revived) are taken out of the deriver's fallen towers. A reader without
    read_towers is read in one call, as before.
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
        self.detect_errors = 0
        self.last_detect_error: BaseException | None = None
        self.in_battle = False
        self.primed_until: int | None = None
        self.towers_kept = False

    def new_battle(self) -> None:
        self.in_battle = True
        tick = None if self._anchor is None else int(self._anchor[0])
        self.primed_until = tick + PRIME_TICKS if tick is not None and tick > CLEAN_START_TICK else None
        if self.towers_kept:
            if self.primed_until is None:              # a clean start: a new battle, new towers
                self.hud_reader.reset_towers()
            self.towers_kept = False
        if self.tracker is not None:
            self.tracker.reset()
        if self.deriver is not None:
            if self.primed_until is None:
                self.deriver.reset()
            else:
                self.deriver.reset(prime_until_tick=self.primed_until)

    def end_battle(self, *, keep_towers: bool = False) -> None:
        self.in_battle = False
        self.primed_until = None
        self.towers_kept = bool(keep_towers) and callable(getattr(self.hud_reader, "reset_towers", None))
        self._reset_hud()
        if self.tracker is not None:
            self.tracker.reset()
        if self.deriver is not None:
            self.deriver.reset()
        self._anchor = None

    def _reset_hud(self) -> None:
        if self.towers_kept:
            self.hud_reader.reset(keep_towers=True)
        else:
            self.hud_reader.reset()

    def row_hand(self, hand: Sequence[int]) -> list[int]:
        """The hand as the model's row shows it (the state decide() reads keeps -1, so the slot
        stays masked): a slot waiting for its refill after a play shows the card that left it,
        as RoyaleSim's state does while a play is in flight (the rows' and the evaluation's
        view); an unread or unknown slot is ROW_PAD_CARD (token 0), never token 1."""
        left = getattr(getattr(self.hud_reader, "hand", None), "left", None) or {}
        out = []
        for slot, c in enumerate(list(hand)[:4]):
            c = int(c)
            if c >= 0:
                out.append(c)
            elif slot in left and int(left[slot]) >= 0:
                out.append(int(left[slot]))
            else:
                out.append(ROW_PAD_CARD)
        return out

    def note_own_tap(self, card: int, tick: int, *, slot: int, aim_mt: Sequence[float] | None,
                     t: float | None) -> None:
        """A tap pair we sent at `t`: the deriver's own play, and the HUD's overlay window (the
        card selected: no tower read until the tile tap is done, overlay.OverlayDetector)."""
        note_tap = getattr(self.hud_reader, "note_tap", None)
        if t is not None and callable(note_tap):
            note_tap(float(t))
        aim = (None, None) if aim_mt is None else (float(aim_mt[0]), float(aim_mt[1]))
        self.deriver.note_own_tap(int(card), int(tick), slot=int(slot), aim_x_mt=aim[0],
                                  aim_y_mt=aim[1], t=t, tracker=self.tracker)

    def track_cards(self, tracks: Iterable[Any]) -> dict[int, int]:
        return self.deriver.track_cards(tracks)

    def tap_info(self, card: int) -> dict[str, Any]:
        """What the v2 observer needs to know of a tap's card (public_state.OwnTaps.sent): the
        track labels a play of it shows, whether it is a spell, and the units one play deploys
        (the deriver's EventTables); {} without them."""
        tables = getattr(self.deriver, "tables", None)
        if tables is None:
            return {}
        card = int(card)
        units = tables.units_per_play(card)
        return {"labels": tuple(sorted(tables.classes_of(card))),
                "spell": tables.kind_of(card) == EV.KIND_SPELL, "units": int(units or 1)}

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
        if (not self.in_battle and self._last_panel_t is not None
                and t - self._last_panel_t > self.hud_reset_gap_s):
            self._reset_hud()                        # back on a battle screen after a while
        self._last_panel_t = t
        two_phase = callable(getattr(self.hud_reader, "read_towers", None))
        vf.hud = self.hud_reader.read(rgb, t, towers=False) if two_phase else self.hud_reader.read(rgb, t)
        vf.tick = vf.hud.clock_tick
        tick = self._track_tick(vf.tick, t)
        res, det_exc = None, None
        if tick is not None and self.detector is not None:
            try:
                res = self.detector.detect(rgb)
            except Exception as exc:                 # re-raised in the board chain below
                det_exc = exc
        if two_phase:
            self.hud_reader.read_towers(rgb, t, vf.hud, self._tower_evidence(res))
            self._revive_towers()
        if tick is None or self.detector is None:
            return vf
        # The whole board chain is guarded, not only detect(): HP bars, ground points, the
        # tracker and the events have never run on the pinned detector's real output, and the
        # detector runs on exactly the frames that start a battle (a tick to track on). An
        # exception here costs the frame its units and events; its HUD and tick still reach
        # BattleState and the decision.
        try:
            if det_exc is not None:
                raise det_exc
            units, spells, bars = res.units(), res.spells(), res.bars()
            reads = hpbars.read_unit_hp(rgb, [d.box for d in units], [b.box for b in bars])
            obs = observations_from_detections(list(units) + list(spells), self.geometry, res.frame_size)
            hpbars.annotate_observations(obs[:len(units)], reads)
            vf.tracks = self.tracker.update(obs, tick)
            vf.detections = list(res.detections)
            vf.events = self.deriver.update(vf.tracks, tick, t=t, tracker=self.tracker,
                                            towers=getattr(vf.hud, "towers", None),
                                            **confirmed_towers(vf.hud))
        except Exception as exc:                     # no units this frame; the HUD still counts
            self.detect_errors += 1
            self.last_detect_error = exc
            vf.events = Events()
            try:
                vf.tracks = [] if self.tracker is None else list(self.tracker.tracks())
            except Exception:
                vf.tracks = []
            return vf
        self.detect_frames += 1
        return vf

    def _tower_evidence(self, res: Any) -> Any:
        """tower_labels.TowerEvidence of a DetectorResult (None for none, or boxes it cannot read)."""
        if res is None:
            return None
        try:
            dets, size = list(getattr(res, "detections", None) or ()), getattr(res, "frame_size", None)
            make = getattr(self.hud_reader, "tower_evidence", None)
            return make(dets, size) if callable(make) else TL.TowerEvidence.from_detections(dets, size)
        except Exception:
            return None

    def _revive_towers(self) -> None:
        revived = getattr(getattr(self.hud_reader, "report", None), "towers_revived", None)
        if revived and self.deriver is not None and callable(getattr(self.deriver, "revive_towers", None)):
            self.deriver.revive_towers(revived)


def confirmed_towers(hud: Any) -> dict[str, Any]:
    """The deriver's destroyed= for a HUD that reports its confirmed towers (Hud.towers_destroyed:
    only those open a pocket), else {} (every 0 counts, as before)."""
    destroyed = getattr(hud, "towers_destroyed", None)
    return {} if destroyed is None else {"destroyed": tuple(destroyed)}


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------
#: The would-play decision's cap: below any elixir, so it never draws a wait while a card is
#: playable (rs_decide's cap rule), and plays the gate's best playable card at its best tile.
WOULD_CAP_MILLI = -(10 ** 9)


class Decider:
    """VisionFrame -> rs_decide.Decision through the public state and the model's row.

    state_fn(vf, ctx) -> the state dict (tools/public_state); row_fn(state, mem_view, tick)
    -> the obs_row (tools/rs_rows); policy / legality / config / rng as rs_decide.decide
    takes them. Our pending taps are pruned against the HUD's hand first (rs_decide), then
    ctx.elixir_cap (rs_decide.ElixirCap, from elixir_cap() per battle: elixir_cap_for; None
    turns it off) gives the frame's cap on our elixir after our own taps: decide spends at
    most min(HUD, cap); no cap while no tap of ours is being paid.
    observer(ctx) -> a new battle's public_state.LiveObserver (v2 rows; make_observer), None
    for legacy rows: the runner keeps it in ctx.observer (observer_for).

    would(): what the bot would play if it played now. The live runner decides on nearly every
    tick, and its wait is drawn with p_wait ** (dt / 20), so almost every decision is a wait
    and a human play would always be compared with one. would() decides the LAST decision's
    row and state again with the cap rule at any elixir (WOULD_CAP_MILLI: no wait drawn while
    a card is playable), on its own rng (the decisions' draws stay as they were): the gate's
    best playable card at its best legal tile, or None (nothing playable or no legal tile).
    It is never tapped."""

    def __init__(self, *, policy: Any, legality: Any, config: Any,
                 state_fn: Callable[[VisionFrame, BattleContext], dict],
                 row_fn: Callable[[dict, dict, int], Mapping[str, Any]], rng: Any = None,
                 team: int = OWN_TEAM, sub: int = RD.SUBTILE_PER_MILLITILE,
                 observer: Callable[[BattleContext], Any] | None = None,
                 elixir_cap: Callable[[], Any] | None = RD.ElixirCap):
        self.policy = RD.as_policy(policy)
        self.legality = legality
        self.config = config
        self.state_fn = state_fn
        self.row_fn = row_fn
        self.rng = rng if rng is not None else np.random.default_rng()
        self.team = int(team)
        self.sub = int(sub)
        self.observer = observer
        self.elixir_cap = elixir_cap
        try:
            base = config if config is not None else RD.DecideConfig()
            self.would_config = base.replace(cap_milli=WOULD_CAP_MILLI)
        except Exception:                            # a config that cannot be copied: no would()
            self.would_config = None
        self.would_rng = np.random.default_rng(0)
        self._last: tuple[Mapping[str, Any], dict, int, list, int | None] | None = None

    def observer_for(self, ctx: BattleContext) -> Any:
        """The battle's LiveObserver (v2 rows), None for legacy rows."""
        return None if self.observer is None else self.observer(ctx)

    def elixir_cap_for(self, ctx: BattleContext) -> Any:
        """The battle's rs_decide.ElixirCap, None when the cap is off (elixir_cap=None)."""
        return None if self.elixir_cap is None else self.elixir_cap()

    def decide(self, vf: VisionFrame, ctx: BattleContext) -> tuple[Any, dict]:
        now_tick = int(vf.tick)
        hand = list(vf.hud.hand)
        ctx.pending = RD.prune_pending(ctx.pending, hand, now_tick)
        ledger = ctx.elixir_cap
        cap = None if ledger is None else ledger.observe(now_tick, hand, vf.hud.elixir_milli)
        state = self.state_fn(vf, ctx)
        row = self.row_fn(state, ctx.memory.view(now_tick, 0), now_tick)
        d = RD.decide(self.policy, row, state, self.team, self.legality, self.config, rng=self.rng,
                      pending=ctx.pending, tick=now_tick,
                      dt=RD.cadence_dt(now_tick, ctx.last_decision_tick), sub=self.sub,
                      **({} if cap is None else {"elixir_cap": cap}))
        ctx.last_decision_tick = now_tick
        ctx.decisions += 1
        self._last = (row, state, now_tick, list(ctx.pending), cap)
        return d, state

    def would(self) -> Any:
        """The would-play Decision for the last decision's row and state (class doc), None."""
        if self._last is None or self.would_config is None:
            return None
        row, state, tick, pending, cap = self._last
        d = RD.decide(self.policy, row, state, self.team, self.legality, self.would_config,
                      rng=self.would_rng, pending=pending, tick=tick, dt=1, sub=self.sub,
                      **({} if cap is None else {"elixir_cap": cap}))
        return d if d.is_play else None


class RefusalLegality:
    """Legality with the refused-tap guard's masks (scripts/vision/tap_refusal.py) on top.

    bind(state, team): the inner legality bound to the state (rs_decide._bind), returned as it
    is while the guard is not active (no mask, no real watch open, or not enforced: a dry run
    or a replay). Decisions and their `legality` names then stay byte-identical. Otherwise a
    Bound named "<inner>+refusals" whose check turns the engine sub-tiles into own-frame model
    tiles (rs_decide.own) and refuses a masked tile without asking the inner check. Decider.would
    goes through the same legality, so would-plays respect the masks too."""

    def __init__(self, inner: Any, guard: Any, *, sub: int = RD.SUBTILE_PER_MILLITILE):
        self.inner = inner
        self.guard = guard
        self.sub = int(sub)

    @property
    def name(self) -> str:
        return f"{getattr(self.inner, 'name', 'callable')}+refusals"

    def bind(self, state: Mapping[str, Any], team: int) -> Any:
        bound = RD._bind(self.inner, state, team)
        guard = self.guard
        raw = state.get("tick") if isinstance(state, Mapping) else None
        tick = None if raw is None else int(raw)
        if guard is None or not guard.active(tick):
            return bound
        team, sub = int(team), self.sub

        def check(slot: int, card: int, x_sub: int, y_sub: int) -> bool:
            ox, oy = RD.own(x_sub, y_sub, team, sub)
            if guard.masked(card, int(ox // 1000), int(oy // 1000), tick):
                return False
            return bound(slot, card, x_sub, y_sub)

        return RD.Bound(f"{bound.name}+refusals", check)


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


#: The own player's key for the hand the ROW shows (LivePerception.row_hand); decide() reads
#: "hand" and never this one.
ROW_HAND_KEY = "row_hand"
#: ... and for the elixir the ROW shows (v2 rows with a tap in flight: LiveObserver.row_elixir);
#: decide() reads "elixir_milli" (the HUD's) and never this one.
ROW_ELIXIR_KEY = "row_elixir_milli"


def make_state_fn(builder: Any, perception: Any, plan: RowPlan
                  ) -> Callable[[VisionFrame, BattleContext], dict]:
    """Decider's state_fn: builder.state (public_state.PublicStateBuilder) of the frame with
    the opponent cards seen, the creating card of each track and the elixir of our taps in
    flight; for v2 rows also the battle's observer (ctx.observer). With a perception that has
    row_hand, the own player also gets ROW_HAND_KEY: the hand the row shows (row_state).
    v2 rows with our taps in flight (public_state module doc, V2 ROWS): the board is the
    observer's (LiveObserver.board: a tap's placement preview left out until the play runs,
    then dated from the run), and the own player gets ROW_ELIXIR_KEY, the row's own bar
    (LiveObserver.row_elixir: min(HUD, ctx.elixir_cap's value) + the costs the HUD has paid
    for plays not run yet), when it differs from the HUD's. Without a tap: as before."""
    row_hand = getattr(perception, "row_hand", None)

    def state_fn(vf: VisionFrame, ctx: BattleContext) -> dict:
        obs = ctx.observer if plan.v2 else None
        kw = {"observer": obs} if plan.v2 else {}
        view = vf
        board = getattr(obs, "board", None)
        if callable(board) and vf.tick is not None:
            tracks = board(vf.tracks, int(vf.tick))
            if tracks is not vf.tracks:
                view = dataclasses.replace(vf, tracks=list(tracks))
        state = builder.state(view, opp_deck=ctx.opp_cards, track_cards=perception.track_cards(vf.tracks),
                              pending_cost=sum(max(0, int(p.cost)) for p in ctx.pending), **kw)
        if callable(row_hand):
            me = state["players"][OWN_TEAM]
            me[ROW_HAND_KEY] = row_hand(me.get("hand") or ())
        row_elixir = getattr(obs, "row_elixir", None)
        if callable(row_elixir) and vf.tick is not None:
            v = row_elixir(vf.hud.elixir_milli, getattr(ctx.elixir_cap, "value", None), int(vf.tick))
            if v is not None:
                state["players"][OWN_TEAM][ROW_ELIXIR_KEY] = int(v)
        return state
    return state_fn


def row_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """The state obs_row reads: `state` itself, or, when the own player has ROW_HAND_KEY or
    ROW_ELIXIR_KEY, a shallow copy whose own hand / elixir_milli are those (the players list
    and the own player are copied; nothing else is). Engine hands are never negative, so the
    rows of training and of the evaluation never held hand token 1; live, an empty or unread
    slot would. The rows' own bar is the engine's, unpaid while a play is in flight; the HUD's
    is paid from the tap."""
    players = state.get("players") if isinstance(state, Mapping) else None
    if not players or len(players) <= OWN_TEAM:
        return state
    me = players[OWN_TEAM]
    if not isinstance(me, Mapping):
        return state
    hand, elixir = me.get(ROW_HAND_KEY), me.get(ROW_ELIXIR_KEY)
    if hand is None and elixir is None:
        return state
    players = list(players)
    mine = dict(me)
    if hand is not None:
        mine["hand"] = [int(c) for c in hand]
    if elixir is not None:
        mine["elixir_milli"] = int(elixir)
    players[OWN_TEAM] = mine
    return {**state, "players": players}


def make_row_fn(obs_row: Callable[..., Mapping[str, Any]], sub: int, plan: RowPlan, *,
                live_compat: bool, units: str,
                row_inputs: Callable[[Mapping[str, Any]], dict] | None = None
                ) -> Callable[[dict, dict, int], Mapping[str, Any]]:
    """Decider's row_fn: obs_row (tools/rs_rows.obs_row) of our side (OWN_TEAM) with the
    checkpoint's unit tokens, on row_state(state) (the row's hand); for v2 rows also
    row_inputs(state) (public_state.row_inputs: schema, opp_elixir and pending from the
    state's observer block)."""
    if not plan.v2:
        def row_fn(state: dict, view: dict, tick: int) -> Mapping[str, Any]:
            return obs_row(row_state(state), OWN_TEAM, sub, view, tick, live=live_compat,
                           units=None if live_compat else units)
        return row_fn
    if row_inputs is None:
        import public_state
        row_inputs = public_state.row_inputs

    def row_fn_v2(state: dict, view: dict, tick: int) -> Mapping[str, Any]:
        return obs_row(row_state(state), OWN_TEAM, sub, view, tick, live=live_compat,
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
    would_every_ticks  while the bot does not tap (MANUAL, a dry run, no tapper), a drawn wait
                is followed, at most once per this many ticks, by Decider.would(): what the
                bot would play now. It goes to ControlState.note_would, where each human play
                is compared with it ("would" in the human_play record), to the frame record
                ("would") and to the status line. 0: never.
    refusals    scripts/vision/tap_refusal.RefusedTapGuard, or None (no guard: as before). It
                watches every tap sent (and, in a dry run, every AUTO(dry) play, virtually),
                is shown each battle-screen frame before the decision, logs tap_landed /
                tap_refused / ... and, when enforced, switches AUTO to MANUAL after k refused
                taps in a battle (tap_guard_manual). Its masks reach the decisions through
                the Decider's legality (RefusalLegality).
    hotkey      the takeover key named in the refused-taps console line (--hotkey)
    tap_gate    scripts/vision/tap_gate.TapGate, or None (no gate: as before). It is shown every
                battle-screen frame of a battle (the HUD reader's label calibration, the
                detector's boxes, the raw elixir reading) and holds AUTO taps until it opens;
                when it closes, AUTO switches to MANUAL (module doc, THE TAP GATE)
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
                 unknown_plays: bool = False, would_every_ticks: int = 10,
                 refusals: Any = None, hotkey: str | None = CT.DEFAULT_HOTKEY, tap_gate: Any = None):
        self.unknown_plays = bool(unknown_plays)
        self.tap_gate = tap_gate                       # TapGate (no AUTO tap before it opens)
        self._gate_gen: int | None = None              # the control generation when the gate closed
        self._gate_took = False                        # the gate itself took AUTO away this battle
        self._det_ran: bool | None = None              # the detector ran on this frame (None: unknown)
        self.refusals = refusals                       # RefusedTapGuard (not tap_guard: ScreenGuard)
        self.hotkey = hotkey
        self._refusal_gen: int | None = None           # the control generation of the guard's switch
        self._refusal_dry_said = False                 # the dry run's "would switch" line, per battle
        self.would_every_ticks = max(0, int(would_every_ticks or 0))
        self._would: Any = None                        # the last would-play Suggestion
        self._would_tick: int | None = None
        self._frame_size_said = False
        self._tower_said: set[str] = set()              # tower_label logged this battle (keys)
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
                      control=dict(getattr(self.control, "counts", {}) or {}),
                      **self._refusal_summary(session=True))
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
        det_errors = getattr(self.perception, "detect_errors", 0)
        det_frames = getattr(self.perception, "detect_frames", None)
        try:
            vf = self.perception.perceive(frame)
        except Exception as exc:                       # no VisionFrame: nothing is decided
            self._error("frame_error", exc, t=float(frame.t))
            return
        self._det_ran = (getattr(self.perception, "detect_frames", None) != det_frames
                         if isinstance(det_frames, int) else None)
        ms = {"stale": round(float(getattr(frame, "stale_s", 0.0)) * 1000.0, 2),
              "perceive": round((self.clock() - t0) * 1000.0, 2)}
        if getattr(self.perception, "detect_errors", 0) != det_errors:
            exc = getattr(self.perception, "last_detect_error", None)
            self._error("detect_error", exc if isinstance(exc, BaseException) else RuntimeError(str(exc)),
                        tick=vf.tick)
        if not self._frame_size_said:
            self._say_frame_size(vf)
        self.last_vf = vf
        out = None
        try:
            out = self._after_perception(vf, frame, ms)
        except Exception as exc:                       # before the tap: no tap this frame
            self._error("decide_error", exc, tick=vf.tick)
        try:
            self._tower_logs(vf)
        except Exception as exc:
            self._error("diagnostics_error", exc)
        self._record(vf, frame, out, ms)
        self._flush_control()

    def _after_perception(self, vf: VisionFrame, frame: Any, ms: dict) -> dict | None:
        events = self.battle.update(vf)
        if "end" in events:
            self._end_battle(vf)
        if "start" in events:
            self._start_battle(vf)
            # This frame's events came from the deriver BEFORE new_battle reset it: the plays
            # in them are reported again by the fresh deriver, so they are not recorded twice.
            vf.events = Events()
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
        amended = getattr(vf.events, "opp_amended", None)
        if amended:                                    # a play's card guessed again (events PHANTOMS)
            EV.amend_plays(ctx.memory, amended)
            plays = getattr(getattr(self.perception, "deriver", None), "opp_plays", None)
            if callable(plays):
                ctx.opp_cards = EV.opp_cards(plays())
        if vf.events.own_landings:
            ctx.own_plays += sum(1 for x in vf.events.own_landings if x.land_tick is not None)
        if ctx.observer is not None:                   # v2 rows: the count and our taps in flight
            ctx.observer.events(vf.events)
            watch = getattr(ctx.observer, "watch", None)
            if callable(watch) and vf.screen == SCREEN_BATTLE and vf.tick is not None:
                watch(int(vf.tick), list(vf.hud.hand), vf.hud.next, vf.tracks)
        if vf.tick is not None or vf.events.own_landings:
            self.control.attribute(vf.events.own_landings, tick=vf.tick)
        if self.refusals is not None and vf.screen == SCREEN_BATTLE and vf.tick is not None:
            try:                                       # before the decision: a switch applies to it
                self._observe_refusals(vf, ctx)
            except Exception as exc:
                self._error("refusal_error", exc, tick=vf.tick)
        if self.tap_gate is not None and vf.screen == SCREEN_BATTLE:
            try:                                       # before the decision: a switch applies to it
                self._observe_gate(vf, ctx)
            except Exception as exc:
                self._error("tap_gate_error", exc, tick=vf.tick)
        if not self.can_decide(vf):
            return None
        return self._decide(vf, frame, ctx, ms)

    # -- the tap gate (scripts/vision/tap_gate.py) ---------------------------------------
    def _gate_enforced(self) -> bool:
        return self.tapper is not None and bool(self.control.taps_enabled)

    def _key(self) -> str:
        return (self.hotkey if self.hotkey and str(self.hotkey).lower() not in ("", "none", "off")
                else "the window button")

    def _observe_gate(self, vf: VisionFrame, ctx: BattleContext) -> None:
        """Shows the gate this frame (labels, boxes, elixir); logs and says what it decided. The
        owner's switch back to AUTO after the gate closed overrides it for the rest of the battle,
        but only once the gate itself has taken AUTO away (so the owner saw its line): a gate that
        closed while MANUAL already turns the owner's next, routine F8 back to MANUAL with the
        line again (tap_gate_handover), and only the F8 after that overrides."""
        g, ctl = self.tap_gate, self.control
        if (g.state == TG.CLOSED and g.overridden is None and self._gate_gen is not None
                and ctl.mode == CT.AUTO and ctl.generation != self._gate_gen):
            sw = getattr(ctl, "last_switch", None) or {}
            src = str(sw.get("source", "?"))
            if self._gate_took:
                g.override(src, vf.tick, vf.t_capture)
                self._log("tap_gate_override", battle=ctx.number, tick=vf.tick, source=src,
                          reasons=list(g.reasons), dry=not self._gate_enforced())
                self.out(f"tower gate overridden ({src}): AUTO taps for the rest of battle {ctx.number}")
            else:
                self._gate_refuse_handover(vf, ctx, src)
        rep = self._hud_report(vf)
        er = getattr(rep, "elixir_read", None)
        detector = getattr(self.perception, "detector", None) is not None
        ran = self._det_ran if self._det_ran is not None else vf.tick is not None
        recs = g.observe(vf.t_capture, vf.tick, calib=TG.calib_status(getattr(self.perception, "hud_reader", None)),
                         detector=detector, detections=list(vf.detections) if detector and ran else None,
                         frame_size=vf.frame_size,
                         elixir=None if er is None else (getattr(er, "digit", None), getattr(er, "bar_frac", None)))
        for kind, rec in recs:
            if kind == "game_mode":
                self._log("game_mode", battle=ctx.number, **rec)
                if rec.get("mode") == "classic":
                    self.out(f"battle {ctx.number}: mode classic ({rec.get('why')})")
            elif kind == "tap_gate":
                self._gate_decided(rec, ctx)
        self._gate_notice()

    def _gate_refuse_handover(self, vf: VisionFrame, ctx: BattleContext, src: str) -> None:
        """The owner's switch to AUTO came after a gate that closed while MANUAL already: back to
        MANUAL (enforced) with the gate's line again; the next switch to AUTO overrides."""
        g, ctl = self.tap_gate, self.control
        enforced = self._gate_enforced()
        switched = bool(ctl.set_mode(CT.MANUAL, source=g.source())) if enforced else False
        self._gate_took, self._gate_gen = True, ctl.generation
        self._log("tap_gate_handover", battle=ctx.number, tick=vf.tick, source=src, reasons=list(g.reasons),
                  enforced=enforced, switched=switched)
        if enforced:
            self.out(f"{g.reason_text()}: tapping off this battle; play yourself (switched back to MANUAL; "
                     f"{self._key()} again hands back to the bot anyway, for this battle only)")
        else:
            self.out(f"dry run: the tower gate would switch back to MANUAL now ({g.reason_text()}); "
                     f"with --tap, {self._key()} a second time would hand back anyway")

    def _gate_notice(self) -> None:
        """The control window's gate line (ControlState.set_notice): OFF or overridden, else none."""
        set_notice = getattr(self.control, "set_notice", None)
        if not callable(set_notice):
            return
        g = self.tap_gate
        word = None if g is None or not self.battle.in_battle else g.status()
        if word == "gate OFF":
            text = f"tap gate OFF this battle ({g.reason_text()})"
        elif word == "gate overridden":
            text = f"tap gate overridden this battle ({(g.overridden or {}).get('source', '?')})"
        else:
            text = None
        set_notice(text)

    def _gate_decided(self, rec: dict, ctx: BattleContext) -> None:
        """tap_gate: log it; on a close, switch AUTO to MANUAL (enforced) and say it once."""
        g, ctl = self.tap_gate, self.control
        enforced = self._gate_enforced()
        mode_before, switched = ctl.mode, False
        if rec.get("state") == TG.CLOSED:
            if enforced and mode_before == CT.AUTO:
                switched = bool(ctl.set_mode(CT.MANUAL, source=g.source()))
            # did the gate take AUTO away (a dry run: would it have)? See _observe_gate.
            self._gate_took = switched or (not enforced and mode_before == CT.AUTO)
            self._gate_gen = ctl.generation
        self._log("tap_gate", battle=ctx.number, enforced=enforced, mode_before=mode_before,
                  switched=switched, **rec)
        if rec.get("state") != TG.CLOSED:
            return
        if not enforced:
            self.out(f"dry run: the tower gate would keep tapping off this battle ({g.reason_text()})")
            return
        if switched:
            self.out(f"{g.reason_text()}: tapping off this battle; play yourself (switched to MANUAL; "
                     f"{self._key()} hands back to the bot anyway, for this battle only)")
        else:
            self.out(f"{g.reason_text()}: tapping off this battle; play yourself ({str(ctl.mode).upper()} "
                     f"already; your next {self._key()} keeps it MANUAL, the one after hands back to the "
                     "bot anyway, for this battle only)")

    def _gate_summary(self) -> dict[str, Any]:
        """{"tap_gate": the gate's summary} for battle_end; {} without one."""
        if self.tap_gate is None:
            return {}
        try:
            return {"tap_gate": self.tap_gate.summary()}
        except Exception as exc:
            self._error("tap_gate_error", exc)
            return {}

    # -- refused taps (scripts/vision/tap_refusal.py) ---------------------------------
    def _observe_refusals(self, vf: VisionFrame, ctx: BattleContext) -> None:
        """Shows the guard this frame's hand, elixir and landings; logs its events; switches
        AUTO to MANUAL on tap_guard_manual when enforced; rearms it once the owner handed back."""
        g = self.refusals
        ctl = self.control
        if (self._refusal_gen is not None and ctl.mode == CT.AUTO
                and ctl.generation != self._refusal_gen):
            g.rearm(int(vf.tick))                      # the owner pressed F8 (or the window) again
            self._log("tap_guard_rearmed", battle=ctx.number, tick=int(vf.tick), gen=ctl.generation)
            self._refusal_gen = None
        events = g.observe(int(vf.tick), float(vf.t_capture), list(vf.hud.hand), vf.hud.elixir_milli,
                           vf.events.own_landings)
        for kind, data in events:
            rec = dict(data)
            if kind == "tap_guard_manual":
                self._refusal_manual(rec, ctx)
                continue
            self._log(kind, battle=ctx.number, name=self.name(rec.get("card")), **rec)
            if kind == "tap_refused" and not rec.get("dry"):
                refused = getattr(ctx.observer, "refused", None)
                if callable(refused):                  # v2 rows: no longer in own_pending
                    refused(int(rec["card"]), int(rec["tap_tick"]), int(rec.get("slot", -1)))
                self.out(self._refused_line(rec))
            elif kind == "tap_refusal_retracted" and not rec.get("dry"):
                retracted = getattr(ctx.observer, "retracted", None)
                if callable(retracted):                # v2 rows: it ran after all
                    retracted(int(rec["card"]), int(rec["tap_tick"]), int(rec.get("slot", -1)),
                              rec.get("tick"))
                self.out(f"tap of {self.name(rec.get('card'))} at tick {rec['tap_tick']} landed after all "
                         f"({rec['evidence']}, {rec['lag']} ticks): refusal withdrawn")

    def _refused_line(self, rec: Mapping[str, Any]) -> str:
        """The console line of a refusal while the bot taps (screen tiles, as the status line)."""
        tile = rec.get("screen_tile") or rec.get("model_tile")
        mask = rec.get("mask")
        if rec.get("toggle_suspect"):
            then = "the card was probably still selected from the last refused tap: no mask"
        elif mask and mask.get("kind") == TR.LANE:
            side = "left" if rec.get("zone") == "enemy_left" else "right"
            then = f"{self.name(rec.get('card'))} stays off the enemy {side} lane for this battle"
        else:
            until = (mask or {}).get("until_tick")
            secs = (int(until) - int(rec.get("tick") or 0)) / 20.0 if until is not None else 0.0
            then = (f"{self.name(rec.get('card'))} stays {TR.MASK_BOX_RADIUS + 1} tiles or more away from "
                    f"there for {secs:.0f} s")
        return (f"tap did nothing: {self.name(rec.get('card'))} -> {tuple(tile) if tile else '?'} at tick "
                f"{rec.get('tap_tick')} (still in its slot {rec.get('age')} ticks later); {then}")

    def _refusal_manual(self, rec: dict, ctx: BattleContext) -> None:
        """tap_guard_manual: switch AUTO to MANUAL (enforced, not dry), log and say it."""
        g, ctl = self.refusals, self.control
        mode_before = ctl.mode
        switched = False
        enforced = bool(g.enforce and not rec.get("dry"))
        if enforced and mode_before == CT.AUTO:
            switched = bool(ctl.set_mode(CT.MANUAL, source="refused taps"))
            if switched:
                g.mark_switched()
        if enforced:
            # The guard disarmed itself on this event. Whether it switched or the owner was in
            # MANUAL already (he took over on the "tap did nothing" lines), the next hand back to
            # AUTO (a new control generation) rearms it: else it never switched again this battle.
            self._refusal_gen = ctl.generation
        self._log("tap_guard_manual", battle=ctx.number, mode_before=mode_before, switched=switched, **rec)
        finals = rec.get("finals") or []
        last = finals[-1] if finals else {}
        tile = last.get("screen_tile") or last.get("tile")
        where = (f"last {self.name(last.get('card'))} -> {tuple(tile) if tile else '?'} at tick "
                 f"{last.get('tap_tick')}")
        if switched:
            key = self.hotkey if self.hotkey and str(self.hotkey).lower() not in ("", "none", "off") else None
            self.out(f"REFUSED TAPS: {len(finals)} taps this battle did nothing in the game ({where}). "
                     f"Switched to MANUAL: play yourself, or press {key or 'the window button'} for AUTO "
                     "once the status line looks right.")
        elif enforced:
            key = self.hotkey if self.hotkey and str(self.hotkey).lower() not in ("", "none", "off") else None
            self.out(f"REFUSED TAPS: {len(finals)} taps this battle did nothing in the game ({where}). "
                     f"Already {str(ctl.mode).upper()}: play yourself; press {key or 'the window button'} "
                     "for AUTO once the status line looks right (the count then starts again).")
        elif rec.get("dry") and not self._refusal_dry_said:
            self._refusal_dry_said = True
            self.out(f"dry run: the refused-tap guard would switch to MANUAL now ({len(finals)} refusals, "
                     f"{where}; no tap was sent, so the plays were watched as if tapped)")

    def _refusal_summary(self, *, session: bool = False) -> dict[str, Any]:
        """{"tap_guard": the guard's summary} for battle_end / session_end; {} without one."""
        if self.refusals is None:
            return {}
        try:
            return {"tap_guard": self.refusals.summary(session=session)}
        except Exception as exc:
            self._error("refusal_error", exc)
            return {}

    def _watch_tap(self, d: Any, vf: VisionFrame, *, virtual: bool) -> None:
        """Opens the guard's watch on a tap just sent (virtual: a dry run's play)."""
        if self.refusals is None:
            return
        try:
            self.refusals.sent(d.card, d.slot, d.cost, int(vf.tick), float(vf.t_capture),
                               tile=(d.tx, d.ty), screen_tile=d.world_tile,
                               elixir_milli=vf.hud.elixir_milli, spendable_milli=d.spendable_milli,
                               virtual=virtual)
        except Exception as exc:
            self._error("refusal_error", exc, tick=vf.tick)

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
        kw = d.suggest_kwargs()
        if d.is_play:                                  # the hand it chose from (human-play pairing)
            kw[CT.HAND_KEY] = [int(c) for c in vf.hud.hand]
        s = self.control.suggest(**kw)
        px = None
        if d.is_play:
            px = {"card": list(self.geometry.card_slot_px(d.slot, size=self.device_size)),
                  "tile": list(self.geometry.tap_px(d.tx, d.ty, size=self.device_size))}
        tapped, why = False, ""
        if d.is_play and self.control.should_tap(s):
            hold = None if self.tap_gate is None else self.tap_gate.hold_reason()
            veto = None if self.tapper is None or hold else self._tap_veto(frame)
            if self.tapper is None:
                why = "no tapper"
            elif hold:
                why = f"not tapped: {hold}"
                self.tap_gate.note_held()
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
            if ctx.elixir_cap is not None:             # its cap goes on once it is paid
                try:                                   # (the plays it was decided with in flight)
                    ctx.elixir_cap.sent(ctx.pending[-1], d.spendable_milli, in_flight=ctx.pending[:-1])
                except Exception as exc:
                    self._error("elixir_cap_error", exc, tick=vf.tick)
            self._watch_tap(d, vf, virtual=False)
            ms["capture_to_tap"] = round((self.clock() - float(frame.t)) * 1000.0, 2)
            self._log("tap", battle=ctx.number, tick=int(vf.tick), card=d.card, name=self.name(d.card),
                      slot=d.slot, model_tile=[d.tx, d.ty], screen_tile=list(d.world_tile),
                      card_px=px["card"], tile_px=px["tile"], ms=dict(ms))
            try:
                self.perception.note_own_tap(d.card, int(vf.tick), slot=d.slot, aim_mt=d.world_mt,
                                             t=self.clock())
            except Exception as exc:
                self._error("note_tap_error", exc, tick=vf.tick)
            if ctx.observer is not None:               # v2 rows: in own_pending until it runs
                try:
                    info = getattr(self.perception, "tap_info", None)
                    flight = dict(info(d.card)) if callable(info) else {}
                    if d.world_mt is not None:
                        flight["aim"] = tuple(d.world_mt)
                    ctx.observer.sent(d.card, int(vf.tick), slot=d.slot, cost=d.cost, **flight)
                except Exception as exc:
                    self._error("observer_error", exc, tick=vf.tick)
        elixir = state["players"][OWN_TEAM].get("elixir_milli")
        extra: dict[str, Any] = {"battle": ctx.number, "px": px, "ms": dict(ms), "elixir": elixir}
        cap = getattr(ctx.elixir_cap, "value", None)
        bites = cap is not None and elixir is not None and cap < elixir
        if bites:                                      # the decision spent min(HUD, cap)
            extra["elixir_cap"] = int(cap)
        seen = state.get("observer")
        if seen is not None:                           # v2 rows: what the row was given
            extra["opp_elixir"] = list(seen.get("opp_elixir") or ())
            extra["own_pending"] = [list(p) for p in seen.get("own_pending") or ()]
            row_elixir = state["players"][OWN_TEAM].get(ROW_ELIXIR_KEY)
            if row_elixir is not None:                 # the row's own bar (a tap in flight)
                extra["row_elixir"] = int(row_elixir)
        if why:
            extra["why"] = why
        rec = self.control.note_decision(s, tapped=tapped, **extra)
        if d.is_play and rec.get("action") == CT.ACTION_DRY:
            self._watch_tap(d, vf, virtual=True)       # a dry run: watched as if tapped (logs only)
        if d.is_play or self.last_decision is None:   # the status line keeps the last play
            self.last_decision = rec
        out = {"decision": d.to_record(), "action": rec.get("action"), "px": px}
        if bites:
            out["elixir_cap"] = int(cap)
        if "row_elixir" in extra:                      # v2 rows: the row's own bar (a tap in flight)
            out["row_elixir"] = extra["row_elixir"]
        if self._would_due(d, int(vf.tick)):
            w = self._would_play(vf, ms)
            if w is not None:
                out["would"] = w.to_dict()
        return out

    def _would_due(self, d: Any, tick: int) -> bool:
        """A drawn wait, while the bot does not tap, once per would_every_ticks."""
        if not self.would_every_ticks or d.is_play or getattr(d, "reason", "") != RD.R_WAIT:
            return False
        if not callable(getattr(self.decider, "would", None)) or not hasattr(self.control, "note_would"):
            return False
        if self.tapper is not None and self.control.should_tap():
            return False                               # AUTO with taps: no time for a second look
        return self._would_tick is None or tick - self._would_tick >= self.would_every_ticks

    def _would_play(self, vf: VisionFrame, ms: dict) -> Any:
        """Decider.would() as a would-play Suggestion (ControlState.note_would), or None."""
        self._would_tick = int(vf.tick)
        t0 = self.clock()
        try:
            w = self.decider.would()
        except Exception as exc:                       # a second look must never cost the frame
            self._error("would_error", exc, tick=vf.tick)
            return None
        finally:
            ms["would"] = round((self.clock() - t0) * 1000.0, 2)
        if w is None:
            return None
        # The hand it was decided with: a human play is compared with the newest would-play
        # whose hand still held the played card (the slot empties when the drag starts).
        kw = w.suggest_kwargs()
        kw[CT.HAND_KEY] = [int(c) for c in vf.hud.hand]
        self._would = self.control.note_would(**kw)
        return self._would

    def _say_frame_size(self, vf: VisionFrame) -> None:
        """The first frame's size, once: the HUD templates and boxes were measured at the
        owner's 747x1328; another size (a resized MuMu window) reads through normalised boxes
        and glyph scaling whose accuracy there is unmeasured."""
        size = tuple(int(v) for v in (vf.frame_size or ()))
        if len(size) != 2 or min(size) <= 0:
            return
        self._frame_size_said = True
        line = f"frame {size[0]}x{size[1]}"
        if size != CALIBRATED_FRAME:
            line += (f" (WARNING: the HUD was calibrated on {CALIBRATED_FRAME[0]}x"
                     f"{CALIBRATED_FRAME[1]} frames; check the status line's elixir and hand)")
        self.out(line)
        self._log("frame_size", size=list(size), calibrated=list(CALIBRATED_FRAME))

    def _start_battle(self, vf: VisionFrame) -> None:
        if self.results.pending:
            self._finish_result()
        n = self.battle.number
        # The observer (v2 rows) watches from the battle's first clock reading: the tracker and
        # the deriver ran from then on (the detector runs once there is a tick), and the plays
        # they saw before this frame are reported again after the reset below.
        first = self.battle.first_tick if self.battle.first_tick is not None else vf.tick
        self.ctx = BattleContext(n, float(vf.t_capture), first)
        kept = bool(getattr(self.perception, "towers_kept", False))
        self.perception.new_battle()
        self.control.new_battle()
        self.last_decision = None
        self._would = None
        self._would_tick = None
        self._tower_said = set()
        self._refusal_gen = None
        self._refusal_dry_said = False
        if self.refusals is not None:
            try:
                self.refusals.new_battle()
            except Exception as exc:
                self._error("refusal_error", exc, tick=vf.tick)
        primed = getattr(self.perception, "primed_until", None)
        if self.tap_gate is not None:
            try:
                if kept and primed is not None and self.tap_gate.decided:
                    self.tap_gate.resume(n)            # the same battle followed again: its verdict stays
                else:
                    self.tap_gate.new_battle(n, first, float(vf.t_capture))
                    self._gate_gen, self._gate_took = None, False
            except Exception as exc:
                self._error("tap_gate_error", exc, tick=vf.tick)
        self._log("battle_start", battle=n, tick=vf.tick, first_tick=first, mode=self.control.mode,
                  **({"joined": True, "primed_until": primed} if primed is not None else {}))
        self.out(f"battle {n} started (tick {vf.tick}, mode {self.control.mode.upper()})"
                 + (f"; joined in the middle: what is on the board until tick {primed} is not "
                    "counted as plays" if primed is not None else ""))
        make = getattr(self.decider, "observer_for", None)
        if make is not None:
            try:
                self.ctx.observer = make(self.ctx)
            except Exception as exc:                   # v2 rows then fail: no decision, no tap
                self._error("observer_error", exc, tick=vf.tick)
        # the deriver asks the battle's opponent elixir count (v2 rows) which plays it could pay
        # (events PHANTOMS); legacy rows have none: the confirmed cards alone
        bind = getattr(getattr(self.perception, "deriver", None), "bind_opp_elixir", None)
        if callable(bind):
            bind(getattr(self.ctx.observer, "opp", None))
        make = getattr(self.decider, "elixir_cap_for", None)
        if make is not None:
            try:
                self.ctx.elixir_cap = make(self.ctx)
            except Exception as exc:                   # no cap: the HUD's elixir as before
                self._error("elixir_cap_error", exc, tick=vf.tick)

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
        cap_summary = getattr(getattr(ctx, "elixir_cap", None), "summary", None)
        cap_info = cap_summary() if callable(cap_summary) else {}
        cap_info = {"elixir_cap": cap_info} if cap_info.get("plays") else {}   # taps were sent
        own_summary = getattr(getattr(getattr(ctx, "observer", None), "own", None), "summary", None)
        own_info = own_summary() if callable(own_summary) else {}
        if own_info.get("sent"):                       # v2 rows: how our taps in flight ended
            cap_info["own_flight"] = own_info
        self.results.open(n, self.battle.ended_reason, float(vf.t_capture), hud.crowns,
                          last_tick=self.battle.last_tick, towers=towers, could_fall=self._could_fall(hud),
                          **stats)
        info = self._towers_info()
        if callable(getattr(self.control, "set_notice", None)):
            self.control.set_notice(None)
        self._log("battle_end", battle=n, reason=self.battle.ended_reason, tick=self.battle.last_tick,
                  hud_crowns=list(hud.crowns), towers=towers, **stats, **self._opp_diagnostics(ctx),
                  **({"towers_info": info} if info else {}), **self._refusal_summary(), **self._gate_summary(),
                  **cap_info)
        self.out(f"battle {n} over ({self.battle.ended_reason}) at tick {self.battle.last_tick}")
        try:
            if keeps_towers(self.battle.ended_reason):   # perhaps the same battle coming back
                self.perception.end_battle(keep_towers=True)
            else:
                self.perception.end_battle()
        except Exception as exc:
            self._error("perception_error", exc)
        self.ctx = None

    def _opp_diagnostics(self, ctx: Any) -> dict[str, Any]:
        """The battle_end record's view of the opponent plays: how the deriver explained the
        enemy tracks (OppStats.summary: plays, the guards' dropped / late / relabelled, ...) and,
        for v2 rows, the elixir count's summary (OpponentElixir.summary: plays it could not have
        paid, the phantom-play diagnostic). Nothing it cannot read."""
        out: dict[str, Any] = {}
        try:
            stats = getattr(getattr(self.perception, "deriver", None), "opp_stats", None)
            if hasattr(stats, "summary"):
                out["opp_events"] = stats.summary()
            opp = getattr(getattr(ctx, "observer", None), "opp", None)
            if hasattr(opp, "summary"):
                out["opp_elixir"] = opp.summary()
        except Exception as exc:
            self._error("diagnostics_error", exc)
        return out

    def _finish_result(self) -> None:
        rec = self.results.finish()
        if rec is None:
            return
        self._log("result", **rec)
        own, enemy = rec["crowns"]
        tally = self.results.tally
        hud = rec["hud_crowns"]
        said = f"; HUD read {hud[0]}-{hud[1]}" if rec.get("crowns_disagree") else ""
        self.out(f"battle {rec['battle']}: {rec['outcome'].upper()} {own}-{enemy} ({rec['source']}{said}); "
                 f"record {tally['win']}-{tally['loss']}-{tally['draw']} (W-L-D)")

    # -- towers -----------------------------------------------------------------
    def _hud_report(self, vf: VisionFrame) -> Any:
        """The HUD reader's report of this frame (None off the battle screen, or none)."""
        if vf.screen != SCREEN_BATTLE:
            return None
        rep = getattr(getattr(self.perception, "hud_reader", None), "report", None)
        return rep if rep is not None and getattr(rep, "t", None) == vf.t_capture else None

    def _could_fall(self, hud: Hud) -> dict[str, list[str]]:
        """The towers the end screen may count although the HUD did not (tower_calib.could_fall_at_end:
        standing, their calibrated label seen in the battle's last 2.5 s); empty without calibration."""
        try:
            cal = getattr(getattr(self.perception, "hud_reader", None), "calib", None)
            absent = cal.absent() if cal is not None and callable(getattr(cal, "absent", None)) else {}
            gone = [k for k, v in (hud.towers or {}).items() if getattr(v, "hp", None) is not None and v.hp <= 0]
            return TC.could_fall_at_end(absent, gone)
        except Exception as exc:
            self._error("diagnostics_error", exc)
            return {"own": [], "enemy": []}

    def _towers_info(self) -> dict[str, Any]:
        """HudReader.towers_info() for the battle_end record ({} without it)."""
        try:
            fn = getattr(getattr(self.perception, "hud_reader", None), "towers_info", None)
            return dict(fn()) if callable(fn) else {}
        except Exception as exc:
            self._error("diagnostics_error", exc)
            return {}

    def _tower_logs(self, vf: VisionFrame) -> None:
        """tower_label once per princess tower and battle whose label is read away from its
        calibrated boxes (a tower skin), with a status line; towers_misread when the HUD's
        overtime check revived towers."""
        rep = self._hud_report(vf)
        if rep is None:
            return
        battle = self.battle.number if self.battle.in_battle else None
        misread = getattr(rep, "towers_misread", None)
        if misread:
            self._log("towers_misread", battle=battle, tick=vf.tick, **dict(misread))
            when = ("its label is back" if misread.get("why") == "label back" else
                    f"overtime began at {misread['crowns_before'][0]}-{misread['crowns_before'][1]} crowns")
            self.out(f"towers misread: {when}; revived {', '.join(misread['revived']) or 'none'}"
                     f" -> {misread['crowns_after'][0]}-{misread['crowns_after'][1]}")
        if battle is None:
            return
        moved = []
        for key, (dx, dy, src) in sorted((getattr(rep, "tower_shift", None) or {}).items()):
            if key in self._tower_said or src == TL.SOURCE_FIXED or not (dx or dy):
                continue
            self._tower_said.add(key)
            moved.append((key, dx, dy, src))
            self._log("tower_label", battle=battle, tick=vf.tick, key=key, shift_px=[dx, dy], source=src)
        for side in ("enemy", "own"):
            these = [m for m in moved if m[0].startswith(side)]
            if not these:
                continue
            dy = sum(m[2] for m in these) / len(these)
            which = f"{side} labels" if len(these) > 1 else f"{these[0][0]} label"
            self.out(f"{which} {abs(dy):.0f} px {'above' if dy < 0 else 'below'} calibration "
                     f"(tower skin; from the {these[0][3]}): HP read there")

    def _record(self, vf: VisionFrame, frame: Any, out: dict | None, ms: dict) -> None:
        active = self.battle.in_battle or vf.screen in (SCREEN_BATTLE, SCREEN_END)
        if not active and self.frames % self.idle_record_every:
            return
        extra: dict[str, Any] = {"battle": self.battle.number if self.battle.in_battle else None,
                                 "state": self.battle.state, "ms": ms}
        rep = self._hud_report(vf)
        if rep is not None and callable(getattr(rep, "towers_compact", None)):
            try:
                extra["towers_raw"] = rep.towers_compact()
            except Exception:
                pass
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
        would = ""
        w = self._would
        if w is not None and self.battle.in_battle:
            would = f" | would {w.name or self.name(w.card)}" + (
                f" -> {tuple(w.tile)}" if w.tile is not None else "") + f" @{w.tick}"
        towers = towers_text(hud) if self.battle.in_battle else ""
        refused = ""
        g = self.refusals
        if g is not None and self.battle.in_battle:
            j = len(getattr(g, "finals", ()) or ())
            if j:
                refused = f" | refused {j}/{g.k}"
        gate = ""
        if self.tap_gate is not None and self.battle.in_battle:
            word = self.tap_gate.status()
            if word:
                gate = f" | {word}"
        return (f"[{where}] tick {tick} | elixir {elixir} | hand {hand} next {self.name(hud.next)}{towers} | "
                f"{self.mode_label()}{refused}{gate} | last {last}{would} | {self._fps:.0f} fps, {self.taps} taps"
                + (f", {errs} errors" if errs else ""))


#: The status line's towers: the princess towers, screen left to right, theirs then ours.
STATUS_TOWERS = (("them", ("enemy_left", "enemy_right")), ("us", ("own_left", "own_right")))


def towers_text(hud: Hud) -> str:
    """The status line's " | towers them 3052 3052 us 3052 0 crowns 1-0" (the princess towers'
    HP as the HUD reads them, "?" unread; crowns ours-theirs), or "" when no tower is read. A 0
    for a standing tower means the bot treats its pocket as open: the owner checks it before
    handing back to AUTO (docs/SPARRING_RUNBOOK.md 18). A 0 the HUD did not confirm (no label
    evidence: Hud.towers_destroyed) keeps its pocket shut and is named after the crowns:
    "; enemy_left 0 (unconfirmed)"."""
    towers = hud.towers or {}
    if not any(towers.get(k) is not None for _, keys in STATUS_TOWERS for k in keys):
        return ""
    parts = []
    for side, keys in STATUS_TOWERS:
        hps = [getattr(towers.get(k), "hp", None) for k in keys]
        parts.append(side + " " + " ".join("?" if hp is None else str(int(hp)) for hp in hps))
    crowns = tuple(hud.crowns or (0, 0))
    confirmed = getattr(hud, "towers_destroyed", None)
    unconfirmed = [] if confirmed is None else [
        k for _, keys in STATUS_TOWERS for k in keys
        if getattr(towers.get(k), "hp", None) is not None and towers[k].hp <= 0 and k not in confirmed]
    return (f" | towers {' '.join(parts)} crowns {int(crowns[0])}-{int(crowns[1])}"
            + "".join(f"; {k} 0 (unconfirmed)" for k in unconfirmed))


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
                    help="auto | printwindow | adb | a recorded session to replay (taps refused). "
                         "auto: PrintWindow, with the adb fallback only under --tap (after "
                         "emulator.verify); with auto or printwindow a dry run opens no adb; "
                         "adb runs emulator.verify before its first screencap")
    ap.add_argument("--replay-fast", action="store_true",
                    help="with a recorded --source: every image once, as fast as the chain runs "
                         "(default: in real time, newer images replace ones the loop was too slow for)")
    ap.add_argument("--geometry", default=None,
                    help="an ArenaGeometry JSON (default data/vision/geometry_default_1080x1920.json)")
    ap.add_argument("--no-window", dest="no_control_window", action="store_true",
                    help="no control window (the same as --no-control-window)")
    ap.add_argument("--seed", type=int, default=None, help="seed of the decision rng")
    ap.add_argument("--decide", choices=(RD.MODE_LIVE, RD.MODE_LEGACY), default=RD.MODE_LIVE,
                    help="the decision rule: live (default: pending-slot mask, fallback, cap rule, "
                         "lane guard) or legacy (the rule rs_eval measured the 46%% with, but for "
                         "the time a card is in flight: its empty HUD slot is masked, so legacy "
                         "plays the next-best card where rs_eval waited; dry runs only: it re-taps "
                         "a card in flight, so --tap refuses it)")
    ap.add_argument("--would-every", type=int, default=10, metavar="TICKS",
                    help="while the bot does not tap, what it would play now is worked out on a wait "
                         "at most once per this many ticks, for the human-versus-bot record (0: never)")
    ap.add_argument("--torch-threads", type=int, default=0, metavar="N",
                    help="CPU threads of the token model (torch.set_num_threads; 0: torch's own "
                         "default). Fewer leaves more of the CPU to MuMu")
    ap.add_argument("--no-lane-guard", action="store_true", help="turn off the Cannon lane guard")
    ap.add_argument("--no-cap-rule", action="store_true",
                    help="allow waiting at full elixir with a playable card")
    ap.add_argument("--unknown-plays", action="store_true",
                    help="an enemy unit no card explains is an UNKNOWN play (row token 1, charged "
                         f"{EV.UNKNOWN_PLAY_COST} elixir; the count is then not exact); its unit stays "
                         "on the board as before; default: counted, not reported")
    ap.add_argument("--unknown-cost", type=int, default=None, metavar="ELIXIR",
                    help="with --unknown-plays: the elixir each UNKNOWN play is charged (default "
                         f"{EV.UNKNOWN_PLAY_COST}, the mean cost of the cards with no KataCR class; an "
                         "unlinked child or ability unit really costs 0, so the gate picks it)")
    add_opp_guard_args(ap)
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


def add_opp_guard_args(ap: argparse.ArgumentParser) -> None:
    """--legacy-opp-plays (tools/vision_replay.py has it too)."""
    ap.add_argument("--legacy-opp-plays", action="store_true",
                    help="derive opponent plays by the rules as they were before the guards "
                         "(events.LIVE_GUARDS off: a play on a track's 2nd sighting, rule 1 by the "
                         "same label only, plays past the river); default: the guards on")


def event_config(args: argparse.Namespace) -> "EV.EventConfig | None":
    """The deriver's EventConfig: EventConfig.live() (events.LIVE_GUARDS and LIVE_OWN on), with
    rule 6 on and each UNKNOWN play charged --unknown-cost (default events.UNKNOWN_PLAY_COST)
    under --unknown-plays. --legacy-opp-plays: the opponent guards off, the defaults but for our
    own side's LIVE_OWN (rule 4b, the drop-in of our own taps: not an opponent rule). ValueError
    for --unknown-cost without --unknown-plays or below 0. tools/vision_replay.py uses it too."""
    cost = getattr(args, "unknown_cost", None)
    over: dict[str, Any] = {}
    if args.unknown_plays:
        over["unknown_plays"] = True
        if cost is not None:
            over["unknown_cost"] = cost
    elif cost is not None:
        raise ValueError("--unknown-cost applies to --unknown-plays")
    if getattr(args, "legacy_opp_plays", False):
        return EV.EventConfig(**{**EV.LIVE_OWN, **over})
    return EV.EventConfig.live(**over)


def opp_guard_line(config: "EV.EventConfig | None") -> str:
    """The setup line of the opponent-play guards (vision_play and vision_replay print it)."""
    g = EV.guard_settings(config)
    if not g["on"]:
        return "legacy rules (--legacy-opp-plays): a play on a track's 2nd sighting"
    parts = []
    if g["min_play_age_ticks"]:
        parts.append(f"an enemy unit is a play once seen for {g['min_play_age_ticks']} ticks")
    if g["reid_any_label"]:
        parts.append("a lost one may come back under another label")
    if g["deploy_zone_mt"] is not None:
        parts.append(f"no play from a unit first seen over {g['deploy_zone_mt']:g} mt past the "
                     "river centre where a reported one could have walked")
    if g["own_reid_cross_team"] or g["own_recent_ticks"]:
        parts.append("our classes re-found as ours")
    return "guards on: " + "; ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        event_config(args)                              # refuse a bad --unknown-cost before anything
    except ValueError as exc:
        print(f"refusing to start: {exc}")
        return 2
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


LIVE_SOURCES = ("auto", "printwindow", "adb")


def is_replay(source: str) -> bool:
    """--source names a recording (anything but auto, printwindow and adb)."""
    return str(source) not in LIVE_SOURCES


def detector_config_from_pin(det_mod: Any, weights: Any) -> Any:
    """The live DetectorConfig at the pinned weights' training imgsz (train_detector.py writes
    it into the pin's meta), or None (explicit --detector weights, no pin, no imgsz: the
    DetectorConfig default, 640)."""
    if weights is not None:
        return None
    doc = det_mod.read_pin()
    imgsz = ((doc or {}).get("meta") or {}).get("imgsz")
    if not imgsz:
        return None
    import dataclasses
    return dataclasses.replace(det_mod.DetectorConfig(), imgsz=int(imgsz))


def detector_lines(detector: Any, det_error: str | None, table: Any) -> list[str]:
    """The setup lines of the detector: weights, digest, device, half, sides, classes, input
    size; a WARNING line for sides that are not joint or names the class list lacks."""
    if detector is None:
        return [f"  detector   NONE ({det_error})"]
    info = detector.info() if callable(getattr(detector, "info", None)) else {}
    digest = str(info.get("classes_digest") or getattr(table, "digest", "") or "")[:8] or "?"
    cfg = info.get("config") or {}
    lines = [f"  detector   {detector.weights} sha256 {str(detector.sha256)[:12]} on {detector.device_desc}, "
             f"half {info.get('half')}, side {info.get('side_mode')}, {info.get('model_classes')} classes, "
             f"{len(info.get('unresolved_names') or ())} unresolved, classes {digest}, "
             f"imgsz {cfg.get('imgsz')}"]
    if info.get("side_mode") not in (None, "joint"):
        lines.append(f"  WARNING    the detector's sides are {info.get('side_mode')!r}, not 'joint': "
                     "enemy and own units are told apart by the deck rule only")
    if info.get("unresolved_names"):
        lines.append("  WARNING    detector names the class list cannot place (ignored): "
                     + ", ".join(map(str, info["unresolved_names"])))
    return lines


def _main(args: argparse.Namespace) -> int:
    mods = import_live_modules()
    PS, RR, CK, RE = mods["public_state"], mods["rs_rows"], mods["rs_checkpoint"], mods["rs_engine"]
    ST, VC, CAP = mods["rs_spawn_table"], mods["vision_classes"], mods["scripts.vision.capture"]
    DET, GEO, HUD = mods["scripts.vision.detector"], mods["scripts.vision.geometry"], mods["scripts.vision.hud"]
    REC, TPL, TRK = mods["scripts.vision.record"], mods["scripts.vision.templates"], mods["scripts.vision.tracker"]

    replay = is_replay(args.source)
    if args.replay_fast and not replay:
        print("refusing to start: --replay-fast replays a recording (--source <session folder>)")
        return 2
    if replay and not Path(args.source).exists():
        print(f"refusing to start: no recording at {args.source} (--source)")
        return 2
    if args.tap and args.decide != RD.MODE_LIVE:
        print(f"refusing to start: --decide {args.decide} is for dry runs (it re-taps a card in "
              "flight); drop --tap or --decide")
        return 2

    # --- the engine and the model ------------------------------------------------
    ckpt = Path(args.ckpt) if args.ckpt else default_ckpt()
    if not ckpt.is_file():
        print(f"no checkpoint at {ckpt} (--ckpt)")
        return 2
    import torch
    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
    try:
        eng = RE.config_from_args(args)
        blob = torch.load(str(ckpt), map_location="cpu")
        meta = blob["meta"]
        plan = row_plan(meta)                           # legacy rows, or v2 (E14) rows
        live_compat = bool(meta.get("live_compat"))
        units = meta.get("units_mode") or RR.UNITS_TYPES
        if units == RR.UNITS_VISION and not live_compat:
            RR.UNITS = RR.UNITS_VISION                  # _setup fills the class lookup
        royalesim, _names, ids, cat = RR._setup(eng)
        sub = int(royalesim.SUBTILE_PER_MILLITILE)
        loaded = CK.load_policy(blob, eng)              # refuses another engine's vocabulary
    except (RE.EngineError, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"refusing to start: {ckpt}: {_short(exc)}")
        return 2
    schema = plan.schema
    policy = RD.TokenPolicyAdapter(loaded.model)
    names = [str(row[0]) for row in cat]
    missing = [n for n in args.deck if n not in ids]
    if missing:
        print(f"--deck: not in the engine's catalogue: {missing}")
        return 2
    deck_ids = [int(ids[n]) for n in args.deck]
    guard = () if args.no_lane_guard else RD.card_ids(cat, RD.GUARD_CARD_NAMES)
    has_args_json = ckpt.with_name("args.json").is_file()
    wait_keep = RD.wait_keep_for(str(ckpt))
    wait_bias = RD.wait_bias_for(str(ckpt))
    dcfg = RD.DecideConfig.for_mode(args.decide, wait_bias, guard_cards=guard)
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
    detector, det_error = None, None
    try:
        det_cfg = detector_config_from_pin(DET, args.detector)
        detector = DET.Detector.load(args.detector, device=args.device, geometry=geometry, table=table,
                                     **({} if det_cfg is None else {"config": det_cfg}))
        detector.warmup()
    except Exception as exc:                           # no weights, no ultralytics, bad pin
        detector, det_error = None, _short(exc)
    store = TPL.TemplateStore()
    # The hand boxes follow --geometry's card slots (the taps do too).
    hud_reader = HUD.HudReader(store, {n: ids[n] for n in args.deck}, deck=deck_ids,
                               regions=HUD.load_regions(geometry=geometry))
    # the live tracker rules (tracker.py CAPTURE: a lost track takes no other unit's box)
    tracker = TRK.Tracker(TRK.TrackerConfig.live(), own_classes=table.deck_labels(args.deck),
                          spell_classes=table.spell_labels())
    # N02 (--unknown-plays): UNKNOWN plays and unknown units; without it, the calls as before
    unknown = bool(args.unknown_plays)
    deriver = EV.EventDeriver(tables, event_config(args), own_deck=deck_ids)
    perception = LivePerception(hud_reader=hud_reader, tracker=tracker, deriver=deriver,
                                geometry=geometry, detector=detector)
    # v2 rows (E14) also show the spells in flight: spell tracks become "spells" rows in the
    # engine's SPELL_FIELDS layout (rs_rows.SF, filled by _setup). Legacy: as before.
    spell_kw = {"spell_fields": RR.SF} if plan.v2 else {}
    if unknown:
        spell_kw["unknown_units"] = True
    builder = PS.PublicStateBuilder(table, spawn, deck=deck_ids, levels=args.level, catalogue=cat,
                                    fields=RR.F, sub=sub, **spell_kw)
    law = RR.elixir_law(eng)
    observer = make_observer(cat, plan, fields=list(getattr(royalesim, "CATALOGUE_FIELDS", ()) or ())
                             or None, law=law if plan.v2 else None,
                             blind=detector is None, observer_cls=PS.LiveObserver)
    # The refused-tap guard (FIX E): its masks reach every decision and would-play through the
    # legality; enforced (masks, MANUAL after k) only once a tapper exists (below).
    tap_refusals = TR.RefusedTapGuard()
    decider = Decider(policy=policy, legality=RefusalLegality(legality, tap_refusals, sub=sub), config=dcfg,
                      state_fn=make_state_fn(builder, perception, plan),
                      row_fn=make_row_fn(RR.obs_row, sub, plan, live_compat=live_compat, units=units,
                                         row_inputs=PS.row_inputs),
                      rng=np.random.default_rng(args.seed), sub=sub, observer=observer,
                      elixir_cap=lambda: RD.ElixirCap(law))

    # --- taps: only with --tap, a live source, a detector, and a verified emulator -------------
    taps = bool(args.tap)
    refusal = None
    if replay:                                         # a recording is never tapped on
        taps, refusal = False, "the source is a recording"
    if taps and detector is None:
        taps, refusal = False, "no detector"
    if taps and not has_args_json:
        taps, refusal = False, (f"no args.json next to {ckpt.name}: the share of waits it trained on "
                                "is unknown, so its wait rate would be off")
    adb, serial, instance = CAP.machine_settings()
    attributor = CT.PlayAttributor(max_lag_ticks=CT.MAX_LAG_TICKS + EV.EventConfig().team_wait_ticks)
    ctl = CT.ControlState.from_args(args, taps_enabled=taps, stop_file=STOP_FILE, names=names,
                                    attributor=attributor)
    tapper = source = recorder = None
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
        tap_refusals.enforce = tapper is not None
        # A dry run never opens adb: "auto" adds its adb fallback only once the tapper has run
        # emulator.verify (above). A recording is replayed on this process's clock ("rebased":
        # its frames and this session's records share one clock).
        kind = args.source
        if kind == "auto" and tapper is None and sys.platform == "win32":
            kind = "printwindow"
        if not replay and tapper is None and kind in ("adb", "auto"):
            # An adb frame source in a dry run (--source adb, or auto off Windows): screencap
            # reads that device, so the serial must first be shown to be the Clash Royale
            # instance, as before a tap (no other MuMu instance is touched).
            try:
                mods["scripts.emulator"].verify(serial, adb)
            except SystemExit as exc:
                print(f"refusing to start: --source {args.source} reads the device over adb, and "
                      f"emulator.verify refused it: {exc}")
                return 2
            except OSError as exc:                     # adb itself would not start
                print(f"refusing to start: --source {args.source} reads the device over adb, and "
                      f"emulator.verify failed: {_short(exc)}")
                return 2
        try:
            if replay:
                source = CAP.FileReplaySource(Path(args.source), realtime=not args.replay_fast,
                                              timestamps="rebased")
            else:
                source = CAP.open_source(kind, instance=instance, adb=adb, serial=serial)
        except Exception as exc:
            print(f"cannot open the frame source {args.source!r}: {_short(exc)}")
            return 2
        # Replays go to their own root, so they never become latency_probe's --last session and
        # stay out of the live sessions' image budget.
        recorder = REC.VisionRecorder(REPLAY_ROOT if replay else REPORT_ROOT,
                                      prefix="replay" if replay else "vision_play",
                                      image_every=max(0, args.record_images))
        if replay:
            where = (f"replay of {args.source} ({len(source)} images, "
                     + ("as fast as the chain runs" if args.replay_fast else "in real time")
                     + "); no device, no adb")
        else:
            where = (f"{kind}" + (" (auto: a dry run opens no adb)" if kind != args.source else "")
                     + f"; device {serial} ({instance})")
        threads = torch.get_num_threads() if callable(getattr(torch, "get_num_threads", None)) else "?"
        lines = [f"vision_play  session {recorder.session_dir}",
                 f"  model      {ckpt} (schema {schema}, live_compat {live_compat}, units {units}); "
                 f"CPU, {threads} torch threads",
                 f"  decide     {dcfg.name}; wait_keep {wait_keep:g} (wait bias {wait_bias:.3f})"
                 + ("" if has_args_json else f"  WARNING: no args.json next to {ckpt.name}, the default")
                 + (f"; would-play every {args.would_every} ticks" if args.would_every > 0 else ""),
                 f"  deck       {', '.join(args.deck)} (level {args.level})",
                 *detector_lines(detector, det_error, table),
                 f"  hud        clock templates {'ok' if hud_reader.clock_reader.ready else 'MISSING'}"
                 f" ({store.root}); regions {hud_reader.regions.name}",
                 "  towers     each princess tower's HP label found at battle start (tower skins) and "
                 "read only where it is; a tower falls only once its label is gone for 2.5 s",
                 "  rows       the row's hand: a played slot shows its card in flight, an unread one "
                 "is a pad (decide still masks both)",
                 f"  geometry   {geometry.name}; taps in device pixels {DEVICE_SIZE[0]}x{DEVICE_SIZE[1]}",
                 f"  source     {where}",
                 "  taps       " + ("ON (verified Clash Royale instance)" if tapper is not None else
                                     f"off ({refusal or 'dry run: no --tap'})")]
        lines.append("  refusals   refused taps: " + (
            f"logged, masked, MANUAL after {tap_refusals.k} in a battle" if tap_refusals.enforce
            else "(dry run: logs only)"))
        cap_cfg = RD.ElixirCap(law).config()
        lines.append(f"  elixir     after each tap of ours is paid: spend at most min(HUD, cap), until the HUD "
                     f"is within {cap_cfg['tol_milli'] / 1000:g} elixir of it or {cap_cfg['timeout_ticks']} "
                     f"ticks pass")
        lines.append("  tap gate   no AUTO tap until the towers are calibrated (and boxed by the detector) "
                     "and the mode is classic; else MANUAL" + ("" if tapper is not None
                                                                else " (dry run: a line only)"))
        if plan.v2:
            lines.append(f"  v2 rows    opponent elixir counted from the plays seen; our taps in flight "
                         f"until they land (--obs-delay {plan.obs_delay} ticks); spells in flight")
        lines.append("  opp plays  " + opp_guard_line(deriver.config))
        if unknown:
            lines.append(f"  unknown    ON: an enemy unit no card explains is an UNKNOWN play (token 1, "
                         f"{deriver.config.unknown_cost} elixir, count not exact)")
        if detector is None:
            lines.append("  no detector: the board stays empty for the model and taps are refused")
        note = hud_reader.tower_note() if callable(getattr(hud_reader, "tower_note", None)) else None
        if note:
            lines.append(f"  towers     note: {note}")
        warn = hud_reader.tower_warning() if callable(getattr(hud_reader, "tower_warning", None)) else None
        if warn:
            lines.append(f"  WARNING    {warn}")
        if not hud_reader.clock_reader.ready:
            lines.append("  no clock templates: the clock cannot be read, so the bot never decides "
                         "(python -m scripts.vision.templates digits --font clock ...)")
        _say(lines)
        recorder.log("config", ckpt=str(ckpt), schema=schema, live_compat=live_compat, units=units,
                     deck=list(args.deck), deck_ids=deck_ids, level=args.level,
                     detector=None if detector is None else detector.info(), detector_error=det_error,
                     geometry=geometry.name, source=args.source, source_kind="replay" if replay else kind,
                     replay=None if not replay else {"path": str(args.source), "images": len(source),
                                                     "realtime": not args.replay_fast},
                     serial=serial, taps=tapper is not None, tap_refusal=refusal,
                     start_mode=args.start_mode, decide=dcfg.name, wait_keep=wait_keep,
                     wait_bias=round(wait_bias, 6), args_json=has_args_json,
                     cap_milli=dcfg.cap_milli, lane_guard=sorted(int(c) for c in dcfg.guard_cards),
                     would_every=args.would_every, torch_threads=threads,
                     row_hand="in flight, else pad", engine=eng.to_json(),
                     vocab_drift=list(loaded.drift),
                     clock_templates=bool(hud_reader.clock_reader.ready),
                     **({"rows": plan.to_json()} if plan.v2 else {}),
                     opp_guards=EV.guard_settings(deriver.config), tap_guard=tap_refusals.config(),
                     elixir_cap=cap_cfg,
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
                            unknown_plays=unknown, would_every_ticks=args.would_every,
                            refusals=tap_refusals, hotkey=args.hotkey, tap_gate=TG.TapGate())
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
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    sys.exit(main())
