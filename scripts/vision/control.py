"""Human takeover for the live runner: AUTO / MANUAL mode with live suggestions (V12).

The owner can take over from the bot in the middle of a friendly and hand
control back whenever they like.

Modes
    AUTO    the bot taps. Taps still need --tap (taps_enabled); without it,
            AUTO is a dry run.
    MANUAL  the bot never taps. It keeps perceiving and deciding (tracker,
            HUD, events, public memory, elixir counter), so it can take over
            again with a full picture. Its decisions become SUGGESTIONS,
            shown in the control window and logged.

Inputs. All are optional and can be combined; each one toggles the same
ControlState.
    hotkey        a global hotkey, F8 by default. It uses Win32
                  RegisterHotKey through ctypes in a daemon thread. Windows
                  only; elsewhere it is skipped and the reason is reported.
                  While the bot runs, other programs do not receive the key.
    window        a small always-on-top tkinter window. It shows the mode, a
                  big toggle button, the bot's current suggestion (card and
                  tile, also while MANUAL), its last decision and your last
                  play next to what the bot had. It is skipped when there is
                  no display, no tkinter, HASTYCR_HEADLESS=1 is set, or on
                  macOS (Tk only runs on the main thread there).
    control file  <repo>/MANUAL, a sibling of the STOP file. When it appears
                  the mode switches to MANUAL; when it disappears, to AUTO.
                  Only the change counts, so the hotkey and the button still
                  work while the file exists. A file already there at start()
                  counts as appearing.

Semantics
* A switch takes effect at the next decision. Every switch bumps a
  generation counter. A Suggestion remembers the mode and generation it was
  made under, and should_tap(s) is True only for a suggestion made in AUTO
  under the current generation. A tap decided before a switch is therefore
  dropped, even if the mode is switched back before the tap is sent.
* STOP (the STOP file, or request_stop()) makes should_tap False in every
  mode. stop_requested stays True even after the file is deleted; the runner
  ends.
* Own plays that match no bot tap are HUMAN plays. A play matches a bot tap
  of the same card that it follows by MIN_LAG_TICKS..MAX_LAG_TICKS, or failing
  that, any bot tap aimed within aim_near_mt of it, in case perception
  misread the card. Each human play is paired with the bot's suggestion at
  the estimated tap tick: the first sighting minus the median tap-to-sighting
  lag of the bot's own plays, or DEFAULT_LAG_TICKS until 3 have been
  measured. That gives a human-versus-bot record for review and future
  training data.
* Every logged decision records the mode it was made in ("mode") and the
  mode at logging time ("mode_now").

Runner integration (scripts/vision_play.py, V10, wired later):

    from scripts.vision import control as CT
    CT.add_control_args(ap)                  # --start-mode --hotkey --no-control-window --manual-file
    ctl = CT.ControlState.from_args(args, taps_enabled=args.tap, stop_file=STOP_FILE, names=card_names)
    print(ctl.start(**CT.start_kwargs(args)))     # {"control_file": "on: ...", "hotkey": ..., "window": ...}
    finder = CT.OwnPlayFinder(card_of)        # card_of(track) -> RoyaleSim card id or None
    try:
        while not ctl.stop_requested:         # STOP, in either mode
            vf = perceive()
            if new_battle: ctl.new_battle(); finder.reset()
            if vf.in_battle and vf.tick is not None:
                ctl.attribute(finder.update(vf.tracks, vf.tick), tick=vf.tick)
                ... decide ...
                s = ctl.suggest(card, slot=slot, tile=(sx, sy), tick=vf.tick, p_wait=p)
                #   (or ctl.suggest(tick=vf.tick, reason="wait") when the bot waits)
                tapped = False
                if ctl.should_tap(s):         # AUTO, --tap, same generation, no STOP
                    tapper.tap_pair(...); tapped = True
                ctl.note_decision(s, tapped=tapped, px=px, ms=ms)   # queued for the log
            ctl.flush_to(recorder)            # mode switches, decisions, human plays
    finally:
        ctl.stop()

Give attribute() one source of own plays: OwnPlayFinder's plays (it reports
every own play), or the event deriver's landings once it also reports the
plays it could not match (an OwnLanding with slot -1). Passing both would
count every bot play twice.

Run the module on its own to try the controls without a battle; it never taps:

    python -m scripts.vision.control [--start-mode auto] [--hotkey F8] [--seconds 60]

Imports only the standard library (tkinter and ctypes lazily, inside the
threads that use them).
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

import argparse  # noqa: E402
import gc  # noqa: E402
import importlib.util  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence  # noqa: E402

from .types import EMPTY_CARD, OWN_TEAM, OwnLanding, Track, now  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STOP_FILE = ROOT / "STOP"
DEFAULT_CONTROL_FILE = ROOT / "MANUAL"

AUTO, MANUAL = "auto", "manual"
MODES = (AUTO, MANUAL)
DEFAULT_START_MODE = MANUAL
DEFAULT_HOTKEY = "F8"

BOT, HUMAN = "bot", "human"

# Decision actions (note_decision / decision_record "action").
ACTION_TAP = "tap"            # the bot tapped
ACTION_WAIT = "wait"          # the bot chose to wait
ACTION_MANUAL = "manual"      # decided in MANUAL: a suggestion only
ACTION_DRY = "dry"            # AUTO without --tap
ACTION_DROPPED = "dropped"    # decided in AUTO, the mode switched before the tap
ACTION_STOPPED = "stopped"    # STOP was requested
ACTION_SKIPPED = "skipped"    # AUTO and allowed, but the runner did not tap (its reason)
ACTIONS = (ACTION_TAP, ACTION_WAIT, ACTION_MANUAL, ACTION_DRY, ACTION_DROPPED,
           ACTION_STOPPED, ACTION_SKIPPED)

DEFAULT_LAG_TICKS = 26        # live_play.LEAD: tap -> first sighting, until measured
MAX_LAG_TICKS = 60            # live_play.PENDING_TICKS: a play unseen after 3 s is lost
MIN_LAG_TICKS = -2            # clock-read noise: a sighting a hair "before" its tap still matches
MIN_LAG_SAMPLES = 3           # bot plays measured before the median replaces the default
PAIR_MAX_AGE_TICKS = 60       # a suggestion older than this is not paired with a human play
REPEAT_TICKS = 20             # an identical non-tap decision is logged once per this many ticks
AIM_NEAR_MT = 2000.0          # a landing this close to a bot tap's aim is that tap's play

_TILES_X, _TILES_Y = 18, 32   # scripts/vision/geometry.TILES_X / TILES_Y
_OFF = frozenset({"", "none", "off", "no", "false"})

# Win32 (winuser.h / winerror.h)
WM_QUIT, WM_HOTKEY, WM_USER = 0x0012, 0x0312, 0x0400
PM_NOREMOVE = 0x0000
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x0001, 0x0002, 0x0004, 0x0008, 0x4000
VK_F1 = 0x70
ERROR_HOTKEY_ALREADY_REGISTERED = 1409
HOTKEY_ID = 0xB0F8            # an application hotkey id must be in 0x0000..0xBFFF

_MOD_NAMES = {"alt": MOD_ALT, "ctrl": MOD_CONTROL, "control": MOD_CONTROL, "shift": MOD_SHIFT,
              "win": MOD_WIN}
_KEY_NAMES = {"pause": 0x13, "scrolllock": 0x91, "insert": 0x2D, "home": 0x24, "end": 0x23,
              "pageup": 0x21, "pagedown": 0x22, "space": 0x20}


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def parse_mode(text: str) -> str:
    """'auto' / 'manual' (any case) -> AUTO / MANUAL."""
    mode = str(text).strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {text!r}")
    return mode


def parse_hotkey(text: str | None) -> tuple[int, int] | None:
    """'F8', 'ctrl+F9', 'alt+shift+m' -> (modifier flags, virtual-key code).

    None, '', 'none' and 'off' mean no hotkey (None). Keys: F1-F24, A-Z, 0-9
    and the names in _KEY_NAMES; modifiers: ctrl, alt, shift, win.
    """
    if text is None:
        return None
    s = str(text).strip().lower().replace(" ", "")
    if s in _OFF:
        return None
    *mods, key = s.split("+")
    flags = 0
    for m in mods:
        if m not in _MOD_NAMES:
            raise ValueError(f"unknown modifier {m!r} in hotkey {text!r} (use ctrl, alt, shift, win)")
        flags |= _MOD_NAMES[m]
    if key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        vk = VK_F1 + int(key[1:]) - 1
    elif len(key) == 1 and key.isascii() and key.isalnum():
        vk = ord(key.upper())
    elif key in _KEY_NAMES:
        vk = _KEY_NAMES[key]
    else:
        raise ValueError(f"unknown key {key!r} in hotkey {text!r} "
                         f"(F1-F24, A-Z, 0-9, {', '.join(sorted(_KEY_NAMES))})")
    return flags, vk


def world_to_screen_tile(x_mt: float | None, y_mt: float | None) -> tuple[int, int] | None:
    """World millitiles -> the screen tile (sx, sy) holding the point (world = screen for team 1)."""
    if x_mt is None or y_mt is None:
        return None
    sx = min(_TILES_X - 1, max(0, int(math.floor(float(x_mt) / 1000.0))))
    sy = min(_TILES_Y - 1, max(0, int(math.floor(float(y_mt) / 1000.0))))
    return sx, sy


def _opt_int(v) -> int | None:
    return None if v is None else int(v)


def _opt_float(v) -> float | None:
    return None if v is None else float(v)


# ---------------------------------------------------------------------------
# suggestions and decision records
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Suggestion:
    """What the bot would play at one decision. card EMPTY_CARD (-1) means wait.

    tile is the SCREEN tile (sx, sy) of the tap (x 0 = screen left; for a model
    tile (tx, ty) that is live_play.screen_tile = (17 - tx, ty)). mode and gen
    are the control mode and generation it was decided under; t is the vision
    clock; seq is a running number.
    """
    tick: int | None
    card: int = EMPTY_CARD
    slot: int = -1
    tile: tuple[int, int] | None = None
    name: str = ""
    p_wait: float | None = None
    reason: str = ""
    mode: str = MANUAL
    gen: int = 0
    t: float = 0.0
    seq: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_wait(self) -> bool:
        return self.card < 0

    def label(self) -> str:
        if self.is_wait:
            return "wait" + (f" ({self.reason})" if self.reason else "")
        name = self.name or f"card {self.card}"
        return name if self.tile is None else f"{name} -> tile ({self.tile[0]}, {self.tile[1]})"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"tick": self.tick, "card": self.card, "name": self.name, "slot": self.slot,
                             "tile": None if self.tile is None else list(self.tile),
                             "p_wait": self.p_wait, "reason": self.reason, "mode": self.mode,
                             "gen": self.gen, "seq": self.seq}
        if self.extra:
            d["extra"] = dict(self.extra)
        return d


def decision_record(s: Suggestion, action: str, /, *, mode_now: str | None = None,
                    taps_enabled: bool | None = None, **extra: Any) -> dict[str, Any]:
    """The log record of one decision: the suggestion, what happened to it, and the mode.

    "mode" is the mode the decision was made in, "mode_now" the mode when it
    was logged. Extra fields (the suggestion's, then the caller's) are kept,
    but they cannot overwrite the core fields (s and action are positional-only,
    so an extra field may even be called "action" or "s").
    """
    core: dict[str, Any] = {"mode": s.mode, "action": action, "tick": s.tick, "card": s.card,
                            "name": s.name, "slot": s.slot,
                            "tile": None if s.tile is None else list(s.tile),
                            "p_wait": s.p_wait, "reason": s.reason, "gen": s.gen, "seq": s.seq}
    if mode_now is not None:
        core["mode_now"] = mode_now
    if taps_enabled is not None:
        core["taps_enabled"] = bool(taps_enabled)
    return {**s.extra, **extra, **core}


# ---------------------------------------------------------------------------
# own plays and their attribution
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OwnPlay:
    """A play of ours first seen on the board at `tick` (world millitiles).

    tap_tick is set when the play was already matched to a bot tap upstream
    (an OwnLanding from the event deriver); then it is a bot play.
    """
    card: int
    tick: int
    x_mt: float | None = None
    y_mt: float | None = None
    uid: int | None = None
    tap_tick: int | None = None


def as_own_play(obj: Any) -> OwnPlay | None:
    """OwnPlay, OwnLanding or a mapping -> OwnPlay. Returns None for a play that was
    not seen (an OwnLanding without land_tick: a missed tap). An OwnLanding with
    slot >= 0 is a matched bot tap; with slot < 0 it is an unmatched own play."""
    if isinstance(obj, OwnPlay):
        return obj
    if isinstance(obj, OwnLanding):
        if obj.land_tick is None:
            return None
        return OwnPlay(int(obj.card), int(obj.land_tick), obj.x_mt, obj.y_mt, obj.uid,
                       int(obj.tap_tick) if obj.slot >= 0 else None)
    if isinstance(obj, Mapping):
        tick = obj.get("tick", obj.get("land_tick"))
        if tick is None:
            return None
        return OwnPlay(int(obj["card"]), int(tick), _opt_float(obj.get("x_mt")),
                       _opt_float(obj.get("y_mt")), _opt_int(obj.get("uid")),
                       _opt_int(obj.get("tap_tick")))
    raise TypeError(f"not an own play: {type(obj).__name__}")


@dataclass(slots=True)
class BotTap:
    """A tap the bot sent, waiting for its unit to be seen."""
    card: int
    tick: int
    slot: int = -1
    t: float | None = None
    suggestion: Suggestion | None = None

    def aim_mt(self) -> tuple[float, float] | None:
        s = self.suggestion
        if s is None or s.tile is None:
            return None
        return (s.tile[0] + 0.5) * 1000.0, (s.tile[1] + 0.5) * 1000.0


@dataclass(slots=True)
class PlayAttribution:
    """An own play and who made it.

    For a bot play, tap_tick is when the tap was decided, lag_ticks the
    measured lag and suggestion the suggestion that was tapped (tapped_card is
    its card; it differs from card when the play was matched by aim). For a
    human play, tap_tick is an ESTIMATE (first sighting minus lag_ticks) and
    suggestion is the bot's suggestion at that tick, or None when it had none
    within PAIR_MAX_AGE_TICKS.
    """
    who: str
    card: int
    tick: int
    mode: str
    tap_tick: int | None = None
    lag_ticks: int | None = None
    x_mt: float | None = None
    y_mt: float | None = None
    uid: int | None = None
    suggestion: Suggestion | None = None
    tapped_card: int | None = None

    @property
    def tile(self) -> tuple[int, int] | None:
        return world_to_screen_tile(self.x_mt, self.y_mt)

    @property
    def same_card(self) -> bool | None:
        """Did the bot suggest this card? (None: no suggestion to compare with.)"""
        return None if self.suggestion is None else self.suggestion.card == self.card

    @property
    def tile_dist(self) -> float | None:
        """Tiles between the play and the centre of the suggested tile."""
        s = self.suggestion
        if s is None or s.is_wait or s.tile is None or self.x_mt is None or self.y_mt is None:
            return None
        return math.hypot(self.x_mt / 1000.0 - (s.tile[0] + 0.5), self.y_mt / 1000.0 - (s.tile[1] + 0.5))

    def to_record(self, name: str = "") -> dict[str, Any]:
        s, tile, dist = self.suggestion, self.tile, self.tile_dist
        age = (None if s is None or s.tick is None or self.tap_tick is None
               else self.tap_tick - s.tick)
        return {"who": self.who, "mode": self.mode, "card": self.card, "name": name,
                "tick": self.tick, "tap_tick": self.tap_tick, "estimated": self.who == HUMAN,
                "lag_ticks": self.lag_ticks, "x_mt": self.x_mt, "y_mt": self.y_mt,
                "tile": None if tile is None else list(tile), "uid": self.uid,
                "tapped_card": self.tapped_card,
                "bot": None if s is None else s.to_dict(), "bot_age_ticks": age,
                "same_card": self.same_card, "tile_dist": None if dist is None else round(dist, 2)}


class PlayAttributor:
    """Matches own plays to the bot's taps; the rest are the human's.

    Not thread-safe on its own; ControlState calls it under its lock. Tap lags
    of matched bot plays are kept across battles (they are a property of the
    rig) and their median estimates when a human tapped.
    """

    def __init__(self, *, default_lag_ticks: int = DEFAULT_LAG_TICKS,
                 min_lag_ticks: int = MIN_LAG_TICKS, max_lag_ticks: int = MAX_LAG_TICKS,
                 min_samples: int = MIN_LAG_SAMPLES, max_samples: int = 50,
                 aim_near_mt: float | None = AIM_NEAR_MT):
        self.default_lag_ticks = int(default_lag_ticks)
        self.min_lag_ticks = int(min_lag_ticks)
        self.max_lag_ticks = int(max_lag_ticks)
        self.min_samples = max(1, int(min_samples))
        self.aim_near_mt = None if aim_near_mt is None else float(aim_near_mt)
        self.pending: list[BotTap] = []
        self.lags: deque[int] = deque(maxlen=max(1, int(max_samples)))

    def note_bot_tap(self, card: int, tick: int, *, slot: int = -1, t: float | None = None,
                     suggestion: Suggestion | None = None) -> BotTap:
        tap = BotTap(int(card), int(tick), int(slot), t, suggestion)
        self.pending.append(tap)
        return tap

    def lag_estimate(self) -> int:
        """Median tap -> first-sighting lag of matched bot plays (default until min_samples)."""
        if len(self.lags) < self.min_samples:
            return self.default_lag_ticks
        return int(round(statistics.median(self.lags)))

    def _in_window(self, tap: BotTap, land_tick: int) -> bool:
        return self.min_lag_ticks <= land_tick - tap.tick <= self.max_lag_ticks

    def _take(self, card: int, land_tick: int, tap_tick: int | None = None) -> BotTap | None:
        """Remove and return the oldest pending tap of `card` that explains the landing."""
        for i, tap in enumerate(self.pending):
            if tap.card != card:
                continue
            if tap_tick is not None:
                if tap.tick != tap_tick:
                    continue
            elif not self._in_window(tap, land_tick):
                continue
            return self.pending.pop(i)
        return None

    def _take_near(self, play: OwnPlay) -> BotTap | None:
        """Remove and return the pending tap aimed closest to the play (within aim_near_mt)."""
        if self.aim_near_mt is None or play.x_mt is None or play.y_mt is None:
            return None
        best, best_d = None, self.aim_near_mt
        for i, tap in enumerate(self.pending):
            aim = tap.aim_mt()
            if aim is None or not self._in_window(tap, play.tick):
                continue
            d = math.hypot(play.x_mt - aim[0], play.y_mt - aim[1])
            if d <= best_d:
                best, best_d = i, d
        return None if best is None else self.pending.pop(best)

    def _sample(self, lag: int) -> None:
        if self.min_lag_ticks <= lag <= self.max_lag_ticks:
            self.lags.append(int(lag))

    def classify(self, play: OwnPlay, *, mode: str,
                 lookup: Callable[[int], Suggestion | None]) -> PlayAttribution:
        if play.tap_tick is not None:                       # matched upstream: a bot play
            tap = self._take(play.card, play.tick, play.tap_tick) or self._take(play.card, play.tick)
            lag = play.tick - play.tap_tick
            self._sample(lag)
            return PlayAttribution(BOT, play.card, play.tick, mode, play.tap_tick, lag, play.x_mt,
                                   play.y_mt, play.uid, None if tap is None else tap.suggestion,
                                   play.card if tap is None else tap.card)
        tap = self._take(play.card, play.tick) or self._take_near(play)
        if tap is not None:
            lag = play.tick - tap.tick
            self._sample(lag)
            return PlayAttribution(BOT, play.card, play.tick, mode, tap.tick, lag, play.x_mt,
                                   play.y_mt, play.uid, tap.suggestion, tap.card)
        return self.human(play, mode=mode, lookup=lookup)

    def human(self, play: OwnPlay, *, mode: str,
              lookup: Callable[[int], Suggestion | None]) -> PlayAttribution:
        lag = self.lag_estimate()
        est = play.tick - lag
        return PlayAttribution(HUMAN, play.card, play.tick, mode, est, lag, play.x_mt, play.y_mt,
                               play.uid, lookup(est))

    def expire(self, tick: int | None) -> list[BotTap]:
        """Drop and return the taps not seen within max_lag_ticks of `tick`."""
        if tick is None:
            return []
        gone = [tap for tap in self.pending if tick - tap.tick > self.max_lag_ticks]
        if gone:
            self.pending = [tap for tap in self.pending if tick - tap.tick <= self.max_lag_ticks]
        return gone

    def reset(self) -> list[BotTap]:
        """Forget the pending taps (a new battle); returns them. Lags are kept."""
        gone, self.pending = self.pending, []
        return gone


class OwnPlayFinder:
    """New own plays from the tracker's tracks.

    The event deriver matches our TAPS to landings; a play the human made has
    no tap, so this reports every new own play and leaves the matching to
    ControlState.attribute(). A new own track is a new play unless:
      * a track of the same card was seen within `recent_ticks` and `near_mt`
        of it: another unit of the same play (Skeletons) or the same unit
        re-acquired under a new uid;
      * the same card was counted within `same_card_ticks` (a rolling Log
        re-acquired far from where it was lost). Hog 2.6 cycles 8 cards, so
        no card comes back that fast.
    With prime=True, the tracks present at the first update after reset() are
    taken as known (a runner started mid-battle reports no phantom plays).

    card_of(track) gives the RoyaleSim card id of an own track (vision class
    -> card via the own deck, tools/vision_classes); None or a negative id
    skips the track (towers, classes no own card makes).
    """

    def __init__(self, card_of: Callable[[Track], int | None], *, near_mt: float = 2500.0,
                 recent_ticks: int = 40, same_card_ticks: int = 40, forget_ticks: int = 400,
                 prime: bool = True):
        self.card_of = card_of
        self.near_mt = float(near_mt)
        self.recent_ticks = int(recent_ticks)
        self.same_card_ticks = int(same_card_ticks)
        self.forget_ticks = int(forget_ticks)
        self.prime = bool(prime)
        self.reset()

    def reset(self) -> None:
        self._seen: dict[int, list] = {}          # uid -> [card, x_mt, y_mt, last tick]
        self._last_play: dict[int, int] = {}      # card -> tick of the last counted play
        self._primed = not self.prime

    def _near(self, card: int, x: float, y: float, tick: int) -> bool:
        for c, sx, sy, last in self._seen.values():
            if (c == card and tick - last <= self.recent_ticks
                    and math.hypot(x - sx, y - sy) <= self.near_mt):
                return True
        return False

    def update(self, tracks: Iterable[Track], tick: int) -> list[OwnPlay]:
        """Feed one frame's tracks; returns the plays first seen on this frame."""
        tick = int(tick)
        plays: list[OwnPlay] = []
        for tr in tracks:
            if tr.team != OWN_TEAM:
                continue
            card = self.card_of(tr)
            if card is None or int(card) < 0:
                continue
            card = int(card)
            seen = self._seen.get(tr.uid)
            if seen is not None:
                seen[1], seen[2], seen[3] = tr.x_mt, tr.y_mt, tick
                continue
            companion = self._near(card, tr.x_mt, tr.y_mt, tick)
            self._seen[tr.uid] = [card, tr.x_mt, tr.y_mt, tick]
            if not self._primed or companion:
                continue
            last = self._last_play.get(card)
            if last is not None and 0 <= tick - last <= self.same_card_ticks:
                continue
            self._last_play[card] = tick
            plays.append(OwnPlay(card, tick, tr.x_mt, tr.y_mt, tr.uid))
        self._primed = True
        stale = [uid for uid, rec in self._seen.items() if tick - rec[3] > self.forget_ticks]
        for uid in stale:
            del self._seen[uid]
        return plays


# ---------------------------------------------------------------------------
# the MANUAL control file
# ---------------------------------------------------------------------------
class ControlFileWatcher:
    """Calls on_change(True) when `path` appears and on_change(False) when it goes.

    poll() checks once; start() polls every poll_s seconds in a daemon thread.
    A file already present at the first poll counts as appearing. Callbacks
    run in order, under the watcher's lock.
    """

    def __init__(self, path: str | os.PathLike, on_change: Callable[[bool], None], *,
                 poll_s: float = 0.25, exists: Callable[[Path], bool] = os.path.exists):
        self.path = Path(path)
        self.on_change = on_change
        self.poll_s = float(poll_s)
        self._exists = exists
        self._present: bool | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    @property
    def present(self) -> bool | None:
        return self._present

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def poll(self) -> str | None:
        """'appeared', 'disappeared' or None (no change)."""
        with self._lock:
            try:
                present = bool(self._exists(self.path))
            except OSError:
                return None
            was, self._present = self._present, present
            if present == was or (was is None and not present):
                return None
            self.on_change(present)
            return "appeared" if present else "disappeared"

    def start(self) -> bool:
        if self.running:
            return True
        if self.poll_s <= 0:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="control-file", daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.poll()
            except Exception as exc:              # keep watching; report the last failure
                self.error = f"{type(exc).__name__}: {exc}"

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout_s)


# ---------------------------------------------------------------------------
# the global hotkey
# ---------------------------------------------------------------------------
class HotkeyBackend(Protocol):
    """register/wait/unregister run on the listener thread; wake() on any thread."""

    def register(self, hotkey_id: int, modifiers: int, vk: int) -> str | None: ...

    def wait(self) -> int | None: ...

    def wake(self) -> None: ...

    def unregister(self, hotkey_id: int) -> None: ...


class Win32HotkeyBackend:
    """RegisterHotKey on the calling thread and a GetMessageW loop for WM_HOTKEY."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self._ct, self._wt = ctypes, wintypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        lpmsg = ctypes.POINTER(wintypes.MSG)
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        user32.RegisterHotKey.restype = wintypes.BOOL
        user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.UnregisterHotKey.restype = wintypes.BOOL
        user32.GetMessageW.argtypes = [lpmsg, wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = wintypes.BOOL
        user32.PeekMessageW.argtypes = [lpmsg, wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
        user32.PeekMessageW.restype = wintypes.BOOL
        user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM,
                                              wintypes.LPARAM]
        user32.PostThreadMessageW.restype = wintypes.BOOL
        kernel32.GetCurrentThreadId.argtypes = []
        kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        self._user32, self._kernel32 = user32, kernel32
        self._tid: int | None = None

    def register(self, hotkey_id: int, modifiers: int, vk: int) -> str | None:
        msg = self._wt.MSG()
        # Give this thread its message queue first, so wake() can post to it.
        self._user32.PeekMessageW(self._ct.byref(msg), None, WM_USER, WM_USER, PM_NOREMOVE)
        self._tid = int(self._kernel32.GetCurrentThreadId())
        if self._user32.RegisterHotKey(None, hotkey_id, modifiers | MOD_NOREPEAT, vk):
            return None
        err = self._ct.get_last_error()
        why = (": another program has registered that key; pick another with --hotkey"
               if err == ERROR_HOTKEY_ALREADY_REGISTERED else "")
        return f"RegisterHotKey failed (error {err}{why})"

    def wait(self) -> int | None:
        msg = self._wt.MSG()
        while True:
            r = self._user32.GetMessageW(self._ct.byref(msg), None, 0, 0)
            if r == 0 or r == -1:                 # WM_QUIT, or an error
                return None
            if msg.message == WM_HOTKEY:
                return int(msg.wParam)

    def wake(self) -> None:
        if self._tid is not None:
            self._user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)

    def unregister(self, hotkey_id: int) -> None:
        self._user32.UnregisterHotKey(None, hotkey_id)


class HotkeyListener:
    """Calls on_press() from a daemon thread each time the global hotkey is pressed.

    Windows only (Win32 RegisterHotKey); elsewhere start() returns False and
    `error` says why. A press within debounce_s of the last one is ignored
    (MOD_NOREPEAT already drops auto-repeat while the key is held).
    """

    def __init__(self, on_press: Callable[[], Any], hotkey: str | None = DEFAULT_HOTKEY, *,
                 backend: HotkeyBackend | None = None, platform: str | None = None,
                 debounce_s: float = 0.25, clock: Callable[[], float] = now,
                 hotkey_id: int = HOTKEY_ID):
        self.on_press = on_press
        self.hotkey = hotkey
        self.platform = sys.platform if platform is None else str(platform)
        self.debounce_s = float(debounce_s)
        self.hotkey_id = int(hotkey_id)
        self._backend = backend
        self._clock = clock
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: float | None = None
        self.error: str | None = None
        self.callback_error: str | None = None
        self.presses = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self.error is None

    def start(self, timeout_s: float = 2.0) -> bool:
        if self._thread is not None:
            return self.running
        try:
            keys = parse_hotkey(self.hotkey)
        except ValueError as exc:
            self.error = str(exc)
            return False
        if keys is None:
            self.error = "disabled"
            return False
        backend = self._backend
        if backend is None:
            if self.platform != "win32":
                self.error = f"global hotkeys need Windows (RegisterHotKey); this is {self.platform}"
                return False
            try:
                backend = Win32HotkeyBackend()
            except Exception as exc:
                self.error = f"no Win32 hotkey API ({type(exc).__name__}: {exc})"
                return False
        self._backend = backend
        self.error = None
        self._ready.clear()
        self._stopping.clear()
        self._thread = threading.Thread(target=self._run, args=(backend, *keys),
                                        name="control-hotkey", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            self.error = "the hotkey thread did not start"
            return False
        return self.error is None

    def _run(self, backend: HotkeyBackend, modifiers: int, vk: int) -> None:
        try:
            err = backend.register(self.hotkey_id, modifiers, vk)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        if err:
            self.error = err
            self._ready.set()
            return
        self._ready.set()
        try:
            while not self._stopping.is_set():
                hid = backend.wait()
                if hid is None or self._stopping.is_set():
                    break
                if hid != self.hotkey_id:
                    continue
                t = self._clock()
                if self._last is not None and t - self._last < self.debounce_s:
                    continue
                self._last = t
                self.presses += 1
                try:
                    self.on_press()
                except Exception as exc:          # a bad callback must not end the listener
                    self.callback_error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                backend.unregister(self.hotkey_id)
            except Exception:
                pass

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stopping.set()
        t, self._thread = self._thread, None
        if t is None:
            return
        try:
            if self._backend is not None:
                self._backend.wake()
        except Exception:
            pass
        if t is not threading.current_thread():
            t.join(timeout_s)


# ---------------------------------------------------------------------------
# the control window
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ControlSnapshot:
    """A consistent copy of what the control window shows."""
    mode: str
    taps_enabled: bool
    stopped: bool
    stop_reason: str | None
    generation: int
    tick: int | None
    suggestion: Suggestion | None
    last_decision: dict[str, Any] | None
    last_human: dict[str, Any] | None
    counts: dict[str, int]


COLORS = {"auto": "#2e7d32", "dry": "#1565c0", "manual": "#e65100", "stopped": "#616161"}


def _card_tile(rec: Mapping[str, Any]) -> str:
    name = rec.get("name") or f"card {rec.get('card')}"
    tile = rec.get("tile")
    return name if not tile else f"{name} ({tile[0]}, {tile[1]})"


def describe_decision(rec: Mapping[str, Any]) -> str:
    """One line for a decision record (decision_record)."""
    action, tick, reason = rec.get("action"), rec.get("tick"), rec.get("reason")
    what = _card_tile(rec)
    why = f" ({reason})" if reason else ""
    text = {
        ACTION_TAP: f"tapped {what}",
        ACTION_MANUAL: f"manual, would play {what}",
        ACTION_DRY: f"dry run, would play {what}",
        ACTION_DROPPED: f"dropped {what} (mode switched before the tap)",
        ACTION_STOPPED: f"not played {what} (stopped)",
        ACTION_WAIT: f"wait{why}",
        ACTION_SKIPPED: f"not played {what}{why}",
    }.get(action, str(action))
    return text if tick is None else f"t={tick} {text}"


def describe_human(rec: Mapping[str, Any]) -> str:
    """One line for a human-play record (PlayAttribution.to_record)."""
    bot = rec.get("bot")
    if bot is None:
        other = "no suggestion"
    elif int(bot.get("card", EMPTY_CARD)) < 0:
        other = "wait"
    else:
        other = _card_tile(bot)
    return f"t={rec.get('tick')} {_card_tile(rec)}; the bot had {other}"


def window_lines(snap: ControlSnapshot, hotkey: str | None = DEFAULT_HOTKEY) -> dict[str, str]:
    """The control window's texts and colour for a snapshot (pure, for tests)."""
    if snap.stopped:
        mode, color = f"STOPPED ({snap.stop_reason})", COLORS["stopped"]
    elif snap.mode == AUTO and snap.taps_enabled:
        mode, color = "AUTO: the bot plays", COLORS["auto"]
    elif snap.mode == AUTO:
        mode, color = "AUTO (dry run: no --tap)", COLORS["dry"]
    else:
        mode, color = "MANUAL: you play", COLORS["manual"]
    key = f"  [{hotkey}]" if hotkey else ""
    button = ("Take over" if snap.mode == AUTO else "Hand back to the bot") + key
    s = snap.suggestion
    suggestion = "Bot suggests: " + ("-" if s is None else s.label())
    if s is not None and s.tick is not None:
        suggestion += f"   (t={s.tick})"
    c = snap.counts
    return {
        "mode": mode, "color": color, "button": button,
        "button_state": "disabled" if snap.stopped else "normal",
        "suggestion": suggestion,
        "decision": "Last decision: " + ("-" if snap.last_decision is None
                                         else describe_decision(snap.last_decision)),
        "human": "Your last play: " + ("-" if snap.last_human is None
                                       else describe_human(snap.last_human)),
        "status": (f"taps {c.get('taps', 0)}   dropped {c.get('dropped', 0)}   "
                   f"your plays {c.get('human_plays', 0)}   switches {c.get('switches', 0)}"),
    }


def window_available(env: Mapping[str, str] | None = None,
                     platform: str | None = None) -> tuple[bool, str]:
    """(True, '') when a tkinter window can be opened here, else (False, why)."""
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else str(platform)
    if str(env.get("HASTYCR_HEADLESS", "")).strip() == "1":
        return False, "HASTYCR_HEADLESS=1"
    if platform == "darwin":
        return False, "macOS runs Tk only on the main thread; use the hotkey or the MANUAL file"
    if platform != "win32" and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        return False, "no display (headless)"
    try:
        found = (importlib.util.find_spec("tkinter") is not None
                 and importlib.util.find_spec("_tkinter") is not None)
    except (ImportError, ValueError):
        found = False
    if not found:
        return False, "tkinter is not installed"
    return True, ""


class ControlWindow:
    """A small always-on-top tkinter window on its own thread.

    It shows the mode, a big toggle button, the bot's current suggestion, its
    last decision and your last play. Every Tk call runs on the window's
    thread; it reads ControlState.snapshot() every refresh_ms. Closing the
    window does not change the mode; the hotkey and the MANUAL file keep
    working.
    """

    def __init__(self, state: "ControlState", *, hotkey_label: str | None = DEFAULT_HOTKEY,
                 refresh_ms: int = 150, env: Mapping[str, str] | None = None,
                 platform: str | None = None, title: str = "Sparring bot", topmost: bool = True):
        self.state = state
        self.hotkey_label = hotkey_label
        self.refresh_ms = max(20, int(refresh_ms))
        self.title = title
        self.topmost = bool(topmost)
        self._env, self._platform = env, platform
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._root = None
        self._widgets: dict[str, Any] | None = None
        self.error: str | None = None

    def available(self) -> tuple[bool, str]:
        return window_available(self._env, self._platform)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._closed.is_set()

    def press_button(self) -> str:
        """What the big button does: toggle the mode. Returns the new mode."""
        return self.state.toggle("button")

    def start(self, timeout_s: float = 5.0) -> bool:
        if self._thread is not None:
            return self.running
        ok, why = self.available()
        if not ok:
            self.error = why
            return False
        self.error = None
        self._ready.clear()
        self._stopping.clear()
        self._closed.clear()
        self._thread = threading.Thread(target=self._run, name="control-window", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            self.error = "the control window did not open in time"
            return False
        return self.error is None

    def _on_close(self) -> None:
        self._closed.set()
        self.state.log("control_input", input="window", event="closed")

    def _build(self, tk) -> None:
        root = tk.Tk()
        self._root = root
        root.title(self.title)
        if self.topmost:
            root.attributes("-topmost", True)
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        big, small = ("Segoe UI", 16, "bold"), ("Segoe UI", 11)
        w = {"mode": tk.Label(root, font=big, width=28, pady=6, fg="white"),
             "button": tk.Button(root, font=big, width=28, height=2, command=self.press_button),
             "suggestion": tk.Label(root, font=small, anchor="w", justify="left", width=56),
             "decision": tk.Label(root, font=small, anchor="w", justify="left", width=56),
             "human": tk.Label(root, font=small, anchor="w", justify="left", width=56),
             "status": tk.Label(root, font=("Segoe UI", 9), anchor="w", fg="#555555", width=64)}
        w["mode"].pack(fill="x")
        w["button"].pack(fill="x", padx=6, pady=6)
        for k in ("suggestion", "decision", "human"):
            w[k].pack(fill="x", padx=6)
        w["status"].pack(fill="x", padx=6, pady=(4, 6))
        self._widgets = w

    def _refresh(self) -> None:
        root, w = self._root, self._widgets
        if root is None or w is None:
            return
        if self._stopping.is_set() or self._closed.is_set():
            root.destroy()
            return
        try:
            lines = window_lines(self.state.snapshot(), self.hotkey_label)
            w["mode"].config(text=lines["mode"], bg=lines["color"])
            w["button"].config(text=lines["button"], state=lines["button_state"])
            for k in ("suggestion", "decision", "human", "status"):
                w[k].config(text=lines[k])
        except Exception as exc:                  # show what we can; never kill the runner
            self.error = f"refresh failed ({type(exc).__name__}: {exc})"
        root.after(self.refresh_ms, self._refresh)

    def _run(self) -> None:
        try:
            import tkinter as tk
            self._build(tk)
            self._refresh()
        except Exception as exc:
            self.error = f"the control window failed to open ({type(exc).__name__}: {exc})"
            self._teardown()
            self._ready.set()
            return
        self._ready.set()
        try:
            self._root.mainloop()
        except Exception as exc:
            self.error = f"the control window failed ({type(exc).__name__}: {exc})"
        finally:
            self._teardown()

    def _teardown(self) -> None:
        # Drop every Tk object on the Tk thread, so none is freed on another one.
        root, self._root, self._widgets = self._root, None, None
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        del root
        gc.collect()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stopping.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout_s)


# ---------------------------------------------------------------------------
# ControlState
# ---------------------------------------------------------------------------
NameSource = Mapping[int, str] | Sequence[str] | Callable[[int], str] | None


class ControlState:
    """AUTO / MANUAL with thread-safe toggles, suggestions, attribution and log records.

    start_mode: MANUAL by default (--start-mode). taps_enabled: --tap; without
    it AUTO never taps. stop_file: the STOP file (None = only request_stop()).
    control_file: the MANUAL file used by start(control_file=True) (None = no
    file). names: card id -> name (a mapping, a list indexed by id, or a
    callable), used for display and logs.

    Thread-safe: the hotkey, window and file-watcher threads call toggle() /
    set_mode() / snapshot(); the runner calls the rest. Log records are queued
    as (kind, data) pairs; drain_log() / flush_to(recorder) hands them out.
    """

    def __init__(self, start_mode: str = DEFAULT_START_MODE, *, taps_enabled: bool = False,
                 stop_file: str | os.PathLike | None = DEFAULT_STOP_FILE,
                 control_file: str | os.PathLike | None = DEFAULT_CONTROL_FILE,
                 names: NameSource = None, history: int = 600,
                 pair_max_age_ticks: int = PAIR_MAX_AGE_TICKS, repeat_ticks: int = REPEAT_TICKS,
                 log_waits: bool = False, attributor: PlayAttributor | None = None,
                 clock: Callable[[], float] = now, exists: Callable[[Path], bool] = os.path.exists,
                 max_log: int = 10_000):
        self._lock = threading.RLock()
        self._mode = parse_mode(start_mode)
        self.start_mode = self._mode
        self._gen = 0
        self._taps_enabled = bool(taps_enabled)
        self.stop_file = None if stop_file is None else Path(stop_file)
        self.control_file = None if control_file is None else Path(control_file)
        self._names = names
        self._history: deque[Suggestion] = deque(maxlen=max(1, int(history)))
        self.pair_max_age_ticks = int(pair_max_age_ticks)
        self.repeat_ticks = int(repeat_ticks)
        self.log_waits = bool(log_waits)
        self.attributor = attributor if attributor is not None else PlayAttributor()
        self._clock = clock
        self._exists = exists
        self._log: deque[tuple[str, dict[str, Any]]] = deque(maxlen=max(1, int(max_log)))
        self.log_overflow = 0
        self._seq = 0
        self._tick: int | None = None
        self._last_decision: dict[str, Any] | None = None
        self._last_logged: tuple[tuple, int | None] | None = None
        self._last_human: dict[str, Any] | None = None
        self._stop_reason: str | None = None
        self.counts = {"switches": 0, "suggestions": 0, "taps": 0, "dropped": 0, "bot_plays": 0,
                       "human_plays": 0, "bot_unseen": 0}
        self._watcher: ControlFileWatcher | None = None
        self._hotkey: HotkeyListener | None = None
        self._window: ControlWindow | None = None
        self.inputs: dict[str, str] = {}

    @classmethod
    def from_args(cls, args: argparse.Namespace, *, taps_enabled: bool,
                  stop_file: str | os.PathLike | None = DEFAULT_STOP_FILE,
                  names: NameSource = None, **kwargs: Any) -> "ControlState":
        """A ControlState from add_control_args() options (start mode, MANUAL file)."""
        manual = str(getattr(args, "manual_file", DEFAULT_CONTROL_FILE))
        return cls(getattr(args, "start_mode", DEFAULT_START_MODE), taps_enabled=taps_enabled,
                   stop_file=stop_file,
                   control_file=None if manual.strip().lower() in _OFF else Path(manual),
                   names=names, **kwargs)

    # -- mode ---------------------------------------------------------------
    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def is_auto(self) -> bool:
        return self.mode == AUTO

    @property
    def is_manual(self) -> bool:
        return self.mode == MANUAL

    @property
    def generation(self) -> int:
        with self._lock:
            return self._gen

    @property
    def taps_enabled(self) -> bool:
        return self._taps_enabled

    @property
    def tick(self) -> int | None:
        with self._lock:
            return self._tick

    def set_mode(self, mode: str, source: str = "api") -> bool:
        """Switch to `mode`; returns False if it was already the mode."""
        mode = parse_mode(mode)
        with self._lock:
            if mode == self._mode:
                return False
            prev, self._mode = self._mode, mode
            self._gen += 1
            self.counts["switches"] += 1
            self._push("mode", {"mode": mode, "prev": prev, "source": str(source), "gen": self._gen,
                                "tick": self._tick, "t": self._clock()})
            return True

    def toggle(self, source: str = "api") -> str:
        """AUTO <-> MANUAL; returns the new mode."""
        with self._lock:
            self.set_mode(MANUAL if self._mode == AUTO else AUTO, source)
            return self._mode

    # -- stop ---------------------------------------------------------------
    def request_stop(self, reason: str = "request") -> None:
        with self._lock:
            if self._stop_reason is None:
                self._stop_reason = str(reason)
                self._push("control_stop", {"reason": self._stop_reason, "tick": self._tick,
                                            "t": self._clock()})

    def _stopped_locked(self) -> bool:
        if self._stop_reason is not None:
            return True
        if self.stop_file is not None:
            try:
                hit = bool(self._exists(self.stop_file))
            except OSError:
                hit = False
            if hit:
                self.request_stop("STOP file")
                return True
        return False

    @property
    def stop_requested(self) -> bool:
        """True once STOP exists (or request_stop() was called); it stays True."""
        with self._lock:
            return self._stopped_locked()

    @property
    def stop_reason(self) -> str | None:
        with self._lock:
            return self._stop_reason

    # -- suggestions and decisions ------------------------------------------
    def card_name(self, card: int) -> str:
        card = int(card)
        if card < 0:
            return "wait"
        names = self._names
        try:
            if callable(names):
                return str(names(card))
            if isinstance(names, Mapping):
                if card in names:
                    return str(names[card])
            elif names is not None and 0 <= card < len(names):
                return str(names[card])
        except Exception:
            pass
        return f"card {card}"

    def suggest(self, card: int = EMPTY_CARD, *, slot: int = -1,
                tile: Sequence[int] | None = None, tick: int | None = None, name: str = "",
                p_wait: float | None = None, reason: str = "", **extra: Any) -> Suggestion:
        """Record what the bot would play now (card -1 = wait) under the current mode.

        Call it at every decision, in both modes, also when the bot waits.
        """
        card = int(card)
        with self._lock:
            self._seq += 1
            if tick is not None:
                self._tick = int(tick)
            wait = card < 0
            s = Suggestion(None if tick is None else int(tick), card, -1 if wait else int(slot),
                           None if wait or tile is None else (int(tile[0]), int(tile[1])),
                           name or self.card_name(card), None if p_wait is None else float(p_wait),
                           str(reason), self._mode, self._gen, self._clock(), self._seq, dict(extra))
            self._history.append(s)
            self.counts["suggestions"] += 1
            return s

    def should_tap(self, suggestion: Suggestion | None = None) -> bool:
        """May the bot tap now? AUTO, --tap, no STOP and, for a suggestion, one that
        is a play decided in AUTO under the current generation (no switch since)."""
        with self._lock:
            if self._stopped_locked() or not self._taps_enabled or self._mode != AUTO:
                return False
            if suggestion is None:
                return True
            return (not suggestion.is_wait and suggestion.mode == AUTO
                    and suggestion.gen == self._gen)

    def _action_locked(self, s: Suggestion, tapped: bool) -> str:
        if tapped:
            return ACTION_TAP
        if s.is_wait:
            return ACTION_WAIT
        if self._stopped_locked():
            return ACTION_STOPPED
        if s.mode == MANUAL:
            return ACTION_MANUAL
        if not self._taps_enabled:
            return ACTION_DRY
        if s.gen != self._gen or self._mode != AUTO:
            return ACTION_DROPPED
        return ACTION_SKIPPED

    def _should_log_locked(self, rec: Mapping[str, Any]) -> bool:
        action, tick = rec["action"], rec["tick"]
        if action == ACTION_WAIT and not self.log_waits:
            return False
        key = (action, rec["card"], rec["slot"], None if rec["tile"] is None else tuple(rec["tile"]))
        last = self._last_logged
        if (action not in (ACTION_TAP, ACTION_DROPPED) and last is not None and last[0] == key
                and tick is not None and last[1] is not None
                and 0 <= tick - last[1] < self.repeat_ticks):
            return False
        self._last_logged = (key, tick)
        return True

    def note_decision(self, suggestion: Suggestion, /, *, tapped: bool, **extra: Any) -> dict[str, Any]:
        """Settle a suggestion: tapped or not, and why. Returns the decision record.

        The record is also queued for the log (a "decision" record; do not log
        it again). Identical non-tap decisions are queued at most once per
        repeat_ticks, waits only with log_waits. A tap registers a pending bot
        tap for attribution. last_decision keeps the last decision that was
        not a wait.
        """
        s = suggestion
        for core in ("mode_now", "taps_enabled"):  # core fields: set below, never the caller's
            extra.pop(core, None)
        with self._lock:
            action = self._action_locked(s, bool(tapped))
            rec = decision_record(s, action, mode_now=self._mode, taps_enabled=self._taps_enabled,
                                  **extra)
            if action != ACTION_WAIT:             # the window shows waits as the suggestion
                self._last_decision = rec
            if action == ACTION_TAP:
                self.counts["taps"] += 1
                if s.tick is not None:
                    self.attributor.note_bot_tap(s.card, s.tick, slot=s.slot, t=self._clock(),
                                                 suggestion=s)
            elif action == ACTION_DROPPED:
                self.counts["dropped"] += 1
            if self._should_log_locked(rec):
                self._push("decision", dict(rec))
            return rec

    @property
    def current_suggestion(self) -> Suggestion | None:
        with self._lock:
            return self._history[-1] if self._history else None

    @property
    def last_decision(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._last_decision is None else dict(self._last_decision)

    def suggestions(self) -> list[Suggestion]:
        with self._lock:
            return list(self._history)

    def _suggestion_at_locked(self, tick: int) -> Suggestion | None:
        for s in reversed(self._history):
            if s.tick is not None and s.tick <= tick:
                return None if tick - s.tick > self.pair_max_age_ticks else s
        return None

    def suggestion_at(self, tick: int) -> Suggestion | None:
        """The latest suggestion made at or before `tick` (None if older than pair_max_age_ticks)."""
        with self._lock:
            return self._suggestion_at_locked(int(tick))

    # -- attribution ----------------------------------------------------------
    def _account_locked(self, a: PlayAttribution) -> None:
        if a.who == HUMAN:
            self.counts["human_plays"] += 1
            rec = a.to_record(self.card_name(a.card))
            self._last_human = rec
            self._push("human_play", dict(rec))
        else:
            self.counts["bot_plays"] += 1

    def attribute(self, plays: Iterable[Any], *, tick: int | None = None) -> list[PlayAttribution]:
        """Attribute this frame's own plays (OwnPlay, OwnLanding or mappings) to the bot
        or the human. Human plays are logged ("human_play") with the bot's suggestion at
        the estimated tap tick; bot taps unseen for MAX_LAG_TICKS are logged ("bot_tap_unseen")."""
        out: list[PlayAttribution] = []
        with self._lock:
            if tick is not None:
                self._tick = int(tick)
            for p in plays:
                play = as_own_play(p)
                if play is None:
                    continue
                a = self.attributor.classify(play, mode=self._mode, lookup=self._suggestion_at_locked)
                self._account_locked(a)
                out.append(a)
            for tap in self.attributor.expire(self._tick):
                self.counts["bot_unseen"] += 1
                self._push("bot_tap_unseen", {"card": tap.card, "name": self.card_name(tap.card),
                                              "slot": tap.slot, "tap_tick": tap.tick, "tick": self._tick})
        return out

    def note_human_play(self, card: int, tick: int, *, x_mt: float | None = None,
                        y_mt: float | None = None, uid: int | None = None) -> PlayAttribution:
        """Record a play the caller already knows is the human's (no bot tap matched it)."""
        play = OwnPlay(int(card), int(tick), _opt_float(x_mt), _opt_float(y_mt), _opt_int(uid))
        with self._lock:
            self._tick = int(tick)
            a = self.attributor.human(play, mode=self._mode, lookup=self._suggestion_at_locked)
            self._account_locked(a)
            return a

    @property
    def last_human(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._last_human is None else dict(self._last_human)

    def new_battle(self) -> None:
        """A new battle: forget suggestions and pending taps (ticks start again)."""
        with self._lock:
            self.attributor.reset()
            self._history.clear()
            self._tick = None
            self._last_logged = None

    # -- log records ----------------------------------------------------------
    def record(self, kind: str, /, **data: Any) -> tuple[str, dict[str, Any]]:
        """The log record helper: (kind, data) stamped with the mode, generation and tick
        (fields already in `data` win). A "kind" field in data is renamed "data_kind"
        (VisionRecorder.log takes kind as its own argument). Not queued; see log()."""
        if "kind" in data:
            data["data_kind"] = data.pop("kind")
        with self._lock:
            data.setdefault("mode", self._mode)
            data.setdefault("gen", self._gen)
            data.setdefault("tick", self._tick)
            return str(kind), data

    def _push(self, kind: str, data: dict[str, Any]) -> None:
        if len(self._log) == self._log.maxlen:
            self.log_overflow += 1
        self._log.append(self.record(kind, **data))

    def log(self, kind: str, /, **data: Any) -> None:
        """Queue a record for the runner's log (stamped by record())."""
        with self._lock:
            self._push(str(kind), dict(data))

    def drain_log(self) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            out = list(self._log)
            self._log.clear()
            return out

    def flush_to(self, recorder: Any) -> int:
        """Write queued records with recorder.log(kind, **data) (a VisionRecorder); returns the count."""
        recs = self.drain_log()
        for kind, data in recs:
            recorder.log(kind, **data)
        return len(recs)

    def snapshot(self) -> ControlSnapshot:
        with self._lock:
            stopped = self._stopped_locked()
            return ControlSnapshot(self._mode, self._taps_enabled, stopped, self._stop_reason,
                                   self._gen, self._tick,
                                   self._history[-1] if self._history else None,
                                   None if self._last_decision is None else dict(self._last_decision),
                                   None if self._last_human is None else dict(self._last_human),
                                   dict(self.counts))

    # -- inputs ---------------------------------------------------------------
    def _on_control_file(self, present: bool) -> None:
        self.set_mode(MANUAL if present else AUTO, "control file")

    def _on_hotkey(self) -> None:
        self.toggle("hotkey")

    def poll(self) -> str:
        """Check the MANUAL file (when start() set it up) and STOP now; returns the mode.
        For a runner that polls instead of using the watcher thread (start(poll_s=None))."""
        w = self._watcher
        if w is not None:
            w.poll()                              # outside our lock: it calls set_mode()
        with self._lock:
            self._stopped_locked()
            return self._mode

    def start(self, *, hotkey: str | None = DEFAULT_HOTKEY, window: bool = True,
              control_file: bool = True, poll_s: float | None = 0.25,
              platform: str | None = None, env: Mapping[str, str] | None = None,
              hotkey_backend: HotkeyBackend | None = None) -> dict[str, str]:
        """Start the inputs; returns {input: 'on: ...' or 'off: why'}. Each input that
        cannot run here (no Windows, no display) is skipped; the others still start.

        poll_s: how often the watcher thread checks the MANUAL file (None or 0:
        no thread; call poll() from the runner loop). platform / env / hotkey_backend
        are for tests.
        """
        self.stop()
        status: dict[str, str] = {}
        if control_file and self.control_file is not None:
            watcher = ControlFileWatcher(self.control_file, self._on_control_file,
                                         poll_s=float(poll_s or 0.0), exists=self._exists)
            self._watcher = watcher
            watcher.poll()                        # a MANUAL file present now applies at once
            threaded = watcher.start()
            status["control_file"] = (f"on: {self.control_file}"
                                      + ("" if threaded else " (polled by the runner)"))
        else:
            status["control_file"] = "off: disabled"
        listener = HotkeyListener(self._on_hotkey, hotkey, backend=hotkey_backend, platform=platform)
        if listener.start():
            self._hotkey = listener
            status["hotkey"] = f"on: {hotkey}"
        else:
            listener.stop()                       # a thread that timed out must not register later
            status["hotkey"] = f"off: {listener.error}"
        if window:
            win = ControlWindow(self, hotkey_label=hotkey if self._hotkey is not None else None,
                                env=env, platform=platform)
            if win.start():
                self._window = win
                status["window"] = "on"
            else:
                win.stop()
                status["window"] = f"off: {win.error}"
        else:
            status["window"] = "off: disabled"
        with self._lock:
            self.inputs = dict(status)
            self._push("control_start", {"inputs": dict(status), "start_mode": self.start_mode,
                                         "taps_enabled": self._taps_enabled})
        return status

    def stop(self) -> None:
        """Stop the inputs (idempotent). The mode and the records stay."""
        parts, self._window, self._hotkey, self._watcher = (
            (self._window, self._hotkey, self._watcher), None, None, None)
        for part in parts:
            if part is not None:
                try:
                    part.stop()
                except Exception:
                    pass

    def __enter__(self) -> "ControlState":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# command-line options
# ---------------------------------------------------------------------------
def _hotkey_arg(text: str) -> str:
    try:
        parse_hotkey(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return text


def add_control_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """--start-mode, --hotkey, --no-control-window, --manual-file."""
    g = parser.add_argument_group("human takeover (scripts/vision/control.py)")
    g.add_argument("--start-mode", choices=MODES, default=DEFAULT_START_MODE,
                   help="who plays first: the bot (auto) or you (manual, the default). "
                        "The bot taps only with --tap.")
    g.add_argument("--hotkey", type=_hotkey_arg, default=DEFAULT_HOTKEY,
                   help="global hotkey that toggles AUTO/MANUAL, e.g. F8 or ctrl+F9; 'none' turns "
                        "it off. Windows only; other programs do not get this key while the bot runs.")
    g.add_argument("--no-control-window", action="store_true",
                   help="do not open the always-on-top control window")
    g.add_argument("--manual-file", default=str(DEFAULT_CONTROL_FILE),
                   help="creating this file switches to MANUAL, deleting it to AUTO "
                        "('none' turns it off)")
    return parser


def start_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """ControlState.start() keywords from add_control_args() options."""
    return {"hotkey": getattr(args, "hotkey", DEFAULT_HOTKEY),
            "window": not getattr(args, "no_control_window", False)}


_DEMO = (("Hog Rider", (3, 15)), ("Musketeer", (8, 26)), ("Cannon", (9, 24)))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Try the AUTO/MANUAL controls without a battle. It prints every switch, "
                    "shows demo suggestions in the window and never taps.")
    add_control_args(ap)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="run this long (0: until Ctrl+C or the STOP file)")
    args = ap.parse_args(argv)
    ctl = ControlState.from_args(args, taps_enabled=False, stop_file=DEFAULT_STOP_FILE)
    if ctl.stop_requested:
        print(f"refusing: {ctl.stop_file} exists. Delete it to run.")
        return 2
    for name, state in ctl.start(**start_kwargs(args)).items():
        print(f"{name:13s} {state}")
    where = f" or create/delete {ctl.control_file}" if ctl.control_file else ""
    print(f"mode {ctl.mode.upper()}. Switch with {args.hotkey}, the window button{where}. "
          f"Stop: Ctrl+C or create {ctl.stop_file}.")
    t0, tick = now(), 0
    try:
        while not ctl.stop_requested:
            if args.seconds and now() - t0 >= args.seconds:
                break
            tick += 4
            k = (tick // 40) % (len(_DEMO) + 1)
            if k == len(_DEMO):
                s = ctl.suggest(tick=tick, reason="demo")
            else:
                s = ctl.suggest(k, slot=k, tile=_DEMO[k][1], tick=tick, name=_DEMO[k][0])
            ctl.note_decision(s, tapped=False)    # taps_enabled=False: nothing is ever tapped
            ctl.poll()
            for kind, data in ctl.drain_log():
                if kind == "mode":
                    print(f"t={tick:5d} {data['prev'].upper()} -> {data['mode'].upper()} "
                          f"({data['source']})")
                elif kind in ("control_stop", "control_input"):
                    print(f"t={tick:5d} {kind} {data}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        ctl.stop()
    print(f"done; mode {ctl.mode.upper()}, {ctl.counts['switches']} switches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
