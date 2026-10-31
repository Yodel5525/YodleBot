"""The tap gate: no AUTO tap until this battle's towers are calibrated and its mode is classic
(skin audit fix list P0-4, P1-12).

Nothing else stops the bot from tapping in a battle it cannot read: the refused-tap guard
(tap_refusal.py) reacts only after 3 taps that did nothing. So every battle starts with the gate
PENDING, and the live runner (scripts/vision_play.py) sends no AUTO tap until it OPENS, which
needs all of:

* THE LABELS: all four princess labels calibrated (tower_calib.TowerCalibrator, P0-1: locked
  from the badge and the outlined bar). A tower whose calibration window ended without a lock
  ("expired") or whose lock was removed ("unlatched"), or one still not locked DEADLINE_S (20) s
  after the battle started, fails it.
* THE BOARD (only with a detector running): a tower body box (tower_labels.BODY_CLASSES) within
  NEAR_TILES (0.5) tile of each of the six tower positions on >= BODY_SHARE (50%) of the
  calibration frames: the frames the detector ran on from the battle's start until the labels
  were calibrated, at least BODY_MIN_FRAMES (3) of them. A position is the bottom centre (the
  ground point) of the median body box of the 2026-10-29 friendly (TOWER_BODY_REF_PX); a box's
  distance is in tiles of the default geometry (51 x 41.4 px at 1080x1920). This checks the
  board's geometry in a new arena or mode, and that the detector sees this battle's towers.
* THE MODE: classic 1v1 (mode.ModeCheck: the first clock reading 2:5x-3:00, the elixir at 1x).

Any of them failing CLOSES the gate for the battle. vision_play then prints one line, e.g.
"towers not calibrated (enemy_left, enemy_right): tapping off this battle; play yourself ...",
and switches AUTO to MANUAL with the mode source "towers" ("game mode" when only the mode
check failed). The owner's own switch back to AUTO (F8, the window, the MANUAL file) after that
overrides the gate for the rest of that battle only (tap_gate_override). A gate that closed while
the owner was in MANUAL already took nothing away, so the owner's next switch to AUTO (the
runbook's routine hand-over) is turned back to MANUAL with the line again (tap_gate_handover),
and only the switch after it overrides. A dry run never switches: it prints "dry run: the tower
gate would keep tapping off this battle (...)" (and what the hand-over would do). The control
window shows "tap gate OFF this battle (...)" / "tap gate overridden this battle" after the mode.

The gate decides once per battle; after it opened, nothing closes it again (a tower that loses
its lock later reads nothing and stays standing, which fails safe). A battle the runner follows
again after it lost it (vision_play: ended without its end screen, then joined in the middle with
the towers kept) is the same battle: resume() keeps its verdict and override.

Records (TapGate.observe returns them; vision_play logs them with the battle number):
  game_mode   the mode verdict (mode.ModeVerdict.to_dict): mode classic / unknown, why, the first
              clock, the elixir fit
  tap_gate    the gate's decision: state open / closed, reasons, the labels' states, the
              detector's body shares, the mode, when (s after the battle start)
summary() is the battle_end record's "tap_gate". Imports numpy only (through mode.py).
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from . import mode as MD
from . import tower_labels as TL
from .geometry import DEFAULT_HOMOGRAPHY_PX

REF_SIZE = TL.REF_SIZE
PRINCESS_KEYS = TL.PRINCESS_KEYS
#: The six towers, as the status line orders them: theirs, then ours.
BODY_KEYS = ("enemy_left", "enemy_king", "enemy_right", "own_left", "own_king", "own_right")
#: px per tile at 1080x1920 (the default geometry: 0.051 px/mt across, 0.0414 down)
TILE_PX = (DEFAULT_HOMOGRAPHY_PX[0][0] * 1000.0, DEFAULT_HOMOGRAPHY_PX[1][1] * 1000.0)
NEAR_TILES = 0.5
BODY_SHARE = 0.5
BODY_MIN_FRAMES = 3
DEADLINE_S = 20.0
#: tower_calib statuses of a tower without a lock whose calibration is over
CALIB_FAILED = ("expired", "unlatched")

PENDING, OPEN, CLOSED = "pending", "open", "closed"
SOURCE_TOWERS, SOURCE_MODE = "towers", "game mode"


def _ground(box: Sequence[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, float(box[3])


#: key -> the ground point (bottom centre, px at 1080x1920) of its reference body box
POSITIONS_PX = {k: _ground(TL.TOWER_BODY_REF_PX[k]) for k in BODY_KEYS}


def body_positions(detections: Iterable[Any], frame_size: Sequence[int] | None, *,
                   near_tiles: float = NEAR_TILES) -> set[str]:
    """The tower keys with a body box (BODY_CLASSES) whose ground point is within `near_tiles`
    of the key's position. detections: objects with .cls and .box (x0, y0, x1, y1 in frame
    pixels of `frame_size` (w, h))."""
    if not frame_size or float(frame_size[0]) <= 0 or float(frame_size[1]) <= 0:
        return set()
    sx, sy = REF_SIZE[0] / float(frame_size[0]), REF_SIZE[1] / float(frame_size[1])
    out: set[str] = set()
    for d in detections or ():
        if str(getattr(d, "cls", "")) not in TL.BODY_CLASSES:
            continue
        b = getattr(d, "box", None)
        if b is None or len(b) < 4:
            continue
        gx, gy = _ground((b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy))
        for key, (px, py) in POSITIONS_PX.items():
            if math.hypot((gx - px) / TILE_PX[0], (gy - py) / TILE_PX[1]) <= near_tiles:
                out.add(key)
    return out


def calib_status(hud_reader: Any) -> dict[str, tuple[str, bool]] | None:
    """princess key -> (tower_calib status, calibrated) of a HudReader that calibrates, else None
    (no calibration: the labels check fails)."""
    cal = getattr(hud_reader, "calib", None)
    if not getattr(hud_reader, "calibrate", False) or cal is None:
        return None
    towers = getattr(cal, "towers", None) or {}
    out = {}
    for key in PRINCESS_KEYS:
        st = towers.get(key)
        if st is None:
            out[key] = ("missing", False)
        else:
            out[key] = (str(getattr(st, "status", "?")), getattr(st, "lock", None) is not None)
    return out


class TowerCheck:
    """The labels and the board of one battle (module doc). start(t) at the battle start; note()
    every battle frame; verdict: None while undecided, else (ok, reasons)."""

    def __init__(self, *, near_tiles: float = NEAR_TILES, body_share: float = BODY_SHARE,
                 body_min_frames: int = BODY_MIN_FRAMES, deadline_s: float = DEADLINE_S):
        self.near_tiles = float(near_tiles)
        self.body_share = float(body_share)
        self.body_min_frames = max(1, int(body_min_frames))
        self.deadline_s = float(deadline_s)
        self.reset()

    def reset(self) -> None:
        self.t0: float | None = None
        self.frames = 0
        self.det_frames = 0
        self.hits = {k: 0 for k in BODY_KEYS}
        self.detector = False
        self.calib: dict[str, tuple[str, bool]] | None = None
        self.verdict: tuple[bool, list[str]] | None = None
        self.t_decided: float | None = None

    def start(self, t: float) -> None:
        self.reset()
        self.t0 = float(t)

    def shares(self) -> dict[str, float | None]:
        if not self.det_frames:
            return {k: None for k in BODY_KEYS}
        return {k: round(self.hits[k] / self.det_frames, 3) for k in BODY_KEYS}

    def note(self, t: float, calib: Mapping[str, tuple[str, bool]] | None, *, detector: bool = False,
             detections: Iterable[Any] | None = None, frame_size: Sequence[int] | None = None
             ) -> tuple[bool, list[str]] | None:
        """One battle frame: the labels' (status, calibrated); detector: one runs (configured);
        detections: this frame's boxes, None when it did not run on this frame."""
        if self.verdict is not None or self.t0 is None:
            return self.verdict
        t = float(t)
        self.frames += 1
        self.detector = self.detector or bool(detector)
        self.calib = None if calib is None else dict(calib)
        if detector and detections is not None:
            self.det_frames += 1
            for key in body_positions(detections, frame_size, near_tiles=self.near_tiles):
                self.hits[key] += 1
        late = t - self.t0 >= self.deadline_s
        if self.calib is None:
            return self._decide(False, ["towers not calibrated (no label calibration)"], t)
        failed = [k for k in PRINCESS_KEYS if not self.calib.get(k, ("missing", False))[1]
                  and self.calib.get(k, ("missing", False))[0] in CALIB_FAILED]
        waiting = [k for k in PRINCESS_KEYS if not self.calib.get(k, ("missing", False))[1]]
        if failed or (waiting and late):
            return self._decide(False, [f"towers not calibrated ({', '.join(failed or waiting)})"], t)
        if waiting:
            return None
        if self.detector:
            if self.det_frames < self.body_min_frames and not late:
                return None
            shares = self.shares()
            low = [k for k in BODY_KEYS if shares[k] is None or shares[k] < self.body_share]
            if low:
                parts = [k if shares[k] is None else f"{k} {round(100 * shares[k])}%" for k in low]
                why = ("the detector boxed no tower" if not self.det_frames
                       else f"towers not boxed by the detector ({', '.join(parts)} of {self.det_frames} frames)")
                return self._decide(False, [why], t)
        return self._decide(True, [], t)

    def _decide(self, ok: bool, reasons: list[str], t: float) -> tuple[bool, list[str]]:
        self.verdict, self.t_decided = (bool(ok), list(reasons)), float(t)
        return self.verdict

    def summary(self) -> dict[str, Any]:
        return {"ok": None if self.verdict is None else self.verdict[0],
                "labels": None if self.calib is None else {k: v[0] for k, v in self.calib.items()},
                "detector": self.detector, "det_frames": self.det_frames, "bodies": self.shares(),
                "decided_s": (None if self.t_decided is None or self.t0 is None
                              else round(self.t_decided - self.t0, 2))}


class TapGate:
    """One battle's tap gate (module doc): new_battle() at each battle start, observe() every
    battle frame, allows() before each AUTO tap, override() on the owner's switch back to AUTO
    after it closed."""

    def __init__(self, *, mode_check: MD.ModeCheck | None = None, tower_check: TowerCheck | None = None):
        self.mode = mode_check if mode_check is not None else MD.ModeCheck()
        self.towers = tower_check if tower_check is not None else TowerCheck()
        self.number: int | None = None
        self._reset_battle()

    def _reset_battle(self) -> None:
        self.state = PENDING
        self.reasons: list[str] = []
        self.overridden: dict[str, Any] | None = None
        self.t0: float | None = None
        self.t_decided: float | None = None
        self.held = 0
        self.resumed = 0
        self._mode_said = False

    def new_battle(self, number: int, first_tick: int | None, t: float) -> None:
        """A new battle: everything is checked again (the override, too, ends here)."""
        self._reset_battle()
        self.number = int(number)
        self.t0 = float(t)
        self.mode.start(first_tick, t)
        self.towers.start(t)

    def resume(self, number: int) -> None:
        """The same battle followed again under a new number: its verdict and override stay."""
        self.number = int(number)
        self.resumed += 1

    @property
    def decided(self) -> bool:
        return self.state != PENDING

    def allows(self) -> bool:
        """May an AUTO tap go out? Open, or closed and overridden by the owner."""
        return self.state == OPEN or (self.state == CLOSED and self.overridden is not None)

    def hold_reason(self) -> str | None:
        """Why an AUTO tap is held now (None: it may go)."""
        if self.allows():
            return None
        if self.state == PENDING:
            return "tap gate: towers and mode not checked yet"
        return "tap gate: " + "; ".join(self.reasons)

    def note_held(self) -> None:
        self.held += 1

    def source(self) -> str:
        """The mode source of the gate's switch to MANUAL."""
        return SOURCE_TOWERS if any(r.startswith(("towers", "the detector")) for r in self.reasons) else SOURCE_MODE

    def reason_text(self) -> str:
        return "; ".join(self.reasons)

    def observe(self, t: float, tick: int | None, *, calib: Mapping[str, tuple[str, bool]] | None,
                detector: bool = False, detections: Iterable[Any] | None = None,
                frame_size: Sequence[int] | None = None,
                elixir: tuple[int | None, float | None] | None = None) -> list[tuple[str, dict[str, Any]]]:
        """One battle frame (module doc); returns the records it decided on this frame:
        ("game_mode", ...) once, ("tap_gate", ...) once."""
        if self.t0 is None:
            return []
        out: list[tuple[str, dict[str, Any]]] = []
        t = float(t)
        if elixir is not None:
            self.mode.note(t, elixir[0], elixir[1])
        verdict = self.mode.update(t, tick)
        if verdict is not None and not self._mode_said:
            self._mode_said = True
            out.append(("game_mode", verdict.to_dict()))
        towers = self.towers.note(t, calib, detector=detector, detections=detections, frame_size=frame_size)
        if self.state != PENDING:
            if self.state == CLOSED:                   # what failed later is recorded too
                extra = [r for r in self._reasons(towers, verdict) if r not in self.reasons]
                self.reasons += extra
            return out
        reasons = self._reasons(towers, verdict)
        if reasons:
            self.state, self.reasons = CLOSED, reasons
        elif towers is not None and towers[0] and verdict is not None and verdict.classic:
            self.state = OPEN
        else:
            return out
        self.t_decided = t
        out.append(("tap_gate", self.record(tick)))
        return out

    @staticmethod
    def _reasons(towers: tuple[bool, list[str]] | None, verdict: MD.ModeVerdict | None) -> list[str]:
        reasons = list(towers[1]) if towers is not None and not towers[0] else []
        if verdict is not None and not verdict.classic:
            reasons.append(verdict.text())
        return reasons

    def override(self, source: str, tick: int | None, t: float) -> None:
        self.overridden = {"source": str(source), "tick": tick, "t": round(float(t), 3)}

    def record(self, tick: int | None = None) -> dict[str, Any]:
        """The tap_gate record of the decision."""
        return {"state": self.state, "reasons": list(self.reasons), "tick": tick,
                "decided_s": None if self.t_decided is None or self.t0 is None else round(self.t_decided - self.t0, 2),
                "towers": self.towers.summary(), "mode": self.mode.summary()}

    def summary(self) -> dict[str, Any]:
        """The battle_end record's "tap_gate"."""
        return {**self.record(), "overridden": None if self.overridden is None else dict(self.overridden),
                "held": self.held, **({"resumed": self.resumed} if self.resumed else {})}

    def status(self) -> str | None:
        """The status line's word: None when open, "gate wait", "gate OFF", "gate overridden"."""
        if self.state == OPEN or self.t0 is None:
            return None
        if self.state == PENDING:
            return "gate wait"
        return "gate overridden" if self.overridden is not None else "gate OFF"
