"""The VisionFrame contract shared by every perception module.

One VisionFrame per processed screen frame:

    VisionFrame
      t_capture   seconds on the vision clock (`now`, time.perf_counter) when
                  the frame's pixels were captured
      tick        battle tick (20 per second) after filtering, or None
      tracks      [Track]      units on the board, one per tracked unit
      hud         Hud          hand, next card, elixir, clock, towers, crowns
      events      Events       opponent plays and our own landings this frame
      screen      "battle" | "end" | "menu" | "unknown"
      detections  [Detection]  raw detector output (optional, for replay/debug)
      frame_size  (width, height) of the captured frame, or None
      stale_s     how long the pixels had been unchanged when read

Conventions (the same as scripts/vision/geometry.py):
* Positions are RoyaleSim WORLD millitiles; the live bot is always RoyaleSim
  team 1, so world orientation equals the screen (x right, y down).
* Teams are RoyaleSim team ids: OWN_TEAM = 1 (us), ENEMY_TEAM = 0, and
  UNKNOWN_TEAM = -1 while a track's team vote is undecided.
* Card ids are RoyaleSim catalogue ids, as in rs_state players[t]["hand"];
  EMPTY_CARD = -1 is an empty hand slot (between a play and its refill).
* Forms: FORM_BASE 0, FORM_EVO 1, FORM_HERO 2 (tools/vision_classes).
* Tower keys name SCREEN sides, which are world sides: slot 1 of
  players[t]["tower_hp"] is "left", slot 2 is "right", for both teams.

Every type has to_dict()/from_dict() producing plain JSON values; from_dict
ignores unknown keys and fills missing ones with defaults, so old recordings
stay readable when fields are added. Imports nothing beyond the standard
library.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

SCHEMA_VERSION = 1

#: The clock for every vision timestamp (t_capture, tap and landing times):
#: monotonic and sub-millisecond on every platform. time.monotonic() on
#: Windows ticks only every ~15.6 ms before Python 3.13, too coarse for
#: 60 fps capture or latency measurements.
now = time.perf_counter

OWN_TEAM = 1
ENEMY_TEAM = 0
UNKNOWN_TEAM = -1

EMPTY_CARD = -1
HAND_SIZE = 4

FORM_BASE, FORM_EVO, FORM_HERO = 0, 1, 2

# Hud.phase values.
PHASE_UNKNOWN = "unknown"      # clock not read
PHASE_PRE = "pre"              # countdown before the battle starts
PHASE_REGULATION = "regulation"
PHASE_OVERTIME = "overtime"
PHASE_ENDED = "ended"
PHASES = (PHASE_UNKNOWN, PHASE_PRE, PHASE_REGULATION, PHASE_OVERTIME, PHASE_ENDED)

# VisionFrame.screen values.
SCREEN_UNKNOWN = "unknown"
SCREEN_BATTLE = "battle"
SCREEN_END = "end"             # battle over: result banner / end screen
SCREEN_MENU = "menu"           # lobby, clan chat, anything outside a battle
SCREENS = (SCREEN_UNKNOWN, SCREEN_BATTLE, SCREEN_END, SCREEN_MENU)

TOWER_KEYS = ("own_king", "own_left", "own_right", "enemy_king", "enemy_left", "enemy_right")


def _tuple_or_none(v, n: int):
    if v is None:
        return None
    t = tuple(float(c) for c in v)
    if len(t) != n:
        raise ValueError(f"expected {n} numbers, got {len(t)}")
    return t


def _opt_float(v):
    return None if v is None else float(v)


def _opt_int(v):
    return None if v is None else int(v)


@dataclass(slots=True)
class Detection:
    """One detector box in frame pixels.

    side_prob is the probability that the unit is the OPPONENT's (0 = surely
    ours, 1 = surely theirs). ground_px is the pixel point the unit stands on
    (ArenaGeometry.ground_px), None until computed.
    """
    cls: str
    side_prob: float
    conf: float
    box: tuple[float, float, float, float]          # x1, y1, x2, y2
    ground_px: tuple[float, float] | None = None
    form: int = FORM_BASE

    def to_dict(self) -> dict[str, Any]:
        return {"cls": self.cls, "side_prob": self.side_prob, "conf": self.conf,
                "box": list(self.box), "ground_px": None if self.ground_px is None else list(self.ground_px),
                "form": self.form}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Detection":
        return cls(str(d["cls"]), float(d.get("side_prob", 0.5)), float(d.get("conf", 0.0)),
                   _tuple_or_none(d["box"], 4), _tuple_or_none(d.get("ground_px"), 2),
                   int(d.get("form", FORM_BASE)))


@dataclass(slots=True)
class Track:
    """A unit followed across frames.

    cls is the vision class (data/vision/classes.json); team is a RoyaleSim
    team id fixed by vote (OWN_TEAM / ENEMY_TEAM / UNKNOWN_TEAM). x_mt, y_mt
    are world millitiles; hp_frac is the HP-bar fill in [0, 1], or None when
    no bar was read. first_seen_tick gives the deploying flag. vx, vy are a
    velocity estimate in millitiles per tick (0 when unknown).
    """
    uid: int
    cls: str
    team: int
    conf: float
    x_mt: float
    y_mt: float
    hp_frac: float | None = None
    form: int = FORM_BASE
    first_seen_tick: int = 0
    last_seen_tick: int = 0
    vx: float = 0.0
    vy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"uid": self.uid, "cls": self.cls, "team": self.team, "conf": self.conf,
                "x_mt": self.x_mt, "y_mt": self.y_mt, "hp_frac": self.hp_frac, "form": self.form,
                "first_seen_tick": self.first_seen_tick, "last_seen_tick": self.last_seen_tick,
                "vx": self.vx, "vy": self.vy}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Track":
        return cls(int(d["uid"]), str(d["cls"]), int(d.get("team", UNKNOWN_TEAM)),
                   float(d.get("conf", 0.0)), float(d["x_mt"]), float(d["y_mt"]),
                   _opt_float(d.get("hp_frac")), int(d.get("form", FORM_BASE)),
                   int(d.get("first_seen_tick", 0)), int(d.get("last_seen_tick", 0)),
                   float(d.get("vx", 0.0)), float(d.get("vy", 0.0)))


@dataclass(slots=True)
class TowerHp:
    """Absolute tower hit points and maximum; None when not read."""
    hp: int | None = None
    max: int | None = None

    @property
    def frac(self) -> float | None:
        if self.hp is None or not self.max:
            return None
        return max(0.0, min(1.0, self.hp / self.max))

    def to_dict(self) -> dict[str, Any]:
        return {"hp": self.hp, "max": self.max}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "TowerHp":
        d = d or {}
        return cls(_opt_int(d.get("hp")), _opt_int(d.get("max")))


def _empty_hand() -> list[int]:
    return [EMPTY_CARD] * HAND_SIZE


@dataclass(slots=True)
class Hud:
    """What the HUD shows.

    hand: 4 card ids in screen order (EMPTY_CARD for an empty slot); next:
    the next card, or EMPTY_CARD. elixir_milli: 0..10000, None when unread.
    clock_tick: the tick the clock reading implies, None when unread. phase:
    one of PHASES. towers: TOWER_KEYS -> TowerHp. crowns: (own, enemy).
    """
    hand: list[int] = field(default_factory=_empty_hand)
    next: int = EMPTY_CARD
    elixir_milli: int | None = None
    clock_tick: int | None = None
    phase: str = PHASE_UNKNOWN
    towers: dict[str, TowerHp] = field(default_factory=dict)
    crowns: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict[str, Any]:
        return {"hand": list(self.hand), "next": self.next, "elixir_milli": self.elixir_milli,
                "clock_tick": self.clock_tick, "phase": self.phase,
                "towers": {k: v.to_dict() for k, v in self.towers.items()},
                "crowns": list(self.crowns)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "Hud":
        d = d or {}
        crowns = d.get("crowns") or (0, 0)
        return cls([int(c) for c in d.get("hand", _empty_hand())], int(d.get("next", EMPTY_CARD)),
                   _opt_int(d.get("elixir_milli")), _opt_int(d.get("clock_tick")),
                   str(d.get("phase", PHASE_UNKNOWN)),
                   {str(k): TowerHp.from_dict(v) for k, v in (d.get("towers") or {}).items()},
                   (int(crowns[0]), int(crowns[1])))


@dataclass(slots=True)
class OppPlay:
    """An opponent card play derived from the board.

    tick is when it was first seen; exec_tick the estimated execution tick
    (what PublicMemory.record takes). kind is "unit" or "spell"; uids are the
    tracks that make up the play.
    """
    card: int
    tick: int
    exec_tick: int | None = None
    x_mt: float | None = None
    y_mt: float | None = None
    kind: str = "unit"
    uids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"card": self.card, "tick": self.tick, "exec_tick": self.exec_tick,
                "x_mt": self.x_mt, "y_mt": self.y_mt, "kind": self.kind, "uids": list(self.uids)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OppPlay":
        return cls(int(d["card"]), int(d["tick"]), _opt_int(d.get("exec_tick")),
                   _opt_float(d.get("x_mt")), _opt_float(d.get("y_mt")), str(d.get("kind", "unit")),
                   [int(u) for u in d.get("uids", [])])


@dataclass(slots=True)
class OwnLanding:
    """One of our taps matched to the unit it produced (or reported missed).

    tap_tick / tap_t: when the tap was sent (battle tick / monotonic s);
    land_tick: first sighting, None if the play was not seen (a miss);
    aim_*: where we aimed, x_mt/y_mt: where it appeared (world mt).
    """
    card: int
    slot: int
    tap_tick: int
    land_tick: int | None = None
    tap_t: float | None = None
    aim_x_mt: float | None = None
    aim_y_mt: float | None = None
    x_mt: float | None = None
    y_mt: float | None = None
    uid: int | None = None

    @property
    def lag_ticks(self) -> int | None:
        return None if self.land_tick is None else self.land_tick - self.tap_tick

    def to_dict(self) -> dict[str, Any]:
        return {"card": self.card, "slot": self.slot, "tap_tick": self.tap_tick,
                "land_tick": self.land_tick, "tap_t": self.tap_t, "aim_x_mt": self.aim_x_mt,
                "aim_y_mt": self.aim_y_mt, "x_mt": self.x_mt, "y_mt": self.y_mt, "uid": self.uid}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OwnLanding":
        return cls(int(d["card"]), int(d.get("slot", -1)), int(d["tap_tick"]), _opt_int(d.get("land_tick")),
                   _opt_float(d.get("tap_t")), _opt_float(d.get("aim_x_mt")), _opt_float(d.get("aim_y_mt")),
                   _opt_float(d.get("x_mt")), _opt_float(d.get("y_mt")), _opt_int(d.get("uid")))


@dataclass(slots=True)
class Events:
    """Events derived on this frame (not cumulative)."""
    opp_plays: list[OppPlay] = field(default_factory=list)
    own_landings: list[OwnLanding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"opp_plays": [p.to_dict() for p in self.opp_plays],
                "own_landings": [p.to_dict() for p in self.own_landings]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "Events":
        d = d or {}
        return cls([OppPlay.from_dict(p) for p in d.get("opp_plays", [])],
                   [OwnLanding.from_dict(p) for p in d.get("own_landings", [])])


@dataclass(slots=True)
class VisionFrame:
    """Everything perception produced for one captured frame."""
    t_capture: float
    tick: int | None = None
    tracks: list[Track] = field(default_factory=list)
    hud: Hud = field(default_factory=Hud)
    events: Events = field(default_factory=Events)
    screen: str = SCREEN_UNKNOWN
    detections: list[Detection] = field(default_factory=list)
    frame_size: tuple[int, int] | None = None
    stale_s: float = 0.0

    @property
    def in_battle(self) -> bool:
        return self.screen == SCREEN_BATTLE

    def own_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.team == OWN_TEAM]

    def enemy_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.team == ENEMY_TEAM]

    def to_dict(self) -> dict[str, Any]:
        return {"v": SCHEMA_VERSION, "t_capture": self.t_capture, "tick": self.tick,
                "tracks": [t.to_dict() for t in self.tracks], "hud": self.hud.to_dict(),
                "events": self.events.to_dict(), "screen": self.screen,
                "detections": [d.to_dict() for d in self.detections],
                "frame_size": None if self.frame_size is None else list(self.frame_size),
                "stale_s": self.stale_s}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VisionFrame":
        v = int(d.get("v", SCHEMA_VERSION))
        if v > SCHEMA_VERSION:
            raise ValueError(f"VisionFrame schema v{v} is newer than this code (v{SCHEMA_VERSION})")
        size = d.get("frame_size")
        return cls(float(d["t_capture"]), _opt_int(d.get("tick")),
                   [Track.from_dict(t) for t in d.get("tracks", [])], Hud.from_dict(d.get("hud")),
                   Events.from_dict(d.get("events")), str(d.get("screen", SCREEN_UNKNOWN)),
                   [Detection.from_dict(x) for x in d.get("detections", [])],
                   None if size is None else (int(size[0]), int(size[1])),
                   float(d.get("stale_s", 0.0)))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "VisionFrame":
        return cls.from_dict(json.loads(text))
