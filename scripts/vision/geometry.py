"""Arena geometry: screen pixels <-> RoyaleSim world millitiles, and tap targets.

Coordinate frames
-----------------
world   RoyaleSim's engine frame in millitiles (1 tile = 1000 mt; RoyaleSim
        sub-tiles are mt * 18). x runs 0..18000 from screen left to right,
        y runs 0..32000 from the top of the screen to the bottom. The live bot
        is always RoyaleSim team 1, the seat that defends high y, so the world
        frame has the same orientation as the screen and nothing is flipped
        between them.
screen  pixels of a 9:16 portrait frame at any resolution. Internally a point
        is normalised to (u, v) = (px / width, py / height), so one geometry
        serves the 1080x1920 framebuffer (adb, taps), a 560x996 PrintWindow
        grab and a 540x960 device alike.
own     the model's frame, tools/rs_rows.own() / tools/rs_eval.to_world() for
        team 1: ox = 18000 - x, oy = y (a mirror in x). A model tile (tx, ty)
        is an own-frame tile; its screen tile is (17 - tx, ty), which is
        live_play.screen_tile().
tiles   tile (sx, sy) covers [sx, sx + 1) x [sy, sy + 1) tiles; its centre,
        where a tap goes, is (sx + 0.5, sy + 0.5).

The map from world to normalised screen is a homography fitted by DLT on four
or more landmark pairs (tower centres, the river banks at the bridges and at
the arena's side edges; LANDMARKS_TILES). The default file
data/vision/geometry_default_1080x1920.json reproduces the author's linear fit
(scripts/brain/arena.py:38-40, scripts/gamestate/live_play.py:52-55):
tile index t -> (103.5 + 51 t, 171.9 + 41.4 t) px at the tile centre. That fit
was measured on another MuMu install; refit it for this PC:

    python -m scripts.vision.geometry template --size 1080x1920 --out points.json
    (edit the px of each landmark from a screenshot of a friendly battle)
    python -m scripts.vision.geometry fit --points points.json --out data/vision/geometry_pc.json
    python -m scripts.vision.geometry overlay --image shot.png --geometry data/vision/geometry_pc.json --out check.png

The most precise landmarks are the corners of the deploy grid the game shows
while a card is held; give those as explicit "world_tiles" points.

The geometry also carries the card-slot centres, the arena tiles covered by
on-screen ability buttons (royalegym action.py UI_BUTTON_TILES; a card tapped
there lands on the button and is not played), and per-class ground offsets
(a flying sprite is drawn above its ground position).

`check_frame_size` refuses any frame that is not 9:16 portrait (a 1080x2400
phone preset, a landscape tablet preset): every position and tap would be off.

Imports numpy only. The `overlay` command imports Pillow lazily.
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__" and not __package__:
    # Run as a file, sys.path[0] is scripts/vision, whose types.py would shadow
    # the standard library's `types`: put the repo root there instead.
    _here = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if sys.path and os.path.normcase(os.path.abspath(sys.path[0] or os.curdir)) == _here:
        sys.path[0] = _root
    else:
        sys.path.insert(0, _root)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterable, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEOMETRY_PATH = ROOT / "data" / "vision" / "geometry_default_1080x1920.json"
SCHEMA = "hastycr.vision.geometry/1"

# ---------------------------------------------------------------------------
# Board constants (RoyaleSim data/derived/arena.json, the ~2018 default arena)
# ---------------------------------------------------------------------------
TILES_X, TILES_Y = 18, 32
MT = 1000                                   # millitiles per tile
WIDTH_MT, HEIGHT_MT = TILES_X * MT, TILES_Y * MT
OWN_TEAM = 1                                # the live bot is always RoyaleSim team 1
ASPECT = 9 / 16                             # width / height of every accepted frame

# River: water half-rows 30-33, i.e. y in [15, 17) tiles. Bridges: half-columns
# 5-8 and 27-30, i.e. x in [2.5, 4.5) and [13.5, 15.5) tiles.
RIVER_Y_MT = (15_000, 17_000)
BRIDGES_X_MT = ((2_500, 4_500), (13_500, 15_500))

# Ground landmarks in world TILES. Kings are arena.json king_blocks centres.
# Princess centres follow royalegym's fallback (x = bridge centre, y = 6.5
# tiles from the back edge), which royalegym marks as a community figure.
# Bridge points are the deck centre line where it meets each river bank, so
# they do not depend on the bridge width the live client draws.
LANDMARKS_TILES: dict[str, tuple[float, float]] = {
    "enemy_king": (9.0, 3.0),
    "own_king": (9.0, 29.0),
    "enemy_left_princess": (3.5, 6.5),
    "enemy_right_princess": (14.5, 6.5),
    "own_left_princess": (3.5, 25.5),
    "own_right_princess": (14.5, 25.5),
    "left_bridge_top": (3.5, 15.0),
    "left_bridge_bottom": (3.5, 17.0),
    "right_bridge_top": (14.5, 15.0),
    "right_bridge_bottom": (14.5, 17.0),
    "river_top_left_edge": (0.0, 15.0),
    "river_top_right_edge": (18.0, 15.0),
    "river_bottom_left_edge": (0.0, 17.0),
    "river_bottom_right_edge": (18.0, 17.0),
}

# ---------------------------------------------------------------------------
# Built-in defaults, identical to the default JSON file (used if it is missing)
# ---------------------------------------------------------------------------
DEFAULT_FRAME_SIZE = (1080, 1920)
# The author's fit, tile index t -> 103.5 + 51 t (x), 171.9 + 41.4 t (y), at the
# tile centre (t + 0.5 tiles). As world mt -> px: 78 + 0.051 x, 151.2 + 0.0414 y.
DEFAULT_HOMOGRAPHY_PX = ((0.051, 0.0, 78.0), (0.0, 0.0414, 151.2), (0.0, 0.0, 1.0))
# Card-slot centres at 1080x1920 (live_play.py:52, from mumu_overnight_bot).
DEFAULT_CARD_SLOTS_PX = ((334, 1711), (538, 1711), (742, 1711), (946, 1711))
# Screen (= world) tiles each ability button covers: royalegym action.py
# UI_BUTTON_TILES (own frame of Red, world = (17 - gx, 31 - gy)), measured there
# on a 1080x1920 screen. "right"/"left" are sides of the screen.
DEFAULT_ABILITY_BUTTONS: dict[str, tuple[tuple[int, int], ...]] = {
    "right": ((16, 30), (16, 31), (17, 30), (17, 31)),
    "left": ((0, 30), (0, 31), (1, 30), (1, 31), (2, 31)),
}
# Where the buttons sit by how many there are (royalegym UI_BUTTON_PLACES). With
# three or more the places are unmeasured and no tile is masked.
DEFAULT_ABILITY_BUTTON_PLACES: dict[int, tuple[str, ...]] = {1: ("right",), 2: ("left", "right")}


class GeometryError(ValueError):
    """A geometry that cannot be built or used (bad fit, bad file, bad input)."""


class FrameSizeError(GeometryError):
    """A frame that is not 9:16 portrait, or too small to read."""


# ---------------------------------------------------------------------------
# Frame size check
# ---------------------------------------------------------------------------
_CHECKED_SIZES: set[tuple[int, int, float, int]] = set()


def check_frame_size(width: int, height: int, *, tol: float = 0.01,
                     min_width: int = 200) -> tuple[int, int]:
    """Return (width, height) as ints, or raise FrameSizeError.

    Accepts portrait frames whose width/height is within `tol` (relative) of
    9/16: 1080x1920, 720x1280, 540x960 and MuMu's 560x996 window all pass.
    1080x2400 (9:20), landscape and square frames are refused, as is anything
    narrower than `min_width` (a thumbnail, not a device).
    """
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError) as exc:
        raise FrameSizeError(f"frame size must be two integers, got {width!r}x{height!r}") from exc
    key = (w, h, float(tol), int(min_width))
    if key in _CHECKED_SIZES:
        return w, h
    if w <= 0 or h <= 0:
        raise FrameSizeError(f"frame {w}x{h} has no area")
    if w >= h:
        raise FrameSizeError(
            f"frame {w}x{h} is not portrait; set the emulator to a 9:16 phone "
            "display (1080x1920) in portrait")
    if abs(w / h - ASPECT) / ASPECT > tol:
        raise FrameSizeError(
            f"frame {w}x{h} is not 9:16 (ratio {w / h:.4f}, need {ASPECT:.4f}); every "
            "screen position assumes 9:16 portrait. Set MuMu's resolution to "
            "1080x1920 (Settings > Display > Phone).")
    if w < min_width:
        raise FrameSizeError(f"frame {w}x{h} is narrower than {min_width} px")
    _CHECKED_SIZES.add(key)
    return w, h


def frame_size_of(image: np.ndarray) -> tuple[int, int]:
    """(width, height) of an HxW[xC] array."""
    return int(image.shape[1]), int(image.shape[0])


def parse_size(text: str) -> tuple[int, int]:
    """'1080x1920' -> (1080, 1920)."""
    try:
        w, h = text.lower().replace(" ", "").split("x")
        return int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"size must look like 1080x1920, not {text!r}") from exc


# ---------------------------------------------------------------------------
# Homography fitting (DLT with Hartley normalisation)
# ---------------------------------------------------------------------------
def _project(H: np.ndarray, x, y):
    """Apply a 3x3 homography to scalars or arrays; returns the same kind."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    den = H[2, 0] * xa + H[2, 1] * ya + H[2, 2]
    u = (H[0, 0] * xa + H[0, 1] * ya + H[0, 2]) / den
    v = (H[1, 0] * xa + H[1, 1] * ya + H[1, 2]) / den
    if u.ndim == 0:
        return float(u), float(v)
    return u, v


def project_points(H: np.ndarray, points) -> np.ndarray:
    """(N, 2) points through H -> (N, 2) array."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    u, v = _project(np.asarray(H, dtype=np.float64), p[:, 0], p[:, 1])
    return np.stack([np.atleast_1d(u), np.atleast_1d(v)], axis=1)


def _normaliser(points: np.ndarray) -> np.ndarray:
    centre = points.mean(axis=0)
    spread = float(np.sqrt(((points - centre) ** 2).sum(axis=1)).mean())
    s = math.sqrt(2.0) / spread if spread > 0 else 1.0
    return np.array([[s, 0.0, -s * centre[0]], [0.0, s, -s * centre[1]], [0.0, 0.0, 1.0]])


def fit_homography(src, dst) -> np.ndarray:
    """3x3 H with dst ~ H @ src, by DLT on >= 4 pairs; H[2, 2] == 1.

    Raises GeometryError for fewer than 4 pairs, non-finite input or a
    degenerate configuration (repeated or collinear points).
    """
    s = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(s) != len(d):
        raise GeometryError(f"{len(s)} source points but {len(d)} destination points")
    if len(s) < 4:
        raise GeometryError(f"a homography needs at least 4 point pairs, got {len(s)}")
    if not (np.isfinite(s).all() and np.isfinite(d).all()):
        raise GeometryError("landmark coordinates must be finite")
    Ts, Td = _normaliser(s), _normaliser(d)
    sn, dn = project_points(Ts, s), project_points(Td, d)
    A = np.zeros((2 * len(s), 9))
    for i, ((x, y), (u, v)) in enumerate(zip(sn, dn)):
        A[2 * i] = (-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u)
        A[2 * i + 1] = (0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v)
    _, sv, vt = np.linalg.svd(A)
    # A unique solution needs rank 8: the 8th singular value must be well above 0.
    if len(sv) < 8 or sv[7] <= 1e-9 * sv[0]:
        raise GeometryError("degenerate landmarks (repeated or collinear points)")
    H = np.linalg.inv(Td) @ vt[-1].reshape(3, 3) @ Ts
    if abs(H[2, 2]) < 1e-12 * np.abs(H).max():
        raise GeometryError("fitted homography maps the origin to infinity")
    return H / H[2, 2]


def reprojection_errors(H: np.ndarray, src, dst) -> np.ndarray:
    """Euclidean distance |H(src) - dst| per pair, in dst units."""
    pred = project_points(H, src)
    return np.sqrt(((pred - np.asarray(dst, dtype=np.float64).reshape(-1, 2)) ** 2).sum(axis=1))


def leave_one_out_errors(src, dst) -> list[float | None]:
    """Held-out error per pair: fit without it, measure it. None when < 5 pairs."""
    s = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    out: list[float | None] = []
    for i in range(len(s)):
        keep = np.arange(len(s)) != i
        if keep.sum() < 4:
            out.append(None)
            continue
        try:
            H = fit_homography(s[keep], d[keep])
        except GeometryError:
            out.append(None)
            continue
        out.append(float(reprojection_errors(H, s[i:i + 1], d[i:i + 1])[0]))
    return out


# ---------------------------------------------------------------------------
# Small pieces carried by the geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GroundOffset:
    """Where on a detection box a unit stands, and a world-space correction.

    The anchor is the box's horizontal centre at `anchor_y` of its height
    (0 = top edge, 1 = bottom edge); (dx_mt, dy_mt) is then added in world
    millitiles (+y is further down the screen).
    """
    anchor_y: float = 1.0
    dx_mt: float = 0.0
    dy_mt: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"anchor_y": self.anchor_y, "dx_mt": self.dx_mt, "dy_mt": self.dy_mt}
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GroundOffset":
        return cls(float(d.get("anchor_y", 1.0)), float(d.get("dx_mt", 0.0)),
                   float(d.get("dy_mt", 0.0)), str(d.get("note", "")))


# Box bottom-centre for ground units, as scripts/brain/vision.py:300-302 does.
# The flyer offset is an UNMEASURED prior (a flying sprite is drawn above its
# shadow); calibrate it by playing a flyer on known tiles in a friendly.
DEFAULT_GROUND = GroundOffset(1.0, 0.0, 0.0, "box bottom-centre (vision.py convention); uncalibrated")
DEFAULT_FLYING = GroundOffset(1.0, 0.0, 600.0, "unmeasured prior: flyers drawn ~0.6 tile above ground")


@dataclass(frozen=True)
class GroundOffsets:
    default: GroundOffset = DEFAULT_GROUND
    flying: GroundOffset = DEFAULT_FLYING
    classes: Mapping[str, GroundOffset] = field(default_factory=dict)

    def lookup(self, cls: str | None, flying: bool = False) -> GroundOffset:
        if cls is not None and cls in self.classes:
            return self.classes[cls]
        return self.flying if flying else self.default

    def to_dict(self) -> dict[str, Any]:
        return {"default": self.default.to_dict(), "flying": self.flying.to_dict(),
                "classes": {k: v.to_dict() for k, v in sorted(self.classes.items())}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "GroundOffsets":
        if not d:
            return cls()
        return cls(GroundOffset.from_dict(d.get("default", {})),
                   GroundOffset.from_dict(d.get("flying", DEFAULT_FLYING.to_dict())),
                   {str(k): GroundOffset.from_dict(v) for k, v in (d.get("classes") or {}).items()})


@dataclass(frozen=True)
class Landmark:
    """One correspondence: a world point (millitiles) and where it is on screen."""
    name: str
    world_mt: tuple[float, float]
    px: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "world_tiles": [self.world_mt[0] / MT, self.world_mt[1] / MT],
                "px": [self.px[0], self.px[1]]}


def landmark_world_mt(name: str, world_tiles: Sequence[float] | None = None) -> tuple[float, float]:
    """World position of a landmark: explicit tiles, else LANDMARKS_TILES[name]."""
    if world_tiles is not None:
        x, y = world_tiles
        return float(x) * MT, float(y) * MT
    if name not in LANDMARKS_TILES:
        raise GeometryError(f"unknown landmark {name!r}; give its world_tiles or use one of "
                            + ", ".join(LANDMARKS_TILES))
    x, y = LANDMARKS_TILES[name]
    return x * MT, y * MT


def on_water(x_mt: float, y_mt: float) -> bool:
    """True if a world point is river water (not a bridge) on the default arena."""
    if not RIVER_Y_MT[0] <= y_mt < RIVER_Y_MT[1]:
        return False
    return not any(lo <= x_mt < hi for lo, hi in BRIDGES_X_MT)


def in_arena(x_mt: float, y_mt: float, margin_mt: float = 0.0) -> bool:
    """True if a world point lies on the 18x32 board (grown by `margin_mt`)."""
    return (-margin_mt <= x_mt <= WIDTH_MT + margin_mt
            and -margin_mt <= y_mt <= HEIGHT_MT + margin_mt)


# ---------------------------------------------------------------------------
# The geometry
# ---------------------------------------------------------------------------
class ArenaGeometry:
    """World millitiles <-> screen pixels for one device, plus tap targets.

    Built from a 3x3 homography `homography_px` that maps world millitiles to
    pixels of a reference frame `frame_size`; every method takes an optional
    `size` (width, height) of the frame or device it is used with, and scales
    through normalised coordinates. Instances are immutable.
    """

    def __init__(self, homography_px, frame_size: Sequence[int] = DEFAULT_FRAME_SIZE, *,
                 card_slots_px: Iterable[Sequence[float]] = DEFAULT_CARD_SLOTS_PX,
                 ability_buttons: Mapping[str, Iterable[Sequence[int]]] | None = None,
                 ability_button_places: Mapping[int, Sequence[str]] | None = None,
                 ground: GroundOffsets | None = None,
                 landmarks: Iterable[Landmark] = (),
                 name: str = "default", meta: Mapping[str, Any] | None = None):
        w, h = check_frame_size(*frame_size)
        H = np.asarray(homography_px, dtype=np.float64)
        if H.shape != (3, 3) or not np.isfinite(H).all():
            raise GeometryError("homography_px must be a finite 3x3 matrix")
        if abs(H[2, 2]) < 1e-12:
            raise GeometryError("homography_px[2][2] must not be 0")
        H = H / H[2, 2]
        self.frame_size = (w, h)
        self.homography_px = H
        self._H = np.diag([1.0 / w, 1.0 / h, 1.0]) @ H          # world -> normalised
        try:
            self._Hinv = np.linalg.inv(self._H)
        except np.linalg.LinAlgError as exc:
            raise GeometryError("homography is singular") from exc
        self._check_orientation()
        slots = [tuple(float(c) for c in p) for p in card_slots_px]
        if len(slots) != 4:
            raise GeometryError(f"need 4 card-slot centres, got {len(slots)}")
        self.card_slots_norm = tuple((x / w, y / h) for x, y in slots)
        buttons = DEFAULT_ABILITY_BUTTONS if ability_buttons is None else ability_buttons
        self.ability_buttons: dict[str, frozenset[tuple[int, int]]] = {}
        for place, tiles in buttons.items():
            cells = frozenset((int(t[0]), int(t[1])) for t in tiles)
            for sx, sy in cells:
                if not (0 <= sx < TILES_X and 0 <= sy < TILES_Y):
                    raise GeometryError(f"ability button {place!r} tile {(sx, sy)} is off the board")
            self.ability_buttons[str(place)] = cells
        places = DEFAULT_ABILITY_BUTTON_PLACES if ability_button_places is None else ability_button_places
        self.ability_button_places = {int(n): tuple(str(p) for p in ps) for n, ps in places.items()}
        for n, ps in self.ability_button_places.items():
            missing = [p for p in ps if p not in self.ability_buttons]
            if missing:
                raise GeometryError(f"ability_button_places[{n}] names unknown places {missing}")
        self.ground = ground if ground is not None else GroundOffsets()
        self.landmarks = tuple(landmarks)
        self.name = str(name)
        self.meta = dict(meta or {})

    # -- sizes ---------------------------------------------------------------
    def _size(self, size: Sequence[int] | None) -> tuple[int, int]:
        if size is None:
            return self.frame_size
        return check_frame_size(size[0], size[1])

    def _check_orientation(self) -> None:
        """Refuse a mirrored or wildly misplaced map (e.g. swapped landmarks)."""
        xs = np.array([0.0, WIDTH_MT, 0.0, WIDTH_MT])
        ys = np.array([0.0, 0.0, HEIGHT_MT, HEIGHT_MT])
        den = self._H[2, 0] * xs + self._H[2, 1] * ys + self._H[2, 2]
        if not (np.all(den > 0) or np.all(den < 0)):
            raise GeometryError("homography puts part of the arena behind the camera")
        u, v = _project(self._H, xs, ys)
        if not (np.all(np.abs(u - 0.5) < 1.5) and np.all(np.abs(v - 0.5) < 1.5)):
            raise GeometryError("homography puts the arena far outside the frame")
        # world x grows to the right and world y grows downwards on screen
        if not (u[1] > u[0] and u[3] > u[2] and v[2] > v[0] and v[3] > v[1]):
            raise GeometryError(
                "homography is mirrored: world x must grow to the right and y downwards "
                "(were left/right or own/enemy landmarks swapped?)")

    # -- world <-> screen ----------------------------------------------------
    def world_to_norm(self, x_mt, y_mt):
        """World millitiles -> normalised screen (u, v) in [0, 1]."""
        return _project(self._H, x_mt, y_mt)

    def norm_to_world(self, u, v):
        """Normalised screen (u, v) -> world millitiles."""
        return _project(self._Hinv, u, v)

    def world_to_px(self, x_mt, y_mt, size: Sequence[int] | None = None):
        """World millitiles -> pixels of a frame of `size` (floats; arrays ok)."""
        w, h = self._size(size)
        u, v = _project(self._H, x_mt, y_mt)
        return u * w, v * h

    def px_to_world(self, px, py, size: Sequence[int] | None = None):
        """Pixels of a frame of `size` -> world millitiles (floats; arrays ok)."""
        w, h = self._size(size)
        return _project(self._Hinv, np.asarray(px, dtype=np.float64) / w,
                        np.asarray(py, dtype=np.float64) / h)

    def tile_center_px(self, sx: int, sy: int, size: Sequence[int] | None = None):
        """Centre of screen (= world) tile (sx, sy) in pixels (floats)."""
        return self.world_to_px(sx * MT + MT / 2, sy * MT + MT / 2, size)

    # -- own (model) frame and taps -------------------------------------------
    @staticmethod
    def own_to_world(ox_mt, oy_mt):
        """Own frame (rs_rows.own, team 1) -> world: (18000 - ox, oy)."""
        return WIDTH_MT - ox_mt, oy_mt

    @staticmethod
    def world_to_own(x_mt, y_mt):
        """World -> own frame (rs_rows.own, team 1): (18000 - x, y)."""
        return WIDTH_MT - x_mt, y_mt

    @staticmethod
    def model_tile_to_screen_tile(tx: int, ty: int) -> tuple[int, int]:
        """Model (own-frame) tile -> screen tile: (17 - tx, ty) (live_play.screen_tile)."""
        return TILES_X - 1 - int(tx), int(ty)

    @staticmethod
    def model_tile_to_world(tx: int, ty: int) -> tuple[int, int]:
        """Centre of a model tile in world mt; equals rs_eval.to_world(tx*1000+500, ty*1000+500, 1)."""
        return WIDTH_MT - (int(tx) * MT + MT // 2), int(ty) * MT + MT // 2

    def tap_px(self, tx: int, ty: int, size: Sequence[int] | None = None) -> tuple[int, int]:
        """Device pixel to tap for a model tile (tx, ty) (own frame, tile head index
        t = ty * 18 + tx). `size` is the DEVICE resolution taps are sent in
        (adb `input tap`), not the size of the captured frame."""
        if not (0 <= int(tx) < TILES_X and 0 <= int(ty) < TILES_Y):
            raise GeometryError(f"model tile {(tx, ty)} is off the 18x32 board")
        x, y = self.model_tile_to_world(tx, ty)
        return self.world_px_int(x, y, size)

    def world_px_int(self, x_mt: float, y_mt: float, size: Sequence[int] | None = None) -> tuple[int, int]:
        """A world point as integer device pixels, clipped to the frame."""
        w, h = self._size(size)
        px, py = self.world_to_px(x_mt, y_mt, (w, h))
        return min(w - 1, max(0, int(round(px)))), min(h - 1, max(0, int(round(py))))

    def card_slot_px(self, slot: int, size: Sequence[int] | None = None) -> tuple[int, int]:
        """Device pixel of hand slot 0-3 (screen order, left to right)."""
        if not 0 <= int(slot) < 4:
            raise GeometryError(f"card slot must be 0-3, got {slot}")
        w, h = self._size(size)
        u, v = self.card_slots_norm[int(slot)]
        return int(round(u * w)), int(round(v * h))

    # -- ability buttons -------------------------------------------------------
    def ability_button_screen_tiles(self, n_buttons: int) -> frozenset[tuple[int, int]]:
        """Screen tiles covered when `n_buttons` ability buttons are shown."""
        out: set[tuple[int, int]] = set()
        for place in self.ability_button_places.get(int(n_buttons), ()):
            out |= self.ability_buttons[place]
        return frozenset(out)

    def ability_button_mask(self, n_buttons: int, frame: str = "model") -> np.ndarray:
        """bool[32, 18] of tiles a card tap cannot use while buttons are shown.

        frame="model" indexes [ty, tx] in the own frame, so `mask.reshape(-1)`
        lines up with the 576-way tile head (t = ty * 18 + tx);
        frame="screen" indexes [sy, sx].
        """
        if frame not in ("model", "screen"):
            raise GeometryError(f"frame must be 'model' or 'screen', not {frame!r}")
        mask = np.zeros((TILES_Y, TILES_X), dtype=bool)
        for sx, sy in self.ability_button_screen_tiles(n_buttons):
            mask[sy, sx if frame == "screen" else TILES_X - 1 - sx] = True
        return mask

    def ability_button_rect_px(self, place: str, size: Sequence[int] | None = None) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) pixel box around a button's tiles (derived, not measured)."""
        if place not in self.ability_buttons:
            raise GeometryError(f"unknown ability button place {place!r}")
        w, h = self._size(size)
        tiles = self.ability_buttons[place]
        x0 = min(t[0] for t in tiles) * MT
        x1 = (max(t[0] for t in tiles) + 1) * MT
        y0 = min(t[1] for t in tiles) * MT
        y1 = (max(t[1] for t in tiles) + 1) * MT
        px, py = self.world_to_px(np.array([x0, x1, x0, x1], float), np.array([y0, y0, y1, y1], float), (w, h))
        return (max(0, int(math.floor(px.min()))), max(0, int(math.floor(py.min()))),
                min(w, int(math.ceil(px.max()))), min(h, int(math.ceil(py.max()))))

    def ability_button_px(self, place: str, size: Sequence[int] | None = None) -> tuple[int, int]:
        """Centre of a button's box: where to tap to press it (derived)."""
        x0, y0, x1, y1 = self.ability_button_rect_px(place, size)
        return (x0 + x1) // 2, (y0 + y1) // 2

    # -- detections -------------------------------------------------------------
    def ground_point(self, box: Sequence[float], size: Sequence[int] | None = None,
                     cls: str | None = None, flying: bool = False) -> tuple[float, float]:
        """World millitiles where the unit in pixel box (x1, y1, x2, y2) stands."""
        x1, y1, x2, y2 = (float(c) for c in box)
        off = self.ground.lookup(cls, flying)
        wx, wy = self.px_to_world((x1 + x2) / 2.0, y1 + off.anchor_y * (y2 - y1), size)
        return wx + off.dx_mt, wy + off.dy_mt

    def ground_px(self, box: Sequence[float], size: Sequence[int] | None = None,
                  cls: str | None = None, flying: bool = False) -> tuple[float, float]:
        """The ground point of `box` in pixels of the same frame."""
        x, y = self.ground_point(box, size, cls, flying)
        return self.world_to_px(x, y, size)

    def arena_rect_px(self, size: Sequence[int] | None = None, margin_mt: float = 0.0) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) pixel box around the board grown by `margin_mt`,
        clipped to the frame: the crop to run the detector on."""
        w, h = self._size(size)
        m = float(margin_mt)
        xs = np.array([-m, WIDTH_MT + m, -m, WIDTH_MT + m])
        ys = np.array([-m, -m, HEIGHT_MT + m, HEIGHT_MT + m])
        px, py = self.world_to_px(xs, ys, (w, h))
        return (max(0, int(math.floor(px.min()))), max(0, int(math.floor(py.min()))),
                min(w, int(math.ceil(px.max()))), min(h, int(math.ceil(py.max()))))

    # -- fit and serialisation -------------------------------------------------------
    @classmethod
    def fit(cls, world_mt, px, frame_size: Sequence[int], *, names: Sequence[str] | None = None,
            base: "ArenaGeometry | None" = None, name: str = "fitted",
            max_rms_px: float | None = None) -> "ArenaGeometry":
        """Fit a geometry to landmark pairs (world millitiles -> pixels at
        frame_size). Card slots, buttons and ground offsets come from `base`
        (default: the default geometry), scaled to this frame size. The RMS
        and leave-one-out errors (px) are kept in `meta`."""
        w, h = check_frame_size(*frame_size)
        world = np.asarray(world_mt, dtype=np.float64).reshape(-1, 2)
        pix = np.asarray(px, dtype=np.float64).reshape(-1, 2)
        H = fit_homography(world, pix)
        errors = reprojection_errors(H, world, pix)
        rms = float(np.sqrt((errors ** 2).mean()))
        if max_rms_px is not None and rms > max_rms_px:
            raise GeometryError(f"fit RMS {rms:.2f} px exceeds {max_rms_px} px")
        base = base if base is not None else cls.default()
        bw, bh = base.frame_size
        names = list(names) if names is not None else [f"p{i}" for i in range(len(world))]
        marks = [Landmark(n, (float(a), float(b)), (float(c), float(d)))
                 for n, (a, b), (c, d) in zip(names, world, pix)]
        meta = {"rms_px": rms, "errors_px": [float(e) for e in errors],
                "loo_px": leave_one_out_errors(world, pix), "points": len(world)}
        return cls(H, (w, h),
                   card_slots_px=[(u * w, v * h) for u, v in base.card_slots_norm],
                   ability_buttons={p: sorted(t) for p, t in base.ability_buttons.items()},
                   ability_button_places=base.ability_button_places,
                   ground=base.ground, landmarks=marks, name=name, meta=meta)

    def to_dict(self) -> dict[str, Any]:
        w, h = self.frame_size
        return {
            "schema": SCHEMA,
            "name": self.name,
            "frame_size": [w, h],
            "homography_px": [[float(c) for c in row] for row in self.homography_px],
            "card_slots_px": [[u * w, v * h] for u, v in self.card_slots_norm],
            "ability_buttons": {p: {"screen_tiles": [list(t) for t in sorted(tiles)]}
                                for p, tiles in sorted(self.ability_buttons.items())},
            "ability_button_places": {str(n): list(ps) for n, ps in sorted(self.ability_button_places.items())},
            "ground_offsets": self.ground.to_dict(),
            "landmarks": [m.to_dict() for m in self.landmarks],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ArenaGeometry":
        schema = d.get("schema", SCHEMA)
        if not str(schema).startswith("hastycr.vision.geometry/"):
            raise GeometryError(f"not a geometry file (schema {schema!r})")
        frame_size = tuple(d.get("frame_size", DEFAULT_FRAME_SIZE))
        marks = [Landmark(str(m.get("name", f"p{i}")),
                          landmark_world_mt(str(m.get("name", "")), m.get("world_tiles")),
                          (float(m["px"][0]), float(m["px"][1])))
                 for i, m in enumerate(d.get("landmarks") or [])]
        buttons = None
        if "ability_buttons" in d:
            buttons = {p: [tuple(t) for t in v["screen_tiles"]] for p, v in d["ability_buttons"].items()}
        places = None
        if "ability_button_places" in d:
            places = {int(n): tuple(ps) for n, ps in d["ability_button_places"].items()}
        common = dict(card_slots_px=d.get("card_slots_px", DEFAULT_CARD_SLOTS_PX),
                      ability_buttons=buttons, ability_button_places=places,
                      ground=GroundOffsets.from_dict(d.get("ground_offsets")),
                      landmarks=marks, name=str(d.get("name", "")), meta=d.get("meta") or {})
        if "homography_px" in d:
            return cls(d["homography_px"], frame_size, **common)
        if len(marks) < 4:
            raise GeometryError("geometry file has neither homography_px nor >= 4 landmarks")
        H = fit_homography([m.world_mt for m in marks], [m.px for m in marks])
        return cls(H, frame_size, **common)

    def save(self, path: str | os.PathLike) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ArenaGeometry":
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GeometryError(f"cannot read geometry {p}: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def builtin(cls) -> "ArenaGeometry":
        """The author's fit from the constants in this module (no file)."""
        return cls(DEFAULT_HOMOGRAPHY_PX, DEFAULT_FRAME_SIZE, name="builtin_default",
                   landmarks=default_landmarks(DEFAULT_HOMOGRAPHY_PX))

    _DEFAULT: "ArenaGeometry | None" = None

    @classmethod
    def default(cls) -> "ArenaGeometry":
        """data/vision/geometry_default_1080x1920.json, else `builtin()`."""
        if cls._DEFAULT is None:
            cls._DEFAULT = cls.load(DEFAULT_GEOMETRY_PATH) if DEFAULT_GEOMETRY_PATH.exists() else cls.builtin()
        return cls._DEFAULT

    def __repr__(self) -> str:
        return f"ArenaGeometry({self.name!r}, frame_size={self.frame_size})"


def default_landmarks(homography_px=DEFAULT_HOMOGRAPHY_PX) -> list[Landmark]:
    """LANDMARKS_TILES with pixels from a homography (synthetic, not clicked)."""
    H = np.asarray(homography_px, dtype=np.float64)
    out = []
    for name, (x, y) in LANDMARKS_TILES.items():
        px, py = _project(H, x * MT, y * MT)
        out.append(Landmark(name, (x * MT, y * MT), (px, py)))
    return out


def load_geometry(path: str | os.PathLike | None = None) -> ArenaGeometry:
    """A geometry from `path`, else the default."""
    return ArenaGeometry.load(path) if path else ArenaGeometry.default()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_check(args) -> int:
    try:
        w, h = check_frame_size(*args.size)
    except FrameSizeError as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(f"ok: {w}x{h} is 9:16 portrait")
    return 0


def _cmd_show(args) -> int:
    geo = load_geometry(args.geometry)
    size = args.size or geo.frame_size
    print(f"{geo!r}  shown at {size[0]}x{size[1]}")
    print("homography_px (world mt -> px at frame_size):")
    for row in geo.homography_px:
        print("   " + "  ".join(f"{c: .8g}" for c in row))
    for k in ("rms_px", "points"):
        if k in geo.meta:
            print(f"{k}: {geo.meta[k]}")
    print("card slots:", [geo.card_slot_px(i, size) for i in range(4)])
    for place in sorted(geo.ability_buttons):
        print(f"ability button {place}: rect {geo.ability_button_rect_px(place, size)}")
    print("model tile -> tap px:")
    for tx, ty in ((0, 31), (17, 31), (3, 17), (14, 17), (8, 24), (0, 0)):
        print(f"   ({tx:2d},{ty:2d}) -> screen tile {geo.model_tile_to_screen_tile(tx, ty)} "
              f"-> px {geo.tap_px(tx, ty, size)}")
    return 0


def _cmd_template(args) -> int:
    geo = load_geometry(args.geometry)
    w, h = check_frame_size(*args.size)
    points = []
    for name, (x, y) in LANDMARKS_TILES.items():
        px, py = geo.world_to_px(x * MT, y * MT, (w, h))
        points.append({"name": name, "px": [round(px, 1), round(py, 1)]})
    doc = {"frame_size": [w, h],
           "note": "Replace each px with where that ground point is on YOUR screenshot. "
                   "Drop points you cannot see; add points as "
                   "{'name': ..., 'world_tiles': [x, y], 'px': [x, y]}.",
           "points": points}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(points)} landmarks at {w}x{h})")
    return 0


def _cmd_fit(args) -> int:
    doc = json.loads(Path(args.points).read_text(encoding="utf-8"))
    try:
        size = check_frame_size(*doc["frame_size"])
    except FrameSizeError as exc:
        print(f"REFUSED: {exc}")
        return 2
    names, world, pix = [], [], []
    try:
        for p in doc["points"]:
            names.append(str(p.get("name", f"p{len(names)}")))
            world.append(landmark_world_mt(names[-1], p.get("world_tiles")))
            pix.append((float(p["px"][0]), float(p["px"][1])))
    except (GeometryError, KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"BAD POINTS FILE ({names[-1] if names else 'points'}): {exc}")
        return 1
    try:
        geo = ArenaGeometry.fit(world, pix, size, names=names, base=load_geometry(args.base),
                                name=args.name or Path(args.out).stem, max_rms_px=args.max_rms)
    except GeometryError as exc:
        print(f"FIT FAILED: {exc}")
        return 1
    print(f"fit on {len(names)} points at {size[0]}x{size[1]}: RMS {geo.meta['rms_px']:.2f} px")
    for n, e, loo in zip(names, geo.meta["errors_px"], geo.meta["loo_px"]):
        held = "  n/a" if loo is None else f"{loo:5.2f}"
        print(f"   {n:26s} fit {e:5.2f} px   held-out {held} px")
    geo.save(args.out)
    print(f"wrote {args.out}")
    return 0


def _cmd_overlay(args) -> int:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("overlay needs Pillow (pip install pillow)")
        return 1
    geo = load_geometry(args.geometry)
    image = Image.open(args.image).convert("RGB")
    size = check_frame_size(*image.size)
    draw = ImageDraw.Draw(image)
    for i in range(TILES_X + 1):
        a, b = geo.world_to_px(i * MT, 0, size), geo.world_to_px(i * MT, HEIGHT_MT, size)
        draw.line([a, b], fill=(255, 255, 0) if i in (0, TILES_X) else (255, 255, 255), width=1)
    for j in range(TILES_Y + 1):
        a, b = geo.world_to_px(0, j * MT, size), geo.world_to_px(WIDTH_MT, j * MT, size)
        draw.line([a, b], fill=(0, 200, 255) if j in (15, 17) else (255, 255, 255), width=1)
    for name, (x, y) in LANDMARKS_TILES.items():
        px, py = geo.world_to_px(x * MT, y * MT, size)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], outline=(255, 0, 0), width=2)
    for slot in range(4):
        x, y = geo.card_slot_px(slot, size)
        draw.line([(x - 12, y), (x + 12, y)], fill=(0, 255, 0), width=3)
        draw.line([(x, y - 12), (x, y + 12)], fill=(0, 255, 0), width=3)
    for place in geo.ability_buttons:
        draw.rectangle(geo.ability_button_rect_px(place, size), outline=(255, 0, 255), width=2)
    image.save(args.out)
    print(f"wrote {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.vision.geometry",
                                 description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="refuse a frame size that is not 9:16 portrait (exit 2)")
    p.add_argument("--size", type=parse_size, required=True, help="WIDTHxHEIGHT, e.g. 1080x1920")
    p.set_defaults(func=_cmd_check)
    p = sub.add_parser("show", help="print a geometry and sample tap pixels")
    p.add_argument("--geometry", help="geometry JSON (default: the default geometry)")
    p.add_argument("--size", type=parse_size, help="frame/device size to show pixels at")
    p.set_defaults(func=_cmd_show)
    p = sub.add_parser("template", help="write a landmark file to edit before `fit`")
    p.add_argument("--size", type=parse_size, default=DEFAULT_FRAME_SIZE, help="screenshot size")
    p.add_argument("--geometry", help="geometry that pre-fills the pixels (default: default)")
    p.add_argument("--out", required=True, help="landmark JSON to write")
    p.set_defaults(func=_cmd_template)
    p = sub.add_parser("fit", help="fit a homography to clicked landmarks and save a geometry")
    p.add_argument("--points", required=True, help="landmark JSON (see `template`)")
    p.add_argument("--out", required=True, help="geometry JSON to write")
    p.add_argument("--base", help="geometry to copy card slots, buttons and ground offsets from")
    p.add_argument("--name", help="name stored in the file (default: the output stem)")
    p.add_argument("--max-rms", type=float, default=None, help="fail if the fit RMS (px) is larger")
    p.set_defaults(func=_cmd_fit)
    p = sub.add_parser("overlay", help="draw the tile grid, landmarks and tap targets on a screenshot")
    p.add_argument("--image", required=True, help="screenshot (9:16 portrait)")
    p.add_argument("--geometry", help="geometry JSON (default: the default geometry)")
    p.add_argument("--out", required=True, help="image to write")
    p.set_defaults(func=_cmd_overlay)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
