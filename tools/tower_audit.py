r"""Audit how a session read its towers: calibration, reads, falls, holds, crowns (skin audit, sec. 2).

    .venvs\buildabot\Scripts\python.exe tools\tower_audit.py reports\vision\replays\skin --all
    ... --sheet reports\vision\replays\skin\sheets      the read boxes drawn on frames
    ... --check calib,destroy,king,overlay,crowns        exit 1 when a check fails
    ... --json reports\vision\replays\skin\audit.json    the whole report as JSON

The owner picks a random tower skin every battle, clanmates use many, and the arena changes with
the season and the mode. This tool shows, from what a session recorded, whether the HUD read each
tower where its label is and destroyed only towers that fell. It reads files only: numpy, and
Pillow for --sheet; it never taps, captures or touches the emulator.

INPUT. PATH is a session written by tools/vision_replay.py or scripts/vision_play.py (a folder
holding frames.jsonl, or the frames.jsonl itself), or a folder holding sessions: the newest
(by frames.jsonl's time), or every one with --all. Per frame it reads the VisionFrame (screen,
tick, the HUD's towers and crowns, the detections, the frame size), the battle number
(extra.battle) and the HUD's tower fields:
  vision_replay  extra.replay.hud_report = HudReport.summary(): towers (accepted fractions),
                 tower_reads, tower_bars, tower_shift (moved labels: [dx px, dy px, source]),
                 tower_present, towers_held (absent: none held), towers_live, clock;
  vision_play    extra.towers_raw = HudReport.towers_compact(): reads, bars, frac, shift,
                 present, calib (no locked_t / frames: the lock is the first frame of the
                 last calibrated run), overlay (no holds: "not recorded"), and its battle_end /
                 result records.
Fields later stages add are read when a frame has them (a session without them still audits):
  calib          {key: {"calibrated": bool, "du": px, "dv": px, "source": str}}, as
                 scripts/vision/tower_calib.py writes it (HudReport.summary / towers_compact),
                 with optionally "state" (searching, locked, re-search, expired, unlatched),
                 "locked_t" (the frame time of the lock), "frames" (the votes it took),
                 "confirmed" (the read gate on this frame) and "seen" (the badge)
  evidence       {key: "label" | "detector" | "hud"} or {key: {"kind": ..., "hold": "body" |
                 "label" | "badge", "absent_s": s}}: a zero's evidence, why the tower was held,
                 how long its calibrated label has been absent at the lock (HudReport.evidence);
                 also read from destroyed_by {key: kind}; calib's absent_s counts too
  overlay        bool (or {"flagged": bool}): the deploy overlay was on this frame and the
                 tower reads were skipped; overlay_info {"on", "mirror", "median", "tap"}
                 (overlay.OverlayRead) says why
  towers_destroyed  the HUD's confirmed set (types.Hud.towers_destroyed, in the frame's hud, or
                 the summary's): the zeros legality and the pockets act on; a zero outside it
                 is "unconfirmed".

REPORT, per battle (frames from extra.battle; the end-screen frames after it count for it) and
per tower:
  calib     the label shift the battle ended on and its source, when it was locked (s after the
            battle's first frame, its tick, frames it took, s after the third clock reading: the
            record's locked_t and frames when it has them), its last state, and the read gate:
            battle frames confirmed / not (gate); a session without calib records shows "fixed"
            (no shift) or the shift and "(no calibration record)"
  reads     frames with digits / with a bar / with neither (battle frames)
  damage    the first frame the accepted fraction fell below 0.995
  zero      the first accepted zero: time, tick, its evidence and where that came from (the
            frame, the battle_end record, or inferred: "detector" when the tower was absent to
            the detector on that frame, else "hud"), whether the HUD confirmed it (in its
            towers_destroyed; None: not recorded), presence then, the detector's label boxes
            at the tower in the 3 s before, a later revive
  absent    the longest label absence at the lock the HUD counted (s; 2.5 s destroys a tower
            read <= 0.75)
  holds     raw zeros held because the tower was seen, by kind: label (a detector label box at
            the tower), body (a body box only), badge (neither: the crown badge's pixels);
            body-only: the longest run of body-only holds (s); body after label: how long body
            boxes stayed after the tower's last label box
  boxes     battle frames the detector ran on, the share with a box at the tower (its grown
            reference body, or its label's window), and the classes of those boxes
  overlay   frames flagged (skipped), per battle, and why (tap, mirror, median) when recorded
  crowns    the HUD's crowns on the battle's last battle frame, the end screen's (per slot the
            most over its frames, as vision_play's ResultBook), whether the battle's end explains
            a difference (explained, why; could_fall: the towers that could fall at the end) and
            vision_play's result record

--sheet DIR draws the boxes the HUD read (tower_* digits and bar_* boxes moved by the frame's
shift, label_* boxes, the kings' boxes; green standing, orange held, red zero) on one frame with
an image every --every s (10) and on the frames within --around s (2) of each accepted zero,
tiled four across into DIR/<session>_b<battle>_p<page>.png. Images: a vision_play session's own;
a replay's from the input recording (its summary.json "input"), or --images DIR.

--check NAMES (comma-separated) turns the report into exit codes (1: a check failed):
  calib    every princess tower of every battle calibrated (needs calib records)
  destroy  every destroyed princess tower destroyed on label-absence evidence ("label"), and
           confirmed by the HUD when the session records its confirmed set
  king     no king reads 0 on a battle frame (its 0 comes from the end screen)
  overlay  overlay records exist, no tower was read on a flagged frame, and every image named
           by --expect-overlay was flagged
  crowns   the HUD's crowns equal the end screen's in every battle that showed one, or differ only
           by what the battle's end explains (tower_calib.end_crowns_explained): the tower whose
           fall ended the battle is never counted by the HUD, so the end screen may show 3 for a
           side (a king fell) and one more per princess tower of the losing side that stood with
           its calibrated label seen in the battle's last 2.5 s (the last battle frame's calib)
A session without a battle fails every check. Exit 2: no session, bad arguments.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                    # scripts.vision
    sys.path.insert(0, str(ROOT))

from scripts.vision import tower_calib as TC  # noqa: E402
from scripts.vision import tower_labels as TL  # noqa: E402

SCHEMA = "hastycr.vision.tower_audit/1"
FRAMES_FILE = "frames.jsonl"
EXIT_OK, EXIT_CHECK, EXIT_INPUT = 0, 1, 2
CHECKS = ("calib", "destroy", "king", "overlay", "crowns")
TOWER_KEYS = ("own_king", "own_left", "own_right", "enemy_king", "enemy_left", "enemy_right")
PRINCESS = TL.PRINCESS_KEYS
KINGS = TL.KING_KEYS
SCREEN_BATTLE, SCREEN_END = "battle", "end"
ZERO = 0.01                                      # hud.TowerHpFilter.ZERO: an accepted zero
FULL = 0.995                                     # below this: damaged
CLOCK_READ = ("read", "resync")                  # hud.ClockState sources that are readings
EVIDENCE_LABEL = "label"
BEFORE_ZERO_S = 3.0                              # label boxes counted this long before a zero
HOLD_KINDS = ("label", "body", "badge")
SHEET_COLS, SHEET_TILE_W, SHEET_ROWS = 4, 374, 6


# ---------------------------------------------------------------------------
# reading a session
# ---------------------------------------------------------------------------
def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _overlay_flag(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, Mapping):
        return bool(v.get("flagged", v.get("on", False)))
    return bool(v)


def _evidence(v: Any) -> tuple[str | None, str | None]:
    """(kind of a destruction, kind of a hold) from one key's evidence entry."""
    if isinstance(v, str):
        return v, None
    if isinstance(v, Mapping):
        kind = v.get("destroyed_by", v.get("kind"))
        hold = v.get("hold")
        return (None if kind is None else str(kind)), (None if hold is None else str(hold))
    return None, None


@dataclass
class FrameView:
    """One frame's tower view, from either session kind."""
    i: int
    t: float
    battle: int | None
    screen: str
    tick: int | None
    image: str | None = None                 # the session's own image (relative path)
    image_name: str | None = None            # a replay's input image name
    frame_size: tuple[int, int] | None = None
    fields: str | None = None                # "summary", "compact" or None (no tower fields)
    frac: dict[str, float | None] = field(default_factory=dict)
    hud_hp: dict[str, tuple[int | None, int | None]] = field(default_factory=dict)
    crowns: tuple[int, int] = (0, 0)
    reads: dict[str, tuple[int, float]] = field(default_factory=dict)
    bars: dict[str, float] = field(default_factory=dict)
    shift: dict[str, tuple[float, float, str]] = field(default_factory=dict)
    present: dict[str, bool | None] = field(default_factory=dict)
    held: list[str] | None = None            # None: not recorded
    clock_source: str | None = None
    calib: dict[str, dict] | None = None
    evidence: dict[str, Any] | None = None
    overlay: bool | None = None
    overlay_why: str | None = None           # tap, mirror or median (overlay_info)
    confirmed: set[str] | None = None        # the HUD's towers_destroyed (None: not recorded)
    detections: list[tuple[str, tuple[float, float, float, float]]] = field(default_factory=list)
    det_ran: bool = False
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def hp_zero(self, key: str) -> bool:
        """The HUD shows `key` destroyed: accepted fraction <= ZERO, or HP 0."""
        f = self.frac.get(key)
        if f is not None and f <= ZERO:
            return True
        hp = self.hud_hp.get(key, (None, None))[0]
        return hp is not None and hp <= 0

    def shift_of(self, key: str) -> tuple[float, float, str]:
        if self.calib and key in self.calib and isinstance(self.calib[key], Mapping):
            c = self.calib[key]
            du, dv = _num(c.get("du")) or 0.0, _num(c.get("dv")) or 0.0
            return du, dv, str(c.get("source", "calib"))
        return self.shift.get(key, (0.0, 0.0, TL.SOURCE_FIXED))

    def calib_field(self, key: str, name: str) -> Any:
        """Field `name` of `key`'s calib record on this frame (None without one)."""
        c = (self.calib or {}).get(key)
        return c.get(name) if isinstance(c, Mapping) else None

    def absent_s(self, key: str) -> float | None:
        """The label absence at the lock this frame records for `key` (evidence or calib)."""
        e = (self.evidence or {}).get(key)
        a = _num(e.get("absent_s")) if isinstance(e, Mapping) else None
        if a is None:
            a = _num(self.calib_field(key, "absent_s"))
        return a

    def calibrated(self, key: str) -> bool | None:
        """None without a calib record for `key` on this frame."""
        if not self.calib or key not in self.calib:
            return None
        c = self.calib[key]
        if isinstance(c, Mapping):
            return bool(c.get("calibrated", c.get("locked", False)))
        return bool(c)


def frame_view(rec: Mapping[str, Any]) -> FrameView:
    """A FrameView from one "frame" record of frames.jsonl (module doc: INPUT)."""
    fr = rec.get("frame") or {}
    extra = rec.get("extra") or {}
    hud = fr.get("hud") or {}
    size = fr.get("frame_size")
    replay = extra.get("replay") if isinstance(extra.get("replay"), Mapping) else None
    v = FrameView(i=int(rec.get("i") or 0), t=float(fr.get("t_capture") or 0.0),
                  battle=None if extra.get("battle") is None else int(extra["battle"]),
                  screen=str(fr.get("screen", "unknown")),
                  tick=None if fr.get("tick") is None else int(fr["tick"]),
                  image=rec.get("image"),
                  image_name=None if replay is None else replay.get("image"),
                  frame_size=None if not size else (int(size[0]), int(size[1])))
    crowns = hud.get("crowns") or (0, 0)
    v.crowns = (int(crowns[0]), int(crowns[1]))
    for k, d in (hud.get("towers") or {}).items():
        d = d or {}
        v.hud_hp[str(k)] = (None if d.get("hp") is None else int(d["hp"]),
                            None if d.get("max") is None else int(d["max"]))
    for d in fr.get("detections") or ():
        box = d.get("box")
        if box and len(box) == 4:
            v.detections.append((str(d.get("cls", "")), tuple(float(c) for c in box)))
    rep = replay.get("hud_report") if replay is not None else None
    if isinstance(rep, Mapping):
        v.fields = "summary"
        frac, reads, bars = rep.get("towers"), rep.get("tower_reads"), rep.get("tower_bars")
        shift, present = rep.get("tower_shift"), rep.get("tower_present")
        v.held = [str(k) for k in rep.get("towers_held") or ()]
        clock = rep.get("clock") or {}
        v.clock_source = clock.get("source") if isinstance(clock, Mapping) else None
        src = rep
    elif isinstance(extra.get("towers_raw"), Mapping):
        raw = extra["towers_raw"]
        v.fields = "compact"
        frac, reads, bars = raw.get("frac"), raw.get("reads"), raw.get("bars")
        shift, present = raw.get("shift"), raw.get("present")
        v.held = None if "towers_held" not in raw else [str(k) for k in raw.get("towers_held") or ()]
        src = raw
    else:
        frac = reads = bars = shift = present = None
        src = {}
    v.frac = {str(k): _num(x) for k, x in (frac or {}).items()}
    for k, r in (reads or {}).items():
        if isinstance(r, (list, tuple)) and r and r[0] is not None:
            v.reads[str(k)] = (int(r[0]), float(r[1]) if len(r) > 1 and r[1] is not None else 0.0)
    v.bars = {str(k): float(x) for k, x in (bars or {}).items() if _num(x) is not None}
    for k, s in (shift or {}).items():
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            v.shift[str(k)] = (float(s[0]), float(s[1]), str(s[2]) if len(s) > 2 else "?")
    v.present = {str(k): (None if x is None else bool(x)) for k, x in (present or {}).items()}
    for name in ("calib", "evidence", "overlay", "destroyed_by"):
        val = src.get(name) if isinstance(src, Mapping) else None
        if val is None:
            val = extra.get(name)
        if name == "calib" and isinstance(val, Mapping):
            v.calib = {str(k): x for k, x in val.items()}
        elif name == "evidence" and isinstance(val, Mapping):
            v.evidence = {str(k): x for k, x in val.items()}
        elif name == "destroyed_by" and isinstance(val, Mapping):
            v.evidence = {**{str(k): x for k, x in val.items()}, **(v.evidence or {})}
        elif name == "overlay":
            v.overlay = _overlay_flag(val)
    info = src.get("overlay_info") if isinstance(src, Mapping) else None
    if isinstance(info, Mapping):
        if v.overlay is None:
            v.overlay = bool(info.get("on"))
        if info.get("on"):
            v.overlay_why = ("tap" if info.get("tap") else
                             "mirror" if (_num(info.get("mirror")) or 0.0) >= 0.5 else "median")
    confirmed = hud.get("towers_destroyed") if "towers_destroyed" in hud else (
        src.get("towers_destroyed") if isinstance(src, Mapping) else None)
    if confirmed is not None:
        v.confirmed = {str(k) for k in confirmed}
    if replay is not None:
        v.det_ran = replay.get("det") is not None
    else:
        v.det_ran = bool(v.detections)
    return v


@dataclass
class Session:
    path: Path                               # the session folder
    frames: list[FrameView]
    logs: list[dict]                         # every non-frame record
    kind: str                                # "replay", "play" or "unknown"
    bad_lines: int = 0
    _input: list = field(default_factory=list, repr=False)     # [the replay's input] once looked up

    @property
    def name(self) -> str:
        return self.path.name


def find_sessions(path: str | Path, every: bool = False) -> list[Path]:
    """The session folder(s) PATH names (module doc: INPUT): [] when none."""
    p = Path(path)
    if p.is_file() and p.name == FRAMES_FILE:
        return [p.parent]
    if (p / FRAMES_FILE).is_file():
        return [p]
    if not p.is_dir():
        return []
    found = sorted(d for d in p.iterdir() if d.is_dir() and (d / FRAMES_FILE).is_file())
    if every or not found:
        return found
    return [max(found, key=lambda d: ((d / FRAMES_FILE).stat().st_mtime, d.name))]


def _records(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                yield {"kind": "__bad__"}
                continue
            if isinstance(rec, dict):
                yield rec


def load_session(folder: str | Path) -> Session:
    folder = Path(folder)
    frames, logs, bad = [], [], 0
    for rec in _records(folder / FRAMES_FILE):
        kind = rec.get("kind")
        if kind == "__bad__":
            bad += 1
        elif kind == "frame":
            frames.append(frame_view(rec))
        elif kind != "session":
            logs.append(rec)
    if any(f.fields == "summary" for f in frames) or any(r.get("kind") == "replay_summary" for r in logs):
        kind = "replay"
    elif any(f.fields == "compact" for f in frames) or any(r.get("kind") == "session_start" for r in logs):
        kind = "play"
    else:
        kind = "unknown"
    return Session(folder, frames, logs, kind, bad)


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------
def _to_ref(box: Sequence[float], size: Sequence[int] | None) -> tuple[float, float, float, float]:
    w, h = (size or TL.REF_SIZE)
    sx, sy = TL.REF_SIZE[0] / float(w), TL.REF_SIZE[1] / float(h)
    return box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy


def _grow(box: Sequence[float], frac: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(c) for c in box)
    gx, gy = (x1 - x0) * frac, (y1 - y0) * frac
    return x0 - gx, y0 - gy, x1 + gx, y1 + gy


def boxes_at(view: FrameView, label_refs: Mapping[str, Sequence[float]] | None = None
             ) -> dict[str, list[str]]:
    """Tower key -> the classes of the frame's detector boxes at that tower: a box whose centre
    lies in the tower's reference body grown BODY_GROW a side, or (princess towers) in its
    label's match window (tower_labels.LABEL_MATCH_*), whatever its class."""
    ck = ("at", id(label_refs))
    if ck in view._cache:
        return view._cache[ck]
    refs = label_refs or TL.LABEL_REF_PX
    out: dict[str, list[str]] = {}
    for cls, box in view.detections:
        x0, y0, x1, y1 = _to_ref(box, view.frame_size)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        for key in TOWER_KEYS:
            hit = False
            b = TL.TOWER_BODY_REF_PX.get(key)
            if b is not None:
                g = _grow(b, TL.BODY_GROW)
                hit = g[0] <= cx <= g[2] and g[1] <= cy <= g[3]
            ref = refs.get(key)
            if not hit and ref is not None:
                rx, ry = (ref[0] + ref[2]) / 2.0, (ref[1] + ref[3]) / 2.0
                hit = (abs(cx - rx) <= TL.LABEL_MATCH_DU_PX
                       and TL.LABEL_MATCH_DV_PX[0] <= cy - ry <= TL.LABEL_MATCH_DV_PX[1])
            if hit:
                out.setdefault(key, []).append(cls)
    view._cache[ck] = out
    return out


def evidence_of(view: FrameView, label_refs: Mapping[str, Sequence[float]] | None = None) -> TL.TowerEvidence:
    """The frame's detector evidence per tower (tower_labels.TowerEvidence: labels, bodies)."""
    ck = ("ev", id(label_refs))
    if ck not in view._cache:
        dets = [_Det(c, b) for c, b in view.detections]
        view._cache[ck] = TL.TowerEvidence.from_detections(dets, view.frame_size or TL.REF_SIZE, label_refs)
    return view._cache[ck]


@dataclass
class _Det:
    cls: str
    box: tuple[float, float, float, float]
    conf: float = 1.0


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------
def battles_of(session: Session) -> dict[int, dict[str, list[FrameView]]]:
    """Battle number -> {"frames": its frames, "end": the end-screen frames after it}."""
    out: dict[int, dict[str, list[FrameView]]] = {}
    last: int | None = None
    for f in session.frames:
        if f.battle is not None:
            last = f.battle
            out.setdefault(f.battle, {"frames": [], "end": []})["frames"].append(f)
            if f.screen == SCREEN_END:
                out[f.battle]["end"].append(f)
        elif f.screen == SCREEN_END and last is not None:
            out[last]["end"].append(f)
    return out


def _log_for(session: Session, kind: str, battle: int) -> dict | None:
    for r in session.logs:
        data = r.get("data") or {}
        if r.get("kind") == kind and data.get("battle") == battle:
            return data
    return None


def _round(x: float | None, n: int = 2) -> float | None:
    return None if x is None else round(float(x), n)


def _calib(key: str, frames: list[FrameView], t0: float, clock3_t: float | None) -> dict[str, Any]:
    recorded = [f for f in frames if f.calibrated(key) is not None]
    extra: dict[str, Any] = {}
    if recorded:
        last = recorded[-1]
        ok = bool(last.calibrated(key))
        du, dv, src = last.shift_of(key)
        lock = None
        locked_t = _num(last.calib_field(key, "locked_t")) if ok else None
        if ok:
            for f in reversed(recorded):
                if not f.calibrated(key) or f.shift_of(key) != (du, dv, src):
                    break
                if locked_t is not None and f.t < locked_t - 1e-6:
                    break
                lock = f
        status = "calibrated" if ok else "uncalibrated"
        state = last.calib_field(key, "state")
        if state is not None:
            extra["state"] = str(state)
        battle = [f for f in recorded if f.screen == SCREEN_BATTLE]
        gate = [f.calib_field(key, "confirmed") for f in battle]
        if any(g is not None for g in gate):
            extra["gate"] = {"confirmed": sum(1 for g in gate if g is True),
                             "unconfirmed": sum(1 for g in gate if g is False)}
        if lock is not None and _num(last.calib_field(key, "frames")) is not None:
            extra["votes"] = int(last.calib_field(key, "frames"))
    else:
        battle = [f for f in frames if f.screen == SCREEN_BATTLE and f.fields] or frames
        du, dv, src = battle[-1].shift_of(key) if battle else (0.0, 0.0, TL.SOURCE_FIXED)
        lock = None
        if src != TL.SOURCE_FIXED:
            for f in reversed(battle):
                if f.shift_of(key) != (du, dv, src):
                    break
                lock = f
        status = "fixed" if src == TL.SOURCE_FIXED else "placed"
    out: dict[str, Any] = {"status": status, "recorded": bool(recorded), "du": _round(du, 1),
                           "dv": _round(dv, 1), "source": src}
    if lock is not None:
        out.update(locked_t=_round(lock.t - t0), locked_tick=lock.tick,
                   frames_to_lock=frames.index(lock) + 1,
                   after_clock3_s=None if clock3_t is None else _round(lock.t - clock3_t))
    out.update(extra)
    return out


def _tower(key: str, frames: list[FrameView], t0: float, clock3_t: float | None, end_log: dict | None,
           label_refs: Mapping[str, Sequence[float]] | None) -> dict[str, Any]:
    battle = [f for f in frames if f.screen == SCREEN_BATTLE and f.fields]
    out: dict[str, Any] = {"calib": _calib(key, frames, t0, clock3_t) if key in PRINCESS else None}
    digits = sum(1 for f in battle if key in f.reads)
    bars = sum(1 for f in battle if key in f.bars)
    out["reads"] = {"digits": digits, "bars": bars,
                    "none": sum(1 for f in battle if key not in f.reads and key not in f.bars)}
    dmg = next((f for f in battle if f.frac.get(key) is not None and f.frac[key] < FULL), None)
    out["first_damage"] = None if dmg is None else {"t": _round(dmg.t - t0), "tick": dmg.tick,
                                                    "frac": _round(dmg.frac[key], 3)}
    zero_i = next((i for i, f in enumerate(battle) if f.hp_zero(key)), None)
    out["zero"] = None
    if zero_i is not None:
        z = battle[zero_i]
        kind, source = None, None
        for f in battle[zero_i:]:
            k, _ = _evidence((f.evidence or {}).get(key))
            if k is not None:
                kind, source = k, "frame"
                break
        if kind is None and end_log:
            by = ((end_log.get("towers_info") or {}).get("destroyed_by") or {})
            if key in by:
                kind, source = str(by[key]), "battle_end"
        if kind is None:
            kind, source = ("detector" if z.present.get(key) is False else "hud"), "inferred"
        before = [f for f in battle[:zero_i] if f.t >= z.t - BEFORE_ZERO_S and f.det_ran]
        with_label = sum(1 for f in before if key in evidence_of(f, label_refs).labels)
        revived = next((f for f in battle[zero_i:] if not f.hp_zero(key) and f.frac.get(key) is not None), None)
        out["zero"] = {"t": _round(z.t - t0), "tick": z.tick, "evidence": kind, "evidence_from": source,
                       "confirmed": None if z.confirmed is None else key in z.confirmed,
                       "present": z.present.get(key), "label_boxes_before": [with_label, len(before)],
                       "revived_t": None if revived is None else _round(revived.t - t0)}
    absent = [a for a in (f.absent_s(key) for f in battle) if a is not None]
    out["absent_max_s"] = _round(max(absent)) if absent else None
    if all(f.held is None for f in battle):
        out["holds"] = None
    else:
        holds: Counter = Counter()
        run = best = 0.0
        run_t0: float | None = None
        for f in battle:
            if f.held is None or key not in f.held:
                run_t0 = None
                continue
            _, hold = _evidence((f.evidence or {}).get(key))
            if hold is None:
                ev = evidence_of(f, label_refs) if f.det_ran else None
                hold = ("label" if ev is not None and key in ev.labels
                        else "body" if ev is not None and key in ev.bodies else "badge")
            holds[hold] += 1
            if hold == "body":
                run_t0 = f.t if run_t0 is None else run_t0
                run = f.t - run_t0
                best = max(best, run)
            else:
                run_t0 = None
        out["holds"] = {k: holds.get(k, 0) for k in HOLD_KINDS}
        out["holds"].update({k: v for k, v in holds.items() if k not in HOLD_KINDS})
        out["body_only_s"] = _round(best)
    det = [f for f in battle if f.det_ran]
    if det:
        classes: Counter = Counter()
        with_box = 0
        last_label = last_body = None
        for f in det:
            here = boxes_at(f, label_refs).get(key)
            if here:
                with_box += 1
                classes.update(here)
            ev = evidence_of(f, label_refs)
            if key in ev.labels:
                last_label = f.t
            if key in ev.bodies:
                last_body = f.t
        out["boxes"] = {"frames": len(det), "with_box": with_box, "share": _round(with_box / len(det), 3),
                        "classes": dict(classes.most_common())}
        out["body_after_label_s"] = (None if last_label is None or last_body is None or last_body <= last_label
                                     else _round(last_body - last_label))
    else:
        out["boxes"] = None
        out["body_after_label_s"] = None
    return out


def _frame_name(f: FrameView) -> str:
    """A frame's image name (00000420.jpg), else #<frame index>."""
    if f.image_name:
        return str(f.image_name)
    return Path(f.image).name if f.image else f"#{f.i}"


def audit_battle(session: Session, number: int, frames: list[FrameView], end: list[FrameView], *,
                 label_refs: Mapping[str, Sequence[float]] | None = None) -> dict[str, Any]:
    t0 = frames[0].t
    battle = [f for f in frames if f.screen == SCREEN_BATTLE]
    reads = [f for f in battle if f.clock_source in CLOCK_READ]
    clock3_t = reads[2].t if len(reads) >= 3 else None
    ticks = [f.tick for f in battle if f.tick is not None]
    end_log = _log_for(session, "battle_end", number)
    towers = {k: _tower(k, frames, t0, clock3_t, end_log, label_refs) for k in TOWER_KEYS}
    flagged = [f for f in battle if f.overlay]
    overlay = None if all(f.overlay is None for f in battle) else {
        "frames": len(flagged), "names": [_frame_name(f) for f in flagged],
        "read_on_flagged": sorted({k for f in flagged for k in list(f.reads) + list(f.bars)}),
        **({"why": dict(Counter(f.overlay_why for f in flagged if f.overlay_why))}
           if any(f.overlay_why for f in flagged) else {})}
    hud_frames = [f for f in battle if f.fields]
    last = (hud_frames or battle or frames)[-1]
    hud_crowns = list(last.crowns)
    end_crowns = None
    if end:
        end_crowns = [max(f.crowns[0] for f in end), max(f.crowns[1] for f in end)]
    result = _log_for(session, "result", number)
    crowns = {"hud": hud_crowns, "end": end_crowns, "end_frames": len(end),
              "agree": None if end_crowns is None else hud_crowns == end_crowns}
    if end_crowns is not None and hud_crowns != end_crowns:
        could = end_could_fall(last)
        crowns["explained"], crowns["why"] = TC.end_crowns_explained(hud_crowns, end_crowns, could)
        crowns["could_fall"] = could
    if result is not None:
        crowns["result"] = {k: result.get(k) for k in ("crowns", "source", "outcome", "crowns_disagree",
                                                       "crowns_explained") if k in result}
    return {"battle": number, "t0": round(t0, 3), "duration_s": _round((battle or frames)[-1].t - t0),
            "frames": len(frames), "battle_frames": len(battle), "first_tick": ticks[0] if ticks else None,
            "last_tick": ticks[-1] if ticks else None,
            "clock3_t": None if clock3_t is None else _round(clock3_t - t0),
            "towers": towers, "overlay": overlay, "crowns": crowns,
            "recorded": {"calib": any(f.calib is not None for f in frames),
                         "evidence": any(f.evidence is not None for f in frames),
                         "overlay": overlay is not None,
                         "confirmed": any(f.confirmed is not None for f in battle),
                         "holds": any(f.held is not None for f in battle),
                         "detections": any(f.det_ran for f in battle)}}


def end_could_fall(f: FrameView) -> dict[str, list[str]]:
    """The towers the end screen may count although the HUD did not, from the battle's last
    battle frame (tower_calib.could_fall_at_end): calibrated, standing, the label's absence at the
    lock under 2.5 s. Nothing for a frame without calib records."""
    absent: dict[str, float | None] = {}
    for k in PRINCESS:
        absent[k] = (f.absent_s(k) or 0.0) if f.calibrated(k) else None
    return TC.could_fall_at_end(absent, [k for k in TOWER_KEYS if f.hp_zero(k)])


def audit_session(session: Session, *, label_refs: Mapping[str, Sequence[float]] | None = None) -> dict[str, Any]:
    battles = battles_of(session)
    return {"session": str(session.path), "name": session.name, "kind": session.kind,
            "frames": len(session.frames), "bad_lines": session.bad_lines,
            "battles": [audit_battle(session, n, b["frames"], b["end"], label_refs=label_refs)
                        for n, b in sorted(battles.items())]}


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def parse_checks(text: str | None) -> list[str]:
    if not text:
        return []
    names = [c.strip().lower() for c in str(text).split(",") if c.strip()]
    bad = [c for c in names if c not in CHECKS]
    if bad:
        raise ValueError(f"unknown check(s) {', '.join(bad)} (choose from {', '.join(CHECKS)})")
    return list(dict.fromkeys(names))


def check_battle(b: Mapping[str, Any], name: str) -> list[str]:
    """The failures of check `name` on one audited battle ([]: it passes)."""
    out: list[str] = []
    towers = b["towers"]
    if name == "calib":
        if not b["recorded"]["calib"]:
            return ["no calibration records (the session predates label calibration)"]
        for k in PRINCESS:
            c = towers[k]["calib"]
            if c["status"] != "calibrated":
                out.append(f"{k} not calibrated" + (f" ({c['state']})" if c.get("state") else ""))
    elif name == "destroy":
        for k in PRINCESS:
            z = towers[k]["zero"]
            if z is not None and z["evidence"] != EVIDENCE_LABEL:
                out.append(f"{k} destroyed at {z['t']} s without label evidence "
                           f"({z['evidence']}, {z['evidence_from']})")
            elif z is not None and z.get("confirmed") is False:
                out.append(f"{k} destroyed at {z['t']} s but not in the HUD's confirmed set")
    elif name == "king":
        for k in KINGS:
            z = towers[k]["zero"]
            if z is not None:
                out.append(f"{k} read 0 at {z['t']} s (tick {z['tick']}) before the end screen")
    elif name == "overlay":
        ov = b["overlay"]
        if ov is None:
            return ["no overlay records"]
        if ov["read_on_flagged"]:
            out.append(f"towers read on overlay frames: {', '.join(ov['read_on_flagged'])}")
    elif name == "crowns":
        c = b["crowns"]
        if c["end"] is not None and c["hud"] != c["end"] and not c.get("explained"):
            out.append(f"HUD crowns {c['hud'][0]}-{c['hud'][1]} differ from the end screen's "
                       f"{c['end'][0]}-{c['end'][1]}")
    return out


def run_checks(report: Mapping[str, Any], names: Sequence[str], expect_overlay: Sequence[str] = ()
               ) -> dict[str, dict[str, Any]]:
    """name -> {"ok", "failures": ["<session> battle <n>: why", ...]} over every session; the
    overlay check also wants each of `expect_overlay` (image names) flagged in some battle."""
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        fails: list[str] = []
        for s in report["sessions"]:
            if not s["battles"]:
                fails.append(f"{s['name']}: no battle")
                continue
            for b in s["battles"]:
                fails += [f"{s['name']} battle {b['battle']}: {why}" for why in check_battle(b, name)]
        if name == "overlay":
            flagged = {n for s in report["sessions"] for b in s["battles"] if b["overlay"]
                       for n in b["overlay"]["names"]}
            fails += [f"{n}: not flagged as an overlay frame" for n in expect_overlay if n not in flagged]
        out[name] = {"ok": not fails, "failures": fails}
    return out


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------
def _calib_text(c: Mapping[str, Any] | None) -> str:
    """"placed +0,-36 detector @1.0s/5f", "calibrated +0,-35 search @2.0s/9f gate 97%", "fixed",
    "uncalibrated (expired)", ..."""
    if c is None:
        return "-"
    if c["status"] == "fixed":
        return "fixed"
    if c["status"] == "uncalibrated":
        text = "uncalibrated" + (f" ({c['state']})" if c.get("state") else "")
    else:
        text = f"{c['status']} {c['du']:+.0f},{c['dv']:+.0f} {c['source']}"
    if c.get("locked_t") is not None:
        text += f" @{c['locked_t']:.1f}s/{c['frames_to_lock']}f"
    g = c.get("gate")
    if g and g["confirmed"] + g["unconfirmed"]:
        text += f" gate {100 * g['confirmed'] / (g['confirmed'] + g['unconfirmed']):.0f}%"
    return text


def _when(x: Mapping[str, Any] | None) -> str:
    if not x:
        return "-"
    return f"{x['t']:.1f}s t{x['tick'] if x['tick'] is not None else '?'}"


def format_report(report: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for s in report["sessions"]:
        lines.append(f"session {s['session']} ({s['kind']}, {s['frames']} frames"
                     + (f", {s['bad_lines']} bad lines" if s["bad_lines"] else "") + ")")
        if not s["battles"]:
            lines.append("  no battle (no frame carries a battle number)")
        for b in s["battles"]:
            rec = b["recorded"]
            missing = [k for k in ("calib", "evidence", "overlay", "holds", "detections") if not rec[k]]
            lines.append(f"  battle {b['battle']}: {b['duration_s']} s, {b['battle_frames']} battle frames, "
                         f"ticks {b['first_tick']}-{b['last_tick']}, third clock reading "
                         f"{'-' if b['clock3_t'] is None else str(b['clock3_t']) + ' s'}"
                         + (f"; not recorded: {', '.join(missing)}" if missing else ""))
            lines.append(f"    {'tower':12s} {'calib':44s} {'reads d/b/-':14s} {'damage':12s} "
                         f"{'zero (evidence)':36s} {'holds l/b/badge':16s} {'body s':7s} boxes")
            for k in TOWER_KEYS:
                t = b["towers"][k]
                r = t["reads"]
                z = t["zero"]
                ztext = "-" if z is None else (
                    f"{_when(z)} {z['evidence']} ({z['evidence_from']})"
                    + (" unconfirmed" if z.get("confirmed") is False else "")
                    + (f" revived {z['revived_t']}s" if z.get("revived_t") is not None else ""))
                h = t["holds"]
                htext = "n/a" if h is None else f"{h['label']}/{h['body']}/{h['badge']}"
                body = "-" if h is None else f"{t.get('body_only_s') or 0:.1f}"
                bx = t["boxes"]
                btext = "n/a" if bx is None else (
                    f"{100 * bx['share']:.0f}% "
                    + ", ".join(f"{c} {n}" for c, n in list(bx["classes"].items())[:3]))
                if t.get("body_after_label_s"):
                    btext += f"; body {t['body_after_label_s']:.1f}s after the label"
                if t.get("absent_max_s"):
                    btext += f"; label absent {t['absent_max_s']:.1f}s"
                rtext = f"{r['digits']}/{r['bars']}/{r['none']}"
                lines.append(f"    {k:12s} {_calib_text(t['calib']):44s} {rtext:14s} "
                             f"{_when(t['first_damage']):12s} {ztext:36s} {htext:16s} {body:7s} {btext}".rstrip())
            ov = b["overlay"]
            lines.append("    overlay: not recorded" if ov is None else
                         f"    overlay: {ov['frames']} frames skipped"
                         + (" (" + ", ".join(f"{w} {n}" for w, n in sorted(ov["why"].items())) + ")"
                            if ov.get("why") else "")
                         + (f"; read on them: {', '.join(ov['read_on_flagged'])}" if ov["read_on_flagged"] else ""))
            c = b["crowns"]
            ctext = f"    crowns: HUD {c['hud'][0]}-{c['hud'][1]}"
            if c["end"] is None:
                ctext += ", no end screen"
            else:
                ctext += (f", end screen {c['end'][0]}-{c['end'][1]} ({c['end_frames']} frames): "
                          + ("agree" if c["agree"] else
                             f"explained ({c['why']})" if c.get("explained") else "DIFFER"
                             + (f" ({c['why']})" if c.get("why") else "")))
            if c.get("result"):
                rr = c["result"]
                ctext += f"; result {rr.get('crowns')} ({rr.get('source')})"
            lines.append(ctext)
    for name, res in (report.get("checks") or {}).items():
        lines.append(f"check {name}: {'PASS' if res['ok'] else 'FAIL'}")
        lines += [f"  {why}" for why in res["failures"]]
    return lines


# ---------------------------------------------------------------------------
# sheets
# ---------------------------------------------------------------------------
def _replay_input(session: Session) -> Path | None:
    """The recording a replay session was made from (its replay_summary record or summary.json)."""
    if not session._input:
        session._input.append(_find_input(session))
    return session._input[0]


def _find_input(session: Session) -> Path | None:
    for r in session.logs:
        if r.get("kind") == "replay_summary":
            inp = (r.get("data") or {}).get("input")
            if inp:
                return Path(inp)
    s = session.path / "summary.json"
    if s.is_file():
        try:
            inp = json.loads(s.read_text(encoding="utf-8")).get("input")
        except (OSError, ValueError):
            inp = None
        if inp:
            return Path(inp)
    return None


def image_path(session: Session, view: FrameView, images: Path | None = None) -> Path | None:
    """Where the frame's image is (module doc: --sheet), None when it is nowhere."""
    cands: list[Path] = []
    if view.image:
        cands.append(session.path / view.image)
    name = view.image_name or (Path(view.image).name if view.image else None)
    if name:
        if images is not None:
            cands += [images / name, images / "images" / name]
        inp = _replay_input(session)
        if inp is not None:
            base = inp.parent if inp.is_file() or inp.suffix == ".jsonl" else inp
            cands += [base / "images" / name, base / name]
    return next((c for c in cands if c.is_file()), None)


def pick_sheet_frames(frames: list[FrameView], zeros: Iterable[float], every_s: float, around_s: float,
                      has_image) -> list[tuple[FrameView, str]]:
    """(frame, why) in time order: the first frame with an image at or after each `every_s` mark,
    and up to three before and three after each zero time within `around_s`."""
    with_img = [f for f in frames if has_image(f)]
    if not with_img:
        return []
    t0 = frames[0].t
    chosen: dict[int, str] = {}
    mark = 0.0
    for f in with_img:
        if f.t - t0 >= mark:
            chosen.setdefault(f.i, f"{f.t - t0:.0f}s")
            mark = (math.floor((f.t - t0) / every_s) + 1) * every_s if every_s > 0 else math.inf
    for z in zeros:
        near = [f for f in with_img if abs(f.t - z) <= around_s]
        before = [f for f in near if f.t < z][-3:]
        after = [f for f in near if f.t >= z][:3]
        for f in before + after:
            chosen[f.i] = f"zero {'-' if f.t < z else '+'}{abs(f.t - z):.1f}s"
    order = {f.i: f for f in with_img}
    return [(order[i], why) for i, why in sorted(chosen.items(), key=lambda kv: order[kv[0]].t)]


def draw_frame(rgb, view: FrameView, regions):
    """The frame as a Pillow image with the HUD's read boxes (module doc: --sheet)."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    w, h = img.size
    for key in TOWER_KEYS:
        du, dv, src = view.shift_of(key)
        if view.hp_zero(key):
            colour = (255, 40, 40)
        elif view.held and key in view.held:
            colour = (255, 160, 0)
        else:
            colour = (40, 230, 60)
        names = [f"tower_{key}"] + ([f"bar_{key}", f"label_{key}"] if key in PRINCESS else [])
        for name in names:
            r = regions.get(name)
            if r is None:
                continue
            x0, y0, x1, y1 = r.shifted(du / 1080.0, dv / 1920.0).px((w, h))
            d.rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], outline=colour,
                        width=1 if name.startswith("label_") else 2)
        r = regions.get(f"tower_{key}")
        if r is not None:
            x0, y0, _, _ = r.shifted(du / 1080.0, dv / 1920.0).px((w, h))
            f = view.frac.get(key)
            hp = view.reads.get(key, (None, 0.0))[0]
            bar = view.bars.get(key)
            text = (f"{'-' if f is None else f'{f:.2f}'}"
                    + (f" d{hp}" if hp is not None else "") + (f" b{bar:.2f}" if bar is not None else "")
                    + (f" {dv:+.0f}{src[0]}" if dv or du else ""))
            d.text((x0, max(0, y0 - 11)), text, fill=colour)
    if view.overlay:
        d.rectangle([0, 0, w - 1, h - 1], outline=(255, 0, 255), width=6)
    return img


def _caption(tile, text: str):
    from PIL import ImageDraw
    d = ImageDraw.Draw(tile)
    d.rectangle([0, 0, tile.size[0], 14], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 255))
    return tile


def write_sheets(session: Session, audit: Mapping[str, Any], out_dir: Path, *, regions=None,
                 images: Path | None = None, every_s: float = 10.0, around_s: float = 2.0,
                 log=print) -> list[Path]:
    """Contact sheets per battle (module doc: --sheet); returns the files written."""
    from PIL import Image

    from scripts.vision.record import load_image
    if regions is None:
        from scripts.vision.hud import load_regions
        regions = load_regions()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    paths: dict[int, Path | None] = {}

    def has_image(f: FrameView) -> bool:
        if f.i not in paths:
            paths[f.i] = image_path(session, f, images)
        return paths[f.i] is not None

    battles = battles_of(session)
    for b in audit["battles"]:
        frames = [f for f in battles[b["battle"]]["frames"] if f.screen == SCREEN_BATTLE]
        if not frames:
            continue
        zeros = [frames[0].t + t["zero"]["t"] for t in b["towers"].values() if t["zero"]]
        picks = pick_sheet_frames(frames, zeros, every_s, around_s, has_image)
        if not picks:
            log(f"  {session.name} battle {b['battle']}: no frame with an image (--images DIR?)")
            continue
        tiles = []
        for view, why in picks:
            try:
                tile = draw_frame(load_image(paths[view.i]), view, regions)
            except Exception as exc:                    # an unreadable image: skip it
                log(f"  {paths[view.i]}: {type(exc).__name__}: {exc}")
                continue
            th = max(1, round(tile.size[1] * SHEET_TILE_W / tile.size[0]))
            tiles.append(_caption(tile.resize((SHEET_TILE_W, th)),
                                  f"{why}  tick {view.tick}  {_frame_name(view)}"
                                  + ("  OVERLAY" if view.overlay else "")))
        per_page = SHEET_COLS * SHEET_ROWS
        for p in range(0, len(tiles), per_page):
            page = tiles[p:p + per_page]
            th = max(t.size[1] for t in page)
            rows = (len(page) + SHEET_COLS - 1) // SHEET_COLS
            sheet = Image.new("RGB", (SHEET_COLS * SHEET_TILE_W, rows * th), (30, 30, 30))
            for j, t in enumerate(page):
                sheet.paste(t, ((j % SHEET_COLS) * SHEET_TILE_W, (j // SHEET_COLS) * th))
            path = out_dir / f"{session.name}_b{b['battle']}_p{p // per_page + 1}.png"
            sheet.save(path)
            written.append(path)
    return written


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tower_audit.py", description=(__doc__ or "").split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="The doc at the top of tools/tower_audit.py describes the "
                                        "fields read, the report and each check.")
    ap.add_argument("path", help="a session folder or its frames.jsonl, or a folder holding sessions")
    ap.add_argument("--all", action="store_true", help="every session under PATH (default: the newest)")
    ap.add_argument("--check", default=None, metavar="NAMES",
                    help=f"comma-separated checks, exit 1 when one fails: {','.join(CHECKS)}")
    ap.add_argument("--expect-overlay", default="", metavar="NAMES",
                    help="--check overlay: image names (00000420.jpg,...) that must be flagged")
    ap.add_argument("--sheet", default=None, metavar="DIR", help="draw the read boxes on frames into DIR")
    ap.add_argument("--every", type=float, default=10.0, metavar="S", help="--sheet: one frame every S s (10)")
    ap.add_argument("--around", type=float, default=2.0, metavar="S",
                    help="--sheet: frames within S s of each accepted zero (2)")
    ap.add_argument("--images", default=None, metavar="DIR", help="--sheet: where a replay's images are")
    ap.add_argument("--regions", default=None, metavar="FILE",
                    help="the HUD regions the session read with (default: the PC's file, else built in)")
    ap.add_argument("--json", default=None, metavar="FILE", help="write the whole report as JSON")
    return ap


def main(argv: Sequence[str] | None = None, out=print) -> int:
    args = build_parser().parse_args(argv)
    try:
        checks = parse_checks(args.check)
    except ValueError as exc:
        out(f"--check: {exc}")
        return EXIT_INPUT
    folders = find_sessions(args.path, args.all)
    if not folders:
        out(f"no session (a folder with {FRAMES_FILE}) at {args.path}")
        return EXIT_INPUT
    regions = label_refs = None
    if args.regions or args.sheet:
        from scripts.vision.hud import load_regions
        try:
            regions = load_regions(args.regions)
        except Exception as exc:                       # missing, unreadable or malformed
            out(f"--regions: {type(exc).__name__}: {exc}")
            return EXIT_INPUT
        label_refs = regions.label_refs()[0]
    sessions = [load_session(f) for f in folders]
    report: dict[str, Any] = {"schema": SCHEMA, "path": str(args.path),
                              "sessions": [audit_session(s, label_refs=label_refs) for s in sessions]}
    expect = [n.strip() for n in args.expect_overlay.split(",") if n.strip()]
    report["checks"] = run_checks(report, checks, expect)
    report["ok"] = all(c["ok"] for c in report["checks"].values())
    for line in format_report(report):
        out(line)
    if args.sheet:
        try:
            import PIL  # noqa: F401
        except ImportError:
            out("--sheet needs Pillow (pip install pillow)")
            return EXIT_INPUT
        for s, a in zip(sessions, report["sessions"]):
            for p in write_sheets(s, a, Path(args.sheet), regions=regions,
                                  images=Path(args.images) if args.images else None,
                                  every_s=args.every, around_s=args.around, log=out):
                out(f"sheet -> {p}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        out(f"report -> {args.json}")
    return EXIT_OK if report["ok"] else EXIT_CHECK


if __name__ == "__main__":
    sys.exit(main())
