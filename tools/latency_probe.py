r"""Latency probe: from the bot's decision to its play running in the game, measured on this rig (V11).

    .venvs\buildabot\Scripts\python.exe tools\latency_probe.py                 every session under reports\vision_play
    .venvs\buildabot\Scripts\python.exe tools\latency_probe.py --last 1        the newest session only
    .venvs\buildabot\Scripts\python.exe tools\latency_probe.py reports\vision_play\vision_play_20261030_201500 --json reports\vision\latency.json

Reads the sessions scripts/vision_play.py writes (reports/vision_play/<session>/frames.jsonl,
the format of scripts/vision/record.py), or a perception replay tools/vision_replay.py wrote
(it copies the recording's taps). It only reads logs: nothing here taps, captures or touches
the emulator. Only sessions with taps (vision_play --tap, while you watch) measure D; a dry run
still gives the pipeline latencies.

D, THE DECISION-TO-RUN DELAY. A "tap" record holds the tick of the frame the bot decided on
(the HUD clock's tick for those pixels) and the card. D is how long after that the play RUNS in
the game: the card cycles (the HUD's next card changes) and the unit is created. That is what
tools/rs_eval.py --delay and tools/rs_rows.py --obs-delay model (the engine runs a play D ticks
after its tap, and adds the unit's deploy time itself). Per tap, the run is the earlier of
  * its next-card change: the HUD shows the plays' changes in tap order, so each change goes
    to the oldest tap whose card left its slot (the client took it) and whose change was not
    counted yet, at least run_min_ticks (20) after that tap and at most next_wait_ticks (45)
    (the second play of a quick pair shows its change 30-40 ticks after its tap); a change
    with no such tap is no play of ours (a human's in MANUAL);
  * its real unit's first sighting: the earliest own track (team 1 in more of the frames it
    was reported in than team 0) of a label the card puts down (data/vision/classes.json
    through vision_classes.ClassTable.deck_labels([card])) first seen run_min_ticks..
    min(max_lag_ticks, 45) after the decision tick, within radius_mt (4000) of the aimed
    tile for troops and buildings (it drops in from 2-3 tiles above the aim), anywhere for a
    spell (a Fireball is seen as it leaves the king).
D = the median over every tap with a run, rounded half up; p10 / p90 give the spread. On the
owner's first --tap friendly (2026-11-01) D is 25 (single taps 23-27).

THE FIRST SIGHTING IS NOT D. Six of the eight Hog 2.6 cards show a translucent placement
preview where they were tapped from about 3 ticks after the tap (the "ghost"), which the
detector boxes as our unit: the first own sighting (match_sighting: the earliest such track
min_lag..max_lag ticks after the decision, -2..60, the window scripts/vision/control.py and
events.py use, no earlier tap claiming it; the other units of the same play claimed with it)
is reported apart ("lag_ticks", "by_card", and "ghost_lag_ticks" for a sighting before
run_min_ticks within ghost_radius_mt of the aim), and so is the event deriver's own match
(scripts/vision/events.py OwnLanding: the same first sighting), with the taps it reported
missed; own plays it matched to no tap are counted and left out. Neither is the --delay.
The slot refilling (another card in the tapped slot, run_min_ticks or more after the tap,
about 2 ticks after the next-card change) is reported too.

LATENCY, from the "ms" vision_play records with every battle frame: stale (capture -> the
runner read the frame), perceive, decide, tap (the adb write), capture_to_decision
(stale + perceive + decide), capture_to_tap and decision_to_tap (capture_to_tap -
capture_to_decision); the processed-frame interval and the decision interval in ticks. Wall
clock tap -> sighting and capture -> sighting take the sighting's time from the frames around
its tick (the vision clock, perf_counter).

Statistics: n, mean, median, nearest-rank p10 and p90, min, max (the p90 of
scripts/vision/events.py's LandingStats). D is the median rounded half up.

Not here: fitting the perception noise profile (V11's fit_noise_profile) needs E17's
profile format, which is not built yet.

Imports numpy (through scripts/vision/record.py) and the standard library.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:          # vision_classes, also when imported as tools.latency_probe
    sys.path.insert(0, str(ROOT / "tools"))
if str(ROOT) not in sys.path:                    # scripts.vision
    sys.path.append(str(ROOT))

from scripts.vision.record import FRAMES_FILE, VisionLogReader, list_sessions  # noqa: E402
from scripts.vision.types import ENEMY_TEAM, OWN_TEAM, SCREEN_BATTLE  # noqa: E402

SCHEMA = "hastycr.latency_probe/1"
REPORT_ROOT = ROOT / "reports" / "vision_play"
TICKS_PER_S = 20
MS_PER_TICK = 1000.0 / TICKS_PER_S
#: live_play.py's LEAD: decision to board, measured with the memory reader (2026-10-10).
LEGACY_LEAD_TICKS = 26
KIND_SPELL = "spell"
#: The per-frame latency stages vision_play records ("ms"), then the derived ones.
RECORDED_STAGES = ("stale", "perceive", "decide", "tap", "capture_to_tap")
STAGES = RECORDED_STAGES + ("capture_to_decision", "decision_to_tap")


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _rank(s: Sequence[float], pct: int) -> float:
    """Nearest-rank percentile `pct` (integer percent) of the sorted values `s`."""
    return s[max(0, -(-len(s) * int(pct) // 100) - 1)]


def describe(values: Iterable[float | None]) -> dict[str, Any] | None:
    """n, mean, median, p10, p90 (nearest rank), min, max of the values that are not None;
    None when there are none."""
    s = sorted(float(v) for v in values if v is not None)
    if not s:
        return None
    return {"n": len(s), "mean": statistics.fmean(s), "median": statistics.median(s),
            "p10": _rank(s, 10), "p90": _rank(s, 90), "min": s[0], "max": s[-1]}


def half_up(x: float) -> int:
    """Round half up (24.5 -> 25), not to even."""
    return int(math.floor(float(x) + 0.5))


# ---------------------------------------------------------------------------
# configuration and the card -> label table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeConfig:
    """Matching parameters: ticks and world millitiles."""
    min_lag_ticks: int = -2            # control.MIN_LAG_TICKS: clock-read noise
    max_lag_ticks: int = 60            # control.MAX_LAG_TICKS: unseen after 3 s is a miss
    radius_mt: float | None = 4000.0   # events.EventConfig.landing_radius_mt (troops, buildings)
    play_window_ticks: int = 20        # events.PLAY_WINDOW: the other units of one play ...
    merge_radius_mt: float = 5000.0    # ... this close (events.EventConfig.merge_radius_mt)
    min_samples: int = 5               # plays with a run before D counts as measured
    run_min_ticks: int = 20            # a run sign sooner than this after the tap is not its run
    next_wait_ticks: int = 45          # a tap's next-card change may come this late
    drop_in_ticks: int = 45            # its real unit first seen at most this late (and max_lag)
    ghost_radius_mt: float = 2000.0    # a sighting before run_min_ticks this close: the ghost
    accept_ticks: int = 8              # a card still in its slot this long after: not taken

    def __post_init__(self):
        if self.min_lag_ticks > self.max_lag_ticks:
            raise ValueError("ProbeConfig needs min_lag_ticks <= max_lag_ticks")
        if self.radius_mt is not None and not self.radius_mt > 0:
            raise ValueError("ProbeConfig.radius_mt must be > 0 (or None: no radius)")


class CardClasses:
    """Card name -> the track labels its play shows, and the card's kind.

    labels / kinds: explicit tables (tests, or overrides); every other name is looked up in
    `table` (a vision_classes.ClassTable): deck_labels([name]) and cards[name]["kind"]
    ("troop" / "building" / "spell"). CardClasses.load() reads data/vision/classes.json.
    """

    def __init__(self, labels: Mapping[str, Iterable[str]] | None = None,
                 kinds: Mapping[str, str] | None = None, *, table: Any = None):
        self._labels = {str(k): frozenset(str(c) for c in v) for k, v in (labels or {}).items()}
        self._kinds = {str(k): str(v) for k, v in (kinds or {}).items()}
        self.table = table

    @classmethod
    def load(cls, path: str | Path | None = None) -> "CardClasses":
        import vision_classes
        return cls(table=vision_classes.load(path))

    def labels(self, name: str | None) -> frozenset[str]:
        if name is None:
            return frozenset()
        name = str(name)
        hit = self._labels.get(name)
        if hit is None:
            hit = frozenset()
            if self.table is not None:
                try:
                    hit = frozenset(self.table.deck_labels([name]))
                except Exception:
                    hit = frozenset()
            self._labels[name] = hit
        return hit

    def kind(self, name: str | None) -> str | None:
        if name is None:
            return None
        name = str(name)
        if name in self._kinds:
            return self._kinds[name]
        if self.table is not None:
            info = getattr(self.table, "cards", {}).get(name) or {}
            kind = info.get("kind")
            return None if kind is None else str(kind)
        return None


# ---------------------------------------------------------------------------
# reading a session
# ---------------------------------------------------------------------------
@dataclass
class TapRec:
    """One "tap" record: the decision tick, the card and where it went."""
    session: str
    battle: int | None
    tick: int
    card: int
    name: str | None
    slot: int
    t: float | None                     # vision clock when it was logged (just after the tap)
    ms: dict[str, float] = field(default_factory=dict)
    tile: tuple[int, int] | None = None  # the SCREEN tile tapped (world = screen for team 1)

    @property
    def aim_mt(self) -> tuple[float, float] | None:
        if self.tile is None:
            return None
        return (self.tile[0] + 0.5) * 1000.0, (self.tile[1] + 0.5) * 1000.0


@dataclass
class Sighting:
    """One track of one battle, over the frames it was reported in."""
    uid: int
    first_tick: int
    x_mt: float                         # where it was first reported
    y_mt: float
    t_first_report: float
    labels: Counter = field(default_factory=Counter)
    teams: Counter = field(default_factory=Counter)
    claimed: bool = False

    @property
    def label(self) -> str:
        return self.labels.most_common(1)[0][0]

    @property
    def own(self) -> bool:
        return self.teams[OWN_TEAM] > self.teams[ENEMY_TEAM]


@dataclass
class BattleLog:
    session: str
    battle: int
    frames: list[tuple[int | None, float]] = field(default_factory=list)   # (tick, t_capture)
    sightings: dict[tuple[int, int], Sighting] = field(default_factory=dict)
    decisions: dict[int, float] = field(default_factory=dict)             # tick -> t_capture
    decision_ticks: list[int] = field(default_factory=list)
    landings: list[Any] = field(default_factory=list)                     # OwnLanding
    taps: list[TapRec] = field(default_factory=list)
    huds: list[tuple[int, tuple[int, ...], int]] = field(default_factory=list)   # (tick, hand, next) a frame

    @property
    def t_range(self) -> tuple[float, float] | None:
        if not self.frames:
            return None
        return self.frames[0][1], self.frames[-1][1]

    def see(self, tr: Any, t: float) -> None:
        # a uid only names one track within a battle, with one first_seen_tick: a pair that
        # differs is another track (the tracker was reset on the battle's first frame)
        key = (int(tr.uid), int(tr.first_seen_tick))
        s = self.sightings.get(key)
        if s is None:
            s = self.sightings[key] = Sighting(int(tr.uid), int(tr.first_seen_tick), float(tr.x_mt),
                                               float(tr.y_mt), float(t))
        s.labels[str(tr.cls)] += 1
        s.teams[int(tr.team)] += 1

    def time_at(self, tick: int) -> float | None:
        """The vision-clock time of battle tick `tick`: the frame showing it, else
        interpolated between the frames around it; None outside the frames."""
        pts = [(k, t) for k, t in self.frames if k is not None]
        for j, (k, t) in enumerate(pts):
            if k < tick:
                continue
            if k == tick:
                return t
            if j == 0:
                return None
            k0, t0 = pts[j - 1]
            return t0 + (tick - k0) / (k - k0) * (t - t0)
        return None


@dataclass
class SessionLog:
    name: str
    path: Path
    frames: int = 0
    battles: dict[int, BattleLog] = field(default_factory=dict)
    taps: list[TapRec] = field(default_factory=list)
    unassigned: list[TapRec] = field(default_factory=list)
    ms: dict[str, list[float]] = field(default_factory=lambda: {k: [] for k in STAGES})
    frame_intervals_ms: list[float] = field(default_factory=list)
    control_unseen: int = 0             # control.py "bot_tap_unseen" records
    human_play_records: int = 0         # control.py "human_play" records
    bad_lines: int = 0

    def battle(self, n: int) -> BattleLog:
        b = self.battles.get(n)
        if b is None:
            b = self.battles[n] = BattleLog(self.name, n)
        return b


def _num(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _add_ms(out: dict[str, list[float]], ms: Mapping[str, Any]) -> None:
    for k in RECORDED_STAGES:
        v = _num(ms.get(k))
        if v is not None:
            out[k].append(v)
    parts = [_num(ms.get(k)) for k in ("stale", "perceive", "decide")]
    if all(p is not None for p in parts):
        c2d = sum(parts)
        out["capture_to_decision"].append(c2d)
        c2t = _num(ms.get("capture_to_tap"))
        if c2t is not None:
            out["decision_to_tap"].append(c2t - c2d)


def _tap_record(session: str, data: Mapping[str, Any]) -> TapRec | None:
    try:
        tick, card = int(data["tick"]), int(data["card"])
    except (KeyError, TypeError, ValueError):
        return None
    tile = data.get("screen_tile")
    try:
        tile = None if tile is None else (int(tile[0]), int(tile[1]))
    except (TypeError, ValueError, IndexError):
        tile = None
    battle = data.get("battle")
    name = data.get("name")
    return TapRec(session, None if battle is None else int(battle), tick, card,
                  None if name is None else str(name), int(data.get("slot", -1)), None,
                  {k: float(v) for k, v in (data.get("ms") or {}).items() if _num(v) is not None}, tile)


def load_session(path: str | Path) -> SessionLog:
    """Read one session (its directory or frames.jsonl) into battles, tracks and taps."""
    reader = VisionLogReader(path)
    log = SessionLog(reader.session_dir.name, reader.session_dir)
    for e in reader:
        if e.kind == "frame":
            log.frames += 1
            extra = e.data.get("extra") or {}
            battle = extra.get("battle")
            if battle is None or e.frame is None:
                continue
            vf = e.frame
            b = log.battle(int(battle))
            _add_ms(log.ms, extra.get("ms") or {})
            if b.frames:
                gap = (vf.t_capture - b.frames[-1][1]) * 1000.0
                if gap > 0:
                    log.frame_intervals_ms.append(gap)
            b.frames.append((vf.tick, float(vf.t_capture)))
            if extra.get("decision") is not None and vf.tick is not None:
                b.decisions.setdefault(int(vf.tick), float(vf.t_capture))
                b.decision_ticks.append(int(vf.tick))
            for tr in vf.tracks:
                b.see(tr, vf.t_capture)
            b.landings.extend(vf.events.own_landings)
            if vf.tick is not None and vf.screen == SCREEN_BATTLE:
                b.huds.append((int(vf.tick), tuple(int(c) for c in vf.hud.hand), int(vf.hud.next)))
        elif e.kind == "tap":
            tap = _tap_record(log.name, e.data.get("data") or {})
            if tap is not None:
                tap.t = _num(e.data.get("t"))
                log.taps.append(tap)
        elif e.kind == "bot_tap_unseen":
            log.control_unseen += 1
        elif e.kind == "human_play":
            log.human_play_records += 1
    log.bad_lines = reader.bad_lines
    for tap in log.taps:
        b = log.battles.get(tap.battle) if tap.battle is not None else None
        if b is None and tap.t is not None:           # no battle field: the battle at that time
            for cand in log.battles.values():
                span = cand.t_range
                if span is not None and span[0] <= tap.t <= span[1] + 1.0:
                    b = cand
                    break
        if b is None:
            log.unassigned.append(tap)
        else:
            b.taps.append(tap)
    return log


def find_sessions(paths: Sequence[str | Path], last: int | None = None) -> list[Path]:
    """Session directories: each path is a session (or its frames.jsonl) or a root of them
    (its newest `last` only, by name, when given)."""
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if (p.is_file() and p.suffix == ".jsonl") or (p / FRAMES_FILE).is_file():
            out.append(p)
            continue
        found = list_sessions(p)
        out.extend(found[-int(last):] if last else found)
    return out


# ---------------------------------------------------------------------------
# matching taps to sightings
# ---------------------------------------------------------------------------
def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def match_sighting(b: BattleLog, tap: TapRec, labels: frozenset[str], kind: str | None,
                   cfg: ProbeConfig) -> Sighting | None:
    """The first own sighting of `tap`'s play (module doc), claimed with its co-units."""
    if not labels:
        return None
    aim = tap.aim_mt
    radius = None if kind == KIND_SPELL else cfg.radius_mt
    best, best_key = None, None
    for s in b.sightings.values():
        if s.claimed or not s.own or s.label not in labels:
            continue
        lag = s.first_tick - tap.tick
        if not cfg.min_lag_ticks <= lag <= cfg.max_lag_ticks:
            continue
        d = 0.0 if aim is None else _dist(aim, (s.x_mt, s.y_mt))
        if radius is not None and d > radius:
            continue
        key = (s.first_tick, d, s.uid)
        if best_key is None or key < best_key:
            best, best_key = s, key
    if best is None:
        return None
    best.claimed = True
    for s in b.sightings.values():
        if (not s.claimed and s.own and s.label == best.label
                and 0 <= s.first_tick - best.first_tick < cfg.play_window_ticks
                and _dist((s.x_mt, s.y_mt), (best.x_mt, best.y_mt)) <= cfg.merge_radius_mt):
            s.claimed = True
    return best


def unit_sighting(b: BattleLog, tap: TapRec, labels: frozenset[str], kind: str | None,
                  cfg: ProbeConfig) -> Sighting | None:
    """`tap`'s real unit (module doc, D): the earliest own sighting of its labels first seen
    run_min_ticks..min(max_lag_ticks, drop_in_ticks) after it, within radius_mt of the aim (a
    spell: anywhere). Claims nothing (it is often the first sighting itself: a Hog Rider)."""
    if not labels:
        return None
    aim = tap.aim_mt
    radius = None if kind == KIND_SPELL else cfg.radius_mt
    hi = min(cfg.max_lag_ticks, cfg.drop_in_ticks)
    best = None
    for s in b.sightings.values():
        if not s.own or s.label not in labels or not cfg.run_min_ticks <= s.first_tick - tap.tick <= hi:
            continue
        if radius is not None and aim is not None and _dist(aim, (s.x_mt, s.y_mt)) > radius:
            continue
        if best is None or (s.first_tick, s.uid) < (best.first_tick, best.uid):
            best = s
    return best


def hud_marks(b: BattleLog, taps: Sequence[TapRec], cfg: ProbeConfig) -> dict[int, dict[str, Any]]:
    """Per tap (by its index in `taps`, oldest first): "accepted" (its card left its slot within
    accept_ticks, or no HUD reading says it stayed), "next" (the tick of its next-card change:
    module doc, D) and "refill" (the first tick run_min_ticks or more after it whose slot shows
    another card), None when not seen. Without HUD readings: accepted, no marks."""
    out: dict[int, dict[str, Any]] = {i: {"accepted": True, "next": None, "refill": None}
                                      for i in range(len(taps))}
    huds = sorted(b.huds, key=lambda h: h[0])           # stable: frames of a tick in their order
    if not huds:
        return out
    for i, tap in enumerate(taps):
        if not 0 <= tap.slot < len(huds[0][1]):
            continue
        early = [hand[tap.slot] for tick, hand, _ in huds if 0 < tick - tap.tick <= cfg.accept_ticks + 2]
        late = [tick for tick, _, _ in huds if cfg.accept_ticks <= tick - tap.tick <= cfg.accept_ticks + 2]
        if late and early and all(c == tap.card for c in early):
            out[i]["accepted"] = False                 # still in its slot: the client did not take it
            continue
        for tick, hand, _ in huds:
            if tick - tap.tick >= cfg.run_min_ticks and hand[tap.slot] >= 0 and hand[tap.slot] != tap.card:
                out[i]["refill"] = tick
                break
    taken: set[int] = set()
    last = None
    for tick, _, nxt in huds:
        if nxt < 0:
            continue
        if last is not None and nxt != last:
            queue = [i for i, tap in enumerate(taps) if out[i]["accepted"] and i not in taken
                     and 0 < tick - tap.tick <= cfg.next_wait_ticks]
            if queue:
                head = min(queue, key=lambda i: (taps[i].tick, i))
                if tick - taps[head].tick >= cfg.run_min_ticks:
                    taken.add(head)
                    out[head]["next"] = tick
        last = nxt
    return out


def battle_plays(b: BattleLog, classes: CardClasses, cfg: ProbeConfig) -> tuple[list[dict], Counter]:
    """One record per tap of battle `b`, oldest first, and the deriver's counts."""
    for s in b.sightings.values():                    # analyse() may run twice on one log
        s.claimed = False
    landed: dict[tuple[int, int], Any] = {}
    missed: set[tuple[int, int]] = set()
    counts: Counter = Counter()
    for ol in b.landings:
        if int(ol.slot) < 0:
            counts["human"] += 1
            continue
        key = (int(ol.card), int(ol.tap_tick))
        if ol.land_tick is None:
            missed.add(key)
        else:
            landed.setdefault(key, ol)
    out = []
    order = sorted(b.taps, key=lambda tp: (tp.tick, tp.t if tp.t is not None else 0.0))
    marks = hud_marks(b, order, cfg)
    for k, tap in enumerate(order):
        kind = classes.kind(tap.name)
        labels = classes.labels(tap.name)
        s = match_sighting(b, tap, labels, kind, cfg)
        u = unit_sighting(b, tap, labels, kind, cfg)
        m = marks[k]
        lags = {"next": None if m["next"] is None else m["next"] - tap.tick,
                "unit": None if u is None else u.first_tick - tap.tick,
                "refill": None if m["refill"] is None else m["refill"] - tap.tick}
        runs = [v for v in (lags["next"], lags["unit"]) if v is not None]
        ghost = None
        if s is not None and s.first_tick - tap.tick < cfg.run_min_ticks:
            aim = tap.aim_mt
            if aim is None or _dist(aim, (s.x_mt, s.y_mt)) <= cfg.ghost_radius_mt:
                ghost = s.first_tick - tap.tick
        key = (tap.card, tap.tick)
        ol = landed.pop(key, None)
        state = "landed" if ol is not None else "missed" if key in missed else None
        if state is not None:
            counts[f"deriver_{state}"] += 1
        t_sight = None if s is None else b.time_at(s.first_tick)
        c2t = tap.ms.get("capture_to_tap")
        t_cap = b.decisions.get(tap.tick)
        if t_cap is None and tap.t is not None and c2t is not None:
            t_cap = tap.t - c2t / 1000.0
        out.append({
            "session": tap.session, "battle": b.battle, "tick": tap.tick, "card": tap.card,
            "name": tap.name, "slot": tap.slot, "kind": kind or "unknown",
            "lag_ticks": None if s is None else s.first_tick - tap.tick,
            "uid": None if s is None else s.uid, "label": None if s is None else s.label,
            "accepted": bool(m["accepted"]), "run_lag_ticks": min(runs) if runs else None,
            "next_lag_ticks": lags["next"], "unit_lag_ticks": lags["unit"],
            "refill_lag_ticks": lags["refill"], "ghost_lag_ticks": ghost,
            "deriver": state,
            "deriver_lag_ticks": None if ol is None or ol.land_tick is None else int(ol.land_tick) - tap.tick,
            "capture_to_tap_ms": c2t,
            "tap_to_sight_ms": None if t_sight is None or tap.t is None else (t_sight - tap.t) * 1000.0,
            "capture_to_sight_ms": None if t_sight is None or t_cap is None else (t_sight - t_cap) * 1000.0,
        })
    return out, counts


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def analyse(sessions: Sequence[SessionLog], classes: CardClasses | None = None,
            config: ProbeConfig | None = None) -> dict[str, Any]:
    """The probe's report over `sessions` (module doc); "plays" lists every tap."""
    cfg = config or ProbeConfig()
    classes = classes if classes is not None else CardClasses.load()
    plays: list[dict] = []
    counts: Counter = Counter()
    ms: dict[str, list[float]] = {k: [] for k in STAGES}
    intervals: list[float] = []
    decision_gaps: list[int] = []
    for log in sessions:
        for k in STAGES:
            ms[k].extend(log.ms[k])
        intervals.extend(log.frame_intervals_ms)
        counts["control_unseen"] += log.control_unseen
        counts["human_play_records"] += log.human_play_records
        counts["unassigned_taps"] += len(log.unassigned)
        for n in sorted(log.battles):
            b = log.battles[n]
            got, c = battle_plays(b, classes, cfg)
            plays.extend(got)
            counts.update(c)
            ticks = sorted(set(b.decision_ticks))
            decision_gaps.extend(y - x for x, y in zip(ticks, ticks[1:]))
        for tap in log.unassigned:
            plays.append({"session": tap.session, "battle": None, "tick": tap.tick, "card": tap.card,
                          "name": tap.name, "slot": tap.slot, "kind": classes.kind(tap.name) or "unknown",
                          "lag_ticks": None, "uid": None, "label": None, "accepted": True,
                          "run_lag_ticks": None, "next_lag_ticks": None, "unit_lag_ticks": None,
                          "refill_lag_ticks": None, "ghost_lag_ticks": None, "deriver": None,
                          "deriver_lag_ticks": None, "capture_to_tap_ms": tap.ms.get("capture_to_tap"),
                          "tap_to_sight_ms": None, "capture_to_sight_ms": None})
    units = [p for p in plays if p["kind"] != KIND_SPELL]
    spells = [p for p in plays if p["kind"] == KIND_SPELL]
    track_lags = [p["lag_ticks"] for p in units if p["lag_ticks"] is not None]
    deriver_lags = [p["deriver_lag_ticks"] for p in units if p["deriver_lag_ticks"] is not None]
    values = [p["run_lag_ticks"] for p in plays if p["run_lag_ticks"] is not None]
    source = "run" if values else None
    stats = describe(values)
    by_card: dict[str, list[int]] = {}
    for p in plays:
        if p["lag_ticks"] is not None:
            by_card.setdefault(str(p["name"] if p["name"] is not None else p["card"]), []).append(p["lag_ticks"])
    d = {"ticks": None if stats is None else half_up(stats["median"]), "source": source,
         "n": len(values), "enough": len(values) >= cfg.min_samples, "min_samples": cfg.min_samples,
         "median": None if stats is None else stats["median"],
         "p10": None if stats is None else stats["p10"], "p90": None if stats is None else stats["p90"],
         "legacy_lead_ticks": LEGACY_LEAD_TICKS}
    seen = sum(1 for p in plays if p["lag_ticks"] is not None)
    return {
        "schema": SCHEMA,
        "sessions": [str(log.path) for log in sessions],
        "frames": sum(log.frames for log in sessions),
        "battles": sum(len(log.battles) for log in sessions),
        "bad_lines": sum(log.bad_lines for log in sessions),
        "D": d,
        "counts": {"taps": len(plays), "seen": seen, "unseen": len(plays) - seen,
                   "deriver_landed": counts["deriver_landed"], "deriver_missed": counts["deriver_missed"],
                   "human_plays": counts["human"], "human_play_records": counts["human_play_records"],
                   "control_unseen": counts["control_unseen"],
                   "unassigned_taps": counts["unassigned_taps"]},
        "run_ticks": {"run": describe(values), "next": describe(p["next_lag_ticks"] for p in plays),
                      "unit": describe(p["unit_lag_ticks"] for p in plays),
                      "refill": describe(p["refill_lag_ticks"] for p in plays),
                      "ghost": describe(p["ghost_lag_ticks"] for p in plays),
                      "ghost_taps": sum(1 for p in plays if p["ghost_lag_ticks"] is not None),
                      "not_accepted": sum(1 for p in plays if not p["accepted"])},
        "lag_ticks": {"tracks": describe(track_lags), "deriver": describe(deriver_lags),
                      "spells_tracks": describe(p["lag_ticks"] for p in spells),
                      "spells_deriver": describe(p["deriver_lag_ticks"] for p in spells),
                      "by_card": {k: describe(v) for k, v in sorted(by_card.items())}},
        "wall_ms": {"tap_to_sight": describe(p["tap_to_sight_ms"] for p in units),
                    "capture_to_sight": describe(p["capture_to_sight_ms"] for p in units)},
        "capture_to_sight_ticks": describe(None if p["capture_to_sight_ms"] is None
                                           else p["capture_to_sight_ms"] / MS_PER_TICK for p in units),
        "pipeline_ms": {k: describe(ms[k]) for k in STAGES},
        "frame_interval_ms": describe(intervals),
        "decision_interval_ticks": describe(decision_gaps),
        "config": asdict(cfg),
        "plays": plays,
    }


def _md(d: Mapping[str, Any] | None, key: str = "median", fmt: str = "{:.1f}") -> str:
    return "-" if not d else fmt.format(d[key])


def format_report(rep: Mapping[str, Any]) -> list[str]:
    """The printed summary of an analyse() report."""
    c, d, lag = rep["counts"], rep["D"], rep["lag_ticks"]
    lines = [f"latency probe: {len(rep['sessions'])} session(s), {rep['battles']} battle(s), "
             f"{rep['frames']} frames; {c['taps']} taps ({c['seen']} seen on the board, {c['unseen']} not)"]
    run = rep["run_ticks"]
    if d["ticks"] is None:
        lines.append("D: not measured (no tap was seen to run: no next-card change and no real unit; a "
                     "dry run taps nothing: run vision_play with --tap while you watch, with the detector)")
    else:
        lines.append(f"D = {d['ticks']} ticks: median {d['median']:g} of {d['n']} plays, the decided frame -> "
                     f"the play running (its next-card change or its real unit's first sighting); p10 "
                     f"{d['p10']:g}, p90 {d['p90']:g}; live_play.LEAD was {d['legacy_lead_ticks']}")
        lines.append(f"  -> the tap-to-run delay tools/rs_eval.py --delay {d['ticks']} and tools/rs_rows.py "
                     f"--obs-delay model (the deploy time comes after it)")
        lines.append(f"  run marks: next-card change median {_md(run['next'], fmt='{:g}')}, real unit "
                     f"{_md(run['unit'], fmt='{:g}')}, slot refilled {_md(run['refill'], fmt='{:g}')}")
        if not d["enough"]:
            lines.append(f"  (only {d['n']} plays: at least {d['min_samples']} before D counts as measured)")
    lines.append(f"  first sighting (not D: most cards show a placement preview first): median "
                 f"{_md(lag['tracks'], fmt='{:g}')}; the preview seen on {run['ghost_taps']} taps, median "
                 f"{_md(run['ghost'], fmt='{:g}')}")
    lines.append(f"  deriver: median {_md(lag['deriver'], fmt='{:g}')} (n {lag['deriver']['n'] if lag['deriver'] else 0}), "
                 f"{c['deriver_landed']} landed, {c['deriver_missed']} missed; human plays {c['human_plays']}; "
                 f"spells median {_md(lag['spells_tracks'], fmt='{:g}')}")
    if lag["by_card"]:
        lines.append("  first sighting per card: " + ", ".join(f"{k} {v['median']:g} (n {v['n']})"
                                                             for k, v in lag["by_card"].items() if v))
    pm = rep["pipeline_ms"]
    parts = [f"{k} {_md(pm[k])}/{_md(pm[k], 'p90')}" for k in STAGES if pm.get(k)]
    if parts:
        lines.append("pipeline ms (median/p90): " + ", ".join(parts))
    w = rep["wall_ms"]
    if w["tap_to_sight"] or w["capture_to_sight"]:
        lines.append(f"wall clock: tap -> sighting median {_md(w['tap_to_sight'], fmt='{:.0f}')} ms, "
                     f"capture -> sighting {_md(w['capture_to_sight'], fmt='{:.0f}')} ms "
                     f"(= {_md(rep['capture_to_sight_ticks'])} ticks)")
    fi, di = rep["frame_interval_ms"], rep["decision_interval_ticks"]
    if fi:
        lines.append(f"frames: interval median {fi['median']:.1f} ms ({1000.0 / fi['median']:.0f} fps), "
                     f"p90 {fi['p90']:.1f} ms; decisions every {_md(di, fmt='{:g}')} ticks (median)")
    if rep["bad_lines"]:
        lines.append(f"({rep['bad_lines']} unreadable log lines skipped)")
    return lines


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="tools/latency_probe.py",
        description="Decision-to-run delay D and pipeline latency from vision_play sessions.")
    ap.add_argument("paths", nargs="*", default=[str(REPORT_ROOT)],
                    help=f"sessions or roots of sessions (default {REPORT_ROOT})")
    ap.add_argument("--last", type=int, default=None, help="only the newest N sessions of each root")
    ap.add_argument("--classes", default=None, help="classes.json (default data/vision/classes.json)")
    ap.add_argument("--min-lag", type=int, default=ProbeConfig.min_lag_ticks, help="ticks (default -2)")
    ap.add_argument("--max-lag", type=int, default=ProbeConfig.max_lag_ticks, help="ticks (default 60)")
    ap.add_argument("--radius-mt", type=float, default=ProbeConfig.radius_mt,
                    help="a troop's sighting this close to the aimed tile (0: any distance)")
    ap.add_argument("--min-samples", type=int, default=ProbeConfig.min_samples)
    ap.add_argument("--json", default=None, help="write the full report (every play) here")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = find_sessions(args.paths, args.last)
    if not paths:
        print(f"no sessions under {', '.join(map(str, args.paths))}")
        return 1
    cfg = ProbeConfig(args.min_lag, args.max_lag, args.radius_mt or None, min_samples=args.min_samples)
    rep = analyse([load_session(p) for p in paths], CardClasses.load(args.classes), cfg)
    for line in format_report(rep):
        print(line)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
