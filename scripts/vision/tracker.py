"""Position-first unit tracker with label and team voting (V03).

The older trackers only match a detection to a track with the same class name
(scripts/brain/tracker.py:59, tools/calibration/core.py:926-927). The detector's
labels agree from one frame to the next only ~94% of the time, so every flicker
started a new track: a new uid, a fresh deploying flag and, downstream, a
phantom opponent play. This tracker matches on position first and votes on the
label and the team over the track's life.

Per update (one call per processed frame):

1. Expiry. Tracks unseen for more than ``max_age_ticks`` (tentative ones:
   ``tentative_ttl_ticks``) are dropped BEFORE association, so a stale track
   cannot capture a new unit that lands where it used to be.
2. Association. Each track predicts its position from a smoothed velocity. The
   cost of pairing track i with observation j is

       d_ij / gate_i  +  class_cost * (1 - sim)  +  team_cost * contra

   where d_ij is the distance (millitiles) to the prediction, gate_i grows with
   the time since the track was last seen (pairs beyond the gate are not
   allowed), sim is 1 for the track's reported label and otherwise the label's
   share of the track's label histogram (or a caller-given similarity), and
   contra is how strongly the side classifier contradicts a decided team.
   The assignment is Hungarian (scipy.optimize.linear_sum_assignment) when
   scipy imports, else greedy cheapest-pair-first. Unmatched observations start
   tentative tracks; a track is reported after ``min_hits`` sightings.
3. Label vote. A decayed histogram of labels weighted by detector confidence.
   The reported label changes only when a different label was seen in
   ``label_switch_frames`` consecutive frames AND outweighs every other label.
4. Team vote. Evidence is summed as a log-likelihood ratio (positive = enemy):
   the side classifier's per-frame P(enemy) (clipped), the HP-bar colour, the
   board half a track was born in, and our own taps (a new track of a tapped
   card near the aim within the tap window is ours). The team is decided when
   the sum reaches ``decide_llr`` and then held: it flips only when the sum
   crosses ``flip_llr`` the other way (the sum is capped at ``llr_cap``).
   Deck membership is a hard rule on top: a reported label that our deck
   cannot produce (``own_classes``) is an enemy.

Conventions are those of scripts/vision/types.py: world millitiles, the live
bot is RoyaleSim team 1 and defends high y; teams OWN_TEAM / ENEMY_TEAM /
UNKNOWN_TEAM; time in battle ticks (20 per second). The caller passes a
monotonic tick; when the clock is unread, derive one from the capture time.

Output is a list of scripts.vision.types.Track: confirmed tracks seen within
``coast_ticks`` (a briefly occluded unit is reported at its predicted
position). ``first_seen_tick`` is the first sighting, which gives the
deploying flag (see deploy_ticks_left). Tracks that expired on the last update
are in ``Tracker.expired`` (for the event deriver's "just-dead parent" rule).
``Tracker.last_assignment`` lists, in the order the last update's observations
came, the uid each went to: the track it was matched to, or the tentative track
it started (reported once it has ``min_hits`` sightings). Harvest tools read it to
carry a detection's box to its track (N05).

Imports only the standard library; scipy (and numpy with it) is optional.

    from scripts.vision.tracker import Tracker, observations_from_detections
    tracker = Tracker(own_classes=HOG26_CLASSES)
    tracker.note_own_tap({"hog-rider"}, x_mt, y_mt, tick)      # after each tap we send
    tracks = tracker.update(observations_from_detections(dets, geometry, size), tick)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Iterable, Sequence

from .types import ENEMY_TEAM, FORM_BASE, OWN_TEAM, UNKNOWN_TEAM, Detection, Track

SOLVERS = ("auto", "scipy", "greedy")

TICK_MS = 50.0                 # RoyaleSim runs 20 ticks per second
MID_Y_MT = 16_000.0            # the river's centre line; we (team 1) own y > MID_Y_MT

_BIG = 1e9                     # stands in for a forbidden pair in the Hungarian matrix

#: A cost-matrix entry that forbids the pair.
FORBIDDEN = math.inf


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------
def _load_scipy_lsa() -> Callable | None:
    """scipy.optimize.linear_sum_assignment, or None when scipy does not import."""
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:          # ImportError, or a scipy built for another numpy
        return None
    return linear_sum_assignment


def scipy_available() -> bool:
    """True when the Hungarian solver (scipy) can be used."""
    return _load_scipy_lsa() is not None


def resolve_solver(solver: str = "auto") -> str:
    """'scipy' or 'greedy' for a requested solver.

    "auto" picks scipy when it imports and greedy otherwise; "scipy" raises
    ImportError without it; "greedy" never touches scipy.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, not {solver!r}")
    if solver == "greedy":
        return "greedy"
    if _load_scipy_lsa() is None:
        if solver == "scipy":
            raise ImportError("solver='scipy' needs scipy (pip install scipy); use 'auto' or 'greedy'")
        return "greedy"
    return "scipy"


def _assign_greedy(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    pairs = sorted((c, i, j) for i, row in enumerate(cost) for j, c in enumerate(row) if math.isfinite(c))
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    out = []
    for _c, i, j in pairs:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        out.append((i, j))
    return sorted(out)


def _assign_scipy(cost: Sequence[Sequence[float]], lsa: Callable) -> list[tuple[int, int]]:
    import numpy as np

    c = np.asarray(cost, dtype=np.float64)
    allowed = np.isfinite(c)
    if not allowed.any():
        return []
    rows, cols = lsa(np.where(allowed, c, _BIG))
    return sorted((int(r), int(k)) for r, k in zip(rows, cols) if allowed[r, k])


def assign(cost: Sequence[Sequence[float]], solver: str = "auto") -> list[tuple[int, int]]:
    """A minimum-cost one-to-one matching of rows (tracks) to columns (observations).

    `cost` is a rectangular list of rows; FORBIDDEN (math.inf) forbids a pair.
    Returns (row, column) pairs sorted by row; forbidden pairs never appear.
    The Hungarian solver first maximises the number of allowed pairs, then
    minimises their total cost; greedy takes the cheapest remaining pair first.
    """
    return _assign(cost, resolve_solver(solver), None)


def _assign(cost, kind: str, lsa: Callable | None) -> list[tuple[int, int]]:
    if not cost or not len(cost[0]):
        return []
    if kind == "scipy":
        lsa = lsa or _load_scipy_lsa()
        if lsa is not None:
            return _assign_scipy(cost, lsa)
    return _assign_greedy(cost)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Observation:
    """One unit sighting in world millitiles, the tracker's input.

    side_prob is the side classifier's P(the unit is the OPPONENT's); 0.5 means
    no opinion. bar_team is the team the HP-bar colour implies (UNKNOWN_TEAM
    when no bar was read). hp_frac is the HP-bar fill, None when unread.
    """
    cls: str
    x_mt: float
    y_mt: float
    conf: float = 1.0
    side_prob: float = 0.5
    form: int = FORM_BASE
    hp_frac: float | None = None
    bar_team: int = UNKNOWN_TEAM


def observations_from_detections(detections: Iterable[Detection], geometry: Any,
                                 size: Sequence[int] | None = None, *,
                                 flying: Collection[str] = ()) -> list[Observation]:
    """Detector boxes -> Observations, through an ArenaGeometry.

    Uses a detection's ground_px when set, else the geometry's ground point of
    its box (flying classes use the geometry's flyer offset). HP fraction and
    bar colour are left unset; the HP-bar reader fills them in.
    """
    out = []
    for d in detections:
        if d.ground_px is not None:
            x, y = geometry.px_to_world(d.ground_px[0], d.ground_px[1], size)
        else:
            x, y = geometry.ground_point(d.box, size, cls=d.cls, flying=d.cls in flying)
        out.append(Observation(d.cls, float(x), float(y), float(d.conf), float(d.side_prob), int(d.form)))
    return out


def deploy_ticks_left(track: Track, tick: int, deploy_time_ms: float) -> int:
    """RoyaleSim deploy_ticks for a track: the card's deploy time minus the track's age.

    deploy_time_ms comes from cards.json (Hog Rider 1000 -> 20 ticks); the age
    counts from the first sighting, so a unit first seen late reads as further
    along than it is.
    """
    total = int(round(float(deploy_time_ms) / TICK_MS))
    return max(0, total - (int(tick) - int(track.first_seen_tick)))


# ---------------------------------------------------------------------------
# Configuration and state
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrackerConfig:
    """Tracker parameters. Distances in world millitiles, times in battle ticks,
    team evidence in natural-log likelihood ratios (positive = enemy)."""
    # association
    gate_mt: float = 1000.0                 # gate one tick after the last sighting ...
    gate_growth_mt_per_tick: float = 50.0   # ... growing this much per unseen tick ...
    max_gate_mt: float = 2000.0             # ... up to this
    class_cost: float = 0.5                 # cost of a label the track has never shown
    team_cost: float = 0.3                  # cost of a side_prob fully against a decided team
    max_speed_mt_per_tick: float = 150.0    # clamp on the velocity estimate (Hog: ~100)
    velocity_alpha: float = 0.5             # weight of the newest velocity sample
    max_predict_ticks: float = 10.0         # never extrapolate further than this
    # life cycle
    min_hits: int = 2                       # sightings before a track is reported
    tentative_ttl_ticks: float = 6.0        # an unconfirmed track dies after this unseen
    coast_ticks: float = 6.0                # a confirmed track is reported this long unseen
    max_age_ticks: float = 20.0             # and forgotten after this
    # labels
    label_switch_frames: int = 3            # agreeing frames before the label may change
    label_decay: float = 0.97               # per-sighting decay of the label histogram
    # team
    decide_llr: float = 3.0                 # |evidence| that decides an undecided team
    flip_llr: float = 6.0                   # evidence the other way that flips a decided team
    llr_cap: float = 12.0                   # |evidence| is clipped here
    cnn_weight: float = 1.0                 # scale of the side classifier's per-frame logit
    cnn_max_llr: float = 2.0                # per-frame clip of that logit (frames correlate)
    bar_llr: float = 2.5                    # per frame with a read HP-bar colour
    spawn_llr: float = 1.0                  # once, for the half a track is born in
    spawn_deadzone_mt: float = 1000.0       # births this close to MID_Y_MT give no half cue
    tap_llr: float = 6.0                    # a new track explained by one of our taps
    tap_window_ticks: float = 60.0          # a track born up to this long after a tap
    tap_radius_mt: float = 2500.0           # and born this close to the aim
    tap_claim_age_ticks: float = 10.0       # a young track may still be claimed (late label fix)

    def __post_init__(self):
        positive = ("gate_mt", "max_gate_mt", "max_speed_mt_per_tick", "decide_llr", "llr_cap",
                    "tap_radius_mt")
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f"TrackerConfig.{name} must be > 0")
        non_negative = ("gate_growth_mt_per_tick", "class_cost", "team_cost", "max_predict_ticks",
                        "tentative_ttl_ticks", "coast_ticks", "max_age_ticks", "cnn_weight",
                        "cnn_max_llr", "bar_llr", "spawn_llr", "spawn_deadzone_mt", "tap_llr",
                        "tap_window_ticks", "tap_claim_age_ticks")
        for name in non_negative:
            if not getattr(self, name) >= 0:
                raise ValueError(f"TrackerConfig.{name} must be >= 0")
        if self.min_hits < 1 or self.label_switch_frames < 1:
            raise ValueError("TrackerConfig.min_hits and label_switch_frames must be >= 1")
        if not 0 < self.velocity_alpha <= 1 or not 0 < self.label_decay <= 1:
            raise ValueError("TrackerConfig.velocity_alpha and label_decay must be in (0, 1]")
        if not self.decide_llr <= self.flip_llr <= self.llr_cap:
            raise ValueError("TrackerConfig needs decide_llr <= flip_llr <= llr_cap")
        if self.max_gate_mt < self.gate_mt:
            raise ValueError("TrackerConfig.max_gate_mt must be >= gate_mt")


@dataclass(slots=True)
class TrackState:
    """The tracker's full view of one track (Track is the reported summary)."""
    uid: int
    x_mt: float
    y_mt: float
    first_tick: float
    last_tick: float
    label: str                                   # the reported (voted) label
    birth_x_mt: float = 0.0
    birth_y_mt: float = 0.0
    vx: float = 0.0                              # millitiles per tick
    vy: float = 0.0
    hits: int = 1
    conf: float = 0.0
    hp_frac: float | None = None
    form: int = FORM_BASE
    label_weights: dict[str, float] = field(default_factory=dict)
    form_weights: dict[int, float] = field(default_factory=dict)
    streak_label: str = ""
    streak: int = 0
    team_llr: float = 0.0                        # summed team evidence, + = enemy
    vote_team: int = UNKNOWN_TEAM                # the team the evidence decided
    cues: dict[str, float] = field(default_factory=dict)   # evidence per cue, unclipped
    tap_claimed: bool = False

    def predict(self, tick: float, max_ticks: float) -> tuple[float, float]:
        """Position extrapolated to `tick` (at most `max_ticks` ahead)."""
        dt = min(max(0.0, tick - self.last_tick), max_ticks)
        return self.x_mt + self.vx * dt, self.y_mt + self.vy * dt


@dataclass(slots=True)
class _Tap:
    classes: frozenset[str]
    x_mt: float
    y_mt: float
    tick: float
    max_units: int | None
    claimed: list[int] = field(default_factory=list)

    @property
    def full(self) -> bool:
        return self.max_units is not None and len(self.claimed) >= self.max_units


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ---------------------------------------------------------------------------
# The tracker
# ---------------------------------------------------------------------------
class Tracker:
    """Position-first multi-unit tracker with label and team voting.

    own_classes: the vision classes our deck can put on the board (None turns
    the deck-membership rule off). similarity(a, b) -> [0, 1]: optional
    label similarity for the association cost (e.g. aliases the detector
    confuses). solver: "auto" | "scipy" | "greedy" (resolve_solver).
    Call reset() at the start of every battle.
    """

    def __init__(self, config: TrackerConfig | None = None, *, solver: str = "auto",
                 own_classes: Collection[str] | None = None,
                 similarity: Callable[[str, str], float] | None = None):
        self.config = config or TrackerConfig()
        self.solver = resolve_solver(solver)
        self._lsa = _load_scipy_lsa() if self.solver == "scipy" else None
        self.own_classes = None if own_classes is None else frozenset(own_classes)
        self.similarity = similarity
        self.reset()

    def reset(self) -> None:
        """Forget every track and tap (new battle). uids restart at 1."""
        self._tracks: dict[int, TrackState] = {}
        self._taps: list[_Tap] = []
        self._next_uid = 1
        self._tick: float | None = None
        self.expired: list[Track] = []
        self.last_assignment: list[int] = []     # uid per observation of the last update
        self.backwards_ticks = 0     # updates whose tick went backwards (clamped)

    # -- inputs ---------------------------------------------------------------
    def note_own_tap(self, classes: Iterable[str], x_mt: float, y_mt: float, tick: float, *,
                     max_units: int | None = None) -> None:
        """Record a card we just played: the vision classes it puts on the board,
        the world point we aimed at and the tick the tap was sent. A track of one
        of those classes born within tap_window_ticks after `tick` and within
        tap_radius_mt of the aim gets strong own-team evidence. max_units caps
        how many tracks one tap may claim (e.g. 3 for Skeletons)."""
        self._taps.append(_Tap(frozenset(classes), float(x_mt), float(y_mt), float(tick), max_units))

    def update(self, observations: Iterable[Observation], tick: float) -> list[Track]:
        """Advance to `tick` with this frame's observations; returns the reported tracks."""
        tick = float(tick)
        if self._tick is not None and tick < self._tick:
            self.backwards_ticks += 1
            tick = self._tick
        self._tick = tick
        cfg = self.config

        self.expired = self._expire(tick)
        obs = list(observations)
        tracks = list(self._tracks.values())
        cost = [[self._cost(t, o, tick) for o in obs] for t in tracks]
        uids = [0] * len(obs)
        matched = set()
        for i, j in _assign(cost, self.solver, self._lsa):
            self._absorb(tracks[i], obs[j], tick)
            matched.add(j)
            uids[j] = tracks[i].uid
        for j, o in enumerate(obs):
            if j not in matched:
                uids[j] = self._birth(o, tick).uid
        self.last_assignment = uids

        horizon = cfg.tap_window_ticks + cfg.tap_claim_age_ticks
        self._taps = [tp for tp in self._taps if not tp.full and tick - tp.tick <= horizon]
        return self.tracks()

    # -- outputs --------------------------------------------------------------
    def tracks(self) -> list[Track]:
        """Confirmed tracks seen within coast_ticks of the last update, by uid."""
        if self._tick is None:
            return []
        cfg = self.config
        return [self._report(t, self._tick) for t in self._tracks.values()
                if t.hits >= cfg.min_hits and self._tick - t.last_tick <= cfg.coast_ticks]

    def states(self) -> list[TrackState]:
        """Every live track, tentative and coasting ones included (read only)."""
        return list(self._tracks.values())

    def state(self, uid: int) -> TrackState | None:
        return self._tracks.get(uid)

    def team_of(self, state: TrackState) -> int:
        """The team reported for a track: deck rule first, then the vote."""
        if self.own_classes is not None and state.label not in self.own_classes:
            return ENEMY_TEAM
        return state.vote_team

    # -- internals ------------------------------------------------------------
    def _report(self, t: TrackState, tick: float) -> Track:
        x, y = t.predict(tick, self.config.max_predict_ticks)
        return Track(uid=t.uid, cls=t.label, team=self.team_of(t), conf=t.conf, x_mt=x, y_mt=y,
                     hp_frac=t.hp_frac, form=t.form, first_seen_tick=int(round(t.first_tick)),
                     last_seen_tick=int(round(t.last_tick)), vx=t.vx, vy=t.vy)

    def _expire(self, tick: float) -> list[Track]:
        cfg = self.config
        gone = []
        for uid, t in list(self._tracks.items()):
            confirmed = t.hits >= cfg.min_hits
            ttl = cfg.max_age_ticks if confirmed else cfg.tentative_ttl_ticks
            if tick - t.last_tick > ttl:
                del self._tracks[uid]
                if confirmed:
                    gone.append(self._report(t, t.last_tick))
        return gone

    def _class_sim(self, t: TrackState, cls: str) -> float:
        if cls == t.label:
            return 1.0
        total = sum(t.label_weights.values())
        sim = t.label_weights.get(cls, 0.0) / total if total > 0 else 0.0
        if self.similarity is not None:
            sim = max(sim, float(self.similarity(t.label, cls)))
        return _clip(sim, 0.0, 1.0)

    def _cost(self, t: TrackState, o: Observation, tick: float) -> float:
        cfg = self.config
        px, py = t.predict(tick, cfg.max_predict_ticks)
        d = math.hypot(o.x_mt - px, o.y_mt - py)
        gate = min(cfg.max_gate_mt, cfg.gate_mt + cfg.gate_growth_mt_per_tick * max(0.0, tick - t.last_tick))
        if d > gate:
            return FORBIDDEN
        c = d / gate + cfg.class_cost * (1.0 - self._class_sim(t, o.cls))
        team = self.team_of(t)
        if team != UNKNOWN_TEAM:
            target = 1.0 if team == ENEMY_TEAM else 0.0
            # 0 when side_prob is 0.5 or agrees, 1 when it is fully against the team
            c += cfg.team_cost * max(0.0, 2.0 * abs(target - o.side_prob) - 1.0)
        return c

    def _birth(self, o: Observation, tick: float) -> TrackState:
        cfg = self.config
        uid = self._next_uid
        self._next_uid += 1
        t = TrackState(uid=uid, x_mt=float(o.x_mt), y_mt=float(o.y_mt), first_tick=tick, last_tick=tick,
                       label=o.cls, birth_x_mt=float(o.x_mt), birth_y_mt=float(o.y_mt),
                       conf=float(o.conf), form=int(o.form),
                       hp_frac=None if o.hp_frac is None else _clip(float(o.hp_frac), 0.0, 1.0))
        t.label_weights[o.cls] = max(float(o.conf), 1e-3)
        t.form_weights[int(o.form)] = max(float(o.conf), 1e-3)
        t.streak_label, t.streak = o.cls, 1
        self._tracks[uid] = t
        if o.y_mt > MID_Y_MT + cfg.spawn_deadzone_mt:
            self._add_cue(t, "spawn", -cfg.spawn_llr)
        elif o.y_mt < MID_Y_MT - cfg.spawn_deadzone_mt:
            self._add_cue(t, "spawn", cfg.spawn_llr)
        self._frame_cues(t, o)
        self._try_claim_tap(t, o.cls, tick)
        self._update_vote(t)
        return t

    def _absorb(self, t: TrackState, o: Observation, tick: float) -> None:
        cfg = self.config
        dt = tick - t.last_tick
        if dt > 0:
            rvx, rvy = (o.x_mt - t.x_mt) / dt, (o.y_mt - t.y_mt) / dt
            if t.hits == 1:
                vx, vy = rvx, rvy
            else:
                a = cfg.velocity_alpha
                vx, vy = a * rvx + (1 - a) * t.vx, a * rvy + (1 - a) * t.vy
            speed = math.hypot(vx, vy)
            if speed > cfg.max_speed_mt_per_tick:
                vx, vy = vx * cfg.max_speed_mt_per_tick / speed, vy * cfg.max_speed_mt_per_tick / speed
            t.vx, t.vy = vx, vy
        t.x_mt, t.y_mt = float(o.x_mt), float(o.y_mt)
        t.last_tick = tick
        t.hits += 1
        t.conf = 0.5 * t.conf + 0.5 * float(o.conf)
        if o.hp_frac is not None:
            t.hp_frac = _clip(float(o.hp_frac), 0.0, 1.0)
        self._vote_label(t, o)
        self._frame_cues(t, o)
        self._try_claim_tap(t, o.cls, tick)
        self._update_vote(t)

    def _vote_label(self, t: TrackState, o: Observation) -> None:
        cfg = self.config
        w = max(float(o.conf), 1e-3)
        for k in t.label_weights:
            t.label_weights[k] *= cfg.label_decay
        t.label_weights[o.cls] = t.label_weights.get(o.cls, 0.0) + w
        for k in t.form_weights:
            t.form_weights[k] *= cfg.label_decay
        f = int(o.form)
        t.form_weights[f] = t.form_weights.get(f, 0.0) + w
        if t.form_weights[f] > t.form_weights.get(t.form, 0.0):
            t.form = f
        if o.cls == t.streak_label:
            t.streak += 1
        else:
            t.streak_label, t.streak = o.cls, 1
        if o.cls != t.label and t.streak >= cfg.label_switch_frames:
            mine = t.label_weights[o.cls]
            if all(mine > v for k, v in t.label_weights.items() if k != o.cls):
                t.label = o.cls

    def _frame_cues(self, t: TrackState, o: Observation) -> None:
        cfg = self.config
        p = _clip(float(o.side_prob), 1e-3, 1.0 - 1e-3)
        self._add_cue(t, "cnn", cfg.cnn_weight * _clip(math.log(p / (1.0 - p)), -cfg.cnn_max_llr, cfg.cnn_max_llr))
        if o.bar_team == ENEMY_TEAM:
            self._add_cue(t, "bar", cfg.bar_llr)
        elif o.bar_team == OWN_TEAM:
            self._add_cue(t, "bar", -cfg.bar_llr)

    def _try_claim_tap(self, t: TrackState, cls: str, tick: float) -> None:
        cfg = self.config
        if t.tap_claimed or tick - t.first_tick > cfg.tap_claim_age_ticks:
            return
        for tp in self._taps:
            if tp.full or not tp.tick <= t.first_tick <= tp.tick + cfg.tap_window_ticks:
                continue
            if cls not in tp.classes and t.label not in tp.classes:
                continue
            if math.hypot(t.birth_x_mt - tp.x_mt, t.birth_y_mt - tp.y_mt) > cfg.tap_radius_mt:
                continue
            tp.claimed.append(t.uid)
            t.tap_claimed = True
            self._add_cue(t, "tap", -cfg.tap_llr)
            return

    def _add_cue(self, t: TrackState, name: str, llr: float) -> None:
        if llr == 0.0:
            return
        cap = self.config.llr_cap
        t.cues[name] = t.cues.get(name, 0.0) + llr
        t.team_llr = _clip(t.team_llr + llr, -cap, cap)

    def _update_vote(self, t: TrackState) -> None:
        """Decide or flip the team from the summed evidence (once per frame, after
        every cue of the frame, so the order of cues cannot matter)."""
        cfg = self.config
        if t.vote_team == UNKNOWN_TEAM:
            if t.team_llr >= cfg.decide_llr:
                t.vote_team = ENEMY_TEAM
            elif t.team_llr <= -cfg.decide_llr:
                t.vote_team = OWN_TEAM
        elif t.vote_team == ENEMY_TEAM and t.team_llr <= -cfg.flip_llr:
            t.vote_team = OWN_TEAM
        elif t.vote_team == OWN_TEAM and t.team_llr >= cfg.flip_llr:
            t.vote_team = ENEMY_TEAM
