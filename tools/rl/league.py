"""The league for self-play RL (R03): who the learner plays next.

    LeagueConfig    cadence, capacity, decay and PFSP settings (defaults below)
    League          the pool: permanent anchors (imitation checkpoints) and learner
                    snapshots, decayed win-rate estimates, PFSP draws, admission every
                    N iterations, eviction by stratified win rate, JSON save / load
    pfsp_weights    the draw weights on their own, for a vector of expected scores
    elo_expected    expected scores from fitted ratings (tools/league/stats.fit_btd), to
                    pass as League.weights(expected=...) in place of the estimates

Why: tools/rs_selfplay.py trains against one frozen opponent (rs_big) with one deck,
so the policy learns to beat that one opponent (audit tokens-15). A pool of its own
earlier selves plus the imitation checkpoints, drawn by how often each still beats the
learner, stops it overfitting one opponent and forgetting how to beat the ones it has
passed. The opponent's deck comes from tools/deck_stats.py's field.

Typical use by the driver (R04):

    league = League(LeagueConfig(admit_every=5), anchors={"il:rs_big": "runs/tokens/rs_big/best.pt"})
    league.promote("il:rs_big")                     # the imitation model starts as champion
    for it in range(iters):
        opponents = league.draw(rng, size=n_battles)            # one id per battle
        ... play, then league.record_many((opponent, learner_score), ...)
        ... PPO update -> checkpoint
        league.end_iteration(snapshot=checkpoint)   # decay; admit when due
        league.save(run / "league.json")

Semantics follow royalelearn 0.5.20 (MIT; not imported):

* Weights (ladder/matchmaker.py MixMatchmaker.weights, defaults config.py:530-605):
  p_i is the learner's expected score against member i, shape_i = (1 - p_i) ** 2
  ("hard"), w = 0.8 * shape / sum(shape) + 0.2 / M, then every weight is floored at
  0.01 / M and the vector renormalised. If every shape is 0 the draw is uniform.
* Anchors are never evicted (ladder/eviction.py keeps the anchors, seeds, the run's
  first snapshot and the champion chain). Eviction only takes a member out of the
  draw: it stays in the archive with its record, and its results can still be filed.
* Eviction keeps a spread across the win-rate range (equal-count strata, the member
  with the most evidence in each), not the newest N: a pool of the last N snapshots
  is one opponent measured N times.

Differences, on purpose:

* p_i is a decayed direct estimate, (w_i + prior / 2) / (n_i + prior): w_i is the
  learner's score and n_i the games against member i, both multiplied by `decay` at
  every end_iteration(), so the estimate follows the learner as it changes, and one
  not played for a while drifts back toward 1/2. Every training battle here is the
  learner against a member and the pool is small (16 snapshots by default), so direct
  records are well populated; royalelearn fits ratings because its pool is 48 and most
  pairs are never played directly.
* Admission is on a fixed cadence (every `admit_every` iterations), not after a gate.
  Promotion (the champion) is the L03 gate's decision, recorded here with promote().
* The newest `protect_newest` snapshots are not evicted, so a snapshot is never
  admitted and dropped before it has been played.
* An optional `mirror_share` of draws returns LEARNER (the live policy on both seats);
  0 by default.

Scores are the learner's: 1 win, 0.5 draw, 0 loss (or "win" / "draw" / "loss").
Pure Python and numpy: no torch, no engine, no memory reader.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "FORMAT_VERSION",
    "KIND_ANCHOR",
    "KIND_SNAPSHOT",
    "LEARNER",
    "League",
    "LeagueConfig",
    "Member",
    "WEIGHTINGS",
    "elo_expected",
    "pfsp_shape",
    "pfsp_weights",
    "score_value",
]

FORMAT_VERSION = 1
#: The live policy. draw() returns it for a mirror battle; record() ignores it.
LEARNER = "learner"
KIND_ANCHOR = "anchor"
KIND_SNAPSHOT = "snapshot"
#: "hard" for training (opponents that still beat the learner), "variance" for
#: measurement (the most even matchups), as royalelearn; "linear" and "uniform" too.
WEIGHTINGS = ("hard", "variance", "linear", "uniform")
SCORE_OF = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def score_value(value: Any) -> float:
    """A learner score in [0, 1]: a number, or "win" / "draw" / "loss"."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in SCORE_OF:
            raise ValueError(f"score {value!r}: use 1 / 0.5 / 0 or win / draw / loss")
        return SCORE_OF[key]
    score = float(value)
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"score {value!r} is outside [0, 1]")
    return score


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------
def pfsp_shape(expected: Any, weighting: str = "hard", power: float = 2.0) -> np.ndarray:
    """Unnormalised PFSP weight of opponents the learner scores `expected` against.

    hard (1 - p) ** power, variance p (1 - p), linear 1 - p, uniform 1. All ones when
    every weight would be 0 (the learner beats everybody), so a draw is still possible.
    """
    p = np.asarray(expected, dtype=np.float64)
    if weighting == "hard":
        shaped = (1.0 - p) ** float(power)
    elif weighting == "variance":
        shaped = p * (1.0 - p)
    elif weighting == "linear":
        shaped = 1.0 - p
    elif weighting == "uniform":
        shaped = np.ones_like(p)
    else:
        raise ValueError(f"unknown PFSP weighting {weighting!r}; use one of {WEIGHTINGS}")
    if not np.any(shaped > 0.0):
        shaped = np.ones_like(p)
    return shaped


def pfsp_weights(expected: Any, *, weighting: str = "hard", power: float = 2.0,
                 uniform_floor: float = 0.2, floor_scale: float = 0.01) -> np.ndarray:
    """Normalised draw probabilities over M opponents with expected scores `expected`.

    (1 - uniform_floor) * shape / sum(shape) + uniform_floor / M, each floored at
    floor_scale / M, renormalised (royalelearn MixMatchmaker.weights).
    """
    p = np.asarray(expected, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"expected scores must be one vector, got shape {p.shape}")
    count = p.size
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(f"expected scores must lie in [0, 1], got {p.tolist()}")
    shaped = pfsp_shape(p, weighting, power)
    shaped = shaped / shaped.sum()
    weights = (1.0 - uniform_floor) * shaped + uniform_floor / count
    weights = np.maximum(weights, floor_scale / count)
    return weights / weights.sum()


def elo_expected(ratings: Mapping[str, float], learner_rating: float) -> dict[str, float]:
    """The learner's expected score against each rated member, on the Elo scale
    (a BTD fit's ratings, tools/league/stats.RatingTable.rating)."""
    return {member: 1.0 / (1.0 + 10.0 ** ((float(rating) - float(learner_rating)) / 400.0))
            for member, rating in ratings.items()}


# ---------------------------------------------------------------------------
# config and members
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LeagueConfig:
    """League settings. PFSP defaults are royalelearn's (config.py LadderConfig)."""

    #: Admit the learner's checkpoint every this many iterations (0: only by admit()).
    admit_every: int = 5
    #: Snapshots in the draw at most (anchors are extra). Protected snapshots (the run's
    #: first, the champion chain, the newest) stay even past it.
    max_snapshots: int = 16
    #: The newest this many snapshots are never evicted.
    protect_newest: int = 1
    #: Evidence is multiplied by this at every end_iteration(): 0.9 keeps about the
    #: last ten iterations' games (half-life 6.6 iterations).
    decay: float = 0.9
    #: Pseudo-games at a score of 1/2 in every estimate: 2 is a Beta(1, 1) prior.
    prior_games: float = 2.0
    weighting: str = "hard"
    power: float = 2.0
    #: The anti-forgetting share, spread evenly over all members.
    uniform_floor: float = 0.2
    #: Every weight is at least this / M before renormalising.
    weight_floor_scale: float = 0.01
    #: Share of draws that return LEARNER (self-play mirror). 0 = pool only.
    mirror_share: float = 0.0

    def __post_init__(self) -> None:
        for name in ("admit_every", "max_snapshots", "protect_newest"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"LeagueConfig.{name} must be a whole number >= 0, got {value!r}")
        if not 0.0 < float(self.decay) <= 1.0:
            raise ValueError(f"LeagueConfig.decay must be in (0, 1], got {self.decay!r}")
        if not float(self.prior_games) > 0.0:
            raise ValueError(f"LeagueConfig.prior_games must be > 0, got {self.prior_games!r}")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"LeagueConfig.weighting {self.weighting!r}: use one of {WEIGHTINGS}")
        if not float(self.power) > 0.0:
            raise ValueError(f"LeagueConfig.power must be > 0, got {self.power!r}")
        for name in ("uniform_floor", "weight_floor_scale", "mirror_share"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"LeagueConfig.{name} must be in [0, 1], got {value!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LeagueConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown LeagueConfig field(s): {unknown}")
        return cls(**dict(data))


@dataclass
class Member:
    """One opponent: an anchor or a snapshot, and the learner's record against it."""

    id: str
    kind: str
    path: str | None = None
    #: League iteration it joined at (anchors: when they were added).
    iteration: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    #: In the draw. An evicted snapshot is inactive; anchors are always active.
    active: bool = True
    #: Decayed games and decayed learner score against it (the estimate's evidence).
    decayed_games: float = 0.0
    decayed_score: float = 0.0
    #: Raw totals, never decayed (for logs).
    games: int = 0
    score: float = 0.0
    last_played: int | None = None

    @property
    def is_anchor(self) -> bool:
        return self.kind == KIND_ANCHOR

    def expected(self, prior_games: float) -> float:
        """The learner's expected score against this member: (w + prior / 2) / (n + prior)."""
        return (self.decayed_score + prior_games / 2.0) / (self.decayed_games + prior_games)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Member:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown Member field(s): {unknown}")
        member = cls(**dict(data))
        if member.kind not in (KIND_ANCHOR, KIND_SNAPSHOT):
            raise ValueError(f"member {member.id!r} has unknown kind {member.kind!r}")
        return member


def _generator(rng: Any) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _path_text(path: str | os.PathLike | None) -> str | None:
    return None if path is None else str(path)


# ---------------------------------------------------------------------------
# the league
# ---------------------------------------------------------------------------
class League:
    """Anchors + snapshots, the learner's decayed record against each, and the draw.

    `anchors` is {id: checkpoint path or None} or a list of ids, added in order. Members
    keep insertion order (anchors given here first, then snapshots as admitted); that
    order is the order of weights() and of the draw, so a draw is a pure function of the
    league state and the generator.
    """

    def __init__(self, config: LeagueConfig | None = None,
                 anchors: Mapping[str, str | os.PathLike | None] | Iterable[str] = ()) -> None:
        self.config = config if config is not None else LeagueConfig()
        #: Completed iterations (end_iteration() calls).
        self.iteration = 0
        self.champion: str | None = None
        self.champion_chain: list[str] = []
        #: The run's first snapshot (never evicted).
        self.v0: str | None = None
        self.evictions = 0
        self._members: dict[str, Member] = {}
        if isinstance(anchors, str):
            anchors = [anchors]
        items = anchors.items() if isinstance(anchors, Mapping) else ((a, None) for a in anchors)
        for member_id, path in items:
            self.add_anchor(member_id, path)

    # -- membership ---------------------------------------------------------
    def _check_new_id(self, member_id: Any) -> str:
        if not isinstance(member_id, str) or not member_id.strip():
            raise ValueError(f"a member id is a non-empty string, got {member_id!r}")
        if member_id == LEARNER:
            raise ValueError(f"{LEARNER!r} is the live policy, not a pool member")
        if member_id in self._members:
            # One id, one set of weights: a reused id would file two policies' games together.
            raise ValueError(f"member {member_id!r} is already in the league")
        return member_id

    def add_anchor(self, member_id: str, path: str | os.PathLike | None = None,
                   meta: Mapping[str, Any] | None = None) -> Member:
        """A permanent member (an imitation checkpoint): drawn like any other, never evicted."""
        member = Member(id=self._check_new_id(member_id), kind=KIND_ANCHOR,
                        path=_path_text(path), iteration=self.iteration, meta=dict(meta or {}))
        self._members[member.id] = member
        return member

    @staticmethod
    def snapshot_id(iteration: int) -> str:
        """The id of the snapshot taken after `iteration` iterations: snap:it0005."""
        return f"snap:it{int(iteration):04d}"

    def admission_due(self, iteration: int | None = None) -> bool:
        """Whether a snapshot is admitted after `iteration` (default: the current) iterations."""
        it = self.iteration if iteration is None else int(iteration)
        every = self.config.admit_every
        return every > 0 and it > 0 and it % every == 0

    def admit(self, path: str | os.PathLike | None, *, member_id: str | None = None,
              iteration: int | None = None,
              meta: Mapping[str, Any] | None = None) -> tuple[str, list[str]]:
        """Add a learner snapshot to the draw. Returns (its id, the ids evicted to make room)."""
        it = self.iteration if iteration is None else int(iteration)
        member_id = self.snapshot_id(it) if member_id is None else member_id
        member = Member(id=self._check_new_id(member_id), kind=KIND_SNAPSHOT,
                        path=_path_text(path), iteration=it, meta=dict(meta or {}))
        self._members[member.id] = member
        if self.v0 is None:
            self.v0 = member.id
        return member.id, self.enforce_capacity()

    def promote(self, member_id: str) -> None:
        """Make a member the champion (the L03 gate's decision). Every champion stays
        protected from eviction."""
        self.member(member_id)
        self.champion = member_id
        if member_id not in self.champion_chain:
            self.champion_chain.append(member_id)

    def evict(self, member_ids: Iterable[str]) -> list[str]:
        """Take snapshots out of the draw (they stay in the archive). Anchors are refused."""
        removed = []
        for member_id in member_ids:
            member = self.member(member_id)
            if member.is_anchor:
                raise ValueError(f"{member_id!r} is an anchor; anchors are never evicted")
            if member.active:
                member.active = False
                removed.append(member_id)
        self.evictions += len(removed)
        return removed

    def member(self, member_id: str) -> Member:
        try:
            return self._members[member_id]
        except KeyError:
            raise KeyError(f"{member_id!r} is not in the league") from None

    def __contains__(self, member_id: object) -> bool:
        return member_id in self._members

    def __len__(self) -> int:
        return len(self._members)

    @property
    def members(self) -> tuple[Member, ...]:
        """Every member ever added, evicted ones included, in insertion order."""
        return tuple(self._members.values())

    def anchor_ids(self) -> tuple[str, ...]:
        return tuple(m.id for m in self._members.values() if m.is_anchor)

    def snapshot_ids(self, active_only: bool = False) -> tuple[str, ...]:
        return tuple(m.id for m in self._members.values()
                     if not m.is_anchor and (m.active or not active_only))

    def active_ids(self) -> tuple[str, ...]:
        """What the draw may return (besides LEARNER), in insertion order."""
        return tuple(m.id for m in self._members.values() if m.active)

    def protected_ids(self) -> set[str]:
        """Never evicted by capacity: anchors, the first snapshot, the champion chain and
        the newest `protect_newest` snapshots."""
        protected = set(self.anchor_ids()) | set(self.champion_chain)
        if self.v0 is not None:
            protected.add(self.v0)
        newest = self.config.protect_newest
        if newest > 0:
            protected |= set(self.snapshot_ids(active_only=True)[-newest:])
        return protected

    def enforce_capacity(self) -> list[str]:
        """Evict down to max_snapshots, keeping a spread across the win-rate range.

        The optional snapshots (not protected) are ranked by expected score and cut into
        as many equal-count strata as there are places left; each stratum keeps the
        member with the most (decayed) evidence, then the most recent. admit() calls it;
        call it after replacing `config` (a smaller max_snapshots on resume). Returns the
        ids evicted.
        """
        active =[m for m in self._members.values() if not m.is_anchor and m.active]
        cap = self.config.max_snapshots
        if len(active) <= cap:
            return []
        protected = self.protected_ids()
        optional = [m for m in active if m.id not in protected]
        budget = max(0, cap - (len(active) - len(optional)))
        if budget >= len(optional):
            return []
        prior = self.config.prior_games
        order = {member_id: k for k, member_id in enumerate(self._members)}
        if budget == 0:
            keep: set[str] = set()
        else:
            ranked = sorted(optional, key=lambda m: (m.expected(prior), m.id))
            keep = set()
            for stratum in range(budget):
                low = stratum * len(ranked) // budget
                high = max(low + 1, (stratum + 1) * len(ranked) // budget)
                best = max(ranked[low:high],
                           key=lambda m: (m.decayed_games, m.iteration, order[m.id]))
                keep.add(best.id)
        doomed = [m.id for m in optional if m.id not in keep]
        return self.evict(doomed)

    # -- results ------------------------------------------------------------
    def record(self, opponent: str, score: Any) -> bool:
        """File one training game: the learner scored `score` against `opponent`.

        LEARNER (a mirror battle) says nothing about the pool and is ignored (False).
        An evicted member's game is still filed: it stays in the archive.
        """
        if opponent == LEARNER:
            return False
        member = self.member(opponent)
        value = score_value(score)
        member.decayed_games += 1.0
        member.decayed_score += value
        member.games += 1
        member.score += value
        member.last_played = self.iteration
        return True

    def record_many(self, results: Iterable[tuple[str, Any]]) -> int:
        """record() for (opponent, learner score) pairs; returns how many were filed."""
        return sum(1 for opponent, score in results if self.record(opponent, score))

    def decay(self, factor: float | None = None) -> None:
        """Multiply every member's evidence by `factor` (default config.decay)."""
        factor = self.config.decay if factor is None else float(factor)
        if not 0.0 < factor <= 1.0:
            raise ValueError(f"decay factor must be in (0, 1], got {factor!r}")
        for member in self._members.values():
            member.decayed_games *= factor
            member.decayed_score *= factor

    def end_iteration(self, snapshot: str | os.PathLike | None = None,
                      meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Close an iteration: decay the evidence, count the iteration, and admit
        `snapshot` (the checkpoint this iteration produced) when admission is due.

        Returns {"iteration", "admitted" (id or None), "evicted" (ids)}.
        """
        self.decay()
        self.iteration += 1
        admitted, evicted = None, []
        if snapshot is not None and self.admission_due():
            admitted, evicted = self.admit(snapshot, meta=meta)
        return {"iteration": self.iteration, "admitted": admitted, "evicted": evicted}

    def expected(self, member_id: str) -> float:
        """The learner's decayed expected score against a member (1/2 with no games)."""
        return self.member(member_id).expected(self.config.prior_games)

    # -- the draw -----------------------------------------------------------
    def weights(self, expected: Mapping[str, float] | None = None) -> dict[str, float]:
        """Draw probabilities over the active members, in insertion order.

        `expected` overrides the estimates for the members it names (for example
        elo_expected() of a rating fit); the rest keep their estimates. The mirror
        share is not included: draw() takes it first.
        """
        ids = self.active_ids()
        if not ids:
            return {}
        given = expected or {}
        p = [float(given[m]) if m in given else self.expected(m) for m in ids]
        cfg = self.config
        probs = pfsp_weights(p, weighting=cfg.weighting, power=cfg.power,
                             uniform_floor=cfg.uniform_floor, floor_scale=cfg.weight_floor_scale)
        return dict(zip(ids, probs.tolist()))

    def draw(self, rng: Any = None, size: int | None = None,
             expected: Mapping[str, float] | None = None) -> str | list[str]:
        """One opponent id (size None) or a list of `size`, drawn with replacement.

        `rng` is a numpy Generator or a seed. With mirror_share > 0 each draw is LEARNER
        with that probability (decided first, from the same generator), else a member by
        weights(). Raises RuntimeError when a member is needed and none is active.
        """
        gen = _generator(rng)
        count = 1 if size is None else int(size)
        if count < 0:
            raise ValueError(f"size must be >= 0, got {size!r}")
        share = float(self.config.mirror_share)
        mirror = gen.random(count) < share if share > 0.0 else np.zeros(count, dtype=bool)
        need = int(count - mirror.sum())
        weights = self.weights(expected)
        if need and not weights:
            raise RuntimeError("the league has no active member to draw "
                               "(add an anchor or admit a snapshot first)")
        ids = list(weights)
        picks = (gen.choice(len(ids), size=need, p=np.fromiter(weights.values(), dtype=np.float64))
                 if need else np.zeros(0, dtype=np.int64))
        out, k = [], 0
        for is_mirror in mirror:
            if is_mirror:
                out.append(LEARNER)
            else:
                out.append(ids[int(picks[k])])
                k += 1
        return out[0] if size is None else out

    # -- reporting and state ------------------------------------------------
    def summary(self, expected: Mapping[str, float] | None = None) -> dict[str, Any]:
        """One JSON-able row for the run log."""
        weights = self.weights(expected)
        return {
            "iteration": self.iteration, "champion": self.champion, "v0": self.v0,
            "active": len(weights), "evictions": self.evictions,
            "members": [{"id": m.id, "kind": m.kind, "active": m.active,
                         "expected": round(m.expected(self.config.prior_games), 4),
                         "games": m.games, "weight": round(weights.get(m.id, 0.0), 4)}
                        for m in self._members.values()],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "config": self.config.to_dict(),
            "iteration": self.iteration,
            "champion": self.champion,
            "champion_chain": list(self.champion_chain),
            "v0": self.v0,
            "evictions": self.evictions,
            "members": [m.to_dict() for m in self._members.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], config: LeagueConfig | None = None) -> League:
        """Rebuild a league from to_dict(). `config` replaces the stored settings (for
        example a new cadence on resume); the members and their records are kept."""
        version = data.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"league state format {version!r}; this code reads {FORMAT_VERSION}")
        league = cls(config if config is not None else LeagueConfig.from_dict(data["config"]))
        league.iteration = int(data["iteration"])
        league.evictions = int(data.get("evictions", 0))
        for raw in data["members"]:
            member = Member.from_dict(raw)
            league._check_new_id(member.id)
            league._members[member.id] = member
        league.v0 = data.get("v0")
        for member_id in data.get("champion_chain", []):
            league.member(member_id)
            league.champion_chain.append(member_id)
        champion = data.get("champion")
        if champion is not None:
            league.member(champion)
        league.champion = champion
        return league

    def save(self, path: str | os.PathLike) -> Path:
        """Write the state as JSON, atomically (a .tmp file, then os.replace)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str | os.PathLike, config: LeagueConfig | None = None) -> League:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), config=config)
