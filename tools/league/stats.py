"""Statistics for comparing agents (L01).

Scores are per game from the first agent's side: 1 win, 0.5 draw, 0 loss
(the strings "win", "draw", "loss" are accepted too).

    wilson(k, n)                 Wilson score interval for k wins in n games
                                 (moved here from tools/rs_eval.py; same numbers)
    score_interval(scores)       wins/draws/losses, mean score and its Wilson interval
    paired_bootstrap_ci(a, b)    CI for mean(a) - mean(b) when A and B played the
                                 SAME pairings (deck pair, seed, seat): resamples
                                 pairings, optionally whole groups (both seats)
    mcnemar(a, b)                exact McNemar test on the same pairings
    fit_btd(results, anchor=..)  Bradley-Terry-Davidson MAP ratings over a whole
                                 result log, Elo scale, one player pinned as anchor
    EloReadout                   online k=32 Elo for a dashboard only

Why paired: 300 unpaired matches give a Wilson half-width of about 5.6 pp;
two checkpoints that play the same deck pairings and seeds are strongly
correlated, so the paired difference is several times tighter at the same cost.

The BTD model, with q_i = 10 ** (r_i / 400) and the draw parameter nu:

    P(i wins) = q_i / D,  P(j wins) = q_j / D,  P(draw) = nu * sqrt(q_i q_j) / D,
    D = q_i + q_j + nu * sqrt(q_i q_j)

In theta = r * ln(10) / 400 the log-posterior (Gaussian priors on theta and on
ln nu) is concave, so Newton's method converges to the unique MAP. The rating
is a pure function of the stored results, in any order. Standard errors come
from the inverse observed information; a difference uses the covariance block,
not the sum of two marginals. When draws are rarer than `min_draw_rate` the
draw term is switched off and a draw counts as half a win. The design follows
royalelearn ladder/rating.py (MIT); nothing is imported from it.
"""
from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: theta = ELO_SCALE * Elo: one Elo point in natural-log odds.
ELO_SCALE = math.log(10.0) / 400.0
#: The two-sided 95% normal quantile.
Z95 = 1.959963984540054

SCORE_OF = {"win": 1.0, "draw": 0.5, "loss": 0.0}
_VALID_SCORES = (0.0, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------
def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for `k` successes in `n` trials.

    The formula of tools/rs_eval.py:42-49, with z = 1.96 by default and
    (0.0, 0.0) for n = 0 as there; the bounds are clipped to [0, 1]. `k` may
    be fractional (draws counted as half a win).
    """
    if not n:
        return 0.0, 0.0
    if n < 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def as_scores(values: Iterable[Any]) -> np.ndarray:
    """Per-game scores as a float array; accepts 0 / 0.5 / 1 and "win"/"draw"/"loss"."""
    if isinstance(values, np.ndarray) and values.dtype.kind in "biuf":
        arr = values.astype(np.float64).ravel()
        if not np.all((arr == 0.0) | (arr == 0.5) | (arr == 1.0)):
            raise ValueError("a score must be 0, 0.5 or 1")
        return arr
    return np.asarray([score_value(v) for v in values], dtype=np.float64)


def score_value(v: Any) -> float:
    """One game's score as 0.0 / 0.5 / 1.0 (also from "win" / "draw" / "loss")."""
    s = SCORE_OF.get(v.lower()) if isinstance(v, str) else float(v)
    if s not in _VALID_SCORES:
        raise ValueError(f"a score must be 0, 0.5, 1, 'win', 'draw' or 'loss', not {v!r}")
    return s


@dataclass(frozen=True)
class ScoreInterval:
    """A score rate and its Wilson interval (draws count half)."""
    n: int
    wins: int
    draws: int
    losses: int
    score: float
    lo: float
    hi: float

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "wins": self.wins, "draws": self.draws, "losses": self.losses,
                "score": self.score, "lo": self.lo, "hi": self.hi}


def score_interval(scores: Iterable[Any], z: float = 1.96) -> ScoreInterval:
    """Counts, mean score and Wilson interval of one agent's per-game scores."""
    s = as_scores(scores)
    n = int(s.size)
    wins, draws = int(np.sum(s == 1.0)), int(np.sum(s == 0.5))
    k = wins + 0.5 * draws
    lo, hi = wilson(k, n, z)
    return ScoreInterval(n, wins, draws, n - wins - draws, k / n if n else 0.0, lo, hi)


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PairedDiff:
    """mean(a) - mean(b) over the same games, with a bootstrap interval.

    se is the standard deviation of the bootstrap replicates; n counts games
    and n_groups the resampled units (pairings, or groups of them).
    """
    diff: float
    lo: float
    hi: float
    se: float
    n: int
    n_groups: int
    n_boot: int
    level: float

    def to_dict(self) -> dict[str, Any]:
        return {"diff": self.diff, "lo": self.lo, "hi": self.hi, "se": self.se, "n": self.n,
                "n_groups": self.n_groups, "n_boot": self.n_boot, "level": self.level}


def paired_bootstrap_ci(a: Iterable[Any], b: Iterable[Any], *, groups: Sequence[Hashable] | None = None,
                        n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> PairedDiff:
    """Percentile-bootstrap CI for the difference in mean score of two agents.

    a[i] and b[i] are the two agents' scores on the same pairing i (same decks,
    seed and seat, each against the same opponent). Pairings are resampled with
    replacement; with `groups` (one key per game, e.g. the pairing id shared by
    both seats) whole groups are resampled and the replicate is the ratio of
    summed differences to summed games. Deterministic for a given seed.
    """
    av, bv = as_scores(a), as_scores(b)
    if av.shape != bv.shape:
        raise ValueError(f"a and b need the same length, got {av.size} and {bv.size}")
    if av.size == 0:
        raise ValueError("no games")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    d = av - bv
    if groups is None:
        sums, counts = d, np.ones_like(d)
    else:
        keys = list(groups)
        if len(keys) != d.size:
            raise ValueError(f"groups has {len(keys)} keys for {d.size} games")
        index: dict[Hashable, int] = {}
        gid = np.fromiter((index.setdefault(k, len(index)) for k in keys), dtype=np.int64, count=len(keys))
        sums = np.bincount(gid, weights=d, minlength=len(index))
        counts = np.bincount(gid, minlength=len(index)).astype(np.float64)
    n_groups = int(sums.size)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=np.float64)
    chunk = max(1, 2_000_000 // n_groups)          # bound the (chunk, n_groups) index array
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = rng.integers(0, n_groups, size=(m, n_groups))
        reps[start:start + m] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = 1.0 - level
    lo, hi = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    se = float(reps.std(ddof=1)) if n_boot > 1 else 0.0
    return PairedDiff(float(d.sum() / d.size), float(lo), float(hi), se, int(d.size), n_groups,
                      int(n_boot), float(level))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b and c discordant pairs.

    Under H0 the b + c discordant pairs split as Binomial(b + c, 1/2):
    p = min(1, 2 * P(X <= min(b, c))). Exact integer arithmetic.
    """
    b, c = int(b), int(c)
    if b < 0 or c < 0:
        raise ValueError("counts must be >= 0")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


@dataclass(frozen=True)
class McNemar:
    """Exact McNemar on paired games: a_better counts pairings where a scored
    more than b, b_better the reverse, ties the rest."""
    a_better: int
    b_better: int
    ties: int
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return {"a_better": self.a_better, "b_better": self.b_better, "ties": self.ties,
                "p_value": self.p_value}


def mcnemar(a: Iterable[Any], b: Iterable[Any]) -> McNemar:
    """Exact McNemar test for two agents' scores on the same pairings."""
    av, bv = as_scores(a), as_scores(b)
    if av.shape != bv.shape:
        raise ValueError(f"a and b need the same length, got {av.size} and {bv.size}")
    up, down = int(np.sum(av > bv)), int(np.sum(av < bv))
    return McNemar(up, down, int(av.size) - up - down, mcnemar_exact_p(up, down))


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------
def elo_of_score(p: float) -> float:
    """The Elo gap a score rate implies: 400 * log10(p / (1 - p))."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return 400.0 * math.log10(p / (1.0 - p))


def score_of_elo(gap: float) -> float:
    """The score rate an Elo gap implies."""
    return 1.0 / (1.0 + 10.0 ** (-float(gap) / 400.0))


class EloReadout:
    """Online zero-sum Elo (k = 32 by default) for dashboards.

    Order dependent and never converging for a frozen player, so it is not a
    decision statistic: gates use fit_btd and the paired CI.
    """

    def __init__(self, k_factor: float = 32.0, initial: float = 1200.0):
        self.k = float(k_factor)
        self.initial = float(initial)
        self._rating: dict[str, float] = {}
        self._games: dict[str, int] = {}

    def rating(self, player: str) -> float:
        return self._rating.get(player, self.initial)

    def ratings(self) -> dict[str, float]:
        return dict(self._rating)

    def games(self, player: str) -> int:
        return self._games.get(player, 0)

    def update(self, a: str, b: str, score_a: Any) -> tuple[float, float]:
        """Record one game (score from a's side) and return the new ratings of a and b."""
        s = score_value(score_a)
        if a == b:
            raise ValueError("a player cannot play itself")
        ra, rb = self.rating(a), self.rating(b)
        delta = self.k * (s - score_of_elo(ra - rb))
        self._rating[a], self._rating[b] = ra + delta, rb - delta
        self._games[a] = self.games(a) + 1
        self._games[b] = self.games(b) + 1
        return self._rating[a], self._rating[b]


# ---------------------------------------------------------------------------
# Bradley-Terry-Davidson
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RatingTable:
    """A BTD fit. Ratings are Elo with the anchor at anchor_rating; se is 0
    for the anchor. nu is the Davidson draw parameter, None when draws were
    counted as half wins."""
    players: tuple[str, ...]
    rating: dict[str, float]
    se: dict[str, float]
    anchor: str
    anchor_rating: float
    nu: float | None
    n_games: dict[str, int]
    converged: bool
    iterations: int
    free: dict[str, int] = field(default_factory=dict, repr=False)
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False, compare=False)

    def _theta(self, player: str) -> float:
        return (self.rating.get(player, self.anchor_rating) - self.anchor_rating) * ELO_SCALE

    def outcome_probabilities(self, a: str, b: str) -> tuple[float, float, float]:
        """(P(a wins), P(b wins), P(draw)) under the fit; an unknown id rates as the anchor."""
        davidson = self.nu is not None
        pa, pb, pd = _outcome_probabilities(np.array([self._theta(a)]), np.array([self._theta(b)]),
                                            math.log(self.nu) if davidson else 0.0, davidson)
        return float(pa[0]), float(pb[0]), float(pd[0])

    def predict(self, a: str, b: str) -> float:
        """P(a scores against b), a draw counted as half."""
        pa, _pb, pd = self.outcome_probabilities(a, b)
        return pa + 0.5 * pd

    def difference_se(self, a: str, b: str) -> float:
        """Standard error of rating[a] - rating[b] in Elo, from the covariance block."""
        ia, ib = self.free.get(a), self.free.get(b)
        var = 0.0
        if ia is not None:
            var += float(self.covariance[ia, ia])
        if ib is not None:
            var += float(self.covariance[ib, ib])
        if ia is not None and ib is not None:
            var -= 2.0 * float(self.covariance[ia, ib])
        return math.sqrt(max(var, 0.0)) / ELO_SCALE

    def to_dict(self) -> dict[str, Any]:
        return {"players": list(self.players), "rating": dict(self.rating), "se": dict(self.se),
                "anchor": self.anchor, "anchor_rating": self.anchor_rating, "nu": self.nu,
                "n_games": dict(self.n_games), "converged": self.converged, "iterations": self.iterations}


def _outcome_probabilities(a: np.ndarray, b: np.ndarray, log_nu: float, davidson: bool):
    """(P(left wins), P(right wins), P(draw)) per pair, through a shifted softmax."""
    draw = log_nu + 0.5 * (a + b) if davidson else np.full_like(a, -np.inf)
    top = np.maximum(np.maximum(a, b), draw)
    ea, eb = np.exp(a - top), np.exp(b - top)
    ed = np.exp(draw - top) if davidson else np.zeros_like(a)
    z = ea + eb + ed
    return ea / z, eb / z, ed / z


def _log_posterior(theta, log_nu, left, right, wl, wr, dr, davidson, prec_theta, prec_nu, free_mask):
    a, b = theta[left], theta[right]
    if davidson:
        draw = log_nu + 0.5 * (a + b)
        top = np.maximum(np.maximum(a, b), draw)
        lz = top + np.log(np.exp(a - top) + np.exp(b - top) + np.exp(draw - top))
        ll = np.sum(wl * (a - lz) + wr * (b - lz) + dr * (draw - lz))
        ll -= 0.5 * prec_nu * log_nu * log_nu
    else:
        top = np.maximum(a, b)
        lz = top + np.log(np.exp(a - top) + np.exp(b - top))
        ll = np.sum((wl + 0.5 * dr) * (a - lz) + (wr + 0.5 * dr) * (b - lz))
    return float(ll - 0.5 * prec_theta * np.sum(theta[free_mask] ** 2))


def fit_btd(results: Iterable[tuple[str, str, Any]], *, anchor: str, anchor_rating: float = 0.0,
            prior_sd: float = 400.0, draws: str = "davidson", min_draw_rate: float = 0.02,
            nu_prior_sd: float = 2.0, max_iterations: int = 100, tolerance: float = 1e-10) -> RatingTable:
    """MAP Bradley-Terry-Davidson ratings from (a, b, score_a) results.

    anchor is pinned at anchor_rating (it is added if it played no game).
    prior_sd: sd in Elo of the Gaussian prior on every other rating, centred on
    the anchor (math.inf = maximum likelihood; then every player needs both a
    win and a loss). draws: "davidson" fits nu when the draw rate is at least
    min_draw_rate, else (and with "half_win") a draw is half a win.
    nu_prior_sd: sd of the prior on ln(nu) (math.inf = none).
    """
    if draws not in ("davidson", "half_win"):
        raise ValueError(f"draws must be 'davidson' or 'half_win', not {draws!r}")
    if not prior_sd > 0 or not nu_prior_sd > 0:
        raise ValueError("prior_sd and nu_prior_sd must be > 0 (math.inf for no prior)")
    tally: dict[tuple[str, str], list[float]] = {}
    for a, b, score in results:
        a, b = str(a), str(b)
        if a == b:
            raise ValueError(f"{a!r} cannot play itself")
        s = score_value(score)
        if a > b:
            a, b, s = b, a, 1.0 - s
        t = tally.setdefault((a, b), [0.0, 0.0, 0.0])
        t[0 if s == 1.0 else 1 if s == 0.0 else 2] += 1.0

    players = tuple(sorted({p for pair in tally for p in pair} | {str(anchor)}))
    index = {p: i for i, p in enumerate(players)}
    free = {p: k for k, p in enumerate(p for p in players if p != anchor)}
    n_free = len(free)
    slot = np.full(len(players), -1, dtype=np.int64)
    for p, k in free.items():
        slot[index[p]] = k
    free_mask = slot >= 0

    pairs = sorted(tally)
    left = np.array([index[a] for a, _ in pairs], dtype=np.int64)
    right = np.array([index[b] for _, b in pairs], dtype=np.int64)
    wl = np.array([tally[p][0] for p in pairs], dtype=np.float64)
    wr = np.array([tally[p][1] for p in pairs], dtype=np.float64)
    dr = np.array([tally[p][2] for p in pairs], dtype=np.float64)
    tot = wl + wr + dr
    n_total = float(tot.sum())
    draw_rate = float(dr.sum()) / n_total if n_total else 0.0
    davidson = draws == "davidson" and n_total > 0 and draw_rate >= min_draw_rate

    prec_theta = 0.0 if math.isinf(prior_sd) else 1.0 / (ELO_SCALE * prior_sd) ** 2
    prec_nu = 0.0 if math.isinf(nu_prior_sd) else 1.0 / nu_prior_sd ** 2
    dim = n_free + (1 if davidson else 0)
    sl, sr = slot[left], slot[right]
    nu_slot = n_free

    def grad_hess(theta, log_nu):
        g = np.zeros(dim)
        h = np.zeros((dim, dim))
        pl, pr, pd = _outcome_probabilities(theta[left], theta[right], log_nu, davidson)
        gl, gr = pl + 0.5 * pd, pr + 0.5 * pd
        _acc(g, sl, wl + 0.5 * dr - tot * gl)
        _acc(g, sr, wr + 0.5 * dr - tot * gr)
        _acc2(h, sl, sl, -tot * (pl + 0.25 * pd - gl ** 2))
        _acc2(h, sr, sr, -tot * (pr + 0.25 * pd - gr ** 2))
        cross = -tot * (0.25 * pd - gl * gr)
        _acc2(h, sl, sr, cross)
        _acc2(h, sr, sl, cross)
        if davidson:
            nu_col = np.full_like(sl, nu_slot)
            g[nu_slot] += float(np.sum(dr - tot * pd)) - prec_nu * log_nu
            hl, hr = -tot * (0.5 * pd - gl * pd), -tot * (0.5 * pd - gr * pd)
            _acc2(h, sl, nu_col, hl)
            _acc2(h, nu_col, sl, hl)
            _acc2(h, sr, nu_col, hr)
            _acc2(h, nu_col, sr, hr)
            h[nu_slot, nu_slot] += float(np.sum(-tot * (pd - pd ** 2))) - prec_nu
        for p, k in free.items():
            g[k] -= prec_theta * theta[index[p]]
            h[k, k] -= prec_theta
        return g, h

    def logpost(theta, log_nu):
        return _log_posterior(theta, log_nu, left, right, wl, wr, dr, davidson, prec_theta, prec_nu, free_mask)

    theta = np.zeros(len(players))
    log_nu = math.log(max(2.0 * draw_rate / max(1.0 - draw_rate, 1e-9), 1e-3)) if davidson else 0.0
    converged, iterations = dim == 0, 0
    current = logpost(theta, log_nu) if dim else 0.0
    while dim and iterations < max_iterations:
        iterations += 1
        g, h = grad_hess(theta, log_nu)
        step = np.linalg.solve(-h, g)
        # Newton on a concave function; halve the step if it ever overshoots.
        scale = 1.0
        for _ in range(40):
            cand = theta.copy()
            cand[free_mask] += scale * step[slot[free_mask]]
            cand_nu = log_nu + scale * step[nu_slot] if davidson else log_nu
            value = logpost(cand, cand_nu)
            if value >= current - 1e-12 * max(1.0, abs(current)):
                break
            scale *= 0.5
        theta, log_nu, current = cand, cand_nu, value
        if float(np.max(np.abs(scale * step))) < tolerance:
            converged = True
            break

    if dim:
        _g, h = grad_hess(theta, log_nu)
        covariance = np.linalg.inv(-h)
        se_theta = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    else:
        covariance = np.zeros((0, 0))
        se_theta = np.zeros(0)

    n_games = {p: 0 for p in players}
    for (a, b), t in tally.items():
        n_games[a] += int(sum(t))
        n_games[b] += int(sum(t))
    rating = {p: float(anchor_rating + theta[index[p]] / ELO_SCALE) for p in players}
    se = {p: float(se_theta[free[p]] / ELO_SCALE) if p in free else 0.0 for p in players}
    return RatingTable(players=players, rating=rating, se=se, anchor=str(anchor),
                       anchor_rating=float(anchor_rating), nu=float(math.exp(log_nu)) if davidson else None,
                       n_games=n_games, converged=converged, iterations=iterations, free=dict(free),
                       covariance=covariance)


def _acc(target: np.ndarray, pos: np.ndarray, values: np.ndarray) -> None:
    keep = pos >= 0                     # -1 = the pinned anchor
    np.add.at(target, pos[keep], values[keep])


def _acc2(target: np.ndarray, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
    keep = (rows >= 0) & (cols >= 0)
    np.add.at(target, (rows[keep], cols[keep]), values[keep])
