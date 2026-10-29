"""L01: statistics and ratings for league evaluation (tools/league/stats.py).

Known closed-form values (Wilson, exact McNemar, Elo, two-player BTD), the
legacy rs_eval.wilson formula, a coverage simulation for the paired bootstrap
and recovery of planted Bradley-Terry-Davidson strengths. numpy only.
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.league import stats as S  # noqa: E402


# ---------------------------------------------------------------------------
# Wilson
# ---------------------------------------------------------------------------
def test_wilson_known_values():
    assert S.wilson(50, 100) == (pytest.approx(0.4038298286, abs=1e-9), pytest.approx(0.5961701714, abs=1e-9))
    lo, hi = S.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and lo >= 0.0
    assert hi == pytest.approx(0.2775401688, abs=1e-9)
    assert S.wilson(10, 10) == (pytest.approx(0.7224598312, abs=1e-9), pytest.approx(1.0, abs=1e-12))
    assert S.wilson(165, 300) == (pytest.approx(0.4934245665, abs=1e-9), pytest.approx(0.6053110905, abs=1e-9))
    assert S.wilson(0, 0) == (0.0, 0.0)                  # rs_eval's convention for no games


def test_wilson_gate_margin_matches_royalelearn():
    # royalelearn ladder/rating.py: 55.2% over 1000 clears a 0.52 lower bound, 55.0% does not
    assert S.wilson(552, 1000, S.Z95)[0] > 0.52
    assert S.wilson(550, 1000, S.Z95)[0] < 0.52


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        S.wilson(11, 10)
    with pytest.raises(ValueError):
        S.wilson(-1, 10)


def test_wilson_reproduces_rs_eval():
    """The interval moved here from tools/rs_eval.py:42-49; the numbers must not change."""
    tree = ast.parse((ROOT / "tools" / "rs_eval.py").read_text(encoding="utf-8"))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "wilson"), None)
    if fn is None:
        pytest.skip("rs_eval.wilson is no longer defined there (rewired to tools.league.stats)")
    ns = {"math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "rs_eval_wilson", "exec"), ns)
    legacy = ns["wilson"]
    for n in (1, 7, 100, 300, 1000):
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
            old, new = legacy(k, n), S.wilson(k, n)
            assert new == (pytest.approx(old[0], abs=1e-12), pytest.approx(old[1], abs=1e-12)), (k, n)


def test_score_interval_counts_draws_as_half():
    r = S.score_interval(["win", "draw", "loss", 1, 1.0, 0.5, 0])
    assert (r.n, r.wins, r.draws, r.losses) == (7, 3, 2, 2)
    assert r.score == pytest.approx(4.0 / 7.0)
    assert (r.lo, r.hi) == S.wilson(4.0, 7)
    with pytest.raises(ValueError):
        S.score_interval([0.7])
    with pytest.raises(ValueError):
        S.as_scores(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------
def test_paired_bootstrap_degenerate_cases_are_exact():
    same = [1, 0, 0.5, 1, 0]
    r = S.paired_bootstrap_ci(same, same)
    assert (r.diff, r.lo, r.hi, r.se, r.n, r.n_groups) == (0.0, 0.0, 0.0, 0.0, 5, 5)
    r = S.paired_bootstrap_ci([1.0] * 8, [0.5] * 8, n_boot=300)
    assert (r.diff, r.lo, r.hi) == (0.5, 0.5, 0.5)


def test_paired_bootstrap_resamples_whole_groups():
    a, b = [1, 0, 1, 1], [0, 0, 1, 0]                  # differences 1, 0, 0, 1
    grouped = S.paired_bootstrap_ci(a, b, groups=["p1", "p1", "p2", "p2"])
    # each pairing (both seats) has difference sum 1 over 2 games: every replicate is 0.5
    assert (grouped.diff, grouped.lo, grouped.hi, grouped.n, grouped.n_groups) == (0.5, 0.5, 0.5, 4, 2)
    single = S.paired_bootstrap_ci(a, b)
    assert single.lo < 0.5 < single.hi and single.n_groups == 4


def test_paired_bootstrap_is_deterministic_and_validates():
    rng = np.random.default_rng(7)
    a, b = rng.integers(0, 2, 60), rng.integers(0, 2, 60)
    assert S.paired_bootstrap_ci(a, b, seed=3) == S.paired_bootstrap_ci(a, b, seed=3)
    with pytest.raises(ValueError):
        S.paired_bootstrap_ci([1, 0], [1])
    with pytest.raises(ValueError):
        S.paired_bootstrap_ci([], [])
    with pytest.raises(ValueError):
        S.paired_bootstrap_ci([1], [0], level=1.0)
    with pytest.raises(ValueError):
        S.paired_bootstrap_ci([1, 0], [0, 1], groups=["g"])


def test_pairing_tightens_the_interval_on_correlated_games():
    a = [1] * 60 + [0] * 40
    b = [0] * 5 + [1] * 55 + [1] * 2 + [0] * 38          # 5 games flipped down, 2 flipped up
    r = S.paired_bootstrap_ci(a, b, n_boot=2000)
    assert r.diff == pytest.approx(0.03)
    unpaired_se = math.sqrt(0.60 * 0.40 / 100 + 0.57 * 0.43 / 100)    # 0.0697
    assert r.se < 0.5 * unpaired_se                                   # paired ~ sqrt(.0691 / 100) = 0.026
    m = S.mcnemar(a, b)
    assert (m.a_better, m.b_better, m.ties) == (5, 2, 93)
    assert m.p_value == pytest.approx(58 / 128)                       # 2 * (1 + 7 + 21) / 2^7


def test_paired_bootstrap_covers_the_true_difference_about_95_percent():
    """Each trial: 150 pairings with a per-pairing difficulty q ~ U(0.2, 0.8);
    A wins with q + 0.05, B with q, independently given q. The true mean
    difference is 0.05; about 95% of the intervals should contain it."""
    rng = np.random.default_rng(20261029)
    delta, n, trials = 0.05, 150, 200
    covered = 0
    for t in range(trials):
        q = rng.uniform(0.2, 0.8, size=n)
        a = (rng.random(n) < q + delta).astype(np.float64)
        b = (rng.random(n) < q).astype(np.float64)
        r = S.paired_bootstrap_ci(a, b, n_boot=500, seed=t)
        covered += r.lo <= delta <= r.hi
    assert 0.88 <= covered / trials <= 0.99


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def test_mcnemar_exact_known_values():
    assert S.mcnemar_exact_p(0, 5) == 0.0625
    assert S.mcnemar_exact_p(1, 9) == 22 / 1024
    assert S.mcnemar_exact_p(3, 7) == 0.34375
    assert S.mcnemar_exact_p(7, 3) == S.mcnemar_exact_p(3, 7)
    assert S.mcnemar_exact_p(5, 5) == 1.0
    assert S.mcnemar_exact_p(0, 0) == 1.0
    assert 0.0 < S.mcnemar_exact_p(600, 400) < 1e-8
    with pytest.raises(ValueError):
        S.mcnemar_exact_p(-1, 3)


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------
def test_elo_helpers_and_readout():
    assert S.elo_of_score(0.75) == pytest.approx(400 * math.log10(3))
    assert S.score_of_elo(S.elo_of_score(0.3)) == pytest.approx(0.3)
    assert S.score_of_elo(0.0) == 0.5
    e = S.EloReadout()
    assert e.update("a", "b", 1) == (1216.0, 1184.0)
    ra, rb = e.update("b", "a", "win")
    assert ra == pytest.approx(1184 + 32 * (1 - 1 / (1 + 10 ** 0.08)))      # 1201.47
    assert ra + rb == pytest.approx(2400.0)
    assert (e.games("a"), e.games("b"), e.rating("new")) == (2, 2, 1200.0)
    with pytest.raises(ValueError):
        e.update("a", "b", 0.7)
    with pytest.raises(ValueError):
        e.update("a", "a", 1)


# ---------------------------------------------------------------------------
# Bradley-Terry-Davidson
# ---------------------------------------------------------------------------
def _games(a: str, b: str, wins: int, losses: int, draws: int = 0) -> list[tuple[str, str, float]]:
    return [(a, b, 1.0)] * wins + [(a, b, 0.0)] * losses + [(a, b, 0.5)] * draws


def test_btd_two_players_maximum_likelihood_is_exact():
    t = S.fit_btd(_games("cand", "ref", 75, 25), anchor="ref", prior_sd=math.inf)
    assert t.converged and t.nu is None
    assert t.rating["ref"] == 0.0 and t.se["ref"] == 0.0
    assert t.rating["cand"] == pytest.approx(400 * math.log10(3), abs=1e-6)      # 190.85
    # observed information 100 * 0.75 * 0.25 = 18.75 logits^-2
    assert t.se["cand"] == pytest.approx(math.sqrt(1 / 18.75) / S.ELO_SCALE, rel=1e-6)
    assert t.predict("cand", "ref") == pytest.approx(0.75, abs=1e-9)
    assert t.difference_se("cand", "ref") == pytest.approx(t.se["cand"])
    assert t.n_games == {"cand": 100, "ref": 100}
    # the same games listed from the other side give the same fit
    flipped = S.fit_btd([(b, a, 1.0 - s) for a, b, s in _games("cand", "ref", 75, 25)],
                        anchor="ref", prior_sd=math.inf)
    assert flipped.rating["cand"] == pytest.approx(t.rating["cand"], abs=1e-9)


def test_btd_map_satisfies_its_first_order_condition():
    t = S.fit_btd(_games("cand", "ref", 75, 25), anchor="ref", anchor_rating=1000.0)
    theta = (t.rating["cand"] - 1000.0) * S.ELO_SCALE
    sigma2 = (S.ELO_SCALE * 400.0) ** 2
    grad = 75 - 100 / (1 + math.exp(-theta)) - theta / sigma2
    assert abs(grad) < 1e-7
    assert 1000.0 < t.rating["cand"] < 1000.0 + 400 * math.log10(3)      # shrunk toward the anchor


def test_btd_davidson_two_players_is_saturated():
    """Two players, three outcomes, two parameters: the MLE reproduces the
    observed frequencies, so theta = ln(50/30) and nu = 0.2 / sqrt(0.5 * 0.3)."""
    t = S.fit_btd(_games("cand", "ref", 50, 30, 20), anchor="ref", prior_sd=math.inf, nu_prior_sd=math.inf)
    assert t.converged
    assert t.rating["cand"] == pytest.approx(400 * math.log10(5 / 3), abs=1e-6)  # 88.74
    assert t.nu == pytest.approx(0.2 / math.sqrt(0.15), rel=1e-7)                 # 0.5164
    pa, pb, pd = t.outcome_probabilities("cand", "ref")
    assert (pa, pb, pd) == (pytest.approx(0.5), pytest.approx(0.3), pytest.approx(0.2))


def test_btd_rare_draws_count_as_half_wins():
    t = S.fit_btd(_games("cand", "ref", 60, 39, 1), anchor="ref", prior_sd=math.inf)
    assert t.nu is None                                                  # 1% < min_draw_rate
    assert t.rating["cand"] == pytest.approx(S.elo_of_score(0.605), abs=1e-6)
    half = S.fit_btd(_games("cand", "ref", 60, 39, 1), anchor="ref", prior_sd=math.inf, draws="half_win")
    assert half.rating["cand"] == pytest.approx(t.rating["cand"], abs=1e-9)
    with pytest.raises(ValueError):
        S.fit_btd([], anchor="ref", draws="bogus")


def test_btd_recovers_planted_strengths():
    planted = {"ref": 0.0, "a": 150.0, "b": -100.0, "c": 300.0}
    nu = 0.3
    rng = np.random.default_rng(11)
    names = sorted(planted)
    results = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            qx, qy = 10 ** (planted[x] / 400), 10 ** (planted[y] / 400)
            dd = nu * math.sqrt(qx * qy)
            tot = qx + qy + dd
            wx, wy, dr = rng.multinomial(1500, [qx / tot, qy / tot, dd / tot])
            results += _games(x, y, int(wx), int(wy), int(dr))
    t = S.fit_btd(results, anchor="ref")
    assert t.converged
    assert t.nu == pytest.approx(nu, abs=0.04)
    for p in ("a", "b", "c"):
        assert 0 < t.se[p] < 20
        assert abs(t.rating[p] - planted[p]) < 4 * t.se[p] + 1, (p, t.rating[p], t.se[p])
    assert sorted(names, key=t.rating.get) == ["b", "ref", "a", "c"]
    # the fitted model predicts the planted one's c-vs-b score (0.877)
    qc, qb = 10 ** (300 / 400), 10 ** (-100 / 400)
    dd = nu * math.sqrt(qc * qb)
    assert t.predict("c", "b") == pytest.approx((qc + 0.5 * dd) / (qc + qb + dd), abs=0.02)
    assert t.difference_se("a", "ref") == pytest.approx(t.se["a"])
    assert 0 < t.difference_se("a", "c") < t.se["a"] + t.se["c"]
    assert t.n_games == {"a": 4500, "b": 4500, "c": 4500, "ref": 4500}
    # a pure function of the results: order does not matter
    shuffled = [results[i] for i in np.random.default_rng(5).permutation(len(results))]
    t2 = S.fit_btd(shuffled, anchor="ref")
    assert all(t2.rating[p] == pytest.approx(t.rating[p], abs=1e-6) for p in names)


def test_btd_anchor_without_games_and_round_trip():
    t = S.fit_btd(_games("a", "b", 10, 10), anchor="x", anchor_rating=1500.0)
    assert t.players == ("a", "b", "x")
    assert t.n_games["x"] == 0 and t.rating["x"] == 1500.0
    assert t.rating["a"] == pytest.approx(1500.0, abs=1e-6)          # symmetric games, prior mean = anchor
    d = t.to_dict()
    assert d["anchor"] == "x" and d["nu"] is None and set(d["rating"]) == {"a", "b", "x"}
    only = S.fit_btd([], anchor="ref")
    assert only.players == ("ref",) and only.converged and only.rating == {"ref": 0.0}
