"""R03: the RL league (tools/rl/league.py) and the opponents' deck field (tools/deck_stats.py).

League: PFSP "hard" weights (1 - p) ** 2 with the 0.2 uniform floor on synthetic win
rates (expected values worked out in exact fractions), decayed estimates, admission
every N iterations, anchors never evicted, eviction that keeps a spread, JSON state.
Deck field: synthetic IL_Replay rows (conftest) ranked per split, the sampler following
the frequencies and never dealing a deck from another split, the split and card-key
rules equal to tools/rs_rows.py's and tools/royalesim_redrive.py's.

numpy only; the parquet round trip needs pyarrow and is skipped without it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.pipeline.conftest import HOG26_KEYS, OPPONENT_KEYS, il_payload, il_replay_row  # noqa: E402
from tools import deck_stats as DS  # noqa: E402
from tools.rl import league as L  # noqa: E402

# ---------------------------------------------------------------------------
# PFSP weights
# ---------------------------------------------------------------------------
#: Synthetic expected scores of the learner against three opponents.
P = (0.2, 0.5, 0.8)
#: (1 - p) ** 2 = 64/100, 25/100, 4/100 (sum 93/100), so 0.8 * shape / sum + 0.2 / 3.
HARD = tuple(float(Fraction(4, 5) * Fraction(n, 93) + Fraction(1, 15)) for n in (64, 25, 4))


def test_pfsp_hard_weights_known_values():
    assert HARD == pytest.approx((0.6172043010752688, 0.2817204301075269, 0.1010752688172043),
                                 abs=1e-15)
    weights = L.pfsp_weights(P)
    assert weights.tolist() == pytest.approx(HARD, abs=1e-12)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_pfsp_uniform_floor_keeps_beaten_members_reachable():
    # The learner always beats nine members and never the tenth: each beaten one keeps 0.2 / 10.
    weights = L.pfsp_weights([0.0] + [1.0] * 9)
    assert weights[0] == pytest.approx(0.8 + 0.02, abs=1e-12)
    assert weights[1:].tolist() == pytest.approx([0.02] * 9, abs=1e-12)


def test_pfsp_weight_floor_scale_binds_without_the_uniform_share():
    weights = L.pfsp_weights([1.0, 0.0], uniform_floor=0.0)
    # shape [0, 1], floored at 0.01 / 2 = 0.005, renormalised by 1.005
    assert weights.tolist() == pytest.approx([0.005 / 1.005, 1.0 / 1.005], abs=1e-12)


def test_pfsp_shapes_and_degenerate_cases():
    assert L.pfsp_weights([1.0, 1.0, 1.0]).tolist() == pytest.approx([1 / 3] * 3, abs=1e-12)
    assert L.pfsp_shape([0.25, 0.5], "variance").tolist() == pytest.approx([0.1875, 0.25])
    assert L.pfsp_shape([0.25, 0.5], "linear").tolist() == pytest.approx([0.75, 0.5])
    assert L.pfsp_shape([0.25, 0.5], "uniform").tolist() == [1.0, 1.0]
    assert L.pfsp_shape([0.5], "hard", power=1.0).tolist() == [0.5]
    assert L.pfsp_weights([]).size == 0
    with pytest.raises(ValueError):
        L.pfsp_shape([0.5], "softmax")
    with pytest.raises(ValueError):
        L.pfsp_weights([1.2])
    with pytest.raises(ValueError):
        L.pfsp_weights([[0.5]])


# ---------------------------------------------------------------------------
# League: estimates, draws, admission, eviction, state
# ---------------------------------------------------------------------------
def _league_with_record(**config):
    """Anchors a, b, c with 1, 4 and 7 learner wins in 8 games: estimates 0.2, 0.5, 0.8."""
    league = L.League(L.LeagueConfig(**config), anchors=["a", "b", "c"])
    for member, wins in (("a", 1), ("b", 4), ("c", 7)):
        league.record_many([(member, 1)] * wins + [(member, "loss")] * (8 - wins))
    return league


def test_estimates_and_weights_follow_synthetic_win_rates():
    league = _league_with_record()
    # (wins + 1) / (games + 2): 2/10, 5/10, 8/10
    assert [league.expected(m) for m in ("a", "b", "c")] == pytest.approx(list(P), abs=1e-12)
    weights = league.weights()
    assert list(weights) == ["a", "b", "c"]
    assert list(weights.values()) == pytest.approx(HARD, abs=1e-12)


def test_draws_follow_the_weights_and_are_seeded():
    league = _league_with_record()
    draws = league.draw(np.random.default_rng(0), size=20_000)
    freq = [draws.count(m) / len(draws) for m in ("a", "b", "c")]
    assert freq == pytest.approx(HARD, abs=0.015)
    assert league.draw(7, size=50) == league.draw(np.random.default_rng(7), size=50)
    assert isinstance(league.draw(3), str)
    assert league.draw(3, size=0) == []


def test_decay_shrinks_old_evidence_toward_one_half():
    league = L.League(L.LeagueConfig(decay=0.5), anchors=["a"])
    league.record_many([("a", 1.0)] * 7 + [("a", 0.0)] * 3)
    assert league.expected("a") == pytest.approx(8 / 12, abs=1e-12)
    assert league.end_iteration() == {"iteration": 1, "admitted": None, "evicted": []}
    member = league.member("a")
    assert (member.decayed_games, member.decayed_score) == (5.0, 3.5)
    assert league.expected("a") == pytest.approx(4.5 / 7, abs=1e-12)
    assert (member.games, member.score) == (10, 7.0)        # raw totals are never decayed
    league.record("a", 0.0)                                 # a new game counts in full
    assert league.expected("a") == pytest.approx(4.5 / 8, abs=1e-12)
    assert member.last_played == 1


def test_admission_every_n_iterations():
    league = L.League(L.LeagueConfig(admit_every=3),
                      anchors={"il:rs_big": "runs/tokens/rs_big/best.pt"})
    admitted = []
    for it in range(1, 10):
        out = league.end_iteration(snapshot=Path("runs") / "rl" / f"it{it:03d}.pt")
        assert out["iteration"] == it
        if out["admitted"]:
            admitted.append(out["admitted"])
    assert admitted == ["snap:it0003", "snap:it0006", "snap:it0009"]
    first = league.member("snap:it0003")
    assert (first.path, first.iteration, first.kind) == (str(Path("runs") / "rl" / "it003.pt"), 3,
                                                         L.KIND_SNAPSHOT)
    assert league.v0 == "snap:it0003"
    assert league.member("il:rs_big").path == "runs/tokens/rs_big/best.pt"
    assert league.active_ids() == ("il:rs_big", "snap:it0003", "snap:it0006", "snap:it0009")
    assert not league.admission_due(0)
    off = L.League(L.LeagueConfig(admit_every=0), anchors=["a"])
    assert not any(off.admission_due(i) for i in range(1, 20))
    assert off.end_iteration(snapshot="x.pt")["admitted"] is None


def test_anchors_are_never_evicted():
    league = L.League(L.LeagueConfig(admit_every=1, max_snapshots=2, protect_newest=1),
                      anchors={"il:rs_big": None, "il:rs_bc": None})
    league.promote("il:rs_big")                  # the imitation model starts as champion
    for it in range(12):
        for opponent in league.active_ids():
            league.record_many([(opponent, 1.0)] * (it % 3) + [(opponent, 0.0)] * 2)
        league.end_iteration(snapshot=f"it{it + 1:03d}.pt")
        assert {"il:rs_big", "il:rs_bc"} <= set(league.active_ids())
        assert len(league.snapshot_ids(active_only=True)) <= 2
    # the run's first snapshot and the newest are protected; the rest made room
    assert league.snapshot_ids(active_only=True) == ("snap:it0001", "snap:it0012")
    assert len(league.snapshot_ids()) == 12                 # evicted ones stay archived
    assert league.evictions == 10
    with pytest.raises(ValueError):
        league.evict(["il:rs_bc"])
    assert "il:rs_bc" in league.active_ids()


def _spread_league():
    """Anchor il; snapshots s0 (v0), o1-o4 and s5 (newest). Learner records against o1-o4:
    1/10, 1/4, 3/4, 9/10 wins, so estimates 2/12, 2/6, 4/6, 10/12."""
    league = L.League(L.LeagueConfig(admit_every=0, max_snapshots=10, decay=1.0), anchors=["il"])
    for member_id in ("s0", "o1", "o2", "o3", "o4", "s5"):
        league.admit(f"{member_id}.pt", member_id=member_id)
    for member_id, wins, games in (("o1", 1, 10), ("o2", 1, 4), ("o3", 3, 4), ("o4", 9, 10)):
        league.record_many([(member_id, 1)] * wins + [(member_id, 0)] * (games - wins))
    return league


def test_eviction_keeps_a_spread_of_win_rates_not_the_newest():
    league = _spread_league()
    assert league.evictions == 0
    league.config = L.LeagueConfig(admit_every=0, max_snapshots=4, decay=1.0)
    # 6 snapshots, 2 protected (s0, s5): 2 places for o1-o4, strata [o1, o2] and [o3, o4],
    # each keeps its best-measured member
    assert league.enforce_capacity() == ["o2", "o3"]
    assert league.active_ids() == ("il", "s0", "o1", "o4", "s5")
    assert league.evictions == 2
    assert league.record("o2", "win")                       # still takes in-flight results
    assert league.member("o2").games == 5
    assert "o2" not in league.weights()


def test_champion_chain_is_protected_from_eviction():
    league = _spread_league()
    league.promote("o2")
    league.config = L.LeagueConfig(admit_every=0, max_snapshots=4, decay=1.0)
    assert "o2" in league.protected_ids()
    # one place left for o1, o3, o4: o1 and o4 have 10 games each, the later o4 wins the tie
    assert league.enforce_capacity() == ["o1", "o3"]
    assert league.active_ids() == ("il", "s0", "o2", "o4", "s5")


def test_record_and_membership_checks():
    league = L.League(anchors=["a"])
    assert league.record(L.LEARNER, 1.0) is False           # a mirror says nothing of the pool
    assert league.record("a", "draw") is True
    assert league.member("a").score == 0.5
    with pytest.raises(KeyError):
        league.record("nobody", 1)
    for bad in (1.5, -0.1, float("nan"), "won"):
        with pytest.raises(ValueError):
            league.record("a", bad)
    assert league.member("a").games == 1
    with pytest.raises(ValueError):
        league.add_anchor("a")                              # one id, one set of weights
    with pytest.raises(ValueError):
        league.add_anchor(L.LEARNER)
    with pytest.raises(ValueError):
        league.admit("x.pt", member_id="a")
    with pytest.raises(KeyError):
        league.promote("nobody")
    for bad in ({"decay": 0.0}, {"decay": 1.5}, {"weighting": "softmax"}, {"uniform_floor": 1.2},
                {"admit_every": -1}, {"max_snapshots": 1.5}, {"prior_games": 0.0},
                {"mirror_share": -0.1}):
        with pytest.raises(ValueError):
            L.LeagueConfig(**bad)


def test_mirror_share_draws_the_live_policy():
    league = L.League(L.LeagueConfig(mirror_share=0.25), anchors=["a", "b"])
    draws = league.draw(5, size=20_000)
    assert draws.count(L.LEARNER) / len(draws) == pytest.approx(0.25, abs=0.015)
    assert set(draws) == {L.LEARNER, "a", "b"}
    assert L.League(L.LeagueConfig(mirror_share=1.0)).draw(0, size=3) == [L.LEARNER] * 3
    with pytest.raises(RuntimeError):
        L.League().draw(0)                                  # nobody to play


def test_weights_can_use_fitted_ratings():
    league = L.League(anchors=["a", "b"])
    expected = L.elo_expected({"a": 0.0, "b": 400.0}, learner_rating=0.0)
    assert expected == pytest.approx({"a": 0.5, "b": 1 / 11})
    weights = league.weights(expected=expected)
    assert list(weights.values()) == pytest.approx(L.pfsp_weights([0.5, 1 / 11]).tolist(),
                                                   abs=1e-12)
    # a member the table does not rate keeps its own estimate (1/2 with no games)
    assert league.weights(expected={"b": 1 / 11}) == weights


def test_state_round_trips_through_json(tmp_path):
    league = _spread_league()
    league.promote("o4")
    league.end_iteration()
    league.evict(["o2"])
    path = league.save(tmp_path / "run" / "league.json")
    assert not (tmp_path / "run" / "league.json.tmp").exists()
    again = L.League.load(path)
    assert again.to_dict() == league.to_dict()
    assert again.weights() == league.weights()
    assert again.draw(11, size=100) == league.draw(11, size=100)
    resumed = L.League.load(path, config=L.LeagueConfig(admit_every=2, decay=1.0))
    assert resumed.config.admit_every == 2 and resumed.member("o4").games == 10
    assert (resumed.champion, resumed.champion_chain, resumed.v0) == ("o4", ["o4"], "s0")
    assert resumed.iteration == 1 and not resumed.member("o2").active
    row = json.loads(json.dumps(league.summary()))          # one JSON-able log row
    assert (row["iteration"], row["champion"], row["active"], row["evictions"]) == (1, "o4", 6, 1)
    assert [m["id"] for m in row["members"] if not m["active"]] == ["o2"]
    assert L.League(anchors="solo").anchor_ids() == ("solo",)      # one anchor given as a string
    data = json.loads(path.read_text(encoding="utf-8"))
    data["format_version"] = 99
    with pytest.raises(ValueError):
        L.League.from_dict(data)


# ---------------------------------------------------------------------------
# deck field
# ---------------------------------------------------------------------------
#: IL_Replay key -> RoyaleSim name for the decks below (royaleimitate replay_cards.json).
NAMES = {"cannon": "Cannon", "fireball": "Fireball", "hog-rider": "HogRider",
         "ice-golem": "IceGolemite", "ice-spirit": "IceSpirits", "musketeer": "Musketeer",
         "skeletons": "Skeletons", "the-log": "Log", "knight": "Knight", "archers": "Archer",
         "giant": "Giant", "arrows": "Arrows", "valkyrie": "Valkyrie", "mini-pekka": "MiniPekka",
         "baby-dragon": "BabyDragon", "zap": "Zap"}
HOG_NAMES = ("Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer",
             "Skeletons", "Log")                            # aligned with sorted keys
D3_KEYS = ("knight", "archers", "giant", "arrows", "valkyrie", "mini-pekka", "baby-dragon",
           "fireball")
D4_KEYS = ("knight", "archers", "giant", "zap", "valkyrie", "mini-pekka", "baby-dragon",
           "the-log")
UNMAPPED_KEYS = ("goblin-barrel", "archers", "giant", "arrows", "valkyrie", "mini-pekka",
                 "baby-dragon", "zap")
HOG, OPP = tuple(sorted(HOG26_KEYS)), tuple(sorted(OPPONENT_KEYS))
D3, D4, UNMAPPED = tuple(sorted(D3_KEYS)), tuple(sorted(D4_KEYS)), tuple(sorted(UNMAPPED_KEYS))
HOG_FORMS = {"skeletons": 1, "musketeer": 2}
#: Forms aligned with HOG (cannon, fireball, hog-rider, ice-golem, ice-spirit, musketeer,
#: skeletons, the-log).
EVO_HERO = (0, 0, 0, 0, 0, 2, 1, 0)
EVO_ONLY = (0, 0, 0, 0, 0, 0, 1, 0)
PLAIN = (0,) * 8


def _tags(split: str, n: int) -> list[str]:
    """The first n tags m-00000, m-00001, ... that fall in `split`."""
    out, i = [], 0
    while len(out) < n:
        tag = f"m-{i:05d}"
        if DS.split_of(tag) == split:
            out.append(tag)
        i += 1
    return out


def _row(tag, team, opponent, *, result="victory", team_forms=None, opponent_forms=None,
         mode="Ladder"):
    return il_replay_row(il_payload(tag=tag, team_deck=team, opponent_deck=opponent,
                                    team_forms=team_forms or {}, opponent_forms=opponent_forms or {},
                                    plays=(), abilities=(), result=result, game_mode=mode))


def _corpus() -> list[dict]:
    """train: Hog 4 seats (score 2.5), OPP 3 (0.5), D3 1 (1); val: D3 3 (3), OPP 2, D4 1;
    test: Hog 2 (a mirror); then one row skipped for each of mode, card and not 1v1."""
    train, val, test = _tags("train", 7), _tags("val", 3), _tags("test", 1)
    two_players = il_payload(tag=train[6], team_forms={}, opponent_forms={}, plays=(),
                             abilities=())
    two_players["battle"]["team"]["players"].append(dict(two_players["battle"]["team"]["players"][0]))
    return [
        _row(train[0], HOG26_KEYS, OPPONENT_KEYS, team_forms=HOG_FORMS),
        _row(train[1], HOG26_KEYS, OPPONENT_KEYS, team_forms=HOG_FORMS),
        _row(train[2], HOG26_KEYS, OPPONENT_KEYS, result="draw"),
        _row(train[3], HOG26_KEYS, D3_KEYS, result="defeat", team_forms={"skeletons": 1}),
        _row(val[0], D3_KEYS, OPPONENT_KEYS),
        _row(val[1], OPPONENT_KEYS, D3_KEYS, result="defeat"),
        _row(val[2], D3_KEYS, D4_KEYS),
        _row(test[0], HOG26_KEYS, HOG26_KEYS),
        _row(train[4], HOG26_KEYS, OPPONENT_KEYS, mode="Friendly"),
        _row(train[5], UNMAPPED_KEYS, HOG26_KEYS),
        il_replay_row(two_players),
    ]


def test_decks_ranked_by_frequency_per_split():
    counter = DS.count_rows(_corpus(), NAMES)
    assert counter.read == 11
    assert dict(counter.skipped) == {"mode": 1, "card": 1, "not 1v1": 1}
    assert counter.matches == {"train": 4, "val": 3, "test": 1}
    stats = counter.stats()
    train = stats.splits["train"]
    assert (train.matches, train.seats, train.distinct) == (4, 8, 3)
    assert [(d.cards, d.count) for d in train.decks] == [(HOG, 4), (OPP, 3), (D3, 1)]
    assert [d.share for d in train.decks] == [0.5, 0.375, 0.125]
    assert [d.score for d in train.decks] == [2.5, 0.5, 1.0]
    hog = train.decks[0]
    assert hog.win_rate == 0.625
    assert hog.names == HOG_NAMES
    assert hog.forms == ((EVO_HERO, 2), (PLAIN, 1), (EVO_ONLY, 1))
    assert hog.modal_forms == EVO_HERO
    val = stats.splits["val"]
    assert [(d.cards, d.count, d.score) for d in val.decks] == [(D3, 3, 3.0), (OPP, 2, 0.0),
                                                               (D4, 1, 0.0)]
    assert [(d.cards, d.count) for d in stats.splits["test"].decks] == [(HOG, 2)]
    assert stats.meta["read"] == 11 and stats.meta["named"] is True
    assert stats.meta["modes"] == sorted(DS.MODES)


def test_deck_field_follows_frequencies_and_respects_the_split():
    stats = DS.count_rows(_corpus(), NAMES).stats()
    train = stats.deck_field("train")
    assert train.probabilities.tolist() == pytest.approx([0.5, 0.375, 0.125], abs=1e-12)
    draws = train.sample(np.random.default_rng(0), size=20_000)
    freq = Counter(d.cards for d in draws)
    for cards, p in ((HOG, 0.5), (OPP, 0.375), (D3, 0.125)):
        assert freq[cards] / len(draws) == pytest.approx(p, abs=0.015)
    # D4 was only played in val matches: the train field never deals it
    assert D4 not in freq and train.probability(D4) == 0.0
    assert {d.split for d in draws} == {"train"}
    val = DS.DeckField.from_stats(stats, "val")
    assert val.probabilities.tolist() == pytest.approx([3 / 6, 2 / 6, 1 / 6], abs=1e-12)
    assert val.probability(D4) == pytest.approx(1 / 6)
    assert val.probability([NAMES[k] for k in D3]) == pytest.approx(3 / 6)   # by names too
    assert DS.DeckField.from_stats(stats, "test").sample(1).cards == HOG
    with pytest.raises(ValueError):
        stats.deck_field("final")


def test_deck_field_options():
    stats = DS.count_rows(_corpus(), NAMES).stats()
    assert stats.deck_field("train", alpha=0.0).probabilities.tolist() == pytest.approx([1 / 3] * 3)
    no_hog = stats.deck_field("train", exclude=[DS.HOG26_KEYS])
    assert [d.cards for d in no_hog.decks] == [OPP, D3]
    assert no_hog.probabilities.tolist() == pytest.approx([0.75, 0.25])
    by_names = stats.deck_field("train", exclude=[set(HOG_NAMES)])
    assert [d.cards for d in by_names.decks] == [OPP, D3]
    assert [d.cards for d in stats.deck_field("train", top=1).decks] == [HOG]
    assert [d.cards for d in stats.deck_field("train", min_count=2).decks] == [HOG, OPP]
    with pytest.raises(ValueError):
        stats.deck_field("train", min_count=5)              # nothing left to deal


def test_deck_field_deals_forms_as_played():
    stats = DS.count_rows(_corpus(), NAMES).stats()
    hog_only = stats.deck_field("train", top=1)
    draws = hog_only.sample(3, size=4_000)
    forms = Counter(d.forms for d in draws)
    assert set(forms) == {EVO_HERO, PLAIN, EVO_ONLY}
    assert forms[EVO_HERO] / len(draws) == pytest.approx(0.5, abs=0.04)
    assert forms[PLAIN] / len(draws) == pytest.approx(0.25, abs=0.04)
    assert draws[0].names == HOG_NAMES and draws[0].cards == HOG
    modal = stats.deck_field("train", top=1, sample_forms=False).sample(3, size=50)
    assert {d.forms for d in modal} == {EVO_HERO}


def test_counters_merge_like_one_pass():
    rows = _corpus()
    whole = DS.count_rows(rows, NAMES)
    parts = DS.count_rows(rows[:5], NAMES).merge(DS.count_rows(rows[5:], NAMES))
    assert parts.stats() == whole.stats()


def test_unnamed_and_every_mode_counting():
    counter = DS.count_rows(_corpus(), names=None, modes=None)
    assert dict(counter.skipped) == {"not 1v1": 1}
    train = counter.stats().splits["train"]
    assert train.matches == 6                               # + the Friendly and unmapped rows
    assert {d.cards: d.count for d in train.decks} == {HOG: 6, OPP: 4, D3: 1, UNMAPPED: 1}
    assert all(d.names is None for d in train.decks)


def test_stats_round_trip_through_json(tmp_path):
    stats = DS.count_rows(_corpus(), NAMES).stats(meta={"corpus": "synthetic"})
    path = stats.save(tmp_path / "deck_stats.json")
    assert DS.load_stats(path) == stats
    field = DS.DeckField.from_file(path, "val")
    assert field.probabilities.tolist() == pytest.approx([3 / 6, 2 / 6, 1 / 6], abs=1e-12)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == DS.SCHEMA and data["corpus"] == "synthetic"
    assert data["splits"]["train"]["decks"][0]["cards"] == list(HOG)
    data["schema"] = 99
    with pytest.raises(ValueError):
        DS.DeckStats.from_dict(data)


def _function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if fn is None:
        pytest.skip(f"{path.name} no longer defines {name}")
    ns = dict(namespace)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


def test_split_rule_is_the_row_builders():
    legacy = _function(ROOT / "tools" / "rs_rows.py", "split_of", {"hashlib": hashlib})
    tags = [f"#{i:08X}" for i in range(2000)] + ["synthetic-0001", "m-00000", "åäö", ""]
    assert [DS.split_of(t) for t in tags] == [legacy(t) for t in tags]
    assert {DS.split_of(t) for t in tags} == {"train", "val", "test"}


def test_card_key_rule_and_modes_are_the_row_builders():
    legacy = _function(ROOT / "tools" / "royalesim_redrive.py", "base_key", {})
    for key in ("hog-rider", "skeletons-ev1", "musketeer-hero", "the-log", "x-ev1-hero"):
        assert DS.base_key(key) == legacy(key)
    tree = ast.parse((ROOT / "tools" / "rs_rows.py").read_text(encoding="utf-8"))
    node = next((n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "MODES" for t in n.targets)), None)
    if node is None:
        pytest.skip("tools/rs_rows.py no longer assigns MODES")
    assert set(ast.literal_eval(node.value)) == set(DS.MODES)
    assert DS.HOG26_KEYS == frozenset(HOG26_KEYS)


def test_the_vendored_card_map_names_hog():
    from tools import rs_cards
    names = rs_cards.load(extras=False)
    counter = DS.count_rows([_row(_tags("train", 1)[0], HOG26_KEYS, OPPONENT_KEYS)], names)
    decks = {d.cards: d.names for d in counter.stats().splits["train"].decks}
    assert decks[HOG] == HOG_NAMES
    assert frozenset(decks[HOG]) == {"Cannon", "Fireball", "HogRider", "IceGolemite",
                                     "IceSpirits", "Musketeer", "Skeletons", "Log"}  # rs_rows.HOG26


def test_parquet_corpus_counts_like_rows(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = _corpus()
    (tmp_path / "replays").mkdir()
    pq.write_table(pa.Table.from_pylist(rows[:6]), str(tmp_path / "replays" / "part-00000.parquet"))
    pq.write_table(pa.Table.from_pylist(rows[6:]), str(tmp_path / "replays" / "part-00001.parquet"))
    files = DS.corpus_parts(tmp_path)
    assert [f.name for f in files] == ["part-00000.parquet", "part-00001.parquet"]
    assert DS.corpus_parts(tmp_path, parts=1) == files[:1]
    assert DS.count_corpus(files, NAMES).stats() == DS.count_rows(rows, NAMES).stats()
    out = tmp_path / "out" / "deck_stats.json"
    assert DS.main(["--corpus", str(tmp_path), "--out", str(out), "--no-names", "--top", "0"]) == 0
    unnamed = DS.load_stats(out)
    assert [(d.cards, d.count) for d in unnamed.splits["train"].decks][:2] == [(HOG, 5), (OPP, 3)]
    assert unnamed.meta["parts"] == ["part-00000.parquet", "part-00001.parquet"]


# ---------------------------------------------------------------------------
# import hygiene
# ---------------------------------------------------------------------------
FORBIDDEN = {"torch", "royalesim", "royalelearn", "royalegym", "royaleimitate", "scripts"}


def _imports(path: Path, top_only: bool) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if top_only else list(ast.walk(tree))
    out: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            out |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_imports_stay_light():
    for rel in ("tools/rl/__init__.py", "tools/rl/league.py", "tools/deck_stats.py"):
        every = _imports(ROOT / rel, top_only=False)
        assert not {name.split(".")[0] for name in every} & FORBIDDEN, rel
        assert not any("memread" in name or "gamestate" in name for name in every), rel
    assert _imports(ROOT / "tools/rl/__init__.py", top_only=False) == set()
    top = {name.split(".")[0] for name in _imports(ROOT / "tools/deck_stats.py", top_only=True)}
    assert "pyarrow" not in top                             # only when a corpus is read
    assert "pyarrow.parquet" in _imports(ROOT / "tools/deck_stats.py", top_only=False)
