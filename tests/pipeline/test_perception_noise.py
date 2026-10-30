"""E17: tools/perception_noise.py, the perception-noise model for rows and evaluation.

What is checked (the plan's test list first): an all-zero profile is the identity (an equal copy,
no draw taken) and gives obs_row's legacy rows byte for byte; a seed gives one result; empirical
rates over 10k draws sit within 4 sigma of the profile (the plan says 3; nothing could be run where
this was written, and 4 keeps a seeded draw from failing one run in 370 per check); the board lag
returns a historical state; the input is never changed. Then every kind on hand-worked values,
the profile file format, the shipped prior and the import rule.

The engine is a small synthetic one (UNITS, CATALOGUE below) with the shipped class table,
data/vision/classes.json; level-11 hit points are the class table's level-1 values x 2.56,
floored (vision_classes level_multiplier_percent[10] = 256): Knight 690 -> 1766, Archer 119 -> 304,
Musketeer 282 -> 721, Skeleton and Bat 32 -> 81, evolved Bat 48 -> 122, Giant 1550 -> 3968,
Cannon 322 -> 824, Hog Rider 663 -> 1697, Baby Dragon 450 -> 1152, Minion 90 -> 230, Ice Golem
480 -> 1228, Ice Spirits 84 -> 215.
"""
from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import pytest

from tests.pipeline.conftest import FIELD, KING_HP, PRINCESS_HP, entity_row, player_dict, state_dict

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import perception_noise as PN  # noqa: E402

import rs_memory  # noqa: E402  (tools/ is on sys.path once perception_noise is imported)

CLASSES = PN.vision_classes.load()
SUB = 18
F = FIELD

# ---------------------------------------------------------------------------
# a small engine
# ---------------------------------------------------------------------------

UNITS = ["Archer", "Archer_EV1", "BabyDragon", "Bat", "Bat_EV1", "Cannon", "Giant", "HogRider",
         "IceGolemite", "IceSpirits", "KingTower", "Knight", "Minion", "Musketeer",
         "MusketeerHero", "PrincessTower", "Skeleton", "Skeleton_EV1", "TowerPrincess"]
U = {n: i for i, n in enumerate(UNITS)}
#: (name, elixir, kind, flying) in catalogue id order; catalogue rows carry the kind in column 8.
CARD_ROWS = (
    ("Cannon", 3, "BUILDING", False), ("Fireball", 4, "SPELL", False),
    ("HogRider", 4, "TROOP", False), ("IceGolemite", 2, "TROOP", False),
    ("IceSpirits", 1, "TROOP", False), ("Musketeer", 4, "TROOP", False),
    ("Skeletons", 1, "TROOP", False), ("Log", 2, "SPELL", False),
    ("Knight", 3, "TROOP", False), ("Archer", 3, "TROOP", False),
    ("Giant", 5, "TROOP", False), ("Arrows", 3, "SPELL", False),
    ("Bats", 2, "TROOP", True), ("Minions", 3, "TROOP", True),
    ("BabyDragon", 4, "TROOP", True), ("Zap", 2, "SPELL", False),
)
CATALOGUE = [[n, 0, cost, 1, 0, fly, 0, 0, kind] for n, cost, kind, fly in CARD_ROWS]
C = {row[0]: i for i, row in enumerate(CATALOGUE)}
MAX_HP = {"Knight": 1766, "Archer": 304, "Archer_EV1": 304, "Musketeer": 721,
          "MusketeerHero": 721, "Skeleton": 81, "Skeleton_EV1": 81, "Bat": 81, "Bat_EV1": 122,
          "Giant": 3968, "Cannon": 824, "HogRider": 1697, "BabyDragon": 1152, "Minion": 230,
          "IceGolemite": 1228, "IceSpirits": 215}
CARD_OF = {"Knight": "Knight", "Archer": "Archer", "Archer_EV1": "Archer", "Musketeer": "Musketeer",
           "MusketeerHero": "Musketeer", "Skeleton": "Skeletons", "Skeleton_EV1": "Skeletons",
           "Bat": "Bats", "Bat_EV1": "Bats", "Giant": "Giant", "Cannon": "Cannon",
           "HogRider": "HogRider", "BabyDragon": "BabyDragon", "Minion": "Minions",
           "IceGolemite": "IceGolemite", "IceSpirits": "IceSpirits"}
FLYING = {"BabyDragon", "Bat", "Bat_EV1", "Minion"}
BUILDINGS = {"Cannon"}


def unit(uid, name, team, x_mt, y_mt, *, hp=None, deploy=0, **extra):
    """A unit entity of the small engine at world millitiles (x_mt, y_mt)."""
    mx = MAX_HP[name]
    return entity_row(uid=uid, team=team, kind=1 if name in BUILDINGS else 0,
                      card_id=C[CARD_OF[name]], x=x_mt * SUB, y=y_mt * SUB,
                      hp=mx if hp is None else hp, max_hp=mx, flying=name in FLYING,
                      deploy_ticks=deploy, unit_type=U[name], level=11, **extra)


def bound(profile, **kw):
    """A model bound to the small engine and the shipped class table."""
    return PN.NoiseModel(profile, unit_types=UNITS, catalogue=CATALOGUE, classes=CLASSES,
                         fields=FIELD, sub=SUB, **kw)


def plain(profile):
    """A model bound to nothing (kinds that name no class)."""
    return PN.NoiseModel(profile, fields=FIELD, sub=SUB)


def units_of(state):
    return [e for e in state["entities"] if e[F["tower_slot"]] < 0 and e[F["hp"]] > 0]


def by_uid(state):
    return {e[F["uid"]]: e for e in units_of(state)}


def assert_rate(k, n, p, sigmas=4.0):
    assert n > 0
    assert abs(k / n - p) <= sigmas * math.sqrt(p * (1 - p) / n), (k, n, p)


def busy_state(tick=1234):
    """Every case at once: damaged, deploying, evolved, hero, flying, a building, a dead unit,
    damaged and fallen towers, fractional elixir."""
    ents = [unit(101, "Knight", 1, 3500, 20000, hp=1000),
            unit(102, "Skeleton_EV1", 0, 9000, 9000, deploy=12),
            unit(103, "MusketeerHero", 0, 14500, 12000, hp=500),
            unit(104, "BabyDragon", 1, 9000, 18000),
            unit(105, "Cannon", 1, 9000, 24000, hp=600),
            unit(106, "Giant", 0, 3500, 14000),
            unit(107, "Archer", 1, 5000, 21000, hp=0)]
    players = [player_dict(0, elixir_milli=6543, tower_hp=(KING_HP, 2731, PRINCESS_HP)),
               player_dict(1, elixir_milli=10000, tower_hp=(4000, 0, 1315))]
    return state_dict(tick, players=players, entities=ents)


def moving_states(n=30, start=100):
    """A match: a Knight and a Baby Dragon walking, a Cannon, four units dropped in later."""
    out = []
    for t in range(start, start + n):
        d = t - start
        ents = [unit(1, "Knight", 1, 3500 + 10 * d, 20000 - 30 * d, hp=1500),
                unit(2, "BabyDragon", 0, 9000, 9000 + 40 * d),
                unit(3, "Cannon", 1, 9000, 24000, hp=700)]
        for k, name in enumerate(("Skeleton_EV1", "MusketeerHero", "Archer", "Giant")):
            spawn = start + 5 + 6 * k
            if t >= spawn:
                ents.append(unit(10 + k, name, k % 2, 4000 + 2500 * k, 12000 + 3000 * k,
                                 deploy=max(0, 20 - (t - spawn))))
        players = [player_dict(0, elixir_milli=min(10000, 3000 + 77 * d)),
                   player_dict(1, elixir_milli=4321, tower_hp=(KING_HP, 2731, PRINCESS_HP))]
        out.append(state_dict(t, players=players, entities=ents))
    return out


@pytest.fixture
def rr(monkeypatch):
    """tools/rs_rows as the tools import it, with conftest's FIELD and this file's catalogue
    flags and class lookup (what _setup would read from royalesim), as test_public_state does."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    monkeypatch.setattr(rr, "F", dict(FIELD))
    monkeypatch.setattr(rr, "LIVE", False)
    monkeypatch.setattr(rr, "CARD_FLY", {i: bool(row[5]) for i, row in enumerate(CATALOGUE)})
    monkeypatch.setattr(rr, "CARD_BUILDING", {i: row[8] == "BUILDING"
                                              for i, row in enumerate(CATALOGUE)})
    monkeypatch.setattr(rr, "VC", CLASSES.lookup(UNITS))
    return rr


def same_row(a, b):
    assert list(a) == list(b)
    for key in a:
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key


# ---------------------------------------------------------------------------
# the plan's tests
# ---------------------------------------------------------------------------

def test_an_all_zero_profile_is_the_identity():
    zero = PN.NoiseProfile()
    assert zero.is_identity() and zero.active_groups() == []
    # settings that add no noise of their own keep it the identity
    assert PN.NoiseProfile.from_dict({
        "size": {"small_max_hp": 10, "large_min_hp": 20}, "board": {"max_hold_ticks": 9},
        "elixir": {"fraction_fill_milli": 500}, "clock": {"quant_ticks": 1},
        "towers": {"quant_hp": 1}, "opponent_plays": {"spell_miss_p": 0.0}}).is_identity()
    for model in (plain(zero), bound(zero), PN.NoiseModel("none")):
        rng = np.random.default_rng(5)
        before = copy.deepcopy(rng.bit_generator.state)
        history = PN.NoiseHistory()
        stats = {}
        for tick in (1234, 1240, 1250):
            st = busy_state(tick)
            frozen = copy.deepcopy(st)
            for team in (0, 1):
                out = model.apply(st, team, rng, history, stats=stats)
                assert out == st and out is not st
                assert out["entities"] is not st["entities"]
                assert all(a is not b for a, b in zip(out["entities"], st["entities"]))
                assert out["players"][team] is not st["players"][team]
                assert out["players"][team]["tower_hp"] is not st["players"][team]["tower_hp"]
            assert model.apply(st, 1, rng) == st              # without a history too
            assert st == frozen
        assert stats == {"observations": 6}
        assert model.play_seen(1, C["Knight"] + 2, 300, rng) == (C["Knight"] + 2, 300)
        assert rng.bit_generator.state == before              # not one draw was taken
    nz = PN.MatchNoise(zero, seed=3)
    assert nz.observe(busy_state(), 1) == busy_state() and nz.tick_shift == 0
    mem = rs_memory.PublicMemory()
    assert nz.record(mem, 0, C["Knight"] + 2, 400)
    assert mem.plays(0) == [(C["Knight"] + 2, 400)]


def test_a_zero_profile_hook_gives_the_legacy_rows(rr):
    """With --noise and a zero profile the hook changes no row (types, live-compat, vision)."""
    mem = rs_memory.empty_view()
    zero = bound(PN.NoiseProfile())
    for team in (0, 1):
        st = busy_state(1500)
        nz = PN.MatchNoise(zero, PN.match_seed("synthetic-0001"))
        seen = nz.observe(st, team)
        for live in (True, False):
            same_row(rr.obs_row(st, team, SUB, mem, 1500, live=live),
                     rr.obs_row(seen, team, SUB, mem, 1500 + nz.tick_shift, live=live))
        same_row(rr.obs_row(st, team, SUB, mem, 1500, live=False, units="vision"),
                 rr.obs_row(seen, team, SUB, mem, 1500 + nz.tick_shift, live=False, units="vision"))
    # and the noise does reach the row: a profile that drops every unit leaves only padding
    gone = plain({"drop": {"small": 1, "medium": 1, "large": 1, "building": 1}})
    row = rr.obs_row(gone.apply(busy_state(1500), 1, np.random.default_rng(0)), 1, SUB, mem,
                     1500, live=False)
    assert not row["unit_ids"].any()


def test_a_seed_gives_one_result():
    prior = PN.NoiseProfile.load("default")

    def run(seed, model):
        nz = PN.MatchNoise(model, seed)
        mem = rs_memory.PublicMemory()
        outs = []
        for st in moving_states():
            outs.extend(nz.observe(st, team) for team in (0, 1))
            nz.record(mem, 0, C["Knight"] + 2, st["tick"])
        return outs, mem.plays(0), dict(nz.stats)

    a = run(7, bound(prior))
    assert a == run(7, bound(prior))           # a fresh model: no state outside the match
    assert a != run(8, bound(prior))
    assert a[2]["observations"] == 60 and a[2]["plays"] == 30


def test_unit_noise_rates_over_10k_draws():
    prof = {"drop": {"small": 0.2, "medium": 0.1, "large": 0.05, "building": 0.02},
            "side_flip": {"p": 0.1}, "confusion": {"unknown_p": 0.05}, "hp": {"no_bar_p": 0.25}}
    model = plain(prof)
    ents, truth = [], {}
    for b, name in enumerate(("Skeleton", "Knight", "Giant", "Cannon"), start=1):
        for k in range(50):
            e = unit(1000 * b + k, name, k % 2, 1000 + 300 * k, 4000 + 6000 * (b - 1),
                     hp=MAX_HP[name] // 2)
            ents.append(e)
            truth[e[F["uid"]]] = e
    st = state_dict(500, entities=ents)
    assert [model.bucket(truth[1000 * b]) for b in (1, 2, 3, 4)] == list(PN.BUCKETS)
    rng = np.random.default_rng(20261030)
    stats = {}
    kept = {b: 0 for b in (1, 2, 3, 4)}
    flips = unknown = no_bar = 0
    for _ in range(200):
        for e in units_of(model.apply(st, 1, rng, stats=stats)):
            t = truth[e[F["uid"]]]
            kept[e[F["uid"]] // 1000] += 1
            flips += e[F["team"]] != t[F["team"]]
            unknown += e[F["unit_type"]] == -1 and e[F["card_id"]] == -1
            no_bar += e[F["hp"]] == e[F["max_hp"]]
    for b, p in zip((1, 2, 3, 4), (0.2, 0.1, 0.05, 0.02)):
        assert_rate(10_000 - kept[b], 10_000, p)
    survivors = sum(kept.values())
    assert_rate(flips, survivors, 0.1)
    assert_rate(unknown, survivors, 0.05)
    assert_rate(no_bar, survivors, 0.25)
    assert stats["observations"] == 200 and stats["units"] == 40_000
    assert stats["dropped"] == 40_000 - survivors
    assert (stats["flipped"], stats["unknown"], stats["no_bar"]) == (flips, unknown, no_bar)


def test_position_noise_rates_and_the_flyer_offset():
    model = plain({"position": {"sigma_x_mt": 300, "sigma_y_mt": 500}})
    st = state_dict(400, entities=[unit(k, "Knight", k % 2, 9000, 8000) for k in range(50)])
    rng = np.random.default_rng(42)
    dx, dy = [], []
    for _ in range(200):
        for e in units_of(model.apply(st, 1, rng)):
            dx.append(e[F["x"]] / SUB - 9000)
            dy.append(e[F["y"]] / SUB - 8000)
    n = len(dx)
    assert n == 10_000
    for d, sigma in ((np.asarray(dx), 300.0), (np.asarray(dy), 500.0)):
        assert abs(d.mean()) <= 4 * sigma / math.sqrt(n)
        assert abs(d.std(ddof=1) - sigma) <= 4 * sigma / math.sqrt(2 * n)
    # A flyer bias is in the observer's screen frame: the world frame for team 1, half a turn
    # for team 0. No sigma, so no draw: exact. Ground units do not move; positions clamp.
    model = plain({"position": {"flyer_dx_mt": -200, "flyer_dy_mt": 600}})
    st = state_dict(400, entities=[unit(1, "BabyDragon", 0, 9000, 18000),
                                   unit(2, "Knight", 1, 3500, 20000),
                                   unit(3, "Minion", 1, 100, 31900)])
    for team, want in ((1, {1: ((9000 - 200) * SUB, (18000 + 600) * SUB),
                            3: (0, 32000 * SUB)}),
                       (0, {1: ((9000 + 200) * SUB, (18000 - 600) * SUB),
                            3: ((100 + 200) * SUB, (31900 - 600) * SUB)})):
        got = by_uid(model.apply(st, team, np.random.default_rng(0)))
        assert {uid: (got[uid][F["x"]], got[uid][F["y"]]) for uid in (1, 3)} == want
        assert (got[2][F["x"]], got[2][F["y"]]) == (3500 * SUB, 20000 * SUB)


def test_false_positives():
    # A double box: a copy of a reported unit (near_sigma 0: on top of it), negative uids.
    model = plain({"false_positives": {"rate": 0.5}})
    st = state_dict(300, entities=[unit(1, "Knight", 1, 3500, 20000), unit(2, "Skeleton", 0, 9000, 9000)])
    sources = [tuple(v for i, v in enumerate(e) if i != F["uid"]) for e in units_of(st)]
    rng = np.random.default_rng(99)
    stats = {}
    total = 0
    for _ in range(10_000):
        ph = [e for e in model.apply(st, 1, rng, stats=stats)["entities"] if e[F["uid"]] < 0]
        assert [e[F["uid"]] for e in ph] == [-(k + 1) for k in range(len(ph))]
        for e in ph:
            assert tuple(v for i, v in enumerate(e) if i != F["uid"]) in sources
        total += len(ph)
    assert abs(total - 5000) <= 4 * math.sqrt(5000)
    assert stats["phantoms"] == total
    # an empty board has nothing to see twice
    empty = state_dict(300)
    assert model.apply(empty, 1, np.random.default_rng(1)) == empty
    # Phantoms of a class: its unit type, card and level-11 hit points, the team of the half
    # it stands in.
    model = bound({"false_positives": {"rate": 3.0, "classes": {"Knight": 1.0}}})
    rng = np.random.default_rng(5)
    total = 0
    for _ in range(200):
        ph = [e for e in model.apply(empty, 0, rng)["entities"] if e[F["uid"]] < 0]
        total += len(ph)
        for e in ph:
            assert (e[F["unit_type"]], e[F["card_id"]], e[F["kind"]], e[F["flying"]]) == (
                U["Knight"], C["Knight"], 0, False)
            assert (e[F["hp"]], e[F["max_hp"]], e[F["tower_slot"]], e[F["deploy_ticks"]],
                    e[F["level"]], e[F["status_flags"]]) == (1766, 1766, -1, 0, 11, 0)
            assert 0 <= e[F["x"]] <= 18000 * SUB and 0 <= e[F["y"]] <= 32000 * SUB
            assert e[F["team"]] == (0 if e[F["y"]] < 16000 * SUB else 1)
    assert abs(total - 600) <= 4 * math.sqrt(600)


def test_board_lag_returns_a_historical_state():
    def lag_state(t):
        players = [player_dict(0, elixir_milli=100 * t), player_dict(1, tower_hp=(KING_HP, 3000 - t, PRINCESS_HP))]
        return state_dict(t, players=players, entities=[unit(7, "Knight", 1, 1000 + 100 * (t - 100), 20000)])

    nz = PN.MatchNoise(plain({"board": {"lag_ticks": 3}}), 1)
    for t in range(100, 111):
        out = nz.observe(lag_state(t), 1)
        if t >= 103:
            assert out == lag_state(t - 3) and nz.tick_shift == -3
        else:                                   # nothing that old yet: the present
            assert out == lag_state(t) and nz.tick_shift == 0
    assert (nz.stats["board_lag"], nz.stats["board_lag_unavailable"]) == (8, 3)
    with pytest.raises(ValueError, match="tick order"):
        nz.observe(lag_state(105), 1)
    # A sparse history: the newest state at or before tick - lag, unless it is more than
    # max_hold_ticks older than that.
    for hold, want in ((2, 110), (10, 100)):
        model = plain({"board": {"lag_ticks": 3, "max_hold_ticks": hold}})
        history = PN.NoiseHistory()
        model.push(history, lag_state(100))
        stats = {}
        assert model.apply(lag_state(110), 1, np.random.default_rng(0), history,
                           stats=stats) == lag_state(want)
        assert stats.get("board_lag_unavailable", 0) == (1 if hold == 2 else 0)
    with pytest.raises(ValueError, match="NoiseHistory"):
        plain({"board": {"lag_ticks": 3}}).apply(lag_state(110), 1, np.random.default_rng(0))
    # the ring keeps max_ticks of the past
    history = PN.NoiseHistory(5)
    for t in range(100, 111):
        plain({}).push(history, lag_state(t))
    assert history.ticks == list(range(105, 111)) and history.at(104) is None
    assert history.at(108) == lag_state(108)


def test_the_input_is_never_changed():
    prior = PN.NoiseProfile.load("default")
    prof = dataclasses.replace(prior, board=PN.Board(lag_ticks=1),
                               false_positives=PN.FalsePositives(rate=2.0, classes={"Knight": 1.0}))
    nz = PN.MatchNoise(bound(prof), 11)
    states = moving_states()
    frozen = copy.deepcopy(states)
    outs = [nz.observe(st, team) for st in states for team in (0, 1)]
    assert states == frozen                     # nor the states the history holds
    mine = {id(x) for st in states for x in st["entities"] + st["players"]}
    mine |= {id(p["tower_hp"]) for st in states for p in st["players"]}
    for out in outs:
        assert not any(id(x) in mine for x in out["entities"] + out["players"])
        assert not any(id(p["tower_hp"]) in mine for p in out["players"])
    last = outs[-1]
    last["entities"][0][F["x"]] = -5
    last["players"][1]["tower_hp"][1] = -5
    last["players"][1]["elixir_milli"] = -5
    assert states == frozen
    assert nz.stats["board_lag"] == 58          # every observation after the first tick


# ---------------------------------------------------------------------------
# each kind, hand-worked
# ---------------------------------------------------------------------------

def test_new_units_hidden_and_the_deploying_flag_from_track_age():
    model = plain({"new_units": {"delay_ticks": 2}, "deploy": {"age_offset_ticks": 2}})
    nz = PN.MatchNoise(model, 0)
    for t in range(100, 126):
        ents = [unit(1, "Knight", 1, 3500, 20000)]
        if t >= 101:                            # dropped in on 101: 20 ticks of deploy
            ents.append(unit(2, "Skeleton", 0, 9000, 9000, deploy=max(0, 20 - (t - 101))))
        got = by_uid(nz.observe(state_dict(t, entities=ents), 1))
        assert got[1][F["deploy_ticks"]] == 0   # on the board before the history: old
        if t in (101, 102):                     # 0 and 1 tick old: not confirmed yet
            assert 2 not in got
        elif t >= 103:
            # the track's age counts from 2 ticks late: the timer reads 2 more, and runs on
            # 2 ticks past its end (121 -> 123)
            assert got[2][F["deploy_ticks"]] == max(0, 123 - t), t
    assert nz.stats["hidden_new"] == 2
    # Without a history, the age comes from the deploy timer alone (20 ticks unbound).
    st = state_dict(500, entities=[unit(k, "Skeleton", 0, 2000 * k, 9000, deploy=d)
                                   for k, d in ((1, 20), (2, 19), (3, 18), (4, 0))])
    got = by_uid(model.apply(st, 1, np.random.default_rng(0)))
    assert sorted(got) == [3, 4]
    assert (got[3][F["deploy_ticks"]], got[4][F["deploy_ticks"]]) == (20, 0)
    # A unit already deploying when the history begins is as old as its timer says.
    history = PN.NoiseHistory()
    got = by_uid(model.apply(state_dict(500, entities=[unit(9, "Knight", 1, 3500, 20000, deploy=5)]),
                             1, np.random.default_rng(0), history))
    assert got[9][F["deploy_ticks"]] == 7 and history.units[9].spawn == 485


def test_unit_bars():
    st = state_dict(200, entities=[unit(k, "Knight", 1, 2000 * k, 20000, hp=hp) for k, hp in
                                   ((1, 1000), (2, 1700), (3, 1), (4, 1766), (5, 1650), (6, 1600),
                                    (7, 1680))])

    def hps(profile):
        got = by_uid(plain(profile).apply(st, 1, np.random.default_rng(0)))
        return [got[k][F["hp"]] for k in range(1, 8)]

    # quant 0.1: 1000/1766 = 0.566 -> 0.6 -> 1060; 0.963 -> 1.0; 0.0006 -> 0, at least 1;
    # full stays; 0.934 -> 0.9 -> 1589; 0.906 -> 0.9 -> 1589; 0.951 -> 1.0
    assert hps({"hp": {"quant": 0.1}}) == [1060, 1766, 1, 1766, 1589, 1589, 1766]
    # full_snap alone: 0.963 and 0.951 read full, the rest is exact
    assert hps({"hp": {"full_snap": 0.95}}) == [1000, 1766, 1, 1766, 1650, 1600, 1766]
    # no bar found: a damaged unit reads full
    assert hps({"hp": {"no_bar_p": 1.0}}) == [1766] * 7


def test_forms_collapse_to_base():
    st = state_dict(300, entities=[unit(1, "MusketeerHero", 0, 9000, 9000, hp=500),
                                   unit(2, "Skeleton_EV1", 0, 9000, 10000),
                                   unit(3, "Bat_EV1", 1, 9000, 20000),
                                   unit(4, "Archer_EV1", 1, 9000, 21000, hp=200),
                                   unit(5, "Knight", 1, 9000, 22000, hp=1000)])

    def forms(profile):
        got = by_uid(bound(profile).apply(st, 1, np.random.default_rng(0)))
        return {k: (UNITS[e[F["unit_type"]]], e[F["hp"]], e[F["max_hp"]], e[F["flying"]],
                    e[F["card_id"]]) for k, e in got.items()}

    # KataCR has no hero: the hero Musketeer reads as the Musketeer (721 hp at level 11, 500
    # of it); the labelled evolutions stay.
    assert forms({"forms": {"unlabelled_to_base": True}}) == {
        1: ("Musketeer", 500, 721, False, C["Musketeer"]),
        2: ("Skeleton_EV1", 81, 81, False, C["Skeletons"]),
        3: ("Bat_EV1", 122, 122, True, C["Bats"]),
        4: ("Archer_EV1", 200, 304, False, C["Archer"]),
        5: ("Knight", 1000, 1766, False, C["Knight"])}
    # every evolution to base: the evolved Bat's 122 hp becomes the Bat's 81 (still full), the
    # Archer keeps 200 / 304 of the same max; the hero stays (hero_to_base_p 0, labelled rule off)
    assert forms({"forms": {"evo_to_base_p": 1.0}}) == {
        1: ("MusketeerHero", 500, 721, False, C["Musketeer"]),
        2: ("Skeleton", 81, 81, False, C["Skeletons"]),
        3: ("Bat", 81, 81, True, C["Bats"]),
        4: ("Archer", 200, 304, False, C["Archer"]),
        5: ("Knight", 1000, 1766, False, C["Knight"])}
    # the rate
    model = bound({"forms": {"evo_to_base_p": 0.25}})
    many = state_dict(300, entities=[unit(k, "Skeleton_EV1", 0, 300 + 300 * k, 9000) for k in range(50)])
    rng = np.random.default_rng(3)
    stats = {}
    for _ in range(200):
        model.apply(many, 1, rng, stats=stats)
    assert_rate(stats["collapsed"], 10_000, 0.25)


def test_class_confusion():
    st = state_dict(300, entities=[unit(1, "Knight", 1, 3500, 20000, hp=883),
                                   unit(2, "Giant", 0, 9000, 9000)])
    # A Knight read as an Archer: the Archer's unit type, card, flags and 304 hp, at the same
    # fill (883 / 1766 = 0.5 -> 152). The Giant has no confusions.
    got = by_uid(bound({"confusion": {"pairs": {"Knight": {"Archer": 1.0}}}}).apply(
        st, 1, np.random.default_rng(0)))
    k = got[1]
    assert (k[F["unit_type"]], k[F["card_id"]], k[F["kind"]], k[F["flying"]], k[F["hp"]],
            k[F["max_hp"]]) == (U["Archer"], C["Archer"], 0, False, 152, 304)
    assert got[2] == st["entities"][-1]
    # "unknown": unit type and card -1 (row token 1), everything else as it was
    got = by_uid(bound({"confusion": {"pairs": {"Knight": {"unknown": 1.0}}}}).apply(
        st, 1, np.random.default_rng(0)))
    want = list(st["entities"][-2])
    want[F["unit_type"]] = want[F["card_id"]] = -1
    assert got[1] == want
    # rates: Archer with 0.3, then unknown with 0.1 of the rest (0.07)
    model = bound({"confusion": {"unknown_p": 0.1, "pairs": {"Knight": {"Archer": 0.3}}}})
    many = state_dict(300, entities=[unit(k, "Knight", 1, 300 + 300 * k, 20000) for k in range(50)])
    rng = np.random.default_rng(17)
    archer = unknown = 0
    for _ in range(200):
        for e in units_of(model.apply(many, 1, rng)):
            archer += e[F["unit_type"]] == U["Archer"]
            unknown += e[F["unit_type"]] == -1
    assert_rate(archer, 10_000, 0.3)
    assert_rate(unknown, 10_000, 0.07)


def test_per_track_decisions_are_held_for_the_unit_life():
    model = bound({"forms": {"evo_to_base_p": 0.5}, "side_flip": {"p": 0.5}})
    nz = PN.MatchNoise(model, 21)
    seen = {}
    for t in (300, 301, 302):
        st = state_dict(t, entities=[unit(k, "Skeleton_EV1", 0, 300 + 15 * k, 9000)
                                     for k in range(1000)])
        for uid, e in by_uid(nz.observe(st, 1)).items():
            seen.setdefault(uid, set()).add((e[F["unit_type"]], e[F["team"]]))
    assert all(len(v) == 1 for v in seen.values())          # one reading per track
    collapsed = sum(next(iter(v))[0] == U["Skeleton"] for v in seen.values())
    flipped = sum(next(iter(v))[1] == 1 for v in seen.values())
    assert_rate(collapsed, 1000, 0.5)
    assert_rate(flipped, 1000, 0.5)
    nz.observe(st, 0)                           # another observer draws its own
    assert set(nz.history.units[5].sticky) == {(1, "form"), (1, "flip"), (0, "form"), (0, "flip")}


def test_towers_elixir_and_clock():
    def players(profile, team=1):
        out = plain(profile).apply(busy_state(1234), team, np.random.default_rng(0))
        return [p["tower_hp"] for p in out["players"]], [p["elixir_milli"] for p in out["players"]]

    # quant 100: a damaged tower rounds (2731 -> 2700, 4000 stays, 49 -> 0 -> at least 1);
    # full towers read full, a fallen one 0
    st = busy_state(1234)
    st["players"][1]["tower_hp"] = [4000, 0, 49]
    out = plain({"towers": {"quant_hp": 100}}).apply(st, 1, np.random.default_rng(0))
    assert [p["tower_hp"] for p in out["players"]] == [[KING_HP, 2700, PRINCESS_HP], [4000, 0, 1]]
    # the bar fallback reading 2 points low: 2731 - 61.04 -> 2670; 4000 - 96.48 -> 3904;
    # 49 - 61.04 -> at least 1
    out = plain({"towers": {"bias_frac": -0.02}}).apply(st, 1, np.random.default_rng(0))
    assert [p["tower_hp"] for p in out["players"]] == [[KING_HP, 2670, PRINCESS_HP], [3904, 0, 1]]
    # elixir: our own only; the fraction lost -> the filter's fill; 10 reads 10000
    assert players({"elixir": {"fraction_lost_p": 1.0, "fraction_fill_milli": 500}}, team=0)[1] == [
        6500, 10000]
    assert players({"elixir": {"fraction_lost_p": 1.0, "fraction_fill_milli": 500}}, team=1)[1] == [
        6543, 10000]
    st = busy_state(1234)
    st["players"][1]["elixir_milli"] = 900
    out = plain({"elixir": {"fraction_lost_p": 1.0}}).apply(st, 1, np.random.default_rng(0))
    assert out["players"][1]["elixir_milli"] == 0
    # the clock: a bias, then the floor to the quantum, never below 0
    for profile, tick, want in (({"clock": {"bias_ticks": 3}}, 1234, 1237),
                                ({"clock": {"bias_ticks": -5, "quant_ticks": 20}}, 1234, 1220),
                                ({"clock": {"bias_ticks": -5}}, 2, 0)):
        nz = PN.MatchNoise(plain(profile), 0)
        out = nz.observe(state_dict(tick), 1)
        assert (out["tick"], out["tick_ms"], nz.tick_shift) == (want, want * 50, want - tick)


def test_hud_rates_over_10k_draws():
    st = state_dict(1234, players=[player_dict(0), player_dict(1, elixir_milli=5500)])
    model = plain({"elixir": {"digit_error_p": 0.2}, "clock": {"sigma_ticks": 2}})
    rng = np.random.default_rng(8)
    elixir, shift = [], []
    for _ in range(10_000):
        out = model.apply(st, 1, rng)
        elixir.append(out["players"][1]["elixir_milli"])
        shift.append(out["tick"] - 1234)
        assert out["players"][0]["elixir_milli"] == 5000
    assert set(elixir) <= {4500, 5500, 6500}
    errors = sum(e != 5500 for e in elixir)
    assert_rate(errors, 10_000, 0.2)
    assert_rate(elixir.count(6500), errors, 0.5)
    s = np.asarray(shift, dtype=float)
    sd = math.sqrt(2.0 ** 2 + 1 / 12)        # a rounded N(0, 2): Sheppard's correction
    assert abs(s.mean()) <= 4 * sd / 100
    assert abs(s.std(ddof=1) - sd) <= 4 * sd / math.sqrt(20_000)


def test_opponent_plays_missed_late_or_unnamed():
    knight, fireball = C["Knight"] + 2, C["Fireball"] + 2
    model = PN.NoiseModel({"opponent_plays": {"miss_p": 0.1, "spell_miss_p": 0.4, "late_ticks": 4,
                                              "unknown_p": 0.05}}, catalogue=CATALOGUE, fields=FIELD)
    rng = np.random.default_rng(12)
    troop = [model.play_seen(1, knight, 1000, rng) for _ in range(10_000)]
    seen = [s for s in troop if s is not None]
    assert_rate(10_000 - len(seen), 10_000, 0.1)
    assert {t for _, t in seen} == {1004}
    assert {c for c, _ in seen} == {knight, PN.UNKNOWN_CARD}
    assert_rate(sum(c == PN.UNKNOWN_CARD for c, _ in seen), len(seen), 0.05)
    spells = [model.play_seen(1, fireball, 1000, rng) for _ in range(10_000)]
    assert_rate(spells.count(None), 10_000, 0.4)
    # into the memory: 4 ticks late, so a view shows it from 304 on, aged from 304
    nz = PN.MatchNoise({"opponent_plays": {"late_ticks": 4}}, 0)
    mem = rs_memory.PublicMemory()
    assert nz.record(mem, 0, knight, 300)
    assert mem.view(303, 0)["last"][0] == [] and mem.view(304, 0)["last"][0] == [(knight, 304)]
    # missed: nothing recorded; spell_miss_p unset: spells miss as troops do
    nz = PN.MatchNoise({"opponent_plays": {"miss_p": 1.0}}, 0)
    mem = rs_memory.PublicMemory()
    assert not nz.record(mem, 0, knight, 300) and not nz.record(mem, 0, fireball, 300)
    assert len(mem) == 0 and nz.stats["plays_missed"] == 2
    # jitter spreads the delay evenly over late +- jitter
    model = PN.NoiseModel({"opponent_plays": {"late_ticks": 4, "late_jitter_ticks": 2}})
    rng = np.random.default_rng(4)
    assert {model.play_seen(0, knight, 50, rng)[1] - 50 for _ in range(1000)} == {2, 3, 4, 5, 6}
    with pytest.raises(ValueError, match="spell_miss_p"):
        PN.NoiseModel({"opponent_plays": {"spell_miss_p": 0.2}})


def test_tower_troops_and_dead_units_are_left_alone():
    troop = entity_row(uid=50, team=1, kind=0, card_id=-1, tower_slot=-1, x=3500 * SUB,
                       y=25500 * SUB, hp=5000, max_hp=5000, unit_type=U["TowerPrincess"])
    st = busy_state(700)
    st["entities"].append(troop)
    every = {"drop": {"small": 1, "medium": 1, "large": 1, "building": 1}}
    out = bound(every).apply(st, 1, np.random.default_rng(0))
    # a bound model knows the Tower Princess is a tower part; towers and the dead Archer stay
    assert [e[F["uid"]] for e in units_of(out)] == [50]
    assert [e for e in out["entities"] if e[F["tower_slot"]] >= 0 or e[F["hp"]] <= 0] == [
        e for e in st["entities"] if e[F["tower_slot"]] >= 0 or e[F["hp"]] <= 0]
    assert units_of(plain(every).apply(st, 1, np.random.default_rng(0))) == []


# ---------------------------------------------------------------------------
# the profile
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data, match", [
    ({"drop": {"smal": 0.1}}, "unknown key 'smal'"),
    ({"wobble": {}}, "unknown group 'wobble'"),
    ({"drop": {"small": 1.5}}, "probability"),
    ({"drop": {"small": True}}, "a number"),
    ({"position": {"sigma_x_mt": -1}}, ">= 0"),
    ({"forms": {"unlabelled_to_base": 1}}, "true or false"),
    ({"new_units": {"delay_ticks": 1.5}}, "whole number"),
    ({"confusion": {"pairs": {"Knight": {"Archer": 0.7, "MiniPekka": 0.4}}}}, "add up to"),
    ({"confusion": {"pairs": {"Knight": {"Knight": 0.1}}}}, "itself"),
    ({"size": {"small_max_hp": 3000, "large_min_hp": 2000}}, "small_max_hp"),
    ({"elixir": {"fraction_fill_milli": 1000}}, "0..999"),
    ({"false_positives": {"rate": 1, "classes": {"Knight": 0}}}, "add up to 0"),
    ({"schema": 2}, "schema"),
])
def test_bad_profiles_are_refused(data, match):
    with pytest.raises(ValueError, match=match):
        PN.NoiseProfile.from_dict(data)


def test_profile_json_round_trip_and_kinds(tmp_path):
    notes = PN.NoiseProfile.from_dict({"_comment": "x", "drop": {"_why": "y", "small": 0.1}})
    assert notes.drop.small == 0.1 and notes.active_groups() == ["drop"]
    prior = PN.NoiseProfile.load("default")
    assert prior == PN.NoiseProfile.load(PN.PRIOR_PATH) and prior.name == "default_prior"
    assert PN.NoiseProfile.from_dict(prior.to_dict()) == prior
    assert json.loads(prior.dumps()) == prior.to_dict()
    assert PN.NoiseProfile.from_dict(json.loads(prior.dumps())).digest() == prior.digest()
    assert dataclasses.replace(prior, name="x", description="y").digest() == prior.digest()
    assert prior.only("drop").digest() != prior.digest()
    assert PN.NoiseProfile.load("none") == PN.NoiseProfile() == PN.as_profile(None)
    assert prior.only("position").active_groups() == ["position"]
    assert prior.only("hp,drop").active_groups() == ["drop", "hp"]
    assert prior.without("position").active_groups() == [
        g for g in prior.active_groups() if g != "position"]
    assert prior.only("drop").size == prior.size
    with pytest.raises(ValueError, match="no noise kind"):
        prior.only("nope")
    with pytest.raises(FileNotFoundError):
        PN.NoiseProfile.load(tmp_path / "missing.json")
    path = tmp_path / "mine.json"
    path.write_text(json.dumps({"hp": {"quant": 0.05}}), encoding="utf-8")
    mine = PN.NoiseProfile.load(path)
    assert mine.name == "mine" and mine.hp.quant == 0.05 and mine.active_groups() == ["hp"]


def test_the_shipped_prior():
    raw = json.loads(PN.PRIOR_PATH.read_text(encoding="utf-8"))
    prior = PN.NoiseProfile.load("default")
    assert raw["schema"] == PN.SCHEMA and raw["_sources"]
    for group in PN.GROUPS + ("size",):
        assert raw[group].get("_why"), group                 # every value says why
    # Capture lag is inside D (E11 rows, the latency probe), so the board lag is off; the
    # rest is on.
    assert prior.active_groups() == [g for g in PN.GROUPS if g != "board"]
    assert prior.forms.hero_to_base_p == 1.0 and prior.forms.unlabelled_to_base
    # every class it names exists and resolves on an engine that has those units
    names = set(prior.confusion.pairs) | {t for row in prior.confusion.pairs.values() for t in row}
    assert all(CLASSES.get(n) is not None and CLASSES.get(n).name == n for n in names)
    units = sorted({CLASSES[n].form(CLASSES[n].base_form).unit for n in names})
    model = PN.NoiseModel(prior, unit_types=units, catalogue=CATALOGUE, classes=CLASSES, fields=FIELD)
    assert model.unresolved == []
    # the prior needs the engine: it names classes and tells spells apart
    with pytest.raises(ValueError, match="unit_types"):
        PN.NoiseModel(prior)


def test_kinds_that_name_classes_need_the_engine():
    pairs = {"confusion": {"pairs": {"Knight": {"Archer": 0.1}}}}
    with pytest.raises(ValueError, match="unit_types"):
        PN.NoiseModel(pairs)
    with pytest.raises(ValueError, match="catalogue"):
        PN.NoiseModel(pairs, unit_types=UNITS, classes=CLASSES)
    with pytest.raises(ValueError, match="unit_types"):
        PN.NoiseModel({"forms": {"unlabelled_to_base": True}})
    with pytest.raises(ValueError, match="unit_types"):
        PN.NoiseModel({"drop": {"by_class": {"Skeleton": 0.5}}})
    # names this engine or table lacks are skipped and listed
    model = bound({"confusion": {"pairs": {"Knight": {"Wizard": 0.1}, "NotAClass": {"Knight": 0.1}}},
                   "drop": {"by_class": {"Skeleton": 1.0, "Nope": 0.5}}})
    assert len(model.unresolved) == 3
    assert any("Wizard" in s for s in model.unresolved)
    assert any("NotAClass" in s for s in model.unresolved)
    assert any("Nope" in s for s in model.unresolved)
    # by_class overrides the bucket: every Skeleton is dropped, the Knight stays
    st = state_dict(10, entities=[unit(1, "Skeleton", 0, 9000, 9000), unit(2, "Knight", 1, 9000, 20000)])
    assert list(by_uid(model.apply(st, 1, np.random.default_rng(0)))) == [2]
    with pytest.raises(ValueError, match="team"):
        model.apply(st, 2, np.random.default_rng(0))
    with pytest.raises(TypeError, match="rng"):
        model.apply(st, 1, None)


# ---------------------------------------------------------------------------
# the command line, the hooks' helpers, the import rule
# ---------------------------------------------------------------------------

def test_noise_args_vocab_and_seeds():
    ap = argparse.ArgumentParser()
    PN.add_noise_args(ap)
    assert PN.profile_from_args(ap.parse_args([])) is None
    assert PN.profile_from_args(ap.parse_args(["--noise", "none"])) is None
    p = PN.profile_from_args(ap.parse_args(["--noise", "default", "--noise-only", "position,hp"]))
    assert p.active_groups() == ["position", "hp"]
    with pytest.raises(ValueError, match="--noise-only"):
        PN.profile_from_args(ap.parse_args(["--noise-only", "hp"]))
    assert ap.parse_args(["--noise-seed", "4"]).noise_seed == 4
    entry = PN.vocab_entry("default", 4)
    prior = PN.NoiseProfile.load("default")
    assert entry == {"schema": 1, "name": "default_prior", "digest": prior.digest(), "seed": 4,
                     "kinds": prior.active_groups(), "profile": prior.to_dict()}
    json.dumps(entry)
    want = int(hashlib.sha1(b"0:synthetic-0001").hexdigest()[:12], 16)
    assert PN.match_seed("synthetic-0001") == want != PN.match_seed("synthetic-0001", 1)


def test_command_line(tmp_path, capsys):
    assert PN.main(["none"]) == 0
    assert "noise zero: none (perfect observer)" in capsys.readouterr().out
    assert PN.main([]) == 0
    assert capsys.readouterr().out.startswith(f"noise default_prior [{PN.NoiseProfile.load('default').digest()}]")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"drop": {"small": 2}}), encoding="utf-8")
    assert PN.main([str(bad)]) == 2
    assert "probability" in capsys.readouterr().err


def test_imports_only_the_standard_library_numpy_and_two_tools():
    tree = ast.parse((TOOLS / "perception_noise.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "argparse", "bisect", "dataclasses", "hashlib", "json", "math",
                    "numbers", "sys", "collections", "pathlib", "typing", "numpy", "rs_engine",
                    "vision_classes"}, mods
