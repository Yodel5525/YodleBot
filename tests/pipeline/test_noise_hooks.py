"""E17 hooks: tools/perception_noise.py wired into tools/rs_rows.py (--noise PROFILE, --noise-only,
--noise-seed) and tools/rs_eval.py (--noise for the agent seat, --opponent-noise for the reactive
opponent seat).

Golden first: without a profile (the default, "none", or a seed alone) rows_for gives the rows,
info and engine calls of before, never makes a MatchNoise and calls obs_row with the keywords of
before; rs_eval.play gives the result, engine calls and state reads of before. An all-zero
profile runs every hook and changes nothing but the counts.

Then by hand, on conftest's fake royalesim (no engine, no corpus, no torch). The match is
test_decision_rows' (tag decision-0001; the team at low y plays blue), plays executing at

    105 red Knight (token 10)   290 red Arrows (13)   300 blue HogRider (4)   320 blue Musketeer (7)

and, in states_u, a red Knight unit of 100 hp (a "small" unit) standing at (9000, 20000). The
rows at execution are test_decision_rows' 52. Profile HOOKS: opponent plays seen 10 ticks late,
the clock read 3 ticks ahead, the elixir fraction always lost (read as 500), every small unit
dropped. None of these outcomes depends on a draw, so every value is exact:

  scalars          [5500 / 10000, (t + 3) * 50 / 300000, 0, 0] on every row (both sides hold 5000)
  unit_ids         all 0 (the Knight unit is never seen); without noise 10 first
  memory           the Knight dated 115, the Arrows 300, the HogRider 310 (the Musketeer 330):
                   blue's row at 300 holds the Knight only, age (303 - 115) / 400; at 320 both,
                   ages 208 / 400 and 23 / 400; red's wait at 320 the HogRider, age 13 / 400
  tick             the true one, so outcome, trust and the wait rows are those of before
  info["noise"]    52 observations, 52 units, 52 dropped, 52 fractions lost, 4 plays

Under --schema v2 (rs_elixir's constant law, test_schema_v2's: 6 elixir at tick 0, 500 of 28000
fine units a tick, milli = fine // 28) the opponent's counted elixir is fed what the memory got:
with the plays 10 ticks late, blue's count of red at 300 has the Knight (paid at 115: 225500 -
84000 + 500 = 142000 at 116, + 500 x 184 = 234000, 8357 milli) and not yet the Arrows (5357 without
noise); at 320 both (150500 at 301, + 500 x 19 = 160000, 5714, as without noise); red's count of
blue at 320 pays the HogRider at 310 from a full bar (280000 - 112000 + 500 = 168500 at 311,
+ 4500 = 173000, 6178; without noise 178000, 6357). An unknown card is not counted at all.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    SUBTILE_PER_MILLITILE,
    entity_row,
    il_payload,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
SUB = SUBTILE_PER_MILLITILE
TAG = "decision-0001"
WAIT = (0, 0, 0)
PLAYS = ((105, "opponent", "knight", 14500, 24500), (290, "opponent", "arrows", 3500, 8000),
         (300, "team", "hog-rider", 3500, 14500), (320, "team", "musketeer", 3500, 7500))
BLUE_EARLY, BLUE_LATE = ([2, 0, 6, 3], 5), ([5, 0, 6, 3], 1)
OPP_HAND, OPP_NEXT = [8, 11, 9, 10], 12
#: The fake catalogue's columns by name (test_schema_v2): the cost is column 1.
FAKE_FIELDS = ("name", "elixir", "c2", "c3", "c4", "flying", "c6", "c7", "card_kind")
#: A red Knight unit (unit type 8, card 8, token 10), 100 hp: the "small" size bucket.
KNIGHT = entity_row(uid=50, team=1, card_id=8, x=9000 * SUB, y=20000 * SUB, hp=100, unit_type=8)
#: Every outcome fixed: plays 10 ticks late, the clock 3 ahead, the fraction lost, small units gone.
HOOKS = {"name": "hooks", "opponent_plays": {"late_ticks": 10}, "clock": {"bias_ticks": 3},
         "elixir": {"fraction_lost_p": 1.0, "fraction_fill_milli": 500}, "drop": {"small": 1.0}}
HOOK_STATS = {"dropped": 52, "elixir_fraction_lost": 52, "observations": 52, "plays": 4,
              "units": 52}
DSTAT_A = {"play_rows": 4, "press_rows": 0, "not_in_hand": 1, "clamped": 0, "pending_rows": 6,
           "exact_rows": 0}
LEGACY_ROW = ("tick", "side", "hand_exact", "unit_ids", "unit_feats", "towers", "scalars",
              "hand", "next", "opp_seen", "opp_last", "opp_last_age")


def f16(values):
    return np.asarray(values, dtype=np.float16).tolist()


def f32(values):
    return np.asarray(values, dtype=np.float32).tolist()


def same(a, b, names):
    for k in names:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.dtype == y.dtype and np.array_equal(x, y), k


def same_row(a, b):
    assert list(a) == list(b)
    same(a, b, a)


def keys(rows):
    return [(r["tick"], r["side"], tuple(int(v) for v in r["action"])) for r in rows]


def by_key(rows):
    return {(r["tick"], r["side"], int(r["action"][0])): r for r in rows}


def payload_a(**kwargs):
    kwargs.setdefault("tag", TAG)
    kwargs.setdefault("plays", PLAYS)
    kwargs.setdefault("abilities", ())
    return il_payload(**kwargs)


def states_a(battle, units=()):
    hand, nxt = BLUE_EARLY if battle.tick <= 300 else BLUE_LATE
    return state_dict(battle.tick, entities=units,
                      players=[player_dict(0, hand=hand, next_card=nxt),
                               player_dict(1, hand=OPP_HAND, next_card=OPP_NEXT)])


def states_u(battle):
    """states_a with the red Knight unit throughout."""
    return states_a(battle, units=[KNIGHT])


@pytest.fixture
def rr(monkeypatch):
    """tools/rs_rows.py as the tools import it, every module setting at its default (restored
    after the test, NOISE and the model cache too), no engine config from the PC and
    rs_elixir's constant law."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "VISIBILITY": rr.VIS_LEGACY, "TRUST": rr.TrustRule(), "OUT": rr.OUT,
                        "CARD_FLY": {}, "CARD_BUILDING": {}, "PRESSES": "off",
                        "BUTTONS": 2, "HAND_SIZE": 4, "DECISION": rr.DecisionRule(),
                        "ROW_SCHEMA": rr.rs_schema.SCHEMA_LEGACY, "SF": None,
                        "UNIT_FORMS": None, "SPELL_CARDS": None, "WAIT_KEEP_AT_BUILD": None,
                        "NOISE": None, "NOISE_SEED": 0, "_NOISE_MODELS": {}}.items():
        monkeypatch.setattr(rr, name, value)
    law = rr.rs_elixir.ElixirLaw.constants()
    monkeypatch.setattr(rr.rs_elixir, "default_law", lambda: law)
    return rr


@pytest.fixture
def engine(monkeypatch):
    """install(**battle_kwargs): a fake royalesim in sys.modules for this test, its catalogue
    read by FAKE_FIELDS."""
    def install(**kwargs):
        mod = make_fake_royalesim(**kwargs)
        mod.CATALOGUE_FIELDS = list(FAKE_FIELDS)
        monkeypatch.setitem(sys.modules, "royalesim", mod)
        return mod
    return install


def cfg(rr):
    return rr.rs_engine.EngineConfig.defaults()


def run(rr, payload=None, **kwargs):
    """rows_for on `payload` (payload_a()): (rows, info)."""
    info = {}
    rows, why = rr.rows_for(payload_a() if payload is None else payload,
                            rr._setup(cfg(rr)), cfg(rr), info=info, **kwargs)
    assert why == "ok"
    return rows, info


def spy_on(rr, monkeypatch):
    """Every obs_row call's (tick argument, sorted keywords)."""
    seen = []
    real = rr.obs_row

    def spy(*args, **kwargs):
        seen.append((args[4], sorted(kwargs)))
        return real(*args, **kwargs)

    monkeypatch.setattr(rr, "obs_row", spy)
    return seen


# ---------------------------------------------------------------------------
# the settings, by hand
# ---------------------------------------------------------------------------

def test_noise_profile_seed_and_vocab_values(rr, tmp_path):
    PN = rr.PN
    for off in (None, "none", " NONE ", "off", "0", ""):
        assert rr.noise_profile(off) is None
    prior = rr.noise_profile("default")
    assert prior.name == "default_prior" and not prior.is_identity()
    again = rr.noise_profile(prior.to_dict())
    assert again.digest() == prior.digest() and again.name == prior.name
    assert rr.noise_profile(prior) is prior
    zero = rr.noise_profile({})
    assert zero.is_identity() and zero.name == "profile"
    foreign = types.SimpleNamespace(to_dict=lambda: {"clock": {"bias_ticks": 2}})
    assert rr.noise_profile(foreign).clock.bias_ticks == 2       # another import's profile
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(HOOKS), encoding="utf-8")
    assert rr.noise_profile(str(path)).digest() == PN.NoiseProfile.from_dict(HOOKS).digest()
    with pytest.raises(FileNotFoundError):
        rr.noise_profile(str(tmp_path / "nope.json"))
    with pytest.raises(TypeError):
        rr.noise_profile(5)
    with pytest.raises(ValueError):
        rr.noise_profile({"clock": {"bias_ticks": 1.5}})
    assert rr.noise_seed_value(3) == 3 and rr.noise_seed_value(np.int64(-2)) == -2
    assert type(rr.noise_seed_value(np.int64(4))) is int
    for bad in (True, 2.5, "3", None):
        with pytest.raises(ValueError, match="noise-seed"):
            rr.noise_seed_value(bad)
    assert rr.noise_vocab(None) == {} and rr.noise_vocab("none", 5) == {}
    entry = rr.noise_vocab(HOOKS, 7)["perception_noise"]
    assert entry == PN.vocab_entry(PN.NoiseProfile.from_dict(HOOKS), 7)
    assert (entry["name"], entry["seed"], entry["schema"]) == ("hooks", 7, PN.SCHEMA)
    assert entry["kinds"] == ["drop", "elixir", "clock", "opponent_plays"]
    assert json.loads(json.dumps(entry)) == entry
    assert rr.describe_noise(None).startswith("none")
    assert "run seed 7" in rr.describe_noise(HOOKS, 7) and "clock" in rr.describe_noise(HOOKS)


def test_ages_floor_at_zero(rr, engine):
    """obs_row's opp_last_age: a play dated after the row's tick (a clock read early) reads 0;
    every other age is as before."""
    engine()
    rr._setup(cfg(rr))
    view = {"seen": {0: set(), 1: {10, 13}}, "last": {0: [], 1: [(10, 100), (13, 210)]}}
    row = rr.obs_row(state_dict(200), 0, SUB, view, 200)
    assert row["opp_last_age"].tolist() == f16([100 / 400, 0.0, 1, 1])
    assert rr.obs_row(state_dict(220), 0, SUB, view, 220)["opp_last_age"].tolist() == \
        f16([120 / 400, 10 / 400, 1, 1])
    old = {"seen": {0: set(), 1: {10}}, "last": {0: [], 1: [(10, 0)]}}
    assert rr.obs_row(state_dict(900), 0, SUB, old, 900)["opp_last_age"].tolist() == \
        f16([1, 1, 1, 1])                                  # still capped at 1


def test_model_is_bound_once_per_profile_and_engine(rr, engine, monkeypatch):
    mod = engine()
    rr._setup(cfg(rr))
    battle = mod.Battle()
    assert rr.noise_model(None, battle, SUB) is None and rr.noise_model("none", battle, SUB) is None
    model = rr.noise_model(HOOKS, battle, SUB)
    assert isinstance(model, rr.PN.NoiseModel) and model.F == rr.F and model.sub == SUB
    assert model.unit_types == tuple(DEFAULT_UNIT_TYPES)
    assert model.card_names == tuple(r[0] for r in DEFAULT_CATALOGUE)
    assert model.unresolved == []
    assert rr.noise_model(dict(HOOKS, name="renamed"), mod.Battle(), SUB) is model   # same digest
    assert rr.noise_model(HOOKS, battle, 9) is not model
    typo = rr.noise_model({"confusion": {"pairs": {"Knigt": {"Archer": 0.5}}}}, battle, SUB)
    assert typo.unresolved == ["confusion.pairs: 'Knigt' is not a class"]   # main prints these
    monkeypatch.setattr(rr, "F", None)
    with pytest.raises(RuntimeError, match="_setup"):
        rr.noise_model(dict(HOOKS, drop={"small": 0.5}), battle, SUB)


# ---------------------------------------------------------------------------
# rs_rows: golden
# ---------------------------------------------------------------------------

def test_without_noise_the_rows_are_the_rows_before(rr, engine, monkeypatch):
    """No profile (the default, "none", a seed alone, NOISE "off"): the same rows, info and
    engine calls; no MatchNoise is made; obs_row gets the keywords and ticks of before."""
    mod = engine(states=states_u)
    base, base_info = run(rr)
    base_calls = mod.battles[-1].calls
    assert len(base) == 52 and base_info == {"trust": {"cut": None, "why": "none"}}
    monkeypatch.setattr(rr.PN, "MatchNoise", None)          # any use would fail
    for kwargs in ({"noise": "none"}, {"noise_seed": 9}, {"noise": None, "noise_seed": None},
                   "module off"):
        if kwargs == "module off":
            monkeypatch.setattr(rr, "NOISE", "off")
            kwargs = {}
        rows, info = run(rr, **kwargs)
        assert info == base_info and mod.battles[-1].calls == base_calls
        assert len(rows) == len(base)
        for a, b in zip(rows, base):
            same_row(a, b)
    seen = spy_on(rr, monkeypatch)
    rows, _ = run(rr)
    assert seen == [(r["tick"], []) for r in rows]
    seen.clear()
    rows, _ = run(rr, decision=21, noise="none")
    assert seen == [(r["tick"], ["hand_exact", "pending"]) for r in rows] and len(rows) == 54


@pytest.mark.parametrize("schema", ["legacy", "v2"])
def test_an_all_zero_profile_runs_the_hooks_and_changes_nothing(rr, engine, schema):
    mod = engine(states=states_u)
    base, base_info = run(rr, schema=schema)
    base_calls = mod.battles[-1].calls
    rows, info = run(rr, schema=schema, noise=rr.PN.NoiseProfile())
    assert mod.battles[-1].calls == base_calls and len(rows) == len(base) == 52
    for a, b in zip(rows, base):
        same_row(a, b)
    assert info == {**base_info, "noise": {"observations": 52, "plays": 4}}


# ---------------------------------------------------------------------------
# rs_rows: the hooks, by hand
# ---------------------------------------------------------------------------

def test_rows_are_observed_through_the_profile(rr, engine, monkeypatch):
    mod = engine(states=states_u)
    plain, plain_info = run(rr)
    plain_calls = mod.battles[-1].calls
    seen = spy_on(rr, monkeypatch)
    rows, info = run(rr, noise=HOOKS)
    assert mod.battles[-1].calls == plain_calls               # the engine runs every play
    assert keys(rows) == keys(plain) and len(rows) == 52
    assert info == {**plain_info, "noise": HOOK_STATS}
    assert seen == [(r["tick"] + 3, []) for r in rows]        # the clock read 3 ticks ahead
    for r, p in zip(rows, plain):
        t = r["tick"]
        assert list(r) == list(p)
        same(r, p, ("side", "hand_exact", "towers", "hand", "next"))
        assert (r["outcome"], r["trusted"], r["won"], r["hog"]) == (p["outcome"], p["trusted"],
                                                                    p["won"], p["hog"])
        assert r["scalars"].tolist() == f32([5500 / 10_000, min(1.0, (t + 3) * 50 / 300_000),
                                             0, 0])
        assert p["scalars"].tolist() == f32([5000 / 10_000, min(1.0, t * 50 / 300_000), 0, 0])
        assert not r["unit_ids"].any() and not r["unit_feats"].any()
        assert p["unit_ids"].tolist()[0] == 10
    by, pb = by_key(rows), by_key(plain)
    hog, musk, red, wait = by[(300, 1, 4)], by[(320, 1, 7)], by[(320, -1, 0)], by[(120, 1, 0)]
    assert hog["opp_last"].tolist() == [10, 0, 0, 0]          # the Arrows (300) not seen yet
    assert hog["opp_seen"].tolist() == [10] + [0] * 7
    assert hog["opp_last_age"].tolist() == f16([188 / 400, 1, 1, 1])
    assert pb[(300, 1, 4)]["opp_last"].tolist() == [10, 13, 0, 0]
    assert pb[(300, 1, 4)]["opp_last_age"].tolist() == f16([195 / 400, 10 / 400, 1, 1])
    assert musk["opp_last"].tolist() == [10, 13, 0, 0]
    assert musk["opp_last_age"].tolist() == f16([208 / 400, 23 / 400, 1, 1])
    assert red["opp_last"].tolist() == [4, 0, 0, 0]
    assert red["opp_last_age"].tolist() == f16([13 / 400, 1, 1, 1])
    assert pb[(320, -1, 0)]["opp_last_age"].tolist() == f16([20 / 400, 1, 1, 1])
    assert wait["opp_last_age"].tolist() == f16([8 / 400, 1, 1, 1])
    assert pb[(120, 1, 0)]["opp_last_age"].tolist() == f16([15 / 400, 1, 1, 1])
    assert not by[(105, -1, 10)]["opp_last"].any() and not by[(300, -1, 0)]["opp_last"].any()
    assert rows[0]["tick"] == 20 and type(rows[0]["tick"]) is int


def test_decision_time_rows_go_through_the_hook(rr, engine, monkeypatch):
    engine(states=states_u)
    plain, plain_info = run(rr, decision=21)
    seen = spy_on(rr, monkeypatch)
    rows, info = run(rr, decision=21, noise=HOOKS)
    assert keys(rows) == keys(plain) and len(rows) == 54
    assert info["decision"] == plain_info["decision"] == DSTAT_A
    assert info["noise"] == {"dropped": 54, "elixir_fraction_lost": 54, "observations": 54,
                             "plays": 4, "units": 54}
    assert seen == [(r["tick"] + 3, ["hand_exact", "pending"]) for r in rows]
    for r, p in zip(rows, plain):
        same(r, p, ("own_pending", "own_pending_land", "hand", "hand_exact"))
        assert r["scalars"][0] == np.float32(0.55) and not r["unit_ids"].any()


def test_a_clock_read_late_floors_the_ages(rr, engine):
    engine(states=states_a)
    rows, info = run(rr, noise={"clock": {"bias_ticks": -20}})
    assert info["noise"] == {"observations": 52, "plays": 4}
    hog = by_key(rows)[(300, 1, 4)]                         # read at 280: the Arrows ran at 290
    assert hog["tick"] == 300
    assert hog["opp_last"].tolist() == [10, 13, 0, 0]
    assert hog["opp_last_age"].tolist() == f16([175 / 400, 0.0, 1, 1])
    assert hog["scalars"].tolist() == f32([0.5, 280 * 50 / 300_000, 0, 0])
    assert rows[0]["scalars"].tolist() == f32([0.5, 0.0, 0, 0])      # tick 20 read as 0


def test_unknown_cards_in_memory_and_not_in_the_count(rr, engine):
    engine(states=states_a)
    unknown = {"opponent_plays": {"unknown_p": 1.0}}
    rows, info = run(rr, noise=unknown)
    assert info["noise"] == {"observations": 52, "plays": 4, "plays_unknown": 4}
    by = by_key(rows)
    assert by[(300, 1, 4)]["opp_last"].tolist() == [1, 1, 0, 0]
    assert by[(300, 1, 4)]["opp_seen"].tolist() == [1] + [0] * 7
    assert by[(300, 1, 4)]["opp_last_age"].tolist() == f16([195 / 400, 10 / 400, 1, 1])
    assert by[(320, -1, 0)]["opp_last"].tolist() == [1, 0, 0, 0]
    rows, info = run(rr, noise=unknown, schema="v2")
    by = by_key(rows)
    # nothing priced: blue's count of red runs from 6 elixir to full (168000 + 500 t, capped)
    assert by[(120, 1, 0)]["scalars"][5] == np.float32(8142 / 10_000)    # 228000 // 28
    assert by[(300, 1, 4)]["scalars"][5] == np.float32(1.0)              # 318000: full
    assert by[(300, 1, 4)]["opp_last"].tolist() == [1, 1, 0, 0]
    assert all(r["scalars"][6] == 1.0 for r in rows)
    assert info["observation"] == {"rows": 52, "opp_inexact": 0, "spell_rows": 0}


def test_v2_counts_the_plays_as_the_memory_got_them(rr, engine):
    engine(states=states_a)
    plain, _ = run(rr, schema="v2")
    rows, info = run(rr, schema="v2", noise={"opponent_plays": {"late_ticks": 10}})
    assert keys(rows) == keys(plain) and len(rows) == 52
    assert info["noise"] == {"observations": 52, "plays": 4}
    assert info["observation"] == {"rows": 52, "opp_inexact": 0, "spell_rows": 0}
    by, pb = by_key(rows), by_key(plain)
    for key, noisy, clean in (((300, 1, 4), 8357, 5357), ((320, 1, 7), 5714, 5714),
                              ((320, -1, 0), 6178, 6357), ((105, -1, 10), 7875, 7875)):
        assert by[key]["scalars"][5] == np.float32(noisy / 10_000), key
        assert pb[key]["scalars"][5] == np.float32(clean / 10_000), key
    assert by[(300, 1, 4)]["opp_last"].tolist() == [10, 0, 0, 0]
    assert all(r["scalars"][6] == 1.0 for r in rows)


def test_the_noise_is_seeded_by_the_match_tag(rr, engine):
    """Position noise (sigma x 300 millitiles): the first row is blue's wait at 20, the first
    draw of default_rng(match_seed(tag, seed)), moving the Knight's x by -18 x the draw in the
    world (blue's screen is the world turned half a turn)."""
    engine(states=states_u)
    PN, wobble = rr.PN, {"position": {"sigma_x_mt": 300}}
    plain, _ = run(rr)
    first, info = run(rr, noise=wobble)
    again, _ = run(rr, noise=wobble)
    for a, b in zip(first, again):
        same_row(a, b)
    assert info["noise"] == {"observations": 52, "plays": 4, "units": 52}
    other, _ = run(rr, noise=wobble, noise_seed=5)
    assert any(not np.array_equal(a["unit_feats"], b["unit_feats"]) for a, b in zip(first, other))
    for rows, seed in ((first, 0), (other, 5)):
        d = np.random.default_rng(PN.match_seed(TAG, seed)).normal(0.0, 300.0)
        x = min(18000 * SUB, max(0, math.floor(9000 * SUB - SUB * d + 0.5)))
        assert keys(rows)[0] == (20, 1, WAIT)
        feats = rows[0]["unit_feats"][0]
        assert feats[1] == np.float16(x / SUB / 18_000)
        assert feats[2] == plain[0]["unit_feats"][0][2]          # y: no noise
    renamed, _ = run(rr, payload_a(tag="another-tag"), noise=wobble)
    assert any(not np.array_equal(a["unit_feats"], b["unit_feats"]) for a, b in zip(first, renamed))


# ---------------------------------------------------------------------------
# rs_rows: build, manifest, vocab.json, the command line
# ---------------------------------------------------------------------------

def test_build_shard_and_manifest_under_noise(rr, engine, tmp_path):
    engine(states=states_u)
    text = [json.dumps(payload_a())]
    out = tmp_path / "rows"
    bid, n, skipped, st = rr.build(3, text, False, str(out), cfg(rr).to_json(),
                                   noise=rr.PN.NoiseProfile.from_dict(HOOKS).to_dict(),
                                   noise_seed=7, stats=True)
    assert (bid, n, skipped) == (3, 52, {}) and st["noise"] == HOOK_STATS
    assert rr.NOISE.digest() == rr.PN.NoiseProfile.from_dict(HOOKS).digest()
    assert rr.NOISE.name == "hooks" and rr.NOISE_SEED == 7
    kept = rr.build(4, text, False, str(out), cfg(rr).to_json(), stats=True)[3]
    assert kept["noise"] == HOOK_STATS                          # None keeps the module's
    off = rr.build(5, text, False, str(out), cfg(rr).to_json(), noise="none", stats=True)[3]
    assert "noise" not in off and rr.NOISE is None
    with np.load(out / "shard-0003.npz") as z, np.load(out / "shard-0005.npz") as y:
        assert list(z.files) == list(y.files)
        assert not z["unit_ids"].any() and (y["unit_ids"][:, 0] == 10).all()
        assert np.array_equal(z["tick"], y["tick"]) and np.array_equal(z["outcome"], y["outcome"])
        assert np.array_equal(z["act_card"], y["act_card"])
    man = rr.manifest(st, [rr.shard_entry(3, st)], {})
    assert man["noise"] == HOOK_STATS and "noise" not in rr.manifest(off, [], {})
    assert rr.describe_manifest(man).endswith(
        "; noise: 52 of 52 units unseen, 0 phantoms, 0 of 4 plays missed")
    merged = rr.merge_stats(rr.merge_stats({}, st), kept)
    assert merged["noise"] == {k: 2 * v for k, v in HOOK_STATS.items()}


def test_one_match_one_history_class_profiles_bind(rr, engine, tmp_path):
    """A profile that names classes (a confusion pair) and tells spells apart binds to the
    engine's unit types and catalogue; the Knight read as an Archer (unit type 9, token 11) or
    not is drawn once per observer and kept for the match: one MatchNoise per match."""
    engine(states=states_u)
    pairs = {"confusion": {"pairs": {"Knight": {"Archer": 0.5}}},
             "opponent_plays": {"spell_miss_p": 0.5}}
    _, n, skipped, st = rr.build(1, [json.dumps(payload_a())], False, str(tmp_path),
                                 cfg(rr).to_json(), noise=pairs, stats=True)
    assert (n, skipped) == (52, {}) and st["noise"]["plays"] == 4
    assert st["noise"]["observations"] == st["noise"]["units"] == 52
    assert st["noise"].get("plays_missed", 0) <= 1                  # only the Arrows can be
    with np.load(tmp_path / "shard-0001.npz") as z:
        for side in (1, -1):
            first = z["unit_ids"][z["side"] == side][:, 0]
            assert len(first) == 26 and set(first.tolist()) <= {10, 11}
            assert len(set(first.tolist())) == 1, side                # held for the match


def test_vocab_records_the_noise_only_when_given(rr):
    cat, units = [list(r) for r in DEFAULT_CATALOGUE], list(DEFAULT_UNIT_TYPES)
    eng, fp = cfg(rr), {"version": "0.0.0+fake", "card_table": None}
    base = rr.vocab_record(cat, units, eng, fp, {}, False)
    assert "perception_noise" not in base
    assert rr.vocab_record(cat, units, eng, fp, {}, False, **rr.noise_vocab(None)) == base
    rec = rr.vocab_record(cat, units, eng, fp, {}, False, **rr.noise_vocab(HOOKS, 7))
    assert list(rec)[-1] == "perception_noise" and {k: v for k, v in rec.items()
                                                    if k != "perception_noise"} == base
    assert rec["perception_noise"] == rr.PN.vocab_entry(rr.PN.NoiseProfile.from_dict(HOOKS), 7)
    v2 = rr.vocab_record(cat, units, eng, fp, {}, False, schema="v2", trust="v2",
                         **rr.noise_vocab(HOOKS, 7))
    assert list(v2)[-3:] == ["perception_noise", "schema_spec", "extra_columns"]
    assert json.loads(json.dumps(rec)) == rec


def test_rows_main_wires_the_noise(rr):
    text = (TOOLS / "rs_rows.py").read_text(encoding="utf-8")
    main = text[text.index("def main()"):]
    for needle in ("PN.add_noise_args(ap)", "noise = PN.profile_from_args(args)",
                   "NOISE, NOISE_SEED = noise, args.noise_seed",
                   "**noise_vocab(NOISE, NOISE_SEED)",
                   "noise=None if NOISE is None else NOISE.to_dict()", "noise_seed=NOISE_SEED",
                   "noise_model(NOISE, vocab_battle"):
        assert needle in main, needle
    ap = argparse.ArgumentParser()
    rr.PN.add_noise_args(ap)
    args = ap.parse_args([])
    assert (args.noise, args.noise_only, args.noise_seed) == (None, None, 0)
    assert rr.PN.profile_from_args(args) is None
    assert rr.PN.profile_from_args(ap.parse_args(["--noise", "none"])) is None
    with pytest.raises(ValueError, match="--noise-only"):
        rr.PN.profile_from_args(ap.parse_args(["--noise-only", "clock"]))
    only = rr.PN.profile_from_args(ap.parse_args(["--noise", "default", "--noise-only", "hp"]))
    assert only.active_groups() == ["hp"]


# ---------------------------------------------------------------------------
# rs_eval
# ---------------------------------------------------------------------------

@pytest.fixture
def tools(monkeypatch):
    """rs_rows and rs_eval as the tools import them, the settings at their defaults (restored
    after the test) and no engine config from the PC."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    ev = importlib.import_module("rs_eval")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "VISIBILITY": rr.VIS_LEGACY, "OUT": rr.OUT, "CARD_FLY": {},
                        "CARD_BUILDING": {}, "ROW_SCHEMA": rr.rs_schema.SCHEMA_LEGACY,
                        "BUTTONS": 2, "HAND_SIZE": 4, "SF": None, "UNIT_FORMS": None,
                        "SPELL_CARDS": None, "NOISE": None, "NOISE_SEED": 0,
                        "_NOISE_MODELS": {}}.items():
        monkeypatch.setattr(rr, name, value)
    return types.SimpleNamespace(rows=rr, eval=ev, engine=eng)


def stub_class(play_at=(), watch=()):
    """An rs_eval agent class: keeps a copy of the memory view each act() is handed at `watch`
    in `records`, and plays hand slot 0 at the ticks in `play_at`."""
    class Stub:
        made: list = []

        def __init__(self, ckpt, seed=0, **_):
            self.ckpt, self.seed, self.records = ckpt, seed, {}
            Stub.made.append(self)

        def act(self, battle, state, team, mem, tick, sub, royalesim):
            if tick in watch:
                self.records[tick] = copy.deepcopy(mem)
            if tick in play_at:
                return (team, 0, 9000 * sub, (10_000 if team == 0 else 22_000) * sub)
            return None
    return Stub


def eval_args(tools, *, mode="agent", opp="", shadow=False, extra=()):
    payload = json.dumps(il_payload(tag="noise-eval-1"))
    return (payload, "", mode, 3, opp, 0, 0, False, shadow, False,
            tools.engine.EngineConfig.defaults().to_json(), 1, None, None) + tuple(extra)


def test_eval_without_noise_is_the_play_of_before(fake_royalesim, tools, monkeypatch):
    ev = tools.eval
    assert ev.RSAgent.noise is None                     # the real class, before the stubs
    runs = []
    for extra in ((), (None, None, None), ("none", "off", 0)):
        agent, opponent = stub_class((200, 300), (210, 260)), stub_class((200, 250), (210, 260))
        monkeypatch.setattr(ev, "AGENT_CLASS", agent)
        monkeypatch.setattr(ev, "RSAgent", opponent)
        if extra:
            monkeypatch.setattr(ev.PN, "MatchNoise", None)     # any use would fail
        out = ev.play(eval_args(tools, opp="opp.pt", extra=extra))
        b = fake_royalesim.battles[-1]
        assert "noise" not in agent.made[0].__dict__ and "noise" not in opponent.made[0].__dict__
        runs.append((out, b.calls, b.state_reads))
    base = runs[0]
    assert "noise" not in base[0] and "opponent_noise" not in base[0]
    assert base[0]["records"][260]["last"][1] == [(10, 200), (10, 250)]
    for out, calls, reads in runs[1:]:
        assert out == base[0] and calls == base[1] and reads == base[2]


def test_eval_agent_sees_the_ghost_through_its_noise(fake_royalesim, tools, monkeypatch):
    """The ghost (red) plays its recorded Knight at 160 and Arrows at 300; the agent's screen
    sees them 4 ticks late."""
    ev, PN = tools.eval, tools.rows.PN
    late = {"opponent_plays": {"late_ticks": 4}}
    outs = {}
    for name, extra in (("plain", ()), ("late", (late, None, None))):
        agent = stub_class(watch=(160, 170, 300, 310))
        monkeypatch.setattr(ev, "AGENT_CLASS", agent)
        outs[name] = (ev.play(eval_args(tools, extra=extra)), agent.made[0],
                      fake_royalesim.battles[-1])
    (plain, _, pb), (out, seat, b) = outs["plain"], outs["late"]
    assert b.calls == pb.calls and b.state_reads == pb.state_reads
    assert out["noise"] == {"plays": 2} and "opponent_noise" not in out
    assert {k: v for k, v in out.items() if k not in ("noise", "records")} == \
        {k: v for k, v in plain.items() if k != "records"}
    rec, old = seat.records, plain["records"]
    assert rec[160]["last"][1] == [] == old[160]["last"][1]
    assert rec[170]["last"][1] == [(10, 164)] and old[170]["last"][1] == [(10, 160)]
    assert rec[300]["last"][1] == [(10, 164)]
    assert rec[310]["last"][1] == [(10, 164), (13, 304)]
    assert old[310]["last"][1] == [(10, 160), (13, 300)]
    nz = seat.noise
    assert isinstance(nz, PN.MatchNoise) and nz.stats == {"plays": 2}
    assert nz.model.profile.digest() == PN.NoiseProfile.from_dict(late).digest()
    assert nz.history.last_tick == 6199                       # every state of the match pushed
    fresh = np.random.default_rng(PN.match_seed("noise-eval-1", 0))
    assert nz.rng.random() == fresh.random()                  # no draw taken; seeded by the tag


def test_eval_opponent_noise_sees_the_agent(fake_royalesim, tools, monkeypatch):
    ev, PN = tools.eval, tools.rows.PN
    miss = {"opponent_plays": {"miss_p": 1.0}}
    agent, opponent = stub_class((200, 300), (260,)), stub_class((200, 250), (210, 310))
    monkeypatch.setattr(ev, "AGENT_CLASS", agent)
    monkeypatch.setattr(ev, "RSAgent", opponent)
    out = ev.play(eval_args(tools, opp="opp.pt", extra=(None, miss, 2)))
    assert out["opponent_noise"] == {"plays": 2, "plays_missed": 2} and "noise" not in out
    a, o = agent.made[0], opponent.made[0]
    assert "noise" not in a.__dict__ and isinstance(o.noise, PN.MatchNoise)
    assert o.records[210]["last"][0] == [] and o.records[310]["last"][0] == []
    assert a.records[260]["last"][1] == [(10, 200), (10, 250)]
    fresh = np.random.default_rng(PN.match_seed("noise-eval-1|opponent", 2))
    fresh.random(), fresh.random()                            # one draw per missed play
    assert o.noise.rng.random() == fresh.random()
    assert out["agent_plays"] == 2


def test_eval_refuses_noise_on_a_seat_without_the_hook(tools):
    ev = tools.eval
    clock = {"clock": {"bias_ticks": 1}}
    base = ("{}", "", "pro", 0, "", 0, 0, False, False, False, None, None, None, None)
    for args, what in ((base + (clock,), "--pro"),
                       (base[:2] + ("search",) + base[3:] + (clock,), "--search"),
                       (base[:2] + ("search_rebuild",) + base[3:] + (clock,), "--search"),
                       (base[:2] + ("agent",) + base[3:8] + (True,) + base[9:] + (clock,),
                        "shadow"),
                       (base[:2] + ("agent",) + base[3:] + (None, clock), "--opponent-ckpt")):
        with pytest.raises(ValueError, match=what):
            ev.play(args)
    ev.check_noise_seats(None, None, "pro")
    ev.check_noise_seats(None, None, "search", "", True)
    profile = tools.rows.noise_profile(clock)
    ev.check_noise_seats(profile, profile, "agent", "opp.pt", False)


def test_rsagent_act_decides_on_what_its_screen_shows(fake_royalesim, tools, monkeypatch):
    """RSAgent.act with a MatchNoise: the row and rs_decide see the noisy state (no Knight, 5500
    elixir) at the tick the clock reads (203); the match's state is not changed; decide gets the
    true tick. Without noise: the state as it is."""
    ev, rr = tools.eval, tools.rows
    rr._setup(tools.engine.EngineConfig.defaults())
    decided = []

    def decide(policy, row, state, team, legality, config, **kw):
        decided.append((row, state, kw["tick"]))
        return types.SimpleNamespace(command=None)

    monkeypatch.setattr(ev.rs_decide, "decide", decide)

    def agent(noise):
        a = object.__new__(ev.RSAgent)
        a.model, a.torch, a.live = None, types.SimpleNamespace(), False
        a.wait_bias, a.rng = 1.0, None
        if noise is not None:
            a.noise = rr.PN.MatchNoise(rr.noise_model(noise, fake_royalesim.Battle(), SUB), 0)
        return a

    battle = fake_royalesim.Battle()
    state = state_dict(200, entities=[KNIGHT])
    before = copy.deepcopy(state)
    view = rr.rs_memory.empty_view()
    noisy = agent(HOOKS)
    assert noisy.act(battle, state, 0, view, 200, SUB, None) is None
    row, seen, tick = decided[-1]
    assert state == before and seen is not state and tick == 200
    assert seen["players"][0]["elixir_milli"] == 5500 and seen["tick"] == 203
    assert row["tick"] == 203 and not row["unit_ids"].any()
    assert row["scalars"].tolist() == f32([0.55, 203 * 50 / 300_000, 0, 0])
    assert noisy.noise.tick_shift == 3
    assert noisy.noise.stats == {"dropped": 1, "elixir_fraction_lost": 1, "observations": 1,
                                 "units": 1}
    plain = agent(None)
    assert plain.noise is None
    plain.act(battle, state, 0, view, 200, SUB, None)
    row, seen, tick = decided[-1]
    assert seen is state and row["tick"] == 200 and row["unit_ids"].tolist()[0] == 10
    assert row["scalars"].tolist() == f32([0.5, 200 * 50 / 300_000, 0, 0])
    # holding for a pending play still observes (the history and counts stay current)
    noisy.hold_pending = True
    battle.pending[0] = [(0, 0, 0, 0)]
    n = len(decided)
    assert noisy.act(battle, state, 0, view, 210, SUB, None) is None and len(decided) == n
    assert noisy.noise.stats["observations"] == 2


def test_eval_main_wires_the_noise(tools):
    text = (TOOLS / "rs_eval.py").read_text(encoding="utf-8")
    main = text[text.index("def main()"):]
    for needle in ("PN.add_noise_args(ap)", '"--opponent-noise"',
                   "noise = PN.profile_from_args(args)",
                   "check_noise_seats(noise, opp_noise, mode_, args.opponent_ckpt, args.shadow)",
                   "args.decide, args.visibility) + extra",
                   'record[key] = PN.vocab_entry(prof, args.noise_seed)',
                   'f" noise {noise.name}"'):
        assert needle in main, needle
    assert main.count("extra = ()") == 1
