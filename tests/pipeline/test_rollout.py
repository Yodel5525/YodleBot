"""tools/rl/rollout.py (R01): vectorised rollouts in RoyaleSim.

What is checked, on conftest's fake royalesim (FakeBattle: a scripted state, every deploy legal,
every command taken) with tiny TokenPolicies (d16, one layer, CPU):
  - the batched forward (Brain.forward -> Heads) gives what rs_decide.TokenPolicyAdapter gives one
    row at a time (gate, critic, the tile logits of all four slots; v2 and legacy), and decide
    draws the same on either
  - plan_battles: the league's draws, then the deck field's, from one seeded generator; the
    learner on Hog 2.6 in base forms with the sampling preset and its noise, an idle seat for a
    member without a checkpoint, both seats for a mirror; friendly level 11, delay 24;
    reproducible by seed, JSON round trips; the refusals
  - a collection: K = 2 battles open at once, one learner forward per round, delay 24 and level 11
    on every battle, every play checked with check_deploy on its own battle; the learner's rows
    in tools/rl/ppo.py's format (ppo.read_rollout checks them): shapes, episodes and steps, done
    and the result on each episode's last row, the true towers as reward parts; the
    log-probabilities recomputed from the stored rows within 1e-5 (ppo's recompute, dropout off,
    and rs_decide's numpy draw); the same seed the same files, another seed other draws
  - a battle cut at max_ticks has no done row; a failing battle is logged and skipped, and
    max_errors stops the worker; a worker stops when the collection is aborted or its parent is
    gone; bench rows
  - the worker task is plain data and run_worker pickles by reference (Windows spawn)
  - the delay, noise and memory settings; the CLI end to end (collect and bench)
Engine-marked: one battle on RoyaleSim, in this process and in one spawn worker.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import math
import pickle
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    KING_HP,
    PRINCESS_HP,
    entity_row,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOG = ("Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons", "Log")
OPP = ("Archer", "Arrows", "BabyDragon", "Giant", "Knight", "MiniPekka", "Valkyrie", "Zap")
#: The fake catalogue's columns by name (conftest keeps the cost in column 1).
FAKE_FIELDS = ("name", "elixir", "c2", "c3", "c4", "flying", "c6", "c7", "card_kind")
NAMES = [str(r[0]) for r in DEFAULT_CATALOGUE]
#: The learner seats see the opponent's plays 4 ticks late (and nothing else is noisy).
NOISE = {"opponent_plays": {"late_ticks": 4}}
END = 200                                   # the scripted battle's last tick: blue wins
VIEW = {"seen": {0: set(), 1: {10}}, "last": {0: [], 1: [(10, 80)]}}


@pytest.fixture
def tools(monkeypatch):
    """rs_rows, rs_eval, rs_match, rs_checkpoint and tools/rl/rollout as the tools import them,
    rs_rows' settings at their defaults (restored after the test), no engine config or drift
    switch from the PC, rs_elixir's constant law and the fake catalogue cached for RSAgent."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    ev = importlib.import_module("rs_eval")
    rm = importlib.import_module("rs_match")
    rc = importlib.import_module("rs_checkpoint")
    ro = importlib.import_module("tools.rl.rollout")
    lg = importlib.import_module("tools.rl.league")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV, rc.ALLOW_DRIFT_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "VISIBILITY": rr.VIS_LEGACY, "TRUST": rr.TrustRule(), "OUT": rr.OUT,
                        "CARD_FLY": {}, "CARD_BUILDING": {}, "PRESSES": "off",
                        "BUTTONS": 2, "HAND_SIZE": 4, "DECISION": rr.DecisionRule(),
                        "ROW_SCHEMA": "rs_rows/legacy", "SF": None, "UNIT_FORMS": None,
                        "SPELL_CARDS": None}.items():
        monkeypatch.setattr(rr, name, value)
    real_law = rr.rs_elixir.default_law
    law = rr.rs_elixir.ElixirLaw.constants()
    monkeypatch.setattr(rr.rs_elixir, "default_law", lambda: law)
    monkeypatch.setattr(ev.RSAgent, "_CAT", [list(r) for r in DEFAULT_CATALOGUE])
    monkeypatch.setattr(rc, "_VOCABS", {})
    redrive = sys.modules.get("royalesim_redrive")
    if redrive is not None and hasattr(redrive, "_ENGINE"):
        monkeypatch.setattr(redrive, "_ENGINE", {})
    return types.SimpleNamespace(rows=rr, eval=ev, match=rm, rc=rc, rollout=ro, league=lg,
                                 ppo=ro.PPO, decide=ro.RD, engine=eng, law=law, real_law=real_law)


def cfg(tools):
    return tools.engine.EngineConfig.defaults()


def fake_engine(monkeypatch, **battle_kwargs):
    """conftest's fake royalesim with FAKE_FIELDS, as sys.modules["royalesim"]."""
    mod = make_fake_royalesim(**battle_kwargs)
    mod.CATALOGUE_FIELDS = list(FAKE_FIELDS)
    monkeypatch.setitem(sys.modules, "royalesim", mod)
    return mod


def red_fraction(tick):
    """Red's mean crown-tower fraction in `script` at `tick`."""
    return (1.0 + (PRINCESS_HP - 5 * max(0, tick - 90)) / PRINCESS_HP + 1.0) / 3


def script(battle):
    """Red's left princess tower loses 5 hp a tick from tick 90; the game ends at END, blue the
    winner with one crown. A Knight of red's and a Hog Rider of blue's stand on the board."""
    t = int(battle.tick)
    over = t >= END
    blue = player_dict(0, crowns=1 if over else 0)
    red = player_dict(1, tower_hp=(KING_HP, PRINCESS_HP - 5 * max(0, t - 90), PRINCESS_HP))
    units = [entity_row(uid=100, team=1, card_id=8, unit_type=8, x=9000 * 18, y=20000 * 18,
                        hp=600),
             entity_row(uid=101, team=0, card_id=2, unit_type=3, x=3500 * 18, y=12000 * 18,
                        hp=1400)]
    return state_dict(t, players=[blue, red], entities=units, game_over=over,
                      winner=0 if over else -1)


def meta_for(**extra) -> dict:
    """A v2 checkpoint meta for conftest's fake catalogue (as tests/pipeline/test_rs_match_v2.py):
    decision-time rows (obs_delay 24) and trained buttons."""
    meta = {"schema": "rs_rows/v2", "n_cards": len(NAMES) + 2,
            "n_units": len(DEFAULT_UNIT_TYPES) + 2, "live_compat": False,
            "cards": {c: i + 2 for i, c in enumerate(NAMES)},
            "units": {u: i + 2 for i, u in enumerate(DEFAULT_UNIT_TYPES)},
            "card_table": "cards-160402017-20261019.1", "d": 16, "layers": 1, "heads": 2,
            "obs_delay": {"ticks": 24, "jitter": 3}, "train_rows": {"press_rows": "trained"}}
    meta.update(extra)
    return meta


def save_ckpt(path, meta, seed=0):
    """A tiny model of `meta` saved as sim/train_tokens saves one; returns it in eval mode."""
    torch = pytest.importorskip("torch")
    from sim import model_tokens as mt
    torch.manual_seed(seed)
    model = mt.policy_from_meta(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta, "epoch": 0}, path)
    return model.eval()


class FixedLeague:
    """The league protocol plan_battles reads, with the draws fixed: `ids` in order."""

    def __init__(self, ids, paths):
        self.ids, self.paths = list(ids), dict(paths)

    def draw(self, rng, size=None):
        return list(self.ids[:size])

    def member(self, member_id):
        return types.SimpleNamespace(path=self.paths[member_id])


def field_of(R, *decks, split="train"):
    """A DeckField of (names, count, forms) decks; forms: ((form tuple, seats), ...)."""
    DS = R.deck_stats
    stats = [DS.DeckStat(cards=tuple(names), count=count, share=0.0, score=0.5,
                         names=tuple(names), forms=forms) for names, count, forms in decks]
    return DS.DeckField(stats, split=split)


EVO_KNIGHT = (0, 0, 0, 0, 1, 0, 0, 0)       # OPP's Knight (position 4) evolved


def opp_field(R):
    return field_of(R, (OPP, 3, (((0,) * 8, 2), (EVO_KNIGHT, 1))))


def collect(tools, tmp_path, *, seed=5, ids=("idle", "snap", "learner"), max_ticks=400,
            out="out", battles=2, field=None, start=0, **settings):
    """plan_battles on a FixedLeague (idle: no checkpoint; snap: the learner's checkpoint as a
    frozen opponent; learner: a mirror) and the OPP field, then an in-process collection."""
    R = tools.rollout
    ckpt = tmp_path / "v2_hog" / "best.pt"
    if not ckpt.exists():
        save_ckpt(ckpt, meta_for())
    league = FixedLeague(ids, {"idle": None, "snap": str(ckpt)})
    plans = R.plan_battles(len(ids), learner=str(ckpt), league=league,
                           deck_field=field or opp_field(R), seed=seed, start=start,
                           noise=NOISE, engine=cfg(tools), max_ticks=max_ticks)
    opts = {"battles": battles, "threads": 0, "engine": cfg(tools), "strict": True}
    opts.update(settings)
    summary = R.collect(plans, R.Settings(learner=str(ckpt), **opts), workers=0,
                        out=tmp_path / out)
    return ckpt, plans, summary


def recompute(tools, model, arrays, meta):
    """(row masks, ppo's recomputed joint log-probability, the stored one, rs_decide's numpy
    one) of a rollout's rows, from the stored observations, masks and candidates alone."""
    torch = pytest.importorskip("torch")
    P, RD = tools.ppo, tools.decide
    a = P.with_defaults(arrays)
    masks = P.row_masks(a)
    n = len(a["gate_a"])
    idx = np.arange(n)
    b = P.obs_batch(a, idx, "cpu")
    cards = P.card_ids(a["hand"], a["gate_a"])
    act = P.action_tensors(a, idx, "cpu", masks.tile, cards)
    model.eval()
    with torch.no_grad():
        gate, tile, _ = model(b, card_ids=act["card"])
    lq, lt = P.log_probs(gate, tile, act, P.Sampling.from_meta(meta))
    joint = P.joint_log_prob(lq, lt, act).double().numpy()
    stored = a["logq_gate"].astype(np.float64) + np.where(masks.tile, a["logq_tile"], 0.0)
    g, tl = gate.double().numpy(), tile.double().numpy()
    numpy_lp = np.zeros(n)
    for i in np.flatnonzero(masks.learn):
        gi = g[i].copy()
        gi[0] += meta["wait_bias"]
        q = RD.gate_probs(gi, a["mask"][i], int(a["dt"][i]), bool(a["capped"][i]))
        numpy_lp[i] = math.log(q[int(a["gate_a"][i])])
        if masks.tile[i]:
            c = a["tile_cands"][i]
            c = c[c >= 0].astype(np.int64)
            pt = RD.tile_probs(tl[i][c], meta["explore"], meta["tile_temp"])
            numpy_lp[i] += math.log(pt[int(a["tile_a"][i])])
    return masks, joint, stored, numpy_lp


# ---------------------------------------------------------------------------
# the batched forward
# ---------------------------------------------------------------------------

def test_one_batched_forward_gives_the_adapters_heads(tools, monkeypatch):
    torch = pytest.importorskip("torch")
    from sim import model_tokens as mt
    R, RR, RD = tools.rollout, tools.rows, tools.decide
    fake_engine(monkeypatch, states=script)
    RR._setup(cfg(tools))
    torch.manual_seed(1)
    v2 = mt.policy_from_meta(meta_for()).eval()
    legacy = mt.TokenPolicy(len(NAMES) + 2, len(DEFAULT_UNIT_TYPES) + 2, d=16, layers=1,
                            heads=2).eval()
    states = [script(types.SimpleNamespace(tick=t)) for t in (90, 130, 170)]
    rows_v2 = [RR.obs_row(st, team, 18, VIEW, st["tick"], schema="v2", opp_elixir=(5000, True),
                          pending=[(4, 12)]) for st in states for team in (0, 1)]
    rows_legacy = [RR.obs_row(st, team, 18, VIEW, st["tick"]) for st in states for team in (0, 1)]
    for model, meta, rows, critic in ((v2, meta_for(), rows_v2, True),
                                      (legacy, {}, rows_legacy, False)):
        brain = R.Brain(model, meta)
        assert brain.v2 is critic and brain.wait_bias == pytest.approx(math.log(1 / 0.35))
        got = brain.forward(rows)
        assert len(got) == 6 and (brain.calls, brain.rows) == (1, 6)
        adapter = RD.TokenPolicyAdapter(model, torch)
        for heads, row in zip(got, rows):
            b = adapter.batch(row)
            want_gate, want_value = adapter.heads(b)
            gate, value = heads.heads(heads.batch(row))
            assert gate.shape == ((7,) if critic else (5,))
            np.testing.assert_allclose(gate, want_gate, atol=1e-5)
            np.testing.assert_allclose(heads.gate(None), want_gate, atol=1e-5)
            if critic:
                assert len(value) == 2
                np.testing.assert_allclose(value, want_value, atol=1e-5)
            else:
                assert value is None and want_value is None
            for slot in range(4):
                assert heads.tiles(None, slot).shape == (576,)
                np.testing.assert_allclose(heads.tiles(None, slot), adapter.tiles(b, slot),
                                           atol=1e-5)
    # decide samples the same from the batched heads as from the model itself
    brain = R.Brain(v2, meta_for())
    dcfg = RD.DecideConfig.sampling(brain.wait_bias, presses=True)
    legal = lambda slot, card, x, y: True  # noqa: E731
    for i, heads in enumerate(brain.forward(rows_v2)):
        team = i % 2
        st = states[i // 2]
        d1 = RD.decide(heads, rows_v2[i], st, team, legal, dcfg, rng=np.random.default_rng(i))
        d2 = RD.decide(RD.TokenPolicyAdapter(v2, torch), rows_v2[i], st, team, legal, dcfg,
                       rng=np.random.default_rng(i))
        assert (d1.kind, d1.gate_a, d1.slot, d1.tile, d1.tile_cands, d1.tile_a, d1.mask) == \
            (d2.kind, d2.gate_a, d2.slot, d2.tile, d2.tile_cands, d2.tile_a, d2.mask)
        assert d1.logq_gate == pytest.approx(d2.logq_gate, abs=1e-6)
        assert d1.logq_tile == pytest.approx(d2.logq_tile, abs=1e-6)
        assert len(d1.value) == 2 and d1.value == pytest.approx(d2.value, abs=1e-5)
    assert R.Brain(v2, meta_for(), wait_keep=0.25).wait_bias == pytest.approx(math.log(4))
    with pytest.raises(ValueError, match="units vision"):
        R.Brain(v2, meta_for(units_mode="vision"))
    with pytest.raises(ValueError, match="model is v2"):
        R.Brain(v2, {})


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------

def test_plans_follow_the_league_and_the_deck_field(tools):
    R, L, M = tools.rollout, tools.league, tools.match
    league = L.League(L.LeagueConfig(mirror_share=0.25), anchors={"il:a": "a.pt", "il:none": None})
    field = field_of(R, (OPP, 3, (((0,) * 8, 2), (EVO_KNIGHT, 1))),
                     (HOG, 1, (((0,) * 8, 1),)))
    noise = R.noise_entry(NOISE)
    plans = R.plan_battles(40, learner="me.pt", league=league, deck_field=field, seed=9,
                           noise=NOISE, engine=cfg(tools))
    rng = np.random.default_rng([9, R.PLAN_STREAM])
    want_ids = league.draw(rng, size=40)
    want_decks = field.sample(rng, size=40)
    assert [p.opponent for p in plans] == want_ids and [p.index for p in plans] == list(range(40))
    assert {"learner", "il:a", "il:none"} <= set(want_ids)           # seeded: all three drawn
    me = M.AgentSpec("me.pt", name="learner", decide="sample", noise=noise, presses=True)
    seeds = set()
    for p, choice in zip(plans, want_decks):
        spec = p.match_spec()
        team = spec.seat
        seeds.add(spec.seed)
        assert spec.agents[team] == me
        assert sorted(spec.decks[team]) == sorted(HOG) and spec.forms[team] == (0,) * 8
        assert sorted(spec.decks[1 - team]) == sorted(choice.names)
        assert dict(zip(spec.decks[1 - team], spec.forms[1 - team])) == \
            dict(zip(choice.names, choice.forms))
        assert (spec.levels, spec.tower_levels, spec.delay, spec.max_ticks, spec.mem_lag) == \
            (11, 11, 24, 6200, 1)
        assert spec.base_engine() == cfg(tools) and spec.tag == spec.pairing == f"rl:9:{p.index}"
        assert 0 <= spec.seed < 2 ** 31
        if p.opponent == "learner":
            assert p.learner == (True, True) and p.mirror and spec.agents[1 - team] == me
        elif p.opponent == "il:none":
            assert spec.agents[1 - team] == M.AgentSpec("", name="il:none")
            assert spec.agents[1 - team].idle
        else:
            assert spec.agents[1 - team] == M.AgentSpec("a.pt", name="il:a", decide="sample",
                                                        noise=noise, presses=True, seed_offset=1)
        if p.opponent != "learner":
            assert p.learner == tuple(t == team for t in (0, 1))
        assert R.BattlePlan.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    assert len(seeds) == 40
    assert {p.match_spec().seat for p in plans} == {0, 1}
    again = R.plan_battles(40, learner="me.pt", league=league, deck_field=field, seed=9,
                           noise=NOISE, engine=cfg(tools))
    other = R.plan_battles(40, learner="me.pt", league=league, deck_field=field, seed=10,
                           noise=NOISE, engine=cfg(tools))
    assert again == plans and other != plans
    shifted = R.plan_battles(3, learner="me.pt", league=league, deck_field=field, seed=9,
                             start=100, opponent_noise="none", opponent_decide="live",
                             presses=False, delay=(22, 26), level=12, max_ticks=500)
    assert [p.index for p in shifted] == [100, 101, 102]
    for p in shifted:
        spec = p.match_spec()
        assert (spec.delay, spec.levels, spec.max_ticks, spec.engine) == ((22, 26), 12, 500, None)
        assert spec.agents[spec.seat].presses is False
        if p.opponent == "il:a":
            assert (spec.agents[1 - spec.seat].noise, spec.agents[1 - spec.seat].decide) == \
                (None, "live")


def test_plan_refusals(tools):
    R, M = tools.rollout, tools.match
    league = FixedLeague(["il:none"], {"il:none": None})
    field = field_of(R, (OPP, 1, (((0,) * 8, 1),)))
    with pytest.raises(ValueError, match="train split"):
        R.plan_battles(1, learner="me.pt", league=league, deck_field=field_of(
            R, (OPP, 1, (((0,) * 8, 1),)), split="val"))
    with pytest.raises(ValueError, match="8 different"):
        R.plan_battles(1, learner="me.pt", league=league, deck_field=field, learner_deck=HOG[:7])
    with pytest.raises(ValueError, match="opponent_decide"):
        R.plan_battles(1, learner="me.pt", league=league, deck_field=field,
                       opponent_decide="argmax")
    (plan,) = R.plan_battles(1, learner="me.pt", league=league, deck_field=field)
    spec, team = plan.match_spec(), plan.match_spec().seat
    with pytest.raises(ValueError, match="legacy"):
        R.BattlePlan(0, spec.replace(mem_lag="legacy"), plan.learner)
    with pytest.raises(ValueError, match="no learner"):
        R.BattlePlan(0, spec, (False, False))
    with pytest.raises(ValueError, match="idle"):
        R.BattlePlan(0, spec, (True, True))
    agents = list(spec.agents)
    agents[team] = dataclasses.replace(agents[team], decide="live")
    with pytest.raises(ValueError, match="samples"):
        R.BattlePlan(0, spec.replace(agents=tuple(agents)), plan.learner)
    with pytest.raises(ValueError, match="no RoyaleSim names"):
        nameless = types.SimpleNamespace(split="train", sample=lambda rng, size: [
            types.SimpleNamespace(names=None, forms=(0,) * 8)] * size)
        R.plan_battles(1, learner="me.pt", league=league, deck_field=nameless)
    assert R.plan_battles(0, learner="me.pt", league=league, deck_field=field) == []
    assert M.MatchSpec.from_dict(plan.spec) == spec


# ---------------------------------------------------------------------------
# a collection
# ---------------------------------------------------------------------------

def test_a_collection_writes_the_learners_rows(tools, monkeypatch, tmp_path):
    pytest.importorskip("torch")
    P = tools.ppo
    mod = fake_engine(monkeypatch, states=script)
    ckpt, plans, s = collect(tools, tmp_path)
    teams = [p.match_spec().seat for p in plans]
    assert [p.learner for p in plans] == [tuple(t == x for x in (0, 1)) for t in teams[:2]] + \
        [(True, True)]
    assert s["errors"] == [] and (s["matches"], s["rows"], s["decisions"]) == (3, 44, 55)
    assert s["files"] == [str(tmp_path / "out" / "rollout-w00-000.npz")]
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))["rows"] == 44
    (w,) = s["workers"]
    # 11 rounds with battles 0 and 1 open, then 11 with battle 2: one learner forward a round.
    assert (w["rounds"], w["learner_forwards"], w["learner_rows"]) == (22, 22, 44)
    assert (w["forwards"], w["forward_rows"]) == (33, 55)
    res = s["results"]
    assert [r["opponent"] for r in res] == ["idle", "snap", "learner"]
    for r, t in zip(res[:2], teams[:2]):
        assert r["learner_teams"] == [t] and r["score"] == (1.0 if t == 0 else 0.0)
    assert res[0]["decisions"] == [11 if t == teams[0] else 0 for t in (0, 1)]
    assert res[1]["decisions"] == res[2]["decisions"] == [11, 11]
    assert res[2]["learner_teams"] == [0, 1] and res[2]["score"] is None
    assert all((r["winner"], r["crowns"], r["game_over"], r["end_tick"]) == (0, [1, 0], True, END)
               for r in res)
    assert s["league_results"] == [["idle", res[0]["score"]], ["snap", res[1]["score"]]]
    assert sum(sum(r["plays"]) for r in res) > 0 and all(r["refused"] == [0, 0] for r in res)

    battles = [b for b in mod.battles if b.reset_kwargs is not None]
    assert len(battles) == 3
    for b, p, team in zip(battles, plans, teams):
        assert b.names()[:2] == ["set_command_delay_ticks", "reset"] and b.delay == (24, 24)
        kw = b.reset_kwargs
        assert kw["levels"] == [[11] * 8, [11] * 8] and kw["tower_levels"] == [11, 11]
        assert sorted(kw["decks"][team]) == list(range(8)) and kw["forms"][team] == [0] * 8
        assert sorted(kw["decks"][1 - team]) == list(range(8, 16))
        checked = set(b.deploy_checks)
        for call in b.calls_named("step"):
            for cmd in call.args[0]:
                assert cmd in checked, cmd                 # legality on this battle itself

    arrays, meta = P.read_rollout(s["files"][0])
    n = 44
    for name, shape in {"unit_ids": (n, 48), "unit_feats": (n, 48, 9), "scalars": (n, 7),
                        "hand_feats": (n, 4, 3), "spell_feats": (n, 8, 6), "own_pending": (n, 4),
                        "ability_card": (n, 2), "ability_state": (n, 2, 5), "mask": (n, 7),
                        "tile_cands": (n, 16), "hp_frac": (n, 2), "crowns": (n, 2),
                        "value": (n, 2), "gate_a": (n,)}.items():
        assert arrays[name].shape == shape, name
    assert arrays["unit_feats"].dtype == np.float16 and arrays["hand"].dtype == np.int16
    assert (meta["format"], meta["schema"], meta["presses"], meta["dt"], meta["mode"]) == \
        (P.FORMAT, "rs_rows/v2", True, 10, "sample")
    assert meta["wait_bias"] == pytest.approx(math.log(1 / 0.35))
    assert (meta["learner"], meta["worker"], meta["part"], meta["tool"]) == \
        (str(ckpt), 0, 0, "tools/rl/rollout.py")
    episodes = {2 * 0 + teams[0], 2 * 1 + teams[1], 4, 5}
    assert {e["episode"] for e in meta["episodes"]} == episodes
    ep, step = arrays["episode"].astype(int), arrays["step"].astype(int)
    for e in episodes:
        assert sorted(step[ep == e].tolist()) == list(range(11)), e
    done = arrays["done"]
    assert done.sum() == 4 and (step[done] == 10).all()
    team = ep % 2
    assert arrays["result"][done].tolist() == [1.0 if t == 0 else -1.0 for t in team[done]]
    assert not arrays["result"][~done].any() and not arrays["crowns"].any()
    tick = 90 + 10 * step
    red = np.array([red_fraction(t) for t in tick])
    np.testing.assert_allclose(arrays["hp_frac"][:, 0], np.where(team == 0, 1.0, red), rtol=1e-6)
    np.testing.assert_allclose(arrays["hp_frac"][:, 1], np.where(team == 0, red, 1.0), rtol=1e-6)
    assert np.isfinite(arrays["value"]).all() and (arrays["dt"] == 10).all()
    assert not arrays["fallback"].any() and not arrays["capped"].any()
    assert arrays["mask"][:, 5:].sum() == 0                     # no ability row: no button

    model = tools.rc.load_policy(str(ckpt), cfg(tools)).model
    masks, joint, stored, numpy_lp = recompute(tools, model, arrays, meta)
    assert masks.learn.all() and masks.tile.any() and (arrays["gate_a"] == 0).any()
    assert np.abs(joint - stored).max() < 1e-5
    assert np.abs(numpy_lp - stored).max() < 1e-5


def test_the_same_seed_gives_the_same_rollout(tools, monkeypatch, tmp_path):
    pytest.importorskip("torch")
    fake_engine(monkeypatch, states=script)
    _, plans_a, a = collect(tools, tmp_path, out="a")
    _, plans_b, b = collect(tools, tmp_path, out="b")
    _, plans_c, c = collect(tools, tmp_path, out="c", seed=6)
    assert plans_a == plans_b != plans_c
    assert a["results"] == b["results"]
    with np.load(a["files"][0]) as za, np.load(b["files"][0]) as zb, \
            np.load(c["files"][0]) as zc:
        assert sorted(za.files) == sorted(zb.files) == sorted(zc.files)
        for k in za.files:
            assert np.array_equal(za[k], zb[k]), k
        assert za["gate_a"].shape == zc["gate_a"].shape
        assert not all(np.array_equal(za[k], zc[k]) for k in ("gate_a", "tile_a", "logq_gate"))


def test_a_battle_cut_at_max_ticks_has_no_done_row(tools, monkeypatch, tmp_path):
    pytest.importorskip("torch")
    P = tools.ppo
    fake_engine(monkeypatch, states=script)
    _, plans, s = collect(tools, tmp_path, ids=("idle",), max_ticks=150)
    (r,) = s["results"]
    assert (r["game_over"], r["end_tick"], r["winner"], r["score"]) == (False, 150, -1, 0.5)
    arrays, _ = P.read_rollout(s["files"][0])
    assert arrays["step"].tolist() == list(range(6))           # ticks 90 .. 140
    assert not arrays["done"].any() and not arrays["result"].any()
    assert s["league_results"] == [["idle", 0.5]]


def test_a_failing_battle_is_logged_and_skipped(tools, monkeypatch, tmp_path):
    pytest.importorskip("torch")
    R = tools.rollout
    fake_engine(monkeypatch, states=script)
    ckpt = tmp_path / "v2_hog" / "best.pt"
    save_ckpt(ckpt, meta_for())
    league = FixedLeague(["idle"], {"idle": None})
    good = R.plan_battles(1, learner=str(ckpt), league=league, deck_field=opp_field(R),
                          engine=cfg(tools), max_ticks=400)
    bad = R.plan_battles(1, learner=str(ckpt), league=league, start=1, engine=cfg(tools),
                         deck_field=field_of(R, (OPP[:7] + ("Mirror",), 1, (((0,) * 8, 1),))))
    settings = R.Settings(learner=str(ckpt), battles=2, threads=0, engine=cfg(tools))
    s = R.collect(good + bad, settings, workers=0, out=tmp_path / "out")
    assert (s["matches"], s["rows"]) == (1, 11)
    (err,) = s["errors"]
    assert err["index"] == 1 and err["opponent"] == "idle" and "Mirror" in err["error"]
    with pytest.raises(RuntimeError, match="1 battles failed"):
        R.collect(good + bad, settings.replace(max_errors=0), workers=0)
    with pytest.raises(ValueError, match="Mirror"):
        R.collect(bad, settings.replace(strict=True), workers=0)
    with pytest.raises(ValueError, match="no battles"):
        R.collect([], settings)


def test_a_worker_stops_when_aborted_or_orphaned(tools, monkeypatch, tmp_path):
    """A spawn worker checks every round: the collection's abort event (collect sets it when a
    worker fails) and its parent's liveness (a killed parent's children must not play on)."""
    pytest.importorskip("torch")
    import threading
    R = tools.rollout
    fake_engine(monkeypatch, states=script)
    assert R._ABORT is None and R._stop_reason() is None          # the parent, inline
    stop = threading.Event()
    monkeypatch.setattr(R, "_ABORT", stop)
    assert R._stop_reason() is None
    stop.set()
    assert "aborted" in R._stop_reason()
    with pytest.raises(R.Aborted, match="1 battle"):
        collect(tools, tmp_path, ids=("idle",))
    assert not list((tmp_path / "out").glob("*.npz"))
    assert not (tmp_path / "out" / "summary.json").exists()
    monkeypatch.setattr(R, "_ABORT", None)
    gone = types.SimpleNamespace(is_alive=lambda: False)
    monkeypatch.setattr(R.multiprocessing, "parent_process", lambda: gone)
    assert "parent process is gone" in R._stop_reason()
    alive = types.SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(R.multiprocessing, "parent_process", lambda: alive)
    assert R._stop_reason() is None
    R._init_pool(None, 0, stop)                    # the initializer hands the event over
    assert R._ABORT is stop


def test_bench_rows_without_files(tools, monkeypatch, tmp_path):
    pytest.importorskip("torch")
    R = tools.rollout
    fake_engine(monkeypatch, states=script)
    _, _, s = collect(tools, tmp_path, write=False)
    assert s["files"] == [] and s["rows"] == 44 and not list((tmp_path / "out").glob("*.npz"))
    row = R.bench_row(s, 2)
    assert (row["workers"], row["battles"], row["matches"], row["errors"]) == (0, 2, 3, 0)
    assert row["rows_per_forward"] == 2.0 and row["matches_per_hour"] > 0
    assert row["rss_mb_max"] is None or row["rss_mb_max"] > 0
    (t,) = row["timings"]
    assert {"engine_s", "rows_s", "decide_s", "record_s", "forward_s"} <= set(t)
    rss, peak = R.rss_mb()
    assert rss is None or rss > 0
    assert peak is None or peak > 0


# ---------------------------------------------------------------------------
# Windows spawn: plain data, functions by reference
# ---------------------------------------------------------------------------

def test_the_worker_task_is_plain_data_for_spawn(tools):
    R = tools.rollout
    assert pickle.loads(pickle.dumps(R.run_worker)) is R.run_worker
    assert pickle.loads(pickle.dumps(R._init_pool)) is R._init_pool
    module = importlib.import_module(R.run_worker.__module__)
    assert module.run_worker is R.run_worker and module.__name__ == "tools.rl.rollout"
    league = FixedLeague(["il:none", "learner"], {"il:none": None})
    plans = R.plan_battles(2, learner="me.pt", league=league, noise="default",
                           deck_field=opp_field(R), engine=cfg(tools))
    settings = R.Settings(learner="me.pt", engine=cfg(tools), meta={"seed": 1, "decks": (1, 2)})
    assert settings.meta == {"seed": 1, "decks": [1, 2]}
    assert settings.engine == cfg(tools).to_json()
    task = {"worker": 0, "settings": settings.to_dict(), "out": None, "prefix": "x",
            "battles": [p.to_dict() for p in plans]}
    assert json.loads(json.dumps(task)) == task
    assert pickle.loads(pickle.dumps(task)) == task
    assert R.Settings.from_dict(json.loads(json.dumps(settings.to_dict()))) == settings
    assert [R.BattlePlan.from_dict(p) for p in json.loads(json.dumps(task["battles"]))] == plans
    for bad in ({"learner": ""}, {"learner": "x", "battles": 0}, {"learner": "x", "wait_keep": 0},
                {"learner": "x", "threads": -1}):
        with pytest.raises(ValueError):
            R.Settings(**bad)
    with pytest.raises(ValueError, match="unknown"):
        R.Settings.from_dict({"learner": "x", "workers": 3})


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def test_delay_noise_and_memory_settings(tools, monkeypatch, tmp_path):
    R = tools.rollout
    c = cfg(tools)
    assert R.resolve_delay(None, c) == 24                       # engine.json: [0, 0]
    assert R.resolve_delay(None, tools.engine.EngineConfig.legacy()) == 24
    assert R.resolve_delay(None, c.replace(delay=(22, 22))) == 22
    assert R.resolve_delay(None, c.replace(delay=(21, 23))) == (21, 23)
    assert R.resolve_delay("21", c.replace(delay=(22, 22))) == 21
    assert R.resolve_delay("21,22", c) == (21, 22) and R.resolve_delay(0, c) == 0
    assert R.resolve_delay("default", c.replace(delay=(22, 22))) == 22
    measured, prior = tmp_path / "measured.json", tmp_path / "prior.json"
    assert R.default_noise(measured, prior) == prior
    measured.write_text("{}", encoding="utf-8")
    assert R.default_noise(measured, prior) == measured
    assert R.default_noise() in (R.MEASURED_NOISE, R.PN.PRIOR_PATH)
    for off in (None, "none", " off ", {}):
        assert R.noise_entry(off) is None, off
    assert R.noise_entry("default") == R.PN.NoiseProfile.load("default").to_dict()
    assert R.noise_entry(NOISE)["opponent_plays"]["late_ticks"] == 4
    assert R.worker_mb(16, 20_000) == 845                       # 600 + 128 + 117.2
    assert R.worker_mb(1, 1024) == 614                          # 600 + 8 + 6
    assert R.ram_check(8, per_worker_mb=900, budget_gb=8, available=16_000)[0] == 8
    got, why = R.ram_check(12, per_worker_mb=900, budget_gb=8, available=16_000)
    assert got == 9 and "budget 9" in why and "not 12" in why       # 8192 // 900
    got, why = R.ram_check(4, per_worker_mb=1000, budget_gb=8, available=4000)
    assert got == 1 and "free memory 1" in why                   # (4000 - 2048) // 1000
    assert R.ram_check(0, per_worker_mb=1000, available=100)[0] == 1
    monkeypatch.setattr(R, "_available_mb", lambda: None)
    got, why = R.ram_check(20, per_worker_mb=1000)
    assert got == 8 and "free unknown" in why


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def test_the_command_line_collects_and_benches(tools, monkeypatch, tmp_path, capsys):
    pytest.importorskip("torch")
    R = tools.rollout
    fake_engine(monkeypatch, states=script)
    ckpt = tmp_path / "v2_hog" / "best.pt"
    save_ckpt(ckpt, meta_for())
    common = ["--learner", str(ckpt), "--matches", "2", "--battles", "2", "--threads", "0",
              "--anchor", "floor=idle", "--opponent-deck", ",".join(OPP), "--noise", "none",
              "--max-ticks", "400", "--seed", "3"]
    out = tmp_path / "it000"
    assert R.main(["collect", "--out", str(out), "--workers", "0", *common]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("engine: ") and "delay 24 ticks, level 11, noise none" in lines[1]
    last = json.loads(lines[-1])
    assert (last["matches"], last["rows"], last["errors"]) == (2, 22, [])
    assert [o for o, _ in last["league_results"]] == ["floor", "floor"]
    assert last["meta"]["delay"] == 24 and last["meta"]["noise"] is None
    assert last["files"] == [str(out / "rollout-w00-000.npz")]
    _, meta = tools.ppo.read_rollout(last["files"][0])
    assert meta["run"]["seed"] == 3 and meta["run"]["presses_switch"] is True
    assert R.main(["bench", "--workers", "0", *common]) == 0
    bench = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    (row,) = bench["bench"]
    assert (row["workers"], row["matches"], row["errors"]) == (0, 2, 0)
    legacy = tmp_path / "legacy" / "best.pt"
    save_ckpt(legacy, meta_for(schema="rs_rows/legacy"))
    assert R.main(["collect", "--out", str(out), "--workers", "0", *common[2:],
                   "--learner", str(legacy)]) == 2
    assert "v2 checkpoint" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

@pytest.mark.engine
@pytest.mark.parametrize("workers", [0, 1])
def test_engine_rollouts_on_royalesim(tools, monkeypatch, tmp_path, workers):
    """One battle against an idle seat on RoyaleSim at delay 24 under the shipped noise prior,
    cut at tick 300: 21 learner rows ppo accepts, whose log-probabilities recompute; workers 1
    plays it in a spawn process (the Windows path)."""
    pytest.importorskip("torch")
    R = tools.rollout
    monkeypatch.setattr(tools.eval.RSAgent, "_CAT", None)        # the engine's own catalogue
    monkeypatch.setattr(tools.rows.rs_elixir, "default_law", tools.real_law)
    c = cfg(tools)
    vocab = tools.rc.engine_vocab(c)
    meta = meta_for(n_cards=len(vocab.cards) + 2, n_units=len(vocab.units) + 2,
                    cards={name: i + 2 for i, name in enumerate(vocab.cards)},
                    units={name: i + 2 for i, name in enumerate(vocab.units)},
                    card_table=vocab.card_table)
    ckpt = tmp_path / "v2" / "best.pt"
    model = save_ckpt(ckpt, meta)
    plans = R.plan_battles(1, learner=str(ckpt), league=FixedLeague(["none"], {"none": None}),
                           deck_field=field_of(R, (OPP, 1, (((0,) * 8, 1),))), seed=1,
                           noise="default", engine=c, max_ticks=300)
    settings = R.Settings(learner=str(ckpt), battles=1, threads=1 if workers else 0, engine=c,
                          allow_drift=True, strict=True)
    s = R.collect(plans, settings, workers=workers, out=tmp_path / "out")
    assert s["errors"] == [] and (s["matches"], s["rows"]) == (1, 21)
    (r,) = s["results"]
    assert (r["game_over"], r["end_tick"]) == (False, 300)
    arrays, meta_ = tools.ppo.read_rollout(s["files"][0])
    masks, joint, stored, numpy_lp = recompute(tools, model, arrays, meta_)
    assert masks.learn.any()
    assert np.abs(joint - stored)[masks.learn].max() < 1e-5
    assert np.abs(numpy_lp - stored)[masks.learn].max() < 1e-5
