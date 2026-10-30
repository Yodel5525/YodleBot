"""R02: tools/rl/ppo.py, PPO with GAE, potential shaping and adaptive KL.

Pure numpy: the potential and the reward parts of a state (conftest's player_dict / state_dict),
GAE against a hand-computed sequence (also through returns_of, rows shuffled, one episode cut
short), the shaped returns telescoping to the terminal reward, the KL controller (up, down, the
dead band's edges, the clamp, no measurement) and the state it carries, the row masks (no draw,
fallback, presses), decision_record from rs_decide Decisions, rollout_meta, check_rollout's
complaints, and write / load / data_files / the RAM estimate.

With torch (importorskip): the gate and tile log-probabilities equal rs_decide.gate_probs /
tile_probs (the rollout's sampling) entry for entry, the entropy and KL by hand; a rollout
sampled by rs_decide.decide (sampling preset, presses on, a legality that refuses some cards and
tiles) through a tiny TokenPolicy v2: the recomputed distributions are the decisions' own, and an
update at lr 0 leaves every ratio at 1, the weights unchanged and the KL to an identical
reference at 0; a positive advantage raises its action's probability, the critic trains value2
and never value_head; the CLI (--check, a full update, a second one continuing the state, the
refusals). GPU-marked: the same at bf16 on CUDA.

No engine, no data, no emulator.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import sys
from pathlib import Path

import pytest

from tests.pipeline.conftest import player_dict, state_dict

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
for _path in (str(ROOT), str(ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tools.rl import ppo as P  # noqa: E402
import rs_decide as RD  # noqa: E402
import rs_schema as RS  # noqa: E402

N_CARDS, N_UNITS, BUTTONS = 18, 20, 2
AVAILABLE = RS.ABILITY_FEATS.index("available")
KING, PRINCESS = 4824, 3052
#: Catalogue ids of conftest's DEFAULT_CATALOGUE (tokens are id + 2): Cannon 0 (3 elixir),
#: HogRider 2 (4), Musketeer 5 (4), Skeletons 6 (1).
CANNON, SKELETONS = 0, 6
META = {"format": P.FORMAT, "schema": RS.SCHEMA_V2, "wait_bias": 0.4, "explore": 0.15,
        "tile_temp": 1.0}


def _torch():
    return pytest.importorskip("torch")


def tiny_model(seed=0):
    torch = _torch()
    from sim import model_tokens as mt
    torch.manual_seed(seed)
    return mt.TokenPolicy(N_CARDS, N_UNITS, d=16, layers=1, heads=2, schema="v2").eval()


def synthetic(episodes=2, steps=3, buttons=BUTTONS, seed=0):
    """Rollout columns by hand: every row draws a wait (logq -0.5), the last row of each episode
    is done (even episodes won, odd lost), towers full, no crowns."""
    rng = np.random.default_rng(seed)
    n = episodes * steps
    a = {}
    for name in P.OBS_FIELDS:
        a[name] = np.zeros((n, *RS.V2.shape(name, buttons)), dtype=RS.V2.dtype(name))
    a["hand"][:] = [4, 7, 8, 2]
    a["next"][:] = 5
    a["unit_ids"][:, :3] = rng.integers(1, N_UNITS, (n, 3))
    a["unit_feats"][:, :3] = rng.random((n, 3, 9))
    a["towers"][:] = rng.random((n, 6))
    a["scalars"][:] = rng.random((n, 7))
    a["opp_seen"][:, :2] = rng.integers(1, N_CARDS, (n, 2))
    a["ability_card"][:, 0] = 7
    a["ability_state"][:, 0, AVAILABLE] = 1.0
    a["episode"] = np.repeat(np.arange(episodes), steps).astype(np.int32)
    a["step"] = np.tile(np.arange(steps), episodes).astype(np.int32)
    a["done"] = a["step"] == steps - 1
    a["result"] = np.where(a["done"], np.where(a["episode"] % 2 == 0, 1.0, -1.0),
                           0.0).astype(np.float32)
    a["hp_frac"] = np.ones((n, 2), dtype=np.float32)
    a["crowns"] = np.zeros((n, 2), dtype=np.int8)
    a["mask"] = np.ones((n, P.GATE_PLAYS + buttons), dtype=bool)
    a["gate_a"] = np.zeros(n, dtype=np.int16)
    a["logq_gate"] = np.full(n, -0.5, dtype=np.float32)
    a["tile_cands"] = np.full((n, P.CANDS), -1, dtype=np.int16)
    a["tile_a"] = np.full(n, -1, dtype=np.int16)
    a["logq_tile"] = np.zeros(n, dtype=np.float32)
    return a


def records_of(a):
    n = len(a["gate_a"])
    return [{k: v[i] for k, v in a.items()} for i in range(n)]


def v2_row(rng, buttons=BUTTONS):
    """An obs_row of schema v2 at the spec's shapes and dtypes, varied by `rng`; button 0 holds
    card token 7 and reads available."""
    row = {"tick": 500}
    for name in RS.V2.model_inputs:
        shape, dtype = RS.V2.shape(name, buttons), RS.V2.dtype(name)
        row[name] = np.zeros(shape, dtype=dtype) if shape else np.dtype(dtype).type(0)
    row["hand"][:] = rng.permutation([4, 7, 8, 2])
    row["next"] = np.int16(5)
    k = int(rng.integers(1, 6))
    row["unit_ids"][:k] = rng.integers(1, N_UNITS, k)
    row["unit_feats"][:k] = rng.random((k, 9))
    row["towers"][:] = rng.random(6)
    row["scalars"][:] = rng.random(7)
    row["opp_seen"][:3] = rng.integers(1, N_CARDS, 3)
    row["opp_last"][:2] = rng.integers(1, N_CARDS, 2)
    row["opp_last_age"][:2] = rng.random(2)
    row["spell_ids"][0] = 3
    row["spell_feats"][0] = rng.random(6)
    row["own_pending"][0] = 7
    row["own_pending_land"][0] = 0.25
    row["ability_card"][0] = 7
    row["ability_state"][0, AVAILABLE] = 1.0
    return row


def legal(slot, card, x_sub, y_sub):
    """Skeletons never have a legal tile (a drawn Skeletons falls back); the Cannon only on even
    world columns; everything else anywhere."""
    if card == SKELETONS:
        return False
    return card != CANNON or (x_sub // 18000) % 2 == 0


def decide_rollout(model, folder, *, seed=0, episodes=3, steps=6):
    """Rows sampled by rs_decide.decide (sampling preset, presses on) with `model` through
    TokenPolicyAdapter, the red seat, as R01 records them; episodes 0 (won) and 1 (lost) end,
    the last one is cut short. Returns (the loaded Rollout, the Decisions, the records)."""
    adapter = RD.TokenPolicyAdapter(model)
    cfg = RD.DecideConfig.sampling(0.4, presses=True)
    rng = np.random.default_rng(seed)
    obs_rng = np.random.default_rng(seed + 100)
    records, decisions = [], []
    for e in range(episodes):
        for s in range(steps):
            row = v2_row(obs_rng)
            hand = [int(t) - 2 for t in row["hand"]]
            elixir = int(obs_rng.choice([0, 1500, 3500, 9000]))
            abilities = [[1, 0, 2, 5, 0]] if s % 2 == 0 else []
            me = player_dict(1, hand=hand, elixir_milli=elixir, abilities=abilities)
            state = state_dict(500, players=[player_dict(0), me])
            d = RD.decide(adapter, row, state, 1, legal, cfg, rng=rng)
            done = e < episodes - 1 and s == steps - 1
            records.append({**{k: row[k] for k in P.OBS_FIELDS}, **P.decision_record(d),
                            "episode": e, "step": s, "done": done,
                            "result": (1.0 if e == 0 else -1.0) if done else 0.0,
                            "hp_frac": [1.0 - 0.05 * s, 1.0 - 0.1 * s * (e == 0)],
                            "crowns": [0, 0]})
            decisions.append(d)
    path = P.write_rollout(Path(folder) / "rollout.npz", records, P.rollout_meta(cfg))
    return P.load_rollout([path]), decisions, records


# ---------------------------------------------------------------------------
# the reward
# ---------------------------------------------------------------------------

def test_potential_reward_parts_and_result():
    assert P.potential((1.0, 1.0), (0, 0)) == 0.0
    assert P.potential((1.0, 0.5), (0, 0)) == pytest.approx(0.05)
    assert P.potential((2 / 3, 1.0), (0, 1)) == pytest.approx(-0.1)
    assert P.potential((1.0, 0.0), (3, 0)) == pytest.approx(0.3)      # the whole board
    both = P.potential([[1.0, 0.5], [0.5, 1.0]], [[1, 0], [0, 1]])
    assert both.shape == (2,)
    assert both.tolist() == pytest.approx([0.05 + 0.2 / 3, -0.05 - 0.2 / 3])
    # blue lost a princess tower and half of the other; red took the crown
    blue = player_dict(0, tower_hp=(KING, 0, PRINCESS // 2), crowns=0)
    red = player_dict(1, tower_hp=(KING, PRINCESS, PRINCESS), crowns=1)
    state = state_dict(1000, players=[blue, red], winner=1, game_over=True)
    hp, cr = P.reward_parts(state, 0)
    assert hp.dtype == np.float32 and cr.dtype == np.int8
    assert hp.tolist() == [0.5, 1.0] and cr.tolist() == [0, 1]
    hp, cr = P.reward_parts(state, 1)
    assert hp.tolist() == [1.0, 0.5] and cr.tolist() == [1, 0]
    assert P.result_of(state, 0) == -1.0 and P.result_of(state, 1) == 1.0
    assert P.result_of(state_dict(1000, winner=-1), 0) == 0.0
    # an engine's negative hp reads as a destroyed tower
    minus = player_dict(0, tower_hp=(-5, PRINCESS, PRINCESS))
    assert P.tower_fraction(minus) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        P.tower_fraction({"tower_hp": [1, 2], "tower_max_hp": [1, 2]})


def test_gae_matches_a_hand_computed_sequence():
    """gamma 0.9, lambda 0.8. Episode A: r (1, 0, 2), V (0.5, 1, 1.5), done: deltas 1.4, 0.35,
    0.5; A = 1.9112, 0.71, 0.5. Episode B, cut short: r 0.25, V (0.2, 0.4): delta 0.41, the last
    row 0."""
    adv, ret = P.gae([1, 0, 2, 0.25, 9.0], [0.5, 1.0, 1.5, 0.2, 0.4], [1.0, 1.5, 0.0, 0.4, 0.0],
                     [False, False, True, False, True], gamma=0.9, lam=0.8,
                     valid=[True, True, True, True, False])
    assert adv.tolist() == pytest.approx([1.9112, 0.71, 0.5, 0.41, 0.0])
    assert ret.tolist() == pytest.approx([2.4112, 1.71, 2.0, 0.61, 0.4])
    # the defaults: gamma 0.997 per decision, lambda 0.95
    assert (P.GAMMA, P.LAMBDA) == (0.997, 0.95)
    adv, _ = P.gae([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [False, True])
    assert adv.tolist() == pytest.approx([0.997 * 0.95, 1.0])
    with pytest.raises(ValueError):
        P.gae([1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [True, False])


def test_returns_of_shapes_the_reward_in_episode_order():
    """Rows given shuffled: B1, A2, A0, B0, A1. Episode 7 (A): Phi (0, 0.05, 0), lost, V (0.2,
    -0.1, -0.5): r = (0.045, -0.05, -1), deltas (-0.245, -0.4, -0.5), A = (-0.7922, -0.76, -0.5).
    Episode 3 (B), cut short: Phi (0, 0.2 / 3), V (0.1, 0.3): r0 = 0.06, A0 = 0.23; B1 invalid."""
    r = P.returns_of(episode=[3, 7, 7, 3, 7], step=[1, 2, 0, 0, 1],
                     done=[False, True, False, False, False], result=[0, -1, 0, 0, 0],
                     hp_frac=[[1, 1], [0.5, 0.5], [1, 1], [1, 1], [1, 0.5]],
                     crowns=[[1, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                     values=[0.3, -0.5, 0.2, 0.1, -0.1], gamma=0.9, lam=0.8)
    assert r.potential.tolist() == pytest.approx([0.2 / 3, 0.0, 0.0, 0.0, 0.05])
    assert r.rewards.tolist() == pytest.approx([0.0, -1.0, 0.045, 0.06, -0.05])
    assert r.advantages.tolist() == pytest.approx([0.0, -0.5, -0.7922, 0.23, -0.76])
    assert r.returns.tolist() == pytest.approx([0.3, -1.0, -0.5922, 0.33, -0.86])
    assert r.valid.tolist() == [False, True, True, True, True]
    assert (r.episodes, r.truncated) == (2, 1)
    with pytest.raises(ValueError):          # done before the episode's last row
        P.returns_of([0, 0], [0, 1], [True, False], [1, 0], [[1, 1]] * 2, [[0, 0]] * 2, [0, 0])


@pytest.mark.parametrize("gamma", [0.997, 1.0])
@pytest.mark.parametrize("result", [1.0, -1.0, 0.0])
def test_shaped_returns_telescope_to_the_terminal_reward(gamma, result):
    """With V = 0 and lambda 1 the return is the discounted shaped reward-to-go, which
    telescopes: G_t = gamma ** (T - 1 - t) * result - Phi_t (Phi of the finished battle is 0);
    from the start, where Phi is 0, that is the terminal reward alone (discounted)."""
    rng = np.random.default_rng(3)
    t_max = 9
    hp = rng.uniform(0.2, 1.0, (t_max, 2))
    hp[0] = 1.0
    crowns = np.zeros((t_max, 2), dtype=int)
    crowns[4:, 0] = 1
    crowns[7:, 1] = 1
    done = [False] * (t_max - 1) + [True]
    res = [0.0] * (t_max - 1) + [result]
    r = P.returns_of(np.zeros(t_max), np.arange(t_max), done, res, hp, crowns,
                     np.zeros(t_max), gamma=gamma, lam=1.0)
    phi = P.potential(hp, crowns)
    t = np.arange(t_max)
    assert phi[0] == 0.0 and r.valid.all() and (r.episodes, r.truncated) == (1, 0)
    assert np.allclose(r.returns, gamma ** (t_max - 1 - t) * result - phi, atol=1e-12)
    assert (gamma ** t * r.rewards).sum() == pytest.approx(gamma ** (t_max - 1) * result)
    if gamma == 1.0:
        assert r.rewards.sum() == pytest.approx(result)


# ---------------------------------------------------------------------------
# the KL controller, the config, the state
# ---------------------------------------------------------------------------

def test_the_kl_controller_moves_by_1_5_outside_the_dead_band():
    c = P.AdaptiveKL(1.0, budget=0.1)                  # band 1.5: no move in [0.0667, 0.15]
    assert (c.band, c.up, c.down, c.lo, c.hi) == (1.5, 1.5, 1.5, 0.01, 10.0)
    assert c.observe(0.2) == pytest.approx(1.5)        # above 0.15: up
    assert c.observe(0.15) == pytest.approx(1.5)       # the band's edge: no move
    assert c.observe(0.1) == pytest.approx(1.5)
    assert c.observe(0.07) == pytest.approx(1.5)
    assert c.observe(0.06) == pytest.approx(1.0)       # below 0.1 / 1.5: down
    assert c.observe(None) == pytest.approx(1.0)       # nothing measured: no move
    assert c.observe(float("nan")) == pytest.approx(1.0)
    for _ in range(20):
        c.observe(1.0)
    assert c.value == 10.0                             # clamped at hi
    for _ in range(40):
        c.observe(0.0)
    assert c.value == 0.01                             # and at lo
    assert P.AdaptiveKL(50.0).value == 10.0
    with pytest.raises(ValueError):
        P.AdaptiveKL(1.0, band=0.5)


def test_the_config_and_the_state_it_carries():
    cfg = P.PPOConfig()
    assert 0.05 <= cfg.kl_budget <= 0.1 and (cfg.kl_up, cfg.kl_down) == (1.5, 1.5)
    assert (cfg.gamma, cfg.lam, cfg.phi_tower, cfg.phi_crown) == (0.997, 0.95, 0.1, 0.2)
    assert cfg.precision == "bf16" and cfg.critic_trunk == "detach" and cfg.value_clip
    s = P.PPOState.new(cfg)
    assert (s.kl.value, s.wait.value, s.kl.budget, s.wait.budget) == (1.0, 1.0, 0.075, 0.005)
    s.kl.observe(1.0)                                  # 1.0 > 1.5 * 0.075
    s.updates = 3
    blob = {P.PPO_KEY: {"state": s.to_json()}}
    back = P.PPOState.from_blob(blob, cfg)
    assert (back.kl.value, back.wait.value, back.updates) == (1.5, 1.0, 3)
    assert back.kl.budget == cfg.kl_budget and back.wait.budget == cfg.wait_kl_budget
    reset = P.PPOState.from_blob(blob, cfg, reset=True)
    assert (reset.kl.value, reset.updates) == (1.0, 3)
    assert P.PPOState.from_blob({}, cfg).to_json() == {"kl_coef": 1.0, "wait_coef": 1.0,
                                                       "updates": 0}
    assert json.loads(json.dumps(cfg.to_json())) == cfg.to_json()
    for bad in ({"critic_trunk": "both"}, {"values": "x"}, {"precision": "fp16"},
                {"epochs": 0}, {"clip": 0.0}, {"gamma": 1.5}, {"kl_band": 0.9}):
        with pytest.raises(ValueError):
            P.PPOConfig(**bad)


# ---------------------------------------------------------------------------
# rows, records, the format
# ---------------------------------------------------------------------------

def test_row_masks_and_the_tile_card():
    """Rows: no draw, wait, slot 0 with a tile, slot 2 fallen back, a press, slot 1 with a tile,
    slot 3 drawn and ended in a wait (no legal tile anywhere: fallback)."""
    a = {"gate_a": np.array([-1, 0, 1, 3, 5, 2, 4]),
         "tile_a": np.array([-1, -1, 4, 0, -1, 2, -1]),
         "fallback": np.array([False, False, False, True, False, False, True])}
    m = P.row_masks(a)
    assert m.choice.tolist() == [False, True, True, True, True, True, True]
    assert m.learn.tolist() == [False, True, True, False, True, True, False]
    assert m.tile.tolist() == [False, False, True, False, False, True, False]
    assert m.fallback.tolist() == [False, False, False, True, False, False, True]
    hand = np.tile([4, 7, 8, 2], (7, 1))
    assert P.card_ids(hand, a["gate_a"]).tolist() == [4, 4, 4, 8, 4, 7, 2]
    assert P.card_ids(np.array([[0, 7, 8, 2]]), np.array([0])).tolist() == [0]


def test_decision_record_from_rs_decide_decisions():
    mask = (True,) * 7
    d = RD.Decision(team=1, tick=500, kind=RD.PLAY, slot=1, card=5, mask=mask, gate_a=2,
                    logq_gate=-1.2, tile_cands=(10, 11, 12), tile_a=1, logq_tile=-0.9,
                    value=(0.3, -0.4))
    rec = P.decision_record(d)
    assert rec["tile_cands"].tolist() == [10, 11, 12] + [-1] * 13
    assert rec["mask"].tolist() == list(mask) and int(rec["gate_a"]) == 2
    assert int(rec["tile_a"]) == 1 and float(rec["logq_tile"]) == pytest.approx(-0.9)
    assert rec["fallback"] is False and int(rec["dt"]) == 10 and rec["capped"] is False
    assert rec["value"].tolist() == pytest.approx([0.3, -0.4])
    assert P.decision_record(d, dt=7)["dt"] == 7
    assert P.decision_record(dataclasses.replace(d, fallback=True))["fallback"] is True
    no_tile = RD.Decision(team=1, tick=500, kind=RD.WAIT, reason=RD.R_NO_TILE, mask=mask,
                          gate_a=3, logq_gate=-2.0)
    assert P.decision_record(no_tile)["fallback"] is True
    drawn_wait = RD.Decision(team=1, tick=500, kind=RD.WAIT, reason=RD.R_WAIT, mask=mask,
                             gate_a=0, logq_gate=-0.1)
    assert P.decision_record(drawn_wait)["fallback"] is False
    press = RD.Decision(team=1, tick=500, kind=RD.PRESS, button=0, mask=mask, gate_a=5,
                        logq_gate=-3.0)
    rec = P.decision_record(press)
    assert rec["fallback"] is False and int(rec["tile_a"]) == -1
    none = P.decision_record(RD.Decision(team=1, tick=500, kind=RD.WAIT, reason=RD.R_NO_CARD))
    assert int(none["gate_a"]) == -1 and np.isnan(none["value"]).all()
    with pytest.raises(ValueError):
        P.decision_record(dataclasses.replace(d, tile_cands=tuple(range(17))))
    cfg = RD.DecideConfig.sampling(0.4, presses=True)
    assert P.rollout_meta(cfg, ckpt="it000.pt") == {
        "format": P.FORMAT, "schema": RS.SCHEMA_V2, "wait_bias": 0.4, "explore": 0.15,
        "tile_temp": 1.0, "dt": 10, "presses": True, "mode": "sample", "ckpt": "it000.pt"}
    with pytest.raises(ValueError):
        P.rollout_meta(RD.DecideConfig.live())


def test_check_rollout_names_what_is_wrong():
    a = synthetic()
    assert P.check_rollout(a, META) == []

    def problems(change, meta=META):
        b = {k: v.copy() for k, v in a.items()}
        change(b)
        return " | ".join(P.check_rollout(b, meta))

    assert "missing fields: mask" in problems(lambda b: b.pop("mask"))
    assert "mask has shape" in problems(lambda b: b.update(mask=b["mask"][:, :5]))

    def masked_draw(b):
        b["gate_a"][1] = 2
        b["mask"][1, 2] = False
    assert "drew a gate entry their mask leaves out" in problems(masked_draw)

    def early_done(b):
        b["done"][0] = True
    assert "done before" in problems(early_done)

    def repeated(b):
        b["step"][1] = 0
    assert "repeat a step" in problems(repeated)

    def wait_masked(b):
        b["mask"][0, 0] = False
    assert "wait masked" in problems(wait_masked)

    def play_without_tile(b):
        b["gate_a"][2] = 1
    assert "plays without a tile_a" in problems(play_without_tile)
    assert "meta format" in problems(lambda b: None, {**META, "format": "x"})
    assert "wait_bias" in problems(lambda b: None, {k: v for k, v in META.items()
                                                     if k != "wait_bias"})
    assert "schema" in problems(lambda b: None, {**META, "schema": "rs_rows/legacy"})


def test_write_load_and_the_data_files(tmp_path):
    a = synthetic()
    pa = P.write_rollout(tmp_path / "w0", records_of(a), META)
    pb = P.write_rollout(tmp_path / "w1.npz", records_of(synthetic(seed=1)), META)
    assert (pa.name, pb.name) == ("w0.npz", "w1.npz")
    files = P.data_files([tmp_path])
    assert files == [str(pa), str(pb)] == P.data_files([str(tmp_path / "*.npz"), str(pa)])
    r = P.load_rollout(files)
    assert r.n == 12 and r.buttons == BUTTONS and r.files == tuple(files)
    assert r.meta["format"] == P.FORMAT and r.meta["wait_bias"] == 0.4
    # episode ids stay unique across the files: (0, 1) and (0, 1) become 0..3
    assert np.unique(r.arrays["episode"]).tolist() == [0, 1, 2, 3]
    assert r.arrays["dt"].tolist() == [10] * 12 and not r.arrays["fallback"].any()
    assert np.isnan(r.arrays["value"]).all() and r.arrays["value"].shape == (12, 2)
    assert np.array_equal(r.arrays["unit_feats"][:6], a["unit_feats"])
    data, rows, width = P.rollout_bytes(files)
    assert rows == 12 and width == P.GATE_PLAYS + BUTTONS + P.CANDS and data > 0
    assert P.ram_need(files) > 2 * data
    assert P.summary(r)["episodes"] == 4 and P.summary(r)["wins"] == 2
    other = P.write_rollout(tmp_path / "other", records_of(a), {**META, "wait_bias": 1.0})
    with pytest.raises(P.RolloutError):
        P.load_rollout([pa, other])
    broken = records_of(a)
    del broken[2]["mask"]
    with pytest.raises(P.RolloutError):
        P.write_rollout(tmp_path / "broken", broken, META)
    with pytest.raises(ValueError):
        P.data_files([tmp_path / "missing.npz"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        P.data_files([tmp_path / "empty"])


# ---------------------------------------------------------------------------
# the distribution (torch)
# ---------------------------------------------------------------------------

def test_gate_and_tile_log_probs_are_rs_decide_sampling():
    torch = _torch()
    rng = np.random.default_rng(7)
    rows, g_width = 40, P.GATE_PLAYS + BUTTONS
    g = rng.normal(0, 2, (rows, g_width)).astype(np.float32)
    g[3, 2] = -1e4                                      # the model's own empty-slot logit
    mask = rng.random((rows, g_width)) < 0.6
    mask[:, 0] = True
    mask[np.arange(rows), 1 + rng.integers(0, 4, rows)] = True
    dt = rng.choice([10, 20, 7], rows)
    capped = np.zeros(rows, dtype=bool)
    capped[5] = True
    wb = 0.7
    lq = P.gate_log_probs(torch.from_numpy(g), torch.from_numpy(mask), wb,
                          torch.from_numpy(dt.astype(np.float32)), torch.from_numpy(capped))
    for i in range(rows):
        gi = g[i].astype(np.float64)
        gi[0] += wb
        q = RD.gate_probs(gi, mask[i], int(dt[i]), bool(capped[i]))
        got = lq[i].double().numpy()
        assert np.allclose(np.exp(got), q, atol=1e-6), i
        # in the log domain only where q > 0.01: a play's log q carries log(1 - q_wait), whose
        # fp32 error grows as 1 / (1 - q_wait) on a row that nearly always waits
        big = q > 1e-2
        assert np.allclose(got[big], np.log(q[big]), atol=1e-5), i
        assert (lq[i].numpy()[~mask[i]] == P.NEG).all()
    assert lq[5, 0].item() == P.NEG                     # capped: never a wait
    c = P.CANDS
    logits = rng.normal(0, 3, (rows, c)).astype(np.float32)
    n_c = rng.integers(1, c + 1, rows)
    cmask = np.arange(c)[None] < n_c[:, None]
    for explore, temp in ((0.15, 1.0), (0.3, 2.0), (0.0, 1.0)):
        lp = P.tile_log_probs(torch.from_numpy(logits), torch.from_numpy(cmask), explore, temp)
        for i in range(rows):
            k = int(n_c[i])
            p = RD.tile_probs(logits[i, :k].astype(np.float64), explore, temp)
            assert np.allclose(np.exp(lp[i, :k].double().numpy()), p, atol=1e-6), (i, explore)
            assert (lp[i, k:].numpy() == P.NEG).all()
    # the same logits on another mask are another distribution: the rollout's mask matters
    one = torch.tensor([[0.0, 1.0, 2.0, 0.5, 0.0]])
    a = P.gate_log_probs(one, torch.tensor([[True, True, False, True, True]]), 0.0)
    b = P.gate_log_probs(one, torch.ones(1, 5, dtype=torch.bool), 0.0)
    assert abs(a[0, 1].item() - b[0, 1].item()) > 0.1


def test_entropy_kl_and_the_joint_by_hand():
    torch = _torch()
    ln = math.log
    lq = torch.tensor([[ln(0.5), ln(0.25), ln(0.25), P.NEG]])
    mask = torch.tensor([[True, True, True, False]])
    assert P.masked_entropy(lq, mask).item() == pytest.approx(1.5 * ln(2), abs=1e-6)
    ref = torch.tensor([[ln(0.25), ln(0.25), ln(0.5), P.NEG]])
    assert P.masked_kl(ref, lq, mask).item() == pytest.approx(0.25 * ln(2), abs=1e-6)
    assert P.masked_kl(lq, lq, mask).item() == pytest.approx(0.0, abs=1e-7)
    want = 0.25 * ln(0.25 / 0.5) + 0.75 * ln(0.75 / 0.5)
    assert P.bernoulli_kl(torch.tensor([ln(0.25)]), torch.tensor([ln(0.5)])).item() == (
        pytest.approx(want, abs=1e-6))
    assert P.log1mexp(torch.tensor([ln(0.25), ln(0.9)])).tolist() == pytest.approx(
        [ln(0.75), ln(0.1)], abs=1e-6)
    lpt = torch.tensor([[ln(0.2), ln(0.3), ln(0.5)], [ln(0.6), ln(0.4), P.NEG]])
    act = {"gate_a": torch.tensor([1, 0]), "tile_a": torch.tensor([2, -1]),
           "tile_rows": torch.tensor([True, False])}
    two = torch.cat([lq, lq])
    assert P.joint_log_prob(two, lpt, act).tolist() == pytest.approx(
        [ln(0.25) + ln(0.5), ln(0.5)], abs=1e-6)


# ---------------------------------------------------------------------------
# the update (torch, CPU)
# ---------------------------------------------------------------------------

def test_the_rollouts_distributions_and_the_ratio_at_lr_0(tmp_path):
    torch = _torch()
    model = tiny_model()
    rollout, decisions, _ = decide_rollout(model, tmp_path)
    assert rollout.n == 18 and rollout.buttons == BUTTONS
    a = rollout.arrays
    m = P.row_masks(a)
    cards = P.card_ids(a["hand"], a["gate_a"])
    idx = np.arange(rollout.n)
    b = P.obs_batch(a, idx, "cpu")
    act = P.action_tensors(a, idx, "cpu", m.tile, cards)
    with torch.no_grad():
        out = model.outputs(b, card_ids=act["card"])
    lq, lpt = P.log_probs(out["gate"], out["tile"], act, P.Sampling.from_meta(rollout.meta))
    for i, d in enumerate(decisions):                   # the decisions' own distributions
        if d.gate_a < 0:
            assert not m.choice[i]
            continue
        assert np.allclose(lq[i].exp().double().numpy(), np.asarray(d.gate_q), atol=1e-5), i
        assert (lq[i] > P.NEG).tolist() == list(d.mask)
        if m.tile[i]:
            assert lpt[i, d.tile_a].exp().item() == pytest.approx(math.exp(d.logq_tile),
                                                                  abs=1e-5)
    valid = np.ones(rollout.n, dtype=bool)
    valid[-1] = False                                   # episode 2 is cut short
    ref = copy.deepcopy(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = P.PPOConfig(lr=0.0, critic_lr=0.0, epochs=2, minibatch=5, chunk=7, precision="fp32")
    state = P.PPOState.new(cfg)
    stats = P.update(model, ref, rollout, cfg, state, device="cpu")
    assert (stats["rows"], stats["episodes"], stats["truncated"]) == (18, 3, 1)
    assert (stats["rows_valid"], stats["episodes_done"]) == (17, 2)
    assert (stats["wins"], stats["losses"], stats["draws"]) == (1, 1, 0)
    assert stats["rows_pg"] == int((m.learn & valid).sum())
    assert stats["rows_choice"] == int((m.choice & valid).sum())
    assert stats["rows_fallback"] == int((m.fallback & valid).sum())
    assert stats["epochs"] == 2 and state.updates == 1 and stats["updates"] == 1
    assert stats["ratio_dev_max"] < 1e-4                # every ratio 1
    if stats["rows_pg"]:
        assert stats["logp_drift_max"] < 1e-5
    assert stats["value_drift_max"] < 1e-5
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k
    if stats["rows_choice"]:
        assert abs(stats["kl"]) < 1e-5 and abs(stats["kl_wait"]) < 1e-5
        # nothing drifted from the reference: both coefficients come down
        assert state.kl.value == pytest.approx(1 / 1.5)
        assert state.wait.value == pytest.approx(1 / 1.5)
        assert stats["q_play"] == pytest.approx(stats["q_play_ref"], abs=1e-5)
    again = P.update(model, ref, rollout, cfg.replace(values="rollout"), P.PPOState.new(cfg))
    assert again["return_mean"] == pytest.approx(stats["return_mean"], abs=1e-5)


def test_a_positive_advantage_raises_its_probability():
    """One trained row (a slot-0 play among three candidates) in a won one-row episode, beside
    a lost row without a draw: its advantage, 1 - V, is positive whatever the critic says, and
    one Adam step at lr 1e-5 raises its log-probability. The critic step moves value2 and never
    value_head."""
    torch = _torch()
    model = tiny_model(1)
    ref = copy.deepcopy(model)
    a = P.with_defaults(synthetic(episodes=2, steps=1))      # dt, capped: action_tensors
    a["gate_a"][:] = [1, -1]
    a["tile_cands"][0, :3] = [5, 9, 13]
    a["tile_a"][0] = 1
    a["logq_gate"][1] = 0.0
    dist = P.Sampling(META["wait_bias"])
    idx = np.arange(2)

    def logp_now():
        m = P.row_masks(a)
        act = P.action_tensors(a, idx, "cpu", m.tile, P.card_ids(a["hand"], a["gate_a"]))
        with torch.no_grad():
            out = model.outputs(P.obs_batch(a, idx, "cpu"), card_ids=act["card"])
        lq, lpt = P.log_probs(out["gate"], out["tile"], act, dist)
        return lq, lpt, P.joint_log_prob(lq, lpt, act)

    lq, lpt, joint = logp_now()
    a["logq_gate"][0] = lq[0, 1].item()                # what the rollout would have stored
    a["logq_tile"][0] = lpt[0, 1].item()
    before = joint[0].item()
    rollout = P.make_rollout(a, META)
    head = {k: v.clone() for k, v in model.value_head.state_dict().items()}
    critic = {k: v.clone() for k, v in model.value2.state_dict().items()}
    cfg = P.PPOConfig(lr=1e-5, critic_lr=1e-3, epochs=1, minibatch=64, ent_coef=0.0,
                      precision="fp32")
    stats = P.update(model, ref, rollout, cfg, P.PPOState.new(cfg))
    assert stats["rows_pg"] == 1 and stats["adv_raw_mean"] > 0
    assert stats["ratio_dev_max"] < 1e-4
    after = logp_now()[2][0].item()
    assert after > before
    for k, v in model.value_head.state_dict().items():
        assert torch.equal(v, head[k]), k
    assert any(not torch.equal(v, critic[k]) for k, v in model.value2.state_dict().items())


# ---------------------------------------------------------------------------
# the CLI and the checkpoint
# ---------------------------------------------------------------------------

def test_the_cli_writes_a_checkpoint_that_loads_and_continues(tmp_path, capsys):
    torch = _torch()
    from sim import model_tokens as mt
    model = tiny_model()
    meta = {"schema": RS.SCHEMA_V2, "n_cards": N_CARDS, "n_units": N_UNITS, "d": 16,
            "layers": 1, "heads": 2}
    ckpt = tmp_path / "it000.pt"
    torch.save({"model": model.state_dict(), "meta": meta, "epoch": 0}, ckpt)
    rollout, _, _ = decide_rollout(model, tmp_path)
    data = rollout.files[0]
    assert P.main(["--check", "--data", str(tmp_path)]) == 0
    shown = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert shown["rows"] == 18 and shown["episodes"] == 3 and shown["format"] == P.FORMAT
    out, opt = tmp_path / "it001.pt", tmp_path / "opt.pt"
    common = ["--ref", str(ckpt), "--data", data, "--device", "cpu", "--lr", "0",
              "--critic-lr", "0", "--epochs", "1", "--minibatch", "8", "--opt-state", str(opt)]
    assert P.main(["--ckpt", str(ckpt), "--out", str(out), *common]) == 0
    stats = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert stats["out"] == str(out) and stats["updates"] == 1 and opt.exists()
    blob = torch.load(out, map_location="cpu")
    assert blob["meta"] == meta and blob["epoch"] == 0
    assert blob["ppo"]["state"]["updates"] == 1 and blob["ppo"]["rollouts"] == [data]
    assert blob["ppo"]["config"]["lr"] == 0.0 and blob["ppo"]["stats"]["rows"] == 18
    loaded = mt.policy_from_meta(blob["meta"], blob["model"])
    loaded.load_state_dict(blob["model"])               # strict
    for k, v in model.state_dict().items():
        assert torch.equal(blob["model"][k], v), k      # lr 0
    kl_coef = blob["ppo"]["state"]["kl_coef"]
    out2 = tmp_path / "it002.pt"
    assert P.main(["--ckpt", str(out), "--out", str(out2), *common]) == 0
    blob2 = torch.load(out2, map_location="cpu")
    assert blob2["ppo"]["state"]["updates"] == 2
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["kl_coef"] == (
        pytest.approx(kl_coef, abs=1e-5))
    # refusals: exit 2, nothing written
    assert P.main(["--check", "--data", str(tmp_path / "missing.npz")]) == 2
    assert P.main(["--check", "--data", data, "--ram-budget-gb", "1e-9"]) == 2
    assert P.main(["--data", data]) == 2
    legacy = tmp_path / "legacy.pt"
    torch.manual_seed(0)
    old = mt.TokenPolicy(N_CARDS, N_UNITS, d=16, layers=1, heads=2)
    torch.save({"model": old.state_dict(), "meta": {k: v for k, v in meta.items()
                                                     if k != "schema"}}, legacy)
    assert P.main(["--ckpt", str(legacy), "--out", str(tmp_path / "x.pt"), *common]) == 2
    other = tmp_path / "other.pt"
    torch.save({"model": model.state_dict(), "meta": {**meta, "cards": {"Cannon": 2}}}, other)
    assert P.main(["--ckpt", str(other), "--out", str(tmp_path / "y.pt"), *common]) == 2
    assert not (tmp_path / "x.pt").exists() and not (tmp_path / "y.pt").exists()


@pytest.mark.gpu
def test_bf16_on_cuda_keeps_the_ratio_near_1(tmp_path):
    torch = _torch()
    model = tiny_model()
    rollout, _, _ = decide_rollout(model, tmp_path)
    ref = copy.deepcopy(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = P.PPOConfig(lr=0.0, critic_lr=0.0, epochs=1, minibatch=8, precision="bf16")
    stats = P.update(model, ref, rollout, cfg, P.PPOState.new(cfg), device="cuda")
    assert stats["ratio_dev_max"] < 0.25
    for k, v in model.state_dict().items():
        assert torch.equal(v.cpu(), before[k]), k
