r"""PPO for the token policy in RoyaleSim (R02): GAE on the v2 critic, a potential-shaped reward,
the clipped ratio over the joint masked action the rollout sampled, and adaptive KL anchors
toward the frozen imitation model.

    .venvs\buildabot\Scripts\python.exe tools\rl\ppo.py --ckpt runs\rl\NAME\it000.pt ^
        --ref runs\tokens\v2_hog\best.pt --data runs\rl\NAME\it000 --out runs\rl\NAME\it001.pt
    .venvs\buildabot\Scripts\python.exe tools\rl\ppo.py --check --data runs\rl\NAME\it000
        (validates the rollout files and prints what they hold; no model, no GPU)

One call is one update: the rollouts of one iteration in (.npz files, a folder of them or a glob;
tools/rl/rollout.py, R01, writes them), one checkpoint out. The driver (R04,
tools/rl/train_loop.py) alternates it with collection, as rs_selfplay.loop did with its train
step, and passes each update's --out as the next one's --ckpt: the KL coefficients travel in the
checkpoint (blob["ppo"]), the Adam moments in --opt-state when it is given. The last stdout line
is the update's statistics as JSON. Exit 0 written; 2 bad arguments, bad rollouts or the RAM
budget. Nothing here touches the emulator; it never imports the memory reader.

WHAT IS LEARNED is exactly the distribution the rollout sampled from: tools/rs_decide.py's
sampling preset (DecideConfig.sampling) on a TokenPolicy v2 (sim/model_tokens.py):
  gate   g = the model's [wait, slot 0..3, button 0..B-1] logits, wait += wait_bias, every entry
         the rollout's `mask` left out at -1e9; p = softmax(g); the decision waits with
         q_wait = p_wait ** (dt / 20) (0 when capped) and otherwise takes entry k >= 1 with
         (1 - q_wait) * softmax(g[1:])[k - 1]                         (rs_decide.gate_probs)
  tile   on a play, among the rollout's `tile_cands` (the legal tiles it drew from, up to 16):
         (1 - explore) * softmax(tile logits / temp) + explore / n    (rs_decide.tile_probs)
         with the tile head conditioned on the played card (hand[slot]), as the rollout's
         TokenPolicyAdapter.tiles was.
  joint  log pi(a) = log q(gate_a) + log p(tile_a) on a play; a wait or a press has no tile.
The update recomputes these on the SAME masks and candidate sets the rollout recorded (the masks
are decide's: affordability, plays in flight, button_mask; the model's own -1e4 for an empty
slot or an unavailable button comes from the observation, which is stored as it was seen).
Treated as part of the state, as rs_selfplay did: the candidate set itself, which depends on the
old tile ranking. The model runs in eval mode (no dropout), so at lr 0 every ratio is 1 up to
float rounding (tests/pipeline/test_ppo.py).

Rows the policy did not choose on are not trained as actions. A row with no draw (gate_a -1:
nothing was playable, decide's "no_card") and a FALLBACK row (the command was not the draw:
decide's fallback took another slot because the drawn one had no legal tile, or no slot had one
and the decision waited) carry no policy-gradient term: the probability of what actually ran is
not log q(gate_a) + log p(tile_a). They stay in the trajectory: they have a state, a value and a
reward, and the gate KL and entropy of a fallback row (its draw was a real choice) still count.

THE ROLLOUT FORMAT (FORMAT = "rl_rollout/v1"). R01 (tools/rl/rollout.py) writes it; it was not
on the branch when this was written, so this is the contract R01 is asked to follow (its
docstring is to name it), and write_rollout / decision_record below are there for it to use. One
row per learner decision (every 10 ticks after tick 90, both seats when both are learners), in
any order; an episode is one battle seen from one learner seat.
  observation   every rs_schema.V2.model_inputs column (unit_ids ... ability_state) at the
                schema's shape and dtype: the v2 obs_row the policy decided on, with the
                perception noise it saw, exactly as the model saw it (B, the ability buttons,
                is ability_card's width)
  episode int32, step int32     the episode id (unique within the file) and the decision's index
                in it (strictly increasing within an episode; gaps are allowed: gamma is per row)
  done bool     the battle ended after this decision: there is no state after it (only on an
                episode's last row). An episode whose last row is not done was cut short: that
                row only bootstraps the one before it and learns nothing itself
  result float32  on the done row: +1 the learner won, -1 lost, 0 a draw (result_of)
  hp_frac float32[2], crowns int8[2]   (own, enemy) at the decision, from the TRUE state, not
                the noisy observation (reward_parts): the mean over a side's three crown towers
                of hp / max hp, and the crowns each side has taken
  mask bool[5 + B]   decide's gate mask (Decision.mask)
  gate_a int16  the gate entry drawn (Decision.gate_a; -1 nothing drawn)
  logq_gate float32, tile_cands int16[C] (-1 pads, C = 16), tile_a int16 (-1 no tile),
  logq_tile float32  Decision's sampling fields
  optional: fallback bool (the command was not the draw; decision_record works it out),
                capped bool (q_wait was 0), dt int16 (the decision's cadence, default 10),
                value float32[2] (the critic's raw [outcome, won logit], Decision.value)
  meta          the .npz entry "rollout_json", JSON text: {"format", "schema": "rs_rows/v2",
                "wait_bias", "explore", "tile_temp", ...} (rollout_meta(DecideConfig), plus
                anything R01 adds, e.g. the checkpoint, engine, delay and noise it ran under)
Other entries are kept when every file has them, and ignored; write_rollout(extra=) writes such
entries (the R05 teacher's: tools/rl/teacher.FIELDS, read only with --teacher-coef).

REWARD. r_t = result on the done row (+1 / 0 / -1) + gamma * Phi(s_t+1) - Phi(s_t), with
Phi = 0.1 * (own - enemy tower-hp fraction) + 0.2 * (own - enemy crowns) / 3 and Phi of the
finished battle taken as 0 (royalelearn rewards.py's rule): summed over an episode the shaping
telescopes to -Phi(s_0), which no policy changes, so it moves when the signal arrives and not
which policy is best (Ng, Harada and Russell 1999); reading Phi off the final state instead
would pay for the margin beside the win.

RETURNS. GAE with gamma 0.997 per decision and lambda 0.95, breaking at every episode end, on
V(s) = tanh(z / 2) = 2 sigmoid(z) - 1 of the critic's won logit z (value2[:, 1]; train_tokens
copies value_head into value2 before every checkpoint, so the critic starts as the imitation win
head read on the +-1 scale). The critic predicts the SHAPED return directly: parameterising it
as V_win - Phi instead would cancel the shaping exactly. Values are recomputed by the current
critic over every row before the epochs ("values": "recompute"; "rollout" uses the stored
`value` where it is finite), and the advantages are standardised once, over the trained rows.

THE LOSS, per minibatch (means over the rows each term covers):
  policy   the clipped ratio exp(log pi_new - log pi_old) with log pi_old the rollout's
           logq_gate + logq_tile, with royalelearn's dual clip (3) for negative advantages
  value    the critic's squared error, value-clipped around the pre-update value (0.2)
  entropy  of the joint over the legal actions: the gate's over its masked entries, plus on a
           play the tile's over the candidates (the chain rule with the sampled slot)
  KL       KL(reference || policy), forward as royaleimitate's regularisers: the gate's exact
           over the masked entries plus, on a play, the tile's over the candidates; in nats per
           decision (a drawn row), the reference under the same masks, wait bias, cadence and
           exploration mixture. Times kl_coef.
  wait KL  the Bernoulli KL of q_wait alone (royaleimitate's noop_marginal factor), times its
           own coefficient: the joint budget can be met while the play rate drifts, and the
           play rate (35-40 plays a match for pros) is what makes the sparring bot human-like.
The reference runs once per update, before the epochs, and its log-probabilities are cached.

KL CONTROL (AdaptiveKL, royaleimitate regularisers.py AdaptiveCoefficient): once per update, the
mean KL the first epoch measured moves the coefficient: above band x budget it is multiplied by
1.5, below budget / band divided by 1.5, inside the dead band left alone, then clamped to
[kl_min, kl_max]. Budgets: 0.075 nats per decision for the joint (the plan's 0.05-0.1), 0.005
for the wait marginal.

THE CRITIC TRUNK. "detach" (default): the value loss trains value2 alone (its input, the
encoder's cls, is detached), so the critic cannot move the policy's encoder and the KL anchors
measure the policy alone. "shared": the value loss also trains the encoder. value_head (the
imitation model's) is never trained. The actor and the critic are clipped separately
(max_grad_norm each) and have their own learning rates.

PRECISION AND MEMORY. bf16 autocast on CUDA (fp32 on CPU, or --precision fp32); the
distribution arithmetic runs in fp32 outside the autocast. The rollouts stay in host RAM and each
minibatch goes to the device. Before loading, the uncompressed size of the rollouts (from the
.npy headers) plus what the update derives is checked against --ram-budget-gb (default 6: the
learner runs while the rollout workers are idle; close MuMu and the browser for big iterations).

THE SEARCH TEACHER (R05, tools/rl/teacher.py; OFF by default, an A/B arm): with --teacher-coef c
> 0 the loss adds c x KL(teacher || policy), teacher.kl_term over the rows whose rollout carries
a search target (the teacher_* entries, write_rollout(extra=teacher.FIELDS)), on the same masks
and candidates as the terms above. teacher.ppo_inputs checks the targets before the epochs;
--teacher-coef on rollouts without them is refused. With c = 0 nothing here reads them.

Not here: mirror augmentation (it would change the action's probability). Pure numpy until a
model is involved; torch is imported where it is used.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import sys
import time
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import rs_decide as RD  # noqa: E402  (the name rs_eval and the tests import it by)
import rs_schema as RS  # noqa: E402

__all__ = [
    "AdaptiveKL",
    "FORMAT",
    "META_KEY",
    "PPOConfig",
    "PPOState",
    "PPO_KEY",
    "Returns",
    "Rollout",
    "RolloutError",
    "Sampling",
    "check_rollout",
    "checkpoint_blob",
    "decision_record",
    "gae",
    "gate_log_probs",
    "load_rollout",
    "main",
    "potential",
    "result_of",
    "returns_of",
    "reward_parts",
    "rollout_meta",
    "tile_log_probs",
    "update",
    "write_rollout",
]

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

FORMAT = "rl_rollout/v1"
#: The .npz entry holding the rollout's meta, as JSON text.
META_KEY = "rollout_json"
#: The checkpoint key the update writes its state under (next to "model" and "meta").
PPO_KEY = "ppo"
PPO_FORMAT = 1
GAMMA, LAMBDA = 0.997, 0.95
PHI_TOWER, PHI_CROWN = 0.1, 0.2
CROWNS = TOWERS = 3
GATE_PLAYS = RD.GATE_PLAYS                    # [wait, slot 0..3]; the buttons follow
TILES = RD.TILES
CANDS = RD.SAMPLE_TOP + RD.SAMPLE_EXTRA      # 16 tile candidates at most
#: A masked log-probability (finite, so no NaN reaches a gradient; exp() of it is exactly 0).
NEG = float(RD.MASKED)
#: |log ratio| is clamped here before exp(): a stale row must not overflow the loss.
MAX_LOG_RATIO = 20.0
OBS_FIELDS: tuple[str, ...] = tuple(RS.V2.model_inputs)
#: Required decision fields: name -> (dtype, per-row shape; "G" the gate width, "C" candidates).
DECISION_FIELDS: dict[str, tuple[str, tuple]] = {
    "episode": ("int32", ()),
    "step": ("int32", ()),
    "done": ("bool", ()),
    "result": ("float32", ()),
    "hp_frac": ("float32", (2,)),
    "crowns": ("int8", (2,)),
    "mask": ("bool", ("G",)),
    "gate_a": ("int16", ()),
    "logq_gate": ("float32", ()),
    "tile_cands": ("int16", ("C",)),
    "tile_a": ("int16", ()),
    "logq_tile": ("float32", ()),
}
#: Optional decision fields: name -> (dtype, per-row shape, the value a missing one reads).
OPTIONAL_FIELDS: dict[str, tuple[str, tuple, Any]] = {
    "fallback": ("bool", (), False),
    "capped": ("bool", (), False),
    "dt": ("int16", (), RD.DECIDE),
    "value": ("float32", (2,), math.nan),
}
KNOWN_FIELDS = frozenset(OBS_FIELDS) | frozenset(DECISION_FIELDS) | frozenset(OPTIONAL_FIELDS)
#: Meta keys every file of one update must agree on.
META_AGREE = ("format", "schema", "wait_bias", "explore", "tile_temp")
CRITIC_TRUNKS = ("detach", "shared")
VALUE_SOURCES = ("recompute", "rollout")
PRECISIONS = ("bf16", "fp32")
RAM_BUDGET_GB = 6.0
GB = 1024 ** 3
#: Bytes per row the update derives beyond the rollout itself, besides the cached reference
#: (float64 returns, advantages, masks and indices).
DERIVED_ROW_BYTES = 160


class RolloutError(ValueError):
    """A rollout file that is not FORMAT, or is inconsistent."""


class CheckpointError(ValueError):
    """A checkpoint this update cannot train (not v2) or a reference that does not match it."""


# ---------------------------------------------------------------------------
# reward parts (for the rollout worker)
# ---------------------------------------------------------------------------

def tower_fraction(player: Mapping[str, Any]) -> float:
    """The mean over a side's three crown towers of hp / max hp, each clipped to [0, 1] (a
    destroyed tower 0), from a state_json player's tower_hp / tower_max_hp."""
    hp = list(player.get("tower_hp") or ())
    mx = list(player.get("tower_max_hp") or ())
    if len(hp) < TOWERS or len(mx) < TOWERS:
        raise ValueError(f"a player needs {TOWERS} tower_hp and tower_max_hp values, got "
                         f"{len(hp)} and {len(mx)}")
    fr = [min(1.0, max(0.0, float(hp[k]) / max(1.0, float(mx[k])))) for k in range(TOWERS)]
    return sum(fr) / TOWERS


def reward_parts(state: Mapping[str, Any], team: int) -> tuple[np.ndarray, np.ndarray]:
    """(hp_frac float32 [own, enemy], crowns int8 [own, enemy]) of the TRUE state for `team`:
    what the rollout records per decision for the potential."""
    team = int(team)
    players = state["players"]
    own, foe = players[team], players[1 - team]
    hp = np.array([tower_fraction(own), tower_fraction(foe)], dtype=np.float32)
    cr = np.array([int(own.get("crowns", 0) or 0), int(foe.get("crowns", 0) or 0)],
                  dtype=np.int8)
    return hp, cr


def result_of(state: Mapping[str, Any], team: int) -> float:
    """The terminal reward for `team` of a finished battle: +1 won, -1 lost, 0 a draw (winner
    not 0 or 1), as rs_match reads the winner."""
    w = state.get("winner")
    w = int(w) if w in (0, 1) else -1
    if w < 0:
        return 0.0
    return 1.0 if w == int(team) else -1.0


def potential(hp_frac: Any, crowns: Any, tower: float = PHI_TOWER,
              crown: float = PHI_CROWN) -> np.ndarray:
    """Phi = tower * (own - enemy hp fraction) + crown * (own - enemy crowns) / 3, over the last
    axis of [..., 2] arrays (own, enemy)."""
    hp = np.asarray(hp_frac, dtype=np.float64)
    cr = np.asarray(crowns, dtype=np.float64)
    return tower * (hp[..., 0] - hp[..., 1]) + crown * (cr[..., 0] - cr[..., 1]) / CROWNS


# ---------------------------------------------------------------------------
# the rollout format
# ---------------------------------------------------------------------------

def rollout_meta(cfg: Any, **extra: Any) -> dict:
    """The meta a rollout sampled under `cfg` (an rs_decide.DecideConfig of the sampling preset)
    records: FORMAT, the schema and the sampling settings the update recomputes the
    probabilities with, plus `extra` (JSON-ready: the checkpoint, engine, delay, noise...)."""
    if not getattr(cfg, "sample", False):
        raise ValueError("rollouts are sampled: use rs_decide.DecideConfig.sampling(...)")
    meta = {"format": FORMAT, "schema": RS.SCHEMA_V2, "wait_bias": float(cfg.wait_bias),
            "explore": float(cfg.explore), "tile_temp": float(cfg.tile_temp),
            "dt": int(cfg.dt), "presses": bool(cfg.presses), "mode": str(cfg.name)}
    meta.update(extra)
    return meta


def decision_record(d: Any, *, dt: int | None = None, cands: int = CANDS) -> dict:
    """The decision fields of one row from an rs_decide.Decision of the sampling preset: mask,
    gate_a, logq_gate, tile_cands (padded to `cands` with -1), tile_a, logq_tile, fallback, capped,
    dt (the cadence decide was given: dt=, default cfg.dt; pass it) and value (NaN without a
    critic). fallback is set when the command is not the draw: decide's own flag, or a drawn
    play that ended in a wait (no legal tile in any slot)."""
    tc = tuple(int(t) for t in d.tile_cands)
    if len(tc) > cands:
        raise ValueError(f"{len(tc)} tile candidates; the record holds {cands}")
    tiles = np.full(cands, -1, dtype=np.int16)
    tiles[:len(tc)] = tc
    gate_a = int(d.gate_a)
    fallback = bool(gate_a > 0 and (bool(d.fallback) or d.kind == RD.WAIT))
    value = (np.asarray(d.value[:2], dtype=np.float32) if len(d.value) >= 2
             else np.full(2, np.nan, dtype=np.float32))
    return {"mask": np.asarray(d.mask, dtype=bool), "gate_a": np.int16(gate_a),
            "logq_gate": np.float32(d.logq_gate), "tile_cands": tiles,
            "tile_a": np.int16(int(d.tile_a)), "logq_tile": np.float32(d.logq_tile),
            "fallback": fallback, "capped": bool(d.capped),
            "dt": np.int16(RD.DECIDE if dt is None else int(dt)), "value": value}


def _column(records: Sequence[Mapping[str, Any]], name: str, dtype: str,
            default: Any = None, shape: tuple = ()) -> np.ndarray:
    """One field of every record, stacked; a record without it takes `default` (None: the
    field is required, RolloutError)."""
    def one(i, r):
        if name in r:
            return np.asarray(r[name], dtype=dtype)
        if default is None:
            raise RolloutError(f"row {i} has no {name!r}")
        return np.full(shape, default, dtype=dtype)
    return np.stack([one(i, r) for i, r in enumerate(records)])


def write_rollout(path: Any, records: Iterable[Mapping[str, Any]], meta: Mapping[str, Any],
                  *, extra: Mapping[str, str] | None = None) -> Path:
    """One .npz of FORMAT from row dicts (each: the observation columns, decision_record's
    fields and episode, step, done, result, hp_frac, crowns), checked by check_rollout first.
    The observation is cast to the schema dtypes: obs_row's rows already are, so nothing the
    model saw changes. `extra` (name -> dtype, e.g. tools/rl/teacher.FIELDS): more entries,
    each written when any row has it, and then every row must. Returns the path written
    (np.savez_compressed adds .npz)."""
    records = list(records)
    if not records:
        raise RolloutError("no rows to write")
    out: dict[str, np.ndarray] = {}
    for name in OBS_FIELDS:
        out[name] = _column(records, name, RS.V2.dtype(name))
    for name, (dtype, _) in DECISION_FIELDS.items():
        out[name] = _column(records, name, dtype)
    for name, (dtype, shape, default) in OPTIONAL_FIELDS.items():
        if any(name in r for r in records):
            out[name] = _column(records, name, dtype, default, shape)
    for name, dtype in (extra or {}).items():
        if name in KNOWN_FIELDS or name == META_KEY:
            raise ValueError(f"write_rollout: extra entry {name!r} is one of the format's own")
        if any(name in r for r in records):
            out[name] = _column(records, name, dtype)
    meta = dict(meta)
    problems = check_rollout(out, meta)
    if problems:
        raise RolloutError("; ".join(problems[:8]))
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out, **{META_KEY: np.array(json.dumps(meta, sort_keys=True))})
    return path


def with_defaults(arrays: Mapping[str, np.ndarray]) -> dict:
    """`arrays` with every OPTIONAL_FIELDS entry present (a missing one at its default)."""
    out = dict(arrays)
    n = int(len(out["gate_a"]))
    for name, (dtype, shape, default) in OPTIONAL_FIELDS.items():
        if name not in out:
            out[name] = np.full((n, *shape), default, dtype=dtype)
    return out


def check_rollout(arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> list[str]:
    """What is wrong with one rollout (arrays and meta), as messages; [] when it is usable."""
    probs: list[str] = []
    if (meta or {}).get("format") != FORMAT:
        probs.append(f"meta format {(meta or {}).get('format')!r}, expected {FORMAT!r}")
    schema = (meta or {}).get("schema", RS.SCHEMA_V2)
    if schema != RS.SCHEMA_V2:
        probs.append(f"schema {schema!r}: the update trains {RS.SCHEMA_V2!r} rows")
    for key in ("wait_bias",):
        v = (meta or {}).get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            probs.append(f"meta {key!r} must be a finite number (rollout_meta records it)")
    for key in ("explore", "tile_temp"):
        v = (meta or {}).get(key)
        if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)):
            probs.append(f"meta {key!r} must be a finite number")
    missing = [k for k in (*OBS_FIELDS, *DECISION_FIELDS) if k not in arrays]
    if missing:
        return probs + [f"missing fields: {', '.join(missing)}"]
    n = int(np.shape(arrays["gate_a"])[0]) if np.ndim(arrays["gate_a"]) else -1
    if n <= 0:
        return probs + ["no rows"]
    card = np.asarray(arrays["ability_card"])
    if card.ndim != 2:
        return probs + [f"ability_card has shape {card.shape}, expected [rows, B]"]
    buttons = int(card.shape[1])
    want: dict[str, tuple] = {name: (n, *RS.V2.shape(name, buttons)) for name in OBS_FIELDS}
    for name, (_, shape) in DECISION_FIELDS.items():
        if "C" in shape:
            continue
        want[name] = (n, *(GATE_PLAYS + buttons if d == "G" else d for d in shape))
    for name, (_, shape, _) in OPTIONAL_FIELDS.items():
        if name in arrays:
            want[name] = (n, *shape)
    for name, shape in want.items():
        got = tuple(np.shape(arrays[name]))
        if got != shape:
            probs.append(f"{name} has shape {got}, expected {shape}")
    tc = np.asarray(arrays["tile_cands"])
    if tc.ndim != 2 or tc.shape[0] != n or tc.shape[1] < 1:
        probs.append(f"tile_cands has shape {tc.shape}, expected [{n}, C >= 1]")
    if probs:
        return probs
    g = GATE_PLAYS + buttons
    mask = np.asarray(arrays["mask"], dtype=bool)
    gate_a = np.asarray(arrays["gate_a"]).astype(np.int64)
    tile_a = np.asarray(arrays["tile_a"]).astype(np.int64)
    fallback = (np.asarray(arrays["fallback"], dtype=bool) if "fallback" in arrays
                else np.zeros(n, dtype=bool))

    def bad(cond: np.ndarray, what: str) -> None:
        k = int(np.count_nonzero(cond))
        if k:
            probs.append(f"{k} rows {what} (first: row {int(np.flatnonzero(cond)[0])})")

    bad((gate_a < -1) | (gate_a >= g), f"have gate_a outside [-1, {g})")
    bad(~mask[:, 0], "have the wait masked (decide never masks it)")
    drawn = gate_a >= 0
    ok = np.ones(n, dtype=bool)
    rows = np.flatnonzero(drawn & (gate_a < g))
    ok[rows] = mask[rows, gate_a[rows]]
    bad(~ok, "drew a gate entry their mask leaves out (not the rollout's masks?)")
    bad((tc < -1) | (tc >= TILES), f"have tile candidates outside [-1, {TILES})")
    c = tc.shape[1]
    play = (gate_a >= 1) & (gate_a < GATE_PLAYS) & ~fallback
    bad(play & ((tile_a < 0) | (tile_a >= c)), "are plays without a tile_a in their candidates")
    rows = np.flatnonzero(play & (tile_a >= 0) & (tile_a < c))
    bad_cand = np.zeros(n, dtype=bool)
    bad_cand[rows] = tc[rows, tile_a[rows]] < 0
    bad(bad_cand, "drew a padding tile candidate")
    for name in ("logq_gate", "logq_tile"):
        lq = np.asarray(arrays[name], dtype=np.float64)
        bad(drawn & ~(np.isfinite(lq) & (lq <= 1e-6)), f"have a non-finite or positive {name}")
    result = np.asarray(arrays["result"], dtype=np.float64)
    bad(~np.isfinite(result) | (np.abs(result) > 1), "have a result outside [-1, 1]")
    hp = np.asarray(arrays["hp_frac"], dtype=np.float64)
    bad(~np.all(np.isfinite(hp) & (hp >= 0) & (hp <= 1), axis=1), "have hp_frac outside [0, 1]")
    ep = np.asarray(arrays["episode"]).astype(np.int64)
    st = np.asarray(arrays["step"]).astype(np.int64)
    done = np.asarray(arrays["done"], dtype=bool)
    order = np.lexsort((st, ep))
    same = ep[order][1:] == ep[order][:-1]
    dup = np.zeros(n, dtype=bool)
    dup[order[1:]] = same & (st[order][1:] == st[order][:-1])
    bad(dup, "repeat a step of their episode")
    early = np.zeros(n, dtype=bool)
    early[order[:-1]] = done[order][:-1] & same
    bad(early, "are done before their episode's last row")
    return probs


@dataclasses.dataclass
class Rollout:
    """One update's rollouts: the columns (every OPTIONAL_FIELDS entry filled), the meta (the
    first file's, the META_AGREE keys checked equal across files) and the files read."""

    arrays: dict
    meta: dict
    files: tuple = ()

    @property
    def n(self) -> int:
        return int(len(self.arrays["gate_a"]))

    @property
    def buttons(self) -> int:
        return int(self.arrays["ability_card"].shape[1])


def make_rollout(arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any],
                 files: Sequence[str] = ()) -> Rollout:
    """A checked Rollout of in-memory columns (RolloutError when check_rollout objects)."""
    problems = check_rollout(arrays, meta)
    if problems:
        raise RolloutError("; ".join(problems[:8]))
    return Rollout(with_defaults(arrays), dict(meta), tuple(str(f) for f in files))


def read_rollout(path: Any) -> tuple[dict, dict]:
    """(columns, meta) of one .npz, checked (RolloutError naming the file)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            if META_KEY not in z.files:
                raise RolloutError(f"{path}: no {META_KEY!r} entry (not a {FORMAT} rollout)")
            meta = json.loads(str(z[META_KEY][()]))
            arrays = {k: z[k] for k in z.files if k != META_KEY}
    except RolloutError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise RolloutError(f"{path}: {type(exc).__name__}: {exc}") from None
    problems = check_rollout(arrays, meta)
    if problems:
        raise RolloutError(f"{path}: " + "; ".join(problems[:8]))
    return arrays, meta


def load_rollout(paths: Iterable[Any]) -> Rollout:
    """The rollouts of `paths` as one Rollout: episodes renumbered so that ids stay unique
    across files, the optional fields filled per file, extra entries kept when every file has
    them. The files must agree on META_AGREE (the same sampling, so one set of probabilities)."""
    paths = [str(p) for p in paths]
    if not paths:
        raise RolloutError("no rollout files")
    parts, meta0 = [], None
    offset = 0
    for path in paths:
        arrays, meta = read_rollout(path)
        if meta0 is None:
            meta0 = meta
        else:
            diff = [k for k in META_AGREE if meta.get(k) != meta0.get(k)]
            if diff:
                raise RolloutError(f"{path}: meta {', '.join(diff)} differ from {paths[0]}'s")
        arrays = with_defaults(arrays)
        _, inv = np.unique(np.asarray(arrays["episode"]).astype(np.int64), return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        arrays["episode"] = (inv + offset).astype(np.int64)
        offset += int(inv.max()) + 1 if inv.size else 0
        parts.append(arrays)
    shared = set(parts[0])
    for p in parts[1:]:
        shared &= set(p)
    widths = {p["ability_card"].shape[1] for p in parts}
    if len(widths) > 1:
        raise RolloutError(f"the files have different button counts {sorted(widths)}")
    cands = {p["tile_cands"].shape[1] for p in parts}
    out = {}
    for name in sorted(shared):
        if name == "tile_cands" and len(cands) > 1:
            c = max(cands)
            out[name] = np.concatenate([np.pad(p[name], ((0, 0), (0, c - p[name].shape[1])),
                                               constant_values=-1) for p in parts])
            continue
        try:
            out[name] = np.concatenate([np.asarray(p[name]) for p in parts])
        except ValueError:
            if name in KNOWN_FIELDS:
                raise RolloutError(f"field {name!r} does not stack across the files") from None
    return Rollout(out, dict(meta0 or {}), tuple(paths))


def data_files(specs: Iterable[Any]) -> list[str]:
    """The .npz files of --data: files, folders (their *.npz, sorted) and globs (expanded here:
    PowerShell does not), each once, in order. ValueError for one that names nothing."""
    out: list[str] = []
    for spec in specs:
        s = str(spec)
        p = Path(s)
        if any(ch in s for ch in "*?["):
            found = sorted(f for f in glob.glob(s) if Path(f).is_file())
        elif p.is_dir():
            found = sorted(str(x) for x in p.glob("*.npz"))
        elif p.is_file():
            found = [s]
        else:
            raise ValueError(f"--data {s}: no such file or folder")
        if not found:
            raise ValueError(f"--data {s}: no .npz files")
        out += [f for f in found if f not in out]
    return out


def rollout_bytes(paths: Iterable[Any]) -> tuple[int, int, int]:
    """(uncompressed bytes, rows, gate width + candidates) of rollout files, from the .npy
    headers alone (tools/rs_rows_compact.npz_columns): what loading them takes before
    anything is inflated."""
    import rs_rows_compact
    total = rows = width = 0
    for path in paths:
        cols = rs_rows_compact.npz_columns(path)
        for dtype, shape in cols.values():
            total += int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if "gate_a" in cols:
            rows += int(cols["gate_a"][1][0]) if cols["gate_a"][1] else 0
        w = 0
        for name in ("mask", "tile_cands"):
            if name in cols and len(cols[name][1]) == 2:
                w += int(cols[name][1][1])
        width = max(width, w)
    return total, rows, width


def ram_need(paths: Iterable[Any]) -> int:
    """The host RAM an update over `paths` needs, in bytes: the rollouts twice (the files and
    their concatenation), the reference cache ((gate + candidates) float32 a row) and
    DERIVED_ROW_BYTES a row."""
    data, rows, width = rollout_bytes(paths)
    return 2 * data + rows * (4 * width + DERIVED_ROW_BYTES)


# ---------------------------------------------------------------------------
# returns: the shaped reward and GAE
# ---------------------------------------------------------------------------

def gae(rewards: Any, values: Any, next_values: Any, ended: Any, gamma: float = GAMMA,
        lam: float = LAMBDA, valid: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """(advantages, returns) of rows in episode order: delta_t = r_t + gamma * next_values_t -
    values_t (0 on a row `valid` leaves out), A_t = delta_t + gamma * lam * A_t+1 within an
    episode; `ended` marks each episode's last row, where the recursion breaks (next_values
    there is the caller's: 0 after a finished battle). Vectorised over episodes."""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    nv = np.asarray(next_values, dtype=np.float64).reshape(-1)
    end = np.asarray(ended, dtype=bool).reshape(-1)
    n = len(r)
    if not (len(v) == len(nv) == len(end) == n):
        raise ValueError("gae: rewards, values, next_values and ended differ in length")
    if n == 0:
        return np.zeros(0), np.zeros(0)
    if not end[-1]:
        raise ValueError("gae: the last row must end its episode")
    ok = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    delta = np.where(ok, r + gamma * nv - v, 0.0)
    seg = np.concatenate([[0], np.cumsum(end[:-1])]).astype(np.int64)
    starts = np.flatnonzero(np.concatenate([[True], end[:-1]]))
    pos = np.arange(n) - starts[seg]
    table = np.zeros((int(seg[-1]) + 1, int(pos.max()) + 1))
    table[seg, pos] = delta
    adv = np.zeros_like(table)
    carry = np.zeros(table.shape[0])
    for t in range(table.shape[1] - 1, -1, -1):
        carry = table[:, t] + gamma * lam * carry
        adv[:, t] = carry
    out = adv[seg, pos]
    return out, out + v


@dataclasses.dataclass
class Returns:
    """returns_of's result, every array in the rows' input order."""

    advantages: np.ndarray
    returns: np.ndarray
    rewards: np.ndarray
    potential: np.ndarray
    valid: np.ndarray            # False: the last row of a cut-short episode (learns nothing)
    episodes: int
    truncated: int


def returns_of(episode: Any, step: Any, done: Any, result: Any, hp_frac: Any, crowns: Any,
               values: Any, *, gamma: float = GAMMA, lam: float = LAMBDA,
               tower: float = PHI_TOWER, crown: float = PHI_CROWN) -> Returns:
    """The shaped rewards (module doc, REWARD) and GAE advantages and returns of a rollout's
    rows, in any order: rows are put in (episode, step) order, an episode's last row ends it
    (done: no state after it, so next value and next potential 0; not done: cut short, the row
    only bootstraps the one before it and is not valid)."""
    episode = np.asarray(episode).astype(np.int64).reshape(-1)
    n = len(episode)
    step = np.asarray(step).astype(np.int64).reshape(-1)
    done = np.asarray(done, dtype=bool).reshape(-1)
    result = np.asarray(result, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    phi_all = np.asarray(potential(hp_frac, crowns, tower, crown), dtype=np.float64).reshape(-1)
    if not (len(step) == len(done) == len(result) == len(values) == len(phi_all) == n):
        raise ValueError("returns_of: the columns differ in length")
    if n == 0:
        z = np.zeros(0)
        return Returns(z, z, z, z, np.zeros(0, dtype=bool), 0, 0)
    order = np.lexsort((step, episode))
    ep = episode[order]
    last = np.ones(n, dtype=bool)
    last[:-1] = ep[1:] != ep[:-1]
    d = done[order]
    if (d & ~last).any():
        raise ValueError("returns_of: a row is done before its episode's last row")
    trunc = last & ~d
    phi = phi_all[order]
    v = values[order]
    next_phi = np.zeros(n)
    next_phi[:-1] = np.where(last[:-1], 0.0, phi[1:])
    next_v = np.zeros(n)
    next_v[:-1] = np.where(last[:-1], 0.0, v[1:])
    r = np.where(d, result[order], 0.0) + gamma * next_phi - phi
    r[trunc] = 0.0
    adv_sorted, _ = gae(r, v, next_v, last, gamma, lam, valid=~trunc)
    adv = np.empty(n)
    adv[order] = adv_sorted
    rew = np.empty(n)
    rew[order] = r
    valid = np.empty(n, dtype=bool)
    valid[order] = ~trunc
    return Returns(adv, adv + values, rew, phi_all, valid, int(last.sum()), int(trunc.sum()))


def standardise(adv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(adv - mean) / std over the rows `mask` selects, once per update (fewer than two: as is)."""
    sel = adv[mask]
    if sel.size < 2:
        return np.array(adv, dtype=np.float64)
    return (adv - sel.mean()) / (sel.std() + 1e-8)


@dataclasses.dataclass
class RowMasks:
    """Which rows each term covers (row_masks): choice (a draw: KL, entropy), learn (a draw that
    ran: the ratio), tile (a play that ran: its tile terms)."""

    choice: np.ndarray
    learn: np.ndarray
    tile: np.ndarray
    fallback: np.ndarray


def row_masks(arrays: Mapping[str, np.ndarray]) -> RowMasks:
    gate_a = np.asarray(arrays["gate_a"]).astype(np.int64)
    n = len(gate_a)
    fallback = (np.asarray(arrays["fallback"], dtype=bool) if "fallback" in arrays
                else np.zeros(n, dtype=bool))
    choice = gate_a >= 0
    learn = choice & ~fallback
    play = (gate_a >= 1) & (gate_a < GATE_PLAYS)
    tile = learn & play & (np.asarray(arrays["tile_a"]).astype(np.int64) >= 0)
    return RowMasks(choice, learn, tile, choice & fallback)


def card_ids(hand: np.ndarray, gate_a: np.ndarray) -> np.ndarray:
    """The card token the tile head is conditioned on per row: hand[gate_a - 1] on a slot draw
    (the rollout's TokenPolicyAdapter.tiles), else hand slot 0 (the gate call's), clamped >= 0."""
    hand = np.asarray(hand).astype(np.int64)
    gate_a = np.asarray(gate_a).astype(np.int64)
    play = (gate_a >= 1) & (gate_a < GATE_PLAYS)
    slot = np.where(play, gate_a - 1, 0)
    return np.maximum(hand[np.arange(len(hand)), slot], 0)


# ---------------------------------------------------------------------------
# the sampled distribution (torch)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Sampling:
    """The sampling settings the rollout drew with (its meta)."""

    wait_bias: float
    explore: float = RD.SAMPLE_EXPLORE
    tile_temp: float = RD.SAMPLE_TEMP

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any]) -> "Sampling":
        return cls(float(meta["wait_bias"]), float(meta.get("explore", RD.SAMPLE_EXPLORE)),
                   float(meta.get("tile_temp", RD.SAMPLE_TEMP)))


def log1mexp(x: Any) -> Any:
    """log(1 - exp(x)) for x <= 0, stable on both sides of log(1/2); x is held a hair below 0
    (royaleimitate regularisers.log1mexp)."""
    import torch
    x = x.clamp(max=-1e-7)
    return torch.where(x > -math.log(2.0), torch.log(-torch.expm1(x)),
                       torch.log1p(-torch.exp(x)))


def gate_log_probs(gate: Any, mask: Any, wait_bias: float, dt: Any = RD.DECIDE,
                   capped: Any = None) -> Any:
    """log q [N, G] of rs_decide.gate_probs from raw gate logits [N, G] (the model's), the
    rollout's mask [N, G] (bool), the wait bias, the cadence dt (a number or [N]) and capped
    ([N] bool, q_wait 0; None: never). Masked entries read NEG. fp32."""
    import torch
    import torch.nn.functional as F
    g = gate.float()
    g = torch.cat([g[:, :1] + float(wait_bias), g[:, 1:]], 1).masked_fill(~mask, NEG)
    lp = F.log_softmax(g, 1)
    if not torch.is_tensor(dt):
        dt = torch.full((g.shape[0],), float(dt), device=g.device)
    lw = lp[:, 0] * (dt.float() / RD.WAIT_REF_TICKS)
    if capped is not None:
        lw = lw.masked_fill(capped, NEG)
    lplay = log1mexp(lw)
    rest = F.log_softmax(g[:, 1:], 1) + lplay[:, None]
    return torch.cat([lw[:, None], rest], 1).masked_fill(~mask, NEG)


def tile_log_probs(logits: Any, cmask: Any, explore: float = RD.SAMPLE_EXPLORE,
                   temp: float = RD.SAMPLE_TEMP) -> Any:
    """log p [N, C] of rs_decide.tile_probs over the candidates: logits [N, C] (the tile head at
    tile_cands), cmask [N, C] the real candidates. Padding reads NEG. fp32."""
    import torch
    import torch.nn.functional as F
    tl = (logits.float() / float(temp)).masked_fill(~cmask, NEG)
    ls = F.log_softmax(tl, 1)
    explore = float(explore)
    if explore > 0:
        n = cmask.sum(1, keepdim=True).clamp(min=1).float()
        floor = torch.log(explore / n).expand_as(ls)
        ls = floor if explore >= 1 else torch.logaddexp(ls + math.log1p(-explore), floor)
    return ls.masked_fill(~cmask, NEG)


def masked_entropy(logp: Any, mask: Any) -> Any:
    """[N]: -sum over the masked-in entries of p log p."""
    import torch
    t = logp.exp() * logp
    return -torch.where(mask, t, torch.zeros_like(t)).sum(1)


def masked_kl(ref: Any, logp: Any, mask: Any) -> Any:
    """[N]: KL(ref || policy) = sum over the masked-in entries of p_ref (log p_ref - log p)."""
    import torch
    t = ref.exp() * (ref - logp)
    return torch.where(mask, t, torch.zeros_like(t)).sum(1)


def bernoulli_kl(ref_log_wait: Any, log_wait: Any) -> Any:
    """[N]: the KL between the wait / play Bernoullis, both given as log q_wait."""
    rp, p = log1mexp(ref_log_wait), log1mexp(log_wait)
    return ref_log_wait.exp() * (ref_log_wait - log_wait) + rp.exp() * (rp - p)


def action_tensors(arrays: Mapping[str, np.ndarray], idx: np.ndarray, device: Any,
                   tile_rows: np.ndarray, cards: np.ndarray) -> dict:
    """The rollout's action columns of rows `idx` on `device`: mask, gate_a, tile_a, cands,
    cmask, dt, capped, tile_rows (bool: the tile terms count) and card (the tile conditioning)."""
    import torch

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    cands = t(np.asarray(arrays["tile_cands"])[idx].astype(np.int64))
    return {"mask": t(np.asarray(arrays["mask"], dtype=bool)[idx]),
            "gate_a": t(np.asarray(arrays["gate_a"])[idx].astype(np.int64)),
            "tile_a": t(np.asarray(arrays["tile_a"])[idx].astype(np.int64)),
            "cands": cands, "cmask": cands >= 0,
            "dt": t(np.asarray(arrays["dt"])[idx].astype(np.float32)),
            "capped": t(np.asarray(arrays["capped"], dtype=bool)[idx]),
            "tile_rows": t(np.asarray(tile_rows, dtype=bool)[idx]),
            "card": t(np.asarray(cards)[idx].astype(np.int64))}


def obs_batch(arrays: Mapping[str, np.ndarray], idx: np.ndarray, device: Any) -> dict:
    """The model batch of rows `idx`: every V2 model input, int columns long and float columns
    float32 (the rule of model_tokens.rows_batch and rs_decide.row_batch)."""
    import torch
    out = {}
    for name in OBS_FIELDS:
        x = torch.from_numpy(np.ascontiguousarray(np.asarray(arrays[name])[idx])).to(device)
        out[name] = x.long() if RS.V2.dtype(name).startswith(("int", "bool")) else x.float()
    return out


def log_probs(gate: Any, tile: Any, act: Mapping[str, Any], dist: Sampling) -> tuple[Any, Any]:
    """(gate log q [N, G], tile log p [N, C]) of model outputs on the rollout's masks."""
    lq = gate_log_probs(gate, act["mask"], dist.wait_bias, act["dt"], act["capped"])
    lt = tile.float().gather(1, act["cands"].clamp(min=0))
    return lq, tile_log_probs(lt, act["cmask"], dist.explore, dist.tile_temp)


def joint_log_prob(lq: Any, lpt: Any, act: Mapping[str, Any]) -> Any:
    """[N]: log q(gate_a) + log p(tile_a) on the tile rows (a wait or a press: the gate alone)."""
    import torch
    lg = lq.gather(1, act["gate_a"].clamp(min=0)[:, None])[:, 0]
    lt = lpt.gather(1, act["tile_a"].clamp(min=0)[:, None])[:, 0]
    return lg + torch.where(act["tile_rows"], lt, torch.zeros_like(lt))


def critic_value(raw: Any) -> Any:
    """V(s) = tanh(z / 2) = 2 sigmoid(z) - 1 of the critic's won logit z = raw[..., 1] (torch or
    numpy)."""
    if isinstance(raw, np.ndarray):
        return np.tanh(np.asarray(raw, dtype=np.float64)[..., 1] / 2)
    import torch
    return torch.tanh(raw[..., 1].float() * 0.5)


# ---------------------------------------------------------------------------
# the KL controller, the config and the state
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AdaptiveKL:
    """A KL coefficient held to a budget (royaleimitate AdaptiveCoefficient): observe(measured)
    multiplies it by `up` when measured > band * budget, divides it by `down` when measured <
    budget / band, leaves it inside the dead band, and clamps it to [lo, hi]. No measurement
    (None, NaN): no move."""

    value: float = 1.0
    budget: float = 0.075
    band: float = 1.5
    up: float = 1.5
    down: float = 1.5
    lo: float = 0.01
    hi: float = 10.0

    def __post_init__(self) -> None:
        if not (self.band >= 1 and self.up >= 1 and self.down >= 1):
            raise ValueError("AdaptiveKL: band, up and down must be >= 1")
        if not 0 <= self.lo <= self.hi:
            raise ValueError("AdaptiveKL: need 0 <= lo <= hi")
        if not self.budget >= 0:
            raise ValueError("AdaptiveKL: the budget must be >= 0")
        self.value = self.clamp(float(self.value))

    def clamp(self, value: float) -> float:
        return min(max(float(value), self.lo), self.hi)

    def observe(self, measured: float | None) -> float:
        value = self.clamp(self.value)
        if measured is not None and math.isfinite(float(measured)):
            if measured > self.band * self.budget:
                value *= self.up
            elif measured < self.budget / self.band:
                value /= self.down
        self.value = self.clamp(value)
        return self.value


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """One update's settings (module doc). The KL "start" values are used only by a checkpoint
    without PPO state (or with --reset-kl)."""

    gamma: float = GAMMA
    lam: float = LAMBDA
    phi_tower: float = PHI_TOWER
    phi_crown: float = PHI_CROWN
    epochs: int = 3
    minibatch: int = 1024
    chunk: int = 2048
    lr: float = 1e-5
    critic_lr: float = 1e-4
    adam_eps: float = 1e-5
    max_grad_norm: float = 1.0
    clip: float = 0.2
    dual_clip: float = 3.0
    value_clip: float | None = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.003
    kl_budget: float = 0.075
    kl_start: float = 1.0
    wait_kl_budget: float = 0.005
    wait_kl_start: float = 1.0
    kl_band: float = 1.5
    kl_up: float = 1.5
    kl_down: float = 1.5
    kl_min: float = 0.01
    kl_max: float = 10.0
    target_kl: float | None = None
    critic_trunk: str = "detach"
    values: str = "recompute"
    precision: str = "bf16"
    seed: int = 0
    teacher_coef: float = 0.0        # R05: KL(search teacher || policy); 0 is off (module doc)

    def __post_init__(self) -> None:
        problems = []
        if not 0 < self.gamma <= 1:
            problems.append("gamma in (0, 1]")
        if not 0 <= self.lam <= 1:
            problems.append("lam in [0, 1]")
        for name in ("epochs", "minibatch", "chunk"):
            if int(getattr(self, name)) < 1:
                problems.append(f"{name} >= 1")
        if not self.clip > 0:
            problems.append("clip > 0")
        if not self.dual_clip > 1:
            problems.append("dual_clip > 1")
        if self.value_clip is not None and not self.value_clip > 0:
            problems.append("value_clip > 0 (or None)")
        for name in ("lr", "critic_lr", "ent_coef", "vf_coef", "kl_budget", "wait_kl_budget",
                     "kl_start", "wait_kl_start", "teacher_coef"):
            if not float(getattr(self, name)) >= 0:
                problems.append(f"{name} >= 0")
        if not self.max_grad_norm > 0:
            problems.append("max_grad_norm > 0")
        if self.critic_trunk not in CRITIC_TRUNKS:
            problems.append(f"critic_trunk one of {CRITIC_TRUNKS}")
        if self.values not in VALUE_SOURCES:
            problems.append(f"values one of {VALUE_SOURCES}")
        if self.precision not in PRECISIONS:
            problems.append(f"precision one of {PRECISIONS}")
        if problems:
            raise ValueError("PPOConfig: need " + ", ".join(problems))
        AdaptiveKL(self.kl_start, self.kl_budget, self.kl_band, self.kl_up, self.kl_down,
                   self.kl_min, self.kl_max)       # the controller's own checks

    def controller(self, value: float, budget: float) -> AdaptiveKL:
        return AdaptiveKL(value, budget, self.kl_band, self.kl_up, self.kl_down, self.kl_min,
                          self.kl_max)

    def replace(self, **changes: Any) -> "PPOConfig":
        return dataclasses.replace(self, **changes)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PPOState:
    """What carries from one update to the next (blob["ppo"]["state"]): the two KL coefficients
    and the number of updates done."""

    kl: AdaptiveKL
    wait: AdaptiveKL
    updates: int = 0

    @classmethod
    def new(cls, cfg: PPOConfig, kl_coef: float | None = None, wait_coef: float | None = None,
            updates: int = 0) -> "PPOState":
        return cls(cfg.controller(cfg.kl_start if kl_coef is None else kl_coef, cfg.kl_budget),
                   cfg.controller(cfg.wait_kl_start if wait_coef is None else wait_coef,
                                  cfg.wait_kl_budget), int(updates))

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None, cfg: PPOConfig) -> "PPOState":
        data = data or {}
        return cls.new(cfg, data.get("kl_coef"), data.get("wait_coef"),
                       int(data.get("updates", 0) or 0))

    @classmethod
    def from_blob(cls, blob: Mapping[str, Any], cfg: PPOConfig, reset: bool = False
                  ) -> "PPOState":
        """The state a checkpoint carries (a new one without; `reset`: the coefficients start
        over from the config, the update count stays)."""
        rec = blob.get(PPO_KEY) if isinstance(blob.get(PPO_KEY), Mapping) else {}
        data = dict(rec.get("state") or {})
        if reset:
            data.pop("kl_coef", None)
            data.pop("wait_coef", None)
        return cls.from_json(data, cfg)

    def to_json(self) -> dict:
        return {"kl_coef": float(self.kl.value), "wait_coef": float(self.wait.value),
                "updates": int(self.updates)}


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def load_checkpoint(path: Any, device: Any = "cpu") -> tuple[Any, dict]:
    """(TokenPolicy v2 in eval mode on `device`, the checkpoint blob) with a strict state_dict
    load (sim/model_tokens.policy_from_meta, as rs_checkpoint.load_policy builds it). A v2
    model without a critic gets one (add_critic: a copy of value_head). CheckpointError for a
    legacy checkpoint: the rollouts are v2 rows."""
    import torch
    from sim.model_tokens import policy_from_meta
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, Mapping) or "model" not in blob or "meta" not in blob:
        raise CheckpointError(f"{path}: not a token-policy checkpoint (no model / meta)")
    model = policy_from_meta(blob["meta"], blob["model"])
    model.load_state_dict(blob["model"])
    if not model.v2:
        raise CheckpointError(f"{path}: schema {blob['meta'].get('schema')!r}; the update "
                              f"trains {RS.SCHEMA_V2!r} checkpoints (TokenPolicy v2)")
    if not model.has_critic:
        model.add_critic()
    return model.to(device).eval(), dict(blob)


def reference_problems(meta: Mapping[str, Any], ref_meta: Mapping[str, Any]) -> list[str]:
    """Why a reference cannot anchor a model: another schema or token vocabulary."""
    out = []
    for key in ("schema", "n_cards", "n_units", "cards", "units"):
        if meta.get(key) != ref_meta.get(key):
            out.append(f"the reference's {key!r} differs from the model's")
    return out


def param_sets(model: Any) -> tuple[list, list]:
    """(actor parameters, critic parameters): the critic is value2; value_head (the imitation
    model's) is in neither and is never trained."""
    critic = list(model.value2.parameters())
    skip = {id(p) for p in critic} | {id(p) for p in model.value_head.parameters()}
    actor = [p for p in model.parameters() if id(p) not in skip]
    return actor, critic


def make_optimizer(model: Any, cfg: PPOConfig) -> Any:
    """AdamW over the actor (lr) and the critic (critic_lr), no weight decay, eps adam_eps."""
    import torch
    actor, critic = param_sets(model)
    return torch.optim.AdamW([{"params": actor, "lr": cfg.lr},
                              {"params": critic, "lr": cfg.critic_lr}],
                             eps=cfg.adam_eps, weight_decay=0.0)


def _autocast(device: Any, precision: str) -> Any:
    import torch
    dev = torch.device(device)
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                          enabled=dev.type == "cuda" and precision == "bf16")


# ---------------------------------------------------------------------------
# the update
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Prepass:
    """The current model and the reference on every row, before the epochs."""

    value: np.ndarray            # V(s) of the current critic
    logp: np.ndarray             # the joint log-prob of the drawn action, current weights
    q_wait: np.ndarray           # the current policy's q_wait
    ref_gate: np.ndarray         # the reference's gate log q [N, G] (fp32)
    ref_tile: np.ndarray         # the reference's tile log p over the candidates [N, C] (fp32)


def prepass(model: Any, ref: Any, arrays: Mapping[str, np.ndarray], tile_rows: np.ndarray,
            cards: np.ndarray, dist: Sampling, cfg: PPOConfig, device: Any) -> Prepass:
    import torch
    n = int(len(arrays["gate_a"]))
    g = int(arrays["mask"].shape[1])
    c = int(arrays["tile_cands"].shape[1])
    value = np.zeros(n)
    logp = np.zeros(n)
    q_wait = np.zeros(n)
    ref_gate = np.zeros((n, g), dtype=np.float32)
    ref_tile = np.zeros((n, c), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, cfg.chunk):
            idx = np.arange(start, min(n, start + cfg.chunk))
            b = obs_batch(arrays, idx, device)
            act = action_tensors(arrays, idx, device, tile_rows, cards)
            with _autocast(device, cfg.precision):
                out = model.outputs(b, card_ids=act["card"])
                rgate, rtile, _ = ref(b, card_ids=act["card"])
            lq, lpt = log_probs(out["gate"], out["tile"], act, dist)
            rq, rpt = log_probs(rgate, rtile, act, dist)
            value[idx] = critic_value(out["value2"]).double().cpu().numpy()
            logp[idx] = joint_log_prob(lq, lpt, act).double().cpu().numpy()
            q_wait[idx] = lq[:, 0].exp().double().cpu().numpy()
            ref_gate[idx] = rq.cpu().numpy()
            ref_tile[idx] = rpt.cpu().numpy()
    return Prepass(value, logp, q_wait, ref_gate, ref_tile)


class _Sums:
    """Per-epoch running sums (python floats) of weighted statistics."""

    def __init__(self) -> None:
        self.s: dict[str, float] = {}
        self.w: dict[str, float] = {}
        self.max: dict[str, float] = {}

    def add(self, name: str, total: float, weight: float) -> None:
        self.s[name] = self.s.get(name, 0.0) + float(total)
        self.w[name] = self.w.get(name, 0.0) + float(weight)

    def peak(self, name: str, value: float) -> None:
        self.max[name] = max(self.max.get(name, 0.0), float(value))

    def mean(self, name: str) -> float | None:
        w = self.w.get(name, 0.0)
        return self.s[name] / w if w > 0 else None


def ppo_terms(gate: Any, tile: Any, value: Any, mb: Mapping[str, Any], dist: Sampling,
              cfg: PPOConfig) -> dict:
    """The per-minibatch terms (module doc, THE LOSS) from the model's outputs (gate, tile
    logits, V) and the minibatch columns `mb` (action_tensors plus old_logp, adv, ret, v_old,
    ref_gate, ref_tile, learn, choice float weights): {"loss", "pg", "value", "entropy", "kl",
    "kl_wait", "ratio", "logp"} (the last two per row) and their weights."""
    import torch
    lq, lpt = log_probs(gate, tile, mb, dist)
    logp = joint_log_prob(lq, lpt, mb)
    learn, choice = mb["learn"], mb["choice"]
    n_learn = learn.sum().clamp(min=1.0)
    n_choice = choice.sum().clamp(min=1.0)
    log_ratio = torch.where(learn > 0, logp - mb["old_logp"], torch.zeros_like(logp))
    ratio = log_ratio.clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO).exp()
    adv = mb["adv"]
    surr = torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv)
    dual = torch.where(adv < 0, torch.max(surr, cfg.dual_clip * adv), surr)
    pg = -(dual * learn).sum() / n_learn
    tiles = mb["tile_rows"]
    zero = torch.zeros_like(logp)
    ent = masked_entropy(lq, mb["mask"]) + torch.where(
        tiles, masked_entropy(lpt, mb["cmask"]), zero)
    kl_rows = masked_kl(mb["ref_gate"], lq, mb["mask"]) + torch.where(
        tiles, masked_kl(mb["ref_tile"], lpt, mb["cmask"]), zero)
    kl_wait_rows = bernoulli_kl(mb["ref_gate"][:, 0], lq[:, 0])
    entropy = (ent * choice).sum() / n_choice
    kl = (kl_rows * choice).sum() / n_choice
    kl_wait = (kl_wait_rows * choice).sum() / n_choice
    err = (value - mb["ret"]).square()
    if cfg.value_clip:
        pulled = mb["v_old"] + (value - mb["v_old"]).clamp(-cfg.value_clip, cfg.value_clip)
        err = torch.max(err, (pulled - mb["ret"]).square())
    v_loss = err.mean()
    loss = (pg + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
            + float(mb["kl_coef"]) * kl + float(mb["wait_coef"]) * kl_wait)
    teach = {}
    if "teacher_gate" in mb:                  # R05, only with teacher_coef > 0 (module doc)
        from tools.rl import teacher as R05
        kl_teacher, n_teacher = R05.kl_term(lq, lpt, mb)
        loss = loss + float(mb["teacher_coef"]) * kl_teacher
        teach = {"kl_teacher": kl_teacher, "n_teacher": n_teacher}
    return {"loss": loss, "pg": pg, "value": v_loss, "entropy": entropy, "kl": kl,
            "kl_wait": kl_wait, "ratio": ratio, "logp": logp, "n_learn": learn.sum(),
            "n_choice": choice.sum(), "rows": logp.shape[0], **teach}


def _f(x: Any, nd: int = 6) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def update(model: Any, ref: Any, rollout: Rollout, cfg: PPOConfig, state: PPOState, *,
           device: Any = "cpu", optimizer: Any = None, log: Any = None) -> dict:
    """One PPO update of `model` on `rollout` (module doc), anchored to the frozen `ref`, the KL
    coefficients from `state` (moved by what the first epoch measured, and state.updates + 1).
    `optimizer`: make_optimizer(model, cfg) by default (pass one to keep its moments). Returns
    the statistics (JSON-ready)."""
    import torch
    t0 = time.time()
    say = log or (lambda *a, **k: None)
    arrays = with_defaults(rollout.arrays)
    n = int(len(arrays["gate_a"]))
    dist = Sampling.from_meta(rollout.meta)
    if not getattr(model, "has_critic", False):
        model.add_critic()                    # a copy of value_head (load_checkpoint does too)
    model.to(device).eval()
    ref.to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    masks = row_masks(arrays)
    cards = card_ids(arrays["hand"], arrays["gate_a"])
    old_logp = (np.asarray(arrays["logq_gate"], dtype=np.float64)
                + np.where(masks.tile, np.asarray(arrays["logq_tile"], dtype=np.float64), 0.0))
    say(f"ppo: {n:,} rows; prepass (critic, reference) in chunks of {cfg.chunk}")
    pre = prepass(model, ref, arrays, masks.tile, cards, dist, cfg, device)
    values = pre.value
    stored = critic_value(np.asarray(arrays["value"], dtype=np.float64))
    has_stored = np.isfinite(stored)
    if cfg.values == "rollout":
        values = np.where(has_stored, stored, values)
    ret = returns_of(arrays["episode"], arrays["step"], arrays["done"], arrays["result"],
                     arrays["hp_frac"], arrays["crowns"], values, gamma=cfg.gamma, lam=cfg.lam,
                     tower=cfg.phi_tower, crown=cfg.phi_crown)
    valid = ret.valid
    learn = masks.learn & valid
    choice = masks.choice & valid
    tile_rows = masks.tile & valid
    adv = standardise(ret.advantages, learn)
    teach = None
    if cfg.teacher_coef > 0:                  # R05: the search teacher's targets, checked
        from tools.rl import teacher as R05
        teach = R05.ppo_inputs(arrays, choice, tile_rows)
    drift = np.abs(pre.logp - old_logp)[learn]
    vdrift = np.abs(pre.value - stored)[has_stored]
    if optimizer is None:
        optimizer = make_optimizer(model, cfg)
    actor, critic = param_sets(model)
    rows = np.flatnonzero(valid)
    rng = np.random.default_rng([int(cfg.seed), int(state.updates)])
    kl_coef, wait_coef = float(state.kl.value), float(state.wait.value)

    def col(x, idx, dtype=np.float32):
        return torch.from_numpy(np.ascontiguousarray(np.asarray(x)[idx]).astype(dtype)).to(device)

    first: _Sums | None = None
    last = _Sums()
    peak = 0.0
    epochs_run = 0
    stopped = False
    for epoch in range(cfg.epochs):
        sums = _Sums()
        perm = rng.permutation(rows)
        for start in range(0, len(perm), cfg.minibatch):
            idx = np.sort(perm[start:start + cfg.minibatch])
            b = obs_batch(arrays, idx, device)
            mb = action_tensors(arrays, idx, device, tile_rows, cards)
            mb.update(old_logp=col(old_logp, idx), adv=col(adv, idx), ret=col(ret.returns, idx),
                      v_old=col(pre.value, idx), ref_gate=col(pre.ref_gate, idx),
                      ref_tile=col(pre.ref_tile, idx), learn=col(learn, idx),
                      choice=col(choice, idx), kl_coef=kl_coef, wait_coef=wait_coef)
            if teach is not None:
                mb.update(teacher_gate=col(teach.gate, idx), teacher_tile=col(teach.tile, idx),
                          teacher_gate_ok=col(teach.gate_ok, idx, bool),
                          teacher_tile_ok=col(teach.tile_ok, idx, bool),
                          teacher_rows=col(teach.rows, idx), teacher_coef=cfg.teacher_coef)
            with _autocast(device, cfg.precision):
                out = model.outputs(b, card_ids=mb["card"])
                raw = (model.value2(out["cls"].detach()) if cfg.critic_trunk == "detach"
                       else out["value2"])
            terms = ppo_terms(out["gate"].float(), out["tile"].float(), critic_value(raw),
                              mb, dist, cfg)
            optimizer.zero_grad(set_to_none=True)
            terms["loss"].backward()
            gn_a = torch.nn.utils.clip_grad_norm_(actor, cfg.max_grad_norm)
            gn_c = torch.nn.utils.clip_grad_norm_(critic, cfg.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                k = len(idx)
                nl = float(terms["n_learn"])
                nc = float(terms["n_choice"])
                ratio = terms["ratio"]
                lmask = mb["learn"] > 0
                sums.add("pg", float(terms["pg"]) * nl, nl)
                sums.add("value", float(terms["value"]) * k, k)
                sums.add("entropy", float(terms["entropy"]) * nc, nc)
                sums.add("kl", float(terms["kl"]) * nc, nc)
                sums.add("kl_wait", float(terms["kl_wait"]) * nc, nc)
                if "kl_teacher" in terms:
                    nt = float(terms["n_teacher"])
                    sums.add("kl_teacher", float(terms["kl_teacher"]) * nt, nt)
                if nl > 0:
                    r = ratio[lmask]
                    sums.add("approx_kl", float(((r - 1) - r.log()).sum()), nl)
                    sums.add("clip_frac", float(((r - 1).abs() > cfg.clip).float().sum()), nl)
                    sums.peak("ratio_dev", float((r - 1).abs().max()))
                sums.add("grad_actor", float(gn_a), 1)
                sums.add("grad_critic", float(gn_c), 1)
        epochs_run += 1
        peak = max(peak, sums.max.get("ratio_dev", 0.0))
        if first is None:
            first = sums
        last = sums
        akl = sums.mean("approx_kl")
        say(f"  epoch {epoch}: pg {_f(sums.mean('pg'), 4)} value {_f(sums.mean('value'), 4)} "
            f"kl {_f(sums.mean('kl'), 4)} wait {_f(sums.mean('kl_wait'), 5)} "
            f"approx_kl {_f(akl, 5)} clip {_f(sums.mean('clip_frac'), 3)}")
        if cfg.target_kl and akl is not None and akl > cfg.target_kl:
            stopped = True
            break
    first = first or _Sums()
    measured_kl, measured_wait = first.mean("kl"), first.mean("kl_wait")
    state.kl.observe(measured_kl)
    state.wait.observe(measured_wait)
    state.updates += 1
    done = np.asarray(arrays["done"], dtype=bool)
    res = np.asarray(arrays["result"], dtype=np.float64)[done]
    sel_v = valid
    ev = None
    if sel_v.sum() > 1:
        var = float(np.var(ret.returns[sel_v]))
        ev = 1.0 - float(np.var(ret.returns[sel_v] - values[sel_v])) / max(var, 1e-8)
    stats = {
        "rows": n, "episodes": ret.episodes, "episodes_done": int(done.sum()),
        "truncated": ret.truncated, "rows_valid": int(valid.sum()),
        "rows_choice": int(choice.sum()), "rows_pg": int(learn.sum()),
        "rows_tile": int(tile_rows.sum()), "rows_fallback": int((masks.fallback & valid).sum()),
        "wins": int((res > 0).sum()), "losses": int((res < 0).sum()),
        "draws": int((res == 0).sum()), "return_mean": _f(ret.returns[valid].mean())
        if valid.any() else None,
        "adv_raw_mean": _f(ret.advantages[learn].mean()) if learn.any() else None,
        "adv_raw_std": _f(ret.advantages[learn].std()) if learn.any() else None,
        "explained_variance": _f(ev),
        "logp_drift_mean": _f(drift.mean()) if drift.size else None,
        "logp_drift_max": _f(drift.max()) if drift.size else None,
        "value_drift_max": _f(vdrift.max()) if vdrift.size else None,
        "q_play": _f(1 - pre.q_wait[choice].mean()) if choice.any() else None,
        "q_play_ref": _f(1 - np.exp(pre.ref_gate[choice, 0].astype(np.float64)).mean())
        if choice.any() else None,
        "pg": _f(last.mean("pg")), "value_loss": _f(last.mean("value")),
        "entropy": _f(last.mean("entropy")),
        "kl": _f(measured_kl), "kl_wait": _f(measured_wait, 8),
        "kl_last": _f(last.mean("kl")), "kl_wait_last": _f(last.mean("kl_wait"), 8),
        "approx_kl": _f(last.mean("approx_kl"), 8), "clip_frac": _f(last.mean("clip_frac")),
        "ratio_dev_max": _f(peak, 8),
        "grad_norm_actor": _f(last.mean("grad_actor")),
        "grad_norm_critic": _f(last.mean("grad_critic")),
        "kl_coef": _f(kl_coef), "wait_coef": _f(wait_coef),
        "kl_coef_next": _f(state.kl.value), "wait_coef_next": _f(state.wait.value),
        "kl_budget": cfg.kl_budget, "wait_kl_budget": cfg.wait_kl_budget,
        "epochs": epochs_run, "stopped_early": stopped, "updates": state.updates,
        "seconds": round(time.time() - t0, 1),
    }
    if teach is not None:
        stats.update({"teacher_coef": cfg.teacher_coef, "rows_teacher": int(teach.rows.sum()),
                      "rows_teacher_gate": int(teach.gate_ok.sum()),
                      "rows_teacher_tile": int(teach.tile_ok.sum()),
                      "kl_teacher": _f(first.mean("kl_teacher")),
                      "kl_teacher_last": _f(last.mean("kl_teacher"))})
    return stats


def checkpoint_blob(blob: Mapping[str, Any], model: Any, state: PPOState, stats: Mapping,
                    cfg: PPOConfig, files: Sequence[str] = ()) -> dict:
    """The checkpoint an update writes: `blob`'s entries (meta and any stamps) with the new
    weights and blob["ppo"] = {format, state, config, stats, rollouts}; builtins and tensors
    only, so torch.load's weights_only loads it. A meta that said "critic": false is set true
    when the model was given one."""
    out = {k: v for k, v in blob.items() if k not in ("model", PPO_KEY)}
    meta = dict(blob["meta"])
    if meta.get("critic") is False and getattr(model, "has_critic", False):
        meta["critic"] = True
    out["meta"] = meta
    out["model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out[PPO_KEY] = {"format": PPO_FORMAT, "state": state.to_json(), "config": cfg.to_json(),
                    "stats": dict(stats), "rollouts": [str(f) for f in files]}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def summary(rollout: Rollout) -> dict:
    """What --check prints: counts of rows, episodes, results and the kinds of rows."""
    a = rollout.arrays
    m = row_masks(a)
    done = np.asarray(a["done"], dtype=bool)
    res = np.asarray(a["result"], dtype=np.float64)[done]
    eps = np.unique(np.asarray(a["episode"]))
    return {"files": len(rollout.files), "rows": rollout.n, "episodes": int(eps.size),
            "episodes_done": int(done.sum()), "wins": int((res > 0).sum()),
            "losses": int((res < 0).sum()), "draws": int((res == 0).sum()),
            "rows_choice": int(m.choice.sum()), "rows_pg": int(m.learn.sum()),
            "rows_fallback": int(m.fallback.sum()), "buttons": rollout.buttons,
            "wait_bias": rollout.meta.get("wait_bias"), "format": rollout.meta.get("format")}


def build_parser() -> argparse.ArgumentParser:
    d = PPOConfig()
    ap = argparse.ArgumentParser(
        description="One PPO update of a TokenPolicy v2 on RoyaleSim rollouts (R02).")
    ap.add_argument("--data", nargs="+", required=True,
                    help="rollout .npz files, folders of them or globs (expanded here)")
    ap.add_argument("--ckpt", help="the checkpoint to update (its blob['ppo'] state is read)")
    ap.add_argument("--ref", help="the frozen imitation reference (runs/tokens/v2_hog/best.pt)")
    ap.add_argument("--out", help="the checkpoint to write")
    ap.add_argument("--check", action="store_true",
                    help="only validate the rollouts and print what they hold")
    ap.add_argument("--opt-state", default="",
                    help="Adam moments: read when the file exists, written after the update")
    ap.add_argument("--device", default="auto", help="auto (cuda if available), cuda or cpu")
    ap.add_argument("--precision", choices=PRECISIONS, default=d.precision,
                    help="bf16 autocast on CUDA (default) or fp32")
    ap.add_argument("--ram-budget-gb", type=float, default=RAM_BUDGET_GB,
                    help=f"host RAM the update may take (default {RAM_BUDGET_GB:g})")
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--minibatch", type=int, default=d.minibatch)
    ap.add_argument("--chunk", type=int, default=d.chunk, help="rows per prepass forward")
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--critic-lr", type=float, default=d.critic_lr)
    ap.add_argument("--adam-eps", type=float, default=d.adam_eps)
    ap.add_argument("--max-grad-norm", type=float, default=d.max_grad_norm)
    ap.add_argument("--clip", type=float, default=d.clip)
    ap.add_argument("--dual-clip", type=float, default=d.dual_clip)
    ap.add_argument("--value-clip", type=float, default=d.value_clip, help="0: off")
    ap.add_argument("--vf-coef", type=float, default=d.vf_coef)
    ap.add_argument("--ent-coef", type=float, default=d.ent_coef)
    ap.add_argument("--kl-budget", type=float, default=d.kl_budget,
                    help="nats per decision of KL(reference || policy) (plan: 0.05-0.1)")
    ap.add_argument("--kl-start", type=float, default=d.kl_start)
    ap.add_argument("--wait-kl-budget", type=float, default=d.wait_kl_budget)
    ap.add_argument("--wait-kl-start", type=float, default=d.wait_kl_start)
    ap.add_argument("--kl-band", type=float, default=d.kl_band)
    ap.add_argument("--kl-up", type=float, default=d.kl_up)
    ap.add_argument("--kl-down", type=float, default=d.kl_down)
    ap.add_argument("--kl-min", type=float, default=d.kl_min)
    ap.add_argument("--kl-max", type=float, default=d.kl_max)
    ap.add_argument("--reset-kl", action="store_true",
                    help="start the KL coefficients from --kl-start / --wait-kl-start even when "
                         "the checkpoint carries PPO state")
    ap.add_argument("--target-kl", type=float, default=0.0,
                    help="stop the epochs once approx KL(old || new) passes this (0: off)")
    ap.add_argument("--gamma", type=float, default=d.gamma)
    ap.add_argument("--lam", type=float, default=d.lam)
    ap.add_argument("--phi-tower", type=float, default=d.phi_tower)
    ap.add_argument("--phi-crown", type=float, default=d.phi_crown)
    ap.add_argument("--critic-trunk", choices=CRITIC_TRUNKS, default=d.critic_trunk)
    ap.add_argument("--values", choices=VALUE_SOURCES, default=d.values)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--teacher-coef", type=float, default=d.teacher_coef,
                    help="R05: weight of KL(search teacher || policy) on the rows the rollout's "
                         "teacher searched (tools/rl/teacher.py); 0 (default): off")
    return ap


def config_from_args(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        gamma=args.gamma, lam=args.lam, phi_tower=args.phi_tower, phi_crown=args.phi_crown,
        epochs=args.epochs, minibatch=args.minibatch, chunk=args.chunk, lr=args.lr,
        critic_lr=args.critic_lr, adam_eps=args.adam_eps, max_grad_norm=args.max_grad_norm,
        clip=args.clip, dual_clip=args.dual_clip, value_clip=args.value_clip or None,
        vf_coef=args.vf_coef, ent_coef=args.ent_coef, kl_budget=args.kl_budget,
        kl_start=args.kl_start, wait_kl_budget=args.wait_kl_budget,
        wait_kl_start=args.wait_kl_start, kl_band=args.kl_band, kl_up=args.kl_up,
        kl_down=args.kl_down, kl_min=args.kl_min, kl_max=args.kl_max,
        target_kl=args.target_kl or None, critic_trunk=args.critic_trunk, values=args.values,
        precision=args.precision, seed=args.seed, teacher_coef=args.teacher_coef)


def _device(name: str) -> str:
    import torch
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("--device cuda: CUDA is not available (run under .venvs/buildabot)")
    return name


def _err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr, flush=True)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = config_from_args(args)
        files = data_files(args.data)
    except ValueError as exc:
        return _err(str(exc))
    try:
        need = ram_need(files)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        return _err(f"reading the rollout headers: {exc}")
    if need > args.ram_budget_gb * GB:
        return _err(f"the update needs about {need / GB:.2f} GB of RAM for {len(files)} rollout "
                    f"files, over --ram-budget-gb {args.ram_budget_gb:g}. Collect fewer battles "
                    "per iteration, or raise the budget with MuMu and the browser closed.")
    try:
        rollout = load_rollout(files)
    except RolloutError as exc:
        return _err(str(exc))
    if cfg.teacher_coef > 0:                  # R05: refused here (exit 2), not mid-update
        from tools.rl import teacher as R05
        masks = row_masks(rollout.arrays)
        try:
            R05.ppo_inputs(rollout.arrays, masks.choice, masks.tile)
        except ValueError as exc:             # RolloutError, of either import of this module
            return _err(str(exc))
    if args.check:
        print(json.dumps({**summary(rollout), "ram_gb": round(need / GB, 3)}), flush=True)
        return 0
    if not (args.ckpt and args.ref and args.out):
        return _err("--ckpt, --ref and --out are needed (or --check)")
    import torch
    try:
        device = _device(args.device)
        torch.manual_seed(cfg.seed)
        model, blob = load_checkpoint(args.ckpt, device)
        ref, ref_blob = load_checkpoint(args.ref, device)
    except Exception as exc:                  # a bad path, a bad blob, a strict-load mismatch
        return _err(f"loading the checkpoints: {type(exc).__name__}: {exc}")
    problems = reference_problems(blob["meta"], ref_blob["meta"])
    if problems:
        return _err("; ".join(problems))
    state = PPOState.from_blob(blob, cfg, reset=args.reset_kl)
    optimizer = make_optimizer(model, cfg)
    opt_path = Path(args.opt_state) if args.opt_state else None
    if opt_path is not None and opt_path.exists():
        try:
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        except Exception as exc:              # another model's moments, a truncated file
            return _err(f"--opt-state {opt_path}: {type(exc).__name__}: {exc} (delete the file "
                        "to start the Adam moments over)")
        for group, lr in zip(optimizer.param_groups, (cfg.lr, cfg.critic_lr)):
            group["lr"] = lr                      # this call's rates, not the saved ones

    def say(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    say(f"ppo: {Path(args.ckpt).name} on {len(files)} rollout files ({rollout.n:,} rows), "
        f"device {device}, {cfg.precision}; kl coef {state.kl.value:.4g} (budget "
        f"{cfg.kl_budget:g}), wait coef {state.wait.value:.4g} (budget {cfg.wait_kl_budget:g})")
    stats = update(model, ref, rollout, cfg, state, device=device, optimizer=optimizer, log=say)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    torch.save(checkpoint_blob(blob, model, state, stats, cfg, files), tmp)
    os.replace(tmp, out)
    if opt_path is not None:
        opt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(optimizer.state_dict(), opt_path)
    print(json.dumps({**stats, "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
