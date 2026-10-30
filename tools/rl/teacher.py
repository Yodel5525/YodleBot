r"""R05: an optional search teacher for distillation, in RoyaleSim only. OFF BY DEFAULT.

    from tools.rl import teacher as T            # or `import teacher` with tools/rl on sys.path
    teacher = T.Teacher(T.TeacherConfig(rate=0.1), seed=worker_seed)    # one per rollout worker
    # per learner decision d = rs_decide.decide(policy, row, ...) of the sampling preset, after
    # decide and BEFORE the tick's battle.step (both seats decide first, as rs_match does):
    rec.update(teacher.record(battle, team, d, policy, row, dcfg=dcfg, sub=sub,
                              delay=delays[team],
                              row_fn=T.seat_row_fn(seat, team, mem_view, sub, royalesim),
                              value_fn=T.forward_values(brain.forward)))
    ppo.write_rollout(path, records, {**meta, "teacher": teacher.config.to_json()},
                      extra=T.FIELDS)
    .venvs\buildabot\Scripts\python.exe tools\rl\ppo.py ... --teacher-coef 0.3

`policy` and `row` are exactly what decide was given (tools/rl/rollout.py: the request's Heads
and row; its Heads answer one row only, so the after-states go through value_fn, the worker's
batched forward; with an rs_decide.TokenPolicyAdapter value_fn may be left out). `mem_view` is
the memory view the seat decided on. add_args / config_from_args give a collector --teacher and
its knobs.

WHY. Search on the true battle copy beat the plain policy 64.3% to 40.3% (PROGRESS:177-181,
tools/rs_eval.py RSSearchAgent). Live there is no true copy, so the gain has to move into a policy
that reads only the public observation: the teacher searches during the rollouts, and its
choice distribution becomes a KL(teacher || student) term in the PPO loss (tools/rl/ppo.py
--teacher-coef). It is an A/B arm of Phase 7, never the default: the teacher is PRIVILEGED (it
steps the true battle, so the opponent's plays in flight, hidden units and the exact engine
are in its scores, none of which the student sees) and it optimises the simulator's own
dynamics, so it can teach simulator-only tricks. The gate (tools/rs_league_eval.py, under delay
and noise) and the KL anchor to the imitation reference decide whether the arm is kept.

THE SEARCH (annotate), for one decision of rs_decide's sampling preset (a Decision with gate_q):
  candidates  the wait (when gate_q gives it mass); the k_cards hand slots with the most gate_q
              mass, plus the drawn slot when it is a play outside them; per slot its k_tiles best
              tiles by the student's tile logits: for the DRAWN slot of a play (not a fallback)
              the best of the decision's own tile_cands (the legal tiles it drew from), plus the
              tile it drew; for any other slot the first k_tiles of its 60 best tiles
              (rs_decide.TOP_TILES) that check_deploy on the battle accepts; and (presses) every
              ability button the decision's mask lets in. Only entries the decision's mask and
              gate_q allow are candidates, so the target never leaves the student's support.
  trial       on the TRUE battle: one battle.save(), then per candidate battle.load(saved) and
              battle.step([command], delay + horizon) (a wait: step([], ...)). Under the command
              delay D the candidate is queued now and lands at now + D, AFTER our plays already
              in flight, as it would in the match; `horizon` counts the ticks after it lands. The
              opponent makes no new plays in the copy (its plays in flight do land: privileged).
              A command the engine refuses at send is dropped (counted "refused"). After the
              last trial battle.load(saved) leaves the battle as it was (verify=True checks the
              state_json is unchanged and raises otherwise; use it on the first run on the PC).
  score       swing_weight x tower swing (RSSearchAgent's: the enemy's crown-tower hp fractions
              lost minus our own, rs_rows.towers_vec arithmetic) + value_weight x value + prior x
              the student's log-probability of the candidate (log gate_q of its entry, plus for a
              play the log-softmax of its tile over all 576 tile logits, as RSSearchAgent's
              prior). value: the battle's result (+1 / 0 / -1) when it ended inside the
              horizon, else the critic's V = tanh(z / 2) of the after-state's row (row_fn), plus
              (potential) Phi of the after-state: tools/rl/ppo.py trains V on the SHAPED return,
              whose expectation is the result less Phi (the shaping telescopes), so V + Phi is
              the critic's estimate of the result. The discount over the horizon is ignored.
  target      softmax(score / temperature). The GATE target: per entry the BEST score of its
              candidates (RSSearchAgent's argmax over cards x tiles; the drawn slot's extra
              drawn tile left out, so every slot is judged on the same k_tiles), normalised over
              the searched entries A. The best, not a log-sum-exp: that would add up to
              log(k_tiles) to an entry with more trials, so a slot of k equal tiles would
              outvote the single-trial wait (or a button) k : 1 and the target would push the
              play rate up for nothing the search found. keep_unsearched (the
              default): the search only re-ranks what it looked at, so A keeps the student's mass
              q(A) and redistributes it, and every entry it did not search keeps its gate_q
              (a card outside the top k is unknown, not bad); off, A gets all the mass. The TILE
              target (a non-fallback play): the same over the drawn slot's searched tiles, within
              the decision's tile_cands, the rest keeping the student's tile probabilities
              (rs_decide.tile_probs with the DecideConfig's explore and tile_temp). A target is
              written only when it compares something: two or more gate entries (teacher_ok),
              two or more tiles (teacher_tile_ok).

COST. A searched decision runs up to 1 + (k_cards + 1) x k_tiles + buttons trials of delay +
horizon ticks each (defaults with D 24: 8 x 84 = 672 engine ticks, against the 10 the match
advances per decision) plus one batched critic forward. `rate` (default 0.1) is the share of the
learner's drawn decisions searched, drawn from the teacher's own generator: the student's
sampling, the battle and the trajectory are exactly those of a run without the teacher.

THE ROLLOUT FIELDS (FIELDS; written by ppo.write_rollout(..., extra=FIELDS), every row, blank()
where nothing was searched): teacher_ok bool, teacher_tile_ok bool, teacher_gate float32[G]
(log-probabilities over the gate entries, NEG = ppo.NEG where 0), teacher_tile float32[C]
(log-probabilities over the row's tile_cands, C = ppo.CANDS, NEG elsewhere), teacher_n int16
(trials scored).

THE LOSS (kl_rows / kl_term, called by tools/rl/ppo.py when --teacher-coef > 0, ppo_inputs
checks the fields first): per row KL(teacher || student) = the gate's exact KL over the row's
mask (teacher_ok rows) + on a play that ran the tile's over its candidates (teacher_tile_ok
rows): forward KL, as the reference anchor is, in the student's own factorised distribution
(ppo.gate_log_probs, ppo.tile_log_probs on the rollout's masks, cadence and exploration). The
tile term is the drawn slot's only (the chain rule with the slot the student sampled, as ppo's
reference KL), mean over the rows with a target, times teacher_coef (fixed; 0 is off). DAgger
rows (the plan's alternative) are not written.

Pure numpy (torch only inside kl_rows / kl_term and the batched critic); never touches the
emulator; never imports the memory reader. Royale* designs are not imported.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import rs_decide as RD  # noqa: E402
from tools.rl import ppo as P  # noqa: E402

__all__ = [
    "FIELDS",
    "Inputs",
    "Target",
    "Teacher",
    "TeacherConfig",
    "TeacherError",
    "Trial",
    "add_args",
    "battle_delay",
    "blank",
    "config_from_args",
    "forward_values",
    "kl_rows",
    "kl_term",
    "policy_values",
    "ppo_inputs",
    "seat_row_fn",
    "tower_fracs",
    "tower_swing",
]

NEG = P.NEG
#: Below this a teacher log-probability is a zero (NEG pads, exp() of it is exactly 0).
HALF_NEG = NEG / 2
CANDS = P.CANDS
HAND_SIZE = RD.HAND_SIZE
GATE_PLAYS = RD.GATE_PLAYS
#: The rollout entries the teacher adds (name -> dtype), for ppo.write_rollout(extra=FIELDS).
FIELDS: dict[str, str] = {
    "teacher_ok": "bool",
    "teacher_tile_ok": "bool",
    "teacher_gate": "float32",
    "teacher_tile": "float32",
    "teacher_n": "int16",
}
#: |log p - logq_tile| allowed between the teacher's recomputed tile distribution of the drawn
#: tile and the decision's own: more means another row, policy or DecideConfig was passed.
TILE_CHECK = 1e-4
#: ppo_inputs: how far a target's probabilities may sum from 1, and the mass it may put where
#: the student cannot go.
SUM_TOL = 1e-3
OUTSIDE_TOL = 1e-6


class TeacherError(RuntimeError):
    """The teacher cannot search this decision as asked (module doc)."""


# ---------------------------------------------------------------------------
# the configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TeacherConfig:
    """The search (module doc, THE SEARCH). Builtins only: it pickles to spawned workers and
    goes into the rollout meta as to_json()."""

    k_cards: int = 2
    k_tiles: int = 3
    horizon: int = 60                # ticks after the candidate lands (the trial is delay + this)
    temperature: float = 0.1
    swing_weight: float = 1.0
    value_weight: float = 1.0
    prior: float = 0.05
    potential: bool = True           # value = V + Phi (the critic's estimate of the result)
    phi_tower: float = P.PHI_TOWER
    phi_crown: float = P.PHI_CROWN
    keep_unsearched: bool = True
    presses: bool = True             # search the ability buttons the mask lets in
    top_tiles: int = RD.TOP_TILES    # tiles scanned for a legal one (slots other than the drawn)
    rate: float = 0.1                # share of drawn decisions searched
    delay: int | None = None         # the command delay; None: annotate's delay= or the battle's
    verify: bool = False             # check the battle's state_json is unchanged after a search
    seed: int = 0

    def __post_init__(self) -> None:
        problems = []
        for name in ("k_cards", "k_tiles", "top_tiles"):
            if int(getattr(self, name)) < 1:
                problems.append(f"{name} >= 1")
        if int(self.horizon) < 1:
            problems.append("horizon >= 1")
        if not self.temperature > 0:
            problems.append("temperature > 0")
        for name in ("swing_weight", "value_weight", "prior", "phi_tower", "phi_crown"):
            if not math.isfinite(float(getattr(self, name))):
                problems.append(f"{name} finite")
        if not 0 <= self.rate <= 1:
            problems.append("rate in [0, 1]")
        if self.delay is not None and int(self.delay) < 0:
            problems.append("delay >= 0 (or None)")
        if problems:
            raise ValueError("TeacherConfig: need " + ", ".join(problems))

    def replace(self, **changes: Any) -> "TeacherConfig":
        return dataclasses.replace(self, **changes)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "TeacherConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dict(data or {}).items() if k in names})


def add_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The teacher's flags on a collector's parser (R01 / R04): --teacher turns it on; the rest
    default to TeacherConfig's."""
    d = TeacherConfig()
    g = ap.add_argument_group("search teacher (R05, tools/rl/teacher.py; off by default)")
    g.add_argument("--teacher", action="store_true",
                   help="search a share of the learner's decisions on the true battle and write "
                        "the targets for ppo.py --teacher-coef (an A/B arm)")
    g.add_argument("--teacher-rate", type=float, default=d.rate,
                   help=f"share of drawn decisions searched (default {d.rate:g})")
    g.add_argument("--teacher-k-cards", type=int, default=d.k_cards)
    g.add_argument("--teacher-k-tiles", type=int, default=d.k_tiles)
    g.add_argument("--teacher-horizon", type=int, default=d.horizon,
                   help="ticks after the candidate lands (the trial is the delay + this)")
    g.add_argument("--teacher-temperature", type=float, default=d.temperature)
    g.add_argument("--teacher-swing-weight", type=float, default=d.swing_weight)
    g.add_argument("--teacher-value-weight", type=float, default=d.value_weight)
    g.add_argument("--teacher-prior", type=float, default=d.prior)
    g.add_argument("--teacher-no-potential", action="store_true",
                   help="score the critic's V alone, not V + Phi")
    g.add_argument("--teacher-search-all", action="store_true",
                   help="the searched entries take all the mass (default: they keep the "
                        "student's and the unsearched ones keep theirs)")
    g.add_argument("--teacher-verify", action="store_true",
                   help="check after every search that the battle is unchanged")
    g.add_argument("--teacher-seed", type=int, default=d.seed)
    return ap


def config_from_args(args: argparse.Namespace) -> TeacherConfig | None:
    """The TeacherConfig of add_args' flags, or None without --teacher (off)."""
    if not getattr(args, "teacher", False):
        return None
    return TeacherConfig(k_cards=args.teacher_k_cards, k_tiles=args.teacher_k_tiles,
                         horizon=args.teacher_horizon, temperature=args.teacher_temperature,
                         swing_weight=args.teacher_swing_weight,
                         value_weight=args.teacher_value_weight, prior=args.teacher_prior,
                         potential=not args.teacher_no_potential,
                         keep_unsearched=not args.teacher_search_all, rate=args.teacher_rate,
                         verify=args.teacher_verify, seed=args.teacher_seed)


# ---------------------------------------------------------------------------
# helpers: towers, the delay, rows and the critic
# ---------------------------------------------------------------------------

def tower_fracs(state: Mapping[str, Any], team: int) -> np.ndarray:
    """[own x 3, enemy x 3] crown-tower hp / max(1, max hp) (rs_rows.towers_vec's arithmetic;
    the order within a side does not matter to the swing)."""
    out: list[float] = []
    for t in (int(team), 1 - int(team)):
        p = state["players"][t]
        hp, mx = list(p["tower_hp"]), list(p["tower_max_hp"])
        out += [float(hp[k]) / max(1.0, float(mx[k])) for k in range(3)]
    return np.asarray(out, dtype=np.float64)


def tower_swing(before: Mapping[str, Any], after: Mapping[str, Any], team: int) -> float:
    """RSSearchAgent's swing: the enemy's tower fractions lost minus our own, before -> after."""
    b, a = tower_fracs(before, team), tower_fracs(after, team)
    return float((b[3:] - a[3:]).sum() - (b[:3] - a[:3]).sum())


def battle_delay(battle: Any, team: int) -> int | None:
    """`team`'s command delay as the battle reports it (royalesim Battle.command_delay_ticks():
    (blue, red), what make_battle set before the reset), None when it reports none."""
    value = getattr(battle, "command_delay_ticks", None)
    if callable(value):
        value = value()
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    seq = list(value)
    return int(seq[int(team)]) if len(seq) > int(team) else None


def seat_row_fn(seat: Any, team: int, mem: Mapping[str, Any], sub: int,
                royalesim: Any = None) -> Callable[[Mapping[str, Any], int], dict]:
    """row_fn for an rs_eval.RSAgent-like seat (rs_match.MatchAgent): the row of an after-state
    exactly as RSSearchAgent builds its look-ahead rows, rs_rows.obs_row(state, team, sub, mem,
    tick, live=seat.live, visibility=seat.visibility, **seat.v2_kw(state, team, mem, tick,
    royalesim)) with `mem` the decision's memory view. Built on the true after-state: the
    teacher's critic reads the battle, not the screen."""
    import rs_rows as RR

    def row_fn(state: Mapping[str, Any], tick: int) -> dict:
        return RR.obs_row(state, team, sub, mem, int(tick), live=getattr(seat, "live", None),
                          visibility=getattr(seat, "visibility", None),
                          **seat.v2_kw(state, team, mem, int(tick), royalesim))
    return row_fn


def _batched_values(policy: Any, rows: Sequence[Mapping[str, Any]]) -> np.ndarray | None:
    """One forward over `rows` for a TokenPolicyAdapter of a model with a critic (value2), the
    per-row batches concatenated; None for any other policy."""
    model, torch = getattr(policy, "model", None), getattr(policy, "torch", None)
    if (model is None or torch is None or not getattr(model, "has_critic", False)
            or not hasattr(model, "outputs")):
        return None
    parts = [policy.batch(r) for r in rows]
    b = {k: torch.cat([x[k] for x in parts], 0) for k in parts[0]}
    with torch.no_grad():
        out = model.outputs(b, card_ids=b["hand"][:, 0].clamp(min=0))
    z = out["value2"][:, 1].float().cpu().numpy().astype(np.float64)
    return np.tanh(z / 2)


def _won_value(raw: Any) -> float:
    """V = tanh(z / 2) of a critic's raw [outcome, won logit z] (ppo.critic_value)."""
    if raw is None or len(raw) < 2:
        raise TeacherError("the policy has no critic (value2): pass value_fn=, or value_weight 0")
    return math.tanh(float(raw[1]) / 2)


def policy_values(policy: Any, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """V = tanh(z / 2) of the critic's won logit z (ppo.critic_value) on each row: the policy's
    own values(rows) when it has one, one batched forward for a TokenPolicyAdapter with a
    critic, else heads(batch(row)) row by row (rs_decide's protocol). A policy whose batch(row)
    is itself (tools/rl/rollout.py's Heads: one row's outputs) cannot read another row:
    TeacherError, pass value_fn=forward_values(...)."""
    rows = list(rows)
    if not rows:
        return np.zeros(0)
    fn = getattr(policy, "values", None)
    if callable(fn):
        return np.asarray(fn(rows), dtype=np.float64).reshape(-1)
    batched = _batched_values(policy, rows)
    if batched is not None:
        return batched
    heads = getattr(policy, "heads", None)
    if not callable(heads):
        raise TeacherError("the policy has no heads(batch): the teacher reads the critic "
                           "through rs_decide.TokenPolicyAdapter (or pass value_fn=)")
    out = []
    for r in rows:
        b = policy.batch(r)
        if b is policy:
            raise TeacherError("this policy answers one row only (its batch(row) is itself, as "
                               "rollout.Heads): pass value_fn=teacher.forward_values(forward)")
        out.append(_won_value(heads(b)[1]))
    return np.asarray(out, dtype=np.float64)


def forward_values(forward: Callable[[list], Sequence[Any]]) -> Callable[[Sequence], np.ndarray]:
    """value_fn from a batched forward giving one object with heads(b) -> (gate, the critic's
    raw value) per row (tools/rl/rollout.py Brain.forward and its Heads): V = tanh(z / 2)."""
    def value_fn(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray([_won_value(h.heads(None)[1]) for h in forward(list(rows))],
                          dtype=np.float64)
    return value_fn


def _reason(row: Any) -> int:
    """The reason of a Battle.step result entry ((card_id, reason, ...); the fake's [team,
    reason]); 0 when it says nothing (rs_match._reason)."""
    try:
        return int(row[1])
    except (TypeError, IndexError, ValueError):
        return 0


def _log(p: np.ndarray) -> np.ndarray:
    """float32 log-probabilities, NEG where p is 0."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, NEG, dtype=np.float32)
    pos = p > 0
    out[pos] = np.log(p[pos]).astype(np.float32)
    return out


def _lse(z: Sequence[float]) -> float:
    z = np.asarray(z, dtype=np.float64)
    m = float(z.max())
    return m + math.log(float(np.exp(z - m).sum()))


def _softmax(z: Sequence[float]) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Trial:
    """One candidate: its gate entry (0 wait, 1 + slot, GATE_PLAYS + button), the tile of a
    play (own frame, -1 otherwise), its index in the decision's tile_cands (the drawn slot's
    tiles, else -1), whether it counts in the gate target, the student's log-probability, the
    command, and what the trial found."""

    entry: int
    tile: int = -1
    cand: int = -1
    in_gate: bool = True
    prior: float = 0.0
    cmd: tuple | None = None
    ran: bool = False
    swing: float = 0.0
    value: float = 0.0
    terminal: bool = False
    score: float = 0.0


@dataclasses.dataclass
class Target:
    """What one search found: the gate target (probabilities over the gate entries, the
    behaviour's gate_q when gate_ok is False), the tile target over the decision's tile_cands
    ([] without a play), whether each is a real comparison, and the trials in the order run."""

    gate: np.ndarray
    tile: np.ndarray
    gate_ok: bool
    tile_ok: bool
    trials: tuple
    refused: int = 0

    @property
    def scored(self) -> list[Trial]:
        return [t for t in self.trials if t.ran]

    @property
    def best(self) -> Trial | None:
        """The trial with the highest score (the first of equals), None when none ran."""
        ran = self.scored
        return max(ran, key=lambda t: t.score) if ran else None

    def record(self, cands: int = CANDS) -> dict:
        """The rollout fields (FIELDS) of this target, the tile padded to `cands`."""
        tile = np.full(int(cands), NEG, dtype=np.float32)
        if self.tile_ok:
            if len(self.tile) > cands:
                raise ValueError(f"{len(self.tile)} tile candidates; the record holds {cands}")
            tile[:len(self.tile)] = _log(self.tile)
        gate = _log(self.gate) if self.gate_ok else np.full(len(self.gate), NEG, np.float32)
        return {"teacher_ok": bool(self.gate_ok), "teacher_tile_ok": bool(self.tile_ok),
                "teacher_gate": gate, "teacher_tile": tile,
                "teacher_n": np.int16(len(self.scored))}


def blank(gate_width: int, cands: int = CANDS) -> dict:
    """The rollout fields of a row the teacher did not search."""
    return {"teacher_ok": False, "teacher_tile_ok": False,
            "teacher_gate": np.full(int(gate_width), NEG, dtype=np.float32),
            "teacher_tile": np.full(int(cands), NEG, dtype=np.float32),
            "teacher_n": np.int16(0)}


class Teacher:
    """The search teacher of one rollout worker (module doc). `seed` (default config.seed)
    seeds its own generator, which only picks the decisions searched (rate)."""

    def __init__(self, config: TeacherConfig | None = None, *, seed: int | None = None):
        self.config = TeacherConfig() if config is None else config
        self.rng = np.random.default_rng(self.config.seed if seed is None else int(seed))
        self.stats: Counter = Counter()

    # ------------------------------------------------------------ which rows
    def wants(self, decision: Any) -> bool:
        """Whether to search this decision: one draw of the teacher's generator per drawn
        decision (gate_a >= 0), searched with probability `rate`; a row without a draw (nothing
        playable) is never searched and draws nothing."""
        if int(decision.gate_a) < 0:
            return False
        return bool(self.rng.random() < self.config.rate)

    def record(self, battle: Any, team: int, decision: Any, policy: Any,
               row: Mapping[str, Any], **kw: Any) -> dict:
        """The rollout fields for this decision: annotate's target when wants() says so and the
        search compared something, else blank(). Keywords as annotate's."""
        width = len(decision.mask)
        if not self.wants(decision):
            self.stats["skipped"] += 1
            return blank(width)
        target = self.annotate(battle, team, decision, policy, row, **kw)
        if target is None or not (target.gate_ok or target.tile_ok):
            self.stats["no_target"] += 1
            return blank(width)
        return target.record()

    # ------------------------------------------------------------ the search
    def annotate(self, battle: Any, team: int, decision: Any, policy: Any,
                 row: Mapping[str, Any], *, row_fn: Callable | None = None,
                 value_fn: Callable | None = None, sub: int = RD.SUBTILE_PER_MILLITILE,
                 delay: int | None = None, dcfg: Any = None) -> Target | None:
        """Search `decision` (module doc, THE SEARCH) on `battle`, the true battle at the
        decision's tick before anything of that tick is stepped, and leave it as it was.

        decision  an rs_decide.Decision of the sampling preset (gate_q, mask, gate_a,
                  tile_cands, tile_a, logq_tile)
        policy    the policy decide was given (batch, tiles; heads or values for the critic)
        row       the row decide was given (the student's view, with its noise)
        row_fn    row_fn(state, tick) -> the row of an after-state (seat_row_fn); needed when
                  value_weight is not 0
        value_fn  value_fn(rows) -> V per row (default policy_values(policy, rows))
        delay     our command delay (default config.delay, else battle.command_delay_ticks())
        dcfg      the DecideConfig decide used (its explore / tile_temp; default sampling())
        Returns None for a decision without a draw or with fewer than two candidates."""
        cfg = self.config
        d = decision
        team = int(team)
        if int(d.gate_a) < 0:
            return None
        q = np.asarray(d.gate_q, dtype=np.float64)
        mask = np.asarray(d.mask, dtype=bool)
        if q.size == 0 or q.size != mask.size:
            raise TeacherError("annotate searches decisions of rs_decide's sampling preset (a "
                               "gate_q as wide as the mask)")
        ticks = delay if delay is not None else cfg.delay
        if ticks is None:
            ticks = battle_delay(battle, team)
        if ticks is None:
            raise TeacherError("the command delay is unknown: pass delay= (the engine "
                               "config's delay[team]) or set TeacherConfig.delay")
        steps = int(ticks) + int(cfg.horizon)
        need_value = cfg.value_weight != 0
        if need_value and row_fn is None:
            raise TeacherError("value_weight is not 0: pass row_fn= (seat_row_fn) so the "
                               "critic can read the after-states")
        dcfg = RD.DecideConfig.sampling() if dcfg is None else dcfg
        before_json = battle.state_json()
        before = json.loads(before_json)
        trials, drawn = self._candidates(battle, before, team, d, q, mask, policy, row, sub,
                                         dcfg)
        if len(trials) < 2:
            self.stats["too_few"] += 1
            return None
        self.stats["searched"] += 1
        rows: list[Any] = []
        waiting: list[Trial] = []
        refused = 0
        saved = battle.save()
        try:
            for tr in trials:
                battle.load(saved)
                cmds = [] if tr.cmd is None else [tr.cmd]
                res = list(battle.step(cmds, steps) or ())
                if tr.cmd is not None and res and _reason(res[0]) != 0:
                    refused += 1
                    continue
                after = json.loads(battle.state_json())
                tr.ran = True
                tr.swing = tower_swing(before, after, team)
                tr.terminal = bool(after.get("game_over"))
                if tr.terminal:
                    tr.value = P.result_of(after, team)
                elif need_value:
                    tr.value = (float(P.potential(*P.reward_parts(after, team), cfg.phi_tower,
                                                  cfg.phi_crown)) if cfg.potential else 0.0)
                    rows.append(row_fn(after, int(after.get("tick", before["tick"] + steps))))
                    waiting.append(tr)
        finally:
            battle.load(saved)
        if cfg.verify and battle.state_json() != before_json:
            raise TeacherError("the battle is not what it was after the search: its save() / "
                               "load() does not restore it")
        if rows:
            values = (value_fn(rows) if value_fn is not None else policy_values(policy, rows))
            values = np.asarray(values, dtype=np.float64).reshape(-1)
            if values.size != len(waiting):
                raise TeacherError(f"value_fn gave {values.size} values for {len(waiting)} rows")
            for tr, v in zip(waiting, values):
                tr.value += float(v)
        for tr in trials:
            if tr.ran:
                tr.score = (cfg.swing_weight * tr.swing + cfg.value_weight * tr.value
                            + cfg.prior * tr.prior)
        self.stats["trials"] += sum(t.ran for t in trials)
        self.stats["refused"] += refused
        self.stats["terminal"] += sum(t.ran and t.terminal for t in trials)
        target = self._target(trials, q, drawn, refused)
        self.stats["gate_targets"] += int(target.gate_ok)
        self.stats["tile_targets"] += int(target.tile_ok)
        return target

    def _candidates(self, battle: Any, state: Mapping[str, Any], team: int, d: Any,
                    q: np.ndarray, mask: np.ndarray, policy: Any, row: Mapping[str, Any],
                    sub: int, dcfg: Any) -> tuple[list[Trial], Any]:
        """(the trials to run in order: the wait, the slots by gate_q mass, the buttons; the
        drawn slot's (tile_cands, the student's probabilities over them) or None)."""
        cfg = self.config
        out: list[Trial] = []
        if mask[0] and q[0] > 0:
            out.append(Trial(entry=0, prior=math.log(q[0])))
        play = d.kind == RD.PLAY and not d.fallback and 1 <= int(d.gate_a) < GATE_PLAYS
        drawn_slot = int(d.gate_a) - 1 if play else -1
        open_slots = [s for s in range(HAND_SIZE)
                      if 1 + s < q.size and mask[1 + s] and q[1 + s] > 0]
        slots = sorted(open_slots, key=lambda s: (-q[1 + s], s))[:cfg.k_cards]
        if drawn_slot >= 0 and drawn_slot not in slots:
            slots.append(drawn_slot)
        hand = list(state["players"][team].get("hand") or [])
        b = policy.batch(row)
        drawn = None
        check = None
        for slot in slots:
            logits = np.asarray(policy.tiles(b, slot), dtype=np.float64).reshape(-1)
            logz = _lse(logits)
            base = math.log(q[1 + slot])
            tc = [int(t) for t in d.tile_cands] if slot == drawn_slot else []
            if tc:
                st = RD.tile_probs(logits[tc], dcfg.explore, dcfg.tile_temp)
                k = int(d.tile_a)
                if 0 <= k < len(tc) and abs(math.log(max(float(st[k]), 1e-300))
                                            - float(d.logq_tile)) > TILE_CHECK:
                    raise TeacherError("the student's tile distribution is not the decision's: "
                                       "pass the row, policy and DecideConfig decide was given")
                top = sorted(range(len(tc)), key=lambda j: (-logits[tc[j]], j))[:cfg.k_tiles]
                picks = [(j, True) for j in top]
                if 0 <= k < len(tc) and k not in top:
                    picks.append((k, False))
                for j, in_gate in picks:
                    t = tc[j]
                    out.append(Trial(entry=1 + slot, tile=t, cand=j, in_gate=in_gate,
                                     prior=base + float(logits[t]) - logz,
                                     cmd=_play_cmd(team, slot, t, sub)))
                drawn = (tc, st)
                continue
            if check is None:
                check = RD.BattleLegality(battle).bind(state, team)
            card = int(hand[slot]) if slot < len(hand) and hand[slot] is not None else -1
            found = 0
            for t in np.argsort(-logits, kind="stable")[:cfg.top_tiles]:
                cmd = _play_cmd(team, slot, int(t), sub)
                if not check(slot, card, cmd[2], cmd[3]):
                    continue
                out.append(Trial(entry=1 + slot, tile=int(t), prior=base + float(logits[t]) - logz,
                                 cmd=cmd))
                found += 1
                if found >= cfg.k_tiles:
                    break
        if cfg.presses:
            for k in range(q.size - GATE_PLAYS):
                e = GATE_PLAYS + k
                if mask[e] and q[e] > 0:
                    out.append(Trial(entry=e, prior=math.log(q[e]),
                                     cmd=(team, HAND_SIZE + k, 0, 0)))
        return out, drawn

    def _target(self, trials: Sequence[Trial], q: np.ndarray, drawn: Any,
                refused: int) -> Target:
        cfg = self.config
        temp = float(cfg.temperature)
        ran = [t for t in trials if t.ran]
        by_entry: dict[int, list[float]] = {}
        for t in ran:
            if t.in_gate:
                by_entry.setdefault(t.entry, []).append(t.score / temp)
        entries = sorted(by_entry)
        gate = q.copy()
        gate_ok = len(entries) >= 2
        if gate_ok:
            # The entry's best trial (module doc, target): count-invariant, so the wait's one
            # trial competes with a slot's k_tiles on equal terms.
            share = _softmax([max(by_entry[e]) for e in entries])
            if cfg.keep_unsearched:
                gate[entries] = float(q[entries].sum()) * share
            else:
                gate = np.zeros_like(q)
                gate[entries] = share
            gate = gate / gate.sum()
        tile = np.zeros(0)
        tile_ok = False
        if drawn is not None:
            tc, st = drawn
            mine = [t for t in ran if t.cand >= 0]
            if len(mine) >= 2:
                share = _softmax([t.score / temp for t in mine])
                js = [t.cand for t in mine]
                if cfg.keep_unsearched:
                    tile = np.asarray(st, dtype=np.float64).copy()
                    tile[js] = float(tile[js].sum()) * share
                else:
                    tile = np.zeros(len(tc))
                    tile[js] = share
                tile = tile / tile.sum()
                tile_ok = True
        return Target(gate, tile, gate_ok, tile_ok, tuple(trials), refused)


def _play_cmd(team: int, slot: int, tile: int, sub: int) -> tuple[int, int, int, int]:
    """(team, slot, x_sub, y_sub) of a play at the centre of model tile `tile`, as decide taps."""
    wx, wy = RD.tile_centre_world(tile, team)
    return (int(team), int(slot), int(wx * sub), int(wy * sub))


# ---------------------------------------------------------------------------
# the loss (tools/rl/ppo.py --teacher-coef)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Inputs:
    """ppo_inputs' result: the targets as the update reads them, every array per rollout row."""

    gate: np.ndarray            # float32 [N, G] log-probabilities
    tile: np.ndarray            # float32 [N, C] log-probabilities over tile_cands
    gate_ok: np.ndarray         # bool [N]: the gate term counts
    tile_ok: np.ndarray         # bool [N]: the tile term counts (a play that ran)

    @property
    def rows(self) -> np.ndarray:
        """The rows with a target (either term): the KL's mean runs over them."""
        return self.gate_ok | self.tile_ok


def ppo_inputs(arrays: Mapping[str, np.ndarray], choice: np.ndarray,
               tile_rows: np.ndarray) -> Inputs:
    """The teacher's fields of a loaded rollout, checked (P.RolloutError naming the problem):
    present, the gate target as wide as the mask and inside it (none on a capped row's wait),
    the tile target inside the row's candidates, each summing to 1. `choice` and `tile_rows` are
    the update's (drawn rows and plays that ran, both valid): a target elsewhere is ignored."""
    missing = [k for k in FIELDS if k not in arrays]
    if missing:
        raise P.RolloutError(f"--teacher-coef needs the search teacher's fields, missing "
                             f"{', '.join(missing)} (collect with the teacher on: "
                             f"tools/rl/teacher.py, written with extra=teacher.FIELDS)")
    mask = np.asarray(arrays["mask"], dtype=bool)
    n, g = mask.shape
    gate = np.asarray(arrays["teacher_gate"], dtype=np.float32)
    if gate.shape != (n, g):
        raise P.RolloutError(f"teacher_gate has shape {gate.shape}, expected {(n, g)}")
    tc = np.asarray(arrays["tile_cands"])
    c = tc.shape[1]
    tile = np.asarray(arrays["teacher_tile"], dtype=np.float32)
    if tile.ndim != 2 or tile.shape[0] != n or tile.shape[1] > c:
        raise P.RolloutError(f"teacher_tile has shape {tile.shape}, expected [{n}, <= {c}]")
    if tile.shape[1] < c:
        tile = np.pad(tile, ((0, 0), (0, c - tile.shape[1])), constant_values=NEG)
    gate_ok = np.asarray(arrays["teacher_ok"], dtype=bool).reshape(-1) & np.asarray(choice,
                                                                                     dtype=bool)
    tile_ok = (np.asarray(arrays["teacher_tile_ok"], dtype=bool).reshape(-1)
               & np.asarray(tile_rows, dtype=bool))
    probs = []

    def bad(cond: np.ndarray, what: str) -> None:
        k = int(np.count_nonzero(cond))
        if k:
            probs.append(f"{k} rows {what} (first: row {int(np.flatnonzero(cond)[0])})")

    pg = np.exp(gate)
    bad(gate_ok & ~np.isfinite(gate).all(1), "have a non-finite teacher_gate")
    bad(gate_ok & ((pg * ~mask).sum(1) > OUTSIDE_TOL),
        "have teacher gate mass outside their mask")
    capped = np.asarray(arrays.get("capped", np.zeros(n, dtype=bool)), dtype=bool).reshape(-1)
    bad(gate_ok & capped & (pg[:, 0] > OUTSIDE_TOL), "are capped and the teacher waits")
    bad(gate_ok & (np.abs((pg * mask).sum(1) - 1) > SUM_TOL),
        "have a teacher gate not summing to 1")
    cmask = tc >= 0
    pt = np.exp(tile)
    bad(tile_ok & ~np.isfinite(tile).all(1), "have a non-finite teacher_tile")
    bad(tile_ok & ((pt * ~cmask).sum(1) > OUTSIDE_TOL),
        "have teacher tile mass outside their candidates")
    bad(tile_ok & (np.abs((pt * cmask).sum(1) - 1) > SUM_TOL),
        "have a teacher tile not summing to 1")
    if probs:
        raise P.RolloutError("the teacher's targets: " + "; ".join(probs))
    return Inputs(gate, tile, gate_ok, tile_ok)


def kl_rows(lq: Any, lpt: Any, t_gate: Any, t_tile: Any, gate_ok: Any, tile_ok: Any,
            mask: Any, cmask: Any) -> Any:
    """[N]: KL(teacher || student) per row (module doc, THE LOSS): over the masked-in gate
    entries the teacher gives mass to, on gate_ok rows, plus over the candidates the teacher
    gives mass to, on tile_ok rows. lq [N, G] and lpt [N, C] are the student's log q / log p
    (ppo.log_probs), t_gate / t_tile the teacher's log-probabilities (NEG pads). fp32."""
    import torch
    lq, lpt = lq.float(), lpt.float()
    t_gate, t_tile = t_gate.float(), t_tile.float()
    on_g = mask & (t_gate > HALF_NEG) & gate_ok[:, None]
    zg = torch.zeros_like(lq)
    kg = torch.where(on_g, t_gate.exp() * (t_gate - lq), zg).sum(1)
    on_t = cmask & (t_tile > HALF_NEG) & tile_ok[:, None]
    zt = torch.zeros_like(lpt)
    kt = torch.where(on_t, t_tile.exp() * (t_tile - lpt), zt).sum(1)
    return kg + kt


def kl_term(lq: Any, lpt: Any, mb: Mapping[str, Any]) -> tuple[Any, Any]:
    """(the mean KL(teacher || student) over the minibatch's rows with a target, their number)
    from a ppo minibatch `mb` (mask, cmask, teacher_gate, teacher_tile, teacher_gate_ok,
    teacher_tile_ok bool, teacher_rows float weights)."""
    rows = kl_rows(lq, lpt, mb["teacher_gate"], mb["teacher_tile"], mb["teacher_gate_ok"],
                   mb["teacher_tile_ok"], mb["mask"], mb["cmask"])
    w = mb["teacher_rows"]
    n = w.sum()
    return (rows * w).sum() / n.clamp(min=1.0), n
