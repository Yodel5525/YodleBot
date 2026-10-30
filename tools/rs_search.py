r"""Search on a rebuilt board (docs/LIVE_SEARCH_PLAN.md). This part: BatchedPolicy (S05).

    from tools import rs_search as SR             # or `import rs_search` (tools/ on sys.path)
    policy = SR.BatchedPolicy(loaded.model)       # rs_checkpoint.load_policy(...).model, eval mode
    d0 = rs_decide.decide(policy, row, state, team, legality, cfg, rng=rng)   # as with the adapter
    enc = policy.encode_rows(rows)                # many rows, one encoder pass
    gate, value = enc.head(2)                     # TokenPolicyAdapter.heads of row 2
    logits = enc.tiles_for([(0, 1), (0, 3)])      # tile logits of (row, hand slot) pairs, one call
    v = policy.values(after_rows, policy.value_kind)    # the search's V per after-row, one pass

WHY (plan 6.1, "The shared encoder pass"). In TokenPolicy._heads (sim/model_tokens.py:251-270)
the gate, the value head and the critic read only the encoder's output, and the tile logits are
tile_head(cat[cls, card_emb(card_ids)]) (:266): encode_all (:195-236) never reads card_ids, so
`cls` is the same for every card. One outputs() call (:246-249, the dict with "cls") therefore
gives the gate, the values and cls of a whole batch, and the 576 tile logits of any (row, card)
cost only the small tile head. rs_decide.TokenPolicyAdapter (rs_decide.py:612-664) runs a full
forward for heads() and another for each tiles(b, slot) the decision tries (rs_decide.py:1285,
1354); the old search ran one per candidate (rs_eval.py:703-704). The model is in eval mode
(rs_checkpoint.load_policy, rs_checkpoint.py:322), so dropout is off and a row's outputs do not
depend on the pass it is in. This module uses sim/model_tokens read-only: it calls outputs(),
tile_head and card_emb and changes nothing on the model.

BatchedPolicy(model, torch=None, *, batch=None, device=None, value_kind=None)
  TokenPolicyAdapter's interface, so rs_decide.decide takes it in place of the adapter:
  batch(row)         the adapter's batch of one row: `batch` (RSAgent.batch) when given, else
                     rs_decide.row_batch(row, torch, model_schema_of(model)); on the model's
                     device
  heads(b)           (gate logits of row 0 [wait, slot 0..3, buttons...] numpy float32, the
                     critic's raw [outcome, won logit] or None): exactly the adapter's, from the
                     batch's one encoder pass
  gate(b)            heads(b)[0]
  tiles(b, slot)     the 576 tile logits of row 0 for the card in hand slot `slot` (card_ids =
                     b["hand"][:, slot], as the adapter has it), from the cached pass of `b`
  schema, has_critic as the adapter's
  and for many rows:
  stack(rows)        one batch of the rows' batch(row)s, concatenated (rows of one batch share
                     their widths: v2 rows the same ability-button count B)
  encode(b)          the Encoded pass of a batch, cached by the batch's identity (the last
                     CACHE_BATCHES batches): heads(b) then tiles(b, slot) run the encoder once
  encode_rows(rows)  encode(stack(rows)), not cached (the caller holds the Encoded)
  values(rows, kind) V per row, float64 [N], one encoder pass (below; kind None: the critic's,
                     tools/rl/teacher.policy_values' protocol)
  stats              Counter: passes, rows, cache_hits, tile_calls, tile_rows
  clear()            drops the cache

A BATCH IS READ AS IMMUTABLE. The cache holds the batch dict itself and compares by identity
(`is`), so a dict that is freed and a new one at the same address never match; a batch changed in
place after its pass would keep the old pass. rs_decide.decide builds a batch and never changes it
(rs_decide.py:1195). A model in training mode is refused (BatchedPolicyError): with dropout on, a
row's outputs would depend on the pass and the shared pass would not be the adapter's.

Encoded (one pass over N rows; numpy float32 unless said)
  n, len()           the rows
  gate [N, G], value [N, 2], value2 [N, 2] or None (a model with a critic), cls (the encoder's
  [N, d] tensor, on the model's device), batch (the stacked tensors)
  head(i)            the adapter's heads() of row i: (gate[i], tuple(value2[i]) or None)
  values(kind)       V per row (below; kind None: the policy's value_kind, the search's V)
  tiles(i, slot)     [576] for the card in row i's hand slot `slot`
  tiles_for(pairs)   [M, 576] for (row, slot) pairs, one tile-head call
  tiles_cards(rows, tokens)   [M, 576] for (row, card token) pairs, one tile-head call; a token
                     is a card id + 2 (rs_rows), what a row's hand column holds

VALUES (plan 3.1 step 9: score = swing + V + ...). kind:
  "swing"   the value head's column 0 (sim/model_tokens.py:149, 267): the predicted tower swing
            over the next HORIZON ticks (rs_rows.py:292, the "outcome" label), in tower_swing's
            units (tools/rl/teacher.py:282). A legacy model's V.
  "critic"  tanh(z / 2) of the critic's won logit z = value2[:, 1] (ppo.critic_value,
            tools/rl/ppo.py:887-893): exactly what tools/rl/teacher.policy_values reads
            (teacher.py:318-366), which calls a policy's own values(rows) first; so this
            BatchedPolicy gives the teacher the same V as its batched TokenPolicyAdapter path.
  None      BatchedPolicy.values(rows): "critic", the teacher's protocol, unless the constructor
            was given a value_kind; a model without a critic then raises BatchedPolicyError, as
            the teacher's adapter path raises TeacherError, instead of handing the teacher the
            value head's swing (another unit) as if it were tanh(z / 2). Encoded.values() and
            value_kind: the search's V (plan 3.1 step 9), "critic" for a model with a critic,
            else "swing"; the search asks values(rows, policy.value_kind) or
            encode_rows(rows).values(). ("critic" without a critic is refused.)
The potential Phi, the terminal result (ppo.result_of, scaled for the value head) and the swing
are the scorer's (S06), not the policy's.

DEVICE. device None: the device of the model's first parameter (CPU when it has none). On the CPU
the batches are the adapter's tensors, unmoved; on another device every batch moves there and the
outputs come back to the CPU as numpy. The adapter itself is CPU only.

Importing this module imports numpy and rs_decide (torch-free), never torch, sim.model_tokens or
royalesim: BatchedPolicy imports torch when it is built, as TokenPolicyAdapter does. Nothing
here reads a battle, taps or touches the emulator.
"""
from __future__ import annotations

import importlib
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:          # also when imported as tools.rs_search
    sys.path.insert(0, str(ROOT / "tools"))

import rs_decide as RD  # noqa: E402  (row_batch, model_schema_of, TILES; numpy only)

__all__ = [
    "BatchedPolicy",
    "BatchedPolicyError",
    "CACHE_BATCHES",
    "Encoded",
    "VALUE_CRITIC",
    "VALUE_KINDS",
    "VALUE_SWING",
    "as_batched",
]

#: values(): V from the value head's column 0 (the predicted tower swing; a legacy model).
VALUE_SWING = "swing"
#: values(): V = tanh(z / 2) of the critic's won logit (ppo.critic_value; a model with value2).
VALUE_CRITIC = "critic"
VALUE_KINDS = (VALUE_SWING, VALUE_CRITIC)
#: Batches whose encoder pass encode() keeps, by identity, most recent first.
CACHE_BATCHES = 4
#: What BatchedPolicy needs of a model (sim/model_tokens.TokenPolicy has them all).
MODEL_PARTS = ("outputs", "tile_head", "card_emb")


class BatchedPolicyError(ValueError):
    """A model, a batch or a value kind BatchedPolicy cannot serve."""


def _model_device(model: Any) -> Any:
    """The device of the model's first parameter, None when it has none."""
    params = getattr(model, "parameters", None)
    if not callable(params):
        return None
    for p in params():
        return p.device
    return None


# ---------------------------------------------------------------------------
# one encoder pass
# ---------------------------------------------------------------------------

class Encoded:
    """One encoder pass over N rows (module doc, Encoded)."""

    __slots__ = ("policy", "batch", "n", "gate", "value", "value2", "cls")

    def __init__(self, policy: "BatchedPolicy", batch: Mapping[str, Any],
                 out: Mapping[str, Any]):
        self.policy = policy
        self.batch = batch
        self.cls = out["cls"]
        self.gate = policy._np(out["gate"])
        self.value = policy._np(out["value"])
        v2 = out.get("value2") if policy.has_critic else None     # the adapter's rule
        self.value2 = None if v2 is None else policy._np(v2)
        self.n = int(self.gate.shape[0])

    def __len__(self) -> int:
        return self.n

    def _row(self, index: int) -> int:
        i = int(index)
        if not 0 <= i < self.n:
            raise IndexError(f"row {index} of a pass over {self.n} rows")
        return i

    def head(self, index: int = 0) -> tuple[np.ndarray, tuple[float, ...] | None]:
        """TokenPolicyAdapter.heads of row `index`: (its gate logits, the critic's raw
        [outcome, won logit] or None without a critic)."""
        i = self._row(index)
        value = None
        if self.value2 is not None:
            value = tuple(float(v) for v in self.value2[i].reshape(-1))
        return self.gate[i].copy(), value

    def values(self, kind: str | None = None) -> np.ndarray:
        """V per row, float64 [N] (module doc, VALUES)."""
        kind = self.policy.value_kind if kind is None else kind
        if kind == VALUE_SWING:
            return self.value[:, 0].astype(np.float64)
        if kind == VALUE_CRITIC:
            if self.value2 is None:
                raise BatchedPolicyError("values: kind 'critic' needs a model with a critic "
                                         "(value2); this one has none: use 'swing'")
            return np.tanh(self.value2[:, 1].astype(np.float64) / 2)
        raise BatchedPolicyError(f"values: kind one of {VALUE_KINDS} or None, got {kind!r}")

    def tiles(self, index: int, slot: int) -> np.ndarray:
        """The 576 tile logits of row `index` for the card in its hand slot `slot`."""
        i = self._row(index)
        return self._tile_head([i], self.batch["hand"][i:i + 1, int(slot)])[0]

    def tiles_for(self, pairs: Iterable[Sequence[int]]) -> np.ndarray:
        """[M, 576] tile logits of (row, hand slot) pairs, in order, one tile-head call."""
        pairs = [(self._row(i), int(s)) for i, s in pairs]
        if not pairs:
            return np.zeros((0, RD.TILES), dtype=np.float32)
        torch = self.policy.torch
        dev = self.batch["hand"].device
        rows = torch.as_tensor([i for i, _ in pairs], dtype=torch.long, device=dev)
        slots = torch.as_tensor([s for _, s in pairs], dtype=torch.long, device=dev)
        return self._tile_head([i for i, _ in pairs], self.batch["hand"][rows, slots])

    def tiles_cards(self, rows: Iterable[int], tokens: Iterable[int]) -> np.ndarray:
        """[M, 576] tile logits of (row, card token) pairs, one tile-head call."""
        rows = [self._row(i) for i in rows]
        tokens = [int(t) for t in tokens]
        if len(rows) != len(tokens):
            raise BatchedPolicyError(f"tiles_cards: {len(rows)} rows for {len(tokens)} tokens")
        if not rows:
            return np.zeros((0, RD.TILES), dtype=np.float32)
        torch = self.policy.torch
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.batch["hand"].device)
        return self._tile_head(rows, ids)

    def _tile_head(self, rows: list[int], card_ids: Any) -> np.ndarray:
        """tile_head(cat[cls, card_emb(card_ids)]) for cls of `rows` (model_tokens.py:266)."""
        p = self.policy
        torch, model = p.torch, p.model
        if rows == list(range(self.n)):
            cls = self.cls
        else:
            cls = self.cls[torch.as_tensor(rows, dtype=torch.long, device=self.cls.device)]
        with torch.no_grad():
            tile = model.tile_head(torch.cat([cls, model.card_emb(card_ids)], -1))
        p.stats["tile_calls"] += 1
        p.stats["tile_rows"] += len(rows)
        return p._np(tile)


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------

class BatchedPolicy:
    """TokenPolicyAdapter's interface with one encoder pass per batch, plus values(rows)
    (module doc)."""

    def __init__(self, model: Any, torch: Any = None, *,
                 batch: Callable[[Mapping[str, Any]], dict] | None = None,
                 device: Any = None, value_kind: str | None = None,
                 cache: int = CACHE_BATCHES):
        missing = [k for k in MODEL_PARTS if not hasattr(model, k)]
        if missing:
            raise BatchedPolicyError(f"BatchedPolicy needs a TokenPolicy (sim/model_tokens.py: "
                                     f"{', '.join(MODEL_PARTS)}); {type(model).__name__} has no "
                                     f"{', '.join(missing)}: use rs_decide.TokenPolicyAdapter")
        if torch is None:
            torch = importlib.import_module("torch")
        self.model = model
        self.torch = torch
        self._batch = batch
        self.device = _model_device(model) if device is None else torch.device(device)
        self._move = self.device is not None and getattr(self.device, "type", "cpu") != "cpu"
        if value_kind not in (None, *VALUE_KINDS):
            raise BatchedPolicyError(f"value_kind one of {VALUE_KINDS} or None, got "
                                     f"{value_kind!r}")
        if value_kind == VALUE_CRITIC and not self.has_critic:
            raise BatchedPolicyError("value_kind 'critic' needs a model with a critic (value2)")
        self._value_kind_given = value_kind
        self.value_kind = value_kind or (VALUE_CRITIC if self.has_critic else VALUE_SWING)
        self.cache_size = max(0, int(cache))
        self._cache: list[tuple[Mapping[str, Any], Encoded]] = []
        self.stats: Counter = Counter()

    # ------------------------------------------------------------ the adapter's properties
    @property
    def schema(self) -> Any:
        """The model's v2 Schema, or None for a legacy model (TokenPolicyAdapter.schema)."""
        return RD.model_schema_of(self.model)

    @property
    def has_critic(self) -> bool:
        return bool(getattr(self.model, "has_critic", False)) and hasattr(self.model, "outputs")

    # ------------------------------------------------------------ batches
    def _cpu_batch(self, row: Mapping[str, Any]) -> dict:
        if self._batch is not None:
            return self._batch(row)
        return RD.row_batch(row, self.torch, self.schema)

    def _to_device(self, b: dict) -> dict:
        if not self._move:
            return b
        return {k: v.to(self.device) for k, v in b.items()}

    def _np(self, t: Any) -> np.ndarray:
        t = t.float()
        if self._move:
            t = t.cpu()
        return t.numpy()

    def batch(self, row: Mapping[str, Any]) -> dict:
        """The batch of one row, as TokenPolicyAdapter.batch builds it (on the model's device)."""
        return self._to_device(self._cpu_batch(row))

    def stack(self, rows: Iterable[Mapping[str, Any]]) -> dict:
        """One batch of `rows`: their batch(row)s concatenated along the first axis."""
        parts = [self._cpu_batch(r) for r in rows]
        if not parts:
            raise BatchedPolicyError("stack: no rows")
        if len(parts) == 1:
            return self._to_device(parts[0])
        keys = list(parts[0])
        for k, p in enumerate(parts[1:], 1):
            if list(p) != keys:
                raise BatchedPolicyError(f"stack: row {k} has the columns {list(p)}, row 0 "
                                         f"{keys} (rows of one schema only)")
        for name in keys:
            widths = {tuple(p[name].shape[1:]) for p in parts}
            if len(widths) > 1:
                raise BatchedPolicyError(f"stack: the rows differ in the width of {name} "
                                         f"{sorted(widths)} (v2 rows of one batch need the same "
                                         f"ability-button count)")
        torch = self.torch
        return self._to_device({name: torch.cat([p[name] for p in parts], 0) for name in keys})

    # ------------------------------------------------------------ the pass
    def _encode(self, b: Mapping[str, Any]) -> Encoded:
        if getattr(self.model, "training", False):
            raise BatchedPolicyError("the model is in training mode (dropout on): call "
                                     "model.eval() (rs_checkpoint.load_policy gives it so)")
        card_ids = b["hand"][:, 0].clamp(min=0)          # the adapter's heads() call
        with self.torch.no_grad():
            out = self.model.outputs(b, card_ids=card_ids)
        enc = Encoded(self, b, out)
        self.stats["passes"] += 1
        self.stats["rows"] += enc.n
        return enc

    def encode(self, b: Mapping[str, Any]) -> Encoded:
        """The Encoded pass of batch `b`, from the cache when `b` is a batch it holds."""
        for held, enc in self._cache:
            if held is b:
                self.stats["cache_hits"] += 1
                return enc
        enc = self._encode(b)
        if self.cache_size:
            self._cache.insert(0, (b, enc))
            del self._cache[self.cache_size:]
        return enc

    def encode_rows(self, rows: Iterable[Mapping[str, Any]]) -> Encoded:
        """One encoder pass over `rows` (stack), not cached."""
        return self._encode(self.stack(rows))

    def clear(self) -> None:
        """Drop the cached passes."""
        self._cache.clear()

    # ------------------------------------------------------------ the adapter's calls
    def heads(self, b: Mapping[str, Any]) -> tuple[np.ndarray, tuple[float, ...] | None]:
        """(the gate logits of row 0, the critic's raw [outcome, won logit] or None)."""
        return self.encode(b).head(0)

    def gate(self, b: Mapping[str, Any]) -> np.ndarray:
        return self.heads(b)[0]

    def tiles(self, b: Mapping[str, Any], slot: int) -> np.ndarray:
        """The tile logits of row 0 for the card in hand slot `slot`, without a new pass."""
        return self.encode(b).tiles(0, slot)

    # ------------------------------------------------------------ values
    def values(self, rows: Iterable[Mapping[str, Any]], kind: str | None = None) -> np.ndarray:
        """V per row, float64 [N], from one encoder pass (module doc, VALUES); [] -> [0]. kind
        None: the constructor's value_kind when it was given one, else "critic" (what
        tools/rl/teacher.policy_values expects of values(rows); BatchedPolicyError without a
        critic). The search's own V: values(rows, self.value_kind)."""
        rows = list(rows)
        if kind is None:
            kind = self._value_kind_given or VALUE_CRITIC
            if kind == VALUE_CRITIC and not self.has_critic:
                raise BatchedPolicyError(
                    "values(rows): the default kind is 'critic' (tanh(z / 2), what "
                    "tools/rl/teacher.policy_values reads) and this model has no critic: pass "
                    "kind='swing' (or value_kind='swing') for the value head's tower swing")
        if not rows:
            return np.zeros(0, dtype=np.float64)
        return self.encode_rows(rows).values(kind)


def as_batched(policy: Any, torch: Any = None, **kw: Any) -> Any:
    """A BatchedPolicy for `policy`: itself when it is one; for a TokenPolicyAdapter (anything
    with a model, torch, batch, gate and tiles) one of its model, torch and batch function; for
    a TokenPolicy one of it. A policy whose model lacks MODEL_PARTS, and any other policy (a
    test's scripted batch / gate / tiles), is returned unchanged."""
    if isinstance(policy, BatchedPolicy):
        return policy
    model = getattr(policy, "model", None)
    if (model is not None and all(hasattr(policy, k) for k in ("batch", "gate", "tiles"))
            and hasattr(policy, "torch")):
        if not all(hasattr(model, k) for k in MODEL_PARTS):
            return policy
        return BatchedPolicy(model, policy.torch if torch is None else torch,
                             batch=getattr(policy, "_batch", None), **kw)
    if all(hasattr(policy, k) for k in MODEL_PARTS):
        return BatchedPolicy(policy, torch, **kw)
    return policy
