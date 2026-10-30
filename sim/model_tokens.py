"""Entity-token policy: what is on the board, what is in hand, what the
opponent has shown -> wait or play (which hand card, which tile), plus value.

Inputs are the columns `sim.replay_rows` writes. Every board object is a token
(unit id embedding plus its features), so the network sees *what* each unit
is, not only where something is - the plane observation in `sim.env` counted
units per cell and could not tell a Skeleton from a Golem.

Heads:
* gate + card: a pointer over the four hand cards plus a wait logit.
* tile: 18 x 32 logits conditioned on the chosen card's embedding.
* value: the 10 s tower outcome the row was labelled with, and the win.

SCHEMAS (E15; tools/rs_schema.py). TokenPolicy(..., schema=) builds the token groups of a row
schema (rs_schema.Schema.groups). `schema` None / "legacy" is rs_rows/legacy, the model above,
module for module and parameter for parameter (a legacy checkpoint loads with a strict
state_dict, and the same seed gives the same weights and outputs). "v2" (rs_rows/v2, tools/
rs_rows.py --schema v2) adds, after every legacy module:
  state       state_in's input is the schema's state_features (towers + scalars: 6 + 7)
  units       unit_in's input is the schema's unit_features (9: the 7, form, burrow)
  hand        + hand_in(hand_feats): deck form, evolution ready, evolution charge
  spells      a token per cast spell in flight: card_emb(spell_ids) + spell_in(spell_feats)
  pending     a token per own play tapped and not run: card_emb(own_pending)
              + pending_in(own_pending_land)
  buttons     a token per ability button: card_emb(ability_card) + button_in(ability_state)
and one group embedding each for the three new groups (group_emb; the six legacy token types
keep type_emb). Id 0 pads, as everywhere. The gate becomes [wait, 4 hand slots, B buttons]:
the button logits are a second pointer (press_query(cls) against the button tokens), and a
button whose card is 0 (no button) or whose "available" feature is 0 is masked to MASK_LOGIT
(-1e4), as an empty hand slot is. B is read from the batch (ability_card's width), so the
weights do not depend on it. A press is the gate target GATE_PLAYS + button (target_slot).

CRITIC (R01 / R02). With critic (default: a v2 model) the model also holds value2, a second
value head built as a copy of value_head. Imitation does not train it: train_tokens copies
value_head into it before each checkpoint (sync_critic), so an RL run starts from a critic
"initialised from value_head" and trains value2 alone, while value_head stays the imitation
model's. forward() returns (gate, tile, value) for every model; outputs() also returns
"value2". add_critic() gives a legacy model one (its checkpoint then holds value2.* keys,
which policy_from_meta reads back).

LOSSES. losses(model, b) of a legacy model is the function it always was. A v2 model's adds the
press CE where the rows are labelled (presses=True: tools/rs_rows.py --presses on): the gate CE
runs over [wait, slots, buttons] with press rows (act_press = button + 1) as button targets,
so the rows where a button was available and not pressed teach "not now". presses=False (rows
built with presses off: no row presses, so nothing says when to press) keeps the button logits
out of the CE and drops any press row from it. A press row whose button is masked has no gate
target (-1), like a played card that is not in the recorded hand.

Batches: row_batch(row, schema) / rows_batch(rows, schema) turn obs_row dicts into the
tensors forward() reads (int columns long, float columns float32), the way RSAgent.batch and
sim/train_tokens.batches feed the model. policy_from_meta(meta) builds the model a checkpoint
meta describes (meta "schema": rs_rows/v2 is v2; anything else, absent included, is legacy).
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_TOOLS = str(Path(__file__).resolve().parents[1] / "tools")
if _TOOLS not in sys.path:          # rs_schema is pure Python; the tools import it by this name
    sys.path.append(_TOOLS)
import rs_schema as RS  # noqa: E402

GRID_W, GRID_H = 18, 32
TILES = GRID_W * GRID_H
HAND_SIZE = RS.HAND_SIZE
#: Gate entries [wait, slot 0..3]; a v2 gate's ability buttons follow them.
GATE_PLAYS = 1 + HAND_SIZE
#: The logit of an empty hand slot and of a masked button.
MASK_LOGIT = -1e4
#: The v2 token groups after the legacy ones, in sequence order (group_emb rows).
EXTRA_GROUPS = ("spells", "pending", "buttons")
#: Where "available" sits in a button's ability_state (rs_schema.ABILITY_FEATS).
AVAILABLE = RS.ABILITY_FEATS.index("available")
#: load_matching: weights whose input features grow from legacy to v2 with the legacy ones
#: first (unit_feats: the 7, then form and burrow; scalars: the 4, then triple, opp_elixir,
#: opp_exact), so a legacy weight fills the first columns and the new ones start at 0.
WIDENABLE = ("state_in.weight", "unit_in.weight")


def _as_schema(value=None) -> RS.Schema:
    """rs_schema.get(value), also for a Schema of another import of rs_schema (by its name)."""
    if value is not None and not isinstance(value, str) and hasattr(value, "name"):
        value = value.name
    return RS.get(value)


def model_schema(meta: Mapping[str, Any] | None) -> RS.Schema:
    """The schema of the model a checkpoint meta (or a vocab.json) describes: rs_schema.V2 when
    its "schema" is rs_rows/v2 (or "v2"), else rs_schema.LEGACY: absent (every checkpoint
    before E14), rs_rows/legacy, or any other name (rows from before rs_schema, a test's).
    The strict state_dict load is what checks the rest."""
    name = (meta or {}).get("schema")
    if isinstance(name, str) and name.strip() in (RS.SCHEMA_V2, "v2"):
        return RS.V2
    return RS.LEGACY


def meta_critic(meta: Mapping[str, Any] | None, state_dict: Mapping[str, Any] | None = None
                ) -> bool:
    """Whether the checkpoint's model has a critic (value2): meta "critic" when it says,
    else whether `state_dict` holds value2.* weights, else whether it is a v2 model."""
    meta = meta or {}
    if meta.get("critic") is not None:
        return bool(meta["critic"])
    if state_dict is not None:
        return any(str(k).startswith("value2.") for k in state_dict)
    return model_schema(meta).v2


class TokenPolicy(nn.Module):
    def __init__(self, n_cards: int, n_units: int, d: int = 192, layers: int = 4,
                 heads: int = 6, unit_features: int | None = None, *, schema=None,
                 critic: bool | None = None):
        super().__init__()
        spec = _as_schema(schema)
        self.schema = spec
        groups = spec.groups
        self.has_hand_feats = "hand_feats" in groups.get("hand", ())
        self.has_spells = "spells" in groups
        self.has_pending = "pending" in groups
        self.has_buttons = "buttons" in groups
        uf = spec.unit_features if unit_features is None else int(unit_features)
        self.card_emb = nn.Embedding(n_cards, d, padding_idx=0)
        self.unit_emb = nn.Embedding(n_units, d, padding_idx=0)
        self.unit_in = nn.Linear(uf, d)
        self.state_in = nn.Linear(spec.state_features, d)   # towers + scalars (legacy 6 + 4)
        self.type_emb = nn.Embedding(6, d)             # cls, unit, state, hand, next, opp
        self.slot_emb = nn.Embedding(4, d)
        self.age_in = nn.Linear(1, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.wait_head = nn.Linear(d, 1)
        self.card_query = nn.Linear(d, d)
        self.tile_head = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(),
                                       nn.Linear(2 * d, TILES))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        # E15, v2 only, after every legacy module: a legacy model's parameters (and the
        # random draws that initialise them) are exactly what they were.
        if self.has_hand_feats:
            self.hand_in = nn.Linear(len(RS.HAND_FEATS), d)
        if self.has_spells:
            self.spell_in = nn.Linear(len(RS.SPELL_FEATS), d)
        if self.has_pending:
            self.pending_in = nn.Linear(1, d)
        if self.has_buttons:
            self.button_in = nn.Linear(len(RS.ABILITY_FEATS), d)
            self.press_query = nn.Linear(d, d)
        if self.has_spells or self.has_pending or self.has_buttons:
            self.group_emb = nn.Embedding(len(EXTRA_GROUPS), d)
        self.has_critic = spec.v2 if critic is None else bool(critic)
        if self.has_critic:
            self.value2 = copy.deepcopy(self.value_head)

    @property
    def v2(self) -> bool:
        return self.schema.v2

    @property
    def input_columns(self) -> tuple:
        """The row columns forward() reads (the schema's model_inputs)."""
        return self.schema.model_inputs

    # ------------------------------------------------------------ critic
    @torch.no_grad()
    def sync_critic(self) -> None:
        """value2 := value_head (no-op without a critic)."""
        if self.has_critic:
            self.value2.load_state_dict(self.value_head.state_dict())

    def add_critic(self) -> None:
        """Give the model a critic, a copy of value_head, when it has none."""
        if not self.has_critic:
            self.value2 = copy.deepcopy(self.value_head)
            self.has_critic = True

    # ------------------------------------------------------------ encoder
    def encode(self, b):
        """(cls [N, d], the four hand slots [N, 4, d]) after the encoder."""
        cls, hand_h, _ = self.encode_all(b)
        return cls, hand_h

    def encode_all(self, b):
        """(cls, hand slots, button tokens [N, B, d] or None without the buttons group)."""
        n = b["unit_ids"].shape[0]
        dev = b["unit_ids"].device
        t = lambda i: self.type_emb.weight[i]
        units = (self.unit_emb(b["unit_ids"]) + self.unit_in(b["unit_feats"].float())
                 + t(1))
        state = self.state_in(torch.cat([b["towers"], b["scalars"]], -1))[:, None] + t(2)
        hand = (self.card_emb(b["hand"]) + self.slot_emb.weight[None] + t(3))
        if self.has_hand_feats:
            hand = hand + self.hand_in(b["hand_feats"].float())
        nxt = self.card_emb(b["next"])[:, None] + t(4)
        seen = self.card_emb(b["opp_seen"]) + t(5)
        last = (self.card_emb(b["opp_last"]) + self.age_in(b["opp_last_age"].float()[..., None])
                + t(5))
        cls = self.cls.expand(n, -1, -1) + t(0)
        parts = [cls, state, hand, nxt, units, seen, last]
        pads = [torch.zeros(n, 2, dtype=torch.bool, device=dev),
                b["hand"] == 0, b["next"][:, None] == 0, b["unit_ids"] == 0,
                b["opp_seen"] == 0, b["opp_last"] == 0]
        g = lambda name: self.group_emb.weight[EXTRA_GROUPS.index(name)]
        if self.has_spells:
            parts.append(self.card_emb(b["spell_ids"]) + self.spell_in(b["spell_feats"].float())
                         + g("spells"))
            pads.append(b["spell_ids"] == 0)
        if self.has_pending:
            parts.append(self.card_emb(b["own_pending"])
                         + self.pending_in(b["own_pending_land"].float()[..., None])
                         + g("pending"))
            pads.append(b["own_pending"] == 0)
        n_buttons = 0
        if self.has_buttons:
            n_buttons = b["ability_card"].shape[1]
            parts.append(self.card_emb(b["ability_card"])
                         + self.button_in(b["ability_state"].float()) + g("buttons"))
            pads.append(b["ability_card"] == 0)
        x = torch.cat(parts, 1)
        pad = torch.cat(pads, 1)
        pad[:, 0] = False
        h = self.norm(self.encoder(x, src_key_padding_mask=pad))
        buttons = h[:, h.shape[1] - n_buttons:] if self.has_buttons else None
        return h[:, 0], h[:, 2:6], buttons

    # ------------------------------------------------------------ heads
    def forward(self, b, card_ids=None):
        """Returns gate logits [N, 5] (wait + 4 hand slots; v2: [N, 5 + B], the ability
        buttons after them), tile logits [N, TILES] for `card_ids` (default: the slot with the
        highest logit), and value [N, 2]."""
        out = self._heads(b, card_ids, critic=False)
        return out["gate"], out["tile"], out["value"]

    def outputs(self, b, card_ids=None) -> dict:
        """forward()'s gate, tile and value as a dict, plus "cls" (the encoder's [N, d]) and,
        with a critic, "value2" [N, 2] (RL)."""
        return self._heads(b, card_ids, critic=self.has_critic)

    def _heads(self, b, card_ids, critic: bool) -> dict:
        cls, hand_h, button_h = self.encode_all(b)
        q = self.card_query(cls)
        slot_logits = torch.einsum("nd,nkd->nk", q, hand_h) / hand_h.shape[-1] ** 0.5
        slot_logits = slot_logits.masked_fill(b["hand"] == 0, MASK_LOGIT)
        parts = [self.wait_head(cls), slot_logits]
        if button_h is not None:
            pq = self.press_query(cls)
            press_logits = torch.einsum("nd,nkd->nk", pq, button_h) / button_h.shape[-1] ** 0.5
            press_logits = press_logits.masked_fill(~button_available(b), MASK_LOGIT)
            parts.append(press_logits)
        gate = torch.cat(parts, 1)
        if card_ids is None:
            slot = slot_logits.argmax(1)
            card_ids = b["hand"].gather(1, slot[:, None])[:, 0]
        tile = self.tile_head(torch.cat([cls, self.card_emb(card_ids)], -1))
        out = {"gate": gate, "tile": tile, "value": self.value_head(cls), "cls": cls}
        if critic:
            out["value2"] = self.value2(cls)
        return out


def is_v2(model) -> bool:
    """Whether `model` is a v2 TokenPolicy (anything else gets the legacy losses)."""
    return bool(getattr(getattr(model, "schema", None), "v2", False))


def policy_from_meta(meta: Mapping[str, Any], state_dict: Mapping[str, Any] | None = None
                     ) -> TokenPolicy:
    """The TokenPolicy a checkpoint meta describes: TokenPolicy(n_cards, n_units, d, layers,
    heads) with the meta's sizes (defaults 192 / 4 / 6), of model_schema(meta), with a critic
    when meta_critic says so. A legacy meta builds exactly the model it always did."""
    spec = model_schema(meta)
    kw: dict = {}
    if spec.v2:
        kw["schema"] = spec.name
    critic = meta_critic(meta, state_dict)
    if critic != spec.v2:
        kw["critic"] = critic
    return TokenPolicy(meta["n_cards"], meta["n_units"], d=meta.get("d", 192),
                       layers=meta.get("layers", 4), heads=meta.get("heads", 6), **kw)


def button_available(b) -> torch.Tensor:
    """bool [N, B]: a button that exists (ability_card > 0) and is available now (its
    ability_state "available" > 0.5: a unit behind it, off cooldown, not mid-ability)."""
    return (b["ability_card"] > 0) & (b["ability_state"][..., AVAILABLE].float() > 0.5)


def target_slot(hand: torch.Tensor, card: torch.Tensor, press: torch.Tensor | None = None,
                available: torch.Tensor | None = None) -> torch.Tensor:
    """Gate target: 0 for wait, 1 + hand slot of the played card, -1 if the
    played card is not in the recorded hand (row skipped for the gate).

    E15: with `press` (act_press: 0, or button + 1) a press row's target is GATE_PLAYS + button
    (1 + hand slots + button), or -1 when the button is not one of `available`'s B columns or
    is masked there (button_available); `available` None takes every button as available."""
    match = (hand == card[:, None]) & (card[:, None] > 0)
    slot = match.float().argmax(1) + 1
    has = match.any(1)
    out = torch.where(card == 0, torch.zeros_like(slot), slot)
    out = torch.where((card > 0) & ~has, torch.full_like(slot, -1), out)
    if press is None:
        return out
    press = press.long()
    k = press - 1
    is_press = press > 0
    button = 1 + hand.shape[1] + k
    if available is None:
        ok = torch.ones_like(is_press)
    elif available.shape[1] == 0:
        ok = torch.zeros_like(is_press)
    else:
        idx = k.clamp(0, available.shape[1] - 1)
        ok = (k < available.shape[1]) & (available.long().gather(1, idx[:, None])[:, 0] > 0)
    return torch.where(is_press, torch.where(ok, button, torch.full_like(button, -1)), out)


def gate_targets(b, presses: bool = True) -> torch.Tensor:
    """A batch's gate targets: target_slot of hand and act_card, and with an act_press column
    the press rows: GATE_PLAYS + button (presses and the batch has buttons), or -1 (no press
    labels: the row has no gate target)."""
    card = b["act_card"].long()
    hand = b["hand"].long()
    if "act_press" not in b:
        return target_slot(hand, card)
    press = b["act_press"].long()
    if presses and "ability_card" in b:
        return target_slot(hand, card, press, button_available(b))
    out = target_slot(hand, card)
    return torch.where(press > 0, torch.full_like(out, -1), out)


def losses(model, b, beta: float = 0.0, max_weight: float = 5.0, presses: bool = True):
    """Imitation losses, optionally outcome-weighted.

    beta == 0: plain imitation. beta > 0: each row's weight is
    exp(advantage / beta) clipped to `max_weight`, where the advantage is the
    row's 10 s tower outcome minus the value head's prediction for that state
    (advantage-weighted regression). The value head learns the outcome either
    way, so the two runs differ only in the weights.

    A v2 model (is_v2) also learns the presses where `presses` says the rows are labelled
    (module doc, LOSSES); a legacy model ignores `presses`.
    """
    if is_v2(model):
        return _losses_v2(model, b, beta, max_weight, presses)
    card = b["act_card"].long()
    gate_t = target_slot(b["hand"], card)
    gate, tile, value = model(b, card_ids=card.clamp(min=0))
    v_out = value[:, 0]
    v_won = value[:, 1]
    outcome = b["outcome"].float()
    if beta > 0:
        adv = (outcome - v_out.detach())
        w = torch.exp(adv / beta).clamp(max=max_weight)
    else:
        w = torch.ones_like(outcome)
    ok = gate_t >= 0
    gate_loss = (F.cross_entropy(gate[ok], gate_t[ok], reduction="none") * w[ok]).sum() / w[ok].sum().clamp(min=1e-6)
    plays = card > 0
    tile_t = (b["act_y"].long() * GRID_W + b["act_x"].long())
    tile_loss = ((F.cross_entropy(tile[plays], tile_t[plays], reduction="none") * w[plays]).sum()
                 / w[plays].sum().clamp(min=1e-6)) if plays.any() else gate_loss * 0
    value_loss = F.mse_loss(v_out, outcome) + F.binary_cross_entropy_with_logits(v_won, b["won"].float())
    with torch.no_grad():
        stats = {
            "gate_acc": (gate[ok].argmax(1) == gate_t[ok]).float().mean().item() if ok.any() else 0.0,
            "card_acc": ((gate[ok & plays][:, 1:].argmax(1) + 1) == gate_t[ok & plays]).float().mean().item()
            if (ok & plays).any() else 0.0,
            "tile_acc": (tile[plays].argmax(1) == tile_t[plays]).float().mean().item() if plays.any() else 0.0,
            "tile_near": _near(tile[plays].argmax(1), tile_t[plays]) if plays.any() else 0.0,
            "mean_w": w.mean().item(),
        }
    return gate_loss + tile_loss + 0.5 * value_loss, stats


def _losses_v2(model, b, beta: float, max_weight: float, presses: bool):
    """losses() of a v2 model: the legacy terms, the gate over the buttons too when the rows
    carry press labels, and "press_acc" (gate argmax right on the press rows with a target)."""
    card = b["act_card"].long()
    press = b["act_press"].long() if "act_press" in b else torch.zeros_like(card)
    with_press = bool(presses) and model.has_buttons and "ability_card" in b
    gate_t = gate_targets(b, with_press)
    gate, tile, value = model(b, card_ids=card.clamp(min=0))
    if not with_press:
        gate = gate[:, :GATE_PLAYS]
    v_out = value[:, 0]
    v_won = value[:, 1]
    outcome = b["outcome"].float()
    if beta > 0:
        adv = (outcome - v_out.detach())
        w = torch.exp(adv / beta).clamp(max=max_weight)
    else:
        w = torch.ones_like(outcome)
    ok = gate_t >= 0
    gate_loss = ((F.cross_entropy(gate[ok], gate_t[ok], reduction="none") * w[ok]).sum()
                 / w[ok].sum().clamp(min=1e-6))
    plays = card > 0
    pressed = ok & (press > 0)
    tile_t = (b["act_y"].long() * GRID_W + b["act_x"].long())
    tile_loss = ((F.cross_entropy(tile[plays], tile_t[plays], reduction="none") * w[plays]).sum()
                 / w[plays].sum().clamp(min=1e-6)) if plays.any() else gate_loss * 0
    value_loss = (F.mse_loss(v_out, outcome)
                  + F.binary_cross_entropy_with_logits(v_won, b["won"].float()))
    with torch.no_grad():
        arg = gate.argmax(1)
        stats = {
            "gate_acc": (arg[ok] == gate_t[ok]).float().mean().item() if ok.any() else 0.0,
            "card_acc": ((gate[ok & plays][:, 1:GATE_PLAYS].argmax(1) + 1)
                         == gate_t[ok & plays]).float().mean().item()
            if (ok & plays).any() else 0.0,
            "tile_acc": (tile[plays].argmax(1) == tile_t[plays]).float().mean().item()
            if plays.any() else 0.0,
            "tile_near": _near(tile[plays].argmax(1), tile_t[plays]) if plays.any() else 0.0,
            "mean_w": w.mean().item(),
            "press_acc": (arg[pressed] == gate_t[pressed]).float().mean().item()
            if pressed.any() else 0.0,
        }
    return gate_loss + tile_loss + 0.5 * value_loss, stats


def _near(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Share of predicted tiles within one tile (Chebyshev) of the pro's."""
    px, py = pred % GRID_W, pred // GRID_W
    tx, ty = true % GRID_W, true // GRID_W
    return (((px - tx).abs() <= 1) & ((py - ty).abs() <= 1)).float().mean().item()


# ---------------------------------------------------------------------------
# rows -> tensors, and warm starts
# ---------------------------------------------------------------------------

def rows_batch(rows: Iterable[Mapping[str, Any]], schema=None, torch_module: Any = None) -> dict:
    """obs_row dicts (tools/rs_rows.py) of `schema` stacked into the batch forward() reads:
    each of the schema's model_inputs, int columns as long, float columns as float32, scalar
    columns (next) as [N]. `torch_module` stands in for torch (its as_tensor and the tensors'
    long() / float() are all that is used)."""
    t = torch if torch_module is None else torch_module
    spec = _as_schema(schema)
    rows = list(rows)
    out = {}
    for name in spec.model_inputs:
        col = spec.column(name)
        if not col.shape:
            arr = np.asarray([int(r[name]) for r in rows], dtype=np.int64)
        else:
            arr = np.stack([np.asarray(r[name]) for r in rows])
        x = t.as_tensor(arr)
        out[name] = x.long() if col.dtype.startswith(("int", "bool")) else x.float()
    return out


def row_batch(row: Mapping[str, Any], schema=None, torch_module: Any = None) -> dict:
    """rows_batch of one row: a batch of one."""
    return rows_batch([row], schema, torch_module)


@torch.no_grad()
def load_matching(model: nn.Module, state: Mapping[str, torch.Tensor]) -> dict:
    """Warm start (sim/train_tokens.py --init-partial): copy every tensor of `state` whose name
    and shape are `model`'s; a WIDENABLE weight with more input columns in `model` gets the old
    ones first and zeros after. A critic missing from `state` is synced from value_head.
    Returns {"loaded", "widened", "skipped" (in state, not taken), "missing" (in model, not
    in state)}, names in order."""
    own = model.state_dict()
    loaded, widened, skipped = [], [], []
    for name, value in state.items():
        mine = own.get(name)
        if mine is None:
            skipped.append(name)
        elif tuple(mine.shape) == tuple(value.shape):
            mine.copy_(value)
            loaded.append(name)
        elif (name in WIDENABLE and mine.dim() == value.dim() == 2
              and mine.shape[0] == value.shape[0] and mine.shape[1] > value.shape[1]):
            mine.zero_()
            mine[:, :value.shape[1]].copy_(value)
            widened.append(name)
        else:
            skipped.append(name)
    missing = [k for k in own if k not in state]
    if getattr(model, "has_critic", False) and any(k.startswith("value2.") for k in missing):
        model.sync_critic()
    return {"loaded": loaded, "widened": widened, "skipped": skipped, "missing": missing}
