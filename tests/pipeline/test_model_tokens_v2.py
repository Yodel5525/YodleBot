"""E15: TokenPolicy v2 (sim/model_tokens.py), its training (sim/train_tokens.py), the v2 load in
tools/rs_checkpoint.py and RSAgent's v2 rows (tools/rs_eval.py).

Golden: the legacy model, target_slot and losses are frozen copies of sim/model_tokens.py at
370a54d (before E15): the same seed gives the same parameters, and the same batch the same
outputs, losses, statistics and gradients, bit for bit, in eval and (dropout, same seed) train
mode; a legacy checkpoint loads strict through rs_checkpoint.load_policy. v2 by hand: forward
shapes at B = 3, 2 and 0, masked buttons at exactly -1e4, the gradients reaching every new
token group, target_slot / gate_targets / eligible_counts with presses, and the loss equal to
the cross-entropies it is made of. train_tokens: the v2 columns, the loader keeping the press
rows, the row record, a tiny CPU run on v2 shards, the schema_spec check and --init /
--init-partial. rs_eval: own_pending and the opponent's counted elixir on a v2 seat (rs_elixir's
constant law: 6 elixir at tick 0 is 168000 fine units, 500 a tick at 1x, an elixir 28000, milli =
fine * 1000 // 28000), and play() giving a v2 seat its buttons and v2 rows on conftest's fake
royalesim.

Needs numpy and torch (both venvs have them); no engine, data or GPU.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import (
    DEFAULT_CATALOGUE,
    DEFAULT_UNIT_TYPES,
    SUBTILE_PER_MILLITILE,
    il_payload,
    make_fake_royalesim,
    player_dict,
    state_dict,
)

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

from sim import model_tokens as mt  # noqa: E402
from sim import train_tokens as tt  # noqa: E402

RS = mt.RS
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
SUB = SUBTILE_PER_MILLITILE
N_CARDS, N_UNITS, UNIT_SLOTS = 12, 10, 6
HAND = (2, 3, 4, 5)
B = 3
V2_TRAIN = ("unit_ids", "unit_feats", "towers", "scalars", "hand", "hand_feats", "next",
            "opp_seen", "opp_last", "opp_last_age", "spell_ids", "spell_feats", "own_pending",
            "own_pending_land", "ability_card", "ability_state", "outcome", "won", "act_card",
            "act_x", "act_y", "act_press")
NEW_MODULES = ("hand_in", "spell_in", "pending_in", "button_in", "press_query", "group_emb",
               "value2")
#: The fake catalogue's columns by name (conftest keeps the cost in column 1).
FAKE_FIELDS = ("name", "elixir", "c2", "c3", "c4", "flying", "c6", "c7", "card_kind")
NAMES = tuple(str(r[0]) for r in DEFAULT_CATALOGUE)
UNITS = tuple(DEFAULT_UNIT_TYPES)
TARGET_REPORTED = "cards-160402017-20261019.1"
#: A red Fireball in flight (SPELL_FIELDS_FALLBACK order), as test_schema_v2 has it.
FIREBALL = [1, 1, 0, 9000 * SUB, 20000 * SUB, 3500 * SUB, 6500 * SUB, 0, 0, 0, 0, 10]


# ---------------------------------------------------------------------------
# frozen: sim/model_tokens.py at 370a54d, verbatim but for the names
# ---------------------------------------------------------------------------

GRID_W = 18
TILES = 18 * 32


class LegacyTokenPolicy(nn.Module):
    def __init__(self, n_cards: int, n_units: int, d: int = 192, layers: int = 4,
                 heads: int = 6, unit_features: int = 7):
        super().__init__()
        self.card_emb = nn.Embedding(n_cards, d, padding_idx=0)
        self.unit_emb = nn.Embedding(n_units, d, padding_idx=0)
        self.unit_in = nn.Linear(unit_features, d)
        self.state_in = nn.Linear(6 + 4, d)            # towers + scalars
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

    def encode(self, b):
        n = b["unit_ids"].shape[0]
        dev = b["unit_ids"].device
        t = lambda i: self.type_emb.weight[i]
        units = (self.unit_emb(b["unit_ids"]) + self.unit_in(b["unit_feats"].float())
                 + t(1))
        state = self.state_in(torch.cat([b["towers"], b["scalars"]], -1))[:, None] + t(2)
        hand = (self.card_emb(b["hand"]) + self.slot_emb.weight[None] + t(3))
        nxt = self.card_emb(b["next"])[:, None] + t(4)
        seen = self.card_emb(b["opp_seen"]) + t(5)
        last = (self.card_emb(b["opp_last"]) + self.age_in(b["opp_last_age"].float()[..., None])
                + t(5))
        cls = self.cls.expand(n, -1, -1) + t(0)
        x = torch.cat([cls, state, hand, nxt, units, seen, last], 1)
        pad = torch.cat([
            torch.zeros(n, 2, dtype=torch.bool, device=dev),
            b["hand"] == 0, b["next"][:, None] == 0, b["unit_ids"] == 0,
            b["opp_seen"] == 0, b["opp_last"] == 0], 1)
        pad[:, 0] = False
        h = self.norm(self.encoder(x, src_key_padding_mask=pad))
        return h[:, 0], h[:, 2:6]

    def forward(self, b, card_ids=None):
        cls, hand_h = self.encode(b)
        q = self.card_query(cls)
        slot_logits = torch.einsum("nd,nkd->nk", q, hand_h) / hand_h.shape[-1] ** 0.5
        slot_logits = slot_logits.masked_fill(b["hand"] == 0, -1e4)
        gate = torch.cat([self.wait_head(cls), slot_logits], 1)
        if card_ids is None:
            slot = slot_logits.argmax(1)
            card_ids = b["hand"].gather(1, slot[:, None])[:, 0]
        tile = self.tile_head(torch.cat([cls, self.card_emb(card_ids)], -1))
        return gate, tile, self.value_head(cls)


def legacy_target_slot(hand, card):
    match = (hand == card[:, None]) & (card[:, None] > 0)
    slot = match.float().argmax(1) + 1
    has = match.any(1)
    out = torch.where(card == 0, torch.zeros_like(slot), slot)
    return torch.where((card > 0) & ~has, torch.full_like(slot, -1), out)


def legacy_losses(model, b, beta: float = 0.0, max_weight: float = 5.0):
    card = b["act_card"].long()
    gate_t = legacy_target_slot(b["hand"], card)
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


def _near(pred, true):
    px, py = pred % GRID_W, pred // GRID_W
    tx, ty = true % GRID_W, true // GRID_W
    return (((px - tx).abs() <= 1) & ((py - ty).abs() <= 1)).float().mean().item()


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------

def act_cards(n):
    """Row i: a wait when i % 3 == 0, else HAND[i % 4]; row 5 plays card 9, not in hand."""
    i = np.arange(n)
    act = np.where(i % 3 == 0, 0, np.asarray(HAND)[i % 4])
    if n > 5:
        act[5] = 9
    return torch.as_tensor(act, dtype=torch.long)


def legacy_batch(n, *, seed=0, uf=7, scalars=4):
    g = torch.Generator().manual_seed(seed)
    hand = torch.tensor(HAND).repeat(n, 1)
    hand[0, 2] = 0                                     # an empty slot (row 0 waits)
    return {
        "unit_ids": torch.randint(0, N_UNITS, (n, UNIT_SLOTS), generator=g),
        "unit_feats": torch.rand((n, UNIT_SLOTS, uf), generator=g),
        "towers": torch.rand((n, 6), generator=g), "scalars": torch.rand((n, scalars), generator=g),
        "hand": hand, "next": torch.full((n,), 6, dtype=torch.long),
        "opp_seen": torch.randint(0, N_CARDS, (n, 8), generator=g),
        "opp_last": torch.randint(0, N_CARDS, (n, 4), generator=g),
        "opp_last_age": torch.rand((n, 4), generator=g),
        "outcome": torch.rand(n, generator=g) - 0.5, "won": (torch.arange(n) % 2).float(),
        "act_card": act_cards(n),
        "act_x": torch.randint(0, 18, (n,), generator=g),
        "act_y": torch.randint(0, 32, (n,), generator=g),
    }


def v2_batch(n, *, seed=0, buttons=B):
    """A v2 batch: every id column non-zero, features in [0.1, 1), every button available."""
    g = torch.Generator().manual_seed(seed + 1)
    b = legacy_batch(n, seed=seed, uf=9, scalars=7)

    def ids(*shape):
        return torch.randint(1, N_CARDS, shape, generator=g)

    def feats(*shape):
        return 0.1 + 0.9 * torch.rand(shape, generator=g)

    b.update(hand_feats=feats(n, 4, 3), spell_ids=ids(n, 8), spell_feats=feats(n, 8, 6),
             own_pending=ids(n, 4), own_pending_land=feats(n, 4), ability_card=ids(n, buttons),
             ability_state=feats(n, buttons, 5), act_press=torch.zeros(n, dtype=torch.long))
    b["ability_state"][..., mt.AVAILABLE] = 1.0
    return b


def press_batch():
    """Six rows (hand HAND each): a wait, card 3 (slot 1), card 9 (not in hand), and presses
    of buttons 0, 2 and 1; buttons 0 and 1 are available, button 2 is not."""
    b = v2_batch(6, seed=3)
    b["hand"][0, 2] = 4                                # the full hand on every row
    b["act_card"] = torch.tensor([0, 3, 9, 0, 0, 0])
    b["act_press"] = torch.tensor([0, 0, 0, 1, 3, 2])
    b["ability_card"] = torch.tensor([[5, 6, 7]] * 6)
    b["ability_state"][..., mt.AVAILABLE] = torch.tensor([1.0, 1.0, 0.0])
    return b


def same_state(a: nn.Module, b: nn.Module):
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k, v in sa.items():
        assert v.shape == sb[k].shape and torch.equal(v, sb[k]), k


# ---------------------------------------------------------------------------
# legacy: the model, target_slot and losses before E15 (golden)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layers", [1, 2])
def test_the_legacy_model_is_the_model_before_e15(layers):
    torch.manual_seed(0)
    old = LegacyTokenPolicy(N_CARDS, N_UNITS, d=16, layers=layers, heads=2)
    torch.manual_seed(0)
    new = mt.TokenPolicy(N_CARDS, N_UNITS, d=16, layers=layers, heads=2)
    same_state(old, new)
    assert not new.v2 and not new.has_critic and new.schema.name == "rs_rows/legacy"
    assert not any(hasattr(new, m) for m in NEW_MODULES)
    assert new.input_columns == tt.COLS[:9]
    b = legacy_batch(7)
    old.eval()
    new.eval()
    with torch.no_grad():
        for card_ids in (None, b["hand"][:, 1], b["act_card"].clamp(min=0)):
            for x, y in zip(old(b, card_ids=card_ids), new(b, card_ids=card_ids)):
                assert torch.equal(x, y)
        ch, hh = old.encode(b)
        cn, hn = new.encode(b)
        assert torch.equal(ch, cn) and torch.equal(hh, hn)
        assert new(b)[0].shape == (7, 5) and new(b)[0][0, 3] == -1e4    # the empty slot
    old.train()
    new.train()                                        # dropout: the same draws, same seed
    for beta in (0.0, 0.5):
        torch.manual_seed(11)
        lo, so = legacy_losses(old, b, beta=beta)
        torch.manual_seed(11)
        ln, sn = mt.losses(new, b, beta=beta)
        assert torch.equal(lo, ln) and so == sn
        old.zero_grad()
        new.zero_grad()
        lo.backward()
        ln.backward()
        for (k, p), q in zip(old.named_parameters(), new.parameters()):
            assert (p.grad is None) == (q.grad is None), k
            assert p.grad is None or torch.equal(p.grad, q.grad), k


def test_legacy_target_slot_is_unchanged():
    g = torch.Generator().manual_seed(5)
    hand = torch.randint(0, 8, (50, 4), generator=g)
    card = torch.randint(0, 8, (50,), generator=g)
    assert torch.equal(mt.target_slot(hand, card), legacy_target_slot(hand, card))
    b = {"hand": hand, "act_card": card}
    assert torch.equal(mt.gate_targets(b), legacy_target_slot(hand, card))


# ---------------------------------------------------------------------------
# building from a meta, and the strict loads
# ---------------------------------------------------------------------------

def meta_for(**extra) -> dict:
    """A checkpoint meta as sim/train_tokens writes it for conftest's fake catalogue."""
    meta = {"n_cards": len(NAMES) + 2, "n_units": len(UNITS) + 2, "live_compat": False,
            "cards": {c: i + 2 for i, c in enumerate(NAMES)},
            "units": {u: i + 2 for i, u in enumerate(UNITS)},
            "card_table": TARGET_REPORTED, "d": 16, "layers": 1, "heads": 2}
    meta.update(extra)
    return meta


@pytest.fixture
def rc(monkeypatch):
    """tools/rs_checkpoint.py as the tools import it: no engine or drift setting from the PC."""
    monkeypatch.syspath_prepend(str(TOOLS))
    module = importlib.import_module("rs_checkpoint")
    eng = module.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), module.ALLOW_DRIFT_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    monkeypatch.setattr(module, "_VOCABS", {})
    return module


def test_model_schema_and_policy_from_meta():
    assert mt.model_schema(None) is RS.LEGACY and mt.model_schema({}) is RS.LEGACY
    for name in ("rs_rows/legacy", "rs_rows/test", "", None, 2):
        assert mt.model_schema({"schema": name}) is RS.LEGACY, name
    assert mt.model_schema({"schema": "rs_rows/v2"}) is RS.V2
    assert mt.model_schema({"schema": " v2 "}) is RS.V2
    legacy = mt.policy_from_meta(meta_for())
    assert not legacy.v2 and not legacy.has_critic
    assert legacy.card_emb.weight.shape == (18, 16) and legacy.unit_emb.weight.shape == (16, 16)
    assert len(legacy.encoder.layers) == 1 and legacy.unit_in.in_features == 7
    v2 = mt.policy_from_meta(meta_for(schema="rs_rows/v2"))
    assert v2.v2 and v2.has_critic and v2.unit_in.in_features == 9
    assert v2.state_in.in_features == 13 and v2.input_columns == V2_TRAIN[:16]
    assert not mt.policy_from_meta(meta_for(schema="rs_rows/v2", critic=False)).has_critic
    # a critic is read from the weights when the meta does not say (an RL-added one)
    critic_state = {"value2.0.weight": None}
    assert mt.meta_critic(meta_for(), critic_state) and not mt.meta_critic(meta_for(), {})
    assert mt.policy_from_meta(meta_for(), critic_state).has_critic
    assert mt.meta_critic(meta_for(schema="rs_rows/v2")) and not mt.meta_critic(meta_for())
    with pytest.raises(RS.SchemaError):
        mt.TokenPolicy(N_CARDS, N_UNITS, d=16, layers=1, heads=2, schema="v3")


def test_a_legacy_checkpoint_loads_strict(rc, tmp_path):
    torch.manual_seed(0)
    old = LegacyTokenPolicy(len(NAMES) + 2, len(UNITS) + 2, d=16, layers=1, heads=2)
    path = tmp_path / "best.pt"
    torch.save({"model": old.state_dict(), "meta": meta_for(), "epoch": 0}, path)
    vocab = rc.EngineVocab(NAMES, UNITS, TARGET_REPORTED)
    model, meta, drift = rc.load_policy(path, vocab)
    assert isinstance(model, mt.TokenPolicy) and (meta, drift) == (meta_for(), [])
    assert not model.v2 and not model.has_critic and not model.training
    same_state(old, model)
    # strict both ways: a legacy state is no v2 model's, nor a v2 state a legacy model's
    v2 = mt.TokenPolicy(len(NAMES) + 2, len(UNITS) + 2, d=16, layers=1, heads=2, schema="v2")
    with pytest.raises(RuntimeError):
        v2.load_state_dict(old.state_dict())
    with pytest.raises(RuntimeError):
        mt.policy_from_meta(meta_for()).load_state_dict(v2.state_dict())
    # and the v2 checkpoint loads as v2
    path2 = tmp_path / "v2.pt"
    torch.save({"model": v2.state_dict(), "meta": meta_for(schema="rs_rows/v2")}, path2)
    model2 = rc.load_policy(path2, vocab).model
    assert model2.v2 and model2.has_critic
    same_state(v2, model2)


# ---------------------------------------------------------------------------
# v2: shapes, masks, the critic, gradients
# ---------------------------------------------------------------------------

def v2_model(seed=0, **kw):
    torch.manual_seed(seed)
    return mt.TokenPolicy(N_CARDS, N_UNITS, d=16, layers=1, heads=2, schema="v2", **kw)


@pytest.mark.parametrize("buttons", [3, 2, 0])
def test_v2_forward_shapes(buttons):
    m = v2_model().eval()
    b = v2_batch(5, buttons=buttons)
    with torch.no_grad():
        gate, tile, value = m(b)
        out = m.outputs(b)
    assert gate.shape == (5, 5 + buttons) and tile.shape == (5, mt.TILES) and value.shape == (5, 2)
    assert set(out) == {"gate", "tile", "value", "value2", "cls"}
    assert out["cls"].shape == (5, 16) and out["value2"].shape == (5, 2)
    assert torch.equal(out["gate"], gate) and torch.equal(out["value"], value)
    assert torch.equal(out["value2"], value)           # the critic starts as value_head
    cls, hand_h, button_h = m.encode_all(b)
    assert hand_h.shape == (5, 4, 16) and button_h.shape == (5, buttons, 16)
    keys = m.state_dict()
    for name in NEW_MODULES:
        assert any(k.startswith(name + ".") for k in keys), name
    assert m.group_emb.weight.shape == (3, 16) and m.hand_in.in_features == 3
    assert (m.spell_in.in_features, m.pending_in.in_features, m.button_in.in_features) == (6, 1, 5)


def test_unavailable_buttons_get_minus_1e4():
    m = v2_model().eval()
    b = v2_batch(2)
    b["ability_card"] = torch.tensor([[5, 0, 7], [1, 6, 7]])
    b["ability_state"][..., mt.AVAILABLE] = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    b["hand"][0] = torch.tensor([2, 3, 0, 5])
    avail = mt.button_available(b)
    assert avail.tolist() == [[True, False, False], [False, True, True]]
    with torch.no_grad():
        gate = m(b)[0]
    assert gate[0, 3].item() == -1e4                   # the empty hand slot, as before
    assert gate[0, 6].item() == gate[0, 7].item() == gate[1, 5].item() == -1e4
    for r, k in ((0, 5), (1, 6), (1, 7), (0, 1), (1, 4)):
        assert gate[r, k].item() != -1e4 and abs(gate[r, k].item()) < 100, (r, k)
    assert mt.MASK_LOGIT == -1e4


def test_the_critic_is_synced_and_added():
    m = v2_model()
    with torch.no_grad():
        m.value_head[2].bias.add_(1.0)
    assert not torch.equal(m.value2[2].bias, m.value_head[2].bias)
    m.sync_critic()
    for k, v in m.value_head.state_dict().items():
        assert torch.equal(m.value2.state_dict()[k], v), k
    legacy = mt.TokenPolicy(N_CARDS, N_UNITS, d=16, layers=1, heads=2)
    legacy.sync_critic()                               # no critic: nothing to do
    assert not legacy.has_critic
    legacy.add_critic()
    assert legacy.has_critic and "value2.2.bias" in legacy.state_dict()
    assert torch.equal(legacy.value2[2].weight, legacy.value_head[2].weight)
    b = legacy_batch(3)
    legacy.eval()
    with torch.no_grad():
        out = legacy.outputs(b)
    assert torch.equal(out["value2"], out["value"]) and out["gate"].shape == (3, 5)


def test_gradients_reach_the_new_tokens():
    m = v2_model().train()
    b = v2_batch(8)
    b["act_press"][3] = 1                              # row 3 (a wait) presses button 0
    loss, stats = mt.losses(m, b, presses=True)
    loss.backward()
    for name in ("hand_in.weight", "spell_in.weight", "pending_in.weight", "button_in.weight",
                 "press_query.weight", "group_emb.weight", "unit_in.weight", "state_in.weight",
                 "card_query.weight", "tile_head.0.weight", "value_head.0.weight"):
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum().item() > 0, name
    for k in range(3):                                  # each new group's own type row
        assert m.group_emb.weight.grad[k].abs().sum().item() > 0, k
    assert all(p.grad is None for p in m.value2.parameters())    # imitation leaves the critic
    assert set(stats) == {"gate_acc", "card_acc", "tile_acc", "tile_near", "mean_w", "press_acc"}


# ---------------------------------------------------------------------------
# targets and the v2 losses, by hand
# ---------------------------------------------------------------------------

def test_target_slot_with_presses():
    hand = torch.tensor([HAND] * 6)
    card = torch.tensor([0, 3, 9, 0, 0, 0])
    press = torch.tensor([0, 0, 0, 1, 3, 2])
    avail = torch.tensor([[True, True, False]] * 6)
    assert mt.target_slot(hand, card).tolist() == [0, 2, -1, 0, 0, 0]
    assert mt.target_slot(hand, card, press, avail).tolist() == [0, 2, -1, 5, -1, 6]
    assert mt.target_slot(hand, card, press).tolist() == [0, 2, -1, 5, 7, 6]
    assert mt.target_slot(hand, card, press, avail[:, :2]).tolist() == [0, 2, -1, 5, -1, 6]
    assert mt.target_slot(hand, card, press, avail[:, :0]).tolist() == [0, 2, -1, -1, -1, -1]
    b = press_batch()
    assert mt.gate_targets(b).tolist() == [0, 2, -1, 5, -1, 6]
    assert mt.gate_targets(b, presses=False).tolist() == [0, 2, -1, -1, -1, -1]


def test_eligible_counts_with_presses():
    b = press_batch()
    assert tt.eligible_counts(b) == {"rows": 6, "gate": 5, "card": 1, "play": 2}     # legacy
    assert tt.eligible_counts(b, presses=True) == {"rows": 6, "gate": 4, "card": 1, "play": 2,
                                                   "press": 2}
    assert tt.eligible_counts(b, presses=False) == {"rows": 6, "gate": 2, "card": 1, "play": 2,
                                                    "press": 0}
    assert tt.STAT_ROWS["press_acc"] == "press"


def _parts(m, b):
    with torch.no_grad():
        gate, tile, value = m(b, card_ids=b["act_card"].clamp(min=0))
    plays = b["act_card"] > 0
    tile_t = b["act_y"] * 18 + b["act_x"]
    rest = (F.cross_entropy(tile[plays], tile_t[plays])
            + 0.5 * (F.mse_loss(value[:, 0], b["outcome"])
                     + F.binary_cross_entropy_with_logits(value[:, 1], b["won"])))
    return gate, rest


def test_v2_losses_are_the_cross_entropies():
    m = v2_model(seed=4).eval()
    b = press_batch()
    gate, rest = _parts(m, b)
    t = torch.tensor([0, 2, -1, 5, -1, 6])
    ok = t >= 0
    want = F.cross_entropy(gate[ok], t[ok]) + rest
    with torch.no_grad():
        loss, stats = mt.losses(m, b, presses=True)
    assert torch.allclose(loss, want, rtol=1e-5, atol=1e-6)
    arg = gate.argmax(1)
    assert stats["press_acc"] == pytest.approx(((arg[3] == 5).float() + (arg[5] == 6).float()).item() / 2)
    assert stats["gate_acc"] == pytest.approx((arg[ok] == t[ok]).float().mean().item())
    assert stats["card_acc"] == pytest.approx(float(gate[1, 1:5].argmax().item() + 1 == 2))
    # no press labels: the buttons stay out of the CE, the press rows have no target
    t2 = torch.tensor([0, 2])
    want2 = F.cross_entropy(gate[:2, :5], t2) + rest
    with torch.no_grad():
        loss2, stats2 = mt.losses(m, b, presses=False)
    assert torch.allclose(loss2, want2, rtol=1e-5, atol=1e-6)
    assert stats2["press_acc"] == 0.0
    assert stats2["gate_acc"] == pytest.approx((gate[:2, :5].argmax(1) == t2).float().mean().item())


# ---------------------------------------------------------------------------
# rows -> batches
# ---------------------------------------------------------------------------

def spec_row(spec, seed=0, buttons=2):
    """A row dict with `spec`'s obs_fields, shapes and dtypes (ids below N_UNITS)."""
    rng = np.random.default_rng(seed)
    row = {}
    for name in spec.obs_fields():
        col = spec.column(name)
        shape = spec.shape(name, buttons)
        if not shape:
            row[name] = True if col.dtype == "bool" else int(rng.integers(1, 5))
        elif col.dtype.startswith("int"):
            row[name] = rng.integers(0, N_UNITS, shape).astype(col.dtype)
        else:
            row[name] = rng.random(shape).astype(col.dtype)
    return row


def frozen_rsagent_batch(row):
    """RSAgent.batch for legacy rows (tools/rs_eval.py), verbatim but for torch."""
    t = torch
    keys = ("unit_ids", "unit_feats", "towers", "scalars", "hand", "opp_seen",
            "opp_last", "opp_last_age")
    b = {k: t.as_tensor(np.asarray(row[k])[None]) for k in keys}
    for k in ("unit_ids", "hand", "opp_seen", "opp_last"):
        b[k] = b[k].long()
    b["unit_feats"] = b["unit_feats"].float()
    b["opp_last_age"] = b["opp_last_age"].float()
    b["next"] = t.as_tensor([int(row["next"])]).long()
    return b


def test_row_batch_legacy_is_rsagent_batch():
    row = spec_row(RS.LEGACY)
    assert RS.LEGACY.check_row(row) == []
    got, want = mt.row_batch(row), frozen_rsagent_batch(row)
    assert set(got) == set(want) == set(RS.LEGACY.model_inputs)
    for k, v in want.items():
        assert got[k].dtype == v.dtype and torch.equal(got[k], v), k


def test_row_batch_v2():
    rows = [spec_row(RS.V2, seed=s) for s in (0, 1)]
    assert RS.V2.check_row(rows[0], buttons=2) == []
    one = mt.row_batch(rows[0], "v2")
    assert list(one) == list(RS.V2.model_inputs)
    assert one["next"].shape == (1,) and one["next"].dtype == torch.long
    assert one["ability_state"].shape == (1, 2, 5) and one["ability_state"].dtype == torch.float32
    assert one["spell_ids"].dtype == torch.long and one["hand_feats"].shape == (1, 4, 3)
    np.testing.assert_array_equal(one["own_pending_land"][0].numpy(),
                                  rows[0]["own_pending_land"].astype(np.float32))
    two = mt.rows_batch(rows, RS.V2)
    assert two["unit_feats"].shape == (2, 48, 9) and two["next"].tolist() == [rows[0]["next"],
                                                                                rows[1]["next"]]
    m = v2_model().eval()
    with torch.no_grad():
        assert m(two)[0].shape == (2, 7)


# ---------------------------------------------------------------------------
# sim/train_tokens.py on v2 rows
# ---------------------------------------------------------------------------

CARDS = {f"Card{i}": i + 2 for i in range(N_CARDS - 2)}
UNIT_NAMES = {f"Unit{i}": i + 2 for i in range(N_UNITS - 2)}
VOCAB_V2 = {
    "n_cards": N_CARDS, "n_units": N_UNITS, "live_compat": False, "cards": CARDS,
    "units": UNIT_NAMES, "engine": "royalesim", "version": "0.0.0+test",
    "schema": "rs_rows/v2", "schema_spec": RS.V2.to_json(B),
    "presses": {"mode": "on", "buttons": B, "slot_base": 4},
    "obs_delay": {"ticks": 24, "jitter": 3}, "wait_keep_at_build": 0.5,
}
#: The press rows of each 40-row shard: two waits, one train (3) and one val (33), both
#: pressing button 0, available on the odd rows.
PRESS_ROWS = (3, 33)
ROWS_V2 = {"trust": "legacy", "trust_column": "trusted", "press_rows": "trained",
           "press_rows_excluded": 0, "press_rows_trained": 4}


def v2_shard(n, *, start=0, seed=0, splits, trusted, hog, press_rows=PRESS_ROWS) -> dict:
    """One shard in rs_schema.V2's columns and dtypes. Row i: a wait when i % 3 == 0, else
    HAND[i % 4]; act_press 1 on `press_rows`; buttons (card tokens 5, 6, none), button 0
    available on the odd rows, button 1 never; outcome (start + i) / 100."""
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    out = {}
    for name in RS.V2.shard_columns():
        col = RS.V2.column(name)
        shape = (n,) + RS.V2.shape(name, B)
        if col.dtype == "bool":
            out[name] = np.ones(shape, dtype=bool)
        elif col.dtype.startswith("int"):
            out[name] = rng.integers(0, N_UNITS, shape).astype(col.dtype)
        elif col.dtype != "str":
            out[name] = rng.random(shape).astype(col.dtype)
    out["ability_state"][:, :, mt.AVAILABLE] = 0
    out["ability_state"][:, 0, mt.AVAILABLE] = i % 2 == 1
    out.update(
        hand=np.tile(np.asarray(HAND, dtype=np.int16), (n, 1)),
        next=np.full(n, 6, dtype=np.int16),
        ability_card=np.tile(np.asarray([5, 6, 0], dtype=np.int16), (n, 1)),
        act_card=np.where(i % 3 == 0, 0, np.asarray(HAND)[i % 4]).astype(np.int16),
        act_press=np.isin(i, press_rows).astype(np.int16),
        act_x=rng.integers(0, 18, n).astype(np.int16),
        act_y=rng.integers(0, 32, n).astype(np.int16),
        outcome=((start + i) / 100).astype(np.float32), won=(i % 2).astype(np.float32),
        trusted=np.asarray(trusted, dtype=bool), hog=np.asarray(hog, dtype=bool),
        match=np.asarray([f"m{(start + k) // 10:05d}" for k in i]), split=np.asarray(splits))
    return {name: out[name] for name in RS.V2.shard_columns()}


@pytest.fixture
def v2_rows(tmp_path):
    """Two v2 shards of 40 rows: 0-29 train (rows 0 and 1 untrusted), 30-39 val; even rows Hog.
    Per shard the val rows are 10 (waits 30, 33 (a press), 36, 39; 6 plays); Hog val rows 30,
    32, 34, 36, 38 (plays 32, 34, 38). vocab wait_keep_at_build 0.5: kept at load with p 1."""
    root = tmp_path / "rows_v2"
    root.mkdir()
    i = np.arange(40)
    for k in (0, 1):
        arrays = v2_shard(40, start=100 * k, seed=10 + k,
                          splits=["train"] * 30 + ["val"] * 10, trusted=i >= 2, hog=i % 2 == 0)
        np.savez_compressed(root / f"shard-{k:04d}.npz", **arrays)
    (root / "vocab.json").write_text(json.dumps(VOCAB_V2), encoding="utf-8")
    return root


def test_cols_for():
    assert tt.cols_for() is tt.COLS and tt.cols_for("legacy") == tt.COLS
    assert tt.COLS == RS.LEGACY.train_columns
    assert tt.cols_for("v2") == tt.cols_for(RS.V2) == tt.cols_for("rs_rows/v2") == V2_TRAIN
    assert V2_TRAIN == RS.V2.train_columns


def test_the_v2_shards_fit_the_spec(v2_rows):
    with np.load(v2_rows / "shard-0000.npz") as z:
        assert RS.V2.check_shard(z, buttons=B) == []


def test_the_loader_keeps_v2_press_rows(v2_rows):
    assert tt.rows_schema(v2_rows) is RS.V2 and tt.rows_schema(v2_rows / "none") is RS.LEGACY
    # wait_keep 0: every wait is dropped, the press rows are not waits
    data = tt.load("train", 0.0, False, False, 0, v2_rows)
    assert list(data) == list(V2_TRAIN)
    assert len(data["act_card"]) == 2 * (20 + 1)                # 20 plays + press row 3
    assert int(data["act_press"].sum()) == 2 and int((data["act_card"] == 0).sum()) == 2
    assert data["ability_state"].shape[1:] == (B, 5) and data["unit_feats"].dtype == np.float16
    # the same shards read as legacy rows: COLS, and the press rows left out
    old = tt.load("train", 0.0, False, False, 0, v2_rows, schema="legacy")
    assert list(old) == list(tt.COLS) and len(old["act_card"]) == 40
    val = tt.load("val", 1.0, True, False, 1, v2_rows)
    assert len(val["act_card"]) == 20 and int(val["act_press"].sum()) == 2


def test_a_v2_shard_without_a_column_is_refused(v2_rows):
    with np.load(v2_rows / "shard-0001.npz") as z:
        arrays = {k: z[k] for k in z.files if k != "spell_ids"}
    np.savez_compressed(v2_rows / "shard-0001.npz", **arrays)
    with pytest.raises(ValueError, match=r"shard-0001.npz .* has no \['spell_ids'\]"):
        tt.load("train", 1.0, False, False, 0, v2_rows)


def test_the_v2_row_record(v2_rows, tmp_path):
    assert tt.row_record(v2_rows) == ROWS_V2
    assert not tt.legacy_rows(ROWS_V2)
    assert "press rows trained: 4 rows with act_press > 0" in tt.describe_rows(ROWS_V2)
    assert tt.row_record(v2_rows, schema="legacy") == {
        "trust": "legacy", "trust_column": "trusted", "press_rows": "excluded",
        "press_rows_excluded": 4}
    # v2 rows built with presses off: no press row and no label, so no press head training
    root = tmp_path / "no_presses"
    root.mkdir()
    i = np.arange(40)
    np.savez_compressed(root / "shard-0000.npz", **v2_shard(
        40, splits=["train"] * 40, trusted=i >= 0, hog=i >= 0, press_rows=()))
    vocab = {k: v for k, v in VOCAB_V2.items() if k != "presses"}
    (root / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    rec = tt.row_record(root)
    assert rec == dict(ROWS_V2, press_rows="none", press_rows_trained=0)
    assert tt.legacy_rows(rec)


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_tiny_v2_run(v2_rows, tmp_path, capsys, rc):
    runs = tmp_path / "runs"
    assert tt.main(["--name", "v2", "--data", str(v2_rows), "--runs", str(runs), "--d", "16",
                    "--layers", "1", "--heads", "2", "--batch", "8", "--device", "cpu",
                    "--epochs", "1"]) == 0
    text = capsys.readouterr().out
    assert "schema rs_rows/v2: 22 columns, state token 13, unit features 9, 3 ability buttons" in text
    assert "rows: trust legacy (column trusted); press rows trained: 4 rows" in text
    out = runs / "v2"
    args = json.loads((out / "args.json").read_text(encoding="utf-8"))
    assert {k: args[k] for k in ROWS_V2} == ROWS_V2 and args["init_partial"] == "False"
    (rec,) = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (rec["val"]["rows"], rec["val"]["plays"]) == (20, 12)
    assert (rec["val_hog"]["rows"], rec["val_hog"]["plays"]) == (10, 6)
    assert 0.0 <= rec["val"]["press_acc"] <= 1.0 and "press_acc" in rec["val_hog"]
    blob = _load(out / "best.pt")
    assert blob["meta"] == dict(json.loads(json.dumps(VOCAB_V2)), d=16, layers=1, heads=2,
                                train_rows=ROWS_V2)
    assert blob["train_tokens"] == args
    state = blob["model"]
    for k in ("0.weight", "0.bias", "2.weight", "2.bias"):
        assert torch.equal(state[f"value2.{k}"], state[f"value_head.{k}"]), k   # synced
    model, meta, drift = rc.load_policy(out / "best.pt",
                                        rc.EngineVocab(tuple(CARDS), tuple(UNIT_NAMES), None))
    assert model.v2 and model.has_critic and not model.training and drift == []
    # --init: the same schema loads strict; a legacy checkpoint is refused for v2 rows
    again = mt.policy_from_meta(blob["meta"])
    assert tt.init_from(again, blob, VOCAB_V2, RS.V2, "best.pt") == {}
    same_state(model, again)


def test_a_schema_spec_from_other_code_is_refused(v2_rows, tmp_path):
    spec = json.loads(json.dumps(VOCAB_V2["schema_spec"]))
    spec["layout"]["spell_slots"] = 9
    (v2_rows / "vocab.json").write_text(json.dumps(dict(VOCAB_V2, schema_spec=spec)),
                                        encoding="utf-8")
    runs = tmp_path / "runs"
    with pytest.raises(SystemExit) as exc:
        tt.main(["--name", "v2", "--data", str(v2_rows), "--runs", str(runs), "--device", "cpu"])
    assert "schema_spec is not this code's rs_rows/v2" in str(exc.value)
    assert "layout" in str(exc.value) and not runs.exists()


def test_init_from(capsys):
    torch.manual_seed(0)
    legacy = mt.TokenPolicy(N_CARDS, N_UNITS, d=16, layers=1, heads=2)
    with torch.no_grad():
        legacy.value_head[2].bias.fill_(0.25)
    blob = {"model": legacy.state_dict(), "meta": {"cards": CARDS, "units": UNIT_NAMES}}
    target = v2_model(seed=1)
    with pytest.raises(SystemExit, match="--init-partial"):
        tt.init_from(target, blob, VOCAB_V2, RS.V2, "live_big.pt")
    report = tt.init_from(target, blob, VOCAB_V2, RS.V2, "live_big.pt", partial=True)
    assert report["widened"] == ["unit_in.weight", "state_in.weight"]
    assert report["skipped"] == []
    assert report["loaded"] == [k for k in legacy.state_dict() if k not in report["widened"]]
    assert {k.split(".")[0] for k in report["missing"]} == set(NEW_MODULES)
    w = target.state_in.weight
    assert torch.equal(w[:, :10], legacy.state_in.weight) and not w[:, 10:].any()
    u = target.unit_in.weight
    assert torch.equal(u[:, :7], legacy.unit_in.weight) and not u[:, 7:].any()
    assert torch.equal(target.unit_in.bias, legacy.unit_in.bias)
    assert torch.equal(target.card_emb.weight, legacy.card_emb.weight)
    assert torch.equal(target.value2[2].bias, torch.full((2,), 0.25))       # synced
    assert "2 widened (unit_in.weight, state_in.weight), 0 skipped" in capsys.readouterr().out
    # another vocabulary: refused
    other = dict(blob, meta={"cards": {"X": 2}, "units": UNIT_NAMES})
    with pytest.raises(SystemExit, match="'cards' is not the rows'"):
        tt.init_from(v2_model(), other, VOCAB_V2, RS.V2, "x.pt", partial=True)
    # legacy into legacy: the strict load it always was
    fresh = mt.TokenPolicy(N_CARDS, N_UNITS, d=16, layers=1, heads=2)
    assert tt.init_from(fresh, {"model": legacy.state_dict(), "meta": {}}, {}, RS.LEGACY,
                        "x.pt") == {}
    same_state(legacy, fresh)


# ---------------------------------------------------------------------------
# tools/rs_eval.py: RSAgent's v2 rows
# ---------------------------------------------------------------------------

@pytest.fixture
def ev(monkeypatch):
    """tools/rs_eval.py with rs_rows at its default settings (restored after the test), no engine
    config from the PC, rs_elixir's constant law, the fake catalogue cached for RSAgent, and a
    fake royalesim (FAKE_FIELDS catalogue; its battle ends at tick 200)."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    mod = importlib.import_module("rs_eval")
    rr = mod.RR
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "VISIBILITY": rr.VIS_LEGACY, "TRUST": rr.TrustRule(), "OUT": rr.OUT,
                        "CARD_FLY": {}, "CARD_BUILDING": {}, "PRESSES": "off",
                        "BUTTONS": 2, "HAND_SIZE": 4, "DECISION": rr.DecisionRule(),
                        "ROW_SCHEMA": RS.SCHEMA_LEGACY, "SF": None, "UNIT_FORMS": None,
                        "SPELL_CARDS": None}.items():
        monkeypatch.setattr(rr, name, value)
    law = rr.rs_elixir.ElixirLaw.constants()
    monkeypatch.setattr(rr.rs_elixir, "default_law", lambda: law)
    rc = importlib.import_module("rs_checkpoint")
    monkeypatch.delenv(rc.ALLOW_DRIFT_ENV, raising=False)
    monkeypatch.setattr(rc, "_VOCABS", {})
    monkeypatch.setattr(mod.RSAgent, "_CAT", [list(r) for r in DEFAULT_CATALOGUE])
    redrive = sys.modules.get("royalesim_redrive")
    if redrive is not None and hasattr(redrive, "_ENGINE"):
        monkeypatch.setattr(redrive, "_ENGINE", {})
    fake = make_fake_royalesim(states=lambda battle: state_dict(battle.tick,
                                                                game_over=battle.tick >= 200))
    fake.CATALOGUE_FIELDS = list(FAKE_FIELDS)
    monkeypatch.setitem(sys.modules, "royalesim", fake)
    threads = torch.get_num_threads()
    yield SimpleNamespace(mod=mod, rr=rr, fake=fake, cfg=eng.EngineConfig.defaults())
    torch.set_num_threads(threads)                  # RSAgent sets 1 for the process


def save_ckpt(path: Path, **meta_extra) -> Path:
    meta = meta_for(**meta_extra)
    torch.manual_seed(0)
    model = mt.policy_from_meta(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta, "epoch": 0}, path)
    return path


def test_checkpoint_v2(ev):
    v2 = ev.mod.checkpoint_v2
    assert v2({"schema": "rs_rows/v2"}) and v2({"schema": "v2"})
    for meta in ({}, None, {"schema": "rs_rows/legacy"}, {"schema": "rs_rows/test"}):
        assert not v2(meta), meta
    assert ev.mod.RSAgent.v2 is False and ev.mod.RSAgent.buttons is None    # class defaults


def test_a_v2_seat_counts_pending_and_the_opponents_elixir(ev, tmp_path):
    mod, rr = ev.mod, ev.rr
    rr.rs_engine.set_active(ev.cfg)
    rr._setup(ev.cfg)
    agent = mod.RSAgent(str(save_ckpt(tmp_path / "v2" / "best.pt", schema="rs_rows/v2")), seed=0)
    assert agent.v2 and agent.model.v2 and agent.model.has_critic and agent.buttons is None
    legacy = mod.RSAgent(str(save_ckpt(tmp_path / "old" / "best.pt")), seed=0)
    assert not legacy.v2 and not legacy.model.v2

    # own_pending: the play rows of the state's pending, soonest first; presses left out
    me = player_dict(0, pending=[[0, 2, 0, 0, 12, 4], [1, 5, 0, 0, 3, 1], [0, 5, 0, 0, 5, 4]],
                     pending_cost=8)
    state = state_dict(100, players=[me, player_dict(1)], spells=[FIREBALL])
    assert agent.own_pending(state, 0) == [(7, 5), (4, 12)]
    agent.lead = 6
    assert agent.own_pending(state, 0) == [(4, 6)]             # Musketeer lands in the look-ahead
    agent.lead = 0

    # the opponent's bar: red's Knight (token 10, 3 elixir) ran at tick 50
    mem = {"seen": {0: set(), 1: {10}}, "last": {0: [], 1: [(10, 50)]}}
    assert agent.opp_elixir(0, mem, 50) == (3892, True)         # (168000 + 25000 - 84000) // 28
    assert agent.opp_elixir(0, mem, 100) == (4785, True)        # (168000 + 50000 - 84000) // 28
    assert agent.opp_elixir(0, mem, 100) == (4785, True)        # fed once
    mem2 = {"seen": {0: set(), 1: {10, 14}}, "last": {0: [], 1: [(10, 50), (14, 120)]}}
    assert agent.opp_elixir(0, mem2, 130) == (1321, True)       # Valkyrie 4: 37000 // 28
    fresh = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}
    assert agent.opp_elixir(0, fresh, 100) == (7785, True)      # a new match: 218000 // 28
    assert legacy.v2_kw(state, 0, mem, 100) == {}

    kw = agent.v2_kw(state, 0, mem, 100)
    assert kw == {"schema": "rs_rows/v2", "opp_elixir": (4785, True), "pending": [(7, 5), (4, 12)]}
    agent.buttons = [5]
    row = rr.obs_row(state, 0, SUB, mem, 100, visibility=agent.visibility,
                     **agent.v2_kw(state, 0, mem, 100))
    assert RS.V2.check_row(row, buttons=2) == []
    assert row["own_pending"].tolist() == [7, 4, 0, 0]
    assert row["own_pending_land"].tolist() == [np.float16(5 / 40), np.float16(12 / 40), 0, 0]
    assert row["scalars"][5] == np.float32(0.4785) and row["scalars"][6] == 1.0
    assert row["spell_ids"].tolist()[:2] == [3, 0] and row["ability_card"].tolist() == [7, 0]
    b = agent.batch(row)
    assert list(b) == list(RS.V2.model_inputs)
    with torch.no_grad():
        assert agent.model(b)[0].shape == (1, 7)


def test_play_gives_a_v2_seat_its_buttons_and_v2_rows(ev, tmp_path, monkeypatch):
    mod = ev.mod
    ckpt = save_ckpt(tmp_path / "v2" / "best.pt", schema="rs_rows/v2")
    made, rows = [], {}

    class Spy(mod.RSAgent):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    real = mod.rs_decide.decide

    def spy(policy, row, state, team, *a, **k):
        rows[int(row["tick"])] = (team, row)
        return real(policy, row, state, team, *a, **k)

    monkeypatch.setattr(mod, "AGENT_CLASS", Spy)
    monkeypatch.setattr(mod.rs_decide, "decide", spy)
    out = mod.play((json.dumps(il_payload()), str(ckpt), "agent", 0, "", 0, 0, False, False,
                    False, ev.cfg.to_json()))
    assert out["hog_team"] == 0 and isinstance(out["agent_plays"], int)
    (agent,) = made
    assert agent.v2 and agent.buttons == [5]                    # the hero Musketeer
    assert sorted(rows) == list(range(90, 200, 10))
    for tick, (team, row) in rows.items():
        assert team == 0 and RS.V2.check_row(row, buttons=2) == [], tick
        assert row["ability_card"].tolist() == [7, 0], tick
    # red's Knight runs at 160 and is on the board (mem lag 1) from 161
    assert rows[160][1]["scalars"][5] == np.float32(0.8857)    # 248000 // 28
    assert rows[170][1]["scalars"][5] == np.float32(0.6035)    # 169000 // 28
    assert agent.last_decision.mask[5:] == (False, False)      # rs_decide: no press yet
