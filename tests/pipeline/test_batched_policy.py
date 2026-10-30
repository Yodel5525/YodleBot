"""S05: tools/rs_search.BatchedPolicy, the shared encoder pass (docs/LIVE_SEARCH_PLAN.md 6.1).

On tiny random TokenPolicies on the CPU (legacy, legacy with a critic, v2; d 16, 2 layers): the
gate, the critic's value and the tile logits of every hand slot equal rs_decide's
TokenPolicyAdapter and sim/model_tokens' own forward within 1e-5, for one row and for many rows
in one pass; heads(b) and four tiles(b, slot) run the encoder once (a forward hook counts the
passes); tiles_for / tiles_cards give each pair's logits from one tile-head call; values() reads
the value head's column 0 ("swing") or tanh(z / 2) of the critic ("critic"), and the teacher's
policy_values reads the same V through it (values(rows) is the critic's, refused for a model
without one, as the teacher's adapter path refuses it); an empty hand slot is masked and tiled as
the adapter has it; rs_decide.decide gives the same decision, draw for draw (the same Decision
and the same generator state after it), in the legacy, live and sampling presets, while the
adapter runs one pass per slot tried. Also: as_batched, the training-mode
refusal, mixed widths, the identity cache, and that importing rs_search imports no torch.

Needs numpy; torch via importorskip; the CUDA test is gpu-marked. No engine or data.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import DEFAULT_CATALOGUE, DEFAULT_UNIT_TYPES, player_dict, state_dict

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for _path in (str(ROOT), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import rs_decide as RD  # noqa: E402  (the name rs_search and rs_eval import it by)
import rs_schema as RS  # noqa: E402
import rs_search as SR  # noqa: E402

SUB = RD.SUBTILE_PER_MILLITILE
N_CARDS = len(DEFAULT_CATALOGUE) + 2         # tokens: catalogue id + 2
N_UNITS = len(DEFAULT_UNIT_TYPES) + 2
KINDS = ("legacy", "legacy_critic", "v2")
MODES = ("legacy", "live", "sampling")
ELIXIRS = (2500, 4500, 7000, 9800)
AVAILABLE = RS.ABILITY_FEATS.index("available")
#: Decision fields compared with a tolerance (the rest exactly).
FLOAT_FIELDS = ("p_wait", "q_wait", "logq_gate", "logq_tile")
FLOAT_TUPLES = ("gate_q", "value")
ATOL = 1e-5


def _torch():
    return pytest.importorskip("torch")


def schema_of(kind: str) -> str:
    return "v2" if kind == "v2" else "legacy"


def make_model(kind: str, seed: int = 0):
    """A tiny TokenPolicy in eval mode; a critic (legacy_critic, v2) differs from value_head."""
    torch = _torch()
    from sim import model_tokens as mt
    torch.manual_seed(seed)
    if kind == "v2":
        model = mt.TokenPolicy(N_CARDS, N_UNITS, d=16, layers=2, heads=2, schema="v2")
    else:
        model = mt.TokenPolicy(N_CARDS, N_UNITS, d=16, layers=2, heads=2)
        if kind == "legacy_critic":
            model.add_critic()
    if model.has_critic:
        with torch.no_grad():
            model.value2[2].bias.add_(0.5)
    return model.eval()


def make_row(rng, schema: str = "legacy", buttons: int = 2) -> dict:
    """An obs_row of `schema` (every model input at its spec's shape and dtype): a hand of four
    distinct catalogue cards (tokens id + 2), the next card, some units, towers, scalars and the
    opponent's cards; v2 also hand features, a spell, a pending play and one available button."""
    spec = RS.get(schema)
    row = {"tick": 500}
    for name in spec.model_inputs:
        shape, dtype = spec.shape(name, buttons), spec.dtype(name)
        row[name] = np.zeros(shape, dtype=dtype) if shape else np.dtype(dtype).type(0)
    cards = rng.permutation(len(DEFAULT_CATALOGUE))[:5] + 2
    row["hand"][:] = cards[:4]
    row["next"] = np.dtype(spec.dtype("next")).type(int(cards[4]))
    k = int(rng.integers(1, 8))
    row["unit_ids"][:k] = rng.integers(1, N_UNITS, k)
    row["unit_feats"][:k] = rng.random((k, row["unit_feats"].shape[1]))
    row["towers"][:] = rng.random(6)
    row["scalars"][:] = rng.random(row["scalars"].shape[0])
    row["opp_seen"][:3] = rng.integers(2, N_CARDS, 3)
    row["opp_last"][:2] = rng.integers(2, N_CARDS, 2)
    row["opp_last_age"][:2] = rng.random(2)
    if spec.v2:
        row["hand_feats"][:] = rng.random((4, 3))
        row["spell_ids"][0] = 3
        row["spell_feats"][0] = rng.random(6)
        row["own_pending"][0] = row["hand"][1]
        row["own_pending_land"][0] = 0.25
        if buttons:
            row["ability_card"][0] = 7
            row["ability_state"][0, AVAILABLE] = 1.0
    return row


def state_for(row: dict, team: int = 1, elixir: int = 7000) -> dict:
    """The present state decide reads: `team` holds the row's hand (catalogue ids)."""
    hand = [int(t) - 2 for t in row["hand"]]
    mine = player_dict(team, hand=hand, next_card=int(row["next"]) - 2, elixir_milli=elixir)
    theirs = player_dict(1 - team)
    return state_dict(int(row["tick"]), players=[mine, theirs] if team == 0 else [theirs, mine])


def some_tiles(refuse=()):
    """A legality callable: about three tiles in four, none for the slots in `refuse`."""
    refuse = frozenset(int(s) for s in refuse)

    def check(slot, card, x, y):
        if slot in refuse:
            return False
        tx, ty = x // (1000 * SUB), y // (1000 * SUB)
        return (3 * tx + ty + slot) % 4 != 0
    return check


def config(mode: str):
    if mode == "legacy":
        return RD.DecideConfig.legacy(0.0)
    if mode == "live":
        return RD.DecideConfig.live(0.0)
    return RD.DecideConfig.sampling(0.0)


@contextlib.contextmanager
def passes(model):
    """The batch size of every encoder pass while the block runs."""
    seen: list[int] = []
    handle = model.encoder.register_forward_hook(
        lambda module, args, out: seen.append(int(args[0].shape[0])))
    try:
        yield seen
    finally:
        handle.remove()


def assert_same_decision(a, b):
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    assert list(da) == list(db)
    for k in da:
        if k in FLOAT_FIELDS:
            assert da[k] == pytest.approx(db[k], abs=1e-6), k
        elif k in FLOAT_TUPLES:
            assert len(da[k]) == len(db[k]) and list(da[k]) == pytest.approx(list(db[k]),
                                                                            abs=1e-6), k
        else:
            assert da[k] == db[k], k


def close(a, b, atol=ATOL):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=atol)


# ---------------------------------------------------------------------------
# one row: the adapter's heads and tiles, and the model's forward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_one_row_equals_the_adapter_and_the_forward_in_one_pass(kind):
    torch = _torch()
    model = make_model(kind)
    adapter, bp = RD.TokenPolicyAdapter(model), SR.BatchedPolicy(model)
    assert bp.schema is adapter.schema and bp.has_critic == adapter.has_critic == (kind != "legacy")
    assert bp.value_kind == (SR.VALUE_SWING if kind == "legacy" else SR.VALUE_CRITIC)
    assert bp.device.type == "cpu"
    row = make_row(np.random.default_rng(5), schema_of(kind))
    ba, bb = adapter.batch(row), bp.batch(row)
    assert list(ba) == list(bb)
    for k in ba:
        assert ba[k].dtype == bb[k].dtype and torch.equal(ba[k], bb[k]), k
    with passes(model) as seen:
        g, value = bp.heads(bb)
        tiles = [bp.tiles(bb, s) for s in range(RD.HAND_SIZE)]
    assert seen == [1]                       # the gate, the value and all four slots: one pass
    assert bp.stats["passes"] == 1 and bp.stats["cache_hits"] == 4
    assert bp.stats["tile_calls"] == 4 and bp.stats["tile_rows"] == 4
    ga, va = adapter.heads(ba)
    assert g.dtype == np.float32 and g.shape == ga.shape == ((7,) if kind == "v2" else (5,))
    close(g, ga)
    close(bp.gate(bb), ga)
    if va is None:
        assert value is None
    else:
        assert len(value) == 2 and list(value) == pytest.approx(list(va), abs=ATOL)
    with torch.no_grad():
        gate_f, _, _ = model(ba, card_ids=ba["hand"][:, 0].clamp(min=0))
    close(g, gate_f[0].numpy())
    for s in range(RD.HAND_SIZE):
        assert tiles[s].shape == (RD.TILES,) and tiles[s].dtype == np.float32
        close(tiles[s], adapter.tiles(ba, s))
        with torch.no_grad():
            _, tile_f, _ = model(ba, card_ids=ba["hand"][:, s])
        close(tiles[s], tile_f[0].numpy())
    assert not np.allclose(tiles[0], tiles[1])          # each slot's own card


# ---------------------------------------------------------------------------
# many rows: one pass, the same per-row outputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_many_rows_in_one_pass_equal_row_by_row(kind):
    torch = _torch()
    model = make_model(kind, seed=1)
    adapter, bp = RD.TokenPolicyAdapter(model), SR.BatchedPolicy(model)
    rng = np.random.default_rng(7)
    rows = [make_row(rng, schema_of(kind)) for _ in range(6)]
    with passes(model) as seen:
        enc = bp.encode_rows(rows)
    assert seen == [6] and len(enc) == 6 and enc.gate.shape[0] == 6
    assert enc.value.shape == (6, 2)
    assert (enc.value2 is None) == (kind == "legacy")
    for i, row in enumerate(rows):
        b = adapter.batch(row)
        ga, va = adapter.heads(b)
        g, v = enc.head(i)
        close(g, ga)
        if va is None:
            assert v is None
        else:
            assert list(v) == pytest.approx(list(va), abs=ATOL)
        for s in range(RD.HAND_SIZE):
            close(enc.tiles(i, s), adapter.tiles(b, s))
    pairs = [(0, 1), (3, 0), (5, 3), (3, 0)]
    calls = bp.stats["tile_calls"]
    got = enc.tiles_for(pairs)
    assert got.shape == (4, RD.TILES) and bp.stats["tile_calls"] == calls + 1
    for k, (i, s) in enumerate(pairs):
        close(got[k], enc.tiles(i, s))
    rows_idx, tokens = [2, 2, 4], [int(rows[2]["hand"][0]), 9, 2]
    by_card = enc.tiles_cards(rows_idx, tokens)
    assert by_card.shape == (3, RD.TILES)
    for k, (i, tok) in enumerate(zip(rows_idx, tokens)):
        with torch.no_grad():
            _, tile_f, _ = model(adapter.batch(rows[i]), card_ids=torch.tensor([tok]))
        close(by_card[k], tile_f[0].numpy())
    close(by_card[0], enc.tiles(2, 0))                   # the card in row 2's slot 0
    assert enc.tiles_for([]).shape == (0, RD.TILES)
    with pytest.raises(IndexError):
        enc.tiles(6, 0)
    with pytest.raises(SR.BatchedPolicyError):
        enc.tiles_cards([0, 1], [5])
    # stack() is the per-row batches concatenated
    b = bp.stack(rows)
    for k, x in b.items():
        assert torch.equal(x, torch.cat([adapter.batch(r)[k] for r in rows], 0)), k


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_values_read_the_value_head_or_the_critic(kind):
    torch = _torch()
    from tools.rl import teacher as T
    model = make_model(kind, seed=2)
    adapter, bp = RD.TokenPolicyAdapter(model), SR.BatchedPolicy(model)
    rows = [make_row(np.random.default_rng(10 + k), schema_of(kind)) for k in range(5)]
    swing, z = [], []
    for row in rows:
        b = adapter.batch(row)
        with torch.no_grad():
            out = model.outputs(b, card_ids=b["hand"][:, 0].clamp(min=0))
        swing.append(float(out["value"][0, 0]))
        if model.has_critic:
            z.append(float(out["value2"][0, 1]))
    with passes(model) as seen:
        v = bp.values(rows, SR.VALUE_SWING)
    assert seen == [5] and v.dtype == np.float64 and v.shape == (5,)
    close(v, swing)
    assert bp.values([], SR.VALUE_SWING).shape == (0,)
    if kind == "legacy":
        assert bp.value_kind == SR.VALUE_SWING
        close(bp.values(rows, bp.value_kind), swing)            # the search's V
        close(bp.encode_rows(rows).values(), swing)
        # values(rows) is the teacher's protocol, the critic's V: a model without a critic is
        # refused, as the teacher's adapter path refuses it, never given the swing instead
        with pytest.raises(SR.BatchedPolicyError, match="critic"):
            bp.values(rows)
        with pytest.raises(SR.BatchedPolicyError, match="critic"):
            T.policy_values(bp, rows)
        with pytest.raises(T.TeacherError, match="critic"):
            T.policy_values(adapter, rows)
        with pytest.raises(SR.BatchedPolicyError, match="critic"):
            bp.values(rows, SR.VALUE_CRITIC)
        with pytest.raises(SR.BatchedPolicyError, match="critic"):
            SR.BatchedPolicy(model, value_kind=SR.VALUE_CRITIC)
        close(SR.BatchedPolicy(model, value_kind=SR.VALUE_SWING).values(rows), swing)
        return
    critic = np.tanh(np.asarray(z) / 2)
    got = bp.values(rows)
    close(got, critic)
    assert not np.allclose(got, swing, atol=1e-3)        # the critic, not the value head
    with pytest.raises(SR.BatchedPolicyError, match="kind"):
        bp.values(rows, "won")
    close(SR.BatchedPolicy(model, value_kind=SR.VALUE_SWING).values(rows), swing)
    # the teacher calls values(rows) first: the V of its own batched adapter path
    close(T.policy_values(bp, rows), got, atol=1e-6)
    close(T.policy_values(adapter, rows), got, atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_an_empty_hand_slot_is_masked_and_tiled_as_the_adapter_does(kind):
    """A row with an empty slot (token 0) in a batch of full hands: the gate masks that slot
    (MASK_LOGIT) and its tiles read the padding card, both as the adapter has them row by row;
    the other rows are untouched."""
    model = make_model(kind, seed=8)                      # torch, or the test skips
    from sim import model_tokens as mt
    adapter, bp = RD.TokenPolicyAdapter(model), SR.BatchedPolicy(model)
    rng = np.random.default_rng(8)
    rows = [make_row(rng, schema_of(kind)) for _ in range(3)]
    rows[1]["hand"][2] = 0
    enc = bp.encode_rows(rows)
    for i, row in enumerate(rows):
        b = adapter.batch(row)
        ga, va = adapter.heads(b)
        g, v = enc.head(i)
        close(g, ga)
        if va is not None:
            assert list(v) == pytest.approx(list(va), abs=ATOL)
        for s in range(RD.HAND_SIZE):
            close(enc.tiles(i, s), adapter.tiles(b, s))
    assert enc.gate[1, 1 + 2] == pytest.approx(mt.MASK_LOGIT)
    assert enc.gate[0, 1 + 2] > mt.MASK_LOGIT / 2 and enc.gate[2, 1 + 2] > mt.MASK_LOGIT / 2
    close(enc.tiles_for([(1, 2), (0, 2)])[0], enc.tiles(1, 2))


# ---------------------------------------------------------------------------
# rs_decide.decide, draw for draw
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", KINDS)
def test_decide_is_the_same_draw_for_draw(kind, mode, seed):
    model = make_model(kind, seed=seed)
    adapter, bp = RD.TokenPolicyAdapter(model), SR.BatchedPolicy(model)
    rng = np.random.default_rng(100 + seed)
    team = seed % 2
    cfg = config(mode)
    legal = some_tiles(refuse=(seed,) if seed >= 2 else ())
    for k in range(6):
        row = make_row(rng, schema_of(kind))
        st = state_for(row, team=team, elixir=ELIXIRS[k % len(ELIXIRS)])
        ra, rb = np.random.default_rng([seed, k]), np.random.default_rng([seed, k])
        da = RD.decide(adapter, row, st, team, legal, cfg, rng=ra)
        db = RD.decide(bp, row, st, team, legal, cfg, rng=rb)
        assert_same_decision(da, db)
        assert ra.bit_generator.state == rb.bit_generator.state      # as many draws, the same


@pytest.mark.parametrize("kind", KINDS)
def test_a_capped_decision_plays_the_same_tile(kind):
    model = make_model(kind, seed=6)
    adapter, bp = RD.TokenPolicyAdapter(model), SR.BatchedPolicy(model)
    row = make_row(np.random.default_rng(6), schema_of(kind))
    st = state_for(row, team=1, elixir=9800)             # capped (live): never a wait
    cfg = RD.DecideConfig.live(0.0)
    da = RD.decide(adapter, row, st, 1, some_tiles(), cfg, rng=np.random.default_rng(1))
    db = RD.decide(bp, row, st, 1, some_tiles(), cfg, rng=np.random.default_rng(1))
    assert da.capped and da.is_play
    assert_same_decision(da, db)
    assert db.command == da.command and db.mode == RD.MODE_LIVE


def test_decide_runs_the_encoder_once_where_the_adapter_runs_it_per_slot():
    model = make_model("v2", seed=3)
    adapter, bp = RD.TokenPolicyAdapter(model), SR.BatchedPolicy(model)
    row = make_row(np.random.default_rng(3), "v2")
    st = state_for(row, team=1, elixir=9800)             # capped: every slot is tried
    cfg = RD.DecideConfig.live(0.0)

    def never(slot, card, x, y):
        return False
    with passes(model) as seen_a:
        da = RD.decide(adapter, row, st, 1, never, cfg, rng=np.random.default_rng(0))
    with passes(model) as seen_b:
        db = RD.decide(bp, row, st, 1, never, cfg, rng=np.random.default_rng(0))
    assert da.kind == RD.WAIT and da.reason == RD.R_NO_TILE and len(da.tried) == 4
    assert seen_a == [1] * 5 and seen_b == [1]           # heads + a forward per slot vs one
    assert bp.stats["passes"] == 1 and bp.stats["tile_calls"] == 4
    assert_same_decision(da, db)


# ---------------------------------------------------------------------------
# construction, as_batched, refusals, the cache
# ---------------------------------------------------------------------------

def test_as_batched_wraps_adapters_and_models_and_keeps_other_policies():
    torch = _torch()
    model = make_model("legacy")

    def own_batch(row):
        return RD.row_batch(row, torch, None)
    adapter = RD.TokenPolicyAdapter(model, batch=own_batch)
    bp = SR.as_batched(adapter)
    assert isinstance(bp, SR.BatchedPolicy) and bp.model is model and bp.torch is adapter.torch
    row = make_row(np.random.default_rng(0))
    assert list(bp.batch(row)) == list(own_batch(row))
    assert SR.as_batched(bp) is bp
    direct = SR.as_batched(model)
    assert isinstance(direct, SR.BatchedPolicy) and direct.model is model
    scripted = SimpleNamespace(batch=lambda r: r, gate=lambda b: None, tiles=lambda b, s: None)
    assert SR.as_batched(scripted) is scripted
    odd = RD.TokenPolicyAdapter(SimpleNamespace(), torch)          # no tile head: kept
    assert SR.as_batched(odd) is odd
    with pytest.raises(SR.BatchedPolicyError, match="TokenPolicy"):
        SR.BatchedPolicy(SimpleNamespace(outputs=None), torch)
    with pytest.raises(SR.BatchedPolicyError, match="value_kind"):
        SR.BatchedPolicy(model, value_kind="won")


def test_a_model_in_training_mode_is_refused_and_left_as_it_is():
    model = make_model("v2").train()
    bp = SR.BatchedPolicy(model)
    b = bp.batch(make_row(np.random.default_rng(1), "v2"))
    with pytest.raises(SR.BatchedPolicyError, match="eval"):
        bp.heads(b)
    assert model.training and bp.stats["passes"] == 0
    model.eval()
    bp.heads(b)
    assert not model.training and bp.stats["passes"] == 1


def test_rows_of_different_widths_are_refused():
    model = make_model("v2")
    bp = SR.BatchedPolicy(model)
    rng = np.random.default_rng(2)
    two, three = make_row(rng, "v2", buttons=2), make_row(rng, "v2", buttons=3)
    with pytest.raises(SR.BatchedPolicyError, match="ability_card"):
        bp.stack([two, three])
    with pytest.raises(SR.BatchedPolicyError, match="no rows"):
        bp.stack([])
    assert bp.encode_rows([three]).gate.shape == (1, 8)           # B is read from the batch


def test_the_cache_is_by_identity_and_bounded():
    model = make_model("legacy")
    bp = SR.BatchedPolicy(model, cache=2)
    row = make_row(np.random.default_rng(3))
    b1, b2, b3 = bp.batch(row), bp.batch(row), bp.batch(row)
    bp.heads(b1)
    bp.tiles(b1, 2)                                      # the pass of b1
    bp.heads(b2)                                         # equal contents, another dict
    assert bp.stats["passes"] == 2 and bp.stats["cache_hits"] == 1
    bp.heads(b3)                                         # b1 leaves a cache of two
    bp.tiles(b1, 0)
    assert bp.stats["passes"] == 4
    bp.clear()
    bp.heads(b3)
    assert bp.stats["passes"] == 5
    none = SR.BatchedPolicy(model, cache=0)
    none.heads(b1)
    none.tiles(b1, 0)
    assert none.stats["passes"] == 2 and none.stats["cache_hits"] == 0


# ---------------------------------------------------------------------------
# CUDA, and the import weight
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_on_cuda_the_batches_move_and_the_outputs_come_back_as_numpy():
    model = make_model("v2", seed=4)
    adapter = RD.TokenPolicyAdapter(model)
    rng = np.random.default_rng(4)
    rows = [make_row(rng, "v2") for _ in range(4)]
    want = [adapter.heads(adapter.batch(r)) for r in rows]
    want_tiles = adapter.tiles(adapter.batch(rows[0]), 2)
    bp = SR.BatchedPolicy(copy.deepcopy(model).cuda().eval())
    assert bp.device.type == "cuda"
    b = bp.batch(rows[0])
    assert b["hand"].device.type == "cuda"
    close(bp.tiles(b, 2), want_tiles, atol=1e-4)
    enc = bp.encode_rows(rows)
    for i, (g, v) in enumerate(want):
        close(enc.head(i)[0], g, atol=1e-4)
        assert list(enc.head(i)[1]) == pytest.approx(list(v), abs=1e-4)
    close(bp.values(rows), [np.tanh(v[1] / 2) for _, v in want], atol=1e-4)
    close(enc.tiles_for([(0, 2)])[0], want_tiles, atol=1e-4)


def test_importing_rs_search_imports_no_torch():
    code = ("import sys\n"
            "import tools.rs_search\n"
            "heavy = ('torch', 'sim.model_tokens', 'royalesim', 'scripts.gamestate.memread',\n"
            "         'gamestate.memread')\n"
            "print(','.join(m for m in heavy if m in sys.modules))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""
