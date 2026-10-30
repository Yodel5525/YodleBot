"""V06 on v2 checkpoints (E15): tools/rs_decide.py's ability buttons, own-pending inputs and
critic value. The legacy decisions are tests/pipeline/test_rs_decide.py's (golden, unchanged).

Pure tests with scripted logits (FakePolicy) and conftest's FakeBattle: button_mask from the
abilities rows and the spendable elixir; presses off (every preset's default) masks every
button and decides exactly as a five-entry gate; an argmax or a sampled draw on a button is a
PRESS (command (team, HAND_SIZE + button, 0, 0), no tile, no legality call); a press in flight
(the state's kind-1 pending rows, or a Pending with a button) masks its button; the fallback goes
on to a button; the control state never taps a press; the critic's value in the record; the
adapter's v2 batch and its one-pass heads; pending_inputs from the live taps, the engine's
pending rows as rs_eval.RSAgent.own_pending reads them; prune_pending for presses;
presses_allowed and obs_delay_ticks from a checkpoint meta.

Torch-marked by importorskip: the v2 batch equals sim/model_tokens.row_batch, a real v2
TokenPolicy through the adapter and decide, and presses_allowed on sim/train_tokens' meta.
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import (DEFAULT_CATALOGUE, DEPLOY_REASONS, FakeBattle, player_dict,
                                     state_dict)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for _path in (str(ROOT), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import rs_decide as RD  # noqa: E402  (the name rs_eval imports it by)
import rs_engine  # noqa: E402
import rs_schema as RS  # noqa: E402

CAT = [list(r) for r in DEFAULT_CATALOGUE]
CID = {row[0]: i for i, row in enumerate(CAT)}
#: HogRider 2 (4 elixir), Musketeer 5 (4), Skeletons 6 (1), Cannon 0 (3).
HAND = [CID["HogRider"], CID["Musketeer"], CID["Skeletons"], CID["Cannon"]]
MUSKETEER, ICE_GOLEM = CID["Musketeer"], CID["IceGolemite"]          # 5, 3
OUT_OF_TERRITORY = DEPLOY_REASONS.index("OUT_OF_TERRITORY")
#: Two ability buttons (royalegym AbilityRow: available, spent, cost, card_id, cooldown_ticks):
#: a hero Musketeer (3 elixir) and a hero Ice Golem (2 elixir), both available.
BUTTONS = ([1, 0, 3, MUSKETEER, 0], [1, 0, 2, ICE_GOLEM, 0])
ROW = {"tick": 500}


@pytest.fixture(autouse=True)
def clean_engine(monkeypatch):
    """No HASTYCR_* engine variable from the PC, and no active config left from a test."""
    for var in (rs_engine.CONFIG_ENV, *rs_engine.ENV.values()):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rs_engine, "_ACTIVE", None)


def ranked(*tiles: int) -> np.ndarray:
    """Tile logits: `tiles` best first, in that order, then every other tile by index."""
    t = -10.0 - np.arange(RD.TILES, dtype=np.float32) / 1000
    for i, k in enumerate(tiles):
        t[k] = 100.0 - i
    return t


class FakePolicy:
    """decide()'s policy interface with scripted logits (gate only: no critic)."""

    def __init__(self, gate, tiles=None):
        self.g = np.asarray(gate, dtype=np.float32)
        self.t = dict(tiles or {})
        self.tile_calls: list[int] = []

    def batch(self, row):
        return row

    def gate(self, b):
        return self.g.copy()

    def tiles(self, b, slot):
        self.tile_calls.append(int(slot))
        return self.t.get(int(slot), ranked())


class ValuePolicy(FakePolicy):
    """A policy with heads(): decide must take the gate and the value from it, not gate()."""

    def __init__(self, gate, value, tiles=None):
        super().__init__(gate, tiles)
        self.value = value

    def heads(self, b):
        return self.g.copy(), self.value

    def gate(self, b):
        raise AssertionError("decide calls heads() when the policy has it")


class StubRng:
    """rng.random() always `u`; counts the draws."""

    def __init__(self, u: float):
        self.u = u
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return self.u


def hero_state(*, team=1, abilities=(), elixir=9000, hand=HAND, costs=None, pending=(),
               pending_cost=0, tick=500):
    """A state_json dict with `team` (default the live seat, 1) holding `hand` and `abilities`."""
    mine = player_dict(team, hand=list(hand), elixir_milli=elixir, hand_costs=costs,
                       pending=[list(r) for r in pending], pending_cost=pending_cost,
                       abilities=[list(r) for r in abilities])
    theirs = player_dict(1 - team)
    players = [mine, theirs] if team == 0 else [theirs, mine]
    return state_dict(tick, players=players)


def always(slot, card, x, y):
    return True


class FakeTensor:
    """Just the torch.Tensor surface the adapter and row_batch use, on numpy."""

    __slots__ = ("a",)

    def __init__(self, a):
        self.a = np.asarray(a)

    def long(self):
        return FakeTensor(self.a.astype(np.int64))

    def float(self):
        return FakeTensor(self.a.astype(np.float32))

    def clamp(self, min=None):
        return FakeTensor(self.a if min is None else np.maximum(self.a, min))

    def numpy(self):
        return self.a

    def __getitem__(self, index):
        return FakeTensor(self.a[index])


FAKE_TORCH = SimpleNamespace(as_tensor=lambda x: FakeTensor(np.asarray(x)),
                             no_grad=contextlib.nullcontext)


def v2_row(buttons: int = 2) -> dict:
    """An obs_row of schema v2 (every model input at its spec's shape and dtype), with a hand,
    a next card, two units, a spell, two pending plays and one available button."""
    row = {"tick": 500}
    for name in RS.V2.model_inputs:
        shape, dtype = RS.V2.shape(name, buttons=buttons), RS.V2.dtype(name)
        row[name] = np.zeros(shape, dtype=dtype) if shape else np.dtype(dtype).type(0)
    row["hand"][:] = [4, 7, 8, 2]
    row["next"] = np.int16(5)
    row["unit_ids"][:2] = [3, 4]
    row["unit_feats"][:2] = 0.5
    row["scalars"][:] = [0.5, 0.1, 0, 0, 0, 0.4785, 1]
    row["spell_ids"][0] = 3
    row["spell_feats"][0] = [1, 0.5, 0.5, 0.5, 0.5, 0.2]
    row["own_pending"][:2] = [7, 4]
    row["own_pending_land"][:2] = [0.125, 0.3]
    row["ability_card"][0] = 7
    row["ability_state"][0] = [1, 1, 0, 0.3, 0]
    return row


# ---------------------------------------------------------------------------
# button_mask and the abilities rows
# ---------------------------------------------------------------------------

def test_button_mask_reads_the_abilities_rows():
    """Row 0 available (3 elixir); row 1 not available; row 2 spent; row 3 short: available,
    cost 0; button 4 has no row."""
    st = hero_state(abilities=[[1, 0, 3, MUSKETEER, 0], [0, 0, 1, ICE_GOLEM, 40],
                               [1, 1, 2, 7, 0], [1]])
    assert RD.button_mask(st, 1, 5, 3000) == [True, False, False, True, False]
    assert RD.button_mask(st, 1, 5, 2999) == [False, False, False, True, False]
    assert RD.button_mask(st, 1, 5, 3000, pressing=[0, 3]) == [False] * 5
    assert RD.button_mask(st, 1, 2, 3000) == [True, False]
    assert RD.button_mask(st, 0, 2, 9000) == [False, False]        # team 0: no abilities rows
    assert RD.ability_row([1]) == (1, 0, 0, -1, -1)
    assert RD.ability_row([1, 0, 3, 5, 0, 99]) == (1, 0, 3, 5, 0)   # appended columns ignored
    assert (RD.press_cost(st, 1, 0), RD.press_cost(st, 1, 3), RD.press_cost(st, 1, 4)) == (3, 0, 0)
    assert (RD.button_card(st, 1, 0), RD.button_card(st, 1, 3), RD.button_card(st, 1, 9)) == (
        MUSKETEER, -1, -1)


def test_engine_pressing_and_the_pending_dataclass():
    st = hero_state(pending=[[0, 2, 100, 200, 12, 4], [1, 5, 0, 0, 3, 2], [1, 4, 0, 0, 1, 3],
                             [1, 2, 0, 0, 1, 0], [1]])
    assert RD.engine_pressing(st, 1) == frozenset({0, 1})     # what 4 and 5; what 2 is no button
    assert RD.engine_pressing(hero_state(), 1) == frozenset()
    # engine_pending is what it was: a press row is Pending(-1, cost), no button
    assert RD.engine_pending(hero_state(pending=[[1, 5, 0, 0, 3, 2]]), 1) == [RD.Pending(-1, 2)]
    tap = RD.Pending(-1, 2, tick=490, button=0)
    assert tap.is_press and not RD.Pending(5, 4).is_press
    assert RD.Pending(6, 1) == RD.Pending(6, 1, -1, None, -1)


# ---------------------------------------------------------------------------
# decide: presses off, argmax presses, pending presses, fallback, sampling
# ---------------------------------------------------------------------------

def test_presses_off_masks_every_button_and_decides_as_before():
    """Presses off (every preset's default): buttons at 50 never win, and the decision is the
    one a five-entry gate [wait, slots] gets, field for field but the mask (and gate_q)."""
    pol = FakePolicy([-5, 3, 2, 1, 0, 50, 50])
    five = FakePolicy([-5, 3, 2, 1, 0])
    st = hero_state(abilities=BUTTONS)
    for cfg in (RD.DecideConfig.legacy(0.0), RD.DecideConfig.live(0.0)):
        assert not cfg.presses
        a = RD.decide(pol, ROW, st, 1, always, cfg, rng=StubRng(0.99))
        b = RD.decide(five, ROW, st, 1, always, cfg, rng=StubRng(0.99))
        assert a.kind == RD.PLAY and a.mask[5:] == (False, False) and a.button == -1
        assert a.value == () and a.slot == 0
        assert dataclasses.replace(a, mask=b.mask) == b
        # presses on cannot change a five-entry (legacy) gate
        c = RD.decide(five, ROW, st, 1, always, cfg.replace(presses=True), rng=StubRng(0.99))
        assert c == b
    cfg = RD.DecideConfig.sampling(0.0, cap_milli=0)
    a = RD.decide(pol, ROW, st, 1, always, cfg, rng=np.random.default_rng(0))
    b = RD.decide(five, ROW, st, 1, always, cfg, rng=np.random.default_rng(0))
    assert a.kind == RD.PLAY and a.gate_q[5:] == (0.0, 0.0) and a.gate_q[:5] == b.gate_q
    assert dataclasses.replace(a, mask=b.mask, gate_q=b.gate_q) == b


def test_an_argmax_on_a_button_presses_it():
    """Gate [-5, 3, 2, 1, 0 | 1, 6]: button 1 (the Ice Golem, 2 elixir) has the best logit."""
    pol = FakePolicy([-5, 3, 2, 1, 0, 1, 6])
    st = hero_state(abilities=BUTTONS)
    bat = FakeBattle()
    d = RD.decide(pol, ROW, st, 1, bat, RD.DecideConfig.live(0.0, presses=True),
                  rng=StubRng(0.99))
    assert d.kind == RD.PRESS and d.is_press and not d.is_play
    assert (d.button, d.card, d.cost, d.slot, d.tile) == (1, ICE_GOLEM, 2, -1, -1)
    assert d.command == (1, RD.HAND_SIZE + 1, 0, 0) == (1, 5, 0, 0)
    assert d.mask == (True,) * 7 and d.spendable_milli == 9000
    assert pol.tile_calls == [] and bat.deploy_checks == []      # no tile, no legality call
    assert d.tried == () and not d.fallback and d.legality == ""
    assert d.world_tile is None and d.model_tile is None and d.world_mt is None
    rec = json.loads(json.dumps(d.to_record()))
    assert rec["command"] == [1, 5, 0, 0] and rec["kind"] == "press" and rec["button"] == 1

    # the control state shows the press and never taps it, in AUTO with taps on
    from scripts.vision import control as C
    kw = d.suggest_kwargs()
    assert kw == {"tick": 500, "p_wait": d.p_wait, "card": -1, "reason": "press", "button": 1,
                  "button_card": ICE_GOLEM}
    ctl = C.ControlState(C.AUTO, taps_enabled=True, stop_file=None, control_file=None,
                         names=[r[0] for r in CAT])
    s = ctl.suggest(**kw)
    assert s.is_wait and s.reason == "press" and s.extra == {"button": 1, "button_card": ICE_GOLEM}
    assert not ctl.should_tap(s)
    play = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.live(0.0), rng=StubRng(0.99))
    assert play.is_play and ctl.should_tap(ctl.suggest(**play.suggest_kwargs()))


def test_a_button_alone_is_pressable_when_no_card_is_affordable():
    """2.5 elixir, every card 4: no card, button 0 (3) unaffordable, button 1 (2) affordable.
    p_wait = 1 / (1 + e^-2) = 0.8808, q = 0.8808 ** 0.5 = 0.9385: the draw 0.99 acts."""
    st = hero_state(abilities=BUTTONS, elixir=2500, costs=[4, 4, 4, 4])
    pol = FakePolicy([0, 3, 2, 1, 0, 9, -2])
    off = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.live(0.0), rng=StubRng(0.99))
    assert (off.kind, off.reason) == (RD.WAIT, RD.R_NO_CARD)
    rng = StubRng(0.99)
    on = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.live(0.0, presses=True), rng=rng)
    assert on.mask == (True, False, False, False, False, False, True)
    assert on.kind == RD.PRESS and (on.button, on.cost, on.card) == (1, 2, ICE_GOLEM)
    assert rng.draws == 1 and not on.capped
    assert on.p_wait == pytest.approx(1 / (1 + math.exp(-2)), rel=1e-5)
    assert on.q_wait == pytest.approx((1 / (1 + math.exp(-2))) ** 0.5, rel=1e-5)
    w = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.live(0.0, presses=True),
                  rng=StubRng(0.5))
    assert (w.kind, w.reason) == (RD.WAIT, RD.R_WAIT)
    # at full elixir the cap rule still needs a playable card: a button alone is not one.
    # Gate [5 | masked x 4 | 0, 0]: p_wait = e^5 / (e^5 + 2) = 0.9867, q = 0.9933 > 0.5: wait
    full = hero_state(abilities=BUTTONS, elixir=9800, costs=[11, 11, 11, 11])
    w = RD.decide(FakePolicy([5, 0, 0, 0, 0, 0, 0]), ROW, full, 1, always,
                  RD.DecideConfig.live(0.0, presses=True), rng=StubRng(0.5))
    assert (w.kind, w.reason, w.capped) == (RD.WAIT, RD.R_WAIT, False)
    assert w.mask == (True, False, False, False, False, True, True)
    assert w.q_wait == pytest.approx((math.exp(5) / (math.exp(5) + 2)) ** 0.5, rel=1e-5)


def test_a_press_in_flight_masks_its_button():
    """The engine holds a press of button 1 (what 5 = HAND_SIZE + 1, 2 elixir): live masks it
    (the engine would refuse CARD_PENDING); legacy has no pending mask, as for cards."""
    st = hero_state(abilities=BUTTONS, pending=[[1, 5, 0, 0, 3, 2]], pending_cost=2)
    pol = FakePolicy([-5, 3, 2, 1, 0, 1, 6])
    live = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.live(0.0, presses=True),
                     rng=StubRng(0.99))
    assert live.mask[5:] == (True, False) and live.spendable_milli == 7000
    assert live.kind == RD.PLAY and live.slot == 0                 # slot 0 (3) > button 0 (1)
    legacy = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.legacy(0.0, presses=True),
                       rng=StubRng(0.99))
    assert legacy.mask[5:] == (True, True) and legacy.kind == RD.PRESS and legacy.button == 1

    # the live runner's own list: a press tap of button 0 blocks it and holds its 2 elixir
    tap = RD.Pending(-1, 2, tick=490, button=0)
    free = hero_state(abilities=BUTTONS)
    d = RD.decide(FakePolicy([-5, 0, 0, 0, 0, 8, 6]), ROW, free, 1, always,
                  RD.DecideConfig.live(0.0, presses=True), rng=StubRng(0.99), pending=[tap])
    assert d.spendable_milli == 7000 and d.mask[5:] == (False, True)
    assert d.kind == RD.PRESS and d.button == 1 and d.command == (1, 5, 0, 0)


def test_the_fallback_goes_on_to_a_button():
    """Gate order over [slots | buttons] = [9, 0, 0, 0 | 5, -1]: slot 0, then button 0. Slot 0
    has no legal tile: live presses button 0; legacy (no fallback) waits."""
    pol = FakePolicy([-5, 9, 0, 0, 0, 5, -1])
    st = hero_state(abilities=BUTTONS)

    def refuse_slot0(team, slot, x, y):
        return OUT_OF_TERRITORY if slot == 0 else 0

    bat = FakeBattle(deploy=refuse_slot0)
    d = RD.decide(pol, ROW, st, 1, bat, RD.DecideConfig.live(0.0, presses=True),
                  rng=StubRng(0.99))
    assert d.kind == RD.PRESS and (d.button, d.tried, d.fallback) == (0, (0,), True)
    assert d.legality == "battle" and d.command == (1, 4, 0, 0)
    assert (d.card, d.cost) == (MUSKETEER, 3)
    assert pol.tile_calls == [0] and len(bat.deploy_checks) == 60
    assert {c[1] for c in bat.deploy_checks} == {0}
    old = RD.decide(pol, ROW, st, 1, FakeBattle(deploy=refuse_slot0),
                    RD.DecideConfig.legacy(0.0, presses=True), rng=StubRng(0.99))
    assert (old.kind, old.reason, old.tried) == (RD.WAIT, RD.R_NO_TILE, (0,))


def test_sampling_draws_a_press_reproducibly():
    """Gate [-50 x 5 | 50, 0]: q(button 0) = 1 - 2e-22, so seed 3 draws it; no tile is drawn."""
    pol = FakePolicy([-50, -50, -50, -50, -50, 50, 0])
    st = hero_state(abilities=BUTTONS)
    cfg = RD.DecideConfig.sampling(0.0, presses=True)
    assert cfg.presses and cfg.sample and cfg.name == RD.MODE_SAMPLE
    runs = [RD.decide(pol, ROW, st, 1, FakeBattle(), cfg, rng=np.random.default_rng(3))
            for _ in range(2)]
    assert runs[0] == runs[1]
    d = runs[0]
    assert d.kind == RD.PRESS and (d.button, d.gate_a) == (0, 5) and d.command == (1, 4, 0, 0)
    assert d.gate_q[5] > 0.99 and sum(d.gate_q) == pytest.approx(1.0)
    assert d.logq_gate == pytest.approx(math.log(d.gate_q[5]))
    assert d.tile_cands == () and d.tile_a == -1 and pol.tile_calls == []
    assert d.mode == RD.MODE_SAMPLE and d.legality == ""
    # presses off: q is 0 on the buttons and a slot is drawn
    off = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.sampling(0.0, cap_milli=0),
                    rng=np.random.default_rng(3))
    assert off.kind == RD.PLAY and off.gate_q[5:] == (0.0, 0.0)


# ---------------------------------------------------------------------------
# the critic's value
# ---------------------------------------------------------------------------

def test_the_critic_value_goes_into_the_record():
    st = hero_state()
    pol = ValuePolicy([-5, 3, 2, 1, 0], (0.25, -1.0))
    d = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.live(0.0), rng=StubRng(0.99))
    assert d.kind == RD.PLAY and d.value == (0.25, -1.0)
    assert d.win_p == pytest.approx(1 / (1 + math.e))
    assert json.loads(json.dumps(d.to_record()))["value"] == [0.25, -1.0]
    w = RD.decide(ValuePolicy([50, 0, 0, 0, 0], (0.5, 2.0)), ROW, st, 1, always,
                  RD.DecideConfig.live(0.0), rng=StubRng(0.5))
    assert w.kind == RD.WAIT and w.value == (0.5, 2.0)
    assert w.win_p == pytest.approx(1 / (1 + math.exp(-2)))
    s = RD.decide(ValuePolicy([0, 3, 2, 1, 0], (0.1, 0.0)), ROW, st, 1, always,
                  RD.DecideConfig.sampling(0.0), rng=np.random.default_rng(1))
    assert s.value == (0.1, 0.0) and s.win_p == 0.5
    plain = RD.decide(FakePolicy([-5, 3, 2, 1, 0]), ROW, st, 1, always,
                      RD.DecideConfig.live(0.0), rng=StubRng(0.99))
    assert plain.value == () and plain.win_p is None
    assert RD.Decision(team=1, tick=0, value=(0.0, -1000.0)).win_p == 0.0       # no overflow
    assert RD.Decision(team=1, tick=0, value=(0.0, 1000.0)).win_p == 1.0


class LegacyModel:
    """A TokenPolicy stand-in without a critic: forward() only."""

    def __init__(self):
        self.cards: list[list[int]] = []

    def __call__(self, b, card_ids=None):
        self.cards.append(card_ids.numpy().tolist())
        return (FakeTensor(np.array([[1, 2, 3, 4, 5]], np.float32)),
                FakeTensor(ranked()[None]), FakeTensor(np.array([[9.0, 9.0]], np.float32)))


class CriticModel:
    """A v2 stand-in with a critic: the gate and value2 from outputs(), tiles from forward()."""

    has_critic = True

    def __init__(self, gate):
        self.g = np.asarray([gate], np.float32)
        self.forward_cards: list[int] = []
        self.outputs_cards: list[int] = []

    def __call__(self, b, card_ids=None):
        self.forward_cards.append(int(card_ids.numpy()[0]))
        return FakeTensor(self.g), FakeTensor(ranked()[None]), FakeTensor(np.zeros((1, 2)))

    def outputs(self, b, card_ids=None):
        self.outputs_cards.append(int(card_ids.numpy()[0]))
        return {"gate": FakeTensor(self.g), "tile": FakeTensor(ranked()[None]),
                "value": FakeTensor(np.zeros((1, 2))), "cls": None,
                "value2": FakeTensor(np.array([[0.5, -2.0]], np.float32))}


def test_the_adapter_takes_the_gate_and_the_critic_from_one_pass():
    b = {"hand": FakeTensor(np.array([[4, 7, 8, 2]]))}
    legacy = LegacyModel()
    a = RD.TokenPolicyAdapter(legacy, FAKE_TORCH)
    g, v = a.heads(b)
    assert g.tolist() == [1, 2, 3, 4, 5] and v is None
    assert not a.has_critic and a.schema is None
    assert a.gate(b).tolist() == [1, 2, 3, 4, 5]
    assert legacy.cards == [[4], [4]]                # card_ids = hand slot 0, as RSAgent.act

    model = CriticModel([-5, 3, 2, 1, 0])
    c = RD.TokenPolicyAdapter(model, FAKE_TORCH, batch=lambda row: b)
    assert c.has_critic
    g, v = c.heads(b)
    assert g.tolist() == [-5, 3, 2, 1, 0] and v == (0.5, -2.0)
    assert model.outputs_cards == [4] and model.forward_cards == []
    d = RD.decide(c, ROW, hero_state(), 1, always, RD.DecideConfig.live(0.0), rng=StubRng(0.99))
    assert d.kind == RD.PLAY and d.slot == 0 and d.value == (0.5, -2.0)
    assert model.outputs_cards == [4, 4] and model.forward_cards == [4]     # the tiles of slot 0
    # has_critic without outputs() is not a critic: forward() as always
    odd = LegacyModel()
    odd.has_critic = True
    assert not RD.TokenPolicyAdapter(odd, FAKE_TORCH).has_critic


# ---------------------------------------------------------------------------
# the v2 batch
# ---------------------------------------------------------------------------

def test_the_v2_batch_has_every_model_input():
    row = v2_row()
    b = RD.row_batch(row, FAKE_TORCH, "v2")
    assert list(b) == list(RS.V2.model_inputs)
    for name in RS.V2.model_inputs:
        want = np.int64 if RS.V2.dtype(name).startswith(("int", "bool")) else np.float32
        assert b[name].numpy().dtype == want, name
        assert b[name].numpy().shape == (1, *RS.V2.shape(name, buttons=2)), name
    assert b["next"].numpy().tolist() == [5] and b["hand"].numpy().tolist() == [[4, 7, 8, 2]]
    assert b["ability_card"].numpy().tolist() == [[7, 0]]
    assert b["scalars"].numpy()[0, 5] == np.float32(0.4785)
    for schema in (RS.V2, RS.SCHEMA_V2):
        assert list(RD.row_batch(row, FAKE_TORCH, schema)) == list(RS.V2.model_inputs)
    # legacy (None, "legacy"): the legacy keys, exactly as before
    keys = [*RD.BATCH_KEYS, "next"]
    for schema in (None, "legacy", RS.LEGACY):
        old = RD.row_batch(row, FAKE_TORCH, schema)
        assert list(old) == keys
        assert old["unit_feats"].numpy().shape == (1, 48, 9)        # the row's own width
    # the adapter's own batch follows the model's schema
    v2_model = SimpleNamespace(schema=RS.V2)
    assert list(RD.TokenPolicyAdapter(v2_model, FAKE_TORCH).batch(row)) == list(RS.V2.model_inputs)
    assert RD.model_schema_of(v2_model) is RS.V2
    for legacy in (SimpleNamespace(schema=RS.LEGACY), SimpleNamespace(), LegacyModel()):
        assert RD.model_schema_of(legacy) is None
        assert list(RD.TokenPolicyAdapter(legacy, FAKE_TORCH).batch(row)) == keys


def test_the_v2_batch_is_model_tokens_row_batch():
    torch = pytest.importorskip("torch")
    from sim.model_tokens import row_batch as mt_row_batch
    row = v2_row()
    ours, theirs = RD.row_batch(row, torch, "v2"), mt_row_batch(row, "v2")
    assert list(ours) == list(theirs)
    for k in ours:
        assert ours[k].dtype == theirs[k].dtype and torch.equal(ours[k], theirs[k]), k


def test_a_v2_token_policy_through_the_adapter_and_decide():
    torch = pytest.importorskip("torch")
    from sim.model_tokens import MASK_LOGIT, TokenPolicy
    torch.manual_seed(0)
    model = TokenPolicy(18, 20, d=16, layers=1, heads=2, schema="v2").eval()
    assert model.v2 and model.has_critic
    adapter = RD.TokenPolicyAdapter(model)
    assert adapter.schema is model.schema and adapter.has_critic
    row = v2_row()
    b = adapter.batch(row)
    assert list(b) == list(RS.V2.model_inputs)
    g, value = adapter.heads(b)
    with torch.no_grad():
        out = model.outputs(b, card_ids=b["hand"][:, 0].clamp(min=0))
    assert g.shape == (7,) and np.array_equal(g, out["gate"][0].numpy())
    assert value == tuple(float(v) for v in out["value2"][0]) and len(value) == 2
    assert g[6] == MASK_LOGIT                         # ability_card 0: no button
    st = hero_state(abilities=BUTTONS[:1], elixir=9800)        # capped: never a wait
    d = RD.decide(adapter, row, st, 1, always, RD.DecideConfig.live(0.0, presses=True),
                  rng=StubRng(0.99))
    assert d.capped
    assert d.value == value and len(d.mask) == 7 and d.mask[5:] == (True, False)
    assert d.kind in (RD.PLAY, RD.PRESS) and (d.command[1] == 4) == d.is_press


# ---------------------------------------------------------------------------
# own pending inputs
# ---------------------------------------------------------------------------

@pytest.fixture
def ev(monkeypatch):
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    mod = importlib.import_module("rs_eval")
    assert mod.rs_decide is RD
    return mod


def test_pending_inputs_from_the_live_taps(ev):
    """delay 24 at tick 500: taps at 480 -> 4, 495 -> 19, unknown -> 24, 450 -> 0 (clipped);
    the press and the tap at 500 (the decision's own) are left out; soonest first."""
    taps = [RD.Pending(5, 4, 1, 480), RD.Pending(2, 4, 0, 495),
            RD.Pending(-1, 2, tick=490, button=0), RD.Pending(6, 1, -1, None),
            RD.Pending(7, 2, -1, 500), RD.Pending(9, 3, 2, 450)]
    got = RD.pending_inputs(taps, 500, 24)
    assert got == [(11, 0), (7, 4), (4, 19), (8, 24)]
    assert RD.pending_inputs(taps, 500, 0) == [] and RD.pending_inputs([], 500, 24) == []
    cards, land = ev.RR.pending_features(got)
    assert cards.tolist() == [11, 7, 4, 8]
    assert land.tolist() == [0.0, np.float16(4 / 40), np.float16(19 / 40), np.float16(24 / 40)]


def test_engine_pending_inputs_are_rsagent_own_pending(ev):
    me = player_dict(0, pending=[[0, 2, 0, 0, 12, 4], [1, 5, 0, 0, 3, 1], [0, 5, 0, 0, 5, 4],
                                 [0, -1, 0, 0, 2, 0], [0, 6, 0, 0]], pending_cost=8)
    state = state_dict(100, players=[me, player_dict(1)])
    assert RD.engine_pending_inputs(state, 0) == [(7, 5), (4, 12)]
    assert RD.engine_pending_inputs(state, 0, 6) == [(4, 6)]
    assert RD.engine_pending_inputs(state, 0, 13) == []
    assert RD.engine_pending_inputs(state, 1) == []
    agent = object.__new__(ev.RSAgent)
    for lead in (0, 5, 6, 12, 13):
        agent.lead = lead
        assert RD.engine_pending_inputs(state, 0, lead) == agent.own_pending(state, 0), lead


def test_v2_row_kwargs_and_the_checkpoint_meta():
    assert RD.v2_row_kwargs((4785, True), [(7, 5), (4, 12)], buttons=[5]) == {
        "schema": "rs_rows/v2", "opp_elixir": (4785, True), "pending": [(7, 5), (4, 12)],
        "buttons": [5]}
    assert RD.v2_row_kwargs(3000) == {"schema": "rs_rows/v2", "opp_elixir": 3000, "pending": []}
    assert RD.obs_delay_ticks({"obs_delay": {"ticks": 24, "jitter": 3, "opp_lag": 1}}) == 24
    for meta in ({}, None, {"obs_delay": {"ticks": "lots"}}, {"obs_delay": 24},
                 {"obs_delay": {"ticks": -3}}, {"obs_delay": {"ticks": None}}):
        assert RD.obs_delay_ticks(meta) == 0, meta
    assert RD.is_v2_meta({"schema": " rs_rows/v2 "}) and RD.is_v2_meta({"schema": "v2"})
    for meta in ({}, None, {"schema": "rs_rows/legacy"}, {"schema": 2}):
        assert not RD.is_v2_meta(meta), meta


def test_prune_pending_keeps_a_press_until_it_runs():
    press = RD.Pending(-1, 2, tick=100, button=1)
    play = RD.Pending(5, 4, 1, 100)
    hand = [9, 5, 6, 0]
    assert RD.prune_pending([press, play], hand, 150) == [press, play]
    assert RD.prune_pending([press], hand, 160) == []                  # 60 ticks: lost
    up = [list(BUTTONS[0]), list(BUTTONS[1])]
    ran = [list(BUTTONS[0]), [0, 1, 2, ICE_GOLEM, 0]]                  # the hero's charge spent
    assert RD.prune_pending([press, play], hand, 150, abilities=up) == [press, play]
    assert RD.prune_pending([press, play], hand, 150, abilities=ran) == [play]
    assert RD.prune_pending([press], hand, 150, abilities=[]) == []
    assert RD.prune_pending([RD.Pending(-1, 2, button=0)], hand, 10_000) == [
        RD.Pending(-1, 2, button=0)]


# ---------------------------------------------------------------------------
# configuration and the checkpoint's press training
# ---------------------------------------------------------------------------

def test_presets_take_presses_and_default_to_off():
    for cfg in (RD.DecideConfig(), RD.DecideConfig.legacy(), RD.DecideConfig.live(),
                RD.DecideConfig.sampling()):
        assert not cfg.presses
    assert RD.DecideConfig.legacy(1.0, presses=True).presses
    assert RD.DecideConfig.live(1.0, presses=True) == RD.DecideConfig.live(1.0).replace(
        presses=True)
    s = RD.DecideConfig.sampling(1.0, presses=True)
    assert s.presses and s.sample and s == RD.DecideConfig.sampling(1.0).replace(presses=True)
    for mode in RD.MODES:
        assert RD.DecideConfig.for_mode(mode, 1.0, presses=True).presses, mode
        assert not RD.DecideConfig.for_mode(mode, 1.0).presses, mode
    assert RD.DecideConfig.for_mode(RD.MODE_SAMPLE, 1.0) == RD.DecideConfig.sampling(1.0)


def test_presses_allowed_reads_the_press_training():
    rows = {"trust": "legacy", "trust_column": "trusted", "press_rows": "trained",
            "press_rows_excluded": 0, "press_rows_trained": 120}
    v2 = {"schema": "rs_rows/v2", "train_rows": rows}
    assert RD.presses_allowed(v2) and RD.presses_allowed({**v2, "schema": "v2"})
    assert not RD.presses_allowed({**v2, "train_rows": {**rows, "press_rows": "none"}})
    assert not RD.presses_allowed({**v2, "train_rows": {**rows, "press_rows": "excluded"}})
    assert not RD.presses_allowed({"schema": "rs_rows/v2"})      # no press rows: no train_rows
    assert not RD.presses_allowed({**v2, "schema": "rs_rows/legacy"})
    assert not RD.presses_allowed({"train_rows": rows})          # no schema: legacy
    assert not RD.presses_allowed(None)
    assert not RD.presses_allowed({"schema": "rs_rows/v2", "train_rows": "trained"})


def test_presses_allowed_on_train_tokens_meta(tmp_path):
    pytest.importorskip("torch")
    from sim import train_tokens as TT
    assert (TT.TRAIN_ROWS_KEY, TT.PRESS_TRAINED) == (RD.TRAIN_ROWS_KEY, RD.PRESS_TRAINED)
    vocab = {"schema": "rs_rows/v2", "n_cards": 18, "n_units": 20}
    labelled = TT._record(RS.V2, "legacy", "trusted", True, 7, tmp_path)
    assert labelled["press_rows"] == TT.PRESS_TRAINED
    assert RD.presses_allowed(TT.checkpoint_meta(vocab, 16, 1, 2, labelled))
    unlabelled = TT._record(RS.V2, "legacy", "trusted", False, 0, tmp_path)   # no vocab.json
    meta = TT.checkpoint_meta(vocab, 16, 1, 2, unlabelled)
    assert TT.TRAIN_ROWS_KEY not in meta and not RD.presses_allowed(meta)
    legacy = TT._record(RS.LEGACY, "legacy", "trusted", True, 7, tmp_path)
    assert not RD.presses_allowed(TT.checkpoint_meta({"n_cards": 18, "n_units": 20}, 16, 1, 2,
                                                     legacy))
