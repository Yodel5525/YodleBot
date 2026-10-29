"""V06: tools/rs_decide.py, the one decision function (rs_eval, the live runner, RL).

Pure tests with scripted logits (FakePolicy) and conftest's FakeBattle: a pending slot is masked
and the next slot plays (missed issue 3), the fallback order, spendable elixir net of our plays
in flight, prune_pending, the p_wait ** (dt / 20) cadence and its one draw per decision, the cap
rule, the lane guard, the towers-only legality battle on a fake royalesim (deck order, reset,
cache, reuse, fallbacks), the pure-python legality tile by tile, suggestions for
scripts/vision/control.py, the sampling mode, the E15 button hook and the wait bias.

Golden: tools/rs_eval.RSAgent.act, now built on decide, against a verbatim copy of the act it
had before V06, on 300 random states per flag combination (live-compat rows, --spell-aim,
--hold-pending): the same command, the same check_deploy calls in the same order and the same
random draws. numpy only, with a numpy stand-in for torch; again with a small real TokenPolicy
when torch imports.

Engine-marked: the towers-only battle on royalesim (own half legal, enemy half not, the pocket
opens when the enemy's world-left princess falls), the pure legality never offering a tile the
engine refuses, and ARENA_TILES against royalesim's arena.json.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import math
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import (DEFAULT_CATALOGUE, DEPLOY_REASONS, FIELD, KING_HP,
                                     PRINCESS_HP, FakeBattle, catalogue_row, entity_row,
                                     make_fake_royalesim, player_dict, state_dict)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for _path in (str(ROOT), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import rs_decide as RD  # noqa: E402  (the name rs_eval imports it by)
import rs_engine  # noqa: E402

SUB = 18
CAT = [list(r) for r in DEFAULT_CATALOGUE]
CID = {row[0]: i for i, row in enumerate(CAT)}
#: HogRider 2 (4 elixir), Musketeer 5 (4), Skeletons 6 (1), Cannon 0 (3).
HAND = [CID["HogRider"], CID["Musketeer"], CID["Skeletons"], CID["Cannon"]]
CARD_PENDING = DEPLOY_REASONS.index("CARD_PENDING")
OUT_OF_TERRITORY = DEPLOY_REASONS.index("OUT_OF_TERRITORY")


@pytest.fixture(autouse=True)
def clean_engine(monkeypatch):
    """No HASTYCR_* engine variable from the PC, and no active config left from a test."""
    for var in (rs_engine.CONFIG_ENV, *rs_engine.ENV.values()):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rs_engine, "_ACTIVE", None)


def T(tx: int, ty: int) -> int:
    """Model tile index (own frame)."""
    return ty * 18 + tx


def ranked(*tiles: int) -> np.ndarray:
    """Tile logits: `tiles` best first, in that order, then every other tile by index."""
    t = -10.0 - np.arange(RD.TILES, dtype=np.float32) / 1000
    for i, k in enumerate(tiles):
        t[k] = 100.0 - i
    return t


class FakePolicy:
    """decide()'s policy interface with scripted logits."""

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


class StubRng:
    """rng.random() always `u`; counts the draws."""

    def __init__(self, u: float):
        self.u = u
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return self.u


def live_state(*, team=1, hand=HAND, elixir=9000, costs=None, pending=(), pending_cost=0,
               entities=(), enemy_towers=None, tick=500, **me):
    """A state_json dict with `team` (default the live seat, 1) holding `hand`."""
    mine = player_dict(team, hand=list(hand), elixir_milli=elixir, hand_costs=costs,
                       pending=[list(r) for r in pending], pending_cost=pending_cost, **me)
    theirs = (player_dict(1 - team) if enemy_towers is None
              else player_dict(1 - team, tower_hp=enemy_towers))
    players = [mine, theirs] if team == 0 else [theirs, mine]
    return state_dict(tick, players=players, entities=entities)


ROW = {"tick": 500}


def always(slot, card, x, y):
    return True


# ---------------------------------------------------------------------------
# masks, pending, fallback
# ---------------------------------------------------------------------------

def test_a_pending_card_is_masked_and_the_next_slot_plays():
    """Missed issue 3: under a delay the engine keeps our HogRider in hand and refuses it
    CARD_PENDING. Legacy picks it, fails all 60 tiles and loses the decision; live masks the
    slot and plays the next card."""
    hog = CID["HogRider"]
    st = live_state(pending=[[0, hog, 0, 0, 12, 4]], pending_cost=4, elixir=9000)
    pol = FakePolicy([-5, 3, 2, 1, 0], {1: ranked(T(9, 24))})

    def refuse_pending(team, slot, x, y):
        return CARD_PENDING if slot == 0 else 0

    old = FakeBattle(deploy=refuse_pending)
    d = RD.decide(pol, ROW, st, 1, old, RD.DecideConfig.legacy(0.0), rng=StubRng(0.99))
    assert (d.kind, d.reason, d.tried, d.command) == (RD.WAIT, RD.R_NO_TILE, (0,), None)
    assert len(old.deploy_checks) == 60 and {c[1] for c in old.deploy_checks} == {0}
    assert d.spendable_milli == 5000 and d.mask == (True,) * 5

    new = FakeBattle(deploy=refuse_pending)
    pol.tile_calls.clear()
    d = RD.decide(pol, ROW, st, 1, new, RD.DecideConfig.live(0.0), rng=StubRng(0.99))
    assert d.mask == (True, False, True, True, True)
    assert d.kind == RD.PLAY and (d.slot, d.card, d.tried, d.fallback) == (
        1, CID["Musketeer"], (1,), False)
    assert (d.tx, d.ty, d.cost) == (9, 24, 4)
    assert d.command == (1, 1, 8500 * 18, 24500 * 18) == (1, 1, 153000, 441000)
    assert new.deploy_checks == [(1, 1, 153000, 441000)]
    assert pol.tile_calls == [1] and d.legality == "battle" and d.mode == RD.MODE_LIVE


def test_no_legal_tile_hands_over_to_the_next_slot_by_gate_logit():
    """Slot logits [3, 0, 2, 1]: slot 0, then 2, then 3. Slots 0 and 2 have no legal tile."""
    st = live_state()
    pol = FakePolicy([-5, 3, 0, 2, 1], {3: ranked(T(9, 24))})

    def refuse(team, slot, x, y):
        return OUT_OF_TERRITORY if slot in (0, 2) else 0

    old = FakeBattle(deploy=refuse)
    d = RD.decide(pol, ROW, st, 1, old, RD.DecideConfig.legacy(0.0), rng=StubRng(0.99))
    assert (d.kind, d.reason, d.tried) == (RD.WAIT, RD.R_NO_TILE, (0,))
    assert len(old.deploy_checks) == 60

    new = FakeBattle(deploy=refuse)
    pol.tile_calls.clear()
    d = RD.decide(pol, ROW, st, 1, new, RD.DecideConfig.live(0.0), rng=StubRng(0.99))
    assert d.kind == RD.PLAY and (d.slot, d.card, d.tried, d.fallback) == (
        3, CID["Cannon"], (0, 2, 3), True)
    assert pol.tile_calls == [0, 2, 3]
    assert [c[1] for c in new.deploy_checks] == [0] * 60 + [2] * 60 + [3]


def test_spendable_elixir_is_net_of_our_plays_in_flight():
    hand_costs = (4, 4, 1, 3)
    tap = RD.Pending(card=CID["Musketeer"], cost=4, slot=1, tick=480)
    # 5.0 elixir, a 4-elixir tap in flight: 1.0 left, only the Skeletons. The state's own
    # pending_cost (3) is the engine's; an explicit list replaces it.
    st = live_state(elixir=5000, pending_cost=3)
    pol = FakePolicy([-5, 1, 3, 2, 0])
    d = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.live(0.0), rng=StubRng(0.99),
                  pending=[tap])
    assert st["players"][1]["hand_costs"] == list(hand_costs)
    assert d.spendable_milli == 1000 and d.mask == (True, False, False, True, False)
    assert (d.slot, d.card, d.legality) == (2, CID["Skeletons"], "callable")
    # pending=None reads the engine's pending_cost: 5.0 - 3 = 2.0 elixir, still the Skeletons
    d = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.live(0.0), rng=StubRng(0.99))
    assert d.spendable_milli == 2000 and d.mask == (True, False, False, True, False)

    # 9.0 elixir: everything affordable; live blocks the tap's slot 1, legacy does not
    st9 = live_state(elixir=9000)
    d = RD.decide(pol, ROW, st9, 1, always, RD.DecideConfig.live(0.0), rng=StubRng(0.99),
                  pending=[tap])
    assert d.spendable_milli == 5000 and d.mask == (True, True, False, True, True)
    assert d.slot == 2
    d = RD.decide(pol, ROW, st9, 1, always, RD.DecideConfig.legacy(0.0), rng=StubRng(0.99),
                  pending=[tap])
    assert d.spendable_milli == 5000 and d.mask == (True,) * 5 and d.slot == 1

    # the tap ran: its card left slot 1 (an Archer came in), so it neither costs nor blocks
    moved = live_state(elixir=9000, hand=[2, CID["Archer"], 6, 0])
    left = RD.prune_pending([tap], moved["players"][1]["hand"], 500)
    assert left == []
    d = RD.decide(pol, ROW, moved, 1, always, RD.DecideConfig.live(0.0), rng=StubRng(0.99),
                  pending=left)
    assert d.spendable_milli == 9000 and d.mask == (True,) * 5
    assert (d.slot, d.card) == (1, CID["Archer"])


def test_prune_pending_follows_the_hand_and_the_age():
    keep = RD.Pending(5, 4, 1, 100)
    gone = RD.Pending(2, 4, 0, 100)           # slot 0 now holds card 9
    old = RD.Pending(6, 1, -1, 50)            # any slot; but 100 ticks old at 150
    hand = [9, 5, 6, 0]
    assert RD.prune_pending([keep, gone, old], hand, 150) == [keep]
    assert RD.prune_pending([keep], hand, 159) == [keep]
    assert RD.prune_pending([keep], hand, 160) == []          # 60 ticks: lost
    assert RD.prune_pending([RD.Pending(6, 1, -1, None)], hand, 10_000) == [RD.Pending(6, 1)]
    assert RD.prune_pending([RD.Pending(7, 2, -1, 140)], hand, 150) == []


def test_engine_pending_rows():
    st = live_state(pending=[[0, 2, 100, 200, 12, 4], [1, 5, 0, 0, 3, 2], [0]])
    assert RD.engine_pending(st, 1) == [RD.Pending(2, 4), RD.Pending(-1, 2)]
    assert RD.engine_pending(live_state(), 1) == []


def test_unplayable_and_empty_slots():
    """hand_costs -1 (a Mirror with nothing to copy) is masked live only; an empty slot (-1)
    and a missing hand_costs entry (99) always."""
    st = live_state(hand=[2, -1, 6, 0], costs=[-1, 3, 1])
    pol = FakePolicy([-5, 3, 2, 1, 0])
    d = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.legacy(0.0), rng=StubRng(0.99))
    assert d.mask == (True, True, False, True, False) and d.slot == 0
    d = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.live(0.0), rng=StubRng(0.99))
    assert d.mask == (True, False, False, True, False) and d.slot == 2
    none = live_state(hand=[-1, -1, -1, -1])
    rng = StubRng(0.0)
    d = RD.decide(pol, ROW, none, 1, always, RD.DecideConfig.live(0.0), rng=rng)
    assert (d.kind, d.reason, rng.draws) == (RD.WAIT, RD.R_NO_CARD, 1)


# ---------------------------------------------------------------------------
# cadence, cap rule, lane guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt, kind", [(10, RD.WAIT), (20, RD.PLAY), (1, RD.WAIT)])
def test_wait_cadence_is_p_wait_to_the_dt_over_20(dt, kind):
    """All logits 0: p_wait 0.2. q = 0.2 ** (dt / 20): 0.447 (dt 10), 0.2 (20), 0.923 (1).
    The draw is 0.40, so the bot waits at 10 and 1 ticks and plays at 20."""
    pol = FakePolicy([0, 0, 0, 0, 0])
    rng = StubRng(0.40)
    d = RD.decide(pol, ROW, live_state(), 1, always, RD.DecideConfig.live(0.0), rng=rng, dt=dt)
    assert d.p_wait == pytest.approx(0.2, rel=1e-6)
    assert d.q_wait == pytest.approx(0.2 ** (dt / 20), rel=1e-6)
    assert d.kind == kind and rng.draws == 1
    if kind == RD.WAIT:
        assert d.reason == RD.R_WAIT and d.suggest_kwargs()["reason"] == "wait"


def test_wait_bias_and_default_cadence():
    """wait_keep 0.25: + log 4 on the wait logit, p_wait 4 / 8; cfg.dt 10 when dt is None."""
    pol = FakePolicy([0, 0, 0, 0, 0])
    d = RD.decide(pol, ROW, live_state(), 1, always, RD.DecideConfig.legacy(math.log(4)),
                  rng=StubRng(0.99))
    assert d.p_wait == pytest.approx(0.5, rel=1e-6)
    assert d.q_wait == pytest.approx(0.5 ** 0.5, rel=1e-6) and d.kind == RD.PLAY
    assert RD.cadence_dt(100, None) == 10 and RD.cadence_dt(100, 95) == 5
    assert RD.cadence_dt(100, 100) == 1 and RD.cadence_dt(100, 50) == 10


def test_cap_rule_plays_at_full_elixir():
    """p_wait = e^5 / (e^5 + 4) = 0.974, q = 0.987 > the draw 0.5: a wait, unless spendable
    elixir is at least 9.5 (live)."""
    pol = FakePolicy([5, 0, 0, 0, 0])
    live = RD.DecideConfig.live(0.0)
    d = RD.decide(pol, ROW, live_state(elixir=9600), 1, always, live, rng=StubRng(0.5))
    assert d.kind == RD.PLAY and d.capped
    d = RD.decide(pol, ROW, live_state(elixir=9400), 1, always, live, rng=StubRng(0.5))
    assert (d.kind, d.reason, d.capped) == (RD.WAIT, RD.R_WAIT, False)
    d = RD.decide(pol, ROW, live_state(elixir=9600), 1, always, RD.DecideConfig.legacy(0.0),
                  rng=StubRng(0.5))
    assert d.kind == RD.WAIT and not d.capped
    tap = RD.Pending(CID["Musketeer"], 4, 1, 490)          # 9.8 - 4 = 5.8 spendable
    d = RD.decide(pol, ROW, live_state(elixir=9800), 1, always, live, rng=StubRng(0.5),
                  pending=[tap])
    assert d.kind == RD.WAIT and not d.capped and d.spendable_milli == 5800


def test_lane_guard_keeps_the_cannon_in_the_threatened_lane():
    """An enemy Giant (2000 hp) at world (14000, 18000) is at own (4000, 18000) for team 1:
    lane 0. Own units, enemy buildings and enemies deep on their side do not count."""
    cannon, hog = CID["Cannon"], CID["HogRider"]
    ents = [entity_row(uid=40, team=0, kind=0, card_id=CID["Giant"], x=14000 * SUB,
                       y=18000 * SUB, hp=2000),
            entity_row(uid=41, team=1, kind=0, x=3000 * SUB, y=20000 * SUB, hp=5000),
            entity_row(uid=42, team=0, kind=1, x=3000 * SUB, y=20000 * SUB, hp=5000),
            entity_row(uid=43, team=0, kind=0, x=3000 * SUB, y=5000 * SUB, hp=5000)]
    st = live_state(hand=[cannon, hog, 6, 5], entities=ents)
    assert RD.threat_lane(st, 1, SUB) == 0
    guard = RD.card_ids(CAT, RD.GUARD_CARD_NAMES)
    assert guard == {cannon}
    cfg = RD.DecideConfig.live(0.0, guard_cards=guard)
    # lane 1 column 14 skipped, centre column 8 allowed
    pol = FakePolicy([-5, 3, 2, 1, 0], {0: ranked(T(14, 24), T(8, 24), T(3, 24))})
    d = RD.decide(pol, ROW, st, 1, always, cfg, rng=StubRng(0.99))
    assert (d.slot, d.tx, d.ty, d.lane) == (0, 8, 24, 0)
    pol = FakePolicy([-5, 3, 2, 1, 0], {0: ranked(T(14, 24), T(3, 24))})
    d = RD.decide(pol, ROW, st, 1, always, cfg, rng=StubRng(0.99))
    assert (d.tx, d.lane) == (3, 0)
    d = RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.live(0.0), rng=StubRng(0.99))
    assert (d.tx, d.lane) == (14, None)                       # guard off
    # a card not guarded keeps its tile
    st_hog = live_state(hand=[hog, cannon, 6, 5], entities=ents)
    d = RD.decide(pol, ROW, st_hog, 1, always, cfg, rng=StubRng(0.99))
    assert (d.card, d.tx, d.lane) == (hog, 14, None)


def test_threat_lane_needs_hp_and_a_clear_share():
    def giant(uid, x, hp):
        return entity_row(uid=uid, team=0, kind=0, x=x * SUB, y=18000 * SUB, hp=hp)
    assert RD.threat_lane(live_state(entities=[giant(40, 14000, 1100)]), 1) is None
    split = [giant(40, 14000, 1000), giant(41, 4000, 1000)]
    assert RD.threat_lane(live_state(entities=split), 1) is None
    lopsided = [giant(40, 14000, 400), giant(41, 4000, 1600)]   # own x 4000 / 14000
    assert RD.threat_lane(live_state(entities=lopsided), 1) == 1


def test_blocked_tiles_are_skipped():
    mask = np.zeros((32, 18), dtype=bool)
    mask[24, 9] = True                                          # [ty, tx], model frame
    pol = FakePolicy([-5, 3, 2, 1, 0], {0: ranked(T(9, 24), T(10, 24))})
    d = RD.decide(pol, ROW, live_state(), 1, always, RD.DecideConfig.live(0.0),
                  rng=StubRng(0.99), blocked_tiles=mask)
    assert (d.tx, d.ty) == (10, 24)


# ---------------------------------------------------------------------------
# the towers-only legality battle (fake royalesim)
# ---------------------------------------------------------------------------

def _fake_engine(deploy):
    """A fake royalesim whose battles deal the reset decks' first four cards as the hand."""
    def states(b):
        decks = (b.reset_kwargs or {}).get("decks")
        players = [player_dict(t, hand=list(decks[t][:4]) if decks else None) for t in (0, 1)]
        return state_dict(b.tick, players=players)
    return make_fake_royalesim(states=states, deploy=deploy)


def test_towers_only_battle_resets_with_towers_hand_and_full_elixir():
    fake = _fake_engine(lambda team, slot, x, y: 0 if slot == 2 else OUT_OF_TERRITORY)
    leg = RD.TowersOnlyLegality(rs_engine.EngineConfig.legacy(), fake)
    st = live_state(hand=[5, 2, 7, 0], next_card=3, deck=list(range(8)))
    check = leg.bind(st, 1)
    assert check.name == "towers" and len(fake.battles) == 1
    b = fake.battles[0]
    kw = b.reset_kwargs
    assert kw["decks"] == [[5, 2, 7, 0, 3, 1, 4, 6]] * 2       # hand, next, rest of the deck
    assert kw["start_tick"] == RD.LEGAL_START_TICK == 100 and kw["start_tick"] >= 90
    assert kw["elixir_milli"] == [10_000, 10_000] and kw["spawns"] == []
    assert kw["tower_hp"] == [[1, 1, 1], [1, 1, 1]] and (kw["seed"], kw["shuffle"]) == (0, 0)
    assert b.names() == ["reset"]                               # legacy config: no delay call
    assert check(0, 5, 100, 200) is False                       # Musketeer: battle slot 0
    assert check(2, 7, 100, 200) is True                        # Log: battle slot 2
    assert check(3, 7, 300, 400) is True                        # by card, not by slot
    assert check(0, 9, 100, 200) is False                       # not in hand, no fallback
    assert b.deploy_checks == [(1, 0, 100, 200), (1, 2, 100, 200), (1, 2, 300, 400)]

    assert leg.bind(st, 1).name == "towers"
    assert (len(fake.battles), leg.builds, leg.hits) == (1, 1, 1)
    dead = live_state(hand=[5, 2, 7, 0], next_card=3, deck=list(range(8)),
                      enemy_towers=[KING_HP, 0, PRINCESS_HP])
    leg.bind(dead, 1)
    assert len(fake.battles) == 2 and leg.builds == 2
    assert fake.battles[1].reset_kwargs["tower_hp"] == [[1, 0, 1], [1, 1, 1]]
    assert RD.TowersOnlyLegality.key(dead, 1) == (
        1, (5, 2, 7, 0), ((False, True, False), (False, False, False)))


def test_towers_only_battles_are_reset_again_when_evicted():
    fake = _fake_engine(None)
    leg = RD.TowersOnlyLegality(rs_engine.EngineConfig.legacy(), fake, cache_size=1)
    for hand in ([5, 2, 7, 0], [2, 7, 0, 3], [7, 0, 3, 1]):
        leg.bind(live_state(hand=hand), 1)
    assert leg.builds == 3 and len(fake.battles) == 2
    assert fake.battles[0].names() == ["reset", "reset"]
    assert fake.battles[0].reset_kwargs["decks"][0][:4] == [7, 0, 3, 1]


def test_towers_only_falls_back_to_pure_legality(monkeypatch):
    st = live_state()
    monkeypatch.setitem(sys.modules, "royalesim", None)          # royalesim missing
    leg = RD.TowersOnlyLegality(rs_engine.EngineConfig.legacy(), catalogue=CAT)
    assert leg.bind(st, 1).name == "pure"
    assert "royalesim does not import" in leg.last_error
    pol = FakePolicy([-5, 3, 2, 1, 0], {0: ranked(T(9, 8), T(9, 24))})
    d = RD.decide(pol, ROW, st, 1, leg, RD.DecideConfig.live(0.0), rng=StubRng(0.99))
    assert (d.legality, d.tx, d.ty) == ("pure", 9, 24)             # (9, 8) is the enemy half
    bare = RD.TowersOnlyLegality(rs_engine.EngineConfig.legacy())
    d = RD.decide(pol, ROW, st, 1, bare, RD.DecideConfig.live(0.0), rng=StubRng(0.99))
    assert (d.kind, d.reason, d.legality) == (RD.WAIT, RD.R_NO_TILE, "none")
    assert d.tried == (0, 1, 2, 3)


def test_a_refused_reset_falls_back_to_pure_legality():
    class Refusing(FakeBattle):
        def reset(self, *args, **kwargs):
            raise ValueError("a spawn on the water")

    fake = make_fake_royalesim()
    fake.Battle = lambda *a, **kw: Refusing(*a, **kw)
    leg = RD.TowersOnlyLegality(rs_engine.EngineConfig.legacy(), fake, catalogue=CAT)
    assert leg.bind(live_state(), 1).name == "pure"
    assert "reset refused" in leg.last_error and leg.builds == 0


def test_legality_deck():
    player = {"hand": [5, -1, 5, 7], "next_card": 3, "deck": list(range(8))}
    assert RD.legality_deck(player, 16) == [5, 3, 0, 7, 1, 2, 4, 6]
    assert RD.legality_deck({"hand": [1, 2]}, 16, skip=[0]) == [1, 2, 3, 4, 5, 6, 7, 8]
    with pytest.raises(ValueError):
        RD.legality_deck({"hand": []}, 4)


# ---------------------------------------------------------------------------
# pure-python legality
# ---------------------------------------------------------------------------

PURE_CAT = CAT + [catalogue_row("Miner", 3, "TROOP"), catalogue_row("GoblinDrill", 4, "BUILDING"),
                  catalogue_row("Mirror", 1, "SPELL"), catalogue_row("GoblinBarrel", 3, "SPELL")]
MINER, DRILL, MIRROR, BARREL = 16, 17, 18, 19


def at(check, card, wtx, wty):
    """Is `card` legal at the centre of WORLD tile (wtx, wty)?"""
    return check(0, card, (wtx * 1000 + 500) * SUB, (wty * 1000 + 500) * SUB)


def test_arena_tiles_shape():
    tiles = RD.ARENA_TILES
    assert len(tiles) == 32 and {len(r) for r in tiles} == {18}
    assert tiles == tiles[::-1] and all(r == r[::-1] for r in tiles)      # both mirrors
    text = "".join(tiles)
    assert (text.count("."), text.count("N"), text.count("W")) == (484, 60, 32)
    assert [(x, y) for y in (15, 16) for x in range(18) if tiles[y][x] == "."] == [
        (3, 15), (14, 15), (3, 16), (14, 16)]                              # the bridges


def test_pure_legality_team1_all_towers_up():
    pure = RD.PureLegality(PURE_CAT)
    c = pure.bind(live_state(), 1)
    hog, cannon, fireball, log = CID["HogRider"], CID["Cannon"], CID["Fireball"], CID["Log"]
    # model tile (9, 24) of team 1 is world tile (8, 24): our half
    assert c(0, hog, 8500 * SUB, 24500 * SUB) and at(c, hog, 8, 24)
    assert not at(c, hog, 8, 8)                    # enemy half (the king's rect)
    assert not at(c, hog, 3, 12)                   # the left pocket, princess standing
    assert not at(c, hog, 0, 17) and not at(c, hog, 8, 28) and not at(c, hog, 2, 31)  # N
    assert at(c, hog, 8, 31)                       # the open middle of the back row
    assert not at(c, hog, 3, 25)                   # on our own princess
    assert at(c, cannon, 8, 20) and not at(c, cannon, 8, 8)
    assert at(c, fireball, 8, 8) and at(c, fireball, 0, 0) and at(c, fireball, 8, 15)
    assert not at(c, log, 8, 8) and at(c, log, 8, 24)         # the Log: troop territory
    assert at(c, MINER, 8, 8) and at(c, MINER, 0, 0) and not at(c, MINER, 8, 15)
    assert not at(c, MINER, 3, 6)                  # on the enemy's left princess
    assert at(c, DRILL, 8, 8)
    assert at(c, BARREL, 8, 8) and not at(c, BARREL, 8, 15)
    assert not c(0, hog, -18, 100) and not c(0, hog, 18000 * SUB, 100)  # off the board
    assert not at(c, 99, 8, 24)                    # not in the catalogue


def test_pure_legality_pocket_opens_when_a_princess_falls():
    pure = RD.PureLegality(PURE_CAT)
    hog, cannon = CID["HogRider"], CID["Cannon"]
    # team 0's slot 1 is its own left, world x 3.5: the pocket is world columns 0-8, rows 11-14
    c = pure.bind(live_state(enemy_towers=[KING_HP, 0, PRINCESS_HP]), 1)
    assert at(c, hog, 3, 12) and at(c, hog, 3, 11) and at(c, hog, 3, 14)
    assert not at(c, hog, 3, 10)                   # the enemy king's rect (y <= 11)
    assert not at(c, hog, 0, 14)                   # river corner, no-deploy
    assert not at(c, hog, 14, 12)                  # the other lane stays shut
    assert not at(c, hog, 3, 15)                   # the bridge: never offered (conservative)
    assert not at(c, cannon, 3, 12)                # buildings: own half only
    assert at(c, MINER, 3, 6)                      # the princess body is gone
    # team 0 acting: red's slot 1 is ITS left, world x 14.5; its pocket is world rows 17-20
    c0 = RD.PureLegality(PURE_CAT).bind(live_state(team=0), 0)
    assert c0(0, hog, 9500 * SUB, 7500 * SUB)      # model (9, 24) of team 0 = world (9, 7)
    assert not at(c0, hog, 14, 18)
    c0 = pure.bind(live_state(team=0, enemy_towers=[KING_HP, 0, PRINCESS_HP]), 0)
    assert at(c0, hog, 14, 18) and at(c0, hog, 14, 20) and not at(c0, hog, 14, 21)
    assert not at(c0, hog, 3, 18)


def test_pure_legality_mirror_follows_its_target():
    pure = RD.PureLegality(PURE_CAT)
    c = pure.bind(live_state(mirror_target=CID["HogRider"]), 1)
    assert at(c, MIRROR, 8, 24) and not at(c, MIRROR, 8, 8)
    c = pure.bind(live_state(mirror_target=CID["Cannon"]), 1)     # card id 0 is a target too
    assert at(c, MIRROR, 8, 20) and not at(c, MIRROR, 3, 12)
    c = pure.bind(live_state(mirror_target=-1), 1)
    assert not at(c, MIRROR, 8, 24)


# ---------------------------------------------------------------------------
# decisions as suggestions, records, sampling, E15
# ---------------------------------------------------------------------------

def test_suggest_kwargs_feed_the_control_state_in_manual():
    from scripts.vision import control as C
    from scripts.vision import geometry as G
    pol = FakePolicy([-50, 1, 3, 2, 0], {1: ranked(T(9, 24))})
    d = RD.decide(pol, ROW, live_state(), 1, always, RD.DecideConfig.live(0.0),
                  rng=StubRng(0.99), tick=500)
    assert d.world_tile == (8, 24) == G.ArenaGeometry.model_tile_to_screen_tile(9, 24)
    assert d.world_mt == (8500, 24500) and d.model_tile == (9, 24)
    ctl = C.ControlState(C.MANUAL, stop_file=None, control_file=None,
                         names=[r[0] for r in CAT])
    s = ctl.suggest(**d.suggest_kwargs())
    assert (s.card, s.slot, s.tile, s.name, s.tick) == (CID["Musketeer"], 1, (8, 24),
                                                       "Musketeer", 500)
    assert s.p_wait == d.p_wait and s.extra["model_tile"] == [9, 24]
    assert s.mode == C.MANUAL and not ctl.should_tap(s)
    assert ctl.note_decision(s, tapped=False)["action"] == C.ACTION_MANUAL
    w = RD.decide(FakePolicy([50, 0, 0, 0, 0]), ROW, live_state(), 1, always,
                  RD.DecideConfig.live(0.0), rng=StubRng(0.5), tick=510)
    s = ctl.suggest(**w.suggest_kwargs())
    assert s.is_wait and s.reason == RD.R_WAIT and s.tile is None and s.tick == 510
    rec = d.to_record()
    assert json.loads(json.dumps(rec))["command"] == [1, 1, 153000, 441000]
    assert rec["mask"] == [True] * 5 and rec["kind"] == RD.PLAY


def test_sampling_draws_slot_and_tile_reproducibly():
    st = live_state()
    logits = np.arange(RD.TILES, dtype=np.float32) / 100            # row 31 best, row 0 worst
    pol = FakePolicy([0, 3, 2, 1, -2], {s: logits for s in range(4)})

    def own_half(team, slot, x, y):
        return 0 if y / SUB >= 17_000 else OUT_OF_TERRITORY

    cfg = RD.DecideConfig.sampling(0.0, cap_milli=0)                  # capped: never waits
    runs = [RD.decide(pol, ROW, st, 1, FakeBattle(deploy=own_half), cfg,
                      rng=np.random.default_rng(7)) for _ in range(2)]
    assert runs[0] == runs[1]
    d = runs[0]
    assert d.kind == RD.PLAY and d.capped and d.mode == RD.MODE_SAMPLE
    q = np.array(d.gate_q)
    assert q[0] == 0.0 and q.sum() == pytest.approx(1.0) and (q[1:] > 0).all()
    assert d.slot == d.gate_a - 1 and d.logq_gate == pytest.approx(math.log(q[d.gate_a]))
    assert d.tile_cands[:8] == tuple(range(575, 567, -1))           # the 8 best legal tiles
    assert len(d.tile_cands) == 16 and len(set(d.tile_cands)) == 16
    assert all(c // 18 >= 17 for c in d.tile_cands)                 # all legal: our half
    assert d.tile_cands[d.tile_a] == d.tile
    pt = RD.tile_probs(logits.astype(np.float64)[list(d.tile_cands)])
    assert d.logq_tile == pytest.approx(math.log(pt[d.tile_a]))
    # not capped and p_wait ~ 1: the wait is drawn
    w = RD.decide(FakePolicy([10, -5, -5, -5, -5]), ROW, st, 1, always,
                  RD.DecideConfig.sampling(0.0), rng=np.random.default_rng(7))
    assert (w.kind, w.reason, w.gate_a) == (RD.WAIT, RD.R_WAIT, 0)
    assert w.logq_gate == pytest.approx(math.log(w.gate_q[0]))


def test_gate_and_tile_probs():
    mask = np.array([True, True, False, True, True])
    q = RD.gate_probs(np.zeros(5), mask, dt=20)
    # p over the 4 unmasked entries is 1/4 each: wait 0.25, the 0.75 left split three ways
    assert q == pytest.approx([0.25, 0.25, 0.0, 0.25, 0.25])
    assert RD.gate_probs(np.zeros(5), mask, dt=20, capped=True) == pytest.approx(
        [0.0, 1 / 3, 0.0, 1 / 3, 1 / 3])
    assert RD.tile_probs(np.zeros(4)) == pytest.approx([0.25] * 4)
    p = RD.tile_probs(np.array([0.0, -1e9]))
    assert p == pytest.approx([0.925, 0.075])                       # 0.85 + 0.075, 0.075


def test_ability_buttons_are_masked_until_e15():
    """A v2 gate [wait, 4 slots, 2 buttons]: the buttons never win, whatever their logits."""
    pol = FakePolicy([-5, 3, 2, 1, 0, 50, 50])
    st = live_state()
    assert RD.button_mask(st, 1, 2, 5000) == [False, False]
    for cfg, rng in ((RD.DecideConfig.legacy(0.0), StubRng(0.99)),
                     (RD.DecideConfig.live(0.0), StubRng(0.99)),
                     (RD.DecideConfig.sampling(0.0, cap_milli=0), np.random.default_rng(0))):
        d = RD.decide(pol, ROW, st, 1, always, cfg, rng=rng)
        assert d.kind == RD.PLAY and d.mask[5:] == (False, False) and d.button == -1
        assert 0 <= d.slot < 4
    assert RD.decide(pol, ROW, st, 1, always, RD.DecideConfig.legacy(0.0),
                     rng=StubRng(0.99)).slot == 0


def test_config_presets():
    legacy = RD.DecideConfig.legacy(1.5)
    assert legacy == RD.DecideConfig(wait_bias=1.5) and RD.DecideConfig() == RD.DecideConfig.legacy()
    assert not (legacy.mask_pending or legacy.fallback or legacy.mask_unplayable or legacy.sample)
    assert legacy.cap_milli is None and legacy.guard_cards == frozenset()
    live = RD.DecideConfig.live(1.5, guard_cards=[0])
    assert live.mask_pending and live.fallback and live.mask_unplayable
    assert live.cap_milli == 9500 and live.guard_cards == frozenset({0}) and not live.sample
    sample = RD.DecideConfig.for_mode(RD.MODE_SAMPLE, 1.5)
    assert sample.sample and sample.cap_milli is None and sample.guard_cards == frozenset()
    assert RD.DecideConfig.for_mode(RD.MODE_LIVE, 1.5, guard_cards=[0]) == live
    assert live.replace(guard_cards=[3]).guard_cards == frozenset({3})
    with pytest.raises(ValueError):
        RD.DecideConfig.for_mode("ladder", 1.0)


def test_wait_bias_from_args_json(tmp_path):
    ck = tmp_path / "run" / "best.pt"
    ck.parent.mkdir()
    args = ck.parent / "args.json"
    assert RD.wait_keep_for(ck) == 0.35 and RD.wait_keep_for(ck, 0.5) == 0.5
    args.write_text(json.dumps({"wait_keep": "0.25"}), encoding="utf-8")
    assert RD.wait_keep_for(ck) == 0.25
    assert RD.wait_bias_for(ck) == pytest.approx(math.log(4))
    for text in (json.dumps({"lr": "1e-3"}), "{not json", json.dumps({"wait_keep": "lots"})):
        args.write_text(text, encoding="utf-8")
        assert RD.wait_keep_for(ck) == 0.35
    assert RD.wait_keep_for(None) == 0.35 and RD.wait_bias_for(None) == pytest.approx(
        math.log(1 / 0.35))


def test_as_policy_keeps_a_policy():
    pol = FakePolicy([0, 0, 0, 0, 0])
    assert RD.as_policy(pol) is pol


# ---------------------------------------------------------------------------
# rs_eval: RSAgent.act on decide, golden against the act before V06
# ---------------------------------------------------------------------------

@pytest.fixture
def ev(monkeypatch):
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    mod = importlib.import_module("rs_eval")
    assert mod.rs_decide is RD
    monkeypatch.setattr(mod.RR, "F", dict(FIELD))
    monkeypatch.setattr(mod.RSAgent, "_CAT", [list(r) for r in DEFAULT_CATALOGUE])
    return mod


class FakeTensor:
    """Just the torch.Tensor surface RSAgent.batch / act use, on numpy."""

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


class FakeModel:
    """A deterministic TokenPolicy stand-in: logits seeded by the batch (and the card)."""

    def __call__(self, b, card_ids=None):
        key = (b["hand"].numpy().tobytes() + b["scalars"].numpy().tobytes()
               + b["unit_ids"].numpy().tobytes())
        gate = np.random.default_rng(zlib.crc32(key)).normal(0.0, 2.0, (1, 5))
        cid = card_ids.numpy().astype(np.int64).tobytes()
        tile = np.random.default_rng(zlib.crc32(key + cid)).normal(0.0, 1.0, (1, RD.TILES))
        return (FakeTensor(gate.astype(np.float32)), FakeTensor(tile.astype(np.float32)),
                FakeTensor(np.zeros((1, 2), np.float32)))


class ScriptedModel:
    """Fixed gate logits; tile logits per hand card id (FakeTensors)."""

    def __init__(self, gate, tiles):
        self.gate = np.asarray([gate], dtype=np.float32)
        self.tiles = tiles

    def __call__(self, b, card_ids=None):
        card = int(card_ids.numpy()[0]) - 2
        tile = self.tiles.get(card, ranked())[None]
        return FakeTensor(self.gate), FakeTensor(tile), FakeTensor(np.zeros((1, 2), np.float32))


def legacy_batch(self, row):
    """RSAgent.batch before V06, verbatim."""
    t = self.torch
    keys = ("unit_ids", "unit_feats", "towers", "scalars", "hand", "opp_seen",
            "opp_last", "opp_last_age")
    b = {k: t.as_tensor(np.asarray(row[k])[None]) for k in keys}
    for k in ("unit_ids", "hand", "opp_seen", "opp_last"):
        b[k] = b[k].long()
    b["unit_feats"] = b["unit_feats"].float()
    b["opp_last_age"] = b["opp_last_age"].float()
    b["next"] = t.as_tensor([int(row["next"])]).long()
    return b


def legacy_act(self, battle, state, team, mem, tick, sub, royalesim, ev):
    """RSAgent.act before V06, verbatim (tools/rs_eval.py at 1054b02), but for its module names."""
    RR, to_world, DECIDE = ev.RR, ev.to_world, ev.DECIDE
    self._team = team
    if getattr(self, "hold_pending", False) and battle.pending_commands(team):
        self.ahead(battle, state)        # keep the velocity memory current
        return None
    view = self.ahead(battle, state)
    row = RR.obs_row(view, team, sub, mem, tick + self.lead, live=self.live)
    b = legacy_batch(self, row)
    with self.torch.no_grad():
        gate, _, _ = self.model(b, card_ids=b["hand"][:, 0].clamp(min=0))
    g = gate[0].float().numpy()
    g[0] += self.wait_bias
    p = state["players"][team]
    pending_cost = int(p.get("pending_cost", 0) or 0)
    for slot in range(4):
        cost = p["hand_costs"][slot] if slot < len(p["hand_costs"]) else 99
        if (slot >= len(p["hand"]) or p["elixir_milli"] - pending_cost * 1000 < cost * 1000
                or p["hand"][slot] < 0):
            g[1 + slot] = -1e9
    pr = np.exp(g - g.max())
    pr /= pr.sum()
    if self.rng.random() < float(pr[0]) ** (DECIDE / 20) or g[1:].max() < -1e8:
        return None
    slot = int(np.argmax(g[1:]))
    with self.torch.no_grad():
        _, tile, _ = self.model(b, card_ids=b["hand"][:, slot])
    order = np.argsort(-tile[0].float().numpy())
    card = p["hand"][slot]
    if getattr(self, "spell_aim", False) and self.cat_kind(card) == "SPELL":
        order = self.aim_spell(order, view, team, card, sub)
    for t in order[:60]:
        tx, ty = int(t % 18), int(t // 18)
        wx, wy = to_world(tx * 1000 + 500, ty * 1000 + 500, team)
        if battle.check_deploy(team, slot, wx * sub, wy * sub) == 0:
            return (team, slot, wx * sub, wy * sub)
    return None


def make_agent(ev, *, seed, live=False, spell_aim=False, hold=False, model=None, torch=None):
    """An RSAgent without a checkpoint file: what __init__ sets, by hand."""
    a = object.__new__(ev.RSAgent)
    a.model = FakeModel() if model is None else model
    a.torch = FAKE_TORCH if torch is None else torch
    a.meta, a.vocab_drift, a.live = {}, [], live
    a.rng = np.random.default_rng(seed)
    a.wait_bias = math.log(1 / 0.35)
    a.spell_aim, a.hold_pending = spell_aim, hold
    return a


def random_case(r, k):
    """A random state for either seat: hands with gaps, odd hand_costs, pending costs, units."""
    team = int(r.integers(0, 2))
    tick = 90 + 10 * k
    hand = [int(c) for c in r.choice(16, size=4, replace=False)]
    if r.random() < 0.15:
        hand[int(r.integers(0, 4))] = -1
    costs = [CAT[c][1] if c >= 0 else 99 for c in hand]
    u = r.random()
    if u < 0.1:
        costs = costs[:int(r.integers(0, 4))]
    elif u < 0.2:
        costs[int(r.integers(0, 4))] = -1
    me = player_dict(team, hand=hand, hand_costs=costs, elixir_milli=int(r.integers(0, 10_001)),
                     next_card=int(r.integers(0, 16)),
                     pending_cost=int(r.integers(0, 5)) if r.random() < 0.3 else 0)
    opp = player_dict(1 - team, next_card=int(r.integers(0, 16)))
    ents = [entity_row(uid=100 + i, team=int(r.integers(0, 2)), kind=int(r.integers(0, 2)),
                       card_id=int(r.integers(0, 16)), x=int(r.integers(0, 18_000)) * SUB,
                       y=int(r.integers(0, 32_000)) * SUB, hp=int(r.integers(1, 3000)),
                       unit_type=int(r.integers(0, 14)), flying=bool(r.random() < 0.3),
                       deploy_ticks=int(r.integers(0, 3)))
            for i in range(int(r.integers(0, 7)))]
    state = state_dict(tick, players=[me, opp] if team == 0 else [opp, me], entities=ents)
    mem = {"seen": {t: {int(c) + 2 for c in r.choice(16, size=int(r.integers(0, 5)),
                                                     replace=False)} for t in (0, 1)},
           "last": {t: [(int(r.integers(2, 18)), int(r.integers(0, tick)))
                        for _ in range(int(r.integers(0, 6)))] for t in (0, 1)}}
    pend = [("command",)] if r.random() < 0.3 else []
    return SimpleNamespace(team=team, state=state, mem=mem, tick=tick, pend=pend)


def deploy_fn(k):
    """About 2% of (team, slot, x, y) legal, so 30% of decisions find no tile in 60."""
    def deploy(team, slot, x, y):
        return 0 if zlib.crc32(f"{k},{team},{slot},{x},{y}".encode()) % 50 == 0 else 8
    return deploy


def run_golden(ev, n, *, seed, live=False, spell_aim=False, hold=False, model=None, torch=None):
    r = np.random.default_rng(seed)
    old = make_agent(ev, seed=seed + 1, live=live, spell_aim=spell_aim, hold=hold, model=model,
                     torch=torch)
    new = make_agent(ev, seed=seed + 1, live=live, spell_aim=spell_aim, hold=hold, model=model,
                     torch=torch)
    seen = {"play": 0, "no_tile": 0, "wait": 0, "held": 0}
    for k in range(n):
        case = random_case(r, k)
        bo, bn = FakeBattle(deploy=deploy_fn(k)), FakeBattle(deploy=deploy_fn(k))
        bo.pending[case.team] = list(case.pend)
        bn.pending[case.team] = list(case.pend)
        want = legacy_act(old, bo, json.loads(json.dumps(case.state)), case.team, case.mem,
                          case.tick, SUB, None, ev)
        got = new.act(bn, json.loads(json.dumps(case.state)), case.team, case.mem, case.tick,
                      SUB, None)
        assert got == want, (k, got, want)
        assert bn.deploy_checks == bo.deploy_checks, k
        if hold and case.pend:
            seen["held"] += 1
        elif want is not None:
            seen["play"] += 1
            assert new.last_decision.command == want
        elif bo.deploy_checks:
            seen["no_tile"] += 1
            assert new.last_decision.reason == RD.R_NO_TILE
        else:
            seen["wait"] += 1
    assert new.rng.bit_generator.state == old.rng.bit_generator.state
    return seen


@pytest.mark.parametrize("live, spell_aim, hold", [(False, False, False), (True, False, False),
                                                   (False, True, False), (False, False, True)])
def test_rs_eval_legacy_decisions_are_unchanged(ev, live, spell_aim, hold):
    seen = run_golden(ev, 300, seed=1000 + 2 * live + 4 * spell_aim + 8 * hold, live=live,
                      spell_aim=spell_aim, hold=hold)
    assert seen["play"] >= 5 and seen["no_tile"] >= 5 and seen["wait"] >= 5, seen
    assert (seen["held"] >= 5) == hold, seen


@pytest.mark.parametrize("live", [False, True])
def test_rs_eval_legacy_decisions_are_unchanged_with_torch(ev, live):
    torch = pytest.importorskip("torch")
    from sim.model_tokens import TokenPolicy
    torch.manual_seed(0)
    model = TokenPolicy(18, 20, d=16, layers=1, heads=2).eval()
    seen = run_golden(ev, 60, seed=77 + live, live=live, model=model, torch=torch)
    assert sum(seen.values()) == 60


def test_row_batch_is_rsagent_batch(ev):
    agent = make_agent(ev, seed=0)
    case = random_case(np.random.default_rng(3), 0)
    row = ev.RR.obs_row(case.state, case.team, SUB, case.mem, case.tick)
    a, b = agent.batch(row), RD.row_batch(row, FAKE_TORCH)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].numpy().dtype == b[k].numpy().dtype
        np.testing.assert_array_equal(a[k].numpy(), b[k].numpy())


def test_rs_eval_live_mode_blocks_the_engine_pending_slot(ev):
    """Team 0 under a delay: HogRider (slot 0) is pending and refused CARD_PENDING."""
    hand = [CID["HogRider"], CID["Musketeer"], CID["Skeletons"], CID["Cannon"]]
    me = player_dict(0, hand=hand, elixir_milli=9000, pending=[[0, CID["HogRider"], 0, 0, 12, 4]],
                     pending_cost=4)
    state = state_dict(500, players=[me, player_dict(1)])
    mem = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}
    model = ScriptedModel([-50, 3, 2, 1, 0], {CID["Musketeer"]: ranked(T(9, 20))})

    def refuse_pending(team, slot, x, y):
        return CARD_PENDING if slot == 0 else 0

    legacy = make_agent(ev, seed=5, model=model)
    assert legacy.act(FakeBattle(deploy=refuse_pending), state, 0, mem, 500, SUB, None) is None
    assert legacy.last_decision.reason == RD.R_NO_TILE and legacy.last_decision.tried == (0,)
    live = make_agent(ev, seed=5, model=model)
    live.decide_mode = RD.MODE_LIVE
    assert live.decide_config().guard_cards == frozenset({CID["Cannon"]})
    # model tile (9, 20) of team 0: world (9500, 11500)
    cmd = live.act(FakeBattle(deploy=refuse_pending), state, 0, mem, 500, SUB, None)
    assert cmd == (0, 1, 9500 * SUB, 11500 * SUB) == (0, 1, 171000, 207000)
    d = live.last_decision
    assert d.mask == (True, False, True, True, True) and d.mode == RD.MODE_LIVE
    live.decide_mode = "ladder"
    with pytest.raises(ValueError):
        live.decide_config()


def test_rs_eval_play_refuses_an_unknown_decide_mode(ev):
    with pytest.raises(ValueError, match="decide"):
        ev.play(("{}", "", "pro", 0, "", 0, 0, False, False, False, None, None, "ladder"))


def test_frames_agree_with_rs_eval(ev):
    for team in (0, 1):
        for t in range(RD.TILES):
            tx, ty = t % 18, t // 18
            assert (RD.to_world(tx * 1000 + 500, ty * 1000 + 500, team)
                    == ev.to_world(tx * 1000 + 500, ty * 1000 + 500, team)
                    == RD.tile_centre_world(t, team))
    for team in (0, 1):
        assert RD.own(9000 * SUB, 20000 * SUB, team) == ev.RR.own(9000 * SUB, 20000 * SUB, team,
                                                                  SUB)


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

@pytest.mark.engine
def test_towers_only_legality_on_royalesim():
    royalesim = importlib.import_module("royalesim")
    cfg = rs_engine.EngineConfig.defaults()
    cat = rs_engine.catalogue(cfg.undelayed(), royalesim)
    ids = {row[0]: i for i, row in enumerate(cat)}
    deck = [ids[n] for n in ("HogRider", "Cannon", "Log", "Fireball", "Musketeer",
                             "IceGolemite", "IceSpirits", "Skeletons")]
    sub = royalesim.SUBTILE_PER_MILLITILE
    towers = RD.TowersOnlyLegality(cfg, royalesim)
    pure = RD.PureLegality(cat, sub=sub)

    def state(dead=()):
        players = [player_dict(0, hand=deck[4:]),
                   player_dict(1, hand=deck[:4], next_card=deck[4], deck=deck)]
        for t, s in dead:
            players[t]["tower_hp"][s] = 0
        return state_dict(200, players=players)

    def legal(check, card, wtx, wty):
        return check(0, card, (wtx * 1000 + 500) * sub, (wty * 1000 + 500) * sub)

    hog = ids["HogRider"]
    up = towers.bind(state(), 1)
    assert up.name == "towers", towers.last_error
    assert legal(up, hog, 8, 24)                   # our half (model tile (9, 24))
    assert not legal(up, hog, 8, 8)                # the enemy half
    assert not legal(up, hog, 3, 12)               # the left pocket, princess standing
    down = towers.bind(state([(0, 1)]), 1)
    assert legal(down, hog, 3, 12)                 # team 0's slot 1 (world left) fell
    assert not legal(down, hog, 14, 12)            # the other lane stays shut
    for dead in ((), ((0, 1),), ((0, 2),), ((0, 1), (0, 2))):
        st = state(dead)
        engine, fallback = towers.bind(st, 1), pure.bind(st, 1)
        for card in deck[:4]:
            bad = [(x, y) for y in range(32) for x in range(18)
                   if legal(fallback, card, x, y) and not legal(engine, card, x, y)]
            assert not bad, (cat[card][0], dead, bad[:5])
    assert towers.builds == 4


@pytest.mark.engine
def test_arena_tiles_match_royalesims_arena():
    royalesim = importlib.import_module("royalesim")
    if not hasattr(royalesim, "data_dir"):
        pytest.skip("royalesim before data_dir()")
    path = Path(royalesim.data_dir()) / "derived" / "arena.json"
    if not path.is_file():
        pytest.skip(f"no {path}")
    arena = json.loads(path.read_text(encoding="utf-8"))
    grid, bits = arena["grid"], arena["bits"]

    def tile(tx, ty):
        cells = [grid[2 * ty + a][2 * tx + b] for a in (0, 1) for b in (0, 1)]
        if any(c & bits["WATER"] for c in cells):
            return "W"
        return "N" if any(c & bits["NO_DEPLOY"] for c in cells) else "."

    assert RD.ARENA_TILES == tuple("".join(tile(x, y) for x in range(18)) for y in range(32))
