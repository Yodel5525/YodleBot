"""S04: tools/rs_rebuild.py, the rebuilder (docs/LIVE_SEARCH_PLAN.md 4.2-4.8).

No engine for most of it: conftest's state_json-shaped dicts (entity rows in royalesim's
ENTITY_FIELDS order, FIELD), a small synthetic catalogue (royalesim's column order: name,
placement, elixir, count, radius, ...; Hog 2.6 is 0-7) and E06 spawn table, S02's public views,
and CycleBattle: conftest's FakeBattle dealing decks as shuffle 0 does, paying and cycling each
play, and listing the spawns as entities. The checks the plan asks for (S04): the deal with pending
cards and recasts gives back exactly the hand and queue; recast costs are added to that side's
elixir; spawns are clamped; a refused reset falls back with a reason; the fidelity arithmetic; R0's
reset arguments equal RSAgent.rebuilt's (a monkeypatched royalesim). Then the root's advance (our
plays in flight at their exec ticks, one command per team per step), legality at B, trials, the
relabel, the spell aim, permanence's kept rows, the probe-driven settings and the import weight.
Marked `engine`: the relabel round trip covers every spawn, a recast deploys, a root rebuilt twice
gives the same state_hash.

The tests that build an R1 / R2 spec need numpy (tools/public_state's sanitiser); they skip
without it.
"""
from __future__ import annotations

import copy
import importlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import (DEPLOY_REASONS, FIELD, KING_HP, PRINCESS_HP, Call, FakeBattle,
                                     entity_row, make_fake_royalesim, player_dict, state_dict)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import rs_rebuild as RB  # noqa: E402

import rs_engine  # noqa: E402  (tools/ is on sys.path once rs_rebuild is imported)
import rs_public_view as PV  # noqa: E402
import rs_spawn_table  # noqa: E402

SUB = 18

# ---------------------------------------------------------------------------
# a small engine: catalogue, unit types, E06 spawn table
# ---------------------------------------------------------------------------

#: (name, elixir, count, radius mt, flying, card_kind) in catalogue id order; Hog 2.6 is 0-7.
CARD_ROWS = (
    ("Cannon", 3, 1, 0, False, "BUILDING"), ("Fireball", 4, 1, 2500, False, "SPELL"),
    ("HogRider", 4, 1, 0, False, "TROOP"), ("IceGolemite", 2, 1, 0, False, "TROOP"),
    ("IceSpirits", 1, 1, 0, False, "TROOP"), ("Musketeer", 4, 1, 0, False, "TROOP"),
    ("Skeletons", 1, 3, 0, False, "TROOP"), ("Log", 2, 1, 0, False, "SPELL"),
    ("Knight", 3, 1, 0, False, "TROOP"), ("Archer", 3, 2, 0, False, "TROOP"),
    ("Giant", 5, 1, 0, False, "TROOP"), ("Arrows", 3, 1, 3500, False, "SPELL"),
    ("Valkyrie", 4, 1, 0, False, "TROOP"), ("MiniPekka", 4, 1, 0, False, "TROOP"),
    ("BabyDragon", 4, 1, 0, True, "TROOP"), ("Zap", 2, 1, 2500, False, "SPELL"),
    ("Witch", 5, 1, 0, False, "TROOP"), ("Tesla", 4, 1, 0, False, "BUILDING"),
    ("Mirror", 1, 1, 0, False, "SPELL"), ("Elixir Collector", 6, 1, 0, False, "BUILDING"),
)
#: catalogue_json rows: name, placement, elixir, count, radius, flying, hitpoints, footprint, kind.
CATALOGUE = [[n, 6 if n == "Mirror" else 0, cost, count, radius, fly, 0, 0, kind]
             for n, cost, count, radius, fly, kind in CARD_ROWS]
NAMES = [r[0] for r in CATALOGUE]
CID = {n: i for i, n in enumerate(NAMES)}
(CANNON, FIREBALL, HOG, ICEGOLEM, SPIRITS, MUSK, SKELS, LOG, KNIGHT, ARCHER, GIANT, ARROWS, VALK,
 MINIPEKKA, BABYD, ZAP, WITCH, TESLA, MIRROR, COLLECTOR) = range(len(NAMES))
KIND = {i: r[8] for i, r in enumerate(CATALOGUE)}
COST = {i: 1000 * r[2] for i, r in enumerate(CATALOGUE)}
UNITS = ["KingTower", "PrincessTower", "Cannon", "HogRider", "IceGolemite", "IceSpirits",
         "Musketeer", "Skeleton", "Knight", "Archer", "Giant", "Valkyrie", "MiniPekka",
         "BabyDragon", "Witch", "Tesla", "ElixirCollector", "Golemite", "Skeleton_EV1"]
UID = {n: i for i, n in enumerate(UNITS)}
PRIMARY = {"Cannon": "Cannon", "HogRider": "HogRider", "IceGolemite": "IceGolemite",
           "IceSpirits": "IceSpirits", "Musketeer": "Musketeer", "Skeletons": "Skeleton",
           "Knight": "Knight", "Archer": "Archer", "Giant": "Giant", "Valkyrie": "Valkyrie",
           "MiniPekka": "MiniPekka", "BabyDragon": "BabyDragon", "Witch": "Witch", "Tesla": "Tesla",
           "Elixir Collector": "ElixirCollector"}
HP11 = {"Cannon": 824, "HogRider": 1697, "IceGolemite": 1228, "IceSpirits": 215, "Musketeer": 721,
        "Skeletons": 81, "Knight": 1766, "Archer": 304, "Giant": 3968, "Valkyrie": 1907,
        "MiniPekka": 1390, "BabyDragon": 1152, "Witch": 839, "Tesla": 1000,
        "Elixir Collector": 1070}
SPAWN_TABLE = {
    "format": 1, "card_names": list(NAMES), "unit_types": list(UNITS),
    "primary": {str(CID[c]): UID[u] for c, u in PRIMARY.items()},
    "hp": {str(CID[c]): {"11": [["own", PRIMARY[c], hp]]} for c, hp in HP11.items()},
}
OWN = {CID[c]: hp for c, hp in HP11.items()}
TABLES = RB.Tables(CATALOGUE, SPAWN_TABLE, level=11)
MIRRORS = PV.mirror_ids(CATALOGUE)
BUILDING_UNITS = ("Cannon", "Tesla", "ElixirCollector")
NOT_ENOUGH = DEPLOY_REASONS.index("NOT_ENOUGH_ELIXIR")
OCCUPIED = DEPLOY_REASONS.index("OCCUPIED")


def mt(x: float, y: float) -> tuple[int, int]:
    return int(round(x * SUB)), int(round(y * SUB))


def unit(uid, team, name, card, x, y, hp, max_hp=None, *, deploy=0, status=0, level=11):
    """An engine entity: unit type `name` (None: -1) put down by `card` at (x, y) millitiles."""
    return entity_row(uid=uid, team=team, kind=1 if name in BUILDING_UNITS else 0, card_id=card,
                      x=int(round(x * SUB)), y=int(round(y * SUB)), hp=hp,
                      max_hp=hp if max_hp is None else max_hp, flying=name == "BabyDragon",
                      deploy_ticks=deploy, status_flags=status, level=level,
                      unit_type=-1 if name is None else UID[name])


def memory(own=(), opp=(), *, own_team=0):
    """A memory view: (card, exec tick) plays of the observer (`own_team`) and of its opponent."""
    last = {own_team: [(c + 2, t) for c, t in own], 1 - own_team: [(c + 2, t) for c, t in opp]}
    return {"last": last, "seen": {k: {c for c, _ in v} for k, v in last.items()}}


def view_of(state, team=0, mem=None, *, opp_elixir=None, sent=()):
    return PV.public_view(state, team, mem, fields=FIELD, opp_elixir=opp_elixir, sent=sent,
                          mirror=MIRRORS)


def play_out(deck, plays):
    """The shuffle-0 cycle in Python: deal `deck`, play each card of `plays` from its slot."""
    hand, queue = list(deck[:4]), list(deck[4:8])
    for card in plays:
        slot = hand.index(card)
        hand[slot] = queue.pop(0)
        queue.append(card)
    return hand, queue


# ---------------------------------------------------------------------------
# CycleBattle: a FakeBattle that deals, pays and cycles
# ---------------------------------------------------------------------------

class CycleBattle(FakeBattle):
    """FakeBattle that deals the reset decks as shuffle 0 does (hand = deck[0:4], queue =
    deck[4:8]), lists the spawns as entities (new uids, in reverse order: matching must go by key)
    and, on a play, checks the elixir, pays it, cycles the card (its slot takes the queue head, the
    card goes to the back) and puts a deploying unit down. No elixir regeneration, no movement.
    refuse(team, card, x, y) -> a reason code for check_deploy and step; formation: a multi-unit
    spawn lays `count` members 500 mt apart in x, the middle one at the spec's point."""

    def __init__(self, *args, refuse=None, reset_error=None, over=False, formation=False, **kw):
        super().__init__(*args, catalogue=CATALOGUE, unit_types=UNITS, **kw)
        self.refuse, self.reset_error = refuse, reset_error
        self.over, self.formation = over, formation
        self.hands, self.queues, self.elixir = [[], []], [[], []], [0, 0]
        self.towers, self.ents, self.played, self.next_uid = None, [], [], 500

    def _unit(self, team, card, x, y, hp, deploy=0):
        name = PRIMARY.get(NAMES[card])
        row = unit(self.next_uid, team, name, card, 0, 0, hp, OWN.get(card, hp), deploy=deploy)
        row[FIELD["x"]], row[FIELD["y"]] = int(x), int(y)
        self.next_uid += 1
        return row

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self.reset_error:
            raise ValueError(self.reset_error)
        decks = kwargs["decks"]
        self.hands = [list(d[:4]) for d in decks]
        self.queues = [list(d[4:8]) for d in decks]
        self.elixir = list(kwargs["elixir_milli"])
        self.towers = kwargs.get("tower_hp")
        self.ents, self.played = [], []
        for team, card, x, y, hp in reversed(list(kwargs["spawns"])):
            k = CATALOGUE[card][3] if self.formation else 1
            for j in range(k):
                self.ents.append(self._unit(team, card, x + (j - (k - 1) // 2) * 500 * SUB, y, hp))

    def _code(self, team, card, x, y):
        code = self.refuse(team, card, x, y) if self.refuse else 0
        if code == 0 and self.elixir[team] < COST[card]:
            code = NOT_ENOUGH
        return code

    def step(self, cmds=(), n=1):
        cmds = [tuple(c) for c in cmds]
        self.calls.append(Call("step", (cmds, n), {}))
        res = []
        for team, slot, x, y in cmds:
            card = self.hands[team][slot]
            code = self._code(team, card, x, y)
            if code == 0:
                self.elixir[team] -= COST[card]
                self.hands[team][slot] = self.queues[team].pop(0)
                self.queues[team].append(card)
                self.played.append((self.tick, team, card, x, y))
                if KIND[card] != "SPELL":
                    self.ents.append(self._unit(team, card, x, y, OWN[card], deploy=20))
            res.append((card, code, self.tick, x, y))
        self.tick += int(n)
        self.steps += 1
        return res

    def check_deploy(self, team, slot, x, y):
        self.deploy_checks.append((int(team), int(slot), int(x), int(y)))
        return self._code(team, self.hands[team][slot], x, y)

    def state(self):
        players = []
        for t in (0, 1):
            hp = list(self.towers[t]) if self.towers else [KING_HP, PRINCESS_HP, PRINCESS_HP]
            players.append(player_dict(t, elixir_milli=self.elixir[t], hand=list(self.hands[t]),
                                       next_card=self.queues[t][0] if self.queues[t] else -1,
                                       tower_hp=hp))
        return state_dict(self.tick, players=players, entities=self.ents, game_over=self.over)

    def save(self):
        self.calls.append(Call("save", (), {}))
        return copy.deepcopy({"tick": self.tick, "steps": self.steps, "hands": self.hands,
                              "queues": self.queues, "elixir": self.elixir, "ents": self.ents,
                              "played": self.played, "uid": self.next_uid})

    def load(self, snap):
        self.calls.append(Call("load", (), {}))
        s = copy.deepcopy(snap)
        self.tick, self.steps = s["tick"], s["steps"]
        self.hands, self.queues, self.elixir = s["hands"], s["queues"], s["elixir"]
        self.ents, self.played, self.next_uid = s["ents"], s["played"], s["uid"]


def cycle_rs(**battle_kw):
    """A fake royalesim whose Battle is a CycleBattle (`battles` lists them)."""
    mod = make_fake_royalesim()

    def Battle(*args, **kwargs):
        b = CycleBattle(**battle_kw)
        mod.battles.append(b)
        return b

    mod.Battle = Battle
    return mod


def rebuilder(settings=None, **battle_kw):
    mod = cycle_rs(**battle_kw)
    rb = RB.Rebuilder(rs_engine.EngineConfig.legacy(), mod, SPAWN_TABLE,
                      settings=settings or RB.RebuildSettings(), catalogue=CATALOGUE, fields=FIELD)
    return rb, mod


def steps(battle):
    return [c.args for c in battle.calls_named("step")]


@pytest.fixture
def ps():
    """tools/public_state (numpy, scripts/vision): what an R1 / R2 spec needs."""
    pytest.importorskip("numpy")
    return importlib.import_module("public_state")


# ---------------------------------------------------------------------------
# the deal
# ---------------------------------------------------------------------------

def test_deal_gives_back_the_hand_and_the_queue():
    hand, queue = [10, 11, 12, 13], [20, 21, 22, 23]
    # two recasts: the queue's last two plays, oldest first
    pre_h, pre_q, notes = RB.deal(hand, queue, [22, 23])
    assert (pre_h, pre_q, notes) == ([10, 11, 23, 22], [13, 12, 20, 21], [])
    assert play_out(pre_h + pre_q, [22, 23]) == (hand, queue)
    # a recast, then our in-flight card still in the hand (the simulator: it cycles when it runs)
    pre_h, pre_q, notes = RB.deal(hand, queue, [23, 11])
    assert (pre_h, pre_q, notes) == ([10, 11, 12, 23], [13, 20, 21, 22], [])
    assert play_out(pre_h + pre_q, [23]) == (hand, queue)
    assert play_out(pre_h + pre_q, [23, 11]) == ([10, 20, 12, 13], [21, 22, 23, 11])
    # the HUD already cycled our in-flight cards (live): they are re-dealt like recasts
    pre_h, pre_q, _ = RB.deal(hand, queue, [21, 22, 23])
    assert play_out(pre_h + pre_q, [21, 22, 23]) == (hand, queue)
    # not the queue's tail: the hand comes back, the queue is reordered (and says so)
    pre_h, pre_q, notes = RB.deal(hand, queue, [22])
    assert play_out(pre_h + pre_q, [22]) == (hand, [20, 21, 23, 22])
    assert notes == ["queue_reordered"]


def test_deal_prefers_mirror_and_collector_and_refuses_what_cannot_be_dealt():
    hand, queue = [10, MIRROR, 12, 13], [20, 21, 22, 23]
    pre_h, pre_q, notes = RB.deal(hand, queue, [23], omit=[MIRROR, COLLECTOR])
    assert MIRROR not in pre_h and pre_q[0] == MIRROR and notes == []
    assert play_out(pre_h + pre_q, [23]) == (hand, queue)
    assert RB.deal(hand, queue, [], omit=[MIRROR])[2] == ["omit_in_hand"]
    assert "deal_interleaved" in RB.deal(hand, queue, [12, 23])[2]
    with pytest.raises(RB.RebuildError, match="neither"):
        RB.deal(hand, queue, [99])
    with pytest.raises(RB.RebuildError, match="slots free"):
        RB.deal(hand, queue, [10, MIRROR, 12, 13, 23])
    with pytest.raises(RB.RebuildError, match="distinct"):
        RB.deal(hand, [20, 21, 22, 10], [])


# ---------------------------------------------------------------------------
# a push: our Hog deploying, an enemy Mini P.E.K.K.A deploying, our Log in flight
# ---------------------------------------------------------------------------

T0 = 1000
HOG_AIM, LOG_AIM = mt(3500, 12000), mt(9000, 9000)
SENT = [(700, MUSK, *mt(6000, 8000)), (800, ICEGOLEM, *mt(9000, 10000)),
        (900, SPIRITS, *mt(9000, 13000)), (995, HOG, *HOG_AIM), (1014, LOG, *LOG_AIM)]
PUSH_MEM = memory([(MUSK, 700), (ICEGOLEM, 800), (SPIRITS, 900), (HOG, 995)],
                  [(GIANT, 600), (VALK, 700), (KNIGHT, 800), (MINIPEKKA, 992)])


def push_state(*, noised=False):
    """Seat 0's state at T0. Our run plays (Musketeer, Ice Golem, Ice Spirits, Hog) are the queue,
    the hand [Cannon, Fireball, Skeletons, Log] still holds the Log in flight (the engine cycles it
    when it runs, 1014). noised: the same board with everything a screen does not show changed."""
    me = player_dict(0, elixir_milli=6000, hand=[CANNON, FIREBALL, SKELS, LOG], next_card=MUSK,
                     deck=list(range(8)))
    them = player_dict(1, elixir_milli=7300, hand=[KNIGHT, ARCHER, GIANT, ARROWS], next_card=VALK,
                       deck=list(range(8, 16)))
    ents = [unit(10, 0, "Musketeer", MUSK, 6000, 8000, 400, 721),
            unit(11, 0, "HogRider", HOG, 3500, 12000, 1697, deploy=15),
            unit(20, 1, "Knight", KNIGHT, 4000, 20000, 1766),
            unit(21, 1, "MiniPekka", MINIPEKKA, 14500, 20000, 1390, deploy=12)]
    state = state_dict(T0, players=[me, them], entities=ents)
    if noised:
        p = state["players"][1]
        p.update(elixir_milli=9100, hand=[ZAP, ARROWS, GIANT, BABYD], next_card=ARCHER,
                 pending=[[0, BABYD, 1, 2, 5, 4]], pending_cost=4)
        for e in state["entities"]:
            e[FIELD["target_uid"]], e[FIELD["stun_ticks"]] = 3, 7
        state["entities"].append(unit(30, 1, "Knight", KNIGHT, 9000, 20000, 900, 1766, status=2))
        state["projectiles"] = [[1, 2, 3]]
    return state


def push_spec(rb=None, **kw):
    view = view_of(push_state(**kw), 0, PUSH_MEM, opp_elixir=(5000, True), sent=SENT)
    opp_fill = [ARCHER, ARROWS, BABYD, ZAP]
    if rb is None:
        return RB.build_spec(view, TABLES, RB.RebuildSettings(level=RB.R2), land_tick=T0 + 24,
                             fields=FIELD, opp_fill=opp_fill)
    return rb.spec(view, land_tick=T0 + 24, level=RB.R2, opp_fill=opp_fill)


def test_spec_deals_both_sides_around_the_recasts_and_our_play_in_flight(ps):
    spec = push_spec()
    assert spec.hands == ((CANNON, FIREBALL, SKELS, LOG), (ARCHER, ARROWS, BABYD, ZAP))
    assert spec.queues == ((MUSK, ICEGOLEM, SPIRITS, HOG), (GIANT, VALK, KNIGHT, MINIPEKKA))
    assert [(r.team, r.card, r.kind, r.order) for r in spec.recasts] == [
        (0, HOG, "deploy", 995), (1, MINIPEKKA, "deploy", 992)]
    hog, pekka = spec.recasts
    assert (hog.x, hog.y) == HOG_AIM                         # our exact aim (own_sent)
    assert (pekka.x, pekka.y) == mt(14500, 20000)            # the opponent's: the centroid
    assert hog.spawns == ((0, HOG, *HOG_AIM, 1697),) and hog.labels == (HOG,) and hog.uids == (11,)
    assert spec.replays == ((1014, LOG, *LOG_AIM),) and spec.later == ()
    assert spec.order == ((HOG, LOG), (MINIPEKKA,))
    assert spec.decks == ((CANNON, FIREBALL, HOG, LOG, SKELS, MUSK, ICEGOLEM, SPIRITS),
                          (ARCHER, ARROWS, BABYD, MINIPEKKA, ZAP, GIANT, VALK, KNIGHT))
    # after the recasts the hands and queues are the view's; after the Log, the Log has cycled
    assert play_out(spec.decks[0], [HOG]) == (list(spec.hands[0]), list(spec.queues[0]))
    assert play_out(spec.decks[1], [MINIPEKKA]) == (list(spec.hands[1]), list(spec.queues[1]))
    assert play_out(spec.decks[0], [HOG, LOG]) == ([CANNON, FIREBALL, SKELS, MUSK],
                                                   [ICEGOLEM, SPIRITS, HOG, LOG])
    # the deploying units are recast, not spawned
    assert spec.spawns == ((0, MUSK, *mt(6000, 8000), 400), (1, KNIGHT, *mt(4000, 20000), 1766))
    assert spec.labels == (MUSK, KNIGHT) and spec.uids == (10, 20)
    assert (spec.units, spec.unit_hp, spec.dropped, spec.lost_hp) == (4, 400 + 1697 + 1766 + 1390,
                                                                      0, 0)
    assert spec.flags == () and spec.refusal is None and spec.opp_exact
    assert RB.fidelity(spec) == 1.0


def test_recast_costs_are_added_to_that_sides_elixir(ps):
    spec = push_spec()
    assert spec.base_elixir == (6000, 5000)                 # the HUD's, the counted opponent's
    assert spec.elixir == (6000 + 4000, 5000 + 4000)
    # a full bar: clamped, and said
    state = push_state()
    state["players"][0]["elixir_milli"] = 8500
    view = view_of(state, 0, PUSH_MEM, opp_elixir=(5000, True), sent=SENT)
    spec = RB.build_spec(view, TABLES, RB.RebuildSettings(level=RB.R2), land_tick=T0 + 24,
                         fields=FIELD, opp_fill=[ARCHER, ARROWS, BABYD, ZAP])
    assert spec.elixir == (10_000, 9000) and "elixir_clamped" in spec.flags
    # an engine that charges a play when it is SENT (probe P16 "send"; not the live client,
    # which charges when the play runs): the Log's cost is added back too
    live = RB.build_spec(view_of(push_state(), 0, PUSH_MEM, opp_elixir=(5000, False), sent=SENT),
                         TABLES, RB.RebuildSettings(level=RB.R2, inflight_paid=True),
                         land_tick=T0 + 24, fields=FIELD, opp_fill=[ARCHER, ARROWS, BABYD, ZAP])
    assert live.base_elixir == (6000 + 2000, 5000) and live.elixir == (10_000, 9000)
    assert not live.opp_exact and RB.fidelity(live) == pytest.approx(0.8)
    # R1: no recasts, nothing added
    r1 = RB.build_spec(view, TABLES, RB.RebuildSettings(level=RB.R1), land_tick=T0 + 24,
                       fields=FIELD)
    assert r1.recasts == () and r1.elixir == (8500, 5000)
    assert len(r1.spawns) == 4 and play_out(r1.decks[0], []) == (list(r1.hands[0]),
                                                                 list(r1.queues[0]))


def test_root_steps_the_recasts_at_t_and_our_play_at_its_exec_tick(ps):
    rb, mod = rebuilder()
    spec = push_spec(rb)
    built = rb.root(spec)
    assert built is not None, rb.last_error
    b = mod.battles[-1]
    assert b is built.battle and b.reset_kwargs["decks"] == [list(d) for d in spec.decks]
    assert b.reset_kwargs["elixir_milli"] == [10_000, 9000] and b.reset_kwargs["start_tick"] == T0
    # both recasts in the first step (one command per team), the Log at 1014, then on to B
    assert steps(b) == [([(0, 2, *HOG_AIM), (1, 3, *mt(14500, 20000))], 1), ([], 13),
                        ([(0, 3, *LOG_AIM)], 1), ([], 9)]
    assert built.tick == T0 + 24 and b.names()[-1] == "save"
    assert built.hand(0) == [CANNON, FIREBALL, SKELS, MUSK] and b.queues[0] == [ICEGOLEM, SPIRITS,
                                                                                 HOG, LOG]
    assert built.hand(1) == [ARCHER, ARROWS, BABYD, ZAP] and b.queues[1] == list(spec.queues[1])
    # each side paid its recast (and we the Log): the counted bars are right again
    assert b.elixir == [6000 - 2000, 5000]
    assert built.report["relabel"] == {"matched": 2, "near": 0, "unmatched": 0, "unused": 0}
    assert built.report["ran"] == 3 and built.fidelity == 1.0
    assert rb.summary()["built"] == 1 and rb.last_error is None


def test_legal_maps_the_card_to_its_slot_at_b_and_trials_restore(ps):
    rb, mod = rebuilder()
    built = rb.root(push_spec(rb))
    b = built.battle
    x, y = mt(6000, 10000)
    assert rb.legal(MUSK, x, y, built) == 0
    assert b.deploy_checks[-1] == (0, 3, x, y)              # the Musketeer is in slot 3 at B
    checks = len(b.deploy_checks)
    assert rb.legal(LOG, x, y, built) == RB.NOT_IN_HAND     # the Log has cycled by B
    assert len(b.deploy_checks) == checks
    assert "load" not in b.names()                          # the root was untouched
    after, run = rb.trial(built, [(T0 + 24, 0, MUSK, x, y)], 20)
    assert after["tick"] == T0 + 44 and run["ran"] == [("play", 0, T0 + 24, 0)]
    new = [e for e in after["entities"] if e[FIELD["uid"]] >= 504]
    assert [(e[FIELD["card_id"]], e[FIELD["x"]], e[FIELD["y"]]) for e in new] == [(MUSK, x, y)]
    assert b.elixir[0] == 0
    # the next use loads the root back
    assert rb.legal(MUSK, x, y, built) == 0 and b.names().count("load") == 1
    assert b.elixir[0] == 4000 and b.hands[0] == built.hand(0)
    again, _ = rb.trial(built, [], 5, verify=True)
    assert again["tick"] == T0 + 29 and b.names().count("load") == 1   # clean after legal()
    rb.trial(built, [], 5, verify=True)
    assert b.names().count("load") == 2


def test_trial_sends_one_command_per_team_per_step(ps):
    rb, mod = rebuilder()
    spec = RB.RootSpec(level=RB.R1, team=0, tick=100, land_tick=100,
                       decks=(tuple(range(8)), tuple(range(8, 16))), elixir=(10_000, 10_000))
    built = rb.root(spec)
    assert built is not None, rb.last_error
    x, y = mt(9000, 10000)
    _, run = rb.trial(built, [(100, 0, CANNON, x, y), (100, 0, FIREBALL, x, y),
                              (100, 1, KNIGHT, x, y)], 10)
    b = built.battle
    assert steps(b) == [([(0, 0, x, y), (1, 0, x, y)], 1), ([(0, 1, x, y)], 1), ([], 8)]
    assert [r[:3] for r in run["ran"]] == [("play", 0, 100), ("play", 2, 100), ("play", 1, 101)]
    assert run["tick"] == 110 and not run["skipped"]


def test_a_refused_recast_is_spawned_instead_and_the_deck_redealt(ps):
    rb, mod = rebuilder(refuse=lambda team, card, x, y: OCCUPIED if card == MINIPEKKA else 0)
    spec = push_spec(rb)
    built = rb.root(spec)
    assert built is not None, rb.last_error
    b = built.battle
    assert b.names()[:2] == ["reset", "reset"]
    assert built.report["resets"] == 2 and built.report["recasts_refused_check"] == 1
    used = built.spec
    assert [r.card for r in used.recasts] == [HOG] and "recast_spawned" in built.flags
    assert used.decks[1] == (ARCHER, ARROWS, BABYD, ZAP, GIANT, VALK, KNIGHT, MINIPEKKA)
    assert used.elixir == (10_000, 5000)
    assert (1, MINIPEKKA, *mt(14500, 20000), 1390) in used.spawns
    assert MINIPEKKA in built.labels.values()                  # the spawned group is relabelled
    assert steps(b)[0] == ([(0, 2, *HOG_AIM)], 1)
    assert built.fidelity == 1.0                              # spawned, not lost


# ---------------------------------------------------------------------------
# multi-member plays: Goblins (staggered), a Goblin Gang (a second summon), a Goblin Giant
# (riders); the catalogue and spawn table above plus four cards and three unit types
# ---------------------------------------------------------------------------

GOBLIN_ROWS = [["Goblins", 0, 2, 4, 0, False, 0, 0, "TROOP"],
               ["GoblinGang", 0, 3, 3, 0, False, 0, 0, "TROOP"],
               ["SpearGoblins", 0, 2, 3, 0, False, 0, 0, "TROOP"],
               ["GoblinGiant", 0, 6, 1, 0, False, 0, 0, "TROOP"]]
CATALOGUE2 = CATALOGUE + GOBLIN_ROWS
NAMES2 = [r[0] for r in CATALOGUE2]
GOBLINS, GANG, SPEARS, GGIANT = range(len(CATALOGUE), len(CATALOGUE2))
UNITS2 = UNITS + ["Goblin", "SpearGoblin", "GoblinGiant"]
UID2 = {n: i for i, n in enumerate(UNITS2)}
PRIMARY2 = {**PRIMARY, "Goblins": "Goblin", "GoblinGang": "Goblin",
            "SpearGoblins": "SpearGoblin", "GoblinGiant": "GoblinGiant"}
HP2 = {**HP11, "Goblins": 202, "GoblinGang": 202, "SpearGoblins": 133, "GoblinGiant": 3000}
#: formation_preview's (unit, deploy ticks): member k waits k x the card's SummonDeployDelay
#: (Goblins 200 ms: 4 ticks; the Gang's goblins and its second summon 100 ms).
MEMBERS = {GOBLINS: (("Goblin", 20), ("Goblin", 24), ("Goblin", 28), ("Goblin", 32)),
           GANG: (("Goblin", 20), ("Goblin", 22), ("Goblin", 24),
                  ("SpearGoblin", 22), ("SpearGoblin", 24), ("SpearGoblin", 26)),
           WITCH: (("Witch", 20),)}


def table2(all_units=None):
    t = {"format": 1, "card_names": list(NAMES2), "unit_types": list(UNITS2),
         "primary": {str(NAMES2.index(c)): UID2[u] for c, u in PRIMARY2.items()},
         "hp": {str(NAMES2.index(c)): {"11": [["own", PRIMARY2[c], hp]]} for c, hp in HP2.items()}}
    if all_units:
        t["all"] = {str(NAMES2.index(c)): sorted(UID2[u] for u in us)
                    for c, us in all_units.items()}
    return t


def unit2(uid, team, name, card, x, y, hp, *, deploy=0):
    row = unit(uid, team, None, card, x, y, hp, deploy=deploy)
    row[FIELD["unit_type"]] = UID2[name]
    return row


def goblin_view(ents, last):
    """Seat 0 at T0: the opponent's latest play is `last` (its formation `ents` on the board)."""
    me = player_dict(0, elixir_milli=6000, hand=[CANNON, FIREBALL, SKELS, LOG], next_card=MUSK,
                     deck=list(range(8)))
    mem = memory([(MUSK, 700), (ICEGOLEM, 800), (SPIRITS, 900), (HOG, 950)],
                 [(GIANT, 600), (VALK, 700), (KNIGHT, 800), (last, T0 - 2)])
    state = state_dict(T0, players=[me, player_dict(1)], entities=ents)
    return view_of(state, 0, mem, opp_elixir=(5000, True))


def goblin_spec(view, tables, level=RB.R2):
    return RB.build_spec(view, tables, RB.RebuildSettings(level=level), land_tick=T0 + 24,
                         fields=FIELD, opp_fill=[ARCHER, ARROWS, BABYD, ZAP],
                         members=MEMBERS.get,
                         deploy_total=lambda c: max(t for _, t in MEMBERS.get(c, (("", 20),))),
                         tap_point=lambda team, card, x, y: (x + 5, y - 7))


GOBLIN_AT = ((8500, 19500), (9500, 19500), (8500, 20500), (9500, 20500))


def test_a_deploy_recast_takes_the_whole_play_or_none_of_it(ps):
    tables = RB.Tables(CATALOGUE2, table2(), level=11)
    # four Goblins two ticks after the play: member k has 30 - 4k ticks left (the stagger)
    goblins = [unit2(40 + k, 1, "Goblin", GOBLINS, *GOBLIN_AT[k], 202, deploy=30 - 4 * k)
               for k in range(4)]
    skeleton = unit2(50, 1, "Skeleton", WITCH, 12000, 22000, 81, deploy=10)   # a Witch's
    spec = goblin_spec(goblin_view(goblins + [skeleton], GOBLINS), tables)
    [rec] = spec.recasts
    assert (rec.team, rec.card, rec.kind, rec.order, rec.cost) == (1, GOBLINS, "deploy", T0 - 2,
                                                                   2000)
    assert (rec.x, rec.y) == (9000 * SUB + 5, 20000 * SUB - 7)       # the formation's tap point
    assert sorted(rec.uids) == [40, 41, 42, 43] and set(rec.labels) == {GOBLINS}
    assert rec.hp == 4 * 202
    # no goblin is spawned beside the recast (it lays all four); the skeleton is spawned
    assert spec.spawns == ((1, SKELS, *mt(12000, 22000), 81),) and spec.labels == (WITCH,)
    assert "recast_partial" not in spec.flags and spec.elixir[1] == 5000 + 2000
    # one goblin gone: not one whole play, so nothing is recast and the three are spawned
    spec = goblin_spec(goblin_view(goblins[:3], GOBLINS), tables)
    assert spec.recasts == () and "recast_partial" in spec.flags
    assert sorted(s[1] for s in spec.spawns) == [GOBLINS] * 3 and spec.elixir[1] == 5000
    # a Goblin Gang: its three goblins and its three Spear Goblins (a second summon, spawned as
    # SpearGoblins) are one play: one recast of the Gang, none of them spawned
    gang = ([unit2(60 + k, 1, "Goblin", GANG, 8500 + 500 * k, 20000, 202, deploy=22 - 2 * k)
             for k in range(3)]
            + [unit2(63 + k, 1, "SpearGoblin", GANG, 8500 + 500 * k, 21000, 133,
                     deploy=24 - 2 * k) for k in range(3)])
    spec = goblin_spec(goblin_view(gang, GANG), tables)
    [rec] = spec.recasts
    assert (rec.card, rec.order, rec.cost) == (GANG, T0 - 2, 3000)
    assert sorted(s[1] for s in rec.spawns) == [GANG] * 3 + [SPEARS] * 3
    assert set(rec.labels) == {GANG} and spec.spawns == ()
    # without formation_preview the play is the catalogue count deploying together: the
    # staggered goblins are spawned, never recast in part
    bare = RB.build_spec(goblin_view(goblins, GOBLINS), tables, RB.RebuildSettings(level=RB.R2),
                         land_tick=T0 + 24, fields=FIELD, opp_fill=[ARCHER, ARROWS, BABYD, ZAP],
                         deploy_total=lambda c: 32)
    assert bare.recasts == () and len(bare.spawns) == 4 and "recast_partial" in bare.flags


def test_a_spec_that_lays_more_than_its_unit_is_substituted_or_carries_it(ps):
    """The spawn table's "all": a Goblin Gang spec lays its Spear Goblins too, so its goblins
    spawn as Goblins; a Goblin Giant spec brings its riders, so they are not spawned again."""
    table = table2(all_units={"Goblins": ["Goblin"], "GoblinGang": ["Goblin", "SpearGoblin"],
                              "SpearGoblins": ["SpearGoblin"],
                              "GoblinGiant": ["GoblinGiant", "SpearGoblin"]})
    tables = RB.Tables(CATALOGUE2, table, level=11)
    assert tables.bundled == {GANG, GGIANT} and tables.alone[UID2["Goblin"]] == GOBLINS
    assert tables.spawn_card(GANG, UID2["Goblin"]) == GOBLINS
    assert tables.spawn_card(GGIANT, UID2["GoblinGiant"]) == GGIANT     # nothing lays it alone
    assert tables.carried(GGIANT, UID2["SpearGoblin"])
    assert not tables.carried(GANG, UID2["SpearGoblin"])                # the goblins are Goblins
    gang = ([unit2(60 + k, 1, "Goblin", GANG, 8500 + 500 * k, 20000, 202) for k in range(3)]
            + [unit2(63 + k, 1, "SpearGoblin", GANG, 8500 + 500 * k, 21000, 133)
               for k in range(3)])
    giant = [unit2(70, 1, "GoblinGiant", GGIANT, 4000, 22000, 3000),
             unit2(71, 1, "SpearGoblin", GGIANT, 3800, 22000, 133),
             unit2(72, 1, "SpearGoblin", GGIANT, 4200, 22000, 133)]
    spec = goblin_spec(goblin_view(gang + giant, GANG), tables, level=RB.R1)
    got = sorted(zip((s[1] for s in spec.spawns), spec.labels))
    assert got == [(GOBLINS, GANG)] * 3 + [(SPEARS, GANG)] * 3 + [(GGIANT, GGIANT)]
    assert {"spec_substituted", "carried"} <= set(spec.flags)
    assert (spec.units, spec.dropped, spec.lost_hp) == (9, 0, 0) and RB.fidelity(spec) == 1.0
    # the giant gone (its riders jumped off): they are spawned as Spear Goblins
    spec = goblin_spec(goblin_view(giant[1:], GGIANT), tables, level=RB.R1)
    assert sorted(s[1] for s in spec.spawns) == [SPEARS, SPEARS] and "carried" not in spec.flags


def test_a_side_without_three_towers_is_refused(ps):
    state = push_state()
    state["players"][1]["tower_hp"] = [KING_HP, PRINCESS_HP]            # a misread lost one
    view = view_of(state, 0, PUSH_MEM, opp_elixir=(5000, True), sent=SENT)
    spec = RB.build_spec(view, TABLES, RB.RebuildSettings(level=RB.R2), land_tick=T0 + 24,
                         fields=FIELD, opp_fill=[ARCHER, ARROWS, BABYD, ZAP])
    assert spec.tower_hp is None and spec.refusal == RB.REFUSED_TOWERS
    assert RB.fidelity(spec) == 0.0
    rb, mod = rebuilder()
    assert rb.root(spec) is None and rb.last_error.reason == "towers_unknown"
    assert mod.battles == []                                  # the battle is never touched


def test_forms_scale_a_spawn_by_its_forms_max_hp(ps):
    """An evolved Knight at half of its form's 2000 hp: with forms on (probe P5) it is reset as
    the evolution and spawned at 1000, not at half the base Knight's 1766."""
    knight = unit(30, 1, "Knight", KNIGHT, 4000, 20000, 1000, 2000, status=PV.STATUS_EVOLVED)
    full = unit(31, 1, "Knight", KNIGHT, 5000, 20000, 2000, 2000, status=PV.STATUS_EVOLVED)
    me = player_dict(0, hand=[CANNON, FIREBALL, SKELS, LOG], next_card=MUSK, deck=list(range(8)))
    view = view_of(state_dict(T0, players=[me, player_dict(1)], entities=[knight, full]), 0,
                   PUSH_MEM, opp_elixir=(5000, True))
    on = RB.build_spec(view, TABLES, RB.RebuildSettings(forms=True), fields=FIELD)
    assert on.forms_of == ((), ((KNIGHT, 1),))
    assert dict(zip(on.uids, (s[4] for s in on.spawns))) == {30: 1000, 31: 2000}
    off = RB.build_spec(view, TABLES, RB.RebuildSettings(), fields=FIELD)
    assert off.forms_of == ((), ()) and dict(zip(off.uids, (s[4] for s in off.spawns))) == {
        30: 883, 31: 1766}


class PlaceBattle(CycleBattle):
    """CycleBattle with building_placement: records (tick, entities on the board) per ask."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.placed = []

    def building_placement(self, team, name, x, y):
        self.placed.append((self.tick, len(self.ents)))
        return (x, y)


def test_snap_asks_on_a_blank_board_even_after_a_root_was_loaded_back(ps):
    mod = make_fake_royalesim()

    def Battle(*args, **kwargs):
        b = PlaceBattle()
        mod.battles.append(b)
        return b
    mod.Battle = Battle
    rb = RB.Rebuilder(rs_engine.EngineConfig.legacy(), mod, SPAWN_TABLE,
                      settings=RB.RebuildSettings(), catalogue=CATALOGUE, fields=FIELD)
    built = rb.root(push_spec(rb))
    assert built is not None, rb.last_error
    b = built.battle
    assert rb.snap(1, CANNON, *mt(9000, 20000)) == mt(9000, 20000)
    assert b.placed[-1] == (RB.BLANK_TICK, 0)                 # a blank board
    assert rb.legal(MUSK, *mt(6000, 10000), built) == 0       # the root is loaded back ...
    assert b.tick == T0 + 24 and len(b.ents) == 4
    assert rb.snap(1, CANNON, *mt(9000, 21000)) == mt(9000, 21000)
    assert b.placed[-1] == (RB.BLANK_TICK, 0)                 # ... and blanked again to ask


class PocketBattle(CycleBattle):
    """CycleBattle with formation_preview: a play's members 500 mt apart in x around the tap,
    each pulled back to its own side of the river (red y >= 17000 mt, blue <= 15000) unless the
    enemy princess of the tap's lane has fallen: royalesim drops its column clamp
    (formation.GROUND_Y_CLAMP) in a pocket. `previews` records (the board's tower_hp, team,
    name, x, y) per ask."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.previews = []

    def formation_preview(self, team, name, x, y):
        team, x, y = int(team), int(x), int(y)
        self.previews.append((copy.deepcopy(self.towers), team, name, x, y))
        n = CATALOGUE[CID[name]][3]
        lane = 1 if x < 9000 * SUB else 2
        if self.towers is None or self.towers[1 - team][lane] > 0:
            y = max(y, 17000 * SUB) if team == 1 else min(y, 15000 * SUB)
        return [(PRIMARY[name], x + (j - (n - 1) // 2) * 500 * SUB, y, 1000) for j in range(n)]


def pocket_rebuilder():
    mod = make_fake_royalesim()

    def Battle(*args, **kwargs):
        b = PocketBattle()
        mod.battles.append(b)
        return b
    mod.Battle = Battle
    rb = RB.Rebuilder(rs_engine.EngineConfig.legacy(), mod, SPAWN_TABLE,
                      settings=RB.RebuildSettings(), catalogue=CATALOGUE, fields=FIELD)
    return rb, mod


def test_tap_points_are_asked_on_a_blank_board_with_the_views_towers():
    """A swarm the opponent played into a pocket (our left princess fell): its tap point is
    asked on a blank board with that tower down, where the members stand where they were
    tapped, and cached per tower pattern. On a board with every tower standing they would be
    pulled back to the river and the tap pushed deeper by as much."""
    players = [player_dict(0, tower_hp=(KING_HP, 0, PRINCESS_HP)),
               player_dict(1, tower_hp=(0, None, PRINCESS_HP))]
    assert RB.standing_towers({"players": players}) == ((True, False, True), (True, False, True))
    short = [player_dict(0), player_dict(1, tower_hp=(KING_HP, PRINCESS_HP))]
    assert RB.standing_towers({"players": short}) == RB.ALL_STANDING
    assert RB.standing_towers({}) == RB.ALL_STANDING
    rb, mod = pocket_rebuilder()
    pocket = ((True, False, True), (True, True, True))
    at = mt(3500, 12000)
    assert rb.tap_point(1, SKELS, *at, towers=pocket) == at
    b = mod.battles[0]
    hp = RB.BLANK_TOWER_HP
    assert b.towers == [[hp, 0, hp], [hp, hp, hp]] and b.tick == RB.BLANK_TICK and b.ents == []
    assert rb.tap_point(1, SKELS, *at) == mt(3500, 7000)          # every tower standing
    assert b.towers is None
    asked = len(b.previews)
    assert rb.tap_point(1, SKELS, *at, towers=pocket) == at       # cached, per pattern
    assert rb.tap_point(1, SKELS, *at) == mt(3500, 7000)
    assert len(b.previews) == asked and len(b.calls_named("reset")) == 2


def test_a_pocket_play_is_recast_where_the_view_shows_it(ps):
    """Through Rebuilder.spec: the view's fallen princess reaches tap_point's board, so red's
    Skeletons deploying in that pocket are recast on their centroid, and the spec rebuilt from
    the same view keeps its key."""
    rb, mod = pocket_rebuilder()
    skels = [unit(70 + k, 1, "Skeleton", SKELS, 3000 + 500 * k, 12000, 81, deploy=18)
             for k in range(3)]
    me = player_dict(0, elixir_milli=6000, hand=[CANNON, FIREBALL, SKELS, LOG], next_card=MUSK,
                     deck=list(range(8)), tower_hp=(KING_HP, 0, PRINCESS_HP))
    mem = memory([(MUSK, 700), (ICEGOLEM, 800), (SPIRITS, 900), (HOG, 950)],
                 [(GIANT, 600), (VALK, 700), (KNIGHT, 800), (SKELS, T0 - 2)])
    view = view_of(state_dict(T0, players=[me, player_dict(1)], entities=skels), 0, mem,
                   opp_elixir=(5000, True))
    fill = [ARCHER, ARROWS, BABYD, ZAP]
    spec = rb.spec(view, land_tick=T0 + 24, level=RB.R2, opp_fill=fill)
    [rec] = spec.recasts
    assert (rec.team, rec.card, rec.kind, rec.order) == (1, SKELS, "deploy", T0 - 2)
    assert (rec.x, rec.y) == mt(3500, 12000)
    hp = RB.BLANK_TOWER_HP
    assert mod.battles[0].previews[-1] == ([[hp, 0, hp], [hp, hp, hp]], 1, "Skeletons",
                                           *mt(3500, 12000))
    assert rb.spec(view, land_tick=T0 + 24, level=RB.R2, opp_fill=fill).key == spec.key


def test_two_states_that_differ_only_in_what_no_screen_shows_give_one_spec(ps):
    clean, noised = push_spec(), push_spec(noised=True)
    assert noised == clean and noised.key == clean.key
    assert pickle.loads(pickle.dumps(clean)) == clean
    back = RB.RootSpec.from_json(json.loads(json.dumps(clean.to_json())))
    assert back == clean and back.key == clean.key
    assert RB.RootSpec(**{**clean.__dict__, "seed": 1}).key != clean.key


# ---------------------------------------------------------------------------
# spawns, refusals, fidelity
# ---------------------------------------------------------------------------

def test_spawns_are_clamped_scaled_and_snapped(ps):
    ents = [unit(40, 1, "Giant", GIANT, 5000, 22000, 5000, 4000),          # above its max
            unit(41, 1, "Valkyrie", VALK, 4000, 24000, 500, 2000, level=13),   # a quarter
            unit(42, 1, "Knight", KNIGHT, 9000, 16000, 1766),               # in the river
            unit(43, 0, "Musketeer", MUSK, -300, 5000, 721),                # off the board
            unit(44, 1, "Golemite", KNIGHT, 9000, 26000, 500),              # no card makes it
            unit(45, 1, "Cannon", CANNON, 9100, 20100, 824),                # a building
            unit(46, 1, "Knight", KNIGHT, 12000, 22000, 2000, 1766),
            unit(47, 1, "Skeleton", WITCH, 9000, 22000, 40, 81)]            # a Witch's skeleton
    players = [player_dict(0, hand=[0, 1, 2, 3], next_card=4, deck=list(range(8))),
               player_dict(1, hand=[], next_card=-1, deck=[])]
    view = view_of(state_dict(1500, players=players, entities=ents), 0)
    asked = []

    def snap(team, card, x, y):
        asked.append((team, card, x, y))
        return mt(9000, 20000) if card == CANNON else None

    spec = RB.build_spec(view, TABLES, RB.RebuildSettings(), fields=FIELD, snap=snap)
    assert asked == [(1, CANNON, *mt(9100, 20100))]
    got = dict(zip(spec.uids, zip(spec.spawns, spec.labels)))
    assert got[40] == ((1, GIANT, *mt(5000, 22000), 3968), GIANT)
    assert got[41] == ((1, VALK, *mt(4000, 24000), 477), VALK)        # 0.25 x 1907 at level 11
    assert got[42] == ((1, KNIGHT, *mt(9000, 17500), 1766), KNIGHT)   # to its own bank
    assert got[43] == ((0, MUSK, *mt(500, 5000), 721), MUSK)
    assert got[45] == ((1, CANNON, *mt(9000, 20000), 824), CANNON)
    assert got[46] == ((1, KNIGHT, *mt(12000, 22000), 1766), KNIGHT)
    assert got[47] == ((1, SKELS, *mt(9000, 22000), 40), WITCH)       # spawns as Skeletons
    assert 44 not in got and len(spec.spawns) == 7
    assert (spec.units, spec.dropped, spec.lost_hp) == (8, 1, 500)
    assert spec.unit_hp == 5000 + 500 + 1766 + 721 + 500 + 824 + 2000 + 40
    assert spec.refusal is None and not spec.opp_exact                 # no count: inexact
    assert RB.fidelity(spec) == pytest.approx((spec.unit_hp - 500) / spec.unit_hp * 0.8)
    for team, card, x, y, hp in spec.spawns:
        assert 1 <= hp <= OWN[card] and KIND[card] in ("TROOP", "BUILDING")


def test_a_spec_refuses_too_many_dropped_units_and_overtime_with_unequal_crowns(ps):
    golemites = [unit(50 + i, 1, "Golemite", KNIGHT, 9000, 25000, 300) for i in range(3)]
    knight = unit(60, 1, "Knight", KNIGHT, 9000, 22000, 1766)
    view = view_of(state_dict(1200, entities=golemites + [knight],
                              players=[player_dict(0, deck=list(range(8))), player_dict(1)]), 0)
    spec = RB.build_spec(view, TABLES, fields=FIELD)
    assert spec.refusal == RB.REFUSED_SPEC_DROPPED and (spec.dropped, spec.units) == (3, 4)
    assert RB.fidelity(spec) == 0.0
    rb, mod = rebuilder()
    assert rb.root(spec) is None and rb.last_error.reason == "units_dropped"
    assert mod.battles == []                                  # the battle is never touched
    # overtime, one princess down on one side only: the game would be over
    players = [player_dict(0, deck=list(range(8))),
               player_dict(1, tower_hp=(KING_HP, 0, PRINCESS_HP))]
    view = view_of(state_dict(3700, players=players), 0)
    spec = RB.build_spec(view, TABLES, fields=FIELD)
    assert spec.refusal == RB.REFUSED_OVERTIME and spec.tower_hp[1] == (KING_HP, 0, PRINCESS_HP)
    assert RB.build_spec(view_of(state_dict(3500, players=players), 0), TABLES,
                         fields=FIELD).refusal is None
    with pytest.raises(RB.RebuildError, match="public view"):
        RB.build_spec(state_dict(100), TABLES, fields=FIELD)


SIMPLE = RB.RootSpec(level=RB.R1, team=0, tick=100, land_tick=110,
                     decks=(tuple(range(8)), tuple(range(8, 16))), elixir=(5000, 5000))


def test_a_refused_reset_falls_back_with_its_reason():
    rb, mod = rebuilder(reset_error="spawn Knight at (1, 2): on water")
    assert rb.root(SIMPLE) is None
    assert rb.last_error == RB.Refusal("reset_refused",
                                       "ValueError: spawn Knight at (1, 2): on water")
    assert mod.battles[-1].names() == ["reset"]
    rb2, _ = rebuilder(over=True)
    assert rb2.root(SIMPLE) is None and rb2.last_error.reason == "game_over_at_reset"
    rb3, _ = rebuilder()
    rb3.battle.command_delay_ticks = lambda: (24, 24)
    assert rb3.root(SIMPLE) is None and rb3.last_error.reason == "delayed_battle"
    rb4, _ = rebuilder()
    rb4.battle.state_json = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    assert rb4.root(SIMPLE) is None and rb4.last_error == RB.Refusal("error", "RuntimeError: boom")
    assert rb.summary() == {"refused:reset_refused": 1, "roots": 1}
    built = rebuilder()[0].root(SIMPLE)
    assert built is not None and steps(built.battle) == [([], 10)] and built.tick == 110


def test_fidelity_arithmetic():
    spec = RB.RootSpec(level=RB.R1, team=0, tick=0, land_tick=0, unit_hp=1000, lost_hp=200)
    assert RB.fidelity(spec) == pytest.approx(0.8)
    assert RB.fidelity(spec, lost_hp=300) == pytest.approx(0.5)
    inexact = RB.RootSpec(**{**spec.__dict__, "opp_exact": False})
    assert RB.fidelity(inexact) == pytest.approx(0.64)
    assert RB.fidelity(inexact, inexact_elixir=0.5) == pytest.approx(0.4)
    assert RB.fidelity(spec, refused=True) == 0.0
    assert RB.fidelity(RB.RootSpec(**{**spec.__dict__, "refusal": "units_dropped"})) == 0.0
    empty = RB.RootSpec(level=RB.R1, team=0, tick=0, land_tick=0, opp_exact=False)
    assert RB.fidelity(empty) == pytest.approx(0.8)            # no units: nothing lost
    assert RB.fidelity(spec, lost_hp=5000) == 0.0
    good = SimpleNamespace(fidelity=0.7)
    assert RB.usable(good) and not RB.usable(SimpleNamespace(fidelity=0.69)) and not RB.usable(None)


# ---------------------------------------------------------------------------
# R0: RSAgent.rebuilt exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("public", [False, True])
def test_r0_reset_arguments_equal_rsagent_rebuilt(monkeypatch, public):
    pytest.importorskip("numpy")
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    ev = importlib.import_module("rs_eval")
    prim = rs_spawn_table.primary_of(SPAWN_TABLE)
    monkeypatch.setattr(ev, "_SPAWN", {"prim": prim, "by_unit": rs_spawn_table.by_unit(prim)})
    monkeypatch.setattr(ev, "_UNITS", list(UNITS))
    monkeypatch.setattr(ev.RR, "F", dict(FIELD))
    ents = [unit(10, 0, "Musketeer", MUSK, 6000, 8000, 400, 721),
            unit(11, 1, "Skeleton", WITCH, 9000, 22000, 81),          # a child: Skeletons
            unit(12, 1, "Skeleton_EV1", SKELS, 8000, 21000, 81),      # a form: the card itself
            unit(13, 1, "Golemite", KNIGHT, 9000, 25000, 300),        # no card: skipped
            unit(14, 1, "Knight", KNIGHT, 4000, 20000, 1766, status=2),   # invisible
            unit(15, 1, None, KNIGHT, 5000, 21000, 900),              # unit_type -1
            unit(16, 1, None, SKELS, 5500, 21000, 50, 81)]            # -1 reads the last name
    players = [player_dict(0, elixir_milli=6000, hand=[CANNON, FIREBALL, SKELS, LOG],
                           next_card=MUSK, deck=list(range(8))),
               player_dict(1, elixir_milli=7000, deck=list(range(8, 16)))]
    state = state_dict(1500, players=players, entities=ents)
    sent = [(1488, HOG, *mt(3500, 12000)), (1510, LOG, *mt(9000, 9000))]
    fake = make_fake_royalesim()
    agent = SimpleNamespace(engine=rs_engine.EngineConfig.legacy(),
                            visibility="public" if public else None)
    old = ev.RSAgent.rebuilt(agent, copy.deepcopy(state), 0, fake,
                             first=[c for (te, c, _, _) in sent if te > 1500])
    assert old is not None, getattr(agent, "rebuild_error", None)
    fake2 = make_fake_royalesim()
    rb = RB.Rebuilder(rs_engine.EngineConfig.legacy(), fake2, SPAWN_TABLE, catalogue=CATALOGUE,
                      fields=FIELD, settings=RB.RebuildSettings())
    spec = rb.spec_r0(copy.deepcopy(state), 0, sent=sent, lead=24, public=public)
    built = rb.root(spec)
    assert built is not None, rb.last_error
    assert fake2.battles[-1].reset_kwargs == old.reset_kwargs
    assert old.reset_kwargs["decks"][0] == [LOG, CANNON, FIREBALL, SKELS, MUSK, HOG, ICEGOLEM,
                                            SPIRITS]
    assert spec.replays == ((1510, LOG, *mt(9000, 9000)),) and spec.land_tick == 1524
    assert built.labels == {} and spec.flags == ("r0",) and not spec.opp_exact
    cards = [s[1] for s in spec.spawns]
    assert cards == [MUSK, SKELS, SKELS] + ([] if public else [KNIGHT]) + [SKELS]


# ---------------------------------------------------------------------------
# relabel, permanence, spells
# ---------------------------------------------------------------------------

def test_relabel_restores_the_creating_card_and_formations_match_the_nearest_spec(ps):
    ents = [unit(70, 1, "Skeleton", WITCH, 9000, 22000, 81),
            unit(71, 1, "Witch", WITCH, 9000, 23000, 839),
            unit(72, 0, "Knight", KNIGHT, 9000, 9000, 1766)]
    players = [player_dict(0, deck=list(range(8))), player_dict(1)]
    view = view_of(state_dict(800, players=players, entities=ents), 0)
    rb, mod = rebuilder()
    built = rb.root(rb.spec(view))
    assert built is not None, rb.last_error
    cards = {e[FIELD["uid"]]: e[FIELD["card_id"]] for e in built.state["entities"]
             if e[FIELD["tower_slot"]] < 0}
    b = built.battle
    spawned = {e[FIELD["uid"]]: e[FIELD["card_id"]] for e in b.ents}
    assert sorted(spawned.values()) == sorted([SKELS, WITCH, KNIGHT])   # spawned as Skeletons
    assert sorted(cards.values()) == sorted([WITCH, WITCH, KNIGHT])       # read back as the Witch's
    assert built.report["relabel"] == {"matched": 3, "near": 0, "unmatched": 0, "unused": 0}
    raw = {"entities": copy.deepcopy(b.ents)}
    assert sorted(e[FIELD["card_id"]] for e in rb.relabel(raw)["entities"]) == sorted(
        [WITCH, WITCH, KNIGHT])                                           # the current root's map
    # relabel off (the M6 ablation): the spawn card stays
    rb_off, _ = rebuilder(RB.RebuildSettings(relabel=False))
    off = rb_off.root(rb_off.spec(view))
    assert sorted(e[FIELD["card_id"]] for e in off.state["entities"]
                  if e[FIELD["tower_slot"]] < 0) == sorted([SKELS, WITCH, KNIGHT])
    # one spec per formation (probe P1 FAIL): three skeletons, one spec, three members matched
    skel = [unit(80 + i, 1, "Skeleton", SKELS, 8500 + 500 * i, 22000, 81) for i in range(3)]
    view = view_of(state_dict(800, players=players, entities=skel), 0)
    rb_f, _ = rebuilder(RB.RebuildSettings(spec_per="formation"), formation=True)
    spec = rb_f.spec(view)
    assert spec.spawns == ((1, SKELS, *mt(9000, 22000), 81),)
    built = rb_f.root(spec)
    assert built.report["relabel"] == {"matched": 1, "near": 2, "unmatched": 0, "unused": 0}
    assert len(built.labels) == 3 and set(built.labels.values()) == {SKELS}


def test_a_hidden_tesla_kept_by_permanence_is_spawned(ps):
    players = [player_dict(0, deck=list(range(8))), player_dict(1)]
    keep = PV.Permanence(fields=FIELD, team=0, hiders=[TESLA])
    seen = view_of(state_dict(1000, players=players,
                              entities=[unit(90, 1, "Tesla", TESLA, 9000, 22000, 1000)]), 0)
    keep.apply(seen)
    hidden = view_of(state_dict(1010, players=players,
                                entities=[unit(90, 1, "Tesla", TESLA, 9000, 22000, 1000,
                                               status=PV.STATUS_HIDDEN)]), 0)
    assert hidden["entities"][-1][FIELD["uid"]] != 90           # the view drops it ...
    kept = keep.apply(hidden)
    assert kept["public"]["kept"] == [90]                       # ... permanence keeps it
    spec = RB.build_spec(kept, TABLES, fields=FIELD)
    assert spec.spawns == ((1, TESLA, *mt(9000, 22000), 983),)  # drained 10 ticks of 600
    assert spec.labels == (TESLA,) and "permanence" in spec.flags


def spell_view():
    """t 2000, seat 0: our Log rolling (sent at 1990), an enemy Arrows flying at our Musketeer
    and king, a troop's effect; Arrows is the opponent's latest play, the Log ours."""
    SF = {n: i for i, n in enumerate(PV.SPELL_FIELDS_FALLBACK)}

    def spell(team, card, x, y, flown=0):
        row = [0] * len(SF)
        row[SF["team"]], row[SF["card_id"]] = team, card
        row[SF["x"]], row[SF["y"]] = mt(x, y)
        row[SF["aim_x"]], row[SF["aim_y"]] = mt(9000, 1000)   # the true aim: never public
        row[SF["ticks_flown"]] = flown
        return row

    players = [player_dict(0, elixir_milli=5000, hand=[CANNON, FIREBALL, HOG, SKELS],
                           next_card=MUSK, deck=list(range(8))), player_dict(1)]
    state = state_dict(2000, players=players,
                       entities=[unit(95, 0, "Musketeer", MUSK, 9000, 9100, 721)],
                       spells=[spell(0, LOG, 9000, 13000), spell(1, ARROWS, 9000, 20100, 12),
                               spell(1, KNIGHT, 4000, 20000)])
    mem = memory([(MUSK, 1700), (ICEGOLEM, 1800), (SPIRITS, 1900), (LOG, 1990)],
                 [(GIANT, 1500), (VALK, 1600), (KNIGHT, 1700), (ARROWS, 1988)])
    return view_of(state, 0, mem, opp_elixir=(6000, False),
                   sent=[(1990, LOG, *mt(9000, 12000))])


def test_infer_aim_walks_the_ray_from_the_casting_king():
    targets = [(9000.0, 9100.0, 721, 0.0), (9000.0, 3000.0, 4824, 2000.0),
               (3500.0, 6500.0, 3052, 1500.0), (14500.0, 6500.0, 3052, 1500.0)]
    # from the red king (9000, 29000) through (9000, 20100): the first point that reaches both
    # the Musketeer and the king (5500 mt) on the 250 mt walk is y 8350
    assert RB.infer_aim(1, 9000, 20100, targets, radius_mt=3500, cap=370) == (9000.0, 8350.0, 740)
    assert RB.infer_aim(1, 9000, 20100, [], radius_mt=3500) is None
    assert RB.infer_aim(1, 9000, 29000, targets) is None        # no ray from the king itself


def test_r2_recasts_spells_at_known_or_inferred_aims_and_r1_drops_them(ps):
    view = spell_view()
    spec = RB.build_spec(view, TABLES, RB.RebuildSettings(level=RB.R2), fields=FIELD)
    got = {(r.team, r.card): r for r in spec.recasts}
    log, arrows = got[(0, LOG)], got[(1, ARROWS)]
    assert (log.kind, log.x, log.y, log.order) == ("log", *mt(9000, 12000), 1990)    # our aim
    assert (arrows.kind, arrows.x, arrows.y, arrows.order) == ("spell", *mt(9000, 8350), 1988)
    assert (log.cost, arrows.cost) == (2000, 3000)
    assert "effect_dropped" in spec.flags and len(spec.recasts) == 2
    assert spec.elixir == (5000 + 2000, 6000 + 3000) and not spec.opp_exact
    assert spec.decks[0][:4] == (CANNON, FIREBALL, HOG, LOG) and ARROWS in spec.decks[1][:4]
    assert play_out(spec.decks[0], [LOG]) == (list(spec.hands[0]), list(spec.queues[0]))
    assert play_out(spec.decks[1], [ARROWS]) == (list(spec.hands[1]), list(spec.queues[1]))
    r1 = RB.build_spec(view, TABLES, RB.RebuildSettings(level=RB.R1), fields=FIELD)
    assert r1.recasts == () and "spells_dropped" in r1.flags and r1.elixir == (5000, 6000)
    # an enemy Log with no aim of ours: its first sighting
    sights = RB.SpellSightings()
    first = copy.deepcopy(view)
    first["tick"] = 1995
    first["spells"] = [[1, LOG] + r[2:] for r in first["spells"][:1]]
    sights.update(first)
    later = copy.deepcopy(first)
    later["tick"] = 2000
    later["spells"][0][PV.SPELL_FIELDS_FALLBACK.index("y")] = mt(9000, 12500)[1]
    sights.update(later)
    assert sights.first(1, LOG) == (1995, *mt(9000, 13000))


# ---------------------------------------------------------------------------
# settings from the S01 probes
# ---------------------------------------------------------------------------

def test_probe_settings_follow_the_engine_report(tmp_path):
    report = {"probes": [
        {"id": "P1", "ok": False}, {"id": "P2", "ok": True},
        {"id": "P13", "ok": True, "answer": {"HogRider": {"deploy_ticks": [19]}}},
        {"id": "P14", "ok": False}, {"id": "P5", "ok": True},
        {"id": "P16", "ok": False, "answer": {"elixir": "send", "hand": "send"}}]}
    s = RB.RebuildSettings.from_report(report)
    assert (s.spec_per, s.deploy_recasts, s.omit_rule, s.forms, s.inflight_paid) == (
        "formation", True, False, True, True)
    assert s.probes == "report" and len(s.notes) == 4
    p2 = {"id": "P2", "ok": False}
    assert not RB.RebuildSettings.from_report({"probes": [p2]}).deploy_recasts
    played = {"id": "P13", "ok": False, "answer": {"HogRider": {"deploy_ticks": [0]}}}
    assert not RB.RebuildSettings.from_report({"probes": [played]}).deploy_recasts
    run = {"id": "P16", "ok": True, "answer": {"elixir": "run"}}
    assert not RB.RebuildSettings.from_report({"probes": [run]}).inflight_paid
    d = RB.RebuildSettings.from_report(search=False)
    assert d == RB.RebuildSettings() and d.probes == "defaults"
    assert RB.RebuildSettings.from_report(search=False, level="R2").level == "R2"
    # the newest engine report of a folder that has probes: a later timings-only run
    # (--no-probes) does not hide them
    (tmp_path / "engine_20261030_000000.json").write_text(json.dumps({"probes": []}))
    (tmp_path / "engine_20261030_000001.json").write_text(json.dumps(report))
    (tmp_path / "engine_20261030_000002.json").write_text(json.dumps({"timings": {"x": 1}}))
    got = RB.load_report(out_dir=tmp_path)
    assert got["probes"] == report["probes"] and got["_path"].endswith("000001.json")
    assert RB.load_report(out_dir=tmp_path / "none") is None
    s2 = RB.RebuildSettings.from_report(path=tmp_path / "engine_20261030_000001.json")
    assert s2.spec_per == "formation" and s2.probes.endswith("000001.json")
    # live: the client charges a play when it runs, so in-flight costs are never added back,
    # whatever the simulator's engine does; the rest is the settings the gate ran
    assert not RB.RebuildSettings.live().inflight_paid
    live = RB.RebuildSettings.live(s)
    assert not live.inflight_paid and (live.spec_per, live.forms) == ("formation", True)
    assert RB.RebuildSettings.live(level="R2").level == "R2"
    assert RB.RebuildSettings.from_json(s.to_json()) == s
    # the key: what changes a rebuild, not where the values came from
    assert s.key == RB.RebuildSettings.from_json(s.to_json()).key
    assert s.replace(probes="elsewhere", notes=("n",)).key == s.key
    assert s.replace(level="R2").key != s.key and len(s.key) == 40
    with pytest.raises(ValueError):
        RB.RebuildSettings(level="R3")
    with pytest.raises(ValueError):
        RB.RebuildSettings(min_fidelity=1.5)


def test_a_rebuilder_without_settings_never_reads_a_report(tmp_path):
    """A report written between two gate runs must not change an arm: no settings, the
    defaults (whatever reports/search_bench holds); the probes only when asked for."""
    (tmp_path / "engine_20261030_000001.json").write_text(json.dumps(
        {"probes": [{"id": "P1", "ok": False}, {"id": "P16", "answer": {"elixir": "send"}}]}))
    rb = RB.Rebuilder(rs_engine.EngineConfig.legacy(), cycle_rs(), SPAWN_TABLE,
                      catalogue=CATALOGUE, fields=FIELD)
    assert rb.settings == RB.RebuildSettings() and rb.settings.probes == "defaults"
    asked = RB.RebuildSettings.from_report(path=tmp_path / "engine_20261030_000001.json")
    assert (asked.spec_per, asked.inflight_paid) == ("formation", True)


def test_with_present_keeps_the_present_hand_and_elixir():
    future = state_dict(124, players=[player_dict(0, elixir_milli=9000, hand=[4, 5, 6, 7]),
                                      player_dict(1, elixir_milli=1000)])
    now = state_dict(100, players=[player_dict(0, elixir_milli=3000, hand=[0, 1, 2, 3],
                                               next_card=4, deck=list(range(8))),
                                   player_dict(1, elixir_milli=0, hand=[], next_card=-1)])
    out = RB.with_present(future, now)
    assert out["tick"] == 124 and out["players"][0]["hand"] == [0, 1, 2, 3]
    assert out["players"][0]["elixir_milli"] == 3000 and out["players"][0]["deck"] == list(range(8))
    assert out["players"][1]["next_card"] == -1 and future["players"][0]["hand"] == [4, 5, 6, 7]


def test_imports_neither_numpy_torch_royalesim_nor_the_memory_reader():
    code = ("import sys\n"
            "import tools.rs_rebuild\n"
            "heavy = ('numpy', 'torch', 'royalesim', 'scripts.gamestate', 'scripts.vision',\n"
            "         'public_state', 'gamestate.memread')\n"
            "print(','.join(m for m in heavy if m in sys.modules))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

ENGINE_CARDS = ("HogRider", "Knight", "Musketeer", "Cannon", "Witch", "MiniPekka")


@pytest.fixture(scope="module")
def engine():
    rs = pytest.importorskip("royalesim")
    pytest.importorskip("numpy")
    cfg = rs_engine.EngineConfig.load().undelayed()
    b = rs_engine.make_battle(cfg, rs)
    cat = json.loads(b.catalogue_json())
    ids = {str(r[0]): i for i, r in enumerate(cat)}
    missing = [n for n in ENGINE_CARDS if n not in ids]
    if missing:
        pytest.skip(f"this engine's catalogue lacks {missing}")
    table = rs_spawn_table.build(cfg, rs, cards=[ids[n] for n in ENGINE_CARDS])
    return SimpleNamespace(rs=rs, cfg=cfg, cat=cat, ids=ids, table=table,
                           F=rs_engine.field_index(rs), sub=int(rs.SUBTILE_PER_MILLITILE),
                           SF=getattr(rs, "SPELL_FIELDS", None),
                           deck=rs_spawn_table.filler_deck(cat))


def _at(E, x, y):
    return int(round(x * E.sub)), int(round(y * E.sub))


def _rebuilder(E, level):
    return RB.Rebuilder(E.cfg, E.rs, E.table, settings=RB.RebuildSettings(level=level), fields=E.F)


def _deploying_view(E):
    """A real battle: red plays a Mini P.E.K.K.A at tick 600; the view of blue at 603 (it is
    deploying), with blue's Knight on the board."""
    pekka = E.ids["MiniPekka"]
    red = ([pekka] + [c for c in E.deck if c != pekka])[:8]
    b = rs_engine.make_battle(E.cfg, E.rs)
    b.reset(seed=5, decks=[list(E.deck), red], shuffle=0, start_tick=600,
            elixir_milli=[5000, 10_000], tower_hp=None,
            spawns=[(0, E.ids["Knight"], *_at(E, 9000, 9000), -1)])
    res = b.step([(1, 0, *_at(E, 14500, 20000))], 1)
    assert int(res[0][1]) == 0, res
    b.step([], 2)
    state = json.loads(b.state_json())
    F = E.F
    mine = [e for e in state["entities"] if e[F["team"]] == 1 and e[F["card_id"]] == pekka
            and e[F["tower_slot"]] < 0]
    if not mine or int(mine[0][F["deploy_ticks"]]) <= 0:
        pytest.skip("a played Mini P.E.K.K.A shows no deploy timer here (probe P13)")
    mem = memory([], [(pekka, 600)])
    return PV.public_view(state, 0, mem, fields=F, spell_fields=E.SF)


@pytest.mark.engine
def test_engine_relabel_round_trip_covers_every_spawn(engine):
    E, F = engine, engine.F
    b = rs_engine.make_battle(E.cfg, E.rs)
    spawns = [(0, E.ids["HogRider"], *_at(E, 3500, 12000), -1),
              (0, E.ids["Musketeer"], *_at(E, 6000, 8000), 300),
              (1, E.ids["Knight"], *_at(E, 4000, 20000), -1),
              (1, E.ids["Knight"], *_at(E, 12000, 21000), -1),
              (1, E.ids["Cannon"], *_at(E, 9000, 24000), -1)]
    b.reset(seed=3, decks=[list(E.deck), list(E.deck)], shuffle=0, start_tick=600,
            elixir_milli=[5000, 5000], tower_hp=None, spawns=spawns)
    b.step([], 5)
    state = json.loads(b.state_json())
    # one Knight reads as a Witch's (a creating card it is not the unit of): relabel restores it
    knights = [e for e in state["entities"] if e[F["team"]] == 1 and e[F["tower_slot"]] < 0
               and e[F["card_id"]] == E.ids["Knight"]]
    knights[0][F["card_id"]] = E.ids["Witch"]
    view = PV.public_view(state, 0, None, fields=F, spell_fields=E.SF)
    rb = _rebuilder(E, RB.R1)
    spec = rb.spec(view, land_tick=view["tick"])
    built = rb.root(spec)
    assert built is not None, rb.last_error
    assert len(spec.spawns) == 5
    assert built.report["relabel"] == {"matched": 5, "near": 0, "unmatched": 0, "unused": 0}
    assert len(built.labels) == 5
    units = [e for e in built.state["entities"] if e[F["tower_slot"]] < 0]
    assert sorted(e[F["card_id"]] for e in units) == sorted(
        [E.ids["HogRider"], E.ids["Musketeer"], E.ids["Witch"], E.ids["Knight"], E.ids["Cannon"]])


@pytest.mark.engine
def test_engine_a_recast_deploys(engine):
    E, F = engine, engine.F
    view = _deploying_view(E)
    pekka = E.ids["MiniPekka"]
    rb = _rebuilder(E, RB.R2)
    spec = rb.spec(view, land_tick=view["tick"] + 1)
    assert (1, pekka) in [(r.team, r.card) for r in spec.recasts]
    built = rb.root(spec)
    assert built is not None, rb.last_error
    assert built.report["recasts_refused_check"] == 0 and built.report["ran"] >= 1
    got = [e for e in built.state["entities"] if e[F["team"]] == 1 and e[F["card_id"]] == pekka
           and e[F["tower_slot"]] < 0]
    assert got and all(int(e[F["deploy_ticks"]]) > 0 for e in got)


@pytest.mark.engine
def test_engine_a_root_rebuilt_twice_gives_the_same_state_hash(engine):
    E = engine
    view = _deploying_view(E)
    rb = _rebuilder(E, RB.R2)
    spec = rb.spec(view, land_tick=view["tick"] + 10)
    one = rb.root(spec)
    assert one is not None, rb.last_error
    if not callable(getattr(one.battle, "state_hash", None)):
        pytest.skip("this engine has no Battle.state_hash")
    h1, s1 = one.battle.state_hash(), one.battle.state_json()
    two = rb.root(rb.spec(view, land_tick=view["tick"] + 10))
    assert two.key == one.key
    assert two.battle.state_hash() == h1 and two.battle.state_json() == s1
    a, _ = rb.trial(two, [], 30)
    b, _ = rb.trial(two, [], 30, verify=True)
    assert a == b
