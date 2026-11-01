"""The decision rules (tools/rs_decide.py DECISION RULES): the wait floor (rule 6) and the stray
cheap-troop guard (rule 7), both off by default, and their switches through rs_eval, rs_match
and rs_league_eval. Synthetic states only.

- the wait floor: holds a play the draw would have made at p_wait >= the floor (reason
  "floor"), not below it; a drawn wait keeps its reason; the cap rule wins (live at 9.5
  spendable, and the same line without a cap rule), in-flight plays lower spendable under it;
  one random() per decision with or without the floor;
- the stray guard: audit episode 9 (game 1 t1587: Skeletons at own (8, 17) while a P.E.K.K.A
  came down the other lane at own (15.3, 13.5), still on its side) is turned down and the next
  tile, own (9, 17), plays; for team 0 too; with no tile left a wait in every preset, never
  another card (not a Hog Rider at the bridge); kept: no threat at all (back-of-king cycle
  plays), a threat still deep on its side, an enemy building alone, the bridge tiles of a lane
  we push in (not the far lane's without one), tiles next to our towers, the enemy half, other
  cards; a building of ours pulls the path, a fallen princess sends it to the king, a flyer
  goes straight;
- the defaults: decide() draw for draw the decide of commit ebf6371 (the last before the rules)
  on random cases in every preset (git show; skipped without git or that commit), and the
  sampling preset refuses the rules (RL unchanged);
- the switches: rs_eval's flags, play() and decide_config (the agent seat only), the counts;
  rs_match AgentSpec (to_dict unchanged while off), configure, MatchAgent.decide_config,
  seat_info, rules_summary, the command line; a league player's keys reach the seat, and the
  same checkpoint plays under two names.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import json
import math
import subprocess
import sys
import types
import zlib
from collections import Counter
from pathlib import Path

import pytest

from tests.pipeline.conftest import (DEFAULT_CATALOGUE, FIELD, KING_HP, PRINCESS_HP, entity_row,
                                     player_dict, state_dict)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for _path in (str(ROOT), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import rs_decide as RD  # noqa: E402
import rs_engine  # noqa: E402

#: The last commit before the decision rules: decide() with the defaults must be its decide.
BASE_COMMIT = "ebf6371"
SUB = 18
CAT = [list(r) for r in DEFAULT_CATALOGUE]
CID = {row[0]: i for i, row in enumerate(CAT)}
SKEL, SPIRITS, CANNON, HOG, MUSK, GOLEM = (CID[n] for n in (
    "Skeletons", "IceSpirits", "Cannon", "HogRider", "Musketeer", "IceGolemite"))
STRAY = RD.card_ids(CAT, RD.STRAY_CARD_NAMES)
ROW = {"tick": 1587}


@pytest.fixture(autouse=True)
def clean_engine(monkeypatch):
    for var in (rs_engine.CONFIG_ENV, *rs_engine.ENV.values()):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rs_engine, "_ACTIVE", None)


def T(tx: int, ty: int) -> int:
    return ty * 18 + tx


def ranked(*tiles: int) -> np.ndarray:
    """Tile logits: `tiles` best first, then every other tile by index."""
    t = -10.0 - np.arange(RD.TILES, dtype=np.float32) / 1000
    for i, k in enumerate(tiles):
        t[k] = 100.0 - i
    return t


class FakePolicy:
    def __init__(self, gate, tiles=None):
        self.g = np.asarray(gate, dtype=np.float32)
        self.t = dict(tiles or {})

    def batch(self, row):
        return row

    def gate(self, b):
        return self.g.copy()

    def tiles(self, b, slot):
        return self.t.get(int(slot), ranked())


class StubRng:
    def __init__(self, u: float):
        self.u, self.draws = u, 0

    def random(self) -> float:
        self.draws += 1
        return self.u


def always(slot, card, x, y):
    return True


def only(team, *tiles):
    """Legality: just these own-frame tiles."""
    allowed = set(tiles)

    def check(slot, card, x, y):
        ox, oy = RD.to_world(x / SUB, y / SUB, team)
        return (int(ox // 1000), int(oy // 1000)) in allowed
    return check


def gate_for(p_wait: float, slot_logits=(3.0, 2.0, 1.0, 0.0)) -> list[float]:
    """Gate logits whose softmax gives the wait `p_wait` (wait bias 0, no slot masked)."""
    s = sum(math.exp(v) for v in slot_logits)
    return [math.log(p_wait / (1 - p_wait) * s), *slot_logits]


def state(*, team=1, hand=(SKEL, CANNON, HOG, MUSK), elixir=4632, entities=(), costs=None,
          my_towers=None, pending=(), pending_cost=0):
    mine = player_dict(team, hand=list(hand), elixir_milli=elixir, hand_costs=costs,
                       pending=[list(r) for r in pending], pending_cost=pending_cost,
                       **({} if my_towers is None else {"tower_hp": list(my_towers)}))
    theirs = player_dict(1 - team)
    return state_dict(ROW["tick"], players=[mine, theirs] if team == 0 else [theirs, mine],
                      entities=list(entities))


def unit(team, own_x, own_y, *, me=1, kind=0, hp=3000, flying=False, uid=50, card=-1):
    """An entity at own-frame millitiles of seat `me`, on `team`."""
    wx, wy = RD.to_world(own_x, own_y, me)
    return entity_row(uid=uid, team=team, kind=kind, card_id=card, x=int(wx * SUB),
                      y=int(wy * SUB), hp=hp, flying=flying)


def episode9(me=1):
    """Game 1 t1587: the P.E.K.K.A (logged as a Dark Prince) at screen (2.7, 13.5), own
    (15.3, 13.5) for the live seat, still on its side; our Musketeer at screen (8.6, 24.8)."""
    return [unit(1 - me, 15_300, 13_500, me=me, uid=50, card=CID["MiniPekka"]),
            unit(me, 9_400, 24_800, me=me, uid=51, hp=700, card=MUSK)]


LIVE = RD.DecideConfig.live(0.0)
LIVE_STRAY = LIVE.replace(stray_cards=STRAY)
LEGACY_STRAY = RD.DecideConfig.legacy(0.0).replace(stray_cards=STRAY)


# ---------------------------------------------------------------------------
# rule 6: the wait floor
# ---------------------------------------------------------------------------

def test_wait_floor_holds_a_play_the_draw_would_make():
    """p_wait 0.986 (the Cannon of game 1 t2178): q = 0.993 at dt 10; a draw of 0.999 plays
    without the floor and waits ("floor") with it; one draw either way."""
    pol = FakePolicy(gate_for(0.986), {0: ranked(T(9, 23))})
    st = state(hand=(CANNON, SKEL, HOG, MUSK), elixir=9000)          # nothing masked
    for cfg in (LIVE, RD.DecideConfig.legacy(0.0)):
        rng = StubRng(0.999)
        d = RD.decide(pol, ROW, st, 1, always, cfg, rng=rng)
        assert d.kind == RD.PLAY and d.card == CANNON and rng.draws == 1
        rng = StubRng(0.999)
        f = RD.decide(pol, ROW, st, 1, always, cfg.replace(wait_floor=0.95), rng=rng)
        assert (f.kind, f.reason, f.command, rng.draws) == (RD.WAIT, RD.R_FLOOR, None, 1)
        assert f.p_wait == pytest.approx(0.986, abs=1e-6) == d.p_wait and f.q_wait == d.q_wait
        # a drawn wait stays a drawn wait
        w = RD.decide(pol, ROW, st, 1, always, cfg.replace(wait_floor=0.95), rng=StubRng(0.5))
        assert (w.kind, w.reason) == (RD.WAIT, RD.R_WAIT)


def test_wait_floor_does_not_fire_below_it():
    pol = FakePolicy(gate_for(0.94), {0: ranked(T(9, 23))})
    st = state(hand=(CANNON, SKEL, HOG, MUSK), elixir=9000)
    d = RD.decide(pol, ROW, st, 1, always, LIVE.replace(wait_floor=0.95), rng=StubRng(0.999))
    assert d.kind == RD.PLAY and d.p_wait == pytest.approx(0.94, abs=1e-6)
    # exactly at the floor it holds
    at = FakePolicy(gate_for(0.95), {0: ranked(T(9, 23))})
    p = RD.decide(at, ROW, st, 1, always, LIVE, rng=StubRng(0.999)).p_wait
    f = RD.decide(at, ROW, st, 1, always, LIVE.replace(wait_floor=p), rng=StubRng(0.999))
    assert f.reason == RD.R_FLOOR


def test_the_cap_rule_wins_over_the_wait_floor():
    """At 9.6 spendable with a card in hand the bot plays (live: capped; legacy, which has no
    cap rule: the floor stands aside at the same line and the draw decides, as without it);
    at 9.4 the floor holds; a play in flight brings spendable under the cap again."""
    pol = FakePolicy(gate_for(0.99), {0: ranked(T(9, 23))})
    floor = 0.95
    live = LIVE.replace(wait_floor=floor)
    d = RD.decide(pol, ROW, state(elixir=9600), 1, always, live, rng=StubRng(0.999))
    assert d.kind == RD.PLAY and d.capped
    d = RD.decide(pol, ROW, state(elixir=9600), 1, always, live, rng=StubRng(0.0))
    assert d.kind == RD.PLAY and d.capped                    # capped: not even a drawn wait
    legacy = RD.DecideConfig.legacy(0.0).replace(wait_floor=floor)
    d = RD.decide(pol, ROW, state(elixir=9600), 1, always, legacy, rng=StubRng(0.999))
    assert d.kind == RD.PLAY and not d.capped
    d = RD.decide(pol, ROW, state(elixir=9600), 1, always, legacy, rng=StubRng(0.5))
    assert (d.kind, d.reason) == (RD.WAIT, RD.R_WAIT)        # as without the floor
    for cfg in (live, legacy):
        d = RD.decide(pol, ROW, state(elixir=9400), 1, always, cfg, rng=StubRng(0.999))
        assert (d.kind, d.reason) == (RD.WAIT, RD.R_FLOOR)
    tap = RD.Pending(MUSK, 4, 3, 1580)                       # 9.8 - 4 = 5.8 spendable
    d = RD.decide(pol, ROW, state(elixir=9800), 1, always, live, rng=StubRng(0.999),
                  pending=[tap])
    assert (d.kind, d.reason, d.capped, d.spendable_milli) == (RD.WAIT, RD.R_FLOOR, False, 5800)
    # nothing playable stays "no_card"
    d = RD.decide(pol, ROW, state(elixir=0), 1, always, live, rng=StubRng(0.999))
    assert d.reason == RD.R_NO_CARD


def test_wait_floor_keeps_the_random_stream():
    """The floor never adds or drops a draw: a seeded run replays the same stream with it."""
    r = np.random.default_rng(3)
    a, b = np.random.default_rng(11), np.random.default_rng(11)
    for _ in range(200):
        pol = FakePolicy(r.normal(0, 3, size=5), {s: r.normal(size=RD.TILES) for s in range(4)})
        st = state(elixir=int(r.integers(0, 10_001)))
        RD.decide(pol, ROW, st, 1, always, LIVE, rng=a)
        RD.decide(pol, ROW, st, 1, always, LIVE.replace(wait_floor=0.9), rng=b)
    assert a.bit_generator.state == b.bit_generator.state


# ---------------------------------------------------------------------------
# rule 7: the stray guard
# ---------------------------------------------------------------------------

def test_episode_9_centre_river_skeletons_are_turned_down():
    """Own (8, 17) is 6.0 tiles from the P.E.K.K.A's path down the right lane (own x 14.5 from
    the bridge exit (14.5, 17) to our princess tower (14.5, 25.5)): turned down; (9, 17), 5.0
    tiles, plays. The live seat (team 1) and team 0 alike."""
    pol = FakePolicy([-5, 3, 2, 1, 0], {0: ranked(T(8, 17), T(9, 17))})
    for team in (1, 0):
        st = state(team=team, entities=episode9(team))
        g = RD.stray_guard(st, team, SUB)
        assert g.paths == (((15_300.0, 13_500.0), (14_500, 17_000), (14_500.0, 25_500.0)),)
        assert (g.reach(8, 17), g.reach(9, 17)) == (6_000.0, 5_000.0)
        off = RD.decide(pol, ROW, st, team, always, LIVE, rng=StubRng(0.99))
        assert (off.card, off.tx, off.ty, off.stray) == (SKEL, 8, 17, 0)
        for cfg in (LIVE_STRAY, LEGACY_STRAY):
            on = RD.decide(pol, ROW, st, team, always, cfg, rng=StubRng(0.99))
            assert (on.kind, on.card, on.tx, on.ty, on.stray) == (RD.PLAY, SKEL, 9, 17, 1)
    # Ice Spirits are guarded too; the Musketeer, the Ice Golem and the Cannon are not.
    for card, guarded in ((SPIRITS, True), (MUSK, False), (GOLEM, False), (CANNON, False)):
        st = state(hand=(card, HOG, SKEL, MUSK), elixir=9000, entities=episode9())
        d = RD.decide(pol, ROW, st, 1, always, LIVE_STRAY, rng=StubRng(0.99))
        assert (d.card, d.tx) == (card, 9 if guarded else 8), card


def test_no_tile_left_waits_and_never_hands_over():
    """Both legal Skeletons tiles are strays: the guard holds the play, a wait (no_tile) in
    live as in legacy, never the next slot (the Musketeer, legal at (9, 24)); stray counts the
    legal tiles turned down, an illegal one is not counted."""
    pol = FakePolicy([-5, 3, 2, 1, 0], {0: ranked(T(8, 17), T(7, 18)), 3: ranked(T(9, 24))})
    st = state(hand=(SKEL, CANNON, HOG, MUSK), elixir=9000, entities=episode9())
    legal = only(1, (8, 17), (7, 18), (9, 24))
    for cfg in (LIVE_STRAY, LEGACY_STRAY):
        w = RD.decide(pol, ROW, st, 1, legal, cfg, rng=StubRng(0.99))
        assert (w.kind, w.reason, w.stray, w.tried, w.command) == \
            (RD.WAIT, RD.R_NO_TILE, 2, (0,), None), cfg.name
    w = RD.decide(pol, ROW, st, 1, only(1, (7, 18), (9, 24)), LIVE_STRAY, rng=StubRng(0.99))
    assert (w.kind, w.stray) == (RD.WAIT, 1)
    # without the guard the live fallback is untouched: no legal Skeletons tile at all (stray 0)
    # hands over to the Musketeer as before
    d = RD.decide(pol, ROW, st, 1, only(1, (9, 24)), LIVE_STRAY, rng=StubRng(0.99))
    assert (d.kind, d.card, d.fallback, d.stray) == (RD.PLAY, MUSK, True, 0)
    assert d == RD.decide(pol, ROW, st, 1, only(1, (9, 24)), LIVE, rng=StubRng(0.99))
    # a card before the Skeletons with no legal tile still hands over to them, and the guard
    # then moves them
    hog_first = FakePolicy([-5, 1, 2, 3, 0], {0: ranked(T(8, 17), T(9, 17)),
                                              2: ranked(T(3, 10))})
    st2 = state(hand=(SKEL, CANNON, HOG, MUSK), elixir=9000, entities=episode9())
    d = RD.decide(hog_first, ROW, st2, 1, only(1, (8, 17), (9, 17)), LIVE_STRAY,
                  rng=StubRng(0.99))
    assert (d.card, d.tx, d.ty, d.tried, d.stray) == (SKEL, 9, 17, (2, 1, 0), 1)


def test_a_fully_rejected_cheap_troop_never_becomes_a_hog_rider():
    """Review finding 2: every one of the model's first 60 Skeletons tiles is a stray (all
    legal); without the waiting rule the live fallback played a 4-elixir Hog Rider at the bridge.
    Now: a wait with stray 60; without the guard, the Skeletons at the model's tile."""
    st = state(hand=(SKEL, HOG, CANNON, MUSK), elixir=6000, entities=episode9())
    g = RD.stray_guard(st, 1, SUB)
    rejected = [t for t in range(RD.TILES) if t // 18 >= 17 and not g.keeps(t % 18, t // 18)]
    assert len(rejected) >= 60
    pol = FakePolicy([-5, 3, 0.5, -1, -2], {0: ranked(*rejected[:60]), 1: ranked(T(14, 17))})
    d = RD.decide(pol, ROW, st, 1, always, LIVE_STRAY, rng=StubRng(0.99))
    assert (d.kind, d.reason, d.stray, d.tried) == (RD.WAIT, RD.R_NO_TILE, 60, (0,))
    plain = RD.decide(pol, ROW, st, 1, always, LIVE, rng=StubRng(0.99))
    assert (plain.card, plain.tile, plain.cost) == (SKEL, rejected[0], 1)


def test_legitimate_plays_are_kept():
    pol_of = (lambda *tiles: FakePolicy([-5, 3, 2, 1, 0], {0: ranked(*tiles)}))

    def tile(entities, *tiles, team=1, cfg=LIVE_STRAY, towers=None):
        st = state(team=team, entities=entities, my_towers=towers)
        d = RD.decide(pol_of(*tiles), ROW, st, team, always, cfg, rng=StubRng(0.99))
        return (d.tx, d.ty, d.stray)

    mine = [unit(1, 9_400, 24_800, uid=51, hp=700, card=MUSK)]
    # nothing threatens: no guard at all; centre river and the back of the king both play
    assert RD.stray_guard(state(entities=mine), 1, SUB) is None
    assert tile(mine, T(8, 17)) == (8, 17, 0) and tile(mine, T(8, 31)) == (8, 31, 0)
    # an enemy still deep on its side (own y < 13): no guard either
    deep = [unit(0, 15_300, 12_900), *mine]
    assert RD.stray_guard(state(entities=deep), 1, SUB) is None
    assert tile(deep, T(8, 17)) == (8, 17, 0)
    ep9 = episode9()
    # with the threat: the threat lane's bridge (near its path), next to our towers (the back
    # of the king, a princess tower's side), and the enemy half
    for kept in ((13, 17), (15, 18), (8, 31), (6, 31), (4, 24), (1, 27), (8, 10)):
        assert tile(ep9, T(*kept)) == (*kept, 0), kept
    for stray in ((5, 17), (3, 19), (2, 22), (8, 22), (0, 28)):
        assert tile(ep9, T(*stray), T(12, 22))[:2] == (12, 22), stray
    # the far lane's bridge only with a push of ours in that lane (review finding 1): the
    # model's (8, 17), (7, 17), (8, 18), (3, 17), (9, 17) plays (9, 17), not the left bridge
    far = (T(8, 17), T(7, 17), T(8, 18), T(3, 17), T(9, 17))
    assert tile(ep9, *far) == (9, 17, 4)
    for bridge in ((3, 17), (2, 18), (4, 17)):
        assert tile(ep9, T(*bridge), T(12, 22)) == (12, 22, 1), bridge
    hog_left = unit(1, 3_500, 14_000, uid=52, card=HOG)          # our Hog across the river
    at_bridge = unit(1, 4_000, 18_500, uid=52, card=HOG)         # ... or on our bridge
    behind = unit(1, 3_500, 20_000, uid=52, card=HOG)            # 1 tile behind: no push yet
    cannon_left = unit(1, 3_500, 18_000, kind=1, uid=52, card=CANNON)   # a building: no push
    for push in (hog_left, at_bridge):
        assert RD.stray_guard(state(entities=[*ep9, push]), 1, SUB).push_lanes == {0}
        for bridge in ((3, 17), (2, 18), (4, 17)):
            assert tile([*ep9, push], T(*bridge), T(12, 22)) == (*bridge, 0), bridge
        assert tile([*ep9, push], T(5, 17), T(12, 22))[:2] == (12, 22)   # not a bridge tile
    assert tile([*ep9, behind], T(3, 17), T(12, 22)) == (12, 22, 1)
    # a building of ours is no push (it keeps (3, 17) only as the end of the path it pulls)
    g = RD.stray_guard(state(entities=[*ep9, cannon_left]), 1, SUB)
    assert g.push_lanes == frozenset() and g.reach(3, 17) < 1_000
    # the push is per team: team 0 with its Hog in its own left lane
    ep9_0 = episode9(0)
    hog_0 = unit(0, 3_500, 14_000, me=0, uid=52, card=HOG)
    assert tile([*ep9_0, hog_0], T(3, 17), T(12, 22), team=0) == (3, 17, 0)
    assert tile(ep9_0, T(3, 17), T(12, 22), team=0) == (12, 22, 1)
    # a fallen princess: its side is no longer "next to a tower" (for the live seat, team 1,
    # the own-frame left princess is the world-right one, slot 2)
    assert tile(ep9, T(4, 24), T(12, 22), towers=(KING_HP, PRINCESS_HP, 0)) == (12, 22, 1)
    assert tile(ep9, T(4, 24), T(12, 22), towers=(KING_HP, 0, PRINCESS_HP)) == (4, 24, 0)
    g = RD.stray_guard(state(entities=ep9, my_towers=(KING_HP, 0, 0)), 1, SUB)
    assert len(g.towers) == 1 and not g.keeps(4, 24) and g.keeps(8, 31)


def test_paths_follow_buildings_fallen_towers_and_flyers():
    # Our Cannon in the centre pulls a ground troop: Skeletons next to it are kept.
    hog = unit(0, 14_500, 16_000, card=HOG)
    cannon = unit(1, 9_500, 21_500, kind=1, uid=60, hp=700, card=CANNON)
    g = RD.stray_guard(state(entities=[hog]), 1, SUB)
    assert not g.keeps(8, 22)
    g = RD.stray_guard(state(entities=[hog, cannon]), 1, SUB)
    assert ((14_500, 17_000), (9_500.0, 21_500.0)) in g.paths and g.keeps(8, 22)
    # our princess tower in the threat's lane is down: the path goes on to the king
    for team in (1, 0):
        lane_slot = next(s for s in (1, 2) if RD.to_world(RD.TOWER_CENTRE[(team, s)][0] * 1000,
                                                          0, team)[0] > 9_000)
        towers = [KING_HP, PRINCESS_HP, PRINCESS_HP]
        towers[lane_slot] = 0
        st = state(team=team, entities=[unit(1 - team, 14_500, 16_000, me=team)],
                   my_towers=towers)
        g = RD.stray_guard(st, team, SUB)
        assert g.paths[0][-1] == (9_000.0, 29_000.0), team
    # a flyer goes straight at the tower of its lane, no bridge
    bat = unit(0, 15_000, 14_000, flying=True)
    g = RD.stray_guard(state(entities=[bat]), 1, SUB)
    assert g.paths == (((15_000.0, 14_000.0), (14_500.0, 25_500.0)),)
    # an enemy building walks nowhere: alone it switches nothing on (review finding 3: an enemy
    # building at their river edge turned 160 of our tiles down), and next to a troop it adds
    # no path; our own units are no threat
    xbow = unit(0, 9_000, 13_500, kind=1, uid=61)
    assert RD.stray_guard(state(entities=[xbow, unit(1, 3_000, 20_000, uid=62)]), 1, SUB) is None
    pol = FakePolicy([-5, 3, 2, 1, 0], {0: ranked(T(1, 20))})
    d = RD.decide(pol, ROW, state(entities=[xbow]), 1, always, LIVE_STRAY, rng=StubRng(0.99))
    assert (d.tx, d.ty, d.stray) == (1, 20, 0)
    g = RD.stray_guard(state(entities=[xbow, bat]), 1, SUB)
    assert g.paths == (((15_000.0, 14_000.0), (14_500.0, 25_500.0)),)
    dead = unit(0, 15_300, 13_500, hp=0)
    assert RD.stray_guard(state(entities=[dead]), 1, SUB) is None


def test_stray_guard_off_by_default_and_on_other_cards():
    st = state(entities=episode9())
    pol = FakePolicy([-5, 3, 2, 1, 0], {0: ranked(T(8, 17))})
    for cfg in (RD.DecideConfig(), LIVE, RD.DecideConfig.live(0.0, guard_cards=[CANNON])):
        d = RD.decide(pol, ROW, st, 1, always, cfg, rng=StubRng(0.99))
        assert (d.tx, d.ty, d.stray) == (8, 17, 0)
    # a guard on another card list leaves the Skeletons alone
    d = RD.decide(pol, ROW, st, 1, always, LIVE.replace(stray_cards=[MUSK]), rng=StubRng(0.99))
    assert (d.tx, d.stray) == (8, 0)


# ---------------------------------------------------------------------------
# the config, and the defaults draw for draw
# ---------------------------------------------------------------------------

def test_config_switches_and_helpers():
    for cfg in (RD.DecideConfig(), RD.DecideConfig.legacy(), RD.DecideConfig.live(),
                RD.DecideConfig.sampling(), RD.DecideConfig.for_mode(RD.MODE_LIVE, 1.0)):
        assert cfg.wait_floor is None and cfg.stray_cards == frozenset() and not cfg.has_rules
    on = RD.apply_rules(LIVE, CAT, wait_floor=0.95, stray_guard=True)
    assert (on.wait_floor, on.stray_cards, on.has_rules) == (0.95, STRAY, True)
    assert STRAY == {SKEL, SPIRITS}
    assert RD.apply_rules(LIVE, CAT) is LIVE and RD.apply_rules(LIVE, (), wait_floor=1) \
        == LIVE.replace(wait_floor=1.0)
    assert LIVE.replace(stray_cards=[SKEL]).stray_cards == frozenset({SKEL})
    with pytest.raises(ValueError, match="none of"):
        RD.apply_rules(LIVE, [["Knight", 3, 0, 0, 0, False, 0, 0, "TROOP"]], stray_guard=True)
    for bad in (0, 0.0, -0.1, 1.01, float("nan"), "x", None, True):
        with pytest.raises(ValueError):
            RD.check_wait_floor(bad)
    with pytest.raises(ValueError):
        LIVE.replace(wait_floor=1.5)
    for changes in ({"wait_floor": 0.95}, {"stray_cards": [SKEL]}):
        with pytest.raises(ValueError, match="sampling preset"):
            RD.DecideConfig.sampling(0.0, **changes)
        with pytest.raises(ValueError, match="sampling preset"):
            RD.DecideConfig.sampling(0.0).replace(**changes)
    assert RD.rules_record() == {} and RD.rules_label() == ""
    assert RD.rules_record(0.95, True) == {"wait_floor": 0.95, "stray_guard": True}
    assert RD.rules_label(0.95, True) == " wait-floor 0.95 stray-guard"
    assert RD.rules_label(stray_guard=True) == " stray-guard"


def _random_case(r, k, mod):
    """One random decide() call for module `mod` (rs_decide now, or as it was): its policy,
    row, state, team, legality, config, rng and keywords. The same r-stream for both."""
    team = int(r.integers(0, 2))
    n_buttons = int(r.choice([0, 0, 2]))
    gate = r.normal(0, 2.5, size=5 + n_buttons).astype(np.float32)
    tiles = {s: r.normal(size=RD.TILES).astype(np.float32) for s in range(4)}
    hand = [int(c) for c in r.choice(16, size=4, replace=False)]
    if r.random() < 0.15:
        hand[int(r.integers(0, 4))] = -1
    costs = [CAT[c][1] if c >= 0 else 99 for c in hand]
    if r.random() < 0.15:
        costs[int(r.integers(0, 4))] = -1
    abilities = [[int(r.integers(0, 2)), int(r.integers(0, 2)), int(r.integers(0, 4)), 2, 0]
                 for _ in range(n_buttons)]
    me = player_dict(team, hand=hand, hand_costs=costs, elixir_milli=int(r.integers(0, 10_001)),
                     pending_cost=int(r.integers(0, 4)) if r.random() < 0.3 else 0,
                     abilities=abilities,
                     tower_hp=[KING_HP, int(r.choice([0, PRINCESS_HP])),
                               int(r.choice([0, PRINCESS_HP]))])
    ents = [entity_row(uid=100 + i, team=int(r.integers(0, 2)), kind=int(r.integers(0, 2)),
                       card_id=int(r.integers(0, 16)), x=int(r.integers(0, 18_000)) * SUB,
                       y=int(r.integers(0, 32_000)) * SUB, hp=int(r.integers(0, 3000)),
                       flying=bool(r.random() < 0.3))
            for i in range(int(r.integers(0, 8)))]
    st = state_dict(90 + 10 * k, players=[me, player_dict(1 - team)] if team == 0
                    else [player_dict(1 - team), me], entities=ents)
    preset = str(r.choice(["legacy", "live", "sample"]))
    presses = bool(r.random() < 0.3)
    bias = float(r.normal(1.0, 0.5))
    dt = int(r.integers(1, 21))
    if preset == "legacy":
        cfg = mod.DecideConfig.legacy(bias, dt=dt, presses=presses)
    elif preset == "live":
        cfg = mod.DecideConfig.live(bias, guard_cards=[CANNON], dt=dt, presses=presses)
    else:
        cfg = mod.DecideConfig.sampling(bias, dt=dt, presses=presses)
    pending = None
    if r.random() < 0.4:
        pending = [mod.Pending(int(hand[int(r.integers(0, 4))]), int(r.integers(0, 5)),
                               int(r.integers(-1, 4)), 90 + 10 * k - int(r.integers(0, 40)))]
    kw = {"pending": pending, "tick": 90 + 10 * k, "sub": SUB}
    if r.random() < 0.3:
        kw["elixir_cap"] = int(r.integers(0, 10_001))
    if r.random() < 0.2:
        kw["blocked_tiles"] = r.random(RD.TILES) < 0.2
    if r.random() < 0.3:
        kw["dt"] = int(r.integers(1, 21))
    salt = int(r.integers(0, 1 << 30))

    def legal(slot, card, x, y):
        return zlib.crc32(f"{salt},{slot},{card},{x},{y}".encode()) % 50 == 0
    return FakePolicy(gate, tiles), {"tick": 90 + 10 * k}, st, team, legal, cfg, kw


def _base_module(monkeypatch):
    """tools/rs_decide.py as it was at BASE_COMMIT, imported under another name (in
    sys.modules while the test runs: its dataclasses look their module up); skip without git or
    that commit (a shallow or exported checkout)."""
    try:
        src = subprocess.run(["git", "show", f"{BASE_COMMIT}:tools/rs_decide.py"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8", timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git: {exc}")
    if src.returncode != 0:
        pytest.skip(f"git show {BASE_COMMIT}: {src.stderr.strip()[:200]}")
    spec = importlib.util.spec_from_loader(f"rs_decide_{BASE_COMMIT}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(TOOLS / "rs_decide.py")
    monkeypatch.setitem(sys.modules, spec.name, mod)
    exec(compile(src.stdout, f"rs_decide@{BASE_COMMIT}", "exec"), mod.__dict__)
    return mod


def test_defaults_decide_draw_for_draw_as_before_the_rules(monkeypatch):
    """decide() with every preset's defaults against the decide of BASE_COMMIT on 600 random
    cases (hands with gaps, odd costs, buttons, plays in flight, caps, blocked tiles, units of
    both teams, fallen towers, a 1-in-50 legality: some decisions find no tile): the same decision record (the new `stray`
    field 0) and the same random stream."""
    old = _base_module(monkeypatch)
    seen = Counter()
    r_new, r_old = np.random.default_rng(2026), np.random.default_rng(2026)
    for k in range(600):
        pol, row, st, team, legal, cfg, kw = _random_case(r_new, k, RD)
        pol_o, row_o, st_o, team_o, legal_o, cfg_o, kw_o = _random_case(r_old, k, old)
        rng_new, rng_old = np.random.default_rng(k), np.random.default_rng(k)
        d = RD.decide(pol, row, st, team, legal, cfg, rng=rng_new, **kw)
        e = old.decide(pol_o, row_o, st_o, team_o, legal_o, cfg_o, rng=rng_old, **kw_o)
        got, want = d.to_record(), e.to_record()
        assert got.pop("stray") == 0
        assert got == want, k
        assert rng_new.bit_generator.state == rng_old.bit_generator.state, k
        seen[(cfg.name, d.kind, d.reason)] += 1
    kinds = Counter(k for (_, k, _) in seen.elements())
    assert kinds[RD.PLAY] >= 50 and kinds[RD.WAIT] >= 50, seen
    assert {name for (name, _, _) in seen} == set(RD.MODES)
    assert any(reason == RD.R_NO_TILE for (_, _, reason) in seen), seen


def test_rules_that_cannot_fire_change_nothing():
    """The stray guard with no stray card in hand, and a wait floor of 1.0 below p_wait 1:
    decision for decision and draw for draw the decide without them."""
    r = np.random.default_rng(9)
    compared = 0
    for k in range(400):
        pol, row, st, team, legal, cfg, kw = _random_case(r, k, RD)
        if cfg.sample:
            continue
        hand = st["players"][team]["hand"]
        hand[:] = [c if c not in STRAY else -1 for c in hand]
        a, b = np.random.default_rng(k), np.random.default_rng(k)
        d = RD.decide(pol, row, st, team, legal, cfg, rng=a, **kw)
        if d.p_wait >= 1.0:                    # (float32 rounding) the floor would hold here
            continue
        e = RD.decide(pol, row, st, team, legal,
                      cfg.replace(stray_cards=STRAY, wait_floor=1.0), rng=b, **kw)
        assert d == e, k
        assert a.bit_generator.state == b.bit_generator.state
        compared += 1
    assert compared >= 100


# ---------------------------------------------------------------------------
# rs_eval: the agent seat's switches
# ---------------------------------------------------------------------------

@pytest.fixture
def ev(monkeypatch):
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    mod = importlib.import_module("rs_eval")
    monkeypatch.setattr(mod.RR, "F", dict(FIELD))
    monkeypatch.setattr(mod.RSAgent, "_CAT", [list(r) for r in DEFAULT_CATALOGUE])
    return mod


def _agent(ev, cls=None):
    a = object.__new__(cls or ev.RSAgent)
    a.wait_bias = math.log(4)
    a.meta = {}
    return a


def test_rs_eval_decide_config_carries_the_rules(ev):
    a = _agent(ev)
    plain = {m: (a.__setattr__("decide_mode", m), a.decide_config())[1]
             for m in (RD.MODE_LEGACY, RD.MODE_LIVE)}
    assert plain[RD.MODE_LEGACY] == RD.DecideConfig.legacy(math.log(4), dt=ev.DECIDE)
    assert plain[RD.MODE_LIVE] == RD.DecideConfig.live(math.log(4), guard_cards=[CANNON],
                                                       dt=ev.DECIDE)
    a.wait_floor, a.stray_guard = 0.95, True
    for m in (RD.MODE_LEGACY, RD.MODE_LIVE):
        a.decide_mode = m
        assert a.decide_config() == plain[m].replace(wait_floor=0.95, stray_cards=STRAY)
    a.stray_guard = False
    assert a.decide_config() == plain[RD.MODE_LIVE].replace(wait_floor=0.95)


def test_rs_eval_counts_what_the_rules_did(ev):
    a = _agent(ev)
    a.note_rules(RD.Decision(team=1, tick=0, kind=RD.WAIT, reason=RD.R_WAIT))
    a.note_rules(RD.Decision(team=1, tick=0, kind=RD.PLAY))
    assert a.rule_counts is None                          # nothing fired: nothing kept
    a.note_rules(RD.Decision(team=1, tick=0, kind=RD.WAIT, reason=RD.R_FLOOR))
    a.note_rules(RD.Decision(team=1, tick=0, kind=RD.PLAY, stray=2))
    a.note_rules(RD.Decision(team=1, tick=0, kind=RD.WAIT, reason=RD.R_NO_TILE, stray=1))
    assert a.rule_counts == {"floor_waits": 1, "stray_moved": 1, "stray_waits": 1}
    assert ev.RSAgent.rule_counts is None


def test_rs_eval_act_feeds_the_counts(ev):
    """RSAgent.act (the rs_eval and league seat) decides with the rules and counts them: a wait
    the floor made, and the episode-9 Skeletons moved to (9, 17)."""
    from tests.pipeline.conftest import FakeBattle
    from tests.pipeline.test_rs_decide import ScriptedModel, make_agent
    mem = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}
    st = state(entities=episode9(), elixir=9000)
    floor = make_agent(ev, seed=1, model=ScriptedModel([5, 0, 0, 0, 0], {}))
    floor.rng = StubRng(0.999)
    floor.decide_mode, floor.wait_floor = RD.MODE_LIVE, 0.95
    assert floor.act(FakeBattle(), st, 1, mem, 1587, SUB, None) is None
    assert floor.last_decision.reason == RD.R_FLOOR and floor.rule_counts == {"floor_waits": 1}
    stray = make_agent(ev, seed=1, model=ScriptedModel([-50, 3, 2, 1, 0],
                                                       {SKEL: ranked(T(8, 17), T(9, 17))}))
    stray.rng = StubRng(0.999)
    stray.decide_mode, stray.stray_guard = RD.MODE_LIVE, True
    cmd = stray.act(FakeBattle(), st, 1, mem, 1587, SUB, None)
    wx, wy = RD.tile_centre_world(T(9, 17), 1)
    assert cmd == (1, 0, wx * SUB, wy * SUB) and stray.rule_counts == {"stray_moved": 1}
    plain = make_agent(ev, seed=1, model=ScriptedModel([-50, 3, 2, 1, 0],
                                                       {SKEL: ranked(T(8, 17), T(9, 17))}))
    plain.rng, plain.decide_mode = StubRng(0.999), RD.MODE_LIVE
    wx, wy = RD.tile_centre_world(T(8, 17), 1)
    assert plain.act(FakeBattle(), st, 1, mem, 1587, SUB, None) == (1, 0, wx * SUB, wy * SUB)
    assert plain.rule_counts is None


def test_rs_eval_rules_need_an_agent_seat(ev):
    ev.check_rule_seats(None, False, "pro")
    ev.check_rule_seats(0.95, True, "agent")
    for mode_, match in (("pro", "no agent seat"), ("search", "act"), ("search_rebuild", "act")):
        with pytest.raises(ValueError, match=match):
            ev.check_rule_seats(0.95, False, mode_)
        with pytest.raises(ValueError, match=match):
            ev.check_rule_seats(None, True, mode_)
    with pytest.raises(ValueError, match="wait floor"):
        ev.check_rule_seats(2.0, False, "agent")
    # play() refuses them before it reads the payload
    args = ("{}", "", "pro", 0, "", 0, 0, False, False, False, None, None, None, None, None,
            None, 0, 0.95, False)
    with pytest.raises(ValueError, match="wait-floor"):
        ev.play(args)


def test_rs_eval_flags_names_and_records():
    tree = ast.parse((TOOLS / "rs_eval.py").read_text(encoding="utf-8"))
    flags = {a.value for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "add_argument"
             for a in n.args if isinstance(a, ast.Constant)}
    assert {"--wait-floor", "--stray-guard"} <= flags
    text = (TOOLS / "rs_eval.py").read_text(encoding="utf-8")
    for needle in ('record["rules"] = rules', 'record["rule_counts"]',
                   "rs_decide.rules_label(args.wait_floor, args.stray_guard)",
                   "extra += (args.wait_floor, bool(args.stray_guard))",
                   'out["rules"] = dict(sorted('):
        assert needle in text, needle


# ---------------------------------------------------------------------------
# rs_match and the league
# ---------------------------------------------------------------------------

@pytest.fixture
def tools(monkeypatch):
    """rs_rows, rs_match and rs_league_eval with module settings at their defaults (as
    tests/pipeline/test_league_eval.py), RSAgent's catalogue the conftest one."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    rm = importlib.import_module("rs_match")
    le = importlib.import_module("rs_league_eval")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    monkeypatch.setattr(rr, "F", dict(FIELD))
    monkeypatch.setattr(rm.rs_eval.RSAgent, "_CAT", [list(r) for r in DEFAULT_CATALOGUE])
    return types.SimpleNamespace(rows=rr, match=rm, league=le, engine=eng)


#: AgentSpec.to_dict's keys before the rules (and without presses, search, latency_ticks).
SPEC_KEYS = {"ckpt", "name", "schema", "lead", "noise", "noise_seed", "decide", "visibility",
             "wait_keep", "spell_aim", "hold_pending", "seed_offset", "allow_drift"}


def test_agent_spec_switches(tools):
    M = tools.match
    plain = M.AgentSpec("runs/tokens/v2_hog/best.pt", name="v2_hog")
    assert set(plain.to_dict()) == SPEC_KEYS and plain.rules == {}
    ruled = M.AgentSpec("runs/tokens/v2_hog/best.pt", name="v2_hog_rules", wait_floor=0.95,
                        stray_guard=True)
    d = ruled.to_dict()
    assert set(d) == SPEC_KEYS | {"wait_floor", "stray_guard"}
    assert (d["wait_floor"], d["stray_guard"]) == (0.95, True)
    assert M.AgentSpec.from_dict(d) == ruled and ruled.rules == {"wait_floor": 0.95,
                                                                 "stray_guard": True}
    assert set(M.AgentSpec("a.pt", wait_floor=1).to_dict()) == SPEC_KEYS | {"wait_floor"}
    assert M.AgentSpec("a.pt", stray_guard=np.bool_(True)).stray_guard is True
    for bad, match in (({"wait_floor": 0}, "wait floor"), ({"wait_floor": "high"}, "wait floor"),
                       ({"stray_guard": "yes"}, "stray_guard"),
                       ({"ckpt": "", "wait_floor": 0.9}, "idle"),
                       ({"ckpt": "idle", "stray_guard": True}, "idle"),
                       ({"decide": "sample", "wait_floor": 0.9}, "sample"),
                       ({"search": "s_root", "wait_floor": 0.9}, "search seat"),
                       ({"search": "s_root", "stray_guard": True}, "search seat")):
        kw = {"ckpt": "a.pt", **bad}
        with pytest.raises(ValueError, match=match):
            M.AgentSpec(**kw)


def _match_seat(M):
    seat = object.__new__(M.MatchAgent)
    seat.wait_bias, seat.meta, seat.cadence = math.log(4), {}, 10
    return seat


def test_configure_reaches_decide_config_and_seat_info(tools):
    M = tools.match
    cfg = tools.engine.EngineConfig.defaults()
    for decide in (RD.MODE_LIVE, RD.MODE_LEGACY):
        plain = M.configure(_match_seat(M), M.AgentSpec("a.pt", decide=decide), cfg)
        ruled = M.configure(_match_seat(M), M.AgentSpec("a.pt", decide=decide, wait_floor=0.95,
                                                        stray_guard=True), cfg)
        assert "wait_floor" not in vars(plain) and "stray_guard" not in vars(plain)
        assert not plain.decide_config().has_rules
        assert ruled.decide_config() == plain.decide_config().replace(wait_floor=0.95,
                                                                      stray_cards=STRAY)
    sample = M.configure(_match_seat(M), M.AgentSpec("a.pt", decide="sample"), cfg)
    assert sample.decide_config() == RD.DecideConfig.sampling(math.log(4), dt=10)
    # seat_info: "rules" only with a rule on, with what the rules did
    spec = M.AgentSpec("a.pt", name="r", wait_floor=0.95)
    seat = M.configure(_match_seat(M), spec, cfg)
    assert "rules" not in M.seat_info(plain, M.AgentSpec("a.pt"))
    assert M.seat_info(seat, spec)["rules"] == {"wait_floor": 0.95}
    seat.note_rules(RD.Decision(team=0, tick=0, kind=RD.WAIT, reason=RD.R_FLOOR))
    assert M.seat_info(seat, spec)["rules"] == {"wait_floor": 0.95, "floor_waits": 1}


def _result(M, agents, info, winner=0):
    return M.MatchResult(id="x", tag="t", pairing="p", seat=0, seed=0, agents=agents,
                         winner=winner, crowns=(1, 0), game_over=True, end_tick=3600,
                         plays=(30, 31), refused=(0, 0), decisions=(300, 300),
                         full_ticks=(0, 0), leak=(0.0, 0.0), towers=((0,) * 3, (0,) * 3),
                         delay=(24, 24), info=info)


def test_rules_summary_and_line(tools):
    M = tools.match
    r1 = _result(M, ("r", "h"), ({"rules": {"wait_floor": 0.95, "stray_guard": True,
                                            "floor_waits": 2, "stray_moved": 1}}, {}))
    r2 = _result(M, ("h", "r"), ({}, {"rules": {"wait_floor": 0.95, "stray_guard": True}}))
    s = M.rules_summary([r1, r2], "r")
    assert s == {"wait_floor": 0.95, "stray_guard": True, "games": 2,
                 "counts": {"floor_waits": {"total": 2, "per_game": 1.0},
                            "stray_moved": {"total": 1, "per_game": 0.5}}}
    assert M.rules_summary([r1, r2], "h") is None
    assert M.rules_line("r", s) == ("r rules (wait-floor 0.95 stray-guard) over 2 games: floor "
                                    "waits 1.00 a game, stray moved 0.50 a game")


def test_rs_match_command_line(tools):
    M = tools.match
    args = M.build_parser().parse_args(["--a", "a.pt", "--b", "a.pt", "--name-a", "rules",
                                        "--name-b", "plain", "--wait-floor-a", "0.95",
                                        "--stray-guard", "a"])
    a, b = M.agents_from_args(args)
    assert (a.wait_floor, a.stray_guard, b.wait_floor, b.stray_guard) == (0.95, True, None, False)
    args = M.build_parser().parse_args(["--a", "a.pt", "--b", "b.pt"])
    a, b = M.agents_from_args(args)
    assert not a.rules and not b.rules and set(a.to_dict()) == SPEC_KEYS


def test_a_league_player_carries_the_rules_to_its_seat(tools, tmp_path):
    """The same checkpoint under two names: "R" with both rules, "H" without. Every planned
    game gives R's seat the switches and H's none, the run header records them, and a seat
    built from a game's spec decides with them."""
    from tests.pipeline.test_league_eval import NAMES, synthetic_rows
    M, LE = tools.match, tools.league
    R = LE.parse_player("R=same.pt,wait_floor=0.95,stray_guard=on")
    assert (R.ckpt, R.wait_floor, R.stray_guard) == ("same.pt", 0.95, True)
    assert LE.parse_player("H=same.pt").rules == {}
    with pytest.raises(ValueError, match="true or false"):
        LE.parse_player("R=same.pt,stray_guard=maybe")
    with pytest.raises(ValueError, match="a number"):
        LE.parse_player("R=same.pt,wait_floor=high")
    with pytest.raises(ValueError, match="wait floor"):
        LE.parse_player("R=same.pt,wait_floor=1.5")
    pairs = M.read_pairs(synthetic_rows(), NAMES, "dev", require=M.HOG26)
    calls = []

    def play(spec):
        calls.append(spec)
        names = tuple(a.label for a in spec.agents)
        info = tuple(M.seat_info(None, a) for a in spec.agents)
        return M.MatchResult(id=spec.id, tag=spec.tag, pairing=spec.pairing, seat=spec.seat,
                             seed=spec.seed, agents=names, winner=names.index("R"),
                             crowns=(1, 0), game_over=True, end_tick=3600, plays=(30, 30),
                             refused=(0, 0), decisions=(300, 300), full_ticks=(0, 0),
                             leak=(0.0, 0.0), towers=((0,) * 3, (0,) * 3),
                             delay=(spec.delay, spec.delay), info=info)

    log = tmp_path / "decide_ab.jsonl"
    rep = LE.evaluate(pairs, LE.Protocol(noise=None, engine=tools.engine.EngineConfig.defaults()),
                      candidates=["R=same.pt,wait_floor=0.95,stray_guard=on"],
                      champion="H=same.pt", config=LE.GateConfig(champion_games=8, n_boot=50),
                      log_path=log, play=play)
    assert len(calls) == 8 and rep["matchups"][0]["score"] == 1.0
    for spec in calls:
        by = {a.label: a for a in spec.agents}
        assert by["R"].rules == {"wait_floor": 0.95, "stray_guard": True}
        assert by["H"].rules == {} and by["R"].ckpt == by["H"].ckpt == "same.pt"
        assert set(by["H"].to_dict()) == SPEC_KEYS
    recs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    head = recs[0]
    assert head["players"]["R"]["wait_floor"] == 0.95 and head["players"]["R"]["stray_guard"]
    assert "wait_floor" not in head["players"]["H"] and "stray_guard" not in head["players"]["H"]
    assert head["players"]["R"]["id"] != head["players"]["H"]["id"]
    game = recs[1]
    team = game["teams"].index("R")
    assert game["result"]["info"][team]["rules"] == {"wait_floor": 0.95, "stray_guard": True}
    assert "rules" not in game["result"]["info"][1 - team]
    # the seat a game's spec builds decides with the rules (the factory's configure step)
    seat = M.configure(_match_seat(M), by["R"], tools.engine.EngineConfig.defaults())
    assert seat.decide_config().wait_floor == 0.95 and seat.decide_config().stray_cards == STRAY
    # the same name with other rules in this log is another player: refused before playing
    with pytest.raises(ValueError, match="another player"):
        LE.evaluate(pairs, LE.Protocol(noise=None, engine=tools.engine.EngineConfig.defaults()),
                    candidates=["R=same.pt,wait_floor=0.9"], champion="H=same.pt",
                    config=LE.GateConfig(champion_games=8, n_boot=50), log_path=log, play=play)
    assert len(calls) == 8
    # without its rules R is H but for the name: the switches are the whole difference
    bare = dataclasses.replace(by["R"], wait_floor=None, stray_guard=False)
    assert {**bare.to_dict(), "name": "H"} == by["H"].to_dict()


def test_the_live_runner_takes_the_rule_switches_off_by_default():
    """scripts/vision_play.py: --wait-floor P and --stray-guard switch the two rules on for live
    play (off by default); a bad floor refuses to start before anything else runs."""
    VP = pytest.importorskip("scripts.vision_play")
    parse = VP.build_parser().parse_args
    off = parse([])
    assert off.wait_floor is None and off.stray_guard is False
    on = parse(["--wait-floor", "0.95", "--stray-guard"])
    assert on.wait_floor == 0.95 and on.stray_guard is True
    assert RD.rules_record(on.wait_floor, on.stray_guard) == {"wait_floor": 0.95, "stray_guard": True}
    assert VP.main(["--wait-floor", "1.5"]) == 2
