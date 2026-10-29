"""E13: the opponent elixir counter (tools/rs_elixir.py).

Analytic values worked by hand in the engine's fine units (one elixir = 28000, a tick
adds 500 at 1x, 1000 at 2x, 1505 at 3x; elixir_milli = fine // 28): regeneration and
the cap, a play inside an interval splitting the regeneration, the double and triple
elixir boundaries, Mirror and press pricing, payouts held at the cap, late reports,
check / resync, and observe() on state_json-shaped dicts. The engine-marked tests drive
scripted RoyaleSim battles and compare the count with players[t].elixir_milli every 10
ticks, exactly, through 1x, the cap, 2x, overtime and 3x, with and without a command delay.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import rs_elixir as E  # noqa: E402

COSTS = {"Knight": 3, "Giant": 5, "Mirror": 1, "Zap": 2, "Hog": 4}
#: conftest DEFAULT_CATALOGUE ids of team 1's cards (Knight, Archer, Giant, Arrows, Valkyrie,
#: MiniPekka) and their elixir.
COST_IDS = {8: 3, 9: 3, 10: 5, 11: 3, 12: 4, 13: 4}


@pytest.fixture
def law():
    return E.ElixirLaw.constants()


def counter(law, **kwargs):
    return E.ElixirCounter(COSTS, law=law, **kwargs)


# ---------------------------------------------------------------------------
# the law
# ---------------------------------------------------------------------------

def test_constants_are_the_engine_units(law):
    assert (law.scale, law.gain_1x, law.gain_2x, law.gain_3x) == (28000, 500, 1000, 1505)
    assert (law.cap_fine, law.start_fine, law.tick_ms) == (280000, 168000, 50)
    assert (law.regular_ticks, law.speedup_ticks, law.triple_ticks) == (3600, 1200, 1200)
    assert (law.double_from, law.triple_from) == (2400, 4800)
    assert law.source == "constants"
    assert law.to_milli(law.start_fine) == 6000 and law.seed_fine(6000) == 168000
    assert [law.rate_at(t) for t in (0, 2399, 2400, 3599, 3600, 4799, 4800, 6000)] == \
        [1, 1, 2, 2, 2, 2, 3, 3]
    assert law.regen(0, 56) == 28000                       # one elixir in 2.8 s
    assert law.regen(2300, 2500) == 100 * 500 + 100 * 1000
    assert law.regen(4700, 4900) == 100 * 1000 + 100 * 1505
    assert law.regen(0, 5000) == 2400 * 500 + 2400 * 1000 + 200 * 1505
    assert law.regen(10, 10) == 0 and law.regen(10, 5) == 0


def test_law_reads_calibration_values_and_overrides():
    cal = {"time": {"TICK_MS": {"value": 50}},
           "match": {"MANA_REGEN_MS_1X": {"value": 30000, "status": "x"},
                     "MANA_REGEN_MS_2X": {"value": 15000}, "START_MANA": 5,
                     "MAX_MANA": {"value": 10}}}
    law = E.ElixirLaw.from_calibration(cal)
    assert (law.scale, law.gain_1x, law.gain_2x) == (30000, 500, 1000)
    assert law.gain_3x == 15_000_000 // 9300 == 1612     # a key it lacks: the constant 9300
    assert (law.cap_fine, law.start_fine, law.regular_ticks) == (300000, 150000, 3600)
    assert law.to_milli(law.start_fine) == 5000
    over = E.ElixirLaw.from_calibration(cal, overrides={
        "match.START_MANA": "7", "match.MANA_SPEED_UP_WHEN_REMAINING_SECONDS": 30})
    assert over.start_fine == 7 * 30000 and over.double_from == 3600 - 600
    assert E.ElixirLaw.from_calibration({}) == E.ElixirLaw.constants()


def test_load_reads_royalesim_data_dir(tmp_path, monkeypatch):
    cal = {"match": {"MANA_REGEN_MS_1X": {"value": 30000}, "MANA_REGEN_MS_2X": {"value": 15000}}}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(cal), encoding="utf-8")
    stub = types.ModuleType("royalesim")
    stub.data_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "royalesim", stub)
    assert E.calibration_path() == path
    law = E.ElixirLaw.load()
    assert law.scale == 30000 and law.source == str(path)
    assert E.ElixirLaw.load(path=path) == law


def test_load_falls_back_to_constants_without_royalesim(monkeypatch, fake_royalesim):
    # conftest's fake royalesim has no data_dir and no __spec__
    assert E.calibration_path() is None
    assert E.ElixirLaw.load().source == "constants"
    monkeypatch.setitem(sys.modules, "royalesim", None)        # not importable
    assert E.calibration_path() is None
    law = E.ElixirLaw.load()
    assert law == E.ElixirLaw.constants() and law.source == "constants"


def test_cli_prints_the_law(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "royalesim", None)
    assert E.main(["--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["scale"] == 28000 and info["double_from"] == 2400 and info["triple_from"] == 4800
    assert E.main([]) == 0
    assert "elixir law from constants" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the count
# ---------------------------------------------------------------------------

def test_regeneration_and_cap(law):
    c = counter(law)
    assert c.bar_milli(1000) == 10000                      # cap, queried first
    assert c.bar_milli(0) == 6000
    assert c.bar_milli(1) == 168500 * 1000 // 28000 == 6017
    assert c.bar_milli(56) == 7000
    assert c.bar_milli(223) == 279500 * 1000 // 28000 == 9982
    assert c.bar_milli(224) == 10000                       # 6 -> 10 in 224 ticks
    assert c.leaked_milli(1000) == (1000 - 224) * 500 * 1000 // 28000 == 13857
    assert c.exact and c.unaffordable == 0 and c.elixir_milli() == 6000


def test_play_mid_interval_splits_regeneration(law):
    c = counter(law)
    c.play(300, "Giant")                                   # at the cap since tick 224
    assert c.bar_milli(300) == 10000                       # state at 300: not paid yet
    assert c.elixir_milli(300) == 5000                     # spendable after the play
    assert c.bar_milli(301) == 140500 * 1000 // 28000 == 5017
    assert c.bar_milli(400) == 190000 * 1000 // 28000 == 6785
    early = counter(law)
    early.play(0, "Giant")                                 # the same play paid at 0 would read
    assert early.bar_milli(400) == 228000 * 1000 // 28000 == 8142
    # a late report before a cached query rewrites what came after it
    late = counter(law)
    assert late.bar_milli(250) == 10000
    late.play(300, "Giant")
    assert late.bar_milli(400) == 6785
    late.play(260, "Zap")
    assert late.bar_milli(400) == 154000 * 1000 // 28000 == 5500
    ordered = counter(law)
    ordered.play(260, "Zap")
    ordered.play(300, "Giant")
    assert ordered.bar_milli(400) == 5500


def test_double_and_triple_elixir_boundaries(law):
    c = counter(law, tick=2300, elixir_milli=2000)
    assert c.bar_milli(2400) == 106000 * 1000 // 28000 == 3785
    assert c.bar_milli(2401) == 107000 * 1000 // 28000 == 3821      # the first 2x tick
    assert c.bar_milli(2500) == 206000 * 1000 // 28000 == 7357
    c.play(2398, "Knight")                                 # straddles the switch
    assert c.bar_milli(2399) == 21500 * 1000 // 28000 == 767
    assert c.bar_milli(2400) == 22000 * 1000 // 28000 == 785
    assert c.bar_milli(2401) == 23000 * 1000 // 28000 == 821
    assert c.bar_milli(2500) == 122000 * 1000 // 28000 == 4357
    overtime = counter(law, tick=3500, elixir_milli=0)
    assert overtime.bar_milli(3700) == 200000 * 1000 // 28000 == 7142  # still 2x
    triple = counter(law, tick=4700, elixir_milli=0)
    assert triple.bar_milli(4800) == 100000 * 1000 // 28000 == 3571
    assert triple.bar_milli(4801) == 101505 * 1000 // 28000 == 3625    # the first 3x tick
    assert triple.bar_milli(4900) == 250500 * 1000 // 28000 == 8946


def test_mirror_costs_the_copied_card_plus_one(law):
    c = counter(law)
    c.play(300, "Knight")
    c.play(400, "Mirror")                                  # Knight 3 + Mirror 1
    c.play(500, "Mirror")                                  # copies the Knight again
    assert c.bar_milli(400) == 246000 * 1000 // 28000 == 8785
    assert c.elixir_milli(400) == 134000 * 1000 // 28000 == 4785
    assert c.bar_milli(600) == 122000 * 1000 // 28000 == 4357
    assert c.last_play == "Knight" and c.exact
    variant = counter(law)
    variant.play(300, "Empress", cost=3)                   # a form's cost, as played
    variant.play(400, "Mirror")
    assert variant.bar_milli(401) == 134500 * 1000 // 28000 == 4803
    assert variant.last_play == "Empress"
    blind = counter(law)
    blind.play(300, "Mirror")                              # nothing to copy: a play was missed
    assert blind.bar_milli(301) == 252500 * 1000 // 28000 == 9017
    assert not blind.exact


def test_press_is_paid_but_not_copied(law):
    c = counter(law)
    c.play(300, "Knight")
    c.press(350, 2)
    c.play(400, "Mirror")
    assert c.bar_milli(401) == 78500 * 1000 // 28000 == 2803
    assert c.last_play == "Knight" and c.exact


def test_unaffordable_play_floors_at_zero_and_is_not_exact(law):
    c = counter(law, elixir_milli=1000)
    c.play(0, "Giant")
    assert c.elixir_milli(0) == 0
    assert c.bar_milli(1) == 500 * 1000 // 28000 == 17
    assert c.unaffordable == 1 and not c.exact


def test_out_of_order_reports_give_the_same_bar(law):
    plays = [(300, "Knight"), (420, "Zap"), (480, "Mirror"), (700, "Giant"), (900, "Hog")]
    ordered = counter(law)
    for tick, card in plays:
        ordered.play(tick, card)
    assert ordered.bar_milli(1000) == 154000 * 1000 // 28000 == 5500
    shuffled = counter(law)
    for tick, card in reversed(plays):
        shuffled.play(tick, card)
        shuffled.bar_milli(1000)                           # a cached query between reports
    assert [shuffled.bar_milli(t) for t in range(0, 1200, 10)] == \
        [ordered.bar_milli(t) for t in range(0, 1200, 10)]
    assert shuffled.last_play == "Hog" and shuffled.exact


def test_credit_after_regen_held_at_cap_overflow_lost(law):
    below = counter(law, elixir_milli=5000)
    below.credit(10, 1000)
    assert below.bar_milli(10) == 145000 * 1000 // 28000 == 5178
    assert below.bar_milli(11) == 173500 * 1000 // 28000 == 6196
    held = counter(law, elixir_milli=10000)
    held.credit(10, 1000)                                  # due at the cap: held
    assert held.bar_milli(11) == 10000
    held.play(50, "Knight")                                # first tick below the cap
    assert held.bar_milli(51) == 224500 * 1000 // 28000 == 8017
    assert held.leaked_milli(51) == 25000 * 1000 // 28000 == 892
    over = counter(law, elixir_milli=9500)
    over.credit(0, 1000)
    assert over.bar_milli(1) == 10000
    assert over.leaked_milli(1) == 14500 * 1000 // 28000 == 517


def test_check_never_repairs_and_resync_reseeds(law):
    c = counter(law)
    c.play(300, "Knight")
    assert c.check(400, 8785) and c.exact
    assert c.check(410, 8000) is False
    assert not c.exact and c.mismatches == [(410, 251000 * 1000 // 28000, 8000)]
    assert c.bar_milli(410) == 8964                        # left wrong, not repaired
    own = counter(law)
    own.play(300, "Knight")
    assert own.resync(400, 5000) is False and not own.exact
    assert own.events == () and own.bar_milli(400) == 5000
    assert own.bar_milli(500) == 190000 * 1000 // 28000 == 6785
    own.play(500, "Mirror")                                # the copy target survives
    assert own.elixir_milli(500) == 78000 * 1000 // 28000 == 2785
    with pytest.raises(ValueError):
        own.play(399, "Knight")                            # before the new seed


def test_counter_validates_inputs(law):
    c = counter(law, tick=100)
    with pytest.raises(ValueError):
        c.play(200, "Unknown")
    with pytest.raises(ValueError):
        c.play(200, "Knight", cost=-1)
    with pytest.raises(ValueError):
        c.press(200, -2)
    with pytest.raises(ValueError):
        c.play(99, "Knight")
    with pytest.raises(ValueError):
        c.bar_milli(50)
    c.advance(150)
    with pytest.raises(ValueError):
        c.advance(149)
    with pytest.raises(ValueError):
        E.ElixirCounter(COSTS, law=E.ElixirLaw.from_calibration(
            {"match": {"MIRROR_COST_RULE": {"value": "something_else"}}}))


def test_from_catalogue_reads_the_engine_layout(law):
    rows = [["Knight", 0, 3, 1, 500, False, 1000, None, "TROOP"],
            ["Mirror", 6, 1, 0, 0, False, 0, None, "SPELL"],
            ["Fireball", 2, 4, 0, 0, False, 0, None, "SPELL"]]
    c = E.ElixirCounter.from_catalogue(rows, law=law)
    assert c.costs == {0: 3, 1: 1, 2: 4} and c.mirror == {1}
    c.play(300, 0)
    c.play(400, 1)
    assert c.bar_milli(401) == 4803
    named = E.ElixirCounter.from_catalogue(rows, law=law, by_name=True)
    assert named.costs == {"Knight": 3, "Mirror": 1, "Fireball": 4} and named.mirror == {"Mirror"}
    moved = E.ElixirCounter.from_catalogue([["Knight", 3, 0], ["Mirror", 1, 6]],
                                           fields=["name", "elixir", "placement"], law=law)
    assert moved.costs == {0: 3, 1: 1} and moved.mirror == {1}
    bare = E.ElixirCounter.from_catalogue([["Knight", 3], ["Mirror", 1]],
                                          fields=["name", "elixir"], law=law)
    assert bare.costs == {0: 3, 1: 1} and bare.mirror == {1}      # the Mirror by name
    with pytest.raises(ValueError):
        E.ElixirCounter.from_catalogue([["Knight", 3]], fields=["name", "cost"], law=law)


# ---------------------------------------------------------------------------
# observe(): state_json dicts
# ---------------------------------------------------------------------------

def _state(make_state, make_player, tick, **team1):
    """A state at `tick` whose team 1 (the counted side) has the given player fields."""
    entities = team1.pop("entities", ())
    return make_state(tick, players=[make_player(0), make_player(1, **team1)], entities=entities)


def test_observe_counts_plays_from_hand_slots(law, make_state, make_player):
    c = E.ElixirCounter(COST_IDS, law=law)
    assert c.observe(_state(make_state, make_player, 100, elixir_milli=7000), 1)
    s1 = _state(make_state, make_player, 110, elixir_milli=4178, hand=[12, 9, 10, 11],
                next_card=13)                              # card 8 (3) left at tick 100
    assert c.observe(s1, 1) and c.exact
    assert c.observe(s1, 1)                                # same tick: nothing moves
    assert [(e.tick, e.card) for e in c.events] == [(100, 8)]
    s2 = _state(make_state, make_player, 120, elixir_milli=4300, hand=[12, 9, 10, 11],
                next_card=13)
    assert c.observe(s2, 1) is False and not c.exact
    assert c.bar_milli(120) == 122000 * 1000 // 28000 == 4357   # never repaired
    # a tick that goes back is a new battle
    assert c.observe(_state(make_state, make_player, 50, elixir_milli=3000), 1)
    assert c.exact and c.bar_milli(50) == 3000 and c.mismatches == []


def test_observe_resyncs_with_repair(law, make_state, make_player):
    c = E.ElixirCounter(COST_IDS, law=law)
    c.observe(_state(make_state, make_player, 100, elixir_milli=7000), 1)
    s1 = _state(make_state, make_player, 110, elixir_milli=4000, hand=[12, 9, 10, 11])
    assert c.observe(s1, 1, repair=True) is False
    assert c.bar_milli(110) == 4000 and c.bar_milli(120) == (112000 + 5000) * 1000 // 28000


def test_observe_counts_presses_from_ability_rows(law, make_state, make_player, make_entity):
    hero = E.ElixirCounter(COST_IDS, law=law)
    hero.observe(_state(make_state, make_player, 100, elixir_milli=7000,
                        abilities=[[1, 0, 2, 9, 0]]), 1)
    spent = _state(make_state, make_player, 110, elixir_milli=145000 * 1000 // 28000,
                   abilities=[[0, 1, 2, 9, 0]])            # the charge turned spent: 2 paid
    assert spent["players"][1]["elixir_milli"] == 5178
    assert hero.observe(spent, 1) and hero.exact
    assert [(e.tick, e.kind, e.amount) for e in hero.events] == [(100, "press", 2)]

    champion = E.ElixirCounter(COST_IDS, law=law)
    champion.observe(_state(make_state, make_player, 100, elixir_milli=7000,
                            abilities=[[1, 0, 3, 13, 0]]), 1)
    dark = _state(make_state, make_player, 110, elixir_milli=4178, abilities=[[0, 0, 3, 13, 0]],
                  entities=[make_entity(uid=50, team=1, card_id=13, hp=500)])
    assert champion.observe(dark, 1) and champion.exact     # dark with him standing: a press

    died = E.ElixirCounter(COST_IDS, law=law)
    died.observe(_state(make_state, make_player, 100, elixir_milli=7000,
                        abilities=[[1, 0, 3, 13, 0]]), 1)
    gone = _state(make_state, make_player, 110, elixir_milli=201000 * 1000 // 28000,
                  abilities=[[0, 0, 3, 13, 0]])            # dark with him gone: a death
    assert gone["players"][1]["elixir_milli"] == 7178
    assert died.observe(gone, 1) and died.exact and died.events == ()


def test_observe_dates_delayed_plays_by_pending_rows(law, make_state, make_player):
    c = E.ElixirCounter(COST_IDS, law=law, delay=21)
    c.observe(_state(make_state, make_player, 100, elixir_milli=10000,
                     pending=[[0, 9, 0, 0, 15, 3]]), 1)    # card 9 runs in 15 ticks
    after = _state(make_state, make_player, 120, elixir_milli=198500 * 1000 // 28000,
                   hand=[8, 12, 10, 11])
    assert after["players"][1]["elixir_milli"] == 7089
    assert c.observe(after, 1) and c.exact
    assert [(e.tick, e.card) for e in c.events] == [(115, 9)]
    # no pending row: accepted at the observation, run `delay` later, clipped to the gap
    d = E.ElixirCounter(COST_IDS, law=law, delay=21)
    d.observe(_state(make_state, make_player, 100, elixir_milli=10000), 1)
    later = _state(make_state, make_player, 120, elixir_milli=196500 * 1000 // 28000,
                   hand=[8, 12, 10, 11])
    assert later["players"][1]["elixir_milli"] == 7017
    assert d.observe(later, 1) and [(e.tick, e.card) for e in d.events] == [(119, 9)]


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

SUB = 18                                    # royalesim subtiles per millitile
SPELLS = ("Fireball", "Arrows", "Zap", "Mirror", "Log", "Poison", "Rocket", "Snowball")
HOG26 = ("HogRider", "Musketeer", "Cannon", "Fireball", "IceGolemite", "IceSpirits",
         "Skeletons", "Log")
OWN_Y = (10000, 22000)                      # millitiles, each side's own half
PERIOD, FIRST = (47, 61), (300, 323)        # each side tries a play this often
PAUSE = ((1400, 1800), (2000, 2300))        # ... except here, so the bars reach the cap again


def _drive(deck, *, end, delay, feed, table):
    """A scripted battle of `deck` against itself; every 10 ticks each side's counted bar
    must equal players[t].elixir_milli. feed "plays": the counters get the accepted plays at
    the tick they run (accepted + delay); "observe": the counters read the states."""
    import royalesim
    from tools import rs_engine
    base = rs_engine.EngineConfig.legacy() if table == "legacy" else rs_engine.EngineConfig.defaults()
    battle = rs_engine.make_battle(base.delayed(delay), royalesim)
    cat = json.loads(battle.catalogue_json())
    ids = {row[0]: i for i, row in enumerate(cat)}
    missing = [name for name in deck if name not in ids]
    assert not missing, f"not in this engine's catalogue: {missing}"
    battle.reset(seed=11, decks=[[ids[name] for name in deck] for _ in (0, 1)], shuffle=0,
                 start_tick=0, elixir_milli=[6000, 6000], tower_hp=None, spawns=[])
    law = E.ElixirLaw.load()
    fields = list(getattr(royalesim, "CATALOGUE_FIELDS", ()) or ()) or None
    counters = [E.ElixirCounter.from_catalogue(cat, fields=fields, law=law, delay=delay)
                for _ in (0, 1)]
    plays, mirrors, turn, nxt = [0, 0], [0, 0], [0, 0], list(FIRST)
    tick, checks, state = 0, 0, {}
    while tick < end:
        state = json.loads(battle.state_json())
        if state["game_over"]:
            break
        assert state["tick"] == tick
        if feed == "observe":
            for team in (0, 1):
                counters[team].observe(state, team)
        if tick % 10 == 0:
            for team in (0, 1):
                truth = state["players"][team]["elixir_milli"]
                got = counters[team].bar_milli(tick)
                assert got == truth, (f"tick {tick} team {team}: counted {got}, engine {truth}; "
                                      f"last events {counters[team].events[-3:]}")
            checks += 1
        cmds, sent = [], []
        for team in (0, 1):
            if tick != nxt[team]:
                continue
            nxt[team] += PERIOD[team]
            if PAUSE[team][0] <= tick < PAUSE[team][1]:
                continue
            hand = state["players"][team]["hand"]
            n = turn[team]
            turn[team] += 1
            slots = [s for s in ((n + j) % len(hand) for j in range(len(hand))) if hand[s] >= 0]
            if not slots:
                continue
            x = (5500, 9000, 12500)[n % 3]
            cmds.append((team, slots[0], x * SUB, OWN_Y[team] * SUB))
            sent.append((team, hand[slots[0]]))
        step = min(nxt[0], nxt[1], (tick // 10 + 1) * 10, end) - tick
        results = battle.step(cmds, step)
        for (team, card), result in zip(sent, results):
            if int(result[1]) != 0:                        # refused (elixir, placement, ...)
                continue
            plays[team] += 1
            mirrors[team] += card in counters[team].mirror
            if feed == "plays":
                counters[team].play(tick + delay, card)
        tick += step
    for team in (0, 1):
        assert counters[team].exact, (team, counters[team].mismatches[:3])
    return {"tick": tick, "checks": checks, "plays": plays, "mirrors": mirrors,
            "overtime": bool(state.get("overtime")), "law": law}


@pytest.mark.engine
def test_installed_calibration_matches_the_constants():
    pytest.importorskip("royalesim")           # imported: calibration_path asks its data_dir()
    path = E.calibration_path()
    assert path is not None and path.is_file()
    law = E.ElixirLaw.load()
    assert law.source == str(path)
    assert law == E.ElixirLaw.constants(), law.summary()


@pytest.mark.engine
@pytest.mark.parametrize("feed,table", [("plays", "legacy"), ("plays", "target"),
                                        ("observe", "legacy")])
def test_engine_bar_matches_count_through_overtime(feed, table):
    """Spells only, cast on each side's own half: no tower falls, so the battle reaches
    overtime at 3600 and triple elixir at 4800."""
    out = _drive(SPELLS, end=4900, delay=0, feed=feed, table=table)
    assert out["tick"] == 4900 and out["overtime"], out
    assert out["checks"] == 490
    assert min(out["plays"]) >= 10 and sum(out["mirrors"]) >= 1, out


@pytest.mark.engine
@pytest.mark.parametrize("feed", ["plays", "observe"])
def test_engine_bar_matches_count_under_command_delay(feed):
    """Hog 2.6 mirror match with the client's 21-tick delay: a play is paid when it runs."""
    out = _drive(HOG26, end=1500, delay=21, feed=feed, table="legacy")
    assert out["tick"] == 1500 and out["checks"] == 150, out
    assert min(out["plays"]) >= 5, out
