"""Self-tests for the shared fakes in tests/pipeline/conftest.py.

Everything here runs without royalesim except the one engine-marked test, which
checks the fakes' layout against the real engine on a machine that has it.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import pytest

from tests.pipeline.conftest import (
    DEPLOY_REASONS,
    ENTITY_FIELDS,
    FIELD,
    HOG26_KEYS,
    KING_HP,
    SUBTILE_PER_MILLITILE,
    FakeBattle,
    entity_row,
    il_action_rows,
    il_payload,
    il_replay_row,
    state_dict,
)


def test_entity_fields_are_the_28_protocol_columns():
    assert len(ENTITY_FIELDS) == 28 == len(set(ENTITY_FIELDS))
    # The positions the existing code reads by index (live_play.threat_lane, rs_rows).
    assert (FIELD["uid"], FIELD["team"], FIELD["kind"], FIELD["card_id"]) == (0, 1, 2, 3)
    assert (FIELD["tower_slot"], FIELD["x"], FIELD["y"], FIELD["hp"]) == (4, 5, 6, 7)
    assert FIELD["status_flags"] == 20 and FIELD["unit_type"] == 27


def test_entity_row_defaults_overrides_and_no_shared_lists(make_entity):
    row = make_entity(team=1, x=36, hp=500)
    assert len(row) == 28
    assert row[FIELD["team"]] == 1 and row[FIELD["x"]] == 36
    assert row[FIELD["hp"]] == 500 and row[FIELD["max_hp"]] == 500
    assert make_entity(hp=10, max_hp=40)[FIELD["max_hp"]] == 40
    with pytest.raises(TypeError):
        make_entity(colour="red")
    row[FIELD["facing"]].append(9)
    assert entity_row()[FIELD["facing"]] == [0, 0]


def test_state_dict_lists_living_towers_first(make_state, make_player):
    state = make_state(tick=40, entities=[entity_row(uid=50, team=0, x=90, y=90)])
    kings = [e for e in state["entities"] if e[FIELD["tower_slot"]] == 0]
    assert sorted(k[FIELD["team"]] for k in kings) == [0, 1]
    blue_king = next(k for k in kings if k[FIELD["team"]] == 0)
    assert blue_king[FIELD["y"]] == 3000 * SUBTILE_PER_MILLITILE
    assert blue_king[FIELD["hp"]] == KING_HP
    assert len(state["entities"]) == 7 and state["entities"][-1][FIELD["uid"]] == 50
    assert state["tick_ms"] == 2000 and state["overtime"] is False
    assert json.loads(json.dumps(state)) == state

    red = make_player(1, tower_hp=(KING_HP, 0, 3052))
    state = make_state(players=[make_player(0), red])
    assert len(state["entities"]) == 5                  # the fallen princess is gone
    assert state["players"][1]["hand"] == [8, 9, 10, 11]
    assert state["players"][0]["hand_costs"] == [3, 4, 4, 2]


def test_fake_battle_records_calls_in_order(make_fake_battle):
    battle = make_fake_battle(card_names=None, slot_of_k=[[0, 1, 2], [0, 1, 2]],
                              states=[state_dict(tick=0), state_dict(tick=5)],
                              step_results=[[[0, 4]]])
    battle.set_command_delay_ticks(21, 21)
    battle.reset(seed=0, start_tick=0, decks=[list(range(8)), list(range(8, 16))])
    assert json.loads(battle.state_json())["tick"] == 0
    assert battle.step([(0, 1, 100, 200)], 5) == [[0, 4]]
    assert json.loads(battle.state_json())["tick"] == 5
    assert battle.step([], 1) == []                     # script used up: default results
    assert battle.step([(1, 0, 5, 5)], 1) == [[1, 0]]
    assert json.loads(battle.state_json())["tick"] == 5  # the last state from then on
    assert battle.names() == ["set_command_delay_ticks", "reset", "step", "step", "step"]
    assert battle.calls[2].args == ([(0, 1, 100, 200)], 5)
    assert battle.calls_named("reset")[0].kwargs["decks"][1][0] == 8
    assert battle.tick == 7 and battle.delay == (21, 21) and battle.state_reads == 3
    assert battle.init_kwargs["slot_of_k"] == [[0, 1, 2], [0, 1, 2]]


def test_fake_battle_callables_deploy_and_save_load():
    sub = SUBTILE_PER_MILLITILE
    battle = FakeBattle(states=lambda b: state_dict(tick=b.tick),
                        deploy=lambda team, slot, x, y: 0 if y < 16_000 * sub else 8)
    battle.reset(start_tick=90)
    assert json.loads(battle.state_json())["tick"] == 90
    assert battle.check_deploy(0, 0, 0, 100) == 0
    assert DEPLOY_REASONS[battle.check_deploy(0, 1, 0, 20_000 * sub)] == "OUT_OF_TERRITORY"
    assert battle.deploy_checks == [(0, 0, 0, 100), (0, 1, 0, 20_000 * sub)]
    snap = battle.save()
    battle.step([], 10)
    assert battle.tick == 100
    battle.load(snap)
    assert battle.tick == 90 and battle.steps == 0
    assert battle.names() == ["reset", "save", "step", "load"]
    assert battle.pending_commands(0) == []
    assert json.loads(battle.catalogue_json())[2][0] == "HogRider"


def test_fake_royalesim_is_installed_for_the_test(fake_royalesim):
    import royalesim
    assert royalesim is fake_royalesim
    fake_royalesim.battle_kwargs["step_results"] = [[[0, 3]]]
    battle = royalesim.Battle(card_names=None, slot_of_k=[[0, 1, 2], [0, 1, 2]])
    assert fake_royalesim.battles == [battle]
    assert battle.init_kwargs == {"card_names": None, "slot_of_k": [[0, 1, 2], [0, 1, 2]]}
    assert royalesim.DEPLOY_REASONS[battle.step([(0, 0, 1, 1)])[0][1]] == "EMPTY_SLOT"
    assert royalesim.ENTITY_FIELDS[FIELD["hp"]] == "hp"


def test_fake_royalesim_is_removed_afterwards():
    mod = sys.modules.get("royalesim")
    assert mod is None or getattr(mod, "__version__", "") != "0.0.0+fake"


def test_il_payload_shape(make_il_payload):
    payload = make_il_payload()
    team = payload["battle"]["team"]["players"][0]
    opp = payload["battle"]["opponent"]["players"][0]
    team_keys = [c["card_key"] for c in team["deck"]]
    assert "musketeer-hero" in team_keys and "skeletons-ev1" in team_keys
    assert {k.removesuffix("-ev1").removesuffix("-hero") for k in team_keys} == set(HOG26_KEYS)
    assert "knight-ev1" in [c["card_key"] for c in opp["deck"]]
    assert [c["level"] for c in team["deck"]] == [11] * 8
    assert [c["level"] for c in opp["deck"]] == [12, 11, 13, 11, 12, 11, 14, 11]
    events = payload["events"]
    assert Counter(e["kind"] for e in events) == {"play_card": 5, "activate_ability": 1}
    assert [e["source_index"] for e in events] == list(range(6))
    assert [e["replay_tick_20hz"] for e in events] == sorted(e["replay_tick_20hz"] for e in events)
    team_plays = [e for e in events if e["side"] == "team" and e["kind"] == "play_card"]
    assert all(e["coordinates"]["native_world_units"]["y"] < 16_000 for e in team_plays)
    assert team_plays[0]["card_key"] == "musketeer-hero"
    ability = next(e for e in events if e["kind"] == "activate_ability")
    assert ability["ability_source_candidates"] == ["musketeer-hero"]
    assert payload["battle"]["team"]["crowns"] == 1 and payload["battle"]["result"] == "victory"
    assert opp["final_tower_hitpoints"]["princess_left"] == 0
    assert json.loads(json.dumps(payload)) == payload


def test_il_payload_team_at_high_y_and_bad_cards():
    payload = il_payload(team_low=False, team_forms={}, abilities=())
    first = payload["events"][0]
    assert first["card_key"] == "musketeer"
    assert first["coordinates"]["native_world_units"] == {"x": 14_500, "y": 24_500}
    with pytest.raises(ValueError):
        il_payload(plays=[(10, "team", "golem", 0, 0)])
    with pytest.raises(ValueError):
        il_payload(team_forms={"golem": 1})


def test_il_parquet_style_rows():
    payload = il_payload(tag="t-7")
    row = il_replay_row(payload)
    assert row["replay_tag"] == "t-7" and row["game_mode"] == "Ladder"
    assert json.loads(row["payload_json"]) == payload
    actions = il_action_rows(payload)
    assert len(actions) == 6
    assert (actions[0]["native_x"], actions[0]["native_y"]) == (3500, 7500)
    assert actions[-1]["kind"] == "activate_ability"


@pytest.mark.engine
def test_fake_layout_matches_the_installed_engine():
    royalesim = pytest.importorskip("royalesim")
    assert tuple(royalesim.ENTITY_FIELDS) == ENTITY_FIELDS
    # Only the names are promised: code looks reasons up through the module it was given.
    assert royalesim.DEPLOY_REASONS[0] == "OK"
    assert set(DEPLOY_REASONS) <= set(royalesim.DEPLOY_REASONS)
    assert royalesim.SUBTILE_PER_MILLITILE == SUBTILE_PER_MILLITILE
    battle = royalesim.Battle(card_names=None, slot_of_k=[[0, 1, 2], [0, 1, 2]])
    ids = {row[0]: i for i, row in enumerate(json.loads(battle.catalogue_json()))}
    deck = [ids[n] for n in ("Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits",
                             "Musketeer", "Skeletons", "Log")]
    battle.reset(seed=0, decks=[deck, deck], shuffle=0, start_tick=0,
                 elixir_milli=[6000, 6000], tower_hp=None, spawns=[],
                 forms=[[0] * 8, [0] * 8], levels=[[11] * 8, [11] * 8], tower_levels=[11, 11])
    state = json.loads(battle.state_json())
    assert {"tick", "players", "entities", "game_over", "winner"} <= set(state)
    assert {"hand", "elixir_milli", "next_card", "crowns", "tower_hp",
            "tower_max_hp"} <= set(state["players"][0])
    assert all(len(e) == len(ENTITY_FIELDS) for e in state["entities"])
    kings = {e[FIELD["team"]]: e for e in state["entities"] if e[FIELD["tower_slot"]] == 0}
    assert kings[0][FIELD["y"]] < kings[1][FIELD["y"]]     # blue owns low y, as the fakes say
