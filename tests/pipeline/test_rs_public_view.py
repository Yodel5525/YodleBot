"""S02: tools/rs_public_view.py, the public view of a state (docs/LIVE_SEARCH_PLAN.md 4.1, 5).

No engine, model or device: conftest's state_json-shaped dicts (entity rows in royalesim's
ENTITY_FIELDS order, FIELD), a small synthetic catalogue (royalesim's column order: name,
placement, elixir, ...; Hog 2.6 is 0-7, the Mirror has placement 6) and rs_memory views. The
checks the plan asks for: a noised state comes out with everything a screen does not show
removed; card cycles (fewer than four plays, unknown plays, the Mirror); a hidden Tesla kept by
permanence with its hit points drained. Then the opponent elixir wrappers, the live identity,
the constants pinned to tools/public_state and tools/rs_rows (those need numpy), and the import
weight.
"""
from __future__ import annotations

import copy
import importlib
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import (ENTITY_FIELDS, FIELD, KING_HP, PRINCESS_HP, entity_row,
                                     player_dict, state_dict)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import rs_public_view as PV  # noqa: E402

import rs_elixir  # noqa: E402  (tools/ is on sys.path once rs_public_view is imported)
import rs_memory  # noqa: E402

SUB = 18
NAMES = ("Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons",
         "Log", "Knight", "Archer", "Giant", "Arrows", "Valkyrie", "MiniPekka", "BabyDragon", "Zap",
         "Tesla", "Mirror")
COSTS = (3, 4, 4, 2, 1, 4, 1, 2, 3, 3, 5, 3, 4, 4, 4, 2, 4, 1)
KINDS = ("BUILDING", "SPELL", "TROOP", "TROOP", "TROOP", "TROOP", "TROOP", "SPELL", "TROOP", "TROOP",
         "TROOP", "SPELL", "TROOP", "TROOP", "TROOP", "SPELL", "BUILDING", "SPELL")
#: catalogue_json rows: name, placement, elixir, count, radius, flying, hitpoints, footprint, kind.
CATALOGUE = [[n, 6 if n == "Mirror" else 0, c, 1, 0, False, 0, 0, k]
             for n, c, k in zip(NAMES, COSTS, KINDS)]
(CANNON, FIREBALL, HOG, ICEGOLEM, SPIRITS, MUSK, SKELS, LOG, KNIGHT, ARCHER, GIANT, ARROWS, VALK,
 MINIPEKKA, BABYD, ZAP, TESLA, MIRROR) = range(len(NAMES))
LAW = rs_elixir.ElixirLaw.constants()
PRIVATE = [n for n in ENTITY_FIELDS if n not in PV.PUBLIC_FIELDS]
TOWERS = [1, 2, 3, 4, 5, 6]          # state_dict's crown tower uids


def milli(fine: int) -> int:
    return fine * 1000 // LAW.scale


def memory(*plays) -> rs_memory.PublicMemory:
    """A PublicMemory with (team, card id or None for an unknown play, exec tick) plays."""
    mem = rs_memory.PublicMemory()
    for team, card, tick in plays:
        mem.record(team, rs_memory.UNKNOWN if card is None else card + rs_memory.TOKEN_OFFSET, tick)
    return mem


# ---------------------------------------------------------------------------
# the view of a noised simulator state
# ---------------------------------------------------------------------------

def noised_state():
    """What rs_match hands seat 1: our state, and everything a screen does not show."""
    them = player_dict(0, elixir_milli=7300, hand=[KNIGHT, ARCHER, GIANT, ARROWS], next_card=VALK,
                       deck=list(range(8, 16)), forms=[0] * 8, mirror_target=GIANT,
                       abilities=[[1, 0, 2]],
                       pending=[[0, MINIPEKKA, 9000 * SUB, 20000 * SUB, 5, 4]], pending_cost=4)
    # Our hand after a Hog from the deal [0, 2, 5, 6 | 4, 1, 3, 7]: the Hog's slot took the
    # Ice Spirits, the Fireball is next.
    me = player_dict(1, elixir_milli=5400, hand=[CANNON, SPIRITS, MUSK, SKELS], next_card=FIREBALL,
                     deck=list(range(8)), forms=[0] * 8,
                     pending=[[0, SKELS, 5000 * SUB, 20000 * SUB, 10, 1]], pending_cost=1)
    hog = entity_row(uid=20, team=1, kind=0, card_id=HOG, x=3500 * SUB, y=14000 * SUB, hp=800,
                     max_hp=1696, target_uid=30, attack_phase=1, stun_ticks=3, knockback_ticks=2,
                     charge=500, buffs=[["rage", 300]], shield=10, dest_x=1, dest_y=2,
                     facing=[1, 2], ability_ticks=4, mount_uid=7, status_flags=64, unit_type=3)
    knight = entity_row(uid=30, team=0, kind=0, card_id=KNIGHT, x=3500 * SUB, y=13000 * SUB,
                        hp=900, max_hp=1766, target_uid=20, attack_phase=2, stun_ticks=5,
                        buffs=[["slow", 100]], status_flags=64 | PV.STATUS_EVOLVED, unit_type=8,
                        deploy_ticks=4)
    cloaked = entity_row(uid=31, team=0, card_id=ARCHER, x=4000 * SUB, y=12000 * SUB, hp=300,
                         status_flags=PV.STATUS_INVISIBLE)
    tesla = entity_row(uid=32, team=0, kind=1, card_id=TESLA, x=9000 * SUB, y=12000 * SUB,
                       hp=1000, status_flags=PV.STATUS_HIDDEN)
    digger = entity_row(uid=33, team=0, card_id=MINIPEKKA, x=6000 * SUB, y=18000 * SUB, hp=1000,
                        status_flags=PV.STATUS_UNDERGROUND | PV.STATUS_HERO, dest_x=5, dest_y=6)
    dead = entity_row(uid=34, team=0, card_id=GIANT, x=9000 * SUB, y=9000 * SUB, hp=0,
                      max_hp=4000)
    spells = [[0, ARROWS, 1, 4000 * SUB, 20000 * SUB, 3500 * SUB, 25500 * SUB, 3, 0, 0, 0, 7],
              [1, FIREBALL, 1, 9000 * SUB, 15000 * SUB, 9000 * SUB, 9000 * SUB, 0, 0, 0, 0, -1]]
    return state_dict(tick=300, players=[them, me],
                      entities=[hog, knight, cloaked, tesla, digger, dead], spells=spells,
                      projectiles=[[1, 100, 200, 300, 400, 30, 0, -1]],
                      debug={"true_deck": list(range(8, 16))})


NOISED_MEMORY = ((0, KNIGHT, 100), (1, HOG, 150), (0, ARROWS, 200))
SENT = [(310, SKELS, 5000 * SUB, 20000 * SUB), (150, HOG, 3500 * SUB, 14500 * SUB),
        (220, LOG, 9000 * SUB, 15000 * SUB)]


def test_a_noised_state_loses_everything_a_screen_does_not_show():
    state = noised_state()
    before = copy.deepcopy(state)
    mem = memory(*NOISED_MEMORY).view(300, rs_memory.BOARD_LAG)
    view = PV.public_view(state, 1, mem, fields=FIELD, opp_elixir=(6500, True), sent=SENT)
    assert state == before                                       # the input is not changed
    assert set(view) == {"tick", "tick_ms", "regular_ticks", "overtime_ticks", "elixir_rate",
                         "overtime", "game_over", "winner", "players", "entities", "spells",
                         "projectiles", "public"}                # no "debug"
    # The opponent's block is the live one: no hand, elixir, next card or pending plays; its
    # deck is the cards its plays showed.
    assert view["players"][0] == {
        "team": 0, "elixir_milli": 0, "hand": [], "hand_costs": [], "next_card": -1,
        "deck": [KNIGHT, ARROWS], "crowns": 0, "tower_hp": [KING_HP, PRINCESS_HP, PRINCESS_HP],
        "tower_max_hp": [KING_HP, PRINCESS_HP, PRINCESS_HP], "king_active": False,
        "pending": [], "pending_cost": 0}
    assert view["players"][1] == state["players"][1]             # ours, whole
    assert view["players"][1] is not state["players"][1]
    # Units: the invisible Archer, the hidden Tesla and the dead Giant are gone; the tunnelling
    # unit stays (its dirt trail is on the screen).
    rows = {e[FIELD["uid"]]: e for e in view["entities"]}
    assert sorted(rows) == TOWERS + [20, 30, 33]
    for e in view["entities"]:
        for name in PRIVATE:
            assert e[FIELD[name]] == PV.ENTITY_DEFAULTS[name], (e[FIELD["uid"]], name)
    hog, knight = state["entities"][6], state["entities"][7]
    for name in PV.PUBLIC_FIELDS:
        if name != "status_flags":
            assert rows[20][FIELD[name]] == hog[FIELD[name]], name
            assert rows[30][FIELD[name]] == knight[FIELD[name]], name
    assert rows[20][FIELD["status_flags"]] == 0                  # a windup bit is not shown
    assert rows[30][FIELD["status_flags"]] == PV.STATUS_EVOLVED
    assert rows[33][FIELD["status_flags"]] == PV.STATUS_UNDERGROUND | PV.STATUS_HERO
    # Spells: the aim is the centre (live's placeholder), ticks_flown kept (-1 reads 0).
    assert view["spells"] == [
        [0, ARROWS, 0, 4000 * SUB, 20000 * SUB, 4000 * SUB, 20000 * SUB, 0, 0, 0, 0, 7],
        [1, FIREBALL, 0, 9000 * SUB, 15000 * SUB, 9000 * SUB, 15000 * SUB, 0, 0, 0, 0, 0]]
    assert view["projectiles"] == [] and view["winner"] == -1
    pub = view["public"]
    assert pub["team"] == 1 and pub["tick"] == 300
    assert pub["opp_elixir"] == [6500, True] and pub["opp_missed"] is False
    assert pub["opp_plays"] == [[KNIGHT, 100], [ARROWS, 200]]
    assert pub["own_inflight"] == [[310, SKELS, 5000 * SUB, 20000 * SUB]]
    assert pub["own_sent"] == [[220, LOG, 9000 * SUB, 15000 * SUB],
                               [310, SKELS, 5000 * SUB, 20000 * SUB]]
    assert pub["dropped"] == {"dead": 1, "projectiles": 1, "unseen": 2}
    assert pub["kept"] == []
    assert pub["own_cycle"]["hand"] == [CANNON, SPIRITS, MUSK, SKELS]
    assert pub["own_cycle"]["queue"] == [FIREBALL, None, None, HOG]
    assert pub["own_cycle"]["unplaced"] == [ICEGOLEM, LOG] and pub["own_cycle"]["exact"]
    assert pub["opp_cycle"]["queue"] == [None, None, KNIGHT, ARROWS]
    assert pub["opp_cycle"]["hand"] == [None] * 4 and pub["opp_cycle"]["exact"]


def test_is_public_names_what_is_privileged_and_the_view_is_idempotent():
    state = noised_state()
    mem = memory(*NOISED_MEMORY).view(300, rs_memory.BOARD_LAG)
    view = PV.public_view(state, 1, mem, fields=FIELD, opp_elixir=(6500, True), sent=SENT)
    assert PV.is_public(view, 1, fields=FIELD) == []
    problems = PV.is_public(state, 1, fields=FIELD)
    for needle in ("opponent elixir 7300", "opponent hand", "opponent next card",
                   "opponent pending", "1 projectiles", "enemy uid 31", "enemy uid 32",
                   "uid 20 target_uid 30", "uid 30 status bits 64", "spell card 11 has an aim"):
        assert any(needle in p for p in problems), (needle, problems)
    again = PV.public_view(view, 1, mem, fields=FIELD, opp_elixir=(6500, True), sent=SENT)
    assert {k: v for k, v in again.items() if k != "public"} == \
        {k: v for k, v in view.items() if k != "public"}
    assert {k: v for k, v in again["public"].items() if k != "dropped"} == \
        {k: v for k, v in view["public"].items() if k != "dropped"}
    assert again["public"]["dropped"] == {}
    # Seat 0's view of the same state: the other side is the opponent.
    zero = PV.public_view(state, 0, mem, fields=FIELD)
    assert zero["players"][1]["hand"] == [] and zero["players"][1]["deck"] == [HOG]
    assert zero["players"][0] == state["players"][0]
    assert {e[FIELD["uid"]] for e in zero["entities"]} == set(TOWERS) | {20, 30, 31, 32, 33}


def test_refused_plays_are_left_out_and_the_winner_shows_only_at_the_end():
    mem = memory(*NOISED_MEMORY).view(300, rs_memory.BOARD_LAG)
    unran = Counter({(0, ARROWS + rs_memory.TOKEN_OFFSET, 200): 1})
    view = PV.public_view(noised_state(), 1, mem, fields=FIELD, unran=unran)
    assert view["players"][0]["deck"] == [KNIGHT]
    assert view["public"]["opp_plays"] == [[KNIGHT, 100]]
    assert view["public"]["opp_cycle"]["queue"] == [None, None, None, KNIGHT]
    assert view["public"]["opp_elixir"] is None
    assert PV.public_view(state_dict(tick=500, winner=0), 1, fields=FIELD)["winner"] == -1
    assert PV.public_view(state_dict(tick=500, game_over=True, winner=1), 1,
                          fields=FIELD)["winner"] == 1
    # the in-flight cut follows `tick` when given
    cut = PV.public_view(noised_state(), 1, mem, fields=FIELD, sent=SENT, tick=320)
    assert cut["public"]["tick"] == 320 and cut["public"]["own_inflight"] == []


def test_an_unreported_status_has_no_public_view():
    own = entity_row(uid=40, team=1, card_id=HOG, x=3500 * SUB, y=20000 * SUB, hp=500,
                     status_flags=-1)
    view = PV.public_view(state_dict(tick=10, entities=[own]), 1, fields=FIELD)
    assert [e[FIELD["status_flags"]] for e in view["entities"] if e[FIELD["uid"]] == 40] == [0]
    enemy = entity_row(uid=41, team=0, card_id=KNIGHT, x=3500 * SUB, y=12000 * SUB, hp=500,
                       status_flags=-1)
    with pytest.raises(PV.PublicViewError, match="status_flags"):
        PV.public_view(state_dict(tick=10, entities=[enemy]), 1, fields=FIELD)
    assert not PV.seen_by(entity_row(team=0, status_flags=PV.STATUS_HIDDEN), 1, FIELD)
    assert PV.seen_by(entity_row(team=1, status_flags=PV.STATUS_HIDDEN), 1, FIELD)


# ---------------------------------------------------------------------------
# the live state is already public
# ---------------------------------------------------------------------------

def live_row(**values) -> list:
    """An entity row as PublicStateBuilder builds it: ENTITY_DEFAULTS, then the given columns."""
    row = [copy.deepcopy(PV.ENTITY_DEFAULTS[name]) for name in ENTITY_FIELDS]
    for name, v in values.items():
        row[FIELD[name]] = v
    return row


def live_state() -> dict:
    towers = [KING_HP, PRINCESS_HP, PRINCESS_HP]
    them = {"team": 0, "elixir_milli": 0, "hand": [], "hand_costs": [], "next_card": -1,
            "deck": [KNIGHT], "crowns": 0, "tower_hp": list(towers), "tower_max_hp": list(towers),
            "king_active": False, "pending_cost": 0}
    me = {"team": 1, "elixir_milli": 5000, "hand": [CANNON, SPIRITS, MUSK, SKELS],
          "hand_costs": [3, 1, 4, 1], "next_card": FIREBALL, "deck": list(range(8)), "crowns": 0,
          "tower_hp": list(towers), "tower_max_hp": list(towers), "king_active": False,
          "pending_cost": 4}
    ents = [live_row(uid=7, team=0, card_id=KNIGHT, x=3500 * SUB, y=13000 * SUB, hp=900,
                     max_hp=1766, unit_type=8),
            live_row(uid=8, team=1, card_id=HOG, x=3500 * SUB, y=14000 * SUB, hp=1696,
                     max_hp=1696, unit_type=3, deploy_ticks=5)]
    return {"tick": 300, "tick_ms": 15000, "regular_ticks": 3600, "overtime_ticks": 2400,
            "elixir_rate": 1, "overtime": False, "players": [them, me], "entities": ents,
            "spells": [[0, ARROWS, 0, 4000 * SUB, 20000 * SUB, 4000 * SUB, 20000 * SUB,
                        0, 0, 0, 0, 5]],
            "projectiles": [], "game_over": False, "winner": -1,
            "perception": {"tracks": 3, "units": 2, "spells": 1, "dropped": {}, "notes": []}}


def test_live_the_view_is_the_state_itself_and_a_check_asserts_it():
    state = live_state()
    assert PV.is_public(state, 1, fields=FIELD) == []
    own = PV.CardCycle.from_known([CANNON, SPIRITS, MUSK, SKELS], [FIREBALL, None, None, HOG])
    mem = memory((0, KNIGHT, 250)).view(300, 0)
    out = PV.public_view(state, 1, mem, fields=FIELD, live=True, own_cycle=own,
                         opp_elixir=(4000, False), sent=[(305, HOG, 3500 * SUB, 14500 * SUB)])
    assert {k: v for k, v in out.items() if k != "public"} == state
    assert out["public"]["own_cycle"] == own.to_json()
    assert out["public"]["opp_cycle"]["queue"] == [None, None, None, KNIGHT]
    assert out["public"]["opp_elixir"] == [4000, False]
    assert out["public"]["own_inflight"] == [[305, HOG, 3500 * SUB, 14500 * SUB]]
    with pytest.raises(PV.PublicViewError, match="not public"):
        PV.public_view(noised_state(), 1, mem, fields=FIELD, live=True)


# ---------------------------------------------------------------------------
# card cycles
# ---------------------------------------------------------------------------

def test_fewer_than_four_plays_fill_the_queue_tail():
    c = PV.CardCycle.from_plays([KNIGHT, ARCHER])
    assert c.hand == (None,) * 4 and c.queue == (None, None, KNIGHT, ARCHER)
    assert c.exact and c.reasons == () and c.plays == 2 and c.next is None
    assert c.seen == (KNIGHT, ARCHER) and c.unplaced == () and c.holes == (4, 2)
    assert PV.CardCycle.from_plays([]).queue == (None,) * 4


def test_the_last_four_plays_are_the_queue_and_older_cards_are_in_hand():
    # Deal [8, 9, 10, 11 | 12, 13, 14, 15]; after 8 9 10 11 12 8 the hand is {9, 13, 14, 15}
    # and the queue 10, 11, 12, 8 (worked through by hand).
    c = PV.CardCycle.from_plays([KNIGHT, ARCHER, GIANT, ARROWS, VALK, KNIGHT])
    assert c.queue == (GIANT, ARROWS, VALK, KNIGHT) and c.next == GIANT
    assert c.hand == (ARCHER, None, None, None) and c.exact
    # (card, tick) pairs read as their card
    assert PV.CardCycle.from_plays([(KNIGHT, 10), (ARCHER, 30)]).queue == (None, None, KNIGHT, ARCHER)


def test_unknown_plays_mirror_repeats_and_missed_plays_make_a_cycle_inexact():
    c = PV.CardCycle.from_plays([KNIGHT, None, GIANT, ARROWS, VALK])
    assert c.queue == (None, GIANT, ARROWS, VALK) and c.hand == (KNIGHT, None, None, None)
    assert not c.exact and c.reasons == ("unknown_play",)
    assert PV.CardCycle.from_plays([KNIGHT, -1]).reasons == ("unknown_play",)
    m = PV.CardCycle.from_plays([KNIGHT, MIRROR, ARCHER], mirror=PV.mirror_ids(CATALOGUE))
    assert PV.mirror_ids(CATALOGUE) == [MIRROR]
    assert m.queue == (None, KNIGHT, MIRROR, ARCHER) and m.reasons == ("mirror",)
    r = PV.CardCycle.from_plays([KNIGHT, ARCHER, KNIGHT])        # a Mirror seen as its copy
    assert r.queue == (None, None, ARCHER, KNIGHT) and r.reasons == ("repeat",)
    assert PV.CardCycle.from_plays([KNIGHT], missed=True).reasons == ("missed_play",)
    nine = PV.CardCycle.from_plays([KNIGHT, ARCHER, GIANT, ARROWS, VALK, MINIPEKKA, BABYD, ZAP,
                                    TESLA])
    assert nine.reasons == ("confusion",) and nine.queue == (MINIPEKKA, BABYD, ZAP, TESLA)
    assert nine.hand == (ARCHER, GIANT, ARROWS, VALK)            # the four most recent
    both = PV.CardCycle.from_plays([None, MIRROR], mirror=[MIRROR], missed=True)
    assert both.reasons == ("unknown_play", "mirror", "missed_play")


def test_our_cycle_from_our_hud_hand_next_card_and_deck():
    # Deal [0, 2, 5, 6 | 4, 1, 3, 7]: after the Hog the hand is 0 4 5 6 and the Fireball next;
    # the slots behind it are the Ice Golem and the Log, in an order not known yet.
    one = PV.CardCycle.from_plays([HOG], hand=[CANNON, SPIRITS, MUSK, SKELS], next_card=FIREBALL,
                                  deck=range(8))
    assert one.hand == (CANNON, SPIRITS, MUSK, SKELS)
    assert one.queue == (FIREBALL, None, None, HOG) and one.unplaced == (ICEGOLEM, LOG)
    assert one.exact
    # After the Skeletons too: the hand 0 4 5 1, the queue 3 7 2 6, the last slot placed.
    two = PV.CardCycle.from_plays([HOG, SKELS], hand=[CANNON, SPIRITS, MUSK, FIREBALL],
                                  next_card=ICEGOLEM, deck=range(8))
    assert two.queue == (ICEGOLEM, LOG, HOG, SKELS) and two.unplaced == () and two.exact
    # A hand that still holds a card the plays say is queued: the hand wins.
    bad = PV.CardCycle.from_plays([HOG], hand=[CANNON, HOG, MUSK, SKELS], next_card=SPIRITS,
                                  deck=range(8))
    assert bad.queue == (SPIRITS, None, None, None) and bad.reasons == ("hand_mismatch",)
    off = PV.CardCycle.from_plays([KNIGHT, ARCHER, GIANT, ARROWS], next_card=ARCHER)
    assert off.reasons == ("next_mismatch",) and off.queue == (KNIGHT, ARCHER, GIANT, ARROWS)
    live = PV.CardCycle.from_known([CANNON, -1, MUSK, SKELS], [SPIRITS, None, LOG, HOG])
    assert live.hand == (CANNON, None, MUSK, SKELS) and live.queue == (SPIRITS, None, LOG, HOG)
    assert live.exact and live.next == SPIRITS


def test_fill_places_unknown_cards_and_keeps_mirror_out_of_the_hand():
    c = PV.CardCycle.from_plays([KNIGHT, ARCHER])
    f = c.fill([VALK, MIRROR, MINIPEKKA, BABYD, ZAP, TESLA, GIANT], omit=[MIRROR])
    assert f.hand == (VALK, MINIPEKKA, BABYD, ZAP) and f.queue == (MIRROR, TESLA, KNIGHT, ARCHER)
    assert f.filled == 6 and f.order() == [VALK, MINIPEKKA, BABYD, ZAP, MIRROR, TESLA, KNIGHT,
                                           ARCHER]
    assert f.exact == c.exact and f.reasons == c.reasons
    # our unplaced cards go to the open queue slots first, in their order
    one = PV.CardCycle.from_plays([HOG], hand=[CANNON, SPIRITS, MUSK, SKELS], next_card=FIREBALL,
                                  deck=range(8))
    assert one.fill().queue == (FIREBALL, ICEGOLEM, LOG, HOG) and one.fill().unplaced == ()


def test_cycle_json_round_trip_and_from_view():
    c = PV.CardCycle.from_plays([KNIGHT, None, GIANT, ARROWS, VALK], mirror=[MIRROR])
    assert PV.CardCycle.from_json(c.to_json()) == c
    mem = memory(*NOISED_MEMORY).view(300, rs_memory.BOARD_LAG)
    view = PV.public_view(noised_state(), 1, mem, fields=FIELD)
    assert PV.CardCycle.from_view(view, 1).queue == (FIREBALL, None, None, HOG)
    assert PV.CardCycle.from_view(view, 0).queue == (None, None, KNIGHT, ARROWS)


def test_plays_of_reads_tokens_unknown_plays_and_json_keys():
    mem = memory((0, None, 50), (0, KNIGHT, 100), (1, HOG, 120), (0, KNIGHT, 400))
    view = mem.view(500, 0)
    assert PV.plays_of(view, 0) == [(None, 50), (KNIGHT, 100), (KNIGHT, 400)]
    assert PV.plays_of(view, 1) == [(HOG, 120)]
    unran = Counter({(0, KNIGHT + 2, 400): 1, (1, KNIGHT + 2, 400): 1})
    assert PV.plays_of(view, 0, unran) == [(None, 50), (KNIGHT, 100)]
    as_json = {"seen": {}, "last": {"0": [[KNIGHT + 2, 100]], "1": []}}
    assert PV.plays_of(as_json, 0) == [(KNIGHT, 100)]
    assert PV.plays_of(None, 1) == []


# ---------------------------------------------------------------------------
# opponent elixir
# ---------------------------------------------------------------------------

def test_opp_elixir_pays_each_seen_play_at_its_execution_tick():
    mem = memory((0, GIANT, 300))
    oe = PV.OppElixir(CATALOGUE, law=LAW)
    # 168000 + 224 x 500 is full (280000) at tick 224; the Giant's 140000 goes at 300
    assert oe.view(1, mem.view(300, 0), 300) == (5000, True, False, 0)
    assert oe.view(1, mem.view(301, 0), 301) == (milli(140500), True, False, 0) == (5017, True,
                                                                                   False, 0)
    assert oe.view(1, mem.view(301, 0), 301).pair == (5017, True)
    # an unknown play cannot be priced: the count is not exact, its bar unchanged
    mem.record(0, rs_memory.UNKNOWN, 400)
    r = oe.view(1, mem.view(400, 0), 400)
    assert r == (milli(140500 + 99 * 500), False, False, 0) == (6785, False, False, 0)
    assert oe.unpriced == 1 and oe.restarts == 1
    # observer 0 counts team 1's plays
    zero = PV.OppElixir(CATALOGUE, law=LAW)
    assert zero.view(0, memory((1, GIANT, 300), (0, GIANT, 100)).view(300, 0), 300).milli == 5000


def test_opp_elixir_flags_a_missed_play_and_is_never_exact_when_blind_or_late():
    # 88000 at tick 120 cannot pay a second Giant: a play was missed
    two = memory((0, GIANT, 100), (0, GIANT, 120)).view(120, 0)
    r = PV.OppElixir(CATALOGUE, law=LAW).view(1, two, 120)
    assert r == (0, False, True, 1)
    cyc = PV.public_view(state_dict(tick=120), 1, two, fields=FIELD, opp_elixir=r)
    assert cyc["public"]["opp_missed"] is True
    # (two Giants 20 ticks apart are also a repeat: a card comes back four plays later)
    assert cyc["public"]["opp_cycle"]["reasons"] == ["missed_play", "repeat"]
    assert cyc["public"]["opp_cycle"]["queue"] == [None, None, None, GIANT]
    empty = rs_memory.empty_view()
    assert PV.OppElixir(CATALOGUE, law=LAW, start_tick=91).view(1, empty, 100) == (7785, False,
                                                                                  False, 0)
    assert PV.OppElixir(CATALOGUE, law=LAW, start_tick=90).view(1, empty, 100).exact
    assert not PV.OppElixir(CATALOGUE, law=LAW, blind=True).view(1, empty, 100).exact


def test_opp_elixir_leaves_out_refused_plays_and_restarts_when_a_fed_play_goes():
    mem = memory((0, GIANT, 300)).view(300, 0)
    oe = PV.OppElixir(CATALOGUE, law=LAW)
    assert oe.view(1, mem, 300).milli == 5000
    unran = Counter({(0, GIANT + rs_memory.TOKEN_OFFSET, 300): 1})
    assert oe.view(1, mem, 300, unran=unran) == (10_000, True, False, 0)
    assert oe.restarts == 2


def test_seat_opp_elixir_reads_the_seats_own_count():
    counter = rs_elixir.ElixirCounter.from_catalogue(CATALOGUE, law=LAW)
    counter.play(100, GIANT)
    counter.play(120, GIANT)                                   # unaffordable

    class Seat:
        def __init__(self, seen):
            self.calls = []
            self._seen = seen

        def opp_elixir(self, team, mem, tick, royalesim=None):
            self.calls.append((team, tick))
            return 4321, False

    seat = Seat(SimpleNamespace(counters={0: counter, 1: None}))
    got = PV.SeatOppElixir(seat).view(1, None, 130, unran=Counter({(0, 12, 100): 1}))
    assert got == (4321, False, True, 1) and seat.calls == [(1, 130)]
    assert PV.SeatOppElixir(Seat(None)).view(1, None, 130) == (4321, False, True, -1)
    assert isinstance(PV.make_opp_elixir(seat, CATALOGUE), PV.SeatOppElixir)
    assert isinstance(PV.make_opp_elixir(None, CATALOGUE, law=LAW), PV.OppElixir)
    with pytest.raises(ValueError):
        PV.make_opp_elixir()
    with pytest.raises(TypeError):
        PV.SeatOppElixir(object())


# ---------------------------------------------------------------------------
# permanence
# ---------------------------------------------------------------------------

def building(uid, card, hp, *, status=0, x_mt=9000, y_mt=12000, max_hp=1000, team=0):
    return entity_row(uid=uid, team=team, kind=1, card_id=card, x=x_mt * SUB, y=y_mt * SUB,
                      hp=hp, max_hp=max_hp, status_flags=status, unit_type=card)


def seen_at(tick, *ents, sent=()):
    return PV.public_view(state_dict(tick=tick, entities=list(ents)), 1, fields=FIELD, sent=sent)


def kept_row(view, uid):
    rows = [e for e in view["entities"] if e[FIELD["uid"]] == uid]
    assert len(rows) == 1, rows
    return rows[0]


def test_a_hidden_tesla_is_kept_with_its_hit_points_drained():
    keep = PV.Permanence(FIELD, team=1, hiders=PV.card_ids(PV.HIDER_NAMES, NAMES),
                         lifetimes={TESLA: 600})
    assert keep.hiders == {TESLA}
    for t in range(0, 101, 10):
        assert keep.apply(seen_at(t, building(32, TESLA, 1000)))["public"]["kept"] == []
    # It goes under ground: seen_by drops it, permanence keeps it, drained 1000 / 600 a tick.
    v = keep.apply(seen_at(110, building(32, TESLA, 1000, status=PV.STATUS_HIDDEN)))
    assert v["public"]["kept"] == [32]
    row = kept_row(v, 32)
    assert row[FIELD["hp"]] == 983 and row[FIELD["max_hp"]] == 1000
    assert row[FIELD["card_id"]] == TESLA and row[FIELD["status_flags"]] & PV.STATUS_HIDDEN
    assert (row[FIELD["x"]], row[FIELD["y"]]) == (9000 * SUB, 12000 * SUB)
    assert PV.is_public(v, 1, fields=FIELD) == []                # the kept row is listed
    v = keep.apply(seen_at(400, building(32, TESLA, 1000, status=PV.STATUS_HIDDEN)))
    assert kept_row(v, 32)[FIELD["hp"]] == 500
    # applying twice at a tick gives the same rows (the kept row is not a sighting)
    again = keep.apply(v)
    assert again["public"]["kept"] == [32] and kept_row(again, 32)[FIELD["hp"]] == 500
    v = keep.apply(seen_at(700))                                 # its lifetime is over
    assert v["public"]["kept"] == [] and keep.stats["expired"] == 1 and keep.memos == {}


def test_a_tesla_seen_dying_or_hit_by_our_spell_is_not_kept():
    dying = PV.Permanence(FIELD, team=1, hiders=[TESLA], lifetimes={TESLA: 600})
    for t, hp in ((0, 1000), (10, 1000), (20, 700), (30, 700), (40, 400), (50, 400), (60, 100)):
        dying.apply(seen_at(t, building(32, TESLA, hp)))
    assert dying.apply(seen_at(70))["public"]["kept"] == []     # a Hog hit takes 300
    assert dying.stats["dying"] == 1
    for aim, kept in (((9000, 12500), []), ((3000, 26000), [32])):
        keep = PV.Permanence(FIELD, team=1, hiders=[TESLA], lifetimes={TESLA: 600},
                             spell_cards=PV.spell_ids(CATALOGUE))
        for t in range(0, 51, 10):
            keep.apply(seen_at(t, building(32, TESLA, 1000)))
        sent = [(45, FIREBALL, aim[0] * SUB, aim[1] * SUB)]
        assert keep.apply(seen_at(60, sent=sent))["public"]["kept"] == kept
        assert keep.stats["spell"] == (0 if kept else 1)
    assert PV.spell_ids(CATALOGUE) == [FIREBALL, LOG, ARROWS, ZAP, MIRROR]


def test_a_building_that_does_not_hide_is_kept_only_through_a_missed_detection():
    keep = PV.Permanence(FIELD, team=1, hiders=[TESLA])
    cannon = building(40, CANNON, 500, max_hp=500, x_mt=3500)
    for t in (0, 10, 20):
        keep.apply(seen_at(t, cannon))
    v = keep.apply(seen_at(30))                                  # one detection missed
    assert v["public"]["kept"] == [40] and kept_row(v, 40)[FIELD["hp"]] == 492
    assert keep.apply(seen_at(40, cannon))["public"]["kept"] == []
    assert keep.apply(seen_at(50))["public"]["kept"] == [40]
    assert keep.apply(seen_at(60))["public"]["kept"] == [40]
    assert keep.apply(seen_at(80))["public"]["kept"] == []       # 40 ticks unseen: gone
    assert keep.stats["gone"] == 1


def test_a_reacquired_track_phantoms_reports_and_a_new_battle():
    keep = PV.Permanence(FIELD, team=1, hiders=[TESLA])
    keep.apply(seen_at(0, building(32, TESLA, 1000)))
    keep.apply(seen_at(10, building(32, TESLA, 1000)))
    assert keep.apply(seen_at(20))["public"]["kept"] == [32]
    # seen again under a new uid 300 mt away: the same Tesla
    assert keep.apply(seen_at(30, building(41, TESLA, 990, x_mt=9300)))["public"]["kept"] == []
    assert sorted(keep.memos) == [41] and keep.stats["merged"] == 1
    # a phantom (negative uid) is never remembered; our own buildings neither
    keep.apply(seen_at(40, building(41, TESLA, 990, x_mt=9300), building(-1, TESLA, 1000),
                       building(50, CANNON, 500, team=1, y_mt=25000)))
    assert sorted(keep.memos) == [41]
    keep.destroyed([41])
    assert keep.memos == {} and keep.stats["reported"] == 1
    assert keep.apply(seen_at(50, building(41, TESLA, 990, x_mt=9300)))["public"]["kept"] == []
    assert keep.memos == {}
    # the clock goes back: a new battle, nothing carried over
    keep.apply(seen_at(60, building(42, TESLA, 1000)))
    keep.apply(seen_at(70))
    assert sorted(keep.memos) == [42]
    assert keep.apply(seen_at(0))["public"]["kept"] == [] and keep.memos == {}


def test_lifetimes_from_a_card_table():
    table = {"cards": [{"name": "Tesla", "lifetime_ms": 25000}, {"name": "Cannon",
                                                                "lifetime_ms": 30000},
                       {"name": "Fireball", "lifetime_ms": None}, {"name": "Nowhere",
                                                                   "lifetime_ms": 9000}]}
    assert PV.lifetimes_from_card_table(table, NAMES) == {TESLA: 500, CANNON: 600}
    assert PV.card_ids(("Tesla", "Elixir Collector", "Mirror"), NAMES) == [TESLA, MIRROR]


# ---------------------------------------------------------------------------
# pinned to the modules it mirrors; import weight
# ---------------------------------------------------------------------------

def test_constants_match_public_state_and_rs_rows(monkeypatch):
    pytest.importorskip("numpy")
    from tools import public_state as PS
    monkeypatch.setitem(sys.modules, "royaleimitate", None)    # as the other rs_rows tests do
    RR = importlib.import_module("rs_rows")
    assert PV.ENTITY_DEFAULTS == PS.ENTITY_DEFAULTS
    assert PV.PUBLIC_FIELDS == PS.REQUIRED_FIELDS
    assert PV.DEPLOY_LOCKOUT_TICKS == PS.DEPLOY_LOCKOUT_TICKS
    assert PV.SPELL_FIELDS_FALLBACK == RR.SPELL_FIELDS_FALLBACK
    assert PV.UNSEEN_STATUS == RR.UNSEEN_STATUS
    assert (PV.STATUS_UNDERGROUND, PV.STATUS_INVISIBLE, PV.STATUS_HIDDEN) == (
        RR.STATUS_UNDERGROUND, RR.STATUS_INVISIBLE, RR.STATUS_HIDDEN)
    for status in (0, 1, 2, 4, 6, 8, 9, 64, 511):
        for team in (0, 1):
            e = entity_row(uid=1, team=0, status_flags=status)
            assert PV.seen_by(e, team, FIELD) == RR.seen_by(e, team, FIELD), (status, team)


def test_imports_neither_numpy_torch_royalesim_nor_the_memory_reader():
    code = ("import sys\n"
            "import tools.rs_public_view\n"
            "heavy = ('numpy', 'torch', 'royalesim', 'scripts.gamestate', 'scripts.vision',\n"
            "         'gamestate.memread')\n"
            "print(','.join(m for m in heavy if m in sys.modules))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""
