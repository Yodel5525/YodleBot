"""V05: tools/public_state.py, the public game state from screen perception.

The round trip: a RoyaleSim-like state (conftest's state_json layout) is rendered into the
VisionFrame a perfect perception would give (render(): one Track per living unit, the HUD
from the players), converted back with PublicStateBuilder, and tools/rs_rows.obs_row must
give the same row from both, for --live-compat rows (the first live model), unit-type rows
and the --units vision class rows (E16). Then the sanitiser (vision-02) and RSAgent.rebuilt
fed with its output, the import rule (scripts/gamestate is never loaded), and, marked
`engine`, the same round trip and rebuild on a real RoyaleSim battle.

The class table is the shipped data/vision/classes.json; catalogue, unit types and the E06
spawn table are small synthetic ones (ids below). Hit points at level 11 follow the class
table's level-1 values x 2.56 (vision_classes level_multiplier_percent[10] = 256), floored.
"""
from __future__ import annotations

import ast
import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.conftest import (ENTITY_FIELDS, FIELD, KING_HP, PRINCESS_HP, entity_row,
                                     make_fake_royalesim, player_dict, state_dict)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import geometry as G  # noqa: E402
from scripts.vision import types as VT  # noqa: E402
from tools import public_state as PS  # noqa: E402

import rs_spawn_table  # noqa: E402  (tools/ is on sys.path once public_state is imported)

CLASSES = PS.load_classes()
SUB = 18

# ---------------------------------------------------------------------------
# a small engine: catalogue, unit types, E06 spawn table
# ---------------------------------------------------------------------------

#: (name, elixir, card_kind, flying) in catalogue id order; Hog 2.6 is 0-7.
CARD_ROWS = (
    ("Cannon", 3, "BUILDING", False), ("Fireball", 4, "SPELL", False),
    ("HogRider", 4, "TROOP", False), ("IceGolemite", 2, "TROOP", False),
    ("IceSpirits", 1, "TROOP", False), ("Musketeer", 4, "TROOP", False),
    ("Skeletons", 1, "TROOP", False), ("Log", 2, "SPELL", False),
    ("Knight", 3, "TROOP", False), ("Archer", 3, "TROOP", False),
    ("Giant", 5, "TROOP", False), ("Arrows", 3, "SPELL", False),
    ("Valkyrie", 4, "TROOP", False), ("MiniPekka", 4, "TROOP", False),
    ("BabyDragon", 4, "TROOP", True), ("Zap", 2, "SPELL", False),
    ("Witch", 5, "TROOP", False), ("Goblins", 2, "TROOP", False),
    ("SkeletonArmy", 3, "TROOP", False),
)
#: catalogue_json rows: name, placement, elixir, count, radius, flying, hitpoints, footprint, kind.
CATALOGUE = [[n, 0, cost, 1, 0, fly, 0, 0, kind] for n, cost, kind, fly in CARD_ROWS]
CARDS = [row[0] for row in CATALOGUE]
CID = {n: i for i, n in enumerate(CARDS)}
UNITS = ["BabyDragon", "Cannon", "Giant", "Goblin", "Goblin_Stab", "HogRider", "IceGolemite",
         "IceSpirits", "KingTower", "Knight", "MiniPekka", "Musketeer", "MusketeerHero",
         "PrincessTower", "Skeleton", "Skeleton_EV1", "TowerPrincess", "Valkyrie", "Witch"]
UID = {n: i for i, n in enumerate(UNITS)}
HOG = list(range(8))
OPP = [CID[n] for n in ("Knight", "Archer", "Giant", "Arrows", "Valkyrie", "MiniPekka",
                        "BabyDragon", "Zap")]
#: card -> its own unit (E06 "primary"); the Goblins card puts down Goblin_Stab.
PRIMARY = {"Cannon": "Cannon", "HogRider": "HogRider", "IceGolemite": "IceGolemite",
           "IceSpirits": "IceSpirits", "Musketeer": "Musketeer", "Skeletons": "Skeleton",
           "Knight": "Knight", "Giant": "Giant", "Valkyrie": "Valkyrie", "MiniPekka": "MiniPekka",
           "BabyDragon": "BabyDragon", "Witch": "Witch", "Goblins": "Goblin_Stab",
           "SkeletonArmy": "Skeleton"}
#: unit_hitpoints(card, 11): [role, unit, hp] rows, the "own" row first.
HP11 = {"Cannon": [["own", "Cannon", 824]], "HogRider": [["own", "HogRider", 1697]],
        "IceGolemite": [["own", "IceGolemite", 1228]], "IceSpirits": [["own", "IceSpirits", 215]],
        "Musketeer": [["own", "Musketeer", 721]], "Skeletons": [["own", "Skeleton", 81]],
        "Knight": [["own", "Knight", 1766]], "Giant": [["own", "Giant", 3968]],
        "Valkyrie": [["own", "Valkyrie", 1907]], "MiniPekka": [["own", "MiniPekka", 1390]],
        "BabyDragon": [["own", "BabyDragon", 1152]],
        "Witch": [["own", "Witch", 839], ["spawn", "Skeleton", 81]],
        "Goblins": [["own", "Goblin_Stab", 202]], "SkeletonArmy": [["own", "Skeleton", 81]]}
SPAWN_TABLE = {
    "format": 1, "card_names": list(CARDS), "unit_types": list(UNITS),
    "primary": {str(CID[c]): UID[u] for c, u in PRIMARY.items()},
    "all": {str(CID[c]): [UID[u]] for c, u in PRIMARY.items()},
    "hp": {str(CID[c]): {"11": rows} for c, rows in HP11.items()},
}
#: royalesim.SPELL_FIELDS as scripts/vision/events.StateTracks documents them.
SPELL_FIELDS = ("team", "card_id", "motion", "x", "y", "aim_x", "aim_y", "delay_ticks",
                "travelled", "length", "hits", "ticks_flown")
#: The opponent memory both rows read (tokens = card id + 2).
MEM = {"seen": {0: {CID["Giant"] + 2, CID["Zap"] + 2}, 1: {CID["HogRider"] + 2}},
       "last": {0: [(CID["Giant"] + 2, 1200), (CID["Zap"] + 2, 1400)],
                1: [(CID["HogRider"] + 2, 1450)]}}


def unit(uid, team, name, card, x, y, hp, max_hp, deploy=0):
    """An engine entity: unit type `name` put down by `card` at (x, y) millitiles."""
    return entity_row(uid=uid, team=team, kind=1 if name == "Cannon" else 0, card_id=CID[card],
                      x=x * SUB, y=y * SUB, hp=hp, max_hp=max_hp, flying=name == "BabyDragon",
                      deploy_ticks=deploy, level=11, unit_type=UID[name])


def board(tick=1500):
    """Both sides mid-push. Skeletons 103 and 104 stand at the same distance from our king,
    so obs_row's sort keeps their order; a princess of the opponent's is down."""
    ents = [
        unit(101, 1, "HogRider", "HogRider", 3500, 14000, 1697, 1697),
        unit(102, 1, "Musketeer", "Musketeer", 6000, 25000, 400, 721),
        unit(103, 1, "Skeleton", "Skeletons", 8600, 21300, 81, 81, deploy=12),
        unit(104, 1, "Skeleton", "Skeletons", 9400, 21300, 81, 81, deploy=12),
        unit(105, 1, "Skeleton", "Skeletons", 9000, 21000, 50, 81, deploy=12),
        unit(106, 1, "Cannon", "Cannon", 9000, 20000, 600, 824),
        unit(107, 1, "IceGolemite", "IceGolemite", 14500, 17600, 1000, 1228),
        unit(201, 0, "Knight", "Knight", 3500, 18000, 1200, 1766),
        unit(202, 0, "BabyDragon", "BabyDragon", 14000, 16000, 1152, 1152, deploy=5),
        unit(203, 0, "Valkyrie", "Valkyrie", 4000, 20000, 1907, 1907, deploy=20),
        unit(204, 0, "Giant", "Giant", 5000, 10000, 3968, 3968),
    ]
    players = [player_dict(0, elixir_milli=6000, hand=[8, 9, 10, 11], next_card=12, crowns=0,
                           tower_hp=(KING_HP, 0, 2101), deck=list(OPP)),
               player_dict(1, elixir_milli=4300, hand=[7, 2, 6, 1], next_card=4, crowns=1,
                           tower_hp=(KING_HP, PRINCESS_HP, 2731), deck=list(HOG))]
    return state_dict(tick, players=players, entities=ents)


def odd_board(tick=1500):
    """Units whose engine type is not their class's plain unit: an alias (Goblin_Stab), an
    evolution, a hero form, and a Witch's skeleton (created by a card it is not the unit of)."""
    ents = [
        unit(301, 0, "Goblin_Stab", "Goblins", 9000, 12000, 150, 202),
        unit(302, 0, "Skeleton_EV1", "SkeletonArmy", 8000, 13000, 81, 81),
        unit(303, 0, "MusketeerHero", "Musketeer", 12000, 9000, 700, 721),
        unit(304, 0, "Skeleton", "Witch", 6000, 9000, 81, 81, deploy=3),
        unit(305, 0, "Witch", "Witch", 6000, 8000, 839, 839),
        unit(306, 1, "HogRider", "HogRider", 14500, 12000, 900, 1697),
    ]
    return state_dict(tick, players=board()["players"], entities=ents)


# ---------------------------------------------------------------------------
# the test renderer and the converter
# ---------------------------------------------------------------------------

def render(state, *, team=1, sub=SUB, unit_names=UNITS, F=FIELD):
    """The VisionFrame a perfect perception of `state` gives, as seat `team`, and the creating
    card of each track (what EventDeriver.track_cards reports when it is right). One Track per
    living non-tower entity: its class and form from the class table, world millitiles, the
    hp fraction, first seen deploy-time-minus-deploy_ticks ago (tracker.deploy_ticks_left
    inverted). The HUD: our hand, next card and elixir, the clock, towers by TOWER_KEYS
    (slot 1 = left, 2 = right), crowns. Returns (vf, cards, unresolved unit names)."""
    tick = int(state["tick"])
    tracks, cards, unresolved = [], {}, []
    for e in state["entities"]:
        if e[F["tower_slot"]] >= 0 or e[F["hp"]] <= 0:
            continue
        name = unit_names[e[F["unit_type"]]] if 0 <= e[F["unit_type"]] < len(unit_names) else None
        cid, form = CLASSES.resolve(name) if name else (None, 0)
        if cid is None:
            unresolved.append(name)
            continue
        c = CLASSES[cid]
        total = int(round(c.form(form).deploy_time_ms / 50))
        uid = int(e[F["uid"]])
        tracks.append(VT.Track(uid=uid, cls=c.name, team=int(e[F["team"]]), conf=1.0,
                               x_mt=e[F["x"]] / sub, y_mt=e[F["y"]] / sub,
                               hp_frac=e[F["hp"]] / max(1, e[F["max_hp"]]), form=form,
                               first_seen_tick=tick - (total - int(e[F["deploy_ticks"]])),
                               last_seen_tick=tick))
        cards[uid] = int(e[F["card_id"]])
    me, them = state["players"][team], state["players"][1 - team]
    towers = {f"{side}_{pos}": VT.TowerHp(p["tower_hp"][slot], p["tower_max_hp"][slot])
              for side, p in (("own", me), ("enemy", them))
              for slot, pos in enumerate(("king", "left", "right"))}
    hud = VT.Hud(hand=list(me["hand"]), next=me["next_card"], elixir_milli=me["elixir_milli"],
                 clock_tick=tick,
                 phase=VT.PHASE_OVERTIME if tick >= 3600 else VT.PHASE_REGULATION,
                 towers=towers, crowns=(me["crowns"], them["crowns"]))
    vf = VT.VisionFrame(t_capture=75.0, tick=tick, tracks=tracks, hud=hud,
                        screen=VT.SCREEN_BATTLE, frame_size=(747, 1328))
    return vf, cards, unresolved


def builder(**kw):
    args = dict(deck=HOG, catalogue=CATALOGUE, fields=FIELD, sub=SUB)
    args.update(kw)
    return PS.PublicStateBuilder(CLASSES, SPAWN_TABLE, **args)


def convert(vf, cards=None, *, opp=OPP, **kw):
    return builder(**kw).state(vf, opp_deck=opp, track_cards=cards)


def track(uid, cls, team, x, y, *, hp=None, form=0, first=1480, last=1500):
    return VT.Track(uid=uid, cls=cls, team=team, conf=0.9, x_mt=float(x), y_mt=float(y),
                    hp_frac=hp, form=form, first_seen_tick=first, last_seen_tick=last)


def frame(tracks, tick=1500, **hud):
    return VT.VisionFrame(t_capture=1.0, tick=tick, tracks=list(tracks), hud=VT.Hud(**hud),
                          screen=VT.SCREEN_BATTLE)


@pytest.fixture
def rr(monkeypatch):
    """tools/rs_rows as the tools import it, with conftest's FIELD as its field index and the
    catalogue flags of CATALOGUE (what _setup reads from royalesim), no royaleimitate."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    monkeypatch.setattr(rr, "F", dict(FIELD))
    monkeypatch.setattr(rr, "LIVE", False)
    monkeypatch.setattr(rr, "CARD_FLY", {i: bool(row[5]) for i, row in enumerate(CATALOGUE)})
    monkeypatch.setattr(rr, "CARD_BUILDING", {i: row[8] == "BUILDING"
                                              for i, row in enumerate(CATALOGUE)})
    monkeypatch.setattr(rr, "VC", CLASSES.lookup(UNITS))
    return rr


def same_row(a, b):
    assert list(a) == list(b)
    for key in a:
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key


def vision_row(rr, state, mem, tick, *, sub=SUB):
    """Our --units vision row (E16): class tokens, class-table flags and the unit_form column,
    through rr.VC (the class lookup of the engine's unit types, set by the fixture)."""
    row = rr.obs_row(state, 1, sub, mem, tick, live=False, units="vision")
    assert "unit_form" in row
    return row


# ---------------------------------------------------------------------------
# the round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("perfect_cards", [True, False], ids=["track_cards", "class_table"])
def test_live_compat_row_survives_the_round_trip(rr, perfect_cards):
    st = board()
    vf, cards, unresolved = render(st)
    assert not unresolved and len(vf.tracks) == 11
    # Without track_cards the class table names the card: our deck for ours, the opponent's
    # known cards for theirs; on this board that is every unit's own card.
    conv = convert(vf, cards if perfect_cards else None)
    assert conv["tick"] == 1500
    same_row(rr.obs_row(st, 1, SUB, MEM, 1500, live=True),
             rr.obs_row(conv, 1, SUB, MEM, conv["tick"], live=True))


def test_unit_type_row_survives_the_round_trip(rr):
    st = board()
    vf, cards, _ = render(st)
    conv = convert(vf, cards)
    row = rr.obs_row(conv, 1, SUB, MEM, 1500, live=False)
    same_row(rr.obs_row(st, 1, SUB, MEM, 1500, live=False), row)
    # the deploying Skeletons (12 ticks left) and the Cannon's building flag came through
    assert sorted(row["unit_feats"][:11, 6].tolist()) == [0.0] * 6 + [1.0] * 5
    assert row["unit_feats"][:11, 5].tolist().count(1.0) == 1


def test_vision_class_row_survives_the_round_trip(rr):
    st = board()
    vf, cards, _ = render(st)
    conv = convert(vf, cards)
    same_row(vision_row(rr, st, MEM, 1500), vision_row(rr, conv, MEM, 1500))


def test_forms_aliases_and_children_survive_the_round_trip(rr):
    st = odd_board()
    vf, cards, unresolved = render(st)
    assert not unresolved
    assert [(t.cls, t.form) for t in vf.tracks] == [
        ("Goblin", 0), ("Skeleton", 1), ("Musketeer", 2), ("Skeleton", 0), ("Witch", 0),
        ("HogRider", 0)]
    conv = convert(vf, cards, opp=[CID[c] for c in ("Goblins", "SkeletonArmy", "Musketeer",
                                                     "Witch")])
    F = FIELD
    # Goblins' goblin is its primary unit Goblin_Stab, not the class's Goblin; the forms keep
    # their own unit types; the Witch's skeleton is a Skeleton the Witch made.
    assert [e[F["unit_type"]] for e in conv["entities"]] == [
        UID["Goblin_Stab"], UID["Skeleton_EV1"], UID["MusketeerHero"], UID["Skeleton"],
        UID["Witch"], UID["HogRider"]]
    assert [e[F["card_id"]] for e in conv["entities"]] == [
        CID["Goblins"], CID["SkeletonArmy"], CID["Musketeer"], CID["Witch"], CID["Witch"],
        CID["HogRider"]]
    # max hp: Goblins' row of Goblin_Stab, the class table's evo and hero forms (32 and 282 at
    # level 1, x 2.56), the Witch's spawn row of Skeleton
    assert [e[F["max_hp"]] for e in conv["entities"]] == [202, 81, 721, 81, 839, 1697]
    for live in (True, False):
        same_row(rr.obs_row(st, 1, SUB, MEM, 1500, live=live),
                 rr.obs_row(conv, 1, SUB, MEM, 1500, live=live))
    same_row(vision_row(rr, st, MEM, 1500), vision_row(rr, conv, MEM, 1500))


def test_a_class_alias_without_its_card_reads_as_the_class_unit(rr):
    """The engine's Goblin_Stab put down by a card perception reports as unknown (-1): the
    entity is the class's own unit, Goblin, with card_id -1. Unit-type rows differ in that
    token alone, live rows in the card token alone (1, "unknown"); vision rows agree."""
    st = odd_board()
    vf, cards, _ = render(st)
    cards[301] = -1
    conv = convert(vf, cards)
    goblin = conv["entities"][0]
    assert (goblin[FIELD["unit_type"]], goblin[FIELD["card_id"]]) == (UID["Goblin"], -1)
    assert goblin[FIELD["max_hp"]] == 202                  # Goblins (primary card): Goblin_Stab
    for live, want in ((False, (UID["Goblin_Stab"] + 2, UID["Goblin"] + 2)),
                       (True, (CID["Goblins"] + 2, 1))):
        a = rr.obs_row(st, 1, SUB, MEM, 1500, live=live)
        b = rr.obs_row(conv, 1, SUB, MEM, 1500, live=live)
        diff = np.nonzero(a["unit_ids"] != b["unit_ids"])[0].tolist()
        assert len(diff) == 1, live
        assert (int(a["unit_ids"][diff[0]]), int(b["unit_ids"][diff[0]])) == want
        assert np.array_equal(a["unit_feats"], b["unit_feats"]), live
    same_row(vision_row(rr, st, MEM, 1500), vision_row(rr, conv, MEM, 1500))


def test_entities_are_built_by_field_name(rr, monkeypatch):
    st = board()
    vf, cards, _ = render(st)
    conv = convert(vf, cards)
    rev = {name: len(ENTITY_FIELDS) - 1 - i for i, name in enumerate(ENTITY_FIELDS)}
    conv_rev = convert(vf, cards, fields=rev)
    assert conv_rev["entities"] == [[e[FIELD[name]] for name in reversed(ENTITY_FIELDS)]
                                    for e in conv["entities"]]
    cannon = next(e for e in conv["entities"] if e[FIELD["uid"]] == 106)
    want = dict(PS.ENTITY_DEFAULTS, uid=106, team=1, kind=1, card_id=CID["Cannon"], tower_slot=-1,
                x=9000 * SUB, y=20000 * SUB, hp=600, max_hp=824, flying=False, deploy_ticks=0,
                status_flags=0, level=11, unit_type=UID["Cannon"])
    assert {name: cannon[FIELD[name]] for name in ENTITY_FIELDS} == want
    a, b = conv["entities"][0][FIELD["facing"]], conv["entities"][1][FIELD["facing"]]
    assert a == b == [0, 0] and a is not b                    # list defaults are per row
    expected = rr.obs_row(st, 1, SUB, MEM, 1500, live=True)
    monkeypatch.setattr(rr, "F", rev)
    same_row(expected, rr.obs_row(conv_rev, 1, SUB, MEM, 1500, live=True))


def test_players_clock_and_the_rest_of_the_state():
    vf, cards, _ = render(board())
    conv = convert(vf, cards)
    assert conv["players"][1] == {
        "team": 1, "elixir_milli": 4300, "hand": [7, 2, 6, 1], "hand_costs": [2, 4, 1, 4],
        "next_card": 4, "deck": HOG, "crowns": 1, "tower_hp": [4824, 3052, 2731],
        "tower_max_hp": [4824, 3052, 3052], "king_active": False, "pending_cost": 0}
    assert conv["players"][0] == {
        "team": 0, "elixir_milli": 0, "hand": [], "hand_costs": [], "next_card": -1,
        "deck": OPP, "crowns": 0, "tower_hp": [4824, 0, 2101],
        "tower_max_hp": [4824, 3052, 3052], "king_active": True, "pending_cost": 0}
    assert {k: v for k, v in conv.items() if k not in ("players", "entities")} == {
        "tick": 1500, "tick_ms": 75000, "regular_ticks": 3600, "overtime_ticks": 2400,
        "elixir_rate": 1, "overtime": False, "spells": [], "projectiles": [],
        "game_over": False, "winner": -1,
        "perception": {"tracks": 11, "units": 11, "spells": 0, "dropped": {}, "notes": []}}
    json.dumps(conv)                                            # plain JSON values only
    later = builder().state(frame([], tick=4900, phase=VT.PHASE_OVERTIME), pending_cost=3)
    assert (later["elixir_rate"], later["overtime"], later["players"][1]["pending_cost"]) == (3, True, 3)
    # 2x from tick 2400 (1:00 left), 3x from 4800 (overtime's last minute): rs_elixir's law
    assert [builder().state(frame([], tick=t))["elixir_rate"] for t in (2399, 2400, 4799, 4800)] == [
        1, 2, 2, 3]
    assert builder().state(frame([], tick=3600))["overtime"]           # past regulation
    ended = builder().state(VT.VisionFrame(t_capture=0.0, tick=2000, screen=VT.SCREEN_END))
    assert ended["game_over"] and ended["elixir_rate"] == 1


def test_tower_keys_are_screen_sides_and_unread_towers_count_as_full(rr):
    towers = {"own_left": VT.TowerHp(1000, 3052), "own_right": VT.TowerHp(2000, None),
              "enemy_king": VT.TowerHp(None, 4824), "enemy_right": VT.TowerHp(3200, 3052)}
    conv = builder().state(frame([], towers=towers))
    me, them = conv["players"][1], conv["players"][0]
    assert (me["tower_hp"], me["tower_max_hp"]) == ([4824, 1000, 2000], [4824, 3052, 3052])
    # a reading above the maximum raises it
    assert (them["tower_hp"], them["tower_max_hp"]) == ([4824, 3052, 3200], [4824, 3052, 3200])
    assert conv["perception"]["notes"] == ["own_king unread: full", "enemy_king unread: full",
                                           "enemy_left unread: full", "elixir unread: 0"]
    # team 1's own frame mirrors x: its own left is the world's (screen's) right, slot 2
    assert np.allclose(rr.towers_vec(conv, 1)[:3], [2000 / 3052, 1000 / 3052, 1.0])


def test_the_creating_card_and_its_hit_points():
    tr = [track(1, "Skeleton", 1, 9000, 21000), track(2, "Skeleton", 0, 9000, 9000)]
    vf = frame(tr)
    F = FIELD

    def cards_and_hp(**kw):
        conv = builder().state(vf, **kw)
        return [(e[F["card_id"]], e[F["max_hp"]]) for e in conv["entities"]]

    skel, witch, army = CID["Skeletons"], CID["Witch"], CID["SkeletonArmy"]
    # ours: our deck's Skeletons; theirs: the first creator of a Skeleton in their known deck
    assert cards_and_hp(opp_deck=["Witch", "Knight"]) == [(skel, 81), (witch, 81)]
    assert cards_and_hp(opp_deck=[]) == [(skel, 81), (skel, 81)]          # the primary card
    assert cards_and_hp(opp_deck=["Witch"], track_cards={2: army}) == [(skel, 81), (army, 81)]


def test_max_hp_fallbacks(monkeypatch):
    b = builder()
    cls = CLASSES.class_id
    assert b.max_hp(CID["Musketeer"], cls("Musketeer"), 2, 11) == 721     # 282 * 256 // 100
    assert b.max_hp(CID["Goblins"], cls("Goblin"), 0, 11) == 202          # a row of its class
    assert b.max_hp(CID["Witch"], cls("Skeleton"), 0, 11) == 81           # the Witch's spawn row
    assert b.max_hp(-1, cls("HogRider"), 0, 11) == 1697                   # the primary card's
    assert b.max_hp(CID["HogRider"], cls("HogRider"), 0, 12) == 1863      # 663 * 281 // 100
    monkeypatch.setattr(CLASSES, "hitpoints", lambda *a, **k: None)
    bare = PS.PublicStateBuilder(CLASSES, dict(SPAWN_TABLE, hp={}), deck=HOG, fields=FIELD)
    conv = bare.state(frame([track(1, "HogRider", 1, 3500, 20000, hp=0.5)]))
    e = conv["entities"][0]
    assert (e[FIELD["hp"]], e[FIELD["max_hp"]]) == (500, PS.UNKNOWN_MAX_HP)
    assert conv["perception"]["notes"][0] == "no hit points for HogRider: max_hp 1000"
    assert conv["players"][1]["hand_costs"] == [0, 0, 0, 0]     # an empty hand costs nothing


@pytest.mark.parametrize("frac, hp", [(None, 1697), (0.25, 424), (0.0, 1), (-0.5, 1),
                                      (1.7, 1697)])
def test_hit_points_from_the_bar(frac, hp):
    conv = builder().state(frame([track(1, "HogRider", 1, 3500, 20000, hp=frac)]))
    assert conv["entities"][0][FIELD["hp"]] == hp


def test_an_undecided_team_goes_by_the_half_or_is_dropped():
    tr = [track(1, "Knight", VT.UNKNOWN_TEAM, 3500, 20000),
          track(2, "Knight", VT.UNKNOWN_TEAM, 3500, 10000)]
    conv = builder().state(frame(tr), opp_deck=["Knight"])
    assert [e[FIELD["team"]] for e in conv["entities"]] == [1, 0]
    dropped = builder(unknown_team="drop").state(frame(tr))
    assert dropped["entities"] == [] and dropped["perception"]["dropped"] == {"team": 2}


def test_spells_towers_and_ui_are_not_units():
    tr = [track(401, "Fireball", 0, 9000, 8000, first=1490),
          track(402, "the-log", 1, 3500, 12000, first=1495),
          track(403, "PrincessTower", 0, 3500, 6500), track(404, "TowerPrincess", 1, 3500, 25500),
          track(405, "king-tower", 0, 9000, 3000), track(406, "bar", 0, 9000, 9000),
          track(407, "zzz-not-a-class", 0, 9000, 9000)]
    conv = builder().state(frame(tr))
    assert conv["entities"] == [] and conv["spells"] == [] and conv["projectiles"] == []
    assert conv["perception"]["dropped"] == {"tower": 3, "ui": 1, "unknown": 1}
    assert conv["perception"]["spells"] == 2
    with_rows = builder(spell_fields=SPELL_FIELDS).state(frame(tr))
    assert with_rows["entities"] == []
    assert with_rows["spells"] == [
        [0, CID["Fireball"], 0, 9000 * SUB, 8000 * SUB, 9000 * SUB, 8000 * SUB, 0, 0, 0, 0, 10],
        [1, CID["Log"], 0, 3500 * SUB, 12000 * SUB, 3500 * SUB, 12000 * SUB, 0, 0, 0, 0, 5]]


def test_the_tick_and_deploy_timers():
    valk = track(1, "Valkyrie", 0, 4000, 20000, first=890, last=890)
    assert builder().state(frame([valk], tick=900))["entities"][0][FIELD["deploy_ticks"]] == 10
    clock = builder().state(frame([valk], tick=None, clock_tick=700))
    assert clock["tick"] == 700 and clock["perception"]["notes"][0] == "tick from the HUD clock"
    # first seen 190 ticks after the state's tick: still within the 20-tick deploy time
    assert clock["entities"][0][FIELD["deploy_ticks"]] == 20
    blind = builder().state(frame([valk, track(2, "Knight", 0, 0, 0, last=905)], tick=None))
    assert blind["tick"] == 905
    assert blind["perception"]["notes"][0] == "tick unread: the newest track sighting"
    given = builder().state(frame([valk], tick=900), tick=880)
    assert given["tick"] == 880 and given["entities"][0][FIELD["deploy_ticks"]] == 20


def test_inputs_are_checked():
    by_name = builder(deck=["Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits",
                            "Musketeer", "Skeletons", "Log"])
    assert by_name.deck == HOG
    with pytest.raises(ValueError, match="different cards"):
        builder(catalogue=CATALOGUE[::-1])
    with pytest.raises(ValueError, match="unit_type"):
        builder(fields=[n for n in ENTITY_FIELDS if n != "unit_type"])
    with pytest.raises(ValueError, match="unknown card name 'Mortar'"):
        builder(deck=["Mortar"])
    with pytest.raises(ValueError, match="unknown card name"):
        builder().state(frame([]), opp_deck=["NotACard"])
    for bad in (dict(own_team=2), dict(unknown_team="guess"), dict(levels=0),
                dict(levels=(11, 12, 13)), dict(sub=0)):
        with pytest.raises(ValueError):
            builder(**bad)
    assert PS.team_levels({1: 13}) == (11, 13) and PS.team_levels([9, 10]) == (9, 10)


def test_the_one_call_form_matches_the_builder_and_leaves_the_frame_alone():
    vf, cards, _ = render(board())
    before = vf.to_dict()
    one = PS.rs_state_from_vision(vf, HOG, CLASSES, SPAWN_TABLE, 11, catalogue=CATALOGUE,
                                  fields=FIELD, opp_deck=OPP, track_cards=cards)
    assert one == convert(vf, cards)
    assert vf.to_dict() == before


def test_own_team_0_fills_the_other_player():
    """An offline run on engine seat 0: the HUD's "own" side is team 0."""
    tr = [track(1, "HogRider", 0, 3500, 12000)]
    conv = builder(own_team=0).state(frame(tr, hand=[2, 5, 6, 7], elixir_milli=5000))
    assert conv["players"][0]["hand"] == [2, 5, 6, 7] and conv["players"][0]["deck"] == HOG
    assert conv["players"][1]["hand"] == [] and conv["entities"][0][FIELD["card_id"]] == 2


# ---------------------------------------------------------------------------
# the rebuild sanitiser
# ---------------------------------------------------------------------------

def awkward(tick=1500):
    """A public state as perception may leave it: units in the river, off the board, above
    their card's hit points, dead, of no team, and a spell read as a unit."""
    ents = [
        unit(1, 1, "HogRider", "HogRider", 9000, 15200, 1697, 1697),       # river, north part
        unit(2, 1, "HogRider", "HogRider", 4400, 16000, 1697, 1697),       # river by a bridge
        unit(3, 1, "Musketeer", "Musketeer", 3500, 16000, 500, 721),       # on the left bridge
        unit(4, 0, "BabyDragon", "BabyDragon", 9000, 16000, 1152, 1152),   # flying over water
        unit(5, 0, "Knight", "Knight", -300, 33000, 1766, 1766),           # off the board
        unit(6, 0, "MusketeerHero", "Musketeer", 12000, 9000, 900, 900),   # above the card's hp
        unit(7, 1, "Skeleton", "Skeletons", 6000, 20000, 0, 81),           # dead
        entity_row(uid=8, team=0, card_id=CID["Fireball"], unit_type=-1, x=9000 * SUB,
                   y=20000 * SUB, hp=572),                                  # a spell
        unit(9, 0, "Skeleton", "Witch", 6000, 9000, 81, 81),               # spawns as Skeletons
        unit(10, 1, "Skeleton", "Skeletons", 9000, 16000, 81, 81),         # mid-river: a tie
        unit(11, 1, "Cannon", "Cannon", 9000, 14800, 824, 824),            # land touching water
        unit(12, -1, "Knight", "Knight", 9000, 9000, 1766, 1766),          # no team
    ]
    players = [player_dict(0, elixir_milli=-5, hand=[8, 9, 10, 11], next_card=12,
                           tower_hp=(5000, 3052, 3100), deck=[16, 17, 16, -1]),
               player_dict(1, elixir_milli=12000, hand=[7, 2, 6, 1], next_card=4,
                           deck=list(HOG))]
    return state_dict(tick, players=players, entities=ents)


def test_sanitize_for_rebuild_cleans_every_spawn():
    st = awkward()
    before = copy.deepcopy(st)
    san = PS.sanitize_for_rebuild(st, SPAWN_TABLE, fields=FIELD, sub=SUB)
    assert st == before                                          # the input is not changed
    F = FIELD
    units = [e for e in san["entities"] if e[F["tower_slot"]] < 0]
    towers = [e for e in san["entities"] if e[F["tower_slot"]] >= 0]
    assert towers == [e for e in before["entities"] if e[F["tower_slot"]] >= 0]
    got = [(e[F["uid"]], e[F["x"]] // SUB, e[F["y"]] // SUB, e[F["hp"]]) for e in units]
    assert got == [(1, 9000, 14500, 1697),        # the nearest bank
                   (2, 4000, 16000, 1697),        # onto the bridge, 500 inside its side
                   (3, 3500, 16000, 500),
                   (4, 9000, 16000, 1152),        # flying: left over the water
                   (5, 500, 31500, 1766),         # into the arena, 500 inside the edge
                   (6, 12000, 9000, 721),         # capped to Musketeer's own 721
                   (9, 6000, 9000, 81),
                   (10, 9000, 17500, 81),         # a tie between the banks: its own (team 1)
                   (11, 9000, 14500, 824)]
    assert all(e[F["x"]] % SUB == 0 and e[F["y"]] % SUB == 0 for e in units)
    assert san["sanitized"] == {"dropped": {"dead": 1, "no spawn card": 1, "team": 1},
                                "into_arena": 1, "off_water": 4, "hp_capped": 1,
                                "padded": {0: 6}}
    # the opponent's deck: its 2 known cards, then Hog 2.6 (rs_spawn_table.FILLER_NAMES)
    assert san["players"][0]["deck"] == [16, 17, CID["HogRider"], CID["Musketeer"],
                                         CID["Cannon"], CID["IceGolemite"], CID["IceSpirits"],
                                         CID["Skeletons"]]
    assert san["players"][1]["deck"] == HOG
    assert [p["elixir_milli"] for p in san["players"]] == [0, 10_000]
    assert san["players"][0]["tower_hp"] == [4824, 3052, 3052]
    for e in units:
        if not e[F["flying"]]:
            assert not G.on_water(e[F["x"]] / SUB, e[F["y"]] / SUB)
        assert G.in_arena(e[F["x"]] / SUB, e[F["y"]] / SUB)


def test_sanitize_odds_and_ends():
    st = state_dict(-3, entities=[unit(1, 1, "HogRider", "HogRider", 3500, 20000, 1697, 1697)],
                    players=[player_dict(0, deck=list(range(10))), player_dict(1, deck=HOG)])
    san = PS.sanitize_for_rebuild(st, SPAWN_TABLE, fields=FIELD, filler=["Zap", 3])
    assert san["tick"] == 0
    assert san["players"][0]["deck"] == list(range(8)) and san["sanitized"]["padded"] == {}
    short = state_dict(10, players=[player_dict(0, deck=[]), player_dict(1, deck=[2])])
    san = PS.sanitize_for_rebuild(short, SPAWN_TABLE, fields=FIELD, filler=["Zap", 3, "Nope"])
    # the given filler first, then the spawnable cards by id
    assert san["players"][0]["deck"] == [CID["Zap"], 3, 0, 2, 4, 5, 6, 8]
    assert san["players"][1]["deck"] == [2, CID["Zap"], 3, 0, 4, 5, 6, 8]
    assert san["sanitized"]["padded"] == {0: 8, 1: 7}
    # reset refuses a king at 0, so a misread king goes to 1; a princess at 0 stays destroyed
    down = state_dict(10, towers=False,
                      players=[player_dict(0, deck=HOG, tower_hp=(0, 0, 3052)),
                               player_dict(1, deck=HOG, tower_hp=(-7, 3052, 9999))])
    san = PS.sanitize_for_rebuild(down, SPAWN_TABLE, fields=FIELD)
    assert [p["tower_hp"] for p in san["players"]] == [[1, 0, 3052], [1, 3052, 3052]]


@pytest.mark.parametrize("point, own_high, want", [
    ((9000, 16000), True, (9000, 17500)),       # a tie: the unit's own bank
    ((9000, 16000), False, (9000, 14500)),
    ((3500, 16000), True, (3500, 16000)),       # on a bridge
    ((2700, 14700), True, (2700, 14500)),       # land, 300 from the water by the bridge
    ((9000, 14400), True, (9000, 14400)),       # clear of the water
    ((9000, 14500), True, (9000, 14500)),       # exactly the margin away
    ((15400, 16000), True, (15000, 16000)),     # past the right bridge's side
])
def test_off_water(point, own_high, want):
    assert PS.off_water(*point, own_high=own_high) == want
    assert not G.on_water(*want)


def test_off_water_margin_and_into_arena():
    assert PS.off_water(9000, 16000, 0) == (9000, 17001)        # at least 1 mt clear
    assert PS.into_arena(-300, 33000) == (500, 31500)
    assert PS.into_arena(9000, 16000) == (9000, 16000)
    assert PS.into_arena(18000, 0, 0) == (18000, 0)


@pytest.mark.parametrize("card, unit, want", [
    ("HogRider", "HogRider", "HogRider"),        # the card's own unit
    ("Musketeer", "MusketeerHero", "Musketeer"),  # a form of it (name starts with the unit's)
    ("Skeletons", "Skeleton_EV1", "Skeletons"),
    ("SkeletonArmy", "Skeleton", "SkeletonArmy"),
    ("Witch", "Skeleton", "Skeletons"),          # a child: the lowest card of that unit
    ("Goblins", "Goblin_Stab", "Goblins"),
    ("Goblins", "Goblin", None),                 # no card puts a plain Goblin down here
    ("Fireball", None, None),                    # unit_type -1
])
def test_rebuild_card_follows_rebuilt(card, unit, want):
    prim = rs_spawn_table.primary_of(SPAWN_TABLE)
    got = PS.rebuild_card(CID[card], -1 if unit is None else UID[unit], prim,
                          rs_spawn_table.by_unit(prim), UNITS)
    assert got == (None if want is None else CID[want])


def test_rebuilt_takes_the_sanitized_state(rr, monkeypatch):
    """tools/rs_eval.RSAgent.rebuilt on the sanitised state, with a fake royalesim: the reset
    gets one spawn per kept unit, every card a troop or building, ground units off the water,
    hp within the card's, both decks of 8."""
    ev = importlib.import_module("rs_eval")
    prim = rs_spawn_table.primary_of(SPAWN_TABLE)
    monkeypatch.setattr(ev, "_SPAWN", {"prim": prim, "by_unit": rs_spawn_table.by_unit(prim)})
    monkeypatch.setattr(ev, "_UNITS", list(UNITS))
    fake = make_fake_royalesim()
    san = PS.sanitize_for_rebuild(awkward(), SPAWN_TABLE, fields=FIELD, sub=SUB)
    agent = SimpleNamespace(engine=ev.rs_engine.EngineConfig.legacy())
    battle = ev.RSAgent.rebuilt(agent, san, 1, fake)
    assert battle is not None, getattr(agent, "rebuild_error", None)
    kw = battle.reset_kwargs
    F = FIELD
    units = [e for e in san["entities"] if e[F["tower_slot"]] < 0]
    want_cards = ["HogRider", "HogRider", "Musketeer", "BabyDragon", "Knight", "Musketeer",
                  "Skeletons", "Skeletons", "Cannon"]
    assert kw["spawns"] == [(e[F["team"]], CID[c], e[F["x"]], e[F["y"]], e[F["hp"]])
                            for e, c in zip(units, want_cards)]
    for team, card, x, y, hp in kw["spawns"]:
        assert CATALOGUE[card][8] in ("TROOP", "BUILDING")
        assert hp <= PS.card_hitpoints(SPAWN_TABLE["hp"], card, 11)
        assert G.in_arena(x / SUB, y / SUB)
        assert CATALOGUE[card][5] or not G.on_water(x / SUB, y / SUB)
    assert [len(d) for d in kw["decks"]] == [8, 8]
    assert kw["decks"][1] == [7, 2, 6, 1, 4, 0, 3, 5]        # hand, next, then the rest
    assert kw["start_tick"] == 1500 and kw["elixir_milli"] == [0, 10_000]


# ---------------------------------------------------------------------------
# the memory reader is never loaded
# ---------------------------------------------------------------------------

FORBIDDEN = ("gamestate", "memread", "cr_bot", "live_bridge", "live_play", "hcr_reader")


def test_public_state_imports_no_memory_reader():
    tree = ast.parse((TOOLS / "public_state.py").read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            names += [f"{node.module}.{a.name}" for a in node.names]
    assert names and not [n for n in names if any(bad in n for bad in FORBIDDEN)]
    assert "scripts.vision" in names


def test_using_public_state_never_loads_scripts_gamestate():
    """A fresh interpreter imports tools.public_state, builds a state and sanitises it; no
    module of scripts/gamestate (or the native reader) is loaded."""
    table = {"card_names": ["HogRider"], "unit_types": ["HogRider"], "primary": {"0": 0},
             "hp": {"0": {"11": [["own", "HogRider", 1697]]}}}
    code = "\n".join([
        "import json, sys",
        f"sys.path.insert(0, {str(ROOT)!r})",
        "from tools import public_state as PS",
        "from scripts.vision import types as VT",
        f"table = json.loads({json.dumps(table)!r})",
        f"fields = {list(ENTITY_FIELDS)!r}",
        "vf = VT.VisionFrame(t_capture=0.0, tick=100,",
        "                    tracks=[VT.Track(1, 'HogRider', 1, 1.0, 3500.0, 20000.0)])",
        "st = PS.rs_state_from_vision(vf, [0], PS.load_classes(), table, fields=fields)",
        "PS.sanitize_for_rebuild(st, table, fields=fields)",
        f"bad = sorted(m for m in sys.modules if any(b in m for b in {FORBIDDEN!r}))",
        "print(json.dumps({'bad': bad, 'units': len(st['entities'])}))",
    ])
    res = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                         text=True, timeout=180)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout.strip().splitlines()[-1]) == {"bad": [], "units": 1}


# ---------------------------------------------------------------------------
# the real engine (marked `engine`; run on the PC)
# ---------------------------------------------------------------------------

ENGINE_HOG = ("Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer",
              "Skeletons", "Log")
ENGINE_OPP = ("Knight", "Archer", "Giant", "Arrows", "Valkyrie", "MiniPekka", "BabyDragon", "Zap")
EMPTY_MEM = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}


@pytest.fixture(scope="module")
def engine_world():
    """One RoyaleSim battle under data/engine.json (level 11): both decks' units spawned and
    stepped 25 ticks, its state, unit types, catalogue and an E06 table for those cards."""
    royalesim = pytest.importorskip("royalesim")
    import rs_engine
    cfg = rs_engine.active().undelayed()
    sub = royalesim.SUBTILE_PER_MILLITILE
    battle = rs_engine.make_battle(cfg, royalesim)
    cat = json.loads(battle.catalogue_json())
    ids = {row[0]: i for i, row in enumerate(cat)}
    missing = [n for n in ENGINE_HOG + ENGINE_OPP if n not in ids]
    if missing:
        pytest.skip(f"cards missing from this engine's catalogue: {missing}")

    def put(team, name, x, y):
        return (team, ids[name], x * sub, y * sub, -1)

    spawns = [put(1, "HogRider", 3500, 20000), put(1, "Musketeer", 6000, 26000),
              put(1, "Skeletons", 9000, 23000), put(1, "Cannon", 9000, 20500),
              put(1, "IceGolemite", 14500, 21000), put(0, "Knight", 3500, 12000),
              put(0, "BabyDragon", 14000, 10000), put(0, "Valkyrie", 5000, 8000)]
    battle.reset(seed=0, decks=[[ids[n] for n in ENGINE_OPP], [ids[n] for n in ENGINE_HOG]],
                 shuffle=0, start_tick=600, elixir_milli=[5000, 5000], tower_hp=None,
                 spawns=spawns)
    battle.step([], 25)
    table = rs_spawn_table.build(cfg, royalesim, cards=sorted({s[1] for s in spawns}))
    return SimpleNamespace(rs=royalesim, cfg=cfg, sub=sub, cat=cat, ids=ids,
                           state=json.loads(battle.state_json()),
                           units=json.loads(battle.unit_types_json()), table=table,
                           hog=[ids[n] for n in ENGINE_HOG], opp=[ids[n] for n in ENGINE_OPP])


@pytest.fixture
def engine(engine_world, monkeypatch):
    """engine_world plus tools/rs_rows set up on it by _setup (F, catalogue flags)."""
    monkeypatch.syspath_prepend(str(TOOLS))
    rr = importlib.import_module("rs_rows")
    for name, value in (("F", None), ("CARD_FLY", {}), ("CARD_BUILDING", {})):
        monkeypatch.setattr(rr, name, value)
    monkeypatch.setattr(rr, "VC", CLASSES.lookup(engine_world.units))
    rr._setup(engine_world.cfg)
    return SimpleNamespace(**vars(engine_world), rr=rr, F=rr.F)


def engine_round_trip(e):
    st = e.state
    odd = []
    for ent in st["entities"]:
        if ent[e.F["tower_slot"]] < 0 and ent[e.F["hp"]] > 0:
            ut = ent[e.F["unit_type"]]
            name = e.units[ut] if 0 <= ut < len(e.units) else None
            cid, _ = CLASSES.resolve(name) if name else (None, 0)
            if cid is None or CLASSES[cid].kind == "tower":
                odd.append(name)
    assert not odd, f"entities obs_row reads as units that a screen does not show as units: {odd}"
    vf, cards, _ = render(st, sub=e.sub, unit_names=e.units, F=e.F)
    conv = PS.PublicStateBuilder(CLASSES, e.table, deck=e.hog, catalogue=e.cat, fields=e.F,
                                 sub=e.sub).state(vf, opp_deck=e.opp, track_cards=cards)
    return st, conv


@pytest.mark.engine
def test_engine_state_round_trip_live_compat(engine):
    st, conv = engine_round_trip(engine)
    assert len(conv["entities"]) >= 8
    rr, sub, tick = engine.rr, engine.sub, st["tick"]
    same_row(rr.obs_row(st, 1, sub, EMPTY_MEM, tick, live=True),
             rr.obs_row(conv, 1, sub, EMPTY_MEM, tick, live=True))


@pytest.mark.engine
def test_engine_state_round_trip_unit_rows(engine):
    st, conv = engine_round_trip(engine)
    rr, sub, tick = engine.rr, engine.sub, st["tick"]
    same_row(rr.obs_row(st, 1, sub, EMPTY_MEM, tick, live=False),
             rr.obs_row(conv, 1, sub, EMPTY_MEM, tick, live=False))
    same_row(vision_row(rr, st, EMPTY_MEM, tick, sub=sub),
             vision_row(rr, conv, EMPTY_MEM, tick, sub=sub))


@pytest.mark.engine
def test_engine_tower_slots_are_screen_sides(engine):
    """What the converter relies on: tower slot 1 stands left of the middle and slot 2 right,
    for both teams (types.TOWER_KEYS), and team 1's first king entity stands at (9000, 29000)
    mt, where obs_row sorts from when a state has no king entity (the converter's)."""
    F, sub = engine.F, engine.sub
    towers = [e for e in engine.state["entities"] if e[F["tower_slot"]] >= 0]
    for e in towers:
        slot, x = e[F["tower_slot"]], e[F["x"]]
        assert {0: x == 9000 * sub, 1: x < 9000 * sub, 2: x > 9000 * sub}[slot], (e[F["team"]], slot, x)
    king = {t: next(e for e in towers if e[F["team"]] == t and e[F["tower_slot"]] == 0)
            for t in (0, 1)}
    assert (king[1][F["x"]], king[1][F["y"]]) == (9000 * sub, 29000 * sub)
    assert (king[0][F["x"]], king[0][F["y"]]) == (9000 * sub, 3000 * sub)


@pytest.mark.engine
def test_engine_rebuilds_from_a_sanitized_state(engine, monkeypatch):
    """Units perception may report in the river, off the board and above their card's hit
    points: sanitised, RSAgent.rebuilt resets a real Battle from them."""
    ev = importlib.import_module("rs_eval")
    prim = rs_spawn_table.primary_of(engine.table)
    monkeypatch.setattr(ev, "_SPAWN", {"prim": prim, "by_unit": rs_spawn_table.by_unit(prim)})
    monkeypatch.setattr(ev, "_UNITS", list(engine.units))
    tr = [track(1, "HogRider", 1, 9000, 16000), track(2, "Musketeer", 1, 4400, 15500),
          track(3, "Cannon", 1, 9000, 14900), track(4, "Knight", 0, -400, 33000),
          track(5, "Musketeer", 0, 12000, 9000, form=VT.FORM_HERO),
          track(6, "Skeleton", 0, 14000, 16500), track(7, "BabyDragon", 0, 9000, 16000)]
    state = PS.PublicStateBuilder(CLASSES, engine.table, deck=engine.hog, catalogue=engine.cat,
                                  fields=engine.F, sub=engine.sub).state(
        frame(tr, hand=engine.hog[:4], next=engine.hog[4], elixir_milli=5000),
        opp_deck=engine.opp)
    san = PS.sanitize_for_rebuild(state, engine.table, fields=engine.F, sub=engine.sub)
    agent = SimpleNamespace(engine=engine.cfg)
    battle = ev.RSAgent.rebuilt(agent, san, 1, engine.rs)
    assert battle is not None, getattr(agent, "rebuild_error", None)
    rebuilt = json.loads(battle.state_json())
    teams = {e[engine.F["team"]] for e in rebuilt["entities"] if e[engine.F["tower_slot"]] < 0}
    assert teams == {0, 1}
