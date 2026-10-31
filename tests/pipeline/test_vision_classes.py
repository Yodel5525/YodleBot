"""E16: tools/vision_classes.py and data/vision/classes.json, the vision class list.

Pure tests read the shipped file and a small synthetic card table (tiny_table). The
`engine`-marked ones check the file against the real royalesim's unit_types_json() under
data/engine.json; one test compares SAME_UNIT_ALIASES with an installed royalegym, and is
skipped without it.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from tests.pipeline.conftest import DEFAULT_CATALOGUE, DEFAULT_UNIT_TYPES, FakeBattle

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def _load(name: str):
    """tools/<name>.py as the module `name`, without putting tools/ on sys.path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vc = _load("vision_classes")

EVO, HERO = vc.FORM_EVO, vc.FORM_HERO
HOG26 = ("Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons", "Log")
#: Hog 2.6's unit classes -> the card that creates them (the hero Musketeer's ability puts
#: down a MusketeerTurret, a building of its own: the 16.402 table's hero_forms[].tables.units).
HOG26_CLASSES = {"Cannon": "Cannon", "HogRider": "HogRider", "IceGolemite": "IceGolemite",
                 "IceSpirits": "IceSpirits", "Musketeer": "Musketeer", "Skeleton": "Skeletons",
                 "MusketeerTurret": "Musketeer"}
HOG26_FORMS = {"Cannon": {0, 1}, "HogRider": {0}, "IceGolemite": {0, 2}, "IceSpirits": {0, 1},
               "Musketeer": {0, 1, 2}, "Skeleton": {0, 1}, "MusketeerTurret": {0}}
#: Hog 2.6's classes that are buildings.
HOG26_BUILDINGS = {"Cannon", "MusketeerTurret"}
#: (class, form) -> KataCR detector class, for every Hog 2.6 form KataCR has.
HOG26_KATACR = {("Cannon", 0): "cannon", ("HogRider", 0): "hog-rider",
                ("IceGolemite", 0): "ice-golem", ("IceSpirits", 0): "ice-spirit",
                ("IceSpirits", 1): "ice-spirit-evolution", ("Musketeer", 0): "musketeer",
                ("Skeleton", 0): "skeleton", ("Skeleton", 1): "skeleton-evolution"}
#: scripts/vision tracker's own-deck labels (tests/pipeline/test_tracker.py HOG26_CLASSES).
TRACKER_HOG26 = {"cannon", "fireball", "hog-rider", "ice-golem", "ice-spirit", "musketeer",
                 "skeleton", "the-log"}

ALIASES = {
    "Graveyard_rework_Skeleton": "Skeleton",
    "SkeletonKingSkeleton": "Skeleton",
    "GoblinCurseGoblin": "Goblin",
    "Goblin_Stab": "Goblin",
    "SpearGoblin_Dummy": "SpearGoblin",
    "GoblinCage_EV1_GoblinBrawler": "GoblinBrawler",
    "DeliveryRecruit": "Recruit",
    "TriWizard": "Wizard",
    "Ghost_EV1_Summon_Right": "Ghost_EV1_Summon_Left",
    "ThreeMusketeer_Rework_Character_2": "ThreeMusketeer_Rework_Character_1",
    "ThreeMusketeer_Rework_Character_3": "ThreeMusketeer_Rework_Character_1",
}


@pytest.fixture(scope="module")
def table():
    return vc.ClassTable.from_file(vc.CLASSES_PATH)


def tiny_table() -> dict:
    """A card table in RoyaleSim's data/derived/cards*.json shape, with only what the
    generator reads: a summon, a spawner, a death spawn, a projectile's spawn, an evolution,
    a hero, a flyer, a building, the towers, and a unit named where it is NOT created
    (GiantBuffer's enchant excludes PhoenixEgg)."""
    def unit(source, hp, **extra):
        return {"source_table": source, "hitpoints": hp, "deploy_time_ms": 1000, **extra}
    return {
        "version": "tiny-1",
        "towers": [{"name": "KingTower", "hitpoints": 2400}, {"name": "PrincessTower", "hitpoints": 1400}],
        "units": {
            "Skeleton": unit("characters", 32),
            "Skeleton_EV1": unit("characters", 32),
            "Witch": unit("characters", 300, spawner={"character": "Skeleton", "number": 4}),
            "Bat": unit("characters", 32, flying_height=2000),
            "Tombstone": unit("buildings", 250, death_spawn={"character": "Skeleton", "count": 4}),
            "Musketeer": unit("characters", 282),
            "Phoenix": unit("characters", 500, flying_height=3000,
                            death_spawn_projectile="PhoenixFireball"),
            "PhoenixEgg": unit("buildings", 100),
            "GiantBuffer": unit("characters", 1000, enchant_friends={"excluded_units": ["PhoenixEgg"]}),
        },
        "projectiles": {"PhoenixFireball": {"name": "PhoenixFireball", "spawn_character": "PhoenixEgg"}},
        "cards": [
            {"name": "Skeletons", "kind": "troop", "elixir": 1, "summon_character": "Skeleton"},
            {"name": "Witch", "kind": "troop", "elixir": 5, "summon_character": "Witch"},
            {"name": "Bats", "kind": "troop", "elixir": 2, "summon_character": "Bat", "flying_height": 2000},
            {"name": "Tombstone", "kind": "building", "elixir": 3, "summon_character": "Tombstone"},
            {"name": "Musketeer", "kind": "troop", "elixir": 4, "summon_character": "Musketeer"},
            {"name": "Phoenix", "kind": "troop", "elixir": 4, "summon_character": "Phoenix"},
            {"name": "GiantBuffer", "kind": "troop", "elixir": 4, "summon_character": "GiantBuffer"},
            {"name": "Fireball", "kind": "spell", "elixir": 4},
        ],
        "evolutions": [{"name": "Skeletons_EV1", "kind": "troop", "summon_character": "Skeleton_EV1"}],
        "hero_forms": [{"name": "Musketeer_hero", "kind": "troop", "summon_character": "MusketeerHero",
                        "hitpoints": 282, "deploy_time_ms": 1000}],
    }


TINY_UNITS = ["Bat", "GiantBuffer", "KingTower", "Musketeer", "MusketeerHero", "Phoenix",
              "PhoenixEgg", "PrincessTower", "Skeleton", "Skeleton_EV1", "Tombstone", "Witch"]


# ---------------------------------------------------------------------------
# names and forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, known, want", [
    ("Skeleton_EV1", None, ("Skeleton", EVO)),
    ("MusketeerHero", None, ("Musketeer", HERO)),
    ("Musketeer_hero", None, ("Musketeer", HERO)),
    ("AngryBarbarian_EV1_2", None, ("AngryBarbarian", EVO)),
    ("HogRider", None, ("HogRider", 0)),
    ("Witch_EV1_Healing_Skeleton", None, ("Witch_EV1_Healing_Skeleton", EVO)),
    ("electro_dragon_ev1", ["ElectroDragon", "electro_dragon_ev1"], ("ElectroDragon", EVO)),
    ("GoblinDrill_EV1_Dig", ["GoblinDrillDig", "GoblinDrill_EV1_Dig"], ("GoblinDrillDig", EVO)),
])
def test_split_form(name, known, want):
    index = None if known is None else vc.NameIndex(known)
    assert vc.split_form(name, index) == want


def test_merges_come_before_the_form_rules():
    assert vc.class_of_unit("GoblinCurseGoblin") == ("Goblin", 0)
    assert vc.class_of_unit("Goblin_Stab") == ("Goblin", 0)
    assert vc.class_of_unit("TriWizard") == ("Wizard", 0)
    assert vc.class_of_unit("Ghost_EV1_Summon_Right") == ("Ghost_EV1_Summon_Left", EVO)
    assert vc.class_of_unit("RoyalHog_EV1_Grounded") == ("RoyalHog", EVO)        # VISION_MERGES
    # the card table's form records, then the base's merges
    assert vc.class_of_unit("MinionHorde_EV1", form_bases={"MinionHorde_EV1": ("Minion", EVO)}) \
        == ("Minion", EVO)
    assert vc.class_of_unit("GoblinHero", form_bases={"GoblinHero": ("Goblin_Stab", HERO)}) \
        == ("Goblin", HERO)
    assert vc.follow("A", {"A": "B", "B": "A"}) == "A"                           # a cycle stops


def test_same_unit_aliases_are_royalegyms():
    assert dict(vc.SAME_UNIT_ALIASES) == ALIASES
    assert json.loads(vc.CLASSES_PATH.read_text(encoding="utf-8"))["aliases"] == ALIASES


def test_same_unit_aliases_match_an_installed_royalegym():
    try:
        obs = importlib.import_module("royalegym.obs")
    except Exception as exc:                     # not installed, or its engine is missing
        pytest.skip(f"royalegym.obs does not import ({type(exc).__name__})")
    theirs = getattr(obs, "SAME_UNIT_ALIASES", None)
    if theirs is None:
        pytest.skip("this royalegym has no obs.SAME_UNIT_ALIASES (added by 0.1.25)")
    # every alias copied is still royalegym's (a newer royalegym may add more)
    assert ALIASES.items() <= dict(theirs).items()


def test_katacr_class_list(tmp_path):
    names = vc.KATACR_CLASSES
    assert len(names) == 201 and len(set(names)) == 201
    assert names[0] == "king-tower" and names[80] == "hog-rider" and names[154] == "zap-evolution"
    assert names[155] == "pad_0" and names[199] == "pad_44" and names[200] == "pad_belong"
    yaml = tmp_path / "ClashRoyale_detection.yaml"
    yaml.write_text("path: x\nnames:\n  0: king-tower\n  1: queen-tower\nnc: 2\n", encoding="utf-8")
    assert vc.read_katacr_yaml(yaml) == ["king-tower", "queen-tower"]
    assert vc.katacr_split("skeleton-evolution") == ("skeleton", EVO)
    assert vc.katacr_split("hog-rider") == ("hog-rider", 0)
    assert vc.norm_key("hog-rider") == vc.norm_key("hog_rider") == vc.norm_key("HogRider")


# ---------------------------------------------------------------------------
# the shipped file
# ---------------------------------------------------------------------------

def test_shipped_file_is_consistent(table):
    doc = table.doc
    assert doc["schema"] == vc.SCHEMA and table.digest == doc["digest"]
    assert doc["katacr_names"] == list(vc.KATACR_CLASSES)
    used = {}
    for unit, (cid, form) in table.units.items():
        assert 0 <= cid < len(table) and form in vc.FORMS, unit
        assert form in table[cid].forms, unit
        used.setdefault(cid, []).append(unit)
    for c in table:
        assert c.kind in vc.KINDS and set(c.forms) <= set(vc.FORMS), c.name
        assert c.retired or used.get(c.id), f"class {c.name} has no unit"
        for form, info in c.forms.items():      # the canonical unit maps back
            assert table.units[info.unit] == (c.id, form), (c.name, form)
        assert c.primary_card is None or c.primary_card == c.cards[0][0]
        assert all(rel in ("summon", "spawn") for _, rel in c.cards)


def test_shipped_file_text_is_what_the_generator_writes():
    text = vc.CLASSES_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert vc.dumps(json.loads(text)) == text


@pytest.mark.parametrize("unit, cls, form", [
    ("Skeleton_EV1", "Skeleton", EVO), ("MusketeerHero", "Musketeer", HERO),
    ("GoblinCurseGoblin", "Goblin", 0), ("Goblin_Stab", "Goblin", 0), ("GoblinHero", "Goblin", HERO),
    ("IceGolemiteHero", "IceGolemite", HERO), ("Cannon_EV1", "Cannon", EVO),
    ("IceSpirits_EV1", "IceSpirits", EVO), ("Musketeer_EV1", "Musketeer", EVO),
    ("HogRider", "HogRider", 0), ("TriWizard", "Wizard", 0), ("MinionHorde_EV1", "Minion", EVO),
    ("electro_dragon_ev1", "ElectroDragon", EVO), ("RoyalHog_EV1_Grounded", "RoyalHog", EVO),
    ("AngryBarbarian_EV1_2", "AngryBarbarian", EVO), ("Furnace_EV1", "Furnace_rework", EVO),
    ("KingTower", "KingTower", 0),
    ("Musketeer_hero", "Musketeer", HERO),       # not in the file: resolved by the rules
])
def test_units_resolve_to_class_and_form(table, unit, cls, form):
    cid, got = table.resolve(unit)
    assert (table[cid].name, got) == (cls, form)


def test_unknown_units_resolve_to_none(table):
    assert table.resolve("NotAUnit") == (None, 0)
    assert table.resolve("NotAUnit_EV1") == (None, EVO)
    assert table.get("NotAUnit") is None and table.token("NotAUnit") == vc.UNKNOWN_TOKEN


def test_hog26_classes(table):
    assert table.deck_classes(HOG26) == set(HOG26_CLASSES)
    for cls, card in HOG26_CLASSES.items():
        c = table[cls]
        assert c.primary_card == card and table.card_for(cls, HOG26) == card
        assert set(c.forms) == HOG26_FORMS[cls]
        assert not c.flying and c.deploy_time_ms == 1000
        building = cls in HOG26_BUILDINGS
        assert c.building == building and c.kind == ("building" if building else "troop")
    assert table["Cannon"].form(EVO).building           # the evolved Cannon is a building too
    assert table.unit_name("Musketeer", HERO) == "MusketeerHero"
    assert table.unit_name("HogRider", EVO) == "HogRider"          # no such form: the base
    elixir = {card: table.cards[card]["elixir"] for card in HOG26}
    assert elixir == {"Cannon": 3, "Fireball": 4, "HogRider": 4, "IceGolemite": 2,
                      "IceSpirits": 1, "Musketeer": 4, "Skeletons": 1, "Log": 2}
    for spell, katacr in (("Fireball", ["fireball"]), ("Log", ["the-log"])):
        assert table.cards[spell]["kind"] == "spell"
        assert table.cards[spell]["classes"] == [] and table.cards[spell]["katacr"] == katacr


def test_card_for_prefers_the_deck(table):
    assert table.card_for("Skeleton") == "Skeletons"
    assert table.card_for("Skeleton", ["Witch", "Knight"]) == "Witch"
    assert table.card_for("Skeleton", ["Witch", "Skeletons"]) == "Skeletons"
    assert table.card_for("Skeleton", ["Knight"]) == "Skeletons"
    assert table.card_for("Skeleton", ["Knight"], strict=True) is None
    assert table.card_for("skeleton-evolution", HOG26) == "Skeletons"         # a detector label
    assert table.card_for("Golemite") == "Golem"
    assert table.card_for("PhoenixEgg") == "Phoenix"
    assert table.card_for("KingTower") is None
    assert table.card_id_for("HogRider", DEFAULT_CATALOGUE) == 2
    assert table.card_id_for("Skeleton", [row[0] for row in DEFAULT_CATALOGUE]) == 6
    assert table.card_id_for("Golemite", DEFAULT_CATALOGUE) == -1


def test_crosswalk_round_trip_hog26(table):
    for (cls, form), name in HOG26_KATACR.items():
        assert table.to_katacr(cls, form) == name
        label = table.from_detector(name)
        assert (label.type, label.name, label.form) == ("unit", cls, form)
        key = table.detector_key(cls, form)
        label = table.from_detector(key)
        assert (label.type, label.name, label.form) == ("unit", cls, form)
    for cls, form in (("Musketeer", HERO), ("IceGolemite", HERO), ("Cannon", EVO), ("Musketeer", EVO)):
        assert table.to_katacr(cls, form) is None          # KataCR has no heroes, no such evos
    assert table.detector_key("Musketeer", HERO) == "MusketeerHero"
    assert table.detector_key("HogRider", EVO) is None
    assert table.from_detector("fireball") == vc.Label("spell", None, 0, "Fireball")
    assert table.from_detector("the-log") == vc.Label("spell", None, 0, "Log")
    assert table.from_detector("Fireball") == vc.Label("spell", None, 0, "Fireball")
    labels = table.deck_labels(HOG26)
    assert TRACKER_HOG26 <= labels and {"Skeleton_EV1", "MusketeerHero"} <= labels
    # the detector names a spell by its card (Detection.cls "Fireball"): ours, not an enemy's
    assert {"Fireball", "Log"} <= labels
    assert "knight" not in labels and "Knight" not in labels


def test_spell_labels_are_the_spell_cards_and_their_katacr_names(table):
    """The tracker's spell_classes (scripts/vision/tracker.py, SPELLS): every name the detector
    gives a spell box, and no class a card puts on the board as a unit."""
    spells = table.spell_labels()
    assert {"Zap", "zap", "zap-evolution", "Freeze", "freeze", "Log", "the-log", "Fireball", "fireball",
            "Rocket", "rocket", "GoblinBarrel", "goblin-barrel", "royal-delivery"} <= spells
    assert {n for n, e in table.katacr.items() if e["type"] == "spell"} <= spells
    # a spell's units keep their own classes; the one shared name is a cardless building
    assert {"Goblin", "BarrelGoblin", "HealSpirit", "Recruit", "Skeleton", "Barbarian"}.isdisjoint(spells)
    assert spells & {c.name for c in table} == {"RoyalDelivery"} and not table["RoyalDelivery"].card_names
    for name in spells - {"RoyalDelivery"}:
        assert table.from_detector(name).type == "spell", name
    assert table.deck_labels(HOG26) & spells == {"Fireball", "fireball", "Log", "the-log"}


def test_katacr_map_covers_every_detector_class(table):
    assert set(table.katacr) == set(vc.KATACR_CLASSES)
    types = {name: e["type"] for name, e in table.katacr.items()}
    assert set(types.values()) == {"unit", "tower", "spell", "object", "ui", "pad"}
    assert {n for n, t in types.items() if t == "pad"} == {n for n in vc.KATACR_CLASSES if n.startswith("pad_")}
    assert {n for n, t in types.items() if t == "tower"} == {
        "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower"}
    assert types["bar"] == types["bar-level"] == "ui"                   # scripts/vision/hpbars.py
    for name, e in table.katacr.items():
        if e["type"] in ("unit", "tower"):
            c = table[e["class"]]
            assert str(e["form"]) in {str(f) for f in c.forms}, name
            assert e["form"] == (EVO if name.endswith("-evolution") else 0), name
        elif e["type"] == "spell":
            assert table.cards[e["card"]]["kind"] == "spell", name
    assert table.from_detector("king-tower") == vc.Label("tower", table.class_id("KingTower"), 0,
                                                         None, "KingTower")
    assert table.from_detector("queen-tower").name == "PrincessTower"
    assert table.from_detector("bar").type == "ui" and table.from_detector("pad_3").type == "pad"
    assert table.from_detector("dirt").type == "object"
    assert table.katacr["bomb"]["also"] == ["BalloonBomb", "BombTowerBomb"]


@pytest.mark.parametrize("label, cls, form", [
    ("hog_rider", "HogRider", 0), ("ice_spirit_evolution", "IceSpirits", EVO),
    ("minipekka", "MiniPekka", 0), ("Skeleton_ev1", "Skeleton", EVO),
    ("IceGolemiteHero", "IceGolemite", HERO), ("elite-barbarian", "AngryBarbarian", 0),
    ("royal-guardian", "ChampionGuard", 0), ("ram-rider", "Ram", 0),
])
def test_from_detector_reads_other_spellings(table, label, cls, form):
    got = table.from_detector(label)
    assert (got.type, got.name, got.form) == ("unit", cls, form)


def test_from_detector_unknown(table):
    assert table.from_detector("nonsense") == vc.Label("unknown", name="nonsense")


def test_lookup_gives_row_tokens_forms_and_flags(table):
    lut = table.lookup(["KingTower", "Skeleton", "Skeleton_EV1", "Nope", "MusketeerHero", "RoyalHog_EV1"])
    sk, mu, rh = table.class_id("Skeleton"), table.class_id("Musketeer"), table.class_id("RoyalHog")
    assert [lut.token(u) for u in range(6)] == [table.class_id("KingTower") + 2, sk + 2, sk + 2,
                                                vc.UNKNOWN_TOKEN, mu + 2, rh + 2]
    assert lut.token(6) == lut.token(-1) == vc.UNKNOWN_TOKEN               # outside the list
    assert lut.forms == (0, 0, EVO, 0, HERO, EVO)
    assert [lut.form_feature(u) for u in range(6)] == [0.0, 0.0, 0.5, 0.0, 1.0, 0.5]
    assert lut.flags(0) == (0.0, 1.0)            # the king tower is a building
    assert lut.flags(1) == (0.0, 0.0)
    assert lut.flags(5) == (1.0, 0.0)            # evolved Royal Hogs fly in
    assert lut.flags(3) == (0.0, 0.0)
    assert lut.unknown == ["Nope"] and lut.digest == table.digest
    assert table.get(sk) is table["Skeleton"] and table.token("Skeleton") == sk + 2


def test_hitpoints_at_level_11(table):
    # floor(level-1 hp * 256 / 100): the table's level 11 multiplier is 256%
    assert table.level_multiplier[10] == 256
    want = {"Skeleton": 81, "HogRider": 1697, "Musketeer": 721, "Cannon": 824,
            "IceGolemite": 1228, "IceSpirits": 215}
    assert {cls: table.hitpoints(cls) for cls in want} == want
    assert table.hitpoints("HogRider", level=1) == 663
    assert table.hitpoints("GiantSkeletonBomb") is None
    # crown towers scale on the tower table (3052 / 4824 at 11), not the card multiplier
    assert table.hitpoints("PrincessTower") is None and table.hitpoints("KingTower") is None


# ---------------------------------------------------------------------------
# building from fixtures and card tables
# ---------------------------------------------------------------------------

def test_every_fixture_unit_maps_to_exactly_one_class():
    names = json.loads(FakeBattle().unit_types_json())
    assert names == list(DEFAULT_UNIT_TYPES)
    doc, report = vc.build(names)
    t = vc.ClassTable(doc)
    assert t.names == sorted(names)              # no forms in the fixture: one class each
    assert sorted(doc["units"]) == sorted(names)
    for name in names:
        cid, form = t.resolve(name)
        assert t[cid].name == name and form == 0
    assert [c.kind for c in t if c.kind == "tower"] == ["tower", "tower"]
    assert report["without_metadata"] == sorted(names)
    lut = t.lookup(names + ["Unseen"])
    assert lut.unknown == ["Unseen"] and lut.token(len(names)) == vc.UNKNOWN_TOKEN


def test_build_from_a_card_table():
    tbl = tiny_table()
    assert vc.table_unit_names(tbl) == TINY_UNITS
    doc, report = vc.build(TINY_UNITS, [tbl])
    t = vc.ClassTable(doc)
    assert t.names == ["Bat", "GiantBuffer", "KingTower", "Musketeer", "Phoenix", "PhoenixEgg",
                       "PrincessTower", "Skeleton", "Tombstone", "Witch"]
    assert t.resolve("Skeleton_EV1") == (t.class_id("Skeleton"), EVO)
    assert t.resolve("MusketeerHero") == (t.class_id("Musketeer"), HERO)
    # creators: the summon first, then the card making fewest classes, then the cheapest
    assert t["Skeleton"].cards == (("Skeletons", "summon"), ("Tombstone", "spawn"), ("Witch", "spawn"))
    assert t["PhoenixEgg"].cards == (("Phoenix", "spawn"),)    # via the projectile, not GiantBuffer
    assert t["Bat"].flying and not t["Skeleton"].flying
    assert t["Tombstone"].kind == "building" and t["Tombstone"].building
    assert t["KingTower"].kind == "tower" and t["KingTower"].primary_card is None
    assert set(t["Musketeer"].forms) == {0, HERO} and t["Musketeer"].form(HERO).hitpoints == 282
    assert doc["form_bases"] == {"MusketeerHero": ["Musketeer", HERO], "Skeleton_EV1": ["Skeleton", EVO]}
    assert doc["cards"]["Fireball"] == {"kind": "spell", "elixir": 4, "flying": False, "forms": [0],
                                        "classes": [], "katacr": ["fireball"]}
    assert doc["cards"]["Witch"]["classes"] == ["Witch", "Skeleton"]
    assert doc["cards"]["Skeletons"]["forms"] == [0, EVO]
    assert doc["cards"]["Bats"]["flying"] is True
    assert doc["katacr"]["skeleton-evolution"] == {"type": "unit", "class": "Skeleton", "form": EVO}
    assert doc["katacr"]["phoenix-egg"] == {"type": "unit", "class": "PhoenixEgg", "form": 0}
    assert doc["katacr"]["queen-tower"] == {"type": "tower", "class": "PrincessTower", "form": 0}
    assert "hog-rider" in report["katacr_unmapped"]
    assert report["katacr_missing_form"] == ["bat-evolution"]
    assert doc["level_multiplier_percent"] is None and t.hitpoints("Skeleton") is None
    assert json.loads(vc.dumps(doc)) == doc


def test_regeneration_keeps_ids_and_retires_merged_classes():
    tbl = tiny_table()
    first, _ = vc.build([n for n in TINY_UNITS if n != "Bat"], [tbl])
    second, report = vc.build(TINY_UNITS, [tbl], previous=first)
    old = [c["name"] for c in first["classes"]]
    assert [c["name"] for c in second["classes"]] == old + ["Bat"]          # appended, not sorted in
    assert report["new_classes"] == ["Bat"]
    third, report = vc.build(TINY_UNITS, [tbl], previous=second, merges={"Bat": "Skeleton"})
    assert [c["name"] for c in third["classes"]] == old + ["Bat"]           # no id moved
    assert report["retired"] == ["Bat"] and third["user_merges"] == {"Bat": "Skeleton"}
    t = vc.ClassTable(third)
    assert t["Bat"].retired and t.resolve("Bat") == (t.class_id("Skeleton"), 0)
    assert t.from_detector("Bat").name == "Skeleton"
    assert t.deck_classes(["Bats"]) == {"Skeleton"}                         # not the retired Bat
    assert t.from_detector("bat") == vc.Label("unknown", name="bat")        # "unmapped" entry
    fourth, _ = vc.build(TINY_UNITS, [tbl], previous=third)                # the merge is kept
    assert fourth["user_merges"] == {"Bat": "Skeleton"} and fourth["classes"] == third["classes"]


def test_write_and_load(tmp_path):
    doc, _ = vc.build(TINY_UNITS, [tiny_table()])
    path = vc.write(doc, tmp_path / "vision" / "classes.json")
    assert b"\r\n" not in path.read_bytes()
    assert json.loads(path.read_text(encoding="utf-8")) == doc
    assert vc.ClassTable.load(path) is vc.load(path)                        # read once


def test_cli_regenerates_from_card_json(tmp_path, capsys):
    card_json = tmp_path / "cards-tiny.json"
    card_json.write_text(json.dumps(tiny_table()), encoding="utf-8")
    out = tmp_path / "classes.json"
    assert vc.main(["--card-json", str(card_json), "--classes", str(out), "--note", "test"]) == 0
    first = vc.ClassTable.from_file(out)
    assert first.doc["source"]["card_tables"] == ["tiny-1"] and first.doc["source"]["note"] == "test"
    assert first.resolve("Skeleton_EV1") == (first.class_id("Skeleton"), EVO)
    assert vc.main(["--card-json", str(card_json), "--classes", str(out), "--merge", "Bat=Skeleton"]) == 0
    second = vc.ClassTable.from_file(out)
    assert second.names == first.names and second["Bat"].retired
    assert vc.main(["--classes", str(out)]) == 0
    assert "10 classes" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        vc.main(["--classes", str(out), "--merge", "Bat=Skeleton"])         # merge needs a rebuild


def test_imports_stay_light():
    tree = ast.parse((TOOLS / "vision_classes.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "argparse", "hashlib", "importlib", "json", "re", "sys",
                        "collections", "dataclasses", "pathlib", "types", "typing", "royalesim"}
    assert "memread" not in (TOOLS / "vision_classes.py").read_text(encoding="utf-8")


def test_records_carried_by_hero_forms_and_evolutions():
    """A 16.402 table keeps some troop and building records inside a hero form or an evolution
    (hero_forms[].tables.units, evo_barrel.extra): they are units of the table (the engine's
    unit_types_json() lists them), made by that card, and a hero form's own units do not make
    its card less primary. TwinMusketeers (3 elixir) summons a Musketeer and a Bat."""
    def rec(name, source, hp, **extra):
        return {"name": name, "source_table": source, "hitpoints": hp, "deploy_time_ms": 1000,
                **extra}
    tbl = tiny_table()
    hero = tbl["hero_forms"][0]
    hero["ability"] = {"effect": {"unit": "MusketeerTurret"}}
    hero["tables"] = {"units": {"MusketeerHero": rec("MusketeerHero", "characters", 282),
                                "MusketeerTurret": rec("MusketeerTurret", "buildings", 600)}}
    tbl["evolutions"][0]["evo_barrel"] = {"extra": rec(
        "SkeletonDrop", "buildings", None, deploy_time_ms=600,
        death_spawn={"character": "Skeleton", "count": 2})}
    tbl["cards"].append({"name": "TwinMusketeers", "kind": "troop", "elixir": 3,
                         "summon_character": "Musketeer", "second_summon": {"character": "Bat"}})
    names = vc.table_unit_names(tbl)
    assert names == sorted(TINY_UNITS + ["MusketeerTurret", "SkeletonDrop"])
    doc, report = vc.build(names, [tbl])
    t = vc.ClassTable(doc)
    turret, drop = t["MusketeerTurret"], t["SkeletonDrop"]
    assert (turret.kind, turret.building, turret.form(0).hitpoints) == ("building", True, 600)
    assert turret.cards == (("Musketeer", "spawn"),) and turret.primary_card == "Musketeer"
    # The dropped barrel is named by no _CREATES key: the record the evolution carries is all.
    assert drop.cards == (("Skeletons", "spawn"),) and drop.deploy_time_ms == 600
    assert (drop.kind, drop.form(0).hitpoints) == ("building", None)
    assert "SkeletonDrop" not in report["no_creator"]
    # Musketeer makes Musketeer and (as a hero) the turret, TwinMusketeers Musketeer and Bat:
    # counted plain, Musketeer makes fewer classes, so the dearer card stays primary.
    assert t["Musketeer"].cards == (("Musketeer", "summon"), ("TwinMusketeers", "summon"))
    assert t["Musketeer"].form(HERO).hitpoints == 282
    assert doc["cards"]["Musketeer"]["classes"] == ["Musketeer", "MusketeerTurret"]
    assert t.deck_classes(["Musketeer"]) == {"Musketeer", "MusketeerTurret"}
    assert t["Skeleton"].cards == (("Skeletons", "summon"), ("Tombstone", "spawn"),
                                   ("Witch", "spawn"))


def test_embedded_table_reads_a_compiled_in_card_table():
    a = {"version": "cards-a.1", "units": {"X": {"name": "X"}}}
    b = {"version": "cards-b.1", "note": 'not "version": "cards-a.1" (inside a string)'}
    blob = (b"\x7fELF\x00{not json " + json.dumps(b, indent=1).encode() + b"cards-a.json"
            + json.dumps(a, indent=1).encode() + b"\xff\xfe")
    assert vc.embedded_table(blob, "cards-a.1") == a
    assert vc.embedded_table(blob, "cards-b.1") == b
    assert vc.embedded_table(blob, "cards-c.1") is None


def test_engine_card_table_reads_the_package_file_then_the_extension(tmp_path):
    ran, own = {"version": "cards-x.1", "units": {}}, {"version": "cards-own.1", "units": {}}
    pkg = tmp_path / "royalesim"
    (pkg / "data" / "derived").mkdir(parents=True)
    (pkg / "data" / "derived" / "cards.json").write_text(json.dumps(own), encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "royalesim.abi3.so").write_bytes(b"\x7fELF" + json.dumps(ran).encode() + b"\x00")
    rs = types.SimpleNamespace(__file__=str(pkg / "__init__.py"), data_dir=lambda: pkg / "data")
    battle = types.SimpleNamespace(card_table=lambda: "cards-x.1")
    assert vc.engine_card_table(battle, rs) == (
        ran, "royalesim royalesim.abi3.so (its compiled-in copy)")
    battle.card_table = lambda: "cards-own.1"
    assert vc.engine_card_table(battle, rs) == (own, "royalesim data/derived/cards.json")
    battle.card_table = lambda: "cards-gone.1"
    doc, why = vc.engine_card_table(battle, rs)
    assert doc is None and "cards-gone.1" in why
    doc, why = vc.engine_card_table(types.SimpleNamespace(), rs)       # royalesim before 0.1.42
    assert doc is None and "card_table" in why


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

def _engine_names():
    rs_engine = _load("rs_engine")
    cfg = rs_engine.EngineConfig.load()
    return rs_engine, cfg, rs_engine.unit_types(cfg)


@pytest.mark.engine
def test_engine_unit_types_resolve_in_the_shipped_file(table):
    _rs_engine, cfg, names = _engine_names()
    assert names
    lut = table.lookup(names)
    assert lut.unknown == [], (f"{len(lut.unknown)} unit types under {cfg.describe()} are not in "
                               f"data/vision/classes.json: {lut.unknown[:10]}; regenerate with "
                               "tools/vision_classes.py --engine")
    for unit, cls, form in (("Skeleton_EV1", "Skeleton", EVO), ("MusketeerHero", "Musketeer", HERO),
                            ("GoblinCurseGoblin", "Goblin", 0), ("HogRider", "HogRider", 0)):
        if unit in names:
            cid, got = table.resolve(unit)
            assert (table[cid].name, got) == (cls, form)


@pytest.mark.engine
def test_engine_every_unit_type_maps_to_exactly_one_class():
    rs_engine, cfg, names = _engine_names()
    royalesim = importlib.import_module("royalesim")
    derived = Path(royalesim.data_dir()) / "derived" / "cards.json"
    fallback = [json.loads(derived.read_text(encoding="utf-8"))] if derived.is_file() else []
    catalogue = [row[0] for row in rs_engine.catalogue(cfg)]
    doc, _ = vc.build(names, fallback_tables=fallback, catalogue=catalogue)
    t = vc.ClassTable(doc)
    assert set(doc["units"]) == set(names)
    for name in names:
        cid, form = t.resolve(name)
        assert cid is not None and form in vc.FORMS, name
    for card in HOG26_CLASSES.values():
        assert card in catalogue


@pytest.mark.engine
def test_engine_hog26_cards_have_catalogue_ids(table):
    rs_engine, cfg, _names = _engine_names()
    catalogue = rs_engine.catalogue(cfg)
    for cls in HOG26_CLASSES:
        assert table.card_id_for(cls, catalogue) >= 0, cls
    names = [row[0] for row in catalogue]
    assert {"Fireball", "Log"} <= set(names)
