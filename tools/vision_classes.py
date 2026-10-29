r"""The vision class list: one vocabulary for the detector, the tracker and the policy (E16).

    .venvs\royalegym\Scripts\python.exe tools\vision_classes.py             summary of the file
    .venvs\royalegym\Scripts\python.exe tools\vision_classes.py --check     does data/vision/classes.json
                                                                           cover the engine's unit types?
    .venvs\royalegym\Scripts\python.exe tools\vision_classes.py --engine    regenerate it: unit names from
                                                                           the engine's unit_types_json()
    python tools\vision_classes.py --card-json T.json [--card-json T2.json] regenerate it from card tables

Why (vision-01 and the plan's missed issue 2): neither row schema could be produced by a
screen. The unit-type rows name RoyaleSim unit types, where Skeleton_EV1 and MusketeerHero
are types of their own; the KataCR detector strips forms (scripts/brain/vision.py _key) and
has never seen a hero. The live-compat rows name the CARD that created a unit. This module
gives every party the same list, data/vision/classes.json:

A CLASS is what a screen can tell apart: a RoyaleSim unit type with its form taken off,
after the merges below. Each unit type maps to exactly one (class, form):

  * forms. FORM_BASE 0, FORM_EVO 1, FORM_HERO 2 (scripts/vision/types.py has the same).
    - The card table's evolution and hero records say it first: the unit an evolution or
      hero summons is that form of the unit its base card summons (MinionHorde_EV1 ->
      (Minion, evo), Furnace_EV1 -> (Furnace_rework, evo), GoblinHero -> Goblin_Stab ->
      (Goblin, hero)).
    - Otherwise by name: a _EV<n> suffix (also _EV<n>_<k>, any case) is evo of the base
      (Skeleton_EV1 -> (Skeleton, evo), electro_dragon_ev1 -> (ElectroDragon, evo)); a
      Hero / _hero suffix is hero (MusketeerHero -> (Musketeer, hero)); an _EV<n>_ infix is
      evo of the base the rest names when that is a unit (GoblinDrill_EV1_Dig ->
      (GoblinDrillDig, evo)), else evo of a class of its own (Witch_EV1_Healing_Skeleton).
  * merges, applied before the form rules:
    - SAME_UNIT_ALIASES, copied from royalegym 0.1.25 (MIT): rows of the characters table
      that are ONE unit to a player (GoblinCurseGoblin -> Goblin, TriWizard -> Wizard);
    - VISION_MERGES, rows a screen cannot tell apart (RoyalHog_EV1_Grounded);
    - user merges, `--merge UNIT=TARGET`, kept in the file across regenerations.

Each class carries: id, name (the canonical unit of its lowest form), kind (troop /
building / tower), flying, building and deploy_time_ms (its lowest form's), hitpoints at
level 1, primary_card, cards, forms and katacr.

  primary_card  the CREATING card, what an engine entity's card_id says and what
                --live-compat rows (tools/rs_rows.py) use as the unit id: a card that
                summons the class directly, the one making the fewest other classes, then
                the cheapest (Skeleton -> Skeletons, not SkeletonArmy or Witch).
  cards         every card that puts the class on the board, [card, "summon" | "spawn"],
                primary first. card_for(cls, deck) picks the deck's card, so a Witch's
                skeleton reads Witch when the deck has a Witch and no Skeletons.
  forms         form -> {unit, flying, building, deploy_time_ms, hitpoints}: the canonical
                unit type of that form and its own stats (building follows the class: the
                evolved Cannon is a characters row but a building on screen).
  katacr        form -> the KataCR detector class ("skeleton", "skeleton-evolution").

The KataCR crosswalk maps all 201 detector classes of the Clash-Royale-Detection-Dataset
(images/part2/ClashRoyale_detection.yaml, MIT) to a unit class and form, a crown tower, a
spell card, an object (dirt, axe), UI (bars, clock, text) or padding. from_detector()
reads KataCR names, BuildABot-style names (hog_rider, by scripts/brain/vision.py _key's
normalisation) and this list's own detector keys (canonical unit names: "Skeleton_EV1").

Ids are positions in "classes", like vocabulary ids. A regeneration keeps every existing
id and appends new classes (sorted); --fresh-ids renumbers. In rows a class is id + 2
(0 pad, 1 unknown), as unit types and cards are in tools/rs_rows.py; ClassTable.lookup()
turns the engine's unit_types_json() into those tokens plus the form and the flags.

The shipped file was generated from royalesim 0.1.42's 16.402.22 card table
(cards-160402017-20261019.1, the table data/engine.json selects), merged with 16.402.19 and
15.535 so that every table's names resolve. Its unit names are every troop and building
record, the crown towers and the hero units: a superset of what unit_types_json() lists
("every troop and building record of the loaded card table, crown towers included, sorted
by name"). `--engine` regenerates it from the engine itself, `--check` verifies coverage.

Imports only the standard library. The command line loads tools/rs_engine.py for its engine
arguments, and royalesim only for --engine and --check. Never imports scripts/gamestate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
CLASSES_PATH = ROOT / "data" / "vision" / "classes.json"
SCHEMA = 1

FORM_BASE, FORM_EVO, FORM_HERO = 0, 1, 2
FORMS = (FORM_BASE, FORM_EVO, FORM_HERO)
FORM_NAMES = ("base", "evo", "hero")

KIND_TROOP, KIND_BUILDING, KIND_TOWER = "troop", "building", "tower"
KINDS = (KIND_TROOP, KIND_BUILDING, KIND_TOWER)

#: Row tokens, as tools/rs_rows.py writes unit types and cards: 0 pad, 1 unknown, id + 2.
PAD_TOKEN, UNKNOWN_TOKEN, TOKEN_OFFSET = 0, 1, 2
TICK_MS = 50
FRIENDLY_LEVEL = 11            # friendly battles play every card at level 11

# Third-party data copied below, both under the MIT License:
#   SAME_UNIT_ALIASES  royalegym 0.1.25, royalegym/obs.py.
#                      Copyright (c) 2026 RoyaleGym contributors.
#   KATACR_NAMED       wty-yy/Clash-Royale-Detection-Dataset, images/part2/
#                      ClashRoyale_detection.yaml (commit 31b4151f). Copyright (c) 2024 wty.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

#: royalegym 0.1.25 obs.SAME_UNIT_ALIASES, verbatim (MIT, see above): characters-table rows
#: that are one unit to a player, mapped to their base.
SAME_UNIT_ALIASES: Mapping[str, str] = MappingProxyType({
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
})

#: Rows a screen cannot tell apart although the engine plays them differently. The evolved
#: Royal Hogs fly in and land; the landed row is the same evolved hog on the ground.
VISION_MERGES: Mapping[str, str] = MappingProxyType({
    "RoyalHog_EV1_Grounded": "RoyalHog_EV1",
})

# ---------------------------------------------------------------------------
# KataCR detector classes
# ---------------------------------------------------------------------------

#: The 155 named classes of ClashRoyale_detection.yaml, ids 0-154 (MIT, see above).
KATACR_NAMED: tuple[str, ...] = (
    'king-tower', 'queen-tower', 'cannoneer-tower', 'dagger-duchess-tower',
    'dagger-duchess-tower-bar', 'tower-bar', 'king-tower-bar', 'bar', 'bar-level', 'clock',
    'emote', 'text', 'elixir', 'selected', 'skeleton-king-bar', 'skeleton', 'skeleton-evolution',
    'electro-spirit', 'fire-spirit', 'ice-spirit', 'heal-spirit', 'goblin', 'spear-goblin',
    'bomber', 'bat', 'bat-evolution', 'zap', 'giant-snowball', 'ice-golem', 'barbarian-barrel',
    'barbarian', 'barbarian-evolution', 'wall-breaker', 'rage', 'the-log', 'archer', 'arrows',
    'knight', 'knight-evolution', 'minion', 'cannon', 'skeleton-barrel', 'firecracker',
    'firecracker-evolution', 'royal-delivery', 'royal-recruit', 'royal-recruit-evolution',
    'tombstone', 'mega-minion', 'dart-goblin', 'earthquake', 'elixir-golem-big',
    'elixir-golem-mid', 'elixir-golem-small', 'goblin-barrel', 'guard', 'clone', 'tornado',
    'miner', 'dirt', 'princess', 'ice-wizard', 'royal-ghost', 'bandit', 'fisherman',
    'skeleton-dragon', 'mortar', 'mortar-evolution', 'tesla', 'fireball', 'mini-pekka',
    'musketeer', 'goblin-cage', 'goblin-brawler', 'valkyrie', 'battle-ram',
    'battle-ram-evolution', 'bomb-tower', 'bomb', 'flying-machine', 'hog-rider', 'battle-healer',
    'furnace', 'zappy', 'baby-dragon', 'dark-prince', 'freeze', 'poison', 'hunter',
    'goblin-drill', 'electro-wizard', 'inferno-dragon', 'phoenix-big', 'phoenix-egg',
    'phoenix-small', 'magic-archer', 'lumberjack', 'night-witch', 'mother-witch', 'hog',
    'golden-knight', 'skeleton-king', 'mighty-miner', 'rascal-boy', 'rascal-girl', 'giant',
    'goblin-hut', 'inferno-tower', 'wizard', 'royal-hog', 'witch', 'balloon', 'prince',
    'electro-dragon', 'bowler', 'executioner', 'axe', 'cannon-cart', 'ram-rider', 'graveyard',
    'archer-queen', 'monk', 'royal-giant', 'royal-giant-evolution', 'elite-barbarian', 'rocket',
    'barbarian-hut', 'elixir-collector', 'giant-skeleton', 'lightning', 'goblin-giant', 'x-bow',
    'sparky', 'pekka', 'electro-giant', 'mega-knight', 'lava-hound', 'lava-pup', 'golem',
    'golemite', 'little-prince', 'royal-guardian', 'archer-evolution', 'ice-spirit-evolution',
    'valkyrie-evolution', 'bomber-evolution', 'wall-breaker-evolution', 'evolution-symbol',
    'mirror', 'tesla-evolution', 'goblin-ball', 'skeleton-king-skill', 'tesla-evolution-shock',
    'ice-spirit-evolution-symbol', 'zap-evolution',
)
#: All 201 detector classes in id order: the named ones, then pad_0..pad_44 and pad_belong.
KATACR_CLASSES: tuple[str, ...] = (KATACR_NAMED + tuple(f"pad_{i}" for i in range(45))
                                   + ("pad_belong",))
KATACR_DATASET = MappingProxyType({
    "repo": "https://github.com/wty-yy/Clash-Royale-Detection-Dataset",
    "commit": "31b4151fedb1b914e99c3c122c16dca61cb2b905",
    "file": "images/part2/ClashRoyale_detection.yaml",
    "sha256": "19d94a420f99229671602c352d0e6d652498b4bece20974716446063e37a597c",
    "license": "MIT, Copyright (c) 2024 wty",
    "label_format": "class x y w h side + 6 unused; side (column 6) 0 = the bottom player, 1 = the top",
})

#: HUD and overlay classes: no unit. "bar" / "bar-level" are unit HP bars (scripts/vision/hpbars.py).
KATACR_UI = frozenset({
    "dagger-duchess-tower-bar", "tower-bar", "king-tower-bar", "bar", "bar-level", "clock",
    "emote", "text", "elixir", "selected", "skeleton-king-bar", "evolution-symbol",
    "ice-spirit-evolution-symbol",
})
#: Spell visuals -> the spell card (catalogue name). A "-evolution" suffix is its evo form.
KATACR_SPELLS: Mapping[str, str] = MappingProxyType({
    "zap": "Zap", "zap-evolution": "Zap", "giant-snowball": "Snowball", "rage": "Rage",
    "the-log": "Log", "arrows": "Arrows", "earthquake": "Earthquake", "clone": "Clone",
    "tornado": "Tornado", "fireball": "Fireball", "freeze": "Freeze", "poison": "Poison",
    "rocket": "Rocket", "lightning": "Lightning", "goblin-barrel": "GoblinBarrel",
    "graveyard": "Graveyard", "barbarian-barrel": "BarbLog", "mirror": "Mirror",
    "royal-delivery": "RoyalDelivery",
})
#: Objects that are not unit types, with the units they belong to (the dataset's notes:
#: dirt = a burrowing Miner, Goblin Drill or Mighty Miner; axe = the Executioner's).
KATACR_OBJECTS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "dirt": ("Miner", "GoblinDrillDig", "MightyMiner"),
    "axe": ("AxeMan",),
    "skeleton-king-skill": ("SkeletonKing",),
    "tesla-evolution-shock": ("Tesla_EV1",),
    "goblin-ball": (),
})
#: KataCR unit classes whose RoyaleSim unit the names do not give away (base names, without
#: "-evolution"). The rest match by normalised name (archer -> Archer, x-bow -> Xbow).
KATACR_UNITS: Mapping[str, str] = MappingProxyType({
    "queen-tower": "PrincessTower", "cannoneer-tower": "Cannoneer",
    "dagger-duchess-tower": "DaggerDuchess",
    "fire-spirit": "FireSpirits", "ice-spirit": "IceSpirits", "ice-golem": "IceGolemite",
    "royal-recruit": "Recruit", "skeleton-barrel": "SkeletonBalloon",
    "dart-goblin": "BlowdartGoblin", "guard": "SkeletonWarrior", "royal-ghost": "Ghost",
    "bandit": "Assassin", "flying-machine": "DartBarrell", "furnace": "Furnace_rework",
    "zappy": "MiniZapMachine", "sparky": "ZapMachine", "magic-archer": "EliteArcher",
    "lumberjack": "RageBarbarian", "night-witch": "DarkWitch", "mother-witch": "WitchMother",
    "hog": "VoodooHog", "executioner": "AxeMan", "cannon-cart": "MovingCannon",
    "elite-barbarian": "AngryBarbarian", "lava-pup": "LavaPups",
    "royal-guardian": "ChampionGuard", "goblin-hut": "GoblinHut_Rework",
    "elixir-golem-big": "ElixirGolem1", "elixir-golem-mid": "ElixirGolem2",
    "elixir-golem-small": "ElixirGolem4", "phoenix-big": "Phoenix",
    "phoenix-small": "PhoenixNoRespawn",
    # The Ram Rider is two engine entities, the Ram (it moves, it has the hit points) and the
    # rider mounted on it; the detector draws one box.
    "ram-rider": "Ram",
    # Three death bombs, one sprite to the dataset; the Giant Skeleton's is the default.
    "bomb": "GiantSkeletonBomb",
})
#: Other classes a KataCR class can be (its entry's "also").
KATACR_ALSO: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "bomb": ("BalloonBomb", "BombTowerBomb"),
    "ram-rider": ("RamRider",),
})

#: Units that are part of a crown tower or replace one (kind "tower"), besides names that
#: contain KingTower / PrincessTower.
TOWER_UNITS = frozenset({"TowerPrincess", "TowerCannoneer", "TowerKnifeThrower", "Cannoneer",
                         "DaggerDuchess", "Chef", "ChefTower", "ChefTowerKing"})
_TOWER_NAME = re.compile(r"(King|Princess)Tower")

_EV_SUFFIX = re.compile(r"^(?P<base>.+?)_[Ee][Vv](?P<n>\d+)(?:_\d+)?$")
_EV_INFIX = re.compile(r"_[Ee][Vv]\d+(?=_)")
_HERO_SUFFIX = re.compile(r"^(?P<base>.+?)_?[Hh][Ee][Rr][Oo]$")

#: Keys of the file written one entry per line.
_ROW_KEYS = frozenset({"_comment", "classes", "units", "cards", "katacr", "form_bases"})


# ---------------------------------------------------------------------------
# names and forms
# ---------------------------------------------------------------------------

def norm_key(name: str) -> str:
    """scripts/brain/vision.py _key's normalisation: lower case, letters and digits only, so
    the dataset's "hog-rider", BuildABot's "hog_rider" and RoyaleSim's "HogRider" compare
    equal. (_key also strips "-evolution"; here that is the form, see katacr_split.)"""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def katacr_split(name: str) -> tuple[str, int]:
    """A KataCR class -> (base name, form): "skeleton-evolution" -> ("skeleton", FORM_EVO)."""
    if name.endswith("-evolution"):
        return name[: -len("-evolution")], FORM_EVO
    return name, FORM_BASE


def _syntactic(name: str) -> tuple[str, int, bool]:
    """(base, form, exact) read off the name alone. exact is False for an _EV<n>_ infix,
    whose remainder is a base only if it names a unit."""
    m = _EV_SUFFIX.match(name)
    if m:
        return m.group("base"), FORM_EVO, True
    m = _HERO_SUFFIX.match(name)
    if m:
        return m.group("base"), FORM_HERO, True
    if _EV_INFIX.search(name):
        return _EV_INFIX.sub("", name, count=1), FORM_EVO, False
    return name, FORM_BASE, True


class NameIndex:
    """Unit names, and the base-form ones by norm_key, for finding the base of a form."""

    def __init__(self, names: Iterable[str]):
        self.names = frozenset(str(n) for n in names)
        self.by_key: dict[str, str] = {}
        for n in sorted(self.names):
            if _syntactic(n)[1] == FORM_BASE:
                self.by_key.setdefault(norm_key(n), n)

    def find(self, name: str) -> str | None:
        if name in self.names:
            return name
        return self.by_key.get(norm_key(name))


def split_form(name: str, index: NameIndex | None = None) -> tuple[str, int]:
    """A unit name -> (base unit name, form) by the name rules (module docstring).

    With an index the base is looked up among real unit names (electro_dragon_ev1 ->
    ElectroDragon); without one, or when no unit matches, a suffix is simply removed and an
    infix name stays its own base."""
    base, form, exact = _syntactic(name)
    if form == FORM_BASE:
        return name, FORM_BASE
    hit = index.find(base) if index is not None else None
    if hit is not None:
        return hit, form
    return (base if exact else name), form


def follow(name: str, aliases: Mapping[str, str]) -> str:
    """`name` through the alias chain (cycles stop where they close)."""
    seen = set()
    while name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    return name


def class_of_unit(name: str, index: NameIndex | None = None,
                  aliases: Mapping[str, str] | None = None,
                  form_bases: Mapping[str, tuple[str, int]] | None = None) -> tuple[str, int]:
    """A unit name -> (class name, form): merges first, then the card table's form records
    (form_bases: unit -> (base unit, form)), then the name rules."""
    aliases = default_aliases() if aliases is None else aliases
    name = follow(str(name), aliases)
    if form_bases and name in form_bases:
        base, form = form_bases[name]
        cls, base_form = split_form(follow(base, aliases), index)
        return cls, max(int(form), base_form)
    return split_form(name, index)


def default_aliases() -> dict[str, str]:
    return {**SAME_UNIT_ALIASES, **VISION_MERGES}


def class_digest(names: Sequence[str]) -> str:
    """SHA-1 of the class names in id order, one per line: equal means the same ids."""
    return hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# reading a card table (RoyaleSim data/derived/cards*.json)
# ---------------------------------------------------------------------------

def _info(rec: Mapping[str, Any], kind: str, building: bool) -> dict[str, Any]:
    hp = rec.get("hitpoints")
    return {"kind": kind, "flying": (rec.get("flying_height") or 0) > 0, "building": building,
            "deploy_time_ms": int(rec.get("deploy_time_ms") or 0),
            "hitpoints": None if hp is None else int(hp)}


def _tower(name: str) -> bool:
    return name in TOWER_UNITS or bool(_TOWER_NAME.search(name))


def table_unit_records(table: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every unit of a card table -> {kind, flying, building, deploy_time_ms, hitpoints}:
    the crown towers, the characters and buildings tables, and the units hero forms summon
    (which have no row of their own)."""
    out: dict[str, dict[str, Any]] = {}
    for rec in table.get("towers") or ():
        out[rec["name"]] = _info(rec, KIND_TOWER, True)
    for name, rec in (table.get("units") or {}).items():
        if name in out:
            continue
        building = rec.get("source_table") == "buildings"
        kind = KIND_TOWER if _tower(name) else KIND_BUILDING if building else KIND_TROOP
        out[name] = _info(rec, kind, building)
    for rec in table.get("hero_forms") or ():
        unit = rec.get("summon_character")
        if unit and unit not in out:
            building = rec.get("kind") == "building"
            out[unit] = _info(rec, KIND_BUILDING if building else KIND_TROOP, building)
    return out


def table_unit_names(table: Mapping[str, Any]) -> list[str]:
    """table_unit_records' names, sorted: a superset of the engine's unit_types_json()."""
    return sorted(table_unit_records(table))


def _card_records(table: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any], int]]:
    """(base card, record, form) for every card, evolution and hero form of a table."""
    cards = {c["name"] for c in table.get("cards") or ()}
    out = [(c["name"], c, FORM_BASE) for c in table.get("cards") or ()]
    for group, form in (("evolutions", FORM_EVO), ("hero_forms", FORM_HERO)):
        for rec in table.get(group) or ():
            base = _syntactic(rec["name"])[0]
            if base in cards:
                out.append((base, rec, form))
    return out


def _summons(rec: Mapping[str, Any]) -> list[str]:
    """What a card record puts down itself: summon_character, second_summon, summon_members."""
    out = [rec.get("summon_character"), (rec.get("second_summon") or {}).get("character")]
    out += [m.get("character") for m in rec.get("summon_members") or () if isinstance(m, Mapping)]
    return [u for u in out if u]


#: Keys of a card-table record whose value names a unit the record puts on the board:
#: summons, spawners, death spawns, projectile and area-effect spawns, action-graph spawns
#: ("CharacterType:X"), morphs and transforms, attached riders, and the evolutions' own
#: blocks (evo_ghost.left, evo_cage.unit, ...). A record also names units it does NOT make
#: (raw.Name, ClonedVersion, form_of, an enchant's multipliers, excluded_units, an
#: earthquake's hidden_split, a parry's attackers), so those keys are left out.
_CREATES = frozenset({
    "summon_character", "character", "character2", "characters", "spawn_character", "spawns",
    "spawn", "SummonCharacter", "SpawnCharacter", "SpawnCharacter2", "DeathSpawnCharacter",
    "DeathSpawnCharacter2", "DeathSpawnCharacter3", "attached_character", "AttachedCharacter",
    "into", "morph", "SpawnPathfindMorph", "unit", "goblin", "general", "spectral", "clone",
    "left", "right", "ghost", "card",
})


#: Keys whose value names a projectile or area-effect object the record fires or leaves
#: (the Phoenix's death projectile spawns the egg); those records are followed too.
_OBJECT_KEYS = frozenset({
    "Projectile", "Projectile2", "Projectile3", "projectile", "ProjectileSpecial",
    "CustomFirstProjectile", "death_spawn_projectile", "DeathSpawnProjectile",
    "spawn_projectile", "deploy_projectile", "buff_projectile", "spawns", "spawn",
    "spawn_area_effect_object", "spawn_area_object", "death_area_effect", "deploy_area_effect",
    "area_effect_on_hit", "deploy_spawn_area", "landing_area", "area", "action",
    "OnStartingAction", "OnAppearAction", "root", "blow", "big", "small",
})


def _refs(obj: Any, keys: frozenset, key: str | None = None, parent: str | None = None):
    """Name candidates under `keys` (and evo_fall.grounded.name), anywhere in a record."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from _refs(v, keys, str(k), key)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _refs(v, keys, key, parent)
    elif isinstance(obj, str) and (key in keys or (parent == "grounded" and key == "name")):
        yield obj.rsplit(":", 1)[-1]          # "CharacterType:Graveyard_rework_Skeleton"


def _made_by(rec: Mapping[str, Any], units: Mapping[str, Mapping[str, Any]],
             objects: Mapping[str, Mapping[str, Any]], known: frozenset) -> set[str]:
    """Every unit `rec` creates, following the units and objects it creates in turn."""
    made: set[str] = set()
    seen_objects: set[str] = set()
    stack = [rec]
    while stack:
        r = stack.pop()
        for name in _refs(r, _CREATES):
            if name in known and name not in made:
                made.add(name)
                if name in units:
                    stack.append(units[name])
        for name in _refs(r, _OBJECT_KEYS):
            if name in objects and name not in seen_objects:
                seen_objects.add(name)
                stack.append(objects[name])
    return made


def table_form_bases(tables: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, int]]:
    """unit -> (base unit, form) from the evolution and hero records: what an evolution or a
    hero summons is that form of what its base card summons (the second summon pairs with
    the second summon, else with the first)."""
    out: dict[str, tuple[str, int]] = {}
    for table in tables:
        cards = {c["name"]: c for c in table.get("cards") or ()}
        for base_card, rec, form in _card_records(table):
            if form == FORM_BASE:
                continue
            base = cards[base_card]
            first = base.get("summon_character")
            second = (base.get("second_summon") or {}).get("character") or first
            for unit, target in ((rec.get("summon_character"), first),
                                 ((rec.get("second_summon") or {}).get("character"), second)):
                if unit and target and unit != target:
                    out.setdefault(unit, (target, form))
    return out


def creators_of(tables: Sequence[Mapping[str, Any]], unit_names: Iterable[str]
                ) -> dict[str, dict[str, str]]:
    """unit -> {card: "summon" | "spawn"}: the cards that put each unit on the board.

    "summon": the card (or its evolution / hero form) deploys it. "spawn": what the card's
    record, or a unit it makes, creates under a _CREATES key (spawner, death spawn, a
    projectile's or an area effect's spawn, an action graph's CharacterType:..., a morph):
    Witch -> Skeleton, Golem -> Golemite, Graveyard -> Graveyard_rework_Skeleton."""
    known = frozenset(unit_names)
    records: dict[str, Mapping[str, Any]] = {}
    objects: dict[str, Mapping[str, Any]] = {}
    for table in tables:
        for name, rec in (table.get("units") or {}).items():
            records.setdefault(name, rec)
        for rec in table.get("hero_forms") or ():
            if rec.get("summon_character"):
                records.setdefault(rec["summon_character"], rec)
        for group in ("projectiles", "area_effect_objects"):
            for name, rec in (table.get(group) or {}).items():
                objects.setdefault(name, rec)
    out: dict[str, dict[str, str]] = {}
    for table in tables:
        for card, rec, _form in _card_records(table):
            direct = set(_summons(rec)) & known
            reach = _made_by(rec, records, objects, known) | direct
            for unit in reach:
                rel = "summon" if unit in direct else "spawn"
                have = out.setdefault(unit, {})
                if have.get(card) != "summon":
                    have[card] = rel
    return out


def _level_multiplier(tables: Sequence[Mapping[str, Any]]) -> list[int] | None:
    """The cards' level_scaling.multiplier_percent_by_level, when they are all one series
    (cards with fewer levels carry a prefix of it); None when they disagree."""
    seen = set()
    for table in tables[:1]:
        for _card, rec, _form in _card_records(table):
            mult = (rec.get("level_scaling") or {}).get("multiplier_percent_by_level")
            if mult:
                seen.add(tuple(int(m) for m in mult))
    if not seen:
        return None
    longest = max(seen, key=len)
    return list(longest) if all(longest[:len(m)] == m for m in seen) else None


# ---------------------------------------------------------------------------
# building classes.json
# ---------------------------------------------------------------------------

def _canonical(cls: str, form: int, units: Iterable[str], aliases: Mapping[str, str]) -> str:
    """The unit a (class, form) is written as: the plain name, then one no alias redirects,
    then the shortest."""
    expected = {FORM_BASE: cls, FORM_EVO: f"{cls}_EV1", FORM_HERO: f"{cls}Hero"}[form]
    return min(units, key=lambda u: (u != expected, u in aliases, len(u), u))


def _katacr_entry(name: str, class_info: Mapping[str, Mapping[str, Any]],
                  resolve, by_key: Mapping[str, str]) -> dict[str, Any]:
    if name.startswith("pad_"):
        return {"type": "pad"}
    if name in KATACR_UI:
        return {"type": "ui"}
    base, form = katacr_split(name)
    if name in KATACR_SPELLS:
        return {"type": "spell", "card": KATACR_SPELLS[name], "form": form}
    if name in KATACR_OBJECTS:
        of = [resolve(u)[0] for u in KATACR_OBJECTS[name]]
        return {"type": "object", "of": [c for c in dict.fromkeys(of) if c in class_info]}
    unit = KATACR_UNITS.get(base)
    if unit is not None:
        cls, unit_form = resolve(unit)
    else:
        cls, unit_form = by_key.get(norm_key(base)), FORM_BASE
    if cls is None or cls not in class_info:
        return {"type": "unmapped"}
    entry: dict[str, Any] = {"type": "tower" if class_info[cls]["kind"] == KIND_TOWER else "unit",
                             "class": cls, "form": form or unit_form}
    also = [resolve(u)[0] for u in KATACR_ALSO.get(name, ())]
    also = [c for c in dict.fromkeys(also) if c in class_info and c != cls]
    if also:
        entry["also"] = also
    return entry


def build(unit_names: Iterable[str], tables: Sequence[Mapping[str, Any]] = (), *,
          fallback_tables: Sequence[Mapping[str, Any]] = (),
          previous: Mapping[str, Any] | None = None, merges: Mapping[str, str] | None = None,
          catalogue: Sequence[str] | None = None, source: Mapping[str, Any] | None = None
          ) -> tuple[dict[str, Any], dict[str, Any]]:
    """(classes.json document, report) for `unit_names`.

    tables     card-table JSON documents, preferred first: unit stats, creators, forms.
    fallback_tables  more tables, whose unit stats rank below `previous`'s (--engine with no
               --card-json reads the wheel's data/derived/cards.json, an older table than the
               one the shipped file was made from).
    previous   an earlier classes.json: its ids are kept (new classes are appended, sorted),
               its classes and units stay in, its user merges apply, and its stats fill in
               units the tables lack.
    merges     more user merges, unit -> target unit.
    catalogue  the engine's card names in id order: creators are limited to these.
    """
    previous = dict(previous or {})
    every_table = list(tables) + list(fallback_tables)
    user_merges = {**dict(previous.get("user_merges") or {}), **dict(merges or {})}
    aliases = {**SAME_UNIT_ALIASES, **VISION_MERGES, **user_merges}

    names = {str(n) for n in unit_names}
    prev_units = dict(previous.get("units") or {})
    prev_classes = list(previous.get("classes") or [])
    names |= set(prev_units)

    info: dict[str, dict[str, Any]] = {}
    for table in tables:
        for name, rec in table_unit_records(table).items():
            info.setdefault(name, rec)
    for name, (cid, form) in prev_units.items():       # stats the tables lack
        if name in info or not 0 <= int(cid) < len(prev_classes):
            continue
        old = prev_classes[int(cid)]
        f = (old.get("forms") or {}).get(str(form))
        if f:
            info[name] = {"kind": old.get("kind", KIND_TROOP), "flying": bool(f["flying"]),
                          "building": bool(f["building"]), "deploy_time_ms": int(f["deploy_time_ms"]),
                          "hitpoints": f.get("hitpoints")}
    for table in fallback_tables:
        for name, rec in table_unit_records(table).items():
            info.setdefault(name, rec)
    described = set(info)
    for name in names:
        if name not in info:
            tower = _tower(name)
            info[name] = {"kind": KIND_TOWER if tower else KIND_TROOP, "flying": False,
                          "building": tower, "deploy_time_ms": 0, "hitpoints": None}

    form_bases = {k: (str(v[0]), int(v[1])) for k, v in (previous.get("form_bases") or {}).items()}
    form_bases.update(table_form_bases(every_table))
    index = NameIndex(names)

    def resolve(unit: str) -> tuple[str, int]:
        return class_of_unit(unit, index, aliases, form_bases)

    groups: dict[str, dict[int, list[str]]] = {}
    unit_class: dict[str, tuple[str, int]] = {}
    for name in sorted(names):
        cls, form = resolve(name)
        unit_class[name] = (cls, form)
        groups.setdefault(cls, {}).setdefault(form, []).append(name)

    creators = creators_of(every_table, names)
    allowed = None if catalogue is None else set(catalogue)
    elixir: dict[str, int] = {}
    card_recs: dict[str, Mapping[str, Any]] = {}
    card_forms: dict[str, set[int]] = {}
    for table in every_table:
        for card, rec, form in _card_records(table):
            card_forms.setdefault(card, set()).add(form)
            if form == FORM_BASE:
                card_recs.setdefault(card, rec)
                if rec.get("elixir") is not None:
                    elixir.setdefault(card, int(rec["elixir"]))

    class_cards: dict[str, dict[str, str]] = {}
    for cls, forms in groups.items():
        merged: dict[str, str] = {}
        for units in forms.values():
            for unit in units:
                for card, rel in creators.get(unit, {}).items():
                    if allowed is not None and card not in allowed:
                        continue
                    if merged.get(card) != "summon":
                        merged[card] = rel
        class_cards[cls] = merged
    makes: dict[str, set[str]] = {}
    for cls, cards in class_cards.items():
        for card in cards:
            makes.setdefault(card, set()).add(cls)

    prev_by_name = {c["name"]: c for c in prev_classes}
    class_info: dict[str, dict[str, Any]] = {}
    for cls, forms in groups.items():
        form_info = {}
        for form in sorted(forms):
            unit = _canonical(cls, form, forms[form], aliases)
            i = info[unit]
            form_info[str(form)] = {"unit": unit, "flying": i["flying"], "building": i["building"],
                                    "deploy_time_ms": i["deploy_time_ms"], "hitpoints": i["hitpoints"]}
        low = form_info[str(min(forms))]
        for f in form_info.values():
            # A form looks like its class: the evolved Cannon is a characters row in the table
            # but a building on screen. Flying stays per form (evolved Royal Hogs fly in).
            f["building"] = low["building"]
        cards = sorted(class_cards[cls].items(),
                       key=lambda kv: (kv[1] != "summon", len(makes.get(kv[0], ())),
                                       elixir.get(kv[0], 99), kv[0]))
        cards_list = [[card, rel] for card, rel in cards]
        if not cards_list and cls in prev_by_name:
            cards_list = [list(x) for x in prev_by_name[cls].get("cards") or []]
        class_info[cls] = {
            "name": cls, "kind": info[low["unit"]]["kind"], "flying": low["flying"],
            "building": low["building"], "deploy_time_ms": low["deploy_time_ms"],
            "hitpoints": low["hitpoints"],
            "primary_card": cards_list[0][0] if cards_list else None,
            "cards": cards_list, "forms": form_info, "katacr": {},
        }

    for old in prev_classes:
        # A class the previous file had and this one does not (a merge folded it into
        # another) keeps its id, retired: no unit maps to it, and no id after it moves.
        if old["name"] not in class_info:
            kept = {k: v for k, v in old.items() if k != "id"}
            kept.update(katacr={}, retired=True)
            class_info[old["name"]] = kept

    # ids: the previous order, then new classes sorted
    order = [c["name"] for c in prev_classes if c["name"] in class_info]
    order += sorted(set(class_info) - set(order))
    ids = {name: i for i, name in enumerate(order)}

    by_key: dict[str, str] = {}
    for cls in sorted(class_info):
        if not class_info[cls].get("retired"):
            by_key.setdefault(norm_key(cls), cls)
    katacr = {name: _katacr_entry(name, class_info, resolve, by_key) for name in KATACR_CLASSES}
    for name, entry in katacr.items():
        if entry["type"] in ("unit", "tower"):
            slots = class_info[entry["class"]]["katacr"]
            slots.setdefault(str(entry["form"]), name)

    spells_of: dict[str, list[str]] = {}
    for name, entry in katacr.items():
        if entry["type"] == "spell":
            spells_of.setdefault(entry["card"], []).append(name)
    card_names = list(catalogue) if catalogue is not None else sorted(card_recs)
    cards_doc = {}
    for card in card_names:
        rec = card_recs.get(card, {})
        made = sorted(makes.get(card, ()), key=lambda c: (class_cards[c].get(card) != "summon", c))
        cards_doc[card] = {"kind": rec.get("kind"), "elixir": elixir.get(card),
                           "flying": (rec.get("flying_height") or 0) > 0,
                           "forms": sorted(card_forms.get(card, {FORM_BASE})),
                           "classes": made, "katacr": spells_of.get(card, [])}

    classes = []
    for name in order:
        c = class_info[name]
        classes.append({"id": ids[name], **c})
    units_doc = {u: [ids[cls], form] for u, (cls, form) in sorted(unit_class.items())}

    doc: dict[str, Any] = {
        "_comment": [
            "Vision classes (E16), written by tools/vision_classes.py; do not edit by hand.",
            "classes[i].id == i; rows write a class as id + 2 (0 pad, 1 unknown).",
            "units: every RoyaleSim unit type name -> [class id, form]; forms 0 base, 1 evo, 2 hero.",
            "primary_card: the creating card (what --live-compat rows use); cards: every creator.",
            "katacr: the 201 KataCR detector classes -> unit / tower / spell / object / ui / pad.",
            "Regenerate: tools/vision_classes.py --engine (ids are kept); check: --check.",
        ],
        "schema": SCHEMA,
        "generator": "tools/vision_classes.py",
        "source": dict(source or {}),
        "attribution": {
            "aliases": "royalegym 0.1.25 obs.SAME_UNIT_ALIASES, MIT, Copyright (c) 2026 RoyaleGym contributors",
            "katacr": f"{KATACR_DATASET['repo']} {KATACR_DATASET['file']}, {KATACR_DATASET['license']}",
        },
        "forms": list(FORM_NAMES),
        "level_multiplier_percent": (_level_multiplier(every_table) if every_table
                                     else previous.get("level_multiplier_percent")),
        "digest": class_digest(order),
        "aliases": dict(SAME_UNIT_ALIASES),
        "vision_merges": dict(VISION_MERGES),
        "user_merges": dict(sorted(user_merges.items())),
        "katacr_dataset": dict(KATACR_DATASET),
        "katacr_names": list(KATACR_CLASSES),
        "classes": classes,
        "units": units_doc,
        "form_bases": {k: [v[0], v[1]] for k, v in sorted(form_bases.items())},
        "cards": cards_doc,
        "katacr": katacr,
    }
    report = {
        "classes": len(classes), "units": len(units_doc),
        "new_classes": [n for n in order if n not in prev_by_name] if prev_classes else [],
        "katacr_unmapped": sorted(n for n, e in katacr.items() if e["type"] == "unmapped"),
        "katacr_missing_form": sorted(n for n, e in katacr.items() if e["type"] in ("unit", "tower")
                                      and str(e["form"]) not in class_info[e["class"]]["forms"]),
        "spell_cards_missing": sorted({e["card"] for e in katacr.values() if e["type"] == "spell"}
                                      - set(cards_doc)),
        "no_creator": sorted(n for n, c in class_info.items()
                             if not c["cards"] and c["kind"] != KIND_TOWER and not c.get("retired")),
        "retired": sorted(n for n, c in class_info.items() if c.get("retired")),
        "without_metadata": sorted(names - described),
    }
    return doc, report


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(", ", ": "))


def dumps(doc: Mapping[str, Any]) -> str:
    """The file's text: one class, unit, card or KataCR entry per line, "\\n" line ends."""
    items = list(doc.items())
    lines = ["{"]
    for i, (key, value) in enumerate(items):
        end = "," if i < len(items) - 1 else ""
        head = f" {json.dumps(key)}: "
        if key in _ROW_KEYS and isinstance(value, list) and value:
            lines.append(head + "[")
            lines += [f"  {_compact(v)}{',' if j < len(value) - 1 else ''}" for j, v in enumerate(value)]
            lines.append(" ]" + end)
        elif key in _ROW_KEYS and isinstance(value, Mapping) and value:
            keys = list(value)
            lines.append(head + "{")
            lines += [f"  {json.dumps(k)}: {_compact(value[k])}{',' if j < len(keys) - 1 else ''}"
                      for j, k in enumerate(keys)]
            lines.append(" }" + end)
        else:
            lines.append(head + _compact(value) + end)
    lines.append("}")
    return "\n".join(lines) + "\n"


def write(doc: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(doc), encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# the runtime table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormInfo:
    """One form of a class: the unit type it is written as, and that unit's own stats."""
    unit: str
    flying: bool
    building: bool
    deploy_time_ms: int
    hitpoints: int | None

    @property
    def deploy_ticks(self) -> int:
        return -(-int(self.deploy_time_ms) // TICK_MS)


@dataclass(frozen=True)
class VisionClass:
    id: int
    name: str
    kind: str
    primary_card: str | None
    cards: tuple[tuple[str, str], ...]
    forms: Mapping[int, FormInfo]
    katacr: Mapping[int, str]
    retired: bool = False            # kept for its id only: no unit maps to it any more

    @property
    def base_form(self) -> int:
        return min(self.forms)

    def form(self, form: int = FORM_BASE) -> FormInfo:
        """That form's info, or the lowest form's when the class has no such form."""
        return self.forms.get(int(form)) or self.forms[self.base_form]

    @property
    def flying(self) -> bool:
        return self.form(self.base_form).flying

    @property
    def building(self) -> bool:
        return self.form(self.base_form).building

    @property
    def deploy_time_ms(self) -> int:
        return self.form(self.base_form).deploy_time_ms

    @property
    def card_names(self) -> tuple[str, ...]:
        return tuple(card for card, _ in self.cards)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VisionClass":
        forms = {int(k): FormInfo(str(v["unit"]), bool(v["flying"]), bool(v["building"]),
                                  int(v["deploy_time_ms"]),
                                  None if v.get("hitpoints") is None else int(v["hitpoints"]))
                 for k, v in (d.get("forms") or {}).items()}
        if not forms:
            raise ValueError(f"class {d.get('name')!r} has no forms")
        return cls(int(d["id"]), str(d["name"]), str(d.get("kind", KIND_TROOP)),
                   d.get("primary_card"), tuple((str(c), str(r)) for c, r in d.get("cards") or ()),
                   MappingProxyType(dict(sorted(forms.items()))),
                   MappingProxyType({int(k): str(v) for k, v in (d.get("katacr") or {}).items()}),
                   bool(d.get("retired", False)))


class Label(NamedTuple):
    """What a detector label means. type: unit, tower, spell, object, ui, pad or unknown."""
    type: str
    class_id: int | None = None
    form: int = FORM_BASE
    card: str | None = None
    name: str | None = None          # the class name for unit / tower labels


@dataclass(frozen=True)
class UnitLookup:
    """unit_type id -> class token, form and flags, for one engine's unit_types_json()."""
    unit_types: tuple[str, ...]
    class_ids: tuple[int, ...]       # -1: not a class of the table
    forms: tuple[int, ...]
    flying: tuple[bool, ...]
    building: tuple[bool, ...]
    deploy_time_ms: tuple[int, ...]
    digest: str                      # the class table's

    def _ok(self, u: int) -> bool:
        return 0 <= int(u) < len(self.class_ids) and self.class_ids[int(u)] >= 0

    def token(self, u: int) -> int:
        """The row token of unit_type u: class id + 2, or UNKNOWN_TOKEN (1)."""
        return self.class_ids[int(u)] + TOKEN_OFFSET if self._ok(u) else UNKNOWN_TOKEN

    def form(self, u: int) -> int:
        return self.forms[int(u)] if self._ok(u) else FORM_BASE

    def form_feature(self, u: int) -> float:
        """The form as one feature: 0 base, 0.5 evo, 1 hero."""
        return self.form(u) / FORM_HERO

    def flags(self, u: int) -> tuple[float, float]:
        """(flying, building) as 0/1 floats, from the class table (what vision can know)."""
        if not self._ok(u):
            return 0.0, 0.0
        return float(self.flying[int(u)]), float(self.building[int(u)])

    @property
    def unknown(self) -> list[str]:
        return [n for n, c in zip(self.unit_types, self.class_ids) if c < 0]


class ClassTable:
    """data/vision/classes.json, loaded.

        table = ClassTable.load()
        table.resolve("Skeleton_EV1")            -> (id of Skeleton, FORM_EVO)
        table.lookup(unit_types).token(u)        row token of an engine unit_type
        table.card_for("Skeleton", deck)         the creating card (live-compat identity)
        table.from_detector("skeleton-evolution") / table.to_katacr("Skeleton", FORM_EVO)
    """

    def __init__(self, doc: Mapping[str, Any], path: Path | None = None):
        if int(doc.get("schema", SCHEMA)) > SCHEMA:
            raise ValueError(f"classes.json schema {doc.get('schema')} is newer than this code ({SCHEMA})")
        self.doc = doc
        self.path = path
        self.classes: tuple[VisionClass, ...] = tuple(VisionClass.from_dict(c) for c in doc["classes"])
        for i, c in enumerate(self.classes):
            if c.id != i:
                raise ValueError(f"classes.json: class {c.name!r} at position {i} has id {c.id}")
        self.by_name = {c.name: c for c in self.classes}
        if len(self.by_name) != len(self.classes):
            raise ValueError("classes.json: duplicate class names")
        self.units: dict[str, tuple[int, int]] = {str(k): (int(v[0]), int(v[1]))
                                                  for k, v in doc["units"].items()}
        self.aliases = {**dict(doc.get("aliases") or SAME_UNIT_ALIASES),
                        **dict(doc.get("vision_merges") or VISION_MERGES),
                        **dict(doc.get("user_merges") or {})}
        self.form_bases = {str(k): (str(v[0]), int(v[1])) for k, v in (doc.get("form_bases") or {}).items()}
        self.index = NameIndex(list(self.units) + list(self.by_name))
        self.cards: dict[str, dict[str, Any]] = {str(k): dict(v) for k, v in (doc.get("cards") or {}).items()}
        self.katacr: dict[str, dict[str, Any]] = {str(k): dict(v) for k, v in (doc.get("katacr") or {}).items()}
        self._katacr_by_key = {}
        for name in self.katacr:
            if not name.startswith("pad_"):
                self._katacr_by_key.setdefault(norm_key(name), name)
        self._class_by_key = {}
        for c in self.classes:
            if not c.retired:
                self._class_by_key.setdefault(norm_key(c.name), c.name)
        self.level_multiplier = tuple(int(m) for m in doc.get("level_multiplier_percent") or ())
        self.digest = class_digest([c.name for c in self.classes])

    # ------------------------------------------------------------ loading
    @classmethod
    def from_file(cls, path: str | Path) -> "ClassTable":
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")), path)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ClassTable":
        """The shipped file (or `path`), read once per path and modification time."""
        path = Path(path) if path is not None else CLASSES_PATH
        key = (str(path.resolve()), path.stat().st_mtime_ns)
        table = _CACHE.get(key)
        if table is None:
            table = _CACHE[key] = cls.from_file(path)
        return table

    def __len__(self) -> int:
        return len(self.classes)

    def __iter__(self):
        return iter(self.classes)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.classes]

    # ------------------------------------------------------------ units -> classes
    def resolve(self, unit: str) -> tuple[int | None, int]:
        """A RoyaleSim unit type name (or a class name) -> (class id or None, form). Names
        the file does not list go through the same rules the file was built with."""
        hit = self.units.get(unit)
        if hit is not None:
            return hit
        cls, form = class_of_unit(unit, self.index, self.aliases, self.form_bases)
        c = self.by_name.get(cls)
        return (None if c is None else c.id), form

    def get(self, key: Any) -> VisionClass | None:
        """A class by id, name, unit type name or detector label; None if it is none."""
        if isinstance(key, VisionClass):
            return key
        if not isinstance(key, (str, bool)) and hasattr(key, "__index__"):    # int, numpy int
            i = key.__index__()
            return self.classes[i] if 0 <= i < len(self.classes) else None
        key = str(key)
        if key in self.by_name:
            return self.by_name[key]
        cid, _ = self.resolve(key)
        if cid is None:
            cid = self.from_detector(key).class_id
        return None if cid is None else self.classes[cid]

    def __getitem__(self, key: Any) -> VisionClass:
        c = self.get(key)
        if c is None:
            raise KeyError(key)
        return c

    def class_id(self, key: Any) -> int | None:
        c = self.get(key)
        return None if c is None else c.id

    def token(self, key: Any) -> int:
        """Row token: class id + 2, or UNKNOWN_TOKEN."""
        c = self.get(key)
        return UNKNOWN_TOKEN if c is None else c.id + TOKEN_OFFSET

    def has_form(self, key: Any, form: int) -> bool:
        c = self.get(key)
        return c is not None and int(form) in c.forms

    def unit_name(self, key: Any, form: int = FORM_BASE) -> str | None:
        """The unit type a (class, form) is written as (to spawn it, or to name it in rows);
        the class's lowest form when it has no such form."""
        c = self.get(key)
        return None if c is None else c.form(form).unit

    def lookup(self, unit_types: Sequence[str]) -> UnitLookup:
        """UnitLookup for an engine's unit_types_json() (names in unit_type id order)."""
        ids, forms, fly, bld, dep = [], [], [], [], []
        for name in unit_types:
            cid, form = self.resolve(str(name))
            info = self.classes[cid].form(form) if cid is not None else None
            ids.append(-1 if cid is None else cid)
            forms.append(form if cid is not None else FORM_BASE)
            fly.append(bool(info and info.flying))
            bld.append(bool(info and info.building))
            dep.append(info.deploy_time_ms if info else 0)
        return UnitLookup(tuple(str(n) for n in unit_types), tuple(ids), tuple(forms), tuple(fly),
                          tuple(bld), tuple(dep), self.digest)

    def hitpoints(self, key: Any, form: int = FORM_BASE, level: int = FRIENDLY_LEVEL) -> int | None:
        """Hit points at `level`: floor(level-1 hp * multiplier / 100), the card table's
        level_scaling (floor rounding, measured on 16.402 captures). Derived; the engine's
        unit_hitpoints (E06, data/rs_rows/spawn_primary.json "hp") is authoritative. None for
        crown towers: they scale with the tower level on their own table (level 11: princess
        3052, king 4824, not 1400 / 2400 times the card multiplier)."""
        c = self.get(key)
        if (c is None or c.kind == KIND_TOWER or c.form(form).hitpoints is None
                or not self.level_multiplier):
            return None
        level = max(1, min(int(level), len(self.level_multiplier)))
        return c.form(form).hitpoints * self.level_multiplier[level - 1] // 100

    # ------------------------------------------------------------ classes -> cards
    def card_for(self, key: Any, deck: Iterable[str] | None = None, *, strict: bool = False) -> str | None:
        """The card that created a unit of this class: the first of its creators in `deck`
        (catalogue names), else its primary card (None with strict=True)."""
        c = self.get(key)
        if c is None:
            return None
        if deck is not None:
            deck = set(deck)
            for card in c.card_names:
                if card in deck:
                    return card
            if strict:
                return None
        return c.primary_card

    def card_id_for(self, key: Any, catalogue: Sequence[Any], deck: Iterable[str] | None = None,
                    *, strict: bool = False) -> int:
        """card_for as a catalogue id (catalogue: names, or catalogue_json() rows); -1 if none."""
        card = self.card_for(key, deck, strict=strict)
        if card is None:
            return -1
        for i, row in enumerate(catalogue):
            if (row[0] if isinstance(row, (list, tuple)) else row) == card:
                return i
        return -1

    def deck_classes(self, deck: Iterable[str]) -> frozenset[str]:
        """Class names any card of `deck` puts on the board (retired classes left out)."""
        deck = set(deck)
        return frozenset(c.name for c in self.classes
                         if not c.retired and deck & set(c.card_names))

    def deck_labels(self, deck: Iterable[str]) -> frozenset[str]:
        """Every label a deck's cards can show: class names, their forms' detector keys, their
        KataCR names, and the deck's spells as card names (what scripts/vision/detector.py
        puts in Detection.cls for a spell: "Fireball") and KataCR names ("fireball"). This
        is the tracker's own_classes: a label outside it is always an enemy."""
        deck = set(deck)
        out: set[str] = set()
        for name in self.deck_classes(deck):
            c = self.by_name[name]
            out.add(name)
            out.update(f.unit for f in c.forms.values())
            out.update(c.katacr.values())
        for card in deck:
            info = self.cards.get(card) or {}
            out.update(info.get("katacr") or ())
            if info.get("kind") == "spell":
                out.add(card)
        return frozenset(out)

    # ------------------------------------------------------------ detector crosswalk
    def _label(self, entry: Mapping[str, Any], name: str) -> Label:
        typ = str(entry.get("type", "unknown"))
        if typ in ("unit", "tower"):
            c = self.by_name.get(entry.get("class"))
            if c is None:
                return Label("unknown", name=name)
            return Label(typ, c.id, int(entry.get("form", FORM_BASE)), None, c.name)
        if typ == "spell":
            return Label("spell", None, int(entry.get("form", FORM_BASE)), entry.get("card"))
        if typ in ("object", "ui", "pad"):
            return Label(typ)
        return Label("unknown", name=name)       # "unmapped" (or a type this code lacks)

    def _unit_label(self, unit: str) -> Label:
        cid, form = self.resolve(unit)
        if cid is None:
            return Label("unknown", name=unit)
        c = self.classes[cid]
        return Label(KIND_TOWER if c.kind == KIND_TOWER else "unit", cid, form, None, c.name)

    def from_detector(self, label: str) -> Label:
        """A detector label -> Label. Accepts KataCR names ("skeleton-evolution"), this
        table's detector keys and class names ("Skeleton_EV1", "MusketeerHero"), spell card
        names ("Fireball"), and the same up to case and punctuation ("hog_rider")."""
        label = str(label)
        entry = self.katacr.get(label)
        if entry is not None:
            return self._label(entry, label)
        if label in self.units or label in self.by_name:
            return self._unit_label(label)
        if (self.cards.get(label) or {}).get("kind") == "spell":
            return Label("spell", None, FORM_BASE, label)
        key = norm_key(label)
        if key in self._katacr_by_key:
            name = self._katacr_by_key[key]
            return self._label(self.katacr[name], name)
        base, form = split_form(label)
        if norm_key(base) in self._class_by_key:
            c = self.by_name[self._class_by_key[norm_key(base)]]
            return Label(KIND_TOWER if c.kind == KIND_TOWER else "unit", c.id, form, None, c.name)
        return Label("unknown", name=label)

    def to_katacr(self, key: Any, form: int = FORM_BASE) -> str | None:
        """The KataCR detector class for (class, form); None when KataCR has none (heroes,
        post-2024 cards and evolutions)."""
        c = self.get(key)
        return None if c is None else c.katacr.get(int(form))

    def detector_key(self, key: Any, form: int = FORM_BASE) -> str | None:
        """This table's own detector label for (class, form): its canonical unit name; None
        when the class has no such form."""
        c = self.get(key)
        if c is None or int(form) not in c.forms:
            return None
        return c.forms[int(form)].unit

    def describe(self) -> str:
        forms = sum(len(c.forms) for c in self.classes)
        kat = {}
        for e in self.katacr.values():
            kat[e.get("type")] = kat.get(e.get("type"), 0) + 1
        src = self.doc.get("source") or {}
        return (f"{len(self.classes)} classes ({forms} class forms), {len(self.units)} unit names, "
                f"digest {self.digest[:12]}; KataCR {', '.join(f'{k} {v}' for k, v in sorted(kat.items()))}; "
                f"from {src.get('unit_names', '?')} of {', '.join(src.get('card_tables') or []) or '?'}")


_CACHE: dict[tuple[str, int], ClassTable] = {}


def load(path: str | Path | None = None) -> ClassTable:
    """ClassTable.load(path)."""
    return ClassTable.load(path)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _rs_engine():
    """tools/rs_engine.py under the name the tools import it by, without putting tools/ on
    sys.path (the module already imported, if it is)."""
    mod = sys.modules.get("rs_engine")
    if mod is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("rs_engine", ROOT / "tools" / "rs_engine.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rs_engine"] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop("rs_engine", None)
            raise
    return mod


def _engine(args):
    """(rs_engine, royalesim, battle, cfg) under the tool's engine arguments."""
    rs_engine = _rs_engine()
    import royalesim
    cfg = rs_engine.config_from_args(args).undelayed()
    return rs_engine, royalesim, rs_engine.make_battle(cfg, royalesim), cfg


def read_katacr_yaml(path: str | Path) -> list[str]:
    """Class names in id order from a dataset yaml (`  id: name` lines under `names:`)."""
    names: dict[int, str] = {}
    inside = False
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("names:"):
            inside = True
            continue
        if inside:
            m = re.match(r"\s+(\d+):\s*(.+?)\s*$", line)
            if m:
                names[int(m.group(1))] = m.group(2)
            elif line.strip() and not line.startswith(" "):
                break
    return [names[i] for i in sorted(names)]


def unit_types_json(cfg: Any = None) -> list[str]:
    """The engine's unit_types_json() under `cfg` (tools/rs_engine.py; needs royalesim)."""
    return _rs_engine().unit_types(cfg)


def _print_report(report: Mapping[str, Any]) -> None:
    for key, value in report.items():
        if isinstance(value, list):
            shown = ", ".join(value[:25]) + (f", ... ({len(value)})" if len(value) > 25 else "")
            print(f"  {key}: {shown or '-'}")
        else:
            print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Vision class list (data/vision/classes.json)")
    ap.add_argument("--classes", type=Path, default=CLASSES_PATH,
                    help="the class file to read, and whose ids a regeneration keeps")
    ap.add_argument("--out", type=Path, default=None, help="where to write (default: --classes)")
    ap.add_argument("--card-json", type=Path, action="append", default=[], metavar="FILE",
                    help="a RoyaleSim card table (data/derived/cards*.json), preferred first; "
                         "regenerates from these")
    ap.add_argument("--engine", action="store_true",
                    help="regenerate with the engine's unit_types_json() and catalogue")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 unless every engine unit type resolves to a class of the file")
    ap.add_argument("--fresh-ids", action="store_true",
                    help="start over: ignore the old file (its ids, classes and user merges)")
    ap.add_argument("--merge", action="append", default=[], metavar="UNIT=TARGET",
                    help="a user merge: UNIT is written as TARGET's class (kept in the file)")
    ap.add_argument("--katacr-yaml", type=Path, default=None,
                    help="the dataset's ClashRoyale_detection.yaml, to check KATACR_CLASSES")
    ap.add_argument("--note", default=None, help="a line for the file's source.note")
    _rs_engine().add_engine_args(ap)
    args = ap.parse_args(argv)

    if args.katacr_yaml is not None:
        got = read_katacr_yaml(args.katacr_yaml)
        if got != list(KATACR_CLASSES):
            diff = [(i, a, b) for i, (a, b) in enumerate(zip(got, KATACR_CLASSES)) if a != b]
            print(f"KataCR yaml differs: {len(got)} names vs {len(KATACR_CLASSES)}; first: {diff[:5]}")
            return 1
        print(f"KataCR yaml matches ({len(got)} classes)")

    if args.check:
        table = ClassTable.load(args.classes)
        _engine_mod, _rs, battle, cfg = _engine(args)
        names = json.loads(battle.unit_types_json())
        lut = table.lookup(names)
        print(f"{table.describe()}\nengine: {cfg.describe()}: {len(names)} unit types, "
              f"{len(lut.unknown)} unknown")
        for name in lut.unknown:
            print(f"  unknown: {name}")
        return 1 if lut.unknown else 0

    merges = {}
    for text in args.merge:
        unit, sep, target = text.partition("=")
        if not sep or not unit.strip() or not target.strip():
            ap.error(f"--merge wants UNIT=TARGET, got {text!r}")
        merges[unit.strip()] = target.strip()
    if merges and not (args.engine or args.card_json):
        ap.error("--merge takes effect when regenerating: add --engine or --card-json")

    if not (args.engine or args.card_json):
        table = ClassTable.load(args.classes)
        print(table.describe())
        return 0

    tables = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.card_json]
    fallback: list = []
    previous = None
    if args.classes.exists() and not args.fresh_ids:
        previous = json.loads(args.classes.read_text(encoding="utf-8"))
    source: dict[str, Any] = {}
    catalogue = None
    if args.engine:
        rs_engine_mod, royalesim, battle, cfg = _engine(args)
        names = json.loads(battle.unit_types_json())
        catalogue = [str(row[0]) for row in json.loads(battle.catalogue_json())]
        if not tables:
            derived = Path(royalesim.data_dir()) / "derived" / "cards.json"
            if derived.is_file():
                fallback = [json.loads(derived.read_text(encoding="utf-8"))]
        fp = rs_engine_mod.fingerprint(battle, royalesim, cfg)
        source = {"unit_names": "engine unit_types_json()",
                  "engine": {k: fp.get(k) for k in ("version", "card_table", "unit_types",
                                                    "unit_types_digest", "catalogue_sha1")}}
    else:
        names = sorted({n for t in tables for n in table_unit_names(t)})
        source = {"unit_names": "card-table records (troops, buildings, crown towers, hero units)"}
    source["card_tables"] = [str(t.get("version", "?")) for t in tables + fallback]
    if args.note:
        source["note"] = args.note
    doc, report = build(names, tables, fallback_tables=fallback, previous=previous,
                        merges=merges, catalogue=catalogue, source=source)
    out = write(doc, args.out or args.classes)
    print(f"wrote {out}: {ClassTable(doc).describe()}")
    _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
