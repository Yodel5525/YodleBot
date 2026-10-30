"""Write data/vision/spawn_children.json: which units each card deploys and which units
create others, from RoyaleSim's card data (V04's EventTables.from_names reads it).

    python tools/vision_spawn_children.py [--card-table TABLE] [--out PATH]
    python tools/vision_spawn_children.py --cards T.json [--cards OLDER.json ...] [--out PATH]

Nothing of the engine runs: the card tables are read as JSON, and royalesim is located
without importing it. By default the table data/engine.json selects (16.402.22,
cards-160402017-20261019.1; --card-table names another, as tools/rs_engine.py reads it) is
read from royalesim's data/derived/ when the package has the file (an sdist or a checkout),
else from the copy compiled into its extension (royalesim.abi3.so / royalesim.pyd carries
every table it can run as JSON text; tools/vision_classes.embedded_table finds it), and the
wheel's own data/derived/cards.json (15.535.29) after it. With --cards, those files in that
order. A LATER TABLE ONLY ADDS: a card, a spell object or a created unit the first table
lacks (the 15.535 Goblin Barrel's Goblin, the children of event objects 16.402 dropped) is
kept, so a regeneration never loses an entry. Names are RoyaleSim card and unit names:

    kinds           card -> "troop" | "building" | "spell"
    compositions    card -> {unit: count one play deploys}: a troop or building card's
                    summon_character x count, plus second_summon, plus the unit that
                    summon turns into (spawn_pathfind.morph: Goblin Drill -> {GoblinDrillDig:
                    1, GoblinDrill: 1}; the dig shows as KataCR's `dirt`, an object no track
                    follows, so the building is the play's first track); and a spell that
                    deploys a troop (spell.spell_as_deploy with spell.summon): Heal ->
                    {HealSpirit: 1}
    variants        a card with forms -> {form card: its elixir}, from spell.variant.options:
                    the Spirit Empress MergeMaiden plays MergeMaiden_Normal (3) below 6 elixir
                    and MergeMaiden_Mounted (6) at 6 or more (royalesim state.rs resolve_play;
                    calibration match.VARIANT_*). The deck, the hand, the command and every
                    play our rows record name MergeMaiden, so EventTables.from_names reports
                    a form's units as the card, at the form's cost.
    spell_objects   spell card -> its projectile and area-effect object names
    spell_children  spell card -> units it spawns (Goblin Barrel -> BarrelGoblin, Goblin)
    children        unit -> units it creates while alive or as it dies (spawner, death
                    spawn, projectile and area spawns, the unit it turns into): Witch ->
                    Skeleton, GoblinDrillDig -> GoblinDrill; and the units a champion's
                    ability creates, by the champion's unit (ability.effect.unit, unless the
                    card deploys that unit itself, as Goblinstein's tether names its doctor):
                    LittlePrince -> ChampionGuard, SkeletonKing -> SkeletonKingSkeleton,
                    MightyMiner -> MightyMinerBomb. Then (SPAWNERS below) the newer spawner
                    fields, the evolution records, the hero forms' abilities, and each
                    unit's descendants.
    child_reach     unit -> {child unit: millitiles}: how far from it a child its attack
                    projectile spawns can land (REACH below): Mortar_EV1 {Goblin: 13600},
                    WitchMother {VoodooHog: 7500}. EventTables reads it per parent and child
                    class, a reach beyond events.EventConfig.child_radius_mt for that child
                    only (the Witch's skeletons and bats keep the radius).

SPAWNERS (16.402's newer data). A unit record's fields interval_spawner (the reworked
Furnace's Fire Spirits every 5 s for its whole life, the evolved Goblin Giant's Goblins below
half HP), life_state_spawner (the Goblin Hut's thrower), transform_at_hp (the Goblin
Demolisher's kamikaze form, the Cannon Cart's broken cannon), stage_schedule (the Goblin Rocket
Silo's stages; "into" names the unit) and action_graph (its "spawns": "CharacterType:X" is a
unit, "ProjectileType:X" / "AreaEffectType:X" an object followed for what it spawns) are read
like the older ones. EVOLUTIONS ("evolutions"): the engine builds an evolved unit from its
evolution record (royalesim card.rs load_evolution), so each record is read by the unit it
summons, its own unit fields (GoblinGiant_EV1's interval spawner and spear goblin riders,
Witch_EV1's skeleton spawner, Wallbreaker_EV1's death spawn exist only there) and its evo_*
blocks' EVO_UNIT_KEYS (evo_soul_drain.unit, evo_army.spectral, evo_ghost.left / .right, ...),
EVO_RECORD_KEYS (the evolved Skeleton Barrel's dropped barrels) and projectile; an evolved
spell's units are its card's spell_children (the evolved Goblin Barrel's decoy GoblinDummy).
Left out, as units the evolution does not create: form_of, cloned_version, evo_ghost.clone (the
copy a Clone spell makes), evo_army.general (deployed with the play: a composition's, not a
child) and evo_fall.grounded (the evolved Royal Hog itself once it lands: a child of the same
vision class would explain a second Royal Hogs play away). HEROES: the units a hero form's
ability creates (ABILITY_UNIT_KEYS anywhere in ability.effect: the hero Musketeer's turret, the
hero Balloon's trooper, the hero Dark Prince's walking form and mount; a spell hero's go to
its card's spell_children), unless it is the hero's own unit. RECORDS: a troop or building
record a hero form (tables.units) or an evolution (evo_barrel.extra) carries inside its own is
a unit as well (the hero Balloon's death spawn BalloonHero_Bomb), with children of its own (the
dropped barrels' Skeletons); data/vision/classes.json counts the same records
(tools/vision_classes.table_unit_records). DESCENDANTS: each unit also lists what its children
create, at unit level (SkeletonBalloon -> SkeletonContainerNew -> Skeleton gives SkeletonBalloon
-> Skeleton: the falling barrel has no detector label, so on screen the skeletons' parent is the
balloon); a spell's spawns list theirs too. REACH: a child created by its parent's attack
projectile lands where the shot does, not next to the parent: the evolved Mortar's shell drops
a Goblin up to its range (11.5 tiles) away, every shot of its 30 s life, and the Mother Witch's
cursed victim turns into a hog where it dies. child_reach gives such a parent, for each
shot-borne child (and what that child creates in turn, and its look-alike), its range_milli +
its collision_radius_milli + REACH_SLACK_MT. merge() takes each (parent, child) reach from the
first table that has it, and works out the descendants (and the reach's) again over the merged
relations, so a chain split across tables keeps its indirect relation. Not read: an evolved or
hero spell's own projectile and area names (spell_objects keeps the base card's: the detector
names an evolved spell by its KataCR label, zap-evolution, and StateTracks an engine spell by
its card, whose least class name DummyZap_EV1_Visual would change), and summon_members (the Three Musketeers deploy C1, C2 and
C3 once each, but compositions keeps summon_character x count + second_summon, {C1: 3, C2: 1}:
changing an entry is for its own change, docs/NEW_CARDS_PLAN.md 1.7; this one only adds).

VISION LOOK-ALIKES (LOOKS_LIKE). A unit the detector can only box as another unit's class
is listed with that unit wherever it is created. 16.402's Goblin Barrel spawns BarrelGoblin
(15.535's spawned Goblin); KataCR boxes both as `goblin`, which data/vision/classes.json
reads as the class Goblin, and gives BarrelGoblin no KataCR name. So the barrel's
spell_children are both: BarrelGoblin for engine states (events.StateTracks), Goblin for the
screen. Likewise KataCR's one `bomb` label is the class GiantSkeletonBomb, and classes.json
lists BalloonBomb and BombTowerBomb as its "also": a Balloon's or a Bomb Tower's death bomb
is boxed as GiantSkeletonBomb, so they create it too. (ram-rider's "also" RamRider is not a
look-alike: it rides the Ram the label already names, and a Ram creating Ram would explain
a second Ram Rider away.)

Most of the units heroes and evolutions create have no detector label yet (the hero
Musketeer's turret, the evolved Wall Breaker's runner); their entries explain them once one
exists. tests/pipeline/test_events_spawners.py checks that no entry is a child of its own
parent's vision class (data/vision/classes.json), which would explain a second play of the
same card away.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "vision" / "spawn_children.json"

#: The card table data/engine.json selects when no engine config can be read.
DEFAULT_TABLE = "cards-160402017-20261019.1"
#: The wheel's own table (15.535.29), read after the selected one.
FALLBACK_FILE = "cards.json"
#: Units the detector boxes as another unit's class (module doc): data/vision/classes.json's
#: KataCR `goblin` (Goblin) and `bomb` (GiantSkeletonBomb, "also" BalloonBomb, BombTowerBomb).
LOOKS_LIKE: dict[str, str] = {"BarrelGoblin": "Goblin", "BalloonBomb": "GiantSkeletonBomb",
                              "BombTowerBomb": "GiantSkeletonBomb"}
#: File suffixes of a compiled Python extension (royalesim.abi3.so, royalesim.pyd).
EXTENSION_SUFFIXES = (".so", ".pyd", ".dylib")

UNIT_KEYS = {"character", "character2", "spawn_character", "death_spawn_character",
             "summon_character", "attached_character", "spawns", "spawn", "morph", "into"}
SKIP = {"raw", "defaults_applied", "level_scaling", "overlays", "source_table"}
UNIT_FIELDS = ("spawner", "death_spawn", "death_spawn_projectile", "spawn_area_object",
               "death_area_effect", "projectile", "attached_character", "area_effect_on_hit",
               "spawn_pathfind",
               # SPAWNERS (module doc): 16.402's newer fields
               "interval_spawner", "life_state_spawner", "transform_at_hp", "stage_schedule",
               "action_graph")
#: Action-graph type prefixes ("CharacterType:FireSpirits") -> what the name is: a unit, a
#: projectile or an area-effect object.
TYPE_PREFIXES = {"CharacterType:": "unit", "ProjectileType:": "projectile",
                 "AreaEffectType:": "area"}
#: Keys of an evolution's evo_* block naming a unit the evolved unit creates (evo_soul_drain.unit,
#: evo_drill.goblin, evo_army.spectral, evo_ghost.left / .right, evo_rage_ghost.ghost,
#: evo_furnace.character); not clone, general or grounded (module doc).
EVO_UNIT_KEYS = frozenset({"unit", "goblin", "spectral", "left", "right", "ghost", "character"})
#: ... holding the record of a unit it creates (evo_barrel.extra / .death, the dropped barrels).
EVO_RECORD_KEYS = frozenset({"extra", "death"})
#: Keys anywhere in a hero's ability.effect naming a unit the ability creates (effect.unit,
#: .air_unit, .mount, .active, cube.unit, spawns[].unit).
ABILITY_UNIT_KEYS = frozenset({"unit", "air_unit", "mount", "active"})
#: Source tables of a troop or building record (its "source_table"), as tools/vision_classes reads.
RECORD_TABLES = ("characters", "buildings")
#: child_reach (module doc, REACH) beyond a shooter's range and its own collision radius: its
#: target's collision radius (a princess tower's 1000, a king tower's 1400) and the error of a
#: birth point the detector reads.
REACH_SLACK_MT = 1500


def package_dir() -> Path:
    """royalesim's package folder, found without importing royalesim."""
    spec = importlib.util.find_spec("royalesim")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("royalesim is not installed here; pass --cards PATH")
    return Path(list(spec.submodule_search_locations)[0])


def default_cards_path() -> Path:
    """royalesim/data/derived/cards.json (the wheel's 15.535.29 table)."""
    return package_dir() / "data" / "derived" / FALLBACK_FILE


def _embedded_table(data: bytes, version: str) -> dict | None:
    """tools/vision_classes.embedded_table (imported here, by name, only when needed)."""
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from vision_classes import embedded_table
    return embedded_table(data, version)


def find_table(version: str, pkg: str | Path | None = None) -> tuple[dict, str]:
    """(the card table whose "version" is `version`, where it was read): a data/derived/
    cards*.json of royalesim's package folder `pkg` (default: the installed one), else the
    copy compiled into its extension. SystemExit when it is in neither."""
    pkg = package_dir() if pkg is None else Path(pkg)
    derived = pkg / "data" / "derived"
    files = sorted(derived.glob("cards*.json")) if derived.is_dir() else []
    for path in sorted(files, key=lambda p: (p.name != FALLBACK_FILE, p.name)):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("version") == version:
            return doc, f"royalesim/data/derived/{path.name}"
    for path in sorted(p for p in pkg.iterdir() if p.suffix.lower() in EXTENSION_SUFFIXES):
        try:
            doc = _embedded_table(path.read_bytes(), version)
        except OSError:
            continue
        if doc is not None:
            return doc, f"royalesim/{path.name} (its compiled-in copy)"
    raise SystemExit(f"card table {version} is neither in {derived} nor compiled into "
                     f"royalesim's extension in {pkg}; pass --cards PATH")


def _is_record(obj: Any) -> bool:
    """A troop or building record: a mapping with a RECORD_TABLES source_table and a name."""
    return (isinstance(obj, Mapping) and obj.get("source_table") in RECORD_TABLES
            and isinstance(obj.get("name"), str) and bool(obj["name"]))


def _records_in(obj: Any):
    """Every troop or building record inside `obj` (obj included), depth first: what
    tools/vision_classes._embedded_records finds in a hero form or an evolution."""
    if isinstance(obj, Mapping):
        if _is_record(obj):
            yield obj
        for v in obj.values():
            yield from _records_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _records_in(v)


def embedded_records(data: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """name -> the troop or building records hero forms and evolutions carry inside their own
    (module doc, RECORDS) that the units table lacks: BalloonHero_Bomb, MusketeerTurret,
    SkeletonBalloonEvoDummyAeO_EXTRA. The first record of a name is kept."""
    units = data.get("units") or {}
    out: dict[str, Mapping[str, Any]] = {}
    for group in ("evolutions", "hero_forms"):           # tools/vision_classes' order
        for form in data.get(group) or ():
            for rec in _records_in(form):
                if rec["name"] not in units:
                    out.setdefault(rec["name"], rec)
    return out


def _typed(name: str) -> tuple[str | None, str]:
    """("unit" | "projectile" | "area" | None, the name without its TYPE_PREFIXES prefix)."""
    for prefix, kind in TYPE_PREFIXES.items():
        if name.startswith(prefix):
            return kind, name[len(prefix):]
    return None, name


class Walker:
    """Collects the unit names an object spawns, following projectile and area names."""

    def __init__(self, data: dict):
        self.units = data["units"]
        self.heroes = {h["summon_character"]: h for h in data.get("hero_forms", [])
                       if h.get("summon_character")}
        self.records = embedded_records(data)
        self.tables = (("p", data["projectiles"]), ("a", data["area_effect_objects"]))

    def is_unit(self, name: str) -> bool:
        return name in self.units or name in self.heroes or name in self.records

    def walk(self, obj, found: set, seen: set, key: str | None = None):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in SKIP:
                    continue
                if k == "name":           # an embedded projectile / area: follow it by name
                    self.follow(v, found, seen)
                    continue
                self.walk(v, found, seen, k)
        elif isinstance(obj, list):
            for v in obj:
                self.walk(v, found, seen, key)
        elif isinstance(obj, str):
            kind, s = _typed(obj)
            if kind in (None, "unit") and key in UNIT_KEYS and self.is_unit(s):
                found.add(s)
            else:
                self.follow(s, found, seen, kind)

    def follow(self, name, found: set, seen: set, kind: str | None = None) -> None:
        """Walk the projectile and the area-effect object named `name` (only the one a
        "ProjectileType:" / "AreaEffectType:" prefix said, `kind`), once each per walk."""
        for tag, table in self.tables:
            if kind == "projectile" and tag != "p" or kind == "area" and tag != "a":
                continue
            if isinstance(name, str) and name in table and (tag, name) not in seen:
                seen.add((tag, name))
                self.walk(table[name], found, seen)

    def unit_children(self, name: str, rec: dict) -> set:
        found: set = set()
        seen: set = set()
        for k in UNIT_FIELDS:
            if rec.get(k):
                self.walk(rec[k], found, seen, k if k == "attached_character" else None)
        found.discard(name)
        return found


def spell_summon(card: Mapping[str, Any]) -> dict[str, int]:
    """{unit: count} of a spell that deploys a troop: spell.spell_as_deploy with a
    spell.summon (Heal -> {HealSpirit: 1}); {} for every other card. Rage's bottle is a
    summon without spell_as_deploy and stays a spell."""
    sp = card.get("spell")
    if not isinstance(sp, Mapping) or not sp.get("spell_as_deploy"):
        return {}
    summon = sp.get("summon")
    if not isinstance(summon, Mapping) or not summon.get("character"):
        return {}
    return {str(summon["character"]): int(summon.get("count") or 1)}


def morph_of(unit: Mapping[str, Any] | None) -> str | None:
    """The unit `unit` turns into where its path ends (spawn_pathfind.morph:
    GoblinDrillDig -> GoblinDrill), else None (the Miner's dig has morph None)."""
    path = unit.get("spawn_pathfind") if isinstance(unit, Mapping) else None
    into = path.get("morph") if isinstance(path, Mapping) else None
    return str(into) if into else None


def _effect_units(obj: Any, key: str | None = None) -> set[str]:
    """The strings under ABILITY_UNIT_KEYS anywhere in an ability effect."""
    if isinstance(obj, Mapping):
        return set().union(*(_effect_units(v, str(k)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return set().union(*(_effect_units(v, key) for v in obj))
    return {obj} if isinstance(obj, str) and obj and key in ABILITY_UNIT_KEYS else set()


def ability_units(card: Mapping[str, Any], deployed: Mapping[str, Any] | Iterable[str]) -> set[str]:
    """Units a champion card's or a hero form's ability creates (ABILITY_UNIT_KEYS in
    ability.effect): the Little Prince's ChampionGuard, the Skeleton King's
    SkeletonKingSkeleton, the Mighty Miner's MightyMinerBomb; the hero Musketeer's
    MusketeerTurret, the hero Dark Prince's DarkPrinceHero_Walking and DarkPrinceHero_Mount, the
    hero Ice Wizard's IceWizardHero_IceCube (effect.cube.unit). Not a unit the card deploys
    itself (`deployed`): Goblinstein's tether names its doctor, which is part of the play."""
    ab = card.get("ability")
    effect = ab.get("effect") if isinstance(ab, Mapping) else None
    if not isinstance(effect, Mapping):
        return set()
    return _effect_units(effect) - set(deployed)


def evo_block_units(w: "Walker", evo: Mapping[str, Any]) -> set[str]:
    """Units the evo_* blocks of evolution record `evo` create (module doc, EVOLUTIONS): an
    EVO_UNIT_KEYS name (Witch_EV1_Skeleton_Interval, Ghost_EV1_Summon_Left, Wallbreaker_mini),
    an EVO_RECORD_KEYS record (SkeletonBalloonEvoDummyAeO_EXTRA) and what a block's projectile
    spawns (the evolved Goblin Barrel's decoy: GoblinDummy)."""
    found: set = set()
    seen: set = set()
    for k, block in evo.items():
        if not str(k).startswith("evo_") or not isinstance(block, Mapping):
            continue
        for kk, v in block.items():
            if kk in EVO_UNIT_KEYS and isinstance(v, str) and w.is_unit(v):
                found.add(v)
            elif kk in EVO_RECORD_KEYS and _is_record(v):
                found.add(v["name"])
            elif kk == "projectile" and isinstance(v, str):
                w.follow(_typed(v)[1], found, seen)
    return found


def _add(table: dict[str, list[str]], name: str, kids: Iterable[str]) -> None:
    kids = set(kids) - {name}
    if kids:
        table[name] = sorted(set(table.get(name, ())) | kids)


def with_descendants(children: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """Each unit's children plus everything they create in turn (module doc, DESCENDANTS),
    at unit level, never the unit itself (a respawn loop)."""
    out: dict[str, list[str]] = {}
    for name, kids in children.items():
        reach: set = set()
        stack = list(kids)
        while stack:
            c = stack.pop()
            if c not in reach:
                reach.add(c)
                stack.extend(children.get(c, ()))
        reach.discard(name)
        out[name] = sorted(reach)
    return out


def card_variants(card: Mapping[str, Any], elixir: Mapping[str, Any]) -> dict[str, int]:
    """{form card: its elixir} of a card with forms (spell.variant.options; the Spirit
    Empress), {} for every other card. A form's cost is its own card's elixir, what the
    engine charges for the play (state.rs resolve_play), else its trigger in whole elixir."""
    sp = card.get("spell")
    variant = sp.get("variant") if isinstance(sp, Mapping) else None
    if not isinstance(variant, Mapping):
        return {}
    out: dict[str, int] = {}
    for opt in variant.get("options") or ():
        name = opt.get("card") if isinstance(opt, Mapping) else None
        if not name:
            continue
        cost = elixir.get(name)
        if cost is None and opt.get("trigger_milli") is not None:
            cost = int(opt["trigger_milli"]) // 1000
        if cost is not None:
            out[str(name)] = int(cost)
    return out


def build_one(data: dict) -> dict:
    """The relations of one card table (build's sections, no about or source)."""
    w = Walker(data)
    children: dict[str, list[str]] = {}
    for name, rec in sorted(w.units.items()):
        kids = w.unit_children(name, rec)
        if kids:
            children[name] = sorted(kids)
    for name, rec in sorted(w.heroes.items()):
        kids = w.unit_children(name, rec)
        if kids:
            children[name] = sorted(set(children.get(name, ())) | kids)
    elixir = {c["name"]: c.get("elixir") for c in data["cards"]}
    kinds, compositions, variants, spell_objects, spell_children = {}, {}, {}, {}, {}
    for c in data["cards"]:
        name, kind = c["name"], c["kind"]
        kinds[name] = kind
        if kind != "spell":
            comp: dict[str, int] = {}
            sc = c.get("summon_character")
            if sc:
                comp[sc] = int(c.get("count") or 1)
                into = morph_of(w.units.get(sc))
                if into and into != sc and w.is_unit(into):
                    comp[into] = comp.get(into, 0) + comp[sc]
            ss = c.get("second_summon")
            if isinstance(ss, dict) and ss.get("character"):
                comp[ss["character"]] = comp.get(ss["character"], 0) + int(ss.get("count") or 1)
            compositions[name] = comp
            made = {u for u in ability_units(c, comp) if u != sc and w.is_unit(u)}
            if sc and made:
                children[sc] = sorted(set(children.get(sc, ())) | made)
            continue
        deploys = spell_summon(c)
        if deploys:
            compositions[name] = deploys
        forms = card_variants(c, elixir)
        if forms:
            variants[name] = forms
        found: set = set()
        seen: set = set()
        for k, v in c.items():
            if k in SKIP or k in ("name", "display_name", "kind", "rarity"):
                continue
            w.walk(v, found, seen, k)
        spell_objects[name] = (sorted(n for t, n in seen if t == "p")
                               + sorted(n for t, n in seen if t == "a"))
        if found:
            spell_children[name] = sorted(found)
    # module doc, SPAWNERS: the embedded records, the heroes' abilities, the evolutions
    for name, rec in sorted(w.records.items()):
        _add(children, name, w.unit_children(name, rec))
    for name, hero in sorted(w.heroes.items()):
        _add(children, name, {u for u in ability_units(hero, {name}) if w.is_unit(u)})
    spell_forms: dict[str, set] = {}
    for evo in data.get("evolutions") or ():
        unit = evo.get("summon_character")
        if unit:
            _add(children, unit, w.unit_children(unit, evo) | evo_block_units(w, evo))
        elif evo.get("form_of") in kinds:                 # an evolved spell: its card's spawns
            spell_forms.setdefault(evo["form_of"], set()).update(
                form_spell_units(w, evo) | evo_block_units(w, evo))
    for hero in data.get("hero_forms") or ():
        if not hero.get("summon_character") and hero.get("form_of") in kinds:     # a spell hero
            spell_forms.setdefault(hero["form_of"], set()).update(
                form_spell_units(w, hero) | {u for u in ability_units(hero, ()) if w.is_unit(u)})
    for name, units in sorted(spell_forms.items()):
        _add(spell_children, name, units)
    # REACH: a shooter's shot-borne spawns land where its shots do
    child_reach: dict[str, dict[str, int]] = {}
    shooters = [*w.units.items(), *w.records.items(), *w.heroes.items(),
                *((e["summon_character"], e) for e in data.get("evolutions") or ()
                  if e.get("summon_character"))]
    for name, rec in shooters:
        got = shot_reach(w, name, rec)
        if got is not None:
            mt, kids = got
            have = child_reach.setdefault(name, {})
            for kid in kids:
                have[kid] = max(have.get(kid, 0), mt)
    # DESCENDANTS: what a unit's (or a spell's) children create counts as its own
    children = with_descendants(children)
    for name, kids in spell_children.items():
        spell_children[name] = sorted(set(kids).union(*(children.get(k, ()) for k in kids)))
    return {"kinds": kinds, "compositions": compositions, "variants": variants,
            "spell_objects": spell_objects, "spell_children": spell_children,
            "children": children, "child_reach": dict(sorted(child_reach.items()))}


def shot_reach(w: Walker, name: str, rec: Mapping[str, Any]) -> tuple[int, set[str]] | None:
    """(reach, the units its shot spawns) of unit `name` (module doc, REACH): when what its
    attack projectile spawns includes a unit (the evolved Mortar's shell -> Goblin, the Mother
    Witch's curse -> VoodooHog), the farthest such a child lands from it, its range_milli, its
    own collision_radius_milli and REACH_SLACK_MT, with those units; else None."""
    proj, reach = rec.get("projectile"), rec.get("range_milli")
    if not proj or isinstance(reach, bool) or not isinstance(reach, (int, float)) or reach <= 0:
        return None
    found: set = set()
    w.walk(proj, found, set(), None)
    found.discard(name)
    if not found:
        return None
    return int(reach) + int(rec.get("collision_radius_milli") or 0) + REACH_SLACK_MT, found


def form_spell_units(w: Walker, form: Mapping[str, Any]) -> set[str]:
    """Units an evolved or hero spell's own record spawns (GoblinBarrel_EV1's projectile ->
    BarrelGoblin, the hero Barbarian Barrel's re-roll -> BarbLogBarbarianHero), walked like a
    spell card but for the names of what it is a form of, its embedded tables and its ability
    (ability_units reads that). Its objects are not spell_objects (module doc, Not read)."""
    found: set = set()
    seen: set = set()
    for k, v in form.items():
        if k in SKIP or k in ("name", "display_name", "kind", "rarity", "form_of", "cloned_version",
                              "tables", "ability"):
            continue
        w.walk(v, found, seen, k)
    return found


def _with_lookalikes(created: Sequence[str]) -> list[str]:
    return sorted(set(created) | {LOOKS_LIKE[u] for u in created if u in LOOKS_LIKE})


def merge(parts: Sequence[Mapping[str, Any]]) -> dict:
    """build_one sections of several tables, the first first: kinds, compositions and
    variants per card (child_reach per unit and child) from the first table that has it;
    spell objects, spell children and children the union over all tables (the first table's
    objects in its order, then the others'); then the descendants over the union (a chain
    split across tables: X -> A in one, A -> B in another gives X -> B), a shot-borne child's
    descendants with its parent's reach; then LOOKS_LIKE on every list of created units and
    on child_reach's children."""
    out: dict[str, dict] = {k: {} for k in ("kinds", "compositions", "variants", "spell_objects",
                                            "spell_children", "children")}
    reach: dict[str, dict[str, int]] = {}
    for part in parts:
        for sec in ("kinds", "compositions", "variants"):
            for name, value in part.get(sec, {}).items():
                out[sec].setdefault(name, value)
        for name, objs in part.get("spell_objects", {}).items():
            have = out["spell_objects"].setdefault(name, [])
            for o in objs:
                if o not in have:
                    have.append(o)
        for sec in ("spell_children", "children"):
            for name, kids in part.get(sec, {}).items():
                out[sec][name] = sorted(set(out[sec].get(name, ())) | set(kids))
        for name, kids in part.get("child_reach", {}).items():
            have = reach.setdefault(name, {})
            for kid, mt in kids.items():
                have.setdefault(kid, mt)
    # DESCENDANTS again, over the union (build_one closed each table on its own)
    children = out["children"] = with_descendants(out["children"])
    out["spell_children"] = {name: sorted(set(kids).union(*(children.get(k, ()) for k in kids)))
                             for name, kids in out["spell_children"].items()}
    for name, kids in reach.items():                # what a shot-borne child creates lands by it
        for kid, mt in list(kids.items()):
            for grand in children.get(kid, ()):
                if grand != name:
                    kids[grand] = max(kids.get(grand, 0), mt)
    for sec in ("spell_children", "children"):
        out[sec] = {name: _with_lookalikes(kids) for name, kids in out[sec].items()}
    for name, kids in reach.items():
        for kid, mt in list(kids.items()):
            if kid in LOOKS_LIKE:
                kids[LOOKS_LIKE[kid]] = max(kids.get(LOOKS_LIKE[kid], 0), mt)
    out["child_reach"] = {name: dict(sorted(kids.items())) for name, kids in sorted(reach.items()) if kids}
    return out


def build(data: dict | Sequence[dict], *, files: Sequence[str] | None = None,
          package: str | None = "royalesim==0.1.42") -> dict:
    """spawn_children.json of one card table, or of several (the first first; later ones
    only add, merge). files: where each table was read, for the source note."""
    tables = [data] if isinstance(data, Mapping) else list(data)
    if not tables:
        raise ValueError("build needs at least one card table")
    if files is None:
        files = ["royalesim/data/derived/" + FALLBACK_FILE] if len(tables) == 1 else []
    files = list(files) + ["?"] * (len(tables) - len(files))
    doc = merge([build_one(t) for t in tables])
    read = [{"version": t.get("version"), "file": f} for t, f in zip(tables, files)]
    source: dict[str, Any] = {"file": read[0]["file"], "version": read[0]["version"],
                              "package": package}
    if len(read) > 1:
        source["tables"] = read
        source["note"] = (f"{read[0]['version']} first, then "
                          + ", ".join(str(r["version"]) for r in read[1:])
                          + ", which only add the cards, spell objects and created units "
                            "the first lacks")
    return {
        "about": ("Spawn relations for scripts/vision/events.py, generated by "
                  "tools/vision_spawn_children.py from RoyaleSim's card tables by reading the "
                  "JSON only. Names are RoyaleSim card and unit names; map units to vision "
                  "classes (tools/vision_classes) and cards to catalogue ids "
                  "(Battle.catalogue_json) with events.EventTables.from_names, which reports "
                  "a variant's forms as the card."),
        "source": source,
        **doc,
    }


def _selected_tables(args) -> tuple[list[dict], list[str]]:
    """The --card-table / data/engine.json table, then the wheel's own cards.json."""
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import rs_engine
    cfg = rs_engine.config_from_args(args)
    pkg = package_dir()
    tables, files = [], []
    if cfg.card_table:
        version = rs_engine.KNOWN_TABLES.get(cfg.card_table)
        if version is None:
            stem = rs_engine.table_version_stem(cfg.card_table)
            version = f"{stem}.1" if stem else str(cfg.card_table)
        doc, where = find_table(version, pkg)
        tables.append(doc)
        files.append(where)
    own = pkg / "data" / "derived" / FALLBACK_FILE
    if own.is_file():
        doc = json.loads(own.read_text(encoding="utf-8"))
        if not tables or doc.get("version") != tables[0].get("version"):
            tables.append(doc)
            files.append(f"royalesim/data/derived/{FALLBACK_FILE}")
    if not tables:
        raise SystemExit(f"no card table found in {pkg}; pass --cards PATH")
    return tables, files


def _package_version() -> str | None:
    try:
        from importlib.metadata import version
        return f"royalesim=={version('royalesim')}"
    except Exception:                   # not installed: the tables came from --cards
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cards", type=Path, action="append", default=None, metavar="PATH",
                    help="a RoyaleSim card table (cards*.json); repeat it: the first is read, "
                         "later ones only add what it lacks (default: the engine's table, "
                         "then royalesim's data/derived/cards.json)")
    ap.add_argument("--out", type=Path, default=OUT)
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import rs_engine
    rs_engine.add_engine_args(ap)
    args = ap.parse_args(argv)
    if args.cards:
        tables = [json.loads(p.read_text(encoding="utf-8")) for p in args.cards]
        files = [str(p) for p in args.cards]
    else:
        tables, files = _selected_tables(args)
    table = build(tables, files=files, package=_package_version())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(table, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{args.out}: {len(table['kinds'])} cards, {len(table['children'])} spawning units, "
          f"{len(table['spell_children'])} spawning spells, {len(table['variants'])} with forms "
          f"(from {', '.join(str(t.get('version')) for t in tables)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
