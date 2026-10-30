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
                    MightyMiner -> MightyMinerBomb

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

Not read: what hero abilities and evolution blocks create (the hero Musketeer's turret, the
evolved Skeleton Army's General); no detector label exists for those units yet, and some
name their own hero's unit (the hero Dark Prince's walking form), which as a child would
explain a second play of the same card away.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
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
             "summon_character", "attached_character", "spawns", "spawn", "morph"}
SKIP = {"raw", "defaults_applied", "level_scaling", "overlays", "source_table"}
UNIT_FIELDS = ("spawner", "death_spawn", "death_spawn_projectile", "spawn_area_object",
               "death_area_effect", "projectile", "attached_character", "area_effect_on_hit",
               "spawn_pathfind")


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


class Walker:
    """Collects the unit names an object spawns, following projectile and area names."""

    def __init__(self, data: dict):
        self.units = data["units"]
        self.heroes = {h["summon_character"]: h for h in data.get("hero_forms", [])
                       if h.get("summon_character")}
        self.tables = (("p", data["projectiles"]), ("a", data["area_effect_objects"]))

    def is_unit(self, name: str) -> bool:
        return name in self.units or name in self.heroes

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
            s = obj.split(":", 1)[1] if obj.startswith("CharacterType:") else obj
            if key in UNIT_KEYS and self.is_unit(s):
                found.add(s)
            else:
                self.follow(s, found, seen)

    def follow(self, name, found: set, seen: set) -> None:
        for tag, table in self.tables:
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


def ability_units(card: Mapping[str, Any], deployed: Mapping[str, Any]) -> set[str]:
    """Units a champion card's ability creates (ability.effect.unit): the Little Prince's
    ChampionGuard, the Skeleton King's SkeletonKingSkeleton, the Mighty Miner's
    MightyMinerBomb. Not a unit the card deploys itself (`deployed`): Goblinstein's tether
    names its doctor, which is part of the play."""
    ab = card.get("ability")
    effect = ab.get("effect") if isinstance(ab, Mapping) else None
    unit = effect.get("unit") if isinstance(effect, Mapping) else None
    if not isinstance(unit, str) or not unit or unit in deployed:
        return set()
    return {unit}


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
    return {"kinds": kinds, "compositions": compositions, "variants": variants,
            "spell_objects": spell_objects, "spell_children": spell_children,
            "children": children}


def _with_lookalikes(created: Sequence[str]) -> list[str]:
    return sorted(set(created) | {LOOKS_LIKE[u] for u in created if u in LOOKS_LIKE})


def merge(parts: Sequence[Mapping[str, Any]]) -> dict:
    """build_one sections of several tables, the first first: kinds, compositions and
    variants per card from the first table that has the card; spell objects, spell
    children and children the union over all tables (the first table's objects in its
    order, then the others'); then LOOKS_LIKE on every list of created units."""
    out: dict[str, dict] = {k: {} for k in ("kinds", "compositions", "variants", "spell_objects",
                                            "spell_children", "children")}
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
    for sec in ("spell_children", "children"):
        out[sec] = {name: _with_lookalikes(kids) for name, kids in out[sec].items()}
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
