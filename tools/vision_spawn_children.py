"""Write data/vision/spawn_children.json: which units each card deploys and which units
create others, from RoyaleSim's card data (V04's EventTables.from_names reads it).

    python tools/vision_spawn_children.py [--cards PATH] [--out PATH]

--cards defaults to royalesim's data/derived/cards.json, located without importing the
package. The file is plain JSON; nothing of the engine runs. Names are RoyaleSim card
and unit names:

    kinds           card -> "troop" | "building" | "spell"
    compositions    troop/building card -> {unit: count one play deploys}
                    (summon_character x count, plus second_summon)
    spell_objects   spell card -> its projectile and area-effect object names
    spell_children  spell card -> units it spawns (Goblin Barrel -> Goblin)
    children        unit -> units it creates while alive or as it dies (spawner,
                    death spawn, projectile and area spawns): Witch -> Skeleton

The default table is the 15.535.29 extraction (cards.json); the 16.402.22 balance is
compiled into the engine with no JSON copy. Spawn relations rarely change with balance.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "vision" / "spawn_children.json"

UNIT_KEYS = {"character", "character2", "spawn_character", "death_spawn_character",
             "summon_character", "attached_character", "spawns", "spawn"}
SKIP = {"raw", "defaults_applied", "level_scaling", "overlays", "source_table"}
UNIT_FIELDS = ("spawner", "death_spawn", "death_spawn_projectile", "spawn_area_object",
               "death_area_effect", "projectile", "attached_character", "area_effect_on_hit")


def default_cards_path() -> Path:
    """royalesim/data/derived/cards.json, found without importing royalesim."""
    spec = importlib.util.find_spec("royalesim")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("royalesim is not installed here; pass --cards PATH")
    return Path(list(spec.submodule_search_locations)[0]) / "data" / "derived" / "cards.json"


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


def build(data: dict) -> dict:
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
    kinds, compositions, spell_objects, spell_children = {}, {}, {}, {}
    for c in data["cards"]:
        name, kind = c["name"], c["kind"]
        kinds[name] = kind
        if kind != "spell":
            comp: dict[str, int] = {}
            sc = c.get("summon_character")
            if sc:
                comp[sc] = int(c.get("count") or 1)
            ss = c.get("second_summon")
            if isinstance(ss, dict) and ss.get("character"):
                comp[ss["character"]] = comp.get(ss["character"], 0) + int(ss.get("count") or 1)
            compositions[name] = comp
            continue
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
    return {
        "about": ("Spawn relations for scripts/vision/events.py, generated from RoyaleSim's "
                  "card data by reading the JSON only. Names are RoyaleSim card and unit names; "
                  "map units to vision classes (tools/vision_classes) and cards to catalogue ids "
                  "(Battle.catalogue_json) with events.EventTables.from_names."),
        "source": {"file": "royalesim/data/derived/cards.json", "package": "royalesim==0.1.42",
                   "version": data.get("version"),
                   "note": ("the default 15.535.29 extraction; the 16.402.22 table ships compiled "
                            "into the engine only, with no JSON file, so it could not be read here")},
        "kinds": kinds,
        "compositions": compositions,
        "spell_objects": spell_objects,
        "spell_children": spell_children,
        "children": children,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cards", type=Path, default=None, help="RoyaleSim cards.json")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    src = args.cards or default_cards_path()
    table = build(json.loads(src.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(table, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{args.out}: {len(table['kinds'])} cards, {len(table['children'])} spawning units, "
          f"{len(table['spell_children'])} spawning spells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
