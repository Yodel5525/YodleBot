r"""The card -> unit-type table and the unit hitpoints, generated from the engine.

    .venvs\royalegym\Scripts\python.exe tools\rs_spawn_table.py            write data/rs_rows/spawn_primary.json
    .venvs\royalegym\Scripts\python.exe tools\rs_spawn_table.py --check    exit 1 when it is missing or stale
    .venvs\royalegym\Scripts\python.exe tools\rs_spawn_table.py --card-table client15535 --out other.json

Why (tokens-07, rsim-04, vision-16): rs_eval's rebuilt look-ahead (--lead -21, --rebuild),
the shadow (tools/rs_shadow.py) and the live converter (scripts/gamestate/live_bridge.py
card_of_form) read data/rs_rows/spawn_primary.json, and nothing in the repo wrote it, so they
failed on a fresh checkout. This tool writes it from the engine, and rs_eval._spawn_tables
calls ensure(), which regenerates it first when it is missing or was made under another
engine (its fingerprint differs).

How: one Battle (tools/rs_engine.make_battle, no command delay) under the engine config.
For each catalogue card whose card_kind (column 8) is TROOP or BUILDING, it is reset with a
filler deck and the single spawn (0, card_id, x, y, -1) at SPAWN_POINT_MILLI, a land point in
blue's half (blue owns low y), and the blue entities that are not crown towers are read; when
there are none yet, after SETTLE_TICKS more ticks. Battle.unit_hitpoints(card_id, level)
(royalesim 0.1.42) is read for every catalogue card at every level 1-16.

The file, ids being THIS engine's catalogue ids and unit_types_json() indexes (JSON keys are
text):

  primary[card]       the card's own unit type: the one unit_hitpoints' "own" row names when
                      the spawn put it down; else the most common spawned type (ties: the
                      lowest id); else the "own" row's unit when no unit appeared. A spell is
                      never in it.
  all[card]           every unit type the spawn put down, ascending ([] when none appeared)
  hp[card][level]     unit_hitpoints(card, level): [role, unit name, hitpoints] rows, the
                      card's "own" row first. Every catalogue card; levels the card's ladder
                      lacks are left out. {} on an engine without unit_hitpoints.
  fingerprint         rs_engine.fingerprint of the battle; ensure() compares its "id"
  unresolved[card]    why a TROOP / BUILDING card has no primary
  card_names, unit_types, engine_config, spawn, notes, format

by_unit(primary) is the inverse rs_eval always used: a unit type several cards make
(Skeletons, Skeleton Army) maps to the lowest card id.

Exit codes of the CLI: 0 ok (or current, with --check); 1 royalesim missing or refused the
config, or --check found the file missing or stale; 2 a bad setting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:      # also when imported as tools.rs_spawn_table
    sys.path.insert(0, str(ROOT / "tools"))
import rs_engine  # noqa: E402

DEFAULT_PATH = ROOT / "data" / "rs_rows" / "spawn_primary.json"
#: Bump when the file's layout changes: ensure() regenerates a file of another format.
FORMAT = 1
#: catalogue_json column of card_kind ("TROOP", "BUILDING", "SPELL"; royalesim CARD_KINDS).
KIND_COLUMN = 8
SPAWN_KINDS = frozenset({"TROOP", "BUILDING"})
#: Where the probe spawn goes, native world millitiles: the middle column, 5 tiles short of
#: the river and clear of blue's crown towers (king 7000-11000 x 1000-5000, princesses
#: centred at y 6500).
SPAWN_POINT_MILLI = (9000, 10000)
SPAWN_TEAM = 0
#: Extra ticks to step when the reset shows no spawned unit yet.
SETTLE_TICKS = 1
LEVELS: tuple[int, ...] = tuple(range(1, rs_engine.MAX_LEVEL + 1))
#: The filler deck both sides hold during the probes: Hog 2.6, a real deck, when the
#: catalogue has it (else the first 8 cards).
FILLER_NAMES = ("HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons",
                "Fireball", "Log")
GENERATOR = "tools/rs_spawn_table.py"

Log = Callable[[str], None]


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _read(obj: Any, name: str) -> Any:
    """obj.name() when it is a method, obj.name when it is a value, None when absent."""
    value = getattr(obj, name, None)
    return value() if callable(value) else value


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def card_kind(row: Sequence[Any]) -> str | None:
    """A catalogue row's card_kind, upper case; None for a row without the column."""
    return str(row[KIND_COLUMN]).upper() if len(row) > KIND_COLUMN else None


def spawn_cards(catalogue: Sequence[Sequence[Any]]) -> list[int]:
    """Catalogue ids of the TROOP and BUILDING cards, in id order."""
    return [i for i, row in enumerate(catalogue) if card_kind(row) in SPAWN_KINDS]


def filler_deck(catalogue: Sequence[Sequence[Any]]) -> list[int]:
    ids = {str(row[0]): i for i, row in enumerate(catalogue)}
    if all(name in ids for name in FILLER_NAMES):
        return [ids[name] for name in FILLER_NAMES]
    if len(catalogue) < 8:
        raise ValueError(f"the catalogue has {len(catalogue)} cards, a deck needs 8")
    return list(range(8))


def spawn_point(sub: int) -> tuple[int, int]:
    """SPAWN_POINT_MILLI in engine subtiles (sub = royalesim.SUBTILE_PER_MILLITILE)."""
    return SPAWN_POINT_MILLI[0] * sub, SPAWN_POINT_MILLI[1] * sub


def pick_primary(observed: Iterable[int], own: int | None) -> int | None:
    """The card's own unit type from what a spawn of it put down (`observed`, one unit type
    per entity) and the unit type unit_hitpoints' "own" row names (`own`, None when it names
    no unit type).

    `own` when it is among the observed; else the most common observed, ties to the lowest
    id; else `own` (nothing appeared); else None.
    """
    counts = Counter(int(u) for u in observed)
    if own is not None and own in counts:
        return own
    if counts:
        top = max(counts.values())
        return min(u for u, n in counts.items() if n == top)
    return own


def by_unit(primary: Mapping[int, int]) -> dict[int, int]:
    """unit type -> card: the inverse of `primary`. A unit type several cards make maps to
    the lowest card id (the rule rs_eval._spawn_tables always used)."""
    out: dict[int, int] = {}
    for card, unit in sorted((int(c), int(u)) for c, u in primary.items()):
        out.setdefault(unit, card)
    return out


def primary_of(table: Mapping[str, Any]) -> dict[int, int]:
    """A table's "primary" with int keys and values."""
    return {int(k): int(v) for k, v in table["primary"].items()}


def table_id(table: Mapping[str, Any] | None) -> str | None:
    """The engine fingerprint id a table was made under (None when it has none)."""
    fp = (table or {}).get("fingerprint")
    if not isinstance(fp, Mapping):
        return None
    return str(fp["id"]) if fp.get("id") else rs_engine.fingerprint_id(fp)


def is_current(table: Mapping[str, Any] | None, fp: Mapping[str, Any]) -> bool:
    """`table` was made by this format under the engine `fp` describes."""
    return (isinstance(table, Mapping) and table.get("format") == FORMAT
            and isinstance(table.get("primary"), Mapping) and table_id(table) == fp.get("id"))


def summary(table: Mapping[str, Any]) -> str:
    unresolved = table.get("unresolved") or {}
    n = len(table.get("primary") or {})
    return (f"{n} of {n + len(unresolved)} troop / building cards resolved"
            + (f" ({len(unresolved)} unresolved)" if unresolved else "")
            + f", hitpoints for {len(table.get('hp') or {})} cards, engine {table_id(table)}")


# ---------------------------------------------------------------------------
# reading the engine
# ---------------------------------------------------------------------------

def spawned_units(battle: Any, card_id: int, deck: Sequence[int], point: tuple[int, int],
                  fields: Mapping[str, int], settle: int = SETTLE_TICKS) -> list[int]:
    """Unit types of the blue non-tower entities after a reset with the single spawn
    (SPAWN_TEAM, card_id, x, y, -1); one per entity, in the engine's order."""
    battle.reset(seed=0, decks=[list(deck), list(deck)], shuffle=0, start_tick=0,
                 elixir_milli=[5000, 5000], tower_hp=None,
                 spawns=[(SPAWN_TEAM, int(card_id), int(point[0]), int(point[1]), -1)])
    F = fields
    for waited in range(settle + 1):
        if waited:
            battle.step([], 1)
        state = json.loads(battle.state_json())
        units = [int(e[F["unit_type"]]) for e in state["entities"]
                 if e[F["tower_slot"]] < 0 and e[F["team"]] == SPAWN_TEAM
                 and int(e[F["unit_type"]]) >= 0]
        if units:
            return units
    return []


def hitpoint_rows(battle: Any, card_id: int, levels: Iterable[int] = LEVELS
                  ) -> dict[str, list[list[Any]]]:
    """{level: [[role, unit name, hitpoints], ...]} from battle.unit_hitpoints; a level the
    card's ladder lacks (ValueError) is left out."""
    out: dict[str, list[list[Any]]] = {}
    for level in levels:
        try:
            rows = battle.unit_hitpoints(int(card_id), int(level))
        except ValueError:
            continue
        out[str(level)] = [[str(role), str(name), int(hp)] for role, name, hp in rows]
    return out


def own_unit_name(hp: Mapping[str, Sequence[Sequence[Any]]], level: int) -> str | None:
    """The unit the card's "own" row names, at `level` or else the nearest level it has."""
    if not hp:
        return None
    order = sorted(hp, key=lambda lv: (abs(int(lv) - level), int(lv)))
    for lv in order:
        for role, name, _ in hp[lv]:
            if role == "own":
                return str(name)
    return None


def _engine_battle(cfg: rs_engine.EngineConfig, rs: ModuleType) -> tuple[Any, dict]:
    battle = rs_engine.make_battle(cfg, rs)
    return battle, rs_engine.fingerprint(battle, rs, cfg)


def build(engine: Any = None, rs: ModuleType | None = None, *,
          cards: Iterable[int] | None = None) -> dict[str, Any]:
    """The table under `engine` (an rs_engine.EngineConfig or its to_json(); None: the
    process's rs_engine.active()), always without command delay. `cards` limits the spawn
    probes to those catalogue ids (all TROOP / BUILDING cards by default); hitpoints are
    read for every card either way."""
    cfg = rs_engine.resolve(engine).undelayed()
    rs = rs_engine.royalesim_module(rs)
    battle, fp = _engine_battle(cfg, rs)
    return _build(battle, rs, cfg, fp, cards)


def _build(battle: Any, rs: ModuleType, cfg: rs_engine.EngineConfig, fp: dict,
           cards: Iterable[int] | None = None) -> dict[str, Any]:
    catalogue = json.loads(battle.catalogue_json())
    unit_types = [str(u) for u in json.loads(battle.unit_types_json())]
    unit_index = {name: i for i, name in enumerate(unit_types)}
    fields = rs_engine.field_index(rs)
    sub = int(getattr(rs, "SUBTILE_PER_MILLITILE", 18))
    point = spawn_point(sub)
    deck = filler_deck(catalogue)
    level = cfg.level or _read(battle, "card_level") or 11
    notes: list[str] = []
    if catalogue and len(catalogue[0]) <= KIND_COLUMN:
        notes.append("the catalogue has no card_kind column: no card was spawned")

    hp: dict[str, dict[str, list[list[Any]]]] = {}
    if callable(getattr(battle, "unit_hitpoints", None)):
        for cid in range(len(catalogue)):
            try:
                rows = hitpoint_rows(battle, cid)
            except Exception as exc:            # one odd card must not lose the others
                notes.append(f"{catalogue[cid][0]} ({cid}): unit_hitpoints failed: "
                             f"{type(exc).__name__}: {exc}")
                continue
            if rows:
                hp[str(cid)] = rows
    else:
        notes.append(f"royalesim {fp.get('version')} has no Battle.unit_hitpoints: hp is empty "
                     "and primary comes from the spawns alone")

    wanted = spawn_cards(catalogue)
    if cards is not None:
        chosen = {int(c) for c in cards}
        wanted = [c for c in wanted if c in chosen]
    primary: dict[str, int] = {}
    seen: dict[str, list[int]] = {}
    unresolved: dict[str, str] = {}
    for cid in wanted:
        name = str(catalogue[cid][0])
        key = str(cid)
        own_name = own_unit_name(hp.get(key, {}), level)
        own = unit_index.get(own_name) if own_name is not None else None
        try:
            observed = spawned_units(battle, cid, deck, point, fields)
        except Exception as exc:                # the engine refused this spawn
            unresolved[key] = f"spawn refused: {type(exc).__name__}: {exc}"
            battle = rs_engine.make_battle(cfg, rs)       # a clean one for the rest
            continue
        seen[key] = sorted(set(observed))
        unit = pick_primary(observed, own)
        if unit is None:
            unresolved[key] = ("the spawn put no unit down and unit_hitpoints names "
                               + (f"{own_name!r}, not a unit type" if own_name else "no unit"))
            continue
        primary[key] = unit
        if not observed:
            notes.append(f"{name} ({cid}): the spawn put no unit down; primary is the own "
                         f"row's {own_name}")
        elif own is not None and own != unit:
            notes.append(f"{name} ({cid}): the own row names {own_name}, the spawn put down "
                         f"{[unit_types[u] for u in seen[key]]}; primary {unit_types[unit]}")
    return {
        "format": FORMAT,
        "generator": GENERATOR,
        "fingerprint": fp,
        "engine_config": cfg.to_json(),
        "spawn": {"team": SPAWN_TEAM, "point_milli": list(SPAWN_POINT_MILLI), "deck": deck,
                  "settle_ticks": SETTLE_TICKS, "levels": [LEVELS[0], LEVELS[-1]],
                  "own_row_level": int(level)},
        "card_names": [str(row[0]) for row in catalogue],
        "unit_types": unit_types,
        "primary": primary,
        "all": seen,
        "hp": hp,
        "unresolved": unresolved,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------

def read(path: str | os.PathLike = DEFAULT_PATH) -> dict | None:
    """The table at `path`; None when it is missing or not a JSON object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):           # missing, unreadable, not JSON
        return None
    return data if isinstance(data, dict) else None


def write(table: Mapping[str, Any], path: str | os.PathLike = DEFAULT_PATH,
          attempts: int = 5) -> Path:
    """Write `table` to `path` through a temporary file and os.replace, so a reader never
    sees half a file. Retries a replace Windows refuses while another process has the file
    open; raises the last OSError when it never succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(table) + "\n", encoding="utf-8")
    for attempt in range(max(1, attempts)):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt >= attempts - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.2 * (attempt + 1))
    return path


def ensure(path: str | os.PathLike = DEFAULT_PATH, engine: Any = None,
           rs: ModuleType | None = None, *, log: Log = _stderr) -> dict[str, Any]:
    """The table at `path` when it was made under `engine` (None: rs_engine.active());
    otherwise -- missing, unreadable, another format, or another engine's fingerprint -- a
    new one, written to `path` and returned.

    Without royalesim an existing file is returned unchecked (what rs_eval did before), and
    a missing one is an rs_engine.EngineError. A file that cannot be written leaves the new
    table in memory only, with a message.
    """
    path = Path(path)
    cfg = rs_engine.resolve(engine).undelayed()
    try:
        rs = rs_engine.royalesim_module(rs)
    except Exception as exc:
        old = read(path)
        if old is not None and isinstance(old.get("primary"), Mapping):
            log(f"spawn table: royalesim does not import ({type(exc).__name__}: {exc}); "
                f"using {path} without checking its engine")
            return old
        raise rs_engine.EngineError(
            f"{path} is missing and royalesim does not import ({type(exc).__name__}: {exc}), "
            "so it cannot be generated. Run with the engine venv, or run "
            r".venvs\royalegym\Scripts\python.exe tools\rs_spawn_table.py") from exc
    battle, fp = _engine_battle(cfg, rs)
    old = read(path)
    if is_current(old, fp):
        return old
    if old is None:
        why = "missing" if not path.exists() else "unreadable"
    elif old.get("format") != FORMAT:
        why = f"format {old.get('format')!r}, not {FORMAT}"
    else:
        why = f"made under engine {table_id(old) or 'unknown (no fingerprint)'}"
    table = _build(battle, rs, cfg, fp)
    try:
        write(table, path)
    except OSError as exc:
        log(f"spawn table: {path} was {why}; regenerated in memory for engine {fp['id']} "
            f"but not written ({type(exc).__name__}: {exc})")
        return table
    log(f"spawn table: {path} was {why}; regenerated: {summary(table)}")
    return table


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Write the card -> unit-type table and unit hitpoints "
                    "(data/rs_rows/spawn_primary.json) from the engine.")
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH,
                    help=f"where to write it (default {DEFAULT_PATH.relative_to(ROOT).as_posix()})")
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 when the file is missing or was made under "
                         "another engine")
    rs_engine.add_engine_args(ap)
    args = ap.parse_args(argv)
    try:
        cfg = rs_engine.config_from_args(args).undelayed()
    except (ValueError, FileNotFoundError) as exc:
        print(f"engine config: {exc}", file=sys.stderr)
        return 2
    try:
        rs = rs_engine.royalesim_module()
    except Exception as exc:
        print(f"royalesim does not import ({type(exc).__name__}: {exc}). Run this with the "
              r"engine venv: .venvs\royalegym\Scripts\python.exe tools\rs_spawn_table.py",
              file=sys.stderr)
        return 1
    try:
        battle, fp = _engine_battle(cfg, rs)
    except Exception as exc:
        print(f"engine: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"engine: {cfg.describe()}\n        {rs_engine.summary(fp)}", flush=True)
    if args.check:
        old = read(args.out)
        if is_current(old, fp):
            print(f"{args.out}: current ({summary(old)})")
            return 0
        print(f"{args.out}: " + ("missing or unreadable" if old is None else
                                 f"made under engine {table_id(old) or 'unknown'} "
                                 f"(format {old.get('format')!r}), this engine is {fp['id']}"))
        return 1
    table = _build(battle, rs, cfg, fp)
    write(table, args.out)
    print(f"{args.out}: {summary(table)}")
    for key, why in sorted(table["unresolved"].items(), key=lambda kv: int(kv[0])):
        print(f"  unresolved {table['card_names'][int(key)]} ({key}): {why}")
    for note in table["notes"]:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
