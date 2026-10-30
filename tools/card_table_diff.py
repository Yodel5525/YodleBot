r"""What a RoyaleSim update changes in the card data (U08 of docs/SEASON_UPDATES.md): the
roster, the values per card, the ledger keys, and the update class the data alone implies.
Standard library only, so the stage can run it with any interpreter, before the new engine is
installed.

    python tools\card_table_diff.py OLD_TABLE.json NEW_TABLE.json
    python tools\card_table_diff.py OLD\calibration.json NEW\calibration.json
    python tools\card_table_diff.py OLD_TABLE.json NEW_TABLE.json ^
        --ledgers OLD\calibration.json NEW\calibration.json ^
        --pool <data>\deck_stats.json --json diff.json --md diff.md

Inputs. A card table is a RoyaleSim data/derived/cards-*.json (the sdist ships every arm's; a
wheel ships cards-15.535.json under royalesim.data_dir()); a ledger is RoyaleSim's
data/calibration.json. Each positional file is recognised by its shape (kind_of): a table has a
"cards" list, a ledger has sections of {"value": ...} entries. --ledgers adds a ledger pair to a
table pair; the positional pair may also be two ledgers.

Tables (diff_tables):
* Roster: names added to and removed from cards, evolutions, hero_forms and towers (records
  keyed by "name") and from units, projectiles, area_effect_objects and buffs (keyed maps). A
  form is its (base card, kind) pair, the record's "form_of" and "evolution" or "hero": a new
  pair is a new form (Evo Giant: ("Giant", "evolution")). Under a pair both sides have, records
  added and removed in equal numbers are "renamed", not a new form; a changed number of records
  is a form record added or removed ("records": a second evolution, Giant_EV2 next to
  Giant_EV1), classed as a new or removed form is.
* Values: every record is flattened to leaf paths (lists of records by index, a list of plain
  values as one leaf) and compared. Provenance is ignored: the keys in IGNORED_KEYS and those
  ending in _provenance or _note, at any depth (prose, extraction notes, display names, the
  table's own version text). A field present on one side only is "added" or "removed".
* Echoes folded onto cards: a balance change shows up in the card record, in its unit and its
  projectile records and in their raw CSV columns (Dart Goblin's damage 59 -> 56 is five leaves
  in the 2026-10-19 table). Each roster record owns the map records it reaches through names
  (family; REF_SKIP keys such as ignore_buffs and form_of are not references). A changed map
  record is listed under every surviving roster record that reaches it, and within one card the
  changes with the same field (lower case, no "_", no ms/milli unit: load_time_ms and raw
  LoadTime agree) and the same old and new values fold into one, the others kept as its echoes.
  Changes no card reaches are "unattached"; rarities and globals changes are "shared".
* Tags: OUR DECK (tools/rs_rows.py HOG26, the Hog 2.6 catalogue names, and their forms; the
  towers too, which every battle has), POOL (the base card carries at least --pool-min, 3%, of
  the opponent pool's weight), else OTHER. --pool reads a deck_stats.json (tools/deck_stats.py,
  the split --pool-split; a deck's share counts for each card in its "names", over the decks
  listed) or a {card name: weight} object; without it nothing is POOL.
* Balance size (docs/SEASON_UPDATES.md 6.3): "minor" when no OUR DECK card changed by more than
  5%, fewer than 3 POOL cards by more than 10%, and no shared value changed; else "major". A
  change without a size (not two numbers, an old value of 0, a field added or removed) counts as
  above both bars.

Ledgers (diff_ledgers): per section, keys added, removed and changed. A key is compared on
LEDGER_FIELDS (value, candidates, status) only: provenance, confidence, history and the other
prose fields are ignored. "meta" (the target game version) and "$comment" entries are listed,
never classed. The cards.CARD_TABLE arms are shown separately (arms added and removed, the
engine's default arm).

Class (classify), the part of docs/SEASON_UPDATES.md 4.4 the data decides; the stage (U09) adds
what only the engines show (a pinned card that stops loading, ENTITY_FIELDS, ABILITY_BUTTONS,
SPELL_FIELDS, STATUS_BITS, the schema digest; and a royalesim version change with byte-equal
tables and ledger, a Rust-only release, which is at least ENGINE_ONLY: checkpoints get the
version note and RL runs a new --name) and takes the worse of the two (worst):
  BREAKING     a base card or a tower removed from the table
  NEW_CARDS    a base card added
  NEW_FORMS    an evolution or hero form (base, kind) added or removed, or a form record added
               or removed under a form both sides have
  BALANCE      a table value changed, a field or a unit / projectile / area effect / buff record
               added or removed, a form record renamed
  ENGINE_ONLY  a ledger value changed, a ledger key added or removed; a value changed in a table
               section this tool does not know ("other"), a table section added or removed
  SAME         none of these (arm, status and meta changes are listed only)
The class is the update's, not each checkpoint's: a unit record added or removed renumbers
the name-sorted unit types (3.3), so a --units types checkpoint is refused in any class that
adds one, BALANCE included. The stage's checkpoint table comes from loading each checkpoint
(tools/rs_vocab.py check), never from the class.

Exit codes: 0 the diff was written (whatever the class); 1 an input is missing, not JSON, not a
table or ledger, or the two files of a pair are of different kinds; 2 bad arguments.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = 1
TOOL = "tools/card_table_diff.py"

#: Update classes, least to most severe (docs/SEASON_UPDATES.md 4.4).
CLASSES: tuple[str, ...] = ("SAME", "ENGINE_ONLY", "BALANCE", "NEW_FORMS", "NEW_CARDS",
                            "BREAKING")
OUR_DECK, POOL, OTHER = "OUR DECK", "POOL", "OTHER"
TAGS: tuple[str, ...] = (OUR_DECK, POOL, OTHER)
#: tools/rs_rows.py HOG26, the sparring deck (catalogue names); tests/pipeline checks the copy.
HOG26 = frozenset({"Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits",
                   "Musketeer", "Skeletons", "Log"})
#: docs/SEASON_UPDATES.md 6.2: a card "matters" from 3% of the opponent pool's weight.
POOL_MIN = 0.03
#: 6.3: BALANCE minor = no Hog 2.6 card past 5%, fewer than 3 POOL cards past 10%.
OUR_DECK_BAR, POOL_BAR, POOL_BAR_CARDS = 5.0, 10.0, 3

#: Table sections of records with a "name" (roster sections), and the kind of each record.
ROSTER: dict[str, str] = {"cards": "card", "evolutions": "evolution",
                          "hero_forms": "hero", "towers": "tower"}
#: Table sections that are maps of name -> record, which roster records reach by name.
MAPS: tuple[str, ...] = ("units", "projectiles", "area_effect_objects", "buffs")
#: Table sections whose values apply to many cards.
SHARED: tuple[str, ...] = ("rarities", "globals")
#: Top-level table keys that are prose about the table, never compared.
IGNORED_TOP = frozenset({"version", "vintage_warning", "provenance", "conventions"})
#: Keys ignored at any depth: provenance, extraction notes and display text.
IGNORED_KEYS = frozenset({"provenance", "vintage", "vintage_warning", "damage_source",
                          "source", "source_table", "defaults_applied", "rounding", "reading",
                          "display_name", "overlays", "unread", "comment", "note", "notes"})
IGNORED_SUFFIXES: tuple[str, ...] = ("_provenance", "_note")
#: Keys whose names are not references to the records they name.
REF_SKIP = frozenset({"form_of", "display_name", "display_attached", "ignore_buffs",
                      "IgnoreBuff", "IgnoreTargetsWithBuff", "included_attackers",
                      "excluded_units", "multipliers"})
#: Ledger entry fields compared; every other field of an entry is prose.
LEDGER_FIELDS: tuple[str, ...] = ("value", "candidates", "status")
#: Fields whose change alone does not class an update (listed only).
LEDGER_LISTED: tuple[str, ...] = ("candidates", "status")
LEDGER_META = "meta"
CARD_TABLE = ("cards", "CARD_TABLE")
UNIT_SUFFIXES: tuple[str, ...] = ("milli", "ms")
MISSING = object()


class DiffError(ValueError):
    """An input that is not a card table or a ledger, or a pair of different kinds."""


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> Any:
    """A JSON file (DiffError when it is missing or not JSON)."""
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise DiffError(f"{path}: cannot read ({exc.strerror or exc})") from None
    except ValueError as exc:
        raise DiffError(f"{path}: not JSON ({exc})") from None


def kind_of(data: Any) -> str | None:
    """"table" (a "cards" list), "ledger" (sections of {"value": ...} entries) or None."""
    if not isinstance(data, Mapping):
        return None
    if isinstance(data.get("cards"), list):
        return "table"
    for name, section in data.items():
        if str(name).startswith("$") or not isinstance(section, Mapping):
            continue
        if any(isinstance(e, Mapping) and "value" in e for e in section.values()):
            return "ledger"
    return None


def ignored(key: Any) -> bool:
    text = str(key)
    return (text in IGNORED_KEYS or text.startswith("$")
            or any(text.endswith(s) for s in IGNORED_SUFFIXES))


def records(table: Mapping[str, Any], section: str) -> dict[str, Any]:
    """A table section as {name: record}: roster lists by "name" (a repeated name gets "#2",
    "#3"), maps as they are, anything else {}."""
    value = table.get(section)
    if isinstance(value, Mapping):
        return dict(value)
    out: dict[str, Any] = {}
    if isinstance(value, list):
        for k, rec in enumerate(value):
            name = str(rec.get("name")) if isinstance(rec, Mapping) and rec.get("name") else f"#{k}"
            key, n = name, 1
            while key in out:
                n += 1
                key = f"{name}#{n}"
            out[key] = rec
    return out


def flatten(value: Any, prefix: tuple = ()) -> dict[tuple, Any]:
    """Leaf paths of a record: dict keys (IGNORED ones dropped) and list indexes; a list holding
    no dict or list is one leaf."""
    out: dict[tuple, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not ignored(key):
                out.update(flatten(item, prefix + (str(key),)))
    elif isinstance(value, list) and any(isinstance(v, (Mapping, list)) for v in value):
        for k, item in enumerate(value):
            out.update(flatten(item, prefix + (k,)))
    else:
        out[prefix] = value
    return out


def path_text(parts: Sequence[Any]) -> str:
    """("units", "Knight", "raw", "Speed") -> 'units.Knight.raw.Speed'; indexes as [k]."""
    text = ""
    for p in parts:
        text += f"[{p}]" if isinstance(p, int) else (f".{p}" if text else str(p))
    return text


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def pct(old: Any, new: Any) -> float | None:
    """(new - old) / |old| in percent, rounded to 0.01; None unless both are numbers and old is
    not 0."""
    if not (number(old) and number(new)) or old == 0:
        return None
    return round(100.0 * (new - old) / abs(old), 2)


def norm_field(parts: Sequence[Any]) -> str:
    """The field a change is folded on: the last key, lower case, "_" removed, a trailing ms or
    milli unit removed (load_time_ms and the raw CSV column LoadTime are both 'loadtime')."""
    leaf = next((p for p in reversed(parts) if isinstance(p, str)), "")
    text = leaf.lower().replace("_", "")
    for suffix in UNIT_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[:-len(suffix)]
    return text


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _order(parts: Sequence[Any]) -> list[tuple]:
    """Sort key of a leaf path: list indexes by number, keys by text."""
    return [(0, p, "") if isinstance(p, int) else (1, 0, str(p)) for p in parts]


def compare(old: Any, new: Any, prefix: tuple = ()) -> list[dict[str, Any]]:
    """Changed leaves of two records, in path order: {"path", "parts", "field", "kind" (changed /
    added / removed), "old" and "new" (absent on the side without it), "pct"}."""
    a, b = flatten(old), flatten(new)
    out = []
    for parts in sorted(set(a) | set(b), key=_order):
        va, vb = a.get(parts, MISSING), b.get(parts, MISSING)
        if va is not MISSING and vb is not MISSING and _key(va) == _key(vb):
            continue
        full = prefix + parts
        change: dict[str, Any] = {"path": path_text(full), "parts": list(full),
                                  "field": path_text(parts) or "(value)"}
        if va is MISSING:
            change.update(kind="added", new=vb, pct=None)
        elif vb is MISSING:
            change.update(kind="removed", old=va, pct=None)
        else:
            change.update(kind="changed", old=va, new=vb, pct=pct(va, vb))
        out.append(change)
    return out


# ---------------------------------------------------------------------------
# references: which map records each roster record reaches
# ---------------------------------------------------------------------------

def direct_refs(value: Any, names: Mapping[str, frozenset]) -> set[tuple[str, str]]:
    """(map section, name) of every string in `value` that names a map record, outside
    REF_SKIP and IGNORED keys."""
    found: set[tuple[str, str]] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            stack.extend(v for k, v in item.items() if k not in REF_SKIP and not ignored(k))
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            for section, pool in names.items():
                if item in pool:
                    found.add((section, item))
    return found


def families(table: Mapping[str, Any]) -> dict[tuple[str, str], set[tuple[str, str]]]:
    """(roster section, name) -> the (map section, name) records it reaches, transitively."""
    maps = {s: records(table, s) for s in MAPS}
    names = {s: frozenset(maps[s]) for s in MAPS}
    refs = {(s, n): direct_refs(rec, names) for s in MAPS for n, rec in maps[s].items()}
    out: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for section in ROSTER:
        for name, rec in records(table, section).items():
            seen: set[tuple[str, str]] = set()
            todo = sorted(direct_refs(rec, names))
            while todo:
                node = todo.pop()
                if node in seen:
                    continue
                seen.add(node)
                todo.extend(sorted(refs.get(node, ()) - seen))
            out[(section, name)] = seen
    return out


# ---------------------------------------------------------------------------
# pool weights and tags
# ---------------------------------------------------------------------------

def pool_weights(data: Any, split: str = "train") -> dict[str, float]:
    """Card name -> share of the pool's weight its decks carry. A deck_stats.json (a "splits"
    object): over the decks listed in `split`, each deck's share counts once for each card in
    its "names", divided by the listed decks' total share. Otherwise a {name: weight} object,
    as given."""
    if not isinstance(data, Mapping):
        raise DiffError("pool: expected deck stats or an object of card name -> weight")
    if "splits" not in data:
        out = {}
        for name, weight in data.items():
            if str(name).startswith(("_", "$")):
                continue
            if not number(weight):
                raise DiffError(f"pool: the weight of {name!r} is not a number")
            out[str(name)] = float(weight)
        return out
    splits = data.get("splits")
    if not isinstance(splits, Mapping) or split not in splits:
        have = ", ".join(sorted(splits)) if isinstance(splits, Mapping) else "none"
        raise DiffError(f"pool: no split {split!r} in the deck stats (it has {have})")
    totals: dict[str, float] = {}
    listed = 0.0
    for deck in splits[split].get("decks") or ():
        names = deck.get("names") if isinstance(deck, Mapping) else None
        share = deck.get("share") if isinstance(deck, Mapping) else None
        if not names or not number(share):
            continue
        listed += float(share)
        for name in set(names):
            totals[str(name)] = totals.get(str(name), 0.0) + float(share)
    if listed <= 0:
        raise DiffError(f"pool: no deck with card names in split {split!r} (deck stats built "
                        "with --no-names cannot be read)")
    return {name: w / listed for name, w in sorted(totals.items())}


def tag_of(section: str, base: str, deck: frozenset, weights: Mapping[str, float] | None,
           pool_min: float = POOL_MIN) -> str:
    """OUR DECK for a card of `deck` or one of its forms, and for a tower; POOL when the base
    card's weight is at least pool_min; else OTHER."""
    if section == "towers" or base in deck:
        return OUR_DECK
    if weights and weights.get(base, 0.0) >= pool_min:
        return POOL
    return OTHER


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def base_of(section: str, name: str, rec: Any) -> str:
    """The card a roster record belongs to: its form_of for a form, else its own name."""
    if section in ("evolutions", "hero_forms") and isinstance(rec, Mapping) and rec.get("form_of"):
        return str(rec["form_of"])
    return name


def forms(table: Mapping[str, Any]) -> dict[tuple[str, str], list[str]]:
    """(base card, "evolution" / "hero") -> the record names of that form."""
    out: dict[tuple[str, str], list[str]] = {}
    for section in ("evolutions", "hero_forms"):
        for name, rec in records(table, section).items():
            out.setdefault((base_of(section, name, rec), ROSTER[section]), []).append(name)
    return out


def fold(changes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Changes with the same norm_field, kind and values folded onto the first: its "echoes"
    are the other paths."""
    out: list[dict[str, Any]] = []
    index: dict[tuple, dict[str, Any]] = {}
    for ch in changes:
        key = (norm_field(ch["parts"]), ch["kind"], _key(ch.get("old")), _key(ch.get("new")))
        head = index.get(key)
        if head is None:
            head = {k: v for k, v in ch.items() if k != "parts"}
            head["echoes"] = []
            index[key] = head
            out.append(head)
        else:
            head["echoes"].append(ch["path"])
    return out


def _roster_names(old: Mapping[str, Any], new: Mapping[str, Any], section: str) -> dict:
    a, b = records(old, section), records(new, section)
    added = [n for n in b if n not in a]
    removed = [n for n in a if n not in b]
    if section in ("evolutions", "hero_forms"):
        return {"added": [{"name": n, "form_of": base_of(section, n, b[n])} for n in added],
                "removed": [{"name": n, "form_of": base_of(section, n, a[n])} for n in removed]}
    return {"added": sorted(added) if section in MAPS else added,
            "removed": sorted(removed) if section in MAPS else removed}


def diff_tables(old: Mapping[str, Any], new: Mapping[str, Any], *,
                weights: Mapping[str, float] | None = None, pool_min: float = POOL_MIN,
                deck: Iterable[str] = HOG26) -> dict[str, Any]:
    """The table part of the diff (module doc): versions, sections, roster, forms, cards (one
    entry per roster record with changes), shared, unattached, other and counts."""
    deck = frozenset(deck)
    known = set(ROSTER) | set(MAPS) | set(SHARED) | IGNORED_TOP
    roster = {s: _roster_names(old, new, s) for s in (*ROSTER, *MAPS)}
    fa, fb = forms(old), forms(new)
    roster["forms"] = {
        "added": [{"base": base, "form": kind, "names": fb[(base, kind)]}
                  for base, kind in fb if (base, kind) not in fa],
        "removed": [{"base": base, "form": kind, "names": fa[(base, kind)]}
                    for base, kind in fa if (base, kind) not in fb]}
    # a form kept on both sides: records added and removed in equal numbers are renamed; a
    # changed number of records is a form record added or removed (Giant_EV2 next to Giant_EV1)
    section_of = {kind: s for s, kind in ROSTER.items()}
    renamed: set[str] = set()
    form_records = []
    for base, kind in sorted(pair for pair in fa if pair in fb):
        section = section_of[kind]
        added = [i["name"] for i in roster[section]["added"] if i["form_of"] == base]
        removed = [i["name"] for i in roster[section]["removed"] if i["form_of"] == base]
        if not (added or removed):
            continue
        if len(fa[(base, kind)]) == len(fb[(base, kind)]):
            renamed.update(added + removed)
        else:
            form_records.append({"base": base, "form": kind, "added": added,
                                 "removed": removed})
    roster["forms"]["renamed"] = sorted(renamed)
    roster["forms"]["records"] = form_records
    raw = 0
    map_changes: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for section in MAPS:
        a, b = records(old, section), records(new, section)
        for name in a:
            if name in b:
                ch = compare(a[name], b[name], (section, name))
                if ch:
                    map_changes[(section, name)] = ch
                    raw += len(ch)
    fam_a, fam_b = families(old), families(new)
    owned: set[tuple[str, str]] = set()
    cards = []
    for section, kind in ROSTER.items():
        a, b = records(old, section), records(new, section)
        for name in a:
            if name not in b:
                continue
            own = compare(a[name], b[name], (section, name))
            raw += len(own)
            family = fam_a.get((section, name), set()) | fam_b.get((section, name), set())
            reached = sorted(node for node in family if node in map_changes)
            owned.update(reached)
            changes = own + [ch for node in reached for ch in map_changes[node]]
            if not changes:
                continue
            heads = fold(changes)
            base = base_of(section, name, b[name])
            sizes = [abs(h["pct"]) for h in heads if h["pct"] is not None]
            cards.append({"name": name, "section": section, "kind": kind, "base": base,
                          "tag": tag_of(section, base, deck, weights, pool_min),
                          "weight": None if weights is None else weights.get(base, 0.0),
                          "changes": heads, "max_pct": max(sizes) if sizes else None,
                          "unsized": sum(1 for h in heads if h["pct"] is None),
                          "via": [path_text(node) for node in reached]})
    cards.sort(key=lambda c: (TAGS.index(c["tag"]), c["base"], list(ROSTER).index(c["section"]),
                              c["name"]))
    unattached = [{k: v for k, v in ch.items() if k != "parts"}
                  for node in sorted(map_changes) if node not in owned
                  for ch in map_changes[node]]
    shared = []
    for section in SHARED:
        ch = compare(old.get(section, {}), new.get(section, {}), (section,))
        raw += len(ch)
        shared += [{k: v for k, v in c.items() if k != "parts"} for c in ch]
    other = []
    for section in sorted((set(old) | set(new)) - known, key=str):
        if ignored(section):
            continue
        ch = compare(old.get(section), new.get(section), (str(section),)) \
            if section in old and section in new else []
        other += [{k: v for k, v in c.items() if k != "parts"} for c in ch]
    folded = sum(len(c["changes"]) for c in cards) + len(unattached) + len(shared)
    return {
        "old_version": old.get("version"), "new_version": new.get("version"),
        "sections": {"added": sorted((str(k) for k in new if k not in old)),
                     "removed": sorted((str(k) for k in old if k not in new))},
        "roster": roster, "cards": cards, "shared": shared, "unattached": unattached,
        "other": other,
        "counts": {"leaves": raw, "folded": folded, "cards": len(cards),
                   "by_tag": {t: sum(1 for c in cards if c["tag"] == t) for t in TAGS}},
    }


def balance_size(table: Mapping[str, Any]) -> str | None:
    """"minor" / "major" by the 6.3 rule (module doc); None when no value changed and no map
    record was added or removed."""
    moved = (any(table["roster"][s]["added"] or table["roster"][s]["removed"] for s in MAPS)
             or bool(table["roster"]["forms"]["renamed"]))
    if not (table["cards"] or table["shared"] or table["unattached"] or moved):
        return None
    if table["shared"]:
        return "major"

    def past(card: Mapping[str, Any], bar: float) -> bool:
        return bool(card["unsized"]) or (card["max_pct"] or 0.0) > bar
    if any(past(c, OUR_DECK_BAR) for c in table["cards"] if c["tag"] == OUR_DECK):
        return "major"
    pool = {c["base"] for c in table["cards"] if c["tag"] == POOL and past(c, POOL_BAR)}
    return "major" if len(pool) >= POOL_BAR_CARDS else "minor"


# ---------------------------------------------------------------------------
# ledgers
# ---------------------------------------------------------------------------

def ledger_entries(ledger: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """section -> {KEY: entry}; "$" sections and keys dropped, a plain value read as
    {"value": it} (prose keys, ending in note or comment, dropped)."""
    out: dict[str, dict[str, Any]] = {}
    for section, items in ledger.items():
        if str(section).startswith("$") or not isinstance(items, Mapping):
            continue
        entries: dict[str, Any] = {}
        for key, entry in items.items():
            text = str(key)
            if text.startswith("$"):
                continue
            if isinstance(entry, Mapping):
                entries[text] = entry
            elif not text.lower().endswith(("note", "comment")):
                entries[text] = {"value": entry}
        out[str(section)] = entries
    return out


def _arms(ledger: Mapping[str, Any]) -> tuple[list[str], Any]:
    entry = ledger.get(CARD_TABLE[0], {})
    entry = entry.get(CARD_TABLE[1]) if isinstance(entry, Mapping) else None
    if not isinstance(entry, Mapping):
        return [], None
    cands = entry.get("candidates")
    return ([str(c) for c in cands] if isinstance(cands, list) else []), entry.get("value")


def diff_ledgers(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """Per section: added and removed keys, and changed keys with each LEDGER_FIELDS field's old
    and new (candidates also as added / removed); "meta" apart; the card-table arms; counts."""
    a, b = ledger_entries(old), ledger_entries(new)
    sections: dict[str, dict[str, Any]] = {}
    counts = {"added": 0, "removed": 0, "value": 0, "listed_only": 0}
    meta: list[dict[str, Any]] = []
    for section in sorted(set(a) | set(b)):
        ea, eb = a.get(section, {}), b.get(section, {})
        added = sorted(k for k in eb if k not in ea)
        removed = sorted(k for k in ea if k not in eb)
        changed = []
        for key in sorted(k for k in ea if k in eb):
            fields: dict[str, Any] = {}
            for name in LEDGER_FIELDS:
                va, vb = ea[key].get(name, MISSING), eb[key].get(name, MISSING)
                if _key(None if va is MISSING else va) == _key(None if vb is MISSING else vb):
                    continue
                item: dict[str, Any] = {}
                if va is not MISSING:
                    item["old"] = va
                if vb is not MISSING:
                    item["new"] = vb
                if name == "candidates" and isinstance(va, list) and isinstance(vb, list):
                    item["added"] = [c for c in vb if c not in va]
                    item["removed"] = [c for c in va if c not in vb]
                fields[name] = item
            if fields:
                changed.append({"key": key, "fields": fields,
                                "listed_only": all(f in LEDGER_LISTED for f in fields)})
        if section == LEDGER_META:
            meta = [{"key": c["key"], **c["fields"].get("value", {})} for c in changed]
            meta += [{"key": k, "new": eb[k].get("value")} for k in added]
            meta += [{"key": k, "old": ea[k].get("value")} for k in removed]
            continue
        if added or removed or changed:
            sections[section] = {"added": added, "removed": removed, "changed": changed}
            counts["added"] += len(added)
            counts["removed"] += len(removed)
            counts["value"] += sum(1 for c in changed if not c["listed_only"])
            counts["listed_only"] += sum(1 for c in changed if c["listed_only"])
    arms_a, default_a = _arms(old)
    arms_b, default_b = _arms(new)
    return {"meta": meta, "sections": sections, "counts": counts,
            "arms": {"old": arms_a, "new": arms_b,
                     "added": [x for x in arms_b if x not in arms_a],
                     "removed": [x for x in arms_a if x not in arms_b],
                     "default_old": default_a, "default_new": default_b}}


# ---------------------------------------------------------------------------
# class
# ---------------------------------------------------------------------------

def worst(*classes: str | None) -> str:
    """The most severe of `classes` (None ignored; SAME for none)."""
    got = [c for c in classes if c]
    for c in got:
        if c not in CLASSES:
            raise ValueError(f"unknown update class {c!r}; the classes are {', '.join(CLASSES)}")
    return max(got, key=CLASSES.index) if got else CLASSES[0]


def classify(table: Mapping[str, Any] | None = None,
             ledger: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """{"class", "reasons" (one line per rule that applied, most severe first), "balance"}."""
    found: list[tuple[str, str]] = []
    balance = None
    if table is not None:
        r = table["roster"]
        gone = r["cards"]["removed"] + r["towers"]["removed"]
        if gone:
            found.append(("BREAKING", "removed from the table: " + ", ".join(gone)))
        if r["cards"]["added"]:
            found.append(("NEW_CARDS", "new cards: " + ", ".join(r["cards"]["added"])))
        if r["towers"]["added"]:
            found.append(("NEW_CARDS", "new towers: " + ", ".join(r["towers"]["added"])))
        for change, verb in (("added", "new"), ("removed", "removed")):
            items = r["forms"][change]
            if items:
                found.append(("NEW_FORMS", f"{verb} forms: " + ", ".join(
                    f"{f['base']} {f['form']}" for f in items)))
        if r["forms"].get("records"):
            found.append(("NEW_FORMS", "form records added or removed: " + ", ".join(
                f"{f['base']} {f['form']} " + " ".join(
                    [f"+{n}" for n in f["added"]] + [f"-{n}" for n in f["removed"]])
                for f in r["forms"]["records"])))
        records_moved = [s for s in MAPS if r[s]["added"] or r[s]["removed"]]
        balance = balance_size(table)
        if balance is not None:
            parts = []
            if table["cards"]:
                by = table["counts"]["by_tag"]
                parts.append(f"{len(table['cards'])} cards changed ("
                             + ", ".join(f"{by[t]} {t}" for t in TAGS if by[t]) + ")")
            if table["shared"]:
                parts.append(f"{len(table['shared'])} shared values")
            if table["unattached"]:
                parts.append(f"{len(table['unattached'])} unattached values")
            for s in records_moved:
                parts.append(f"{s} +{len(r[s]['added'])} -{len(r[s]['removed'])}")
            if r["forms"]["renamed"]:
                parts.append("form records renamed: " + ", ".join(r["forms"]["renamed"]))
            found.append(("BALANCE", "; ".join(parts) + f" ({balance})"))
        # table bytes that no rule above reads (sections this tool does not know, sections
        # added or removed) still change the table's FNV: ENGINE_ONLY, never SAME
        others = sorted({str(c["path"]).split(".")[0].split("[")[0] for c in table["other"]})
        secs = {change: [s for s in table["sections"][change]
                         if s not in IGNORED_TOP and not ignored(s)]
                for change in ("added", "removed")}
        parts = []
        if others:
            n = len(table["other"])
            parts.append(f"{n} value{'s' if n != 1 else ''} changed in other sections ("
                         + ", ".join(others) + ")")
        parts += [f"sections {change}: " + ", ".join(names)
                  for change, names in secs.items() if names]
        if parts:
            found.append(("ENGINE_ONLY", "table: " + "; ".join(parts)))
    if ledger is not None:
        c = ledger["counts"]
        if c["value"] or c["added"] or c["removed"]:
            found.append(("ENGINE_ONLY", f"ledger: {c['value']} keys changed, {c['added']} added, "
                                         f"{c['removed']} removed"))
    found.sort(key=lambda item: -CLASSES.index(item[0]))
    return {"class": worst(*(cls for cls, _ in found)),
            "reasons": [f"{cls}: {why}" for cls, why in found], "balance": balance}


# ---------------------------------------------------------------------------
# the result, as text and markdown
# ---------------------------------------------------------------------------

def diff(old_table: Mapping | None = None, new_table: Mapping | None = None,
         old_ledger: Mapping | None = None, new_ledger: Mapping | None = None, *,
         weights: Mapping[str, float] | None = None, pool_min: float = POOL_MIN,
         deck: Iterable[str] = HOG26, sources: Mapping[str, Any] | None = None
         ) -> dict[str, Any]:
    """Everything: {"schema", "tool", "sources", "table", "ledger", "class", "reasons",
    "balance"} (table / ledger None when that pair was not given)."""
    table = None if old_table is None or new_table is None else diff_tables(
        old_table, new_table, weights=weights, pool_min=pool_min, deck=deck)
    ledger = None if old_ledger is None or new_ledger is None else diff_ledgers(
        old_ledger, new_ledger)
    verdict = classify(table, ledger)
    return {"schema": SCHEMA, "tool": TOOL, "sources": dict(sources or {}),
            "pool": None if weights is None else {"cards": len(weights), "min": pool_min},
            "table": table, "ledger": ledger, **verdict}


def show(value: Any, limit: int = 60) -> str:
    """A value for a line of text: JSON, cut to `limit` characters."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def describe_change(ch: Mapping[str, Any], limit: int = 60) -> str:
    """'damage 59 -> 56 (-5.08%)'; 'x added: 1'; 'x removed (was 1)'."""
    if ch["kind"] == "added":
        return f"{ch['field']} added: {show(ch.get('new'), limit)}"
    if ch["kind"] == "removed":
        return f"{ch['field']} removed (was {show(ch.get('old'), limit)})"
    size = f" ({ch['pct']:+.2f}%)" if ch.get("pct") is not None else ""
    return f"{ch['field']} {show(ch.get('old'), limit)} -> {show(ch.get('new'), limit)}{size}"


def changed_list(item: Mapping[str, Any]) -> str:
    """'added a, b; removed c' from an item's "added" and "removed" lists ('reordered' when
    both are empty)."""
    bits = [f"{what} {', '.join(map(str, item[what]))}" for what in ("added", "removed")
            if item.get(what)]
    return "; ".join(bits) or "reordered"


def _names(items: Sequence[Any]) -> str:
    return ", ".join(i["name"] if isinstance(i, Mapping) else str(i) for i in items)


def summary_lines(result: Mapping[str, Any], *, per_card: int = 3) -> list[str]:
    """The console summary."""
    L: list[str] = []
    for side in ("old", "new"):
        src = (result.get("sources") or {}).get(side) or {}
        if src:
            L.append(f"{side:<4} {src.get('version') or src.get('kind') or '?'}  "
                     f"{src.get('path') or ''}".rstrip())
    t = result.get("table")
    if t is not None:
        r = t["roster"]
        L.append("roster: " + ", ".join(
            f"{s.replace('_', ' ')} +{len(r[s]['added'])} -{len(r[s]['removed'])}"
            for s in (*ROSTER, *MAPS)))
        for change in ("added", "removed"):
            for f in r["forms"][change]:
                L.append(f"  form {change}: {f['base']} {f['form']} ({', '.join(f['names'])})")
        if r["forms"]["renamed"]:
            L.append("  form records renamed: " + ", ".join(r["forms"]["renamed"]))
        for f in r["forms"].get("records", ()):
            L.append(f"  form records of {f['base']} {f['form']}: {changed_list(f)}")
        for s in ROSTER:
            for change in ("added", "removed"):
                if r[s][change]:
                    L.append(f"  {s} {change}: {_names(r[s][change])}")
        by = t["counts"]["by_tag"]
        L.append(f"values: {t['counts']['cards']} cards changed ("
                 + ", ".join(f"{by[tag]} {tag}" for tag in TAGS)
                 + f"), {len(t['shared'])} shared, {len(t['unattached'])} unattached; "
                 f"{t['counts']['leaves']} changed leaves, {t['counts']['folded']} changes "
                 "listed after folding")
        for c in t["cards"]:
            heads = c["changes"]
            more = f"; +{len(heads) - per_card} more" if len(heads) > per_card else ""
            L.append(f"  {c['tag']:<8} {c['name']:<22} "
                     + "; ".join(describe_change(h, 24) for h in heads[:per_card]) + more)
        for ch in t["shared"][:per_card]:
            L.append(f"  shared   {describe_change(ch, 40)}")
        for ch in t["unattached"][:per_card]:
            L.append(f"  unattached {ch['path']}: {describe_change(ch, 24)}")
    led = result.get("ledger")
    if led is not None:
        c = led["counts"]
        per = ", ".join(f"{s} {len(v['changed']) + len(v['added']) + len(v['removed'])}"
                        for s, v in led["sections"].items())
        L.append(f"ledger: {c['value']} keys changed, {c['added']} added, {c['removed']} "
                 f"removed, {c['listed_only']} candidates / status only"
                 + (f" ({per})" if per else ""))
        arms = led["arms"]
        if arms["added"] or arms["removed"]:
            L.append("  card-table arms " + changed_list(arms))
        if arms["default_old"] != arms["default_new"]:
            L.append(f"  default arm {arms['default_old']} -> {arms['default_new']}")
    L.append(f"class: {result['class']}"
             + (f" ({result['balance']})" if result.get("balance") else ""))
    L.extend(f"  {why}" for why in result["reasons"])
    return L


def _md(text: Any) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def render_md(result: Mapping[str, Any]) -> str:
    """The card-data section of the update report (docs/SEASON_UPDATES.md 4.3)."""
    L: list[str] = []
    w = L.append
    w("## Card data\n")
    w(f"Generated by `{TOOL}`. Class **{result['class']}**"
      + (f" (balance {result['balance']})" if result.get("balance") else "") + ".\n")
    for why in result["reasons"]:
        w(f"- {_md(why)}")
    if result["reasons"]:
        w("")
    src = result.get("sources") or {}
    if src:
        w("| side | kind | version | file |")
        w("|---|---|---|---|")
        for side in ("old", "new", "old_ledger", "new_ledger"):
            s = src.get(side)
            if s:
                w(f"| {side} | {s.get('kind') or ''} | {_md(s.get('version') or '')} | "
                  f"`{_md(s.get('path') or '')}` |")
        w("")
    t = result.get("table")
    if t is not None:
        r = t["roster"]
        w("### Roster\n")
        w("| section | added | removed |")
        w("|---|---|---|")
        for s in (*ROSTER, *MAPS):
            w(f"| {s} | {_md(_names(r[s]['added'])) or '-'} | "
              f"{_md(_names(r[s]['removed'])) or '-'} |")
        for change in ("added", "removed"):
            for f in r["forms"][change]:
                w(f"\nForm {change}: {f['base']} {f['form']} ({', '.join(f['names'])}).")
        for f in r["forms"].get("records", ()):
            w(f"\nForm records of {f['base']} {f['form']}: {changed_list(f)}.")
        w("")
        w("### Values per card\n")
        if not t["cards"]:
            w("No card's values changed.\n")
        else:
            w("Echoes (the same change in the card's unit, projectile and raw CSV records) are "
              "folded onto the card. Tags: OUR DECK (Hog 2.6 and its forms, and the towers), "
              f"POOL (at least {100 * POOL_MIN:.0f}% of the opponent pool), OTHER.\n")
            w("| tag | card | change | old | new | % | echoes |")
            w("|---|---|---|---|---|---|---|")
            for c in t["cards"]:
                for h in c["changes"]:
                    size = f"{h['pct']:+.2f}" if h.get("pct") is not None else ""
                    w(f"| {c['tag']} | {c['name']} | {_md(h['field'])} | "
                      f"{_md(show(h.get('old'))) if 'old' in h else '(absent)'} | "
                      f"{_md(show(h.get('new'))) if 'new' in h else '(absent)'} | {size} | "
                      f"{len(h['echoes'])} |")
            w("")
        for title, items in (("Shared values (rarities, globals)", t["shared"]),
                             ("Unattached values (no card reaches them)", t["unattached"]),
                             ("Other sections", t["other"])):
            if items:
                w(f"### {title}\n")
                for ch in items:
                    w(f"- `{_md(ch['path'])}`: {_md(describe_change(ch))}")
                w("")
    led = result.get("ledger")
    if led is not None:
        w("### Ledger\n")
        c = led["counts"]
        w(f"{c['value']} keys changed, {c['added']} added, {c['removed']} removed; "
          f"{c['listed_only']} changed in candidates or status only (not classed).\n")
        arms = led["arms"]
        w(f"Card-table arms: {', '.join(arms['new']) or 'none'}"
          + (f" (added {', '.join(arms['added'])})" if arms["added"] else "")
          + (f" (removed {', '.join(arms['removed'])})" if arms["removed"] else "")
          + f"; the engine's default arm {arms['default_new']}.\n")
        if led["sections"]:
            w("| section | key | change |")
            w("|---|---|---|")
            for s, v in led["sections"].items():
                for k in v["added"]:
                    w(f"| {s} | {k} | added |")
                for k in v["removed"]:
                    w(f"| {s} | {k} | removed |")
                for ch in v["changed"]:
                    parts = []
                    for name, f in ch["fields"].items():
                        if name == "candidates" and ("added" in f or "removed" in f):
                            parts.append("candidates " + changed_list(f))
                        else:
                            parts.append(f"{name} {show(f.get('old'), 40)} -> "
                                         f"{show(f.get('new'), 40)}")
                    w(f"| {s} | {ch['key']} | {_md('; '.join(parts))} |")
            w("")
        for m in led["meta"]:
            w(f"- meta `{m['key']}`: {show(m.get('old'))} -> {show(m.get('new'))}")
        if led["meta"]:
            w("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _source(path: Path, data: Any, kind: str) -> dict[str, Any]:
    version = data.get("version") if kind == "table" else None
    if kind == "ledger":
        meta = data.get(LEDGER_META)
        version = meta.get("target_game_version") if isinstance(meta, Mapping) else None
    return {"path": str(path), "kind": kind, "version": version}


def read_pair(old: Path, new: Path) -> tuple[str, Any, Any]:
    """(kind, old data, new data); DiffError unless both are tables or both ledgers."""
    a, b = load_json(old), load_json(new)
    ka, kb = kind_of(a), kind_of(b)
    for path, kind in ((old, ka), (new, kb)):
        if kind is None:
            raise DiffError(f"{path}: neither a card table (a 'cards' list) nor a ledger "
                            "(sections of {'value': ...} entries)")
    if ka != kb:
        raise DiffError(f"{old} is a {ka}, {new} a {kb}: give two of one kind")
    return ka, a, b


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Diff two RoyaleSim card tables and/or ledgers: roster, values per card "
                    "(echoes folded), ledger keys, and the update class (see the module doc).")
    ap.add_argument("old", type=Path, help="the old card table or ledger")
    ap.add_argument("new", type=Path, help="the new card table or ledger")
    ap.add_argument("--ledgers", type=Path, nargs=2, metavar=("OLD", "NEW"), default=None,
                    help="also diff these two ledgers (calibration.json) with the two tables")
    ap.add_argument("--pool", type=Path, default=None, metavar="FILE",
                    help="opponent pool weights: a deck_stats.json or a {card: weight} object "
                         "(default none: no card is POOL)")
    ap.add_argument("--pool-split", default="train",
                    help="the deck_stats split to weigh (default train)")
    ap.add_argument("--pool-min", type=float, default=POOL_MIN, metavar="SHARE",
                    help=f"pool weight that makes a card POOL (default {POOL_MIN})")
    ap.add_argument("--json", type=Path, default=None, metavar="FILE",
                    help="write the whole diff as JSON")
    ap.add_argument("--md", type=Path, default=None, metavar="FILE",
                    help="write the report section as markdown")
    args = ap.parse_args(argv)
    if not 0.0 <= args.pool_min <= 1.0:
        ap.error(f"--pool-min: a share from 0 to 1, got {args.pool_min}")
    try:
        kind, old, new = read_pair(args.old, args.new)
        sources = {"old": _source(args.old, old, kind), "new": _source(args.new, new, kind)}
        tables = (old, new) if kind == "table" else (None, None)
        ledgers = (old, new) if kind == "ledger" else (None, None)
        if args.ledgers:
            if kind == "ledger":
                raise DiffError("--ledgers: the positional files are ledgers already")
            lkind, la, lb = read_pair(*args.ledgers)
            if lkind != "ledger":
                raise DiffError(f"--ledgers: {args.ledgers[0]} and {args.ledgers[1]} are "
                                f"{lkind}s, not ledgers")
            ledgers = (la, lb)
            sources["old_ledger"] = _source(args.ledgers[0], la, "ledger")
            sources["new_ledger"] = _source(args.ledgers[1], lb, "ledger")
        weights = None
        if args.pool is not None:
            weights = pool_weights(load_json(args.pool), args.pool_split)
            sources["pool"] = {"path": str(args.pool), "kind": "pool",
                               "version": args.pool_split}
    except DiffError as exc:
        print(f"{TOOL}: {exc}", file=sys.stderr)
        return 1
    result = diff(*tables, *ledgers, weights=weights, pool_min=args.pool_min, sources=sources)
    for line in summary_lines(result):
        print(line)
    for path, text in ((args.json, lambda: json.dumps(result, indent=1, ensure_ascii=False)),
                       (args.md, lambda: render_md(result))):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text(), encoding="utf-8")
            print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
