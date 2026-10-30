r"""Load a token-policy checkpoint only into the engine it was trained for.

    import rs_checkpoint
    model, meta, drift = rs_checkpoint.load_policy("runs/tokens/rs_bc/best.pt", engine)

    .venvs\royalegym\Scripts\python.exe tools\rs_checkpoint.py runs\tokens\rs_bc\best.pt
        checks checkpoints against data/engine.json's engine (--card-table / --engine-config
        as the other tools) and prints each one's notes; exit 1 on a mismatch

Why (tokens-12, rsim-01): a row's card ids are catalogue ids + 2, and its unit ids
unit_types_json() indexes + 2, of the engine that built it (tools/rs_rows.py). Another
royalesim or card table renumbers them (the unit vocabulary differs between tables, 237
unit types on the default table on the PC), and a policy loaded into it reads every unit as
another one and plays nonsense, silently. vocab.json records both lists, the card table and
the engine fingerprint, and sim/train_tokens.py copies vocab.json into every checkpoint's
meta.

load_policy(ckpt, engine, allow_drift=False) builds the TokenPolicy exactly as RSAgent did,
after checking, against the engine (an rs_engine.EngineConfig, None for the process's
rs_engine.active(), a Battle, or an EngineVocab):

  * meta["cards"] (name -> token id) in token order equals the engine's catalogue names in
    id order, and meta["units"] its unit_types_json(), by name and order;
  * the card table the checkpoint's rows were built under (meta "card_table", else its
    engine_fingerprint's, else its engine_config's) is the one the engine runs; aliases and
    revisions compare equal, as rs_engine.table_name reads them. A side that does not say
    (royalesim before 0.1.42, a checkpoint from before E01) is not compared. This one refuses
    only when a name differs too (U01, below).

Otherwise it raises VocabMismatch, naming the first difference. allow_drift=True loads anyway
with a VocabDriftWarning listing them; allow_drift=None (the default) reads
HASTYCR_ALLOW_VOCAB_DRIFT=1, which rs_eval --allow-vocab-drift sets for its workers. When every
card (and unit type) the checkpoint knows is still in the engine and only new ones were added,
the refusal names tools/rs_vocab.py grow, which carries it across by copying its embedding rows
by name (docs/SEASON_UPDATES.md 6.1).

NOTES (VocabDriftWarning, never a refusal; vocab_differences' second list):
  * a --live-compat checkpoint's unit tokens are card ids, so a different unit list is a note;
  * U01: another card table when every name matches (the cards in id order, and the unit types
    unless --live-compat). The guard exists to catch renumbering, which the name comparisons
    catch; a balance update keeps the names and changes values, so the checkpoint loads and
    says it learned under the other table;
  * U01: the royalesim version its rows were built under is not the engine's, or the table has
    the same arm but another card_table_fnv1a64 (a same-named table whose bytes changed). Both
    are compared only when the meta and the engine record them (royalesim 0.1.40+ for the FNV).

A checkpoint whose meta has no cards / units (from before vocab.json recorded them) cannot be
checked and is refused unless drift is allowed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:      # also when imported as tools.rs_checkpoint
    sys.path.insert(0, str(ROOT / "tools"))
import rs_engine  # noqa: E402

ALLOW_DRIFT_ENV = "HASTYCR_ALLOW_VOCAB_DRIFT"
_TRUE = frozenset({"1", "true", "yes", "on"})


class VocabMismatch(rs_engine.EngineError):
    """The checkpoint's card / unit vocabulary or card table is not the engine's."""

    def __init__(self, message: str, differences: Sequence[str] = ()):
        super().__init__(message)
        self.differences = list(differences)


class VocabDriftWarning(UserWarning):
    """A checkpoint loaded although its vocabulary differs from the engine's."""


def allow_drift_from_env(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return (env.get(ALLOW_DRIFT_ENV) or "").strip().lower() in _TRUE


def resolve_allow_drift(allow_drift: bool | None) -> bool:
    """allow_drift, or HASTYCR_ALLOW_VOCAB_DRIFT when it is None."""
    return allow_drift_from_env() if allow_drift is None else bool(allow_drift)


def set_allow_drift(allow: bool = True) -> None:
    """Allow (or stop allowing) drift in this process and the worker processes it starts
    later: they inherit the environment."""
    if allow:
        os.environ[ALLOW_DRIFT_ENV] = "1"
    else:
        os.environ.pop(ALLOW_DRIFT_ENV, None)


def _read(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    return value() if callable(value) else value


def _arm(table: Any) -> str | None:
    """A card-table name as rs_engine.table_name reads it; None for 'unknown'."""
    if not isinstance(table, str):
        return None
    try:
        return rs_engine.table_name(table)
    except ValueError:
        return None


def _fnv(value: Any) -> str | None:
    """An FNV-1a 64 as 16 lowercase hex digits (an int, or the text recorded); None if unknown."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"{value & 0xFFFF_FFFF_FFFF_FFFF:016x}"
    text = str(value).strip().lower().removeprefix("0x")
    return text or None


def _version(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return None if text in ("", "?") else text


# ---------------------------------------------------------------------------
# the engine's vocabulary
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EngineVocab:
    """What the ids mean in an engine: catalogue names in card id order, unit_types_json()
    in unit_type order, and the card table it runs (None when it does not say), with its
    card_table_fnv1a64 when the Battle reports one (royalesim 0.1.40+)."""

    cards: tuple[str, ...]
    units: tuple[str, ...]
    card_table: str | None = None
    fingerprint: Mapping[str, Any] | None = None
    card_table_fnv1a64: str | None = None

    @classmethod
    def from_battle(cls, battle: Any, rs: ModuleType | None = None,
                    cfg: rs_engine.EngineConfig | None = None) -> "EngineVocab":
        cards = tuple(str(row[0]) for row in json.loads(battle.catalogue_json()))
        units = tuple(str(u) for u in json.loads(battle.unit_types_json()))
        table = _read(battle, "card_table")
        if table is None and cfg is not None:
            table = cfg.card_table          # what was asked; the engine reports nothing
        fp = rs_engine.fingerprint(battle, rs, cfg) if rs is not None else None
        return cls(cards, units, None if table is None else str(table), fp,
                   _fnv(_read(battle, "card_table_fnv1a64")))

    @classmethod
    def from_vocab(cls, vocab: Mapping[str, Any]) -> "EngineVocab":
        """From a vocab.json (or a checkpoint meta): what the rows' ids meant."""
        cards, _ = _ordered("card", vocab.get("cards"))
        units, _ = _ordered("unit type", vocab.get("units"))
        return cls(tuple(cards or ()), tuple(units or ()), meta_card_table(vocab),
                   vocab.get("engine_fingerprint"), meta_fnv(vocab))

    @property
    def version(self) -> str | None:
        """The royalesim version, when the fingerprint says."""
        return _version((self.fingerprint or {}).get("version"))

    @property
    def fnv(self) -> str | None:
        """The card table's FNV-1a 64, from the Battle or the fingerprint; None if unknown."""
        return (_fnv(self.card_table_fnv1a64)
                or _fnv((self.fingerprint or {}).get("card_table_fnv1a64")))

    def describe(self) -> str:
        fp = self.fingerprint or {}
        parts = [f"royalesim {fp['version']}" if fp.get("version") else "the engine",
                 f"card table {self.card_table or 'n/a'}"
                 + (f" (fnv1a64 {self.fnv})" if self.fnv else ""),
                 f"{len(self.cards)} cards, {len(self.units)} unit types"]
        if fp.get("id"):
            parts.append(f"id {fp['id']}")
        return ", ".join(parts)


_VOCABS: dict[str, tuple[ModuleType, EngineVocab]] = {}


def engine_vocab(engine: Any = None, rs: ModuleType | None = None) -> EngineVocab:
    """`engine` as an EngineVocab: an EngineVocab as is; a Battle (anything with
    catalogue_json and unit_types_json) read directly; a vocab.json-like mapping (with
    "cards" and "units"); else an engine config (rs_engine.resolve: None = active()), whose
    Battle is built once per process and config."""
    if isinstance(engine, EngineVocab):
        return engine
    if hasattr(engine, "catalogue_json") and hasattr(engine, "unit_types_json"):
        return EngineVocab.from_battle(engine, rs)
    if isinstance(engine, Mapping) and "cards" in engine and "units" in engine:
        return EngineVocab.from_vocab(engine)
    cfg = rs_engine.resolve(engine).undelayed()
    rs = rs_engine.royalesim_module(rs)
    hit = _VOCABS.get(cfg.key())
    if hit is not None and hit[0] is rs:
        return hit[1]
    vocab = EngineVocab.from_battle(rs_engine.make_battle(cfg, rs), rs, cfg)
    _VOCABS[cfg.key()] = (rs, vocab)
    return vocab


# ---------------------------------------------------------------------------
# the checkpoint's vocabulary, and the comparison
# ---------------------------------------------------------------------------

def _ordered(kind: str, mapping: Any) -> tuple[list[str] | None, str | None]:
    """A vocab.json name -> token id map as names in token order, and what is wrong with it
    (None: tokens are 2 .. N+1, each once)."""
    if not isinstance(mapping, Mapping) or not mapping:
        return None, (f"the checkpoint's meta has no {kind} vocabulary (made before "
                      "vocab.json recorded it), so its ids cannot be checked")
    try:
        items = sorted(((str(k), int(v)) for k, v in mapping.items()), key=lambda kv: kv[1])
    except (TypeError, ValueError):
        return None, f"the checkpoint's {kind} vocabulary has a token that is not a number"
    tokens = [t for _, t in items]
    if tokens != list(range(2, len(tokens) + 2)):
        bad = next(i for i, t in enumerate(tokens) if t != i + 2)
        return [k for k, _ in items], (f"the checkpoint's {kind} tokens are not 2..{len(tokens) + 1}: "
                                       f"{items[bad][0]!r} has token {items[bad][1]}")
    return [k for k, _ in items], None


def names_of(meta: Mapping[str, Any], key: str) -> tuple[list[str] | None, str | None]:
    """meta["cards"] / meta["units"] as names in token order, and what is wrong with it
    (None when its tokens are 2 .. N+1, each once)."""
    return _ordered("card" if key == "cards" else "unit type", meta.get(key))


def meta_card_table(meta: Mapping[str, Any]) -> str | None:
    """The card table a checkpoint's rows were built under, as recorded: "card_table" (what
    the Battle reported), else engine_fingerprint's, else engine_config's arm."""
    for value in (meta.get("card_table"),
                  (meta.get("engine_fingerprint") or {}).get("card_table"),
                  (meta.get("engine_config") or {}).get("card_table")):
        if isinstance(value, str) and value.strip():
            return value
    return None


def meta_version(meta: Mapping[str, Any]) -> str | None:
    """The royalesim version a checkpoint's rows were built under: its engine_fingerprint's,
    else the vocab.json "version"; None when neither says ("?" is not saying)."""
    return (_version((meta.get("engine_fingerprint") or {}).get("version"))
            or _version(meta.get("version")))


def meta_fnv(meta: Mapping[str, Any]) -> str | None:
    """The card_table_fnv1a64 of the table a checkpoint's rows were built under, as its
    engine_fingerprint recorded it (royalesim 0.1.40+); None when it did not."""
    return (_fnv((meta.get("engine_fingerprint") or {}).get("card_table_fnv1a64"))
            or _fnv(meta.get("card_table_fnv1a64")))


@dataclasses.dataclass(frozen=True)
class NameChanges:
    """How a name list (catalogue names in card id order, unit types in unit id order) changed
    from `old` to `new`: names only in new (in new order), names only in old (in old order),
    and names in both at another index, as (name, old index, new index)."""

    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    moved: tuple[tuple[str, int, int], ...] = ()
    appended: bool = False      # new starts with all of old, in order, and is longer

    @property
    def same(self) -> bool:
        return not (self.added or self.removed or self.moved)

    def describe(self, kind: str) -> str:
        """'1 card appended (Witch)', '2 unit types added (A, B), 5 moved', 'the same'."""
        def some(names: Sequence[str]) -> str:
            shown = ", ".join(names[:3])
            return shown + (f", ... {len(names) - 3} more" if len(names) > 3 else "")
        parts = []
        for names, verb in ((self.added, "appended" if self.appended else "added"),
                            (self.removed, "removed")):
            if names:
                parts.append(f"{len(names)} {kind}{'s' if len(names) != 1 else ''} {verb} "
                             f"({some(names)})")
        if self.moved:
            parts.append(f"{len(self.moved)} moved")
        return ", ".join(parts) or "the same"


def name_changes(old: Sequence[str], new: Sequence[str]) -> NameChanges:
    old, new = [str(n) for n in old], [str(n) for n in new]
    at_old = {n: i for i, n in enumerate(old)}
    at_new = {n: i for i, n in enumerate(new)}
    added = tuple(n for n in new if n not in at_old)
    removed = tuple(n for n in old if n not in at_new)
    moved = tuple((n, i, at_new[n]) for i, n in enumerate(old) if n in at_new and at_new[n] != i)
    appended = len(new) > len(old) and new[:len(old)] == old
    return NameChanges(added, removed, moved, appended)


def grow_blockers(meta: Mapping[str, Any], vocab: EngineVocab) -> list[str]:
    """Why tools/rs_vocab.py grow cannot carry the checkpoint `meta` describes into `vocab`'s
    engine by name ([] when it can): a vocabulary it does not record or that is malformed, a
    card (or, unless --live-compat, a unit type) it knows that the engine no longer has, or a
    --units vision checkpoint (its unit tokens are vision classes, not the engine's units)."""
    out: list[str] = []
    live = bool(meta.get("live_compat", False))
    keys = ("cards",) if live else ("cards", "units")
    if not live and meta.get("units_mode", "types") != "types":
        out.append(f"a --units {meta.get('units_mode')} checkpoint: its unit tokens are not the "
                   "engine's unit types, so they cannot be carried by name")
        keys = ("cards",)
    for key in keys:
        kind = "card" if key == "cards" else "unit type"
        names, problem = names_of(meta, key)
        if problem:
            out.append(problem)
            continue
        gone = name_changes(names or (), vocab.cards if key == "cards" else vocab.units).removed
        if gone:
            out.append(f"{len(gone)} {kind}(s) it knows are not in the engine: "
                       + ", ".join(repr(n) for n in gone[:5]) + (", ..." if len(gone) > 5 else ""))
    return out


def grow_hint(meta: Mapping[str, Any], vocab: EngineVocab, source: str = "CKPT") -> str | None:
    """The sentence a refusal adds when the engine only added cards (or unit types) after the
    checkpoint's: tools/rs_vocab.py grow carries it across. None otherwise."""
    if grow_blockers(meta, vocab):
        return None
    live = bool(meta.get("live_compat", False))
    cards = name_changes(names_of(meta, "cards")[0] or (), vocab.cards)
    units = NameChanges() if live else name_changes(names_of(meta, "units")[0] or (), vocab.units)
    if not (cards.added or units.added):
        return None
    what = [c for c in (cards.describe("card") if cards.added else "",
                        units.describe("unit type") if units.added else "") if c]
    return (f"Every card{'' if live else ' and unit type'} it knows is still in the engine, "
            f"which has {'; '.join(what)}: tools/rs_vocab.py grow {source} --out DIR carries it "
            "across, copying its embedding rows by name (a new one starts as the unknown row).")


def _first_difference(kind: str, mine: Sequence[str], theirs: Sequence[str]) -> str | None:
    for i, (a, b) in enumerate(zip(mine, theirs)):
        if a != b:
            return f"{kind} {i} (token {i + 2}): the checkpoint has {a!r}, the engine {b!r}"
    if len(mine) != len(theirs):
        i = min(len(mine), len(theirs))
        where, name = (("only in the checkpoint", mine[i]) if len(mine) > i
                       else ("only in the engine", theirs[i]))
        return (f"{kind}s: the checkpoint has {len(mine)}, the engine {len(theirs)}; "
                f"{kind} {i} (token {i + 2}) is {where}: {name!r}")
    return None


def other_arm(meta: Mapping[str, Any], vocab: EngineVocab) -> bool:
    """Whether the checkpoint's rows and the engine name different card tables (both must say)."""
    mine, theirs = _arm(meta_card_table(meta)), _arm(vocab.card_table)
    return bool(mine and theirs and mine != theirs)


def drift_notes(meta: Mapping[str, Any], vocab: EngineVocab) -> list[str]:
    """U01's warn-only notes on what the ids do not show: another royalesim version than the
    rows', and, on the same card table (or one side not naming it), another card_table_fnv1a64.
    Each is compared only when both sides record it."""
    notes: list[str] = []
    mine, theirs = meta_version(meta), vocab.version
    if mine and theirs and mine != theirs:
        notes.append(f"royalesim: the checkpoint's rows were built under royalesim {mine}, the "
                     f"engine is {theirs}; the ids are checked, the simulation may differ")
    if not other_arm(meta, vocab):
        mine, theirs = meta_fnv(meta), vocab.fnv
        if mine and theirs and mine != theirs:
            table = meta_card_table(meta) or vocab.card_table or "the card table"
            notes.append(f"card table bytes: the checkpoint's rows were built under {table} with "
                         f"FNV-1a 64 {mine}, the engine's has {theirs}: the same table, other "
                         "values")
    return notes


def vocab_differences(meta: Mapping[str, Any], vocab: EngineVocab
                      ) -> tuple[list[str], list[str]]:
    """(differences that refuse the load, most telling first; notes, only warned about).

    Errors, in order: the cards, then the unit types (a note for a --live-compat checkpoint,
    whose unit tokens are card ids), then the card table, which is an error only next to one
    of them. When every name matches, another card table is a note (U01): the ids mean the
    same cards and units, only the values they were learned under differ. drift_notes follow.
    """
    errors: list[str] = []
    notes: list[str] = []
    live = bool(meta.get("live_compat", False))
    for kind, key, engine_names in (("card", "cards", vocab.cards),
                                    ("unit type", "units", vocab.units)):
        names, problem = _ordered(kind, meta.get(key))
        diff = problem or _first_difference(kind, names or [], engine_names)
        if diff is None:
            continue
        if key == "units" and live:
            notes.append(f"{diff} (a --live-compat checkpoint's unit tokens are card ids, so "
                         "its unit list is not used)")
        else:
            errors.append(diff)
    if other_arm(meta, vocab):
        table = (f"card table: the checkpoint's rows were built under {meta_card_table(meta)}, "
                 f"the engine runs {vocab.card_table}")
        if errors:
            errors.append(table)
        else:
            names = "card names are" if live else "card and unit type names are"
            notes.append(f"{table}; its {names} the engine's in id order, so it loads, with what "
                         "it learned under the other table's values")
    notes.extend(drift_notes(meta, vocab))
    return errors, notes


def _built_under(meta: Mapping[str, Any]) -> str:
    fp = meta.get("engine_fingerprint") or {}
    version = fp.get("version") or meta.get("version")
    parts = [f"royalesim {version}" if version else "an unrecorded engine",
             f"card table {meta_card_table(meta) or 'n/a'}"]
    if fp.get("id"):
        parts.append(f"id {fp['id']}")
    return ", ".join(parts)


def check_meta(meta: Mapping[str, Any], engine: Any = None, allow_drift: bool | None = None,
               *, rs: ModuleType | None = None, source: str = "checkpoint") -> list[str]:
    """Raise VocabMismatch unless `meta`'s vocabulary is `engine`'s (see the module doc).
    With drift allowed, warn (VocabDriftWarning) instead. Returns the differences that
    were allowed ([] when there are none)."""
    vocab = engine_vocab(engine, rs)
    errors, notes = vocab_differences(meta, vocab)
    for note in notes:
        warnings.warn(VocabDriftWarning(f"{source}: {note}"), stacklevel=2)
    if not errors:
        return []
    more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
    if not resolve_allow_drift(allow_drift):
        hint = grow_hint(meta, vocab, source if source != "checkpoint" else "CKPT")
        raise VocabMismatch(
            f"{source} does not fit this engine: {errors[0]}{more}. Its rows were built under "
            f"{_built_under(meta)}; this engine is {vocab.describe()}. "
            + (f"{hint} Or rebuild" if hint else "Rebuild")
            + " the rows (tools/rs_rows.py) and retrain under this engine, or run the engine it "
            "was made for (--card-table / --engine-config), or load it anyway with "
            f"rs_eval --allow-vocab-drift or {ALLOW_DRIFT_ENV}=1.", errors)
    warnings.warn(VocabDriftWarning(
        f"{source}: loading despite {len(errors)} vocabulary difference(s): "
        + "; ".join(errors[:5]) + ("; ..." if len(errors) > 5 else "")), stacklevel=2)
    return errors


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

class LoadedPolicy(NamedTuple):
    model: Any                  # sim.model_tokens.TokenPolicy, in eval mode
    meta: dict
    drift: list[str]            # differences allowed by allow_drift ([] when none)


def _shown(ckpt: Any) -> str:
    if isinstance(ckpt, Mapping):
        return "checkpoint"
    return str(ckpt)


def _load_blob(ckpt: Any, map_location: Any = "cpu") -> Mapping[str, Any]:
    if isinstance(ckpt, Mapping):
        return ckpt
    import torch
    return torch.load(ckpt, map_location=map_location)


def check_file(ckpt: Any, engine: Any = None, allow_drift: bool | None = None, *,
               rs: ModuleType | None = None) -> list[str]:
    """check_meta on a checkpoint file (or an already loaded blob)."""
    return check_meta(_load_blob(ckpt)["meta"], engine, allow_drift, rs=rs, source=_shown(ckpt))


def load_policy(ckpt: Any, engine: Any = None, allow_drift: bool | None = None, *,
                rs: ModuleType | None = None, map_location: Any = "cpu") -> LoadedPolicy:
    """(TokenPolicy in eval mode, meta, allowed differences) from a checkpoint file (or an
    already loaded {"model", "meta"} blob), after check_meta against `engine`. The model is
    built exactly as RSAgent built it: TokenPolicy(n_cards, n_units, d, layers, heads) with
    the meta's sizes (defaults 192 / 4 / 6); a meta whose "schema" is rs_rows/v2 builds the v2
    model (E15, sim/model_tokens.policy_from_meta). The state_dict load is strict."""
    blob = _load_blob(ckpt, map_location)
    meta = blob["meta"]
    drift = check_meta(meta, engine, allow_drift, rs=rs, source=_shown(ckpt))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from sim.model_tokens import policy_from_meta
    model = policy_from_meta(meta, blob["model"]).eval()
    model.load_state_dict(blob["model"])
    return LoadedPolicy(model, meta, drift)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check token-policy checkpoints against the engine's card and unit "
                    "vocabulary and card table.")
    ap.add_argument("ckpt", nargs="+", type=Path, help="checkpoint .pt files")
    rs_engine.add_engine_args(ap)
    args = ap.parse_args(argv)
    try:
        cfg = rs_engine.config_from_args(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"engine config: {exc}", file=sys.stderr)
        return 2
    try:
        vocab = engine_vocab(cfg)
    except Exception as exc:                # royalesim missing, or it refused the config
        print(f"engine: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"engine: {cfg.describe()}\n        {vocab.describe()}")
    bad = 0
    for path in args.ckpt:
        try:
            meta = _load_blob(path)["meta"]
            with warnings.catch_warnings():         # the notes are printed below
                warnings.simplefilter("ignore", VocabDriftWarning)
                check_meta(meta, vocab, allow_drift=False, source=str(path))
        except VocabMismatch as exc:
            bad += 1
            print(f"MISMATCH {exc}")
            for diff in exc.differences[1:]:
                print(f"         also: {diff}")
            continue
        except Exception as exc:
            bad += 1
            print(f"ERROR    {path}: {type(exc).__name__}: {exc}")
            continue
        notes = vocab_differences(meta, vocab)[1]
        print(f"ok       {path}" + (f" ({len(notes)} note{'s' if len(notes) != 1 else ''})"
                                    if notes else ""))
        for note in notes:
            print(f"  note   {note}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
