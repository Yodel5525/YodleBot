r"""Load a token-policy checkpoint only into the engine it was trained for.

    import rs_checkpoint
    model, meta, drift = rs_checkpoint.load_policy("runs/tokens/rs_bc/best.pt", engine)

    .venvs\royalegym\Scripts\python.exe tools\rs_checkpoint.py runs\tokens\rs_bc\best.pt
        checks checkpoints against data/engine.json's engine (--card-table / --engine-config
        as the other tools); exit 1 on a mismatch

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
    (royalesim before 0.1.42, a checkpoint from before E01) is not compared.

Otherwise it raises VocabMismatch, naming the first difference. allow_drift=True loads anyway
with a VocabDriftWarning listing them; allow_drift=None (the default) reads
HASTYCR_ALLOW_VOCAB_DRIFT=1, which rs_eval --allow-vocab-drift sets for its workers.

A --live-compat checkpoint's unit tokens are card ids, so a different unit list is only
warned about. A checkpoint whose meta has no cards / units (from before vocab.json recorded
them) cannot be checked and is refused unless drift is allowed.
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


# ---------------------------------------------------------------------------
# the engine's vocabulary
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EngineVocab:
    """What the ids mean in an engine: catalogue names in card id order, unit_types_json()
    in unit_type order, and the card table it runs (None when it does not say)."""

    cards: tuple[str, ...]
    units: tuple[str, ...]
    card_table: str | None = None
    fingerprint: Mapping[str, Any] | None = None

    @classmethod
    def from_battle(cls, battle: Any, rs: ModuleType | None = None,
                    cfg: rs_engine.EngineConfig | None = None) -> "EngineVocab":
        cards = tuple(str(row[0]) for row in json.loads(battle.catalogue_json()))
        units = tuple(str(u) for u in json.loads(battle.unit_types_json()))
        table = _read(battle, "card_table")
        if table is None and cfg is not None:
            table = cfg.card_table          # what was asked; the engine reports nothing
        fp = rs_engine.fingerprint(battle, rs, cfg) if rs is not None else None
        return cls(cards, units, None if table is None else str(table), fp)

    @classmethod
    def from_vocab(cls, vocab: Mapping[str, Any]) -> "EngineVocab":
        """From a vocab.json (or a checkpoint meta): what the rows' ids meant."""
        cards, _ = _ordered("card", vocab.get("cards"))
        units, _ = _ordered("unit type", vocab.get("units"))
        return cls(tuple(cards or ()), tuple(units or ()), meta_card_table(vocab),
                   vocab.get("engine_fingerprint"))

    def describe(self) -> str:
        fp = self.fingerprint or {}
        parts = [f"royalesim {fp['version']}" if fp.get("version") else "the engine",
                 f"card table {self.card_table or 'n/a'}",
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


def meta_card_table(meta: Mapping[str, Any]) -> str | None:
    """The card table a checkpoint's rows were built under, as recorded: "card_table" (what
    the Battle reported), else engine_fingerprint's, else engine_config's arm."""
    for value in (meta.get("card_table"),
                  (meta.get("engine_fingerprint") or {}).get("card_table"),
                  (meta.get("engine_config") or {}).get("card_table")):
        if isinstance(value, str) and value.strip():
            return value
    return None


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


def vocab_differences(meta: Mapping[str, Any], vocab: EngineVocab
                      ) -> tuple[list[str], list[str]]:
    """(differences that refuse the load, most telling first; differences only warned about).

    Order: the cards, then the unit types, then the card table.
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
    mine, theirs = meta_card_table(meta), vocab.card_table
    if _arm(mine) and _arm(theirs) and _arm(mine) != _arm(theirs):
        errors.append(f"card table: the checkpoint's rows were built under {mine}, the engine "
                      f"runs {theirs}")
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
        raise VocabMismatch(
            f"{source} does not fit this engine: {errors[0]}{more}. Its rows were built under "
            f"{_built_under(meta)}; this engine is {vocab.describe()}. Rebuild the rows "
            "(tools/rs_rows.py) and retrain under this engine, or run the engine it was "
            "made for (--card-table / --engine-config), or load it anyway with "
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
    the meta's sizes (defaults 192 / 4 / 6)."""
    blob = _load_blob(ckpt, map_location)
    meta = blob["meta"]
    drift = check_meta(meta, engine, allow_drift, rs=rs, source=_shown(ckpt))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from sim.model_tokens import TokenPolicy
    model = TokenPolicy(meta["n_cards"], meta["n_units"], d=meta.get("d", 192),
                        layers=meta.get("layers", 4), heads=meta.get("heads", 6)).eval()
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
            check_file(path, vocab, allow_drift=False)
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
        print(f"ok       {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
