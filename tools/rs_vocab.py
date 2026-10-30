r"""Carry token-policy checkpoints across engine updates by card and unit NAME (U14).

    .venvs\royalegym\Scripts\python.exe tools\rs_vocab.py diff runs\tokens\x\best.pt
        the checkpoint's vocabulary against data/engine.json's engine (--card-table /
        --engine-config as the other tools); `diff A B` compares two checkpoints or vocab.json
        files (a folder means its vocab.json)
    .venvs\royalegym\Scripts\python.exe tools\rs_vocab.py grow runs\tokens\x\best.pt --out runs\tokens\x_carried
        writes runs\tokens\x_carried\best.pt and args.json: the checkpoint in the engine's
        catalogue order, every row copied by name, a new card's row the unknown row
    .venvs\royalegym\Scripts\python.exe tools\rs_vocab.py reset-cold runs\tokens\x\best.pt --cards MinionGiant --out DIR
        the cards' rows set to the unknown row (cards the model never learned)
    .venvs\royalegym\Scripts\python.exe tools\rs_vocab.py check runs\tokens\x\best.pt [...] [--cold NAME ...] [--pool NAME ...]
        OK, REFUSED (with the fix), ENGINE_DRIFT or COLD for each checkpoint

Why (docs/SEASON_UPDATES.md 3.3, 6.1, 6.2): a checkpoint's card tokens are catalogue ids + 2 and
its unit tokens unit_types_json() indexes + 2 (or, --live-compat, creating-card ids + 2) of the
engine its rows were built under. RoyaleSim appends a new card after every other ("the next id",
py.rs:241-246), and sorts unit types by name, so a new form's unit renumbers the units after it.
tools/rs_checkpoint.py then refuses the checkpoint, rightly: its ids would mean other cards.
grow makes them mean the same ones again, by name.

WHAT IS VOCABULARY-SIZED in sim/model_tokens.TokenPolicy (:133-134, read through :195-270):
  card_emb.weight  [n_cards, d]. Token 0 pads (padding_idx), 1 is "unknown", 2.. a card. Every
                   card-token input reads it: hand, next, opp_seen, opp_last, and in v2
                   spell_ids, own_pending and ability_card (a press button has no embedding of
                   its own: card_emb of its card plus button_in of its state); the tile head
                   reads card_emb of the chosen card.
  unit_emb.weight  [n_units, d]. --units types: unit_type + 2; --live-compat: the creating
                   card + 2, n_units == n_cards, grown with the card map.
  Nothing else. Forms are features, not embeddings (v2 unit_feats and hand_feats "form"; the
  legacy unit_form column is no model input). The card choice is a pointer over the four
  hand tokens (card_query(cls) . hand), and every head has a fixed width (wait 1, tile 576,
  value 2, value2 2, one press logit per button token). grow checks this: every other tensor
  of the carried model has its old shape, or it refuses.

GROW (CKPT into the engine): refused (BREAKING) when a card the checkpoint knows is not in the
engine (or, unless --live-compat, a unit type), when its vocabulary is missing or malformed, and
for --units vision checkpoints (rs_checkpoint.grow_blockers). Otherwise each table row of a name
is copied to that name's new token; the pad row stays; a new name's row is a copy of the
unknown row (token 1), so the model reads a new card exactly as it reads an unnamed one. The
meta gets the engine's n_cards, cards, n_units and units; what says what the model learned
under (card_table, engine_fingerprint, engine_config, version) is kept, so loading it still
notes the drift (rs_checkpoint, U01); "vocab_carry" lists every grow and reset-cold, and
"cold_cards" the cards whose rows were never trained (the added ones). Before anything is
written: the source loads strictly, every other tensor keeps its shape, the largest |change|
of gate, tile, value (and value2) over PROBE_ROWS synthetic rows of old tokens is below
TOLERANCE (1e-4), every row is the old row of its name read through the two metas alone
(rows_by_name: the probe reads the old tokens through the same map the rows were copied by,
so it cannot see a wrong map), and the result passes a strict rs_checkpoint.load_policy into
the engine. It writes DIR/best.pt and DIR/args.json: the source's args.json (when it has one)
with "wait_keep", which rs_decide.wait_keep_for reads, as the source's (its default when it
had none), and the carry record; args.json lands first, so a best.pt there always has one, and
an existing best.pt or args.json in DIR needs --force.

RESET-COLD: the named cards' card_emb rows (and, --live-compat, their unit_emb rows; --units
for named unit tokens otherwise) become the unknown row; every other row is unchanged, which
the same probe checks on rows without them. It fixes a card no row ever trained, such as
Minion Giant (no IL_Replay key reaches it), whose row is its random N(0, 1) start (6.2). Hog
2.6 cards, which every row trains, are refused unless --allow-deck; the record says what each
reset row was (the unknown row, marked cold, or its own row), and the command line names the
cards that had their own row and no cold mark.

CHECK: REFUSED when rs_checkpoint refuses it (the fix: grow when every name it knows is still
in the engine, which added or moved some, as rs_checkpoint's refusal says; else rebuild and
retrain); COLD when a cold card is in the pool that matters (Hog 2.6 and --pool);
ENGINE_DRIFT when it loads with another card table, table FNV or royalesim version than its
rows'; else OK. Cold cards: meta "cold_cards", rows equal to the unknown row, and --cold.

Exit codes: 0 ok (diff: SAME); 1 refused, BREAKING or an error; 2 bad arguments; 3 diff LOADS or
check ENGINE_DRIFT; 4 check COLD; 5 diff GROW. check returns the worst over its checkpoints.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:      # also when imported as tools.rs_vocab
    sys.path.insert(0, str(ROOT / "tools"))
import rs_checkpoint as RC  # noqa: E402
import rs_engine  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_ARGS, EXIT_DRIFT, EXIT_COLD, EXIT_GROW = 0, 1, 2, 3, 4, 5
TOOL = "tools/rs_vocab.py"
PAD, UNKNOWN = 0, 1
TOLERANCE = 1e-4
PROBE_ROWS = 64
PROBE_BUTTONS = 2
CARD_EMB, UNIT_EMB = "card_emb.weight", "unit_emb.weight"
#: The model inputs that hold card tokens / unit tokens (sim/model_tokens.encode_all).
CARD_TOKEN_COLUMNS = ("hand", "next", "opp_seen", "opp_last", "spell_ids", "own_pending",
                      "ability_card")
UNIT_TOKEN_COLUMNS = ("unit_ids",)
#: Meta keys this tool writes.
CARRY_KEY, COLD_KEY = "vocab_carry", "cold_cards"
#: diff verdicts and check verdicts.
SAME, LOADS, GROW, BREAKING = "SAME", "LOADS", "GROW", "BREAKING"
OK, REFUSED, ENGINE_DRIFT, COLD = "OK", "REFUSED", "ENGINE_DRIFT", "COLD"
_CHECK_EXIT = {OK: EXIT_OK, ENGINE_DRIFT: EXIT_DRIFT, COLD: EXIT_COLD, REFUSED: EXIT_FAILED}
_SEVERITY = (OK, ENGINE_DRIFT, COLD, REFUSED)


class VocabError(ValueError):
    """grow or reset-cold refused: the checkpoint cannot be carried, or the result would not be
    the same model."""


# ---------------------------------------------------------------------------
# the difference
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class VocabDiff:
    """A checkpoint's (or vocab.json's) vocabulary against an engine's (or another's)."""

    cards: RC.NameChanges
    units: RC.NameChanges
    live: bool
    units_mode: str
    table: tuple[str | None, str | None]
    fnv: tuple[str | None, str | None]
    version: tuple[str | None, str | None]
    other_table: bool
    blockers: tuple[str, ...]

    @staticmethod
    def _changed(pair: tuple[str | None, str | None]) -> bool:
        return bool(pair[0] and pair[1] and pair[0] != pair[1])

    @property
    def fnv_changed(self) -> bool:
        return not self.other_table and self._changed(self.fnv)

    @property
    def version_changed(self) -> bool:
        return self._changed(self.version)

    @property
    def drift(self) -> bool:
        """Loads with another card table, table FNV or royalesim version than its rows'."""
        return self.other_table or self.fnv_changed or self.version_changed

    @property
    def ids_same(self) -> bool:
        """Every token means what it meant (a --live-compat checkpoint's units are cards)."""
        return self.cards.same and (self.live or self.units.same)

    @property
    def verdict(self) -> str:
        """BREAKING (grow cannot carry it), SAME, LOADS (the same ids, with notes) or GROW."""
        if self.blockers:
            return BREAKING
        if self.ids_same:
            return LOADS if self.drift or not self.units.same else SAME
        return GROW

    def lines(self) -> list[str]:
        def pair(p: tuple[str | None, str | None], changed: bool) -> str:
            a, b = (x or "n/a" for x in p)
            return f"{a} -> {b}" if changed else (f"{a} (the same)" if p[0] and p[1] else
                                                  f"{a} -> {b} (not compared)")
        units = self.units.describe("unit type")
        if self.live:
            units += " (--live-compat: its unit tokens are card ids; not used)"
        elif self.units_mode != "types":
            units += f" (--units {self.units_mode})"
        out = [f"cards       {self.cards.describe('card')}",
               f"unit types  {units}",
               f"card table  {pair(self.table, self.other_table)}",
               f"fnv1a64     {pair(self.fnv, self._changed(self.fnv))}",
               f"royalesim   {pair(self.version, self.version_changed)}",
               f"verdict     {self.verdict}"]
        out += [f"  blocked: {b}" for b in self.blockers]
        return out

    def to_json(self) -> dict:
        def names(c: RC.NameChanges) -> dict:
            return {"added": list(c.added), "removed": list(c.removed), "moved": len(c.moved),
                    "appended": c.appended}
        return {"verdict": self.verdict, "cards": names(self.cards), "units": names(self.units),
                "live_compat": self.live, "units_mode": self.units_mode,
                "card_table": list(self.table), "other_table": self.other_table,
                "fnv1a64": list(self.fnv), "fnv_changed": self.fnv_changed,
                "royalesim": list(self.version), "version_changed": self.version_changed,
                "blockers": list(self.blockers)}


def vocab_diff(meta: Mapping[str, Any], engine: Any) -> VocabDiff:
    """`meta` (a checkpoint meta or vocab.json) against `engine` (what rs_checkpoint.engine_vocab
    takes: an EngineVocab, a Battle, another vocab.json-like mapping, an engine config)."""
    vocab = RC.engine_vocab(engine)
    return VocabDiff(
        cards=RC.name_changes(RC.names_of(meta, "cards")[0] or (), vocab.cards),
        units=RC.name_changes(RC.names_of(meta, "units")[0] or (), vocab.units),
        live=bool(meta.get("live_compat", False)),
        units_mode=str(meta.get("units_mode") or "types"),
        table=(RC.meta_card_table(meta), vocab.card_table),
        fnv=(RC.meta_fnv(meta), vocab.fnv),
        version=(RC.meta_version(meta), vocab.version),
        other_table=RC.other_arm(meta, vocab),
        blockers=tuple(RC.grow_blockers(meta, vocab)))


def token_map(old: Sequence[str], new: Sequence[str]) -> list[int | None]:
    """For each token of `new` (0 .. len(new) + 1), the token of `old` whose row it takes:
    0 -> 0 (pad), 1 -> 1 (unknown), a name -> its old token, None for a name `old` lacks."""
    at = {str(n): i + 2 for i, n in enumerate(old)}
    return [PAD, UNKNOWN] + [at.get(str(n)) for n in new]


def take_rows(table: Any, mapping: Sequence[int | None], unknown: int = UNKNOWN) -> Any:
    """A new embedding table: row j is table[mapping[j]], or table[unknown] where it is None."""
    import torch
    idx = torch.tensor([unknown if m is None else int(m) for m in mapping], dtype=torch.long,
                       device=table.device)
    return table.index_select(0, idx).clone()


# ---------------------------------------------------------------------------
# function preservation
# ---------------------------------------------------------------------------

def probe_batch(spec: Any, card_tokens: Sequence[int], unit_tokens: Sequence[int], *,
                rows: int = PROBE_ROWS, buttons: int = PROBE_BUTTONS, seed: int = 0) -> dict:
    """A synthetic batch of `rows` rows of the schema `spec` (its model_inputs): card-token
    columns drawn from `card_tokens`, unit_ids from `unit_tokens` (each token at least once
    when the column has room: one place per token is never padded; about a quarter of the
    others is padding, except in the hand), features uniform in [0, 1). An integer input that
    is neither is refused: it might hold tokens this tool does not remap."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    out = {}
    for name in spec.model_inputs:
        col = spec.column(name)
        shape = (int(rows), *col.resolved(buttons))
        if name in CARD_TOKEN_COLUMNS or name in UNIT_TOKEN_COLUMNS:
            pool = list(card_tokens if name in CARD_TOKEN_COLUMNS else unit_tokens) or [PAD]
            pool_t = torch.tensor(pool, dtype=torch.long)
            n = math.prod(shape)
            perm = torch.randperm(n, generator=g)
            values = pool_t[perm % len(pool)].reshape(shape)    # place k: pool[perm[k] % len]
            if name != "hand":
                first = (perm < len(pool)).reshape(shape)       # each token's one sure place
                pad = (torch.rand(shape, generator=g) < 0.25) & ~first
                values = values.masked_fill(pad, PAD)
            out[name] = values
        elif col.dtype.startswith(("int", "bool")):
            raise VocabError(f"model input {name!r} ({col.dtype}) is neither a card nor a unit "
                             f"token column of {TOOL}; check whether it holds tokens before "
                             "carrying the checkpoint")
        else:
            out[name] = torch.rand(shape, generator=g)
    return out


def _remap(batch: Mapping[str, Any], card_lut: Any, unit_lut: Any) -> dict:
    out = dict(batch)
    for name, value in batch.items():
        if name in CARD_TOKEN_COLUMNS:
            out[name] = card_lut[value]
        elif name in UNIT_TOKEN_COLUMNS:
            out[name] = unit_lut[value]
    return out


def _lut(mapping: Sequence[int | None], old_rows: int) -> tuple[Any, list[int]]:
    """(old token -> new token as a LongTensor of old_rows, the old tokens kept)."""
    import torch
    lut = torch.zeros(old_rows, dtype=torch.long)
    kept = []
    for new, old in enumerate(mapping):
        if old is not None and 0 <= old < old_rows:
            lut[old] = new
            kept.append(old)
    return lut, sorted(set(kept))


def preservation_delta(old_model: Any, new_model: Any, card_map: Sequence[int | None],
                       unit_map: Sequence[int | None], *, skip_cards: Iterable[int] = (),
                       skip_units: Iterable[int] = (), rows: int = PROBE_ROWS,
                       seed: int = 0) -> float:
    """The largest |difference| of gate, tile, value (and value2) between `old_model` on a probe
    batch of its tokens and `new_model` on the same batch with the tokens mapped (card_map /
    unit_map: new token -> old token, as token_map gives). skip_*: old tokens left out of the
    probe (reset-cold's). Both models are put in eval mode."""
    import torch
    old_model.eval()
    new_model.eval()
    card_rows = old_model.card_emb.weight.shape[0]
    unit_rows = old_model.unit_emb.weight.shape[0]
    card_lut, cards = _lut(card_map, card_rows)
    unit_lut, units = _lut(unit_map, unit_rows)
    skip_c, skip_u = set(skip_cards) | {PAD}, set(skip_units) | {PAD}
    batch = probe_batch(old_model.schema, [t for t in cards if t not in skip_c],
                        [t for t in units if t not in skip_u], rows=rows, seed=seed)
    moved = _remap(batch, card_lut, unit_lut)
    with torch.no_grad():
        a = old_model.outputs(batch, card_ids=batch["hand"][:, 0])
        b = new_model.outputs(moved, card_ids=moved["hand"][:, 0])
    keys = [k for k in ("gate", "tile", "value", "value2") if k in a]
    return max(float((a[k] - b[k]).abs().max()) for k in keys)


# ---------------------------------------------------------------------------
# grow and reset-cold
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Carried:
    """What grow / reset_cold give: the checkpoint to write, the record "vocab_carry" got, the
    probe's largest change, the strict load's notes, and the diff (grow)."""

    blob: dict
    record: dict
    max_delta: float
    notes: list[str]
    diff: VocabDiff | None = None


def _model_tools():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from sim.model_tokens import policy_from_meta
    return policy_from_meta


def _strict_model(meta: Mapping[str, Any], state: Mapping[str, Any], what: str) -> Any:
    policy_from_meta = _model_tools()
    try:
        model = policy_from_meta(meta, state)
        model.load_state_dict(state)
    except (RuntimeError, KeyError, TypeError, ValueError) as exc:
        raise VocabError(f"{what} does not load as the TokenPolicy its meta describes: "
                         f"{type(exc).__name__}: {exc}") from None
    return model.eval()


def _check_shapes(meta: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    """Every tensor of `state` is the model's (by name and shape): a vocabulary-sized tensor
    grow does not know would differ here."""
    want = _model_tools()(meta, state).state_dict()
    wrong = [k for k in want if k in state and tuple(want[k].shape) != tuple(state[k].shape)]
    extra = [k for k in state if k not in want]
    missing = [k for k in want if k not in state]
    if wrong or extra or missing:
        raise VocabError("the carried weights are not the carried model's: "
                         + "; ".join(f"{what} {keys[:5]}" for what, keys in
                                     (("shape differs:", wrong), ("not in the model:", extra),
                                      ("missing:", missing)) if keys)
                         + f" ({TOOL} grows only {CARD_EMB} and {UNIT_EMB})")


def rows_by_name(old_state: Mapping[str, Any], new_state: Mapping[str, Any],
                 old_meta: Mapping[str, Any], new_meta: Mapping[str, Any]) -> list[str]:
    """What is wrong with the carried embedding rows, read through the two metas' name maps
    only (not through token_map, which made them): the pad and unknown rows are the old ones,
    each old name's row is its old row at its new token, and a new name's row is the unknown
    row; card_emb by card name, unit_emb by unit type name (by card for --live-compat). The
    probe cannot see a wrong map, since it reads the old tokens through the same map; this
    can. [] when every row is right."""
    import torch
    live = bool(new_meta.get("live_compat", False))
    wrong: list[str] = []
    for key, names in ((CARD_EMB, "cards"), (UNIT_EMB, "cards" if live else "units")):
        a, b = old_state[key], new_state[key]
        old_at = {n: i + 2 for i, n in enumerate(RC.names_of(old_meta, names)[0] or ())}
        new_at = {n: i + 2 for i, n in enumerate(RC.names_of(new_meta, names)[0] or ())}
        if int(b.shape[0]) != len(new_at) + 2:
            wrong.append(f"{key} has {int(b.shape[0])} rows, the new meta {len(new_at) + 2}")
            continue
        wrong += [f"{key} row {t}" for t in (PAD, UNKNOWN) if not torch.equal(a[t], b[t])]
        for name, t in new_at.items():
            want = a[old_at[name]] if name in old_at else a[UNKNOWN]
            if not torch.equal(b[t], want):
                wrong.append(f"{key} {name!r}")
        wrong += [f"{key} {name!r} is gone" for name in old_at if name not in new_at]
    return wrong


def _source_sha(source: Any) -> str | None:
    if isinstance(source, (str, os.PathLike)) and Path(source).is_file():
        h = hashlib.sha256()
        with open(source, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    return None


def _cold(meta: Mapping[str, Any], names: Sequence[str], extra: Iterable[str]) -> list[str]:
    """meta's cold_cards and `extra`, in `names` order, only names in `names`."""
    cold = {str(n) for n in (meta.get(COLD_KEY) or ())} | {str(n) for n in extra}
    return [n for n in names if n in cold]


def _load_quietly(blob: Mapping[str, Any], vocab: RC.EngineVocab) -> tuple[Any, list[str]]:
    """rs_checkpoint.load_policy, strict, drift refused; its notes returned, not warned."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RC.VocabDriftWarning)
        loaded = RC.load_policy(blob, vocab, allow_drift=False)
    notes = []
    for w in caught:
        if issubclass(w.category, RC.VocabDriftWarning):
            notes.append(str(w.message))
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return loaded.model, notes


def grow(blob: Mapping[str, Any], engine: Any, *, source: Any = "checkpoint",
         tolerance: float = TOLERANCE, rows: int = PROBE_ROWS, seed: int = 0) -> Carried:
    """The checkpoint `blob` ({"model", "meta", ...}) carried into `engine`'s vocabulary by
    name (module doc, GROW). VocabError when it cannot be, or would not be the same model."""
    meta, state = dict(blob["meta"]), blob["model"]
    vocab = RC.engine_vocab(engine)
    diff = vocab_diff(meta, vocab)
    shown = str(source)
    if diff.blockers:
        raise VocabError(f"{shown} cannot be carried into this engine (BREAKING): "
                         + "; ".join(diff.blockers))
    old_cards = RC.names_of(meta, "cards")[0] or []
    card_map = token_map(old_cards, vocab.cards)
    if diff.live:
        unit_map, old_units_rows = card_map, len(old_cards) + 2
    else:
        old_units = RC.names_of(meta, "units")[0] or []
        unit_map, old_units_rows = token_map(old_units, vocab.units), len(old_units) + 2
    tables = {CARD_EMB: (card_map, len(old_cards) + 2), UNIT_EMB: (unit_map, old_units_rows)}
    for key, (_, n) in tables.items():
        value = state.get(key)
        if value is None or value.dim() != 2 or int(value.shape[0]) != n:
            got = None if value is None else tuple(value.shape)
            raise VocabError(f"{shown}: {key} is {got}, its meta's vocabulary says {n} rows")
    old_model = _strict_model(meta, state, shown)
    new_state = {k: (take_rows(v, tables[k][0]) if k in tables else v) for k, v in state.items()}
    added = list(diff.cards.added)
    record = {"op": "grow", "tool": TOOL, "from": shown, "from_sha256": _source_sha(source),
              "engine": vocab.describe(), "engine_card_table": vocab.card_table,
              "engine_fnv1a64": vocab.fnv, "engine_royalesim": vocab.version,
              "cards_added": added, "cards_moved": len(diff.cards.moved),
              "units_added": [] if diff.live else list(diff.units.added),
              "units_moved": 0 if diff.live else len(diff.units.moved),
              "new_rows": "the unknown row (token 1)"}
    new_meta = dict(meta)
    new_meta["n_cards"] = len(vocab.cards) + 2
    new_meta["cards"] = {name: i + 2 for i, name in enumerate(vocab.cards)}
    new_meta["n_units"] = (len(vocab.cards) if diff.live else len(vocab.units)) + 2
    new_meta["units"] = {name: i + 2 for i, name in enumerate(vocab.units)}
    new_meta[CARRY_KEY] = [*list(meta.get(CARRY_KEY) or ()), record]
    new_meta[COLD_KEY] = _cold(meta, vocab.cards, added)
    _check_shapes(new_meta, new_state)
    out = {k: v for k, v in blob.items() if k not in ("model", "meta")}
    out.update(model=new_state, meta=new_meta)
    try:
        new_model, notes = _load_quietly(out, vocab)
    except RC.VocabMismatch as exc:                 # cannot happen: the names are the engine's
        raise VocabError(f"the carried checkpoint does not load strictly: {exc}") from None
    delta = preservation_delta(old_model, new_model, card_map, unit_map, rows=rows, seed=seed)
    if not delta < tolerance:
        raise VocabError(f"{shown}: the carried model's outputs on the old tokens moved by "
                         f"{delta:.3g} (at most {tolerance:g}); nothing written")
    wrong = rows_by_name(state, new_state, meta, new_meta)
    if wrong:
        raise VocabError(f"{shown}: the carried rows are not the old rows by name ("
                         + ", ".join(wrong[:5]) + (", ..." if len(wrong) > 5 else "")
                         + "); nothing written")
    record["max_delta"] = delta
    return Carried(out, record, delta, notes, diff)


def deck_cards() -> frozenset[str]:
    """Hog 2.6, the sparring deck, as the rows mark it (tools/rs_rows.py HOG26)."""
    import rs_rows
    return frozenset(rs_rows.HOG26)


def reset_cold(blob: Mapping[str, Any], cards: Iterable[str], *, units: Iterable[str] = (),
               source: Any = "checkpoint", tolerance: float = TOLERANCE,
               rows: int = PROBE_ROWS, seed: int = 0, protected: Iterable[str] | None = None,
               allow_protected: bool = False) -> Carried:
    """The checkpoint `blob` with the named cards' rows set to the unknown row (module doc,
    RESET-COLD). VocabError for a name the checkpoint does not know, and for a `protected`
    card (default Hog 2.6, deck_cards(): every row trains them) unless allow_protected. The
    record's "was" says what each card's row was before: "the unknown row", "marked cold" (the
    meta's cold_cards) or "its own row" (trained, or a random start no row reached)."""
    meta, state = dict(blob["meta"]), blob["model"]
    shown = str(source)
    live = bool(meta.get("live_compat", False))
    names, problem = RC.names_of(meta, "cards")
    if problem:
        raise VocabError(f"{shown}: {problem}")
    cards, units = [str(c) for c in cards], [str(u) for u in units]
    at = {n: i + 2 for i, n in enumerate(names)}
    unknown = [c for c in cards if c not in at]
    if unknown or not cards:
        raise VocabError(f"{shown}: not in its card vocabulary: {unknown}" if unknown
                         else "reset-cold: name at least one card")
    guard = deck_cards() if protected is None else frozenset(str(n) for n in protected)
    deck = [n for n in names if n in set(cards) & guard]
    if deck and not allow_protected:
        it = "it" if len(deck) == 1 else "them"
        where = ("Hog 2.6, which every row trains," if protected is None
                 else "the protected cards,")
        raise VocabError(f"{shown}: {', '.join(deck)} {'is' if len(deck) == 1 else 'are'} in "
                         f"{where} so the model would forget {it} (--allow-deck resets {it} "
                         "anyway)")
    card_tokens = sorted({at[c] for c in cards})
    unit_tokens: list[int] = []
    if live:
        if units:
            raise VocabError("--units: a --live-compat checkpoint's unit tokens are its cards, "
                             "reset with them")
        unit_tokens = list(card_tokens)
    elif units:
        unit_names, problem = RC.names_of(meta, "units")
        if problem:
            raise VocabError(f"{shown}: {problem}")
        at_u = {n: i + 2 for i, n in enumerate(unit_names)}
        missing = [u for u in units if u not in at_u]
        if missing:
            raise VocabError(f"{shown}: not in its unit vocabulary: {missing}")
        unit_tokens = sorted({at_u[u] for u in units})
    old_model = _strict_model(meta, state, shown)
    marked = {str(n) for n in (meta.get(COLD_KEY) or ())}
    already = unknown_rows(state, names)
    was = {n: ("the unknown row" if n in already else "marked cold" if n in marked
               else "its own row") for n in names if n in set(cards)}
    new_state = dict(state)
    for key, tokens in ((CARD_EMB, card_tokens), (UNIT_EMB, unit_tokens)):
        if tokens:
            # a new table, as grow builds one (take_rows: None takes the unknown row), not an
            # indexed write of the table's own row into itself: torch refuses that write
            # (index_put_'s value would alias the tensor it writes)
            reset = set(tokens)
            new_state[key] = take_rows(state[key], [None if t in reset else t
                                                    for t in range(int(state[key].shape[0]))])
    record = {"op": "reset-cold", "tool": TOOL, "from": shown, "from_sha256": _source_sha(source),
              "cards": [n for n in names if n in set(cards)], "was": was,
              "unit_tokens": unit_tokens, "rows": "the unknown row (token 1)"}
    new_meta = dict(meta)
    new_meta[CARRY_KEY] = [*list(meta.get(CARRY_KEY) or ()), record]
    new_meta[COLD_KEY] = _cold(meta, names, cards)
    new_model = _strict_model(new_meta, new_state, "the reset checkpoint")
    identity = list(range(state[CARD_EMB].shape[0]))
    unit_identity = list(range(state[UNIT_EMB].shape[0]))
    delta = preservation_delta(old_model, new_model, identity, unit_identity,
                               skip_cards=card_tokens, skip_units=unit_tokens, rows=rows,
                               seed=seed)
    if not delta < tolerance:
        raise VocabError(f"{shown}: rows without the reset cards moved by {delta:.3g}")
    record["max_delta"] = delta
    out = {k: v for k, v in blob.items() if k not in ("model", "meta")}
    out.update(model=new_state, meta=new_meta)
    return Carried(out, record, delta, [])


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _wait_keep(value: Any) -> float | None:
    try:
        keep = float(value)
    except (TypeError, ValueError):
        return None
    return keep if math.isfinite(keep) else None


def carried_args(source: Any, record: Mapping[str, Any]) -> dict:
    """args.json for a carried checkpoint: the source's (next to it; {} when it has none or it
    does not read) with "wait_keep" kept when it reads as a number, else set to
    rs_decide.DEFAULT_WAIT_KEEP, which is what rs_decide.wait_keep_for gave the source; and
    the carry record appended to "vocab_carry"."""
    import rs_decide
    doc: dict = {}
    if isinstance(source, (str, os.PathLike)):
        path = Path(source).with_name("args.json")
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                doc = dict(loaded)
    if _wait_keep(doc.get("wait_keep")) is None:
        doc["wait_keep"] = str(rs_decide.DEFAULT_WAIT_KEEP)
    doc[CARRY_KEY] = [*list(doc.get(CARRY_KEY) or ()), dict(record)]
    return doc


def write_carried(carried: Carried, out_dir: Any, source: Any = None, *,
                  force: bool = False) -> tuple[Path, Path]:
    """DIR/best.pt and DIR/args.json (carried_args). Refuses the source's own folder, and an
    existing best.pt or args.json unless `force`. Both are written to temporary files first;
    args.json replaces its file before best.pt does, so a best.pt there always has its
    args.json (without it rs_decide.wait_keep_for would fall back to the default wait bias)."""
    import torch
    out = Path(out_dir)
    best, args_json = out / "best.pt", out / "args.json"
    if isinstance(source, (str, os.PathLike)) and out.resolve() == Path(source).resolve().parent:
        raise VocabError(f"--out {out} is the source's own folder; name another")
    for path in (best, args_json):
        if path.exists() and not force:
            raise VocabError(f"{path} exists (--force replaces it)")
    doc = carried_args(source, carried.record)
    out.mkdir(parents=True, exist_ok=True)
    tmp_best = best.with_name(best.name + ".tmp")
    tmp_args = args_json.with_name(args_json.name + ".tmp")
    try:
        torch.save(carried.blob, tmp_best)
        tmp_args.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        os.replace(tmp_args, args_json)
        os.replace(tmp_best, best)
    finally:
        for tmp in (tmp_best, tmp_args):
            tmp.unlink(missing_ok=True)
    return best, args_json


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Check:
    """One checkpoint against one engine (module doc, CHECK)."""

    source: str
    verdict: str
    errors: list[str]
    notes: list[str]
    fix: str | None
    cold: list[str]
    cold_in_pool: list[str]
    diff: VocabDiff

    def lines(self) -> list[str]:
        out = [f"{self.verdict:12s} {self.source}"]
        out += [f"  refused: {e}" for e in self.errors]
        out += [f"  note:    {n}" for n in self.notes]
        if self.cold:
            out.append(f"  cold:    {', '.join(self.cold)}"
                       + (f" (in the pool: {', '.join(self.cold_in_pool)})"
                          if self.cold_in_pool else ""))
        if self.fix:
            out.append(f"  fix:     {self.fix}")
        return out

    def to_json(self) -> dict:
        return {"source": self.source, "verdict": self.verdict, "errors": self.errors,
                "notes": self.notes, "fix": self.fix, "cold": self.cold,
                "cold_in_pool": self.cold_in_pool, "diff": self.diff.to_json()}


def unknown_rows(state: Mapping[str, Any] | None, names: Sequence[str]) -> set[str]:
    """The cards whose card_emb row is the unknown row exactly (grow's new cards, reset-cold's)."""
    table = (state or {}).get(CARD_EMB)
    if table is None or not hasattr(table, "dim") or table.dim() != 2 \
            or int(table.shape[0]) != len(names) + 2:
        return set()
    same = (table[2:] == table[UNKNOWN]).all(dim=1)
    return {names[i] for i in same.nonzero().flatten().tolist()}


def check(blob: Mapping[str, Any], engine: Any, *, source: str = "checkpoint",
          cold: Iterable[str] = (), matters: Iterable[str] = ()) -> Check:
    """`blob` ({"meta"} and, for the cold rows, "model") against `engine`."""
    meta = blob["meta"]
    vocab = RC.engine_vocab(engine)
    errors, notes = RC.vocab_differences(meta, vocab)
    diff = vocab_diff(meta, vocab)
    names = RC.names_of(meta, "cards")[0] or []
    found = _cold(meta, names, [*cold, *unknown_rows(blob.get("model"), names)])
    wanted = {str(n) for n in matters}
    in_pool = [n for n in found if n in wanted]
    fix = None
    if errors:
        verdict = REFUSED
        if diff.verdict == GROW:
            fix = (f"python {TOOL} grow {source} --out DIR (every name it knows is still in the "
                   "engine: grow copies its rows by name, a new name's row starts as the "
                   "unknown row)")
            if diff.cards.moved:
                fix += ("; cards at other ids are unexpected (RoyaleSim only appends cards), "
                        "so find out why first")
        else:
            fix = ("rebuild the rows (tools/rs_rows.py) and retrain under this engine, or run "
                   "the engine it was made for (--card-table / --engine-config)")
    elif in_pool:
        verdict = COLD
        fix = (f"the model never learned {', '.join(in_pool)}: {TOOL} reset-cold makes an "
               "untrained row the unknown row; a fine-tune with them in the field teaches them "
               "(docs/SEASON_UPDATES.md 6.2, 6.3)")
    elif diff.drift:
        verdict = ENGINE_DRIFT
    else:
        verdict = OK
    return Check(source, verdict, errors, notes, fix, found, in_pool, diff)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def read_meta(path: Any) -> dict:
    """A vocabulary to compare: a checkpoint's meta (.pt), a vocab.json (or a checkpoint-like
    {"meta": ...} JSON), or a folder's vocab.json."""
    p = Path(path)
    if p.is_dir():
        p = p / "vocab.json"
    if p.suffix.lower() == ".json":
        doc = json.loads(p.read_text(encoding="utf-8-sig"))
        if isinstance(doc, dict) and isinstance(doc.get("meta"), dict):
            doc = doc["meta"]
        if not isinstance(doc, dict):
            raise ValueError(f"{p}: not a vocab.json object")
        return doc
    return dict(RC._load_blob(p)["meta"])


def _engine(args: argparse.Namespace) -> RC.EngineVocab:
    return RC.engine_vocab(rs_engine.config_from_args(args))


def _pool_names(extra: Iterable[str]) -> set[str]:
    return set(deck_cards()) | {str(n) for n in extra}


def cmd_diff(args: argparse.Namespace) -> int:
    old = read_meta(args.a)
    if args.b is not None:
        new_meta = read_meta(args.b)
        vocab, shown = RC.EngineVocab.from_vocab(new_meta), str(args.b)
    else:
        vocab = _engine(args)
        shown = vocab.describe()
    diff = vocab_diff(old, vocab)
    if args.json:
        print(json.dumps({"old": str(args.a), "new": shown, **diff.to_json()}, indent=1))
    else:
        print(f"old         {args.a}\nnew         {shown}")
        print("\n".join(diff.lines()))
    return {SAME: EXIT_OK, LOADS: EXIT_DRIFT, GROW: EXIT_GROW}.get(diff.verdict, EXIT_FAILED)


def _write(carried: Carried, args: argparse.Namespace) -> int:
    best, args_json = write_carried(carried, args.out, args.ckpt, force=args.force)
    print(f"wrote {best} and {args_json} (largest output change on the old tokens "
          f"{carried.max_delta:.2g})")
    for note in carried.notes:
        print(f"  note: {note}")
    return EXIT_OK


def cmd_grow(args: argparse.Namespace) -> int:
    vocab = _engine(args)
    blob = RC._load_blob(args.ckpt)
    carried = grow(blob, vocab, source=args.ckpt, tolerance=args.tolerance)
    d = carried.diff
    print(f"{args.ckpt} -> {vocab.describe()}\n  cards: {d.cards.describe('card')}"
          + ("" if d.live else f"\n  unit types: {d.units.describe('unit type')}"))
    return _write(carried, args)


def cmd_reset_cold(args: argparse.Namespace) -> int:
    blob = RC._load_blob(args.ckpt)
    carried = reset_cold(blob, args.cards, units=args.units, source=args.ckpt,
                         tolerance=args.tolerance, allow_protected=args.allow_deck)
    print(f"{args.ckpt}: {', '.join(carried.record['cards'])} reset to the unknown row")
    own = [n for n, was in carried.record["was"].items() if was == "its own row"]
    if own:
        print(f"  note: {', '.join(own)} had {'its' if len(own) == 1 else 'their'} own row and "
              "no cold mark: what the model learned of them is gone in the reset copy")
    return _write(carried, args)


def cmd_check(args: argparse.Namespace) -> int:
    vocab = _engine(args)
    matters = _pool_names(args.pool)
    results, worst = [], OK
    for path in args.ckpt:
        try:
            result = check(RC._load_blob(path), vocab, source=str(path), cold=args.cold,
                           matters=matters)
        except Exception as exc:            # a missing, truncated or foreign file
            print(f"ERROR        {path}: {type(exc).__name__}: {exc}")
            worst = REFUSED
            continue
        results.append(result)
        if _SEVERITY.index(result.verdict) > _SEVERITY.index(worst):
            worst = result.verdict
        if not args.json:
            print("\n".join(result.lines()))
    if args.json:
        print(json.dumps({"engine": vocab.describe(), "verdict": worst,
                          "checkpoints": [r.to_json() for r in results]}, indent=1))
    return _CHECK_EXIT[worst]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Diff, grow, reset-cold or check token-policy checkpoints against an "
                    "engine's card and unit vocabulary, by name (docs/SEASON_UPDATES.md 6.1).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diff", help="a checkpoint or vocab.json against the engine or another")
    d.add_argument("a", help="checkpoint .pt, vocab.json or a rows folder")
    d.add_argument("b", nargs="?", default=None, help="another one; default: the engine")
    d.add_argument("--json", action="store_true")
    rs_engine.add_engine_args(d)
    g = sub.add_parser("grow", help="carry a checkpoint into the engine's vocabulary by name")
    g.add_argument("ckpt")
    g.add_argument("--out", required=True, help="folder for best.pt and args.json")
    g.add_argument("--force", action="store_true", help="replace an existing best.pt there")
    g.add_argument("--tolerance", type=float, default=TOLERANCE)
    rs_engine.add_engine_args(g)
    r = sub.add_parser("reset-cold", help="set cards' rows to the unknown row")
    r.add_argument("ckpt")
    r.add_argument("--cards", nargs="+", required=True, help="catalogue names")
    r.add_argument("--units", nargs="*", default=[],
                   help="unit names to reset too (not --live-compat: its cards' are)")
    r.add_argument("--out", required=True)
    r.add_argument("--force", action="store_true")
    r.add_argument("--allow-deck", action="store_true",
                   help="reset Hog 2.6 cards too (refused by default: every row trains them)")
    r.add_argument("--tolerance", type=float, default=TOLERANCE)
    c = sub.add_parser("check", help="OK, REFUSED, ENGINE_DRIFT or COLD per checkpoint")
    c.add_argument("ckpt", nargs="+")
    c.add_argument("--cold", nargs="*", default=[], help="more cards the model never learned")
    c.add_argument("--pool", nargs="*", default=[],
                   help="cards that matter besides Hog 2.6 (the opponent pool's)")
    c.add_argument("--json", action="store_true")
    rs_engine.add_engine_args(c)
    return ap


COMMANDS = {"diff": cmd_diff, "grow": cmd_grow, "reset-cold": cmd_reset_cold, "check": cmd_check}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.cmd](args)
    except (ValueError, FileNotFoundError) as exc:
        if isinstance(exc, VocabError):
            print(f"refused: {exc}", file=sys.stderr)
            return EXIT_FAILED
        print(f"{args.cmd}: {exc}", file=sys.stderr)
        return EXIT_ARGS
    except rs_engine.EngineError as exc:
        print(f"engine: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
