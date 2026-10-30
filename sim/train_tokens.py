"""Train `sim.model_tokens.TokenPolicy` on the shards from tools/rs_rows.py.

    .venvs/buildabot/Scripts/python.exe -m sim.train_tokens --name rs_bc --wait-keep 0.25
    .venvs/buildabot/Scripts/python.exe -m sim.train_tokens --name awr --beta 0.1
    ... --finetune-hog --init runs/tokens/awr/best.pt --name awr_hog
    smoke run: ... --name smoke --device cpu --d 16 --layers 1 --heads 2 --epochs 2 --max-rows 5000

Only numpy and torch are needed, so it runs in any environment with CUDA
PyTorch; it does not import the engine. `--device cuda` (the default) stops
with an error when CUDA is unavailable, rather than training on the CPU unnoticed.

Rows: by default only *trusted* rows (before the match's first divergence)
are used, and wait rows are thinned to `--wait-keep` of all wait rows so the
dataset fits in memory. Validation is the hash split written by the builder,
by match. `--max-rows N` keeps a seeded uniform sample of at most N rows per
split (train, val, val Hog), for smoke runs or a PC with little RAM.

`--trust legacy` (the default) reads the `trusted` column, as before. `--trust v2`
reads `trusted_v2` instead (tools/rs_rows.py --trust v2, E09: also cut where the
engine's towers or game end leave the recording, and no trust for the waits after
the last recorded play); it stops with an error when a shard has no such column.
Press rows (tools/rs_rows.py --presses on, E10: act_press > 0) have act_card 0 but
are not waits, so they are left out of every split, train and validation; rows
without an act_press column are loaded exactly as before. args.json records
"trust", "trust_column", "press_rows" ("excluded" or "none") and
"press_rows_excluded"; the checkpoint meta has them as "train_rows" unless they
are the legacy ones (trust legacy, no press column), so a legacy meta is still
vocab.json plus the architecture.

Schema v2 (E15; tools/rs_rows.py --schema v2, tools/rs_schema.py): the rows' vocab.json
"schema" picks the model. rs_rows/v2 trains sim/model_tokens.TokenPolicy(schema="v2") on the
v2 columns (cols_for: rs_schema.V2.train_columns, after a check that vocab.json's
schema_spec is this code's v2, rs_schema.spec_problems); anything else (rs_rows/legacy, no key,
another name) is legacy and trains exactly as before, on COLS (rs_schema.LEGACY.train_columns).
Under v2 the press rows (act_press > 0) are not left out: they are never thinned like waits,
and when the rows are press-labelled (vocab.json "presses" mode on, or any press row) they
train the ability buttons (model_tokens.losses presses=True); otherwise the button logits stay
out of the loss. The row record then says "press_rows": "trained" (or "none") with
"press_rows_trained", and the validation lines add press_acc. Each checkpoint's critic (value2)
is copied from value_head before it is saved (TokenPolicy.sync_critic). --init takes a
checkpoint of the same schema (strict load); --init-partial takes the tensors whose names and
shapes fit (model_tokens.load_matching: a legacy model's state_in / unit_in fill the first
columns of the v2 ones), from a checkpoint with the same card and unit vocabulary.

`--wait-keep` is the share of ALL wait rows that training sees. When vocab.json
says the builder already thinned waits (`wait_keep_at_build`), training keeps
each remaining wait with probability wait_keep / wait_keep_at_build, and the
default is wait_keep_at_build itself (no further thinning). Either way
args.json records the overall share as "wait_keep", which is what RSAgent
(tools/rs_eval.py) adds back as log(1 / wait_keep) on the wait logit.

Full corpus (E18; tokens-13). By default the kept rows are loaded into RAM (load), as before.
  --memmap        --data is a store of tools/rs_rows_compact.py (per-column .npy files, memory
                  mapped read-only): memmap_keep picks exactly the rows load() would (the same
                  filters, wait draws and --max-rows sample), the two validation sets are read
                  into RAM (gather), and each epoch's training rows are read from disk a window
                  (--memmap-window rows, a multiple of --batch) at a time in epoch_order: runs of
                  CHUNK_ROWS consecutive kept rows in a random order, shuffled within each window
                  (every row once an epoch); the batches reach CUDA through two pinned buffers
                  (PinnedStager). With --phase-balance the order is balanced_order's, read the same
                  way. What this holds in RAM (memmap_ram_need: the ids and the order, the
                  validation rows, a window twice and two batches) must fit --ram-budget-gb
                  (default 6, for a 16 GB PC with MuMu and the browser closed), else it stops
                  before training, naming --val-max-rows and --memmap-window.
  --gpu-resident  the training rows are copied to the GPU once (from RAM, or from the store a
                  window at a time) and each batch is an index_select there, in the order RAM
                  training uses (np.random.permutation, or balanced_order); refused (SystemExit,
                  before anything is written) when they need more than --gpu-budget-gb (default
                  7, of the RTX 4070 Super's 12) with their int64 order (resident_gpu_need).
  --val-max-rows  at most this many rows of each validation set (0: --max-rows's cap, as before).
  --mirror P      each training row is flipped left-right with probability P (Mirror, seeded by
                  --seed, its own stream): by tools/rs_schema.py's mirror rules, unit x -> 1 - x
                  (own-frame x / 18000, the tile x -> 17 - x of the board) where a unit is,
                  towers left <-> right on both sides, spell centre and aim x -> 1 - x, and a
                  play's tile x -> 17 - x; nothing else has a side (own_pending holds cards and
                  land times only, the hand, buttons and scalars none), and the units keep their
                  order (sorted by distance from the own king, on the centre line). Unit and
                  spell features are then float32 (the model reads them as float32 anyway), so a
                  flip twice is the identity. 0 (the default) draws nothing: the run is the one
                  before E18.
args.json records these flags; the checkpoint meta does not change (the store's vocab.json is a
copy of the rows').

--drop-pending-waits (the decision-time diagnosis of 2026-10-30, docs/SPARRING_RUNBOOK.md section
19). Decision-time rows (tools/rs_rows.py --obs-delay 24, E11) keep a wait row on the 20-tick grid
wherever the side taps nothing in the next 10 ticks, so the waits between a tap and the tick its
play runs are kept and labelled wait. The engine re-drive has no command delay: in those rows the
tapped card is still in hand, the bar unspent and its unit not on the board, the tap row's state a
little later, and only own_pending (which the legacy model does not read) tells them apart. The
flag leaves those waits (act_card 0 and an own_pending slot > 0, pending_waits) out of the
TRAINING split, after the same wait draws, so every other row is the one the run without it
trains on; the validation sets keep them (their card and tile figures, and best.pt, read plays
only). Legacy schema only: SystemExit for v2 rows (the v2 model reads own_pending and learns from
these waits not to tap twice) and for rows without the column (rows at execution). args.json
gets "pending_waits_dropped", the count before --max-rows. Off (the default): the rows of before.

Output in <--runs>/<--name>/:
* ep01.pt, ep02.pt, ...: the model after each epoch (ep01.pt is the one printed
  as "epoch 0"; the "epoch" stored inside a checkpoint stays 0-based);
* last.pt (the latest epoch) and best.pt (best val Hog card_acc + tile_near);
* args.json: the arguments as strings, the resolved wait_keep, the row record
  (trust, trust_column, press_rows, press_rows_excluded), and torch / CUDA /
  numpy / Python versions and the git commit;
* log.jsonl: one line of validation metrics per epoch.
Each checkpoint is {"model", "meta", "epoch", "train_tokens"}: meta is vocab.json
(ids, engine fingerprint, schema, observation delay, wait_keep_at_build, whatever
the builder wrote) plus d / layers / heads (and "train_rows" when not legacy), and
"train_tokens" repeats args.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .model_tokens import (RS, TokenPolicy, gate_targets, load_matching, losses, model_schema,
                           target_slot)
import rs_rows_compact as RC  # noqa: E402  (tools/ is on sys.path once model_tokens is imported)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rs_rows"
RUNS = ROOT / "runs" / "tokens"
#: The legacy columns (rs_schema.LEGACY.train_columns; a literal, which tests read with ast).
COLS = ("unit_ids", "unit_feats", "towers", "scalars", "hand", "next", "opp_seen",
        "opp_last", "opp_last_age", "outcome", "won", "act_card", "act_x", "act_y")
#: Columns stored as float16 whatever the shard holds (the largest one, ~84% of a row).
HALF_COLS = ("unit_feats",)
DEFAULT_WAIT_KEEP = 0.35
#: vocab.json keys that describe how the rows were made (E01 / E04 / E08 / E09 / E10 / E11 /
#: E12 / E18). The whole of vocab.json goes into the checkpoint meta; these are the ones
#: printed at start and the ones a checkpoint must not lose. A key the rows' vocab.json
#: lacks (rows built without that option) is not printed.
PROVENANCE_KEYS = ("engine", "version", "fingerprint", "engine_fingerprint", "card_table",
                   "schema", "obs_delay", "obs_delay_jitter", "wait_keep_at_build",
                   "live_compat", "opp_memory", "visibility", "decision_rows", "trust",
                   "presses", "wait_thinning", "selection")
#: --trust (E09): the column the trusted-rows filter reads. legacy: `trusted`, as before;
#: v2: `trusted_v2`, which tools/rs_rows.py --trust v2 writes next to `trusted`.
TRUST_LEGACY, TRUST_V2 = "legacy", "v2"
TRUST_COLUMNS = {TRUST_LEGACY: "trusted", TRUST_V2: "trusted_v2"}
TRUST_MODES = tuple(TRUST_COLUMNS)
#: E10 press rows (tools/rs_rows.py --presses on): act_press is the pressed button + 1, 0 on
#: every other row. A press row's act_card is 0, but it is no wait, so the loader leaves it
#: out (E15's press head will train on them).
PRESS_COLUMN = "act_press"
PRESS_EXCLUDED, PRESS_NONE = "excluded", "none"
#: E15 (schema v2): press rows train the ability buttons.
PRESS_TRAINED = "trained"
#: The checkpoint meta key of the row record (row_record) when it is not the legacy one.
TRAIN_ROWS_KEY = "train_rows"
#: The row record key a v2 record adds: the press rows the buttons train on.
PRESS_TRAINED_KEY = "press_rows_trained"
#: --drop-pending-waits (tokens-dt diagnosis, docs/SPARRING_RUNBOOK.md section 19): the column of
#: decision-time rows (tools/rs_rows.py --obs-delay > 0, E11) holding the side's own plays tapped
#: and not yet run, card tokens, 0 for an empty slot; and the args.json key of the waits dropped.
PENDING_COLUMN = "own_pending"
PENDING_DROPPED_KEY = "pending_waits_dropped"


def cols_for(schema=None) -> tuple:
    """The columns load() reads for rows of `schema` (None: legacy): COLS, or (v2) the
    schema's train_columns, model inputs then labels."""
    spec = _spec(schema)
    return spec.train_columns if spec.v2 else COLS


def _spec(schema):
    """`schema` as an rs_schema.Schema (a name, short name, Schema, or None for legacy)."""
    if schema is not None and not isinstance(schema, str) and hasattr(schema, "name"):
        schema = schema.name
    return RS.get(schema)


def _vocab_of(root) -> dict:
    """vocab.json of the rows in `root`, {} when there is none."""
    path = Path(root) / "vocab.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows_schema(root) -> "RS.Schema":
    """The schema of the rows in `root`: model_schema of their vocab.json (legacy without one)."""
    return model_schema(_vocab_of(root))


def presses_labelled(vocab: dict) -> bool:
    """Whether rows were built with presses replayed (vocab.json "presses" mode "on")."""
    entry = vocab.get("presses")
    return isinstance(entry, dict) and entry.get("mode") == "on"


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def trust_column(trust: str = TRUST_LEGACY) -> str:
    """The shard column --trust `trust` filters on; ValueError for another mode."""
    try:
        return TRUST_COLUMNS[trust]
    except (KeyError, TypeError):
        raise ValueError(f"--trust: one of {', '.join(TRUST_MODES)}, got {trust!r}") from None


def _files(z) -> set:
    """The column names of an open shard (np.load's NpzFile; a dict works too)."""
    files = getattr(z, "files", None)
    return set(z if files is None else files)


def missing_trust(trust: str, missing: list, root, shards: int | None = None) -> str:
    """The error for --trust `trust` on shards without its column: `missing` shard names,
    out of `shards` (None: the loader, which stops at the first)."""
    col = trust_column(trust)
    where = (f"{len(missing)} of {shards} shard(s) in {root} have no {col} column "
             f"(first: {missing[0]})" if shards is not None
             else f"{missing[0]} in {root} has no {col} column")
    return (f"--trust {trust}: {where}. Build the rows with tools/rs_rows.py --trust {trust}, "
            f"or train with --trust {TRUST_LEGACY} (the `trusted` column).")


def pending_waits(z) -> np.ndarray:
    """Bool per row of shard `z`: a wait row (act_card 0) while the side has a play of its own
    tapped and not yet run (any own_pending slot > 0). Decision-time rows only (PENDING_COLUMN)."""
    pend = np.asarray(z[PENDING_COLUMN])
    return (np.asarray(z["act_card"]) == 0) & (pend.reshape(len(pend), -1) > 0).any(axis=1)


def _keep_mask(z, split: str, wait_keep: float, trusted_only: bool, hog_only: bool, rng,
               trust_col: str = "trusted", presses: str = PRESS_EXCLUDED,
               drop_pending: bool = False, counts: dict | None = None):
    """The rows of one shard the loader keeps. One rng draw per row of the shard,
    whatever the other filters say, so the same seed keeps the same waits.
    `trust_col` is the trusted-rows column (trust_column). Press rows (act_press > 0,
    when the shard has that column) are left out after the draw: they are not waits.
    presses PRESS_TRAINED (schema v2, E15): press rows are kept, and not thinned as waits.

    `drop_pending` (--drop-pending-waits; legacy schema only, ValueError under PRESS_TRAINED):
    after the same draws, the wait rows with an own play in flight (pending_waits) are left out
    too; `counts` (a dict) then gets PENDING_DROPPED_KEY += the rows this leaves out (rows the
    other filters and the draw had kept). False: the mask of before."""
    keep = z["split"] == split
    if trusted_only:
        keep &= z[trust_col]
    if hog_only:
        keep &= z["hog"]
    waits = z["act_card"] == 0
    if presses == PRESS_TRAINED:
        if drop_pending:
            raise ValueError("--drop-pending-waits is for legacy-schema rows: a v2 model reads "
                             "own_pending and learns from these waits not to tap twice")
        waits &= ~(z[PRESS_COLUMN] > 0)
        keep &= ~waits | (rng.random(len(waits)) < wait_keep)
        return keep
    keep &= ~waits | (rng.random(len(waits)) < wait_keep)
    if PRESS_COLUMN in _files(z):
        keep &= ~(z[PRESS_COLUMN] > 0)
    if drop_pending:
        pend = pending_waits(z)
        if counts is not None:
            counts[PENDING_DROPPED_KEY] = (counts.get(PENDING_DROPPED_KEY, 0)
                                           + int((keep & pend).sum()))
        keep &= ~pend
    return keep


def missing_pending(missing: list, root, shards: int | None = None) -> str:
    """The error for --drop-pending-waits on rows without an own_pending column: `missing` shard
    names, out of `shards` (None: the loader, which stops at the first)."""
    where = (f"{len(missing)} of {shards} shard(s) in {root} have no {PENDING_COLUMN} column "
             f"(first: {missing[0]})" if shards is not None
             else f"{missing[0]} in {root} has no {PENDING_COLUMN} column")
    return (f"--drop-pending-waits: {where}. Only decision-time rows (tools/rs_rows.py "
            f"--obs-delay 24) have own plays in flight; train these rows without the flag.")


def check_drop_pending(root=None, store=None, schema=None) -> None:
    """SystemExit unless --drop-pending-waits can run: legacy-schema rows (`schema`, a v2 model
    reads own_pending itself) whose every shard in `root` (or the compact `store`) has the
    own_pending column."""
    spec = _spec(schema)
    if spec.v2:
        raise SystemExit(f"--drop-pending-waits: these rows are {spec.name}. The v2 model reads "
                         f"{PENDING_COLUMN}, and the waits with a play in flight teach it not to "
                         "tap twice: train v2 rows without the flag.")
    if store is not None:
        if PENDING_COLUMN not in store:
            names = store.shard_names
            raise SystemExit(missing_pending(names, store.root, len(names)))
        return
    paths = sorted(Path(root).glob("shard-*.npz"))
    missing = []
    for path in paths:
        with np.load(path) as z:
            if PENDING_COLUMN not in _files(z):
                missing.append(path.name)
    if missing:
        raise SystemExit(missing_pending(missing, root, len(paths)))


def row_record(root: Path, trust: str = TRUST_LEGACY, schema=None) -> dict:
    """What --trust and the press rows come to on the shards in `root`, read before any
    row is loaded: {"trust", "trust_column", "press_rows": "excluded" when a shard has an
    act_press column, else "none", "press_rows_excluded": rows with act_press > 0 over
    every split}. args.json gets it as it is, the checkpoint meta as "train_rows" when it
    is not the legacy record (legacy_rows). SystemExit when --trust is not legacy and a
    shard lacks its column (rows built without --trust v2). No shards: the legacy counts
    (load then says there are none).

    Schema v2 (`schema`; None: the rows' own, rows_schema): nothing is excluded, so
    "press_rows" is "trained" when the rows are press-labelled (presses_labelled, or any press
    row) and "none" otherwise, "press_rows_excluded" 0, and "press_rows_trained" counts the
    press rows."""
    spec = rows_schema(root) if schema is None else _spec(schema)
    col = trust_column(trust)
    paths = sorted(Path(root).glob("shard-*.npz"))
    missing, press_shards, presses = [], 0, 0
    for path in paths:
        with np.load(path) as z:
            files = _files(z)
            if trust != TRUST_LEGACY and col not in files:
                missing.append(path.name)
            if PRESS_COLUMN in files:
                press_shards += 1
                presses += int((z[PRESS_COLUMN] > 0).sum())
    if missing:
        raise SystemExit(missing_trust(trust, missing, root, len(paths)))
    return _record(spec, trust, col, bool(press_shards), presses, root)


def _record(spec, trust: str, col: str, press_column: bool, presses: int, root) -> dict:
    """row_record's dict for rows of `spec` with (or without) an act_press column holding
    `presses` press rows."""
    if spec.v2:
        labelled = bool(presses) or presses_labelled(_vocab_of(root))
        return {"trust": trust, "trust_column": col,
                "press_rows": PRESS_TRAINED if labelled else PRESS_NONE,
                "press_rows_excluded": 0, PRESS_TRAINED_KEY: presses}
    return {"trust": trust, "trust_column": col,
            "press_rows": PRESS_EXCLUDED if press_column else PRESS_NONE,
            "press_rows_excluded": presses}


def row_record_store(store, trust: str = TRUST_LEGACY, schema=None) -> dict:
    """row_record of a compact store (--memmap): the record row_record gives for the shards it
    was made from (SystemExit the same way when --trust names a column the rows lack)."""
    spec = rows_schema(store.root) if schema is None else _spec(schema)
    col = trust_column(trust)
    if trust != TRUST_LEGACY and col not in store:
        names = store.shard_names
        raise SystemExit(missing_trust(trust, names, store.root, len(names)))
    has = PRESS_COLUMN in store
    return _record(spec, trust, col, has, store.count_positive(PRESS_COLUMN) if has else 0,
                   store.root)


def _row_entry(rows: dict) -> dict:
    """The row record as args.json and the meta store it: plain str / int values."""
    out = {"trust": str(rows["trust"]), "trust_column": str(rows["trust_column"]),
           "press_rows": str(rows["press_rows"]),
           "press_rows_excluded": int(rows["press_rows_excluded"])}
    if PRESS_TRAINED_KEY in rows:
        out[PRESS_TRAINED_KEY] = int(rows[PRESS_TRAINED_KEY])
    return out


def legacy_rows(rows: dict | None) -> bool:
    """Whether a row_record changes nothing: trust legacy and no press column (or no record)."""
    return rows is None or (rows.get("trust", TRUST_LEGACY) == TRUST_LEGACY
                            and rows.get("press_rows", PRESS_NONE) == PRESS_NONE)


def describe_rows(rows: dict) -> str:
    """One line for the console."""
    text = f"trust {rows['trust']} (column {rows['trust_column']})"
    if rows["press_rows"] == PRESS_EXCLUDED:
        text += (f"; press rows left out: {rows['press_rows_excluded']:,} rows with "
                 f"{PRESS_COLUMN} > 0 (not waits)")
    elif rows["press_rows"] == PRESS_TRAINED:
        text += (f"; press rows trained: {rows.get(PRESS_TRAINED_KEY, 0):,} rows with "
                 f"{PRESS_COLUMN} > 0 (the ability buttons)")
    return text


def _cap(picks: list, max_rows: int, seed: int) -> list:
    """At most `max_rows` of the picked rows, a uniform sample in row order. Its own
    rng, so the wait draws above do not depend on the cap."""
    sizes = [len(p) for p in picks]
    total = sum(sizes)
    if not max_rows or total <= max_rows:
        return picks
    chosen = np.sort(np.random.default_rng([seed, 7919]).choice(total, max_rows, replace=False))
    bounds = np.cumsum([0] + sizes)
    return [p[chosen[(chosen >= a) & (chosen < z)] - a]
            for p, a, z in zip(picks, bounds[:-1], bounds[1:])]


def load(split: str, wait_keep: float, trusted_only: bool, hog_only: bool, seed: int = 0,
         root: Path = DATA, *, max_rows: int = 0, trust: str = TRUST_LEGACY, schema=None,
         drop_pending: bool = False, counts: dict | None = None):
    """The rows of one split, every column in COLS, shards in name order.

    The same rows, order and dtypes as concatenating every shard's kept rows and
    then casting unit_feats to float16 (the loader before E05), with a lower
    peak: a first pass reads only the filter columns, then each column is
    written into an array of the final size, one shard at a time, so neither
    the per-shard parts nor a float16 copy are held next to the result.
    `max_rows` > 0 keeps a seeded uniform sample of at most that many rows.
    `trust` (--trust) names the column `trusted_only` reads (trust_column);
    ValueError when a shard lacks it. Press rows are left out (_keep_mask).

    `schema` (None: the rows' own, rows_schema) v2: the columns are cols_for("v2"), press rows
    are kept (and not thinned), and a shard without one of those columns is a ValueError.

    `drop_pending` (--drop-pending-waits) leaves out the waits with an own play in flight as
    _keep_mask does, after the same draws (`counts` gets how many); ValueError for v2 rows or a
    shard without own_pending. False (the default): the rows of before.
    """
    spec = rows_schema(root) if schema is None else _spec(schema)
    cols = cols_for(spec)
    presses = PRESS_TRAINED if spec.v2 else PRESS_EXCLUDED
    if drop_pending and spec.v2:
        raise ValueError(f"--drop-pending-waits: {root} holds {spec.name} rows; legacy only")
    col = trust_column(trust)
    rng = np.random.default_rng(seed)
    paths = sorted(Path(root).glob("shard-*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shard-*.npz in {root}")
    picks = []
    for path in paths:
        with np.load(path) as z:
            if trusted_only and col not in _files(z):
                raise ValueError(missing_trust(trust, [path.name], root))
            if spec.v2:
                lacking = [c for c in cols if c not in _files(z)]
                if lacking:
                    raise ValueError(f"{path.name} in {root} has no {lacking}: not rows of "
                                     f"{spec.name} (tools/rs_rows.py --schema v2)")
            if drop_pending and PENDING_COLUMN not in _files(z):
                raise ValueError(missing_pending([path.name], root))
            picks.append(np.flatnonzero(_keep_mask(z, split, wait_keep, trusted_only,
                                                   hog_only, rng, col, presses,
                                                   drop_pending, counts)))
    found = sum(len(p) for p in picks)
    picks = _cap(picks, max_rows, seed)
    total = sum(len(p) for p in picks)
    if total < found:
        print(f"  {split}{' (hog)' if hog_only else ''}: {total:,} of {found:,} rows "
              f"(--max-rows {max_rows:,})", flush=True)
    data: dict[str, np.ndarray] = {}
    pos = 0
    for path, idx in zip(paths, picks):
        with np.load(path) as z:
            for c in cols:
                part = z[c][idx]
                if c in HALF_COLS:
                    part = part.astype(np.float16, copy=False)
                out = data.get(c)
                if out is None:
                    out = data[c] = np.empty((total,) + part.shape[1:], dtype=part.dtype)
                elif out.shape[1:] != part.shape[1:]:
                    raise ValueError(f"{path.name}: column {c} has shape {part.shape[1:]}, "
                                     f"earlier shards {out.shape[1:]}")
                elif c not in HALF_COLS:
                    wider = np.result_type(out.dtype, part.dtype)
                    if wider != out.dtype:          # as np.concatenate would promote
                        out = data[c] = out.astype(wider)
                out[pos:pos + len(idx)] = part
        pos += len(idx)
    return data


def resolve_wait_keep(cli: float | None, vocab: dict) -> tuple[float, float, float | None]:
    """(overall share of wait rows trained on, per-row keep probability at load,
    wait_keep_at_build or None).

    Rows built with `wait_keep_at_build` p already hold only p of the waits, so
    an overall share q is p's rows kept with probability q / p; q defaults to p.
    Without it (rows from before E18) the probability is q itself, default 0.35.
    """
    at_build = vocab.get("wait_keep_at_build")
    if at_build is not None:
        at_build = float(at_build)
        if not 0.0 < at_build <= 1.0:
            raise SystemExit(f"vocab.json wait_keep_at_build {at_build} is not in (0, 1]")
    if cli is None:
        cli = at_build if at_build is not None else DEFAULT_WAIT_KEEP
    cli = float(cli)
    if not 0.0 < cli <= 1.0:
        raise SystemExit(f"--wait-keep {cli} is not in (0, 1]")
    if at_build is None:
        return cli, cli, None
    if cli > at_build:
        print(f"warning: --wait-keep {cli} is more than the rows hold "
              f"(wait_keep_at_build {at_build}); training on all of them, "
              f"wait_keep = {at_build}", flush=True)
        return at_build, 1.0, at_build
    return cli, min(1.0, cli / at_build), at_build


PHASE_NAMES = [f"{t}/{w}" for t in ("normal", "double", "overtime")
               for w in ("all up", "theirs down", "ours down", "both down")]


def phase_of(data):
    """Phase bucket per row: clock (normal / double elixir / overtime) x towers
    (all princess towers up / one of theirs down / one of ours down / both).

    `scalars` = [elixir, clock, double elixir flag, overtime flag];
    `towers` = [own L, own R, own king, enemy L, enemy R, enemy king] as hp
    fractions, in the acting player's frame.
    """
    sc, tw = data["scalars"], data["towers"]
    clock = np.where(sc[:, 3] > 0.5, 2, np.where(sc[:, 2] > 0.5, 1, 0))
    theirs = (tw[:, 3] <= 0) | (tw[:, 4] <= 0)
    ours = (tw[:, 0] <= 0) | (tw[:, 1] <= 0)
    return clock * 4 + theirs.astype(int) + 2 * ours.astype(int)


def balanced_order(data, alpha: float, rng):
    """One epoch of row indices, drawn so rare phases weigh more.

    Each row's weight is (1 / rows in its phase) ** alpha: alpha 0 is the
    data as it is, 1 is every phase equally often. Repeats of a tiny phase
    are capped at 20x its size per epoch, so a handful of rows is not drilled
    thousands of times.
    """
    return balanced_from_phases(phase_of(data), alpha, rng)


def balanced_from_phases(ph, alpha: float, rng):
    """balanced_order of rows whose phase_of is `ph` (TrainFeed computes it once, --memmap from
    the store)."""
    counts = np.bincount(ph, minlength=12).astype(float)
    w = (1.0 / np.maximum(counts[ph], 1)) ** alpha
    n = len(ph)
    share = w / w.sum()
    # cap: expected draws of a phase <= 20 x its rows
    per_phase = np.bincount(ph, weights=share, minlength=12) * n
    scale = np.where(per_phase > 20 * counts, 20 * counts / np.maximum(per_phase, 1e-9), 1.0)
    share = share * scale[ph]
    share /= share.sum()
    return rng.choice(n, n, replace=True, p=share)


def batches(data, size: int, shuffle: bool, device, order=None):
    n = len(data["act_card"])
    if order is None:
        order = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, size):
        idx = order[i:i + size]
        b = {}
        for c, v in data.items():
            t = torch.from_numpy(np.ascontiguousarray(v[idx]))
            if t.dtype in (torch.int16, torch.int32):
                t = t.long()
            if t.dtype == torch.float64:
                t = t.float()
            b[c] = t.to(device, non_blocking=True)
        yield b


# ---------------------------------------------------------------------------
# E18: rows on disk (--memmap), on the GPU (--gpu-resident), and the mirror
# ---------------------------------------------------------------------------

#: --memmap-window default (rounded down to a multiple of --batch), the runs of consecutive kept
#: rows an epoch is cut into (epoch_order), and the rows a split is picked from at a time.
WINDOW_ROWS = 1 << 18
CHUNK_ROWS = 1024
KEEP_CHUNK = 1 << 22
#: --ram-budget-gb / --gpu-budget-gb defaults for the PC (16 GB RAM, RTX 4070 Super 12 GB): what
#: the trainer's own arrays may take next to Windows, the CUDA context and the file cache; what
#: the rows may take on the GPU next to the model, its optimiser and its activations.
RAM_BUDGET_GB, GPU_BUDGET_GB = 6.0, 7.0
GB = 2 ** 30
#: Mirror's first seed word (default_rng([MIRROR_SALT, --seed])), so its draws are its own.
MIRROR_SALT = 0x4D52
_SMALL_INTS = (torch.int8, torch.uint8, torch.int16, torch.int32)


def open_store(root):
    """The compact store (tools/rs_rows_compact.py) in `root`, for --memmap; SystemExit naming
    the command that makes one when `root` holds none (or one this code cannot read)."""
    try:
        return RC.CompactRows(root)
    except FileNotFoundError:
        rows = root if any(Path(root).glob("shard-*.npz")) else "<rows folder>"
        raise SystemExit(f"--memmap: no {RC.COMPACT_JSON} in {root}. Make the store with "
                         f"tools/rs_rows_compact.py --rows {rows}, then train with --memmap "
                         f"--data <rows folder>/compact") from None
    except RC.CompactError as exc:
        raise SystemExit(f"--memmap: {exc}") from None


def memmap_keep(store, split: str, wait_keep: float, trusted_only: bool, hog_only: bool,
                seed: int = 0, *, max_rows: int = 0, trust: str = TRUST_LEGACY, schema=None,
                chunk: int = KEEP_CHUNK, drop_pending: bool = False,
                counts: dict | None = None) -> np.ndarray:
    """The store rows load() keeps for the same arguments, as ascending int64 row ids: the same
    filters (_keep_mask on `chunk` rows at a time, so one wait draw per row in row order, the
    stream load() draws a shard at a time) and the same --max-rows sample (_cap), so gather()
    of these ids is load()'s rows in load()'s order. ValueError where load() raises one.
    `drop_pending` and `counts` as load()'s; only then is own_pending read."""
    spec = rows_schema(store.root) if schema is None else _spec(schema)
    presses = PRESS_TRAINED if spec.v2 else PRESS_EXCLUDED
    col = trust_column(trust)
    if trusted_only and col not in store:
        raise ValueError(missing_trust(trust, ["the compact store"], store.root))
    if spec.v2:
        lacking = [c for c in cols_for(spec) if c not in store]
        if lacking:
            raise ValueError(f"the compact store in {store.root} has no {lacking}: not rows of "
                             f"{spec.name} (tools/rs_rows.py --schema v2)")
    if drop_pending and spec.v2:
        raise ValueError(f"--drop-pending-waits: {store.root} holds {spec.name} rows; legacy only")
    if drop_pending and PENDING_COLUMN not in store:
        raise ValueError(missing_pending(["the compact store"], store.root))
    need = (["split", "act_card", "hog"] + ([col] if trusted_only else [])
            + ([PRESS_COLUMN] if PRESS_COLUMN in store else [])
            + ([PENDING_COLUMN] if drop_pending else []))
    rng = np.random.default_rng(seed)
    picks = []
    for a in range(0, store.rows, chunk):
        z = {c: np.asarray(store[c][a:a + chunk]) for c in need}
        z["split"] = store.decode_split(z["split"])
        keep = _keep_mask(z, split, wait_keep, trusted_only, hog_only, rng, col, presses,
                          drop_pending, counts)
        picks.append(np.flatnonzero(keep) + a)
    found = sum(len(p) for p in picks)
    picks = _cap(picks, max_rows, seed)
    ids = (np.concatenate(picks) if picks else np.zeros(0)).astype(np.int64)
    if len(ids) < found:
        print(f"  {split}{' (hog)' if hog_only else ''}: {len(ids):,} of {found:,} rows "
              f"(--max-rows {max_rows:,})", flush=True)
    return ids


def gather(store, cols, ids, window: int = WINDOW_ROWS) -> dict:
    """Columns `cols` of the store rows `ids` (ascending) in RAM at the store's dtypes, read
    `window` rows at a time: load()'s dict (values, keys in `cols` order) for memmap_keep's ids,
    with the compacted dtypes (tools/rs_rows_compact.py) the batches widen anyway."""
    out = {}
    for c in cols:
        src = store[c]
        dst = out[c] = np.empty((len(ids),) + tuple(src.shape[1:]), dtype=src.dtype)
        for a in range(0, len(ids), window):
            dst[a:a + window] = src[ids[a:a + window]]
    return out


def store_phases(store, ids, window: int = WINDOW_ROWS) -> np.ndarray:
    """phase_of of the store rows `ids`, read `window` rows at a time."""
    ph = np.zeros(len(ids), dtype=np.int64)
    for a in range(0, len(ids), window):
        rows = ids[a:a + window]
        ph[a:a + len(rows)] = phase_of({"scalars": np.asarray(store["scalars"][rows]),
                                        "towers": np.asarray(store["towers"][rows])})
    return ph


def epoch_order(n: int, rng, chunk: int = CHUNK_ROWS, window: int = WINDOW_ROWS) -> np.ndarray:
    """One epoch of positions 0..n-1, each exactly once, for rows read from disk: runs of `chunk`
    consecutive positions in a random order, then each `window` of that shuffled within. A
    window's rows lie in about window / chunk runs of the file, and a batch draws from all of
    them."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    chunk, window = max(1, int(chunk)), max(1, int(window))
    runs = -(-n // chunk)
    starts = rng.permutation(runs).astype(np.int64) * chunk
    order = (starts[:, None] + np.arange(chunk, dtype=np.int64)[None, :]).reshape(-1)
    order = order[order < n]
    for w in range(0, n, window):
        seg = order[w:w + window]
        order[w:w + window] = seg[rng.permutation(len(seg))]
    return order


def to_model(b: dict) -> dict:
    """The dtypes batches() gives the model: small ints (int8 / uint8 / int16 / int32) as int64,
    float64 as float32, anything else as it is."""
    out = {}
    for c, t in b.items():
        if t.dtype in _SMALL_INTS:
            t = t.long()
        elif t.dtype == torch.float64:
            t = t.float()
        out[c] = t
    return out


class PinnedStager:
    """Host batches (dicts of numpy arrays) to `device` tensors. On CUDA every column goes
    through one of `slots` pinned host buffers and an asynchronous copy; before a buffer is
    filled again, the event recorded after its last copy is waited on, so the copy of one batch
    overlaps the step on the one before. Elsewhere: torch.from_numpy."""

    def __init__(self, device, slots: int = 2):
        self.device = device
        self.cuda = str(device).startswith("cuda")
        self._bufs = [{} for _ in range(max(1, int(slots)))]
        self._events = [None] * len(self._bufs)
        self._k = 0

    def __call__(self, arrays: dict) -> dict:
        if not self.cuda:
            return {c: torch.from_numpy(np.ascontiguousarray(a)) for c, a in arrays.items()}
        slot = self._k % len(self._bufs)
        self._k += 1
        if self._events[slot] is not None:
            self._events[slot].synchronize()
        bufs, out = self._bufs[slot], {}
        for c, a in arrays.items():
            src = torch.from_numpy(np.ascontiguousarray(a))
            buf = bufs.get(c)
            if (buf is None or buf.dtype != src.dtype or buf.shape[1:] != src.shape[1:]
                    or buf.shape[0] < src.shape[0]):
                buf = bufs[c] = torch.empty(tuple(src.shape), dtype=src.dtype).pin_memory()
            view = buf[:src.shape[0]]
            view.copy_(src)
            out[c] = view.to(self.device, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._events[slot] = event
        return out


def window_batches(store, cols, ids, order, size: int, window: int, put):
    """Batches of `size` of the store rows ids[order] (columns `cols`), read a window at a time:
    each window of `window` positions (a multiple of `size`) is read once, its rows in file
    order, then cut into batches in `order`'s order and handed to `put` (PinnedStager) and
    to_model. Every position of `order` is in one batch; all but the last are full."""
    window = max(size, int(window) // size * size)
    for w in range(0, len(order), window):
        rows = ids[order[w:w + window]]
        uniq, inv = np.unique(rows, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        data = {c: np.asarray(store[c][uniq]) for c in cols}
        for i in range(0, len(rows), size):
            pos = inv[i:i + size]
            yield to_model(put({c: v[pos] for c, v in data.items()}))


def to_device(data: dict, device) -> dict:
    """Rows in RAM (load()'s dict) as `device` tensors at their dtypes (--gpu-resident)."""
    return {c: torch.from_numpy(np.ascontiguousarray(v)).to(device) for c, v in data.items()}


def store_to_device(store, cols, ids, device, window: int = WINDOW_ROWS) -> dict:
    """Columns `cols` of the store rows `ids` as `device` tensors at the store's dtypes, read and
    copied `window` rows at a time (--memmap --gpu-resident: no host copy of the whole set)."""
    out = {}
    for c in cols:
        src = store[c]
        kind = torch.from_numpy(np.zeros(0, dtype=src.dtype)).dtype
        dst = torch.empty((len(ids),) + tuple(src.shape[1:]), dtype=kind, device=device)
        for a in range(0, len(ids), window):
            dst[a:a + window].copy_(torch.from_numpy(np.asarray(src[ids[a:a + window]])))
        out[c] = dst
    return out


def resident_batches(data: dict, order, size: int):
    """Batches of `size` of tensors `data` in `order`, by index_select where they are, then
    to_model: for the same order, the batches batches() makes from the same rows in RAM."""
    dev = next(iter(data.values())).device
    index = torch.from_numpy(np.ascontiguousarray(order, dtype=np.int64)).to(dev)
    for i in range(0, len(order), size):
        idx = index[i:i + size]
        yield to_model({c: t.index_select(0, idx) for c, t in data.items()})


def memmap_ram_need(n_train: int, n_eval: int, row_bytes: int, window: int, batch: int) -> int:
    """Bytes --memmap holds in RAM: the train ids and an epoch's order (int64 each), the
    validation rows (both sets), a window twice over (the rows read and the batches cut from
    them) and two pinned batches. The file cache comes on top of it."""
    return int(16 * n_train + n_eval * row_bytes + 2 * window * row_bytes + 2 * batch * row_bytes)


def resident_gpu_need(data_bytes: int, n_train: int) -> int:
    """Bytes --gpu-resident puts on the GPU: the training rows and an epoch's int64 order."""
    return int(data_bytes + 8 * n_train)


def budget_problem(what: str, need: int, budget_gb: float, hint: str = "") -> str | None:
    """Why `need` bytes do not fit a `budget_gb` GB (2**30 bytes) budget, or None when they do.
    `what` is "RAM" (--ram-budget-gb) or "GPU" (--gpu-budget-gb)."""
    if need <= float(budget_gb) * GB:
        return None
    flag = "--gpu-budget-gb" if what == "GPU" else "--ram-budget-gb"
    return (f"{what} budget: these rows need {need / GB:.2f} GB, more than {flag} "
            f"{float(budget_gb):g}. {hint}").strip()


class TrainFeed:
    """One epoch of training batches at a time (main) from rows in RAM (load(): batches(), as
    before E18), on the GPU (resident: resident_batches) or on disk (a store and its train ids:
    window_batches). `alpha` > 0 is --phase-balance, with the phases taken once."""

    def __init__(self, size: int, device, alpha: float = 0.0, *, train=None, store=None,
                 ids=None, cols=(), resident: bool = False, window: int = WINDOW_ROWS,
                 chunk: int = CHUNK_ROWS):
        self.size, self.device, self.alpha = int(size), device, float(alpha)
        self.store, self.ids, self.cols = store, ids, tuple(cols)
        self.window = max(self.size, int(window) // self.size * self.size)
        self.chunk = int(chunk)
        if store is None:
            self.n = len(train["act_card"])
            self.phases = phase_of(train) if self.alpha > 0 else None
        else:
            self.n = len(ids)
            self.phases = store_phases(store, ids, self.window) if self.alpha > 0 else None
        self.mode = "resident" if resident else ("memmap" if store is not None else "ram")
        self.train = train if self.mode == "ram" else None
        self.data = None
        if resident:
            self.data = (to_device(train, device) if store is None
                         else store_to_device(store, self.cols, ids, device, self.window))
        self.put = PinnedStager(device) if self.mode == "memmap" else None

    def phase_counts(self) -> np.ndarray:
        return np.bincount(self.phases, minlength=12)

    def epoch(self, rng):
        """The batches of one epoch; `rng` is main's generator (balanced_order, epoch_order)."""
        balanced = (balanced_from_phases(self.phases, self.alpha, rng) if self.alpha > 0
                    else None)
        if self.mode == "ram":
            yield from batches(self.train, self.size, True, self.device, order=balanced)
        elif self.mode == "resident":
            order = np.random.permutation(self.n) if balanced is None else balanced
            yield from resident_batches(self.data, order, self.size)
        else:
            order = (epoch_order(self.n, rng, self.chunk, self.window) if balanced is None
                     else balanced)
            yield from window_batches(self.store, self.cols, self.ids, order, self.size,
                                      self.window, self.put)


def mirror_batch(b: dict, mask, rules: dict) -> dict:
    """Batch `b` with the rows of bool `mask` [N] flipped left-right by `rules`
    (rs_schema.Schema.mirror(), module doc of tools/rs_schema.py, MIRROR): "flip" features k ->
    1 - k in the slots whose "where" id is not 0 (the column then float32), "swap" pairs of
    features, "tile" t -> tile - t where the "where" column is > 0. Columns without a rule, or
    whose rule's columns the batch lacks, are the batch's own; `b` is not changed."""
    out = dict(b)
    m = mask.to(torch.bool)
    for col, rule in rules.items():
        if col not in b or rule.get("where", col) not in b:
            continue
        t = b[col]
        if "flip" in rule:
            x = t.float()
            flipped = x.clone()
            k = [int(i) for i in rule["flip"]]
            flipped[..., k] = 1.0 - x[..., k]
            sel = (m[:, None] & (b[rule["where"]] != 0))[..., None]
            out[col] = torch.where(sel, flipped, x)
        elif "swap" in rule:
            perm = list(range(t.shape[-1]))
            for i, j in rule["swap"]:
                perm[i], perm[j] = perm[j], perm[i]
            sel = m.view((-1,) + (1,) * (t.dim() - 1))
            out[col] = torch.where(sel, t[..., perm], t)
        elif "tile" in rule:
            sel = m & (b[rule["where"]] > 0)
            out[col] = torch.where(sel, int(rule["tile"]) - t, t)
    return out


class Mirror:
    """--mirror p: each training row is flipped left-right with probability p (mirror_batch with
    the schema's rules), one draw per row from default_rng([MIRROR_SALT, seed]). p 0 draws
    nothing and returns every batch as it is."""

    def __init__(self, p: float = 0.0, schema=None, seed: int = 0):
        self.p = float(p)
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"--mirror: a probability in [0, 1], got {p!r}")
        self.rules = _spec(schema).mirror()
        self.rng = np.random.default_rng([MIRROR_SALT, int(seed)])

    def __call__(self, b: dict) -> dict:
        if self.p <= 0.0:
            return b
        n = int(b["act_card"].shape[0])
        mask = torch.from_numpy(self.rng.random(n) < self.p).to(b["act_card"].device)
        return mirror_batch(b, mask, self.rules)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

#: The rows each statistic of model_tokens.losses() is a mean over: "gate" rows
#: whose played card is in the recorded hand (waits included), "card" such rows
#: that are plays, "play" every play. Other statistics are means over all rows.
STAT_ROWS = {"gate_acc": "gate", "card_acc": "card", "tile_acc": "play", "tile_near": "play",
             "press_acc": "press"}
STATS = ("gate_acc", "card_acc", "tile_acc", "tile_near", "mean_w")


def eligible_counts(b, presses: bool | None = None) -> dict[str, int]:
    """How many rows of batch `b` each kind of statistic is a mean over. `presses` (a v2
    model's losses(presses=)): the gate rows are model_tokens.gate_targets', and "press"
    counts the press rows with a gate target (press_acc); None: the legacy counts."""
    card = b["act_card"].long()
    if presses is None:
        ok = target_slot(b["hand"].long(), card) >= 0
        plays = card > 0
        return {"rows": int(card.numel()), "gate": int(ok.sum()),
                "card": int((ok & plays).sum()), "play": int(plays.sum())}
    ok = gate_targets(b, presses) >= 0
    plays = card > 0
    pressed = (b["act_press"].long() > 0) if "act_press" in b else torch.zeros_like(plays)
    return {"rows": int(card.numel()), "gate": int(ok.sum()), "card": int((ok & plays).sum()),
            "play": int(plays.sum()), "press": int((ok & pressed).sum())}


@torch.no_grad()
def evaluate(model, data, device, beta, batch_size: int = 2048, presses: bool | None = None):
    """Validation metrics over all of `data`: correct / eligible rows per statistic,
    summed over batches (so a batch with few plays weighs little in card_acc),
    plus the row and play counts. `presses` (v2 models; None: legacy) goes to losses and
    eligible_counts, and adds press_acc."""
    model.eval()
    hits = {k: 0.0 for k in STATS}
    den = {k: 0 for k in STATS}
    rows = plays = 0
    kw = {} if presses is None else {"presses": bool(presses)}
    for b in batches(data, batch_size, False, device):
        _, s = losses(model, b, beta=beta, **kw)
        n = eligible_counts(b, **kw)
        rows += n["rows"]
        plays += n["play"]
        for key, v in s.items():
            k = n[STAT_ROWS.get(key, "rows")]
            # an accuracy is a float32 mean of 0/1 over k rows: round back to the count
            hits[key] = hits.get(key, 0.0) + (round(v * k) if key in STAT_ROWS else v * k)
            den[key] = den.get(key, 0) + k
    model.train()
    out = {k: (hits[k] / den[k] if den[k] else 0.0) for k in hits}
    out.update(rows=rows, plays=plays)
    return out


# ---------------------------------------------------------------------------
# run setup and records
# ---------------------------------------------------------------------------

def resolve_device(name: str) -> str:
    """The torch device for --device: "cpu", or "cuda" when CUDA works. A CUDA
    request without CUDA stops the run (SystemExit) rather than training on the CPU."""
    if name == "cpu":
        return "cpu"
    if name != "cuda":
        raise SystemExit(f"--device {name!r}: use cuda or cpu")
    if not torch.cuda.is_available():
        raise SystemExit(
            f"--device cuda, but CUDA is not available (torch {torch.__version__}, CUDA build "
            f"{torch.version.cuda}). Install the CUDA build of torch in .venvs/buildabot "
            f"(scripts/setup_windows.ps1), or pass --device cpu to train on the CPU.")
    return "cuda"


def git_commit(root: Path = ROOT) -> tuple[str | None, bool | None]:
    """(HEAD commit, whether tracked files differ from it), or None where git cannot say."""
    def git(*cmd):
        try:
            r = subprocess.run(["git", "-C", str(root), *cmd], capture_output=True, text=True,
                               timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    if not head:
        return None, None
    status = git("status", "--porcelain", "--untracked-files=no")
    return head, (None if status is None else bool(status))


def run_record(args, device: str, wait_keep: float, wait_keep_train: float,
               wait_keep_at_build: float | None, rows: dict | None = None) -> dict:
    """args.json: every argument as a string (as before E05; "wait_keep" is the
    resolved overall share), then `rows` (row_record: trust, trust_column, press_rows,
    press_rows_excluded) when given, then the run's environment as JSON values.

    Plain str / int / float / bool / None only: the record is also stored in every
    checkpoint, and torch.load's weights_only unpickler (torch >= 2.6's default, used by
    rs_eval and rs_selfplay) refuses any other class. torch.__version__ is a TorchVersion
    (a str subclass), so it is converted with str()."""
    rec = {k: str(v) for k, v in vars(args).items()}
    rec["wait_keep"] = str(wait_keep)
    if rows is not None:
        rec.update(_row_entry(rows))
    commit, dirty = git_commit()
    cuda = torch.version.cuda
    rec.update({
        "wait_keep_train": float(wait_keep_train),
        "wait_keep_at_build": None if wait_keep_at_build is None else float(wait_keep_at_build),
        "torch_version": str(torch.__version__),
        "torch_cuda": None if cuda is None else str(cuda),
        "numpy_version": str(np.__version__),
        "python_version": str(platform.python_version()),
        "device_name": str(torch.cuda.get_device_name(0)) if device == "cuda" else "cpu",
        "git_commit": commit,
        "git_dirty": dirty,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    })
    return rec


def checkpoint_meta(vocab: dict, d: int, layers: int, heads: int,
                    rows: dict | None = None) -> dict:
    """The checkpoint meta: all of vocab.json (ids, n_cards / n_units, and the
    engine fingerprint, schema, observation delay and wait_keep_at_build when the
    builder wrote them) plus the architecture, plus "train_rows" (the row_record)
    when the trust column or the press rows differ from the legacy ones; without it
    the meta is exactly what it was before E09 / E10 reached training."""
    meta = dict(vocab, d=d, layers=layers, heads=heads)
    if not legacy_rows(rows):
        meta[TRAIN_ROWS_KEY] = _row_entry(rows)
    return meta


def one_cycle_pct(steps: int, pct: float = 0.05) -> float:
    """OneCycleLR's warm-up share. Its warm-up phase ends at pct * steps - 1, and at
    exactly 0 (20 steps for 0.05) the scheduler divides by zero, so that case warms
    up a little longer."""
    return pct if float(pct * steps) - 1 != 0 else pct * 1.5


def read_vocab(data: Path) -> dict:
    path = Path(data) / "vocab.json"
    if not path.exists():
        raise SystemExit(f"no vocab.json in {data}: build rows with tools/rs_rows.py, "
                         f"or pass --data <rows folder>")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _nbytes(data: dict) -> int:
    return sum(v.nbytes for v in data.values())


def init_from(model, blob: dict, vocab: dict, schema, source, partial: bool = False) -> dict:
    """--init: load checkpoint `blob` into `model`. A strict load of a checkpoint of the rows'
    schema (legacy: exactly the load this script always did); another schema is refused
    (SystemExit) unless `partial` (--init-partial), which takes model_tokens.load_matching's
    tensors from a checkpoint with the rows' card and unit vocabulary (and unit token kind).
    Returns load_matching's report ({} for a strict load)."""
    meta = blob.get("meta") if isinstance(blob, dict) else None
    if not partial:
        have = model_schema(meta)
        if have != _spec(schema):
            raise SystemExit(f"--init {source}: a {have.name} checkpoint, and these rows are "
                             f"{_spec(schema).name}; pass --init-partial to take the weights "
                             "that fit")
        model.load_state_dict(blob["model"])
        return {}
    meta = meta or {}
    # absent means what tools/rs_rows.py wrote before the key: not live-compatible, unit types
    defaults = {"live_compat": False, "units_mode": "types"}
    for key in ("cards", "units", "live_compat", "units_mode"):
        if meta.get(key, defaults.get(key)) != vocab.get(key, defaults.get(key)):
            raise SystemExit(f"--init-partial {source}: its {key!r} is not the rows' (the token "
                             "ids would mean other cards or units)")
    report = load_matching(model, blob["model"])
    print(f"--init-partial {source}: {len(report['loaded'])} tensors loaded, "
          f"{len(report['widened'])} widened ({', '.join(report['widened']) or '-'}), "
          f"{len(report['skipped'])} skipped, {len(report['missing'])} new", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="0 = plain imitation; >0 = outcome-weighted (AWR temperature)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wait-keep", type=float, default=None,
                    help="share of all wait rows to train on (default: vocab.json "
                         f"wait_keep_at_build when the builder thinned waits, else {DEFAULT_WAIT_KEEP})")
    ap.add_argument("--all-rows", action="store_true",
                    help="include rows after the match's divergence point")
    ap.add_argument("--trust", choices=TRUST_MODES, default=TRUST_LEGACY,
                    help="which rows are trusted: 'legacy' (default) the `trusted` column, as "
                         "before; 'v2' the `trusted_v2` column of tools/rs_rows.py --trust v2 "
                         "(refused when the rows have none)")
    ap.add_argument("--finetune-hog", action="store_true")
    ap.add_argument("--init", type=Path)
    ap.add_argument("--init-partial", action="store_true",
                    help="with --init: take the checkpoint's tensors whose names and shapes fit "
                         "(a legacy model into a v2 one: its state_in / unit_in fill the first "
                         "columns); the card and unit vocabulary must be the rows'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--phase-balance", type=float, default=0.0,
                    help="0 = rows as they come; >0 = draw rare game phases (overtime, "
                         "a tower down) more often, weight (1/phase rows)**this")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                    help="cuda (default) stops with an error when CUDA is unavailable")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="at most this many rows per split (train, val, val Hog), a seeded "
                         "uniform sample: smoke runs, or less RAM; 0 = all")
    ap.add_argument("--runs", type=Path, default=RUNS,
                    help="folder that holds the run folders (default runs/tokens)")
    ap.add_argument("--memmap", action="store_true",
                    help="E18: --data is a store of tools/rs_rows_compact.py; the training rows "
                         "are read from disk a window at a time (memory-mapped), so they need "
                         "not fit in RAM")
    ap.add_argument("--memmap-window", type=int, default=WINDOW_ROWS, metavar="ROWS",
                    help=f"--memmap: rows read at a time (default {WINDOW_ROWS:,}; a multiple of "
                         "--batch)")
    ap.add_argument("--gpu-resident", action="store_true",
                    help="E18: copy the training rows to the GPU once (refused when they need "
                         "more than --gpu-budget-gb); needs --device cuda")
    ap.add_argument("--ram-budget-gb", type=float, default=RAM_BUDGET_GB, metavar="GB",
                    help=f"--memmap: what the trainer may hold in RAM (default {RAM_BUDGET_GB:g}, "
                         "for a 16 GB PC); a run that needs more stops before training")
    ap.add_argument("--gpu-budget-gb", type=float, default=GPU_BUDGET_GB, metavar="GB",
                    help=f"--gpu-resident: what the rows may take on the GPU (default "
                         f"{GPU_BUDGET_GB:g}, for 12 GB)")
    ap.add_argument("--val-max-rows", type=int, default=0,
                    help="at most this many rows of each validation set, a seeded uniform "
                         "sample (default 0: as --max-rows)")
    ap.add_argument("--mirror", type=float, default=0.0, metavar="P",
                    help="E18: flip each training row left-right with probability P (default 0, "
                         "as before; 0.5 doubles the data the model sees)")
    ap.add_argument("--drop-pending-waits", action="store_true",
                    help="legacy-schema decision-time rows only (tools/rs_rows.py --obs-delay "
                         "24): leave the wait rows with an own play in flight (own_pending) out "
                         "of training, after the same wait draws; the validation sets keep "
                         "them. A legacy model cannot see own_pending, so such a wait looks like "
                         "the tap it follows (runbook section 19). Refused for v2 rows and for "
                         "rows without own_pending")
    args = ap.parse_args(argv)
    if args.max_rows < 0:
        ap.error("--max-rows must be 0 (all rows) or more")
    if args.epochs < 1:
        ap.error("--epochs must be 1 or more")
    if args.init_partial and not args.init:
        ap.error("--init-partial needs --init")
    if args.val_max_rows < 0:
        ap.error("--val-max-rows must be 0 (as --max-rows) or more")
    if not 0.0 <= args.mirror <= 1.0:
        ap.error("--mirror is a probability, 0 to 1")
    if args.memmap_window < 1 or args.ram_budget_gb <= 0 or args.gpu_budget_gb <= 0:
        ap.error("--memmap-window, --ram-budget-gb and --gpu-budget-gb must be more than 0")
    if args.gpu_resident and args.device != "cuda":
        ap.error("--gpu-resident needs --device cuda")

    device = resolve_device(args.device)        # before anything slow
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    vocab = read_vocab(args.data)
    wait_keep, keep_p, at_build = resolve_wait_keep(args.wait_keep, vocab)
    args.wait_keep = wait_keep
    prov = {k: vocab[k] for k in PROVENANCE_KEYS if k in vocab}
    print(f"device {device} (torch {torch.__version__}, CUDA {torch.version.cuda}); rows "
          f"{args.data}: " + ", ".join(f"{k} {json.dumps(v)[:80]}" for k, v in prov.items()),
          flush=True)
    print(f"wait_keep {wait_keep} (rows built with {at_build}; kept at load with p {keep_p:g})",
          flush=True)
    schema = model_schema(vocab)              # E15: rs_rows/v2 rows train the v2 model
    if schema.v2:
        problems = RS.spec_problems(vocab)
        if problems:
            raise SystemExit(f"{args.data / 'vocab.json'}: its schema_spec is not this code's "
                             f"{schema.name}: " + "; ".join(problems[:5])
                             + ". Rebuild the rows with this code's tools/rs_rows.py.")
        print(f"schema {schema.name}: {len(cols_for(schema))} columns, state token "
              f"{schema.state_features}, unit features {schema.unit_features}, "
              f"{RS.buttons_of(vocab)} ability buttons", flush=True)
    store = open_store(args.data) if args.memmap else None     # E18
    if store is not None:
        rows_rec = row_record_store(store, args.trust, schema)
    else:
        if (Path(args.data) / RC.COMPACT_JSON).exists() and not any(
                Path(args.data).glob("shard-*.npz")):
            raise SystemExit(f"{args.data} is a compact store (tools/rs_rows_compact.py): train "
                             "on it with --memmap")
        rows_rec = row_record(args.data, args.trust, schema)   # SystemExit: --trust v2, no column
    if args.drop_pending_waits:       # before anything is loaded or written
        check_drop_pending(args.data, store, schema)
    if not legacy_rows(rows_rec):
        print(f"rows: {describe_rows(rows_rec)}", flush=True)
    dropped: dict = {}                # --drop-pending-waits: the train waits it left out
    dkw = ({"drop_pending": True, "counts": dropped} if args.drop_pending_waits else {})
    # v2: whether the press head learns (the rows carry press labels); None: the legacy losses.
    presses = rows_rec["press_rows"] == PRESS_TRAINED if schema.v2 else None
    lkw = {} if presses is None else {"presses": presses}
    val_cap = args.val_max_rows or args.max_rows
    cols = cols_for(schema)
    t0 = time.time()
    if store is None:
        train = load("train", keep_p, not args.all_rows, args.finetune_hog, args.seed, args.data,
                     max_rows=args.max_rows, trust=args.trust, schema=schema, **dkw)
        val = load("val", keep_p, True, args.finetune_hog, 1, args.data, max_rows=val_cap,
                   trust=args.trust, schema=schema)
        val_hog = load("val", keep_p, True, True, 1, args.data, max_rows=val_cap,
                       trust=args.trust, schema=schema)
        train_ids, n_train = None, len(train["act_card"])
        train_bytes = _nbytes(train)
        mb = (train_bytes + _nbytes(val) + _nbytes(val_hog)) / 2 ** 20
        where = ""
    else:
        # E18 --memmap: the ids of load()'s rows; only the validation rows are read into RAM.
        train = None
        train_ids = memmap_keep(store, "train", keep_p, not args.all_rows, args.finetune_hog,
                                args.seed, max_rows=args.max_rows, trust=args.trust,
                                schema=schema, **dkw)
        val_ids = memmap_keep(store, "val", keep_p, True, args.finetune_hog, 1,
                              max_rows=val_cap, trust=args.trust, schema=schema)
        hog_ids = memmap_keep(store, "val", keep_p, True, True, 1, max_rows=val_cap,
                              trust=args.trust, schema=schema)
        n_train, row_bytes = len(train_ids), store.row_bytes(cols)
        window = max(args.batch, args.memmap_window // args.batch * args.batch)
        problem = budget_problem(
            "RAM", memmap_ram_need(n_train, len(val_ids) + len(hog_ids), row_bytes, window,
                                   args.batch), args.ram_budget_gb,
            "Cap the validation sets (--val-max-rows, held in RAM), lower --memmap-window or "
            "--max-rows, or raise --ram-budget-gb with MuMu and the browser closed.")
        if problem:
            raise SystemExit(problem)
        val = gather(store, cols, val_ids, window)
        val_hog = gather(store, cols, hog_ids, window)
        train_bytes = n_train * row_bytes
        mb = (_nbytes(val) + _nbytes(val_hog)) / 2 ** 20
        where = f"; train read from {store.root} ({train_bytes / GB:.2f} GB, --memmap)"
    print(f"rows: train {n_train:,}  val {len(val['act_card']):,}  "
          f"val(hog) {len(val_hog['act_card']):,}  ({mb:,.0f} MB, {time.time() - t0:.0f}s)"
          + where, flush=True)
    if args.drop_pending_waits:
        print(f"--drop-pending-waits: {dropped.get(PENDING_DROPPED_KEY, 0):,} wait rows with an "
              f"own play in flight left out of training (before --max-rows); the validation "
              f"sets keep theirs", flush=True)
    if not n_train:
        raise SystemExit(f"no training rows in {args.data} (split 'train' after the filters)")
    if args.gpu_resident:
        problem = budget_problem(
            "GPU", resident_gpu_need(train_bytes, n_train), args.gpu_budget_gb,
            "Train without --gpu-resident (--memmap reads the rows from disk), or with "
            "--max-rows.")
        if problem:
            raise SystemExit(problem)
    feed = TrainFeed(args.batch, device, args.phase_balance, train=train, store=store,
                     ids=train_ids, cols=cols, resident=args.gpu_resident,
                     window=args.memmap_window)
    train = None                          # the feed holds what it needs (resident: on the GPU)
    mirror = Mirror(args.mirror, schema, args.seed)

    meta = checkpoint_meta(vocab, args.d, args.layers, args.heads, rows=rows_rec)
    mkw = {"schema": schema.name} if schema.v2 else {}
    model = TokenPolicy(meta["n_cards"], meta["n_units"], d=args.d, layers=args.layers,
                        heads=args.heads, **mkw).to(device)
    if args.init:
        init_from(model, torch.load(args.init, map_location=device), vocab, schema,
                  args.init, partial=args.init_partial)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((n_train + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps,
                                                pct_start=one_cycle_pct(steps))
    out = Path(args.runs) / args.name
    out.mkdir(parents=True, exist_ok=True)
    record = run_record(args, device, wait_keep, keep_p, at_build, rows=rows_rec)
    if args.drop_pending_waits:       # the count only with the flag
        record[PENDING_DROPPED_KEY] = int(dropped.get(PENDING_DROPPED_KEY, 0))
    (out / "args.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    best = -1.0
    step = 0
    rng = np.random.default_rng(args.seed)
    if args.phase_balance > 0:
        counts = feed.phase_counts()
        print("train rows per phase: " + ", ".join(f"{PHASE_NAMES[i]} {c:,}"
                                                   for i, c in enumerate(counts) if c), flush=True)
    with (out / "log.jsonl").open("a", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            for b in feed.epoch(rng):
                b = mirror(b)                 # E18: --mirror 0 returns the batch as it is
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss, s = losses(model, b, beta=args.beta, **lkw)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % 500 == 0:
                    print(f"  ep {epoch} step {step}/{steps} loss {loss.item():.3f} "
                          f"gate {s['gate_acc']:.3f} card {s['card_acc']:.3f} "
                          f"tile {s['tile_acc']:.3f} near {s['tile_near']:.3f} w {s['mean_w']:.2f}"
                          + (f" press {s['press_acc']:.3f}" if "press_acc" in s else ""),
                          flush=True)
            v = evaluate(model, val, device, 0.0, presses=presses)
            vh = evaluate(model, val_hog, device, 0.0, presses=presses)
            # Per phase, so a weak phase is not hidden in the average.
            ph = phase_of(val_hog)
            per_phase = {}
            for i in range(12):
                m = ph == i
                if m.sum() >= 200:
                    sub = {c: arr[m] for c, arr in val_hog.items()}
                    e = evaluate(model, sub, device, 0.0, presses=presses)
                    per_phase[PHASE_NAMES[i]] = {"rows": int(m.sum()), "card": round(e["card_acc"], 3),
                                                 "near": round(e["tile_near"], 3)}
            print("  hog per phase: " + "; ".join(f"{k} ({v_['rows']}) card {v_['card']} near {v_['near']}"
                                                  for k, v_ in per_phase.items()), flush=True)
            score = vh["card_acc"] + vh["tile_near"]
            rec = {"epoch": epoch, "val": v, "val_hog": vh, "val_hog_phase": per_phase,
                   "time": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"epoch {epoch}: val card {v['card_acc']:.3f} tile {v['tile_acc']:.3f} "
                  f"near {v['tile_near']:.3f} gate {v['gate_acc']:.3f} | hog card {vh['card_acc']:.3f} "
                  f"near {vh['tile_near']:.3f}"
                  + (f" | press {v['press_acc']:.3f} (hog {vh['press_acc']:.3f})"
                     if "press_acc" in v else ""), flush=True)
            model.sync_critic()          # E15: the critic starts as value_head (no-op legacy)
            blob = {"model": model.state_dict(), "meta": meta, "epoch": epoch,
                    "train_tokens": record}
            torch.save(blob, out / f"ep{epoch + 1:02d}.pt")
            torch.save(blob, out / "last.pt")
            is_best = score > best
            if is_best:
                best = score
                torch.save(blob, out / "best.pt")
            print(f"  saved ep{epoch + 1:02d}.pt, last.pt{', best.pt' if is_best else ''}",
                  flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
