r"""Compact tools/rs_rows.py shards into per-column files for sim/train_tokens.py --memmap (E18).

    .venvs\royalegym\Scripts\python.exe tools\rs_rows_compact.py --rows data\rs_rows_v2
    .venvs\royalegym\Scripts\python.exe tools\rs_rows_compact.py --rows data\rs_rows_v2 --verify
    .venvs\buildabot\Scripts\python.exe -m sim.train_tokens --memmap --data data\rs_rows_v2\compact ...

Why (tokens-13): the shards are zipped .npz files, so a column can only be read by inflating it
whole, and sim/train_tokens.py's default loader puts every kept row in RAM. That stops at about
30k matches on a 16 GB PC. Here every column of every shard goes into one uncompressed .npy file
(np.save's format), which np.load(..., mmap_mode="r") maps without reading, so the trainer reads
only the rows of each batch (sim/train_tokens.py, "--memmap").

LAYOUT of <out> (default <rows>/compact):
  columns/<name>.npy  one file per shard column, the rows of every shard in shard name order (the
                      order sim/train_tokens.load reads them), first axis the row
  index/<name>.npy    int64 row ids, ascending: split_<split> for each split name (split_train,
                      split_val, split_test), and for each bool filter column the rows where it
                      is True (trusted, hog, and trusted_v2 when the rows have it)
  match_names.npy     the replay tags the match column's codes point into
  vocab.json, manifest.json
                      byte copies of the rows folder's (the trainer reads vocab.json from here,
                      so a checkpoint's meta is the same as when it trains on the shards)
  compact.json        FORMAT, rows, shards (name, rows, start row, and the dtype of each column
                      the shard stored at another dtype), columns (dtype, per-row shape, how
                      it was stored), splits, matches, index counts, the rows' schema and the
                      sha1 of their vocab.json. Written last: a folder without it is unfinished,
                      and CompactRows refuses it.

DTYPES. A column is stored at the widest dtype its shards hold (np.result_type, the promotion
np.concatenate and the loader do), except:
  * unit_feats (HALF_COLS) is float16, the dtype sim/train_tokens.load casts it to; rows from
    tools/rs_rows.py are float16 already;
  * a column whose schema column has a `compact` dtype (tools/rs_schema.py: legacy tick int32,
    side int8, next int16, outcome and won float32) is stored at it when every shard's values
    survive the cast exactly (checked in the first pass), else at its widest dtype;
  * split and match, strings in the shards, become codes: split int8 into compact.json "splits"
    (train, val, test, then any other name in the order met), match int32 into match_names.npy
    (each shard's distinct tags, sorted, in shard order).
So every value comes back exactly (CompactRows.shard, verify), unit_feats of a shard stored
wider than float16 as the trainer reads it. The first pass reads only the .npy headers inside
the shards and the small columns that may be compacted; the second writes each shard's columns
into the open files (np.lib.format.open_memmap).

Pure numpy and the standard library (and tools/rs_schema.py), so both venvs run it and the trainer
imports CompactRows from here.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat

_TOOLS = str(Path(__file__).resolve().parent)
if _TOOLS not in sys.path:
    sys.path.append(_TOOLS)
import rs_schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rs_rows"
#: compact.json "format"; CompactRows refuses another.
FORMAT = "rs_rows_compact/1"
COMPACT_JSON = "compact.json"
COLUMNS_DIR, INDEX_DIR = "columns", "index"
MATCH_NAMES = "match_names.npy"
#: The split names tools/rs_rows.split_of gives, codes 0, 1, 2.
SPLITS = ("train", "val", "test")
#: sim/train_tokens.HALF_COLS: loaded as float16 whatever the shard holds (a test checks).
HALF_COLS = ("unit_feats",)
#: The string columns and the dtype of their codes.
CODE_COLUMNS = {"split": "int8", "match": "int32"}
#: The bool columns that get an index of their True rows (when the rows have them).
INDEX_FILTERS = ("trusted", "trusted_v2", "hog")
#: The files copied from the rows folder.
COPIED = ("vocab.json", "manifest.json")


class CompactError(ValueError):
    """Shards that cannot be compacted together, or a folder that is no finished compact store."""


# ---------------------------------------------------------------------------
# reading shards
# ---------------------------------------------------------------------------

def shard_paths(rows) -> list:
    """The shards of a rows folder in the order sim/train_tokens.load reads them."""
    return sorted(Path(rows).glob("shard-*.npz"))


def _header(f):
    """(dtype, shape) from an open .npy stream (np.lib.format), or None for a header version
    this reader does not know."""
    version = npformat.read_magic(f)
    if version == (1, 0):
        shape, _, dtype = npformat.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, _, dtype = npformat.read_array_header_2_0(f)
    else:
        return None
    return np.dtype(dtype), tuple(int(d) for d in shape)


def npz_columns(path) -> dict:
    """{column: (dtype, shape)} of an .npz, in the archive's order, from the members' .npy
    headers alone (the arrays are not inflated, except for an unknown header version)."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            name = info.filename[:-len(".npy")]
            with zf.open(info) as f:
                head = _header(f)
            if head is None:
                with np.load(path) as z:
                    arr = z[name]
                head = (arr.dtype, tuple(arr.shape))
            out[name] = head
    return out


def row_schema(vocab) -> rs_schema.Schema:
    """The schema of rows whose vocab.json is `vocab`: rs_schema.V2 for "rs_rows/v2" (or "v2"),
    else LEGACY (sim/model_tokens.model_schema's rule: absent and unknown names are legacy)."""
    name = (vocab or {}).get("schema")
    if isinstance(name, str) and name.strip() in (rs_schema.SCHEMA_V2, "v2"):
        return rs_schema.V2
    return rs_schema.LEGACY


def _vocab(rows) -> dict:
    path = Path(rows) / "vocab.json"
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def _lossless(arr: np.ndarray, dtype) -> bool:
    """Whether every value of `arr` comes back unchanged through `dtype`."""
    back = arr.astype(dtype).astype(arr.dtype)
    if arr.dtype.kind == "f":
        return bool(np.array_equal(back, arr, equal_nan=True))
    return bool(np.array_equal(back, arr))


# ---------------------------------------------------------------------------
# pass 1: the plan
# ---------------------------------------------------------------------------

def plan(rows) -> dict:
    """What compact() writes for the shards in `rows`, from their headers and the columns that
    may be compacted: {"paths", "rows" (per shard), "names" (the first shard's column order),
    "shape" (per row), "dtype" (the store's), "how" (per column: "as stored", "compact",
    "float16 as the trainer loads it", "codes"), "shard_dtypes" (per shard, per column)}.
    CompactError for no shards, shards whose columns or row shapes disagree, a column whose
    first axis is not the shard's row count, or a string column that is not split or match."""
    paths = shard_paths(rows)
    if not paths:
        raise CompactError(f"no shard-*.npz in {rows}")
    heads = [npz_columns(p) for p in paths]
    names = list(heads[0])
    if not names:
        raise CompactError(f"{paths[0].name} holds no arrays")
    want = set(names)
    counts = []
    for p, h in zip(paths, heads):
        if set(h) != want:
            raise CompactError(f"{p.name}: columns {sorted(set(h) ^ want)} differ from "
                               f"{paths[0].name}'s; build every shard with the same flags")
        n = {h[c][1][0] if h[c][1] else -1 for c in names}
        if len(n) != 1 or -1 in n:
            raise CompactError(f"{p.name}: its columns do not all have the same row count ({n})")
        counts.append(n.pop())
    shape, dtype, how = {}, {}, {}
    spec = row_schema(_vocab(rows))
    for c in names:
        shapes = {h[c][1][1:] for h in heads}
        if len(shapes) != 1:
            raise CompactError(f"column {c}: rows of shapes {sorted(shapes)} in different shards")
        shape[c] = shapes.pop()
        kinds = {h[c][0].kind for h in heads}
        if kinds & {"U", "S", "O"}:
            if c not in CODE_COLUMNS or kinds - {"U"}:
                raise CompactError(f"column {c}: strings ({sorted(kinds)}); only "
                                   f"{sorted(CODE_COLUMNS)} may be")
            dtype[c], how[c] = np.dtype(CODE_COLUMNS[c]), "codes"
            continue
        widest = np.result_type(*[h[c][0] for h in heads])
        if c in HALF_COLS:
            dtype[c], how[c] = np.dtype(np.float16), (
                "as stored" if widest == np.float16 else "float16 as the trainer loads it")
            continue
        dtype[c], how[c] = widest, "as stored"
        small = spec.column(c).compact if c in spec else None
        if small is not None and np.dtype(small) != widest:
            ok = True
            for p in paths:
                with np.load(p) as z:
                    if not _lossless(z[c], small):
                        ok = False
                        break
            if ok:
                dtype[c], how[c] = np.dtype(small), "compact"
    shard_dtypes = [{c: str(h[c][0]) for c in names} for h in heads]
    return {"paths": paths, "rows": counts, "names": names, "shape": shape, "dtype": dtype,
            "how": how, "shard_dtypes": shard_dtypes}


# ---------------------------------------------------------------------------
# pass 2: writing
# ---------------------------------------------------------------------------

class _Codes:
    """String -> code, names in the order they are first met (split starts with SPLITS)."""

    def __init__(self, column: str):
        self.column = column
        self.names: list = list(SPLITS) if column == "split" else []
        self.code = {n: i for i, n in enumerate(self.names)}
        self.limit = np.iinfo(CODE_COLUMNS[column]).max

    def encode(self, arr: np.ndarray) -> np.ndarray:
        uniq, inv = np.unique(arr, return_inverse=True)
        for name in uniq.tolist():
            if name not in self.code:
                if len(self.names) > self.limit:
                    raise CompactError(f"column {self.column}: more than {self.limit + 1} names")
                self.code[name] = len(self.names)
                self.names.append(name)
        table = np.asarray([self.code[n] for n in uniq.tolist()], dtype=CODE_COLUMNS[self.column])
        return table[np.asarray(inv).reshape(-1)]


def _sha1(path: Path) -> str | None:
    return hashlib.sha1(path.read_bytes()).hexdigest() if path.exists() else None


def compact(rows, out=None, *, force: bool = False, log=print) -> dict:
    """Writes the compact store of the shards in `rows` into `out` (default <rows>/compact) and
    returns its compact.json. CompactError when `out` is `rows` itself, or already holds a
    finished store and not `force` (with `force` its columns/ and index/ are removed first)."""
    rows = Path(rows)
    out = rows / "compact" if out is None else Path(out)
    if out.resolve() == rows.resolve():
        raise CompactError(f"--out {out} is the rows folder; pick another (default "
                           f"{rows / 'compact'})")
    if (out / COMPACT_JSON).exists():
        if not force:
            raise CompactError(f"{out} already holds a compact store; pass --force to replace it")
        (out / COMPACT_JSON).unlink()           # unfinished until the new one is written
    for sub in (COLUMNS_DIR, INDEX_DIR):
        if force and (out / sub).exists():
            shutil.rmtree(out / sub)
    t0 = time.time()
    p = plan(rows)
    total = sum(p["rows"])
    if total == 0:
        raise CompactError(f"the shards in {rows} hold no rows")
    (out / COLUMNS_DIR).mkdir(parents=True, exist_ok=True)
    (out / INDEX_DIR).mkdir(parents=True, exist_ok=True)
    log(f"compact: {len(p['paths'])} shards, {total:,} rows, {len(p['names'])} columns -> {out}")
    files = {}
    for c in p["names"]:
        shape = (total,) + p["shape"][c]
        path = out / COLUMNS_DIR / f"{c}.npy"
        if math.prod(shape) == 0:
            np.save(path, np.zeros(shape, dtype=p["dtype"][c]))     # nothing to map
        else:
            files[c] = npformat.open_memmap(path, mode="w+", dtype=p["dtype"][c],
                                                 shape=shape)
    codes = {c: _Codes(c) for c in p["names"] if p["how"][c] == "codes"}
    index: dict = {}
    filters = [c for c in INDEX_FILTERS if c in p["names"]]
    shards, start = [], 0
    try:
        for k, (path, n) in enumerate(zip(p["paths"], p["rows"])):
            with np.load(path) as z:
                for c in p["names"]:
                    arr = z[c]
                    if c in codes:
                        arr = codes[c].encode(arr)
                    else:
                        arr = arr.astype(p["dtype"][c], copy=False)
                    if c in files:
                        files[c][start:start + n] = arr
                    if c == "split":
                        for code in np.unique(arr).tolist():
                            name = f"split_{codes['split'].names[code]}"
                            index.setdefault(name, []).append(np.flatnonzero(arr == code) + start)
                    elif c in filters:
                        index.setdefault(c, []).append(np.flatnonzero(arr) + start)
            other = {c: d for c, d in p["shard_dtypes"][k].items() if d != str(p["dtype"][c])}
            shards.append({"name": path.name, "rows": int(n), "start": int(start), "dtypes": other})
            start += n
            if (k + 1) % 50 == 0 or k + 1 == len(p["paths"]):
                log(f"  {k + 1}/{len(p['paths'])} shards ({time.time() - t0:.0f}s)")
        for f in files.values():
            f.flush()
    finally:
        files.clear()                       # unmapped, so Windows lets the files go
    names = [f"split_{s}" for s in (codes["split"].names if "split" in codes else ())]
    counts = {}
    for name in names + filters:
        ids = (np.concatenate(index[name]) if index.get(name) else np.zeros(0)).astype(np.int64)
        np.save(out / INDEX_DIR / f"{name}.npy", ids)
        counts[name] = int(len(ids))
    if "match" in codes:
        np.save(out / MATCH_NAMES, np.asarray(codes["match"].names, dtype=str))
    for name in COPIED:
        if (rows / name).exists():
            shutil.copyfile(rows / name, out / name)
    vocab = _vocab(rows)
    meta = {
        "format": FORMAT, "rows": int(total), "source": str(rows),
        "schema": vocab.get("schema"), "vocab_sha1": _sha1(rows / "vocab.json"),
        "shards": shards,
        "columns": {c: {"dtype": str(p["dtype"][c]), "shape": list(p["shape"][c]),
                        "stored": p["how"][c]} for c in p["names"]},
        "splits": list(codes["split"].names) if "split" in codes else [],
        "matches": len(codes["match"].names) if "match" in codes else 0,
        "index": counts,
        "made_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    (out / COMPACT_JSON).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"done: {total:,} rows, {store_bytes(meta) / 2 ** 30:.2f} GB of columns "
        f"({time.time() - t0:.0f}s)")
    return meta


def store_bytes(meta: dict, columns=None) -> int:
    """Bytes of the store's `columns` (default all), from compact.json."""
    total = 0
    for c, spec in meta["columns"].items():
        if columns is None or c in columns:
            total += meta["rows"] * math.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize
    return total


# ---------------------------------------------------------------------------
# reading a store
# ---------------------------------------------------------------------------

def _map(path: Path) -> np.ndarray:
    """A .npy file memory-mapped read-only, or read when it holds no bytes (nothing to map)."""
    with open(path, "rb") as f:
        head = _header(f)
    if head is not None and math.prod(head[1]) * head[0].itemsize == 0:
        return np.load(path)
    return np.load(path, mmap_mode="r")


class CompactRows:
    """A finished compact store (module doc), every column mapped read-only (np.load mmap_mode
    "r"). FileNotFoundError without compact.json, CompactError for another format or a column
    file that does not match it."""

    def __init__(self, root):
        self.root = Path(root)
        path = self.root / COMPACT_JSON
        if not path.exists():
            raise FileNotFoundError(f"no {COMPACT_JSON} in {self.root}: make the store with "
                                    f"tools/rs_rows_compact.py --rows <rows folder>")
        self.meta = json.loads(path.read_text(encoding="utf-8"))
        if self.meta.get("format") != FORMAT:
            raise CompactError(f"{path}: format {self.meta.get('format')!r}, this code reads "
                               f"{FORMAT!r}; compact the rows again")
        self.rows = int(self.meta["rows"])
        self.splits = tuple(self.meta.get("splits", ()))
        self.columns = {}
        for c, spec in self.meta["columns"].items():
            arr = _map(self.root / COLUMNS_DIR / f"{c}.npy")
            want = (self.rows,) + tuple(spec["shape"])
            if arr.shape != want or str(arr.dtype) != spec["dtype"]:
                raise CompactError(f"{c}.npy: {arr.shape} {arr.dtype}, compact.json says "
                                   f"{want} {spec['dtype']}")
            self.columns[c] = arr

    def __contains__(self, name) -> bool:
        return name in self.columns

    def __getitem__(self, name) -> np.ndarray:
        return self.columns[name]

    @property
    def names(self) -> tuple:
        return tuple(self.columns)

    @property
    def shard_names(self) -> list:
        return [s["name"] for s in self.meta["shards"]]

    def split_code(self, split: str) -> int:
        """The code of `split` in the split column, -1 for a name the rows do not have."""
        return self.splits.index(split) if split in self.splits else -1

    def decode_split(self, codes) -> np.ndarray:
        """Split names for codes of the split column."""
        return np.asarray(self.splits, dtype=str)[np.asarray(codes, dtype=np.int64)]

    def match_names(self) -> np.ndarray:
        return np.load(self.root / MATCH_NAMES)

    def index(self, name: str) -> np.ndarray:
        """index/<name>.npy (module doc): int64 row ids, ascending."""
        return np.load(self.root / INDEX_DIR / f"{name}.npy")

    def row_bytes(self, columns=None) -> int:
        """Bytes per row of `columns` (default all) at the store's dtypes."""
        return sum(math.prod(self.columns[c].shape[1:]) * self.columns[c].dtype.itemsize
                   for c in (self.columns if columns is None else columns))

    def count_positive(self, name: str, chunk: int = 1 << 22) -> int:
        """Rows where column `name` is > 0, read `chunk` rows at a time."""
        col = self.columns[name]
        return int(sum(int((np.asarray(col[a:a + chunk]) > 0).sum())
                       for a in range(0, self.rows, chunk)))

    def shard(self, k: int) -> dict:
        """Shard k as tools/rs_rows.py wrote it: every column at the shard's dtype, codes back
        to strings (unit_feats of a shard stored wider than float16: as the trainer reads it,
        widened back)."""
        entry = self.meta["shards"][k]
        a, n = entry["start"], entry["rows"]
        out = {}
        names = None
        for c, arr in self.columns.items():
            part = np.asarray(arr[a:a + n])
            if c == "split":
                part = self.decode_split(part)
            elif c == "match":
                names = self.match_names() if names is None else names
                part = names[part.astype(np.int64)]
            dtype = entry.get("dtypes", {}).get(c)
            out[c] = part.astype(dtype) if dtype is not None else part
        return out

    def close(self) -> None:
        """Drops the maps (Windows keeps a mapped file open)."""
        self.columns = {}


def verify(rows, out=None, log=print) -> list:
    """Every shard of `rows` against the store in `out` (default <rows>/compact): [] when each
    column comes back with the same values and dtype (unit_feats of a shard stored wider than
    float16: equal to the shard's cast to float16), else the problems."""
    rows = Path(rows)
    store = CompactRows(rows / "compact" if out is None else out)
    problems = []
    try:
        paths = shard_paths(rows)
        if [p.name for p in paths] != store.shard_names:
            return [f"shards {[p.name for p in paths]}, the store has {store.shard_names}"]
        for k, path in enumerate(paths):
            back = store.shard(k)
            with np.load(path) as z:
                if list(z.files) != list(back):
                    problems.append(f"{path.name}: columns {list(z.files)}, the store "
                                    f"{list(back)}")
                    continue
                for c in z.files:
                    arr = z[c]
                    want = (arr.astype(np.float16).astype(arr.dtype) if c in HALF_COLS
                            and arr.dtype != np.float16 else arr)
                    same = (np.array_equal(back[c], want, equal_nan=True)
                            if want.dtype.kind == "f" else np.array_equal(back[c], want))
                    if not same or back[c].dtype != arr.dtype or back[c].shape != arr.shape:
                        problems.append(f"{path.name}: column {c} differs")
            if (k + 1) % 50 == 0:
                log(f"  verified {k + 1}/{len(paths)} shards")
    finally:
        store.close()
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compact rs_rows shards for train_tokens --memmap.")
    ap.add_argument("--rows", type=Path, default=DATA,
                    help="the rows folder (shard-*.npz and vocab.json; default data/rs_rows)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where the store goes (default <rows>/compact)")
    ap.add_argument("--force", action="store_true", help="replace a store already in --out")
    ap.add_argument("--verify", action="store_true",
                    help="afterwards, read every shard again and compare it with the store")
    args = ap.parse_args(argv)
    try:
        meta = compact(args.rows, args.out, force=args.force)
    except (CompactError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for c, spec in meta["columns"].items():
        print(f"  {c:18s} {'x'.join(map(str, spec['shape'])) or '-':>6s} {spec['dtype']:8s} "
              f"{spec['stored']}")
    print("  index: " + ", ".join(f"{k} {v:,}" for k, v in meta["index"].items()))
    if args.verify:
        problems = verify(args.rows, args.out)
        if problems:
            print("verify: " + "; ".join(problems[:10])
                  + (f" ... ({len(problems)} in all)" if len(problems) > 10 else ""))
            return 1
        print(f"verify: every shard comes back exactly ({meta['rows']:,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
