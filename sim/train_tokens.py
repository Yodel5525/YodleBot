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

`--wait-keep` is the share of ALL wait rows that training sees. When vocab.json
says the builder already thinned waits (`wait_keep_at_build`), training keeps
each remaining wait with probability wait_keep / wait_keep_at_build, and the
default is wait_keep_at_build itself (no further thinning). Either way
args.json records the overall share as "wait_keep", which is what RSAgent
(tools/rs_eval.py) adds back as log(1 / wait_keep) on the wait logit.

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

from .model_tokens import TokenPolicy, losses, target_slot

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rs_rows"
RUNS = ROOT / "runs" / "tokens"
COLS = ("unit_ids", "unit_feats", "towers", "scalars", "hand", "next", "opp_seen",
        "opp_last", "opp_last_age", "outcome", "won", "act_card", "act_x", "act_y")
#: Columns stored as float16 whatever the shard holds (the largest one, ~84% of a row).
HALF_COLS = ("unit_feats",)
DEFAULT_WAIT_KEEP = 0.35
#: vocab.json keys that describe how the rows were made (E01 / E04 / E09 / E10 / E11 / E18).
#: The whole of vocab.json goes into the checkpoint meta; these are the ones
#: printed at start and the ones a checkpoint must not lose.
PROVENANCE_KEYS = ("engine", "version", "fingerprint", "engine_fingerprint", "card_table",
                   "schema", "obs_delay", "obs_delay_jitter", "wait_keep_at_build",
                   "live_compat", "trust", "presses")
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
#: The checkpoint meta key of the row record (row_record) when it is not the legacy one.
TRAIN_ROWS_KEY = "train_rows"


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


def _keep_mask(z, split: str, wait_keep: float, trusted_only: bool, hog_only: bool, rng,
               trust_col: str = "trusted"):
    """The rows of one shard the loader keeps. One rng draw per row of the shard,
    whatever the other filters say, so the same seed keeps the same waits.
    `trust_col` is the trusted-rows column (trust_column). Press rows (act_press > 0,
    when the shard has that column) are left out after the draw: they are not waits."""
    keep = z["split"] == split
    if trusted_only:
        keep &= z[trust_col]
    if hog_only:
        keep &= z["hog"]
    waits = z["act_card"] == 0
    keep &= ~waits | (rng.random(len(waits)) < wait_keep)
    if PRESS_COLUMN in _files(z):
        keep &= ~(z[PRESS_COLUMN] > 0)
    return keep


def row_record(root: Path, trust: str = TRUST_LEGACY) -> dict:
    """What --trust and the press rows come to on the shards in `root`, read before any
    row is loaded: {"trust", "trust_column", "press_rows": "excluded" when a shard has an
    act_press column, else "none", "press_rows_excluded": rows with act_press > 0 over
    every split}. args.json gets it as it is, the checkpoint meta as "train_rows" when it
    is not the legacy record (legacy_rows). SystemExit when --trust is not legacy and a
    shard lacks its column (rows built without --trust v2). No shards: the legacy counts
    (load then says there are none)."""
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
    return {"trust": trust, "trust_column": col,
            "press_rows": PRESS_EXCLUDED if press_shards else PRESS_NONE,
            "press_rows_excluded": presses}


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
         root: Path = DATA, *, max_rows: int = 0, trust: str = TRUST_LEGACY):
    """The rows of one split, every column in COLS, shards in name order.

    The same rows, order and dtypes as concatenating every shard's kept rows and
    then casting unit_feats to float16 (the loader before E05), with a lower
    peak: a first pass reads only the filter columns, then each column is
    written into an array of the final size, one shard at a time, so neither
    the per-shard parts nor a float16 copy are held next to the result.
    `max_rows` > 0 keeps a seeded uniform sample of at most that many rows.
    `trust` (--trust) names the column `trusted_only` reads (trust_column);
    ValueError when a shard lacks it. Press rows are left out (_keep_mask).
    """
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
            picks.append(np.flatnonzero(_keep_mask(z, split, wait_keep, trusted_only,
                                                   hog_only, rng, col)))
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
            for c in COLS:
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
    ph = phase_of(data)
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
# metrics
# ---------------------------------------------------------------------------

#: The rows each statistic of model_tokens.losses() is a mean over: "gate" rows
#: whose played card is in the recorded hand (waits included), "card" such rows
#: that are plays, "play" every play. Other statistics are means over all rows.
STAT_ROWS = {"gate_acc": "gate", "card_acc": "card", "tile_acc": "play", "tile_near": "play"}
STATS = ("gate_acc", "card_acc", "tile_acc", "tile_near", "mean_w")


def eligible_counts(b) -> dict[str, int]:
    """How many rows of batch `b` each kind of statistic is a mean over."""
    card = b["act_card"].long()
    ok = target_slot(b["hand"].long(), card) >= 0
    plays = card > 0
    return {"rows": int(card.numel()), "gate": int(ok.sum()), "card": int((ok & plays).sum()),
            "play": int(plays.sum())}


@torch.no_grad()
def evaluate(model, data, device, beta, batch_size: int = 2048):
    """Validation metrics over all of `data`: correct / eligible rows per statistic,
    summed over batches (so a batch with few plays weighs little in card_acc),
    plus the row and play counts."""
    model.eval()
    hits = {k: 0.0 for k in STATS}
    den = {k: 0 for k in STATS}
    rows = plays = 0
    for b in batches(data, batch_size, False, device):
        _, s = losses(model, b, beta=beta)
        n = eligible_counts(b)
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
        rec.update({"trust": str(rows["trust"]), "trust_column": str(rows["trust_column"]),
                    "press_rows": str(rows["press_rows"]),
                    "press_rows_excluded": int(rows["press_rows_excluded"])})
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
        meta[TRAIN_ROWS_KEY] = {"trust": str(rows["trust"]),
                                "trust_column": str(rows["trust_column"]),
                                "press_rows": str(rows["press_rows"]),
                                "press_rows_excluded": int(rows["press_rows_excluded"])}
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
    args = ap.parse_args(argv)
    if args.max_rows < 0:
        ap.error("--max-rows must be 0 (all rows) or more")
    if args.epochs < 1:
        ap.error("--epochs must be 1 or more")

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
    rows_rec = row_record(args.data, args.trust)      # SystemExit: --trust v2 without the column
    if not legacy_rows(rows_rec):
        print(f"rows: {describe_rows(rows_rec)}", flush=True)
    t0 = time.time()
    train = load("train", keep_p, not args.all_rows, args.finetune_hog, args.seed, args.data,
                 max_rows=args.max_rows, trust=args.trust)
    val = load("val", keep_p, True, args.finetune_hog, 1, args.data, max_rows=args.max_rows,
               trust=args.trust)
    val_hog = load("val", keep_p, True, True, 1, args.data, max_rows=args.max_rows,
                   trust=args.trust)
    mb = (_nbytes(train) + _nbytes(val) + _nbytes(val_hog)) / 2 ** 20
    print(f"rows: train {len(train['act_card']):,}  val {len(val['act_card']):,}  "
          f"val(hog) {len(val_hog['act_card']):,}  ({mb:,.0f} MB, {time.time() - t0:.0f}s)",
          flush=True)
    if not len(train["act_card"]):
        raise SystemExit(f"no training rows in {args.data} (split 'train' after the filters)")

    meta = checkpoint_meta(vocab, args.d, args.layers, args.heads, rows=rows_rec)
    model = TokenPolicy(meta["n_cards"], meta["n_units"], d=args.d, layers=args.layers,
                        heads=args.heads).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train["act_card"]) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps,
                                                pct_start=one_cycle_pct(steps))
    out = Path(args.runs) / args.name
    out.mkdir(parents=True, exist_ok=True)
    record = run_record(args, device, wait_keep, keep_p, at_build, rows=rows_rec)
    (out / "args.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    best = -1.0
    step = 0
    rng = np.random.default_rng(args.seed)
    if args.phase_balance > 0:
        counts = np.bincount(phase_of(train), minlength=12)
        print("train rows per phase: " + ", ".join(f"{PHASE_NAMES[i]} {c:,}"
                                                   for i, c in enumerate(counts) if c), flush=True)
    with (out / "log.jsonl").open("a", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            order = balanced_order(train, args.phase_balance, rng) if args.phase_balance > 0 else None
            for b in batches(train, args.batch, True, device, order=order):
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss, s = losses(model, b, beta=args.beta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % 500 == 0:
                    print(f"  ep {epoch} step {step}/{steps} loss {loss.item():.3f} "
                          f"gate {s['gate_acc']:.3f} card {s['card_acc']:.3f} "
                          f"tile {s['tile_acc']:.3f} near {s['tile_near']:.3f} w {s['mean_w']:.2f}",
                          flush=True)
            v = evaluate(model, val, device, 0.0)
            vh = evaluate(model, val_hog, device, 0.0)
            # Per phase, so a weak phase is not hidden in the average.
            ph = phase_of(val_hog)
            per_phase = {}
            for i in range(12):
                m = ph == i
                if m.sum() >= 200:
                    sub = {c: arr[m] for c, arr in val_hog.items()}
                    e = evaluate(model, sub, device, 0.0)
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
                  f"near {vh['tile_near']:.3f}", flush=True)
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
