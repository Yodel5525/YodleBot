"""sim/train_tokens.py --drop-pending-waits (the decision-time diagnosis, runbook section 19).

* off (the default), the keep mask is the one before the flag (a verbatim copy of it), with
  the same rng draws;
* on, it leaves out exactly the wait rows with an own play in flight (any own_pending slot > 0),
  after the same draws: play rows with a play in flight stay, every other row stays, and the
  count it reports is the rows it took away;
* load() and memmap_keep() (the compact store) keep the same rows with it, and refuse rows
  without own_pending or of schema v2; main() records it in args.json, trains on the training
  rows less those waits and leaves the validation sets as they were.

Needs numpy and torch (both venvs have them); no engine, data or GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from sim import train_tokens as tt  # noqa: E402
import rs_rows_compact as RC  # noqa: E402  (tools/ is on sys.path once train_tokens is imported)

N_CARDS, N_UNITS, UNIT_SLOTS = 12, 10, 6
HAND = (2, 3, 4, 5)
NOT_IN_HAND = 9
#: The waits with an own play in flight in a shard of 40 rows (shard_arrays): waits are
#: i % 3 == 0, a play is in flight in slot 0 when i % 4 == 0 and in slot 2 when i % 10 == 9.
PENDING_WAITS_40 = (0, 9, 12, 24, 36, 39)


def shard_arrays(n: int, *, start: int = 0, seed: int = 0, splits=None, trusted=None, hog=None,
                 pending: bool = True) -> dict:
    """One legacy shard as test_train_tokens.py's, plus (with `pending`) the decision-time
    columns: own_pending card 5 in slot 0 when i % 4 == 0 and card 7 in slot 2 when
    i % 10 == 9 (zeros elsewhere), own_pending_land 0.5 where a slot is filled. Row i: a wait
    when i % 3 == 0, else HAND[i % 4] (a card not in the hand when i % 11 == 10); split by
    i % 5 unless given; trusted unless i % 7 == 6; Hog when i % 4 < 2; outcome
    (start + i) / 100 names the row."""
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    act = np.where(i % 3 == 0, 0, np.asarray(HAND)[i % 4])
    act = np.where((i % 11 == 10) & (act > 0), NOT_IN_HAND, act).astype(np.int16)
    if splits is None:
        splits = np.asarray(["train", "train", "train", "val", "test"])[i % 5]
    out = {
        "tick": (i * 20).astype(np.int64),
        "side": np.where(i % 2 == 0, 1, -1).astype(np.int64),
        "hand_exact": np.ones(n, dtype=bool),
        "unit_ids": rng.integers(0, N_UNITS, (n, UNIT_SLOTS)).astype(np.int16),
        "unit_feats": rng.random((n, UNIT_SLOTS, 7)).astype(np.float16),
        "towers": rng.random((n, 6)).astype(np.float32),
        "scalars": rng.random((n, 4)).astype(np.float32),
        "hand": np.tile(np.asarray(HAND, dtype=np.int16), (n, 1)),
        "next": np.full(n, 6, dtype=np.int64),
        "opp_seen": rng.integers(0, N_CARDS, (n, 8)).astype(np.int16),
        "opp_last": rng.integers(0, N_CARDS, (n, 4)).astype(np.int16),
        "opp_last_age": rng.random((n, 4)).astype(np.float16),
        "outcome": ((start + i) / 100).astype(np.float64),
        "trusted": np.asarray(i % 7 != 6 if trusted is None else trusted, dtype=bool),
        "won": (i % 2).astype(np.float64),
        "hog": np.asarray(i % 4 < 2 if hog is None else hog, dtype=bool),
        "act_card": act,
        "act_x": rng.integers(0, 18, n).astype(np.int16),
        "act_y": rng.integers(0, 32, n).astype(np.int16),
        "match": np.asarray([f"m{(start + k) // 10:05d}" for k in i]),
        "split": np.asarray(splits),
    }
    if pending:
        own = np.zeros((n, 4), dtype=np.int16)
        own[:, 0] = np.where(i % 4 == 0, 5, 0)
        own[:, 2] = np.where(i % 10 == 9, 7, 0)
        out["own_pending"] = own
        out["own_pending_land"] = ((own > 0) * 0.5).astype(np.float16)
    return out


def write_shard(folder: Path, k: int, arrays: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"shard-{k:04d}.npz"
    np.savez_compressed(path, **arrays)
    return path


def old_keep_mask(z, split, wait_keep, trusted_only, hog_only, rng, trust_col="trusted"):
    """sim/train_tokens._keep_mask (legacy schema) before --drop-pending-waits, verbatim."""
    keep = z["split"] == split
    if trusted_only:
        keep &= z[trust_col]
    if hog_only:
        keep &= z["hog"]
    waits = z["act_card"] == 0
    keep &= ~waits | (rng.random(len(waits)) < wait_keep)
    if "act_press" in set(getattr(z, "files", z)):
        keep &= ~(z["act_press"] > 0)
    return keep


def names(data: dict) -> list:
    """The rows of a load() dict by name: round(outcome * 100)."""
    return np.round(np.asarray(data["outcome"], dtype=np.float64) * 100).astype(int).tolist()


def quiet(*_):
    pass


@pytest.fixture
def dt_rows(tmp_path):
    """Three decision-time shards (40, 40 and 25 rows) with own_pending."""
    root = tmp_path / "dt"
    write_shard(root, 0, shard_arrays(40, start=0, seed=1))
    write_shard(root, 1, shard_arrays(40, start=100, seed=2))
    write_shard(root, 2, shard_arrays(25, start=200, seed=3, splits=["train"] * 25))
    return root


@pytest.fixture
def plain_rows(tmp_path):
    """The same shards without the decision-time columns (rows at execution)."""
    root = tmp_path / "plain"
    write_shard(root, 0, shard_arrays(40, start=0, seed=1, pending=False))
    write_shard(root, 1, shard_arrays(40, start=100, seed=2, pending=False))
    return root


def test_the_fixture_holds_what_its_doc_says():
    z = shard_arrays(40)
    pend = tt.pending_waits(z)
    assert np.flatnonzero(pend).tolist() == list(PENDING_WAITS_40)
    assert (z["act_card"][pend] == 0).all()
    # plays with a play in flight exist, and are no pending waits
    with_play = (z["own_pending"] > 0).any(axis=1) & (z["act_card"] > 0)
    assert np.flatnonzero(with_play).tolist() == [4, 8, 16, 19, 20, 28, 29, 32]
    assert not (pend & with_play).any()


LOADS = [
    ("train", 0.35, True, False, 0),
    ("val", 0.35, True, False, 1),
    ("val", 0.35, True, True, 1),
    ("train", 1.0, False, False, 5),
    ("test", 0.5, False, True, 2),
]


@pytest.mark.parametrize("split, wait_keep, trusted_only, hog_only, seed", LOADS)
def test_off_is_the_mask_of_before(split, wait_keep, trusted_only, hog_only, seed):
    z = shard_arrays(40, seed=seed)
    a, b = np.random.default_rng(seed), np.random.default_rng(seed)
    new = tt._keep_mask(z, split, wait_keep, trusted_only, hog_only, a)
    old = old_keep_mask(z, split, wait_keep, trusted_only, hog_only, b)
    assert np.array_equal(new, old)
    assert a.random() == b.random()               # the same draws taken


@pytest.mark.parametrize("split, wait_keep, trusted_only, hog_only, seed", LOADS)
def test_on_leaves_out_only_the_pending_waits(split, wait_keep, trusted_only, hog_only, seed):
    z = shard_arrays(40, seed=seed)
    a, b = np.random.default_rng(seed), np.random.default_rng(seed)
    counts: dict = {}
    on = tt._keep_mask(z, split, wait_keep, trusted_only, hog_only, a, drop_pending=True,
                       counts=counts)
    off = old_keep_mask(z, split, wait_keep, trusted_only, hog_only, b)
    pend = tt.pending_waits(z)
    assert np.array_equal(on, off & ~pend)        # every other row as before
    plays = z["act_card"] > 0
    assert np.array_equal(on[plays], off[plays])  # plays untouched, in flight or not
    assert counts == {tt.PENDING_DROPPED_KEY: int((off & pend).sum())}
    assert a.random() == b.random()               # no extra draws: the waits kept are the same
    # counts add up over calls
    tt._keep_mask(z, split, wait_keep, trusted_only, hog_only, np.random.default_rng(seed),
                  drop_pending=True, counts=counts)
    assert counts[tt.PENDING_DROPPED_KEY] == 2 * int((off & pend).sum())


def test_on_without_counts_and_a_one_dimensional_column():
    z = shard_arrays(40)
    z1 = dict(z, own_pending=z["own_pending"][:, 0])      # a single slot reads the same way
    flat = tt._keep_mask(z1, "test", 1.0, False, False, np.random.default_rng(0),
                         drop_pending=True)
    keep = tt._keep_mask(z, "test", 1.0, False, False, np.random.default_rng(0),
                         drop_pending=True)
    # slot 2 alone (rows i % 10 == 9, both test rows) is a play in flight only in the full
    # column; slot 0's waits (24 is the test row among them) are left out by both
    assert np.flatnonzero(flat & ~keep).tolist() == [9, 39]
    assert not (keep & tt.pending_waits(z)).any()


@pytest.mark.parametrize("split, wait_keep, trusted_only, hog_only, seed", LOADS)
def test_load_with_the_flag_is_load_less_the_pending_waits(dt_rows, split, wait_keep,
                                                           trusted_only, hog_only, seed):
    before = tt.load(split, wait_keep, trusted_only, hog_only, seed, dt_rows)
    counts: dict = {}
    after = tt.load(split, wait_keep, trusted_only, hog_only, seed, dt_rows,
                    drop_pending=True, counts=counts)
    pending = {start + i for start in (0, 100) for i in PENDING_WAITS_40}
    pending |= {200 + i for i in PENDING_WAITS_40 if i < 25}
    gone = [r for r in names(before) if r in pending]
    assert names(after) == [r for r in names(before) if r not in pending]
    assert counts == {tt.PENDING_DROPPED_KEY: len(gone)}
    assert list(after) == list(tt.COLS)           # the legacy columns, own_pending not loaded
    for c in tt.COLS:
        assert after[c].dtype == before[c].dtype, c
    # off, explicitly: the rows of before
    again = tt.load(split, wait_keep, trusted_only, hog_only, seed, dt_rows, drop_pending=False)
    assert names(again) == names(before)


def test_load_by_hand(dt_rows):
    # every train row of every shard, waits all kept: exactly the pending train waits go
    every = tt.load("train", 1.0, False, False, 0, dt_rows)
    counts: dict = {}
    some = tt.load("train", 1.0, False, False, 0, dt_rows, drop_pending=True, counts=counts)
    gone = sorted(set(names(every)) - set(names(some)))
    # shards 0 and 1 split by i % 5 (train: 0-2), shard 2 is all train
    want = sorted([s + i for s in (0, 100) for i in PENDING_WAITS_40 if i % 5 < 3]
                  + [200 + i for i in PENDING_WAITS_40 if i < 25])
    assert gone == want == [0, 12, 36, 100, 112, 136, 200, 209, 212, 224]
    assert counts[tt.PENDING_DROPPED_KEY] == 10


def test_max_rows_samples_after_the_drop(dt_rows):
    counts: dict = {}
    capped = tt.load("train", 1.0, False, False, 0, dt_rows, max_rows=9, drop_pending=True,
                     counts=counts)
    assert len(capped["outcome"]) == 9
    assert counts[tt.PENDING_DROPPED_KEY] == 10        # counted before the cap
    pending = {0, 12, 36, 100, 112, 136, 200, 209, 212, 224}
    assert not pending & set(names(capped))


@pytest.mark.parametrize("split, wait_keep, trusted_only, hog_only, seed", LOADS)
def test_memmap_keeps_the_rows_load_keeps(dt_rows, tmp_path, split, wait_keep, trusted_only,
                                          hog_only, seed):
    RC.compact(dt_rows, tmp_path / "store", log=quiet)
    store = tt.open_store(tmp_path / "store")
    want_counts: dict = {}
    want = tt.load(split, wait_keep, trusted_only, hog_only, seed, dt_rows, drop_pending=True,
                   counts=want_counts)
    for chunk in (7, 1000):
        counts: dict = {}
        ids = tt.memmap_keep(store, split, wait_keep, trusted_only, hog_only, seed, chunk=chunk,
                             drop_pending=True, counts=counts)
        got = tt.gather(store, tt.COLS, ids, window=5)
        assert list(got) == list(want)
        for c in tt.COLS:
            assert got[c].shape == want[c].shape and np.array_equal(got[c], want[c]), c
        assert counts == want_counts
    # off: the ids of before
    plain = tt.memmap_keep(store, split, wait_keep, trusted_only, hog_only, seed, chunk=7)
    assert names(tt.gather(store, tt.COLS, plain)) == names(
        tt.load(split, wait_keep, trusted_only, hog_only, seed, dt_rows))
    store.close()


def test_refusals(dt_rows, plain_rows, tmp_path):
    with pytest.raises(ValueError, match="shard-0000.npz in .* has no own_pending column"):
        tt.load("train", 1.0, False, False, 0, plain_rows, drop_pending=True)
    tt.load("train", 1.0, False, False, 0, plain_rows)           # off: loads as before
    with pytest.raises(SystemExit, match=r"--drop-pending-waits: 2 of 2 shard\(s\) .* no "
                                         r"own_pending column"):
        tt.check_drop_pending(plain_rows)
    assert tt.check_drop_pending(dt_rows) is None
    with pytest.raises(SystemExit, match="rs_rows/v2"):
        tt.check_drop_pending(dt_rows, schema="v2")
    with pytest.raises(ValueError, match="legacy"):
        tt.load("train", 1.0, False, False, 0, dt_rows, schema="v2", drop_pending=True)
    z = dict(shard_arrays(10), act_press=np.zeros(10, dtype=np.int16))
    with pytest.raises(ValueError, match="v2 model reads own_pending"):
        tt._keep_mask(z, "train", 1.0, False, False, np.random.default_rng(0),
                      presses=tt.PRESS_TRAINED, drop_pending=True)
    RC.compact(plain_rows, tmp_path / "store", log=quiet)
    store = tt.open_store(tmp_path / "store")
    with pytest.raises(ValueError, match="no own_pending column"):
        tt.memmap_keep(store, "train", 1.0, False, False, 0, drop_pending=True)
    with pytest.raises(SystemExit, match=r"2 of 2 shard\(s\)"):
        tt.check_drop_pending(store=store)
    store.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

VOCAB = {
    "n_cards": N_CARDS, "n_units": N_UNITS, "live_compat": True,
    "cards": {f"Card{i}": i + 2 for i in range(N_CARDS - 2)},
    "units": {f"Unit{i}": i + 2 for i in range(N_UNITS - 2)},
    "engine": "royalesim", "version": "0.0.0+test",
    "schema": "rs_rows/legacy", "obs_delay": {"ticks": 24, "jitter": 3, "opp_lag": 1},
    "wait_keep_at_build": 0.5,
}


def smoke(root: Path, pending: bool = True) -> Path:
    """Two shards of 40: rows 0-29 train (0 and 1 untrusted), 30-39 val, even rows Hog;
    wait_keep_at_build 0.5, so every wait passing the filters is kept at load. Train: 28 rows
    a shard; the trusted pending train waits are 9, 12 and 24 of each."""
    i = np.arange(40)
    for k in (0, 1):
        write_shard(root, k, shard_arrays(40, start=100 * k, seed=10 + k,
                                          splits=["train"] * 30 + ["val"] * 10,
                                          trusted=i >= 2, hog=i % 2 == 0, pending=pending))
    (root / "vocab.json").write_text(json.dumps(VOCAB), encoding="utf-8")
    return root


def _run(rows, runs, *extra):
    return tt.main(["--name", "smoke", "--data", str(rows), "--runs", str(runs), "--d", "16",
                    "--layers", "1", "--heads", "2", "--batch", "8", "--device", "cpu",
                    "--epochs", "1", *extra])


def _log(out):
    return [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8")
            .splitlines()]


def test_main_trains_without_the_pending_waits(tmp_path, capsys):
    rows = smoke(tmp_path / "rows")
    assert _run(rows, tmp_path / "off") == 0
    text = capsys.readouterr().out
    assert "rows: train 56  val 20  val(hog) 10" in text and "--drop-pending-waits" not in text
    off = json.loads((tmp_path / "off" / "smoke" / "args.json").read_text(encoding="utf-8"))
    assert off["drop_pending_waits"] == "False" and tt.PENDING_DROPPED_KEY not in off

    assert _run(rows, tmp_path / "on", "--drop-pending-waits") == 0
    text = capsys.readouterr().out
    assert "rows: train 50  val 20  val(hog) 10" in text           # 3 waits a shard left out
    assert "--drop-pending-waits: 6 wait rows with an own play in flight left out" in text
    out = tmp_path / "on" / "smoke"
    on = json.loads((out / "args.json").read_text(encoding="utf-8"))
    assert on["drop_pending_waits"] == "True" and on[tt.PENDING_DROPPED_KEY] == 6
    (rec,) = _log(out)
    assert (rec["val"]["rows"], rec["val_hog"]["rows"]) == (20, 10)   # validation as it was
    blob = tt.torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert blob["train_tokens"] == on
    assert blob["meta"] == dict(VOCAB, d=16, layers=1, heads=2)       # the meta is not changed


def test_main_with_the_flag_on_a_compact_store(tmp_path, capsys):
    rows = smoke(tmp_path / "rows")
    RC.compact(rows, log=quiet)
    assert _run(rows / "compact", tmp_path / "runs", "--memmap", "--drop-pending-waits") == 0
    text = capsys.readouterr().out
    assert "rows: train 50  val 20  val(hog) 10" in text
    args = json.loads((tmp_path / "runs" / "smoke" / "args.json").read_text(encoding="utf-8"))
    assert args[tt.PENDING_DROPPED_KEY] == 6


def test_main_refuses_rows_without_own_pending_before_any_output(tmp_path):
    rows = smoke(tmp_path / "rows", pending=False)
    runs = tmp_path / "runs"
    with pytest.raises(SystemExit, match=r"--drop-pending-waits: 2 of 2 shard\(s\)"):
        _run(rows, runs, "--drop-pending-waits")
    assert not runs.exists()
    assert _run(rows, runs) == 0                                       # off: trains as before
