"""sim/train_tokens.py (E05, and the train_tokens part of E04).

* the loader keeps the pre-E05 rows, order and dtypes (golden, against a copy of
  the old loader) and --max-rows takes a seeded sample in row order;
* --wait-keep resolution against vocab.json's wait_keep_at_build;
* evaluate() is correct / eligible per statistic, not batch-size weighted;
* --device cuda without CUDA stops before anything is read;
* a tiny CPU run on two synthetic shards writes ep01.pt, ep02.pt, last.pt, best.pt,
  args.json with versions, and checkpoints whose meta carries vocab.json;
* --trust v2 reads trusted_v2 (and refuses rows without it), press rows (act_press > 0)
  are left out after the wait draw, and rows without these columns load as before;
  args.json and the meta record both.

Needs numpy and torch (both venvs have them); no engine, data or GPU, except the
one gpu-marked run.
"""
from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from sim import train_tokens as tt  # noqa: E402
from sim.model_tokens import TokenPolicy, target_slot  # noqa: E402

N_CARDS, N_UNITS, UNIT_SLOTS = 12, 10, 6
HAND = (2, 3, 4, 5)
NOT_IN_HAND = 9


# ---------------------------------------------------------------------------
# synthetic shards in tools/rs_rows.py's layout and dtypes
# ---------------------------------------------------------------------------

def shard_arrays(n: int, *, start: int = 0, seed: int = 0, splits=None, trusted=None, hog=None,
                 unit_dtype=np.float16, outcome_dtype=np.float64) -> dict:
    """One shard. Row i: a wait when i % 3 == 0, else a hand card, except a card not
    in the hand when i % 11 == 10. outcome = (start + i) / 100 identifies the row.
    Default flags are fixed patterns, so every filter keeps some play rows (plays are
    never thinned): split by i % 5 (0-2 train, 3 val, 4 test), trusted unless
    i % 7 == 6, Hog when i % 4 < 2. Only the features are random."""
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    act = np.where(i % 3 == 0, 0, np.asarray(HAND)[i % 4])
    act = np.where((i % 11 == 10) & (act > 0), NOT_IN_HAND, act).astype(np.int16)
    if splits is None:
        splits = np.asarray(["train", "train", "train", "val", "test"])[i % 5]
    if trusted is None:
        trusted = i % 7 != 6
    if hog is None:
        hog = i % 4 < 2
    return {
        "tick": (i * 20).astype(np.int64),
        "side": np.where(i % 2 == 0, 1, -1).astype(np.int64),
        "hand_exact": np.ones(n, dtype=bool),
        "unit_ids": rng.integers(0, N_UNITS, (n, UNIT_SLOTS)).astype(np.int16),
        "unit_feats": rng.random((n, UNIT_SLOTS, 7)).astype(unit_dtype),
        "towers": rng.random((n, 6)).astype(np.float32),
        "scalars": rng.random((n, 4)).astype(np.float32),
        "hand": np.tile(np.asarray(HAND, dtype=np.int16), (n, 1)),
        "next": np.full(n, 6, dtype=np.int64),
        "opp_seen": rng.integers(0, N_CARDS, (n, 8)).astype(np.int16),
        "opp_last": rng.integers(0, N_CARDS, (n, 4)).astype(np.int16),
        "opp_last_age": rng.random((n, 4)).astype(np.float16),
        "outcome": ((start + i) / 100).astype(outcome_dtype),
        "trusted": np.asarray(trusted, dtype=bool),
        "won": (i % 2).astype(np.float64),
        "hog": np.asarray(hog, dtype=bool),
        "act_card": act,
        "act_x": rng.integers(0, 18, n).astype(np.int16),
        "act_y": rng.integers(0, 32, n).astype(np.int16),
        "match": np.asarray([f"m{(start + k) // 10:05d}" for k in i]),
        "split": np.asarray(splits),
    }


def write_shard(folder: Path, k: int, arrays: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"shard-{k:04d}.npz"
    np.savez_compressed(path, **arrays)
    return path


def legacy_load(split, wait_keep, trusted_only, hog_only, seed, root):
    """sim/train_tokens.load before E05 (files closed, otherwise verbatim)."""
    rng = np.random.default_rng(seed)
    parts = {c: [] for c in tt.COLS}
    for path in sorted(root.glob("shard-*.npz")):
        with np.load(path) as z:
            keep = z["split"] == split
            if trusted_only:
                keep &= z["trusted"]
            if hog_only:
                keep &= z["hog"]
            waits = z["act_card"] == 0
            keep &= ~waits | (rng.random(len(waits)) < wait_keep)
            for c in tt.COLS:
                parts[c].append(z[c][keep])
    data = {c: np.concatenate(v) for c, v in parts.items()}
    data["unit_feats"] = data["unit_feats"].astype(np.float16)
    return data


@pytest.fixture
def mixed_rows(tmp_path):
    """Three shards: float16 / float32 unit_feats, float32 then float64 outcome (so the
    loader must widen as np.concatenate does), and one shard with only train rows."""
    root = tmp_path / "rows"
    write_shard(root, 0, shard_arrays(50, start=0, seed=1, outcome_dtype=np.float32))
    write_shard(root, 1, shard_arrays(40, start=100, seed=2, unit_dtype=np.float32))
    write_shard(root, 2, shard_arrays(30, start=200, seed=3, splits=["train"] * 30))
    return root


def assert_same(new: dict, old: dict):
    assert list(new) == list(old) == list(tt.COLS)
    for c in tt.COLS:
        assert new[c].dtype == old[c].dtype, c
        assert new[c].shape == old[c].shape, c
        assert np.array_equal(new[c], old[c]), c


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split, wait_keep, trusted_only, hog_only, seed", [
    ("train", 0.35, True, False, 0),          # the train load
    ("val", 0.35, True, False, 1),            # val
    ("val", 0.35, True, True, 1),             # val (hog)
    ("train", 1.0, False, False, 5),          # --all-rows, every wait
    ("test", 0.5, False, True, 2),
])
def test_load_matches_the_old_loader(mixed_rows, split, wait_keep, trusted_only, hog_only, seed):
    new = tt.load(split, wait_keep, trusted_only, hog_only, seed, mixed_rows)
    old = legacy_load(split, wait_keep, trusted_only, hog_only, seed, mixed_rows)
    assert_same(new, old)
    assert new["unit_feats"].dtype == np.float16
    assert new["outcome"].dtype == np.float64            # widened from shard 0's float32
    assert len(new["act_card"]) > 0


def test_load_without_shards_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no shard"):
        tt.load("train", 0.35, True, False, 0, tmp_path)


def test_max_rows_is_a_seeded_sample_in_row_order(mixed_rows):
    full = tt.load("train", 1.0, False, False, 0, mixed_rows)
    n = len(full["outcome"])
    assert n > 10
    capped = tt.load("train", 1.0, False, False, 0, mixed_rows, max_rows=10)
    assert len(capped["outcome"]) == 10
    assert np.all(np.diff(capped["outcome"]) > 0)        # row order kept
    idx = np.searchsorted(full["outcome"], capped["outcome"])
    for c in tt.COLS:                                     # whole rows, not mixed columns
        assert capped[c].dtype == full[c].dtype
        assert np.array_equal(capped[c], full[c][idx]), c
    again = tt.load("train", 1.0, False, False, 0, mixed_rows, max_rows=10)
    assert_same(again, capped)
    assert_same(tt.load("train", 1.0, False, False, 0, mixed_rows, max_rows=n), full)
    assert_same(tt.load("train", 1.0, False, False, 0, mixed_rows, max_rows=0), full)


def test_max_rows_does_not_change_which_waits_are_kept(mixed_rows):
    thinned = tt.load("train", 0.35, True, False, 0, mixed_rows)
    capped = tt.load("train", 0.35, True, False, 0, mixed_rows, max_rows=5)
    assert len(capped["outcome"]) == 5
    assert np.isin(capped["outcome"], thinned["outcome"]).all()


# ---------------------------------------------------------------------------
# --trust v2 and press rows (E09 / E10 rows in training)
# ---------------------------------------------------------------------------

#: Rows with i % 9 == 8 lose trusted_v2; waits with i % 9 == 0 are presses.
V2_CUT, PRESS_EVERY = 8, 9


def with_v2_and_presses(a: dict) -> dict:
    """A shard_arrays shard as tools/rs_rows.py --trust v2 --presses on writes it: trusted_v2
    is `trusted` less the rows with i % 9 == 8, and the waits with i % 9 == 0 are press rows
    (act_press 1 or 2, act_card 0; act_press 0 on every other row)."""
    i = np.arange(len(a["act_card"]))
    a = dict(a)
    a["trusted_v2"] = a["trusted"] & (i % PRESS_EVERY != V2_CUT)
    press = (a["act_card"] == 0) & (i % PRESS_EVERY == 0)
    a["act_press"] = np.where(press, 1 + (i // PRESS_EVERY) % 2, 0).astype(np.int16)
    return a


@pytest.fixture
def press_rows(tmp_path):
    """Two shards with trusted_v2 and act_press: 50 rows (presses at i = 0, 9, 18, 27, 36, 45)
    and 40 rows (0, 9, 18, 27, 36): 11 press rows, 7 of them in the train split (i % 5 < 3:
    0, 27, 36, 45 and 0, 27, 36)."""
    root = tmp_path / "press"
    write_shard(root, 0, with_v2_and_presses(shard_arrays(50, start=0, seed=21)))
    write_shard(root, 1, with_v2_and_presses(shard_arrays(40, start=100, seed=22)))
    return root


def outcomes_of_presses(root: Path) -> np.ndarray:
    """The outcome (row id) of every press row in `root`."""
    out = []
    for path in sorted(root.glob("shard-*.npz")):
        with np.load(path) as z:
            out.append(z["outcome"][z["act_press"] > 0])
    return np.concatenate(out)


def without(data: dict, outcomes: np.ndarray) -> dict:
    keep = ~np.isin(data["outcome"], outcomes)
    return {c: data[c][keep] for c in tt.COLS}


LOADS = [
    ("train", 0.35, True, False, 0),
    ("val", 0.35, True, True, 1),
    ("train", 1.0, False, False, 5),          # --all-rows, every wait
    ("test", 0.5, True, False, 2),
]


def test_the_fixture_holds_what_its_doc_says(press_rows):
    presses = outcomes_of_presses(press_rows)
    assert sorted(np.round(presses * 100).astype(int).tolist()) == [
        0, 9, 18, 27, 36, 45, 100, 109, 118, 127, 136]
    for path in sorted(press_rows.glob("shard-*.npz")):
        with np.load(path) as z:
            assert (z["act_card"][z["act_press"] > 0] == 0).all()     # presses look like waits
            assert z["act_press"].dtype == np.int16
            assert (z["trusted_v2"] <= z["trusted"]).all() and (z["trusted_v2"] < z["trusted"]).any()


@pytest.mark.parametrize("split, wait_keep, trusted_only, hog_only, seed", LOADS)
def test_press_rows_are_left_out_after_the_same_wait_draws(press_rows, split, wait_keep,
                                                          trusted_only, hog_only, seed):
    # The old loader ignores act_press: its rows less the press rows are what the loader keeps,
    # so the press rows still take their rng draw and every other wait is kept as before.
    new = tt.load(split, wait_keep, trusted_only, hog_only, seed, press_rows)
    old = legacy_load(split, wait_keep, trusted_only, hog_only, seed, press_rows)
    assert_same(new, without(old, outcomes_of_presses(press_rows)))


def test_press_rows_left_out_by_hand(press_rows):
    every = tt.load("train", 1.0, False, False, 0, press_rows)
    old = legacy_load("train", 1.0, False, False, 0, press_rows)
    assert (len(old["outcome"]), len(every["outcome"])) == (30 + 24, 30 + 24 - 7)
    gone = np.setdiff1d(old["outcome"], every["outcome"])
    assert np.round(gone * 100).astype(int).tolist() == [0, 27, 36, 45, 100, 127, 136]


def test_rows_with_no_presses_load_as_before(tmp_path):
    root = tmp_path / "rows"
    for k, n in ((0, 30), (1, 25)):
        a = shard_arrays(n, start=100 * k, seed=30 + k)
        a["act_press"] = np.zeros(n, dtype=np.int16)          # presses on, none replayed
        write_shard(root, k, a)
    for args in LOADS:
        assert_same(tt.load(*args, root), legacy_load(*args, root))


@pytest.mark.parametrize("split, wait_keep, trusted_only, hog_only, seed", LOADS)
def test_trust_v2_reads_trusted_v2(press_rows, tmp_path, split, wait_keep, trusted_only,
                                   hog_only, seed):
    # The same shards with trusted_v2 moved into `trusted`: the legacy trust on them is what
    # --trust v2 gives on the originals.
    swapped = tmp_path / "swapped"
    for k, path in enumerate(sorted(press_rows.glob("shard-*.npz"))):
        with np.load(path) as z:
            a = {c: z[c] for c in z.files}
        a["trusted"] = a.pop("trusted_v2")
        write_shard(swapped, k, a)
    v2 = tt.load(split, wait_keep, trusted_only, hog_only, seed, press_rows, trust="v2")
    assert_same(v2, tt.load(split, wait_keep, trusted_only, hog_only, seed, swapped))
    assert_same(v2, without(legacy_load(split, wait_keep, trusted_only, hog_only, seed, swapped),
                            outcomes_of_presses(press_rows)))
    legacy = tt.load(split, wait_keep, trusted_only, hog_only, seed, press_rows)
    if trusted_only:      # each of these loads keeps a play that trusted_v2 cuts (i % 9 == 8)
        assert len(v2["outcome"]) < len(legacy["outcome"])
        assert np.isin(v2["outcome"], legacy["outcome"]).all()
    else:                 # no trust filter: the trust column does not matter
        assert_same(v2, legacy)


def test_legacy_trust_ignores_trusted_v2(press_rows):
    for args in LOADS:
        assert_same(tt.load(*args, press_rows, trust="legacy"), tt.load(*args, press_rows))


def test_trust_v2_without_the_column_is_refused(mixed_rows):
    with pytest.raises(ValueError, match="no trusted_v2 column"):
        tt.load("train", 0.35, True, False, 0, mixed_rows, trust="v2")
    with pytest.raises(ValueError, match="no trusted_v2 column"):
        tt.load("val", 0.35, True, True, 1, mixed_rows, trust="v2")
    # --all-rows' train load filters on no trust column, so it needs none
    assert_same(tt.load("train", 1.0, False, False, 0, mixed_rows, trust="v2"),
                tt.load("train", 1.0, False, False, 0, mixed_rows))
    with pytest.raises(SystemExit) as exc:
        tt.row_record(mixed_rows, "v2")
    msg = str(exc.value)
    assert "--trust v2: 3 of 3 shard(s)" in msg and "shard-0000.npz" in msg
    assert "rs_rows.py --trust v2" in msg and "--trust legacy" in msg


def test_trust_v2_names_the_shards_that_lack_the_column(press_rows, tmp_path):
    root = tmp_path / "half"
    with np.load(sorted(press_rows.glob("shard-*.npz"))[0]) as z:
        write_shard(root, 0, {c: z[c] for c in z.files})
    write_shard(root, 1, shard_arrays(20, start=300, seed=4))         # built before E09
    with pytest.raises(SystemExit) as exc:
        tt.row_record(root, "v2")
    assert "1 of 2 shard(s)" in str(exc.value) and "shard-0001.npz" in str(exc.value)
    with pytest.raises(ValueError, match="shard-0001.npz"):
        tt.load("train", 0.35, True, False, 0, root, trust="v2")


def test_unknown_trust_is_refused(mixed_rows):
    assert tt.trust_column("legacy") == "trusted" and tt.trust_column("v2") == "trusted_v2"
    assert tt.TRUST_MODES == ("legacy", "v2")
    for bad in ("v3", None, ""):
        with pytest.raises(ValueError, match="--trust"):
            tt.trust_column(bad)
    with pytest.raises(ValueError, match="--trust"):
        tt.load("train", 0.35, True, False, 0, mixed_rows, trust="v3")


def test_row_record(mixed_rows, press_rows, tmp_path):
    legacy = {"trust": "legacy", "trust_column": "trusted", "press_rows": "none",
              "press_rows_excluded": 0}
    assert tt.row_record(mixed_rows) == tt.row_record(mixed_rows, "legacy") == legacy
    assert tt.legacy_rows(legacy) and tt.legacy_rows(None)
    assert tt.row_record(tmp_path / "empty") == legacy          # load says there are no shards
    got = tt.row_record(press_rows)
    assert got == {"trust": "legacy", "trust_column": "trusted", "press_rows": "excluded",
                   "press_rows_excluded": 11}                   # every split
    assert not tt.legacy_rows(got)
    v2 = tt.row_record(press_rows, "v2")
    assert v2 == dict(got, trust="v2", trust_column="trusted_v2")
    assert not tt.legacy_rows(v2)
    assert type(v2["press_rows_excluded"]) is int
    assert "11 rows with act_press > 0" in tt.describe_rows(v2)
    assert "press" not in tt.describe_rows(legacy)


# ---------------------------------------------------------------------------
# --wait-keep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cli, vocab, expected", [
    (None, {}, (0.35, 0.35, None)),                                   # rows before E18
    (0.25, {}, (0.25, 0.25, None)),
    (None, {"wait_keep_at_build": None}, (0.35, 0.35, None)),
    (None, {"wait_keep_at_build": 0.25}, (0.25, 1.0, 0.25)),          # as built
    (0.25, {"wait_keep_at_build": 0.5}, (0.25, 0.5, 0.5)),            # thin further
    (1.0, {"wait_keep_at_build": 1.0}, (1.0, 1.0, 1.0)),
])
def test_resolve_wait_keep(cli, vocab, expected):
    assert tt.resolve_wait_keep(cli, vocab) == expected


def test_wait_keep_above_the_build_share_trains_on_what_there_is(capsys):
    assert tt.resolve_wait_keep(0.5, {"wait_keep_at_build": 0.25}) == (0.25, 1.0, 0.25)
    assert "warning" in capsys.readouterr().out


@pytest.mark.parametrize("cli, vocab", [
    (0.0, {}), (-0.1, {}), (1.5, {}),
    (None, {"wait_keep_at_build": 0.0}), (None, {"wait_keep_at_build": 2.0}),
])
def test_resolve_wait_keep_rejects_shares_outside_0_1(cli, vocab):
    with pytest.raises(SystemExit):
        tt.resolve_wait_keep(cli, vocab)


# ---------------------------------------------------------------------------
# evaluate(): correct / eligible, summed over batches
# ---------------------------------------------------------------------------

#: (act_card, gate ok, card ok, tile ok, near ok) per row; hand is HAND for every row.
#: Batches of 4: rows 0-3 (one play), 4-7 (four plays, row 7 not in hand), 8-9.
EVAL_ROWS = (
    (0, 1, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 0, 0, 0), (2, 1, 1, 0, 1),
    (3, 0, 0, 1, 1), (4, 1, 0, 0, 0), (5, 1, 1, 1, 1), (NOT_IN_HAND, 1, 1, 1, 1),
    (0, 0, 0, 0, 0), (2, 0, 0, 0, 0),
)


def _eval_data():
    a = np.asarray(EVAL_ROWS, dtype=np.int16)
    return {"hand": np.tile(np.asarray(HAND, dtype=np.int16), (len(a), 1)),
            "act_card": a[:, 0], "g": a[:, 1], "c": a[:, 2], "t": a[:, 3], "n": a[:, 4]}


def _fake_losses(model, b, beta=0.0):
    """model_tokens.losses' statistics, with the 'correct' flags read from the rows."""
    card = b["act_card"].long()
    ok = target_slot(b["hand"], card) >= 0
    plays = card > 0

    def mean(flag, mask):
        return b[flag][mask].float().mean().item() if mask.any() else 0.0

    return torch.zeros(()), {"gate_acc": mean("g", ok), "card_acc": mean("c", ok & plays),
                             "tile_acc": mean("t", plays), "tile_near": mean("n", plays),
                             "mean_w": 1.0}


class _Mode:
    def __init__(self):
        self.calls = []

    def eval(self):
        self.calls.append("eval")

    def train(self):
        self.calls.append("train")


def test_evaluate_is_correct_over_eligible_rows(monkeypatch):
    monkeypatch.setattr(tt, "losses", _fake_losses)
    model = _Mode()
    got = tt.evaluate(model, _eval_data(), "cpu", 0.0, batch_size=4)
    # gate: the 9 rows whose card is a wait or in hand (not row 7), 5 right
    # card: plays among them (3, 4, 5, 6, 9), 2 right; tile / near: all 6 plays
    assert got["gate_acc"] == pytest.approx(5 / 9, rel=1e-12)
    assert got["card_acc"] == pytest.approx(2 / 5, rel=1e-12)
    assert got["tile_acc"] == pytest.approx(3 / 6, rel=1e-12)
    assert got["tile_near"] == pytest.approx(4 / 6, rel=1e-12)
    assert got["mean_w"] == pytest.approx(1.0)
    assert (got["rows"], got["plays"]) == (10, 6)
    assert model.calls == ["eval", "train"]
    # The old batch-size weighting gives other numbers on these batches, so the
    # test tells the two apart: card (4*1 + 4*1/3 + 2*0) / 10, tile (4*0 + 4*3/4) / 10.
    legacy_card, legacy_tile = (4 * 1 + 4 / 3) / 10, (4 * 3 / 4) / 10
    assert abs(legacy_card - got["card_acc"]) > 0.1
    assert abs(legacy_tile - got["tile_acc"]) > 0.1


def test_evaluate_does_not_depend_on_the_batch_size(monkeypatch):
    monkeypatch.setattr(tt, "losses", _fake_losses)
    data = _eval_data()
    one = tt.evaluate(_Mode(), data, "cpu", 0.0, batch_size=len(EVAL_ROWS))
    for size in (1, 3, 4, 7):
        got = tt.evaluate(_Mode(), data, "cpu", 0.0, batch_size=size)
        for k in tt.STATS:
            assert got[k] == pytest.approx(one[k], rel=1e-12), (size, k)


def test_evaluate_of_no_rows_is_zeros(monkeypatch):
    monkeypatch.setattr(tt, "losses", _fake_losses)
    empty = {k: v[:0] for k, v in _eval_data().items()}
    got = tt.evaluate(_Mode(), empty, "cpu", 0.0)
    assert {k: got[k] for k in tt.STATS} == {k: 0.0 for k in tt.STATS}
    assert (got["rows"], got["plays"]) == (0, 0)


def test_eligible_counts():
    b = {k: torch.from_numpy(v).long() for k, v in _eval_data().items()}
    assert tt.eligible_counts(b) == {"rows": 10, "gate": 9, "card": 5, "play": 6}


# ---------------------------------------------------------------------------
# device, provenance, meta
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extra", [[], ["--device", "cuda"]])
def test_cuda_requested_without_cuda_exits_first(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        tt.main(["--name", "x", "--data", str(tmp_path / "missing"),
                 "--runs", str(tmp_path / "runs"), *extra])
    assert "CUDA is not available" in str(exc.value)
    assert not (tmp_path / "runs").exists()


def test_cpu_device_never_needs_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert tt.resolve_device("cpu") == "cpu"


@pytest.mark.gpu
def test_cuda_device_when_cuda_works():
    assert tt.resolve_device("cuda") == "cuda"


def test_missing_vocab_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        tt.main(["--name", "x", "--device", "cpu", "--data", str(tmp_path),
                 "--runs", str(tmp_path / "runs")])
    assert "no vocab.json" in str(exc.value)


@pytest.mark.parametrize("steps", [1, 7, 14, 19, 20, 21, 40, 1000])
def test_one_cycle_schedule_builds_for_any_step_count(steps):
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.1)   # as main()
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=steps,
                                                pct_start=tt.one_cycle_pct(steps))
    for _ in range(steps):
        opt.step()
        sched.step()
    assert tt.one_cycle_pct(14) == 0.05 and tt.one_cycle_pct(1000) == 0.05
    assert tt.one_cycle_pct(20) != 0.05


def test_default_data_is_rs_rows():
    assert tt.DATA == tt.ROOT / "data" / "rs_rows"


def _completed(cmd, code, out):
    return subprocess.CompletedProcess(cmd, code, stdout=out, stderr="")


def test_git_commit(monkeypatch):
    def ok(cmd, **kw):
        return _completed(cmd, 0, "abc123\n" if "rev-parse" in cmd else " M sim/train_tokens.py\n")
    monkeypatch.setattr(tt.subprocess, "run", ok)
    assert tt.git_commit(Path(".")) == ("abc123", True)

    def clean(cmd, **kw):
        return _completed(cmd, 0, "abc123\n" if "rev-parse" in cmd else "")
    monkeypatch.setattr(tt.subprocess, "run", clean)
    assert tt.git_commit(Path(".")) == ("abc123", False)

    monkeypatch.setattr(tt.subprocess, "run", lambda cmd, **kw: _completed(cmd, 128, ""))
    assert tt.git_commit(Path(".")) == (None, None)

    def missing(cmd, **kw):
        raise FileNotFoundError("git")
    monkeypatch.setattr(tt.subprocess, "run", missing)
    assert tt.git_commit(Path(".")) == (None, None)


def test_run_record_keeps_args_as_strings_and_adds_versions(monkeypatch):
    monkeypatch.setattr(tt, "git_commit", lambda root=tt.ROOT: ("abc123", False))
    args = Namespace(name="rs_bc", wait_keep=0.25, data=Path("data/rs_rows"), init=None,
                     epochs=6, all_rows=False)
    rec = tt.run_record(args, "cpu", 0.25, 1.0, 0.25)
    for k in ("name", "data", "init", "epochs", "all_rows"):
        assert rec[k] == str(getattr(args, k))
    assert rec["wait_keep"] == "0.25"          # what RSAgent reads: float(args["wait_keep"])
    assert float(rec["wait_keep"]) == 0.25
    assert rec["wait_keep_train"] == 1.0 and rec["wait_keep_at_build"] == 0.25
    assert rec["torch_version"] == torch.__version__
    assert rec["torch_cuda"] == torch.version.cuda
    assert rec["numpy_version"] == np.__version__
    assert (rec["git_commit"], rec["git_dirty"]) == ("abc123", False)
    assert rec["device_name"] == "cpu"
    json.dumps(rec)                             # JSON as it is
    # Exact builtin types: the record goes into every checkpoint, and torch.load's
    # weights_only unpickler refuses torch.__version__'s own class (TorchVersion).
    for k, v in rec.items():
        assert type(v) in (str, int, float, bool, type(None)), (k, type(v))


def test_checkpoint_meta_is_vocab_plus_architecture():
    # Synthetic sizes: the real ones come from the engine's catalogue / unit_types_json.
    legacy = {"n_cards": N_CARDS, "n_units": N_UNITS, "live_compat": False, "cards": {"Cannon": 2},
              "units": {"KingTower": 2}, "engine": "royalesim", "version": "0.1.42"}
    assert tt.checkpoint_meta(legacy, 192, 4, 6) == dict(legacy, d=192, layers=4, heads=6)
    v2 = dict(legacy, fingerprint={"card_table": "cards-15535.1", "unit_types_digest": "ab"},
              schema="rs_rows/v2", obs_delay={"ticks": 24, "jitter": 3}, wait_keep_at_build=0.25)
    meta = tt.checkpoint_meta(v2, 256, 6, 8)
    for k, v in v2.items():
        assert meta[k] == v
    assert (meta["d"], meta["layers"], meta["heads"]) == (256, 6, 8)


ROWS_LEGACY = {"trust": "legacy", "trust_column": "trusted", "press_rows": "none",
               "press_rows_excluded": 0}
ROWS_V2 = {"trust": "v2", "trust_column": "trusted_v2", "press_rows": "none",
           "press_rows_excluded": 0}
ROWS_PRESSES = dict(ROWS_LEGACY, press_rows="excluded", press_rows_excluded=11)


def test_checkpoint_meta_records_the_rows_unless_they_are_legacy():
    vocab = {"n_cards": N_CARDS, "n_units": N_UNITS, "live_compat": True, "cards": {"Cannon": 2},
             "trust": {"mode": "v2", "sides": "unordered"}}          # rs_rows' rule stays as it is
    plain = dict(vocab, d=192, layers=4, heads=6)
    assert tt.checkpoint_meta(vocab, 192, 4, 6, rows=ROWS_LEGACY) == plain
    assert tt.checkpoint_meta(vocab, 192, 4, 6, rows=None) == plain
    for rows in (ROWS_V2, ROWS_PRESSES, dict(ROWS_V2, press_rows="excluded")):
        meta = tt.checkpoint_meta(vocab, 192, 4, 6, rows=rows)
        assert meta == dict(plain, train_rows=rows)
        assert meta["trust"] == vocab["trust"]
        assert all(type(v) in (str, int) for v in meta["train_rows"].values())
    assert "train_rows" not in vocab                                 # the vocab is not changed


def test_run_record_has_the_rows(monkeypatch):
    monkeypatch.setattr(tt, "git_commit", lambda root=tt.ROOT: ("abc123", False))
    args = Namespace(name="dt_big", wait_keep=0.25, trust="v2", all_rows=False)
    rec = tt.run_record(args, "cpu", 0.25, 0.25, None, rows=dict(ROWS_V2, press_rows="excluded",
                                                                 press_rows_excluded=3))
    assert (rec["trust"], rec["trust_column"]) == ("v2", "trusted_v2")
    assert (rec["press_rows"], rec["press_rows_excluded"]) == ("excluded", 3)
    assert rec["name"] == "dt_big" and rec["wait_keep"] == "0.25"
    for k, v in rec.items():
        assert type(v) in (str, int, float, bool, type(None)), (k, type(v))
    old = tt.run_record(Namespace(name="x", wait_keep=0.35), "cpu", 0.35, 0.35, None)
    assert not {"trust", "trust_column", "press_rows", "press_rows_excluded"} & set(old)


# ---------------------------------------------------------------------------
# a tiny run end to end
# ---------------------------------------------------------------------------

VOCAB = {
    "n_cards": N_CARDS, "n_units": N_UNITS, "live_compat": False,
    "cards": {f"Card{i}": i + 2 for i in range(N_CARDS - 2)},
    "units": {f"Unit{i}": i + 2 for i in range(N_UNITS - 2)},
    "engine": "royalesim", "version": "0.0.0+test",
    "fingerprint": {"version": "0.0.0+test", "card_table": "cards-test.1",
                    "unit_types_digest": "0123456789abcdef"},
    "schema": "rs_rows/test", "obs_delay": {"ticks": 0, "jitter": 0},
    "wait_keep_at_build": 0.5,
}


@pytest.fixture
def smoke_rows(tmp_path):
    """Two shards of 40 rows: 0-29 train (rows 0 and 1 untrusted), 30-39 val; even rows
    Hog. Per shard the val rows hold 4 waits (30, 33, 36, 39) and 6 plays; the Hog val
    rows are 30, 32, 34, 36, 38, three of them plays. wait_keep_at_build 0.5 makes the
    default wait_keep 0.5, kept at load with p 1, so every row passing the filters is used."""
    root = tmp_path / "rows"
    i = np.arange(40)
    for k in (0, 1):
        write_shard(root, k, shard_arrays(
            40, start=100 * k, seed=10 + k, splits=["train"] * 30 + ["val"] * 10,
            trusted=i >= 2, hog=i % 2 == 0))
    (root / "vocab.json").write_text(json.dumps(VOCAB), encoding="utf-8")
    return root


def _run(rows, runs, *extra):
    return tt.main(["--name", "smoke", "--data", str(rows), "--runs", str(runs), "--d", "16",
                    "--layers", "1", "--heads", "2", "--batch", "8", *extra])


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_tiny_cpu_run_writes_every_epoch_and_versions(smoke_rows, tmp_path):
    runs = tmp_path / "runs"
    assert _run(smoke_rows, runs, "--device", "cpu", "--epochs", "2") == 0
    out = runs / "smoke"
    for name in ("ep01.pt", "ep02.pt", "last.pt", "best.pt", "args.json", "log.jsonl"):
        assert (out / name).is_file(), name
    assert not (out / "ep03.pt").exists()

    args = json.loads((out / "args.json").read_text(encoding="utf-8"))
    assert args["torch_version"] == torch.__version__
    assert args["torch_cuda"] == torch.version.cuda
    assert "git_commit" in args and (args["git_commit"] is None or len(args["git_commit"]) >= 7)
    assert args["device"] == "cpu" and args["device_name"] == "cpu"
    assert args["wait_keep"] == "0.5"                    # vocab's wait_keep_at_build
    assert (args["wait_keep_train"], args["wait_keep_at_build"]) == (1.0, 0.5)
    assert args["max_rows"] == "0" and args["epochs"] == "2"
    assert {k: args[k] for k in ROWS_LEGACY} == ROWS_LEGACY     # the default, recorded

    ep1, ep2, last, best = (_load(out / n) for n in ("ep01.pt", "ep02.pt", "last.pt", "best.pt"))
    assert (ep1["epoch"], ep2["epoch"], last["epoch"]) == (0, 1, 1)
    assert best["epoch"] in (0, 1)
    for k, v in ep2["model"].items():
        assert torch.equal(v, last["model"][k]), k
    for blob in (ep1, ep2, last, best):
        meta = blob["meta"]
        for k, v in VOCAB.items():                      # E04: vocab.json reaches the meta
            assert meta[k] == v, k
        assert (meta["d"], meta["layers"], meta["heads"]) == (16, 1, 2)
        assert meta == dict(VOCAB, d=16, layers=1, heads=2)     # no train_rows: legacy rows
        assert blob["train_tokens"] == args
    net = TokenPolicy(ep2["meta"]["n_cards"], ep2["meta"]["n_units"], d=16, layers=1, heads=2)
    net.load_state_dict(ep2["model"])

    log = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["epoch"] for r in log] == [0, 1]
    for r in log:
        assert (r["val"]["rows"], r["val"]["plays"]) == (20, 12)
        assert (r["val_hog"]["rows"], r["val_hog"]["plays"]) == (10, 6)
        for k in tt.STATS:
            assert 0.0 <= r["val"][k] <= 1.0 or k == "mean_w"


def test_tiny_run_with_max_rows(smoke_rows, tmp_path):
    runs = tmp_path / "runs"
    assert _run(smoke_rows, runs, "--device", "cpu", "--epochs", "1", "--max-rows", "8") == 0
    out = runs / "smoke"
    assert (out / "ep01.pt").is_file() and (out / "best.pt").is_file()
    args = json.loads((out / "args.json").read_text(encoding="utf-8"))
    assert args["max_rows"] == "8"
    (rec,) = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rec["val"]["rows"] == 8 and rec["val_hog"]["rows"] == 8


def test_negative_max_rows_is_refused(smoke_rows, tmp_path):
    with pytest.raises(SystemExit):
        _run(smoke_rows, tmp_path / "runs", "--device", "cpu", "--max-rows", "-1")


@pytest.fixture
def smoke_v2_rows(tmp_path):
    """smoke_rows as rs_rows --trust v2 --presses on would write them: trusted_v2 also cuts
    rows 37-39, and rows 3 (train) and 33 (val), both waits, are presses. Per shard the val
    rows then are, under --trust legacy, 30-39 less 33: 9 rows, plays 31, 32, 34, 35, 37, 38;
    Hog (even) 30, 32, 34, 36, 38, plays 32, 34, 38. Under --trust v2, 30-36 less 33: 6 rows,
    plays 31, 32, 34, 35; Hog 30, 32, 34, 36, plays 32, 34."""
    root = tmp_path / "rows_v2"
    i = np.arange(40)
    for k in (0, 1):
        a = shard_arrays(40, start=100 * k, seed=10 + k, splits=["train"] * 30 + ["val"] * 10,
                         trusted=i >= 2, hog=i % 2 == 0)
        assert (a["act_card"][[3, 33]] == 0).all()
        a["trusted_v2"] = (i >= 2) & (i < 37)
        a["act_press"] = np.where(np.isin(i, (3, 33)), 1, 0).astype(np.int16)
        write_shard(root, k, a)
    (root / "vocab.json").write_text(json.dumps(VOCAB), encoding="utf-8")
    return root


@pytest.mark.parametrize("trust, column, val, val_hog", [
    ("legacy", "trusted", (18, 12), (10, 6)),
    ("v2", "trusted_v2", (12, 8), (8, 4)),
])
def test_tiny_run_with_v2_trust_and_press_rows(smoke_v2_rows, tmp_path, capsys, trust, column,
                                              val, val_hog):
    runs = tmp_path / "runs"
    extra = [] if trust == "legacy" else ["--trust", trust]
    assert _run(smoke_v2_rows, runs, "--device", "cpu", "--epochs", "1", *extra) == 0
    assert f"rows: trust {trust} (column {column}); press rows left out: 4 rows" in capsys.readouterr().out
    out = runs / "smoke"
    args = json.loads((out / "args.json").read_text(encoding="utf-8"))
    rows = {"trust": trust, "trust_column": column, "press_rows": "excluded",
            "press_rows_excluded": 4}
    assert {k: args[k] for k in rows} == rows
    (rec,) = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (rec["val"]["rows"], rec["val"]["plays"]) == val
    assert (rec["val_hog"]["rows"], rec["val_hog"]["plays"]) == val_hog
    for name in ("ep01.pt", "best.pt"):
        blob = _load(out / name)
        assert blob["meta"] == dict(VOCAB, d=16, layers=1, heads=2, train_rows=rows)
        assert blob["train_tokens"] == args


def test_trust_v2_on_rows_without_the_column_stops_first(smoke_rows, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run(smoke_rows, tmp_path / "runs", "--device", "cpu", "--trust", "v2")
    assert "--trust v2: 2 of 2 shard(s)" in str(exc.value)
    assert "no trusted_v2 column" in str(exc.value)
    assert not (tmp_path / "runs").exists()                # before anything is written
    with pytest.raises(SystemExit):                        # argparse: not a --trust mode
        _run(smoke_rows, tmp_path / "runs", "--device", "cpu", "--trust", "v3")


@pytest.mark.gpu
def test_tiny_cuda_run(smoke_rows, tmp_path):
    runs = tmp_path / "runs"
    assert _run(smoke_rows, runs, "--device", "cuda", "--epochs", "1") == 0
    args = json.loads((runs / "smoke" / "args.json").read_text(encoding="utf-8"))
    assert args["device"] == "cuda" and args["torch_cuda"]
    assert (runs / "smoke" / "ep01.pt").is_file()
