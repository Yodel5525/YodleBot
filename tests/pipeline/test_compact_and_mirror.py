"""E18: the full-corpus data path.

tools/rs_rows.py: the seeded random pick of the non-Hog matches and --all-matches (select_matches;
"first", the default, is main's rule before E18), the two-pass corpus read (scan_part,
selected_payloads, chunked, bounded) and --wait-keep-at-build (thin_waits, seeded by the match
tag; build, manifest.json and vocab.json). tools/rs_rows_compact.py: npz shards -> per-column .npy
files with compact dtypes and index arrays, and back, exactly. sim/train_tokens.py: --memmap picks
exactly load()'s rows (memmap_keep + gather), the epoch samplers cover every row once
(epoch_order, window_batches, resident_batches, TrainFeed), the RAM and GPU budgets refuse what
does not fit, and --mirror flips a row left-right (mirror twice is the identity; towers swap left
and right on both sides; own_pending does not change).

The rs_rows tests use conftest's fake royalesim (no engine, no corpus); the store and mirror tests
need numpy (and torch for sim/train_tokens.py); one test needs CUDA (gpu marker).
"""
from __future__ import annotations

import ast
import importlib
import json
import math
import sys
from concurrent.futures import Future
from pathlib import Path

import pytest

from tests.pipeline.conftest import il_payload, il_replay_row, make_fake_royalesim, player_dict, \
    state_dict, HOG26_KEYS, OPPONENT_KEYS

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.append(str(TOOLS))             # as sim/model_tokens.py does
import rs_rows_compact as RC  # noqa: E402
import rs_schema as RS  # noqa: E402

N_CARDS, N_UNITS, UNIT_SLOTS = 12, 10, 6
HAND = (2, 3, 4, 5)
NOT_IN_HAND = 9
B = 3


# ---------------------------------------------------------------------------
# fixtures: tools/rs_rows.py on the fake engine
# ---------------------------------------------------------------------------

@pytest.fixture
def rr(monkeypatch):
    """tools/rs_rows.py as the tools import it, its module settings at the defaults (restored
    after the test, WAIT_KEEP_AT_BUILD too) and no engine config from the PC."""
    monkeypatch.syspath_prepend(str(TOOLS))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    rr = importlib.import_module("rs_rows")
    eng = rr.rs_engine
    for var in (eng.CONFIG_ENV, *eng.ENV.values(), rr.rs_cards.EXTRAS_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(eng, "_ACTIVE", None)
    for name, value in {"F": None, "LIVE": False, "UNITS": rr.UNITS_TYPES,
                        "MEM_LAG": rr.rs_memory.BOARD_LAG, "VC": None, "VCT": None,
                        "VISIBILITY": rr.VIS_LEGACY, "TRUST": rr.TrustRule(), "OUT": rr.OUT,
                        "CARD_FLY": {}, "CARD_BUILDING": {}, "PRESSES": "off",
                        "BUTTONS": 2, "HAND_SIZE": 4, "DECISION": rr.DecisionRule(),
                        "ROW_SCHEMA": rr.rs_schema.SCHEMA_LEGACY,
                        "WAIT_KEEP_AT_BUILD": None}.items():
        monkeypatch.setattr(rr, name, value)
    return rr


@pytest.fixture
def engine(monkeypatch):
    """install(**battle_kwargs): a fake royalesim in sys.modules for this test."""
    def install(**kwargs):
        mod = make_fake_royalesim(**kwargs)
        monkeypatch.setitem(sys.modules, "royalesim", mod)
        return mod
    return install


#: tests/pipeline/test_decision_rows.py's match: four plays; at execution (D = 0) the fake gives
#: 52 rows, the 4 plays and 48 waits.
TAG = "decision-0001"
PLAYS = ((105, "opponent", "knight", 14500, 24500), (290, "opponent", "arrows", 3500, 8000),
         (300, "team", "hog-rider", 3500, 14500), (320, "team", "musketeer", 3500, 7500))
BLUE_EARLY, BLUE_LATE = ([2, 0, 6, 3], 5), ([5, 0, 6, 3], 1)
OPP_HAND, OPP_NEXT = [8, 11, 9, 10], 12


def payload_a():
    return il_payload(tag=TAG, plays=PLAYS, abilities=())


def states_a(battle):
    hand, nxt = BLUE_EARLY if battle.tick <= 300 else BLUE_LATE
    return state_dict(battle.tick, players=[player_dict(0, hand=hand, next_card=nxt),
                                            player_dict(1, hand=OPP_HAND, next_card=OPP_NEXT)])


def cfg(rr):
    return rr.rs_engine.EngineConfig.defaults()


def shard(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# rs_rows: which matches
# ---------------------------------------------------------------------------

#: (part, row, tag, hog) in parquet order: 3 Hog 2.6 matches (rows 0, 3, 7), 7 others.
HOGS = (0, 3, 7)
CANDS = [(k // 5, k % 5, f"t{k}", k in HOGS) for k in range(10)]
OTHERS = [c for c in CANDS if not c[3]]


def legacy_pick(cands, others):
    """main's rule before E18, over the candidates in parquet order."""
    hog, rest = [], []
    for c in cands:
        if c[3]:
            hog.append(c)
        elif len(rest) < others:
            rest.append(c)
    return hog + rest


def test_first_pick_is_the_old_rule(rr):
    for n in (0, 1, 3, 7, 100):
        assert rr.select_matches(CANDS, n) == legacy_pick(CANDS, n), n
    # by hand: the Hog matches t0, t3, t7, then the first three others t1, t2, t4
    assert [c[2] for c in rr.select_matches(CANDS, 3)] == ["t0", "t3", "t7", "t1", "t2", "t4"]
    assert rr.select_matches(CANDS, 3, "first", seed=99) == rr.select_matches(CANDS, 3)


def test_random_pick_is_seeded_and_all_matches_takes_every_one(rr):
    pick = rr.select_matches(CANDS, 3, "random", seed=5)
    assert pick[:3] == [CANDS[k] for k in HOGS]                 # every Hog match, first
    rest = pick[3:]
    assert len(rest) == 3 and len(set(rest)) == 3 and set(rest) <= set(OTHERS)
    assert rest == sorted(rest, key=CANDS.index)                # in parquet order
    idx = np.sort(np.random.default_rng([rr.SELECT_SALT, 5]).choice(7, 3, replace=False))
    assert rest == [OTHERS[int(i)] for i in idx]
    assert rr.select_matches(CANDS, 3, "random", seed=5) == pick             # same seed
    picks = {tuple(rr.select_matches(CANDS, 3, "random", seed=s)) for s in range(10)}
    assert len(picks) > 1                                       # the seed matters
    assert rr.select_matches(CANDS, 7, "random", seed=1) == legacy_pick(CANDS, 7)   # all of them
    assert rr.select_matches(CANDS, 50, "random") == legacy_pick(CANDS, 7)
    assert rr.select_matches(CANDS, 0, "random") == [CANDS[k] for k in HOGS]
    every = [CANDS[k] for k in HOGS] + OTHERS
    for pick_mode in rr.PICK_MODES:
        assert rr.select_matches(CANDS, 1, pick_mode, all_matches=True) == every
    with pytest.raises(ValueError, match="--others-pick"):
        rr.select_matches(CANDS, 3, "best")
    with pytest.raises(ValueError, match="--others"):
        rr.select_matches(CANDS, -1)
    with pytest.raises(ValueError, match="--select-seed"):
        rr.select_rng(-1)


def test_selection_record(rr):
    assert rr.selection_record(rr.select_matches(CANDS, 3), 10, 2, 3, "first", 0, False) is None
    pick = rr.select_matches(CANDS, 3, "random", seed=5)
    order = "Hog 2.6 first, then the others, each in parquet order"
    assert rr.selection_record(pick, 10, 2, 3, "random", 5, False) == {
        "pick": "random", "others": 3, "seed": 5, "parts": 2, "candidates": 10, "hog": 3,
        "other_candidates": 7, "others_taken": 3, "order": order,
        "rng": "numpy default_rng([18, seed]).choice(other candidates, others, replace=False), "
               "sorted"}
    every = rr.select_matches(CANDS, 3, all_matches=True)
    assert rr.selection_record(every, 10, 2, 3, "first", 0, True) == {
        "pick": "all", "others": None, "seed": None, "parts": 2, "candidates": 10, "hog": 3,
        "other_candidates": 7, "others_taken": 7, "order": order}


#: IL_Replay key -> catalogue name for the test decks ("mystery" is left out: unmapped).
NAMES = dict(zip(HOG26_KEYS, ("Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits",
                              "Musketeer", "Skeletons", "Log")))
NAMES.update({k: k.title() for k in OPPONENT_KEYS})


def part_rows():
    """One parquet part: Hog vs others (a candidate, hog), others vs others (a candidate),
    a 2v2 match, an unmapped card, two players a side, others vs Hog (a candidate, hog)."""
    plain = dict(team_forms={}, opponent_forms={}, plays=(), abilities=())
    two = il_payload(tag="m4", **plain)
    two["battle"]["opponent"]["players"].append(dict(two["battle"]["opponent"]["players"][0]))
    payloads = [
        il_payload(tag="m0"),
        il_payload(tag="m1", team_deck=OPPONENT_KEYS, **plain),
        il_payload(tag="m2", game_mode="2v2", **plain),
        il_payload(tag="m3", opponent_deck=OPPONENT_KEYS[:7] + ("mystery",), **plain),
        two,
        il_payload(tag="m5", team_deck=OPPONENT_KEYS, opponent_deck=HOG26_KEYS, **plain),
    ]
    return payloads, [il_replay_row(p) for p in payloads]


def test_scan_part_keeps_the_matches_main_kept(rr):
    payloads, rows = part_rows()
    assert rr.scan_part(rows, NAMES) == [(0, "m0", True), (1, "m1", False), (5, "m5", True)]
    assert rr.match_decks(payloads[0], NAMES) == (frozenset(NAMES[k] for k in HOG26_KEYS),
                                                  frozenset(NAMES[k] for k in OPPONENT_KEYS))
    assert rr.match_decks(payloads[3], NAMES) is None          # "mystery" has no name
    assert rr.match_decks(payloads[4], NAMES) is None          # two players on a side
    assert set(rr.SCAN_COLUMNS) <= set(rows[0]) and set(rr.PAYLOAD_COLUMNS) <= set(rows[0])


def test_selected_payloads_read_a_part_when_the_order_comes_to_it(rr):
    payloads, rows = part_rows()
    tables = {"A": rows[:3], "B": rows[3:]}
    reads = []

    def read(path, columns):
        reads.append((path, tuple(columns)))
        return tables[path]

    picked = [(0, 0, "m0", True), (1, 2, "m5", True), (0, 1, "m1", False), (1, 0, "m3", False)]
    texts = list(rr.selected_payloads(picked, ["A", "B"], read))
    assert [json.loads(t) for t in texts] == [payloads[0], payloads[5], payloads[1], payloads[3]]
    assert reads == [("A", rr.PAYLOAD_COLUMNS), ("B", rr.PAYLOAD_COLUMNS),
                     ("A", rr.PAYLOAD_COLUMNS), ("B", rr.PAYLOAD_COLUMNS)]
    # the source tag is set as main always set it
    bare = dict(payloads[1])
    del bare["source"]
    row = dict(rows[1], payload_json=json.dumps(bare))
    (text,) = rr.selected_payloads([(0, 0, "m1", False)], ["C"], lambda p, c: [row])
    assert json.loads(text)["source"] == {"replay_tag": "m1"}
    with pytest.raises(RuntimeError, match="scanned as 'm9'"):
        list(rr.selected_payloads([(0, 0, "m9", False)], ["A"], read))


def test_chunked(rr):
    assert list(rr.chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(rr.chunked(range(6), 3)) == [[0, 1, 2], [3, 4, 5]]
    assert list(rr.chunked([], 3)) == []
    # the legacy batches: picked[i:i + batch]
    picked = list(range(11))
    assert list(rr.chunked(picked, 4)) == [picked[i:i + 4] for i in range(0, 11, 4)]
    with pytest.raises(ValueError, match="--batch"):
        list(rr.chunked([1], 0))


def test_bounded_keeps_at_most_limit_futures_out(rr, monkeypatch):
    out = []

    def submit(k, job):
        assert sum(1 for f in out if not f.done()) < 2         # room for this one
        f = Future()
        f.job = (k, job)
        out.append(f)
        return f

    def fake_wait(pending, return_when):
        assert return_when == rr.FIRST_COMPLETED
        first = min(pending, key=lambda f: f.job[0])           # the oldest finishes
        first.set_result(first.job)
        return {first}, set(pending) - {first}

    monkeypatch.setattr(rr, "wait_futures", fake_wait)
    got = [f.result() for f in rr.bounded(submit, "abcde", 2)]
    assert got == [(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e")]
    assert [f.job[0] for f in out] == [0, 1, 2, 3, 4]
    assert list(rr.bounded(submit, "", 2)) == []


# ---------------------------------------------------------------------------
# rs_rows: --wait-keep-at-build
# ---------------------------------------------------------------------------

def test_wait_keep_value_and_rule(rr):
    assert rr.wait_keep_value(None) is None
    assert rr.wait_keep_value(0.25) == 0.25 and rr.wait_keep_value("0.5") == 0.5
    assert rr.wait_keep_value(1) == 1.0
    for bad in (0, 0.0, 1.5, -0.1, True, "x", "nan", [0.5]):
        with pytest.raises(ValueError, match="--wait-keep-at-build"):
            rr.wait_keep_value(bad)
    assert rr.wait_rule(None) is None
    assert rr.wait_rule(0.3) == {
        "keep": 0.3, "waits": "rows with act_card 0 and no press",
        "seed": "int(sha1(replay_tag)[:16], 16)", "stream": 2,
        "draws": "one per row of the match, in row order; a wait is kept when its draw < keep"}
    assert rr.STREAM_WAITS not in (rr.STREAM_PLAYS, rr.STREAM_PRESSES)


def thin_rows():
    """12 rows: plays at 1, 5, 9; a press at 6; waits elsewhere (one with press 0 given)."""
    rows = []
    for i in range(12):
        row = {"action": (4, 3, 17) if i % 4 == 1 else (0, 0, 0), "i": i}
        if i == 6:
            row["press"] = 2
        elif i == 8:
            row["press"] = 0
        rows.append(row)
    return rows


def test_thin_waits(rr):
    rows = thin_rows()
    assert [rr.is_wait(r) for r in rows] == [i not in (1, 5, 6, 9) for i in range(12)]
    kept, dropped = rr.thin_waits(rows, "thin-0001", None)
    assert kept == rows and dropped == 0
    draws = np.random.default_rng([rr.tap_seed("thin-0001"), 2]).random(12)
    want = [r for r, u in zip(rows, draws) if r["i"] in (1, 5, 6, 9) or u < 0.5]
    kept, dropped = rr.thin_waits(rows, "thin-0001", 0.5)
    assert kept == want and dropped == 12 - len(want)
    assert [r["i"] for r in kept] == sorted(r["i"] for r in kept)       # the rows' order
    assert {1, 5, 6, 9} <= {r["i"] for r in kept}                       # plays and the press
    assert rr.thin_waits(rows, "thin-0001", 0.5) == (kept, dropped)     # seeded by the tag
    assert rr.thin_waits(rows, "thin-0001", 1.0) == (rows, 0)           # draws are < 1
    kept, dropped = rr.thin_waits(rows, "thin-0001", 1e-12)
    assert [r["i"] for r in kept] == [1, 5, 6, 9] and dropped == 8
    assert rr.thin_waits([], "thin-0001", 0.5) == ([], 0)


def test_build_thins_waits_by_the_match_tag(rr, engine, tmp_path):
    engine(states=states_a)
    text = json.dumps(payload_a())
    eng = cfg(rr).to_json()
    full = rr.build(1, [text], False, str(tmp_path / "all"), eng, stats=True)
    assert full[:3] == (1, 52, {}) and "wait_thinning" not in full[3]
    thin = rr.build(2, [text], False, str(tmp_path / "thin"), eng, stats=True,
                    wait_keep_at_build=0.5)
    assert rr.WAIT_KEEP_AT_BUILD == 0.5
    a, b = shard(tmp_path / "all" / "shard-0001.npz"), shard(tmp_path / "thin" / "shard-0002.npz")
    assert int((a["act_card"] > 0).sum()) == 4 and int((a["act_card"] == 0).sum()) == 48
    draws = np.random.default_rng([rr.tap_seed(TAG), rr.STREAM_WAITS]).random(52)
    kept = [i for i in range(52) if a["act_card"][i] > 0 or draws[i] < 0.5]
    assert thin[:3] == (2, len(kept), {})
    assert list(b) == list(a)
    for k in a:
        assert np.array_equal(b[k], a[k][kept]), k
    waits = sum(1 for i in kept if a["act_card"][i] == 0)
    assert thin[3]["wait_thinning"] == {"kept": waits, "dropped": 48 - waits}
    assert (thin[3]["rows"], thin[3]["play_rows"]) == (len(kept), 4)
    # None keeps the setting: the same tag, the same rows
    rr.build(3, [text], False, str(tmp_path / "again"), eng)
    c = shard(tmp_path / "again" / "shard-0003.npz")
    for k in a:
        assert np.array_equal(c[k], b[k]), k
    man = rr.manifest(thin[3], [rr.shard_entry(2, thin[3])], thin[2], wait_keep=0.5)
    assert man["wait_thinning"] == {"keep": 0.5, "kept": waits, "dropped": 48 - waits}
    assert (man["rows"], man["wait_rows"]) == (len(kept), len(kept) - 4)
    assert (f"waits kept at build with p 0.5: {waits:,} kept, {48 - waits:,} dropped"
            in rr.describe_manifest(man))
    assert "wait_thinning" not in rr.manifest(full[3], [rr.shard_entry(1, full[3])], {})
    with pytest.raises(ValueError, match="--wait-keep-at-build"):
        rr.build(4, [text], False, str(tmp_path / "bad"), eng, wait_keep_at_build=0)
    assert not (tmp_path / "bad").exists()


def test_vocab_record_records_thinning_and_selection(rr):
    cat, units, fp = [["Knight"], ["Archers"]], ["u0"], {"version": "9.9", "card_table": "t1"}
    plain = rr.vocab_record(cat, units, cfg(rr), fp, {}, False)
    assert plain["wait_keep_at_build"] is None
    assert "wait_thinning" not in plain and "selection" not in plain
    sel = rr.selection_record(rr.select_matches(CANDS, 3, all_matches=True), 10, 2, 3, "first",
                              0, True)
    rec = rr.vocab_record(cat, units, cfg(rr), fp, {}, False, wait_keep_at_build=0.25,
                          selection=sel)
    assert rec["wait_keep_at_build"] == 0.25 and rec["wait_thinning"] == rr.wait_rule(0.25)
    assert rec["selection"] == sel
    assert {k: rec[k] for k in plain if k != "wait_keep_at_build"} == \
        {k: v for k, v in plain.items() if k != "wait_keep_at_build"}


def _argument(main, flag):
    hits = [n for n in ast.walk(main) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", "") == "add_argument"
            and n.args and getattr(n.args[0], "value", None) == flag]
    assert len(hits) == 1, flag
    return {k.arg: k.value for k in hits[0].keywords}


def _main(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def test_rows_main_wires_selection_and_thinning():
    text = (TOOLS / "rs_rows.py").read_text(encoding="utf-8")
    main = text[text.index("def main()"):]
    for needle in ("scan_part(_read_part(part, SCAN_COLUMNS)", "select_matches(cands, args.others, "
                   "args.others_pick, args.select_seed", "selection=selection",
                   "selected_payloads(picked, parts)", "chunked(", "bounded(submit, jobs, "
                   "2 * args.workers)", "WAIT_KEEP_AT_BUILD = wait_keep",
                   "manifest(totals, shards, skipped, TRUST, wait_keep=WAIT_KEEP_AT_BUILD)"):
        assert needle in main, needle
    assert main.count("wait_keep_at_build=WAIT_KEEP_AT_BUILD") == 2     # vocab.json, workers
    tree = _main(TOOLS / "rs_rows.py")
    assert _argument(tree, "--others")["default"].value == 20000
    assert _argument(tree, "--others-pick")["default"].id == "PICK_FIRST"
    assert _argument(tree, "--select-seed")["default"].value == 0
    assert _argument(tree, "--wait-keep-at-build")["default"].value is None
    assert _argument(tree, "--all-matches")["action"].value == "store_true"


# ---------------------------------------------------------------------------
# shards in tools/rs_rows.py's layout (tests/pipeline/test_train_tokens.py's)
# ---------------------------------------------------------------------------

def shard_arrays(n: int, *, start: int = 0, seed: int = 0, splits=None, trusted=None, hog=None,
                 unit_dtype=np.float16, outcome_dtype=np.float64, press=None,
                 trusted_v2=None) -> dict:
    """One legacy shard. Row i: a wait when i % 3 == 0, else HAND[i % 4], except a card not in
    the hand when i % 11 == 10; split by i % 5 (0-2 train, 3 val, 4 test), trusted unless
    i % 7 == 6, Hog when i % 4 < 2; outcome (start + i) / 100, won i % 2; match tag
    m{(start + i) // 10:05d}. `press`: rows (waits) with act_press 1."""
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
        "unit_feats": rng.random((n, UNIT_SLOTS, 7)).astype(unit_dtype),
        "towers": rng.random((n, 6)).astype(np.float32),
        "scalars": rng.random((n, 4)).astype(np.float32),
        "hand": np.tile(np.asarray(HAND, dtype=np.int16), (n, 1)),
        "next": np.full(n, 6, dtype=np.int64),
        "opp_seen": rng.integers(0, N_CARDS, (n, 8)).astype(np.int16),
        "opp_last": rng.integers(0, N_CARDS, (n, 4)).astype(np.int16),
        "opp_last_age": rng.random((n, 4)).astype(np.float16),
        "outcome": ((start + i) / 100).astype(outcome_dtype),
        "trusted": np.asarray(i % 7 != 6 if trusted is None else trusted, dtype=bool),
        "won": (i % 2).astype(np.float64),
        "hog": np.asarray(i % 4 < 2 if hog is None else hog, dtype=bool),
    }
    if trusted_v2 is not None:
        out["trusted_v2"] = np.asarray(trusted_v2, dtype=bool)
    out.update(act_card=act, act_x=rng.integers(0, 18, n).astype(np.int16),
               act_y=rng.integers(0, 32, n).astype(np.int16))
    if press is not None:
        assert (act[list(press)] == 0).all()
        out["act_press"] = np.isin(i, press).astype(np.int16)
    out.update(match=np.asarray([f"m{(start + k) // 10:05d}" for k in i]),
               split=np.asarray(splits))
    return out


def v2_shard(n, *, start=0, seed=0, splits, trusted, hog, press_rows=(3, 33)) -> dict:
    """One shard in rs_schema.V2's columns and dtypes (test_model_tokens_v2.py's): a wait when
    i % 3 == 0, else HAND[i % 4]; act_press 1 on `press_rows`; buttons (5, 6, none), button 0
    available on the odd rows."""
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    out = {}
    for name in RS.V2.shard_columns():
        col = RS.V2.column(name)
        shape = (n,) + RS.V2.shape(name, B)
        if col.dtype == "bool":
            out[name] = np.ones(shape, dtype=bool)
        elif col.dtype.startswith("int"):
            out[name] = rng.integers(0, N_UNITS, shape).astype(col.dtype)
        elif col.dtype != "str":
            out[name] = rng.random(shape).astype(col.dtype)
    avail = RS.ABILITY_FEATS.index("available")
    out["ability_state"][:, :, avail] = 0
    out["ability_state"][:, 0, avail] = i % 2 == 1
    out.update(
        hand=np.tile(np.asarray(HAND, dtype=np.int16), (n, 1)), next=np.full(n, 6, dtype=np.int16),
        ability_card=np.tile(np.asarray([5, 6, 0], dtype=np.int16), (n, 1)),
        act_card=np.where(i % 3 == 0, 0, np.asarray(HAND)[i % 4]).astype(np.int16),
        act_press=np.isin(i, press_rows).astype(np.int16),
        act_x=rng.integers(0, 18, n).astype(np.int16), act_y=rng.integers(0, 32, n).astype(np.int16),
        outcome=((start + i) / 100).astype(np.float32), won=(i % 2).astype(np.float32),
        trusted=np.asarray(trusted, dtype=bool), hog=np.asarray(hog, dtype=bool),
        match=np.asarray([f"m{(start + k) // 10:05d}" for k in i]), split=np.asarray(splits))
    return {name: out[name] for name in RS.V2.shard_columns()}


def write_shard(folder: Path, k: int, arrays: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"shard-{k:04d}.npz"
    np.savez_compressed(path, **arrays)
    return path


def quiet(*_):
    pass


@pytest.fixture
def mixed_rows(tmp_path):
    """test_train_tokens.py's three shards: 50 rows (float32 outcome), 40 (float32 unit_feats),
    30 (train only); 120 rows."""
    root = tmp_path / "mixed"
    write_shard(root, 0, shard_arrays(50, start=0, seed=1, outcome_dtype=np.float32))
    write_shard(root, 1, shard_arrays(40, start=100, seed=2, unit_dtype=np.float32))
    write_shard(root, 2, shard_arrays(30, start=200, seed=3, splits=["train"] * 30))
    return root


@pytest.fixture
def plain_rows(tmp_path):
    """Two shards as tools/rs_rows.py writes them (float16 unit_feats, float64 outcome), and a
    vocab.json and manifest.json."""
    root = tmp_path / "plain"
    write_shard(root, 0, shard_arrays(30, start=0, seed=4))
    write_shard(root, 1, shard_arrays(20, start=300, seed=5))
    (root / "vocab.json").write_text(json.dumps({"schema": "rs_rows/legacy", "n_cards": 12}),
                                     encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"rows": 50}), encoding="utf-8")
    return root


@pytest.fixture
def press_rows(tmp_path):
    """Two legacy shards of 40 rows with act_press on waits 3 and 33 and a trusted_v2 column
    (rows 2-36); rows 0-29 train, 30-39 val; Hog on even rows."""
    root = tmp_path / "press"
    i = np.arange(40)
    for k in (0, 1):
        write_shard(root, k, shard_arrays(
            40, start=100 * k, seed=10 + k, splits=["train"] * 30 + ["val"] * 10,
            trusted=i >= 2, hog=i % 2 == 0, press=(3, 33), trusted_v2=(i >= 2) & (i < 37)))
    return root


@pytest.fixture
def v2_rows(tmp_path):
    root = tmp_path / "v2"
    i = np.arange(40)
    for k in (0, 1):
        write_shard(root, k, v2_shard(40, start=100 * k, seed=10 + k,
                                      splits=["train"] * 30 + ["val"] * 10, trusted=i >= 2,
                                      hog=i % 2 == 0))
    (root / "vocab.json").write_text(json.dumps({"schema": "rs_rows/v2"}), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# tools/rs_rows_compact.py
# ---------------------------------------------------------------------------

#: The mixed shards' store: (dtype, how it was stored) per column, in shard order.
MIXED_COLUMNS = {
    "tick": ("int32", "compact"), "side": ("int8", "compact"), "hand_exact": ("bool", "as stored"),
    "unit_ids": ("int16", "as stored"),
    "unit_feats": ("float16", "float16 as the trainer loads it"),
    "towers": ("float32", "as stored"), "scalars": ("float32", "as stored"),
    "hand": ("int16", "as stored"), "next": ("int16", "compact"),
    "opp_seen": ("int16", "as stored"), "opp_last": ("int16", "as stored"),
    "opp_last_age": ("float16", "as stored"),
    "outcome": ("float64", "as stored"),          # (100 + i) / 100 is no float32
    "trusted": ("bool", "as stored"), "won": ("float32", "compact"), "hog": ("bool", "as stored"),
    "act_card": ("int16", "as stored"), "act_x": ("int16", "as stored"),
    "act_y": ("int16", "as stored"), "match": ("int32", "codes"), "split": ("int8", "codes"),
}


def test_npz_headers_without_inflating(plain_rows):
    arrays = shard_arrays(30, start=0, seed=4)
    head = RC.npz_columns(plain_rows / "shard-0000.npz")
    assert list(head) == list(arrays)
    assert head["unit_feats"] == (np.dtype("float16"), (30, UNIT_SLOTS, 7))
    assert head["match"] == (np.dtype("<U6"), (30,)) and head["tick"] == (np.dtype("int64"), (30,))


def test_compact_layout_dtypes_and_indexes(mixed_rows, tmp_path):
    out = tmp_path / "store"
    meta = RC.compact(mixed_rows, out, log=quiet)
    assert meta["format"] == RC.FORMAT and meta["rows"] == 120
    assert {c: (s["dtype"], s["stored"]) for c, s in meta["columns"].items()} == MIXED_COLUMNS
    assert list(meta["columns"]) == list(MIXED_COLUMNS)
    assert meta["columns"]["unit_feats"]["shape"] == [UNIT_SLOTS, 7]
    assert [(s["name"], s["rows"], s["start"]) for s in meta["shards"]] == [
        ("shard-0000.npz", 50, 0), ("shard-0001.npz", 40, 50), ("shard-0002.npz", 30, 90)]
    assert meta["shards"][0]["dtypes"] == {"tick": "int64", "side": "int64", "next": "int64",
                                           "outcome": "float32", "won": "float64",
                                           "match": "<U6", "split": "<U5"}
    assert meta["shards"][1]["dtypes"] == {"tick": "int64", "side": "int64",
                                           "unit_feats": "float32", "next": "int64",
                                           "won": "float64", "match": "<U6", "split": "<U5"}
    assert meta["splits"] == ["train", "val", "test"] and meta["matches"] == 12
    # by hand: train 30 + 24 + 30, val 10 + 8, test 10 + 8; untrusted i % 7 == 6 (7, 5, 4 rows);
    # Hog i % 4 < 2 (26, 20, 16)
    assert meta["index"] == {"split_train": 84, "split_val": 18, "split_test": 18,
                             "trusted": 104, "hog": 62}
    assert (out / RC.COMPACT_JSON).exists() and not (out / "vocab.json").exists()
    store = RC.CompactRows(out)
    full = {c: np.concatenate([shard(mixed_rows / f"shard-{k:04d}.npz")[c] for k in range(3)])
            for c in MIXED_COLUMNS}
    for name, want in (("split_train", full["split"] == "train"),
                       ("split_val", full["split"] == "val"),
                       ("split_test", full["split"] == "test"),
                       ("trusted", full["trusted"]), ("hog", full["hog"])):
        ids = store.index(name)
        assert ids.dtype == np.int64 and np.array_equal(ids, np.flatnonzero(want)), name
    assert store.match_names().tolist() == ([f"m{k:05d}" for k in range(5)]
                                            + [f"m{k:05d}" for k in range(10, 14)]
                                            + [f"m{k:05d}" for k in range(20, 23)])
    assert store["tick"].dtype == np.int32 and isinstance(store["tick"], np.memmap)
    assert np.array_equal(store["tick"], full["tick"])
    assert np.array_equal(store.decode_split(store["split"]), full["split"])
    assert store.split_code("val") == 1 and store.split_code("dev") == -1
    assert store.row_bytes(["unit_feats", "act_card"]) == UNIT_SLOTS * 7 * 2 + 2
    assert RC.store_bytes(meta, ["act_card"]) == 120 * 2
    store.close()


def test_round_trip_is_exact(plain_rows, tmp_path):
    out = plain_rows / "compact"
    meta = RC.compact(plain_rows, log=quiet)                 # default <rows>/compact
    assert (out / RC.COMPACT_JSON).exists()
    assert (out / "vocab.json").read_bytes() == (plain_rows / "vocab.json").read_bytes()
    assert (out / "manifest.json").read_bytes() == (plain_rows / "manifest.json").read_bytes()
    assert meta["schema"] == "rs_rows/legacy" and meta["source"] == str(plain_rows)
    store = RC.CompactRows(out)
    for k in (0, 1):
        want = shard(plain_rows / f"shard-{k:04d}.npz")
        back = store.shard(k)
        assert list(back) == list(want)
        for c in want:
            assert back[c].dtype == want[c].dtype, c
            assert back[c].shape == want[c].shape, c
            assert np.array_equal(back[c], want[c]), c
    store.close()
    assert RC.verify(plain_rows, log=quiet) == []


def test_round_trip_of_mixed_shards_is_what_the_trainer_reads(mixed_rows, tmp_path):
    out = tmp_path / "store"
    RC.compact(mixed_rows, out, log=quiet)
    assert RC.verify(mixed_rows, out, log=quiet) == []
    store = RC.CompactRows(out)
    want = shard(mixed_rows / "shard-0001.npz")
    back = store.shard(1)
    assert back["unit_feats"].dtype == np.float32
    assert np.array_equal(back["unit_feats"], want["unit_feats"].astype(np.float16)
                          .astype(np.float32))
    assert np.array_equal(back["outcome"], want["outcome"]) and back["tick"].dtype == np.int64
    assert store.shard(0)["outcome"].dtype == np.float32           # shard 0 kept float32
    store.close()
    # a store that differs from its shards is found
    np.savez_compressed(mixed_rows / "shard-0002.npz",
                        **dict(shard(mixed_rows / "shard-0002.npz"), act_x=np.zeros(30, np.int16)))
    assert RC.verify(mixed_rows, out, log=quiet) == ["shard-0002.npz: column act_x differs"]


def test_compact_refuses(mixed_rows, tmp_path):
    with pytest.raises(RC.CompactError, match="is the rows folder"):
        RC.compact(mixed_rows, mixed_rows, log=quiet)
    with pytest.raises(RC.CompactError, match="no shard"):
        RC.compact(tmp_path / "nothing", tmp_path / "out0", log=quiet)
    out = tmp_path / "out1"
    RC.compact(mixed_rows, out, log=quiet)
    with pytest.raises(RC.CompactError, match="--force"):
        RC.compact(mixed_rows, out, log=quiet)
    assert RC.compact(mixed_rows, out, force=True, log=quiet)["rows"] == 120
    other = tmp_path / "odd"
    write_shard(other, 0, shard_arrays(10))
    write_shard(other, 1, {k: v for k, v in shard_arrays(10).items() if k != "won"})
    with pytest.raises(RC.CompactError, match=r"shard-0001.npz: columns \['won'\] differ"):
        RC.compact(other, tmp_path / "out2", log=quiet)
    with pytest.raises(FileNotFoundError, match="compact.json"):
        RC.CompactRows(tmp_path / "out2")
    (tmp_path / "out3").mkdir()
    (tmp_path / "out3" / RC.COMPACT_JSON).write_text(json.dumps({"format": "x/0"}),
                                                     encoding="utf-8")
    with pytest.raises(RC.CompactError, match="format"):
        RC.CompactRows(tmp_path / "out3")


def test_compact_keeps_v2_dtypes(v2_rows, tmp_path):
    out = tmp_path / "store"
    meta = RC.compact(v2_rows, out, log=quiet)
    for c in RS.V2.shard_columns():
        col = RS.V2.column(c)
        assert meta["columns"][c]["dtype"] == (RC.CODE_COLUMNS[c] if col.dtype == "str"
                                               else col.dtype), c
    assert meta["columns"]["ability_state"]["shape"] == [B, 5]
    assert RC.verify(v2_rows, out, log=quiet) == []


def test_the_compact_rules_agree_with_the_trainer():
    tt = _train_tokens()
    mt = importlib.import_module("sim.model_tokens")
    assert RC.HALF_COLS == tt.HALF_COLS
    for vocab in ({}, {"schema": "rs_rows/v2"}, {"schema": "v2"}, {"schema": "rs_rows/test"},
                  {"schema": "rs_rows/legacy"}):
        assert RC.row_schema(vocab) == mt.model_schema(vocab), vocab
    assert tt.RC is RC


# ---------------------------------------------------------------------------
# sim/train_tokens.py --memmap: the same rows as load()
# ---------------------------------------------------------------------------

def _train_tokens():
    pytest.importorskip("torch")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("sim.train_tokens")


@pytest.fixture
def tt():
    return _train_tokens()


def assert_same_rows(got: dict, want: dict, cols):
    assert list(got) == list(want) == list(cols)
    for c in cols:
        assert got[c].shape == want[c].shape, c
        assert np.array_equal(got[c], want[c]), c


@pytest.mark.parametrize("split, wait_keep, trusted_only, hog_only, seed", [
    ("train", 0.35, True, False, 0),
    ("val", 0.35, True, False, 1),
    ("val", 0.35, True, True, 1),
    ("train", 1.0, False, False, 5),
    ("test", 0.5, False, True, 2),
])
def test_memmap_keeps_the_rows_load_keeps(tt, mixed_rows, tmp_path, split, wait_keep,
                                          trusted_only, hog_only, seed):
    RC.compact(mixed_rows, tmp_path / "store", log=quiet)
    store = tt.open_store(tmp_path / "store")
    want = tt.load(split, wait_keep, trusted_only, hog_only, seed, mixed_rows)
    for chunk in (7, 1000):                 # chunks that cut across the shards, and one
        ids = tt.memmap_keep(store, split, wait_keep, trusted_only, hog_only, seed, chunk=chunk)
        assert ids.dtype == np.int64 and np.all(np.diff(ids) > 0)
        assert_same_rows(tt.gather(store, tt.COLS, ids, window=5), want, tt.COLS)
    capped = tt.load(split, wait_keep, trusted_only, hog_only, seed, mixed_rows, max_rows=9)
    ids = tt.memmap_keep(store, split, wait_keep, trusted_only, hog_only, seed, max_rows=9)
    assert_same_rows(tt.gather(store, tt.COLS, ids), capped, tt.COLS)
    store.close()


@pytest.mark.parametrize("split, wait_keep, trusted_only, trust", [
    ("train", 0.5, True, "legacy"), ("val", 1.0, True, "v2"), ("train", 0.0, False, "legacy"),
])
def test_memmap_leaves_out_legacy_press_rows_as_load_does(tt, press_rows, tmp_path, split,
                                                          wait_keep, trusted_only, trust):
    RC.compact(press_rows, tmp_path / "store", log=quiet)
    store = tt.open_store(tmp_path / "store")
    want = tt.load(split, wait_keep, trusted_only, False, 3, press_rows, trust=trust)
    ids = tt.memmap_keep(store, split, wait_keep, trusted_only, False, 3, trust=trust, chunk=11)
    assert_same_rows(tt.gather(store, tt.COLS, ids), want, tt.COLS)
    assert tt.row_record_store(store, trust) == tt.row_record(press_rows, trust) == {
        "trust": trust, "trust_column": tt.trust_column(trust), "press_rows": "excluded",
        "press_rows_excluded": 4}
    store.close()


def test_memmap_keeps_v2_press_rows_as_load_does(tt, v2_rows, tmp_path):
    RC.compact(v2_rows, tmp_path / "store", log=quiet)
    store = tt.open_store(tmp_path / "store")
    cols = tt.cols_for("v2")
    for args in (("train", 0.0, False, False, 0), ("val", 1.0, True, False, 1),
                 ("val", 0.5, True, True, 2)):
        want = tt.load(*args, v2_rows)
        ids = tt.memmap_keep(store, *args, chunk=13)
        assert_same_rows(tt.gather(store, cols, ids), want, cols)
    # wait_keep 0: 20 plays and press row 3 of each shard's 30 train rows
    assert len(tt.memmap_keep(store, "train", 0.0, False, False, 0)) == 42
    assert tt.row_record_store(store) == tt.row_record(v2_rows) == {
        "trust": "legacy", "trust_column": "trusted", "press_rows": "trained",
        "press_rows_excluded": 0, "press_rows_trained": 4}
    store.close()


def test_memmap_refuses_like_load(tt, mixed_rows, tmp_path):
    RC.compact(mixed_rows, tmp_path / "store", log=quiet)
    store = tt.open_store(tmp_path / "store")
    with pytest.raises(ValueError, match="no trusted_v2 column"):
        tt.memmap_keep(store, "train", 1.0, True, False, 0, trust="v2")
    with pytest.raises(SystemExit, match=r"--trust v2: 3 of 3 shard\(s\)"):
        tt.row_record_store(store, "v2")
    assert tt.row_record_store(store) == tt.row_record(mixed_rows) == {
        "trust": "legacy", "trust_column": "trusted", "press_rows": "none",
        "press_rows_excluded": 0}
    with pytest.raises(ValueError, match=r"has no \['hand_feats'"):
        tt.memmap_keep(store, "train", 1.0, False, False, 0, schema="v2")
    store.close()
    with pytest.raises(SystemExit, match="tools/rs_rows_compact.py --rows"):
        tt.open_store(mixed_rows)


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, chunk, window", [(10, 3, 4), (1, 5, 5), (64, 64, 64),
                                              (1000, 7, 64), (37, 1, 37), (12, 3, 6)])
def test_epoch_order_takes_every_row_once(tt, n, chunk, window):
    order = tt.epoch_order(n, np.random.default_rng(0), chunk, window)
    assert order.dtype == np.int64 and np.array_equal(np.sort(order), np.arange(n))
    again = tt.epoch_order(n, np.random.default_rng(0), chunk, window)
    assert np.array_equal(order, again)                         # seeded
    assert len(tt.epoch_order(0, np.random.default_rng(0))) == 0


def test_epoch_order_reads_runs_of_rows(tt):
    # window 1: nothing shuffled within, so each run of 3 (the last one: [9]) stays in order
    order = tt.epoch_order(10, np.random.default_rng(1), chunk=3, window=1)
    for a, b in zip(order[:-1], order[1:]):
        if a % 3 != 2 and a != 9:
            assert b == a + 1
    # window 6 of runs of 3: each window holds two whole runs
    order = tt.epoch_order(12, np.random.default_rng(2), chunk=3, window=6)
    for w in (0, 6):
        part = np.sort(order[w:w + 6])
        assert {tuple(part[:3] // 3), tuple(part[3:] // 3)} <= {(k, k, k) for k in range(4)}
        assert part[0] % 3 == 0 and part[3] % 3 == 0


def test_window_batches_cover_every_row_once(tt):
    store = {"outcome": np.arange(100, dtype=np.float64),
             "act_card": (np.arange(100) % 7).astype(np.int16)}
    ids = np.arange(0, 100, 3)                                  # 34 rows
    order = tt.epoch_order(34, np.random.default_rng(4), 4, 10)
    got = list(tt.window_batches(store, ("outcome", "act_card"), ids, order, 5, 10,
                                 tt.PinnedStager("cpu")))
    assert [len(b["outcome"]) for b in got] == [5, 5, 5, 5, 5, 5, 4]
    seen = np.concatenate([b["outcome"].numpy() for b in got])
    assert np.array_equal(seen, ids[order].astype(np.float32))  # in the order's order
    assert all(b["outcome"].dtype.is_floating_point and b["act_card"].dtype == tt.torch.int64
               for b in got)
    assert np.array_equal(np.concatenate([b["act_card"].numpy() for b in got]), ids[order] % 7)
    # a window that is no multiple of the batch is rounded down to one (10 -> 8 for 4)
    assert [len(b["outcome"]) for b in tt.window_batches(store, ("outcome",), ids, order, 4,
                                                         10, tt.PinnedStager("cpu"))] == \
        [4] * 8 + [2]


def legacy_dict(n=30, seed=3):
    a = shard_arrays(n, seed=seed)
    return {c: a[c] for c in ("unit_ids", "unit_feats", "towers", "scalars", "hand", "next",
                              "opp_seen", "opp_last", "opp_last_age", "outcome", "won",
                              "act_card", "act_x", "act_y")}


def assert_same_batches(a, b, torch):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert list(x) == list(y)
        for c in x:
            assert x[c].dtype == y[c].dtype, c
            assert torch.equal(x[c], y[c]), c


def test_resident_batches_are_the_ram_batches(tt):
    torch = tt.torch
    data = legacy_dict()
    assert list(data) == list(tt.COLS)
    order = np.random.default_rng(2).permutation(30)
    ram = list(tt.batches(data, 8, True, "cpu", order=order))
    res = list(tt.resident_batches(tt.to_device(data, "cpu"), order, 8))
    assert [len(b["act_card"]) for b in res] == [8, 8, 8, 6]
    assert_same_batches(ram, res, torch)


def test_train_feed_gives_the_batches_of_before(tt):
    torch = tt.torch
    data = legacy_dict()
    np.random.seed(11)
    legacy = list(tt.batches(data, 8, True, "cpu"))
    for resident in (False, True):
        np.random.seed(11)
        feed = tt.TrainFeed(8, "cpu", train=data, resident=resident)
        assert feed.mode == ("resident" if resident else "ram") and feed.n == 30
        assert_same_batches(legacy, list(feed.epoch(np.random.default_rng(0))), torch)
    order = tt.balanced_order(data, 0.5, np.random.default_rng(1))
    want = list(tt.batches(data, 8, True, "cpu", order=order))
    feed = tt.TrainFeed(8, "cpu", 0.5, train=data)
    assert np.array_equal(feed.phase_counts(), np.bincount(tt.phase_of(data), minlength=12))
    assert_same_batches(want, list(feed.epoch(np.random.default_rng(1))), torch)


def test_train_feed_reads_every_kept_row_once_from_disk(tt, mixed_rows, tmp_path):
    RC.compact(mixed_rows, tmp_path / "store", log=quiet)
    store = tt.open_store(tmp_path / "store")
    ids = tt.memmap_keep(store, "train", 1.0, False, False, 0)
    assert len(ids) == 84
    feed = tt.TrainFeed(8, "cpu", store=store, ids=ids, cols=tt.COLS, window=16, chunk=5)
    assert (feed.mode, feed.n, feed.window) == ("memmap", 84, 16)
    got = list(feed.epoch(np.random.default_rng(0)))
    assert [len(b["act_card"]) for b in got] == [8] * 10 + [4]   # ceil(84 / 8) steps
    seen = np.sort(np.concatenate([b["outcome"].numpy() for b in got]))
    assert np.array_equal(seen, np.sort(np.asarray(store["outcome"][ids]).astype(np.float32)))
    assert list(got[0]) == list(tt.COLS) and got[0]["next"].dtype == tt.torch.int64
    on_device = tt.store_to_device(store, ("act_card", "outcome"), ids[:10], "cpu", window=3)
    assert np.array_equal(on_device["act_card"].numpy(), np.asarray(store["act_card"][ids[:10]]))
    assert on_device["outcome"].dtype == tt.torch.float64
    store.close()


# ---------------------------------------------------------------------------
# budgets
# ---------------------------------------------------------------------------

def test_budgets(tt):
    # 16 x 1000 + 200 x 100 + 2 x 64 x 100 + 2 x 8 x 100
    assert tt.memmap_ram_need(1000, 200, 100, 64, 8) == 16000 + 20000 + 12800 + 1600
    assert tt.resident_gpu_need(100_000, 1000) == 108_000
    # 1e-4 GB = 107,374.18 bytes: 108,000 does not fit, so GPU-resident mode is refused
    text = tt.budget_problem("GPU", 108_000, 1e-4, "Train with --memmap.")
    assert text == ("GPU budget: these rows need 0.00 GB, more than --gpu-budget-gb 0.0001. "
                    "Train with --memmap.")
    assert tt.budget_problem("GPU", 108_000, 2e-4) is None
    assert tt.budget_problem("RAM", 50_400, 50_400 / 2 ** 30) is None          # exactly full
    assert tt.budget_problem("RAM", 50_401, 50_400 / 2 ** 30).startswith("RAM budget")
    assert "--ram-budget-gb" in tt.budget_problem("RAM", 10 ** 12, 6)
    assert (tt.RAM_BUDGET_GB, tt.GPU_BUDGET_GB) == (6.0, 7.0)


# ---------------------------------------------------------------------------
# --mirror
# ---------------------------------------------------------------------------

def legacy_batch(torch):
    """Two rows, two unit slots (row 0's second is padding), as a batch reaches the model."""
    f = [[[1, 0.25, 0.5, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
         [[0, 0.75, 0.125, 0.5, 1, 0, 1], [1, 0.5, 0.875, 0.25, 0, 1, 0]]]
    return {"unit_ids": torch.tensor([[3, 0], [4, 5]]),
            "unit_feats": torch.tensor(f, dtype=torch.float16),
            "towers": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 0, 0.5, 0.25, 0.75, 1]]),
            "hand": torch.tensor([HAND, HAND]), "act_card": torch.tensor([2, 0]),
            "act_x": torch.tensor([3, 7]), "act_y": torch.tensor([10, 0])}


def test_mirror_rules_come_from_the_schema(tt):
    legacy = {"unit_feats": {"flip": [1], "where": "unit_ids"},
              "towers": {"swap": [[0, 1], [3, 4]]}, "act_x": {"tile": 17, "where": "act_card"}}
    assert tt.Mirror(0.5).rules == legacy
    assert tt.Mirror(0.5, "v2").rules == dict(legacy, spell_feats={"flip": [1, 3],
                                                                    "where": "spell_ids"})
    with pytest.raises(ValueError, match="--mirror"):
        tt.Mirror(1.5)


def test_mirror_by_hand(tt):
    torch = tt.torch
    b = legacy_batch(torch)
    out = tt.mirror_batch(b, torch.tensor([True, True]), tt.Mirror(1).rules)
    want = torch.tensor([[[1, 0.75, 0.5, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
                         [[0, 0.25, 0.125, 0.5, 1, 0, 1], [1, 0.5, 0.875, 0.25, 0, 1, 0]]])
    assert out["unit_feats"].dtype == torch.float32 and torch.equal(out["unit_feats"], want)
    swap = [1, 0, 2, 4, 3, 5]                                    # left <-> right, both sides
    assert torch.equal(out["towers"], b["towers"][:, swap])
    assert out["act_x"].tolist() == [14, 7]                      # 17 - 3; a wait keeps its 7
    for c in ("unit_ids", "hand", "act_card", "act_y"):
        assert out[c] is b[c]
    assert b["unit_feats"].dtype == torch.float16 and b["act_x"].tolist() == [3, 7]   # unchanged
    half = tt.mirror_batch(b, torch.tensor([False, True]), tt.Mirror(1).rules)
    assert torch.equal(half["unit_feats"][0], b["unit_feats"][0].float())
    assert torch.equal(half["unit_feats"][1], want[1])
    assert torch.equal(half["towers"][0], b["towers"][0])
    assert torch.equal(half["towers"][1], b["towers"][1, swap])
    assert half["act_x"].tolist() == [3, 7]


def test_a_mirrored_unit_stands_on_the_mirrored_tile(tt):
    torch = tt.torch
    x = torch.tensor([(c + 0.5) / 18 for c in range(18)], dtype=torch.float16)
    feats = torch.zeros(1, 18, 7, dtype=torch.float16)
    feats[0, :, 1] = x
    b = {"unit_ids": torch.ones(1, 18, dtype=torch.long), "unit_feats": feats,
         "act_card": torch.tensor([0])}
    out = tt.mirror_batch(b, torch.tensor([True]), tt.Mirror(1).rules)
    tiles = torch.floor(out["unit_feats"][0, :, 1] * 18).long()
    assert tiles.tolist() == [17 - c for c in range(18)]


def test_mirror_twice_is_the_identity(tt):
    torch = tt.torch
    rng = np.random.default_rng(6)
    n, s = 6, 4
    b = {"unit_ids": torch.from_numpy(rng.integers(0, 4, (n, s))),
         "unit_feats": torch.from_numpy(rng.random((n, s, 9)).astype(np.float16)),
         "towers": torch.from_numpy(rng.random((n, 6)).astype(np.float32)),
         "spell_ids": torch.from_numpy(rng.integers(0, 3, (n, 8))),
         "spell_feats": torch.from_numpy(rng.random((n, 8, 6)).astype(np.float16)),
         "own_pending": torch.from_numpy(rng.integers(0, 9, (n, 4))),
         "own_pending_land": torch.from_numpy(rng.random((n, 4)).astype(np.float16)),
         "act_card": torch.tensor([0, 2, 3, 0, 4, 2]),
         "act_x": torch.from_numpy(rng.integers(0, 18, n))}
    m = torch.tensor([True, False, True, True, False, True])
    rules = tt.Mirror(1, "v2").rules
    once = tt.mirror_batch(b, m, rules)
    twice = tt.mirror_batch(once, m, rules)
    for c in ("unit_feats", "spell_feats"):
        assert torch.equal(twice[c], b[c].float()), c
    for c in ("towers", "act_x", "unit_ids", "spell_ids", "own_pending", "own_pending_land"):
        assert torch.equal(twice[c], b[c]), c
    # own_pending (cards and land times) has nothing to flip; the spells' x and aim x flip
    assert once["own_pending"] is b["own_pending"]
    assert once["own_pending_land"] is b["own_pending_land"]
    sp = b["spell_feats"].float()
    live = (b["spell_ids"] != 0) & m[:, None]
    for k in range(6):
        want = torch.where(live, 1 - sp[..., k], sp[..., k]) if k in (1, 3) else sp[..., k]
        assert torch.equal(once["spell_feats"][..., k], want), k
    assert torch.equal(once["act_x"], torch.where(m & (b["act_card"] > 0), 17 - b["act_x"],
                                                  b["act_x"]))


def test_mirror_draws(tt):
    torch = tt.torch
    b = legacy_batch(torch)
    off = tt.Mirror(0.0, seed=3)
    state = off.rng.bit_generator.state
    assert off(b) is b and off.rng.bit_generator.state == state  # p 0 draws nothing
    every = tt.Mirror(1.0, seed=3)(b)
    want = tt.mirror_batch(b, torch.tensor([True, True]), tt.Mirror(1).rules)
    for c in want:
        assert torch.equal(every[c], want[c]), c
    a, c = tt.Mirror(0.5, seed=3), tt.Mirror(0.5, seed=3)
    for _ in range(3):
        x, y = a(b), c(b)
        for k in x:
            assert torch.equal(x[k], y[k]), k
    mask = np.random.default_rng([tt.MIRROR_SALT, 3]).random(2) < 0.5
    first = tt.Mirror(0.5, seed=3)(b)
    assert first["act_x"].tolist() == [14 if mask[0] else 3, 7]


# ---------------------------------------------------------------------------
# main, end to end on the CPU
# ---------------------------------------------------------------------------

VOCAB = {
    "n_cards": N_CARDS, "n_units": N_UNITS, "live_compat": False,
    "cards": {f"Card{i}": i + 2 for i in range(N_CARDS - 2)},
    "units": {f"Unit{i}": i + 2 for i in range(N_UNITS - 2)},
    "engine": "royalesim", "version": "0.0.0+test",
    "schema": "rs_rows/test", "obs_delay": {"ticks": 0, "jitter": 0},
    "wait_keep_at_build": 0.5,
}


@pytest.fixture
def smoke_rows(tmp_path):
    """test_train_tokens.py's smoke rows: two shards of 40, rows 0-29 train (0 and 1 untrusted),
    30-39 val, even rows Hog; per shard the val rows are 30-39 (4 waits, 6 plays), the Hog val
    rows 30, 32, 34, 36, 38 (3 plays); wait_keep_at_build 0.5, so every wait is kept at load.
    Train: 28 rows a shard, 56 in all."""
    root = tmp_path / "rows"
    i = np.arange(40)
    for k in (0, 1):
        write_shard(root, k, shard_arrays(40, start=100 * k, seed=10 + k,
                                          splits=["train"] * 30 + ["val"] * 10,
                                          trusted=i >= 2, hog=i % 2 == 0))
    (root / "vocab.json").write_text(json.dumps(VOCAB), encoding="utf-8")
    return root


def _run(tt, rows, runs, *extra):
    return tt.main(["--name", "smoke", "--data", str(rows), "--runs", str(runs), "--d", "16",
                    "--layers", "1", "--heads", "2", "--batch", "8", "--device", "cpu",
                    "--epochs", "1", *extra])


def _log(out):
    return [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8")
            .splitlines()]


def test_a_tiny_memmap_run_trains_on_the_same_rows(tt, smoke_rows, tmp_path, capsys):
    torch = tt.torch
    RC.compact(smoke_rows, log=quiet)
    store = smoke_rows / "compact"
    runs = tmp_path / "runs"
    assert _run(tt, store, runs, "--memmap", "--memmap-window", "16", "--mirror", "0.5") == 0
    text = capsys.readouterr().out
    assert "rows: train 56  val 20  val(hog) 10" in text and "--memmap)" in text
    out = runs / "smoke"
    (rec,) = _log(out)
    assert (rec["val"]["rows"], rec["val"]["plays"]) == (20, 12)        # as from the shards
    assert (rec["val_hog"]["rows"], rec["val_hog"]["plays"]) == (10, 6)
    args = json.loads((out / "args.json").read_text(encoding="utf-8"))
    assert (args["memmap"], args["mirror"], args["gpu_resident"]) == ("True", "0.5", "False")
    blob = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert blob["meta"] == dict(VOCAB, d=16, layers=1, heads=2)        # the rows' meta
    # the same rows from the shards, in RAM, with --mirror: the same validation rows
    assert _run(tt, smoke_rows, tmp_path / "ram", "--mirror", "1") == 0
    (ram,) = _log(tmp_path / "ram" / "smoke")
    assert (ram["val"]["rows"], ram["val_hog"]["rows"]) == (20, 10)


def test_main_refuses_what_does_not_fit_or_is_the_wrong_folder(tt, smoke_rows, tmp_path):
    RC.compact(smoke_rows, log=quiet)
    store = smoke_rows / "compact"
    runs = tmp_path / "runs"
    with pytest.raises(SystemExit, match=r"RAM budget: .*\(--val-max-rows"):
        _run(tt, store, runs, "--memmap", "--ram-budget-gb", "1e-9")
    with pytest.raises(SystemExit, match="train on it with --memmap"):
        _run(tt, store, runs)
    with pytest.raises(SystemExit, match="no compact.json"):
        _run(tt, smoke_rows, runs, "--memmap")
    with pytest.raises(SystemExit):                             # argparse: CUDA only
        _run(tt, store, runs, "--memmap", "--gpu-resident")
    for bad in (("--mirror", "1.5"), ("--val-max-rows", "-1"), ("--memmap-window", "0")):
        with pytest.raises(SystemExit):
            _run(tt, smoke_rows, runs, *bad)
    assert not runs.exists()                                    # refused before any output


def test_val_max_rows_caps_the_validation_sets(tt, smoke_rows, tmp_path):
    runs = tmp_path / "runs"
    assert _run(tt, smoke_rows, runs, "--val-max-rows", "8") == 0
    (rec,) = _log(runs / "smoke")
    assert rec["val"]["rows"] == 8 and rec["val_hog"]["rows"] == 8


def test_train_tokens_main_defaults_are_the_old_run():
    tree = _main(ROOT / "sim" / "train_tokens.py")
    assert _argument(tree, "--mirror")["default"].value == 0.0
    assert _argument(tree, "--val-max-rows")["default"].value == 0
    for flag in ("--memmap", "--gpu-resident"):
        assert _argument(tree, flag)["action"].value == "store_true", flag


@pytest.mark.gpu
def test_gpu_resident_and_pinned_batches(tt, smoke_rows, tmp_path):
    torch = tt.torch
    put = tt.PinnedStager("cuda")
    for n in (8, 5, 8):
        arrays = {"a": np.arange(n, dtype=np.int16), "b": np.ones((n, 3), dtype=np.float16)}
        got = put(arrays)
        assert got["a"].device.type == "cuda" and got["a"].tolist() == list(range(n))
        assert got["b"].shape == (n, 3)
    torch.cuda.synchronize()
    with pytest.raises(SystemExit, match="GPU budget"):
        tt.main(["--name", "g", "--data", str(smoke_rows), "--runs", str(tmp_path / "g"),
                 "--d", "16", "--layers", "1", "--heads", "2", "--batch", "8", "--epochs", "1",
                 "--gpu-resident", "--gpu-budget-gb", "1e-9"])
    assert not (tmp_path / "g").exists()
    RC.compact(smoke_rows, log=quiet)
    for extra in ((), ("--memmap",)):
        data = smoke_rows / "compact" if extra else smoke_rows
        assert tt.main(["--name", "r", "--data", str(data), "--runs", str(tmp_path / "r"),
                        "--d", "16", "--layers", "1", "--heads", "2", "--batch", "8",
                        "--epochs", "1", "--gpu-resident", *extra]) == 0
    assert math.isfinite(_log(tmp_path / "r" / "r")[-1]["val"]["card_acc"])
