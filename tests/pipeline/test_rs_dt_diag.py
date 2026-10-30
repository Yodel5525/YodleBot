"""tools/rs_dt_diag.py: the read-only checks of the decision-time diagnosis (runbook section 19).

* headroom in whole ticks of regeneration, by rs_elixir's law, rounded down, with the cost of
  the side's own plays in flight;
* rows: which rows count (trusted Hog 2.6 plays of a priced card, the bar not at the cap), the
  waits with an own play in flight (press rows are no waits), and the lines it prints;
* manifest.json's decision counts, and rows at execution (no own_pending column);
* ghost: the last rs_eval --pro line without a delay against the last one at --delay D, match by
  match.

Needs numpy only; no engine, data or GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.append(str(TOOLS))
import rs_dt_diag as DD  # noqa: E402
import rs_elixir  # noqa: E402

LAW = rs_elixir.ElixirLaw.constants()
#: vocab.json "cards" of the synthetic rows: catalogue name -> id + 2.
CARDS = {"Knight": 3, "HogRider": 5, "Musketeer": 6, "Log": 9}
HOG, LOG, KNIGHT = 5, 9, 3


def test_headroom_ticks_by_hand():
    # 1x: 500 fine units a tick, 28000 an elixir: 375 milli is 21 ticks
    cases = [  # (milli, cost, pending cost, tick, ticks)
        (4000, 4, 0, 100, 0),
        (4375, 4, 0, 100, 21),
        (4374, 4, 0, 100, 20),
        (3999, 4, 0, 100, -1),        # rounded down: short by under a tick is -1
        (4375, 4, 0, 2400, 10),       # 2x from tick 2400
        (4375, 4, 0, 4800, 6),        # 3x from 4800: 1505 fine units a tick
        (7000, 4, 3, 100, 0),         # a 3-elixir play in flight counts against the bar
        (3500, 4, 0, 2399, -28),
    ]
    milli, cost, pend, tick, want = (np.asarray(c) for c in zip(*cases))
    assert DD.headroom_ticks(milli, cost, pend, tick, LAW).tolist() == want.tolist()
    assert DD.rate_of([0, 2399, 2400, 4799, 4800, 6000], LAW).tolist() == [1, 1, 2, 2, 3, 3]
    assert [LAW.rate_at(t) for t in (0, 2399, 2400, 4799, 4800, 6000)] == [1, 1, 2, 2, 3, 3]


def test_prices_and_the_bar():
    price = DD.price_table(CARDS)
    assert len(price) == 11 and price[HOG] == 4 and price[LOG] == 2 and price[KNIGHT] == 0
    assert DD.priced(price, [5, 9, 3, 0, 40, -1]).tolist() == [4, 2, 0, 0, 0, 0]
    assert DD.priced(price, [[9, 5, 0, 0]]).sum(axis=1).tolist() == [6]
    scalars = (np.asarray([[4375, 0, 0, 0], [6000, 0, 0, 0], [1, 0, 0, 0]]) / 10_000
               ).astype(np.float32)
    assert DD.milli_of(scalars).tolist() == [4375, 6000, 1]
    assert DD.HOG_COSTS == {"HogRider": 4, "Musketeer": 4, "Cannon": 3, "Fireball": 4,
                            "IceGolemite": 2, "IceSpirits": 1, "Skeletons": 1, "Log": 2}


def test_pick_shards_spreads_them():
    paths = [Path(f"shard-{k:04d}.npz") for k in range(10)]
    assert [p.name[6:10] for p in DD.pick_shards(paths, 3)] == ["0000", "0004", "0009"]
    assert DD.pick_shards(paths, 0) == paths and DD.pick_shards(paths, 20) == paths


#: The synthetic shard, row by row: (act_card, tick, bar milli, hog, trusted, own_pending,
#: act_press). Counted plays: 0 (21 ticks), 1 (0), 2 (-14 at 2x), 10 (56, a Log in flight).
ROWS = [
    (HOG, 100, 4375, True, True, 0, 0),       # 0
    (LOG, 100, 2000, True, True, 0, 0),       # 1
    (HOG, 2400, 3500, True, True, 0, 0),      # 2
    (HOG, 100, 10000, True, True, 0, 0),      # 3 at the cap
    (HOG, 100, 5000, True, False, 0, 0),      # 4 untrusted
    (HOG, 100, 5000, False, True, 0, 0),      # 5 not Hog 2.6
    (KNIGHT, 100, 5000, True, True, 0, 0),    # 6 no price
    (0, 120, 3000, True, True, HOG, 0),       # 7 a pending wait
    (0, 140, 3000, True, True, 0, 0),         # 8 a wait
    (0, 160, 3000, False, True, LOG, 0),      # 9 a pending wait, not Hog 2.6
    (HOG, 100, 7000, True, True, LOG, 0),     # 10 a play with a Log in flight
    (0, 180, 3000, True, True, 0, 1),         # 11 a press row: no wait
]


def write_rows(root: Path, pending: bool = True, press: bool = True, v2: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    act, tick, milli, hog, trusted, own, presses = (np.asarray(c) for c in zip(*ROWS))
    n = len(ROWS)
    arrays = {
        "act_card": act.astype(np.int16), "tick": tick.astype(np.int64),
        "scalars": np.column_stack([milli / 10_000, np.zeros((n, 3))]).astype(np.float32),
        "hog": hog.astype(bool), "trusted": trusted.astype(bool),
    }
    if pending:
        slots = np.zeros((n, 4), dtype=np.int16)
        slots[:, 0] = own
        arrays["own_pending"] = slots
    if press:
        arrays["act_press"] = presses.astype(np.int16)
    if v2:
        arrays["trusted_v2"] = trusted.astype(bool)
    np.savez_compressed(root / "shard-0000.npz", **arrays)
    (root / "vocab.json").write_text(json.dumps({"cards": CARDS}), encoding="utf-8")
    return root


def test_scan_counts_what_its_doc_says(tmp_path):
    s = DD.scan(write_rows(tmp_path / "dt"), law=LAW)
    assert (s.shards, s.rows, s.has_pending) == (1, 12, True)
    assert (s.waits, s.pending_waits, s.hog_waits, s.hog_pending_waits) == (3, 2, 2, 1)
    assert (s.plays, s.capped, s.with_pending) == (4, 1, 1)
    head, rate = s.headroom()
    assert head.tolist() == [21, 0, -14, 56] and rate.tolist() == [1, 1, 2, 1]


def test_report_lines(tmp_path):
    root = write_rows(tmp_path / "dt")
    text = "\n".join(DD.report(root, DD.scan(root, law=LAW), "legacy"))
    assert "1 shard(s) read, 12 rows" in text
    assert ("pending waits: 2 of 3 wait rows (66.7%) have an own play in flight; trusted Hog "
            "2.6: 1 of 2 (50.0%)") in text
    assert ("headroom: 4 trusted (legacy) Hog 2.6 play rows, 1 of them with an own play in "
            "flight; 1 more at the cap left out") in text
    assert "< 0: 0.250  < 6: 0.500  < 15: 0.500  < 21: 0.500  (min -14, median 10 ticks)" in text
    assert "headroom at 1x: 3 rows, < 0: 0.000, < 21: 0.333" in text
    assert "headroom at 2x: 1 rows, < 0: 1.000, < 21: 1.000" in text
    assert "at 3x" not in text
    bins = dict(DD.histogram(np.asarray([21, 0, -14, 56, -31, 60])))
    assert (bins["< -30"], bins["[-15, -10)"], bins["[0, 5)"], bins["[20, 25)"],
            bins["[55, 60)"], bins[">= 60"]) == (1, 1, 1, 1, 1, 1)
    assert sum(bins.values()) == 6


def test_rows_at_execution(tmp_path, capsys):
    root = write_rows(tmp_path / "live", pending=False, press=False)
    s = DD.scan(root, law=LAW)
    assert not s.has_pending and s.pending_waits == 0 and s.waits == 4    # row 11 is a wait now
    head, _ = s.headroom()
    # row 10 has no Log in flight now: 7000 - 4000 = 3000 milli, 168 ticks
    assert head.tolist() == [21, 0, -14, 168]
    assert DD.main(["rows", str(root)]) == 0
    out = capsys.readouterr().out
    assert "pending waits: no own_pending column (rows at execution)" in out
    assert "manifest: none in" in out


def test_manifest_lines(tmp_path):
    root = write_rows(tmp_path / "dt")
    (root / "manifest.json").write_text(json.dumps({
        "rows": 100, "play_rows": 20, "wait_rows": 80,
        "decision": {"play_rows": 20, "press_rows": 0, "pending_rows": 18, "not_in_hand": 1,
                     "clamped": 0, "exact_rows": 50}}), encoding="utf-8")
    assert DD.manifest_lines(root) == [
        "manifest: 100 rows, 20 play rows, 80 wait rows",
        "decision: 18 rows with an own play in flight (18.0% of the rows); not in hand 1 (5.0% "
        "of 20 play rows); clamped 0; exact hand 50 (50.0%)"]
    (root / "manifest.json").write_text(json.dumps({"rows": 10, "play_rows": 4}),
                                        encoding="utf-8")
    assert DD.manifest_lines(root) == ["manifest: 10 rows, 4 play rows, 6 wait rows",
                                       "decision: none (rows at execution, --obs-delay 0)"]


def test_trust_and_missing_folders(tmp_path, capsys):
    root = write_rows(tmp_path / "dt")
    with pytest.raises(ValueError, match="no trusted_v2 column"):
        DD.scan(root, trust="v2", law=LAW)
    assert DD.main(["rows", str(root), "--trust", "v2"]) == 2
    v2 = write_rows(tmp_path / "v2", v2=True)
    assert DD.scan(v2, trust="v2", law=LAW).plays == 4
    assert DD.main(["rows", str(tmp_path / "nothing")]) == 2
    assert "no shard-*.npz" in capsys.readouterr().err


def run(tags_results, delay):
    return {"name": DD.PRO_NAME, "engine": {"delay": None if delay is None else [delay, delay]},
            "wins": sum(r == "win" for _, r, _, _ in tags_results), "n": len(tags_results),
            "rows": [{"tag": t, "result": r, "crowns": list(c), "end_tick": e}
                     for t, r, c, e in tags_results]}


def test_ghost_compares_the_last_runs(tmp_path, capsys):
    base = run([("a", "win", (1, 0), 3600), ("b", "loss", (0, 1), 3600),
                ("c", "win", (3, 0), 2000)], None)
    old = run([("a", "loss", (0, 1), 3600)], 21)
    late = run([("a", "win", (1, 0), 3600), ("b", "loss", (0, 2), 3600),
                ("c", "win", (3, 0), 2100), ("d", "win", (1, 0), 3600)], 21)
    model = dict(run([("a", "loss", (0, 1), 3600)], 21), name="dt_big_hog delay 21 vs ghost")
    assert DD.run_delay(base) == 0 and DD.run_delay({"engine": {"delay": [0, 0]}}) == 0
    assert DD.last_pro([base, old, late, model], 21) is late
    assert DD.compare_runs(base, late) == {"matches": 3, "result": 3, "crowns": 2,
                                           "end_tick": 2, "all": 1}
    log = tmp_path / "rs_ghost.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in (base, old, late, model)) + "\n\nnot json\n",
                   encoding="utf-8")
    assert DD.main(["ghost", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "no delay (2/3 wins) vs --delay 21 (3/4 wins), 3 matches in both" in out
    assert ("same result 3 (100.0%), same crowns 2 (66.7%), same end tick 2 (66.7%), all three "
            "1 (33.3%)") in out
    assert DD.main(["ghost", "--log", str(log), "--delay", "24"]) == 2
    assert DD.main(["ghost", "--log", str(tmp_path / "none.jsonl")]) == 2
