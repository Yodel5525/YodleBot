r"""Read-only checks for the decision-time diagnosis (2026-10-30, docs/SPARRING_RUNBOOK.md section
19): what the rows on disk and rs_eval's log say about why dt_big_hog lost to live_big_hog.
numpy and tools/rs_elixir.py only: no engine, no model, nothing written.

    .venvs\royalegym\Scripts\python.exe tools\rs_dt_diag.py rows data\rs_rows_dt --trust v2
    .venvs\royalegym\Scripts\python.exe tools\rs_dt_diag.py rows data\rs_rows_live --shards 30
    .venvs\royalegym\Scripts\python.exe tools\rs_dt_diag.py ghost --delay 21

rows FOLDER: one rows folder of tools/rs_rows.py.
  1. manifest.json's counts: rows, play and wait rows, and for decision-time rows (E11) its
     "decision" counts as shares: pending_rows (rows with an own play in flight) of the rows,
     not_in_hand of the play rows, clamped taps.
  2. From --shards shards spread evenly over the folder (default 30; 0: every shard): the wait
     rows with an own play in flight (act_card 0 and an own_pending slot > 0; decision-time rows
     only, others have no such column), of all waits and of the trusted Hog 2.6 waits. These are
     the rows sim/train_tokens.py --drop-pending-waits leaves out.
  3. Elixir headroom of the trusted (--trust) Hog 2.6 play rows: the bar the row shows
     (scalars[0], the raw bar), less the cost of the side's own plays in flight (own_pending,
     when the rows have it), less the played card's cost (HOG_COSTS), in ticks of regeneration
     at the row's tick (rs_elixir's law: 56 ticks an elixir at 1x, 28 at 2x, 18.6 at 3x),
     rounded down. Rows whose bar is at the cap (>= CAP_MILLI) say nothing and are counted
     apart. A histogram in 5-tick bins, the shares below 0, 6, 15 and 21 ticks, and per rate;
     for decision-time rows also the share below TAP_FLOOR (-7) ticks. Every share is of the
     rows below the cap. --trust v2 picks the rows the decision-time models trained on
     (sim/train_tokens.py --trust v2); rows built with --trust legacy have no such column.

  How to read 3 (the meaning of IL_Replay's play tick, docs/SPARRING_PLAN.md):
  * rows at execution (data\rs_rows_live): if the recorded tick is the tick the play RAN, its
    player tapped it about 21 ticks earlier with the cost in the bar, so the bar at the recorded
    tick holds the cost plus about 21 ticks of regeneration: almost no row below 15-21. If the
    recorded tick is the TAP, the engine ran the play there, so the floor is 0 and many rows sit
    just above it.
  * decision-time rows (data\rs_rows_dt): each row sits 24 +- 3 ticks before the recorded
    tick, so the floor moves down by that much: about -6 if the recorded tick is the one the
    play ran (the row is then 0-6 ticks before the real tap), about -27 if it is the tap. So
    these rows answer the tick question too, without rows at execution: the "< -7" line is
    about 0 (a few rows at a rate change) if the recorded tick is the tick a play ran, and a
    large share (every play made within about 1 s of affording it) if it is the tap.
    "share < 0" is the share of play labels (below the cap; a capped row is short only with
    about 6 elixir of plays in flight) whose card rs_eval's decide masks at the tap (spendable
    elixir below the cost, tools/rs_decide.py): those labels push the model's probability into
    waits or a cheaper card.

ghost: rs_eval --pro (the recorded plays of both seats, no model) writes one line of
reports\rs_ghost.jsonl per run. Under --delay D it sends every recorded play D ticks early, so
that it still runs on its recorded tick. This compares the last undelayed --pro line with the
last one at --delay D, match by match (result, crowns, end tick): if the recorded tick is the
tick a play ran, the sends D ticks early are affordable and the two runs agree on nearly every
match; if it is the tap, every play made within about 1 s of being affordable is refused at
its early send and many matches diverge. Matches whose first play is before tick 90 + D differ
under either reading (the deploy lockout refuses those sends); this log cannot tell them apart,
so a few percent of disagreement is expected either way.

Exit code 0, or 2 for a folder without shards (rows) or a log without both runs (ghost).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import rs_elixir  # noqa: E402  (tools/ is sys.path[0] when this runs as a script)

ROOT = Path(__file__).resolve().parents[1]
GHOST_LOG = ROOT / "reports" / "rs_ghost.jsonl"
#: Hog 2.6's cards (tools/rs_rows.HOG26, catalogue names) and their elixir costs.
HOG_COSTS = {"HogRider": 4, "Musketeer": 4, "Cannon": 3, "Fireball": 4, "IceGolemite": 2,
             "IceSpirits": 1, "Skeletons": 1, "Log": 2}
#: A bar at or above this (milli-elixir) is at the cap: its headroom says nothing.
CAP_MILLI = 9900
#: The thresholds the shares are printed for, in ticks of regeneration.
THRESHOLDS = (0, 6, 15, 21)
#: Decision-time rows: a play row more ticks short than this says the recorded tick is the tap
#: (module doc, How to read 3). With execution ticks the floor is about -6: the row is at most
#: 27 - 21 ticks before the real tap.
TAP_FLOOR = -7
#: Histogram bins: 5 ticks wide from LOW to HIGH, plus one bin below and one above.
BIN, LOW, HIGH = 5, -30, 60
DEFAULT_SHARDS = 30
TRUST_COLUMNS = {"legacy": "trusted", "v2": "trusted_v2"}
PENDING_COLUMN = "own_pending"
PRESS_COLUMN = "act_press"
PRO_NAME = "recorded pro"


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def price_table(cards: Mapping[str, int], costs: Mapping[str, int] = HOG_COSTS) -> np.ndarray:
    """Elixir cost by row token (vocab.json "cards": catalogue name -> id + 2), 0 for a token
    whose card is not in `costs`; one entry past the largest token."""
    tokens = [int(t) for t in cards.values()] or [0]
    price = np.zeros(max(tokens) + 2, dtype=np.int64)
    for name, cost in costs.items():
        if name in cards:
            price[int(cards[name])] = int(cost)
    return price


def priced(price: np.ndarray, tokens) -> np.ndarray:
    """price[tokens], 0 for a token outside the table (or below 0)."""
    t = np.asarray(tokens, dtype=np.int64)
    ok = (t >= 0) & (t < len(price))
    return np.where(ok, price[np.clip(t, 0, len(price) - 1)], 0)


def milli_of(scalars) -> np.ndarray:
    """The own bar in milli-elixir from a rows' scalars (column 0 is elixir_milli / 10000)."""
    s = np.asarray(scalars, dtype=np.float64)
    return np.rint(s[:, 0] * 10_000).astype(np.int64)


def rate_of(ticks, law: rs_elixir.ElixirLaw) -> np.ndarray:
    """The elixir rate (1, 2 or 3) of the tick that runs from each tick: law.rate_at, for an
    array of ticks."""
    t = np.asarray(ticks, dtype=np.int64).reshape(-1)
    rate = np.where(t >= law.double_from, 2, 1)
    if law.triple_from is not None:
        rate = np.where(t >= law.triple_from, 3, rate)
    return rate.astype(np.int64)


def headroom_ticks(milli, cost, pending_cost, ticks, law: rs_elixir.ElixirLaw) -> np.ndarray:
    """(bar - pending_cost - cost) in whole ticks of regeneration at each row's rate, rounded
    down (negative: short of the cost). `milli` the bar in milli-elixir, the costs in elixir.
    Integer arithmetic: a milli is law.scale / 1000 fine units, a tick law.gain(rate) of them."""
    milli = np.asarray(milli, dtype=np.int64).reshape(-1)
    spare = milli - 1000 * (np.asarray(cost, dtype=np.int64).reshape(-1)
                            + np.asarray(pending_cost, dtype=np.int64).reshape(-1))
    gains = np.asarray([0, law.gain(1), law.gain(2), law.gain(3)], dtype=np.int64)
    return np.floor_divide(spare * int(law.scale), 1000 * gains[rate_of(ticks, law)])


def pick_shards(paths: Sequence[Path], n: int) -> list:
    """At most `n` of `paths` spread evenly (first and last included); every one for n <= 0."""
    paths = list(paths)
    if n <= 0 or n >= len(paths):
        return paths
    idx = np.unique(np.rint(np.linspace(0, len(paths) - 1, n)).astype(int))
    return [paths[k] for k in idx]


class Scan:
    """What `rows` adds up over the shards it reads."""

    def __init__(self):
        self.shards = 0
        self.rows = 0
        self.waits = 0
        self.pending_waits = 0
        self.hog_waits = 0
        self.hog_pending_waits = 0
        self.has_pending = False
        self.plays = 0                  # trusted Hog 2.6 play rows of a priced card
        self.capped = 0
        self.with_pending = 0           # of those, rows with an own play in flight
        self.head: list = []            # their headroom, ticks
        self.rate: list = []            # and the elixir rate at their tick

    def add(self, z, trust_col: str, price: np.ndarray, law: rs_elixir.ElixirLaw) -> None:
        files = set(getattr(z, "files", z))
        act = np.asarray(z["act_card"]).astype(np.int64)
        n = len(act)
        self.shards += 1
        self.rows += n
        trusted = np.asarray(z[trust_col]).astype(bool)
        hog = np.asarray(z["hog"]).astype(bool)
        waits = act == 0
        if PRESS_COLUMN in files:
            waits &= ~(np.asarray(z[PRESS_COLUMN]) > 0)
        pend_cost = np.zeros(n, dtype=np.int64)
        in_flight = np.zeros(n, dtype=bool)
        if PENDING_COLUMN in files:
            self.has_pending = True
            own = np.asarray(z[PENDING_COLUMN]).astype(np.int64).reshape(n, -1)
            in_flight = (own > 0).any(axis=1)
            pend_cost = priced(price, own).sum(axis=1)
        self.waits += int(waits.sum())
        self.pending_waits += int((waits & in_flight).sum())
        hw = waits & trusted & hog
        self.hog_waits += int(hw.sum())
        self.hog_pending_waits += int((hw & in_flight).sum())
        cost = priced(price, act)
        m = trusted & hog & (act > 0) & (cost > 0)
        milli = milli_of(z["scalars"])
        cap = m & (milli >= CAP_MILLI)
        self.capped += int(cap.sum())
        m &= ~cap
        self.plays += int(m.sum())
        self.with_pending += int((m & in_flight).sum())
        ticks = np.asarray(z["tick"]).astype(np.int64)
        if m.any():
            self.head.append(headroom_ticks(milli[m], cost[m], pend_cost[m], ticks[m], law))
            self.rate.append(rate_of(ticks[m], law))

    def headroom(self) -> tuple:
        """(headroom ticks, rates) of every play row counted."""
        if not self.head:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return np.concatenate(self.head), np.concatenate(self.rate)


def scan(root, shards: int = DEFAULT_SHARDS, trust: str = "legacy",
         law: rs_elixir.ElixirLaw | None = None) -> Scan:
    """Scan of `shards` shards of the rows in `root` (pick_shards). FileNotFoundError for a
    folder without shards or vocab.json; ValueError for --trust v2 on rows without the column."""
    root = Path(root)
    paths = sorted(root.glob("shard-*.npz"))
    if not paths:
        raise FileNotFoundError(f"no shard-*.npz in {root}")
    vocab_path = root / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"no vocab.json in {root}")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8-sig"))
    price = price_table(vocab.get("cards") or {})
    law = law or rs_elixir.default_law()
    col = TRUST_COLUMNS[trust]
    out = Scan()
    for path in pick_shards(paths, shards):
        with np.load(path) as z:
            if col not in z.files:
                raise ValueError(f"{path.name} has no {col} column (--trust {trust}); use "
                                 f"--trust legacy")
            out.add(z, col, price, law)
    return out


def share_below(head: np.ndarray, value: int) -> float:
    return float((head < value).mean()) if len(head) else 0.0


def histogram(head: np.ndarray) -> list:
    """[(label, count)] of `head` in BIN-tick bins from LOW to HIGH, with an open bin each side."""
    out = [(f"< {LOW}", int((head < LOW).sum()))]
    for a in range(LOW, HIGH, BIN):
        out.append((f"[{a}, {a + BIN})", int(((head >= a) & (head < a + BIN)).sum())))
    out.append((f">= {HIGH}", int((head >= HIGH).sum())))
    return out


def pct(k, n) -> str:
    return f"{100.0 * k / n:.1f}%" if n else "-"


def manifest_lines(root) -> list:
    """The manifest.json lines of `rows` (none when the folder has no manifest.json)."""
    path = Path(root) / "manifest.json"
    if not path.exists():
        return [f"manifest: none in {root} (an unfinished rows run?)"]
    man = json.loads(path.read_text(encoding="utf-8-sig"))
    rows, plays = int(man.get("rows", 0)), int(man.get("play_rows", 0))
    lines = [f"manifest: {rows:,} rows, {plays:,} play rows, "
             f"{int(man.get('wait_rows', rows - plays)):,} wait rows"]
    d = man.get("decision")
    if isinstance(d, Mapping):
        dp = int(d.get("play_rows", 0))
        lines.append(f"decision: {int(d.get('pending_rows', 0)):,} rows with an own play in flight "
                     f"({pct(d.get('pending_rows', 0), rows)} of the rows); not in hand "
                     f"{int(d.get('not_in_hand', 0)):,} ({pct(d.get('not_in_hand', 0), dp)} of "
                     f"{dp:,} play rows); clamped {int(d.get('clamped', 0)):,}; exact hand "
                     f"{int(d.get('exact_rows', 0)):,} ({pct(d.get('exact_rows', 0), rows)})")
    else:
        lines.append("decision: none (rows at execution, --obs-delay 0)")
    return lines


def report(root, s: Scan, trust: str) -> list:
    """The lines `rows` prints for scan `s` of `root`."""
    lines = [f"rows: {root}: {s.shards} shard(s) read, {s.rows:,} rows"]
    if s.has_pending:
        lines.append(f"pending waits: {s.pending_waits:,} of {s.waits:,} wait rows "
                     f"({pct(s.pending_waits, s.waits)}) have an own play in flight; trusted "
                     f"Hog 2.6: {s.hog_pending_waits:,} of {s.hog_waits:,} "
                     f"({pct(s.hog_pending_waits, s.hog_waits)})")
    else:
        lines.append("pending waits: no own_pending column (rows at execution)")
    head, rate = s.headroom()
    lines.append(f"headroom: {s.plays:,} trusted ({trust}) Hog 2.6 play rows"
                 + (f", {s.with_pending:,} of them with an own play in flight" if s.has_pending
                    else "")
                 + f"; {s.capped:,} more at the cap left out")
    if len(head):
        lines.append("headroom: " + "  ".join(f"< {v}: {share_below(head, v):.3f}"
                                                for v in THRESHOLDS)
                     + f"  (min {int(head.min())}, median {int(np.median(head))} ticks)")
        if s.has_pending:
            lines.append(f"headroom: < {TAP_FLOOR}: {share_below(head, TAP_FLOOR):.3f} of the "
                         "play rows (decision-time rows: about 0 if the recorded tick is the "
                         "tick a play ran, a large share if it is the tap)")
        for r in (1, 2, 3):
            sub = head[rate == r]
            if len(sub):
                lines.append(f"headroom at {r}x: {len(sub):,} rows, < 0: "
                             f"{share_below(sub, 0):.3f}, < 21: {share_below(sub, 21):.3f}")
        lines.append("headroom histogram (ticks: rows):")
        for label, k in histogram(head):
            lines.append(f"  {label:>10}: {k:>9,} ({pct(k, len(head))})")
    return lines


# ---------------------------------------------------------------------------
# ghost
# ---------------------------------------------------------------------------

def run_delay(record: Mapping[str, Any]) -> int:
    """The command delay of an rs_eval line (its engine config's "delay"; None or absent: 0)."""
    delay = (record.get("engine") or {}).get("delay")
    return int(delay[0]) if delay else 0


def last_pro(records: Iterable[Mapping[str, Any]], delay: int):
    """The last rs_eval --pro line run at `delay`, or None."""
    found = None
    for rec in records:
        if rec.get("name") == PRO_NAME and run_delay(rec) == int(delay):
            found = rec
    return found


def compare_runs(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict:
    """Match-by-match agreement of two rs_eval lines' rows, on the tags both played: counts of
    "matches", "result", "crowns", "end_tick" (the same in both) and "all" (all three)."""
    rows_a = {r["tag"]: r for r in a.get("rows") or ()}
    rows_b = {r["tag"]: r for r in b.get("rows") or ()}
    tags = sorted(set(rows_a) & set(rows_b))
    out = {"matches": len(tags), "result": 0, "crowns": 0, "end_tick": 0, "all": 0}
    for t in tags:
        x, y = rows_a[t], rows_b[t]
        same = {"result": x.get("result") == y.get("result"),
                "crowns": list(x.get("crowns") or ()) == list(y.get("crowns") or ()),
                "end_tick": x.get("end_tick") == y.get("end_tick")}
        for k, v in same.items():
            out[k] += int(v)
        out["all"] += int(all(same.values()))
    return out


def read_log(path) -> list:
    """The JSON lines of an rs_eval log (blank or broken lines skipped)."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def ghost_lines(records: list, delay: int) -> list | None:
    """The lines `ghost` prints, or None when the log lacks one of the two runs."""
    base, late = last_pro(records, 0), last_pro(records, delay)
    if base is None or late is None:
        return None
    c = compare_runs(base, late)
    n = c["matches"]
    return [f"ghost: recorded pro, no delay ({base.get('wins')}/{base.get('n')} wins) vs "
            f"--delay {delay} ({late.get('wins')}/{late.get('n')} wins), {n} matches in both",
            f"ghost: same result {c['result']} ({pct(c['result'], n)}), same crowns "
            f"{c['crowns']} ({pct(c['crowns'], n)}), same end tick {c['end_tick']} "
            f"({pct(c['end_tick'], n)}), all three {c['all']} ({pct(c['all'], n)})"]


# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows", help="pending waits and elixir headroom of a rows folder")
    r.add_argument("folder", type=Path)
    r.add_argument("--shards", type=int, default=DEFAULT_SHARDS,
                   help=f"shards to read, spread evenly (default {DEFAULT_SHARDS}; 0: all)")
    r.add_argument("--trust", choices=tuple(TRUST_COLUMNS), default="legacy",
                   help="which trusted column picks the play rows (default legacy: `trusted`, "
                        "which every rows folder has)")
    g = sub.add_parser("ghost", help="rs_eval --pro with and without --delay, match by match")
    g.add_argument("--log", type=Path, default=GHOST_LOG)
    g.add_argument("--delay", type=int, default=21)
    args = ap.parse_args(argv)
    if args.cmd == "rows":
        try:
            s = scan(args.folder, args.shards, args.trust)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for line in manifest_lines(args.folder) + report(args.folder, s, args.trust):
            print(line)
        return 0
    if not args.log.exists():
        print(f"error: no {args.log}: run tools\\rs_eval.py --pro first", file=sys.stderr)
        return 2
    lines = ghost_lines(read_log(args.log), args.delay)
    if lines is None:
        print(f"error: {args.log} needs an rs_eval --pro line without a delay and one at "
              f"--delay {args.delay}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
