r"""RoyaleSim fidelity grid (E03): the same recorded matches re-driven under each engine setting,
and how often each ending agrees with the recording.

    .venvs\royalegym\Scripts\python.exe tools\rs_fidelity_grid.py
    .venvs\royalegym\Scripts\python.exe tools\rs_fidelity_grid.py --sample 2000 --split val ^
        --seed 0 --tables 16.402.22,15.535 --ticks exec,tap --presses off,on
    .venvs\royalegym\Scripts\python.exe tools\rs_fidelity_grid.py --dry-run      the commands only

Why (rsim-01, rsim-03, tokens-01): every fidelity number so far (winner agreement 62.6%, exact
crowns 49.0% on the author's b2000 matches) was measured on royalesim 0.1.22, and about 120
calibration values changed in 0.1.42; the card table is now a choice (15.535 or 16.402.22),
ability presses can be replayed (E10), and whether IL_Replay's play ticks are execution ticks
or tap ticks is an open question. Phase 2 of docs/SPARRING_PLAN.md picks the card table and
settings from this grid. The grid's exec / tap axis says little about that question: moving
every play by the same ticks keeps the board's timing, so the endings and the elixir refusals
hardly change. tools/rs_dt_diag.py's elixir headroom of the rows (and its ghost comparison of
rs_eval --pro with and without --delay) is the check (docs/SPARRING_RUNBOOK.md section 19).

What it does:

1. The matches. --sample N --split S --seed K (default 2000, val, 0) picks N corpus matches by
   tag hash, royalesim_redrive.sample_tags (the ranked modes of tools/rs_rows.py unless
   --modes), and writes their payloads once to <work>/payloads.jsonl (needs pyarrow, so run the
   grid with the engine venv's Python). --payloads FILE uses a file royalesim_redrive.py
   --save-payloads wrote instead. <work> defaults to <runs root>/fidelity_grid/<sample name>
   (src/hastycr/machine.py; runs/ is gitignored).
2. The cells: every --tables x --ticks x --presses combination on the engine interpreter
   (--python, default the machine's py_engine), then, when a royalesim 0.1.22 venv exists
   (--rs022 auto: .venvs/rs022, through hastycr.machine.venv_python; or an interpreter path;
   "off"), every --ticks x --presses combination on it with the engine's own card table (0.1.22
   has no card-table choice). Each cell is one subprocess, run one after the other (16 GB RAM):
       <python> tools/royalesim_redrive.py --payloads <work>/payloads.jsonl
           --out <work>/<cell>.jsonl --pair <cell> --card-table <table> --tick-semantics <ticks>
           --tap-delay <d> --presses <presses> --workers <w>
   with its console in <work>/<cell>.log. A cell that exits non-zero is reported as failed and
   the grid goes on. --reuse keeps a cell whose output and meta are there and match the
   payload file and the cell's table, ticks, tap delay and presses (not the interpreter).
3. The report, reports/fidelity_grid.md (and the numbers as reports/fidelity_grid.json): per
   cell, matches run, winner and exact-crown agreement with 95% Wilson intervals
   (tools/league/stats.wilson), refusals per 100 plays (all reasons, and by reason), 3-crown
   endings in the engine and in the recording, and the difference to the first cell (the base)
   on the matches both ran, with an exact McNemar p (tools/league/stats.mcnemar); the same
   per tower troop (both sides' tower_card; friendlies are Tower Princess on both sides) and
   per game mode; failed matches by error; ability presses; card levels in the sample; and
   each cell's engine fingerprint. The best cell by winner agreement is named against the
   62.6% target.

Exit codes: 0 every cell ran; 1 a cell failed; 2 a bad argument, or no match to re-drive.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for _path in (str(ROOT), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import royalesim_redrive as RD  # noqa: E402  (standard library; pyarrow only to pick the sample)
import rs_cards  # noqa: E402
import rs_engine  # noqa: E402
import rs_presses  # noqa: E402
from tools.league import stats as LS  # noqa: E402  (numpy)

SCHEMA = 1
TOOL = "tools/rs_fidelity_grid.py"
REDRIVE = TOOLS / "royalesim_redrive.py"
REPORT_MD = ROOT / "reports" / "fidelity_grid.md"
#: RoyaleSim 0.1.22 on the author's b2000 matches (docs/PROGRESS_2026-10-09.md): phase 2 of
#: docs/SPARRING_PLAN.md wants a setting at this winner agreement or better.
TARGET_WINNER = 0.626
TARGET_CROWNS = 0.490
DEFAULT_SAMPLE = 2000
DEFAULT_TABLE = "default"
DEFAULT_TABLES = (rs_engine.TARGET_TABLE, "client15535")
MAIN, RS022 = "main", "rs022"
RS022_VENV = Path(".venvs") / "rs022"
#: The tower troop of both seats in the owner's friendlies (pc_observations, 2026-10-29).
FRIENDLY_TOWER = "tower-princess"
UNKNOWN = "unknown"
LOG_TAIL = 15


# ---------------------------------------------------------------------------
# settings and cells
# ---------------------------------------------------------------------------

def split_list(text) -> list[str]:
    """'a, b,,c' -> ['a', 'b', 'c']."""
    return [p.strip() for p in str(text or "").split(",") if p.strip()]


def parse_choices(text, choices: Sequence[str], what: str) -> list[str]:
    """A comma-separated subset of `choices`, in the order given, without repeats."""
    out: list[str] = []
    for item in split_list(text):
        if item not in choices:
            raise ValueError(f"{what}: {item!r} is not one of {', '.join(choices)}")
        if item not in out:
            out.append(item)
    if not out:
        raise ValueError(f"{what}: at least one of {', '.join(choices)}")
    return out


def parse_tables(text) -> list[str]:
    """--tables: card-table names as tools/rs_engine.py takes them (arms, content versions such
    as 16.402.22, or 'default' for the engine's own), as arm names or DEFAULT_TABLE."""
    out: list[str] = []
    for item in split_list(text):
        arm = rs_engine.table_name(item)
        name = DEFAULT_TABLE if arm is None else arm
        if name not in out:
            out.append(name)
    if not out:
        raise ValueError("--tables: at least one card table (e.g. 16.402.22,15.535 or default)")
    return out


def table_label(table) -> str:
    """A card table as a short label: its longest content-version alias ("16.402.22",
    "15.535.29"), DEFAULT_TABLE for the engine's own, else the arm name."""
    arm = rs_engine.table_name(table)
    if arm is None:
        return DEFAULT_TABLE
    best = ""
    for alias, name in rs_engine.TABLE_ALIASES.items():
        if name == arm and len(alias) > len(best):
            best = alias
    return best or arm


@dataclasses.dataclass(frozen=True)
class Cell:
    """One re-drive of the grid: interpreter label and path, card table (an arm or "default"),
    tick semantics and presses mode."""
    interp: str
    python: str
    table: str
    ticks: str
    presses: str

    @property
    def name(self) -> str:
        return f"{self.interp}-{table_label(self.table)}-{self.ticks}-presses-{self.presses}"

    def settings(self) -> dict[str, str]:
        return {"interp": self.interp, "python": str(self.python), "table": self.table,
                "ticks": self.ticks, "presses": self.presses}


def grid_cells(tables: Sequence[str], ticks: Sequence[str], presses: Sequence[str],
               python, rs022=None) -> list[Cell]:
    """Every table x ticks x presses cell on `python` (the first is the base), then every
    ticks x presses cell on `rs022` (when given) with the engine's own table."""
    cells = [Cell(MAIN, str(python), table, tick, press)
             for table in tables for tick in ticks for press in presses]
    if rs022:
        cells += [Cell(RS022, str(rs022), DEFAULT_TABLE, tick, press)
                  for tick in ticks for press in presses]
    return cells


def cell_command(cell: Cell, payloads, out, *, workers: int = 6,
                 tap_delay: int = RD.TAP_DELAY, script: Path = REDRIVE) -> list[str]:
    """The argv that runs `cell` (module doc, step 2)."""
    return [str(cell.python), str(script), "--payloads", str(payloads), "--out", str(out),
            "--pair", cell.name, "--card-table", cell.table, "--tick-semantics", cell.ticks,
            "--tap-delay", str(int(tap_delay)), "--presses", cell.presses,
            "--workers", str(int(workers))]


def _machine():
    """src/hastycr/machine.py (standard library only)."""
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from hastycr import machine
    return machine


def default_python() -> str:
    """The machine's engine interpreter (py_engine) when it exists, else this one."""
    try:
        path = Path(_machine().py_engine())
        if path.is_file():
            return str(path)
    except Exception:                       # a broken machine.json: this interpreter
        pass
    return sys.executable


def rs022_python(choice: str | None = "auto", *, root: Path = ROOT) -> Path | None:
    """--rs022: "off" -> None; "auto" -> <root>/.venvs/rs022's interpreter
    (hastycr.machine.venv_python) when it exists, else None; anything else is an interpreter
    path, which must exist."""
    text = str(choice or "auto").strip()
    if text.lower() == "off":
        return None
    if text.lower() == "auto":
        path = Path(_machine().venv_python(Path(root) / RS022_VENV))
        return path if path.is_file() else None
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"--rs022 {text}: no such interpreter")
    return path


def default_work(name: str) -> Path:
    """<runs root>/fidelity_grid/<name> (runs root: src/hastycr/machine.py, else runs/)."""
    try:
        runs = Path(_machine().runs_root())
    except Exception:
        runs = ROOT / "runs"
    return runs / "fidelity_grid" / name


def default_corpus() -> Path:
    """<data root>/il_replay (src/hastycr/machine.py, else data/il_replay)."""
    try:
        return Path(_machine().data_root()) / "il_replay"
    except Exception:
        return RD.CORPUS


# ---------------------------------------------------------------------------
# statistics (pure)
# ---------------------------------------------------------------------------

def rate(k: int, n: int) -> dict[str, Any]:
    """k of n with its 95% Wilson interval; rate None when n is 0."""
    lo, hi = LS.wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "lo": lo, "hi": hi}


def per100(count: float, plays: int) -> float:
    """Refusals per 100 plays, as royalesim_redrive.main prints them."""
    return 100.0 * count / max(1, int(plays))


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """royalesim_redrive.totals(rows) plus: winner and crowns (rate), refused_total,
    refusals_per_100 and refusals_per_100_by_reason, and presses (press_totals of the matches
    that ran)."""
    tot = RD.totals(rows)
    n = tot["ran"]
    refused_total = sum(tot["refused"].values())
    return {**tot,
            "winner": rate(tot["winner_agree"], n),
            "crowns": rate(tot["crowns_agree"], n),
            "refused_total": refused_total,
            "refusals_per_100": per100(refused_total, tot["plays"]),
            "refusals_per_100_by_reason": {k: per100(v, tot["plays"])
                                           for k, v in tot["refused"].items()},
            "presses": RD.press_totals([r for r in rows if r.get("ok")])}


def tower_stratum(row: Mapping[str, Any]) -> str:
    """Both seats' tower troops, sorted and joined with " + " (one name when they are the same;
    UNKNOWN for a missing one)."""
    cards = row.get("tower_card")
    if not isinstance(cards, Mapping):
        return UNKNOWN
    return " + ".join(sorted({str(cards.get(side) or "") or UNKNOWN for side in RD.SIDES}))


def mode_stratum(row: Mapping[str, Any]) -> str:
    return str(row.get("game_mode") or "") or UNKNOWN


#: (key, heading, row -> stratum) of the stratified tables.
STRATA: tuple[tuple[str, str, Callable[[Mapping[str, Any]], str]], ...] = (
    ("tower", "tower troop", tower_stratum),
    ("mode", "game mode", mode_stratum),
)


def stratify(rows: Sequence[Mapping[str, Any]],
             key: Callable[[Mapping[str, Any]], str]) -> dict[str, dict[str, Any]]:
    """{stratum: summarize(its rows)}, most matches first (then by name)."""
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    order = sorted(groups, key=lambda s: (-len(groups[s]), s))
    return {s: summarize(groups[s]) for s in order}


def paired(base_rows: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
           key: str = "winner_agree") -> dict[str, Any] | None:
    """`key` (winner_agree / crowns_agree) of a cell against the base on the tags both ran:
    n, rate, base_rate, diff (rate - base_rate), better / worse (matches only this cell / only
    the base agrees on) and the exact McNemar p. None when no tag ran in both."""
    mine = {r.get("tag"): bool(r.get(key)) for r in rows if r.get("ok")}
    base = {r.get("tag"): bool(r.get(key)) for r in base_rows if r.get("ok")}
    common = sorted(t for t in set(mine) & set(base) if t is not None)
    if not common:
        return None
    a = [1.0 if mine[t] else 0.0 for t in common]
    b = [1.0 if base[t] else 0.0 for t in common]
    test = LS.mcnemar(a, b)
    n = len(common)
    return {"n": n, "rate": sum(a) / n, "base_rate": sum(b) / n, "diff": (sum(a) - sum(b)) / n,
            "better": test.a_better, "worse": test.b_better, "p_value": test.p_value}


def level_histogram(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Card levels over both decks of every row ({"level": cards}, by level)."""
    hist: Counter = Counter()
    for r in rows:
        levels = r.get("levels") if isinstance(r.get("levels"), Mapping) else {}
        for side in RD.SIDES:
            for level, n in (levels.get(side) or {}).items():
                hist[str(level)] += int(n)

    def order(level: str):
        return (0, int(level), level) if level.isdigit() else (1, 0, level)
    return {k: hist[k] for k in sorted(hist, key=order)}


def best_cell(cells: Sequence[Mapping[str, Any]], target: float = TARGET_WINNER
              ) -> dict[str, Any] | None:
    """The cell with the highest winner agreement (then crowns agreement; the earlier cell on a
    tie) among those that ran a match, and whether it reaches `target` (its rate, and its
    interval's lower bound)."""
    ran = [c for c in cells if c["summary"]["ran"] > 0]
    if not ran:
        return None
    best = ran[0]
    for c in ran[1:]:
        if ((c["summary"]["winner"]["rate"], c["summary"]["crowns"]["rate"])
                > (best["summary"]["winner"]["rate"], best["summary"]["crowns"]["rate"])):
            best = c
    w = best["summary"]["winner"]
    return {"name": best["name"], "winner": w, "crowns": best["summary"]["crowns"],
            "meets_target": w["rate"] >= target, "ci_above_target": w["lo"] >= target}


def fresh(meta: Any, cell: Cell, sha256: str, tap_delay: int = RD.TAP_DELAY) -> bool:
    """--reuse: True when a cell's meta (royalesim_redrive's <out>.meta.json) is for this
    payload file and this cell's card table, tick semantics, tap delay and presses."""
    if not isinstance(meta, Mapping):
        return False
    source = meta.get("source") if isinstance(meta.get("source"), Mapping) else {}
    engine = meta.get("engine") if isinstance(meta.get("engine"), Mapping) else {}
    try:
        same_table = rs_engine.table_name(engine.get("card_table")) == \
            rs_engine.table_name(cell.table)
        delay = int(meta.get("tap_delay") or 0)
    except (TypeError, ValueError):
        return False
    want_delay = int(tap_delay) if cell.ticks == RD.TICK_TAP else 0
    return (source.get("sha256") == sha256 and meta.get("presses") == cell.presses
            and meta.get("tick_semantics") == cell.ticks and delay == want_delay and same_table)


def build_report(sample: Mapping[str, Any], entries: Sequence[Mapping[str, Any]], *,
                 generated: str | None = None, target: float = TARGET_WINNER) -> dict[str, Any]:
    """The grid's numbers. entries: one per cell, in grid order, with "cell" (Cell), "status"
    ("ran" / "reused" / "failed"), "returncode", "seconds", "rows" (the cell's output rows),
    "meta" (its meta.json, or {}), and optionally "out", "log", "log_tail"."""
    entries = list(entries)
    base_rows = entries[0].get("rows") or [] if entries else []
    cells = []
    for k, e in enumerate(entries):
        rows = e.get("rows") or []
        meta = e.get("meta") if isinstance(e.get("meta"), Mapping) else {}
        fp = meta.get("fingerprint") if isinstance(meta.get("fingerprint"), Mapping) else {}
        cells.append({
            "name": e["cell"].name, "settings": e["cell"].settings(),
            "status": e.get("status"), "returncode": e.get("returncode"),
            "seconds": e.get("seconds"), "out": str(e.get("out") or ""),
            "log": str(e.get("log") or ""), "log_tail": list(e.get("log_tail") or []),
            "engine": {key: fp.get(key) for key in ("version", "card_table",
                                                     "card_table_fnv1a64", "unit_types",
                                                     "command_delay_ticks", "id")},
            "summary": summarize(rows),
            "vs_base": None if k == 0 else {"winner": paired(base_rows, rows, "winner_agree"),
                                            "crowns": paired(base_rows, rows, "crowns_agree")},
            "strata": {key: stratify(rows, fn) for key, _, fn in STRATA},
        })
    levels = level_histogram(next((e.get("rows") for e in entries if e.get("rows")), []))
    return {"schema": SCHEMA, "tool": TOOL,
            "generated": generated or datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%d %H:%M UTC"),
            "sample": dict(sample), "target": {"winner": target, "crowns": TARGET_CROWNS},
            "base": cells[0]["name"] if cells else None, "cells": cells,
            "best": best_cell(cells, target), "levels": levels}


# ---------------------------------------------------------------------------
# the report (markdown)
# ---------------------------------------------------------------------------

def fmt_rate(r: Mapping[str, Any] | None) -> str:
    """'75.0% [30.1, 95.4]'; 'n/a' without matches."""
    if not r or not r.get("n"):
        return "n/a"
    return f"{100 * r['rate']:.1f}% [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]"


def fmt_diff(p: Mapping[str, Any] | None) -> str:
    """'+20.0 pp (p 1, n 5)'; 'n/a' when no match ran in both."""
    if not p:
        return "n/a"
    return f"{100 * p['diff']:+.1f} pp (p {p['p_value']:.3g}, n {p['n']})"


def _cellsafe(text: Any) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _sample_line(sample: Mapping[str, Any]) -> str:
    sha = str(sample.get("sha256") or "")[:12]
    if sample.get("kind") == "payloads":
        return f"- Matches: the payloads of `{sample.get('path')}` (sha256 `{sha}`)."
    modes = sample.get("modes")
    modes = "all" if modes in (None, "all") else ", ".join(modes)
    return (f"- Matches: {sample.get('n')} of split `{sample.get('split')}`, seed "
            f"{sample.get('seed')}, modes {modes} ({sample.get('found', '?')} payloads found "
            f"of {sample.get('picked', '?')} picked by tag hash); payload file sha256 `{sha}`.")


def render(report: Mapping[str, Any]) -> str:
    """reports/fidelity_grid.md."""
    cells = report["cells"]
    target = report["target"]
    L: list[str] = []
    w = L.append
    w("# RoyaleSim fidelity grid\n")
    w(f"Generated by `{TOOL}` (E03) on {report['generated']}. Each cell re-drives the same "
      "recorded IL_Replay matches with `tools/royalesim_redrive.py` and compares each ending "
      "with the recording: winner agreement and exact crown agreement, with 95% Wilson "
      "intervals (`tools/league/stats.py`), and the recorded plays the engine refused per 100 "
      "plays (a refused play is dropped, never forced). \"vs base\" compares a cell with the "
      "first cell on the matches both ran (exact McNemar p).\n")
    w(_sample_line(report["sample"]))
    w(f"- Target (docs/SPARRING_PLAN.md, phase 2): winner agreement >= "
      f"{100 * target['winner']:.1f}%, RoyaleSim 0.1.22 on the author's b2000 matches (exact "
      f"crowns {100 * target['crowns']:.1f}% there).")
    w(f"- Friendlies are Tower Princess on both sides (the owner's frames), so the "
      f"`{FRIENDLY_TOWER}` row of \"By tower troop\" is the one that matters for sparring. "
      "RoyaleSim always plays the Tower Princess.")
    w("- Ticks: `exec` reads IL_Replay play ticks as execution ticks (no command delay, as "
      "every run before E03); `tap` reads them as tap ticks (command delay set before reset, "
      "each play sent at its recorded tick).\n")
    w("## Cells\n")
    w("| cell | royalesim | card table | ticks | presses | ran | winner agree | crowns agree "
      "| refusals / 100 plays | 3-crown engine / recorded | winner vs base | crowns vs base |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, c in enumerate(cells):
        s, eng, st = c["summary"], c["engine"], c["settings"]
        vs = c["vs_base"] or {}
        base_w = "base" if k == 0 else fmt_diff(vs.get("winner"))
        base_c = "base" if k == 0 else fmt_diff(vs.get("crowns"))
        status = "" if c["status"] in ("ran", "reused") else f" ({c['status']})"
        w(f"| {c['name']}{status} | {eng.get('version') or 'n/a'} | "
          f"{eng.get('card_table') or table_label(st['table'])} | {st['ticks']} | "
          f"{st['presses']} | {s['ran']} / {s['matches']} | {fmt_rate(s['winner'])} | "
          f"{fmt_rate(s['crowns'])} | {s['refusals_per_100']:.2f} | "
          f"{s['three_crowns']['engine']} / {s['three_crowns']['recorded']} | {base_w} | "
          f"{base_c} |")
    w("")
    best = report.get("best")
    if best is None:
        w("**No cell ran a match.**\n")
    else:
        verdict = ("reaches" if best["meets_target"] else "is below")
        bound = (" (and so does its interval's lower bound)" if best["ci_above_target"]
                 else "")
        w(f"**Best winner agreement: `{best['name']}`, {fmt_rate(best['winner'])} (crowns "
          f"{fmt_rate(best['crowns'])}); it {verdict} the {100 * target['winner']:.1f}% "
          f"target{bound}.** The card table and settings go into data/engine.json only by the "
          "owner's decision.\n")

    reasons: Counter = Counter()
    for c in cells:
        reasons.update(c["summary"]["refused"])
    w("## Refusals per 100 plays, by reason\n")
    if not reasons:
        w("No recorded play was refused.\n")
    else:
        names = [k for k, _ in reasons.most_common()]
        w("| cell | plays | " + " | ".join(names) + " |")
        w("|---|---|" + "---|" * len(names))
        for c in cells:
            by = c["summary"]["refusals_per_100_by_reason"]
            w(f"| {c['name']} | {c['summary']['plays']:,} | "
              + " | ".join(f"{by.get(k, 0.0):.2f}" for k in names) + " |")
        w("\n`hand`: the card was not in the engine's hand (its cycle disagreed); `hand_pending`: "
          "the same while a delayed play of that side was still waiting (tap only).\n")

    pressing = [c for c in cells if c["settings"]["presses"] == rs_presses.PRESS_ON]
    if pressing:
        w("## Ability presses (cells with presses on)\n")
        w("| cell | events | sent | accepted | refused | skipped |")
        w("|---|---|---|---|---|---|")
        for c in pressing:
            p = c["summary"]["presses"] or {}
            refused = ", ".join(f"{k} {v}" for k, v in sorted((p.get("refused") or {}).items()))
            skipped = ", ".join(f"{k} {v}" for k, v in sorted((p.get("skipped") or {}).items()))
            w(f"| {c['name']} | {p.get('events', 0)} | {p.get('sent', 0)} | {p.get('ok', 0)} | "
              f"{refused or '-'} | {skipped or '-'} |")
        w("")

    errors = [(c["name"], kind, n) for c in cells for kind, n in c["summary"]["errors"].items()]
    w("## Matches that did not run\n")
    if not errors:
        w("Every match of every cell that ran was re-driven.\n")
    else:
        w("| cell | error | matches |")
        w("|---|---|---|")
        for name, kind, n in errors:
            w(f"| {name} | {_cellsafe(kind)} | {n} |")
        w("")

    failed = [c for c in cells if c["status"] == "failed"]
    if failed:
        w("## Cells that failed\n")
        for c in failed:
            w(f"- `{c['name']}`: exit code {c['returncode']} after {c['seconds'] or 0:.0f} s; "
              f"log `{c['log']}`.")
            if c["log_tail"]:
                w("\n```")
                L.extend(_cellsafe(line) for line in c["log_tail"])
                w("```")
        w("")

    for key, heading, _ in STRATA:
        w(f"## By {heading}\n")
        w(f"| {heading} | cell | matches | ran | winner agree | crowns agree | "
          "refusals / 100 plays |")
        w("|---|---|---|---|---|---|---|")
        strata: list[str] = []
        for c in cells:
            for s in c["strata"][key]:
                if s not in strata:
                    strata.append(s)
        for s in strata:
            for c in cells:
                row = c["strata"][key].get(s)
                if row is None or c["status"] == "failed":
                    continue
                w(f"| {_cellsafe(s)} | {c['name']} | {row['matches']} | {row['ran']} | "
                  f"{fmt_rate(row['winner'])} | {fmt_rate(row['crowns'])} | "
                  f"{row['refusals_per_100']:.2f} |")
        w("")

    levels = report.get("levels") or {}
    w("## Card levels in the sample\n")
    if not levels:
        w("No match was read.\n")
    else:
        total = sum(levels.values())
        w("Both decks of every match; the re-drive plays each card at its recorded level and "
          "the crown towers at the match's most common card level. Friendlies are level 11.\n")
        w("| level | cards | share |")
        w("|---|---|---|")
        for level, n in levels.items():
            w(f"| {_cellsafe(level)} | {n:,} | {100 * n / max(1, total):.1f}% |")
        w("")

    w("## Engines\n")
    w("| cell | interpreter | royalesim | card table | card table fnv1a64 | unit types | "
      "command delay | fingerprint id |")
    w("|---|---|---|---|---|---|---|---|")
    for c in cells:
        eng = c["engine"]
        delay = eng.get("command_delay_ticks")
        w(f"| {c['name']} | `{_cellsafe(c['settings']['python'])}` | "
          f"{eng.get('version') or 'n/a'} | {eng.get('card_table') or 'n/a'} | "
          f"{eng.get('card_table_fnv1a64') or 'n/a'} | {eng.get('unit_types') or 'n/a'} | "
          f"{'/'.join(str(d) for d in delay) if delay else 'n/a'} | {eng.get('id') or 'n/a'} |")
    w("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_lines(path: Path) -> int:
    """Non-empty lines of a file (the matches of a payload file)."""
    with Path(path).open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_json(path: Path) -> Any:
    """A JSON file, or None when it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def log_tail(path: Path, n: int = LOG_TAIL) -> list[str]:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


def run_cell(cmd: Sequence[str], log: Path, timeout: float | None = None) -> tuple[int, float]:
    """Run one cell's command from the repo root, its console to `log`: (exit code, seconds).
    A timeout gives -9, an interpreter that does not start -1."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log.open("w", encoding="utf-8") as f:
        try:
            proc = subprocess.run(list(cmd), cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                                  env=env, timeout=timeout or None, check=False)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[{TOOL}] timed out after {timeout} s\n")
            code = -9
        except OSError as exc:
            f.write(f"\n[{TOOL}] could not start {cmd[0]}: {exc}\n")
            code = -1
    return code, time.time() - t0


def ensure_sample(work: Path, payloads: Path, *, n: int, split: str, seed: int, modes,
                  corpus: Path, reuse: bool = False) -> dict[str, Any]:
    """Write the sample's payloads to `payloads` (royalesim_redrive.sample_payloads) and its
    description to <work>/sample.json; under --reuse keep both when they describe the same
    sample."""
    spec = {"kind": "sample", "n": int(n), "split": split, "seed": int(seed),
            "modes": "all" if modes is None else sorted(modes), "corpus": str(corpus),
            "rule": RD.SAMPLE_RULE}
    spec_path = Path(work) / "sample.json"
    old = read_json(spec_path)
    if reuse and Path(payloads).is_file() and isinstance(old, dict) and \
            {k: old.get(k) for k in spec} == spec:
        return old
    tags, texts = RD.sample_payloads(corpus, n, split, seed, modes)
    RD.write_payload_file(payloads, texts)
    spec.update(picked=len(tags), found=len(texts))
    spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return spec


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-drive the same IL_Replay matches under each engine setting and write "
                    "reports/fidelity_grid.md (see the module doc).")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, metavar="N",
                    help=f"matches to pick by tag hash (default {DEFAULT_SAMPLE})")
    ap.add_argument("--split", choices=(*RD.SPLITS, RD.SPLIT_ALL), default=RD.DEFAULT_SPLIT,
                    help=f"the split to pick from (default {RD.DEFAULT_SPLIT})")
    ap.add_argument("--seed", type=int, default=0, help="the hash seed (default 0)")
    ap.add_argument("--modes", default=None,
                    help="game modes, comma-separated, or 'all' (default: tools/rs_rows.py's)")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="IL_Replay folder (default <data root>/il_replay)")
    ap.add_argument("--payloads", type=Path, default=None, metavar="FILE",
                    help="re-drive this payload file (royalesim_redrive.py --save-payloads) "
                         "instead of picking a sample")
    ap.add_argument("--tables", default=",".join(DEFAULT_TABLES),
                    help="card tables of the main interpreter's cells, comma-separated "
                         f"(default {','.join(DEFAULT_TABLES)}; 'default' = the engine's own)")
    ap.add_argument("--ticks", default=",".join(RD.TICK_MODES),
                    help="tick semantics, comma-separated (default exec,tap)")
    ap.add_argument("--presses", default=",".join(rs_presses.PRESS_MODES),
                    help="presses modes, comma-separated (default off,on)")
    ap.add_argument("--tap-delay", type=int, default=RD.TAP_DELAY, metavar="TICKS",
                    help=f"command delay of the tap cells (default {RD.TAP_DELAY})")
    ap.add_argument("--python", default=None,
                    help="interpreter of the main cells (default the machine's py_engine)")
    ap.add_argument("--rs022", default="auto",
                    help="royalesim 0.1.22 interpreter: 'auto' (default: .venvs/rs022 when it "
                         "exists), 'off', or a path")
    ap.add_argument("--workers", type=int, default=6, help="worker processes per cell")
    ap.add_argument("--timeout", type=float, default=0, metavar="S",
                    help="seconds a cell may run (default 0: no limit)")
    ap.add_argument("--work", type=Path, default=None,
                    help="folder for the payloads and each cell's output and log (default "
                         "<runs root>/fidelity_grid/<sample name>)")
    ap.add_argument("--out", type=Path, default=REPORT_MD,
                    help="the report (default reports/fidelity_grid.md; the numbers go beside "
                         "it as .json)")
    ap.add_argument("--reuse", action="store_true",
                    help="keep the sample and every cell output that matches its settings")
    ap.add_argument("--dry-run", action="store_true", help="print the cells and commands only")
    args = ap.parse_args(argv)
    try:
        tables = parse_tables(args.tables)
        ticks = parse_choices(args.ticks, RD.TICK_MODES, "--ticks")
        presses = parse_choices(args.presses, rs_presses.PRESS_MODES, "--presses")
        tap_delay = RD.tap_ticks(args.tap_delay)
        modes = RD.parse_modes(args.modes)
        if not args.payloads and args.sample < 1:
            raise ValueError(f"--sample: at least 1 match, got {args.sample}")
        if args.payloads and not args.payloads.is_file():
            raise FileNotFoundError(f"--payloads {args.payloads} does not exist")
        if args.workers < 1:
            raise ValueError(f"--workers: at least 1, got {args.workers}")
        python = args.python or default_python()
        rs022 = rs022_python(args.rs022)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    name = (Path(args.payloads).stem if args.payloads
            else RD.sample_name(args.sample, args.split, args.seed))
    work = Path(args.work or default_work(name)).resolve()
    payloads = Path(args.payloads or work / "payloads.jsonl").resolve()
    cells = grid_cells(tables, ticks, presses, python, rs022)
    print(f"grid: {len(cells)} cells, work {work}", flush=True)
    if rs022 is None:
        print(f"      no royalesim 0.1.22 cells ({RS022_VENV} not found or --rs022 off)",
              flush=True)
    if args.dry_run:
        for cell in cells:
            out = work / f"{cell.name}.jsonl"
            print(f"{cell.name}\n  " + " ".join(cell_command(
                cell, payloads, out, workers=args.workers, tap_delay=tap_delay)), flush=True)
        return 0
    work.mkdir(parents=True, exist_ok=True)
    if args.payloads:
        sample: dict[str, Any] = {"kind": "payloads", "path": str(payloads)}
    else:
        corpus = Path(args.corpus or default_corpus())
        print(f"sample: {args.sample} matches of {args.split}, seed {args.seed}, from {corpus}",
              flush=True)
        sample = ensure_sample(work, payloads, n=args.sample, split=args.split, seed=args.seed,
                               modes=modes, corpus=corpus, reuse=args.reuse)
    sample["sha256"] = rs_cards.file_sha256(payloads)
    matches = count_lines(payloads)
    print(f"payloads: {matches} matches, {payloads} (sha256 {sample['sha256'][:12]})",
          flush=True)
    if not matches:
        print(f"{TOOL}: no match to re-drive (check --corpus, --split, --modes or --payloads)",
              file=sys.stderr)
        return 2
    entries = []
    for k, cell in enumerate(cells, 1):
        out, log = work / f"{cell.name}.jsonl", work / f"{cell.name}.log"
        meta = read_json(RD.meta_path(out))
        if args.reuse and out.is_file() and fresh(meta, cell, sample["sha256"], tap_delay):
            status, code, seconds = "reused", 0, float(meta.get("seconds") or 0.0)
        else:
            for stale in (out, RD.meta_path(out)):
                stale.unlink(missing_ok=True)
            print(f"[{k}/{len(cells)}] {cell.name} ...", flush=True)
            code, seconds = run_cell(cell_command(cell, payloads, out, workers=args.workers,
                                                  tap_delay=tap_delay),
                                     log, args.timeout or None)
            status = "ran" if code == 0 and out.is_file() else "failed"
            meta = read_json(RD.meta_path(out))
        rows = read_jsonl(out) if status != "failed" else []
        entries.append({"cell": cell, "status": status, "returncode": code, "seconds": seconds,
                        "rows": rows, "meta": meta or {}, "out": out, "log": log,
                        "log_tail": log_tail(log) if status == "failed" else []})
        s = summarize(rows)
        print(f"  {cell.name}: {status}, ran {s['ran']}/{s['matches']}, winner "
              f"{fmt_rate(s['winner'])}, crowns {fmt_rate(s['crowns'])}, refusals "
              f"{s['refusals_per_100']:.2f} per 100 plays ({seconds:.0f}s)", flush=True)
    report = build_report(sample, entries)
    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render(report), encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    best = report["best"]
    if best:
        print(f"best: {best['name']}, winner {fmt_rate(best['winner'])} "
              f"({'reaches' if best['meets_target'] else 'below'} the "
              f"{100 * TARGET_WINNER:.1f}% target)", flush=True)
    print(f"report -> {out_md}", flush=True)
    return 0 if all(e["status"] != "failed" for e in entries) else 1


if __name__ == "__main__":
    sys.exit(main())
