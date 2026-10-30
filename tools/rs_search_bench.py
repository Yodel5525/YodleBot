r"""Engine benchmark and engine probes for the live search plan (S01, docs/LIVE_SEARCH_PLAN.md).

    .venvs\royalegym\Scripts\python.exe tools\rs_search_bench.py engine             timings, thread
                                                                                    scaling, probes
    .venvs\royalegym\Scripts\python.exe tools\rs_search_bench.py engine --quick     a short smoke run
    .venvs\royalegym\Scripts\python.exe tools\rs_search_bench.py engine --probes P8,P16 --no-timings
    .venvs\royalegym\Scripts\python.exe tools\rs_search_bench.py report             the table and the
                                                                                    recommendation
    .venvs\buildabot\Scripts\python.exe tools\rs_search_bench.py search ...         needs S08 (not built)

READ-ONLY AND OFFLINE. The tool builds RoyaleSim Battles in this process and nothing else: it
never taps, captures or touches the emulator, and it imports no scripts/ code. Close MuMu and the
browser first (16 GB RAM), and run no training at the same time, so the numbers are the PC's.

engine (royalegym venv). One EngineConfig (tools/rs_engine.py: data/engine.json, --card-table,
...), always undelayed for the rebuild battles (cfg.undelayed(), plan section 4.2). It times:
  construct           rs_engine.make_battle
  reset N             Battle.reset with N = 0, 10, 20, 40 spawns (half per side, land points)
  step k              step([], k), k = 1, 10, 60, on a busy board (40 spawns, warmed 40 ticks;
                      the board is load()ed back before every timed call), and k = 60 on an empty one
  state_json, json.loads, save (and its size), load, state_hash, check_deploy (per call, over
  every tile of blue's half and the four hand slots), building_placement (per call)
  root                reset 20 + step D + save + state_json + loads: the rebuild of one decision
  trial H             load + step([cmd], H) + state_json + loads, H = 40, 60, 80
  threads             1-16 Battles, each built and stepped in its own thread (step releases the
                      GIL): aggregate ticks per second and the efficiency against one thread
  memory              RSS per Battle and the free memory (psutil when installed)
and runs the probes below, each PASS (the answer the plan assumes), FAIL (it is not: fix the
rebuilder before anything else, plan section 7.2 M0) or "not run" (a card the catalogue lacks, an
accessor the engine lacks, or an error, which is recorded). Every probe records its raw answer.

  P1   does one Skeletons spec make 1 entity? (S04: one spec per unit, or per formation)
  P2   does a spawned unit have deploy_ticks 0? (S04: recasts on or off)
  P3   does a spawn's max_hp match unit_hitpoints(card, level)? (hp 0 and above max never sent)
  P4   do reset `levels` change a spawn's level?
  P5   do reset `forms` reach spawns?
  P6   what is king_active after a reset with the king below max?
  P7   elixir rate and overtime at ticks 2399/2400, 3599/3600, 4799/4800; do unequal crowns in
       overtime end the battle at once?
  P8   does set_command_delay_ticks persist across resets? (and a delay set after a reset)
  P9   save -> step 100 -> load: the same state_json and state_hash? a spawned Tesla hidden?
  P10  two Battles, the same seed and commands: the same state_hash?
  P11  does a RamRider or GoblinGiant spec bring its rider?
  P12  does a Dark Prince spawn with its shield (the one a played Dark Prince has)?
  P13  does a played card get its full deploy time and match formation_preview?
  P14  are Mirror and Elixir Collector kept out of the first four under shuffle 0?
  P15  does a blob saved on one Battle load on a sibling Battle?
  P16  under set_command_delay_ticks, are the elixir and the hand charged when a play is SENT
       or when it RUNS? (added by the S01 task; the report repeats it as "pending_charge")

It writes reports/search_bench/engine_<stamp>.json and the readable summary beside it as
engine_<stamp>.txt, and prints the summary.

report (any venv, no engine). Reads the newest engine_*.json (or --engine) and, when one exists,
the newest search_*.json (or --search), prints the table and a recommendation, and writes
report_<stamp>.json:
  per preset (s_root, s_idle, s_reply; plan section 3.2) the estimated search time against its
  deadline (80 / 120 ms, section 6.3), the cuts needed in the plan's order (H 40, 5 trials,
  K 2; then the asynchronous mode), the most trials within the budget, latency_ticks =
  ceil(ms / 50) and its break-even (0.51 pp per tick, section 2.1);
  the gate: workers (memory and CPUs, as rs_league_eval.worker_cap counts them, at --worker-mb
  1800 for a seat with an opponent model, section 6.2) and torch threads per worker.
The estimate is the engine components (p95 by default) plus two priors the search benchmark
replaces: ROW_MS_PRIOR per row (obs_row + relabel) and FORWARD_MS_PRIOR per batched forward
(CUDA, section 6.1). A search_*.json from the `search` subcommand may carry "row_ms" and
"forward_ms" summaries and "presets": {name: {"p95_ms": ...}}; those replace the priors and the
estimate.

search (buildabot venv): needs S08 (SearchSeat, rs_match's search options) and S13 for the
procs backend. Until then it parses the runbook's options, says so and exits 3.

Exit codes: 0 ok; 1 royalesim missing or refused the config, no engine report to read, or
--strict and a probe failed; 2 a bad setting; 3 the search subcommand before S08.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import sys
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:      # also when imported as tools.rs_search_bench
    sys.path.insert(0, str(ROOT / "tools"))
import rs_engine  # noqa: E402  (standard library only; royalesim is imported by `engine`)
import rs_spawn_table  # noqa: E402  (standard library and rs_engine only)

FORMAT = 1
TOOL = "tools/rs_search_bench.py"
OUT_DIR = ROOT / "reports" / "search_bench"
TICK_MS = 50

# ---------------------------------------------------------------------------
# board geometry, native world millitiles (engine = millitiles x SUBTILE_PER_MILLITILE, no flip;
# blue, team 0, owns low y). River water covers y 15000-17000 (arena.json water_half_rows).
# ---------------------------------------------------------------------------

BOARD_W, BOARD_H = 18_000, 32_000
#: A land point in blue's half, the middle column (rs_spawn_table.SPAWN_POINT_MILLI).
SPAWN_MT = tuple(rs_spawn_table.SPAWN_POINT_MILLI)
#: Where probe plays go: blue's left lane, own half, clear of its princess tower (y 5000-8000).
LANE_MT = (3500, 12000)
#: Red's right lane in its own half (a legal red troop tap).
RED_LANE_MT = (14500, 20000)
#: Red's half, in front of its king and away from everything: the Tesla of P9.
RED_BACK_MT = (9000, 24000)
#: Busy-board spawn points for blue; red's are these turned half a turn. Clear of the towers and
#: a tile and a half off the water, so every ground spec is on land.
BLUE_GRID: tuple[tuple[int, int], ...] = tuple(
    (x, y) for y in (13500, 12000, 10500, 9000) for x in (2500, 5500, 9000, 12500, 15500))
#: Single-unit troops for the busy board (missing ones are skipped).
BUSY_NAMES = ("Knight", "Musketeer", "Valkyrie", "MiniPekka", "Giant", "HogRider",
              "BabyDragon", "Wizard")
#: Plays and probes run after the deploy lockout (calibration match.DEPLOY_LOCKOUT_TICKS 90).
START_TICK = 600
#: The command delay P8 and P16 use: L03's delay (rs_league_eval.py:142-143).
PROBE_DELAY = 24

RESET_SPAWNS = (0, 10, 20, 40)
STEP_TICKS = (1, 10, 60)
TRIAL_HORIZONS = (40, 60, 80)
THREADS = (1, 2, 4, 8, 16)
#: The spawns of a typical rebuild, for the root composite and the cost model.
ROOT_SPAWNS = 20
BARRIER_S = 300.0

#: royalesim.CATALOGUE_FIELDS when the engine does not export it (rs_orient.KIND_COLUMN's note).
CATALOGUE_FIELDS_FALLBACK = ("name", "placement", "elixir", "count", "radius", "flying",
                             "hitpoints", "footprint_tiles", "card_kind")
#: royalesim.STATUS_BITS when the engine does not export it (royalegym protocol STATUS_BIT_NAMES).
STATUS_FALLBACK = ("underground", "invisible", "hidden", "evolved", "hero", "clone",
                   "ability_windup", "ability_active", "charged", "grounded")
#: step_commands_run kind of a card play, and its reason for one that ran (rs_memory.RAN_*).
RAN_DEPLOY, RAN_OK = "deploy", 0
#: An elixir fall larger than this (milli) between two ticks is a debit, not noise.
DEBIT_MIN_MILLI = 500

# ---------------------------------------------------------------------------
# the recommendation's model (plan sections 2.1, 3.2, 6.1, 6.3)
# ---------------------------------------------------------------------------

#: trials, horizon H, worlds K, batched forwards, deadline: plan 3.2, 6.1 and 6.3. The plan names
#: deadlines for s_idle (80 ms) and s_reply (120 ms); s_root, inline with no trials, gets 80.
PRESETS: dict[str, dict[str, Any]] = {
    "s_root": {"trials": 0, "horizon": 0, "worlds": 1, "forwards": 1, "deadline_ms": 80.0,
               "reply": False},
    "s_idle": {"trials": 8, "horizon": 60, "worlds": 1, "forwards": 2, "deadline_ms": 80.0,
               "reply": False},
    "s_reply": {"trials": 8, "horizon": 80, "worlds": 4, "forwards": 4, "deadline_ms": 120.0,
                "reply": True},
}
#: Fewer trials than this and the bot plays d0 (plan 3.1 step 7, 4.6).
MIN_TRIALS = 2
#: The cut order when the budget is exceeded (plan 6.3), after the procs backend (cut 1, which
#: needs S13 and is not estimated here) and before the asynchronous mode (cut 5).
CUTS: tuple[tuple[str, str, int], ...] = (("H 40", "horizon", 40), ("5 trials", "trials", 5),
                                          ("K 2", "worlds", 2))
#: Priors the search benchmark replaces: one obs_row + relabel, one batched value forward of
#: live_big on CUDA (plan 6.1: 2-5 ms for 8 rows).
ROW_MS_PRIOR = 1.5
FORWARD_MS_PRIOR = 5.0
#: What one tick of delay costs: live_big_hog 56.7% -> 46.0% over 21 ticks (plan 2.1).
PP_PER_TICK = 0.51
#: The decision-to-landing delay of the gate (plan 7, L03).
GATE_DELAY = 24
#: Gate worker memory: a seat with an opponent model and Battles (plan 6.2), the reserve and the
#: cap of rs_league_eval (rs_league_eval.py:166-168), and its assumption when memory is unknown.
WORKER_MB = 1800
RESERVE_MB = 3072
MAX_WORKERS = 6
ASSUMED_AVAILABLE_MB = 8192
#: Thread scaling counts as useful down to this efficiency.
GOOD_EFFICIENCY = 0.7


class Skip(Exception):
    """A probe that cannot run here: a card the catalogue lacks, an accessor the engine lacks."""


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def percentile(values: Sequence[float], q: float) -> float | None:
    """The q-th percentile (0-100) with linear interpolation (numpy's default); None when empty."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    pos = (len(xs) - 1) * float(q) / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def summarize(ms: Sequence[float]) -> dict[str, Any]:
    """n, mean, p50, p95, p99, min and max of a list of milliseconds."""
    xs = [float(v) for v in ms]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean_ms": _r(sum(xs) / len(xs)), "p50_ms": _r(percentile(xs, 50)),
            "p95_ms": _r(percentile(xs, 95)), "p99_ms": _r(percentile(xs, 99)),
            "min_ms": _r(min(xs)), "max_ms": _r(max(xs))}


def scaled(summary: Mapping[str, Any], factor: float) -> dict[str, Any]:
    """A summary with every *_ms value multiplied by `factor` (a batch -> per call)."""
    return {k: (_r(v * factor) if k.endswith("_ms") and v is not None else v)
            for k, v in summary.items()}


def timed(fn: Callable[[], Any], reps: int, before: Callable[[], Any] | None = None) -> list[float]:
    """Milliseconds of `reps` calls of fn(); before() runs untimed ahead of each."""
    out = []
    for _ in range(max(0, int(reps))):
        if before is not None:
            before()
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def latency_ticks(ms: float | None, tick_ms: float = TICK_MS) -> int | None:
    """ceil(ms / tick): the ticks of command delay a search of `ms` is charged (plan 6.3)."""
    if ms is None:
        return None
    if ms <= 0:
        return 0
    return int(math.ceil(round(float(ms) / float(tick_ms), 6)))


# ---------------------------------------------------------------------------
# the lab: one engine, its catalogue, and small helpers every probe uses
# ---------------------------------------------------------------------------

def _read(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    return value() if callable(value) else value


def ran_rows(battle: Any) -> list[tuple[int, int, str, int, int]]:
    """The last step's Battle.step_commands_run() rows (tick, team, kind, what, reason); [] on an
    engine or fake without it (rs_match._ran_rows)."""
    fn = getattr(battle, "step_commands_run", None)
    if not callable(fn):
        return []
    out = []
    for row in fn() or ():
        row = tuple(row)
        if len(row) < 5:
            continue
        try:
            out.append((int(row[0]), int(row[1]), str(row[2]), int(row[3]), int(row[4])))
        except (TypeError, ValueError):
            continue
    return out


def pending_of(battle: Any, state: Mapping[str, Any], team: int) -> list[list[Any]]:
    """`team`'s commands accepted and not run yet: Battle.pending_commands, else the state's
    players[team]["pending"]."""
    fn = getattr(battle, "pending_commands", None)
    if callable(fn):
        return [list(p) for p in fn(int(team)) or ()]
    return [list(p) for p in state["players"][team].get("pending") or ()]


class Lab:
    """One engine config and royalesim module, read once: catalogue, unit names, field index,
    reasons, the filler deck (Hog 2.6, rs_spawn_table.FILLER_NAMES) and the card level."""

    def __init__(self, cfg: Any, rs: ModuleType):
        self.cfg = rs_engine.resolve(cfg).undelayed()
        self.rs = rs
        self.sub = int(getattr(rs, "SUBTILE_PER_MILLITILE", 18))
        b = self.battle()
        self.catalogue = json.loads(b.catalogue_json())
        self.ids = {str(row[0]): i for i, row in enumerate(self.catalogue)}
        self.units = [str(u) for u in json.loads(b.unit_types_json())]
        self.F = rs_engine.field_index(rs)
        self.reasons = [str(r) for r in (getattr(rs, "DEPLOY_REASONS", None) or ())]
        self.status_names = [str(n) for n in (getattr(rs, "STATUS_BITS", None) or STATUS_FALLBACK)]
        fields = getattr(rs, "CATALOGUE_FIELDS", None)
        self.cat_fields = [str(f) for f in fields] if fields else list(CATALOGUE_FIELDS_FALLBACK)
        self.deck = rs_spawn_table.filler_deck(self.catalogue)
        self.level = int(self.cfg.level or _read(b, "card_level") or 11)
        self._busy: list[int] | None = None

    # -------------------------------------------------------------- building
    def battle(self, delay: int | None = None) -> Any:
        """A new Battle; `delay` sets (delay, delay) BEFORE its first reset (make_battle)."""
        cfg = self.cfg if not delay else self.cfg.replace(delay=(int(delay), int(delay)))
        return rs_engine.make_battle(cfg, self.rs)

    def reset(self, b: Any, *, spawns: Sequence[Sequence[int]] = (), start_tick: int = START_TICK,
              elixir: Sequence[int] = (5000, 5000), tower_hp: Any = None,
              decks: Sequence[Sequence[int]] | None = None, seed: int = 0, read: bool = True,
              **extra: Any) -> dict | None:
        """Battle.reset with shuffle 0 (hand = deck[0:4]); the state after it unless read=False."""
        b.reset(seed=int(seed), decks=[list(d) for d in (decks or (self.deck, self.deck))],
                shuffle=0, start_tick=int(start_tick), elixir_milli=[int(e) for e in elixir],
                tower_hp=tower_hp, spawns=[tuple(int(v) for v in s) for s in spawns], **extra)
        return self.state(b) if read else None

    @staticmethod
    def state(b: Any) -> dict:
        return json.loads(b.state_json())

    # -------------------------------------------------------------- cards
    def card(self, *names: str) -> int | None:
        """The catalogue id of the first of `names` the catalogue has."""
        for name in names:
            if name in self.ids:
                return self.ids[name]
        return None

    def need(self, *names: str) -> int:
        cid = self.card(*names)
        if cid is None:
            raise Skip(f"none of {list(names)} is in this engine's catalogue")
        return cid

    def name(self, cid: Any) -> str:
        try:
            return str(self.catalogue[int(cid)][0])
        except (TypeError, ValueError, IndexError):
            return str(cid)

    def column(self, field: str) -> int:
        if field in self.cat_fields:
            return self.cat_fields.index(field)
        return CATALOGUE_FIELDS_FALLBACK.index(field)

    def cost(self, cid: int) -> int:
        return int(self.catalogue[cid][self.column("elixir")])

    def deck_with(self, cid: int, *, first: bool = True) -> list[int]:
        """The filler deck with `cid` in it: first (hand slot 0) or kept where it is."""
        rest = [c for c in self.deck if c != cid]
        if not first and cid in self.deck:
            return list(self.deck)
        return ([cid] + rest)[:8]

    def reason(self, code: Any) -> str:
        try:
            code = int(code)
        except (TypeError, ValueError):
            return str(code)
        return self.reasons[code] if 0 <= code < len(self.reasons) else str(code)

    def bit(self, name: str) -> int:
        names = self.status_names
        return 1 << names.index(name) if name in names else 1 << STATUS_FALLBACK.index(name)

    # -------------------------------------------------------------- board
    def xy(self, mx: float, my: float) -> tuple[int, int]:
        return int(round(mx * self.sub)), int(round(my * self.sub))

    def spawn(self, team: int, cid: int, mt: Sequence[float], hp: int = -1) -> tuple:
        x, y = self.xy(*mt)
        return (int(team), int(cid), x, y, int(hp))

    def busy_cards(self) -> list[int]:
        if self._busy is None:
            cards = [self.ids[n] for n in BUSY_NAMES if n in self.ids]
            if len(cards) < 2:                  # another table: any single-unit troop
                kind, count = self.column("card_kind"), self.column("count")
                cards = [i for i, row in enumerate(self.catalogue)
                         if len(row) > max(kind, count) and str(row[kind]).upper() == "TROOP"
                         and row[count] == 1][:len(BUSY_NAMES)]
            if not cards:
                raise Skip("no single-unit troop in the catalogue for the busy board")
            self._busy = cards
        return self._busy

    def board(self, n: int) -> list[tuple]:
        """`n` spawns, alternating blue and red, each side cycling through busy_cards."""
        if n <= 0:
            return []
        cards = self.busy_cards()
        out = []
        for i in range(int(n)):
            team, k = i % 2, i // 2
            mx, my = BLUE_GRID[k % len(BLUE_GRID)]
            if team == 1:
                mx, my = BOARD_W - mx, BOARD_H - my
            out.append(self.spawn(team, cards[k % len(cards)], (mx, my)))
        return out

    def busy(self, b: Any, n: int = 40, *, warm: int = 0, seed: int = 0,
             start_tick: int = START_TICK) -> None:
        """Reset `b` to the busy board (both sides at 10 elixir) and step `warm` ticks."""
        self.reset(b, spawns=self.board(n), elixir=(10000, 10000), seed=seed,
                   start_tick=start_tick, read=False)
        if warm:
            b.step([], int(warm))

    # -------------------------------------------------------------- entities
    def units_of(self, state: Mapping[str, Any], team: int | None = None,
                 card: int | None = None) -> list[list]:
        """Living non-tower entities, optionally of one team and one card."""
        F = self.F
        return [e for e in state["entities"] if e[F["tower_slot"]] < 0 and e[F["hp"]] > 0
                and (team is None or e[F["team"]] == team)
                and (card is None or e[F["card_id"]] == card)]

    def unit_name(self, e: Sequence[Any]) -> str | None:
        ut = e[self.F["unit_type"]]
        return self.units[ut] if isinstance(ut, int) and 0 <= ut < len(self.units) else None

    def row(self, e: Sequence[Any], *names: str) -> dict:
        out = {n: e[self.F[n]] for n in names}
        if "unit_type" in names:
            out["unit"] = self.unit_name(e)
        if "card_id" in names:
            out["card"] = self.name(e[self.F["card_id"]])
        return out

    def own_hp(self, b: Any, cid: int, level: int | None = None) -> int | None:
        """The "own" row's hitpoints of unit_hitpoints(cid, level); None without the accessor."""
        fn = getattr(b, "unit_hitpoints", None)
        if not callable(fn):
            return None
        rows = list(fn(int(cid), int(level or self.level)))
        for role, _name, hp in rows:
            if str(role) == "own":
                return int(hp)
        return int(rows[0][2]) if rows else None


# ---------------------------------------------------------------------------
# watching one command (P8, P16)
# ---------------------------------------------------------------------------

def first_change(before: Any, series: Sequence[tuple[int, Any]],
                 changed: Callable[[Any, Any], bool]) -> int | None:
    """The tick at which the step began that first changed a value: `series` holds (tick after
    the step, value) for consecutive one-tick steps, `before` the value ahead of the first."""
    prev = before
    for tick, value in series:
        if changed(prev, value):
            return int(tick) - 1
        prev = value
    return None


def moment(tick: int | None, sent: int, ran: int | None) -> str:
    """"send" when a change came in the step the command was sent in, "run" in the step it ran
    in (both when the two are one step: "send=run"), "never", or "tick N"."""
    if tick is None:
        return "never"
    if ran is not None and tick == ran:
        return "send=run" if ran == sent else "run"
    if tick == sent:
        return "send"
    return f"tick {tick}"


def observe_command(lab: Lab, b: Any, team: int, slot: int, mt: Sequence[float], *,
                    max_wait: int, extra: int = 0, other_slot: int | None = None,
                    trace: bool = False) -> dict[str, Any]:
    """Send (team, slot, mt) now, then step one tick at a time until it has run (plus `extra`
    ticks) or `max_wait` ticks passed. "ran" is the tick its step began: step_commands_run's
    tick, else (an engine that does not report it, or no delay) the step after which nothing
    is pending and the card has left the slot. trace=True records every tick's elixir, hand,
    pending block and the check_deploy verdicts of `slot` and `other_slot`."""
    st = lab.state(b)
    me = st["players"][team]
    sent = int(st["tick"])
    hand0 = [int(c) for c in me["hand"]]
    card = hand0[slot]
    x, y = lab.xy(*mt)
    out: dict[str, Any] = {"sent_tick": sent, "slot": int(slot), "card": lab.name(card),
                           "cost": lab.cost(card) if 0 <= card < len(lab.catalogue) else None,
                           "elixir_before": int(me["elixir_milli"]),
                           "hand_before": [lab.name(c) for c in hand0],
                           "pending_cost_before": int(me.get("pending_cost") or 0)}
    res = list(b.step([(int(team), int(slot), x, y)], 1) or ())
    try:
        code = int(res[0][1]) if res else 0
    except (TypeError, IndexError, ValueError):
        code = 0
    out["send_reason"] = lab.reason(code)
    out["resolved"] = [int(v) for v in res[0][3:5]] if res and len(res[0]) >= 5 else None
    out["accepted"] = code == 0
    ticks: list[dict[str, Any]] = []
    ran: int | None = None
    ran_reason: str | None = None
    reported = False
    waited, start, after_run = 1, sent, 0
    while True:
        st = lab.state(b)
        me = st["players"][team]
        pend = pending_of(b, st, team)
        hand = [int(c) for c in me["hand"]]
        if ran is None:
            rows = [r for r in ran_rows(b) if r[1] == team and r[2] == RAN_DEPLOY]
            if rows:
                ran, ran_reason, reported = rows[0][0], lab.reason(rows[0][4]), True
            elif out["accepted"] and not pend and hand[slot] != card:
                ran = start
        else:
            after_run += 1
        if trace:
            rec = {"tick": int(st["tick"]), "elixir_milli": int(me["elixir_milli"]),
                   "hand": [lab.name(c) for c in hand], "hand_ids": hand,
                   "next": lab.name(me.get("next_card", -1)),
                   "pending": [list(p) for p in me.get("pending") or ()],
                   "pending_cost": int(me.get("pending_cost") or 0),
                   "pending_commands": pend,
                   "check_slot": lab.reason(b.check_deploy(int(team), int(slot), x, y))}
            if other_slot is not None:
                rec["check_other"] = lab.reason(b.check_deploy(int(team), int(other_slot), x, y))
            ticks.append(rec)
        if not out["accepted"] or waited >= max_wait or (ran is not None and after_run >= extra):
            break
        start = int(st["tick"])
        b.step([], 1)
        waited += 1
    out.update({"ran_tick": ran, "ran_reason": ran_reason, "ran_reported": reported,
                "delay": None if ran is None else ran - sent, "waited": waited})
    if trace:
        out["trace"] = ticks
    return out


def charge_moments(obs: Mapping[str, Any]) -> dict[str, Any]:
    """From an observe_command(trace=True) record: when the elixir fell by the play's cost and
    when the card left its slot, as moment() labels them."""
    sent, ran = int(obs["sent_tick"]), obs.get("ran_tick")
    trace = obs.get("trace") or []
    slot = int(obs["slot"])
    elixir = first_change(obs["elixir_before"], [(t["tick"], t["elixir_milli"]) for t in trace],
                          lambda a, b: b < a - DEBIT_MIN_MILLI)
    card0 = obs["hand_before"][slot]
    hand = first_change(card0, [(t["tick"], t["hand"][slot]) for t in trace],
                        lambda a, b: b != a)
    at_send = trace[0] if trace else {}
    return {"elixir": moment(elixir, sent, ran), "hand": moment(hand, sent, ran),
            "elixir_tick": elixir, "hand_tick": hand, "sent_tick": sent, "ran_tick": ran,
            "pending_after_send": bool(at_send.get("pending_commands") or at_send.get("pending")),
            "pending_cost_after_send": at_send.get("pending_cost"),
            "check_slot_while_pending": at_send.get("check_slot"),
            "check_other_while_pending": at_send.get("check_other")}


# ---------------------------------------------------------------------------
# the probes
# ---------------------------------------------------------------------------

ProbeFn = Callable[[Lab, "BenchSettings"], tuple[Any, dict, str]]


@dataclasses.dataclass(frozen=True)
class Probe:
    id: str
    question: str
    expect: str
    plan: str
    run: ProbeFn


def _p1(lab: Lab, s: "BenchSettings"):
    answer: dict[str, Any] = {}
    for name in ("Skeletons", "Barbarians", "Minions"):
        cid = lab.card(name)
        if cid is None:
            continue
        b = lab.battle()
        spec = lab.spawn(0, cid, SPAWN_MT)
        n0 = len(lab.units_of(lab.reset(b, spawns=[spec]), 0))
        b.step([], 1)
        n1 = len(lab.units_of(lab.state(b), 0))
        rec: dict[str, Any] = {"entities_at_reset": n0, "entities_after_1": n1}
        preview = getattr(b, "formation_preview", None)
        if callable(preview):
            try:
                rec["formation_members"] = len(preview(0, name, spec[2], spec[3]))
            except Exception as exc:            # the preview refused: say so, keep the count
                rec["formation_members"] = f"{type(exc).__name__}: {exc}"
        answer[name] = rec
    if "Skeletons" not in answer:
        raise Skip("no Skeletons in this engine's catalogue")
    sk = answer["Skeletons"]
    n = sk["entities_at_reset"] or sk["entities_after_1"]
    ok = n == 1
    note = ("one spec makes 1 Skeleton: spawn one spec per unit" if ok else
            f"one Skeletons spec makes {n} entities: spawn one spec per formation (S04 switch)")
    return ok, answer, note


def _p2(lab: Lab, s: "BenchSettings"):
    picks = [(0, "HogRider", LANE_MT), (0, "Knight", SPAWN_MT), (1, "Cannon", RED_LANE_MT)]
    spawns, wanted = [], []
    for team, name, mt in picks:
        cid = lab.card(name)
        if cid is not None:
            spawns.append(lab.spawn(team, cid, mt))
            wanted.append(name)
    if not spawns:
        raise Skip(f"none of {[p[1] for p in picks]} is in this engine's catalogue")
    b = lab.battle()
    st = lab.reset(b, spawns=spawns)
    rows = [lab.row(e, "team", "card_id", "deploy_ticks") for e in lab.units_of(st)]
    b.step([], 1)
    after = [lab.row(e, "team", "card_id", "deploy_ticks") for e in lab.units_of(lab.state(b))]
    ok = bool(rows) and all(r["deploy_ticks"] == 0 for r in rows)
    note = ("spawns start deployed (deploy_ticks 0): deploying units need recasts (S04 R2)" if ok
            else f"spawns carry deploy timers {[r['deploy_ticks'] for r in rows]}: recasts may be "
                 "unnecessary; re-check plan 4.4")
    return ok, {"cards": wanted, "at_reset": rows, "after_1": after}, note


def _p3(lab: Lab, s: "BenchSettings"):
    names = ("HogRider", "Musketeer", "Knight", "Cannon", "IceGolemite", "Giant")
    hp_col = lab.column("hitpoints")
    out, bad = {}, []
    b = lab.battle()
    if not callable(getattr(b, "unit_hitpoints", None)):
        raise Skip("this engine has no Battle.unit_hitpoints")
    for name in names:
        cid = lab.card(name)
        if cid is None:
            continue
        own = lab.own_hp(b, cid)
        units = lab.units_of(lab.reset(b, spawns=[lab.spawn(0, cid, SPAWN_MT)]), 0)
        if len(units) != 1 or own is None:
            out[name] = {"units": len(units), "own_hp": own, "skipped": "not one unit"}
            continue
        full = lab.row(units[0], "hp", "max_hp")
        sent = max(1, own // 2)
        half = lab.units_of(lab.reset(b, spawns=[lab.spawn(0, cid, SPAWN_MT, hp=sent)]), 0)
        rec = {"own_hp": own, "catalogue_hp": lab.catalogue[cid][hp_col]
               if len(lab.catalogue[cid]) > hp_col else None,
               "max_hp": full["max_hp"], "hp_full": full["hp"], "hp_sent": sent,
               "hp_read": half[0][lab.F["hp"]] if len(half) == 1 else None}
        out[name] = rec
        if not (rec["max_hp"] == own and rec["hp_full"] == own and rec["hp_read"] == sent):
            bad.append(name)
    checked = [n for n, r in out.items() if "skipped" not in r]
    if not checked:
        raise Skip("no single-unit card to check")
    ok = not bad
    note = (f"max_hp = unit_hitpoints(card, {lab.level}) and hp as sent, for {checked}" if ok
            else f"spawn hp differs from unit_hitpoints for {bad}: hp = fraction x max needs "
                 "another max")
    return ok, {"level": lab.level, "cards": out}, note


def _p4(lab: Lab, s: "BenchSettings"):
    hog = lab.need("HogRider")
    deck = lab.deck_with(hog, first=False)
    other = next((c for c in (lab.card("Knight"), lab.card("Giant"), lab.card("Valkyrie"))
                  if c is not None and c not in deck), None)
    spawns = [lab.spawn(0, hog, LANE_MT)]
    if other is not None:
        spawns.append(lab.spawn(0, other, SPAWN_MT))
    level = min(rs_engine.MAX_LEVEL, lab.level + 1)
    b = lab.battle()

    def read(st):
        return {lab.name(e[lab.F["card_id"]]): lab.row(e, "level", "max_hp")
                for e in lab.units_of(st, 0)}

    base = read(lab.reset(b, spawns=spawns, decks=(deck, lab.deck)))
    got = read(lab.reset(b, spawns=spawns, decks=(deck, lab.deck),
                         levels=[[level] * len(deck), [lab.level] * len(lab.deck)]))
    want_hp = lab.own_hp(b, hog, level)
    name = lab.name(hog)
    in_deck = got.get(name, {})
    follows = in_deck.get("level") == level or (
        want_hp is not None and in_deck.get("max_hp") == want_hp
        and want_hp != base.get(name, {}).get("max_hp"))
    answer = {"level_asked": level, "base": base, "with_levels": got,
              "own_hp_at_level": want_hp, "not_in_deck": None if other is None else lab.name(other)}
    note = (f"a deck card's spawn takes its reset level ({level})" if follows else
            "reset levels do not reach spawns: rebuilt units keep the object's level")
    if other is not None:
        note += f"; a card outside the deck reads level {got.get(lab.name(other), {}).get('level')}"
    return bool(follows), answer, note


#: (card, form) pairs P5 tries, those in the filler deck: 1 evolution, 2 hero.
FORM_CANDIDATES = (("Skeletons", 1), ("Musketeer", 2), ("IceGolemite", 2), ("Cannon", 1),
                   ("HogRider", 1), ("IceSpirits", 1))


def _p5(lab: Lab, s: "BenchSettings"):
    b = lab.battle()
    out, reached, ran = {}, [], []
    bits = {1: lab.bit("evolved"), 2: lab.bit("hero")}
    for name, form in FORM_CANDIDATES:
        cid = lab.card(name)
        if cid is None or cid not in lab.deck:
            continue
        spawn = [lab.spawn(0, cid, SPAWN_MT)]
        base = sorted(str(lab.unit_name(e)) for e in lab.units_of(lab.reset(b, spawns=spawn), 0))
        forms = [0] * len(lab.deck)
        forms[lab.deck.index(cid)] = form
        try:
            st = lab.reset(b, spawns=spawn, forms=[forms, [0] * len(lab.deck)])
        except Exception as exc:                # this table has no such form: try the next
            out[f"{name}/{form}"] = {"refused": f"{type(exc).__name__}: {exc}"}
            continue
        units = lab.units_of(st, 0)
        flags = [int(e[lab.F["status_flags"]]) for e in units]
        names = sorted(str(lab.unit_name(e)) for e in units)
        hit = any(f >= 0 and f & bits[form] for f in flags) or names != base
        out[f"{name}/{form}"] = {"base_units": base, "units": names, "status_flags": flags,
                                 "reached": hit}
        ran.append(name)
        if hit:
            reached.append(f"{name}/{form}")
    if not ran:
        raise Skip(f"every form was refused or absent: {out}")
    ok = len(reached) == len(ran)
    note = (f"forms reach spawns ({reached})" if ok else
            f"forms reach spawns only for {reached or 'none'} of {ran}: an evolution or hero is "
            "rebuilt as the base card")
    return ok, out, note


def _p6(lab: Lab, s: "BenchSettings"):
    b = lab.battle()
    st = lab.reset(b)
    mx = [[int(v) for v in st["players"][t]["tower_max_hp"]] for t in (0, 1)]

    def king(tower_hp):
        st0 = lab.reset(b, tower_hp=tower_hp)
        rec = {"king_hp": int(st0["players"][0]["tower_hp"][0]),
               "at_reset": bool(st0["players"][0].get("king_active"))}
        b.step([], 1)
        rec["after_1"] = bool(lab.state(b)["players"][0].get("king_active"))
        b.step([], 79)
        rec["after_80"] = bool(lab.state(b)["players"][0].get("king_active"))
        return rec

    full = king([list(mx[0]), list(mx[1])])
    damaged = king([[mx[0][0] - 500, mx[0][1], mx[0][2]], list(mx[1])])
    princess = king([[mx[0][0], 0, mx[0][2]], list(mx[1])])
    awake = damaged["at_reset"] or damaged["after_1"]
    answer = {"tower_max_hp": mx, "full": full, "king_damaged": damaged,
              "princess_down": princess}
    note = ("a damaged king is awake after the reset" if awake else
            "a damaged king is asleep after the reset" + (" (it woke within 4 s)"
                                                          if damaged["after_80"] else "")
            + ": king activation is lost (plan 4.7)")
    return awake, answer, note


def rate_label(gain: float, unit: float) -> int | None:
    """1, 2 or 3: the elixir multiplier of a gain against the 1x gain `unit` (same ticks)."""
    if not unit:
        return None
    ratio = float(gain) / float(unit)
    return 1 if ratio < 1.5 else 2 if ratio < 2.5 else 3


def _p7(lab: Lab, s: "BenchSettings"):
    b = lab.battle()
    st = lab.reset(b)
    mx = [[int(v) for v in st["players"][t]["tower_max_hp"]] for t in (0, 1)]

    def per_tick(start, ticks):
        st0 = lab.reset(b, start_tick=start, elixir=(0, 0))
        seq = [(int(st0["tick"]), int(st0["players"][0]["elixir_milli"]),
                st0.get("elixir_rate"), bool(st0.get("overtime")))]
        for _ in range(ticks):
            b.step([], 1)
            st1 = lab.state(b)
            seq.append((int(st1["tick"]), int(st1["players"][0]["elixir_milli"]),
                        st1.get("elixir_rate"), bool(st1.get("overtime"))))
        return seq

    lab.reset(b, start_tick=START_TICK, elixir=(0, 0))
    e0 = int(lab.state(b)["players"][0]["elixir_milli"])
    b.step([], 56)
    unit = (int(lab.state(b)["players"][0]["elixir_milli"]) - e0) / 56.0
    ticks: dict[str, dict[str, Any]] = {}
    for boundary in (2400, 3600, 4800):
        seq = per_tick(boundary - 1, 2)
        for k in (0, 1):
            tick, e, rate, ot = seq[k]
            gain = seq[k + 1][1] - e
            ticks[str(tick)] = {"state_rate": rate, "overtime": ot, "gain_milli": gain,
                                "rate": rate_label(gain, unit)}
    windows = {}
    for start in (START_TICK, 2400, 3600, 4800):
        seq = per_tick(start, 40) if start != START_TICK else None
        if seq is None:
            windows[str(start)] = {"mult": 1.0}
            continue
        windows[str(start)] = {"gain_40": seq[-1][1] - seq[0][1],
                               "mult": _r((seq[-1][1] - seq[0][1]) / (40 * unit), 3)}
    expected = {"2399": (1, False), "2400": (2, False), "3599": (2, False), "3600": (2, True),
                "4799": (2, True), "4800": (3, True)}
    wrong = [t for t, (rate, ot) in expected.items()
             if ticks.get(t, {}).get("rate") != rate or ticks.get(t, {}).get("overtime") != ot]

    def crowns(start, red_down, blue_down):
        th = [list(mx[0]), list(mx[1])]
        if red_down:
            th[1][1] = 0
        if blue_down:
            th[0][1] = 0
        st0 = lab.reset(b, start_tick=start, tower_hp=th)
        rec = {"start_tick": start, "crowns": [int(p.get("crowns", 0)) for p in st0["players"]],
               "game_over_at_reset": bool(st0["game_over"]), "winner": st0.get("winner")}
        if not st0["game_over"]:
            b.step([], 1)
            st1 = lab.state(b)
            rec.update({"game_over_after_1": bool(st1["game_over"]), "winner": st1.get("winner")})
        return rec

    answer = {"unit_milli_per_tick": _r(unit, 3), "ticks": ticks, "windows": windows,
              "expected": {t: {"rate": r, "overtime": o} for t, (r, o) in expected.items()},
              "crowns_overtime_unequal": crowns(3600, True, False),
              "crowns_overtime_equal": crowns(3600, True, True),
              "crowns_regular_unequal": crowns(START_TICK, True, False)}
    unequal = answer["crowns_overtime_unequal"]
    ends = unequal["game_over_at_reset"] or unequal.get("game_over_after_1")
    note = ("rates follow the tick (2x from 2400, overtime from 3600, 3x from 4800)" if not wrong
            else f"rate or overtime differs from plan 4.2 at ticks {wrong}")
    note += ("; unequal crowns in overtime end the battle "
             + ("at the reset" if unequal["game_over_at_reset"] else
                "on the first tick" if ends else "NOT at once"))
    return not wrong, answer, note


def _p8(lab: Lab, s: "BenchSettings"):
    k = int(s.delay)
    b = lab.battle(delay=k)
    runs = []
    for i in (1, 2):
        lab.reset(b, elixir=(10000, 10000), read=False)
        got = _read(b, "command_delay_ticks")
        obs = observe_command(lab, b, 0, 0, LANE_MT, max_wait=k + 10)
        runs.append({"reset": i, "command_delay_ticks": None if got is None else list(got),
                     "send_reason": obs["send_reason"], "ran_tick": obs["ran_tick"],
                     "delay": obs["delay"], "ran_reported": obs["ran_reported"]})
    late = lab.battle()
    lab.reset(late, elixir=(10000, 10000), read=False)
    late.set_command_delay_ticks(k, k)
    now = observe_command(lab, late, 0, 0, LANE_MT, max_wait=k + 10)
    lab.reset(late, elixir=(10000, 10000), read=False)
    nxt = observe_command(lab, late, 0, 0, LANE_MT, max_wait=k + 10)
    answer = {"delay": k, "runs": runs,
              "set_after_reset": {"same_battle_delay": now["delay"],
                                  "next_reset_delay": nxt["delay"]}}
    ok = all(r["delay"] == k for r in runs)
    note = (f"the delay ({k}) holds across resets" if ok else
            f"after a second reset the play ran {runs[-1]['delay']} ticks late, not {k}")
    note += (f"; set after a reset it applies from the next reset (now {now['delay']}, "
             f"next {nxt['delay']})")
    return ok, answer, note


def _p9(lab: Lab, s: "BenchSettings"):
    b = lab.battle()
    lab.busy(b, s.busy_spawns, warm=20)
    if not callable(getattr(b, "state_hash", None)):
        raise Skip("this engine has no Battle.state_hash")
    blob = b.save()
    h0, s0 = b.state_hash(), b.state_json()
    b.step([], 100)
    h100 = b.state_hash()
    b.load(blob)
    h1, s1 = b.state_hash(), b.state_json()
    b.step([], 100)
    h100b = b.state_hash()
    restore = {"hash_equal": h0 == h1, "state_json_equal": s0 == s1,
               "replay_hash_equal": h100 == h100b, "save_bytes": len(blob)}
    exact = all(v for k, v in restore.items() if k != "save_bytes")
    tesla: dict[str, Any] = {}
    cid = lab.card("Tesla")
    if cid is not None and callable(getattr(b, "hidden_uids", None)):
        t = lab.battle()
        st = lab.reset(t, spawns=[lab.spawn(1, cid, RED_BACK_MT)])
        mine = lab.units_of(st, 1, cid)
        uid = mine[0][lab.F["uid"]] if mine else None
        hidden0 = uid in set(t.hidden_uids())
        flag0 = bool(mine and mine[0][lab.F["status_flags"]] >= 0
                     and mine[0][lab.F["status_flags"]] & lab.bit("hidden"))
        t.step([], 1)
        hidden1 = uid in set(t.hidden_uids())
        tesla = {"uid": uid, "hidden_at_reset": hidden0, "hidden_after_1": hidden1,
                 "status_hidden_at_reset": flag0}
    else:
        tesla = {"skipped": "no Tesla in the catalogue or no Battle.hidden_uids"}
    hidden = bool(tesla.get("hidden_at_reset") or tesla.get("hidden_after_1"))
    ok = exact if "skipped" in tesla else exact and hidden
    note = ("save/load is exact" if exact else f"save/load is NOT exact: {restore}")
    if "skipped" in tesla:
        note += f"; Tesla not checked ({tesla['skipped']})"
    elif tesla.get("uid") is None:
        note += "; the Tesla spec put no Tesla on the board"
    else:
        note += "; a spawned Tesla starts hidden" if hidden else "; a spawned Tesla is NOT hidden"
    return ok, {"restore": restore, "tesla": tesla}, note


def _script(lab: Lab, b: Any, seed: int) -> list[int]:
    """P10's game: the busy board, a play each at once, another after 60 ticks; hashes."""
    lab.busy(b, 20, seed=seed)
    out = []
    bx, by = lab.xy(*LANE_MT)
    rx, ry = lab.xy(*RED_LANE_MT)
    b.step([(0, 0, bx, by), (1, 0, rx, ry)], 1)
    b.step([], 59)
    out.append(int(b.state_hash()))
    b.step([(0, 1, bx, by)], 1)
    b.step([], 139)
    out.append(int(b.state_hash()))
    return out


def _p10(lab: Lab, s: "BenchSettings"):
    b1, b2, b3 = lab.battle(), lab.battle(), lab.battle()
    if not callable(getattr(b1, "state_hash", None)):
        raise Skip("this engine has no Battle.state_hash")
    h1, h2, h3 = _script(lab, b1, 7), _script(lab, b2, 7), _script(lab, b3, 8)
    same = h1 == h2 and b1.state_json() == b2.state_json()
    answer = {"hashes_a": h1, "hashes_b": h2, "same": same, "other_seed_differs": h1 != h3}
    note = ("two Battles agree tick for tick (common random numbers hold)" if same else
            "two Battles with the same seed and commands DIVERGE")
    note += "; another seed " + ("changes" if h1 != h3 else "does not change") + " the hash"
    return same, answer, note


def _p11(lab: Lab, s: "BenchSettings"):
    out = {}
    for name in ("RamRider", "GoblinGiant"):
        cid = lab.card(name)
        if cid is None:
            continue
        b = lab.battle()
        st = lab.reset(b, spawns=[lab.spawn(0, cid, SPAWN_MT)])
        units = lab.units_of(st, 0)
        links = [lab.row(e, "uid", "mount_uid", "unit_type") for e in units]
        riders = [list(r) for r in (_read(b, "rider_states") or ())]
        out[name] = {"entities": len(units), "units": links, "rider_states": riders,
                     "has_rider": any(r["mount_uid"] >= 0 for r in links) or bool(riders)}
    if not out:
        raise Skip("no RamRider or GoblinGiant in this engine's catalogue")
    ok = all(r["has_rider"] for r in out.values())
    missing = [n for n, r in out.items() if not r["has_rider"]]
    note = ("a spec brings its rider" if ok else
            f"no rider for {missing}: the rebuild must spawn the rider too, or loses it")
    return ok, out, note


def _p12(lab: Lab, s: "BenchSettings"):
    dp = lab.need("DarkPrince")
    b = lab.battle()
    spawned = lab.units_of(lab.reset(b, spawns=[lab.spawn(0, dp, SPAWN_MT)]), 0, dp)
    at_spawn = spawned[0][lab.F["shield"]] if spawned else None
    deck = lab.deck_with(dp)
    lab.reset(b, decks=(deck, lab.deck), elixir=(10000, 10000), read=False)
    played = observe_command(lab, b, 0, 0, SPAWN_MT, max_wait=2)
    b.step([], 30)
    units = lab.units_of(lab.state(b), 0, dp)
    at_play = units[0][lab.F["shield"]] if units else None
    answer = {"spawned_shield": at_spawn, "played_shield": at_play,
              "play_reason": played["send_reason"]}
    ok = bool(at_spawn) and (at_play is None or at_spawn == at_play)
    note = (f"a spawned Dark Prince has its shield ({at_spawn})" if ok else
            f"a spawned Dark Prince has shield {at_spawn}, a played one {at_play}")
    return ok, answer, note


def match_formation(preview: Sequence[Sequence[Any]], units: Sequence[Mapping[str, Any]], *,
                    elapsed: int = 1, tick_ms: int = TICK_MS, pos_tol: int = 4500,
                    tick_tol: int = 2) -> dict[str, Any]:
    """Compare formation_preview members (unit, x, y, deploy ms) with the units a play laid
    (dicts with x, y, deploy_ticks), `elapsed` ticks after the play: counts, the largest
    nearest-member distance (subtiles; the tolerance is a quarter tile) and the largest
    deploy-timer gap (ticks; a member's timer should read ceil(ms / tick) - elapsed)."""
    want = [(int(m[1]), int(m[2]), int(m[3])) for m in preview]
    got = [(int(u["x"]), int(u["y"]), int(u["deploy_ticks"])) for u in units]
    out: dict[str, Any] = {"members": len(want), "units": len(got)}
    if not want or not got:
        out.update({"ok": False, "max_dist": None, "max_tick_gap": None})
        return out
    free = list(range(len(want)))
    dists, gaps = [], []
    for gx, gy, gticks in got:
        j = min(free or range(len(want)),
                key=lambda i: (want[i][0] - gx) ** 2 + (want[i][1] - gy) ** 2)
        if j in free:
            free.remove(j)
        wx, wy, wms = want[j]
        dists.append(math.hypot(wx - gx, wy - gy))
        gaps.append(abs((math.ceil(wms / tick_ms) - elapsed) - gticks))
    out.update({"max_dist": _r(max(dists), 1), "max_tick_gap": max(gaps),
                "deploy_ticks": [g[2] for g in got],
                "preview_ticks": [math.ceil(w[2] / tick_ms) for w in want]})
    out["ok"] = (len(want) == len(got) and max(dists) <= pos_tol and max(gaps) <= tick_tol
                 and all(g[2] > 0 for g in got))
    return out


def _p13(lab: Lab, s: "BenchSettings"):
    out = {}
    for name in ("Skeletons", "HogRider"):
        cid = lab.card(name)
        if cid is None:
            continue
        b = lab.battle()
        preview = getattr(b, "formation_preview", None)
        if not callable(preview):
            raise Skip("this engine has no Battle.formation_preview")
        st = lab.reset(b, decks=(lab.deck_with(cid), lab.deck), elixir=(10000, 10000))
        before = {e[lab.F["uid"]] for e in lab.units_of(st)}
        x, y = lab.xy(*SPAWN_MT)
        res = list(b.step([(0, 0, x, y)], 1) or ())
        code = int(res[0][1]) if res else 0
        rx, ry = (int(res[0][3]), int(res[0][4])) if res and len(res[0]) >= 5 else (x, y)
        members = [list(m) for m in preview(0, name, rx, ry)]
        new = [lab.row(e, "x", "y", "deploy_ticks", "unit_type")
               for e in lab.units_of(lab.state(b), 0) if e[lab.F["uid"]] not in before]
        rec = match_formation(members, new)
        rec.update({"reason": lab.reason(code), "tap": [x, y], "resolved": [rx, ry],
                    "preview": members})
        out[name] = rec
    if not out:
        raise Skip("no Skeletons or HogRider in this engine's catalogue")
    ok = all(r["ok"] for r in out.values())
    bad = [n for n, r in out.items() if not r["ok"]]
    note = ("a played card lays formation_preview's members with its full deploy time" if ok
            else f"play and preview differ for {bad}")
    return ok, out, note


def _p14(lab: Lab, s: "BenchSettings"):
    omit = [c for c in (lab.card("Mirror"), lab.card("Elixir Collector", "ElixirCollector"))
            if c is not None]
    if not omit:
        raise Skip("no Mirror or Elixir Collector in this engine's catalogue")
    deck = (omit + [c for c in lab.deck if c not in omit])[:8]
    b = lab.battle()
    st = lab.reset(b, decks=(deck, lab.deck))
    me = st["players"][0]
    hand = [int(c) for c in me["hand"]]
    kept_out = not (set(omit) & set(hand))
    answer = {"deck": [lab.name(c) for c in deck], "hand": [lab.name(c) for c in hand],
              "next": lab.name(me.get("next_card", -1)),
              "expected_without_rule": [lab.name(c) for c in deck[:4]]}
    note = ("Mirror / Elixir Collector are swapped out of the first four: a rebuilt hand holding "
            "one cannot be dealt as deck[0:4] (S04 deck order)" if kept_out else
            "Mirror / Elixir Collector stay in the first four under shuffle 0")
    return kept_out, answer, note


def _p15(lab: Lab, s: "BenchSettings"):
    b1 = lab.battle()
    lab.busy(b1, s.busy_spawns, warm=20)
    if not callable(getattr(b1, "state_hash", None)):
        raise Skip("this engine has no Battle.state_hash")
    blob = b1.save()
    h, raw = b1.state_hash(), b1.state_json()
    b2 = lab.battle()
    lab.reset(b2, read=False)
    b2.load(blob)
    same = b2.state_hash() == h and b2.state_json() == raw
    b1.step([], 50)
    b2.step([], 50)
    stepped = b1.state_hash() == b2.state_hash()
    fresh: dict[str, Any] = {}
    b3 = lab.battle()
    try:
        b3.load(blob)
        fresh = {"loaded": True, "hash_equal": b3.state_hash() == h}
    except Exception as exc:                    # a never-reset Battle may refuse it
        fresh = {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}
    ok = same and stepped
    answer = {"sibling_equal": same, "sibling_stepped_equal": stepped, "never_reset": fresh,
              "save_bytes": len(blob)}
    note = ("a blob loads on a sibling Battle and runs on identically" if ok else
            "a blob does NOT carry over to a sibling Battle")
    return ok, answer, note


def _p16(lab: Lab, s: "BenchSettings"):
    k = int(s.delay)
    b = lab.battle(delay=k)
    st = lab.reset(b, elixir=(5000, 5000))
    hand = [int(c) for c in st["players"][0]["hand"]]
    cost = lab.cost(hand[0])
    other = next((i for i in range(1, len(hand)) if lab.cost(hand[i]) > 5 - cost), None)
    obs = observe_command(lab, b, 0, 0, LANE_MT, max_wait=k + 6, extra=2, other_slot=other,
                          trace=True)
    if not obs["accepted"]:
        raise Skip(f"the probe play was refused at send: {obs['send_reason']}")
    m = charge_moments(obs)
    answer = {"delay": k, **m, "card": obs["card"], "cost": cost,
              "other_slot_card": None if other is None else lab.name(hand[other]),
              "ran_reported": obs["ran_reported"], "elixir_before": obs["elixir_before"],
              "trace": [{key: t[key] for key in ("tick", "elixir_milli", "hand", "pending_cost",
                                                   "check_slot") if key in t}
                        for t in obs["trace"]]}
    ok = m["elixir"] == "run" and m["hand"] == "run"
    note = (f"elixir is charged at {m['elixir'].upper()}, the hand cycles at "
            f"{m['hand'].upper()} (sent {m['sent_tick']}, ran {m['ran_tick']}); after the send "
            f"pending_cost {m['pending_cost_after_send']}, the slot reads "
            f"{m['check_slot_while_pending']}")
    if other is not None:
        note += f", a {lab.name(hand[other])} reads {m['check_other_while_pending']}"
    return ok, answer, note


PROBES: tuple[Probe, ...] = (
    Probe("P1", "Does one Skeletons spec make 1 entity?",
          "1 entity per spec", "S04: one spec per unit, or one per formation", _p1),
    Probe("P2", "Does a spawned unit have deploy_ticks 0?",
          "0: spawns start deployed", "S04: recasts on or off (plan 4.4)", _p2),
    Probe("P3", "Does a spawn's max_hp match unit_hitpoints(card, level)?",
          "max_hp = the own row; hp as sent", "plan 4.2: hp = fraction x unit_hitpoints", _p3),
    Probe("P4", "Do reset `levels` change a spawn's level?",
          "a deck card's spawn takes its reset level", "informational (friendlies play 11)", _p4),
    Probe("P5", "Do reset `forms` reach spawns?",
          "an evolved / hero deck card spawns in its form", "plan 4.7: forms", _p5),
    Probe("P6", "What is king_active after a reset with the king below max?",
          "awake", "plan 4.7: king activation", _p6),
    Probe("P7", "Elixir rate and overtime at 2399/2400, 3599/3600, 4799/4800; unequal crowns "
          "in overtime?", "1x, 2x from 2400, overtime from 3600, 3x from 4800",
          "plan 4.2 start_tick; 4.2 refusals (game over at reset)", _p7),
    Probe("P8", "Does set_command_delay_ticks persist across resets?",
          "yes", "plan 4.2: one reused Battle", _p8),
    Probe("P9", "Does save -> step 100 -> load give the same state_json and state_hash? Is a "
          "spawned Tesla in hidden_uids?", "exact; hidden", "plan 3.1 trials; 4.1 permanence", _p9),
    Probe("P10", "Do two Battles reset with the same seed and commands give the same "
          "state_hash?", "yes", "plan 4.7: common random numbers", _p10),
    Probe("P11", "Does a RamRider or GoblinGiant spec bring its rider?",
          "yes", "S04 spawns", _p11),
    Probe("P12", "Does a Dark Prince spawn with its shield?",
          "the shield a played one has", "plan 4.7: shields", _p12),
    Probe("P13", "Does a played card get its full deploy time and match formation_preview?",
          "yes", "plan 4.4: recasts", _p13),
    Probe("P14", "Are Mirror and Elixir Collector kept out of the first four under shuffle 0?",
          "kept out", "plan 5 card cycle; S04 deck order", _p14),
    Probe("P15", "Does a blob saved on one Battle load on a sibling Battle?",
          "yes", "plan 6.3: procs backend", _p15),
    Probe("P16", "Under set_command_delay_ticks, are elixir and the hand charged at SEND or "
          "when the play RUNS?", "both at RUN (Battle docstring)",
          "plan 4.3: own in-flight plays pay when stepped in; public_state pending_cost", _p16),
)
PROBE_IDS = tuple(p.id for p in PROBES)


def run_probes(lab: Lab, settings: "BenchSettings", only: Sequence[str] | None = None,
               probes: Sequence[Probe] = PROBES, log: Callable[[str], None] | None = None
               ) -> list[dict[str, Any]]:
    """Every probe (or those in `only`), each isolated: a Skip or an error is recorded and the
    rest still run."""
    wanted = None if not only else {p.strip().upper() for p in only}
    out = []
    for probe in probes:
        if wanted is not None and probe.id.upper() not in wanted:
            continue
        rec: dict[str, Any] = {"id": probe.id, "question": probe.question,
                               "expect": probe.expect, "plan": probe.plan}
        t0 = time.perf_counter()
        try:
            ok, answer, note = probe.run(lab, settings)
            rec.update({"ok": None if ok is None else bool(ok), "answer": answer, "note": note})
        except Skip as exc:
            rec.update({"ok": None, "answer": {}, "note": f"not run: {exc}"})
        except Exception as exc:                # one probe's failure must not lose the others
            rec.update({"ok": None, "answer": {}, "note": "not run: error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-8:]})
        rec["ms"] = _r((time.perf_counter() - t0) * 1000.0, 1)
        if log is not None:
            log(f"  {probe.id:<4} {verdict(rec):<8} {rec['note']}")
        out.append(rec)
    return out


def verdict(rec: Mapping[str, Any]) -> str:
    ok = rec.get("ok")
    return "PASS" if ok is True else "FAIL" if ok is False else "not run"


def probe_summary(probes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(verdict(p) for p in probes)
    return {"probes": len(probes), "pass": counts.get("PASS", 0), "fail": counts.get("FAIL", 0),
            "not_run": counts.get("not run", 0),
            "failed": [p["id"] for p in probes if p.get("ok") is False],
            "not_run_ids": [p["id"] for p in probes if p.get("ok") is None]}


def pending_charge(probes: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """P16's answer, for the investigations that need it without reading the probe list."""
    for p in probes:
        if p.get("id") == "P16":
            a = p.get("answer") or {}
            return {"elixir": a.get("elixir"), "hand": a.get("hand"), "delay": a.get("delay"),
                    "sent_tick": a.get("sent_tick"), "ran_tick": a.get("ran_tick"),
                    "pending_cost_after_send": a.get("pending_cost_after_send"),
                    "check_slot_while_pending": a.get("check_slot_while_pending"),
                    "check_other_while_pending": a.get("check_other_while_pending"),
                    "ok": p.get("ok"), "note": p.get("note")}
    return None


# ---------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BenchSettings:
    reps: int = 50                  # samples per timing
    construct_reps: int = 5
    thread_reps: int = 10           # 60-tick steps per thread
    threads: tuple[int, ...] = THREADS
    busy_spawns: int = 40
    warm_ticks: int = 40
    delay: int = PROBE_DELAY        # D: the root's advance; P8 and P16's command delay


QUICK = BenchSettings(reps=3, construct_reps=2, thread_reps=2, threads=(1, 2))


def engine_timings(lab: Lab, s: BenchSettings, log: Callable[[str], None] | None = None
                   ) -> dict[str, Any]:
    say = log or (lambda _m: None)
    t: dict[str, Any] = {}
    say("  construct")
    t["construct"] = summarize(timed(lab.battle, s.construct_reps))
    b = lab.battle()
    say("  reset")
    resets = {}
    for n in RESET_SPAWNS:
        spawns = lab.board(n)
        resets[str(n)] = summarize(timed(
            lambda spawns=spawns: lab.reset(b, spawns=spawns, elixir=(10000, 10000), read=False),
            s.reps))
    t["reset"] = resets

    say("  step")
    lab.busy(b, s.busy_spawns, warm=s.warm_ticks)
    blob = b.save()

    def restore() -> None:
        b.load(blob)

    busy = {str(k): summarize(timed(lambda k=k: b.step([], k), s.reps, before=restore))
            for k in STEP_TICKS}
    empty_b = lab.battle()
    lab.reset(empty_b, read=False)
    empty_blob = empty_b.save()
    empty = {"60": summarize(timed(lambda: empty_b.step([], 60), s.reps,
                                   before=lambda: empty_b.load(empty_blob)))}
    t["step"] = {"busy": busy, "empty": empty}
    t["ticks_per_s"] = {
        "busy": _r(60_000.0 / busy["60"]["p50_ms"], 1) if busy["60"].get("p50_ms") else None,
        "empty": _r(60_000.0 / empty["60"]["p50_ms"], 1) if empty["60"].get("p50_ms") else None}

    say("  state_json, save, load")
    restore()
    raw = b.state_json()
    t["state_json"] = summarize(timed(b.state_json, s.reps))
    t["json_loads"] = summarize(timed(lambda: json.loads(raw), s.reps))
    t["state_bytes"] = len(raw)
    t["save"] = summarize(timed(b.save, s.reps))
    t["save_bytes"] = len(blob)
    t["load"] = summarize(timed(restore, s.reps))
    t["state_hash"] = (summarize(timed(b.state_hash, s.reps))
                       if callable(getattr(b, "state_hash", None)) else None)

    say("  check_deploy, building_placement")
    restore()
    tiles = [lab.xy(tx * 1000 + 500, ty * 1000 + 500) for ty in range(15) for tx in range(18)]
    calls = [(0, slot, x, y) for slot in range(4) for x, y in tiles]
    reasons = Counter(lab.reason(b.check_deploy(*c)) for c in calls)
    batches = max(1, s.reps // 10)
    t["check_deploy"] = scaled(summarize(timed(lambda: [b.check_deploy(*c) for c in calls],
                                               batches)), 1.0 / len(calls))
    t["check_deploy_calls"] = len(calls)
    t["check_deploy_reasons"] = dict(sorted(reasons.items()))
    place = getattr(b, "building_placement", None)
    cannon = lab.card("Cannon")
    if callable(place) and cannon is not None:
        some = tiles[::5]
        t["building_placement"] = scaled(summarize(timed(
            lambda: [place(0, lab.name(cannon), x, y) for x, y in some], batches)), 1.0 / len(some))
    else:
        t["building_placement"] = None

    say("  root and trials")
    board = lab.board(ROOT_SPAWNS)

    def root() -> None:
        lab.reset(b, spawns=board, elixir=(10000, 10000), read=False)
        b.step([], int(s.delay))
        b.save()
        json.loads(b.state_json())

    composite: dict[str, Any] = {"root": summarize(timed(root, s.reps)),
                                 "root_spawns": ROOT_SPAWNS, "delay": int(s.delay)}
    root()
    root_blob = b.save()
    cx, cy = lab.xy(*LANE_MT)
    trials = {}
    for h in TRIAL_HORIZONS:
        def trial(h=h) -> None:
            b.load(root_blob)
            b.step([(0, 0, cx, cy)], h)
            json.loads(b.state_json())
        trials[str(h)] = summarize(timed(trial, s.reps))
    composite["trial"] = trials
    t["composite"] = composite
    return t


def _thread_run(lab: Lab, n: int, s: BenchSettings, ticks: int = 60) -> dict[str, Any]:
    """`n` threads, each with its own Battle built in that thread, stepping its busy board
    `thread_reps` x `ticks` after a common start; the wall time of the whole."""
    barrier = threading.Barrier(n + 1)
    done: list[tuple[float, float] | None] = [None] * n
    errors: list[str] = []

    def work(i: int) -> None:
        try:
            b = lab.battle()
            lab.busy(b, s.busy_spawns, warm=s.warm_ticks)
            blob = b.save()
        except Exception as exc:
            errors.append(f"battle {i}: {type(exc).__name__}: {exc}")
            barrier.abort()
            return
        try:
            barrier.wait(timeout=BARRIER_S)
        except threading.BrokenBarrierError:
            return
        step_s = 0.0
        try:
            for _ in range(s.thread_reps):
                b.load(blob)
                t1 = time.perf_counter()
                b.step([], ticks)
                step_s += time.perf_counter() - t1
        except Exception as exc:
            errors.append(f"battle {i}: {type(exc).__name__}: {exc}")
            return
        done[i] = (time.perf_counter(), step_s)

    workers = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(n)]
    for w in workers:
        w.start()
    try:
        barrier.wait(timeout=BARRIER_S)
    except threading.BrokenBarrierError:
        pass
    t0 = time.perf_counter()
    for w in workers:
        w.join(timeout=BARRIER_S)
    wall = time.perf_counter() - t0
    finished = [d for d in done if d is not None]
    total = len(finished) * s.thread_reps * ticks
    per = [s.thread_reps * ticks / d[1] for d in finished if d[1] > 0]
    return {"battles": n, "finished": len(finished), "wall_s": _r(wall),
            "ticks": total, "ticks_per_s": _r(total / wall, 1) if finished and wall > 0 else None,
            "step_ticks_per_s_per_battle": _r(sum(per) / len(per), 1) if per else None,
            "errors": errors}


def thread_scaling(lab: Lab, s: BenchSettings, log: Callable[[str], None] | None = None
                   ) -> list[dict[str, Any]]:
    lab.busy_cards()                            # read once, before the threads share the lab
    rows = []
    base = None
    for n in s.threads:
        if log is not None:
            log(f"  threads {n}")
        row = _thread_run(lab, int(n), s)
        if base is None and row["ticks_per_s"] and int(n) == 1:
            base = row["ticks_per_s"]
        rows.append(row)
    for row in rows:
        row["efficiency"] = (_r(row["ticks_per_s"] / (row["battles"] * base), 3)
                             if base and row["ticks_per_s"] else None)
    return rows


def available_mb() -> int | None:
    """Free physical memory in MB: psutil, else rs_league_eval.available_mb, else None."""
    try:
        import psutil
        return int(psutil.virtual_memory().available // (1 << 20))
    except Exception:
        pass
    try:
        import rs_league_eval
        return rs_league_eval.available_mb()
    except Exception:
        return None


def memory_info(lab: Lab, s: BenchSettings) -> dict[str, Any]:
    out: dict[str, Any] = {"available_mb": available_mb(), "rss_mb_per_battle": None}
    try:
        import psutil
    except Exception:
        out["note"] = "psutil not installed: no RSS per Battle"
        return out
    proc = psutil.Process()
    before = proc.memory_info().rss
    keep = []
    for _ in range(8):
        b = lab.battle()
        lab.busy(b, s.busy_spawns, warm=1)
        keep.append(b)
    after = proc.memory_info().rss
    out["rss_mb_per_battle"] = _r((after - before) / len(keep) / (1 << 20), 2)
    out["rss_mb"] = _r(after / (1 << 20), 1)
    return out


def machine_info() -> dict[str, Any]:
    """What ran the benchmark, with no path or user name in it (reports may be committed)."""
    return {"system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "processor": platform.processor(),
            "python": platform.python_version(), "cpus": os.cpu_count(),
            "available_mb": available_mb()}


# ---------------------------------------------------------------------------
# the recommendation (report)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Costs:
    """Milliseconds of each piece of a searched decision (a p95 by default)."""
    reset_ms: float
    tick_ms: float
    save_ms: float
    load_ms: float
    state_ms: float
    row_ms: float = ROW_MS_PRIOR
    forward_ms: float = FORWARD_MS_PRIOR

    @classmethod
    def from_engine(cls, report: Mapping[str, Any], *, stat: str = "p95_ms",
                    spawns: int = ROOT_SPAWNS, row_ms: float | None = None,
                    forward_ms: float | None = None) -> "Costs":
        t = report["timings"]
        resets = {int(k): v[stat] for k, v in t["reset"].items() if v.get(stat) is not None}
        n = min(resets, key=lambda k: (abs(k - spawns), -k))
        busy = t["step"]["busy"]
        k = max(int(x) for x, v in busy.items() if v.get(stat) is not None)
        return cls(reset_ms=float(resets[n]), tick_ms=float(busy[str(k)][stat]) / k,
                   save_ms=float(t["save"][stat]), load_ms=float(t["load"][stat]),
                   state_ms=float(t["state_json"][stat]) + float(t["json_loads"][stat]),
                   row_ms=ROW_MS_PRIOR if row_ms is None else float(row_ms),
                   forward_ms=FORWARD_MS_PRIOR if forward_ms is None else float(forward_ms))


def estimate_ms(costs: Costs, preset: Mapping[str, Any], *, trials: int | None = None,
                delay: int = GATE_DELAY) -> float:
    """One searched decision: K roots (reset, advance D, save), the root row; then trials x K
    trials (load, step H, state, row), the reply rows (s_reply) and the batched forwards."""
    n = int(preset["trials"] if trials is None else trials)
    k = max(1, int(preset["worlds"]))
    c = costs
    root = k * (c.reset_ms + delay * c.tick_ms + c.save_ms) + c.state_ms + c.row_ms
    trial = c.load_ms + int(preset["horizon"]) * c.tick_ms + c.state_ms + c.row_ms
    reply = n * k * c.row_ms if preset.get("reply") else 0.0
    return root + n * k * trial + reply + int(preset["forwards"]) * c.forward_ms


def max_trials(costs: Costs, preset: Mapping[str, Any], *, delay: int = GATE_DELAY
               ) -> int | None:
    """The most trials (up to the preset's) within its deadline; None when even none fit."""
    for n in range(int(preset["trials"]), -1, -1):
        if estimate_ms(costs, preset, trials=n, delay=delay) <= float(preset["deadline_ms"]):
            return n
    return None


def plan_preset(name: str, costs: Costs, *, delay: int = GATE_DELAY,
                measured_p95: float | None = None) -> dict[str, Any]:
    """A preset against its deadline: the plan's cuts in order until it fits, the most trials
    then, the time, latency_ticks and their break-even."""
    base = dict(PRESETS[name])
    cur = dict(base)
    budget = float(cur["deadline_ms"])
    cuts: list[str] = []
    fits = estimate_ms(costs, cur, delay=delay) <= budget
    if not fits and cur["trials"] > 0:
        for label, key, value in CUTS:
            if int(cur[key]) <= value:
                continue
            cur[key] = value
            cuts.append(label)
            if estimate_ms(costs, cur, delay=delay) <= budget:
                fits = True
                break
    most = max_trials(costs, cur, delay=delay)
    if fits:
        trials = int(cur["trials"])
    else:
        trials = most or 0
    searchable = int(base["trials"]) == 0 or trials >= MIN_TRIALS
    est = estimate_ms(costs, cur, trials=trials, delay=delay)
    ms = float(measured_p95) if measured_p95 is not None else est
    ticks = latency_ticks(ms)
    notes = []
    if not fits and int(base["trials"]) > 0:
        notes.append("over budget after the cuts: the procs backend (S13) or the asynchronous "
                     "mode (plan 6.3, cuts 1 and 5)")
    if not searchable:
        notes.append(f"fewer than {MIN_TRIALS} trials fit: the search would play d0")
    if int(base["trials"]) == 0 and not fits:
        notes.append("the root alone is over the deadline")
    return {"preset": name, "config": {k: cur[k] for k in ("trials", "horizon", "worlds")},
            "cuts": cuts, "fits": bool(fits), "trials": trials, "most_trials": most,
            "searchable": bool(searchable), "est_ms": _r(est, 2), "ms": _r(ms, 2),
            "source": "measured" if measured_p95 is not None else "estimate",
            "deadline_ms": budget, "latency_ticks": ticks,
            "break_even_pp": _r(ticks * PP_PER_TICK, 2) if ticks is not None else None,
            "notes": notes}


def gate_workers(cpus: int | None, available: int | None, *, worker_mb: int = WORKER_MB,
                 reserve_mb: int = RESERVE_MB, max_workers: int = MAX_WORKERS) -> dict[str, Any]:
    """Gate worker processes (rs_league_eval.worker_cap's caps: memory, CPUs less one, the
    maximum) and torch threads per worker (the spare CPUs shared out, at least 1)."""
    ncpu = max(1, int(cpus or 1))
    avail = ASSUMED_AVAILABLE_MB if available is None else int(available)
    caps = {"memory": max(1, (avail - int(reserve_mb)) // int(worker_mb)),
            "cpus": max(1, ncpu - 1), "max": max(1, int(max_workers))}
    workers = min(caps.values())
    return {"workers": workers, "torch_threads": max(1, (ncpu - 1) // workers),
            "caps": caps, "limited_by": sorted(k for k, v in caps.items() if v == workers),
            "cpus": ncpu, "available_mb": avail, "available_assumed": available is None,
            "worker_mb": int(worker_mb), "reserve_mb": int(reserve_mb)}


def scaling_limit(rows: Sequence[Mapping[str, Any]], good: float = GOOD_EFFICIENCY) -> int | None:
    """The most Battles that still step at `good` efficiency or better (None without rows)."""
    ok = [int(r["battles"]) for r in rows if r.get("efficiency") is not None
          and float(r["efficiency"]) >= good]
    return max(ok) if ok else None


def _search_ms(search: Mapping[str, Any] | None, key: str) -> float | None:
    if not search:
        return None
    value = search.get(key)
    if isinstance(value, Mapping):
        value = value.get("p95_ms")
    return None if value is None else float(value)


def recommend(engine: Mapping[str, Any], search: Mapping[str, Any] | None = None, *,
              delay: int = GATE_DELAY, row_ms: float | None = None,
              forward_ms: float | None = None, available: int | None = None,
              worker_mb: int = WORKER_MB, stat: str = "p95_ms") -> dict[str, Any]:
    """The knobs the plan sets from the benchmark (plan 10 step 3)."""
    row = _search_ms(search, "row_ms")
    fwd = _search_ms(search, "forward_ms")
    row = row if row is not None else row_ms
    fwd = fwd if fwd is not None else forward_ms
    costs = Costs.from_engine(engine, stat=stat, row_ms=row, forward_ms=fwd)
    measured = (search or {}).get("presets") or {}
    plans = {name: plan_preset(name, costs, delay=delay,
                               measured_p95=(measured.get(name) or {}).get("p95_ms"))
             for name in PRESETS}
    machine = engine.get("machine") or {}
    mem = engine.get("memory") or {}
    avail = next((v for v in (available, mem.get("available_mb"), machine.get("available_mb"))
                  if v is not None), None)
    gate = gate_workers(machine.get("cpus"), avail, worker_mb=worker_mb)
    threads = engine.get("threads") or []
    probes = engine.get("probes") or []
    summary = engine.get("probe_summary") or (probe_summary(probes) if probes else None)
    charge = engine.get("pending_charge") or pending_charge(probes)
    return {"delay": int(delay), "stat": stat, "costs": dataclasses.asdict(costs),
            "priors": {"row_ms": "measured" if _search_ms(search, "row_ms") is not None else
                       ("given" if row_ms is not None else f"prior {ROW_MS_PRIOR}"),
                       "forward_ms": "measured" if _search_ms(search, "forward_ms") is not None
                       else ("given" if forward_ms is not None else f"prior {FORWARD_MS_PRIOR}")},
            "presets": plans,
            "latency_ticks": {n: p["latency_ticks"] for n, p in plans.items()},
            "gate": gate, "thread_scaling_limit": scaling_limit(threads),
            "probe_summary": summary, "pending_charge": charge}


# ---------------------------------------------------------------------------
# rendering and files
# ---------------------------------------------------------------------------

def _fmt(summary: Mapping[str, Any] | None) -> str:
    if not summary or not summary.get("n"):
        return "n/a"
    return f"{summary['p50_ms']:9.3f} {summary['p95_ms']:9.3f} {summary['p99_ms']:9.3f}"


def render_engine(report: Mapping[str, Any]) -> str:
    fp = (report.get("engine") or {}).get("fingerprint") or {}
    m = report.get("machine") or {}
    lines = [f"RoyaleSim search bench, engine ({report.get('created', '?')})",
             f"engine   royalesim {fp.get('version', '?')}, card table "
             f"{fp.get('card_table') or 'n/a'}, id {fp.get('id', '?')}",
             f"machine  {m.get('system')} {m.get('release')} {m.get('machine')}, "
             f"{m.get('cpus')} CPUs, Python {m.get('python')}, "
             f"{m.get('available_mb')} MB free"]
    t = report.get("timings")
    if t:
        lines += ["", f"{'timings (ms)':<34} {'p50':>9} {'p95':>9} {'p99':>9}"]

        def add(label: str, summary: Any, extra: str = "") -> None:
            lines.append(f"  {label:<32} {_fmt(summary)}{('  ' + extra) if extra else ''}")

        add("construct (make_battle)", t.get("construct"))
        for n, sm in (t.get("reset") or {}).items():
            add(f"reset, {n} spawns", sm)
        for k, sm in ((t.get("step") or {}).get("busy") or {}).items():
            add(f"step {k}, busy board", sm)
        for k, sm in ((t.get("step") or {}).get("empty") or {}).items():
            add(f"step {k}, empty board", sm)
        tps = t.get("ticks_per_s") or {}
        lines.append(f"  {'ticks per second (p50)':<32} busy {tps.get('busy')}, "
                     f"empty {tps.get('empty')}")
        add("state_json", t.get("state_json"), f"{t.get('state_bytes')} bytes")
        add("json.loads", t.get("json_loads"))
        add("save", t.get("save"), f"{t.get('save_bytes')} bytes")
        add("load", t.get("load"))
        add("state_hash", t.get("state_hash"))
        add("check_deploy (per call)", t.get("check_deploy"))
        add("building_placement (per call)", t.get("building_placement"))
        comp = t.get("composite") or {}
        add(f"root (reset {comp.get('root_spawns')} + step {comp.get('delay')} + save + state)",
            comp.get("root"))
        for h, sm in (comp.get("trial") or {}).items():
            add(f"trial H {h} (load + step + state)", sm)
    rows = report.get("threads") or []
    if rows:
        lines += ["", "threads  battles  ticks/s  per battle  efficiency"]
        for r in rows:
            lines.append(f"         {r['battles']:>7}  {r.get('ticks_per_s')!s:>7}  "
                         f"{r.get('step_ticks_per_s_per_battle')!s:>10}  "
                         f"{r.get('efficiency')!s:>10}"
                         + (f"  errors: {r['errors']}" if r.get("errors") else ""))
    mem = report.get("memory")
    if mem:
        lines.append(f"memory   {mem.get('rss_mb_per_battle')} MB RSS per Battle, "
                     f"{mem.get('available_mb')} MB free" + (f" ({mem['note']})"
                                                             if mem.get("note") else ""))
    probes = report.get("probes") or []
    if probes:
        lines += ["", "probes"]
        for p in probes:
            lines.append(f"  {p['id']:<4} {verdict(p):<8} {p.get('note', '')}")
            if p.get("error"):
                lines.append(f"       {p['error']}")
        s = probe_summary(probes)
        lines.append(f"  {s['probes']} probes: {s['pass']} pass, {s['fail']} fail, "
                     f"{s['not_run']} not run")
        charge = pending_charge(probes)
        if charge:
            lines.append(f"  pending charge (P16): elixir at {charge['elixir']}, hand at "
                         f"{charge['hand']}")
    return "\n".join(lines)


def render_recommendation(rec: Mapping[str, Any]) -> str:
    c = rec["costs"]
    lines = ["", f"recommendation (engine {rec['stat']}, delay D {rec['delay']}; row "
             f"{c['row_ms']} ms [{rec['priors']['row_ms']}], forward {c['forward_ms']} ms "
             f"[{rec['priors']['forward_ms']}])",
             f"  components: reset {c['reset_ms']:.3f}, tick {c['tick_ms']:.4f}, save "
             f"{c['save_ms']:.3f}, load {c['load_ms']:.3f}, state {c['state_ms']:.3f} ms",
             f"  {'preset':<8} {'config':<22} {'ms':>8} {'budget':>7} {'trials':>6} "
             f"{'latency':>7} {'break-even':>10}  cuts"]
    for name, p in rec["presets"].items():
        cfg = p["config"]
        conf = f"{cfg['trials']} x H {cfg['horizon']} x K {cfg['worlds']}"
        lines.append(f"  {name:<8} {conf:<22} {p['ms']:>8.2f} {p['deadline_ms']:>7.0f} "
                     f"{p['trials']:>6} {p['latency_ticks']!s:>7} {p['break_even_pp']!s:>8} pp"
                     f"  {', '.join(p['cuts']) or '-'}"
                     + (f"  ({p['source']})" if p["source"] != "estimate" else ""))
        for note in p["notes"]:
            lines.append(f"           {note}")
    g = rec["gate"]
    lines.append(f"  gate: {g['workers']} workers x {g['torch_threads']} torch threads "
                 f"({g['cpus']} CPUs, {g['available_mb']} MB free"
                 + (" assumed" if g["available_assumed"] else "")
                 + f", {g['worker_mb']} MB each; limited by {', '.join(g['limited_by'])})")
    if rec.get("thread_scaling_limit") is not None:
        lines.append(f"  engine threads scale to {rec['thread_scaling_limit']} Battles at "
                     f">= {GOOD_EFFICIENCY:.0%} efficiency")
    s = rec.get("probe_summary") or {}
    if s:
        lines.append(f"  probes: {s.get('pass')} pass, {s.get('fail')} fail"
                     + (f" ({', '.join(s.get('failed') or [])}: fix the rebuilder first, "
                        "plan 7.2 M0)" if s.get("fail") else "")
                     + f", {s.get('not_run')} not run")
    return "\n".join(lines)


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_report(report: Mapping[str, Any], out_dir: Path, kind: str, text: str | None = None,
                 when: str | None = None) -> tuple[Path, Path | None]:
    """reports/search_bench/<kind>_<stamp>.json (and .txt with `text`); a clash gets _2, _3."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{kind}_{when or stamp()}"
    name, k = base, 1
    while (out_dir / f"{name}.json").exists():
        k += 1
        name = f"{base}_{k}"
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    txt = None
    if text is not None:
        txt = out_dir / f"{name}.txt"
        txt.write_text(text + "\n", encoding="utf-8")
    return path, txt


def newest(out_dir: Path, kind: str) -> Path | None:
    files = sorted(Path(out_dir).glob(f"{kind}_*.json"))
    return files[-1] if files else None


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _ints(text: str) -> tuple[int, ...]:
    try:
        out = tuple(int(p) for p in str(text).replace(" ", "").split(",") if p)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: comma-separated integers") from None
    if not out or any(v < 1 for v in out):
        raise argparse.ArgumentTypeError(f"{text!r}: integers of at least 1")
    return out


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_engine(args: argparse.Namespace) -> int:
    try:
        cfg = rs_engine.config_from_args(args)
    except (ValueError, FileNotFoundError) as exc:
        _say(f"engine config: {exc}")
        return 2
    only = [p for p in (args.probes or "").split(",") if p.strip()]
    unknown = sorted({p.strip().upper() for p in only} - set(PROBE_IDS))
    if unknown:
        _say(f"--probes: unknown {unknown}; the probes are {list(PROBE_IDS)}")
        return 2
    s = QUICK if args.quick else BenchSettings()
    changes = {k: v for k, v in (("reps", args.reps), ("threads", args.threads),
                                 ("delay", args.delay)) if v is not None}
    s = dataclasses.replace(s, **changes)
    try:
        rs = rs_engine.royalesim_module()
    except Exception as exc:
        _say(f"royalesim does not import ({type(exc).__name__}: {exc}). Run this with the engine "
             r"venv: .venvs\royalegym\Scripts\python.exe tools\rs_search_bench.py engine")
        return 1
    try:
        lab = Lab(cfg, rs)
        fp = rs_engine.fingerprint(lab.battle(), rs, lab.cfg)
    except Exception as exc:
        _say(f"engine: {type(exc).__name__}: {exc}")
        return 1
    _say(f"rs_search_bench engine: {rs_engine.summary(fp)}")
    report: dict[str, Any] = {
        "format": FORMAT, "kind": "engine", "tool": TOOL,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "machine": machine_info(),
        "engine": {"config": lab.cfg.to_json(), "fingerprint": fp},
        "settings": {**dataclasses.asdict(s), "quick": bool(args.quick)},
    }
    if not args.no_timings:
        _say("timings")
        report["timings"] = engine_timings(lab, s, _say)
        report["threads"] = thread_scaling(lab, s, _say)
        report["memory"] = memory_info(lab, s)
    if not args.no_probes:
        _say("probes")
        report["probes"] = run_probes(lab, s, only or None, log=_say)
        report["probe_summary"] = probe_summary(report["probes"])
        report["pending_charge"] = pending_charge(report["probes"])
    text = render_engine(report)
    path, txt = write_report(report, Path(args.out), "engine", text)
    print(text)
    print(f"\nwrote {path}" + (f" and {txt.name}" if txt else ""))
    if args.strict and (report.get("probe_summary") or {}).get("fail"):
        return 1
    return 0


def _load(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a JSON object")
    return data


def run_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.dir)
    epath = Path(args.engine) if args.engine else newest(out_dir, "engine")
    if epath is None or not epath.is_file():
        _say(f"no engine report in {out_dir}: run `tools/rs_search_bench.py engine` first")
        return 1
    spath = Path(args.search) if args.search else newest(out_dir, "search")
    try:
        engine = _load(epath)
        search = _load(spath) if spath is not None and spath.is_file() else None
    except (OSError, ValueError) as exc:
        _say(f"report: {exc}")
        return 1
    text = render_engine(engine)
    if not engine.get("timings"):
        print(text)
        _say(f"{epath.name} has no timings (run without --no-timings): no recommendation")
        return 1
    try:
        rec = recommend(engine, search, delay=args.delay, row_ms=args.row_ms,
                        forward_ms=args.forward_ms, available=args.available_mb,
                        worker_mb=args.worker_mb)
    except (KeyError, TypeError, ValueError) as exc:
        _say(f"report: {epath.name} is not an engine report this tool can read "
             f"({type(exc).__name__}: {exc})")
        return 1
    rec["sources"] = {"engine": epath.name, "search": spath.name if search else None}
    text += "\n" + render_recommendation(rec)
    if args.json:
        print(json.dumps(rec, indent=2))
    else:
        print(text)
    if not args.no_write:
        path, _ = write_report({"format": FORMAT, "kind": "report", "tool": TOOL,
                                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "recommendation": rec}, out_dir, "report")
        _say(f"wrote {path}")
    return 0


def run_search(args: argparse.Namespace) -> int:
    _say("rs_search_bench search needs S08 (SearchSeat in tools/rs_search.py and rs_match's "
         "search options; docs/LIVE_SEARCH_PLAN.md section 9) and, for --backend procs, S13. "
         "It is not built yet: nothing was run. `engine` and `report` work now.")
    return 3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="RoyaleSim timings and engine probes for the live search plan (S01).")
    sub = ap.add_subparsers(dest="command", required=True)

    e = sub.add_parser("engine", help="time the engine, scale threads, run the probes")
    e.add_argument("--quick", action="store_true", help="a short run: 3 samples, 1-2 threads")
    e.add_argument("--reps", type=int, default=None, help="samples per timing (default 50)")
    e.add_argument("--threads", type=_ints, default=None, metavar="N,N",
                   help="Battle counts to scale over (default 1,2,4,8,16)")
    e.add_argument("--delay", type=int, default=None,
                   help=f"command delay of P8 / P16 and the root's advance (default {PROBE_DELAY})")
    e.add_argument("--probes", default="", metavar="P1,P8",
                   help="only these probes (default all)")
    e.add_argument("--no-timings", action="store_true", help="probes only")
    e.add_argument("--no-probes", action="store_true", help="timings only")
    e.add_argument("--strict", action="store_true", help="exit 1 when a probe fails")
    e.add_argument("--out", default=str(OUT_DIR),
                   help="report folder (default reports/search_bench)")
    rs_engine.add_engine_args(e, levels=True)

    r = sub.add_parser("report", help="print the table and the recommendation")
    r.add_argument("--dir", default=str(OUT_DIR), help="where the reports are")
    r.add_argument("--engine", default=None, help="an engine_*.json (default the newest)")
    r.add_argument("--search", default=None, help="a search_*.json (default the newest, if any)")
    r.add_argument("--delay", type=int, default=GATE_DELAY, help="D, decision to landing ticks")
    r.add_argument("--row-ms", type=float, default=None, help=f"obs_row + relabel ms (prior "
                   f"{ROW_MS_PRIOR}; a search report's measurement wins)")
    r.add_argument("--forward-ms", type=float, default=None, help=f"one batched forward, ms "
                   f"(prior {FORWARD_MS_PRIOR}, CUDA)")
    r.add_argument("--available-mb", type=int, default=None,
                   help="free memory for the gate (default: what the engine run measured)")
    r.add_argument("--worker-mb", type=int, default=WORKER_MB,
                   help=f"memory per gate worker (default {WORKER_MB}, plan 6.2)")
    r.add_argument("--json", action="store_true", help="print the recommendation as JSON")
    r.add_argument("--no-write", action="store_true", help="do not write report_<stamp>.json")

    s = sub.add_parser("search", help="profile SearchSeat games (needs S08; not built yet)")
    s.add_argument("--ckpt", default=None)
    s.add_argument("--opp-ckpt", default=None)
    s.add_argument("--games", type=int, default=6)
    s.add_argument("--delay", type=int, default=GATE_DELAY)
    s.add_argument("--noise", default="default")
    s.add_argument("--backend", choices=("inline", "procs"), default="inline")
    s.add_argument("--workers", type=int, default=4)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(None if argv is None else list(argv))
    if args.command == "engine":
        if args.reps is not None and args.reps < 1:
            ap.error("--reps: at least 1")
        return run_engine(args)
    if args.command == "report":
        return run_report(args)
    return run_search(args)


if __name__ == "__main__":
    sys.exit(main())
