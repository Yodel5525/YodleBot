r"""Vectorised rollouts in RoyaleSim (R01): K battles at once in each worker process, one batched
CPU forward per decision round, rs_decide's sampled decision, and the learner's transitions in
tools/rl/ppo.py's rollout format.

    .venvs\royalegym\Scripts\python.exe tools\rl\rollout.py collect --learner runs\rl\NAME\it003.pt ^
        --league runs\rl\NAME\league.json --out runs\rl\NAME\it003 --matches 400 --seed 3
    .venvs\royalegym\Scripts\python.exe tools\rl\rollout.py bench --learner runs\tokens\v2_hog\best.pt ^
        --workers 1,4,8 --matches 64

Pure simulation: nothing here starts, touches or taps the emulator, and nothing imports the
memory reader (scripts/gamestate). The worker interpreter is the engine venv's
(src/hastycr/machine.py py_engine: CPU torch, 1-2 threads per worker); the PPO update (R02,
tools/rl/ppo.py) runs in the GPU venv on what this writes, and the driver (R04) alternates them.

Why (tokens-15): tools/rs_selfplay.py plays 48 single-forward matches per iteration against one
frozen opponent with one deck, rebuilding an expensive look-ahead per decision. RL past
imitation needs tens of thousands of matches under the live conditions (command delay,
perception noise, friendly levels) against a league and the corpus's deck field.

WHAT A BATTLE IS (plan_battles, BattlePlan). One tools/rs_match.MatchSpec: two decks in deal
order with their forms, every card and crown tower at FRIENDLY_LEVEL (11: friendlies), the
command delay D for both seats (resolve_delay: --delay, else data/engine.json's delay when it
sets one, else DEFAULT_DELAY = 24 ticks: the client's 21-22 plus capture, inference and adb),
the opponent memory lag (rs_memory.BOARD_LAG), a reset seed and the engine config. The learner
(a v2 checkpoint, decide "sample") plays the learner deck (default Hog 2.6, all base forms: the
bot's account has no evolutions or heroes) on a team drawn per battle. Its opponent comes from
the league (tools/rl/league.py: League.draw, PFSP over snapshots and anchors): a member with a
checkpoint is a frozen seat (decide `opponent_decide`, default "sample"); a member with no path
(or "idle") is the scripted "none", an idle seat that never plays (a floor); LEARNER (the
league's mirror share) puts the learner on both seats, and both are recorded. The opponent's
deck is drawn from the train split's deck field (tools/deck_stats.py DeckField: IL_Replay decks
by frequency, their recorded forms sampled); a dev or final split is refused, those decks are
the evaluation's (tools/rs_league_eval.py). Everything is drawn from numpy generators seeded by
(seed, stream) and (seed, battle index, stream), so a plan is a pure function of its arguments.

ONE WORKER (run_worker -> Worker.run). Up to K battles (Settings.battles, --battles) are open at
once; when one ends the next plan takes its place. A round is:
  1. advance   every open battle steps (10 ticks, single ticks past tick 6000, as rs_match) to
               its next decision tick: every DECIDE (10) ticks from tick 90, while not over
  2. prepare   each deciding seat (team `first`, then the other) builds its row: the state its
               screen reader reports (perception_noise.MatchNoise.observe, per seat, seeded per
               seat by rs_match.noise_seed), its memory view (taken before anything is sent this
               tick), then rs_eval.RSAgent's row (rs_rows.obs_row, v2 keywords: the opponent's
               bar counted from the plays the engine ran, own plays in flight, the buttons of
               its deal), exactly what rs_match.MatchAgent.act builds
  3. forward   ONE batched CPU forward per policy (Brain) over the rows of every battle: all
               learner seats in one call, each frozen opponent in one call. It gives the gate
               [wait, slot 0..3, buttons], the critic (value2) and the tile logits of all four
               hand slots (Heads), so decide never calls the model again
  4. resolve   rs_decide.decide on those outputs, its sampling preset (DecideConfig.sampling:
               gate, slot, tile and, when the checkpoint's buttons trained, press are drawn;
               masks for elixir, plays in flight, buttons), legality from check_deploy on the
               battle itself (BattleLegality); the commands go into the memories (each seat's
               through its own noise: missed, late), the battle steps, and refusals at send or
               when a delayed command runs are counted (an opponent's refused play leaves that
               seat's elixir count, rs_match V2 SEATS)
This is rs_match.run_match's loop with the seats' decisions split around the batched forward;
its rules for memories, noise, refusals and presses are the same code (rs_match's helpers). It
differs only in the order of the noise draws within a tick (both seats observe before either's
play is recorded) and in counting no elixir leak. mem_lag "legacy" is refused: its view is
taken at each act, which a batched round cannot keep.

THE TRANSITIONS (tools/rl/ppo.py, FORMAT "rl_rollout/v1"). One row per learner decision: every
rs_schema.V2.model_inputs column of the row the model saw (noise included); decide's mask, the
drawn gate entry and its log q, the tile candidates (padded with -1 to 16), the drawn tile and
its log p, the fallback and capped flags, the cadence dt and the critic's raw value
(ppo.decision_record); episode (2 x battle index + team, so a mirror's two seats are two
episodes) and step (the seat's decision count); and the reward parts from the TRUE state at the
decision (ppo.reward_parts: each side's mean crown-tower hit-point fraction and crowns, own then
enemy; their differences are the potential Phi's terms) plus on an episode's last row `done`
and the terminal result (+1 won, -1 lost, 0 a draw; ppo.result_of). A battle cut at max_ticks
before game over has no done row (ppo bootstraps it). Files are written whole episodes at a time
(ppo renumbers episodes per file), a part every Settings.flush_rows rows, as
<out>/<prefix>-w<worker>-<part>.npz, each written to <out>/.tmp first and moved into place, with
the meta ppo.rollout_meta(the learner's sampling config) plus the learner, engine, delay, noise,
worker, part and the episodes it holds. A collection also writes <out>/summary.json and prints
it (results trimmed) as the last stdout line: per battle the opponent, the learner's teams and
score (1 / 0.5 / 0; None for a mirror), winner, crowns, end tick, plays; league_results pairs
(opponent, score) for League.record_many.

DETERMINISM. A worker's battles, seat rngs, noise rngs and deals are seeded, and the batches are
the same for the same plans, workers and K, so a collection is reproducible run for run on one
machine (torch CPU kernels at a fixed thread count). Another K or worker count changes the batch
composition and so the last bits of the logits; draws may then differ.

WINDOWS (spawn). Workers are spawn processes (also off Windows, for one behaviour):
ProcessPoolExecutor(mp_context=spawn, max_tasks_per_child=1), the pool initializer _init_pool
sets the engine config (rs_engine.set_active) and the BLAS thread variables before torch loads,
and the task is plain data (Settings.to_dict, BattlePlan.to_dict: JSON-ready), run by the
module-level run_worker. Workers load their own checkpoints: models never cross processes.

MEMORY (16 GB PC). Each worker holds torch, royalesim, the learner, up to Settings.max_models
opponent models (LRU), K battles and at most flush_rows transition rows: worker_mb() estimates
it (BASE_WORKER_MB + K x BATTLE_MB + rows x ROW_KB), ram_check caps the workers by the budget
(--ram-budget-gb, default 8 GB for the workers while the learner is idle) and by the free memory
less RESERVE_MB, and says so. bench measures the real figure: matches per hour, decisions per
second and each worker's resident and peak memory (rss_mb), for setting --workers and
--worker-mb.

The design follows royalelearn's actor (MIT; not imported). Never imports scripts/gamestate.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import multiprocessing
import os
import sys
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import deck_stats  # noqa: E402  (the deck field; rs_match imports it by this name)
import perception_noise as PN  # noqa: E402
import rs_decide as RD  # noqa: E402
import rs_engine  # noqa: E402
import rs_eval  # noqa: E402  (read-only: checkpoint_v2, seat_visibility)
import rs_match as RM  # noqa: E402  (read-only: MatchSpec, AgentSpec, MatchAgent and its helpers)
import rs_memory  # noqa: E402
import rs_presses  # noqa: E402
import rs_rows as RR  # noqa: E402
from tools.rl import league as LG  # noqa: E402
from tools.rl import ppo as PPO  # noqa: E402  (the rollout format, written with its helpers)

__all__ = [
    "BattlePlan", "BattleSlot", "Brain", "DEFAULT_DELAY", "FORMAT", "Heads", "Request",
    "RolloutWriter", "Seat", "Settings", "Worker", "bench_row", "collect", "default_noise",
    "main", "noise_entry", "plan_battles", "ram_check", "resolve_delay", "rss_mb", "run_worker",
    "worker_mb",
]

FORMAT = PPO.FORMAT
TOOL = "tools/rl/rollout.py"
#: Ticks between a decision and its play running live: 21-22 client delay + capture and taps.
DEFAULT_DELAY = 24
FRIENDLY_LEVEL = RM.FRIENDLY_LEVEL
#: The league's id for the live policy; also the learner seats' agent name.
LEARNER = LG.LEARNER
LEARNER_DECK = RR.HOG26
DEFAULT_WORKERS = 8
DEFAULT_BATTLES = 16
DEFAULT_THREADS = 1
DEFAULT_MATCHES = 128
DEFAULT_FLUSH_ROWS = 20_000
DEFAULT_MAX_MODELS = 6
MAX_ERRORS = 10
#: Memory: the workers' budget, what is kept free for the parent and the system, and the
#: per-worker estimate's parts (torch + royalesim + models; a battle; a buffered row).
RAM_BUDGET_GB = 8.0
RESERVE_MB = 2048
BASE_WORKER_MB = 600
BATTLE_MB = 8
ROW_KB = 3.0
#: Generator streams: a seat's decisions, the plan's league and deck draws, a battle's deals.
SEAT_STREAM, PLAN_STREAM, DEAL_STREAM = 31, 32, 33
MEASURED_NOISE = ROOT / "data" / "noise" / "measured.json"
#: opponent_noise: the learner's profile for the opponent seats too.
SAME = "same"
TMP_DIR = ".tmp"
SUMMARY_JSON = "summary.json"
MB = 1 << 20


def _int(value: Any, what: str, lo: int | None = None) -> int:
    """`value` as a plain int (numpy ints too); bools, other types and values below `lo` refused."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{what}: an int, got {value!r}")
    value = int(value)
    if lo is not None and value < lo:
        raise ValueError(f"{what}: at least {lo}, got {value}")
    return value


def _two(value: Any, what: str) -> tuple:
    try:
        items = tuple(value)
    except TypeError:
        raise ValueError(f"{what}: two items (team 0, team 1), got {value!r}") from None
    if len(items) != 2:
        raise ValueError(f"{what}: two items (team 0, team 1), got {len(items)}")
    return items


def _same_path(a: Any, b: Any) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


# ---------------------------------------------------------------------------
# settings: delay, noise, memory
# ---------------------------------------------------------------------------

def resolve_delay(value: Any = None, engine: Any = None) -> Any:
    """The command delay D in ticks: `value` when it names one (21, "21", "21,22"); else the
    engine config's delay when it sets one above 0 (data/engine.json, HASTYCR_COMMAND_DELAY);
    else DEFAULT_DELAY. An int when both seats are equal, else (blue, red)."""
    d = rs_engine.parse_delay(value) if value is not None else None
    if d is None:
        cfg = rs_engine.resolve(engine)
        if cfg.delay is not None and max(cfg.delay) > 0:
            d = tuple(int(x) for x in cfg.delay)
    if d is None:
        return DEFAULT_DELAY
    return int(d[0]) if d[0] == d[1] else (int(d[0]), int(d[1]))


def default_noise(measured: Any = MEASURED_NOISE, prior: Any = PN.PRIOR_PATH) -> Path:
    """The perception-noise profile of the live screen reader: data/noise/measured.json
    (tools/fit_noise_profile.py) when it exists, else the shipped prior."""
    path = Path(measured)
    return path if path.is_file() else Path(prior)


def noise_entry(noise: Any) -> dict | None:
    """An AgentSpec's noise as plain data: a profile's to_dict(), or None for a perfect observer
    (None, "none", an all-zero profile). Paths and "default" are read here, in the parent."""
    if noise is None or (isinstance(noise, str) and noise.strip().lower() in PN.NO_NOISE):
        return None
    profile = PN.as_profile(noise)
    return None if profile.is_identity() else profile.to_dict()


def worker_mb(battles: int = DEFAULT_BATTLES, flush_rows: int = DEFAULT_FLUSH_ROWS) -> int:
    """The estimated memory of one worker in MB (module doc, MEMORY)."""
    return int(BASE_WORKER_MB + int(battles) * BATTLE_MB + int(flush_rows) * ROW_KB / 1024 + 0.5)


def _available_mb() -> int | None:
    """Free physical memory in MB (tools/rs_league_eval.available_mb), None when unknown."""
    import rs_league_eval
    return rs_league_eval.available_mb()


def ram_check(workers: int, *, per_worker_mb: int, budget_gb: float = RAM_BUDGET_GB,
              available: int | None = None, reserve_mb: int = RESERVE_MB) -> tuple[int, str]:
    """(workers, why): `workers` capped by the budget (budget_gb / per_worker_mb) and, when the
    free memory is known (`available` MB, default the machine's), by (free - reserve_mb) /
    per_worker_mb; at least 1."""
    want = max(1, int(workers))
    per = max(1, int(per_worker_mb))
    caps = {"budget": max(1, int(float(budget_gb) * 1024) // per)}
    free = _available_mb() if available is None else int(available)
    if free is not None:
        caps["free memory"] = max(1, (free - int(reserve_mb)) // per)
    got = min(want, *caps.values())
    shown = "unknown" if free is None else f"{free} MB"
    if got == want:
        return got, (f"{got} worker(s) x {per} MB = {got * per} MB (budget {budget_gb:g} GB, "
                     f"free {shown})")
    limits = ", ".join(f"{k} {v}" for k, v in caps.items() if v == got)
    return got, (f"{got} worker(s), not {want}: capped by {limits} ({per} MB each, budget "
                 f"{budget_gb:g} GB, free {shown}, {reserve_mb} MB kept)")


def rss_mb() -> tuple[float | None, float | None]:
    """(resident, peak resident) memory of this process in MB; None where it cannot be read
    (Windows GetProcessMemoryInfo, Linux /proc/self/status, else getrusage's peak)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            info = k32.K32GetProcessMemoryInfo
            info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            info.restype = wintypes.BOOL
            if info(k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / MB, counters.PeakWorkingSetSize / MB
            return None, None
        status = Path("/proc/self/status")
        if status.is_file():
            vals: dict[str, float] = {}
            for line in status.read_text(encoding="ascii", errors="replace").splitlines():
                key, _, rest = line.partition(":")
                if key in ("VmRSS", "VmHWM") and rest.split():
                    vals[key] = int(rest.split()[0]) / 1024
            return vals.get("VmRSS"), vals.get("VmHWM")
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return None, (peak / MB if sys.platform == "darwin" else peak / 1024)
    except Exception:                       # memory figures are a report, never a failure
        return None, None


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BattlePlan:
    """One battle of a collection (module doc, WHAT A BATTLE IS): its index (unique in the
    collection; the episodes are 2 x index + team), its rs_match.MatchSpec as plain data, which
    teams the learner plays, and the league id of the opponent (LEARNER for a mirror)."""

    index: int
    spec: Any
    learner: tuple = (True, False)
    opponent: str = ""

    def __post_init__(self) -> None:
        put = object.__setattr__
        put(self, "index", _int(self.index, "index", 0))
        spec = RM.MatchSpec.from_dict(self.spec)
        learner = tuple(bool(x) for x in _two(self.learner, "learner"))
        if not any(learner):
            raise ValueError(f"battle {self.index}: no learner seat")
        for t in (0, 1):
            if not learner[t]:
                continue
            if spec.agents[t].idle:
                raise ValueError(f"battle {self.index}: the learner's seat {t} is idle")
            if spec.agents[t].decide != RD.MODE_SAMPLE:
                raise ValueError(f"battle {self.index}: the learner samples (decide "
                                 f"{RD.MODE_SAMPLE!r}), got {spec.agents[t].decide!r}")
        if spec.mem_lag == RR.MEM_LEGACY:
            raise ValueError("mem_lag 'legacy' reads the memory at each act, which a batched round "
                             "cannot keep: use a lag in ticks (rs_memory.BOARD_LAG)")
        put(self, "spec", spec.to_dict())
        put(self, "learner", learner)
        put(self, "opponent", str(self.opponent or ""))

    def match_spec(self) -> RM.MatchSpec:
        return RM.MatchSpec.from_dict(self.spec)

    def episode(self, team: int) -> int:
        return 2 * self.index + int(team)

    @property
    def mirror(self) -> bool:
        return all(self.learner)

    def to_dict(self) -> dict:
        return {"index": self.index, "spec": self.spec, "learner": list(self.learner),
                "opponent": self.opponent}

    @classmethod
    def from_dict(cls, data: Any) -> "BattlePlan":
        if isinstance(data, cls):
            return data
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"BattlePlan: unknown field(s) {unknown}")
        return cls(**dict(data))


def plan_battles(n: int, *, learner: Any, league: Any, deck_field: Any, seed: int = 0,
                 start: int = 0, learner_deck: Iterable[str] = LEARNER_DECK,
                 delay: Any = DEFAULT_DELAY, level: int = FRIENDLY_LEVEL, noise: Any = None,
                 opponent_noise: Any = SAME, noise_seed: int = 0, presses: bool = True,
                 opponent_decide: str = RD.MODE_SAMPLE, engine: Any = None,
                 mem_lag: Any = rs_memory.BOARD_LAG,
                 max_ticks: int = RM.MAX_TICKS) -> list[BattlePlan]:
    """`n` battles, indexes start .. start + n - 1 (module doc, WHAT A BATTLE IS).

    league        anything with draw(rng, size) -> ids and member(id).path (tools/rl/league.py)
    deck_field    anything with sample(rng, size) -> DeckChoices (names, forms) and split "train"
    noise         the learner seats' perception noise (a profile, its dict, a path, "default",
                  None / "none"); opponent_noise the opponent seats' (SAME: the learner's)
    presses       the seats may press their buttons (only where the checkpoint's buttons
                  trained: rs_decide.presses_allowed)
    delay         ticks (an int or (blue, red)); None keeps the engine config's
    """
    n = _int(n, "n", 0)
    seed = _int(seed, "seed", 0)
    start = _int(start, "start", 0)
    split = getattr(deck_field, "split", "train")
    if split != "train":
        raise ValueError(f"the opponents' decks come from the train split, not {split!r}: the val "
                         "(dev) and test (final) decks are the evaluation's")
    if opponent_decide not in RD.MODES:
        raise ValueError(f"opponent_decide: one of {', '.join(RD.MODES)}, got {opponent_decide!r}")
    mine_deck = sorted(str(c) for c in learner_deck)
    if len(mine_deck) != 8 or len(set(mine_deck)) != 8:
        raise ValueError(f"the learner deck is 8 different cards, got {mine_deck}")
    own_noise = noise_entry(noise)
    opp_noise = own_noise if (isinstance(opponent_noise, str) and opponent_noise == SAME) \
        else noise_entry(opponent_noise)
    engine_json = None if engine is None else rs_engine.EngineConfig.coerce(engine).to_json()
    rng = np.random.default_rng([seed, PLAN_STREAM])
    ids = [str(i) for i in league.draw(rng, size=n)] if n else []
    choices = list(deck_field.sample(rng, size=n)) if n else []
    me = RM.AgentSpec(str(learner), name=LEARNER, decide=RD.MODE_SAMPLE, noise=own_noise,
                      noise_seed=noise_seed, presses=presses)
    plans = []
    for i in range(n):
        index = start + i
        brng = np.random.default_rng([seed, index, DEAL_STREAM])
        team = int(brng.integers(2))
        mine = RM.deal(mine_deck, [0] * len(mine_deck), brng)
        names = getattr(choices[i], "names", None)
        if not names:
            raise ValueError("the deck field has no RoyaleSim names (deck stats made with "
                             "tools/deck_stats.py --no-names?)")
        forms = list(getattr(choices[i], "forms", None) or [0] * len(names))
        theirs = RM.deal(list(names), forms, brng)
        battle_seed = int(brng.integers(2 ** 31))
        opp = ids[i]
        if opp == LEARNER:
            other, seats = me, (True, True)
        else:
            path = league.member(opp).path
            if path is None or not str(path).strip() or str(path).strip().lower() == RM.IDLE:
                other = RM.AgentSpec("", name=opp)
            else:
                other = RM.AgentSpec(str(path), name=opp, decide=opponent_decide, noise=opp_noise,
                                     noise_seed=noise_seed, presses=presses, seed_offset=1)
            seats = tuple(t == team for t in (0, 1))
        agents, decks, deck_forms = [None, None], [None, None], [None, None]
        agents[team], agents[1 - team] = me, other
        decks[team], decks[1 - team] = mine[0], theirs[0]
        deck_forms[team], deck_forms[1 - team] = mine[1], theirs[1]
        tag = f"rl:{seed}:{index}"
        spec = RM.MatchSpec(decks=tuple(decks), agents=tuple(agents), forms=tuple(deck_forms),
                            levels=level, tower_levels=level, delay=delay, seed=battle_seed,
                            engine=engine_json, mem_lag=mem_lag, max_ticks=max_ticks, tag=tag,
                            pairing=tag, seat=team)
        plans.append(BattlePlan(index, spec, seats, opp))
    return plans


# ---------------------------------------------------------------------------
# policies: one Brain per checkpoint, the batched forward and its per-row heads
# ---------------------------------------------------------------------------

def _model_tokens() -> Any:
    """sim/model_tokens.py (imports torch), loaded when a policy is."""
    return importlib.import_module("sim.model_tokens")


class Heads:
    """One row's outputs of a batched forward, as rs_decide.decide asks a policy for them:
    batch(row) (itself), heads(b) -> (gate logits [5 + B], the critic's raw value or None),
    gate(b) and tiles(b, slot) -> the 576 tile logits for the card in that hand slot."""

    __slots__ = ("_gate", "_value", "_tiles")

    def __init__(self, gate: np.ndarray, value: tuple | None, tiles: np.ndarray):
        self._gate = gate
        self._value = value
        self._tiles = tiles

    def batch(self, row: Mapping[str, Any]) -> "Heads":
        return self

    def heads(self, b: Any) -> tuple[np.ndarray, tuple | None]:
        return self._gate, self._value

    def gate(self, b: Any) -> np.ndarray:
        return self._gate

    def tiles(self, b: Any, slot: int) -> np.ndarray:
        return self._tiles[int(slot)]


class Brain:
    """One loaded TokenPolicy (eval mode) shared by every seat of a worker that plays it, with
    what rs_eval.RSAgent reads from its meta: visibility, live tokens, v2 rows, the wait bias
    (log(1 / wait_keep), wait_keep from the run's args.json unless given)."""

    def __init__(self, model: Any, meta: Mapping[str, Any], *, ckpt: str = "",
                 wait_keep: float | None = None, drift: Sequence[str] = (), torch: Any = None):
        self.torch = importlib.import_module("torch") if torch is None else torch
        self.model = model.eval()
        self.meta = dict(meta or {})
        self.ckpt = str(ckpt or "")
        self.drift = list(drift)
        units = self.meta.get("units_mode", RR.UNITS_TYPES)
        if units != RR.UNITS_TYPES:
            raise ValueError(f"{self.ckpt or 'checkpoint'}: a --units {units} checkpoint; the "
                             f"rollouts build --units {RR.UNITS_TYPES} rows only (as rs_eval)")
        self.visibility = RR.visibility_of(self.meta)
        self.live = bool(self.meta.get("live_compat", False))
        self.v2 = rs_eval.checkpoint_v2(self.meta)
        self.schema = RD.model_schema_of(model)
        if self.v2 != (self.schema is not None):
            raise ValueError(f"{self.ckpt or 'checkpoint'}: the meta says schema "
                             f"{self.meta.get('schema')!r} but the model is "
                             f"{'v2' if self.schema is not None else 'legacy'}")
        keep = (RD.wait_keep_for(self.ckpt or None, RD.DEFAULT_WAIT_KEEP) if wait_keep is None
                else float(wait_keep))
        self.wait_keep = keep
        self.wait_bias = math.log(1 / keep)
        self.has_critic = bool(getattr(model, "has_critic", False))
        self.forward_s = 0.0
        self.calls = 0
        self.rows = 0

    @classmethod
    def from_checkpoint(cls, ckpt: Any, engine: Any = None, *, allow_drift: bool | None = None,
                        wait_keep: float | None = None) -> "Brain":
        """rs_checkpoint.load_policy (the vocabulary checked against `engine`, a strict load)."""
        import rs_checkpoint
        model, meta, drift = rs_checkpoint.load_policy(ckpt, rs_engine.resolve(engine),
                                                       allow_drift=allow_drift)
        return cls(model, meta, ckpt=str(ckpt), wait_keep=wait_keep, drift=drift)

    def forward(self, rows: Sequence[Mapping[str, Any]]) -> list[Heads]:
        """ONE forward over `rows` (obs_row dicts of this model's schema): the gate and the
        critic as TokenPolicyAdapter.heads gives them (outputs() with card_ids = hand slot 0),
        and the tile head on each hand slot's card (TokenPolicyAdapter.tiles), all from one
        encoder pass."""
        if not rows:
            return []
        t0 = time.perf_counter()
        t = self.torch
        b = _model_tokens().rows_batch(rows, self.schema, t)
        with t.inference_mode():
            hand = b["hand"]
            out = self.model.outputs(b, card_ids=hand[:, 0].clamp(min=0))
            cls = out["cls"]
            n, k = int(hand.shape[0]), int(hand.shape[1])
            tile = self.model.tile_head(t.cat([cls.repeat_interleave(k, dim=0),
                                               self.model.card_emb(hand.reshape(-1))], -1))
            tiles = tile.reshape(n, k, -1).float().numpy()
            gate = out["gate"].float().numpy()
            value = out.get("value2")
            value = None if value is None else value.float().numpy()
        self.forward_s += time.perf_counter() - t0
        self.calls += 1
        self.rows += n
        return [Heads(gate[i], None if value is None else tuple(float(v) for v in value[i]),
                      tiles[i]) for i in range(n)]


@dataclasses.dataclass
class Request:
    """One seat's decision of one round: its row and the state it decided on (noisy), the
    tick it read (tick + clock noise), the spell-aim hook, and (after the forward) its heads."""

    slot: Any
    seat: Any
    team: int
    row: dict
    state: dict
    tick: int
    aim: Any = None
    heads: Any = None


class Seat(RM.MatchAgent):
    """A battle seat played by a Brain: rs_match.MatchAgent (RSAgent's rows, the match cadence,
    the sampling preset, v2 own_pending / opp_elixir / presses) whose act() is split in two
    around the batched forward, prepare() -> Request and finish() -> rs_decide.Decision. It
    never loads a model (RSAgent.__init__ is not called): the model is the brain's."""

    def __init__(self, brain: Brain, agent: RM.AgentSpec, team: int, spec: RM.MatchSpec,
                 engine: Any):
        self.brain = brain
        self.model, self.meta, self.torch = brain.model, brain.meta, brain.torch
        self.vocab_drift = brain.drift
        self.visibility, self.live, self.v2 = brain.visibility, brain.live, brain.v2
        self.wait_bias = brain.wait_bias
        self.rng = np.random.default_rng([int(spec.seed), int(team), SEAT_STREAM,
                                          int(agent.seed_offset) & 0xFFFFFFFF])
        self.cadence = int(spec.cadence)
        self._dcfg = None
        RM.configure(self, agent, rs_engine.EngineConfig.coerce(engine).undelayed())

    def decide_config(self) -> RD.DecideConfig:
        """MatchAgent.decide_config, made once: the switches do not change within a battle."""
        if self._dcfg is None:
            self._dcfg = super().decide_config()
        return self._dcfg

    def prepare(self, battle: Any, state: Mapping[str, Any], team: int, mem: Mapping[str, Any],
                tick: int, sub: int, royalesim: Any) -> Request | None:
        """RSAgent.act up to the decision: the hold, the look-ahead and the row. None when the
        seat holds (hold_pending with its own play in flight)."""
        self._team = team
        if getattr(self, "hold_pending", False) and battle.pending_commands(team):
            self.ahead(battle, state)
            return None
        view = self.ahead(battle, state)
        row = RR.obs_row(view, team, sub, mem, tick + self.lead, live=self.live,
                         visibility=self.visibility,
                         **self.v2_kw(state, team, mem, tick + self.lead, royalesim))
        aim = self._spell_aim(view, team, sub) if getattr(self, "spell_aim", False) else None
        return Request(None, self, int(team), row, state, int(tick), aim)

    def finish(self, req: Request, battle: Any, sub: int) -> RD.Decision:
        """rs_decide.decide on the request's heads, check_deploy on the battle itself."""
        if req.heads is None:
            raise RuntimeError("finish() before the forward gave this request its heads")
        d = RD.decide(req.heads, req.row, req.state, req.team, RD.BattleLegality(battle),
                      self.decide_config(), rng=self.rng, tick=req.tick, sub=sub, aim=req.aim)
        self.last_decision = d
        return d


# ---------------------------------------------------------------------------
# one battle
# ---------------------------------------------------------------------------

def _sampling_key(cfg: RD.DecideConfig) -> tuple:
    return (round(float(cfg.wait_bias), 12), float(cfg.explore), float(cfg.tile_temp),
            int(cfg.dt), bool(cfg.presses), int(cfg.sample_top), int(cfg.sample_extra))


class BattleSlot:
    """One open battle of a worker: rs_match.run_match's state and loop (module doc, ONE
    WORKER), stepped by the worker a round at a time: advance() -> prepare() -> (the worker's
    forward) -> resolve()."""

    def __init__(self, plan: BattlePlan, worker: "Worker"):
        self.plan = plan
        self.spec = spec = plan.match_spec()
        cfg = spec.engine_config()
        royalesim, _names, ids, cat = worker.ctx
        self.royalesim = royalesim
        self.sub = royalesim.SUBTILE_PER_MILLITILE
        self.lag = spec.mem_lag
        self.delays = tuple(int(d) for d in (cfg.delay or (0, 0)))
        self.cadence, self.first_decision = spec.cadence, spec.first_decision
        self.max_ticks = spec.max_ticks
        self.learner = tuple(plan.learner)
        self.timings: Counter = Counter()
        t0 = time.perf_counter()
        self.battle = rs_engine.make_battle(cfg, royalesim)
        dealt = spec.reset_kwargs(ids)
        self.battle.reset(**dealt)
        self.timings["engine_s"] += time.perf_counter() - t0
        self.seats: list[Seat | None] = []
        champs = None
        for t in (0, 1):
            agent = spec.agents[t]
            if agent.idle:
                self.seats.append(None)
                continue
            if self.learner[t]:
                if not _same_path(agent.ckpt, worker.settings.learner):
                    raise ValueError(f"battle {plan.index}: the learner seat plays {agent.ckpt}, "
                                     f"this worker's learner is {worker.settings.learner}")
                brain = worker.learner
            else:
                brain = worker.brain(agent.ckpt)
            seat = Seat(brain, agent, t, spec, cfg)
            if seat.v2 or rs_eval.seat_visibility(seat) == RR.VIS_PUBLIC:
                RR.check_status_bits(royalesim)       # a v2 row reads the burrow bit
            if seat.v2:
                if champs is None:
                    champs = rs_presses.champion_ids(cat, getattr(royalesim, "CATALOGUE_FIELDS",
                                                                  None))
                seat.buttons = rs_presses.side_buttons(dealt["decks"][t], dealt["forms"][t],
                                                       champs)
            if self.learner[t]:
                worker.note_sampling(seat.decide_config())
            self.seats.append(seat)
        self.unran = [Counter(), Counter()]
        for t, seat in enumerate(self.seats):
            if seat is not None:
                seat.unran = self.unran[t]
        self.noise = []
        for t in (0, 1):
            model = RM.noise_model(spec.agents[t].noise, self.battle, self.sub,
                                   worker.noise_models)
            self.noise.append(None if model is None else PN.MatchNoise(model, RM.noise_seed(spec, t)))
        self.hand_size = rs_presses.hand_size(royalesim)
        n_buttons = rs_presses.button_count(royalesim)
        self.press_slots = [n_buttons if spec.agents[t].presses else 0 for t in (0, 1)]
        self.mems = [rs_memory.PublicMemory(), rs_memory.PublicMemory()]
        self.order = (spec.first, 1 - spec.first)
        self.plays, self.refused, self.decisions = [0, 0], [0, 0], [0, 0]
        self.pressed, self.press_refused = [0, 0], [0, 0]
        self.log: list[list[int]] = []
        self.entries: dict[int, tuple] = {}
        self.exec_of: dict[int, int] = {}
        self.flying: list[int] = []
        self.rows: tuple[list, list] = ([], [])
        self.tick = 0
        self.state: dict | None = None
        self.final: dict | None = None
        self.reqs: list[Request] = []
        self.done = False
        self.game_over = False
        self.failed = False

    # ------------------------------------------------------------ engine
    def _read(self) -> dict:
        t0 = time.perf_counter()
        state = json.loads(self.battle.state_json())
        self.timings["engine_s"] += time.perf_counter() - t0
        return state

    def _never_ran(self, k: int) -> None:
        """Play k was refused: each seat's memory entry of it goes into that seat's unran."""
        who, got = self.entries.pop(k, (None, ()))
        for obs, entry in enumerate(got):
            if entry is not None:
                self.unran[obs][(who, entry[0], entry[1])] += 1

    def _step(self, cmds: list, sent: list) -> None:
        """One step to the next decision tick (single ticks past END_TICK) with `cmds`, and the
        refusals of what was sent and of the delayed commands that ran (rs_match.run_match)."""
        tick = self.tick
        step = max(1, min(RM.END_TICK, (tick // self.cadence + 1) * self.cadence) - tick)
        step = max(1, min(step, self.max_ticks - tick))
        t0 = time.perf_counter()
        res = list(self.battle.step(cmds, step) or ())
        ran = RM._ran_rows(self.battle)
        self.timings["engine_s"] += time.perf_counter() - t0
        for i, (team, k, is_press) in enumerate(sent):
            reason = RM._reason(res[i]) if i < len(res) else 0
            self.log[k][6] = reason
            if is_press:
                if reason != 0:
                    self.press_refused[team] += 1
                continue
            if reason == 0:
                if self.delays[team] > 0:
                    self.flying.append(k)
            else:
                self.refused[team] += 1
                self._never_ran(k)
        for run_tick, team, kind, what, reason in ran:
            if kind == rs_memory.RAN_DEPLOY:
                k = RM._ran_play(self.flying, self.log, self.exec_of, team, what, run_tick)
                if reason != rs_memory.RAN_OK:
                    self.refused[team] += 1
                    if k is not None:
                        self._never_ran(k)
            elif reason != rs_memory.RAN_OK:
                self.press_refused[team] += 1
        self.tick += step

    # ------------------------------------------------------------ the round
    def advance(self) -> bool:
        """Step until a decision tick with a seat to decide: True; False when the battle is
        over (game over: the learner's last rows become done) or cut at max_ticks."""
        while True:
            state = self._read()
            if state["game_over"]:
                self._finish(state, over=True)
                return False
            if self.tick >= self.max_ticks:
                self._finish(state, over=False)
                return False
            self.state = state
            deciding = self.tick % self.cadence == 0 and self.tick >= self.first_decision
            if deciding and any(seat is not None for seat in self.seats):
                return True
            for nz in self.noise:
                if nz is not None:
                    nz.push(state)
            self._step([], [])

    def prepare(self) -> list[Request]:
        """Each deciding seat's request (module doc, ONE WORKER, 2)."""
        t0 = time.perf_counter()
        state, tick = self.state, self.tick
        views = [self.mems[t].view(tick, self.lag) for t in (0, 1)]
        reqs = []
        for team in self.order:
            seat, nz = self.seats[team], self.noise[team]
            if seat is None:
                if nz is not None:
                    nz.push(state)
                continue
            seen, shift = state, 0
            if nz is not None:
                seen = nz.observe(state, team)
                shift = nz.tick_shift
            req = seat.prepare(self.battle, seen, team, views[team], tick + shift, self.sub,
                               self.royalesim)
            if req is not None:
                req.slot = self
                reqs.append(req)
        self.reqs = reqs
        self.timings["rows_s"] += time.perf_counter() - t0
        return reqs

    def resolve(self) -> None:
        """Decide each request on its heads, record the learner's, send, step (module doc, 4)."""
        state, tick = self.state, self.tick
        cmds, sent = [], []
        for req in self.reqs:
            team, seat = req.team, req.seat
            t0 = time.perf_counter()
            d = seat.finish(req, self.battle, self.sub)
            self.timings["decide_s"] += time.perf_counter() - t0
            self.decisions[team] += 1
            if self.learner[team]:
                t0 = time.perf_counter()
                self._record(team, req, d)
                self.timings["record_s"] += time.perf_counter() - t0
            cmd = d.command
            if cmd is None:
                continue
            cmd = RM._command(cmd, team, state, buttons=self.press_slots[team],
                              hand_size=self.hand_size)
            hand = state["players"][team]["hand"]
            cmds.append(cmd)
            exec_tick = tick + self.delays[team]
            if cmd[1] >= len(hand):
                # A press (its spec lets the seat press): no card play, no memory entry.
                self.pressed[team] += 1
                sent.append((team, len(self.log), True))
                self.log.append([tick, team, RD.button_card(state, team, cmd[1] - self.hand_size),
                                 cmd[1], cmd[2], cmd[3], 0])
                continue
            card = int(hand[cmd[1]])
            seat.sent = list(getattr(seat, "sent", None) or []) + [(exec_tick, card, cmd[2], cmd[3])]
            token = card + rs_memory.TOKEN_OFFSET
            got = []
            for obs in (0, 1):
                if obs != team and self.noise[obs] is not None:
                    seen_as = RM._Recorder(self.mems[obs])
                    self.noise[obs].record(seen_as, team, token, exec_tick)
                    got.append(seen_as.got)                 # None: that seat missed the play
                else:
                    self.mems[obs].record(team, token, exec_tick)
                    got.append((token, exec_tick))
            self.plays[team] += 1
            self.entries[len(self.log)] = (team, tuple(got))
            self.exec_of[len(self.log)] = exec_tick
            sent.append((team, len(self.log), False))
            self.log.append([tick, team, card, cmd[1], cmd[2], cmd[3], 0])
        self.reqs = []
        self._step(cmds, sent)

    # ------------------------------------------------------------ the learner's rows
    def _record(self, team: int, req: Request, d: RD.Decision) -> None:
        rec = {name: req.row[name] for name in PPO.OBS_FIELDS}
        rec.update(PPO.decision_record(d, dt=req.seat.cadence))
        hp, crowns = PPO.reward_parts(self.state, team)        # the TRUE state
        rows = self.rows[team]
        rec.update(episode=np.int32(self.plan.episode(team)), step=np.int32(len(rows)),
                   done=False, result=np.float32(0.0), hp_frac=hp, crowns=crowns)
        rows.append(rec)

    def _finish(self, state: dict, *, over: bool) -> None:
        self.final = state
        self.done = True
        self.game_over = bool(over)
        if over:
            for team in (0, 1):
                if self.learner[team] and self.rows[team]:
                    last = self.rows[team][-1]
                    last["done"] = True
                    last["result"] = np.float32(PPO.result_of(state, team))

    def episodes(self) -> list[tuple[dict, list]]:
        """(info, rows) of each learner seat that decided."""
        out = []
        for team in (0, 1):
            rows = self.rows[team]
            if not (self.learner[team] and rows):
                continue
            out.append(({"episode": self.plan.episode(team), "index": self.plan.index,
                         "team": team, "opponent": self.plan.opponent, "rows": len(rows),
                         "done": bool(rows[-1]["done"]), "result": float(rows[-1]["result"])},
                        rows))
        return out

    def result(self) -> dict:
        """The battle's record for the summary (module doc, THE TRANSITIONS)."""
        st = self.final or self.state or {}
        w = st.get("winner")
        winner = int(w) if w in (0, 1) else -1
        players = st.get("players") or [{}, {}]
        teams = [t for t in (0, 1) if self.learner[t]]
        score = None
        if len(teams) == 1:
            score = 0.5 if winner < 0 else (1.0 if winner == teams[0] else 0.0)
        return {"index": self.plan.index, "opponent": self.plan.opponent,
                "learner_teams": teams, "score": score, "winner": winner,
                "crowns": [int(players[t].get("crowns", 0) or 0) for t in (0, 1)],
                "game_over": self.game_over, "end_tick": int(st.get("tick", self.tick)),
                "plays": list(self.plays), "refused": list(self.refused),
                "presses": list(self.pressed), "press_refused": list(self.press_refused),
                "decisions": list(self.decisions),
                "rows": [len(self.rows[t]) for t in (0, 1)],
                "decks": [list(d) for d in self.spec.decks]}


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

class RolloutWriter:
    """Whole episodes into FORMAT .npz parts (tools/rl/ppo.write_rollout), a part once
    `flush_rows` rows are buffered; each part written under <out>/.tmp and moved into place."""

    def __init__(self, out: Any, prefix: str, worker: int, meta: Mapping[str, Any],
                 flush_rows: int = DEFAULT_FLUSH_ROWS):
        self.out = Path(out)
        self.prefix = str(prefix)
        self.worker = int(worker)
        self.meta = dict(meta)
        self.flush_rows = max(1, int(flush_rows))
        self.sampling: RD.DecideConfig | None = None
        self.records: list[dict] = []
        self.episodes: list[dict] = []
        self.files: list[str] = []
        self.part = 0

    def add(self, info: Mapping[str, Any], rows: Sequence[dict]) -> None:
        self.records.extend(rows)
        self.episodes.append(dict(info))
        if len(self.records) >= self.flush_rows:
            self.flush()

    def flush(self) -> Path | None:
        if not self.records:
            return None
        if self.sampling is None:
            raise RuntimeError("no learner sampling config: no learner seat was opened")
        meta = PPO.rollout_meta(self.sampling, **self.meta, worker=self.worker, part=self.part,
                                episodes=self.episodes)
        name = f"{self.prefix}-w{self.worker:02d}-{self.part:03d}.npz"
        tmp = self.out / TMP_DIR / name
        written = PPO.write_rollout(tmp, self.records, meta)
        final = self.out / name
        os.replace(written, final)
        self.files.append(str(final))
        self.part += 1
        self.records, self.episodes = [], []
        return final


# ---------------------------------------------------------------------------
# one worker
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Settings:
    """What every worker of a collection shares (plain data: Windows spawn). learner: the
    learner's checkpoint; battles: K, the battles open at once (the forward batch is up to 2K
    rows); threads: torch CPU threads per worker (0 leaves torch's); flush_rows: rows buffered
    before a part is written; engine: the engine config (to_json; None: the process's);
    allow_drift: rs_checkpoint's (None: HASTYCR_ALLOW_VOCAB_DRIFT); wait_keep: the learner's
    (None: its args.json); max_models: opponent models kept per worker; write: write the rows
    (bench: False); strict: re-raise a battle's error (tests); max_errors: battle errors
    tolerated per worker; meta: provenance for the files' meta."""

    learner: str
    battles: int = DEFAULT_BATTLES
    threads: int = DEFAULT_THREADS
    flush_rows: int = DEFAULT_FLUSH_ROWS
    engine: Any = None
    allow_drift: bool | None = None
    wait_keep: float | None = None
    max_models: int = DEFAULT_MAX_MODELS
    write: bool = True
    strict: bool = False
    max_errors: int = MAX_ERRORS
    meta: Any = None

    def __post_init__(self) -> None:
        put = object.__setattr__
        put(self, "learner", "" if self.learner is None else str(self.learner))
        if not self.learner.strip():
            raise ValueError("Settings.learner: the learner's checkpoint")
        for what, lo in (("battles", 1), ("threads", 0), ("flush_rows", 1), ("max_models", 1),
                         ("max_errors", 0)):
            put(self, what, _int(getattr(self, what), what, lo))
        if self.engine is not None:
            engine = json.loads(self.engine) if isinstance(self.engine, str) else self.engine
            put(self, "engine", rs_engine.EngineConfig.coerce(engine).to_json())
        if self.allow_drift is not None:
            put(self, "allow_drift", bool(self.allow_drift))
        if self.wait_keep is not None:
            keep = float(self.wait_keep)
            if not 0.0 < keep <= 1.0:
                raise ValueError(f"wait_keep: in (0, 1], got {self.wait_keep!r}")
            put(self, "wait_keep", keep)
        put(self, "write", bool(self.write))
        put(self, "strict", bool(self.strict))
        put(self, "meta", json.loads(json.dumps(dict(self.meta or {}))))

    def replace(self, **changes: Any) -> "Settings":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, data: Any) -> "Settings":
        if isinstance(data, cls):
            return data
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"Settings: unknown field(s) {unknown}")
        return cls(**dict(data))


def _set_threads(threads: int) -> None:
    import torch
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:                    # already set, or parallel work already ran
        pass


class Worker:
    """K battles at a time (module doc, ONE WORKER). `learner` (a Brain) stands in for loading
    Settings.learner (tests); `out` None writes nothing."""

    def __init__(self, settings: Any, worker: int = 0, *, out: Any = None,
                 prefix: str = "rollout", learner: Brain | None = None):
        self.settings = s = Settings.from_dict(settings)
        self.id = int(worker)
        if s.engine is not None:
            rs_engine.set_active(s.engine)
        self.cfg = rs_engine.active()
        if s.threads > 0:
            _set_threads(s.threads)
        self.ctx = RR._setup(self.cfg)
        self.learner = learner if learner is not None else Brain.from_checkpoint(
            s.learner, self.cfg, allow_drift=s.allow_drift, wait_keep=s.wait_keep)
        if not self.learner.v2:
            raise ValueError(f"{s.learner}: the learner must be a v2 checkpoint (rs_rows/v2): "
                             "tools/rl/ppo.py trains v2 rows")
        self.brains: OrderedDict = OrderedDict()
        self.noise_models: dict = {}
        self.sampling: RD.DecideConfig | None = None
        self.writer = None
        if s.write and out is not None:
            meta = {"tool": TOOL, "learner": s.learner, "engine": self.cfg.to_json(),
                    "run": s.meta}
            self.writer = RolloutWriter(out, prefix, self.id, meta, s.flush_rows)
        self.results: list[dict] = []
        self.errors: list[dict] = []
        self.timings: Counter = Counter()
        self.rows = 0
        self.decisions = 0
        self.rounds = 0
        self.forwards = 0
        self.forward_rows = 0

    def brain(self, ckpt: str) -> Brain:
        """The frozen opponent policy of `ckpt`, loaded once and kept (LRU, max_models); a
        checkpoint that failed to load fails again without a reload."""
        key = str(ckpt)
        hit = self.brains.get(key)
        if hit is None:
            try:
                hit = Brain.from_checkpoint(key, self.cfg, allow_drift=self.settings.allow_drift)
            except Exception as exc:
                hit = exc
            self.brains[key] = hit
            while len(self.brains) > self.settings.max_models:
                self.brains.popitem(last=False)
        else:
            self.brains.move_to_end(key)
        if isinstance(hit, Exception):
            raise hit
        return hit

    def note_sampling(self, cfg: RD.DecideConfig) -> None:
        """Every learner seat must sample under one config: the update recomputes the
        probabilities with the meta's (tools/rl/ppo.py META_AGREE)."""
        if not cfg.sample:
            raise ValueError("the learner's decide config does not sample")
        if self.sampling is None:
            self.sampling = cfg
            if self.writer is not None:
                self.writer.sampling = cfg
        elif _sampling_key(cfg) != _sampling_key(self.sampling):
            raise ValueError(f"learner seats sample under different settings: {cfg} vs "
                             f"{self.sampling}")

    # ------------------------------------------------------------ errors
    def _fail(self, slot: Any, plan: BattlePlan, exc: BaseException) -> None:
        if self.settings.strict:
            raise exc
        if slot is not None:
            slot.failed = True
        self.errors.append({"index": plan.index, "opponent": plan.opponent,
                            "tick": None if slot is None else slot.tick,
                            "error": f"{type(exc).__name__}: {exc}"})
        if len(self.errors) > self.settings.max_errors:
            raise RuntimeError(f"worker {self.id}: {len(self.errors)} battles failed (more than "
                               f"{self.settings.max_errors}); the last: {self.errors[-1]}") from exc

    def _open(self, plan: BattlePlan) -> BattleSlot | None:
        try:
            return BattleSlot(plan, self)
        except Exception as exc:
            self._fail(None, plan, exc)
            return None

    def _close(self, slot: BattleSlot) -> None:
        for info, rows in slot.episodes():
            self.rows += len(rows)
            if self.writer is not None:
                self.writer.add(info, rows)
        self.results.append(slot.result())
        self.decisions += sum(slot.decisions)
        self.timings.update(slot.timings)

    def _forward(self, reqs: Sequence[Request]) -> None:
        """One forward per brain over its requests, in order of first appearance (a brain's
        error is the worker's: every battle shares it)."""
        groups: dict[int, tuple[Brain, list]] = {}
        for r in reqs:
            groups.setdefault(id(r.seat.brain), (r.seat.brain, []))[1].append(r)
        for brain, group in groups.values():
            t0 = time.perf_counter()
            heads = brain.forward([r.row for r in group])
            self.timings["forward_s"] += time.perf_counter() - t0
            self.forwards += 1
            self.forward_rows += len(group)
            for r, h in zip(group, heads):
                r.heads = h

    # ------------------------------------------------------------ the loop
    def run(self, plans: Iterable[Any]) -> "Worker":
        queue = deque(BattlePlan.from_dict(p) for p in plans)
        active: list[BattleSlot] = []
        while queue or active:
            while queue and len(active) < self.settings.battles:
                slot = self._open(queue.popleft())
                if slot is not None:
                    active.append(slot)
            ready = []
            for slot in active:
                try:
                    more = slot.advance()
                except Exception as exc:
                    self._fail(slot, slot.plan, exc)
                    continue
                if more:
                    ready.append(slot)
                else:
                    self._close(slot)
            reqs: list[Request] = []
            active = []
            for slot in ready:
                try:
                    reqs.extend(slot.prepare())
                except Exception as exc:
                    self._fail(slot, slot.plan, exc)
                    continue
                active.append(slot)
            if not active:
                continue
            self.rounds += 1
            self._forward(reqs)
            alive = []
            for slot in active:
                try:
                    slot.resolve()
                except Exception as exc:
                    self._fail(slot, slot.plan, exc)
                    continue
                alive.append(slot)
            active = alive
        if self.writer is not None:
            self.writer.flush()
        return self

    def summary(self, wall_s: float) -> dict:
        rss, peak = rss_mb()
        timings = {k: round(float(v), 6) for k, v in sorted(self.timings.items())}
        return {"worker": self.id, "battles": len(self.results), "errors": list(self.errors),
                "rows": self.rows, "decisions": self.decisions, "rounds": self.rounds,
                "forwards": self.forwards, "forward_rows": self.forward_rows,
                "learner_forwards": self.learner.calls, "learner_rows": self.learner.rows,
                "files": [] if self.writer is None else list(self.writer.files),
                "wall_s": round(float(wall_s), 6), "timings": timings,
                "rss_mb": None if rss is None else round(rss, 1),
                "peak_rss_mb": None if peak is None else round(peak, 1),
                "results": sorted(self.results, key=lambda r: r["index"])}


def run_worker(task: Mapping[str, Any]) -> dict:
    """A worker process's task (plain data): {"worker", "settings" (Settings.to_dict()),
    "battles" ([BattlePlan.to_dict()]), "out" (folder or None), "prefix"} -> its summary.
    Module level, so a spawn child imports it by name (Windows)."""
    t0 = time.perf_counter()
    task = dict(task)
    worker = Worker(task["settings"], int(task.get("worker", 0)), out=task.get("out"),
                    prefix=str(task.get("prefix") or "rollout"))
    worker.run(task.get("battles") or ())
    return worker.summary(time.perf_counter() - t0)


def _init_pool(engine: Any, threads: int) -> None:
    """The pool initializer: the parent's engine config, and the BLAS threads set before torch
    loads (the worker sets torch's own once it does)."""
    if engine is not None:
        rs_engine.set_active(engine)
    if int(threads) > 0:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(int(threads))


# ---------------------------------------------------------------------------
# a collection
# ---------------------------------------------------------------------------

def _merge(parts: Sequence[Mapping[str, Any]], wall_s: float, workers: int) -> dict:
    results = sorted((r for p in parts for r in p["results"]), key=lambda r: r["index"])
    decisions = sum(int(p["decisions"]) for p in parts)
    hours = wall_s / 3600 if wall_s > 0 else None
    return {"tool": TOOL, "format": FORMAT, "workers_run": workers, "matches": len(results),
            "errors": [e for p in parts for e in p["errors"]],
            "rows": sum(int(p["rows"]) for p in parts), "decisions": decisions,
            "files": [f for p in parts for f in p["files"]], "wall_s": round(wall_s, 3),
            "matches_per_hour": None if hours is None else round(len(results) / hours, 1),
            "decisions_per_s": None if not wall_s else round(decisions / wall_s, 1),
            "rss_mb": [p.get("rss_mb") for p in parts],
            "peak_rss_mb": [p.get("peak_rss_mb") for p in parts],
            "workers": [{k: v for k, v in p.items() if k != "results"} for p in parts],
            "league_results": [[r["opponent"], r["score"]] for r in results
                               if r["score"] is not None and r["opponent"] != LEARNER],
            "results": results}


def collect(plans: Iterable[Any], settings: Any, *, workers: int = 0, out: Any = None,
            prefix: str = "rollout") -> dict:
    """Play `plans` and write the learner's rows under `out` (module doc). workers 0 plays them
    here, in one worker; workers >= 1 in that many spawn processes (plans dealt round-robin).
    Returns the summary (also <out>/summary.json)."""
    settings = Settings.from_dict(settings)
    plans = [BattlePlan.from_dict(p) for p in plans]
    if not plans:
        raise ValueError("no battles to play")
    workers = _int(workers, "workers", 0)
    if settings.engine is None:                 # the workers run the parent's engine
        settings = settings.replace(engine=rs_engine.active().to_json())
    n = max(1, min(workers, len(plans)))
    out_text = None if out is None else str(out)
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
    tasks = [{"worker": w, "settings": settings.to_dict(),
              "battles": [p.to_dict() for p in plans[w::n]], "out": out_text, "prefix": prefix}
             for w in range(n)]
    t0 = time.perf_counter()
    if workers <= 0:
        parts = [run_worker(task) for task in tasks]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(n, mp_context=ctx, initializer=_init_pool,
                                 initargs=(settings.engine, settings.threads),
                                 max_tasks_per_child=1) as pool:
            parts = list(pool.map(run_worker, tasks))
    summary = _merge(parts, time.perf_counter() - t0, workers)
    summary["settings"] = {k: v for k, v in settings.to_dict().items() if k != "meta"}
    summary["meta"] = settings.meta
    if out is not None:
        path = Path(out) / SUMMARY_JSON
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    return summary


def bench_row(summary: Mapping[str, Any], battles: int) -> dict:
    """One line of the bench table from a collection's summary."""
    rss = [x for x in summary.get("rss_mb") or () if x is not None]
    peak = [x for x in summary.get("peak_rss_mb") or () if x is not None]
    parts = summary.get("workers") or ()
    return {"workers": summary.get("workers_run"), "battles": int(battles),
            "matches": summary.get("matches"), "errors": len(summary.get("errors") or ()),
            "wall_s": summary.get("wall_s"), "matches_per_hour": summary.get("matches_per_hour"),
            "decisions_per_s": summary.get("decisions_per_s"),
            "rss_mb_max": max(rss) if rss else None, "peak_rss_mb_max": max(peak) if peak else None,
            "rows_per_forward": (round(sum(p["learner_rows"] for p in parts)
                                       / max(1, sum(p["learner_forwards"] for p in parts)), 2)
                                 if parts else None),
            "timings": [p.get("timings") for p in parts]}


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _anchors(texts: Sequence[str]) -> dict:
    out: dict[str, str | None] = {}
    for text in texts:
        member, sep, path = str(text).partition("=")
        member, path = member.strip(), path.strip()
        if not sep or not member:
            raise ValueError(f"--anchor {text!r}: ID=CHECKPOINT (or ID=idle)")
        out[member] = None if path.lower() in ("", RM.IDLE, "none") else path
    return out


def league_from_args(args: argparse.Namespace) -> Any:
    """--league FILE (tools/rl/league.py's JSON), else a league of the --anchor members (default:
    the learner itself as the one anchor), with --mirror-share."""
    if args.league:
        return LG.League.load(args.league)
    anchors = _anchors(args.anchor or ())
    if not anchors:
        anchors = {"il:learner": str(args.learner)}
    return LG.League(LG.LeagueConfig(mirror_share=float(args.mirror_share)), anchors=anchors)


def _one_deck_field(names: Sequence[str]) -> Any:
    names = tuple(str(c) for c in names)
    stat = deck_stats.DeckStat(cards=names, count=1, share=1.0, score=0.5, names=names,
                               forms=(((0,) * len(names), 1),))
    return deck_stats.DeckField([stat], split="train")


def _deck_names(text: str) -> tuple[str, ...]:
    t = (text or "").strip()
    if t.lower() in ("hog26", "hog", "hog2.6"):
        return tuple(sorted(RR.HOG26))
    names = tuple(c.strip() for c in t.split(",") if c.strip())
    if len(names) != 8 or len(set(names)) != 8:
        raise ValueError(f"a deck: 'hog26' or 8 different comma-separated card names, got {text!r}")
    return names


def deck_field_from_args(args: argparse.Namespace, *, fallback: bool = False) -> Any:
    """--opponent-deck NAMES (one deck), else the train split of --deck-stats (default
    <data>/deck_stats.json); `fallback` (bench): Hog 2.6 when the stats file is missing."""
    if args.opponent_deck:
        return _one_deck_field(_deck_names(args.opponent_deck))
    path = Path(args.deck_stats) if args.deck_stats else deck_stats.data_root() / "deck_stats.json"
    if not path.is_file():
        if fallback:
            print(f"note: no deck stats at {path}; the opponents play Hog 2.6", flush=True)
            return _one_deck_field(sorted(RR.HOG26))
        raise FileNotFoundError(f"no deck stats at {path}: run tools/deck_stats.py, or pass "
                                "--deck-stats FILE or --opponent-deck NAMES")
    exclude = [deck_stats.HOG26_KEYS] if args.exclude_hog else ()
    return deck_stats.DeckField.from_file(path, "train", top=args.deck_top or None,
                                          alpha=args.deck_alpha, exclude=exclude)


def _learner_meta(path: Any, cfg: rs_engine.EngineConfig, allow_drift: bool | None) -> dict:
    """The learner checkpoint's meta, its vocabulary checked against the engine (VocabMismatch),
    and v2 (ValueError)."""
    import rs_checkpoint
    import torch
    meta = dict(torch.load(str(path), map_location="cpu")["meta"])
    rs_checkpoint.check_meta(meta, cfg, allow_drift, source=str(path))
    if not RD.is_v2_meta(meta):
        raise ValueError(f"{path}: the learner must be a v2 checkpoint (schema rs_rows/v2), got "
                         f"{meta.get('schema')!r}")
    return meta


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="RL rollouts in RoyaleSim (R01): K battles per "
                                             "worker, batched forwards, ppo rollout files.")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("collect", "bench"):
        p = sub.add_parser(name, help="write rollouts" if name == "collect" else
                           "measure matches/hour and worker memory (writes nothing)")
        p.add_argument("--learner", required=True, metavar="CKPT", help="the v2 learner checkpoint")
        if name == "collect":
            p.add_argument("--out", required=True, type=Path, help="folder for the .npz parts")
            p.add_argument("--prefix", default="rollout")
            p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                           help="spawn processes (0: in this process); capped by the RAM check")
        else:
            p.add_argument("--workers", default="1,4,8",
                           help="comma-separated worker counts to measure in turn")
        p.add_argument("--matches", type=int, default=DEFAULT_MATCHES if name == "collect" else 64)
        p.add_argument("--battles", type=int, default=DEFAULT_BATTLES,
                       help="K: battles open at once per worker (the forward batch is up to 2K)")
        p.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                       help="torch CPU threads per worker (1-2)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--start", type=int, default=0, help="the first battle index")
        p.add_argument("--league", default=None, help="tools/rl/league.py JSON (the driver's)")
        p.add_argument("--anchor", action="append", default=[], metavar="ID=CKPT",
                       help="a league anchor without --league (ID=idle: an idle seat); default "
                            "the learner itself")
        p.add_argument("--mirror-share", type=float, default=0.0,
                       help="without --league: share of battles the learner plays itself")
        p.add_argument("--deck-stats", default=None, help="tools/deck_stats.py JSON (default "
                       "<data>/deck_stats.json); the train split is dealt")
        p.add_argument("--deck-top", type=int, default=0, help="only the N most played decks")
        p.add_argument("--deck-alpha", type=float, default=1.0,
                       help="decks dealt in proportion to count ** alpha")
        p.add_argument("--exclude-hog", action="store_true", help="no Hog 2.6 opponent decks")
        p.add_argument("--opponent-deck", default=None, metavar="NAMES",
                       help="one opponent deck instead of the field: hog26 or 8 card names")
        p.add_argument("--learner-deck", default="hog26", metavar="NAMES")
        p.add_argument("--delay", default=None, metavar="TICKS",
                       help=f"command delay (default: the engine config's, else {DEFAULT_DELAY})")
        p.add_argument("--level", default=str(FRIENDLY_LEVEL),
                       help="card and crown-tower level of both sides (friendlies: 11)")
        p.add_argument("--noise", default=None, metavar="PROFILE",
                       help="the learner's perception noise: a profile JSON, 'default' or 'none' "
                            "(default: data/noise/measured.json, else the prior)")
        p.add_argument("--opponent-noise", default=SAME, metavar="PROFILE",
                       help="the opponents' noise ('same': the learner's)")
        p.add_argument("--noise-seed", type=int, default=0)
        p.add_argument("--opponent-decide", choices=RD.MODES, default=RD.MODE_SAMPLE)
        p.add_argument("--no-presses", action="store_true", help="never press ability buttons")
        p.add_argument("--wait-keep", type=float, default=None,
                       help="the learner's wait_keep (default: its args.json)")
        p.add_argument("--max-ticks", type=int, default=RM.MAX_TICKS)
        p.add_argument("--flush-rows", type=int, default=DEFAULT_FLUSH_ROWS)
        p.add_argument("--max-models", type=int, default=DEFAULT_MAX_MODELS)
        p.add_argument("--ram-budget-gb", type=float, default=RAM_BUDGET_GB)
        p.add_argument("--worker-mb", type=int, default=None,
                       help="memory per worker for the RAM check (default: worker_mb(), or the "
                            "bench's peak)")
        p.add_argument("--no-ram-check", action="store_true")
        p.add_argument("--allow-vocab-drift", action="store_true")
        rs_engine.add_engine_args(p)
    return ap


def _prepare(args: argparse.Namespace, *, bench: bool) -> tuple:
    cfg = rs_engine.config_from_args(args)
    rs_engine.set_active(cfg)
    delay = resolve_delay(args.delay, cfg)
    level = rs_engine.parse_level(args.level, "--level")
    if level is None:
        raise ValueError(f"--level: a level 1-{rs_engine.MAX_LEVEL} (friendlies: "
                         f"{FRIENDLY_LEVEL}), got {args.level!r}")
    noise_src = default_noise() if args.noise is None else args.noise
    noise = noise_entry(noise_src)
    opp = args.opponent_noise
    opp_noise = noise if str(opp).strip().lower() == SAME else noise_entry(opp)
    allow = True if args.allow_vocab_drift else None
    _learner_meta(args.learner, cfg, allow)
    league = league_from_args(args)
    for member_id in league.active_ids():
        path = league.member(member_id).path
        if path and str(path).strip().lower() not in ("", RM.IDLE) and not Path(path).is_file():
            raise FileNotFoundError(f"league member {member_id}: no checkpoint {path}")
    field = deck_field_from_args(args, fallback=bench)
    plans = plan_battles(args.matches, learner=args.learner, league=league, deck_field=field,
                         seed=args.seed, start=args.start, learner_deck=_deck_names(args.learner_deck),
                         delay=delay, level=level, noise=noise, opponent_noise=opp_noise,
                         noise_seed=args.noise_seed, presses=not args.no_presses,
                         opponent_decide=args.opponent_decide, engine=cfg,
                         max_ticks=args.max_ticks)

    def shown(entry):
        if entry is None:
            return None
        p = PN.as_profile(entry)
        return {"name": p.name, "digest": p.digest(), "kinds": p.active_groups()}

    meta = {"seed": args.seed, "start": args.start, "delay": delay, "level": level,
            "noise": shown(noise), "noise_source": str(noise_src), "opponent_noise": shown(opp_noise),
            "noise_seed": args.noise_seed, "league": args.league or _anchors(args.anchor or ()),
            "decks": {"split": "train", "count": len(getattr(field, "decks", ())),
                      "alpha": args.deck_alpha, "top": args.deck_top,
                      "exclude_hog": bool(args.exclude_hog)},
            "max_ticks": args.max_ticks, "opponent_decide": args.opponent_decide,
            "presses_switch": not args.no_presses}
    settings = Settings(learner=args.learner, battles=args.battles, threads=args.threads,
                        flush_rows=args.flush_rows, engine=cfg.to_json(), allow_drift=allow,
                        wait_keep=args.wait_keep, max_models=args.max_models,
                        write=not bench, meta=meta)
    print(f"engine: {cfg.describe()}\ndelay {delay} ticks, level {level}, noise "
          f"{(shown(noise) or {}).get('name', 'none')} ({noise_src}), opponents' noise "
          f"{(shown(opp_noise) or {}).get('name', 'none')}\nleague: {len(league.active_ids())} "
          f"active member(s); decks: {len(getattr(field, 'decks', ()))} (train split)", flush=True)
    return cfg, plans, settings


def _workers(args: argparse.Namespace, wanted: int) -> int:
    if wanted <= 0 or args.no_ram_check:
        return wanted
    per = args.worker_mb or worker_mb(args.battles, args.flush_rows)
    got, why = ram_check(wanted, per_worker_mb=per, budget_gb=args.ram_budget_gb)
    print(f"workers: {why}", flush=True)
    return got


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.matches < 1 or args.battles < 1 or args.threads < 0 or args.flush_rows < 1:
        ap.error("--matches, --battles and --flush-rows >= 1, --threads >= 0")
    import rs_checkpoint
    bench = args.command == "bench"
    try:
        _cfg, plans, settings = _prepare(args, bench=bench)
    except (rs_checkpoint.VocabMismatch, rs_engine.EngineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if bench:
        try:
            counts = [int(x) for x in str(args.workers).split(",") if x.strip()]
        except ValueError:
            ap.error(f"--workers: comma-separated counts, got {args.workers!r}")
        rows = []
        for wanted in counts:
            workers = _workers(args, wanted)
            summary = collect(plans, settings, workers=workers)
            row = bench_row(summary, args.battles)
            rows.append(row)
            print(f"workers {row['workers']:>2} x K {row['battles']}: {row['matches']} matches in "
                  f"{row['wall_s']:.1f} s = {row['matches_per_hour']} matches/hour, "
                  f"{row['decisions_per_s']} decisions/s, {row['rows_per_forward']} rows per "
                  f"learner forward, rss max {row['rss_mb_max']} MB (peak "
                  f"{row['peak_rss_mb_max']} MB), errors {row['errors']}", flush=True)
        print(json.dumps({"tool": TOOL, "bench": rows}))
        return 0
    workers = _workers(args, args.workers)
    summary = collect(plans, settings, workers=workers, out=args.out, prefix=args.prefix)
    print(f"{summary['matches']} matches, {summary['rows']} rows in {len(summary['files'])} "
          f"file(s), {summary['wall_s']:.0f} s ({summary['matches_per_hour']} matches/hour), "
          f"{len(summary['errors'])} error(s) -> {args.out}", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("results", "workers")}))
    return 0 if summary["matches"] else 1


if __name__ == "__main__":
    sys.exit(main())
