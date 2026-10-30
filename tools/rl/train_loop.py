r"""The RL driver (R04): collection and PPO updates in turn, the league, the promotion gate every
M iterations, checkpoints, resume and a bench.

    .venvs\royalegym\Scripts\python.exe tools\rl\train_loop.py run --name hog_rl1 --bench 10m
    .venvs\royalegym\Scripts\python.exe tools\rl\train_loop.py run --name hog_rl1 --hours 48
    .venvs\royalegym\Scripts\python.exe tools\rl\train_loop.py run --name hog_rl1 --resume
    .venvs\royalegym\Scripts\python.exe tools\rl\train_loop.py status --name hog_rl1

scripts\rl_train.ps1 wraps it (docs/SPARRING_RUNBOOK.md section 21). Pure simulation: nothing
here starts, touches or taps the emulator, and nothing imports the memory reader
(scripts/gamestate). It replaces tools/rs_selfplay.py's loop (48 matches against one frozen
opponent with one deck).

ONE ITERATION (learner it<n>.pt, n from 0; it0000.pt is a copy of --start):
  1 collect  tools/rl/rollout.py (R01): --matches battles planned here (plan_battles: the learner
             on Hog 2.6 in base forms with rs_decide's sampling preset, every card and tower at
             level 11, command delay D, the run's perception noise, the opponent from the league
             mix below, its deck from the train split's deck field), played by --workers spawn
             processes of the engine venv (hastycr.machine.py_engine, CPU torch, --threads each).
             The collection runs in a subprocess, `train_loop.py collect-task` (rollout.collect on
             a task file this process writes: the plans, the settings, the worker count), so the
             plans, and with them the mix, are made here. Rows go to runs/rl/<name>/it<n>/.
  2 update   tools/rl/ppo.py (R02) in the GPU venv (py_train; in this process when this IS that
             interpreter): --ckpt it<n>.pt --ref the imitation model (--ref, default --start)
             --data it<n>/ --out it<n+1>.pt, the Adam moments in adam/it<n+1>.pt (copied from
             adam/it<n>.pt first, so a repeated update starts from the same moments); the KL
             coefficients travel in the checkpoint (blob["ppo"]).
  3 commit   the league files the learner's scores (League.record_many of the collection's
             league_results), ends the iteration (decay; a snapshot of it<n+1>.pt is admitted
             every --admit-every iterations), and state.json moves to iteration n+1, in one atomic
             write: a crash anywhere repeats at most the step that was running.
Every --gate-every (M) iterations the gate runs on the newest checkpoint before the next
collection.

THE OPPONENTS (MixDraw, royalelearn's ladder mix (mirror, pool, scripted) = (0.50, 0.35, 0.15),
the plan's "50% recent selves / 35% pool / 15% imitation anchors"): --mix shares of the learner
itself on both seats (rollout's mirror: both seats are recorded), of the league's active
snapshots and of its anchors, each group drawn by PFSP "hard" weights within it (tools/rl/league.
pfsp_weights with the league's settings). Until a snapshot is admitted the pool's share goes to
the anchors. The anchors are the imitation models: --start itself and every --anchor NAME=CKPT
(league ids il:NAME), never evicted.

THE GATE (tools/rs_league_eval.py run, L03, in the engine venv): the candidate (it<c>.pt, named
it<c>-<6 hex of its SHA-256>) against the champion (at first --start, then the last candidate
that passed), the anchors (--start and every --anchor, the champion left out) as the regression
check, on the dev split's frozen pairings (runs/rl/<name>/gate.pairs.json), both seats, at delay
D with the run's noise, --gate-champion-games (1,000) and --gate-anchor-games (400), one JSONL
log for the whole run (gate.jsonl: the champion's games against the anchors are played once and
reused). PASS: the candidate is admitted to the league (if it was not), promoted, and copied to
champion.pt (+ champion.json). FAIL, INCOMPLETE or a gate that could not run: the champion stays.

THE RUN FOLDER, runs/rl/<name>/ (<runs> from src/hastycr/machine.py):
  args.json       the frozen settings ("flags": what decides the games; "frozen": the start's
                  schema and wait_keep, SHA-256 digests of the checkpoints and the noise, the
                  engine config and fingerprint id) and every launch. Its top-level "wait_keep"
                  is what rs_decide.wait_keep_for reads for every checkpoint in this folder, so
                  it<n>.pt and champion.pt play with the start's wait bias everywhere (rollouts,
                  the gate, the live runner).
  state.json      the iteration, the pending step, the league, the champion, the gates
  it<n>.pt        checkpoints; it0000.pt carries blob["rl"] (run, engine and its fingerprint,
                  schema, start), which ppo.py copies into every later one
  it<n>.json      the state at the start of iteration n (resume, --rollback)
  it<n>/          the iteration's collect.json (the task), summary.json and rollout parts
                  (deleted after the update unless --keep-rollouts)
  adam/           the optimizer's moments of the last two updates
  league.json     the league (a copy of state.json's, for reading)
  engine.json, noise.json   the frozen engine config and noise profile, given to the gate
  gate.jsonl, gate.pairs.json, gates/it<c>.report.json   the gate's log, pairings, reports
  champion.pt, champion.json   the current champion (the live runner's --ckpt)
  log.jsonl       one record per iteration and per gate; train_loop.log the console;
                  logs/it<n>-collect.log, it<n>-update.log, gate-it<c>.log each step's output
  bench-<time>/   a --bench run's files and bench.json

RESUME AND ROLLBACK. --resume continues a run from state.json; the frozen settings come from
args.json, and a launch that names one with another value (or whose engine, start, reference,
anchors or noise file changed) is refused: start a new --name. --resume --rollback N goes back
to iteration N: the later checkpoints, states, rollouts, Adam moments and gate reports move to
rollback-<time>/, and the league, champion and gates are those of it<N>.json. A new run on an
existing one is refused without --resume. One driver per run (a lock on runs/rl/<name>/.lock).

STOP. A file runs/rl/<name>/STOP, or C:\hcr\STOP (the live runner's), ends the run cleanly at
the next step boundary: the collection, update or gate that is running finishes first (Ctrl+C
stops at once; resume repeats the interrupted step). Like the live runner, the driver refuses
to start while either exists. --hours H stops before a collection once H hours have passed
since the launch; --iterations N once N iterations are done (a gate that is due still runs).

BENCH (--bench 10m): about that long of collections at a few worker counts (--bench-workers,
default 1, half and all of what the RAM and CPU allow), no training, then one PPO update on the
last one's rows; it prints matches/hour, the peak memory of a worker, a recommended --workers
and how long an iteration of --matches would take, and writes bench-<time>/bench.json.

MEMORY (16 GB PC). Collection and the update never run at the same time. Workers: rollout's
ram_check (--ram-budget-gb, default 8, and the free memory less 2 GB; --worker-mb from the
bench) before every collection; the update: ppo.py's own --ram-budget-gb (--ppo-ram-gb,
default 6); the gate: rs_league_eval's worker cap (about 1.2 GB a worker, at most 6). This
process holds no model while the steps run (torch is imported only to check and copy
checkpoints, unless the update runs in this process).

WINDOWS. Every step is a subprocess with a list argv (no shell), UTF-8 output, streamed to the
console and its log. The collection's workers are rollout.py's spawn pool; the task is plain
JSON, and this module imports nothing heavy at the top, so the spawned children (which import
the main module) start fast.

Exit codes: 0 done (or stopped, or the bench); 1 a step failed, or the engine or a checkpoint's
vocabulary does not match; 2 bad arguments, files or settings (refused before anything ran); 3
finished, but a gate could not run (see its log); 130 interrupted (Ctrl+C). The design follows
royalelearn's coordinator and ladder (MIT; not imported).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import deck_stats  # noqa: E402  (data_root; the deck field itself is read through rollout)
import perception_noise as PN  # noqa: E402
import rs_decide as RD  # noqa: E402  (the name rollout and ppo import it by)
import rs_engine  # noqa: E402
from tools.rl import league as LG  # noqa: E402
from tools.rl import ppo as P  # noqa: E402  (pure numpy until a model is involved)
from tools.rl import rollout as R  # noqa: E402

__all__ = [
    "EXIT_ARGS", "EXIT_FAILED", "EXIT_GATE", "EXIT_INTERRUPTED", "EXIT_OK", "FROZEN",
    "InlineSteps", "MixDraw", "Refusal", "RunPaths", "StepError", "Steps", "SubprocessSteps",
    "Trainer", "build_parser", "collect_from_task", "last_json", "main", "parse_duration",
    "parse_mix", "recommend", "run_logged",
]

TOOL = "tools/rl/train_loop.py"
FORMAT = 1
#: The checkpoint key it0000.pt is stamped under (ppo.py carries it into every later one).
STAMP_KEY = "rl"
DEFAULT_NAME = "hog_rl1"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
DEFAULT_MATCHES = 400
#: mirror (the learner on both seats) / pool (snapshots) / anchors (imitation models).
MIX_GROUPS = ("mirror", "pool", "anchors")
DEFAULT_MIX = (0.50, 0.35, 0.15)
DEFAULT_ADMIT_EVERY = 5
DEFAULT_MAX_SNAPSHOTS = 16
DEFAULT_DECAY = 0.9
DEFAULT_GATE_EVERY = 10
GATE_WORKERS = 6
GATE_SPLIT = "dev"
CHAMPION_GAMES, ANCHOR_GAMES = 1000, 400        # rs_league_eval's CHAMPION_GAMES, ANCHOR_GAMES
PASS, CYCLE = "pass", "cycle"                   # rs_league_eval's decisions that admit
ANCHOR_PREFIX = "il:"
STOP_NAME = "STOP"
REPO_STOP = ROOT / STOP_NAME
#: The plan's floor for the opponents' deck field (Phase 7): fewer is said at launch.
MIN_DECKS = 1000
DEFAULT_BENCH = "10m"
BENCH_SEED = 7_777_777
#: The bench's margin on a worker's measured peak memory, and the matches a slice may grow to
#: (in waves of K battles a worker).
BENCH_MARGIN = 1.15
BENCH_MAX_WAVES = 8
#: Lines of a step's output kept to find its JSON summary.
TAIL_LINES = 400
EXIT_OK, EXIT_FAILED, EXIT_ARGS, EXIT_GATE, EXIT_INTERRUPTED = 0, 1, 2, 3, 130

#: Settings fixed when a run is created (args.json "flags"): a resumed run takes them from there,
#: and a launch that gives one another value is refused (module doc, RESUME AND ROLLBACK).
FROZEN = ("start", "ref", "anchor", "delay", "level", "noise", "opponent_noise", "learner_deck",
          "deck_stats", "deck_top", "deck_alpha", "exclude_hog", "opponent_deck", "mix",
          "admit_every", "max_snapshots", "decay", "presses", "opponent_decide", "max_ticks",
          "seed", "gate_champion_games", "gate_anchor_games", "gate_decide", "gate_seed")
#: What is derived from the frozen settings and checked again at every launch.
COMPARED = ("start_name", "schema", "wait_keep", "start_sha256", "ref_sha256", "anchor_sha256",
            "noise_digest", "opponent_noise_digest", "engine", "engine_id")
ENGINE_KEYS = ("engine", "engine_id")


class Refusal(Exception):
    """A launch refused before anything ran (exit `code`: EXIT_ARGS, or EXIT_FAILED for an
    engine or vocabulary mismatch)."""

    def __init__(self, message: str, code: int = EXIT_ARGS):
        super().__init__(message)
        self.code = int(code)


class StepError(RuntimeError):
    """A collection, update or gate step that failed; the message names its log."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (Path, os.PathLike)):
        return str(x)
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    return str(x)


def write_json(path: Any, data: Any) -> Path:
    """`data` as JSON at `path`, atomically (a .tmp file, then os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=_jsonable), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Any) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: Any, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=_jsonable) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def file_digest(path: Any) -> str:
    """16 hex digits of SHA-256 over a file (rs_league_eval.ckpt_digest's)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def copy_atomic(src: Any, dst: Any) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst


def _abspath(p: Any) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


def _same_file(a: Any, b: Any) -> bool:
    try:
        return os.path.samefile(str(a), str(b))
    except OSError:
        return os.path.normcase(_abspath(a)) == os.path.normcase(_abspath(b))


def _safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._-")
    return name or "start"


def parse_duration(text: Any) -> float:
    """Seconds from "10m", "600s", "1.5h" or a bare number of minutes."""
    t = str(text).strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d*)?|\.\d+)\s*([smh]?)", t)
    if not m:
        raise ValueError(f"a duration: 10m, 600s, 1.5h or minutes, got {text!r}")
    value = float(m.group(1)) * {"s": 1.0, "m": 60.0, "h": 3600.0, "": 60.0}[m.group(2)]
    if value <= 0:
        raise ValueError(f"a duration above 0, got {text!r}")
    return value


def parse_mix(value: Any) -> list[float]:
    """--mix "MIRROR,POOL,ANCHORS" (or three numbers): shares >= 0 with a positive sum."""
    items = value.split(",") if isinstance(value, str) else list(value)
    try:
        mix = [round(float(x), 6) for x in items]
    except (TypeError, ValueError):
        raise ValueError(f"--mix: three numbers MIRROR,POOL,ANCHORS, got {value!r}") from None
    if len(mix) != 3 or any(not math.isfinite(x) or x < 0 for x in mix) or sum(mix) <= 0:
        raise ValueError(f"--mix: three shares >= 0 with a positive sum, got {value!r}")
    return mix


def last_json(lines: Iterable[str]) -> dict | None:
    """The last line that is a JSON object (the tools print their summary that way)."""
    for line in reversed(list(lines)):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                data = json.loads(s)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
    return None


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# ---------------------------------------------------------------------------
# the run folder
# ---------------------------------------------------------------------------

class RunPaths:
    """The files of runs/rl/<name>/ (module doc, THE RUN FOLDER)."""

    def __init__(self, root: Any, name: str):
        self.root = Path(root)
        self.name = str(name)
        self.run = self.root / self.name
        self.args_json = self.run / "args.json"
        self.state_json = self.run / "state.json"
        self.league_json = self.run / "league.json"
        self.log_jsonl = self.run / "log.jsonl"
        self.driver_log = self.run / "train_loop.log"
        self.logs = self.run / "logs"
        self.gates = self.run / "gates"
        self.adam_dir = self.run / "adam"
        self.champion = self.run / "champion.pt"
        self.champion_json = self.run / "champion.json"
        self.noise_json = self.run / "noise.json"
        self.engine_json = self.run / "engine.json"
        self.gate_log = self.run / "gate.jsonl"
        self.stop = self.run / STOP_NAME
        self.lock = self.run / ".lock"

    def ckpt(self, n: int) -> Path:
        return self.run / f"it{int(n):04d}.pt"

    def sidecar(self, n: int) -> Path:
        return self.run / f"it{int(n):04d}.json"

    def data(self, n: int) -> Path:
        return self.run / f"it{int(n):04d}"

    def adam(self, n: int) -> Path:
        return self.adam_dir / f"it{int(n):04d}.pt"

    def gate_report(self, n: int) -> Path:
        return self.gates / f"it{int(n):04d}.report.json"

    def step_log(self, n: int, what: str) -> Path:
        return self.logs / f"it{int(n):04d}-{what}.log"


def iteration_of(path: Path) -> int | None:
    """n of it<n>, it<n>.pt or it<n>.json; None for anything else."""
    m = re.fullmatch(r"it(\d{4,})(?:\.pt|\.json)?", path.name)
    return int(m.group(1)) if m else None


class RunLock:
    """One driver per run: an exclusive lock on runs/rl/<name>/.lock, released when this
    process ends (msvcrt on Windows, fcntl elsewhere)."""

    def __init__(self, path: Any):
        self.path = Path(path)
        self.fd: int | None = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise Refusal(f"another train_loop.py is running this run ({self.path} is locked); "
                          "stop it first (or wait for it)") from None
        self.fd = fd
        return self

    def __exit__(self, *exc: Any) -> None:
        fd, self.fd = self.fd, None
        if fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


class Echo:
    """The driver's console: print, and append to train_loop.log once `path` is set. A step's
    own output (child lines) goes to the console and to that step's log only."""

    def __init__(self, path: Any = None):
        self.path = None if path is None else Path(path)

    @staticmethod
    def _print(line: str) -> None:
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)

    def __call__(self, text: str = "") -> None:
        for line in str(text).splitlines() or [""]:
            self._print(line)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

    def child(self, line: str) -> None:
        self._print("    " + line)


# ---------------------------------------------------------------------------
# the opponents' mix
# ---------------------------------------------------------------------------

class MixDraw:
    """What rollout.plan_battles asks of a league (draw(rng, size) -> ids, member(id).path) over a
    tools/rl/league.League, with the mix of module doc THE OPPONENTS: `mix` shares of LEARNER
    (a mirror), of the active snapshots and of the anchors, PFSP within each group. The draw is
    one generator call over the combined weights, so it is a pure function of the league state
    and the generator."""

    def __init__(self, league: Any, mix: Sequence[float] = DEFAULT_MIX):
        self.league = league
        self.mix = tuple(parse_mix(mix))

    def groups(self) -> dict[str, tuple[str, ...]]:
        members = [m for m in self.league.members if m.active]
        return {"mirror": (LG.LEARNER,),
                "pool": tuple(m.id for m in members if not m.is_anchor),
                "anchors": tuple(m.id for m in members if m.is_anchor)}

    def shares(self) -> dict[str, float]:
        """The groups' shares, normalised: an empty pool's share goes to the anchors (and an
        empty anchor group's to the pool)."""
        g = self.groups()
        share = dict(zip(MIX_GROUPS, self.mix))
        if not g["pool"]:
            share["anchors"] += share["pool"]
            share["pool"] = 0.0
        if not g["anchors"]:
            share["pool"] += share["anchors"]
            share["anchors"] = 0.0
        if not g["pool"] and not g["anchors"]:
            share["pool"] = share["anchors"] = 0.0
        total = sum(share.values())
        if total <= 0:
            raise RuntimeError("nothing to draw: the mirror share is 0 and the league has no "
                               "active member")
        return {k: v / total for k, v in share.items()}

    def weights(self) -> dict[str, float]:
        """Draw probability of every id (LEARNER for the mirror), summing to 1."""
        share, g, cfg = self.shares(), self.groups(), self.league.config
        out: dict[str, float] = {}
        if share["mirror"] > 0:
            out[LG.LEARNER] = share["mirror"]
        for name in ("pool", "anchors"):
            ids = g[name]
            if not ids or share[name] <= 0:
                continue
            w = LG.pfsp_weights([self.league.expected(i) for i in ids], weighting=cfg.weighting,
                                power=cfg.power, uniform_floor=cfg.uniform_floor,
                                floor_scale=cfg.weight_floor_scale)
            for member_id, p in zip(ids, w):
                out[member_id] = out.get(member_id, 0.0) + share[name] * float(p)
        return out

    def draw(self, rng: Any = None, size: int | None = None) -> str | list[str]:
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        count = 1 if size is None else int(size)
        w = self.weights()
        ids = list(w)
        p = np.array([w[i] for i in ids], dtype=np.float64)
        picks = gen.choice(len(ids), size=count, p=p / p.sum()) if count else []
        out = [ids[int(k)] for k in picks]
        return out[0] if size is None else out

    def member(self, member_id: str) -> Any:
        return self.league.member(member_id)

    def kinds(self) -> dict[str, str]:
        """Every id's group (evicted members too): mirror, pool or anchors."""
        out = {LG.LEARNER: "mirror"}
        for m in self.league.members:
            out[m.id] = "anchors" if m.is_anchor else "pool"
        return out


# ---------------------------------------------------------------------------
# running the steps
# ---------------------------------------------------------------------------

def run_logged(cmd: Sequence[Any], log_path: Any, echo: Any, *,
               poll: Callable[[], Any] | None = None,
               env: Mapping[str, str] | None = None) -> tuple[int, dict | None]:
    """Run `cmd` (a list argv, no shell) from the repo root with its stdout and stderr streamed
    to the console (echo.child) and appended to `log_path`; `poll` runs after every line.
    Returns (exit code, the last JSON object line of its output)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = dict(os.environ if env is None else env)
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    run_env["PYTHONIOENCODING"] = "utf-8"
    argv = [str(c) for c in cmd]
    tail: deque = deque(maxlen=TAIL_LINES)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"== {_now()} {subprocess.list2cmdline(argv)}\n")
        fh.flush()
        proc = subprocess.Popen(argv, cwd=str(ROOT), env=run_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                fh.write(line + "\n")
                fh.flush()
                tail.append(line)
                echo.child(line)
                if poll is not None:
                    poll()
            code = proc.wait()
            fh.write(f"== exit {code}\n")
        finally:
            if proc.poll() is None:            # interrupted: the child goes too
                proc.kill()
                proc.wait()
    return int(code), last_json(tail)


def _inline_main(main_fn: Callable[[list[str]], int], argv: Sequence[str], log_path: Any,
                 echo: Any) -> tuple[int, dict | None]:
    """A tool's main(argv) in this process, its stdout captured (and logged), argparse's exits
    turned into codes."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            code = main_fn(list(argv))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else EXIT_ARGS
    lines = buf.getvalue().splitlines()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"== {_now()} (in process) {' '.join(str(a) for a in argv)}\n")
        fh.writelines(line + "\n" for line in lines)
        fh.write(f"== exit {code}\n")
    for line in lines:
        echo.child(line)
    return int(code or 0), last_json(lines)


def collect_from_task(task: Mapping[str, Any]) -> dict:
    """One collection from a task (train_loop's collect.json: plans, settings, workers, out,
    prefix): rollout.collect, which also writes <out>/summary.json. Returns its summary."""
    settings = R.Settings.from_dict(task["settings"])
    if settings.engine is not None:
        rs_engine.set_active(settings.engine)
    return R.collect(task["plans"], settings, workers=int(task.get("workers") or 0),
                     out=task.get("out"), prefix=str(task.get("prefix") or "rollout"))


def _printed(summary: Mapping[str, Any]) -> dict:
    """What collect-task prints: the summary without the per-battle results (summary.json has
    them); the per-worker parts stay (rollout.bench_row reads them)."""
    return {k: v for k, v in summary.items() if k != "results"}


class Steps:
    """How the steps run. This base runs nothing; SubprocessSteps (the default) runs each in its
    venv, InlineSteps in this process (tests, --inline)."""

    inline = False

    def collect(self, task_path: Path, log_path: Path, echo: Any,
                poll: Callable[[], Any] | None = None) -> dict:
        raise NotImplementedError

    def update(self, argv: Sequence[str], log_path: Path, echo: Any,
               poll: Callable[[], Any] | None = None) -> dict:
        raise NotImplementedError

    def gate(self, argv: Sequence[str], log_path: Path, echo: Any,
             poll: Callable[[], Any] | None = None) -> int:
        raise NotImplementedError

    def describe(self) -> str:
        return type(self).__name__


class InlineSteps(Steps):
    """Every step in this process: rollout.collect (its own workers, 0 = this process),
    ppo.main and rs_league_eval.main with their stdout captured."""

    inline = True

    def collect(self, task_path, log_path, echo, poll=None):
        summary = collect_from_task(read_json(task_path))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"== {_now()} (in process) collect {task_path}: {summary['matches']} matches,"
                     f" {summary['rows']} rows\n")
        return _printed(summary)

    def update(self, argv, log_path, echo, poll=None):
        code, stats = _inline_main(P.main, argv, log_path, echo)
        if code != 0 or stats is None:
            raise StepError(f"the PPO update failed (exit {code}); see {log_path}")
        return stats

    def gate(self, argv, log_path, echo, poll=None):
        import rs_league_eval
        return _inline_main(rs_league_eval.main, argv, log_path, echo)[0]

    def describe(self) -> str:
        return "in this process"


class SubprocessSteps(Steps):
    """Each step a subprocess: the collection and the gate in the engine venv (py_engine), the
    update in the GPU venv (py_train), or in this process when `ppo_inline` (this interpreter is
    py_train)."""

    def __init__(self, py_engine: Any, py_train: Any, *, ppo_inline: bool = False):
        self.py_engine = Path(py_engine)
        self.py_train = Path(py_train)
        self.ppo_inline = bool(ppo_inline)

    def collect(self, task_path, log_path, echo, poll=None):
        cmd = [self.py_engine, Path(__file__).resolve(), "collect-task", "--task", task_path]
        code, last = run_logged(cmd, log_path, echo, poll=poll)
        if code != 0 or last is None:
            raise StepError(f"the collection failed (exit {code}); see {log_path}")
        return last

    def update(self, argv, log_path, echo, poll=None):
        if self.ppo_inline:
            return InlineSteps().update(argv, log_path, echo, poll)
        code, last = run_logged([self.py_train, ROOT / "tools" / "rl" / "ppo.py", *argv],
                                log_path, echo, poll=poll)
        if code != 0 or last is None:
            raise StepError(f"the PPO update failed (exit {code}); see {log_path}")
        return last

    def gate(self, argv, log_path, echo, poll=None):
        return run_logged([self.py_engine, ROOT / "tools" / "rs_league_eval.py", *argv],
                          log_path, echo, poll=poll)[0]

    def describe(self) -> str:
        ppo = "in this process" if self.ppo_inline else str(self.py_train)
        return f"collection and gate {self.py_engine}; update {ppo}"


# ---------------------------------------------------------------------------
# settings: the frozen flags, what they derive, checkpoints, the engine
# ---------------------------------------------------------------------------

def runs_dir(value: Any = None) -> Path:
    """--runs-dir, else <runs>/rl (src/hastycr/machine.py), else runs/rl."""
    if value is not None:
        return Path(value)
    try:
        from hastycr import machine
        return Path(machine.runs_root()) / "rl"
    except Exception:                       # a broken machine.json: the repo's runs/
        return ROOT / "runs" / "rl"


def default_start() -> Path:
    """The v2 Hog 2.6 imitation model (scripts/third_model.ps1): <runs>/tokens/v2_hog/best.pt."""
    return runs_dir().parent / "tokens" / "v2_hog" / "best.pt"


def frozen_defaults() -> dict:
    return {"start": str(default_start()), "ref": None, "anchor": [], "delay": None,
            "level": R.FRIENDLY_LEVEL, "noise": str(R.default_noise()),
            "opponent_noise": R.SAME, "learner_deck": "hog26", "deck_stats": None,
            "deck_top": 0, "deck_alpha": 1.0, "exclude_hog": False, "opponent_deck": None,
            "mix": list(DEFAULT_MIX), "admit_every": DEFAULT_ADMIT_EVERY,
            "max_snapshots": DEFAULT_MAX_SNAPSHOTS, "decay": DEFAULT_DECAY, "presses": True,
            "opponent_decide": RD.MODE_SAMPLE, "max_ticks": R.RM.MAX_TICKS, "seed": 0,
            "gate_champion_games": CHAMPION_GAMES, "gate_anchor_games": ANCHOR_GAMES,
            "gate_decide": RD.MODE_LIVE, "gate_seed": 0}


def _noise_source(value: Any, *, same_ok: bool = False) -> str:
    text = str(value).strip()
    low = text.lower()
    if same_ok and low == R.SAME:
        return R.SAME
    if low in PN.NO_NOISE:
        return "none"
    if low in PN.PRIOR_NAMES:
        return str(PN.PRIOR_PATH)
    return _abspath(text)


def _deck(value: Any) -> list[str]:
    if isinstance(value, str):
        names = R._deck_names(value)
    else:
        names = tuple(str(c) for c in value)
        if len(names) != 8 or len(set(names)) != 8:
            raise ValueError(f"a deck is 8 different cards, got {list(names)}")
    return sorted(names)


def _anchor_list(value: Any) -> list[list[str]]:
    out = []
    for item in value or ():
        if isinstance(item, str):
            name, sep, path = item.partition("=")
            if not sep:
                raise ValueError(f"--anchor {item!r}: NAME=CKPT")
        else:
            name, path = item
        name, path = str(name).strip(), str(path).strip()
        if not name or not path:
            raise ValueError(f"--anchor {item!r}: NAME=CKPT")
        if _safe_name(name) != name:
            raise ValueError(f"--anchor name {name!r}: letters, digits, '_', '.' and '-' only")
        out.append([name, _abspath(path)])
    return out


def _delay(value: Any, cfg: Any) -> int:
    d = R.resolve_delay(value, cfg)
    return int(max(d)) if isinstance(d, tuple) else int(d)


def _onoff(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    low = str(value).strip().lower()
    if low not in ("on", "off", "true", "false", "1", "0"):
        raise ValueError(f"--presses: on or off, got {value!r}")
    return low in ("on", "true", "1")


def normalize_flags(raw: Mapping[str, Any], cfg: Any) -> dict:
    """The frozen flags in their stored form (absolute paths, ints, sorted decks); applying it
    twice changes nothing."""
    f = dict(raw)
    f["start"] = _abspath(f["start"])
    f["ref"] = _abspath(f["ref"]) if f.get("ref") else f["start"]
    f["anchor"] = _anchor_list(f.get("anchor"))
    f["delay"] = _delay(f.get("delay"), cfg)
    f["level"] = int(f["level"])
    f["noise"] = _noise_source(f["noise"])
    f["opponent_noise"] = _noise_source(f["opponent_noise"], same_ok=True)
    f["learner_deck"] = _deck(f["learner_deck"])
    f["opponent_deck"] = ",".join(_deck(f["opponent_deck"])) if f.get("opponent_deck") else None
    if f["opponent_deck"]:
        f["deck_stats"] = None
    else:
        f["deck_stats"] = _abspath(f.get("deck_stats") or
                                   Path(deck_stats.data_root()) / "deck_stats.json")
    f["deck_top"] = int(f["deck_top"] or 0)
    f["deck_alpha"] = float(f["deck_alpha"])
    f["exclude_hog"] = bool(f["exclude_hog"])
    f["mix"] = parse_mix(f["mix"])
    for key in ("admit_every", "max_snapshots", "max_ticks", "seed", "gate_champion_games",
                "gate_anchor_games", "gate_seed"):
        f[key] = int(f[key])
    f["decay"] = float(f["decay"])
    f["presses"] = _onoff(f["presses"])
    for key in ("opponent_decide", "gate_decide"):
        f[key] = str(f[key])
        if f[key] not in RD.MODES:
            raise ValueError(f"--{key.replace('_', '-')}: one of {', '.join(RD.MODES)}")
    if not 1 <= f["level"] <= rs_engine.MAX_LEVEL:
        raise ValueError(f"--level: 1-{rs_engine.MAX_LEVEL} (friendlies: {R.FRIENDLY_LEVEL})")
    for key in ("max_ticks", "gate_champion_games", "gate_anchor_games"):
        if f[key] < 1:
            raise ValueError(f"--{key.replace('_', '-')}: at least 1")
    if f["seed"] < 0 or f["gate_seed"] < 0 or f["deck_top"] < 0 or f["admit_every"] < 0:
        raise ValueError("--seed, --gate-seed, --deck-top and --admit-every: at least 0")
    for key in ("start", "ref"):
        if "," in f[key]:             # rs_league_eval's NAME=CKPT[,KEY=VALUE] player form
            raise ValueError(f"--{key} {f[key]}: the gate cannot name a path with ','")
    for name, path in f["anchor"]:
        if "," in path:
            raise ValueError(f"--anchor {name}: the gate cannot name a path with ','")
    return f


def resolve_flags(args: argparse.Namespace, stored: Mapping[str, Any] | None,
                  cfg: Any) -> tuple[dict, list[str]]:
    """(the frozen flags of this launch, the names of those it gives with another value than
    `stored`, a resumed run's args.json "flags"). A flag the launch leaves out (None) is taken
    from `stored`, else its default."""
    base = dict(stored) if stored is not None else frozen_defaults()
    defaults = frozen_defaults()
    given = {k for k in FROZEN if getattr(args, k, None) is not None}
    raw = {k: (getattr(args, k) if k in given else base.get(k, defaults[k])) for k in FROZEN}
    flags = normalize_flags(raw, cfg)
    changed = []
    if stored is not None:
        changed = [k for k in FROZEN if k in given and flags[k] != stored.get(k)]
    return flags, changed


def load_meta(path: Any) -> dict:
    """A checkpoint's meta (torch, CPU)."""
    import torch
    try:
        blob = torch.load(str(path), map_location="cpu")
    except Exception as exc:                # a missing, truncated or foreign file
        raise Refusal(f"{path}: cannot read the checkpoint ({type(exc).__name__}: {exc})") from None
    if not isinstance(blob, Mapping) or "meta" not in blob or "model" not in blob:
        raise Refusal(f"{path}: not a token-policy checkpoint (no model / meta)")
    return dict(blob["meta"])


def check_vocab(meta: Mapping[str, Any], path: Any, cfg: Any, allow: bool | None) -> None:
    import rs_checkpoint
    units = meta.get("units_mode", R.RR.UNITS_TYPES)
    if units != R.RR.UNITS_TYPES:
        raise Refusal(f"{path}: a --units {units} checkpoint; the rollouts and the gate build "
                      f"--units {R.RR.UNITS_TYPES} rows only")
    try:
        rs_checkpoint.check_meta(meta, cfg, allow, source=str(path))
    except rs_checkpoint.VocabMismatch as exc:
        raise Refusal(str(exc), EXIT_FAILED) from None


def engine_record(cfg: Any) -> dict:
    """{"engine": cfg.to_json(), "engine_fingerprint", "engine_id"}: rs_engine.fingerprint of an
    undelayed Battle under `cfg` (as tools/rs_selfplay.engine_record)."""
    try:
        rs = rs_engine.royalesim_module()
    except Exception as exc:
        raise Refusal(f"royalesim does not import ({type(exc).__name__}: {exc}); run "
                      r"train_loop.py with the engine venv: .venvs\royalegym\Scripts\python.exe",
                      EXIT_FAILED) from None
    try:
        fp = rs_engine.fingerprint(rs_engine.make_battle(cfg.undelayed(), rs), rs, cfg)
    except Exception as exc:                # EngineError, or the engine's own refusal
        raise Refusal(f"engine: {type(exc).__name__}: {exc}", EXIT_FAILED) from None
    return {"engine": cfg.to_json(), "engine_fingerprint": fp, "engine_id": fp["id"]}


def derive(flags: Mapping[str, Any], cfg: Any, allow: bool | None,
           record: Mapping[str, Any]) -> dict:
    """What the frozen flags give, checked: the start and reference metas (v2, the engine's
    vocabulary, one schema), the anchors, the noise profiles, the digests and the engine."""
    start, ref = Path(flags["start"]), Path(flags["ref"])
    for what, p in (("--start", start), ("--ref", ref)):
        if not p.is_file():
            raise Refusal(f"{what} {p}: no such checkpoint (the v2 Hog 2.6 model comes from "
                          r"scripts\third_model.ps1, runbook section 20)")
    meta = load_meta(start)
    if not RD.is_v2_meta(meta):
        raise Refusal(f"{start}: schema {meta.get('schema')!r}; the learner must be a v2 "
                      "checkpoint (rs_rows/v2): tools/rl/ppo.py trains v2 rows")
    check_vocab(meta, start, cfg, allow)
    ref_meta = meta if _same_file(start, ref) else load_meta(ref)
    if not RD.is_v2_meta(ref_meta):
        raise Refusal(f"{ref}: the reference must be a v2 checkpoint, got schema "
                      f"{ref_meta.get('schema')!r}")
    problems = P.reference_problems(meta, ref_meta)
    if problems:
        raise Refusal(f"--ref {ref}: " + "; ".join(problems))
    start_name = _safe_name(start.parent.name or start.stem)
    names, anchor_sha = {start_name}, {}
    for name, path in flags["anchor"]:
        if name in names or name == LG.LEARNER:
            raise Refusal(f"--anchor {name}: the name is taken (the start is {start_name!r}); "
                          "name it otherwise")
        names.add(name)
        if not Path(path).is_file():
            raise Refusal(f"--anchor {name}={path}: no such checkpoint")
        check_vocab(load_meta(path), path, cfg, allow)
        anchor_sha[name] = file_digest(path)
    try:
        noise = R.noise_entry(None if flags["noise"] == "none" else flags["noise"])
        if flags["opponent_noise"] == R.SAME:
            opp = noise
        else:
            opp = R.noise_entry(None if flags["opponent_noise"] == "none"
                                else flags["opponent_noise"])
    except (OSError, ValueError) as exc:
        raise Refusal(f"noise: {exc}") from None
    return {"start_name": start_name, "schema": str(meta.get("schema")),
            "wait_keep": float(RD.wait_keep_for(str(start))),
            "start_sha256": file_digest(start), "ref_sha256": file_digest(ref),
            "anchor_sha256": anchor_sha,
            "noise_digest": None if noise is None else PN.as_profile(noise).digest(),
            "opponent_noise_digest": None if opp is None else PN.as_profile(opp).digest(),
            "engine": record["engine"], "engine_id": record["engine_id"],
            "presses_trained": bool(RD.presses_allowed(meta)), "noise": noise,
            "opponent_noise": opp}


def stamp_start(src: Any, dst: Any, stamp: Mapping[str, Any]) -> Path:
    """it0000.pt: the start checkpoint with blob[STAMP_KEY] = `stamp` and no PPO state (a new
    run starts its KL coefficients from ppo's --kl-start)."""
    import torch
    blob = torch.load(str(src), map_location="cpu")
    if not isinstance(blob, Mapping) or "model" not in blob or "meta" not in blob:
        raise Refusal(f"{src}: not a token-policy checkpoint (no model / meta)")
    blob = dict(blob)
    blob.pop(P.PPO_KEY, None)
    blob[STAMP_KEY] = json.loads(json.dumps(dict(stamp), default=_jsonable))
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, dst)
    return dst


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def collect_digest(full: Mapping[str, Any], kinds: Mapping[str, str]) -> dict:
    """What an iteration keeps of a collection's summary: counts, speed, memory, the league's
    results and the learner's score by opponent group, plays and refusals per learner seat."""
    groups: dict[str, list[float]] = defaultdict(list)
    plays, refused = [], []
    for r in full.get("results") or ():
        for t in r.get("learner_teams") or ():
            plays.append(int(r["plays"][t]))
            refused.append(int(r["refused"][t]))
        if r.get("score") is not None:
            groups[kinds.get(str(r.get("opponent")), "other")].append(float(r["score"]))
    errors = list(full.get("errors") or ())
    peaks = [float(x) for x in full.get("peak_rss_mb") or () if x is not None]
    return {"matches": int(full.get("matches") or 0), "rows": int(full.get("rows") or 0),
            "decisions": int(full.get("decisions") or 0), "files": len(full.get("files") or ()),
            "errors": len(errors), "error_examples": errors[:3],
            "wall_s": full.get("wall_s"), "matches_per_hour": full.get("matches_per_hour"),
            "decisions_per_s": full.get("decisions_per_s"),
            "workers_run": full.get("workers_run"),
            "peak_rss_mb_max": max(peaks) if peaks else None,
            "league_results": [[str(o), s] for o, s in full.get("league_results") or ()],
            "score": {g: [len(v), round(sum(v) / len(v), 4)] for g, v in sorted(groups.items())},
            "plays_per_seat": round(sum(plays) / len(plays), 2) if plays else None,
            "refused_per_seat": round(sum(refused) / len(refused), 2) if refused else None}


def describe_iteration(rec: Mapping[str, Any]) -> str:
    c, u, lg = rec.get("collect") or {}, rec.get("update") or {}, rec.get("league") or {}
    score = ", ".join(f"{g} {s:.2f} ({n})" for g, (n, s) in (c.get("score") or {}).items())
    peak = c.get("peak_rss_mb_max")
    lines = [
        f"it {rec['iteration']:04d} -> {rec['out']}: {c.get('matches')} matches in "
        f"{(rec.get('collect_s') or 0) / 60:.1f} min ({_fmt(c.get('matches_per_hour'), 0)}/h"
        f"{'' if peak is None else f', peak {peak:.0f} MB a worker'}), {c.get('rows')} rows, "
        f"{c.get('errors')} error(s)",
        f"  learner score: {score or '-'}; plays per seat {_fmt(c.get('plays_per_seat'), 1)} "
        f"(pros 35-40), refused {_fmt(c.get('refused_per_seat'), 2)}",
        f"  ppo {(rec.get('update_s') or 0):.0f} s: kl {_fmt(u.get('kl'), 4)} (coef "
        f"{_fmt(u.get('kl_coef'), 3)} -> {_fmt(u.get('kl_coef_next'), 3)}), wait kl "
        f"{_fmt(u.get('kl_wait'), 5)}, q_play {_fmt(u.get('q_play'))} (ref "
        f"{_fmt(u.get('q_play_ref'))}), explained var {_fmt(u.get('explained_variance'))}, "
        f"clip {_fmt(u.get('clip_frac'))}",
        f"  league: {lg.get('active')} active, admitted {lg.get('admitted') or '-'}, evicted "
        f"{', '.join(lg.get('evicted') or ()) or '-'}"
        + ("; gate due" if rec.get("gate_due") else ""),
    ]
    return "\n".join(lines)


def _condition_lines(gate: Mapping[str, Any]) -> list[str]:
    try:
        import rs_league_eval
        return ["    " + rs_league_eval._condition_line(name, c)
                for name, c in (gate.get("conditions") or {}).items()]
    except Exception:                       # a report of another shape: the decision alone
        return []


def recommend(rows: Sequence[Mapping[str, Any]], *, battles: int, flush_rows: int,
              threads: int, budget_gb: float, cpus: int | None = None,
              available: int | None = None) -> dict:
    """The bench's --workers: as many as the RAM budget (the measured peak a worker plus
    BENCH_MARGIN) and the CPU (all logical CPUs but two, over --threads) allow, unless the
    measured throughput flattens (the last step's scaling under 50%): then the best measured."""
    peaks = [float(r["peak_rss_mb_max"]) for r in rows if r.get("peak_rss_mb_max")]
    per = (int(math.ceil(max(peaks) * BENCH_MARGIN)) if peaks
           else R.worker_mb(battles, flush_rows))
    cpus = (os.cpu_count() or 2) if cpus is None else int(cpus)
    cpu_cap = max(1, (cpus - 2) // max(1, int(threads)))
    ram_cap, why = R.ram_check(cpu_cap, per_worker_mb=per, budget_gb=budget_gb,
                               available=available)
    measured = sorted((max(1, int(r.get("workers") or 0)), float(r["matches_per_hour"]))
                      for r in rows if r.get("matches_per_hour"))
    rec, basis, eff = ram_cap, "the RAM and CPU caps", None
    if len(measured) >= 2:
        (w1, m1), (w2, m2) = measured[-2], measured[-1]
        if w2 > w1 and m1 > 0:
            eff = (m2 / m1) / (w2 / w1)
            if eff < 0.5:
                best = max(measured, key=lambda x: x[1])[0]
                rec = min(ram_cap, best)
                basis = f"throughput flattens past {best} workers (scaling {eff:.0%})"
    mph = None
    if measured:
        # The measured count nearest below the recommendation (else the smallest), scaled
        # linearly to it: exact at a measured count, optimistic above the largest.
        below = [x for x in measured if x[0] <= rec]
        w0, m0 = below[-1] if below else measured[0]
        mph = m0 * rec / w0
    top = measured[-1][0] if measured else None
    return {"recommended_workers": int(rec), "basis": basis, "per_worker_mb": per,
            "cpu_cap": cpu_cap, "ram_cap": ram_cap, "ram_note": why, "scaling": eff,
            "measured_max": top, "extrapolated": bool(top is not None and rec > top),
            "matches_per_hour_est": None if mph is None else round(mph, 1)}


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------

class Trainer:
    """One launch of a run (module doc): prepare() creates or resumes it, loop() trains,
    bench() measures."""

    def __init__(self, args: argparse.Namespace, paths: RunPaths, steps: Steps, echo: Echo,
                 argv: Sequence[str] = ()):
        self.args = args
        self.paths = paths
        self.steps = steps
        self.echo = echo
        self.argv = [str(a) for a in argv]
        self.t0 = time.time()
        self.gate_errors = 0
        self._stop_noted = False
        self.state: dict = {}
        self.flags: dict = {}
        self.frozen: dict = {}
        self.record: dict = {}

    # ------------------------------------------------------------ stop
    def stop_reason(self) -> str | None:
        for path in (self.paths.stop, REPO_STOP):
            if path.exists():
                return str(path)
        return None

    def poll(self) -> None:
        if not self._stop_noted and self.stop_reason():
            self._stop_noted = True
            self.echo(f"STOP seen ({self.stop_reason()}): the step that is running finishes "
                      "first, then the run stops")

    # ------------------------------------------------------------ setup
    def prepare(self, *, bench: bool = False) -> None:
        a, paths = self.args, self.paths
        why = self.stop_reason()
        if why:
            raise Refusal(f"refusing to start: {why} exists. Delete it to run: Remove-Item {why}")
        try:
            self.cfg = rs_engine.config_from_args(a)
        except (ValueError, FileNotFoundError) as exc:
            raise Refusal(f"engine config: {exc}") from None
        rs_engine.set_active(self.cfg)
        self.allow = True if a.allow_vocab_drift else None
        has_state = paths.state_json.is_file()
        if not bench:
            if a.resume and not has_state:
                raise Refusal(f"no run to resume at {paths.run} (no state.json): start it "
                              "without --resume")
            if not a.resume and has_state:
                raise Refusal(f"run {paths.name!r} already exists at {paths.run}: pass --resume to "
                              "continue it (rl_train.ps1 -From resume), or choose another --name")
        if a.resume and has_state and not paths.args_json.is_file():
            raise Refusal(f"{paths.args_json} is missing: the run's settings are gone; start a new "
                          "--name")
        stored = read_json(paths.args_json) if (a.resume and has_state) else None
        try:
            flags, changed = resolve_flags(a, None if stored is None else stored.get("flags"),
                                           self.cfg)
        except ValueError as exc:
            raise Refusal(str(exc)) from None
        if changed:
            shown = "; ".join(f"--{k.replace('_', '-')} {flags[k]!r} (the run's: "
                              f"{stored['flags'].get(k)!r})" for k in changed)
            raise Refusal(f"run {paths.name!r} was started with other settings: {shown}. Leave "
                          "these flags out to keep the run's, or start a new --name.")
        self.flags = flags
        self.record = engine_record(self.cfg)
        derived = derive(flags, self.cfg, self.allow, self.record)
        if stored is not None:
            was = stored.get("frozen") or {}
            diff = [k for k in COMPARED if derived.get(k) != was.get(k)]
            if diff:
                code = EXIT_FAILED if any(k in ENGINE_KEYS for k in diff) else EXIT_ARGS
                raise Refusal(f"run {paths.name!r} cannot resume: {', '.join(diff)} differ from "
                              f"when it started (engine id {was.get('engine_id')} then, "
                              f"{derived.get('engine_id')} now). Restore them (the same "
                              "--card-table / --engine-config, HASTYCR_* settings and checkpoint "
                              "files) or start a new --name.", code)
        self.frozen = derived
        self.noise, self.opp_noise = derived["noise"], derived["opponent_noise"]
        ns = types.SimpleNamespace(opponent_deck=flags["opponent_deck"],
                                   deck_stats=flags["deck_stats"],
                                   exclude_hog=flags["exclude_hog"], deck_top=flags["deck_top"],
                                   deck_alpha=flags["deck_alpha"])
        try:
            self.field = R.deck_field_from_args(ns, fallback=bench)
        except FileNotFoundError as exc:
            raise Refusal(f"{exc} (run: .venvs\\royalegym\\Scripts\\python.exe tools\\"
                          "deck_stats.py --top 2000 --workers 4)") from None
        except ValueError as exc:
            raise Refusal(f"deck field: {exc}") from None
        try:
            self.league_config = LG.LeagueConfig(admit_every=flags["admit_every"],
                                                 max_snapshots=flags["max_snapshots"],
                                                 decay=flags["decay"], mirror_share=0.0)
        except ValueError as exc:
            raise Refusal(str(exc)) from None
        if bench:
            return
        if a.resume:
            self.state = read_json(paths.state_json)
            stored.setdefault("launches", []).append(self.launch_record())
            write_json(paths.args_json, stored)
            if a.rollback is not None:
                self.rollback(int(a.rollback))
        else:
            self.create()

    def launch_record(self) -> dict:
        a = self.args
        return {"time": _now(), "argv": self.argv, "resume": bool(a.resume),
                "rollback": a.rollback, "steps": self.steps.describe(),
                "settings": {k: v for k, v in sorted(vars(a).items())
                             if k not in FROZEN and k != "command"}}

    def new_league(self) -> Any:
        anchors = {ANCHOR_PREFIX + self.frozen["start_name"]: self.flags["start"]}
        anchors.update((ANCHOR_PREFIX + n, p) for n, p in self.flags["anchor"])
        league = LG.League(self.league_config, anchors=anchors)
        league.promote(ANCHOR_PREFIX + self.frozen["start_name"])
        return league

    def create(self) -> None:
        paths, fl, fr = self.paths, self.flags, self.frozen
        paths.run.mkdir(parents=True, exist_ok=True)
        stamp = {"format": FORMAT, "tool": TOOL, "run": paths.name, "created": _now(),
                 "schema": fr["schema"], "engine": self.record["engine"],
                 "engine_id": self.record["engine_id"],
                 "engine_fingerprint": self.record["engine_fingerprint"],
                 "start": fl["start"], "start_sha256": fr["start_sha256"], "ref": fl["ref"]}
        doc = {"tool": TOOL, "format": FORMAT, "name": paths.name, "created": _now(),
               "wait_keep": fr["wait_keep"], "schema": fr["schema"],
               "engine": self.record["engine"], "engine_id": self.record["engine_id"],
               "engine_fingerprint": self.record["engine_fingerprint"], "flags": fl,
               "frozen": {k: v for k, v in fr.items() if k not in ("noise", "opponent_noise")},
               "noise_profile": fr["noise"], "launches": [self.launch_record()]}
        write_json(paths.args_json, doc)
        write_json(paths.engine_json, self.cfg.to_json())
        if self.noise is not None:
            paths.noise_json.write_text(PN.as_profile(self.noise).dumps(), encoding="utf-8")
        stamp_start(fl["start"], paths.ckpt(0), stamp)
        champion = {"id": ANCHOR_PREFIX + fr["start_name"], "name": fr["start_name"],
                    "path": fl["start"], "iteration": 0}
        league = self.new_league()
        self.state = {"format": FORMAT, "name": paths.name, "iteration": 0, "pending": None,
                      "gate_due": None, "league": league.to_dict(), "champion": champion,
                      "gates": [], "created": _now()}
        self.install_champion(champion, None)
        write_json(paths.sidecar(0), self.state)
        self.save_state()

    def save_state(self) -> None:
        self.state["updated"] = _now()
        write_json(self.paths.state_json, self.state)
        write_json(self.paths.league_json, self.state["league"])

    def install_champion(self, champ: Mapping[str, Any], gate: Mapping[str, Any] | None) -> None:
        copy_atomic(champ["path"], self.paths.champion)
        write_json(self.paths.champion_json, {
            "name": champ["name"], "id": champ["id"], "iteration": champ["iteration"],
            "source": champ["path"], "sha256": file_digest(self.paths.champion),
            "time": _now(), "gate": None if gate is None else {
                k: gate.get(k) for k in ("iteration", "candidate", "champion", "decision",
                                         "report")}})

    def rollback(self, k: int) -> None:
        paths, st = self.paths, self.state
        if not 0 <= k <= int(st["iteration"]):
            raise Refusal(f"--rollback {k}: the run is at iteration {st['iteration']}")
        if not (paths.sidecar(k).is_file() and paths.ckpt(k).is_file()):
            raise Refusal(f"--rollback {k}: {paths.sidecar(k).name} or {paths.ckpt(k).name} is "
                          "missing")
        archive = paths.run / f"rollback-{_stamp()}"
        moved = []
        for p in sorted(paths.run.iterdir()):
            j = iteration_of(p)
            if j is None or j < k or (j == k and p.suffix in (".pt", ".json")):
                continue
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(archive / p.name))
            moved.append(p.name)
        for folder in (paths.adam_dir, paths.gates):
            for p in sorted(folder.glob("it*")) if folder.is_dir() else ():
                j = iteration_of(Path(p.name.split(".")[0]))
                if j is not None and j > k:
                    (archive / folder.name).mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(archive / folder.name / p.name))
                    moved.append(f"{folder.name}/{p.name}")
        state = read_json(paths.sidecar(k))
        state.pop("record", None)
        state["pending"] = None
        self.state = state
        self.save_state()
        champ = state["champion"]
        if not (paths.champion.is_file()
                and file_digest(paths.champion) == file_digest(champ["path"])):
            self.install_champion(champ, None)
        append_jsonl(paths.log_jsonl, {"kind": "rollback", "iteration": k, "time": _now(),
                                       "archive": archive.name, "moved": moved})
        self.echo(f"rollback to iteration {k}: moved {len(moved)} file(s) to {archive.name}; "
                  f"champion {champ['name']}")

    # ------------------------------------------------------------ pieces
    def iter_seed(self, n: int) -> int:
        return int(self.flags["seed"]) * 1_000_003 + int(n)

    def workers_for(self, wanted: int) -> tuple[int, str]:
        """`wanted` capped by rollout.ram_check, per worker --worker-mb (else rollout's estimate),
        or the peak the last collection measured (plus BENCH_MARGIN) when that is more."""
        a = self.args
        if wanted <= 0 or a.no_ram_check:
            return wanted, ""
        per = a.worker_mb or R.worker_mb(a.battles, a.flush_rows)
        seen = self.state.get("peak_rss_mb")
        if seen:
            per = max(per, int(math.ceil(float(seen) * BENCH_MARGIN)))
        return R.ram_check(wanted, per_worker_mb=per, budget_gb=a.ram_budget_gb)

    def plans(self, count: int, *, learner: Path, league: Any, seed: int) -> list:
        fl = self.flags
        opp = R.SAME if fl["opponent_noise"] == R.SAME else self.opp_noise
        return R.plan_battles(count, learner=str(learner), league=league, deck_field=self.field,
                              seed=seed, start=0, learner_deck=fl["learner_deck"],
                              delay=fl["delay"], level=fl["level"], noise=self.noise,
                              opponent_noise=opp, noise_seed=fl["seed"], presses=fl["presses"],
                              opponent_decide=fl["opponent_decide"], engine=self.cfg,
                              max_ticks=fl["max_ticks"])

    def settings(self, learner: Path, *, write: bool, meta: Mapping[str, Any]) -> Any:
        a = self.args
        return R.Settings(learner=str(learner), battles=a.battles, threads=a.threads,
                          flush_rows=a.flush_rows, engine=self.cfg.to_json(),
                          allow_drift=self.allow, wait_keep=self.frozen["wait_keep"],
                          max_models=a.max_models, write=write, max_errors=a.max_errors,
                          meta=meta)

    def ppo_argv(self, ckpt: Path, data: Path, out: Path, opt: Path | None) -> list[str]:
        a = self.args
        argv = ["--ckpt", str(ckpt), "--ref", self.flags["ref"], "--data", str(data),
                "--out", str(out), "--device", a.ppo_device, "--precision", a.precision,
                "--ram-budget-gb", f"{a.ppo_ram_gb:g}", "--seed", str(self.flags["seed"])]
        if opt is not None:
            argv += ["--opt-state", str(opt)]
        for flag, value in (("--lr", a.lr), ("--critic-lr", a.critic_lr),
                            ("--epochs", a.ppo_epochs), ("--minibatch", a.minibatch),
                            ("--kl-budget", a.kl_budget), ("--wait-kl-budget", a.wait_kl_budget),
                            ("--ent-coef", a.ent_coef), ("--clip", a.clip),
                            ("--target-kl", a.target_kl)):
            if value is not None:
                argv += [flag, f"{value:g}" if isinstance(value, float) else str(value)]
        if a.ppo_extra:
            argv += shlex.split(a.ppo_extra, posix=os.name != "nt")
        return argv

    def anchors(self) -> list[tuple[str, str]]:
        """(gate name, checkpoint) of every imitation anchor: the start, then each --anchor."""
        return ([(self.frozen["start_name"], self.flags["start"])]
                + [(n, p) for n, p in self.flags["anchor"]])

    def gate_workers(self) -> int:
        a = self.args
        if a.gate_workers:
            return int(a.gate_workers)
        return max(1, min(GATE_WORKERS, a.workers or GATE_WORKERS))

    def gate_argv(self, cand: str, cand_path: Path, champ: Mapping[str, Any],
                  anchors: Sequence[tuple[str, str]], report: Path) -> list[str]:
        fl, paths = self.flags, self.paths
        argv = ["run", "--candidate", f"{cand}={cand_path}",
                "--champion", f"{champ['name']}={champ['path']}"]
        for name, path in anchors:
            argv += ["--anchor", f"{name}={path}"]
        argv += ["--split", GATE_SPLIT, "--seed", str(fl["gate_seed"]),
                 "--delay", str(fl["delay"]),
                 "--noise", str(paths.noise_json) if self.noise is not None else "none",
                 "--max-ticks", str(fl["max_ticks"]),
                 "--champion-games", str(fl["gate_champion_games"]),
                 "--anchor-games", str(fl["gate_anchor_games"]), "--decide", fl["gate_decide"],
                 "--workers", str(self.gate_workers()), "--log", str(paths.gate_log),
                 "--report", str(report), "--engine-config", str(paths.engine_json)]
        if fl["presses"]:
            argv.append("--presses")
        if self.allow:
            argv.append("--allow-vocab-drift")
        return argv

    def _clear_data(self, folder: Path) -> None:
        for f in folder.glob("*.npz"):
            f.unlink()
        shutil.rmtree(folder / R.TMP_DIR, ignore_errors=True)
        (folder / R.SUMMARY_JSON).unlink(missing_ok=True)

    def _data_ready(self, n: int) -> bool:
        try:
            P.data_files([self.paths.data(n)])
        except ValueError:
            return False
        return (self.paths.data(n) / R.SUMMARY_JSON).is_file()

    # ------------------------------------------------------------ the steps
    def collect(self, n: int) -> dict:
        a, paths = self.args, self.paths
        folder = paths.data(n)
        folder.mkdir(parents=True, exist_ok=True)
        self._clear_data(folder)
        learner = paths.ckpt(n)
        if not learner.is_file():
            raise StepError(f"the learner {learner} is missing")
        mix = MixDraw(LG.League.from_dict(self.state["league"]), self.flags["mix"])
        workers, why = self.workers_for(a.workers)
        if why:
            self.echo(f"workers: {why}")
        seed = self.iter_seed(n)
        plans = self.plans(a.matches, learner=learner, league=mix, seed=seed)
        kinds = mix.kinds()
        drawn = Counter(kinds.get(p.opponent, "other") for p in plans)
        meta = {"run": paths.name, "iteration": n, "seed": seed, "delay": self.flags["delay"],
                "level": self.flags["level"], "noise_digest": self.frozen["noise_digest"],
                "mix": mix.shares(), "drawn": dict(drawn)}
        settings = self.settings(learner, write=True, meta=meta)
        task = {"plans": [p.to_dict() for p in plans], "settings": settings.to_dict(),
                "workers": workers, "out": str(folder), "prefix": "rollout"}
        task_path = write_json(folder / "collect.json", task)
        self.echo(f"it {n:04d}: collect {len(plans)} matches with {learner.name} on "
                  f"{workers if workers > 0 else 'no extra'} worker(s) x K {a.battles} "
                  f"(opponents: {', '.join(f'{k} {v}' for k, v in sorted(drawn.items()))})")
        self.steps.collect(task_path, paths.step_log(n, "collect"), self.echo, poll=self.poll)
        try:
            full = read_json(folder / R.SUMMARY_JSON)
        except (OSError, ValueError) as exc:
            raise StepError(f"the collection wrote no summary ({exc}); see "
                            f"{paths.step_log(n, 'collect')}") from None
        engine = (full.get("settings") or {}).get("engine")
        if engine != self.record["engine"]:
            raise StepError(f"iteration {n} was collected under engine {json.dumps(engine)}, the "
                            f"run under {json.dumps(self.record['engine'])}")
        if not full.get("matches") or not full.get("rows"):
            raise StepError(f"the collection played {full.get('matches')} matches and wrote "
                            f"{full.get('rows')} rows; see {paths.step_log(n, 'collect')}")
        digest = collect_digest(full, kinds)
        peak = digest["peak_rss_mb_max"]
        if (peak and workers > 0 and not a.no_ram_check
                and peak * workers > a.ram_budget_gb * 1024):
            self.echo(f"warning: {workers} workers peaked at {peak:.0f} MB each "
                      f"({peak * workers / 1024:.1f} GB, over --ram-budget-gb "
                      f"{a.ram_budget_gb:g}); the next collection plans with that peak")
        return digest

    def update(self, n: int) -> dict:
        paths = self.paths
        out, opt, prev = paths.ckpt(n + 1), paths.adam(n + 1), paths.adam(n)
        paths.adam_dir.mkdir(parents=True, exist_ok=True)
        if prev.is_file():
            copy_atomic(prev, opt)              # the same moments every time this update runs
        elif opt.exists():
            opt.unlink()
        argv = self.ppo_argv(paths.ckpt(n), paths.data(n), out, opt)
        self.echo(f"it {n:04d}: PPO update -> {out.name}")
        stats = self.steps.update(argv, paths.step_log(n, "update"), self.echo, poll=self.poll)
        if not out.is_file():
            raise StepError(f"the update wrote no {out}; see {paths.step_log(n, 'update')}")
        for old in paths.adam_dir.glob("it*.pt"):
            j = iteration_of(old)
            if j is not None and j < n:
                old.unlink()
        return stats

    def commit(self, n: int) -> dict:
        a, paths, st = self.args, self.paths, self.state
        pend = st["pending"]
        league = LG.League.from_dict(st["league"])
        filed = league.record_many((str(o), s) for o, s in pend["collect"]["league_results"])
        c = n + 1
        info = league.end_iteration(snapshot=str(paths.ckpt(c)), meta={"iteration": c})
        gate_due = c if a.gate_every > 0 and c % a.gate_every == 0 else None
        rec = {"kind": "iteration", "iteration": n, "learner": paths.ckpt(n).name,
               "out": paths.ckpt(c).name, "time": _now(), "collect_s": pend.get("collect_s"),
               "update_s": pend.get("update_s"), "collect": pend["collect"],
               "update": pend["update"],
               "league": {"filed": filed, "admitted": info["admitted"],
                          "evicted": info["evicted"], "active": len(league.active_ids()),
                          "iteration": league.iteration, "champion": league.champion},
               "gate_due": gate_due}
        st.update(iteration=c, pending=None, gate_due=gate_due, league=league.to_dict())
        peak = pend["collect"].get("peak_rss_mb_max")
        if peak and (pend["collect"].get("workers_run") or 0) > 0:
            st["peak_rss_mb"] = peak           # a worker process's, not this one's (workers 0)
        write_json(paths.sidecar(c), {**st, "record": rec})
        self.save_state()
        append_jsonl(paths.log_jsonl, rec)
        self.echo(describe_iteration(rec))
        keep = int(a.keep_rollouts)
        old = n - keep
        if old >= 0:
            for f in paths.data(old).glob("*.npz"):
                f.unlink()
            shutil.rmtree(paths.data(old) / R.TMP_DIR, ignore_errors=True)
        return rec

    def iterate(self, n: int) -> None:
        st = self.state
        pend = st.get("pending")
        if pend is not None and pend.get("iteration") != n:
            self.echo(f"warning: a pending step of iteration {pend.get('iteration')} at "
                      f"iteration {n}: dropped")
            pend = None
        if pend is not None and pend.get("update") is None and not self._data_ready(n):
            pend = None                     # its rollouts are gone: collect again
        if pend is None:
            t0 = time.time()
            digest = self.collect(n)
            pend = {"iteration": n, "collect": digest, "collect_s": round(time.time() - t0, 1),
                    "update": None}
            st["pending"] = pend
            self.save_state()
        else:
            self.echo(f"it {n:04d}: the collection was done before; resuming at the "
                      f"{'commit' if pend.get('update') else 'update'}")
        if self.stop_reason():
            return
        if pend.get("update") is None or not self.paths.ckpt(n + 1).is_file():
            t0 = time.time()
            pend["update"] = self.update(n)
            pend["update_s"] = round(time.time() - t0, 1)
            self.save_state()
        self.commit(n)

    def run_gate(self, c: int) -> dict:
        paths, fl, st = self.paths, self.flags, self.state
        cand_path = paths.ckpt(c)
        champ = dict(st["champion"])
        rec: dict[str, Any] = {"kind": "gate", "iteration": c, "path": str(cand_path),
                               "champion": champ["name"], "time": _now(), "decision": None,
                               "error": None, "report": str(paths.gate_report(c))}
        t0 = time.time()
        if not cand_path.is_file():
            rec["error"] = f"{cand_path} is missing"
        elif _same_file(cand_path, champ["path"]):
            rec["error"] = "the candidate is the champion"
        else:
            cand = f"it{c:04d}-{file_digest(cand_path)[:6]}"
            rec["candidate"] = cand
            anchors = [(n, p) for n, p in self.anchors() if n != champ["name"]]
            report = paths.gate_report(c)
            report.unlink(missing_ok=True)
            argv = self.gate_argv(cand, cand_path, champ, anchors, report)
            self.echo(f"gate at iteration {c}: {cand} vs the champion {champ['name']}, anchors "
                      f"{', '.join(n for n, _ in anchors) or 'none'}; "
                      f"{fl['gate_champion_games']} games, then {fl['gate_anchor_games']} per "
                      f"anchor, dev split, delay {fl['delay']}, decide {fl['gate_decide']}")
            code = self.steps.gate(argv, paths.logs / f"gate-it{c:04d}.log", self.echo,
                                   poll=self.poll)
            gate = None
            if code == 0 and report.is_file():
                try:
                    gate = (read_json(report).get("gates") or {}).get(cand)
                except (OSError, ValueError):
                    gate = None
            if gate is None:
                rec["error"] = (f"exit {code}" if code != 0 else "no decision in the report")
                rec["error"] += f"; see {paths.logs / f'gate-it{c:04d}.log'}"
            else:
                rec["decision"] = gate.get("decision")
                rec["turned_on"] = gate.get("turned_on")
                rec["conditions"] = {
                    name: {k: cond.get(k) for k in ("passed", "n", "observed", "bound",
                                                    "reference", "missing", "skipped")}
                    for name, cond in (gate.get("conditions") or {}).items()}
                rec["lines"] = _condition_lines(gate)
        rec["seconds"] = round(time.time() - t0, 1)
        if rec["error"]:
            self.gate_errors += 1
        league = LG.League.from_dict(st["league"])
        if rec["decision"] in (PASS, CYCLE):
            member = LG.League.snapshot_id(c)
            if member not in league:
                league.admit(str(cand_path), member_id=member, iteration=c,
                             meta={"gate": rec["decision"]})
            if rec["decision"] == PASS:
                league.promote(member)
                st["champion"] = {"id": member, "name": rec["candidate"],
                                  "path": str(cand_path), "iteration": c}
                self.install_champion(st["champion"], rec)
        st["league"] = league.to_dict()
        st.setdefault("gates", []).append(rec)
        if st.get("gate_due") == c:
            st["gate_due"] = None
        self.save_state()
        append_jsonl(paths.log_jsonl, rec)
        if rec["error"]:
            self.echo(f"gate at iteration {c}: COULD NOT RUN ({rec['error']}); the champion "
                      f"stays {champ['name']}")
        else:
            verdict = ("PASS: the new champion, copied to champion.pt" if rec["decision"] == PASS
                       else f"{str(rec['decision']).upper()}: the champion stays {champ['name']}")
            self.echo("\n".join([f"gate at iteration {c}: {rec['candidate']} {verdict} "
                                 f"({rec['seconds'] / 60:.0f} min)", *rec.get("lines", [])]))
        return rec

    # ------------------------------------------------------------ the loop
    def banner(self) -> None:
        fl, fr, st, a = self.flags, self.frozen, self.state, self.args
        noise = "none" if self.noise is None else (
            f"{PN.as_profile(self.noise).name} ({fr['noise_digest']}) from {fl['noise']}")
        mix = MixDraw(LG.League.from_dict(st["league"]), fl["mix"])
        shares = " / ".join(f"{k} {100 * v:.0f}%" for k, v in mix.shares().items())
        anchors = ", ".join(n for n, _ in self.anchors())
        decks = (f"one deck ({fl['opponent_deck']})" if fl["opponent_deck"]
                 else f"{len(self.field)} train-split decks from {fl['deck_stats']}")
        self.echo("\n".join([
            f"train_loop: run {self.paths.name} at iteration {st['iteration']} in {self.paths.run}",
            f"  start {fr['start_name']} ({fl['start']}), reference {fl['ref']}; schema "
            f"{fr['schema']}, wait_keep {fr['wait_keep']:g}"
            f"{', presses trained' if fr['presses_trained'] else ''}",
            f"  engine: {self.cfg.describe()} (id {self.record['engine_id']})",
            f"  delay {fl['delay']} ticks, level {fl['level']}, noise {noise}; opponents' noise "
            f"{'same' if fl['opponent_noise'] == R.SAME else fl['opponent_noise']}",
            f"  opponents: {shares}; anchors {anchors}; decks: {decks}",
            f"  champion {st['champion']['name']}; gate every {a.gate_every or 'never'} "
            f"iteration(s): {fl['gate_champion_games']} + {fl['gate_anchor_games']} per anchor "
            f"games, {GATE_SPLIT} split, decide {fl['gate_decide']}",
            f"  per iteration: {a.matches} matches on {a.workers} worker(s) x K {a.battles}, "
            f"{a.threads} thread(s) each; steps: {self.steps.describe()}",
            f"  stop: create {self.paths.stop} (or {REPO_STOP}); the running step finishes first",
        ]))
        if not fl["opponent_deck"] and len(self.field) < MIN_DECKS:
            self.echo(f"warning: {len(self.field)} opponent decks, the plan asks for at least "
                      f"{MIN_DECKS}: rebuild with tools\\deck_stats.py --top 2000 (a new run)")

    def loop(self) -> int:
        a, st = self.args, self.state
        self.banner()
        if a.gate_now is not None:
            k = int(st["iteration"]) if a.gate_now < 0 else int(a.gate_now)
            if not 1 <= k <= int(st["iteration"]):
                raise Refusal(f"--gate-now {k}: iterations 1-{st['iteration']} have checkpoints")
            self.run_gate(k)
            return EXIT_GATE if self.gate_errors else EXIT_OK
        deadline = self.t0 + a.hours * 3600 if a.hours > 0 else None
        reason = ""
        while True:
            why = self.stop_reason()
            if why:
                reason = f"STOP ({why}); delete it before resuming"
                break
            if st.get("gate_due") is not None:
                self.run_gate(int(st["gate_due"]))
                continue
            n = int(st["iteration"])
            if a.iterations and n >= a.iterations:
                reason = f"{n} iterations done (--iterations {a.iterations})"
                break
            if deadline is not None and time.time() >= deadline:
                reason = f"{a.hours:g} hour(s) passed (--hours)"
                break
            self.iterate(n)
            st = self.state
        champ = st["champion"]
        self.echo(f"stopped: {reason}. Iteration {st['iteration']}, champion {champ['name']} "
                  f"({self.paths.champion}). Resume: train_loop.py run --name {self.paths.name} "
                  "--resume (rl_train.ps1 -From resume)")
        if self.gate_errors:
            self.echo(f"warning: {self.gate_errors} gate(s) could not run; see the gate logs in "
                      f"{self.paths.logs}")
            return EXIT_GATE
        return EXIT_OK

    # ------------------------------------------------------------ the bench
    def bench_counts(self) -> list[int]:
        a = self.args
        if a.bench_workers:
            try:
                counts = sorted({int(x) for x in str(a.bench_workers).split(",") if x.strip()})
            except ValueError:
                raise Refusal(f"--bench-workers: comma-separated counts, got "
                              f"{a.bench_workers!r}") from None
            if not counts or counts[0] < 0:
                raise Refusal("--bench-workers: counts >= 0 (0: in this process)")
            return counts
        cpus = os.cpu_count() or 2
        cpu_cap = max(1, (cpus - 2) // max(1, a.threads))
        per = a.worker_mb or R.worker_mb(a.battles, a.flush_rows)
        cap = R.ram_check(cpu_cap, per_worker_mb=per, budget_gb=a.ram_budget_gb)[0]
        return sorted({1, max(1, cap // 2), cap})

    def bench(self) -> int:
        a, fl = self.args, self.flags
        budget = parse_duration(a.bench)
        folder = self.paths.run / f"bench-{_stamp()}"
        folder.mkdir(parents=True, exist_ok=True)
        self.echo.path = folder / "bench.log"
        counts = self.bench_counts()
        league = self.new_league()
        mix = MixDraw(league, fl["mix"])
        learner = Path(fl["start"])
        slice_s = budget / len(counts)
        self.echo(f"bench: about {budget / 60:.0f} min on {learner}, K {a.battles} battles a "
                  f"worker, {a.threads} thread(s) each; worker counts "
                  f"{', '.join(str(w) for w in counts)}; {len(self.field)} opponent deck(s); "
                  f"files in {folder}")
        rows, rate, t_start, ppo, out_dir = [], None, time.time(), None, None
        peak_seen = None
        for i, w in enumerate(counts):
            if i and time.time() - t_start > budget:
                self.echo(f"bench: time is up; {', '.join(str(x) for x in counts[i:])} not "
                          "measured")
                break
            if w > 1 and peak_seen and not a.no_ram_check:
                # What the smaller counts measured, not the estimate, decides whether this fits.
                fits, why = R.ram_check(w, per_worker_mb=int(math.ceil(peak_seen * BENCH_MARGIN)),
                                        budget_gb=a.ram_budget_gb)
                if fits < w:
                    self.echo(f"bench: {w} workers would not fit ({why}); {fits} instead")
                    if any(r["workers"] == fits for r in rows):
                        continue
                    w = fits
            eff = max(1, w)
            wave = eff * a.battles
            matches = wave if rate is None else int(max(wave, min(BENCH_MAX_WAVES * wave,
                                                                  rate * eff * slice_s)))
            last = i == len(counts) - 1 and not a.no_bench_ppo
            out = folder / f"w{w:02d}" if last else None
            plans = self.plans(matches, learner=learner, league=mix, seed=BENCH_SEED + i)
            settings = self.settings(learner, write=out is not None,
                                     meta={"bench": True, "workers": w})
            task = {"plans": [p.to_dict() for p in plans], "settings": settings.to_dict(),
                    "workers": w, "out": None if out is None else str(out), "prefix": "bench"}
            task_path = write_json(folder / f"w{w:02d}.json", task)
            self.echo(f"bench: {matches} matches on {w} worker(s)")
            try:
                summary = self.steps.collect(task_path, folder / f"w{w:02d}.log", self.echo)
            except StepError as exc:
                self.echo(f"bench: {w} worker(s) failed: {exc}")
                continue
            row = R.bench_row(summary, a.battles)
            row["workers"] = w
            row["rows"] = summary.get("rows")
            rows.append(row)
            if row.get("wall_s"):
                rate = row["matches"] / float(row["wall_s"]) / eff
            peak = row.get("peak_rss_mb_max")
            if peak:
                peak_seen = max(peak_seen or 0.0, float(peak))
            self.echo(f"  workers {w:>2}: {row['matches']} matches in {_fmt(row['wall_s'], 0)} s"
                      f" = {_fmt(row['matches_per_hour'], 0)} matches/hour, "
                      f"{_fmt(row['decisions_per_s'], 0)} decisions/s, RAM per worker "
                      f"{'-' if peak is None else f'{peak:.0f} MB'} at peak, errors "
                      f"{row['errors']}")
            if out is not None and summary.get("rows"):
                out_dir = out
        if out_dir is not None:
            argv = self.ppo_argv(learner, out_dir, folder / "ppo_bench.pt", None)
            try:
                stats = self.steps.update(argv, folder / "ppo.log", self.echo)
                ppo = {"rows": stats.get("rows"), "seconds": stats.get("seconds")}
                self.echo(f"  ppo: {ppo['rows']} rows in {_fmt(ppo['seconds'], 1)} s")
            except StepError as exc:
                ppo = {"error": str(exc)}
                self.echo(f"  ppo: {exc}")
        if not rows:
            self.echo("bench: nothing was measured")
            return EXIT_FAILED
        rec = recommend(rows, battles=a.battles, flush_rows=a.flush_rows, threads=a.threads,
                        budget_gb=a.ram_budget_gb)
        w = rec["recommended_workers"]
        lines = [f"RAM per worker: {rec['per_worker_mb']} MB with a {BENCH_MARGIN - 1:.0%} margin;"
                 f" {rec['ram_note']}",
                 f"recommended: --workers {w} ({rec['basis']}"
                 + ("; more than measured, so bench it once at that count" if rec["extrapolated"]
                    else "") + ")"]
        last = rows[-1]
        rows_per_match = (round(float(last["rows"]) / float(last["matches"]), 1)
                          if last.get("rows") and last.get("matches") else None)
        if rec["matches_per_hour_est"]:
            minutes = 60.0 * a.matches / rec["matches_per_hour_est"]
            text = (f"at --workers {w}: about {rec['matches_per_hour_est']:.0f} matches/hour; an "
                    f"iteration of --matches {a.matches} collects in about {minutes:.0f} min")
            if ppo and ppo.get("seconds") and ppo.get("rows") and rows_per_match:
                upd = float(ppo["seconds"]) / float(ppo["rows"]) * rows_per_match * a.matches
                text += f", and updates in about {upd / 60:.1f} min ({rows_per_match} rows a match)"
            lines.append(text)
        self.echo("\n".join(lines))
        result = {"tool": TOOL, "bench": rows, "ppo": ppo, **rec, "matches": a.matches,
                  "battles": a.battles, "threads": a.threads, "budget_s": budget,
                  "rows_per_match": rows_per_match}
        write_json(folder / "bench.json", result)
        for f in folder.rglob("*.npz"):
            f.unlink()
        (folder / "ppo_bench.pt").unlink(missing_ok=True)
        print(json.dumps(result, default=_jsonable), flush=True)
        return EXIT_OK


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The RL driver (R04): collection and PPO in turn, "
                                             "the league, the promotion gate, resume, bench.")
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="train (or --bench); module doc")
    run.add_argument("--name", default=DEFAULT_NAME, help="the run: runs/rl/<name>/")
    run.add_argument("--runs-dir", type=Path, default=None,
                     help="where runs live (default <runs>/rl, src/hastycr/machine.py)")
    run.add_argument("--resume", action="store_true", help="continue the run from state.json")
    run.add_argument("--rollback", type=int, default=None, metavar="N",
                     help="with --resume: go back to iteration N (later files move to "
                          "rollback-<time>/)")
    run.add_argument("--gate-now", type=int, nargs="?", const=-1, default=None, metavar="N",
                     help="with --resume: run the gate on iteration N (default the newest), "
                          "then exit")
    run.add_argument("--bench", nargs="?", const=DEFAULT_BENCH, default=None, metavar="DURATION",
                     help=f"measure instead of training, for about DURATION (default "
                          f"{DEFAULT_BENCH}): matches/hour, RAM per worker, a --workers")
    run.add_argument("--bench-workers", default=None, metavar="COUNTS",
                     help="worker counts to bench, e.g. 1,4,8 (0: in this process)")
    run.add_argument("--no-bench-ppo", action="store_true", help="the bench times no update")
    g = run.add_argument_group("how long")
    g.add_argument("--iterations", type=int, default=0, help="stop once this many are done")
    g.add_argument("--hours", type=float, default=0.0,
                   help="stop before a collection once this launch has run this long")
    g = run.add_argument_group("collection (may change on resume)")
    g.add_argument("--matches", type=int, default=DEFAULT_MATCHES, help="battles per iteration")
    g.add_argument("--workers", type=int, default=R.DEFAULT_WORKERS,
                   help="rollout worker processes (0: in the collect process); RAM-checked")
    g.add_argument("--battles", type=int, default=R.DEFAULT_BATTLES,
                   help="K: battles open at once per worker")
    g.add_argument("--threads", type=int, default=R.DEFAULT_THREADS,
                   help="torch CPU threads per worker (1-2)")
    g.add_argument("--flush-rows", type=int, default=R.DEFAULT_FLUSH_ROWS)
    g.add_argument("--max-models", type=int, default=R.DEFAULT_MAX_MODELS)
    g.add_argument("--max-errors", type=int, default=R.MAX_ERRORS,
                   help="failed battles a worker tolerates")
    g.add_argument("--ram-budget-gb", type=float, default=R.RAM_BUDGET_GB,
                   help="memory for all rollout workers together")
    g.add_argument("--worker-mb", type=int, default=None,
                   help="memory per worker for the RAM check (the bench measures it)")
    g.add_argument("--no-ram-check", action="store_true")
    g.add_argument("--keep-rollouts", type=int, default=0,
                   help="iterations whose rollout files are kept after the update (0: none)")
    g = run.add_argument_group("frozen when the run starts (module doc, RESUME AND ROLLBACK)")
    g.add_argument("--start", default=None,
                   help="the v2 imitation checkpoint the learner starts from (default "
                        "<runs>/tokens/v2_hog/best.pt); also the first champion and an anchor")
    g.add_argument("--ref", default=None, help="the KL reference (default --start)")
    g.add_argument("--anchor", action="append", default=None, metavar="NAME=CKPT",
                   help="an imitation anchor: a league opponent and the gate's regression check")
    g.add_argument("--delay", type=int, default=None,
                   help=f"command delay in ticks (default data/engine.json's, else "
                        f"{R.DEFAULT_DELAY})")
    g.add_argument("--level", type=int, default=None,
                   help=f"card and tower level of both sides (default {R.FRIENDLY_LEVEL})")
    g.add_argument("--noise", default=None, metavar="PROFILE",
                   help="perception noise: a profile JSON, default (the prior) or none "
                        "(default data/noise/measured.json, else the prior)")
    g.add_argument("--opponent-noise", default=None, metavar="PROFILE",
                   help="the opponents' noise: same (default), none or a profile")
    g.add_argument("--learner-deck", default=None, metavar="NAMES", help="default hog26")
    g.add_argument("--deck-stats", default=None,
                   help="tools/deck_stats.py JSON (default <data>/deck_stats.json; train split)")
    g.add_argument("--deck-top", type=int, default=None, help="only the N most played (0: all)")
    g.add_argument("--deck-alpha", type=float, default=None,
                   help="decks dealt in proportion to count ** alpha (default 1)")
    g.add_argument("--exclude-hog", action="store_const", const=True, default=None,
                   help="no Hog 2.6 opponent decks")
    g.add_argument("--opponent-deck", default=None, metavar="NAMES",
                   help="one opponent deck instead of the field (tests, checks)")
    g.add_argument("--mix", default=None, metavar="MIRROR,POOL,ANCHORS",
                   help="opponent shares (default 0.5,0.35,0.15)")
    g.add_argument("--admit-every", type=int, default=None,
                   help=f"a snapshot joins the league every N iterations (default "
                        f"{DEFAULT_ADMIT_EVERY})")
    g.add_argument("--max-snapshots", type=int, default=None)
    g.add_argument("--decay", type=float, default=None)
    g.add_argument("--presses", choices=("on", "off"), default=None,
                   help="ability presses where the checkpoint's buttons trained (default on)")
    g.add_argument("--opponent-decide", choices=RD.MODES, default=None,
                   help="frozen opponents' decision rule (default sample)")
    g.add_argument("--max-ticks", type=int, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--gate-champion-games", type=int, default=None,
                   help=f"default {CHAMPION_GAMES}")
    g.add_argument("--gate-anchor-games", type=int, default=None, help=f"default {ANCHOR_GAMES}")
    g.add_argument("--gate-decide", choices=RD.MODES, default=None,
                   help="the gate's decision rule (default live, the live runner's)")
    g.add_argument("--gate-seed", type=int, default=None)
    g = run.add_argument_group("the gate (may change on resume)")
    g.add_argument("--gate-every", type=int, default=DEFAULT_GATE_EVERY,
                   help="run the promotion gate every M iterations (0: never)")
    g.add_argument("--gate-workers", type=int, default=None,
                   help=f"rs_league_eval workers (default min(--workers, {GATE_WORKERS}))")
    g = run.add_argument_group("the update, tools/rl/ppo.py (may change on resume)")
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--critic-lr", type=float, default=None)
    g.add_argument("--ppo-epochs", type=int, default=None)
    g.add_argument("--minibatch", type=int, default=None)
    g.add_argument("--kl-budget", type=float, default=None)
    g.add_argument("--wait-kl-budget", type=float, default=None)
    g.add_argument("--ent-coef", type=float, default=None)
    g.add_argument("--clip", type=float, default=None)
    g.add_argument("--target-kl", type=float, default=None)
    g.add_argument("--ppo-device", default="auto", help="auto, cuda or cpu")
    g.add_argument("--precision", choices=P.PRECISIONS, default="bf16")
    g.add_argument("--ppo-ram-gb", type=float, default=P.RAM_BUDGET_GB,
                   help="ppo.py --ram-budget-gb")
    g.add_argument("--ppo-extra", default="", help="more ppo.py flags, as one string")
    g = run.add_argument_group("processes")
    g.add_argument("--inline", action="store_true",
                   help="run every step in this process (tests, debugging)")
    g.add_argument("--allow-vocab-drift", action="store_true")
    rs_engine.add_engine_args(run)
    st = sub.add_parser("status", help="print a run's state (read only)")
    st.add_argument("--name", default=DEFAULT_NAME)
    st.add_argument("--runs-dir", type=Path, default=None)
    ct = sub.add_parser("collect-task", help="(internal) one collection from a task file")
    ct.add_argument("--task", type=Path, required=True)
    return ap


def _check_run_args(a: argparse.Namespace) -> str | None:
    if not NAME_RE.match(a.name or ""):
        return f"--name {a.name!r}: letters, digits, '_', '.' and '-' (up to 64)"
    if a.matches < 1 or a.battles < 1 or a.flush_rows < 1 or a.max_models < 1:
        return "--matches, --battles, --flush-rows and --max-models: at least 1"
    if a.threads < 0 or a.workers < 0 or a.max_errors < 0 or a.keep_rollouts < 0:
        return "--threads, --workers, --max-errors and --keep-rollouts: at least 0"
    if a.iterations < 0 or a.hours < 0 or a.gate_every < 0 or a.ram_budget_gb <= 0:
        return "--iterations, --hours and --gate-every: at least 0; --ram-budget-gb above 0"
    if (a.rollback is not None or a.gate_now is not None) and not a.resume:
        return "--rollback and --gate-now go with --resume"
    if a.bench is not None and (a.rollback is not None or a.gate_now is not None):
        return "--bench does not go with --rollback or --gate-now"
    if a.bench is not None:
        try:
            parse_duration(a.bench)
        except ValueError as exc:
            return f"--bench: {exc}"
    return None


def make_steps(args: argparse.Namespace) -> Steps:
    """InlineSteps for --inline, else SubprocessSteps with hastycr.machine's interpreters (the
    update in this process when this interpreter is py_train)."""
    if args.inline:
        return InlineSteps()
    from hastycr import machine
    py_engine, py_train = machine.py_engine(), machine.py_train()
    for what, py in (("engine", py_engine), ("GPU", py_train)):
        if not Path(py).is_file():
            raise Refusal(f"the {what} venv's python {py} is missing: run "
                          r"scripts\setup_windows.ps1 (or set HASTYCR_PY_ENGINE / "
                          "HASTYCR_PY_TRAIN)")
    return SubprocessSteps(py_engine, py_train, ppo_inline=_same_file(sys.executable, py_train))


def cmd_run(args: argparse.Namespace, argv: Sequence[str], steps: Steps | None = None) -> int:
    problem = _check_run_args(args)
    if problem:
        print(f"error: {problem}", file=sys.stderr, flush=True)
        return EXIT_ARGS
    paths = RunPaths(runs_dir(args.runs_dir), args.name)
    echo = Echo()
    bench = args.bench is not None
    try:
        steps = steps if steps is not None else make_steps(args)
        with RunLock(paths.lock):
            trainer = Trainer(args, paths, steps, echo, argv)
            if not bench:
                echo.path = paths.driver_log
            trainer.prepare(bench=bench)
            if bench:
                return trainer.bench()
            try:
                return trainer.loop()
            except StepError as exc:
                echo(f"error: {exc}")
                echo(f"the run stays at iteration {trainer.state.get('iteration')}; fix the cause, "
                     f"then resume: train_loop.py run --name {paths.name} --resume")
                return EXIT_FAILED
            except KeyboardInterrupt:
                echo(f"interrupted at iteration {trainer.state.get('iteration')}; resume with "
                     f"train_loop.py run --name {paths.name} --resume (the interrupted step runs "
                     "again)")
                return EXIT_INTERRUPTED
    except Refusal as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return exc.code


def cmd_status(args: argparse.Namespace) -> int:
    paths = RunPaths(runs_dir(args.runs_dir), args.name)
    if not paths.state_json.is_file():
        print(f"error: no run at {paths.run} (no state.json)", file=sys.stderr)
        return EXIT_ARGS
    st = read_json(paths.state_json)
    doc = read_json(paths.args_json) if paths.args_json.is_file() else {}
    pend = st.get("pending")
    league = LG.League.from_dict(st["league"])
    lines = [f"run {paths.name}: {paths.run}",
             f"  iteration {st['iteration']} (learner it{int(st['iteration']):04d}.pt)"
             + ("" if not pend else f"; iteration {pend.get('iteration')} "
                f"{'updated' if pend.get('update') else 'collected'}, not committed"),
             f"  champion {st['champion']['name']} (iteration {st['champion']['iteration']}) -> "
             f"{paths.champion}",
             f"  engine id {doc.get('engine_id')}, schema {doc.get('schema')}, wait_keep "
             f"{doc.get('wait_keep')}, launches {len(doc.get('launches') or ())}"]
    if st.get("gate_due"):
        lines.append(f"  gate due at iteration {st['gate_due']}")
    for g in st.get("gates") or ():
        lines.append(f"  gate it{int(g['iteration']):04d}: "
                     f"{(g.get('decision') or 'error').upper()} vs {g.get('champion')}"
                     + (f" ({g['error']})" if g.get("error") else
                        f" (turned on {g.get('turned_on')})"))
    lines.append(f"  league: {len(league.active_ids())} active of {len(league)}")
    for m in league.summary()["members"]:
        if m["active"]:
            lines.append(f"    {m['id']}: expected {m['expected']:.3f} over {m['games']} games, "
                         f"weight {m['weight']:.3f}")
    recs = []
    if paths.log_jsonl.is_file():
        for line in paths.log_jsonl.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("kind") == "iteration":
                recs.append(rec)
    for rec in recs[-5:]:
        lines.append(describe_iteration(rec))
    print("\n".join(lines))
    print(json.dumps({"run": str(paths.run), "iteration": st["iteration"],
                      "champion": st["champion"], "gates": st.get("gates") or [],
                      "pending": bool(pend), "gate_due": st.get("gate_due")},
                     default=_jsonable))
    return EXIT_OK


def cmd_collect_task(args: argparse.Namespace) -> int:
    """`collect-task --task FILE`: the collection subprocess (the engine venv)."""
    import rs_checkpoint
    try:
        summary = collect_from_task(read_json(args.task))
    except (rs_checkpoint.VocabMismatch, rs_engine.EngineError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return EXIT_FAILED
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return EXIT_ARGS
    print(f"{summary['matches']} matches, {summary['rows']} rows in {len(summary['files'])} "
          f"file(s), {summary['wall_s']:.0f} s ({summary['matches_per_hour']} matches/hour), "
          f"{len(summary['errors'])} error(s)", flush=True)
    print(json.dumps(_printed(summary), default=_jsonable), flush=True)
    return EXIT_OK if summary["matches"] else EXIT_FAILED


def main(argv: Sequence[str] | None = None, *, steps: Steps | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.command == "collect-task":
        return cmd_collect_task(args)
    if args.command == "status":
        return cmd_status(args)
    return cmd_run(args, argv, steps)


if __name__ == "__main__":
    sys.exit(main())
