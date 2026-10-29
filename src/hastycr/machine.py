"""Machine settings: which adb, which emulator, which Python environments, where data lives.

The same tree runs on more than one PC, so nothing machine-specific belongs in the
scripts. Each setting is resolved, first match wins, from:

  1. an environment variable   HASTYCR_ADB, HASTYCR_SERIAL, ... (table below)
  2. machine.json              at the repo root (gitignored; start from
                               machine.example.json), or the file HASTYCR_MACHINE names
  3. a default                 for adb: `adb` on PATH, then MuMu's install folders

  setting       variable              default
  adb           HASTYCR_ADB           adb on PATH, else the newest MuMu shell\\adb.exe
                                      under Program Files\\Netease (MUMU_ADB_GLOBS)
  serial        HASTYCR_SERIAL        127.0.0.1:16384, MuMu instance 0 (instance i
                                      answers on 16384 + 32 * i: mumu_serial)
  instance      HASTYCR_INSTANCE      "Android Device", the MuMu instance's name (messages)
  py_engine     HASTYCR_PY_ENGINE     .venvs/royalegym/Scripts/python.exe (bin/python off Windows)
  py_train      HASTYCR_PY_TRAIN      .venvs/buildabot/Scripts/python.exe (bin/python off Windows)
  data          HASTYCR_DATA          data/
  runs          HASTYCR_RUNS          runs/
  mumu_manager  HASTYCR_MUMU_MANAGER  MuMuManager.exe in MuMu's install folders, if found

Relative paths, from either source, are taken from the repo root; a bare command
name for adb ("adb") is looked up on PATH. Surrounding quotes are stripped, and
%VARS% / ~ are expanded. In machine.json a key starting with "_" is a comment and a
null value means "not set".

MuMu ships its own adb. A second adb of another version on PATH restarts the adb
server whenever the two take turns, so if PATH finds one while MuMu has its own,
the result carries a warning; set HASTYCR_ADB to MuMu's to silence it.

From the command line (stdlib only, so it also runs as `python src/hastycr/machine.py`):

    python -m hastycr.machine              every value and where it came from
    python -m hastycr.machine --get adb    one value and nothing else (for .ps1 / .cmd)
    python -m hastycr.machine --json       all of it as JSON

FRIENDLY_ONLY is not a setting. The bot plays friendly battles that a person starts.
Nothing in this repo presses the lobby Battle button or queues ladder or any other
matchmade mode, and there is no option that would: code asked to queue calls
refuse_queue(), which raises FriendlyOnlyError.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, NoReturn

# ---------------------------------------------------------------------------
# friendly battles only
# ---------------------------------------------------------------------------

FRIENDLY_ONLY: Final = True
FRIENDLY_ONLY_MESSAGE: Final = (
    "Friendly battles only: this bot never presses Battle and never queues ladder or any "
    "other matchmade mode. Start a friendly battle yourself (clan chat or a friend's "
    "profile); the bot plays the battle that is on screen.")


class FriendlyOnlyError(RuntimeError):
    """Something was asked to queue a matchmade battle. There is no way to allow it."""


def refuse_queue(what: str = "Queueing a battle") -> NoReturn:
    """Raise FriendlyOnlyError; every former queueing path calls this instead."""
    raise FriendlyOnlyError(f"{what} is disabled. {FRIENDLY_ONLY_MESSAGE}")


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    here = Path(__file__).resolve().parents[2]          # src/hastycr/machine.py -> repo
    return here if (here / "pyproject.toml").is_file() else Path.cwd()


ROOT: Final = _repo_root()
CONFIG_NAME: Final = "machine.json"
CONFIG_ENV: Final = "HASTYCR_MACHINE"

MUMU_BASE_PORT: Final = 16384
MUMU_PORT_STEP: Final = 32


def mumu_serial(index: int) -> str:
    """adb serial of MuMu instance `index` (what `MuMuManager.exe adb -v <index>` reports)."""
    if index < 0:
        raise ValueError(f"MuMu instance index must be >= 0, got {index}")
    return f"127.0.0.1:{MUMU_BASE_PORT + MUMU_PORT_STEP * index}"


DEFAULT_SERIAL: Final = mumu_serial(0)
DEFAULT_INSTANCE: Final = "Android Device"

#: setting -> environment variable. Also the list of settings machine.json may hold.
ENV: Final[Mapping[str, str]] = {
    "adb": "HASTYCR_ADB",
    "serial": "HASTYCR_SERIAL",
    "instance": "HASTYCR_INSTANCE",
    "py_engine": "HASTYCR_PY_ENGINE",
    "py_train": "HASTYCR_PY_TRAIN",
    "data": "HASTYCR_DATA",
    "runs": "HASTYCR_RUNS",
    "mumu_manager": "HASTYCR_MUMU_MANAGER",
}

#: Where MuMu installs adb, relative to a Program Files folder. In order; inside one
#: pattern the highest Android image version (nx_device\<version>) comes first.
MUMU_ADB_GLOBS: Final = (
    "Netease/MuMuPlayer/nx_device/*/shell/adb.exe",
    "Netease/MuMu Player 12/nx_device/*/shell/adb.exe",
    "Netease/MuMuPlayer-12.0/nx_device/*/shell/adb.exe",
    "Netease/MuMuPlayer-12.0/shell/adb.exe",
    "Netease/MuMu Player 12/shell/adb.exe",
    "Netease/MuMuPlayer/shell/adb.exe",
)
MUMU_MANAGER_GLOBS: Final = (
    "Netease/MuMuPlayer/nx_main/MuMuManager.exe",
    "Netease/MuMu Player 12/nx_main/MuMuManager.exe",
    "Netease/MuMuPlayer-12.0/shell/MuMuManager.exe",
    "Netease/MuMu Player 12/shell/MuMuManager.exe",
)


class MachineConfigError(ValueError):
    """machine.json cannot be used as it is (not JSON, not an object, a non-string value)."""


@dataclass(frozen=True)
class Machine:
    """The resolved settings. `sources[name]` says where each came from."""

    adb: Path
    serial: str
    instance: str
    py_engine: Path
    py_train: Path
    data: Path
    runs: Path
    mumu_manager: Path | None
    root: Path
    config_path: Path
    config_found: bool
    sources: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def values(self) -> dict[str, str]:
        """Every setting as text ("" for a missing mumu_manager)."""
        out = {}
        for name in ENV:
            value = getattr(self, name)
            out[name] = "" if value is None else str(value)
        return out


# ---------------------------------------------------------------------------
# resolution helpers
# ---------------------------------------------------------------------------

_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def _expand(raw: str) -> str:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return os.path.expandvars(os.path.expanduser(text))


def _is_absolute(text: str) -> bool:
    # A Windows drive or UNC path counts on any OS, so a machine.json reads the same everywhere.
    return Path(text).is_absolute() or bool(_DRIVE.match(text)) or text.startswith("\\\\")


def _as_path(raw: str, root: Path) -> Path:
    text = _expand(raw)
    return Path(text) if _is_absolute(text) else root / text


def _as_command(raw: str, root: Path, which: Callable[[str], str | None]) -> Path:
    text = _expand(raw)
    if _is_absolute(text):
        return Path(text)
    if "/" not in text and "\\" not in text:            # a bare name: look it up on PATH
        found = which(text)
        return Path(found) if found else Path(text)
    return root / text


def venv_python(venv: Path, os_name: str = os.name) -> Path:
    """The interpreter of a virtual environment: Scripts/python.exe on Windows, bin/python
    elsewhere; whichever exists wins, so a venv made on the other OS still resolves."""
    windows, posix = venv / "Scripts" / "python.exe", venv / "bin" / "python"
    order = (windows, posix) if os_name == "nt" else (posix, windows)
    return next((p for p in order if p.exists()), order[0])


def install_roots(env: Mapping[str, str] | None = None, os_name: str = os.name) -> list[Path]:
    """Program Files folders to look for MuMu in (none off Windows)."""
    if os_name != "nt":
        return []
    env = os.environ if env is None else env
    names = (env.get("ProgramFiles"), env.get("ProgramW6432"), env.get("ProgramFiles(x86)"),
             r"C:\Program Files", r"D:\Program Files")
    roots, seen = [], set()
    for name in names:
        if name and os.path.normcase(name) not in seen:
            seen.add(os.path.normcase(name))
            roots.append(Path(name))
    return roots


def _version_key(path: Path) -> tuple[int, ...]:
    # .../nx_device/15.0/shell/adb.exe -> (15, 0); .../MuMuPlayer-12.0/shell/adb.exe -> (12, 0)
    return tuple(int(n) for n in re.findall(r"\d+", path.parent.parent.name))


def find_installed(globs: Sequence[str], roots: Iterable[Path]) -> list[Path]:
    """Existing files matching `globs` under `roots`: pattern order, newest version first."""
    roots = [Path(r) for r in roots]
    found, seen = [], set()
    for pattern in globs:
        for root in roots:
            try:
                hits = [p for p in root.glob(pattern) if p.is_file()]
            except OSError:
                continue
            for path in sorted(hits, key=_version_key, reverse=True):
                key = os.path.normcase(str(path))
                if key not in seen:
                    seen.add(key)
                    found.append(path)
    return found


def mumu_adb_candidates(roots: Iterable[Path] | None = None) -> list[Path]:
    """MuMu's own adb.exe files, best first."""
    return find_installed(MUMU_ADB_GLOBS, install_roots() if roots is None else roots)


def read_config(path: Path) -> tuple[dict[str, str], list[str]]:
    """machine.json as ({setting: text}, [unknown keys]); ({}, []) when there is no file."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}, []
    except OSError as exc:
        raise MachineConfigError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MachineConfigError(
            f"{path} is not valid JSON ({exc}). In JSON a backslash must be doubled "
            f"(\"C:\\\\Program Files\\\\...\"), or use forward slashes.") from exc
    if not isinstance(data, dict):
        raise MachineConfigError(f"{path} must hold one JSON object, got {type(data).__name__}")
    values, unknown = {}, []
    for key, value in data.items():
        if key.startswith("_"):
            continue                                    # comments
        if key not in ENV:
            unknown.append(key)
            continue
        if value is None:
            continue                                    # null: use the default
        if not isinstance(value, str):
            raise MachineConfigError(f"{path}: {key!r} must be a string, got {value!r}")
        values[key] = value
    return values, unknown


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(str(a)) == os.path.normcase(str(b))


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def load(env: Mapping[str, str] | None = None, *, config_path: Path | str | None = None,
         root: Path | str | None = None, mumu_roots: Iterable[Path] | None = None,
         which: Callable[[str], str | None] = shutil.which) -> Machine:
    """Resolve every setting. All arguments exist for tests; the defaults read this machine:
    os.environ, <repo>/machine.json (or HASTYCR_MACHINE), Program Files and PATH."""
    env = os.environ if env is None else env
    root = ROOT if root is None else Path(root)
    if config_path is None:
        named = env.get(CONFIG_ENV, "").strip()
        config_path = _as_path(named, root) if named else root / CONFIG_NAME
    config_path = Path(config_path)
    config, unknown = read_config(config_path)
    config_found = config_path.is_file()
    roots = install_roots(env) if mumu_roots is None else [Path(r) for r in mumu_roots]
    warnings = [f"config: {config_path}: unknown setting {k!r} ignored "
                f"(known: {', '.join(ENV)})" for k in unknown]
    sources: dict[str, str] = {}

    def given(name: str) -> str | None:
        raw = (env.get(ENV[name]) or "").strip()
        if raw:
            sources[name] = f"env {ENV[name]}"
            return raw
        raw = (config.get(name) or "").strip()
        if raw:
            sources[name] = CONFIG_NAME if config_path.name == CONFIG_NAME else str(config_path)
            return raw
        sources[name] = "default"
        return None

    # adb: env -> machine.json -> PATH -> MuMu's install folders
    raw = given("adb")
    if raw:
        adb = _as_command(raw, root, which)
        if not adb.exists():
            warnings.append(f"adb: {adb} ({sources['adb']}) does not exist")
    else:
        on_path = which("adb")
        mumu = find_installed(MUMU_ADB_GLOBS, roots)
        if on_path:
            adb, sources["adb"] = Path(on_path), "PATH"
            if mumu and not _same_file(adb, mumu[0]):
                warnings.append(
                    f"adb: using {adb} from PATH, but MuMu has its own at {mumu[0]}. Two adb "
                    f"versions restart each other's server; set {ENV['adb']} to MuMu's.")
        elif mumu:
            adb, sources["adb"] = mumu[0], "MuMu install"
        else:
            adb, sources["adb"] = Path("adb"), "not found"
            warnings.append(f"adb: no adb on PATH and no MuMu install found; set {ENV['adb']} "
                            f"or \"adb\" in {CONFIG_NAME}")

    raw = given("serial")
    serial = _expand(raw) if raw else DEFAULT_SERIAL
    raw = given("instance")
    instance = _expand(raw) if raw else DEFAULT_INSTANCE

    raw = given("py_engine")
    py_engine = _as_path(raw, root) if raw else venv_python(root / ".venvs" / "royalegym")
    raw = given("py_train")
    py_train = _as_path(raw, root) if raw else venv_python(root / ".venvs" / "buildabot")
    raw = given("data")
    data = _as_path(raw, root) if raw else root / "data"
    raw = given("runs")
    runs = _as_path(raw, root) if raw else root / "runs"

    raw = given("mumu_manager")
    if raw:
        mumu_manager: Path | None = _as_path(raw, root)
    else:
        managers = find_installed(MUMU_MANAGER_GLOBS, roots)
        mumu_manager = managers[0] if managers else None
        sources["mumu_manager"] = "MuMu install" if managers else "not found"

    return Machine(adb=adb, serial=serial, instance=instance, py_engine=py_engine,
                   py_train=py_train, data=data, runs=runs, mumu_manager=mumu_manager,
                   root=root, config_path=config_path, config_found=config_found,
                   sources=dict(sources), warnings=tuple(warnings))


@functools.cache
def get() -> Machine:
    """This machine's settings (os.environ, machine.json), read once per process."""
    return load()


def reload() -> Machine:
    """Forget the cached settings and read them again."""
    get.cache_clear()
    return get()


def adb() -> Path:
    return get().adb


def serial() -> str:
    return get().serial


def instance() -> str:
    return get().instance


def py_engine() -> Path:
    """Interpreter of the engine venv (CPU torch, royalesim): rows, evaluation, rollouts."""
    return get().py_engine


def py_train() -> Path:
    """Interpreter of the GPU venv (CUDA torch, vision): training and live play."""
    return get().py_train


def data_root() -> Path:
    return get().data


def runs_root() -> Path:
    return get().runs


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def describe(m: Machine) -> str:
    state = "found" if m.config_found else "not found; copy machine.example.json to create it"
    lines = ["HastyCR machine settings", f"  config        {m.config_path} ({state})"]
    for name, value in m.values().items():
        lines.append(f"  {name:<13} {value or '-':<56} [{m.sources.get(name, '?')}]")
    lines.append(f"  friendly_only {FRIENDLY_ONLY} (not a setting: nothing queues a matchmade battle)")
    lines += [f"warning: {w}" for w in m.warnings]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m hastycr.machine",
        description="Show this machine's HastyCR settings: environment variables, then "
                    "machine.json, then defaults.")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--get", choices=list(ENV), metavar="NAME",
                     help="print only this setting's value, for scripts: "
                          + ", ".join(ENV) + ". Exit 1 if it has none")
    out.add_argument("--json", action="store_true",
                     help="print the values, their sources and any warnings as JSON")
    args = ap.parse_args(argv)
    try:
        m = load()
    except MachineConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.get:
        print(m.values()[args.get])
        for warning in m.warnings:
            if warning.startswith(f"{args.get}:"):
                print(f"warning: {warning}", file=sys.stderr)
        return 0 if getattr(m, args.get) is not None else 1
    if args.json:
        print(json.dumps({"settings": m.values(), "sources": dict(m.sources),
                          "config": str(m.config_path), "config_found": m.config_found,
                          "friendly_only": FRIENDLY_ONLY, "warnings": list(m.warnings)},
                         indent=2))
        return 0
    print(describe(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
