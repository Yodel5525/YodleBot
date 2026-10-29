r"""Check that this PC is ready: Python, torch and CUDA, RoyaleSim, pins, the emulator, friendly-only.

Run it once with each venv's interpreter, from the repo root:

    .venvs\buildabot\Scripts\python.exe scripts\selfcheck.py            GPU venv, and the emulator
    .venvs\royalegym\Scripts\python.exe scripts\selfcheck.py --offline  engine venv, no device

Each check prints PASS, WARN, FAIL, SKIP or INFO:

  python     Python 3.12 (pyproject.toml requires >=3.12,<3.13).
  friendly   hastycr.machine.FRIENDLY_ONLY is True and refuse_queue() raises.
  machine    machine.json (if there is one) parses; any warnings from hastycr.machine.
  role       Which venv this is. "train" is .venvs/buildabot and "engine" is
             .venvs/royalegym (paths from hastycr.machine), taken from the interpreter's
             location; --role overrides it. Outside both venvs the role is guessed (WARN).
  torch      Version and build.
             train: needs a CUDA build, CUDA available, bf16 support, a CUDA torchvision,
             and a small bf16 matmul on the GPU; otherwise FAIL, because training would
             silently run on the CPU.
             engine: a CUDA build is a WARN, since every worker process would load the
             CUDA libraries.
  packages   Every pin in requirements/engine.txt (engine) or requirements/gpu.txt (train)
             is installed at that version. Missing is a FAIL; another version is a WARN.
  royalesim  Imports at the pinned version (otherwise FAIL). Reports the default Battle's
             card_table(), card_level(), tower_level() and number of unit types.
  hastycr    The repo is installed in this venv (pip install --no-deps -e .); otherwise WARN.
  adb        The adb from hastycr.machine (or --adb) reaches the serial (or --serial).
  screen     wm size is 1080x1920 portrait with no override. Another 9:16 portrait
             physical size is a WARN; anything else, or an override, is a FAIL.
  density    wm density is 480, the DPI this PC was set up with (otherwise WARN).
  game       Clash Royale is installed, and its versionCode. A build other than
             RECORDED_GAME_BUILD (docs/SPARRING_PLAN.md) is a WARN: the game has updated.
  emulator   scripts/emulator.verify(): the serial is the Clash Royale instance.

--offline skips the last five (the device checks). They only read: adb get-state, wm size,
wm density, dumpsys package and pm list packages. When a host:port serial is not attached,
"adb connect" is tried once. Nothing starts the emulator, and nothing taps.

Exit code: 0 when nothing failed (warnings are allowed; --strict counts them as failures),
1 when a check failed, 2 for bad arguments.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)                    # hastycr, also without `pip install -e .`

from hastycr import machine  # noqa: E402

ROLES = ("engine", "train")
REQUIREMENTS = {"engine": ROOT / "requirements" / "engine.txt",
                "train": ROOT / "requirements" / "gpu.txt"}
VENV_NAME = {"engine": ".venvs/royalegym", "train": ".venvs/buildabot"}

SCREEN = (1080, 1920)                           # every screen position assumes this
DENSITY = 480                                   # what this PC runs (docs/SPARRING_PLAN.md)
GAME_PACKAGE = "com.supercell.clashroyale"
RECORDED_GAME_BUILD = 160635001                 # versionCode measured on the PC, 2026-10-29
ADB_TIMEOUT = 20.0                              # seconds, per adb call

PASS, WARN, FAIL, SKIP, INFO = "PASS", "WARN", "FAIL", "SKIP", "INFO"
DEVICE_CHECKS = ("adb", "screen", "density", "game", "emulator")


@dataclass
class Check:
    """One result line. `data` holds the measured values (shown with --json)."""

    name: str
    status: str
    detail: str
    data: dict = field(default_factory=dict)

    def line(self) -> str:
        first, *rest = self.detail.splitlines() or [""]
        out = f"{self.status:<4}  {self.name:<9}  {first}"
        return "\n".join([out] + [" " * 17 + more for more in rest])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)\s*(?:#.*)?$")


def normalize(name: str) -> str:
    """PEP 503 name: 'opencv_python' and 'OpenCV-Python' are both 'opencv-python'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: Path) -> dict[str, str]:
    """{normalized name: version} for each name==version line of a requirements file."""
    pins: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN.match(line)
        if match:
            pins[normalize(match.group(1))] = match.group(2)
    return pins


def base_version(version: str) -> str:
    """'2.14.1+cu130' -> '2.14.1' (the local build label does not change the release)."""
    return str(version).split("+", 1)[0].strip()


def installed_version(name: str) -> str | None:
    """The installed distribution's version, or None when it is not installed."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _import(name: str):
    return importlib.import_module(name)


def _same_dir(a: str | Path, b: str | Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _same_file_name(a: str | Path, b: str | Path) -> bool:
    # No symlink resolution: a POSIX venv's bin/python links to the base interpreter.
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


# ---------------------------------------------------------------------------
# environment checks
# ---------------------------------------------------------------------------

def check_python(version_info: Sequence[int] | None = None) -> Check:
    vi = tuple(sys.version_info[:3] if version_info is None else version_info[:3])
    text = ".".join(str(n) for n in vi)
    if vi[:2] == (3, 12):
        return Check("python", PASS, f"Python {text} ({sys.executable})", {"version": text})
    return Check("python", FAIL,
                 f"Python {text}: this repo needs 3.12 (pyproject.toml). scripts/setup_windows.ps1 "
                 "makes both venvs with `py -3.12`.", {"version": text})


def check_friendly() -> Check:
    if machine.FRIENDLY_ONLY is not True:
        return Check("friendly", FAIL, "hastycr.machine.FRIENDLY_ONLY is not True")
    try:
        machine.refuse_queue("selfcheck")
    except machine.FriendlyOnlyError:
        return Check("friendly", PASS,
                     "FRIENDLY_ONLY: nothing presses Battle or queues a matchmade mode. "
                     "Start each friendly battle yourself.")
    except Exception as exc:
        return Check("friendly", FAIL, f"refuse_queue() raised {type(exc).__name__}, "
                                       "not FriendlyOnlyError")
    return Check("friendly", FAIL, "refuse_queue() returned instead of raising")


def load_machine() -> tuple[machine.Machine | None, Check]:
    """hastycr.machine's settings, read fresh (environment, machine.json, defaults)."""
    try:
        m = machine.load()
    except Exception as exc:                    # MachineConfigError, or worse
        return None, Check("machine", FAIL, f"machine settings cannot be read: {exc}")
    state = "found" if m.config_found else "not found (environment variables and defaults)"
    detail = (f"{m.config_path.name} {state}; adb {m.adb} [{m.sources.get('adb', '?')}], "
              f"serial {m.serial} [{m.sources.get('serial', '?')}]")
    data = {"settings": m.values(), "sources": dict(m.sources), "warnings": list(m.warnings)}
    if m.warnings:
        return m, Check("machine", WARN, "\n".join([detail] + list(m.warnings)), data)
    return m, Check("machine", PASS, detail, data)


def role_of_interpreter(m: machine.Machine, executable: str | None = None,
                        prefix: str | None = None) -> str | None:
    """'train' or 'engine' when this interpreter is that venv's, else None."""
    executable = sys.executable if executable is None else executable
    prefix = sys.prefix if prefix is None else prefix
    for role, python in (("train", m.py_train), ("engine", m.py_engine)):
        if _same_file_name(executable, python) or _same_dir(prefix, Path(python).parent.parent):
            return role
    return None


def resolve_role(choice: str, m: machine.Machine | None) -> tuple[str, Check]:
    if choice in ROLES:
        return choice, Check("role", INFO, f"{choice} (--role); pins from "
                                           f"requirements/{REQUIREMENTS[choice].name}")
    if m is not None:
        role = role_of_interpreter(m)
        if role:
            return role, Check("role", INFO, f"{role}: this interpreter is {VENV_NAME[role]}; "
                                             f"pins from requirements/{REQUIREMENTS[role].name}")
    try:
        cuda = getattr(getattr(_import("torch"), "version", None), "cuda", None)
    except Exception:
        cuda = None
    role = "train" if cuda else "engine"
    return role, Check("role", WARN,
                       f"{role} (guessed from the torch build): {sys.prefix} is neither "
                       f"{VENV_NAME['engine']} nor {VENV_NAME['train']}. Run this with a venv's "
                       "python, or pass --role engine|train.")


def cuda_smoke(torch) -> str | None:
    """None when a small bf16 matmul runs on the GPU and is right, else what went wrong."""
    try:
        x = torch.ones((8, 8), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = x @ x
        total = float(y.float().sum().item())
        torch.cuda.synchronize()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if total != 512.0:
        return f"ones(8, 8) @ ones(8, 8) summed to {total}, expected 512"
    return None


def check_torch(role: str) -> Check:
    try:
        torch = _import("torch")
    except Exception as exc:
        return Check("torch", FAIL, f"torch does not import ({type(exc).__name__}: {exc}). "
                                    "Run scripts/setup_windows.ps1.")
    version = str(getattr(torch, "__version__", "?"))
    cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
    data: dict = {"version": version, "cuda_build": cuda_build}
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False
    data["cuda_available"] = available
    summary = f"torch {version}, CUDA build {cuda_build or 'none (CPU)'}, CUDA available {available}"
    if available:
        try:
            data["device"] = str(torch.cuda.get_device_name(0))
        except Exception as exc:
            data["device"] = f"? ({type(exc).__name__})"
        try:
            data["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        except Exception:
            pass
        try:
            data["bf16"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            data["bf16"] = False
        summary += f", {data['device']}, bf16 {data['bf16']}"

    if role == "engine":
        if cuda_build:
            return Check("torch", WARN, summary + ". A CUDA torch in the engine venv makes "
                         "every worker process load the CUDA libraries (memory). "
                         "setup_windows.ps1 installs the CPU build here.", data)
        return Check("torch", PASS, summary + " (the CPU build, as intended here)", data)

    problems = []
    if not cuda_build:
        problems.append("this is a CPU-only torch, so training would silently run on the CPU. "
                        "Re-run: scripts\\setup_windows.ps1 -Only buildabot")
    elif not available:
        problems.append(f"CUDA is not available. Update the NVIDIA driver (nvidia-smi must show "
                        f"CUDA Version >= {cuda_build}), or install torch from an index your "
                        "driver supports (setup_windows.ps1 -GpuIndexUrl ...)")
    else:
        if not data.get("bf16"):
            problems.append(f"bf16 is not supported on {data['device']}; train_tokens trains "
                            "under bf16 autocast")
        error = cuda_smoke(torch)
        if error:
            problems.append(f"a bf16 matmul on the GPU failed ({error})")
    vision = installed_version("torchvision")
    data["torchvision"] = vision
    if vision is None:
        problems.append("torchvision is not installed (YOLO needs it)")
    elif "+cu" in version and "+cu" not in vision:
        problems.append(f"torchvision {vision} is a CPU build next to CUDA torch {version}; "
                        "YOLO's NMS on GPU tensors fails. setup_windows.ps1 reinstalls it")
    if problems:
        return Check("torch", FAIL, "\n".join([summary] + problems), data)
    return Check("torch", PASS, f"{summary}, torchvision {vision}", data)


def check_packages(role: str, pins: dict[str, str]) -> Check:
    source = f"requirements/{REQUIREMENTS[role].name}"
    missing, other, matched = [], [], []
    for name, want in pins.items():
        have = installed_version(name)
        if have is None:
            missing.append(f"{name}=={want}")
        elif base_version(have) != base_version(want):
            other.append(f"{name} {have} (pinned {want})")
        else:
            matched.append(name)
    data = {"source": source, "missing": missing, "other_version": other, "matched": matched}
    if missing:
        lines = [f"missing from {source}: {', '.join(missing)}. Run scripts/setup_windows.ps1."]
        lines += [f"not at the pinned version: {', '.join(other)}"] if other else []
        return Check("packages", FAIL, "\n".join(lines), data)
    if other:
        return Check("packages", WARN,
                     f"not at the pinned version in {source}: {', '.join(other)}. Re-run "
                     "scripts/setup_windows.ps1, or change the pin if this version is intended.",
                     data)
    return Check("packages", PASS, f"all {len(matched)} pins in {source} are installed", data)


def check_royalesim(pin: str | None) -> Check:
    try:
        rs = _import("royalesim")
    except Exception as exc:
        return Check("royalesim", FAIL, f"royalesim does not import ({type(exc).__name__}: {exc}). "
                                        "Run scripts/setup_windows.ps1.")
    version = str(getattr(rs, "__version__", "?"))
    data: dict = {"version": version, "pin": pin}
    problems = []
    if pin and base_version(version) != base_version(pin):
        problems.append(f"royalesim {version} is installed but the pin is {pin}. Rows, vocab ids "
                        "and checkpoints are tied to the engine version. Re-run "
                        "scripts/setup_windows.ps1.")
    try:
        battle = rs.Battle(card_names=None, slot_of_k=[[0, 1, 2], [0, 1, 2]])
        data["card_table"] = str(battle.card_table()) if hasattr(battle, "card_table") else None
        if hasattr(battle, "card_table_fnv1a64"):
            data["card_table_fnv1a64"] = battle.card_table_fnv1a64()
        data["card_level"] = int(battle.card_level())
        data["tower_level"] = int(battle.tower_level())
        data["unit_types"] = len(json.loads(battle.unit_types_json()))
    except Exception as exc:
        problems.append(f"a default Battle could not be built or read ({type(exc).__name__}: {exc})")
    if hasattr(rs, "provenance"):
        try:
            data["provenance"] = list(rs.provenance())
        except Exception:
            pass
    detail = f"royalesim {version}"
    if "unit_types" in data:
        detail += (f": card table {data['card_table'] or 'n/a (before 0.1.42)'}, card level "
                   f"{data['card_level']}, tower level {data['tower_level']}, "
                   f"{data['unit_types']} unit types")
    if problems:
        return Check("royalesim", FAIL, "\n".join([detail] + problems), data)
    return Check("royalesim", PASS, detail, data)


def check_hastycr() -> Check:
    have = installed_version("hastycr")
    where = Path(machine.__file__).resolve().parent
    if have is None:
        return Check("hastycr", WARN, f"the repo is not installed in this venv (using {where}). "
                                      "Scripts still find src/, but `python -m hastycr.machine` "
                                      "needs: python -m pip install --no-deps -e .")
    return Check("hastycr", PASS, f"hastycr {have} ({where})", {"version": have})


# ---------------------------------------------------------------------------
# device checks
# ---------------------------------------------------------------------------

def _run(cmd: Sequence[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def adb_call(adb: Path, serial: str | None, *args: str,
             timeout: float = ADB_TIMEOUT) -> tuple[int, str, str]:
    """(returncode, stdout, stderr); a timeout is 124 and a missing program 127."""
    cmd = [str(adb)] + (["-s", serial] if serial else []) + list(args)
    try:
        out = _run(cmd, timeout)
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout:.0f} s: {' '.join(cmd)}"
    except OSError as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return out.returncode, out.stdout or "", out.stderr or ""


_SIZE = re.compile(r"(Physical|Override) size:\s*(\d+)\s*x\s*(\d+)", re.IGNORECASE)
_DENSITY = re.compile(r"(Physical|Override) density:\s*(\d+)", re.IGNORECASE)


def parse_wm_size(text: str) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """`wm size` output -> (physical, override); either may be None."""
    physical = override = None
    for kind, w, h in _SIZE.findall(text):
        if kind.lower() == "physical":
            physical = (int(w), int(h))
        else:
            override = (int(w), int(h))
    return physical, override


def parse_wm_density(text: str) -> tuple[int | None, int | None]:
    """`wm density` output -> (physical, override); either may be None."""
    physical = override = None
    for kind, value in _DENSITY.findall(text):
        if kind.lower() == "physical":
            physical = int(value)
        else:
            override = int(value)
    return physical, override


def parse_package_version(text: str, package: str = GAME_PACKAGE) -> tuple[int | None, str | None]:
    """(versionCode, versionName) from `dumpsys package <package>`, read only inside that
    package's own section; (None, None) when the package is not there."""
    start = text.find(f"Package [{package}]")
    if start < 0:
        return None, None
    section = text[start:]
    nxt = section.find("Package [", 1)
    section = section if nxt < 0 else section[:nxt]
    code = re.search(r"versionCode=(\d+)", section)
    name = re.search(r"versionName=(\S+)", section)
    return (int(code.group(1)) if code else None), (name.group(1) if name else None)


def _size(wh: tuple[int, int] | None) -> str:
    return "-" if wh is None else f"{wh[0]}x{wh[1]}"


def _is_9_16_portrait(w: int, h: int) -> bool:
    return 0 < w < h and abs((w / h) / (9 / 16) - 1.0) <= 0.01


def judge_screen(physical, override, reset_hint: str = "adb shell wm size reset") -> Check:
    effective = override or physical
    data = {"physical": _size(physical), "override": _size(override) if override else None}
    if effective is None:
        return Check("screen", FAIL, "could not read `wm size`", data)
    w, h = effective
    note = f"physical {_size(physical)}" + (f", override {_size(override)}" if override else "")
    if (w, h) == SCREEN:
        return Check("screen", PASS, f"{w}x{h} portrait ({note})", data)
    if override:
        return Check("screen", FAIL,
                     f"a size override of {w}x{h} is active ({note}). Every screen position "
                     f"assumes {_size(SCREEN)}. Undo it with: {reset_hint}", data)
    if _is_9_16_portrait(w, h):
        return Check("screen", WARN,
                     f"{w}x{h} is 9:16 portrait, but not {_size(SCREEN)}. Positions scale, but "
                     f"the screen readers were measured at {_size(SCREEN)}. Set MuMu Settings > "
                     "Display to 1080x1920.", data)
    return Check("screen", FAIL,
                 f"{w}x{h} is not 9:16 portrait, and every screen position assumes "
                 f"{_size(SCREEN)} portrait. Set MuMu Settings > Display to Phone, 1080x1920, "
                 "portrait.", data)


def judge_density(physical: int | None, override: int | None) -> Check:
    effective = override or physical
    data = {"physical": physical, "override": override}
    if effective is None:
        return Check("density", WARN, "could not read `wm density`", data)
    note = f" (override of physical {physical})" if override else ""
    if effective == DENSITY:
        return Check("density", PASS, f"{effective} dpi{note}", data)
    return Check("density", WARN,
                 f"{effective} dpi{note}. This PC was set up at {DENSITY} dpi, where the screen "
                 f"positions were checked. Set DPI {DENSITY} in MuMu Settings > Display.", data)


def judge_game(code: int | None, name: str | None) -> Check:
    data = {"version_code": code, "version_name": name, "recorded": RECORDED_GAME_BUILD}
    if code is None:
        return Check("game", FAIL, f"Clash Royale ({GAME_PACKAGE}) is not installed, or dumpsys "
                                   "printed no versionCode for it", data)
    if code == RECORDED_GAME_BUILD:
        return Check("game", PASS, f"Clash Royale versionCode {code} ({name}), the build recorded "
                                   "in docs/SPARRING_PLAN.md", data)
    return Check("game", WARN,
                 f"Clash Royale versionCode {code} ({name}); docs/SPARRING_PLAN.md recorded "
                 f"{RECORDED_GAME_BUILD}. The game has updated, so card stats and screen "
                 "positions may have moved: re-check the geometry and re-run the fidelity grid, "
                 "then update RECORDED_GAME_BUILD in scripts/selfcheck.py.", data)


def emulator_verify(serial: str, adb: Path) -> None:
    """scripts/emulator.verify(serial, adb). It raises SystemExit with the reason when it
    refuses the device."""
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    _import("emulator").verify(serial, adb)


def check_emulator(serial: str, adb: Path) -> Check:
    try:
        emulator_verify(serial, adb)
    except SystemExit as exc:
        return Check("emulator", FAIL, str(exc.code if exc.code is not None else exc))
    except Exception as exc:
        return Check("emulator", FAIL, f"emulator.verify() failed ({type(exc).__name__}: {exc})")
    return Check("emulator", PASS, f"{serial} is the Clash Royale instance (emulator.verify)")


def _skip_rest(names: Sequence[str], why: str) -> list[Check]:
    return [Check(name, SKIP, why) for name in names]


def check_device(m: machine.Machine | None, adb_arg: str | None,
                 serial_arg: str | None) -> list[Check]:
    if m is None and not (adb_arg and serial_arg):
        return _skip_rest(DEVICE_CHECKS, "machine settings unreadable (see machine); fix them or "
                                         "pass --adb and --serial")
    adb = Path(adb_arg) if adb_arg else m.adb
    serial = serial_arg or m.serial
    source = "--adb" if adb_arg else (m.sources.get("adb", "?") if m else "?")
    if not (adb.is_file() or shutil.which(str(adb))):
        return [Check("adb", FAIL, f"adb not found: {adb} [{source}]. Set HASTYCR_ADB (or \"adb\" "
                                   "in machine.json) to MuMu's adb.exe; `python -m hastycr.machine`"
                                   " shows where it looks.")] + \
            _skip_rest(DEVICE_CHECKS[1:], "adb not found")

    rc, out, err = adb_call(adb, serial, "get-state")
    how = ""
    if out.strip() != "device" and ":" in serial:
        adb_call(adb, None, "connect", serial)         # MuMu's port needs attaching once
        rc, out, err = adb_call(adb, serial, "get-state")
        how = " (after adb connect)"
    if out.strip() != "device":
        said = (err or out).strip().splitlines()
        return [Check("adb", FAIL,
                      f"cannot reach {serial} with {adb}: {said[-1] if said else 'no answer'}\n"
                      f"Start the MuMu instance yourself, then: \"{adb}\" connect {serial}\n"
                      "Its port: MuMuManager.exe adb -v <index> (instance i is on "
                      "16384 + 32 * i).", {"serial": serial, "adb": str(adb)})] + \
            _skip_rest(DEVICE_CHECKS[1:], "adb cannot reach the device")
    checks = [Check("adb", PASS, f"{serial} is attached{how} ({adb} [{source}])",
                    {"serial": serial, "adb": str(adb)})]

    _, out, _ = adb_call(adb, serial, "shell", "wm", "size")
    checks.append(judge_screen(*parse_wm_size(out),
                               reset_hint=f"\"{adb}\" -s {serial} shell wm size reset"))
    _, out, _ = adb_call(adb, serial, "shell", "wm", "density")
    checks.append(judge_density(*parse_wm_density(out)))
    _, out, _ = adb_call(adb, serial, "shell", "dumpsys", "package", GAME_PACKAGE)
    checks.append(judge_game(*parse_package_version(out)))
    checks.append(check_emulator(serial, adb))
    return checks


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def run_checks(args: argparse.Namespace) -> list[Check]:
    checks = [check_python(), check_friendly()]
    m, machine_check = load_machine()
    checks.append(machine_check)
    role, role_check = resolve_role(args.role, m)
    checks.append(role_check)
    try:
        pins = read_pins(REQUIREMENTS[role])
    except OSError as exc:
        pins = {}
        checks.append(Check("packages", FAIL, f"cannot read {REQUIREMENTS[role]}: {exc}"))
    checks.append(check_torch(role))
    if pins:
        checks.append(check_packages(role, pins))
    checks += [check_royalesim(pins.get("royalesim")), check_hastycr()]
    if args.offline:
        checks += _skip_rest(DEVICE_CHECKS, "--offline")
    else:
        checks += check_device(m, args.adb, args.serial)
    return checks


def exit_code(checks: Sequence[Check], strict: bool = False) -> int:
    bad = {FAIL, WARN} if strict else {FAIL}
    return 1 if any(c.status in bad for c in checks) else 0


def report(checks: Sequence[Check], code: int) -> str:
    counts = {s: sum(c.status == s for c in checks) for s in (PASS, WARN, FAIL, SKIP)}
    lines = [f"HastyCR selfcheck ({sys.executable})"] + [c.line() for c in checks]
    lines.append(f"{counts[PASS]} passed, {counts[WARN]} warnings, {counts[FAIL]} failed, "
                 f"{counts[SKIP]} skipped: {'READY' if code == 0 else 'NOT READY'}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="selfcheck.py",
        description="Check this PC and venv: Python 3.12, torch/CUDA/bf16, RoyaleSim and the "
                    "pinned packages, the emulator (adb, 1080x1920, density, Clash Royale build, "
                    "emulator.verify) and FRIENDLY_ONLY. Exit 0 ready, 1 a check failed.")
    ap.add_argument("--offline", action="store_true",
                    help="skip the device checks (adb, screen size, density, game build, "
                         "emulator.verify)")
    ap.add_argument("--role", choices=("auto",) + ROLES, default="auto",
                    help="which venv's expectations to apply: train (.venvs/buildabot: CUDA "
                         "required, requirements/gpu.txt) or engine (.venvs/royalegym: CPU torch, "
                         "requirements/engine.txt). auto (default) goes by the interpreter's path")
    ap.add_argument("--adb", help="adb executable (default: hastycr.machine, i.e. HASTYCR_ADB, "
                                  "machine.json, PATH, then MuMu's install folders)")
    ap.add_argument("--serial", help="adb serial (default: hastycr.machine, i.e. HASTYCR_SERIAL, "
                                     "machine.json, then 127.0.0.1:16384)")
    ap.add_argument("--strict", action="store_true", help="count warnings as failures")
    ap.add_argument("--json", action="store_true", help="print the results as JSON")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")    # a cp1252 console must not crash a report
    except (AttributeError, ValueError):
        pass
    checks = run_checks(args)
    code = exit_code(checks, args.strict)
    if args.json:
        print(json.dumps({"exit_code": code, "python": sys.executable,
                          "checks": [{"name": c.name, "status": c.status, "detail": c.detail,
                                      "data": c.data} for c in checks]},
                         indent=2, default=str))
    else:
        print(report(checks, code))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
