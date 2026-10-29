"""scripts/selfcheck.py (P04): the pass and fail paths, and the exit codes.

These tests need no torch, royalesim or device:
- torch and royalesim are stub modules in sys.modules;
- adb is a fake subprocess.run that answers from a table;
- installed versions come from a fake installed_version().

Machine settings go through the real hastycr.machine, using environment variables
and a temporary machine.json.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hastycr import machine

ROOT = Path(__file__).resolve().parents[2]
SELFCHECK = ROOT / "scripts" / "selfcheck.py"
SERIAL = "127.0.0.1:16384"
PACKAGE = "com.supercell.clashroyale"
ALL_CHECKS = {"python", "friendly", "machine", "role", "torch", "packages", "royalesim",
              "hastycr", "adb", "screen", "density", "game", "emulator"}

DUMPSYS = (
    "Activity Resolver Table:\n"
    "  versionCode=1 (another section; must not be read)\n"
    "Packages:\n"
    f"  Package [{PACKAGE}] (8c2f1a0):\n"
    "    userId=10123\n"
    "    versionCode=160635001 minSdk=24 targetSdk=35\n"
    "    versionName=16.635.1\n"
    "  Package [com.android.settings] (1b2c3d):\n"
    "    versionCode=35 minSdk=35 targetSdk=35\n"
)

HEALTHY_ADB = {
    ("get-state",): (0, "device\n", ""),
    ("shell", "wm", "size"): (0, "Physical size: 1080x1920\n", ""),
    ("shell", "wm", "density"): (0, "Physical density: 480\n", ""),
    ("shell", "dumpsys", "package", PACKAGE): (0, DUMPSYS, ""),
    ("shell", "pm", "list", "packages"):
        (0, f"package:com.android.settings\npackage:{PACKAGE}\n", ""),
}


@pytest.fixture(scope="module")
def sc():
    """scripts/selfcheck.py as a module (scripts/ is not a package)."""
    name = "_selfcheck_under_test"
    spec = importlib.util.spec_from_file_location(name, SELFCHECK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module              # dataclasses look the module up by name
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeAdb:
    """subprocess.run stand-in. Answers adb commands, keyed by the arguments after
    `adb [-s serial]`, from a table. An answer can be (rc, out, err), an exception to
    raise, or a callable that returns one of those. Every call is recorded."""

    def __init__(self, answers=None):
        self.answers = dict(HEALTHY_ADB if answers is None else answers)
        self.calls: list[tuple[list[str], dict]] = []
        self.seen: list[tuple[str, ...]] = []

    def __call__(self, cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        self.calls.append((cmd, kwargs))
        args = cmd[1:]
        if args[:1] == ["-s"]:
            args = args[2:]
        self.seen.append(tuple(args))
        answer = self.answers.get(tuple(args), (1, "", "error: unknown command"))
        if callable(answer):
            answer = answer()
        if isinstance(answer, BaseException):
            raise answer
        rc, out, err = answer
        return subprocess.CompletedProcess(cmd, rc, out, err)


def make_torch(version="2.14.1+cu130", cuda="13.0", available=True, bf16=True,
               device="NVIDIA GeForce RTX 4070 SUPER"):
    torch = types.ModuleType("torch")
    torch.__version__ = version
    torch.version = types.SimpleNamespace(cuda=cuda)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_name=lambda index=0: device,
        get_device_properties=lambda index=0: types.SimpleNamespace(total_memory=12 * 2**30),
        is_bf16_supported=lambda: bf16)
    return torch


def make_royalesim(version="0.1.42", broken=False):
    rs = types.ModuleType("royalesim")
    rs.__version__ = version

    class Battle:
        def __init__(self, card_names=None, slot_of_k=None, **kwargs):
            if broken:
                raise RuntimeError("engine exploded")
            self.slot_of_k = slot_of_k

        def card_table(self):
            return "cards-15535.1"

        def card_table_fnv1a64(self):
            return 0x1234

        def card_level(self):
            return 11

        def tower_level(self):
            return 11

        def unit_types_json(self):
            return json.dumps([{"id": i} for i in range(237)])

    rs.Battle = Battle
    rs.provenance = lambda: ("commit", "tree")
    return rs


@pytest.fixture
def world(sc, monkeypatch, tmp_path):
    """A ready PC for the train role: CUDA torch, royalesim 0.1.42, every pin installed,
    a reachable 1080x1920 device on the recorded build. Tests then break one thing."""
    for var in machine.ENV.values():
        monkeypatch.delenv(var, raising=False)
    adb_exe = tmp_path / "adb.exe"
    adb_exe.write_text("", encoding="utf-8")
    config = tmp_path / "machine.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HASTYCR_MACHINE", str(config))
    monkeypatch.setenv("HASTYCR_ADB", str(adb_exe))
    monkeypatch.setenv("HASTYCR_SERIAL", SERIAL)

    pinned = {**sc.read_pins(sc.REQUIREMENTS["engine"]), **sc.read_pins(sc.REQUIREMENTS["train"])}
    versions = dict(pinned, hastycr="0.1.0")
    versions["torch"] = pinned["torch"] + "+cu130"
    versions["torchvision"] = pinned["torchvision"] + "+cu130"
    monkeypatch.setattr(sc, "installed_version", lambda name: versions.get(sc.normalize(name)))

    monkeypatch.setitem(sys.modules, "torch", make_torch())
    monkeypatch.setitem(sys.modules, "royalesim", make_royalesim())
    monkeypatch.setattr(sc, "cuda_smoke", lambda torch: None)
    monkeypatch.setattr(sc, "check_python",
                        lambda version_info=None: sc.Check("python", sc.PASS, "3.12 (test)"))
    verified = []
    monkeypatch.setattr(sc, "emulator_verify", lambda serial, adb: verified.append(serial))
    adb = FakeAdb()
    monkeypatch.setattr(subprocess, "run", adb)
    return types.SimpleNamespace(adb=adb, versions=versions, adb_exe=adb_exe, config=config,
                                 verified=verified)


def run(sc, capsys, *argv):
    """main(--json ...) -> (exit code, {check name: check dict})."""
    code = sc.main(["--json", *argv])
    report = json.loads(capsys.readouterr().out)
    assert report["exit_code"] == code
    return code, {c["name"]: c for c in report["checks"]}


def statuses(checks):
    return {name: c["status"] for name, c in checks.items()}


# ---------------------------------------------------------------------------
# a ready PC
# ---------------------------------------------------------------------------

def test_ready_pc_passes_every_check_and_exits_0(sc, world, capsys):
    code, checks = run(sc, capsys, "--role", "train")
    assert code == 0, statuses(checks)
    assert set(checks) == ALL_CHECKS
    assert {n for n, s in statuses(checks).items() if s not in ("PASS", "INFO")} == set()
    assert checks["royalesim"]["data"]["card_table"] == "cards-15535.1"
    assert checks["royalesim"]["data"]["card_level"] == 11
    assert checks["royalesim"]["data"]["tower_level"] == 11
    assert checks["royalesim"]["data"]["unit_types"] == 237
    assert checks["game"]["data"]["version_code"] == 160635001
    assert checks["game"]["data"]["version_name"] == "16.635.1"
    assert checks["torch"]["data"]["bf16"] is True
    assert world.verified == [SERIAL]
    assert world.adb.calls, "the device checks ran adb"
    for cmd, kwargs in world.adb.calls:
        assert cmd[0] == str(world.adb_exe)
        assert kwargs.get("timeout"), f"adb call without a timeout: {cmd}"
    assert ("connect", SERIAL) not in world.adb.seen       # attached already


def test_offline_skips_the_device_and_never_runs_adb(sc, world, capsys):
    code = sc.main(["--offline", "--role", "train"])
    out = capsys.readouterr().out
    assert code == 0
    assert world.adb.calls == [] and world.verified == []
    assert "READY" in out and "NOT READY" not in out
    for name in ("adb", "screen", "density", "game", "emulator"):
        assert f"SKIP  {name}" in out


def test_engine_venv_with_cpu_torch_passes(sc, world, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", make_torch("2.14.1+cpu", cuda=None, available=False))
    code, checks = run(sc, capsys, "--offline", "--role", "engine")
    assert code == 0, statuses(checks)
    assert checks["torch"]["status"] == "PASS"
    assert "requirements/engine.txt" in checks["packages"]["detail"]


# ---------------------------------------------------------------------------
# torch
# ---------------------------------------------------------------------------

def test_cpu_torch_in_the_gpu_venv_fails(sc, world, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", make_torch("2.14.1+cpu", cuda=None, available=False))
    code, checks = run(sc, capsys, "--offline", "--role", "train")
    assert code == 1
    assert checks["torch"]["status"] == "FAIL"
    assert "CPU-only torch" in checks["torch"]["detail"]


@pytest.mark.parametrize("breakage, needle", [
    ("no_cuda", "CUDA is not available"),
    ("no_bf16", "bf16 is not supported"),
    ("smoke", "bf16 matmul on the GPU failed"),
    ("cpu_torchvision", "is a CPU build"),
    ("no_torchvision", "torchvision is not installed"),
])
def test_gpu_problems_fail_the_train_role(sc, world, monkeypatch, capsys, breakage, needle):
    if breakage == "no_cuda":
        monkeypatch.setitem(sys.modules, "torch", make_torch(available=False))
    elif breakage == "no_bf16":
        monkeypatch.setitem(sys.modules, "torch", make_torch(bf16=False))
    elif breakage == "smoke":
        monkeypatch.setattr(sc, "cuda_smoke", lambda torch: "RuntimeError: no kernel image")
    elif breakage == "cpu_torchvision":
        world.versions["torchvision"] = "0.29.1"
    else:
        del world.versions["torchvision"]
    code, checks = run(sc, capsys, "--offline", "--role", "train")
    assert code == 1
    assert checks["torch"]["status"] == "FAIL"
    assert needle in checks["torch"]["detail"]


def test_cuda_torch_in_the_engine_venv_is_a_warning(sc, world, capsys):
    code, checks = run(sc, capsys, "--offline", "--role", "engine")
    assert code == 0
    assert checks["torch"]["status"] == "WARN"
    assert sc.main(["--offline", "--role", "engine", "--strict"]) == 1
    capsys.readouterr()


def test_missing_torch_fails(sc, world, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)          # import torch -> ImportError
    code, checks = run(sc, capsys, "--offline", "--role", "engine")
    assert code == 1
    assert checks["torch"]["status"] == "FAIL"
    assert "does not import" in checks["torch"]["detail"]


def test_cuda_smoke_runs_a_bf16_matmul(sc):
    calls = []

    class Tensor:
        def __init__(self, value):
            self.value = value

        def __matmul__(self, other):
            calls.append("matmul")
            return Tensor(self.value * other.value * 8)

        def float(self):
            return self

        def sum(self):
            return Tensor(self.value * 64)

        def item(self):
            return self.value

    class Autocast:
        def __init__(self, device, dtype):
            calls.append(("autocast", device, dtype))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch = types.SimpleNamespace(
        ones=lambda shape, device: calls.append(("ones", shape, device)) or Tensor(1.0),
        autocast=Autocast, bfloat16="bf16",
        cuda=types.SimpleNamespace(synchronize=lambda: calls.append("sync")))
    assert sc.cuda_smoke(torch) is None
    assert calls == [("ones", (8, 8), "cuda"), ("autocast", "cuda", "bf16"), "matmul", "sync"]
    torch.ones = lambda shape, device: (_ for _ in ()).throw(RuntimeError("no kernel image"))
    assert "no kernel image" in sc.cuda_smoke(torch)


# ---------------------------------------------------------------------------
# royalesim and the pins
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module, needle", [
    (make_royalesim("0.1.22"), "the pin is 0.1.42"),
    (make_royalesim(broken=True), "could not be built"),
    (None, "does not import"),
])
def test_royalesim_problems_fail(sc, world, monkeypatch, capsys, module, needle):
    monkeypatch.setitem(sys.modules, "royalesim", module)
    code, checks = run(sc, capsys, "--offline", "--role", "engine")
    assert code == 1
    assert checks["royalesim"]["status"] == "FAIL"
    assert needle in checks["royalesim"]["detail"]


def test_a_missing_pin_fails_and_another_version_warns(sc, world, capsys):
    world.versions["numpy"] = "2.4.0"
    code, checks = run(sc, capsys, "--offline", "--role", "train")
    assert code == 0 and checks["packages"]["status"] == "WARN"
    assert "numpy 2.4.0 (pinned 2.5.4)" in checks["packages"]["detail"]
    assert sc.main(["--offline", "--role", "train", "--strict"]) == 1
    capsys.readouterr()
    del world.versions["pyarrow"]
    code, checks = run(sc, capsys, "--offline", "--role", "train")
    assert code == 1 and checks["packages"]["status"] == "FAIL"
    assert "pyarrow==26.0.0" in checks["packages"]["detail"]


def test_read_pins_and_base_version(sc, tmp_path):
    path = tmp_path / "r.txt"
    path.write_text("# comment\n\nTorch==2.14.1\nopencv_python == 4.14.0.94  # note\n",
                    encoding="utf-8")
    assert sc.read_pins(path) == {"torch": "2.14.1", "opencv-python": "4.14.0.94"}
    assert sc.base_version("2.14.1+cu130") == "2.14.1"
    assert sc.base_version("0.29.1") == "0.29.1"


def test_hastycr_not_installed_is_a_warning(sc, world, capsys):
    del world.versions["hastycr"]
    code, checks = run(sc, capsys, "--offline", "--role", "train")
    assert code == 0
    assert checks["hastycr"]["status"] == "WARN"
    assert "--no-deps -e ." in checks["hastycr"]["detail"]


# ---------------------------------------------------------------------------
# the device
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, status", [
    ("Physical size: 1080x1920\n", "PASS"),
    ("Physical size: 1440x2560\nOverride size: 1080x1920\n", "PASS"),
    ("Physical size: 1080x1920\nOverride size: 540x960\n", "FAIL"),
    ("Physical size: 720x1280\n", "WARN"),
    ("Physical size: 1920x1080\n", "FAIL"),
    ("Physical size: 1080x2400\n", "FAIL"),
    ("error: closed\n", "FAIL"),
])
def test_screen_size(sc, text, status):
    check = sc.judge_screen(*sc.parse_wm_size(text))
    assert check.status == status, check.detail
    if "Override size: 540x960" in text:
        assert "wm size reset" in check.detail


@pytest.mark.parametrize("text, status", [
    ("Physical density: 480\n", "PASS"),
    ("Physical density: 480\nOverride density: 320\n", "WARN"),
    ("Physical density: 320\n", "WARN"),
    ("", "WARN"),
])
def test_screen_density(sc, text, status):
    assert sc.judge_density(*sc.parse_wm_density(text)).status == status


def test_game_version_is_read_from_its_own_section(sc):
    assert sc.parse_package_version(DUMPSYS) == (160635001, "16.635.1")
    assert sc.judge_game(160635001, "16.635.1").status == "PASS"
    newer = sc.judge_game(160700001, "16.700.1")
    assert newer.status == "WARN" and "updated" in newer.detail
    missing = f"Unable to find package: {PACKAGE}\n  versionCode=5\n"
    assert sc.parse_package_version(missing) == (None, None)
    assert sc.judge_game(None, None).status == "FAIL"


def test_unreachable_device_fails_after_one_connect(sc, world, capsys):
    world.adb.answers[("get-state",)] = (1, "", f"error: device '{SERIAL}' not found")
    world.adb.answers[("connect", SERIAL)] = (0, f"failed to connect to '{SERIAL}'\n", "")
    code, checks = run(sc, capsys, "--role", "train")
    assert code == 1
    assert checks["adb"]["status"] == "FAIL"
    assert "connect" in checks["adb"]["detail"] and "not found" in checks["adb"]["detail"]
    for name in ("screen", "density", "game", "emulator"):
        assert checks[name]["status"] == "SKIP"
    assert world.adb.seen == [("get-state",), ("connect", SERIAL), ("get-state",)]
    assert world.verified == []


def test_device_attached_by_adb_connect_passes(sc, world, capsys):
    states = iter([(1, "", "error: device not found"), (0, "device\n", "")])
    world.adb.answers[("get-state",)] = lambda: next(states)
    world.adb.answers[("connect", SERIAL)] = (0, f"connected to {SERIAL}\n", "")
    code, checks = run(sc, capsys, "--role", "train")
    assert code == 0, statuses(checks)
    assert "after adb connect" in checks["adb"]["detail"]


def test_adb_timeout_is_a_failure_not_a_hang(sc, world, capsys):
    world.adb.answers[("get-state",)] = subprocess.TimeoutExpired(cmd="adb", timeout=20)
    code, checks = run(sc, capsys, "--role", "train")
    assert code == 1
    assert checks["adb"]["status"] == "FAIL"
    assert "timed out" in checks["adb"]["detail"]


def test_missing_adb_fails_without_running_anything(sc, world, monkeypatch, capsys):
    monkeypatch.setenv("HASTYCR_ADB", str(world.adb_exe.parent / "nowhere" / "adb.exe"))
    code, checks = run(sc, capsys, "--role", "train")
    assert code == 1
    assert checks["adb"]["status"] == "FAIL" and "adb not found" in checks["adb"]["detail"]
    assert checks["machine"]["status"] == "WARN"           # machine.py warns about it too
    assert world.adb.calls == []


def test_wrong_screen_size_fails(sc, world, capsys):
    world.adb.answers[("shell", "wm", "size")] = (
        0, "Physical size: 1080x1920\nOverride size: 540x960\n", "")
    code, checks = run(sc, capsys, "--role", "train")
    assert code == 1
    assert checks["screen"]["status"] == "FAIL"
    assert f"-s {SERIAL} shell wm size reset" in checks["screen"]["detail"]


def test_emulator_refusal_fails(sc, world, monkeypatch, capsys):
    def refuse(serial, adb):
        raise SystemExit(f"refusing to run: {serial} is the CLASH OF CLANS instance.")
    monkeypatch.setattr(sc, "emulator_verify", refuse)
    code, checks = run(sc, capsys, "--role", "train")
    assert code == 1
    assert checks["emulator"]["status"] == "FAIL"
    assert "CLASH OF CLANS" in checks["emulator"]["detail"]


def test_emulator_check_runs_the_real_verify(sc, monkeypatch, tmp_path):
    fake = machine.load(env={}, config_path=tmp_path / "none.json", root=tmp_path,
                        mumu_roots=[], which=lambda name: None)
    monkeypatch.setattr(machine, "get", lambda: fake)        # what emulator.py reads at import
    monkeypatch.setattr(sys, "path", list(sys.path))
    adb = FakeAdb()
    monkeypatch.setattr(subprocess, "run", adb)
    saved = sys.modules.pop("emulator", None)
    try:
        ok = sc.check_emulator(SERIAL, Path("adb"))
        assert ok.status == "PASS", ok.detail
        assert ("shell", "pm", "list", "packages") in adb.seen
        adb.answers[("shell", "pm", "list", "packages")] = (
            0, "package:com.supercell.clashofclans\n", "")
        refused = sc.check_emulator(SERIAL, Path("adb"))
        assert refused.status == "FAIL" and "CLASH OF CLANS" in refused.detail
    finally:
        sys.modules.pop("emulator", None)
        if saved is not None:
            sys.modules["emulator"] = saved


# ---------------------------------------------------------------------------
# machine settings, role, friendly-only, the command line
# ---------------------------------------------------------------------------

def test_broken_machine_json_fails_and_skips_the_device(sc, world, capsys):
    world.config.write_text("{not json", encoding="utf-8")
    code, checks = run(sc, capsys, "--role", "train")
    assert code == 1
    assert checks["machine"]["status"] == "FAIL"
    assert checks["adb"]["status"] == "SKIP" and world.adb.calls == []
    code, checks = run(sc, capsys, "--role", "train", "--adb", str(world.adb_exe),
                       "--serial", SERIAL)
    assert checks["adb"]["status"] == "PASS"                 # explicit flags still work
    assert code == 1                                         # machine.json is still broken


def test_role_comes_from_the_interpreter_location(sc, tmp_path):
    engine, train = tmp_path / "royalegym", tmp_path / "buildabot"
    for venv in (engine, train):
        (venv / "Scripts").mkdir(parents=True)
    m = machine.load(env={"HASTYCR_PY_ENGINE": str(engine / "Scripts" / "python.exe"),
                          "HASTYCR_PY_TRAIN": str(train / "Scripts" / "python.exe")},
                     config_path=tmp_path / "none.json", root=tmp_path, mumu_roots=[],
                     which=lambda name: None)
    other = str(tmp_path / "python.exe")
    assert sc.role_of_interpreter(m, executable=str(train / "Scripts" / "python.exe"),
                                  prefix=str(tmp_path / "elsewhere")) == "train"
    assert sc.role_of_interpreter(m, executable=other, prefix=str(engine)) == "engine"
    assert sc.role_of_interpreter(m, executable=other, prefix=str(tmp_path)) is None


def test_role_outside_both_venvs_is_guessed_with_a_warning(sc, monkeypatch, tmp_path):
    m = machine.load(env={"HASTYCR_PY_ENGINE": str(tmp_path / "e" / "Scripts" / "python.exe"),
                          "HASTYCR_PY_TRAIN": str(tmp_path / "t" / "Scripts" / "python.exe")},
                     config_path=tmp_path / "none.json", root=tmp_path, mumu_roots=[],
                     which=lambda name: None)
    monkeypatch.setitem(sys.modules, "torch", make_torch())
    role, check = sc.resolve_role("auto", m)
    assert (role, check.status) == ("train", "WARN")
    monkeypatch.setitem(sys.modules, "torch", make_torch("2.14.1+cpu", cuda=None))
    assert sc.resolve_role("auto", m)[0] == "engine"
    assert sc.resolve_role("engine", m) == ("engine", sc.resolve_role("engine", None)[1])


def test_friendly_only(sc, monkeypatch):
    assert sc.check_friendly().status == "PASS"
    monkeypatch.setattr(machine, "FRIENDLY_ONLY", False)
    assert sc.check_friendly().status == "FAIL"


def test_python_version(sc):
    assert sc.check_python((3, 12, 7)).status == "PASS"
    assert sc.check_python((3, 11, 9)).status == "FAIL"
    assert sc.check_python((3, 13, 0)).status == "FAIL"


def test_bad_arguments_exit_2(sc, capsys):
    with pytest.raises(SystemExit) as stop:
        sc.main(["--role", "ladder"])
    assert stop.value.code == 2
    capsys.readouterr()


def test_exit_code_rules(sc):
    passed, warned, failed = (sc.Check("a", s, "") for s in ("PASS", "WARN", "FAIL"))
    info, skipped = sc.Check("b", "INFO", ""), sc.Check("c", "SKIP", "")
    assert sc.exit_code([passed, info, skipped]) == 0
    assert sc.exit_code([passed, warned]) == 0
    assert sc.exit_code([passed, warned], strict=True) == 1
    assert sc.exit_code([passed, failed]) == 1


def test_selfcheck_loads_only_the_stdlib_and_hastycr():
    source = SELFCHECK.read_text(encoding="utf-8")
    top = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            top |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert top <= {"__future__", "argparse", "importlib", "json", "os", "re", "shutil",
                   "subprocess", "sys", "collections", "dataclasses", "pathlib", "hastycr"}, top
    assert "memread" not in source and "hcr_reader" not in source
