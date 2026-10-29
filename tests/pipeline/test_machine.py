"""hastycr.machine: precedence (env > machine.json > default), adb discovery, the CLI,
and the two readers wired to it in P01 (scripts/emulator.py, tools/rs_selfplay.py).

Every load() here gets an explicit env, root, mumu_roots and which, so a real
HASTYCR_* variable, machine.json, PATH or MuMu install on the PC cannot change a result.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hastycr import machine

ROOT = Path(__file__).resolve().parents[2]


def _load(tmp_path, env=None, **kw):
    kw.setdefault("root", tmp_path)
    kw.setdefault("mumu_roots", [])
    kw.setdefault("which", lambda name: None)
    return machine.load(env or {}, **kw)


def _write(path: Path, data, encoding="utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding=encoding)
    return path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.fixture
def isolated_machine(monkeypatch, tmp_path):
    """os.environ without HASTYCR_* and machine.json pointed at tmp; cache cleared around."""
    for var in (*machine.ENV.values(), machine.CONFIG_ENV):
        monkeypatch.delenv(var, raising=False)
    config = tmp_path / "machine.json"
    monkeypatch.setenv(machine.CONFIG_ENV, str(config))
    machine.get.cache_clear()
    yield config
    machine.get.cache_clear()


def test_defaults_without_env_or_file(tmp_path):
    m = _load(tmp_path)
    assert m.serial == "127.0.0.1:16384" == machine.DEFAULT_SERIAL
    assert m.instance == "Android Device"
    assert m.adb == Path("adb") and m.sources["adb"] == "not found"
    assert any(w.startswith("adb:") for w in m.warnings)
    exe = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    assert m.py_engine == tmp_path.joinpath(".venvs", "royalegym", *exe)
    assert m.py_train == tmp_path.joinpath(".venvs", "buildabot", *exe)
    assert (m.data, m.runs) == (tmp_path / "data", tmp_path / "runs")
    assert m.mumu_manager is None and m.sources["mumu_manager"] == "not found"
    assert m.config_path == tmp_path / "machine.json" and not m.config_found
    for name in ("serial", "instance", "py_engine", "py_train", "data", "runs"):
        assert m.sources[name] == "default"
    assert m.values()["mumu_manager"] == ""


def test_env_beats_machine_json_beats_default(tmp_path):
    _write(tmp_path / "machine.json", {"serial": "127.0.0.1:16416", "instance": "Bot",
                                       "data": "D2", "adb": "tools/adb.exe", "runs": None,
                                       "_comment": ["ignored"]})
    elsewhere = tmp_path / "big disk" / "data"
    m = _load(tmp_path, {"HASTYCR_SERIAL": "127.0.0.1:16448", "HASTYCR_DATA": str(elsewhere),
                         "HASTYCR_INSTANCE": "   "})
    assert m.config_found
    assert (m.serial, m.sources["serial"]) == ("127.0.0.1:16448", "env HASTYCR_SERIAL")
    assert (m.instance, m.sources["instance"]) == ("Bot", "machine.json")   # blank env = unset
    assert (m.data, m.sources["data"]) == (elsewhere, "env HASTYCR_DATA")
    assert (m.adb, m.sources["adb"]) == (tmp_path / "tools" / "adb.exe", "machine.json")
    assert any("does not exist" in w for w in m.warnings)
    assert (m.runs, m.sources["runs"]) == (tmp_path / "runs", "default")    # null = default
    assert not any(w.startswith("config:") for w in m.warnings)

    m = _load(tmp_path)                                                     # file alone
    assert (m.serial, m.data) == ("127.0.0.1:16416", tmp_path / "D2")


def test_machine_json_named_by_HASTYCR_MACHINE(tmp_path):
    runs = tmp_path / "elsewhere"
    cfg = _write(tmp_path / "cfg" / "pc.json", {"runs": str(runs), "sereal": "typo"})
    m = _load(tmp_path, {"HASTYCR_MACHINE": str(cfg)})
    assert m.config_path == cfg and m.config_found
    assert m.runs == runs and m.sources["runs"] == str(cfg)
    assert any("'sereal'" in w for w in m.warnings)
    m = _load(tmp_path, {"HASTYCR_MACHINE": "cfg/pc.json"})                 # relative to root
    assert m.config_path == cfg and m.runs == runs


def test_bad_machine_json_is_an_error(tmp_path):
    cfg = tmp_path / "machine.json"
    _write(cfg, '{"adb": "C:\\Program Files\\adb.exe"}')                   # single backslashes
    with pytest.raises(machine.MachineConfigError, match="backslash"):
        _load(tmp_path)
    _write(cfg, "[1, 2]")
    with pytest.raises(machine.MachineConfigError, match="object"):
        _load(tmp_path)
    _write(cfg, {"serial": 16384})
    with pytest.raises(machine.MachineConfigError, match="string"):
        _load(tmp_path)
    _write(cfg, {"serial": "127.0.0.1:16416"}, encoding="utf-8-sig")       # Notepad's BOM
    assert _load(tmp_path).serial == "127.0.0.1:16416"


def _fake_mumu(pf: Path) -> dict[str, Path]:
    """A Program Files folder with two Android images and an older MuMu 12 layout."""
    images = machine.MUMU_ADB_GLOBS[0]                  # .../<android image>/shell/adb.exe
    legacy = next(p for p in machine.MUMU_ADB_GLOBS if "*" not in p and "12.0" in p)
    return {"15.0": _touch(pf / images.replace("*", "15.0")),
            "12.0": _touch(pf / images.replace("*", "12.0")),
            "legacy": _touch(pf / legacy)}


def test_adb_order_env_json_path_mumu(tmp_path):
    pf = tmp_path / "Program Files"
    mumu = _fake_mumu(pf)
    assert machine.mumu_adb_candidates([pf]) == [mumu["15.0"], mumu["12.0"], mumu["legacy"]]

    m = _load(tmp_path, mumu_roots=[pf])
    assert (m.adb, m.sources["adb"]) == (mumu["15.0"], "MuMu install")
    assert not m.warnings

    on_path = str(tmp_path / "platform-tools" / "adb.exe")
    m = _load(tmp_path, mumu_roots=[pf], which=lambda name: on_path if name == "adb" else None)
    assert (m.adb, m.sources["adb"]) == (Path(on_path), "PATH")
    assert any("MuMu has its own" in w for w in m.warnings)

    _write(tmp_path / "machine.json", {"adb": str(mumu["12.0"])})
    m = _load(tmp_path, mumu_roots=[pf], which=lambda name: on_path)
    assert (m.adb, m.sources["adb"]) == (mumu["12.0"], "machine.json")

    m = _load(tmp_path, {"HASTYCR_ADB": str(mumu["legacy"])}, mumu_roots=[pf],
              which=lambda name: on_path)
    assert (m.adb, m.sources["adb"]) == (mumu["legacy"], "env HASTYCR_ADB")
    assert not m.warnings


def test_adb_quotes_drives_and_bare_names(tmp_path):
    m = _load(tmp_path, {"HASTYCR_ADB": '"C:\\MuMu\\shell\\adb.exe"'})
    assert str(m.adb) == "C:\\MuMu\\shell\\adb.exe"          # absolute on any OS, quotes gone
    m = _load(tmp_path, {"HASTYCR_ADB": "adb"},
              which=lambda name: "/opt/android/adb" if name == "adb" else None)
    assert m.adb == Path("/opt/android/adb")


def test_mumu_manager_is_found_next_to_mumu(tmp_path):
    pf = tmp_path / "pf"
    manager = _touch(pf / machine.MUMU_MANAGER_GLOBS[0])
    m = _load(tmp_path, mumu_roots=[pf])
    assert (m.mumu_manager, m.sources["mumu_manager"]) == (manager, "MuMu install")


def test_venv_python_fallbacks(tmp_path):
    venv = tmp_path / "v"
    assert machine.venv_python(venv, os_name="nt") == venv / "Scripts" / "python.exe"
    assert machine.venv_python(venv, os_name="posix") == venv / "bin" / "python"
    _touch(venv / "bin" / "python")
    assert machine.venv_python(venv, os_name="nt") == venv / "bin" / "python"
    _touch(venv / "Scripts" / "python.exe")
    assert machine.venv_python(venv, os_name="nt") == venv / "Scripts" / "python.exe"
    assert machine.venv_python(venv, os_name="posix") == venv / "bin" / "python"


def test_mumu_serial_and_install_roots():
    assert machine.mumu_serial(0) == "127.0.0.1:16384"
    assert machine.mumu_serial(1) == "127.0.0.1:16416"
    assert machine.mumu_serial(3).endswith(":16480")
    with pytest.raises(ValueError):
        machine.mumu_serial(-1)
    assert machine.install_roots({}, os_name="posix") == []
    roots = machine.install_roots({"ProgramFiles": r"E:\Apps"}, os_name="nt")
    assert roots[0] == Path(r"E:\Apps") and Path(r"C:\Program Files") in roots


def test_get_is_cached_until_reload(isolated_machine, monkeypatch):
    monkeypatch.setenv("HASTYCR_SERIAL", "127.0.0.1:16416")
    assert machine.serial() == "127.0.0.1:16416"
    monkeypatch.setenv("HASTYCR_SERIAL", "127.0.0.1:16448")
    assert machine.serial() == "127.0.0.1:16416"
    assert machine.reload().serial == "127.0.0.1:16448"
    assert machine.get().config_path == isolated_machine


def test_cli_get_json_and_table(isolated_machine, capsys):
    _write(isolated_machine, {"serial": "127.0.0.1:16416"})
    assert machine.main(["--get", "serial"]) == 0
    assert capsys.readouterr().out == "127.0.0.1:16416\n"
    assert machine.main(["--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["settings"]["serial"] == "127.0.0.1:16416"
    assert report["sources"]["serial"] == "machine.json" and report["friendly_only"] is True
    assert machine.main([]) == 0
    table = capsys.readouterr().out
    assert "127.0.0.1:16416" in table and "friendly_only" in table
    _write(isolated_machine, "{not json")
    assert machine.main(["--get", "serial"]) == 2


def test_imports_without_torch_and_runs_as_a_file(tmp_path):
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import hastycr.machine as m; "
            "m.load({}, mumu_roots=[], which=lambda n: None); "
            "print(','.join(n for n in ('torch', 'numpy', 'royalesim') if n in sys.modules))")
    out = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "src")],
                         capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
    env = {**os.environ, "HASTYCR_SERIAL": "127.0.0.1:16416",
           "HASTYCR_MACHINE": str(tmp_path / "none.json")}
    out = subprocess.run([sys.executable, "-I", str(ROOT / "src" / "hastycr" / "machine.py"),
                          "--get", "serial"], capture_output=True, text=True,
                         encoding="utf-8", env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "127.0.0.1:16416"           # what run.ps1 reads


def test_emulator_defaults_come_from_machine(monkeypatch, tmp_path):
    # A fresh import against known settings: an `emulator` imported earlier in the
    # session keeps the values machine.get() had then, whatever this test sets.
    adb = _touch(tmp_path / "mumu" / "adb.exe")
    fake = _load(tmp_path, {"HASTYCR_ADB": str(adb), "HASTYCR_SERIAL": "127.0.0.1:16448",
                            "HASTYCR_INSTANCE": "Sparring"})
    monkeypatch.setattr(machine, "get", lambda: fake)
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    saved = sys.modules.pop("emulator", None)
    try:
        import emulator
        assert (emulator.DEFAULT_ADB, emulator.DEFAULT_SERIAL, emulator.DEFAULT_INSTANCE) == (
            adb, "127.0.0.1:16448", "Sparring")
        assert emulator.packages.__defaults__ == ("127.0.0.1:16448", adb)
    finally:                                # later tests get the real settings back
        sys.modules.pop("emulator", None)
        if saved is not None:
            sys.modules["emulator"] = saved


def test_emulator_verify_still_refuses_other_devices(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import emulator
    cases = {
        frozenset(): "cannot reach",
        frozenset({"com.supercell.clashofclans", "com.android.chrome"}): "CLASH OF CLANS",
        frozenset({"com.android.chrome"}): "not installed",
    }
    for found, message in cases.items():
        monkeypatch.setattr(emulator, "packages", lambda serial, adb, found=found: set(found))
        with pytest.raises(SystemExit, match=message):
            emulator.verify("127.0.0.1:16384", "adb")
    monkeypatch.setattr(emulator, "packages",
                        lambda serial, adb: {"com.supercell.clashroyale", "com.android.chrome"})
    assert emulator.verify("127.0.0.1:16384", "adb") is None


def test_emulator_packages_parses_pm_output(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import emulator
    seen = []

    def fake_run(argv, **kw):
        seen.append(argv)
        return SimpleNamespace(stdout="package:com.supercell.clashroyale\n\npackage:com.x\n")

    monkeypatch.setattr(emulator.subprocess, "run", fake_run)
    assert emulator.packages("127.0.0.1:16384", "adb") == {"com.supercell.clashroyale", "com.x"}
    assert seen[0][:3] == ["adb", "-s", "127.0.0.1:16384"]

    def broken(argv, **kw):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(emulator.subprocess, "run", broken)
    assert emulator.packages("127.0.0.1:16384", "adb") == set()


def test_rs_selfplay_interpreters_come_from_machine(isolated_machine, monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    engine, train = tmp_path / "eng" / "python.exe", tmp_path / "gpu" / "python.exe"
    _write(isolated_machine, {"py_engine": str(engine), "py_train": str(train)})
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import rs_selfplay
    monkeypatch.setattr(rs_selfplay, "RUNS", tmp_path / "runs")
    init = tmp_path / "init.pt"
    init.write_bytes(b"not a checkpoint")
    calls = []

    def fake_run(argv, **kw):
        calls.append([str(a) for a in argv])
        if argv[2] == "collect":
            return SimpleNamespace(returncode=0, stdout='{"matches": 2}\n', stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="stop the test here")

    monkeypatch.setattr(rs_selfplay.subprocess, "run", fake_run)
    args = argparse.Namespace(name="t", init=str(init), ref="", opponent="opp.pt", iters=1,
                              matches=2, workers=1, lr=1e-5, kl=0.05, eval_every=0,
                              eval_matches=10, eval_opponent="ep1.pt")
    assert rs_selfplay.loop(args) == 1
    assert [c[0] for c in calls] == [str(engine), str(train)]
    assert [c[2] for c in calls] == ["collect", "train"]
