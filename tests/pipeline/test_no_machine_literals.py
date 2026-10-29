"""No machine-specific adb path or serial outside src/hastycr/machine.py (P02).

The original tree carried the author's MuMu adb path (with its Android image
folder) in 18 places and his instance-3 serial in about 20. They now come from
hastycr.machine; this keeps them from creeping back. docs/ may still quote them
as history. The needles are built from pieces so this file does not match itself.
"""
from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

from hastycr import machine

ROOT = Path(__file__).resolve().parents[2]
NEEDLES = ("nx_" + "device", "127.0.0.1:" + "16480")
ALLOWED_FILES = {"src/hastycr/machine.py"}
ALLOWED_TOP = {"docs"}                      # prose; history may quote the old values
SKIP_TOP = {"data", "vendor"}               # datasets, and upstream code installed separately
SKIP_DIRS = {".git", ".venvs", "vendor", "data", "tmp", "runs", "checkpoints", "__pycache__",
             ".pytest_cache", "node_modules", "_references", ".claude"}
BINARY = {".mp4", ".png", ".jpg", ".jpeg", ".gif", ".pt", ".pth", ".onnx", ".npz", ".npy",
          ".parquet", ".pdf", ".zip", ".so", ".dll", ".exe", ".pyd", ".whl", ".pyc"}
MAX_BYTES = 4 * 1024 * 1024
LAUNCHERS = ("run.ps1", "scrcpy-preview.ps1", "scripts/live_probe.ps1",
             "scripts/overnight_supervisor.ps1", "scripts/watch_katacr_overnight.ps1")


def _repo_files() -> list[Path]:
    """Tracked plus untracked-but-not-ignored files; a directory walk without git."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others",
                              "--exclude-standard"], capture_output=True, timeout=60)
        if out.returncode == 0 and out.stdout:
            return [ROOT / p for p in out.stdout.decode("utf-8", "replace").split("\0") if p]
    except (OSError, subprocess.SubprocessError):
        pass
    files = []
    for folder, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        files += [Path(folder) / n for n in names]
    return files


def _scanned() -> list[tuple[str, Path]]:
    out = []
    for path in _repo_files():
        rel = path.relative_to(ROOT).as_posix()
        top = rel.split("/", 1)[0]
        if rel in ALLOWED_FILES or top in ALLOWED_TOP | SKIP_TOP:
            continue
        if path.suffix.lower() in BINARY or not path.is_file():
            continue
        if path.stat().st_size > MAX_BYTES:
            continue
        out.append((rel, path))
    return out


def test_needles_are_what_machine_resolves():
    assert machine.mumu_serial(3) == NEEDLES[1]            # the author's instance 3
    assert NEEDLES[0] in (ROOT / "src" / "hastycr" / "machine.py").read_text(encoding="utf-8")


def test_no_adb_path_or_old_serial_outside_machine_py():
    files = _scanned()
    assert any(rel == "scripts/emulator.py" for rel, _ in files)   # the scan sees the scripts
    offenders = []
    for rel, path in files:
        data = path.read_bytes()
        offenders += [f"{rel}: {needle}" for needle in NEEDLES if needle.encode() in data]
    assert not offenders, ("machine-specific literals; use hastycr.machine "
                           "(or emulator.DEFAULT_ADB / DEFAULT_SERIAL):\n" + "\n".join(offenders))


def test_adb_and_serial_options_have_no_literal_defaults():
    bad = []
    for top in ("scripts", "tools", "src", "sim"):
        for path in sorted((ROOT / top).rglob("*.py")):
            if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
                continue
            try:
                text = path.read_text(encoding="utf-8-sig")
                if "--adb" not in text and "--serial" not in text:     # either quote style
                    continue
                tree = ast.parse(text, str(path))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "add_argument" and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value in ("--adb", "--serial")):
                    continue
                for kw in node.keywords:
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                        bad.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
                    if kw.arg == "default" and isinstance(kw.value, ast.Call) and any(
                            isinstance(a, ast.Constant) and isinstance(a.value, str)
                            for a in kw.value.args):
                        bad.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert not bad, "argparse --adb/--serial defaults must come from hastycr.machine: " + ", ".join(bad)


def test_launchers_ask_machine_py():
    for rel in LAUNCHERS:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "src\\hastycr\\machine.py" in text and "--get serial" in text, rel
        assert "--get adb" in text, rel
    watchdog = (ROOT / "scripts" / "emulator_watchdog.ps1").read_text(encoding="utf-8")
    assert "$env:HASTYCR_SERIAL" in watchdog and "$env:HASTYCR_ADB" in watchdog
