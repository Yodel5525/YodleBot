"""Pinned requirements, the Windows setup script and the runbook agree (P04).

requirements/*.txt hold every version the venvs install. Each file has one exact
name==version pin per line: no ranges, no local build labels, no index options.
scripts/setup_windows.ps1 reads the torch pins from these files, so the versions live
in one place, and passes PyTorch's index for torch itself. These tests only read
files. One test also asks PowerShell, when it is on PATH, to parse the setup script
without running it.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REQ = ROOT / "requirements"
FILES = {"engine": REQ / "engine.txt", "gpu": REQ / "gpu.txt",
         "reference": REQ / "rs022-reference.txt"}
SETUP = ROOT / "scripts" / "setup_windows.ps1"
RUNBOOK = ROOT / "docs" / "SPARRING_RUNBOOK.md"

PIN_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9]+(?:\.[0-9]+)*)$")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(path: Path) -> dict[str, str]:
    """{normalized name: version}; asserts that every other line is blank or a comment."""
    out: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN_LINE.match(line)
        assert match, f"{path.name}:{number}: {raw!r} is not an exact name==version pin"
        name = _normalize(match.group(1))
        assert name not in out, f"{path.name}:{number}: {name} is pinned twice"
        out[name] = match.group(2)
    return out


# ---------------------------------------------------------------------------
# the requirements files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(FILES))
def test_every_line_is_an_exact_pin(key):
    path = FILES[key]
    assert pins(path), f"{path.name} pins nothing"
    assert path.read_bytes().isascii(), f"{path.name} has non-ASCII bytes"


# What the PC's venvs run, measured there on 2026-10-29 (plan P04). A pin changes only
# after the new version was tried on the PC, so a change here is a deliberate one.
PC_ENGINE = {"torch": "2.14.1", "royalesim": "0.1.42", "numpy": "2.5.4", "pyarrow": "26.0.0",
             "pytest": "9.2.0"}
PC_GPU = {"torch": "2.14.1", "torchvision": "0.29.1", "royalesim": "0.1.42", "numpy": "2.5.4",
          "pyarrow": "26.0.0", "scipy": "1.18.1", "opencv-python": "5.0.0.93",
          "ultralytics": "8.4.192", "pillow": "13.0.0", "matplotlib": "3.11.2",
          "pytest": "9.2.0"}


def test_engine_venv_pins():
    assert pins(FILES["engine"]) == PC_ENGINE


def test_gpu_venv_pins():
    assert pins(FILES["gpu"]) == PC_GPU


def test_shared_pins_are_identical_in_both_venvs():
    engine, gpu = pins(FILES["engine"]), pins(FILES["gpu"])
    shared = set(engine) & set(gpu)
    assert {"torch", "royalesim", "numpy", "pyarrow", "pytest"} <= shared
    assert {n: engine[n] for n in shared} == {n: gpu[n] for n in shared}


def test_reference_venv_pins_the_authors_engine_and_nothing_heavy():
    reference, engine = pins(FILES["reference"]), pins(FILES["engine"])
    assert reference["royalesim"] == "0.1.22"
    assert "torch" not in reference
    for name in (set(reference) & set(engine)) - {"royalesim"}:
        assert reference[name] == engine[name], name


# ---------------------------------------------------------------------------
# pyproject.toml ranges must admit the pins
# ---------------------------------------------------------------------------

_CLAUSE = re.compile(r"(==|!=|>=|<=|>|<)\s*([0-9][0-9.]*)")
_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")


def _release(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.strip(".").split("."))


def satisfies(version: str, specifier: str) -> bool:
    """Release-segment comparison for the simple clauses pyproject.toml uses."""
    have = _release(version)
    for op, text in _CLAUSE.findall(specifier):
        want = _release(text)
        width = max(len(have), len(want))
        a, b = have + (0,) * (width - len(have)), want + (0,) * (width - len(want))
        if not {"==": a == b, "!=": a != b, ">=": a >= b, "<=": a <= b,
                ">": a > b, "<": a < b}[op]:
            return False
    return True


def test_satisfies_helper():
    assert satisfies("2.14.1", ">=2.6")
    assert satisfies("2.5.4", ">=1.26")
    assert not satisfies("1.25.0", ">=1.26")
    assert satisfies("0.1.42", "==0.1.42") and not satisfies("0.1.22", "==0.1.42")
    assert satisfies("8.4.175", ">=8.3,<9") and not satisfies("9.0", ">=8.3,<9")


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_pyproject_ranges_admit_the_pins():
    project = _pyproject()
    declared = list(project["dependencies"])
    for group in project["optional-dependencies"].values():
        declared += group
    pinned = {**pins(FILES["engine"]), **pins(FILES["gpu"])}
    checked = 0
    for requirement in declared:
        match = _REQUIREMENT.match(requirement.split(";")[0])
        name, specifier = _normalize(match.group(1)), match.group(2)
        if name in pinned:
            assert satisfies(pinned[name], specifier), \
                f"pyproject.toml {requirement!r} rejects the pin {name}=={pinned[name]}"
            checked += 1
    assert checked >= 6


def test_pyproject_engine_extra_pins_the_same_royalesim():
    engine = [r.replace(" ", "") for r in _pyproject()["optional-dependencies"]["engine"]]
    assert f"royalesim=={pins(FILES['engine'])['royalesim']}" in engine


# ---------------------------------------------------------------------------
# scripts/setup_windows.ps1
# ---------------------------------------------------------------------------

def _setup() -> str:
    return SETUP.read_text(encoding="utf-8")


def _default(param: str) -> str:
    match = re.search(r"\$" + param + r"\s*=\s*'([^']+)'", _setup())
    assert match, f"setup_windows.ps1 has no default for -{param}"
    return match.group(1)


def test_setup_script_is_ascii():
    # Windows PowerShell 5.1 reads a .ps1 without a BOM in the ANSI code page.
    assert SETUP.read_bytes().isascii()


def test_setup_script_takes_versions_only_from_the_requirements():
    text = _setup()
    assert not re.search(r"(torch|torchvision|royalesim|numpy|pyarrow)==\d", text)
    assert "Get-Pin $req 'torch'" in text and "Get-Pin $req 'torchvision'" in text


def test_setup_script_builds_both_venvs_from_their_files():
    text = _setup()
    for needle in ("'royalegym'", "'buildabot'", "'rs022'", "'engine.txt'", "'gpu.txt'",
                   "'rs022-reference.txt'", "--index-url", "--force-reinstall", "--no-deps",
                   "'freeze-'", "scripts\\selfcheck.py", "--offline", "-Recreate"):
        assert needle in text, needle


def test_setup_script_installs_torch_first_and_checks_it_again():
    body = _setup().split("function Install-Venv", 1)[1]
    first = body.index("Repair-Torch")
    requirements = body.index("'-r', $req")
    assert first < requirements < body.rindex("Repair-Torch")


def test_index_urls_agree_between_script_requirements_and_runbook():
    gpu, cpu = _default("GpuIndexUrl"), _default("CpuIndexUrl")
    assert re.fullmatch(r"https://download\.pytorch\.org/whl/cu\d+", gpu)
    assert cpu == "https://download.pytorch.org/whl/cpu"
    assert gpu in FILES["gpu"].read_text(encoding="utf-8")
    assert cpu in FILES["engine"].read_text(encoding="utf-8")
    assert gpu in RUNBOOK.read_text(encoding="utf-8")


def _powershell() -> str | None:
    for name in ("powershell", "pwsh"):
        found = shutil.which(name)
        if found:
            return found
    return None


def test_setup_script_parses_in_powershell():
    exe = _powershell()
    if exe is None:
        pytest.skip("no PowerShell on PATH")
    path = str(SETUP).replace("'", "''")
    script = ("$e = $null; $t = $null; "
              f"[void][System.Management.Automation.Language.Parser]::ParseFile('{path}', "
              "[ref]$t, [ref]$e); "
              "foreach ($x in $e) { Write-Output ('line {0}: {1}' -f "
              "$x.Extent.StartLineNumber, $x.Message) }; exit $e.Count")
    out = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-Command", script],
                         capture_output=True, text=True, encoding="utf-8", errors="replace",
                         timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr


# ---------------------------------------------------------------------------
# docs/SPARRING_RUNBOOK.md
# ---------------------------------------------------------------------------

LATER = "\n## Later steps"
_PATH = re.compile(r"(?<![\w./\\-])((?:scripts|tools|sim|src)[\\/][\w./\\-]+?\.(?:py|ps1))")
_MODULE = re.compile(r"-m\s+((?:sim|tools|hastycr)\.[\w.]*\w)")


def _runbook() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def test_runbook_clones_the_fork_without_the_video_dumps():
    text = _runbook()
    for needle in (
            "git clone --filter=blob:none --no-checkout https://github.com/Yodel5525/YodleBot.git C:\\hcr",
            'git sparse-checkout set --no-cone "/*" "!/data/vod/" "!/demo.mp4"',
            "git checkout sparring"):
        assert needle in text, needle


def test_runbook_covers_setup_selfcheck_tests_and_memory():
    text = _runbook()
    for needle in ("scripts\\setup_windows.ps1", "scripts\\selfcheck.py", "--offline",
                   "-m pytest tests/pipeline", "16 GB"):
        assert needle in text, needle


def test_runbook_steps_only_run_scripts_that_exist():
    head, sep, _ = _runbook().partition(LATER)
    assert sep, "the runbook needs a '## Later steps' section for what is not built yet"
    missing = []
    for line in head.splitlines():
        if "later batch" in line:
            continue                      # names something that is announced, not run
        for raw in _PATH.findall(line):
            rel = raw.replace("\\", "/")
            if not (ROOT / rel).exists():
                missing.append(rel)
        for module in _MODULE.findall(line):
            parts = module.split(".")
            base = ROOT / "src" if parts[0] == "hastycr" else ROOT
            if not (base.joinpath(*parts).with_suffix(".py").is_file()
                    or base.joinpath(*parts, "__init__.py").is_file()):
                missing.append(module)
    assert not missing, f"runbook steps name files that do not exist yet: {missing}"
