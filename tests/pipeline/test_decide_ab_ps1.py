"""scripts/decide_ab.ps1, read as text, in the style of tests/pipeline/test_model_check_ps1.py.

Windows PowerShell 5.1 runs the script on the owner's PC and nothing here can run it, so these
tests read it (and ask PowerShell to parse it when PowerShell is on PATH):

- PS 5.1: ASCII without a BOM, no && / || chains, files written without a BOM, no regex match
  operators, every function reads script variables as $script:NAME and no function parameter
  has a script variable's name (dynamic scoping);
- its parameters are the documented ones with the documented defaults (the A/B the owner asked
  for: runs\\tokens\\v2_hog\\best.pt, both rules, a 0.95 floor = rs_decide.WAIT_FLOOR_AB, 1,000
  games, 4 workers, its own league log, the noise prior), and the runbook's decide_ab.ps1
  commands use only those;
- the tools it names exist, every --flag it passes to tools/rs_league_eval.py and
  tools/decide_ab.py is one their argparse defines (two #region blocks, no flag literal outside);
- the players it builds parse (rs_league_eval.parse_player) to the same checkpoint with the
  rules on for each candidate and off for the champion, under distinct names;
- the -WaitFloor pattern takes 0.5-1 with two decimals at most and nothing else;
- the summary copies the lines tools/decide_ab.py prints ("verdict:", "  INCOMPLETE:"), and the
  exit codes it reads are decide_ab's;
- no tap, memory reader or adb.
"""
from __future__ import annotations

import ast
import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.pipeline.test_model_check_ps1 import (AUTOMATIC, FLAG, _argparse_flags, _assigned,
                                                 _functions, _script_level, _strip)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "decide_ab.ps1"
LEAGUE = ROOT / "tools" / "rs_league_eval.py"
VERDICT = ROOT / "tools" / "decide_ab.py"
RUNBOOK = ROOT / "docs" / "SPARRING_RUNBOOK.md"

PARAMS = ["Ckpt", "Variant", "WaitFloor", "Games", "Workers", "Log", "Noise"]
REGION = re.compile(r"^#region (rs_league_eval|decide_ab) flags\b[^\n]*\n(.*?)^#endregion",
                    re.M | re.S)


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _params(text: str) -> list[str]:
    m = re.search(r"^param\((.*?)^\)", text, re.M | re.S)
    assert m, "decide_ab.ps1 has no param( ... ) block at the start of a line"
    return re.findall(r"\]\s*\$(\w+)", m.group(1))


def _regions(text: str) -> dict[str, tuple[str, int, int]]:
    return {m.group(1): (m.group(2), m.start(2), m.end(2)) for m in REGION.finditer(text)}


# ---------------------------------------------------------------------------
# PowerShell 5.1
# ---------------------------------------------------------------------------

def test_script_is_ascii_and_writes_without_a_bom():
    raw = SCRIPT.read_bytes()
    assert raw.isascii(), [i for i, b in enumerate(raw) if b > 127][:5]
    text = _text()
    assert "&&" not in text and "||" not in text
    assert "New-Object System.Text.UTF8Encoding($false)" in text
    assert "UTF8Encoding($true)" not in text
    assert not re.search(r"\b(Out-File|Set-Content|Add-Content)\b", _strip(text))
    assert "$ErrorActionPreference = 'Continue'" in text


def test_no_regex_match_operators():
    code = _strip(_text())
    assert not re.search(r"-[ci]?(not)?match\b", code, re.I)
    assert not re.search(r"\bswitch\s+-regex\b", code, re.I)
    assert "$Matches" not in code and "$matches" not in code


def test_functions_read_script_variables_with_script_scope():
    text = _text()
    _, script_vars = _script_level(text)
    code = _strip(text)
    funcs = _functions(text)
    assert {"Write-Both", "Stop-Run", "Invoke-Step", "Resolve-Model"} <= {f[0] for f in funcs}
    bad = []
    for name, params, _, start, end in funcs:
        body = code[start:end]
        local = _assigned(body) | {p.lower() for p in params}
        read = {v.lower() for v in re.findall(r"(?<![\w:`])\$(\w+)", body)}
        leaked = sorted((read - local - AUTOMATIC) & script_vars)
        if leaked:
            bad.append((name, leaked))
    assert not bad, f"read without $script: (dynamic scoping): {bad}"
    clashes = [(name, p) for name, params, _, _, _ in funcs for p in params
               if p.lower() in script_vars]
    assert not clashes, clashes


def test_parses_in_powershell():
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if exe is None:
        pytest.skip("no PowerShell on PATH")
    path = str(SCRIPT).replace("'", "''")
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
# parameters, tools and flags
# ---------------------------------------------------------------------------

def test_param_block_defaults_and_help(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    rs_decide = importlib.import_module("rs_decide")
    text = _text()
    assert _params(text) == PARAMS
    assert sorted(re.findall(r"^\.PARAMETER\s+(\w+)", text, re.M)) == sorted(PARAMS)
    for needle in ("[string]$Ckpt = 'runs\\tokens\\v2_hog\\best.pt'",
                   "[ValidateSet('both', 'floor', 'stray', 'all')][string]$Variant = 'both'",
                   "[string]$WaitFloor = '0.95'", "[int]$Games = 1000", "[int]$Workers = 4",
                   "[string]$Log = 'runs\\league\\decide_ab.jsonl'",
                   "[string]$Noise = 'data\\noise\\default_prior.json'"):
        assert needle in text, needle
    assert float(re.search(r"\$WaitFloor = '([0-9.]+)'", text).group(1)) == rs_decide.WAIT_FLOOR_AB
    assert (ROOT / "data" / "noise" / "default_prior.json").is_file()
    # its own league log: not section 20's (third_model.jsonl) nor model_check.ps1's
    for other in ("third_model.jsonl", "model_check.jsonl"):
        assert other not in re.search(r"\$Log = '[^']*'", text).group(0)


def test_named_tools_exist():
    names = set(re.findall(r"(?<![\w.\\/-])((?:tools|scripts)[\\/][\w.\\/-]+?\.(?:py|ps1))",
                           _text()))
    assert {"tools\\rs_league_eval.py", "tools\\decide_ab.py"} <= names
    missing = sorted(n for n in names if not (ROOT / n.replace("\\", "/")).is_file())
    assert not missing, missing


def test_flags_exist_in_the_tools_argparse():
    regions = _regions(_text())
    assert set(regions) == {"rs_league_eval", "decide_ab"}
    for tool, path in (("rs_league_eval", LEAGUE), ("decide_ab", VERDICT)):
        body = regions[tool][0]
        assert f"'tools\\{tool}.py'" in body
        flags = set(FLAG.findall(body))
        assert flags, tool
        unknown = sorted(flags - _argparse_flags(path))
        assert not unknown, f"{tool}: {unknown}"
    league = regions["rs_league_eval"][0]
    assert "'tools\\rs_league_eval.py', 'run'" in league
    for needle in ("'--champion-games', \"$Games\"", "'--lower-bound', '0.5'", "'--delay', '24'",
                   "'--decide', 'live'", "'--noise', $Noise", "'--log', $logModel.Rel",
                   "'--report', $reportModel.Rel", "'--candidate', $c.Spec",
                   "'--champion', $champion.Spec", "'--workers', \"$leagueWorkers\""):
        assert needle in league, needle
    assert "[Math]::Min($Workers, 6)" in _text()
    verdict = regions["decide_ab"][0]
    for needle in ("'--log', $logModel.Rel", "'--champion', $champion.Name",
                   "'--games', \"$Games\"", "'--candidate', $c.Name"):
        assert needle in verdict, needle


def test_every_flag_literal_is_in_a_checked_region():
    text = _text()
    spans = [(start, end) for _, start, end in _regions(text).values()]
    outside = [m.group(1) for m in FLAG.finditer(text)
               if not any(start <= m.start() < end for start, end in spans)]
    assert not outside, outside


def test_players_parse_to_one_checkpoint_with_and_without_the_rules(monkeypatch):
    """The script's player templates, filled in as -f fills them, are league players: each
    candidate the same checkpoint as the champion, with its rules on, the champion with none."""
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    monkeypatch.setitem(sys.modules, "royaleimitate", None)
    LE = importlib.import_module("rs_league_eval")
    text = _text()
    specs = re.findall(r"Spec = \('([^']*)' -f ([^)]*)\)", text)
    assert len(specs) == 4, specs
    names = set(re.findall(r"\$candName = '([^']*)' -f \$model\.Name", text))
    assert names == {"{0}_rules{1}", "{0}_floor{1}", "{0}_stray"}
    ckpt = "runs\\tokens\\v2_hog\\best.pt"
    filled = {}
    for template, args in specs:
        values = {"$candName": None, "$model.Name": "v2_hog", "$model.Rel": ckpt,
                  "$floorText": "0.95"}
        parts = [a.strip() for a in args.split(",")]
        assert all(p in values for p in parts), parts
        fmt = template.replace("{0}", "\x00").replace("{1}", "\x01").replace("{2}", "\x02")
        for i, p in enumerate(parts):
            fmt = fmt.replace(chr(i), "NAME" if p == "$candName" else values[p])
        filled[template] = LE.parse_player(fmt)
    champ = filled["{0}={1}"]
    assert (champ.label, champ.ckpt, champ.rules) == ("v2_hog", ckpt, {})
    cands = [a for t, a in filled.items() if t != "{0}={1}"]
    assert sorted(a.rules.get("wait_floor", 0) for a in cands) == [0, 0.95, 0.95]
    assert sorted(bool(a.rules.get("stray_guard")) for a in cands) == [False, True, True]
    assert all(a.ckpt == ckpt and a.label == "NAME" for a in cands)
    assert any(a.rules == {"wait_floor": 0.95, "stray_guard": True} for a in cands)


def test_wait_floor_pattern():
    m = re.search(r"\[regex\]::IsMatch\(\$WaitFloor\.Trim\(\), '([^']*)'\)", _text())
    assert m, "decide_ab.ps1's -WaitFloor check"
    pat = re.compile(m.group(1))
    for ok in ("0.95", "0.5", "0.9", "0.99", ".95", "1", "1.0", "1.00"):
        assert pat.search(ok), ok
    for bad in ("0.4", "0,95", "0.955", "1.01", "2", "95", "0.95x", "", "-0.95"):
        assert not pat.search(bad), bad
    assert "'_wf{0:000}' -f [int][Math]::Round($floorValue * 100)" in _text()


def test_summary_reads_what_decide_ab_prints():
    text = _text()
    for needle in ("StartsWith('verdict:')", "StartsWith('  INCOMPLETE:')",
                   "StartsWith('verdict: STRONGER')", "StartsWith('verdict: WEAKER')",
                   "$verdict.Code -eq 2"):
        assert needle in text, needle
    source = VERDICT.read_text(encoding="utf-8")
    for needle in ('f"verdict: {word}: ', '"  INCOMPLETE: ', '"STRONGER"', '"WEAKER"',
                   "code = 2", "return 1"):
        assert needle in source, needle
    tree = ast.parse(source)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "format_ab" for n in tree.body)
    # the league step's failure stops the verdict (no older games stand in)
    assert "if (-not $league.Ok) {" in text


def test_runbook_section_and_commands():
    book = RUNBOOK.read_text(encoding="utf-8")
    assert "## 22. Decision rules: the A/B in the simulator" in book
    assert "section 22" in _text()
    lines = [ln for ln in book.splitlines() if "decide_ab.ps1" in ln]
    assert any("scripts\\decide_ab.ps1" in ln for ln in lines)
    option = re.compile(r"(?<![\w-])-([A-Z][A-Za-z]+)\b")
    used = set()
    for ln in lines:
        for part in ln.split("decide_ab.ps1")[1:]:
            used |= set(option.findall(part.split("`")[0]))
    assert "Variant" in used and used <= set(PARAMS), sorted(used - set(PARAMS))


def test_friendly_only_and_no_memory_reader():
    text = _text()
    for needle in ("--tap", "memread", "hcr_reader", "adb ", "vision_play"):
        assert needle not in text, needle
