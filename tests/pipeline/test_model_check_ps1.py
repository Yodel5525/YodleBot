"""scripts/model_check.ps1, read as text: what commonly breaks in an unattended PowerShell run.

Windows PowerShell 5.1 runs the script on the owner's PC, and nothing in the cloud sandbox can
run it, so these tests read it (one test also asks PowerShell to parse it when PowerShell is on
PATH, as tests/pipeline/test_requirements.py does for scripts/setup_windows.ps1):

- the tools and scripts it names exist, and every --flag it passes to tools/rs_eval.py and
  tools/rs_league_eval.py is one their argparse defines. Those flags sit in two #region blocks,
  and no flag literal sits outside them;
- the league step plays at scripts/third_model.ps1's settings (the same flags, delay, games,
  worker cap, noise default, champion and anchor), and both write their report per matchup
  under one name (--report), so neither writes over step 1's third_model.report.json;
- its parameters are the documented ones, and the runbook's model_check.ps1 commands use only
  those;
- PS 5.1 reads a .ps1 without a byte-order mark in the ANSI code page, so the file is ASCII; it
  has no && or || pipeline chains (PowerShell 7 only); it writes files without a BOM;
- the -Matches parameter has the name of the automatic variable that PowerShell's regex match
  operators fill, so the script uses none of them;
- dynamic scoping: a function that reads a script variable without $script: can be handed a
  caller's local of the same name, case ignored. scripts/dt_diagnose.ps1's summary lost its
  first wait-bias run that way (Invoke-WaitBias's -Keep stood in for the script's $keep regex
  inside Invoke-Check). Here every function reads script variables as $script:NAME, and no
  function parameter has a script variable's name;
- the result-line pattern reads what rs_eval prints (its own f-string, evaluated), the gate-line
  pattern what rs_league_eval.format_report prints, and the lines the summary copies exist;
- the wait_keep probe, run in a subprocess, reads a copy's args.json (written as the script
  writes it) the way rs_decide.wait_keep_for does, and says 'none' without one;
- an existing wait_keep copy (perhaps the live model) is never overwritten from another or a
  changed source, and two checkpoints of one run never share a name;
- a name clash stops the run only where the league would refuse it (-LeagueLog) or for the
  candidate, and no advice moves a league log;
- scripts/dt_diagnose.ps1's summary pattern is no longer shadowed by a parameter;
- since 2026-10-30 (the freeze at 8 rs_eval workers; the champion v2_hog_20k): model_check.ps1,
  third_model.ps1, dt_diagnose.ps1, rl_train.ps1, first_model.ps1 and second_model.ps1 default
  to 4 workers, with a help line that points at the runbook's freeze note, and rl_train.ps1's
  bench plays at most -Workers; model_check.ps1's plan: line counts the times measured at 4
  workers; third_model.ps1 takes -Champion (default v2_hog_20k), names it as model_check.ps1's
  Resolve-Model does, checks it before any step (a file; not the candidate, the anchor or the
  general model; its games in the league log of the same file and path), and plays it in the
  eval step's reference run and the league; its start-of-run refusal of a candidate name with
  games in the log stays.
"""
from __future__ import annotations

import ast
import json
import math
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "model_check.ps1"
RS_EVAL = ROOT / "tools" / "rs_eval.py"
LEAGUE = ROOT / "tools" / "rs_league_eval.py"
THIRD = ROOT / "scripts" / "third_model.ps1"
RUNBOOK = ROOT / "docs" / "SPARRING_RUNBOOK.md"
DT_DIAGNOSE = ROOT / "scripts" / "dt_diagnose.ps1"

PARAMS = ["Ckpt", "Keeps", "Rules", "Delay", "Matches", "Workers", "Opponent", "League",
          "LeagueOnly", "LeagueKeep", "LeagueName", "LeagueLog", "Champion", "Noise",
          "NoBaseline"]
#: Variables PowerShell itself provides, and scope prefixes ($script:x reads as "script").
AUTOMATIC = {"script", "global", "env", "_", "null", "true", "false", "lastexitcode",
             "psscriptroot", "erroractionpreference"}
FLAG = re.compile(r"'(--[a-z][a-z0-9-]*)'")
REGION = re.compile(r"^#region (rs_eval|rs_league_eval) flags\b[^\n]*\n(.*?)^#endregion",
                    re.M | re.S)


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _blank(s: str) -> str:
    return "".join("\n" if c == "\n" else " " for c in s)


def _strip(text: str, keep_double: bool = True) -> str:
    """The script with comments and single-quoted strings blanked (same length, newlines kept);
    double-quoted strings (which PowerShell expands) kept, or blanked too."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("<#", i):
            j = text.find("#>", i + 2)
            j = n if j < 0 else j + 2
            out.append(_blank(text[i:j]))
        elif c == "#":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(_blank(text[i:j]))
        elif c == "'":
            j = i + 1
            while j < n and not (text[j] == "'" and text[j + 1:j + 2] != "'"):
                j += 2 if text[j] == "'" else 1
            out.append("'" + _blank(text[i + 1:j]) + "'")
            j += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "`" else 1
            out.append(text[i:j + 1] if keep_double else '"' + _blank(text[i + 1:j]) + '"')
            j += 1
        else:
            out.append(c)
            j = i + 1
        i = j
    return "".join(out)


def _functions(text: str) -> list[tuple[str, list[str], int, int, int]]:
    """(name, parameter names, header start, body start, body end) of every `function Name(...) {`
    at the start of a line, the body found by counting braces outside strings and comments."""
    code, bare = _strip(text), _strip(text, keep_double=False)
    out = []
    for m in re.finditer(r"^function\s+([\w-]+)\s*\(([^)]*)\)\s*\{", code, re.M):
        depth, k = 0, m.end() - 1
        while True:
            if bare[k] == "{":
                depth += 1
            elif bare[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        out.append((m.group(1), re.findall(r"\$(\w+)", m.group(2)), m.start(), m.end(), k))
    return out


def _assigned(code: str) -> set[str]:
    """Names assigned in `code`: $x = / += / ..., foreach ($x in ...), lower-cased."""
    names = set(re.findall(r"(?<![\w:`])\$(\w+)\s*[-+*/]?=(?!=)", code))
    names |= set(re.findall(r"foreach\s*\(\s*\$(\w+)\s+in\b", code, re.I))
    return {n.lower() for n in names}


def _script_level(text: str) -> tuple[str, set[str]]:
    """The code outside every function (header and body), and the names assigned there or
    declared as script parameters (lower-cased)."""
    code = _strip(text)
    spans = [(head, end) for _, _, head, _, end in _functions(text)]
    outside, last = [], 0
    for start, end in spans:
        outside.append(code[last:start])
        last = end + 1
    outside.append(code[last:])
    rest = "".join(outside)
    return rest, _assigned(rest) | {p.lower() for p in _params(text)}


def _params(text: str) -> list[str]:
    m = re.search(r"^param\((.*?)^\)", text, re.M | re.S)
    assert m, "model_check.ps1 has no param( ... ) block at the start of a line"
    return re.findall(r"\]\s*\$(\w+)", m.group(1))


def _argparse_flags(path: Path) -> set[str]:
    """Every "--flag" an add_argument call in the tool defines (all its parsers and groups)."""
    flags = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                    flags.add(arg.value)
    return flags


def _regions(text: str) -> dict[str, tuple[str, int, int]]:
    return {m.group(1): (m.group(2), m.start(2), m.end(2)) for m in REGION.finditer(text)}


def _single_quoted(text: str, var: str) -> str:
    """The value of `$var = '...'` (a single-quoted literal, '' unescaped)."""
    m = re.search(r"^\$" + var + r" = '((?:[^']|'')*)'", text, re.M)
    assert m, f"model_check.ps1 has no ${var} = '...' line"
    return m.group(1).replace("''", "'")


# ---------------------------------------------------------------------------
# PowerShell 5.1
# ---------------------------------------------------------------------------

def test_script_is_ascii_without_bom():
    # Windows PowerShell 5.1 reads a .ps1 without a BOM in the ANSI code page.
    raw = SCRIPT.read_bytes()
    assert raw.isascii(), [i for i, b in enumerate(raw) if b > 127][:5]


def test_no_pipeline_chain_operators():
    text = _text()
    assert "&&" not in text and "||" not in text


def test_no_addition_onto_a_generic_list():
    # On the owner's PC (Windows PowerShell 5.1, 2026-10-30)
    # 'foreach ($m in @($models) + @($championModel))', with $models a List[object], threw
    # 'Argument types do not match' and skipped the loop: the name checks never ran. The script
    # adds onto plain arrays only (@($list | ForEach-Object { $_ }) first, then +=).
    code = _strip(_text())
    lists = set(re.findall(r"\$(\w+)\s*=\s*New-Object\s+System\.Collections\.Generic\.List", code))
    assert lists >= {"models", "keepSpecs", "plan"}
    for name in lists:
        bad = re.compile(r"(@\(\s*)?\$" + name + r"\b\s*\)?\s*\+(?![+=])")
        hits = [m.group(0) for m in bad.finditer(code)]
        assert not hits, (name, hits)
    assert "$checkModels = @($models | ForEach-Object { $_ })" in code
    assert code.count("foreach ($m in $checkModels)") == 2


def test_writes_files_without_a_bom():
    text = _text()
    assert "New-Object System.Text.UTF8Encoding($false)" in text
    assert "UTF8Encoding($true)" not in text
    # PS 5.1's Out-File / Set-Content / Add-Content write UTF-16 or a BOM by default.
    assert not re.search(r"\b(Out-File|Set-Content|Add-Content)\b", _strip(text))
    assert "$ErrorActionPreference = 'Continue'" in text


def test_no_regex_match_operators():
    # They fill the automatic variable named like the -Matches parameter (at script level they
    # would overwrite the parameter, which is typed [int]).
    code = _strip(_text())
    assert not re.search(r"-[ci]?(not)?match\b", code, re.I)
    assert not re.search(r"\bswitch\s+-regex\b", code, re.I)
    assert "$matchCount = $Matches" in code
    rest, _ = _script_level(_text())
    # The parameter's declaration and the one read; no function reads it.
    assert len(re.findall(r"\$Matches\b", rest, re.I)) == 2
    assert len(re.findall(r"\$Matches\b", code, re.I)) == 2


def test_functions_read_script_variables_with_script_scope():
    text = _text()
    _, script_vars = _script_level(text)
    code = _strip(text)
    bad = []
    for name, params, _, start, end in _functions(text):
        body = code[start:end]
        local = _assigned(body) | {p.lower() for p in params}
        read = {v.lower() for v in re.findall(r"(?<![\w:`])\$(\w+)", body)}
        leaked = sorted((read - local - AUTOMATIC) & script_vars)
        if leaked:
            bad.append((name, leaked))
    assert not bad, f"read without $script: (dynamic scoping): {bad}"


def test_function_parameters_do_not_share_script_variable_names():
    text = _text()
    _, script_vars = _script_level(text)
    funcs = _functions(text)
    expected = {"Invoke-Check", "Write-Both", "New-KeepCopy", "Find-NameClashes"}
    assert expected <= {f[0] for f in funcs}
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

def test_param_block_and_help():
    text = _text()
    assert _params(text) == PARAMS
    helped = re.findall(r"^\.PARAMETER\s+(\w+)", text, re.M)
    assert sorted(helped) == sorted(PARAMS)
    for needle in ("[string[]]$Ckpt = @('runs\\tokens\\dt_big_hog_np\\best.pt')",
                   "[string]$Keeps = 'own,1.0'", "[string]$Rules = 'plain,live'",
                   "[int]$Delay = 21", "[int]$Matches = 300", "[int]$Workers = 4",
                   "$Opponent = 'runs\\tokens\\live_big\\ep01.pt'",
                   "$LeagueLog = 'runs\\league\\model_check.jsonl'",
                   "$Champion = 'runs\\tokens\\v2_hog_20k\\best.pt'",
                   "$Noise = 'data\\noise\\default_prior.json'"):
        assert needle in text, needle


def test_named_tools_and_scripts_exist():
    names = set(re.findall(r"(?<![\w.\\/-])((?:tools|scripts|sim)[\\/][\w.\\/-]+?\.(?:py|ps1))",
                           _text()))
    assert "tools\\rs_eval.py" in names and "tools\\rs_league_eval.py" in names
    missing = sorted(n for n in names if not (ROOT / n.replace("\\", "/")).is_file())
    assert not missing, missing


def test_flags_exist_in_the_tools_argparse():
    text = _text()
    regions = _regions(text)
    assert set(regions) == {"rs_eval", "rs_league_eval"}
    for tool, path in (("rs_eval", RS_EVAL), ("rs_league_eval", LEAGUE)):
        body = regions[tool][0]
        assert f"'tools\\{tool}.py'" in body
        flags = set(FLAG.findall(body))
        assert flags, tool
        unknown = sorted(flags - _argparse_flags(path))
        assert not unknown, f"{tool}: {unknown}"
    assert "'tools\\rs_league_eval.py', 'run'" in regions["rs_league_eval"][0]
    league_tree = ast.parse(LEAGUE.read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_parser"
               and n.args and getattr(n.args[0], "value", None) == "run"
               for n in ast.walk(league_tree))


def test_every_flag_literal_is_in_a_checked_region():
    text = _text()
    spans = [(start, end) for _, start, end in _regions(text).values()]
    outside = [m.group(1) for m in FLAG.finditer(text)
               if not any(start <= m.start() < end for start, end in spans)]
    assert not outside, outside


def test_decide_live_is_one_of_rs_evals_choices(monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import importlib
    rs_decide = importlib.import_module("rs_decide")
    assert "'--decide', 'live'" in _regions(_text())["rs_eval"][0]
    for node in ast.walk(ast.parse(RS_EVAL.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                and node.args and getattr(node.args[0], "value", None) == "--decide"):
            choices = next(k.value for k in node.keywords if k.arg == "choices")
            names = [getattr(e, "attr", None) for e in choices.elts]
            assert "live" in [getattr(rs_decide, n) for n in names]
            break
    else:
        pytest.fail("rs_eval.py defines no --decide")


def test_league_step_matches_third_model():
    ours = _regions(_text())["rs_league_eval"][0]
    third = THIRD.read_text(encoding="utf-8")
    m = re.search(r"Invoke-Step \"league:.*?'--log', \$leagueLog\)\)", third, re.S)
    assert m, "third_model.ps1's league step"
    # The same flags, --report too: both scripts write one report per matchup next to the log,
    # under the same name (test_league_report_goes_next_to_the_log_per_matchup), so neither
    # writes over step 1's third_model.report.json, rs_league_eval's default.
    assert set(FLAG.findall(ours)) == set(FLAG.findall(m.group(0)))
    assert "'--report', $leagueReport, '--log', $leagueLog))" in m.group(0)
    for needle in ("'--delay', '24'", "'--champion-games', '1000'", "'--anchor-games', '400'",
                   "'--full'", "'--noise', $Noise"):
        assert needle in ours and needle in m.group(0), needle
    text = _text()
    assert "[Math]::Min($Workers, 6)" in text and "[Math]::Min($Workers, 6)" in third
    assert "$Noise = 'data\\noise\\default_prior.json'" in third
    # The champion and the anchor under third_model.ps1's names and paths, so that one name
    # means one checkpoint in both logs: the same -Champion default, named the same way
    # (test_third_model_names_the_champion_as_resolve_model_does).
    default = "[ValidateNotNullOrEmpty()][string]$Champion = 'runs\\tokens\\v2_hog_20k\\best.pt'"
    assert default in third and default in text
    assert "'--champion', \"$championName=$championCkpt\"" in m.group(0)
    assert "('{0}={1}' -f $championModel.Name, $championModel.Rel)" in ours
    assert "$anchor = 'runs\\tokens\\live_big\\ep01.pt'" in third
    assert "\"live_big=$anchor\"" in third
    assert "$anchorModel = Resolve-Model 'runs\\tokens\\live_big\\ep01.pt'" in text
    assert "$anchorModel.Name = 'live_big'" in text


def test_runbook_commands_use_only_its_parameters():
    """The options after model_check.ps1 (to the end of its code span or line), and the code
    spans that start with an option on a line that names no other script."""
    lines = [ln for ln in RUNBOOK.read_text(encoding="utf-8").splitlines()
             if "model_check.ps1" in ln]
    assert any("scripts\\model_check.ps1" in ln for ln in lines)
    option = re.compile(r"(?<![\w-])-([A-Z][A-Za-z]+)\b")
    used = set()
    for ln in lines:
        for part in ln.split("model_check.ps1")[1:]:
            used |= set(option.findall(part.split("`")[0]))
        if len(set(re.findall(r"[\w-]+\.ps1", ln))) == 1:
            for span in re.findall(r"`(-[A-Z][^`]*)`", ln):
                used |= set(option.findall(span))
    assert {"League", "LeagueOnly", "LeagueKeep", "Ckpt", "Champion"} <= used
    assert used <= set(PARAMS), sorted(used - set(PARAMS))


def test_friendly_only_and_no_memory_reader():
    text = _text()
    for needle in ("--tap", "memread", "hcr_reader", "adb "):
        assert needle not in text, needle


# ---------------------------------------------------------------------------
# what the script reads back
# ---------------------------------------------------------------------------

def _rs_eval_line(wins: int, n: int, lo: float, hi: float, draws: int, cd: float,
                  plays: float, name: str) -> str:
    """rs_eval's result line, from its own f-string (the print that says 'wins (')."""
    for node in ast.walk(ast.parse(RS_EVAL.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"
                and node.args and isinstance(node.args[0], ast.JoinedStr)
                and "wins (" in ast.unparse(node.args[0])):
            expr = ast.Expression(body=node.args[0])
            break
    else:
        pytest.fail("rs_eval.py prints no result line")
    rows = [{"agent_plays": plays}] * n
    env = {"__builtins__": __builtins__, "name": name, "wins": wins, "rows": rows, "lo": lo,
           "hi": hi, "draws": draws, "cd": cd, "t0": 0.0,
           "time": types.SimpleNamespace(time=lambda: 586.0)}
    return eval(compile(expr, "rs_eval", "eval"), env)


def test_result_pattern_reads_rs_eval_lines():
    pattern = re.compile(_single_quoted(_text(), "evalPattern"))
    cases = [
        ((242, 300, 0.758, 0.847, 1, 0.79, 37.4, "dt_big_hog_np delay 21 vs reactive live_big"),
         ("242", "300", "80.7", "75.8", "84.7", "1", "+0.79", "37.4")),
        ((155, 300, 0.46, 0.573, 0, -0.15, 38.3,
          "live_big_hog delay 21 decide live vs reactive live_big"),
         ("155", "300", "51.7", "46.0", "57.3", "0", "-0.15", "38.3")),
        ((300, 300, 0.987, 1.0, 0, 2.5, 40.0, "x_wk100 delay 21 vs reactive live_big"),
         ("300", "300", "100.0", "98.7", "100.0", "0", "+2.50", "40.0")),
    ]
    for args, groups in cases:
        line = _rs_eval_line(*args)
        m = pattern.search(line)
        assert m and m.groups() == groups, line
    # The owner's log of 2026-10-30, as printed.
    logged = ("dt_big_hog_wk100 delay 21 vs reactive live_big: 204/300 wins (68.0%, 95% [62.5%, "
              "73.0%]) 0 draws, crown diff +0.51, agent plays/match 43.3 (573s)")
    assert pattern.search(logged).groups() == ("204", "300", "68.0", "62.5", "73.0", "0", "+0.51",
                                               "43.3")
    real = ("  real result of the Hog player in these matches: 147/300 (49.0%); unfinished in "
            "engine: 0")
    assert not pattern.search(real)


def test_gate_pattern_reads_format_report(monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import importlib
    LE = importlib.import_module("rs_league_eval")
    text = _text()
    m = re.search(r"\[regex\]::Match\(\$leagueLine, '((?:[^']|'')*)'\)", text)
    assert m, "model_check.ps1's gate-line pattern"
    gate = re.compile(m.group(1).replace("''", "'"))
    cand, champ, anchor = "dt_big_hog_np_wk100", "live_big_hog", "live_big"

    def g(a, b, winner, pairing, holder):
        return LE.Game(key=f"{a}|{b}|{pairing}", teams=(a, b), winner=winner, pairing=pairing,
                       holder=holder, protocol="P", split="val")

    games = [g(cand, champ, 0, f"val:p{i:04d}", cand) for i in range(4)]
    for i in range(2):
        pairing = f"val:p{i:04d}"
        games += [g(cand, anchor, 0, pairing, cand), g(champ, anchor, 1, pairing, champ)]
    config = LE.GateConfig(champion_games=4, anchor_games=2, n_boot=20)
    rep = LE.report(games, candidates=[cand], champion=champ, anchors=[anchor], config=config)
    lines = LE.format_report(rep).splitlines()
    found = [(x.group(1), x.group(2), x.group(3)) for x in map(gate.search, lines) if x]
    assert found == [(cand, rep["gates"][cand]["decision"].upper(), champ)]
    # The block the summary copies: from "league evaluation:" to "best:".
    assert lines[0].startswith("league evaluation:")
    assert any(x.startswith("best:") for x in lines)
    source = LEAGUE.read_text(encoding="utf-8")
    for needle in ("\\nprotocol: ", "\\npairs: ", "\\nworkers: ", 'print(f"report: '):
        assert needle in source, needle
    for needle in ("StartsWith('league evaluation:')", "StartsWith('best:')",
                   "StartsWith('protocol:')", "StartsWith('pairs:')", "StartsWith('workers:')",
                   "StartsWith('report:')"):
        assert needle in text, needle


def test_keep_probe_and_copy_args_json(tmp_path):
    pytest.importorskip("numpy")
    text = _text()
    probe = _single_quoted(text, "keepProbe")
    assert '"' not in probe                 # PowerShell 5.1 passes double quotes unescaped
    compile(probe, "keepProbe", "exec")
    # The copy's args.json, put together as the script writes it.
    m = re.search(r"\$json = '([^']*)' \+ \$keepText \+ '([^']*)' \+ "
                  r"\$SourceModel\.Rel\.Replace\('\\', '/'\) \+ '([^']*)'", text)
    assert m, "model_check.ps1's args.json line"
    copy_json = m.group(1) + "1.0" + m.group(2) + "runs/tokens/dt_big_hog_np/best.pt" + m.group(3)
    doc = json.loads(copy_json)
    assert doc["wait_keep"] == "1.0" and doc["made_by"] == "scripts/model_check.ps1"
    assert '"made_by": "scripts/model_check.ps1"' in copy_json    # what marks a folder as a copy
    copy_re = re.search(r"\[regex\]::IsMatch\(\$old\.Trim\(\), '([^']*)'\)", text).group(1)
    assert re.search(copy_re, '{"wait_keep": "1.0"}')             # dt_diagnose.ps1's copies
    assert not re.search(copy_re, '{"wait_keep": "0.25", "epochs": "3"}')

    def probe_reads(args_json):
        folder = tmp_path / f"m{len(list(tmp_path.iterdir()))}"
        folder.mkdir()
        if args_json is not None:
            (folder / "args.json").write_text(args_json, encoding="utf-8")
        out = subprocess.run([sys.executable, "-c", probe, str(folder / "best.pt")], cwd=ROOT,
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        return out.stdout.strip().splitlines()[-1]

    assert probe_reads(copy_json) == "1.0"
    assert probe_reads('{"wait_keep": "0.5"}') == "0.5"
    assert probe_reads(json.dumps({"wait_keep": "0.25", "epochs": "3"})) == "0.25"
    assert probe_reads(None) == "none"
    assert probe_reads('{"epochs": "3"}') == "none"


# ---------------------------------------------------------------------------
# copies, names and clashes
# ---------------------------------------------------------------------------

def _function(text: str, name: str) -> tuple[str, str]:
    """(the raw body, the body with comments and single-quoted strings blanked) of `name`."""
    code = _strip(text)
    for fname, _, _, start, end in _functions(text):
        if fname == name:
            return text[start:end], code[start:end]
    pytest.fail(f"no function {name}")


def test_keep_copy_never_replaces_another_or_a_changed_source():
    # A copy may be the live model (vision_play --ckpt runs\tokens\<name>_wk100\best.pt): a
    # retrained source, or another checkpoint of the same name, must stop the run, not silently
    # replace the copy's weights.
    raw, body = _function(_text(), "New-KeepCopy")
    source = re.search(r"if \(\$oldSource -and \$oldSource -ine \$sourceRel\) \{\s*Stop-Run", body)
    digest = re.search(r"if \(\$oldDigest -ne \$sourceDigest\) \{\s*Stop-Run", body)
    assert source and digest
    assert "$oldDigest = Get-CkptDigest $bestPath" in body
    assert "$sourceDigest = Get-CkptDigest $SourceModel.Full" in body
    assert "ConvertFrom-Json).copy_of" in body
    assert "$sourceRel = $SourceModel.Rel.Replace(" in body
    # The one Copy-Item runs only when the folder holds no best.pt, after both checks; the
    # args.json is written after them too.
    assert body.count("Copy-Item") == 1
    copy = re.search(r"if \(-not \(Test-Path -LiteralPath \$bestPath -PathType Leaf\)\) \{\s*"
                     r"Copy-Item -LiteralPath \$SourceModel\.Full -Destination \$bestPath", body)
    assert copy
    assert max(source.start(), digest.start()) < copy.start() < body.index("WriteAllText")
    # The copy_of it compares is the one it writes (forward slashes).
    assert "copy_of" in raw and "$SourceModel.Rel.Replace('\\', '/')" in raw


def test_two_checkpoints_of_one_run_never_share_a_name():
    rest, _ = _script_level(_text())
    # $checkModels: the checkpoints and the champion as one plain array (44e7c02; see
    # test_no_addition_onto_a_generic_list).
    m = re.search(r"\$byName = @\{\}\s*foreach \(\$m in \$checkModels\) \{"
                  r"\s*\$nameKey = \$m\.Name\.ToLowerInvariant\(\)\s*"
                  r"if \(\$byName\.ContainsKey\(\$nameKey\) -and \$byName\[\$nameKey\]\.Full -ine "
                  r"\$m\.Full\) \{\s*Stop-Run", rest)
    assert m
    # Before the first copy is made (Add-Plan, Get-EvalModel and New-KeepCopy run at plan time).
    assert m.start() < rest.index("Add-Plan $model $spec")


def test_name_clashes_stop_only_where_the_league_would_refuse_or_for_the_candidate():
    text = _text()
    raw, _ = _function(text, "Find-NameClashes")
    assert "Fatal = ($isOwn -or $player.Role -eq 'candidate')" in raw
    rest, _ = _script_level(text)
    assert "$fatalClashes = @($clashes | Where-Object { $_.Fatal })" in rest
    assert "@($clashes | Where-Object { -not $_.Fatal })" in rest
    # An RL run's gate.jsonl is part of its resumable state: no advice moves a league log.
    assert "Move-Item <log>" not in text


def test_verdict_lines_name_the_champion_rule_and_a_league_only_rerun():
    text = _text()
    assert "is the new champion: every later candidate plays it (-Champion {2})" in text
    assert "after a section 17 dry run" in text
    # A FAIL keeps the champion; which model the live runner plays is the runbook's business.
    assert "league verdict: {0}. Keep {1} as the champion." in text
    assert "in the live runner. The beats_champion" not in text
    assert "Run it again with -LeagueOnly and the same -Ckpt, -LeagueKeep, -LeagueName, -LeagueLog and -Champion" in text
    # 'own' refusal: the live runner refuses taps only when args.json is missing.
    assert "the live runner refuses taps only when args.json is missing" in text


# ---------------------------------------------------------------------------
# scripts/dt_diagnose.ps1
# ---------------------------------------------------------------------------

def test_dt_diagnose_summary_pattern_is_not_shadowed():
    # 2026-10-30: Invoke-WaitBias's -Keep ('0.5', '1.0') stood in for the script's $keep inside
    # Invoke-Check, so the summary lost the wait_keep 0.5 result line and kept engine: lines.
    text = DT_DIAGNOSE.read_text(encoding="utf-8")
    assert text.isascii()
    _, script_vars = _script_level(text)
    funcs = _functions(text)
    assert {"Invoke-Check", "Invoke-WaitBias"} <= {f[0] for f in funcs}
    clashes = [(name, p) for name, params, _, _, _ in funcs for p in params
               if p.lower() in script_vars]
    assert not clashes, clashes
    assert "keep" in script_vars
    _, body = _function(text, "Invoke-Check")
    assert "$line -match $script:keep" in body
    assert not re.search(r"(?<![\w:`])\$keep\b", body, re.I)
    assert "$script:summary.Add(" in body and "$script:failed.Add(" in body


# ---------------------------------------------------------------------------
# 2026-10-30: 4 workers, the per-matchup report, the champion in third_model.ps1
# ---------------------------------------------------------------------------

RL_TRAIN = ROOT / "scripts" / "rl_train.ps1"
TRAIN_LOOP = ROOT / "tools" / "rl" / "train_loop.py"
FIRST = ROOT / "scripts" / "first_model.ps1"
SECOND = ROOT / "scripts" / "second_model.ps1"
#: Each engine script and the line in its param block that sets the -Workers default.
WORKER_DEFAULTS = {SCRIPT: "[ValidateRange(1, 64)][int]$Workers = 4",
                   THIRD: "[ValidateRange(1, 64)][int]$Workers = 4",
                   DT_DIAGNOSE: "[ValidateRange(1, 64)][int]$Workers = 4",
                   RL_TRAIN: "[ValidateRange(0, 64)][int]$Workers = 4",
                   FIRST: "[int]$Workers = 4",
                   SECOND: "[int]$Workers = 4"}


def _help_block(text: str, name: str) -> str:
    """The text of a .PARAMETER block, up to the next one or the end of the help."""
    m = re.search(r"^\.PARAMETER " + name + r"\n(.*?)(?=^\.PARAMETER |^#>)", text, re.M | re.S)
    assert m, f".PARAMETER {name}"
    return " ".join(m.group(1).split())


def test_engine_scripts_default_to_4_workers():
    # The PC froze 6 minutes into an 8-worker v2 rs_eval run on 2026-10-30; runs at 4 since were
    # stable and as fast (runbook, "If something breaks", "If the PC freezes").
    for path, line in WORKER_DEFAULTS.items():
        text = path.read_text(encoding="utf-8")
        assert text.isascii(), path.name
        m = re.search(r"^param\((.*?)^\)", text, re.M | re.S)
        assert m and line in m.group(1), path.name
        assert len(re.findall(r"\]\$Workers = ", m.group(1))) == 1, path.name
        block = _help_block(text, "Workers")
        assert "default 4" in block and '"If the PC freezes"' in block, (path.name, block)
        assert "default 8" not in block and "default 0" not in block, (path.name, block)
        # The examples in the help start no 8-worker run.
        assert "-Workers 8" not in text, path.name
    # A -Workers 0 still means the RL driver's own default.
    assert "0: the driver's own" in _help_block(RL_TRAIN.read_text(encoding="utf-8"), "Workers")
    assert "if ($Workers -gt 0) { $argList += @('--workers', \"$Workers\") }" in \
        RL_TRAIN.read_text(encoding="utf-8")
    # The runbook's commands of the scripts and tools start no 8-worker job either.
    runbook = RUNBOOK.read_text(encoding="utf-8")
    assert not re.search(r"--workers 8\b|-Workers 8\b|default 8\b", runbook)
    for stale in ("uses less RAM", "re-run with `--workers 6`"):
        assert stale not in runbook, stale


def test_rl_bench_measures_at_most_the_workers_asked_for():
    # Without --bench-workers, train_loop's bench plays 1, half and all of what RAM and CPU allow
    # (bench_counts), about 9 workers on the 16 GB PC: more than the 8 that froze it. rl_train.ps1
    # passes 1, half and all of -Workers instead (1,2,4 by default); -Workers 0 leaves it.
    text = RL_TRAIN.read_text(encoding="utf-8")
    m = re.search(r"^if \(\$Bench\) \{\n(.*?)^\}", text, re.M | re.S)
    assert m, "rl_train.ps1's bench arguments"
    block = m.group(1)
    assert "$argList += @('--bench', (\"{0}m\" -f $BenchMinutes))" in block
    counts_line = ("$benchCounts = @(@(1, [Math]::Max(1, [int][Math]::Floor($Workers / 2.0)), "
                   "$Workers) | Sort-Object -Unique)")
    assert re.search(r"if \(\$Workers -gt 0\) \{\s*" + re.escape(counts_line) + r"\s*"
                     + re.escape("$argList += @('--bench-workers', ($benchCounts -join ','))")
                     + r"\s*\}", block)
    assert text.index("'--bench-workers'") < text.index("$code = Invoke-Py $title $RG $argList")

    def ps_counts(workers):        # the line above: Floor of a positive int / 2 is //, -Unique sorts
        return ",".join(str(c) for c in sorted({1, max(1, workers // 2), workers}))

    assert ps_counts(4) == "1,2,4"            # the default
    assert [ps_counts(w) for w in (1, 2, 3, 5, 6)] == ["1", "1,2", "1,3", "1,2,5", "1,3,6"]
    for w in range(1, 65):
        assert max(int(c) for c in ps_counts(w).split(",")) == w
    # train_loop reads them as its --bench-workers (comma-separated counts) before its own caps.
    assert "--bench-workers" in _argparse_flags(TRAIN_LOOP)
    loop = TRAIN_LOOP.read_text(encoding="utf-8")
    body = loop[loop.index("    def bench_counts(self)"):loop.index("    def bench(self)")]
    assert body.index("if a.bench_workers:") < body.index("R.ram_check(")
    assert ('counts = sorted({int(x) for x in str(a.bench_workers).split(",") if x.strip()})'
            in body)
    assert sorted({int(x) for x in ps_counts(4).split(",") if x.strip()}) == [1, 2, 4]
    help_block = _help_block(text, "Bench")
    assert "1, 2 and 4 by default" in help_block


def _ps_number(expr: str, env: dict) -> float:
    """A PowerShell arithmetic expression of this script ([Math]::Max / Ceiling / Round, $names)
    evaluated in Python. .NET's Math.Round rounds half to even, as Python's round does."""
    py = (expr.replace("[Math]::Max", "max").replace("[Math]::Ceiling", "math.ceil")
          .replace("[Math]::Round", "round"))
    py = re.sub(r"\$(\w+)", r"\1", py)
    return eval(py, {"__builtins__": {}, "max": max, "round": round, "math": math}, env)


def test_plan_line_counts_the_times_measured_at_4_workers():
    text = _text()
    rest, _ = _script_level(text)
    # rs_eval about 10-13 min per 300 matches at 4 workers (8 were no faster), the league about
    # 2.2 s a game at 4: 1,000 games in 37 min, 1,400 in about 51, 1,800 in about 66.
    assert "8.0 / $Workers" not in text and "at 8 workers){2}" not in text
    assert "about 10-13 min per 300 matches at 4 workers, measured 2026-10-30" in text
    exprs = {}
    for name in ("evalScale", "evalLow", "evalHigh", "leagueScale"):
        m = re.search(r"^\$" + name + r" = (.+)$", rest, re.M)
        assert m, name
        exprs[name] = m.group(1)
    m = re.search(r"\$leagueText = \('[^']*' -f\s*(\[Math\]::Round\([^\n]*?\)), "
                  r"(\[Math\]::Round\([^\n]*\))\)\s*\}", text)
    assert m, "the league's part of the plan line"
    games_1800, games_1400 = m.group(1), m.group(2)
    assert "1800 * 2.2" in games_1800 and "1400 * 2.2" in games_1400

    def plan(runs, matches, workers):
        env = {"toRun": runs, "matchCount": matches, "Workers": workers,
               "leagueWorkers": min(workers, 6)}
        for name in ("evalScale", "evalLow", "evalHigh", "leagueScale"):
            env[name] = _ps_number(exprs[name], env)
        return (env["evalLow"], env["evalHigh"], _ps_number(games_1800, env),
                _ps_number(games_1400, env))

    assert plan(8, 300, 4) == (80, 104, 66, 51)          # the default run: about 1.5 h, then the league
    assert plan(8, 300, 6) == (80, 104, 66, 51)          # more workers: not counted as faster
    assert plan(8, 300, 8) == (80, 104, 66, 51)
    assert plan(8, 300, 2) == (160, 208, 132, 103)       # fewer: slower in proportion
    assert plan(1, 150, 4) == (5, 7, 66, 51)
    assert plan(0, 300, 4)[:2] == (0, 0)                 # -LeagueOnly


def test_league_report_goes_next_to_the_log_per_matchup(monkeypatch):
    text = _text()
    region = _regions(text)["rs_league_eval"][0]
    assert "'--report', $leagueReportModel.Rel" in region
    m = re.search(r"\$leagueReportFile = '([^']*)' -f \[System\.IO\.Path\]::"
                  r"GetFileNameWithoutExtension\(\$leagueLogModel\.Full\), \$candName, "
                  r"\$championModel\.Name", text)
    assert m, "the report's file name"
    pattern = m.group(1)
    assert ("$leagueReportModel = Resolve-Model (Join-Path (Split-Path -Parent "
            "$leagueLogModel.Full) $leagueReportFile)") in text
    # Named at plan time, before the clash checks and long before the league step uses it.
    rest, _ = _script_level(text)
    assert rest.index("$leagueReportModel = ") < rest.index("$clashes = @(Find-NameClashes")
    assert rest.index("$leagueReportModel = ") < rest.index("$leagueArgs = @(")

    def name(log_stem, cand, champ):        # PowerShell's -f with {0} {1} {2}, as str.format
        return pattern.format(log_stem, cand, champ)

    # Section 20 step 2's v2_hog against the champion on the third model's log, and the default.
    assert name("third_model", "v2_hog", "v2_hog_20k") == "third_model.v2_hog_vs_v2_hog_20k.report.json"
    assert name("model_check", "dt_big_hog_np", "v2_hog_20k") == \
        "model_check.dt_big_hog_np_vs_v2_hog_20k.report.json"
    # third_model.ps1 names its league's report the same way on its own log (a literal there: its
    # log is fixed), so one matchup on that log has one report file, whichever script wrote it.
    third = _third()
    t = re.search(r"^\$leagueReport = '([^']*)' -f \$candidateName, \$championName$", third, re.M)
    assert t, "third_model.ps1's report name"
    assert "$leagueLog = 'runs\\league\\third_model.jsonl'" in third
    for cand, champ in (("v2_hog", "v2_hog_20k"), ("v2_hog_full", "dt_big_hog_np")):
        assert t.group(1).format(cand, champ) == \
            "runs\\league\\" + name("third_model", cand, champ)
    third_code = _strip(third)
    assert third_code.index("$leagueReport = ") < third_code.index("Invoke-Step '")
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import argparse
    import importlib
    LE = importlib.import_module("rs_league_eval")
    log = Path("runs/league/third_model.jsonl")
    # Without --report the league writes next to its log: step 1's third_model.report.json.
    assert LE._report_path(argparse.Namespace(report=None), log).name == "third_model.report.json"
    assert name("third_model", "v2_hog", "v2_hog_20k") != "third_model.report.json"
    args = LE.build_parser().parse_args(["run", "--candidate", "a=x.pt", "--champion", "b=y.pt",
                                         "--report", "runs/league/r.report.json"])
    assert LE._report_path(args, log) == Path("runs/league/r.report.json")


def _third() -> str:
    return THIRD.read_text(encoding="utf-8")


def test_third_model_champion_parameter_and_help():
    third = _third()
    params = _params(third)
    assert params[-1] == "Champion" and params.count("Champion") == 1
    helped = re.findall(r"^\.PARAMETER\s+(\w+)", third, re.M)
    assert sorted(helped) == sorted(params)
    block = _help_block(third, "Champion")
    assert "default runs\\tokens\\v2_hog_20k\\best.pt" in block
    assert "must not be the candidate" in block
    # live_big_hog is a former champion: no help or comment calls it the live model or the best.
    for stale in ("the live model and the best so far", "beats live_big_hog:",
                  "hard-wired"):
        assert stale not in third, stale
    # PowerShell names ignore case: $champion would be the -Champion parameter, so no script code
    # assigns it (the path the tools get is $championCkpt); the one match is the default.
    rest, assigned = _script_level(third)
    assert re.findall(r"(?<![\w:`])\$champion\s*[-+*/]?=(?!=)", rest, re.I) == ["$Champion ="]
    assert {"championckpt", "championname", "championfull"} <= assigned


def test_third_model_names_the_champion_as_resolve_model_does():
    """Resolve-Model's lines, renamed, are third_model.ps1's: one path form and one league name
    for a checkpoint in both scripts' logs."""
    raw, _ = _function(_text(), "Resolve-Model")
    rename = {"PathText": "Champion", "clean": "championText", "joined": "championJoined",
              "full": "championFull", "prefix": "rootPrefix", "rel": "championCkpt",
              "file": "championFile", "folder": "championFolder", "modelName": "championName",
              "script:root": "root"}
    lines = [ln.strip() for ln in raw.splitlines()
             if re.match(r"\s*\$(clean|joined|full|prefix|rel|file|folder|modelName) = ", ln)]
    assert len(lines) == 8
    third = _third()
    for line in lines:
        ours = re.sub(r"\$(script:root|\w+)", lambda m: "$" + rename.get(m.group(1), m.group(1)),
                      line)
        assert ours in third, ours

    # The same rules in Python (ntpath: the PC's paths), for the names the runbook uses.
    import ntpath

    def resolve(text, root="C:\\hcr"):
        clean = text.strip().strip("\"'")
        full = ntpath.normpath(clean if ntpath.isabs(clean) else ntpath.join(root, clean))
        prefix = root.rstrip("\\") + "\\"
        rel = full[len(prefix):] if full.lower().startswith(prefix.lower()) else full
        file, folder = ntpath.basename(full), ntpath.basename(ntpath.dirname(full))
        name = folder if file.lower() == "best.pt" else f"{folder}_{ntpath.splitext(file)[0]}"
        return rel, name

    assert resolve("runs\\tokens\\v2_hog_20k\\best.pt") == ("runs\\tokens\\v2_hog_20k\\best.pt",
                                                            "v2_hog_20k")
    assert resolve("C:\\hcr\\runs\\tokens\\live_big_hog\\best.pt") == \
        ("runs\\tokens\\live_big_hog\\best.pt", "live_big_hog")
    assert resolve("runs/rl/hog_rl1/it0010.pt") == ("runs\\rl\\hog_rl1\\it0010.pt",
                                                    "hog_rl1_it0010")
    assert resolve("D:\\keep\\v2_hog_20k\\best.pt") == ("D:\\keep\\v2_hog_20k\\best.pt",
                                                        "v2_hog_20k")


def test_third_model_checks_the_champion_before_any_step():
    third = _third()
    code = _strip(third)
    first_step = code.index("Invoke-Step '")
    checks = [
        r"if \(-not \(Test-Path -LiteralPath \$championFull -PathType Leaf\)\) \{\s*Stop-Run",
        r"if \(-not \[regex\]::IsMatch\(\$championName, '[^']*'\)\) \{\s*Stop-Run",
        r"if \(\$championFull -ieq \$candidateFull -or \$championName -ieq \$candidateName\) "
        r"\{\s*Stop-Run",
        r"if \(\$championFull -ieq \$anchorFull -or \$championName -ieq '[^']*'\) \{\s*Stop-Run",
        # Not this run's general model either: the run retrains it in place (-Champion
        # runs\tokens\v2_big\best.pt), and an all-deck model is no Hog 2.6 champion.
        r"if \(\$championFull -ieq \$generalFull -or \$championName -ieq \$generalName\) "
        r"\{\s*Stop-Run",
    ]
    for pattern in checks:
        m = re.search(pattern, code)
        assert m and m.start() < first_step, pattern
    assert "$candidateFull = [System.IO.Path]::GetFullPath((Join-Path $root $candidate))" in third
    assert "$anchorFull = [System.IO.Path]::GetFullPath((Join-Path $root $anchor))" in third
    assert "$generalFull = [System.IO.Path]::GetFullPath((Join-Path $root $general))" in third
    assert code.index("$general = ") < code.index("$generalFull = ")
    assert "-ieq 'live_big')" in third              # the anchor's league name
    # The checks use the champion's own full path, not Join-Path $root (an absolute -Champion).
    m = re.search(r"^\$need = @\(([^)]*)\)", third, re.M)
    assert m and "champion" not in m.group(1).lower()


def test_third_model_checks_the_champions_games_in_the_league_log(tmp_path, monkeypatch):
    """rs_league_eval refuses a player whose games in the log were played under another identity
    (check_identities: the AgentSpec with the checkpoint path as given, and the file's digest),
    but only at the league step, after rows, training and eval. third_model.ps1 reads the log's
    run headers at the start, as model_check.ps1's Find-NameClashes does, and stops on another
    file (digest) or another spelling of the path for the champion's name."""
    third = _third()
    code = _strip(third)
    first_step = code.index("Invoke-Step '")
    seq = ["$leagueLogFull = Join-Path $root $leagueLog",
           "if (Test-Path -LiteralPath $leagueLogFull) {",
           "$championDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath $championFull).Hash"
           ".ToLowerInvariant().Substring(0, 16)",
           "$runHeaders = @(Select-String -LiteralPath $leagueLogFull -SimpleMatch -Pattern '",
           "foreach ($hit in $runHeaders) {",
           "try { $header = $hit.Line | ConvertFrom-Json } catch { $header = $null }",
           "foreach ($prop in $header.players.PSObject.Properties) {",
           "if ($prop.Name -cne $championName) { continue }",
           "$oldCkpt = [string]$prop.Value.ckpt",
           "$oldDigest = [string]$prop.Value.ckpt_digest"]
    at = 0
    for needle in seq:
        at = code.index(needle, at)
        assert at < first_step, needle
    for pattern in (r"if \(\$oldDigest -and \$oldDigest -ne \$championDigest\) \{\s*Stop-Run",
                    r"if \(\$oldCkpt -cne \$championCkpt\) \{\s*Stop-Run"):
        m = re.search(pattern, code)
        assert m and at < m.start() < first_step, pattern
    assert "-SimpleMatch -Pattern '\"kind\":\"league_run\"'" in third
    # The digest as model_check.ps1 takes it (Get-CkptDigest) and rs_league_eval's ckpt_digest.
    assert ("(Get-FileHash -Algorithm SHA256 -LiteralPath $DigestPath).Hash.ToLowerInvariant()"
            ".Substring(0, 16)") in _text()
    league = LEAGUE.read_text(encoding="utf-8")
    for needle in ('KIND_RUN, KIND_GAME, KIND_ERROR = "league_run", "league_game", "league_error"',
                   '"players": {n: {**i, "id": ids[n]} for n, i in idents.items()}',
                   'return {**agent.to_dict(), "ckpt_digest": digest}',
                   "return h.hexdigest()[:16]"):
        assert needle in league, needle

    import hashlib
    import ntpath

    def resolve(text, root="C:\\hcr"):          # Resolve-Model's Rel and Name (ntpath: the PC)
        clean = text.strip().strip("\"'")
        full = ntpath.normpath(clean if ntpath.isabs(clean) else ntpath.join(root, clean))
        prefix = root.rstrip("\\") + "\\"
        rel = full[len(prefix):] if full.lower().startswith(prefix.lower()) else full
        file, folder = ntpath.basename(full), ntpath.basename(ntpath.dirname(full))
        return rel, folder if file.lower() == "best.pt" else f"{folder}_{ntpath.splitext(file)[0]}"

    def refusal(lines, champion_text, digest):
        """The script's loop: None, or why it stops."""
        ckpt, name = resolve(champion_text)
        for line in lines:
            if '"kind":"league_run"' not in line:
                continue
            for player, value in json.loads(line)["players"].items():
                if player != name:                      # -cne: names are case-sensitive
                    continue
                old_ckpt, old_digest = str(value.get("ckpt") or ""), str(value.get("ckpt_digest") or "")
                if old_digest and old_digest != digest:
                    return "file"
                if old_ckpt != ckpt:
                    return "path"
        return None

    blob = b"v2_hog_20k weights"
    d20k = hashlib.sha256(blob).hexdigest()[:16]

    def header(players):
        return json.dumps({"kind": "league_run", "players": players}, separators=(",", ":"))

    step1 = header({"v2_hog_20k": {"ckpt": "runs\\tokens\\v2_hog_20k\\best.pt", "ckpt_digest": d20k},
                    "live_big_hog": {"ckpt": "runs\\tokens\\live_big_hog\\best.pt",
                                     "ckpt_digest": "1" * 16},
                    "live_big": {"ckpt": "runs\\tokens\\live_big\\ep01.pt", "ckpt_digest": "2" * 16}})
    game = json.dumps({"kind": "league_game", "teams": ["v2_hog_20k", "live_big"]},
                      separators=(",", ":"))
    log = [step1, game]
    # Step 2 with the default: the same path and file as step 1's candidate games.
    assert refusal(log, "runs\\tokens\\v2_hog_20k\\best.pt", d20k) is None
    assert refusal(log, "C:\\hcr\\runs\\tokens\\v2_hog_20k\\best.pt", d20k) is None
    # Another spelling of the path: the league compares it as written.
    assert refusal(log, "Runs\\Tokens\\v2_hog_20k\\best.pt", d20k) == "path"
    assert refusal(log, "D:\\keep\\v2_hog_20k\\best.pt", d20k) == "path"
    # The same path, another file (retrained or restored from elsewhere).
    assert refusal(log, "runs\\tokens\\v2_hog_20k\\best.pt", "f" * 16) == "file"
    # A champion new to the log, or a name the league tells apart (case-sensitive), plays.
    assert refusal(log, "runs\\tokens\\dt_big_hog_np\\best.pt", "3" * 16) is None
    assert refusal(log, "runs\\tokens\\V2_hog_20k\\best.pt", "f" * 16) is None
    assert refusal([], "runs\\tokens\\v2_hog_20k\\best.pt", d20k) is None

    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import importlib
    LE = importlib.import_module("rs_league_eval")
    ck = tmp_path / "best.pt"
    ck.write_bytes(blob)
    assert LE.ckpt_digest(str(ck)) == d20k                 # Get-FileHash, lower-cased, 16 digits
    # The path's spelling alone changes the identity the league compares (check_identities).
    same = {"ckpt": "runs\\tokens\\v2_hog_20k\\best.pt", "ckpt_digest": d20k}
    assert LE.identity_digest(same) != LE.identity_digest(
        {**same, "ckpt": "Runs\\Tokens\\v2_hog_20k\\best.pt"})
    assert LE.identity_digest(same) != LE.identity_digest({**same, "ckpt_digest": "f" * 16})


def test_third_model_plays_the_champion_in_the_reference_run_and_the_league():
    third = _third()
    runs = [ln for ln in third.splitlines() if ln.strip().startswith("Invoke-Run ")]
    assert len(runs) == 5
    reference = runs[-1]
    assert "(reference)" in reference and "$championName" in reference
    assert "'--ckpt', $championCkpt" in reference and "'--decide', 'live'" in reference
    m = re.search(r"Invoke-Step \"league: \$candidateName vs \$championName, anchor live_big\".*?"
                  r"'--log', \$leagueLog\)\)", third, re.S)
    assert m and "'--champion', \"$championName=$championCkpt\"" in m.group(0)
    assert "'--report', $leagueReport" in m.group(0)
    for line in third.splitlines():
        if re.match(r"\s*(Invoke-Run|Invoke-Step)\b", line) or "'--champion'" in line:
            assert "live_big_hog" not in line, line
    # The start line names the champion.
    assert "champion {9}={10}. Log: {11}" in third and "$championName, $championCkpt, $log" in third


def test_third_model_still_refuses_a_candidate_name_with_games_in_the_log():
    third = _third()
    code = third                    # raw: the quotes around the name are single-quoted strings
    m = re.search(r"if \(\$start -le 3\) \{\s*\$leaguePath = Join-Path \$root \$leagueLog\s*"
                  r"if \(\(Test-Path \$leaguePath\) -and \(Select-String -LiteralPath \$leaguePath "
                  r"-SimpleMatch -Quiet -Pattern \('\"' \+ \$candidateName \+ '\"'\)\)\) \{\s*"
                  r"Stop-Run", code)
    assert m and m.start() < code.index("Invoke-Step '")
    # The log is JSON without spaces (rs_league_eval append_records), so the quoted name matches
    # one player exactly: the champion v2_hog_20k's games do not stop the whole-corpus run's v2_hog,
    # and v2_hog_20k's own name (a -Tag 20k run) is refused.
    header = json.dumps({"kind": "league_run", "players": {"v2_hog_20k": {"ckpt": "x"},
                                                           "live_big": {"ckpt": "y"}}},
                        separators=(",", ":"))
    game = json.dumps({"kind": "league_game", "teams": ["v2_hog_20k", "live_big"]},
                      separators=(",", ":"))
    for line in (header, game):
        assert '"v2_hog"' not in line and '"v2_hog_20k"' in line
    source = LEAGUE.read_text(encoding="utf-8")
    assert 'f.write(json.dumps(rec, separators=(",", ":")) + "\\n")' in source
