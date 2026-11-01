<#
.SYNOPSIS
    A/B of the two decision rules in the simulator, unattended: the wait floor and the stray
    cheap-troop guard (tools\rs_decide.py DECISION RULES), one checkpoint with the rules on
    against itself with them off, 1,000 league games, and a plain verdict: stronger, no
    difference or weaker, with the score and its 95% interval.

.DESCRIPTION
    tools\rs_league_eval.py run plays the candidate, -Ckpt with the rules on, against the
    champion, the same -Ckpt with the rules off, under two league names: the champion is named
    after the checkpoint's folder (v2_hog for the default -Ckpt), the candidate <name>_rules
    (both rules; -Variant floor: <name>_floor, the wait floor only; -Variant stray:
    <name>_stray, the stray guard only; -Variant all: the three, each against the champion).
    -Games games per candidate (default 1,000: every pairing on both seats and again with the
    decks exchanged), a 24-tick delay, the -Noise profile on every seat, the live decision rule
    (the live runner's: the cap rule, the lane guard), at most 6 workers. The league's log is
    -Log (default runs\league\decide_ab.jsonl, its own: not section 20's third_model.jsonl),
    its frozen pairings go next to it, and a rerun plays only the games the log lacks; the
    league refuses a name that the log holds with other rules or another checkpoint, before
    anything is played.

    Then tools\decide_ab.py reads the log and prints, per candidate, its score against the
    champion with two 95% intervals (Wilson over the games, the bootstrap over the pairings),
    plays and elixir leak a game, how often the rules fired, and the verdict:
      STRONGER       both intervals above 50%
      NO DIFFERENCE  an interval holds 50%
      WEAKER         both intervals below 50%
    The league's own report block comes first; its gate there uses a 50% lower bound
    (--lower-bound 0.5), the Wilson half of STRONGER. The verdict line is the one to read.

    Everything printed also goes to reports\decide_ab_<time>.log. Exit code 0 when every
    verdict is complete, 2 when a verdict has fewer games than -Games (run the same command
    again), 1 when a step failed. Only the CPU venv (.venvs\royalegym) runs; close MuMu and the
    browser first. Nothing here touches the emulator.

        powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\decide_ab.ps1

    Time: the league played about 2.2 s a game at 4 workers (2026-10-30), so 1,000 games take
    about 40 minutes and -Variant all (3,000 games) about 2 hours.

.PARAMETER Ckpt
    The checkpoint both players play (default runs\tokens\v2_hog\best.pt). Relative paths are
    read from the repo root.

.PARAMETER Variant
    Which rules the candidate has: both (default, <name>_rules), floor (the wait floor only,
    <name>_floor), stray (the stray guard only, <name>_stray), or all (the three candidates in
    one run, -Games games each).

.PARAMETER WaitFloor
    The wait floor (default 0.95): from 0.5 to 1 with at most two decimals. A floor other than
    0.95 adds _wfNNN to the candidate's name (wf090 = 0.90), so its games stay apart.

.PARAMETER Games
    Games per candidate against the champion (default 1000). A larger number than an earlier
    run on the same -Log needs more pairings than its frozen pairings file holds: give another
    -Log.

.PARAMETER Workers
    Worker processes (default 4; the league uses at most 6, and lowers the count itself when
    free memory is short).

.PARAMETER Log
    The league's JSONL log (default runs\league\decide_ab.jsonl). The report goes next to it as
    <log name>.report.json.

.PARAMETER Noise
    The perception-noise profile of every seat (default data\noise\default_prior.json, the
    prior, as the other leagues play).
#>
param(
    [ValidateNotNullOrEmpty()][string]$Ckpt = 'runs\tokens\v2_hog\best.pt',
    [ValidateSet('both', 'floor', 'stray', 'all')][string]$Variant = 'both',
    [ValidateNotNullOrEmpty()][string]$WaitFloor = '0.95',
    [ValidateRange(100, 100000)][int]$Games = 1000,
    [ValidateRange(1, 64)][int]$Workers = 4,
    [ValidateNotNullOrEmpty()][string]$Log = 'runs\league\decide_ab.jsonl',
    [ValidateNotNullOrEmpty()][string]$Noise = 'data\noise\default_prior.json'
)

# Continue, not Stop: native programs write warnings to stderr, and under Stop Windows
# PowerShell would end the script on the first one. Failures are caught from exit codes.
$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$RG = Join-Path $root '.venvs\royalegym\Scripts\python.exe'
if (-not (Test-Path $RG)) {
    Write-Host "ERROR: $RG is missing. Run scripts\setup_windows.ps1 first." -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path (Join-Path $root 'reports') | Out-Null
$runLog = Join-Path $root ("reports\decide_ab_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$writer = New-Object System.IO.StreamWriter($runLog, $true, (New-Object System.Text.UTF8Encoding($false)))
$writer.AutoFlush = $true

$env:PYTHONUNBUFFERED = '1'     # show progress as it happens, also in the log
$env:PYTHONIOENCODING = 'utf-8'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture

# Every function reads the script's own variables as $script:NAME, so that no caller's local
# variable of the same name (PowerShell looks names up in the callers' scopes, ignoring case)
# can stand in for them.

function Write-Both([string]$Text, [string]$Color = '') {
    if ($Color) { Write-Host $Text -ForegroundColor $Color } else { Write-Host $Text }
    $script:writer.WriteLine($Text)
}

function Stop-Run([string]$Text) {
    Write-Both $Text 'Red'
    $script:writer.Close()
    exit 1
}

# One step: its output to the console and the log, every line kept for the caller, with its
# exit code. The caller decides what a code means.
function Invoke-Step([string]$StepName, [string]$Exe, [string[]]$ExeArgs) {
    Write-Both ("== {0}  ({1})  {2} {3}" -f $StepName, (Get-Date -Format 'HH:mm:ss'), $Exe, ($ExeArgs -join ' ')) 'Cyan'
    $kept = New-Object System.Collections.Generic.List[string]
    & $Exe @ExeArgs 2>&1 | ForEach-Object {
        $outLine = "$_"
        Write-Both $outLine
        $kept.Add($outLine)
    }
    $code = $LASTEXITCODE
    return [pscustomobject]@{ Ok = ($code -eq 0); Code = $code; Lines = $kept }
}

# A path as the tools are given it (scripts\model_check.ps1's Resolve-Model): relative to the
# repo root when it is under it, with backslashes, else absolute. Name: the checkpoint's folder
# for best.pt, else folder_file.
function Resolve-Model([string]$PathText) {
    $clean = $PathText.Trim().Trim([char[]]('"', "'"))
    $joined = $(if ([System.IO.Path]::IsPathRooted($clean)) { $clean } else { Join-Path $script:root $clean })
    $full = [System.IO.Path]::GetFullPath($joined)
    $prefix = $script:root.TrimEnd('\') + '\'
    $rel = $(if ($full.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) { $full.Substring($prefix.Length) } else { $full })
    $file = [System.IO.Path]::GetFileName($full)
    $folder = Split-Path -Leaf (Split-Path -Parent $full)
    $modelName = $(if ($file -ieq 'best.pt') { $folder } else { '{0}_{1}' -f $folder, [System.IO.Path]::GetFileNameWithoutExtension($full) })
    return [pscustomobject]@{ Full = $full; Rel = $rel; Name = $modelName }
}

# --- the settings, checked before anything runs -------------------------------------------
if (-not [regex]::IsMatch($WaitFloor.Trim(), '^(0?\.[5-9][0-9]?|1(\.0{1,2})?)$')) {
    Stop-Run ("ERROR: -WaitFloor '{0}': a number from 0.5 to 1 with at most two decimals, written with a point (for example 0.95)." -f $WaitFloor)
}
$floorValue = [double]::Parse($WaitFloor.Trim(), $invariant)
$floorText = $floorValue.ToString('0.0#', $invariant)
$floorTag = ''
if ([Math]::Abs($floorValue - 0.95) -gt 1e-9) { $floorTag = '_wf{0:000}' -f [int][Math]::Round($floorValue * 100) }

$model = Resolve-Model $Ckpt
if (-not (Test-Path -LiteralPath $model.Full -PathType Leaf)) {
    Stop-Run ("ERROR: {0} is missing (-Ckpt). v2_hog comes from scripts\third_model.ps1 (runbook section 20)." -f $model.Full)
}
if ($model.Rel.Contains(',') -or $model.Rel.Contains('=')) {
    Stop-Run ("ERROR: {0}: the league reads a player as NAME=CKPT,KEY=VALUE, so the path may hold no ',' or '='." -f $model.Rel)
}
if (-not [regex]::IsMatch($model.Name, '^[A-Za-z0-9_.-]+$')) {
    Stop-Run ("ERROR: {0}: the name '{1}' made from its path is not letters, digits, '.', '_' and '-' only; move the checkpoint into a folder with such a name." -f $model.Full, $model.Name)
}
$logModel = Resolve-Model $Log
$reportModel = Resolve-Model (Join-Path (Split-Path -Parent $logModel.Full) ('{0}.report.json' -f [System.IO.Path]::GetFileNameWithoutExtension($logModel.Full)))
# The names tools\perception_noise.py reads as the prior or as no noise; anything else is a file.
if (@('default', 'prior', 'default_prior', 'none', 'off', 'zero', '0') -notcontains $Noise.Trim().ToLowerInvariant()) {
    $noisePath = $(if ([System.IO.Path]::IsPathRooted($Noise)) { $Noise } else { Join-Path $root $Noise })
    if (-not (Test-Path $noisePath)) { Stop-Run ("ERROR: the noise profile {0} is missing (-Noise)." -f $noisePath) }
}

# The players: the champion (rules off) and one candidate per variant, the same checkpoint.
$champion = [pscustomobject]@{ Name = $model.Name; Spec = ('{0}={1}' -f $model.Name, $model.Rel); Rules = 'none' }
$variantList = $(if ($Variant -eq 'all') { @('both', 'floor', 'stray') } else { @($Variant) })
$candidates = @()
foreach ($v in $variantList) {
    if ($v -eq 'both') {
        $candName = '{0}_rules{1}' -f $model.Name, $floorTag
        $candidates += [pscustomobject]@{ Name = $candName; Rules = ('wait floor {0} and stray guard' -f $floorText)
                                          Spec = ('{0}={1},wait_floor={2},stray_guard=on' -f $candName, $model.Rel, $floorText) }
    } elseif ($v -eq 'floor') {
        $candName = '{0}_floor{1}' -f $model.Name, $floorTag
        $candidates += [pscustomobject]@{ Name = $candName; Rules = ('wait floor {0}' -f $floorText)
                                          Spec = ('{0}={1},wait_floor={2}' -f $candName, $model.Rel, $floorText) }
    } else {
        $candName = '{0}_stray' -f $model.Name
        $candidates += [pscustomobject]@{ Name = $candName; Rules = 'stray guard'
                                          Spec = ('{0}={1},stray_guard=on' -f $candName, $model.Rel) }
    }
}

$leagueWorkers = [Math]::Min($Workers, 6)
$totalGames = $Games * $candidates.Count
$minutes = [Math]::Round($totalGames * 2.2 * [Math]::Max(1.0, 4.0 / $leagueWorkers) / 60.0)
Write-Both ("decide_ab.ps1: {0} ({1}) with the rules off as {2}; candidate(s): {3}. {4} games each against {2}, 24-tick delay, noise {5}, live rule, {6} workers; about {7} min (2.2 s a game at 4 workers). League log {8}, report {9}. Log: {10}" -f
    $model.Rel, $model.Name, $champion.Name,
    (($candidates | ForEach-Object { '{0} ({1})' -f $_.Name, $_.Rules }) -join ', '),
    $Games, $Noise, $leagueWorkers, $minutes, $logModel.Rel, $reportModel.Rel, $runLog) 'Cyan'

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logModel.Full) | Out-Null

#region rs_league_eval flags (tests\pipeline\test_decide_ab_ps1.py checks each against tools\rs_league_eval.py)
# Each candidate against the champion: -Games games, both seats, the decks exchanged (the
# league's default --h2h-decks both), no anchors. A 50% lower bound for the gate's PASS line.
$leagueArgs = @('tools\rs_league_eval.py', 'run')
foreach ($c in $candidates) { $leagueArgs += @('--candidate', $c.Spec) }
$leagueArgs += @('--champion', $champion.Spec, '--decide', 'live', '--delay', '24',
                 '--noise', $Noise, '--champion-games', "$Games", '--lower-bound', '0.5',
                 '--workers', "$leagueWorkers", '--log', $logModel.Rel, '--report', $reportModel.Rel)
#endregion
$league = Invoke-Step 'league' $RG $leagueArgs

$failed = @()
$incomplete = $false
$verdictLines = @()
if (-not $league.Ok) {
    $failed += 'league'
    Write-Both ("ERROR: the league step failed with exit code {0} (see its error line above): no verdict is read, so that no older games of these names stand in for this run's. A name the log holds with other rules or another checkpoint is refused before anything is played: give another -Log." -f $league.Code) 'Red'
} else {
#region decide_ab flags (tests\pipeline\test_decide_ab_ps1.py checks each against tools\decide_ab.py)
    $verdictArgs = @('tools\decide_ab.py', '--log', $logModel.Rel, '--champion', $champion.Name,
                     '--games', "$Games")
    foreach ($c in $candidates) { $verdictArgs += @('--candidate', $c.Name) }
#endregion
    $verdict = Invoke-Step 'verdict' $RG $verdictArgs
    if ($verdict.Code -eq 2) {
        $incomplete = $true
    } elseif (-not $verdict.Ok) {
        $failed += 'verdict'
    }
    foreach ($verdictLine in $verdict.Lines) {
        if ($verdictLine.StartsWith('verdict:') -or $verdictLine.StartsWith('  INCOMPLETE:')) { $verdictLines += $verdictLine }
    }
}

# --- the summary ---------------------------------------------------------------------------
Write-Both ("== summary ({0}): {1} with the rules on against itself with them off ({2}), {3} games each. How to read it: docs\SPARRING_RUNBOOK.md section 22." -f
    (Get-Date -Format 'HH:mm:ss'), $model.Rel, $champion.Name, $Games) 'Green'
foreach ($verdictLine in $verdictLines) {
    $lineColor = 'Yellow'
    if ($verdictLine.StartsWith('verdict: STRONGER')) { $lineColor = 'Green' }
    if ($verdictLine.StartsWith('verdict: WEAKER')) { $lineColor = 'Red' }
    Write-Both $verdictLine $lineColor
}
if ($failed.Count -gt 0) {
    Write-Both ("{0} step(s) failed: {1}. Send this log back: {2}" -f $failed.Count, ($failed -join ', '), $runLog) 'Red'
    $writer.Close()
    exit 1
}
if ($incomplete) {
    Write-Both ("A verdict has fewer than {0} games: run the same command again (the league plays only the games its log lacks). Log: {1}" -f $Games, $runLog) 'Yellow'
    $writer.Close()
    exit 2
}
Write-Both ("== done ({0}). Send back the log: {1}" -f (Get-Date -Format 'HH:mm:ss'), $runLog) 'Green'
$writer.Close()
