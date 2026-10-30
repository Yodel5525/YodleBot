<#
.SYNOPSIS
    RL fine-tuning in RoyaleSim (Phase 7, R04): the bench first, then the long run, with
    resume. Runs tools\rl\train_loop.py and writes everything to reports\rl_train_<time>.log.

.DESCRIPTION
    docs\SPARRING_RUNBOOK.md section 21. Pure simulation: nothing here starts, touches or taps
    the emulator. The run lives in runs\rl\<Name>\ (checkpoints, the league, the gate's log,
    champion.pt); train_loop.log there and this script's log hold the console.

    Each iteration collects -Matches battles (Hog 2.6 at level 11 against the train split's
    deck field, command delay 24 ticks, perception noise) on rollout workers of the engine venv
    and then runs one PPO update in the GPU venv. Every -GateEvery iterations the promotion
    gate (tools\rs_league_eval.py) plays the newest checkpoint against the champion (at first
    the start model, v2_hog) and the imitation anchors; a pass copies it to champion.pt.

    Close MuMu and the browser first: the workers use about 8 GB.

        powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\rl_train.ps1 -Bench
        powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\rl_train.ps1 -Workers 8 -Hours 48
        powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\rl_train.ps1 -From resume -Hours 24

    To stop cleanly: New-Item -ItemType File -Force C:\hcr\runs\rl\hog_rl1\STOP
    (the step that is running finishes first; delete the file before resuming).

.PARAMETER Bench
    Measure instead of training (about -BenchMinutes): matches/hour, RAM per worker and a
    recommended -Workers. Writes runs\rl\<Name>\bench-<time>\bench.json. Run it first.

.PARAMETER BenchMinutes
    How long the bench measures (default 10).

.PARAMETER Workers
    Rollout worker processes (default 0: the driver's 8, lowered when RAM is short). Use the
    bench's recommendation.

.PARAMETER Hours
    Stop before a new collection once this launch has run this long (default 0: no limit).

.PARAMETER Iterations
    Stop once the run has done this many iterations in all (default 0: no limit).

.PARAMETER Matches
    Battles per iteration (default 400).

.PARAMETER From
    new (default): start the run (refused if it exists). resume: continue it from where it
    stopped. A number N: resume from iteration N, moving later iterations to a rollback folder.

.PARAMETER Name
    The run's folder name under runs\rl (default hog_rl1).

.PARAMETER Start
    The v2 imitation checkpoint to start from (default runs\tokens\v2_hog\best.pt). Also the
    first champion, the KL reference and an anchor. Only for a new run.

.PARAMETER Noise
    The perception-noise profile (default: data\noise\measured.json when it exists, else the
    prior). Only for a new run: a run keeps its profile.

.PARAMETER GateEvery
    Run the promotion gate every N iterations (default 10; 0: never).

.PARAMETER NoAnchors
    Do not add dt_big_hog and live_big_hog as anchors (new run only).
#>
param(
    [switch]$Bench,
    [ValidateRange(1, 240)][int]$BenchMinutes = 10,
    [ValidateRange(0, 64)][int]$Workers = 0,
    [ValidateRange(0, 10000)][double]$Hours = 0,
    [ValidateRange(0, 1000000)][int]$Iterations = 0,
    [ValidateRange(1, 100000)][int]$Matches = 400,
    [ValidateNotNullOrEmpty()][string]$From = 'new',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')][string]$Name = 'hog_rl1',
    [string]$Start = '',
    [string]$Noise = '',
    [ValidateRange(0, 100000)][int]$GateEvery = 10,
    [switch]$NoAnchors
)

# Continue, not Stop: native programs write warnings to stderr, and under Stop Windows
# PowerShell would end the script on the first one. Failures are caught from exit codes.
$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$RG = Join-Path $root '.venvs\royalegym\Scripts\python.exe'
$BB = Join-Path $root '.venvs\buildabot\Scripts\python.exe'
foreach ($py in @($RG, $BB)) {
    if (-not (Test-Path $py)) {
        Write-Host "ERROR: $py is missing. Run scripts\setup_windows.ps1 first." -ForegroundColor Red
        exit 1
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $root 'reports') | Out-Null
$log = Join-Path $root ("reports\rl_train_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$writer = New-Object System.IO.StreamWriter($log, $true, (New-Object System.Text.UTF8Encoding($false)))
$writer.AutoFlush = $true

$env:PYTHONUNBUFFERED = '1'     # show progress as it happens, also in the log
$env:PYTHONIOENCODING = 'utf-8'

function Write-Both([string]$Text, [string]$Color = '') {
    if ($Color) { Write-Host $Text -ForegroundColor $Color } else { Write-Host $Text }
    $writer.WriteLine($Text)
}

function Stop-Run([string]$Text) {
    Write-Both $Text 'Red'
    $writer.Close()
    exit 1
}

# Runs one python command, everything it prints to the console and the log. Returns the exit
# code and the last line that looks like a JSON object (the tools print their summary so).
function Invoke-Py([string]$Title, [string]$Python, [string[]]$ArgList) {
    Write-Both ("== {0}  ({1})  {2} {3}" -f $Title, (Get-Date -Format 'HH:mm:ss'), $Python, ($ArgList -join ' ')) 'Cyan'
    $script:lastJson = ''
    & $Python @ArgList 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Both $line
        if ($line.StartsWith('{') -and $line.EndsWith('}')) { $script:lastJson = $line }
    }
    return $LASTEXITCODE
}

# The machine's data and runs folders (src\hastycr\machine.py; HASTYCR_DATA / HASTYCR_RUNS or
# machine.json may move them), else the repo's.
function Get-MachinePath([string]$Key, [string]$Fallback) {
    $value = & $RG (Join-Path $root 'src\hastycr\machine.py') --get $Key 2>$null | Select-Object -Last 1
    if ($LASTEXITCODE -eq 0 -and $value) { return "$value".Trim() }
    return (Join-Path $root $Fallback)
}

$fromKey = $From.Trim().ToLowerInvariant()
$rollback = -1
if ($fromKey -match '^\d+$') {
    $rollback = [int]$fromKey
    $fromKey = 'rollback'
} elseif (@('new', 'resume') -notcontains $fromKey) {
    Stop-Run ("ERROR: -From {0}: new, resume or an iteration number." -f $From)
}

$dataRoot = Get-MachinePath 'data' 'data'
$runsRoot = Get-MachinePath 'runs' 'runs'
$runDir = Join-Path $runsRoot ("rl\{0}" -f $Name)
$stateFile = Join-Path $runDir 'state.json'
$mode = $(if ($Bench) { 'bench' } else { $fromKey })
Write-Both ("rl_train.ps1: run {0} ({1}), workers {2}, matches {3}, hours {4}, iterations {5}, gate every {6}. Run folder: {7}. Log: {8}" -f $Name, $mode, $(if ($Workers -gt 0) { $Workers } else { 'default' }), $Matches, $(if ($Hours -gt 0) { $Hours } else { 'no limit' }), $(if ($Iterations -gt 0) { $Iterations } else { 'no limit' }), $GateEvery, $runDir, $log) 'Cyan'

# Read-only checks of what else holds memory. Nothing here starts, stops or touches MuMu.
$mumu = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -match '^(MuMuPlayer|MuMuNxDevice|MuMuVMMHeadless)' })
if ($mumu.Count -gt 0) {
    Write-Both ("warning: MuMu is running ({0} process(es)). Close it yourself before a long run: the workers need the memory." -f $mumu.Count) 'Yellow'
}
try {
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
    $freeGb = [Math]::Round($os.FreePhysicalMemory / 1MB, 1)
    Write-Both ("memory: {0} GB free" -f $freeGb)
    if ($freeGb -lt 10) {
        Write-Both "warning: under 10 GB free. Close MuMu and the browser; the driver lowers the worker count to what fits." 'Yellow'
    }
} catch {
    Write-Both ("warning: could not read the free memory: {0}" -f $_) 'Yellow'
}

$newRun = ($Bench -or $fromKey -eq 'new')
if (-not $Bench) {
    if ($fromKey -eq 'new' -and (Test-Path $stateFile)) {
        Stop-Run ("ERROR: the run {0} already exists ({1}). Continue it with -From resume, or choose another -Name." -f $Name, $stateFile)
    }
    if ($fromKey -ne 'new' -and -not (Test-Path $stateFile)) {
        Stop-Run ("ERROR: there is no run {0} to resume ({1} is missing). Start it with -From new." -f $Name, $stateFile)
    }
    if (Test-Path (Join-Path $runDir 'STOP')) {
        Stop-Run ("ERROR: {0} exists (it stopped the last launch). Delete it to run: Remove-Item {0}" -f (Join-Path $runDir 'STOP'))
    }
}
if (Test-Path (Join-Path $root 'STOP')) {
    Stop-Run ("ERROR: {0} exists (the live runner's stop file; it stops this run too). Delete it to run: Remove-Item {0}" -f (Join-Path $root 'STOP'))
}

$argList = @('tools\rl\train_loop.py', 'run', '--name', $Name, '--matches', "$Matches",
             '--gate-every', "$GateEvery")
if ($Workers -gt 0) { $argList += @('--workers', "$Workers") }
if ($Hours -gt 0) { $argList += @('--hours', $Hours.ToString([System.Globalization.CultureInfo]::InvariantCulture)) }
if ($Iterations -gt 0) { $argList += @('--iterations', "$Iterations") }

if ($newRun -and $fromKey -eq 'new') {
    # What a new run (or the bench) reads: the start, the anchors, the deck field.
    $startPath = $(if ($Start) { $Start } else { Join-Path $runsRoot 'tokens\v2_hog\best.pt' })
    if (-not (Test-Path $startPath)) {
        Stop-Run ("ERROR: the start checkpoint {0} is missing. It comes from scripts\third_model.ps1 (runbook section 20)." -f $startPath)
    }
    $argList += @('--start', $startPath)
    if (-not $NoAnchors) {
        foreach ($anchor in @('dt_big_hog', 'live_big_hog')) {
            $anchorPath = Join-Path $runsRoot ("tokens\{0}\best.pt" -f $anchor)
            if (Test-Path $anchorPath) {
                $argList += @('--anchor', ("{0}={1}" -f $anchor, $anchorPath))
            } else {
                Write-Both ("warning: anchor {0} is missing ({1}); the run goes on without it" -f $anchor, $anchorPath) 'Yellow'
            }
        }
    }
    if ($Noise) {
        # The names tools\perception_noise.py reads as the prior or as no noise; else a file.
        if (@('default', 'prior', 'default_prior', 'none', 'off', 'zero', '0') -notcontains $Noise.Trim().ToLowerInvariant()) {
            if (-not (Test-Path $Noise)) { Stop-Run ("ERROR: the noise profile {0} is missing (-Noise)." -f $Noise) }
        }
        $argList += @('--noise', $Noise)
    }
    $deckStats = Join-Path $dataRoot 'deck_stats.json'
    if (-not (Test-Path $deckStats)) {
        # The opponents' deck field: IL_Replay decks of the train split, by frequency.
        $code = Invoke-Py 'deck stats (the opponents deck field, once)' $RG @('tools\deck_stats.py', '--top', '2000', '--workers', '4')
        if ($code -ne 0) {
            Stop-Run ("ERROR: tools\deck_stats.py failed with exit code {0}. Send this log back: {1}" -f $code, $log)
        }
    }
} elseif ($Bench) {
    $argList += @('--resume')      # the bench of an existing run uses its settings
} elseif ($fromKey -eq 'resume') {
    $argList += @('--resume')
} else {
    $argList += @('--resume', '--rollback', "$rollback")
}
if ($Start -and -not $newRun) {
    Write-Both "note: -Start is only for a new run; a resumed run keeps its start." 'Yellow'
}
if ($Bench) {
    $argList += @('--bench', ("{0}m" -f $BenchMinutes))
}

$title = $(if ($Bench) { 'bench' } else { 'RL training' })
$code = Invoke-Py $title $RG $argList

if ($Bench -and $code -eq 0 -and $script:lastJson) {
    try {
        $result = $script:lastJson | ConvertFrom-Json
        Write-Both ("bench done. Next: powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\rl_train.ps1 -Workers {0} -Hours 48" -f $result.recommended_workers) 'Green'
    } catch {
        Write-Both 'bench done (its summary line did not parse; see the lines above).' 'Yellow'
    }
}
if (-not $Bench -and (Test-Path $stateFile)) {
    $null = Invoke-Py 'status' $RG @('tools\rl\train_loop.py', 'status', '--name', $Name)
}

if ($code -eq 0) {
    Write-Both ("== done ({0}). Champion: {1}. Send back the log: {2}" -f (Get-Date -Format 'HH:mm:ss'), (Join-Path $runDir 'champion.pt'), $log) 'Green'
} elseif ($code -eq 3) {
    Write-Both ("== done ({0}), but a gate could not run: see runs\rl\{1}\logs\gate-*.log. Send back the log: {2}" -f (Get-Date -Format 'HH:mm:ss'), $Name, $log) 'Yellow'
} elseif ($code -eq 130) {
    Write-Both ("== interrupted. Resume with -From resume. Log: {0}" -f $log) 'Yellow'
} else {
    Write-Both ("ERROR: train_loop.py failed with exit code {0}. Fix the cause (send this log back: {1}), then resume with -From resume." -f $code, $log) 'Red'
}
$writer.Close()
exit $code
