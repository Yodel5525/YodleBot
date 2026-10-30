<#
.SYNOPSIS
    The third model: v2 rows from the whole corpus. Builds the rows, compacts them for
    memory-mapped training, trains a general v2 model, fine-tunes it on Hog 2.6, and
    evaluates it in the simulator, one step after another.

.DESCRIPTION
    These are the steps of docs\SPARRING_RUNBOOK.md section 20, in order. The script
    stops at the first step that fails, except the rs_eval runs of the eval step: a failed
    one is logged, the league still runs, and the script lists the failures and exits 1
    at the end. Everything printed also goes to reports\third_model_<time>.log, so the run
    can be left alone overnight and the log sent back afterwards.

    The rows are schema v2 (E14): on top of section 19's decision-time rows (24 ticks
    before each play lands, the screen's view of the board, trust v2), each row shows the
    spells in flight, the opponent's elixir as counted from its plays, our own plays still
    in flight, hand evolution, the ability buttons (E10, --presses on) and unit form and
    burrow. Every row is observed through the perception-noise profile (E17), and the rows
    come from every usable IL_Replay match, not 20,000 (E18). Unit tokens stay
    --live-compat, as in the first and second models: the live runner reads that and the
    v2 schema, and the simulator's evaluators (rs_eval, rs_match, rs_league_eval) build
    only --live-compat or unit-type rows, not --units vision ones.

    The rows are compacted into per-column files (data\rs_cols_v2) and the trainer reads
    them from disk (--memmap), so they need not fit in 16 GB of RAM. Training flips half
    the rows left-right (--mirror 0.5) and keeps only the trust-v2 rows.

    The evaluation: first tools\rs_eval.py, 300 held-out matches against
    runs\tokens\live_big\ep01.pt at a 21-tick delay (the same instrument as section 19's
    runs A and B, so the numbers compare), then the same without a delay, at a 24-tick
    delay, and at a 21-tick delay under the live runner's decision rule (--decide live),
    with live_big_hog under that rule for reference: a lazy gate or a timing offset shows
    up in the same log. Then the league (tools\rs_league_eval.py): v2_hog as the candidate,
    live_big_hog (section 14, the live model and the best so far) as the champion and
    live_big ep01 as the anchor, paired games on both seats at a 24-tick delay with
    perception noise. dt_big_hog (section 19, 29.3%) is not the champion: beating it would
    not mean v2_hog is better than the live model.

    Run scripts\dt_diagnose.ps1 first (section 19, "Why the second model lost"): this
    script uses the same decision-time rows, and the checks say whether v2 should fix them.

    Close MuMu and the browser first: these jobs use most of a 16 GB PC's memory. The
    shorter first run (section 20), then the whole corpus once its v2_hog_20k beats
    live_big_hog:

        powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\third_model.ps1 -Others 20000 -Tag 20k
        powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\third_model.ps1

.PARAMETER Workers
    Processes for row building and rs_eval (default 8). Use 6 if RAM runs short. The
    league step uses at most 6, and tools\rs_league_eval.py lowers that further when free
    memory is short.

.PARAMETER Others
    0 (default): every usable match (--all-matches). N: every Hog 2.6 match plus N others
    drawn at random (--others N --others-pick random), for a shorter run. Give a shorter
    run its own -Tag, so that the whole-corpus run after it keeps its models.

.PARAMETER Epochs
    Epochs of the general model (default 2). The Hog 2.6 fine-tune always runs 3.

.PARAMETER MaxRows
    Train on a uniform sample of at most this many rows per split (default 0 = all). Use it
    to shorten training, e.g. -From general -MaxRows 15000000.

.PARAMETER Noise
    The perception-noise profile of the rows and the league (default
    data\noise\default_prior.json, the prior). A profile fitted by
    tools\fit_noise_profile.py can be given instead.

.PARAMETER From
    Start at a later step, skipping the earlier ones: rows, compact, general, hog, eval or
    league. Example: -From league re-runs the league only.

.PARAMETER SkipDiskCheck
    Start even when the drive has less free space than the estimate: 90 GB when the run
    starts at the rows, 60 GB when it starts at the compact step (-From compact).

.PARAMETER Tag
    A name for this run's files, so that a second run keeps the first one's. Empty (the
    default): data\rs_rows_v2, data\rs_cols_v2, runs\tokens\v2_big and v2_hog, and the
    league player v2_hog. -Tag 20k: data\rs_rows_v2_20k, data\rs_cols_v2_20k,
    runs\tokens\v2_big_20k and v2_hog_20k, and the league player v2_hog_20k. Every run
    shares the league log runs\league\third_model.jsonl, so live_big_hog's games against
    live_big are played once. The league refuses a name whose games it holds when the
    checkpoint has changed, so the script stops at the start when it would train a
    candidate whose name already has games there. Letters, digits, _ and - only.
    Example: the shorter first run, -Others 20000 -Tag 20k, then the whole corpus with no
    -Tag.
#>
param(
    [ValidateRange(1, 64)][int]$Workers = 8,
    [ValidateRange(0, 1000000)][int]$Others = 0,
    [ValidateRange(1, 20)][int]$Epochs = 2,
    [long]$MaxRows = 0,
    [ValidateNotNullOrEmpty()][string]$Noise = 'data\noise\default_prior.json',
    [ValidateSet('rows', 'compact', 'general', 'hog', 'eval', 'league')][string]$From = 'rows',
    [switch]$SkipDiskCheck,
    [ValidatePattern('^[A-Za-z0-9_-]*$')][string]$Tag = ''
)

# Continue, not Stop: native programs write warnings to stderr, and under Stop
# Windows PowerShell would end the script on the first one. Failures are caught
# from each step's exit code instead.
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
$log = Join-Path $root ("reports\third_model_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
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

function Invoke-Step([string]$Name, [string]$Python, [string[]]$ArgList) {
    Write-Both ("== {0}  ({1})  {2} {3}" -f $Name, (Get-Date -Format 'HH:mm:ss'), $Python, ($ArgList -join ' ')) 'Cyan'
    & $Python @ArgList 2>&1 | ForEach-Object { Write-Both ("$_") }
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Both ("ERROR: step '{0}' failed with exit code {1}. Send this log back: {2}" -f $Name, $code, $log) 'Red'
        $writer.Close()
        exit $code
    }
}

# An rs_eval run of the eval step: a failure is logged and the script goes on, so that one
# failed evaluation (the v2 and --decide live paths have not run on the PC) does not cost the
# league after a night of rows and training. The failures are listed at the end.
$failedRuns = New-Object System.Collections.Generic.List[string]
function Invoke-Run([string]$Name, [string]$Python, [string[]]$ArgList) {
    Write-Both ("== {0}  ({1})  {2} {3}" -f $Name, (Get-Date -Format 'HH:mm:ss'), $Python, ($ArgList -join ' ')) 'Cyan'
    & $Python @ArgList 2>&1 | ForEach-Object { Write-Both ("$_") }
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Both ("ERROR: '{0}' failed with exit code {1}; going on with the next step." -f $Name, $code) 'Red'
        $failedRuns.Add($Name)
    }
}

# Whether tools\rs_match.py (which rs_league_eval plays every game through) builds v2 rows: its
# ROW_SCHEMAS must hold rs_rows/v2, else every game of v2_hog is refused at its start. Exit 3
# means it does not; any other failure of the probe (an import error) is left to the league
# step itself to report. No double quotes in the probe: PowerShell 5.1 passes them to python
# unescaped.
function Test-LeagueV2 {
    $probe = "import sys; sys.path[:0] = ['tools']; import rs_match; " +
             "sys.exit(0 if 'rs_rows/v2' in getattr(rs_match, 'ROW_SCHEMAS', ['rs_rows/v2']) else 3)"
    $null = & $RG -c $probe 2>&1
    return ($LASTEXITCODE -ne 3)
}
$leagueV2Text = "tools\rs_match.py on this branch builds no v2 rows (ROW_SCHEMAS lacks rs_rows/v2), so the league cannot play v2_hog"

$order = @('rows', 'compact', 'general', 'hog', 'eval', 'league')
# ValidateSet accepts any case (-From Hog), but IndexOf compares exactly: without the
# lower-casing, -From Hog would give -1 and run every step from the rows again.
$fromKey = $From.ToLowerInvariant()
$start = [array]::IndexOf($order, $fromKey)
# -Tag: this run's own rows, store, checkpoints and league name (the .PARAMETER Tag help).
$sfx = $(if ($Tag) { "_$Tag" } else { '' })
$tagArg = $(if ($Tag) { " -Tag $Tag" } else { '' })      # for the restart advice
$rowsDir = "data\rs_rows_v2$sfx"
$colsDir = "data\rs_cols_v2$sfx"
# The champion is the live model, the first model (scripts\first_model.ps1): the second model
# (dt_big_hog, 29.3% against its 46.0%) would make the gate easy to pass and the pass mean
# nothing. The anchor is the first model's general checkpoint, which is also rs_eval's
# reactive opponent.
$champion = 'runs\tokens\live_big_hog\best.pt'
$anchor = 'runs\tokens\live_big\ep01.pt'
$opponent = 'runs\tokens\live_big\ep01.pt'         # rs_eval's reactive opponent, as in section 19
$generalName = "v2_big$sfx"
$candidateName = "v2_hog$sfx"
$general = "runs\tokens\$generalName\best.pt"
$candidate = "runs\tokens\$candidateName\best.pt"
$leagueLog = 'runs\league\third_model.jsonl'
# d256, 6 layers, 8 heads as live_big and dt_big: the comparison is then about the rows, and
# the simulator's CPU workers (league, later RL) run the model as fast as the champion. About
# 3-4 GB of the 12 GB VRAM at batch 512. --wait-keep 0.25 equals --wait-keep-at-build, so the
# trainer keeps every wait row the builder kept. --val-max-rows: the all-deck validation set
# is a 500,000-row sample, held in RAM; the Hog 2.6 one is smaller and whole.
$model = @('--memmap', '--mirror', '0.5', '--trust', 'v2', '--wait-keep', '0.25',
           '--d', '256', '--layers', '6', '--heads', '8', '--val-max-rows', '500000')
if ($MaxRows -gt 0) { $model += @('--max-rows', "$MaxRows") }
$matchText = $(if ($Others -gt 0) { "Hog 2.6 + $Others others" } else { 'all' })
Write-Both ("third_model.ps1: from '{0}', workers {1}, matches {2}, epochs {3}, max rows {4}, noise {5}, models {6} and {7}, rows {8}. Log: {9}" -f $fromKey, $Workers, $matchText, $Epochs, $(if ($MaxRows -gt 0) { $MaxRows } else { 'all' }), $Noise, $generalName, $candidateName, $rowsDir, $log) 'Cyan'

# A run that trains the candidate (it starts at the hog step or before) gives it a new
# checkpoint. If the league log already holds games of that name, tools\rs_league_eval.py
# refuses the new checkpoint under the old name, after hours of work: stop now instead. The
# log is JSON lines written without spaces, so the quoted name matches the player exactly
# (v2_hog does not match v2_hog_20k).
if ($start -le 3) {
    $leaguePath = Join-Path $root $leagueLog
    if ((Test-Path $leaguePath) -and (Select-String -LiteralPath $leaguePath -SimpleMatch -Quiet -Pattern ('"' + $candidateName + '"'))) {
        Stop-Run ("ERROR: {0} already holds league games of {1}. This run trains {1} again, and the league would refuse the new checkpoint under that name after hours of work. Give this run its own name with -Tag (for example -Tag full), or, to throw the old games away, move the log: Move-Item {2} {2}.old" -f $leagueLog, $candidateName, $leaguePath)
    }
}

# What the steps after -From read, checked now rather than after hours of work.
# manifest.json, not vocab.json: rs_rows writes vocab.json first and manifest.json only once
# every shard is done; compact.json is likewise the last file rs_rows_compact writes.
$need = @($champion, $anchor)
if ($start -le 4) { $need += $opponent }
if ($start -eq 1) { $need += "$rowsDir\manifest.json" }
if ($start -eq 2 -or $start -eq 3) { $need += "$colsDir\compact.json" }
if ($start -eq 3) { $need += $general }
if ($start -ge 4) { $need += $candidate }
foreach ($f in $need) {
    if (-not (Test-Path (Join-Path $root $f))) {
        Stop-Run ("ERROR: {0} is missing. live_big_hog and live_big come from scripts\first_model.ps1 (runbook section 14); the v2_ files, {1} and {2} from the earlier steps of this script (start without -From)." -f $f, $rowsDir, $colsDir)
    }
}
# The names tools\perception_noise.py reads as the prior or as no noise; anything else is a file.
if (@('default', 'prior', 'default_prior', 'none', 'off', 'zero', '0') -notcontains $Noise.Trim().ToLowerInvariant()) {
    $noisePath = $(if ([System.IO.Path]::IsPathRooted($Noise)) { $Noise } else { Join-Path $root $Noise })
    if (-not (Test-Path $noisePath)) {
        Stop-Run ("ERROR: the noise profile {0} is missing (-Noise)." -f $noisePath)
    }
}
if (-not (Test-LeagueV2)) {
    if ($start -ge 5) {
        Stop-Run ("ERROR: {0}. Update the branch (git pull), then run -From league{1} again." -f $leagueV2Text, $tagArg)
    }
    Write-Both ("warning: {0}. The steps before it run; the script then stops before the league. Update the branch (git pull) and run -From league{1}." -f $leagueV2Text, $tagArg) 'Yellow'
}
if ($start -le 0) {
    # Finished rows are hours of work: rebuilding them must be asked for by deleting them.
    if (Test-Path (Join-Path $root "$rowsDir\manifest.json")) {
        Stop-Run ("ERROR: {0} already holds finished rows. Start at the next step with -From compact, or delete the folder to build them again: Remove-Item -Recurse -Force {1}" -f $rowsDir, (Join-Path $root $rowsDir))
    }
}

# Disk: about 15-20 GB of shards plus about 50 GB of compact columns (runbook section 20).
$diskNeed = @{ 'rows' = 90; 'compact' = 60 }
$needGb = $diskNeed[$fromKey]
if ($needGb) {
    $freeGb = -1
    try {
        $drive = New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot($root))
        $freeGb = [Math]::Floor($drive.AvailableFreeSpace / 1GB)
    } catch {
        Write-Both ("warning: could not read the free space of {0}: {1}" -f $root, $_) 'Yellow'
    }
    if ($freeGb -ge 0) {
        Write-Both ("disk: {0} GB free on {1}; this run needs about {2} GB" -f $freeGb, [System.IO.Path]::GetPathRoot($root), $needGb)
        if ($freeGb -lt $needGb -and -not $SkipDiskCheck) {
            Stop-Run ("ERROR: {0} GB free, about {1} GB needed. Free some space (runbook section 20 lists what can go), or pass -SkipDiskCheck to start anyway." -f $freeGb, $needGb)
        }
    }
}

if ($start -le 0) {
    # Shards of an interrupted earlier run (no manifest.json, checked above) would be mixed
    # into this run's: rs_rows_compact and the trainer read every shard-*.npz of the folder.
    $stale = @(Get-ChildItem -Path (Join-Path $root $rowsDir) -Filter 'shard-*.npz' -ErrorAction SilentlyContinue)
    if ($stale.Count -gt 0) {
        Write-Both ("removing {0} shards of an unfinished earlier run from {1}" -f $stale.Count, $rowsDir) 'Yellow'
        $stale | Remove-Item -Force
    }
    $rowsArgs = @('tools\rs_rows.py', '--schema', 'v2', '--live-compat', '--obs-delay', '24',
                  '--obs-delay-jitter', '3', '--opp-lag', '1', '--visibility', 'public',
                  '--trust', 'v2', '--presses', 'on', '--noise', $Noise,
                  '--wait-keep-at-build', '0.25', '--workers', "$Workers", '--out', $rowsDir)
    if ($Others -gt 0) {
        $rowsArgs += @('--others', "$Others", '--others-pick', 'random')
    } else {
        $rowsArgs += @('--all-matches')
    }
    Invoke-Step 'rows (schema v2)' $RG $rowsArgs
}
if ($start -le 1) {
    # --force: a store already in data\rs_cols_v2 is from older rows (or an earlier try).
    Invoke-Step 'compact' $RG (@('tools\rs_rows_compact.py', '--rows', $rowsDir, '--out', $colsDir,
                                 '--force'))
}
if ($start -le 2) {
    Invoke-Step 'general model' $BB (@('-m', 'sim.train_tokens', '--name', $generalName,
                                       '--data', $colsDir, '--epochs', "$Epochs") + $model)
}
if ($start -le 3) {
    Invoke-Step 'Hog 2.6 fine-tune' $BB (@('-m', 'sim.train_tokens', '--name', $candidateName,
                                           '--data', $colsDir, '--epochs', '3') + $model +
                                         @('--finetune-hog', '--init', $general))
}
if ($start -le 4) {
    # Section 19's instrument: the same reactive opponent, the same 300 held-out matches and the
    # same 21-tick delay as its runs A (live_big_hog) and B (dt_big_hog), so the numbers compare
    # directly. Then the arms that tell a lazy gate or a timing offset apart (section 19, "Why the
    # second model lost"): no delay, a 24-tick delay, and the live runner's rule. --decide live
    # switches the opponent seat too, so the two live-rule runs compare only with each other.
    # Invoke-Run: a failed evaluation is logged and the league still runs.
    $vs = @('--opponent-ckpt', $opponent, '--matches', '300', '--workers', "$Workers")
    $evalNew = @('tools\rs_eval.py', '--ckpt', $candidate) + $vs
    Invoke-Run "evaluation: $candidateName, 21-tick delay" $RG ($evalNew + @('--delay', '21'))
    Invoke-Run "evaluation: $candidateName, no delay" $RG $evalNew
    Invoke-Run "evaluation: $candidateName, 24-tick delay" $RG ($evalNew + @('--delay', '24'))
    Invoke-Run "evaluation: $candidateName, 21-tick delay, live rule" $RG ($evalNew + @('--delay', '21', '--decide', 'live'))
    Invoke-Run 'evaluation: live_big_hog, 21-tick delay, live rule (reference)' $RG (@('tools\rs_eval.py', '--ckpt', $champion) + $vs + @('--delay', '21', '--decide', 'live'))
}
if ($start -le 5) {
    # The gate: 1,000 paired games against the champion (both seats, both decks), 400 against
    # the anchor for each of the candidate and the champion. --full: the anchor games are
    # played even when the candidate does not beat the champion, so their numbers come back.
    if (-not (Test-LeagueV2)) {
        Stop-Run ("ERROR: {0}. {1} and {2} are trained and kept: update the branch (git pull), then run -From league{3}. Send this log back: {4}" -f $leagueV2Text, $generalName, $candidateName, $tagArg, $log)
    }
    $leagueWorkers = [Math]::Min($Workers, 6)
    Invoke-Step "league: $candidateName vs live_big_hog, anchor live_big" $RG (@('tools\rs_league_eval.py', 'run',
        '--candidate', "$candidateName=$candidate", '--champion', "live_big_hog=$champion",
        '--anchor', "live_big=$anchor", '--delay', '24', '--noise', $Noise,
        '--champion-games', '1000', '--anchor-games', '400', '--full',
        '--workers', "$leagueWorkers", '--log', $leagueLog))
}

if ($failedRuns.Count -gt 0) {
    Write-Both ("== done ({0}), but {1} evaluation run(s) failed: {2}. Send this log back: {3}" -f (Get-Date -Format 'HH:mm:ss'), $failedRuns.Count, ($failedRuns -join '; '), $log) 'Red'
    $writer.Close()
    exit 1
}
Write-Both ("== done ({0}). Send back the log: {1}" -f (Get-Date -Format 'HH:mm:ss'), $log) 'Green'
$writer.Close()
