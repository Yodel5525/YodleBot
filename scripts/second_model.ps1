<#
.SYNOPSIS
    The second model: decision-time rows. Builds the rows, trains a general model,
    fine-tunes it on Hog 2.6, and runs the simulator A/B against the first model,
    one step after another.

.DESCRIPTION
    These are the steps of docs\SPARRING_RUNBOOK.md section 19, in order. The script
    stops at the first step that fails. Everything printed also goes to
    reports\second_model_<time>.log, so the run can be left alone (overnight, say) and
    the log sent back afterwards.

    The rows are decision-time rows (E11): each play is observed about 24 ticks before
    it lands, where its player tapped it, with the screen's view of the board (no
    invisible enemy, E12) and trust v2 (E09). The evaluation is an A/B: the first model
    runs\tokens\live_big_hog\best.pt is measured again under the current code, then the
    new runs\tokens\dt_big_hog\best.pt, both against the same reactive opponent
    runs\tokens\live_big\ep01.pt on the same 300 held-out matches at a 21-tick command
    delay; then the new model once more without the delay.

    Close MuMu and the browser first: these jobs use most of a 16 GB PC's memory.

        powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\second_model.ps1

.PARAMETER Workers
    Processes for row building and evaluation (default 8). Use 6 if RAM runs short.
    The league step uses at most 4.

.PARAMETER Others
    Non-Hog matches added to the rows (default 20000, as for the first model).

.PARAMETER MaxRows
    Train on a uniform sample of at most this many rows (default 0 = all). Use it if
    training runs out of memory, e.g. -From general -MaxRows 4000000.

.PARAMETER From
    Start at a later step, skipping the earlier ones: rows, general, hog, eval or league.
    Example: -From eval re-runs the evaluations only. -From league runs the league only.

.PARAMETER League
    Also run the optional last step: a league evaluation (tools\rs_league_eval.py) of
    dt_big_hog against live_big_hog, both seats and both decks, 24-tick delay and the
    default perception noise, 400 games. It adds roughly half an hour to an hour.
#>
param(
    [int]$Workers = 8,
    [int]$Others = 20000,
    [long]$MaxRows = 0,
    [ValidateSet('rows', 'general', 'hog', 'eval', 'league')][string]$From = 'rows',
    [switch]$League
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
$log = Join-Path $root ("reports\second_model_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$writer = New-Object System.IO.StreamWriter($log, $true, (New-Object System.Text.UTF8Encoding($false)))
$writer.AutoFlush = $true

$env:PYTHONUNBUFFERED = '1'     # show progress as it happens, also in the log
$env:PYTHONIOENCODING = 'utf-8'

function Write-Both([string]$Text, [string]$Color = '') {
    if ($Color) { Write-Host $Text -ForegroundColor $Color } else { Write-Host $Text }
    $writer.WriteLine($Text)
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

$order = @('rows', 'general', 'hog', 'eval', 'league')
$start = [array]::IndexOf($order, $From)
$runLeague = $League -or ($From -eq 'league')
$rowsDir = 'data\rs_rows_dt'
$champion = 'runs\tokens\live_big_hog\best.pt'     # the first model (scripts\first_model.ps1)
$opponent = 'runs\tokens\live_big\ep01.pt'         # the reactive opponent of every evaluation
$general = 'runs\tokens\dt_big\best.pt'
$candidate = 'runs\tokens\dt_big_hog\best.pt'
$model = @('--wait-keep', '0.25', '--epochs', '3', '--d', '256', '--layers', '6', '--heads', '8',
           '--trust', 'v2')
if ($MaxRows -gt 0) { $model += @('--max-rows', "$MaxRows") }
Write-Both ("second_model.ps1: from '{0}', workers {1}, others {2}, max rows {3}, league {4}. Log: {5}" -f $From, $Workers, $Others, $(if ($MaxRows -gt 0) { $MaxRows } else { 'all' }), $(if ($runLeague) { 'yes' } else { 'no' }), $log) 'Cyan'

# What the steps after -From read, checked now rather than after an hour of work.
$need = @($champion)
if ($start -le 3) { $need += $opponent }
if ($start -eq 1 -or $start -eq 2) { $need += "$rowsDir\vocab.json" }
if ($start -eq 2) { $need += $general }
if ($start -ge 3) { $need += $candidate }
foreach ($f in $need) {
    if (-not (Test-Path (Join-Path $root $f))) {
        Write-Both ("ERROR: {0} is missing. live_big_hog and live_big come from scripts\first_model.ps1 (runbook section 14); the dt_ files and {1} from the earlier steps of this script (start without -From)." -f $f, $rowsDir) 'Red'
        $writer.Close()
        exit 1
    }
}

if ($start -le 0) {
    Invoke-Step 'rows (decision time)' $RG (@('tools\rs_rows.py', '--live-compat', '--obs-delay', '24',
                                              '--obs-delay-jitter', '3', '--opp-lag', '1',
                                              '--visibility', 'public', '--trust', 'v2',
                                              '--presses', 'off', '--others', "$Others",
                                              '--workers', "$Workers", '--out', $rowsDir))
}
if ($start -le 1) {
    Invoke-Step 'general model' $BB (@('-m', 'sim.train_tokens', '--name', 'dt_big',
                                       '--data', $rowsDir) + $model)
}
if ($start -le 2) {
    Invoke-Step 'Hog 2.6 fine-tune' $BB (@('-m', 'sim.train_tokens', '--name', 'dt_big_hog',
                                           '--data', $rowsDir) + $model +
                                         @('--finetune-hog', '--init', $general))
}
if ($start -le 3) {
    # The A/B: the same opponent, the same 300 held-out matches and the same delay for both.
    # The first model is measured again because the evaluation code has changed since its
    # 46.0% (opponent memory now dated at execution).
    $vs = @('--opponent-ckpt', $opponent, '--matches', '300', '--workers', "$Workers")
    $evalOld = @('tools\rs_eval.py', '--ckpt', $champion) + $vs
    $evalNew = @('tools\rs_eval.py', '--ckpt', $candidate) + $vs
    Invoke-Step 'evaluation A: live_big_hog (first model), 21-tick delay' $RG ($evalOld + @('--delay', '21'))
    Invoke-Step 'evaluation B: dt_big_hog, 21-tick delay' $RG ($evalNew + @('--delay', '21'))
    Invoke-Step 'evaluation C: dt_big_hog, no delay' $RG $evalNew
}
if ($runLeague -and $start -le 4) {
    # Optional: paired games, both seats and both decks, under delay and perception noise.
    # 400 games (100 pairings) instead of the gate's 1,000, so its pass / fail is a first
    # reading only. Its own log, so the full gate later starts from fresh pairings.
    $leagueWorkers = [Math]::Min($Workers, 4)
    Invoke-Step 'league: dt_big_hog vs live_big_hog' $RG (@('tools\rs_league_eval.py', 'run',
        '--candidate', "dt_big_hog=$candidate", '--champion', "live_big_hog=$champion",
        '--delay', '24', '--noise', 'default', '--champion-games', '400', '--anchor-games', '200',
        '--workers', "$leagueWorkers", '--log', 'runs\league\second_model.jsonl'))
}

Write-Both ("== done ({0}). Send back the log: {1}" -f (Get-Date -Format 'HH:mm:ss'), $log) 'Green'
$writer.Close()
