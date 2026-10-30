<#
.SYNOPSIS
    Phase 3, the first live-compatible model. Builds the rows, trains a general model,
    fine-tunes it on Hog 2.6, and runs two simulator evaluations, one after another.

.DESCRIPTION
    These are the steps of docs\SPARRING_RUNBOOK.md section 14, in order. The script
    stops at the first step that fails. Everything printed also goes to
    reports\first_model_<time>.log, so the run can be left alone (overnight, say) and
    the log sent back afterwards.

    Close MuMu and the browser first: these jobs use most of a 16 GB PC's memory.

        powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\first_model.ps1

.PARAMETER Workers
    Processes for row building and evaluation (default 4: the PC froze at 8 rs_eval workers
    on 2026-10-30, docs\SPARRING_RUNBOOK.md "If something breaks", "If the PC freezes"). 6
    only after a watched 4-worker run left more than about 7 GB of memory available; not 8
    until the freeze is explained.

.PARAMETER Others
    Non-Hog matches added to the rows (default 20000, as the author used).

.PARAMETER MaxRows
    Train on a uniform sample of at most this many rows (default 0 = all). Use it if
    training runs out of memory, e.g. -From general -MaxRows 4000000.

.PARAMETER From
    Start at a later step, skipping the earlier ones: rows, general, hog or eval.
    Example: -From hog re-runs the fine-tune and the evaluations only.
#>
param(
    [int]$Workers = 4,
    [int]$Others = 20000,
    [long]$MaxRows = 0,
    [ValidateSet('rows', 'general', 'hog', 'eval')][string]$From = 'rows'
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
$log = Join-Path $root ("reports\first_model_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
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

$order = @('rows', 'general', 'hog', 'eval')
# ValidateSet accepts any case (-From Eval) but IndexOf compares exactly; lower-case it.
$start = [array]::IndexOf($order, $From.ToLowerInvariant())
$model = @('--wait-keep', '0.25', '--epochs', '3', '--d', '256', '--layers', '6', '--heads', '8')
if ($MaxRows -gt 0) { $model += @('--max-rows', "$MaxRows") }
Write-Both ("first_model.ps1: from '{0}', workers {1}, others {2}, max rows {3}. Log: {4}" -f $From, $Workers, $Others, $(if ($MaxRows -gt 0) { $MaxRows } else { 'all' }), $log) 'Cyan'

if ($start -le 0) {
    Invoke-Step 'rows' $RG (@('tools\rs_rows.py', '--live-compat', '--others', "$Others",
                              '--workers', "$Workers", '--out', 'data\rs_rows_live'))
}
if ($start -le 1) {
    Invoke-Step 'general model' $BB (@('-m', 'sim.train_tokens', '--name', 'live_big',
                                       '--data', 'data\rs_rows_live') + $model)
}
if ($start -le 2) {
    Invoke-Step 'Hog 2.6 fine-tune' $BB (@('-m', 'sim.train_tokens', '--name', 'live_big_hog',
                                           '--data', 'data\rs_rows_live') + $model +
                                         @('--finetune-hog', '--init', 'runs\tokens\live_big\best.pt'))
}
if ($start -le 3) {
    $eval = @('tools\rs_eval.py', '--ckpt', 'runs\tokens\live_big_hog\best.pt',
              '--opponent-ckpt', 'runs\tokens\live_big\ep01.pt', '--matches', '300',
              '--workers', "$Workers")
    Invoke-Step 'evaluation, no delay' $RG $eval
    Invoke-Step 'evaluation, 21-tick delay' $RG ($eval + @('--delay', '21'))
}

Write-Both ("== done ({0}). Send back the log: {1}" -f (Get-Date -Format 'HH:mm:ss'), $log) 'Green'
$writer.Close()
