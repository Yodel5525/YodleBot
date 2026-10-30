<#
.SYNOPSIS
    Why the second model lost: the checks of docs\SPARRING_RUNBOOK.md section 19 ("Why the
    second model lost"), one after another, before the third model is run.

.DESCRIPTION
    dt_big_hog (decision-time rows) won 29.3% where live_big_hog won 46.0%, both at a
    21-tick delay. The leading explanation: in decision-time rows the waits between a tap
    and its landing are labelled "wait" while the tapped card is still in hand and the bar
    unspent, and the legacy model cannot see that a play is in flight, so it learns to hold
    back at the very moment it should play. These checks test that and the alternatives.
    Nothing here changes the rows or the earlier models; the new models get new names.

    Each check that fails is reported and the script goes on with the next one. Everything
    printed also goes to reports\dt_diagnose_<time>.log; the last block of the log is a
    summary of the result lines. Close MuMu and the browser first.

        powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\dt_diagnose.ps1

    The checks (-Steps, comma-separated, in this order whatever the order given):
      headroom  tools\rs_dt_diag.py rows on data\rs_rows_dt and data\rs_rows_live: the share of
                waits with an own play in flight, and the elixir headroom of the Hog 2.6 plays
                (what the recorded tick means). Read-only, a few minutes.
      waitbias  dt_big_hog with a smaller wait bias (wait_keep 0.5 and 1.0 instead of 0.25),
                rs_eval at --delay 21. Copies of best.pt in runs\tokens\dt_big_hog_wk050 and
                dt_big_hog_wk100. About 16 min.
      live      live_big_hog and dt_big_hog at --delay 21 --decide live (the live runner's
                decision rule, for both seats). About 18 min.
      offset    dt_big_hog at --delay 24. About 8 min.
      ablation  the Hog 2.6 fine-tune again without the waits with an own play in flight
                (sim.train_tokens --drop-pending-waits, runs\tokens\dt_big_hog_np), then
                rs_eval at --delay 21. About 20 min.
      trust     the Hog 2.6 fine-tune again with --trust legacy (runs\tokens\dt_big_hog_tl),
                then rs_eval at --delay 21. About 20 min.
      ticks     rs_eval --pro with and without --delay 21, then tools\rs_dt_diag.py ghost.
                About 10 min.
    Default: headroom,waitbias,live,offset,ablation (about 75 min). 'all' runs every check.

.PARAMETER Workers
    Processes for rs_eval (default 8). Use 6 if RAM runs short.

.PARAMETER Steps
    The checks to run, comma-separated (see above), or 'all'. Example: -Steps trust,ticks
#>
param(
    [ValidateRange(1, 64)][int]$Workers = 8,
    [string]$Steps = 'headroom,waitbias,live,offset,ablation'
)

# Continue, not Stop: native programs write warnings to stderr, and under Stop
# Windows PowerShell would end the script on the first one. Failures are caught
# from each check's exit code instead.
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
$log = Join-Path $root ("reports\dt_diagnose_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
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

# The lines the summary repeats: rs_eval's result line, rs_dt_diag's numbers, the trainer's
# row count, its --drop-pending-waits count and its last validation line.
$keep = '(: \d+/\d+ wins \()|^(manifest:|decision:|pending waits:|headroom: [<0-9]|headroom at|ghost:|--drop-pending-waits:|rows: train|epoch 2: val)'
$summary = New-Object System.Collections.Generic.List[string]
$failed = New-Object System.Collections.Generic.List[string]

# One check: its output to the console and the log, its result lines to the summary. $true
# when it exited 0; a failure is logged and the script goes on.
function Invoke-Check([string]$Name, [string]$Python, [string[]]$ArgList) {
    Write-Both ("== {0}  ({1})  {2} {3}" -f $Name, (Get-Date -Format 'HH:mm:ss'), $Python, ($ArgList -join ' ')) 'Cyan'
    & $Python @ArgList 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Both $line
        if ($line -match $keep) { $summary.Add(("{0}: {1}" -f $Name, $line.Trim())) }
    }
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Both ("ERROR: '{0}' failed with exit code {1}; going on with the next check." -f $Name, $code) 'Red'
        $failed.Add($Name)
        return $false
    }
    return $true
}

$known = @('headroom', 'waitbias', 'live', 'offset', 'ablation', 'trust', 'ticks')
$asked = @()
# Commas or spaces: -File passes "trust,ticks" as it is, PowerShell itself joins an array
# given to a [string] parameter with spaces.
foreach ($s in ($Steps -split '[,\s]+')) {
    $s = $s.Trim().ToLowerInvariant()
    if (-not $s) { continue }
    if ($s -eq 'all') { $asked += $known; continue }
    if ($known -notcontains $s) {
        Stop-Run ("ERROR: unknown check '{0}' in -Steps. The checks: {1}, or all." -f $s, ($known -join ', '))
    }
    $asked += $s
}
$run = @($known | Where-Object { $asked -contains $_ })
if ($run.Count -eq 0) { Stop-Run "ERROR: -Steps names no check." }

$rowsDt = 'data\rs_rows_dt'
$rowsLive = 'data\rs_rows_live'
$first = 'runs\tokens\live_big_hog\best.pt'        # the first model (scripts\first_model.ps1)
$opponent = 'runs\tokens\live_big\ep01.pt'         # the reactive opponent of every evaluation
$general = 'runs\tokens\dt_big\best.pt'            # the second model (scripts\second_model.ps1)
$second = 'runs\tokens\dt_big_hog\best.pt'
# The Hog 2.6 fine-tune of scripts\second_model.ps1, flags for flags; the checks add one each.
$fineTune = @('-m', 'sim.train_tokens', '--data', $rowsDt, '--wait-keep', '0.25', '--epochs', '3',
              '--d', '256', '--layers', '6', '--heads', '8', '--finetune-hog', '--init', $general)
$vs = @('--opponent-ckpt', $opponent, '--matches', '300', '--workers', "$Workers")

Write-Both ("dt_diagnose.ps1: checks {0}, workers {1}. Log: {2}" -f ($run -join ', '), $Workers, $log) 'Cyan'

# What the checks read, checked now rather than an hour in.
$need = @()
if ($run | Where-Object { @('waitbias', 'live', 'offset', 'ablation', 'trust') -contains $_ }) { $need += $opponent }
if ($run | Where-Object { @('waitbias', 'live', 'offset') -contains $_ }) { $need += $second }
if ($run -contains 'live') { $need += $first }
if ($run | Where-Object { @('headroom', 'ablation', 'trust') -contains $_ }) { $need += "$rowsDt\manifest.json" }
if ($run | Where-Object { @('ablation', 'trust') -contains $_ }) { $need += $general }
foreach ($f in ($need | Select-Object -Unique)) {
    if (-not (Test-Path (Join-Path $root $f))) {
        Stop-Run ("ERROR: {0} is missing. live_big_hog and live_big come from scripts\first_model.ps1 (runbook section 14); dt_big, dt_big_hog and {1} from scripts\second_model.ps1 (section 19)." -f $f, $rowsDt)
    }
}

if ($run -contains 'headroom') {
    Invoke-Check 'headroom: decision-time rows' $RG @('tools\rs_dt_diag.py', 'rows', $rowsDt) | Out-Null
    if (Test-Path (Join-Path $root "$rowsLive\vocab.json")) {
        Invoke-Check 'headroom: rows at execution (first model)' $RG @('tools\rs_dt_diag.py', 'rows', $rowsLive) | Out-Null
    } else {
        Write-Both ("note: {0} is not there (deleted to save disk?); the tick check on rows at execution is skipped. -Steps ticks checks the same question another way." -f $rowsLive) 'Yellow'
    }
}

if ($run -contains 'waitbias') {
    # rs_eval adds log(1 / wait_keep) to the wait logit, wait_keep read from the args.json next
    # to the checkpoint (tools\rs_decide.py wait_keep_for). A copy of best.pt with another
    # args.json plays the same model with a smaller wait bias: 0.69 and 0 instead of 1.39.
    $biases = [ordered]@{ '0.5' = 'dt_big_hog_wk050'; '1.0' = 'dt_big_hog_wk100' }
    foreach ($k in $biases.Keys) {
        $dir = Join-Path $root ('runs\tokens\' + $biases[$k])
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Copy-Item -Force (Join-Path $root $second) (Join-Path $dir 'best.pt')
        # No byte-order mark: json.loads refuses one, and rs_decide would fall back to 0.35.
        [System.IO.File]::WriteAllText((Join-Path $dir 'args.json'), ('{"wait_keep": "' + $k + '"}'), (New-Object System.Text.UTF8Encoding($false)))
        $ck = Join-Path $dir 'best.pt'
        # No double quotes in the probe: PowerShell 5.1 passes them to python unescaped.
        $probe = 'import sys; sys.path[:0] = [''tools'']; import rs_decide; print(rs_decide.wait_keep_for(sys.argv[1]))'
        $got = "$(& $RG -c $probe $ck 2>&1 | Select-Object -Last 1)".Trim()
        # A string comparison: python prints 0.5 and 1.0, whatever Windows' decimal separator.
        if ($got -eq $k) {
            Write-Both ("{0}: rs_eval reads wait_keep {1}" -f $ck, $got)
        } else {
            Write-Both ("warning: {0}: rs_decide.wait_keep_for read '{1}', not {2}; this run's bias is not the one asked for." -f $ck, $got, $k) 'Yellow'
        }
        Invoke-Check ("wait bias: dt_big_hog, wait_keep {0}, 21-tick delay" -f $k) $RG (@('tools\rs_eval.py', '--ckpt', $ck) + $vs + @('--delay', '21')) | Out-Null
    }
}

if ($run -contains 'live') {
    # --decide live switches both seats, the opponent too: compare these two with each other.
    Invoke-Check 'live rule: live_big_hog, 21-tick delay' $RG (@('tools\rs_eval.py', '--ckpt', $first) + $vs + @('--delay', '21', '--decide', 'live')) | Out-Null
    Invoke-Check 'live rule: dt_big_hog, 21-tick delay' $RG (@('tools\rs_eval.py', '--ckpt', $second) + $vs + @('--delay', '21', '--decide', 'live')) | Out-Null
}

if ($run -contains 'offset') {
    Invoke-Check 'offset: dt_big_hog, 24-tick delay' $RG (@('tools\rs_eval.py', '--ckpt', $second) + $vs + @('--delay', '24')) | Out-Null
}

if ($run -contains 'ablation') {
    $ok = Invoke-Check 'ablation: fine-tune without the waits with a play in flight' $BB ($fineTune + @('--name', 'dt_big_hog_np', '--trust', 'v2', '--drop-pending-waits'))
    if ($ok) {
        Invoke-Check 'ablation: dt_big_hog_np, 21-tick delay' $RG (@('tools\rs_eval.py', '--ckpt', 'runs\tokens\dt_big_hog_np\best.pt') + $vs + @('--delay', '21')) | Out-Null
    }
}

if ($run -contains 'trust') {
    $ok = Invoke-Check 'trust: fine-tune with --trust legacy' $BB ($fineTune + @('--name', 'dt_big_hog_tl', '--trust', 'legacy'))
    if ($ok) {
        Invoke-Check 'trust: dt_big_hog_tl, 21-tick delay' $RG (@('tools\rs_eval.py', '--ckpt', 'runs\tokens\dt_big_hog_tl\best.pt') + $vs + @('--delay', '21')) | Out-Null
    }
}

if ($run -contains 'ticks') {
    $pro = @('tools\rs_eval.py', '--pro', '--matches', '300', '--workers', "$Workers")
    $a = Invoke-Check 'ticks: recorded pro, no delay' $RG $pro
    $b = Invoke-Check 'ticks: recorded pro, 21-tick delay' $RG ($pro + @('--delay', '21'))
    if ($a -and $b) {
        Invoke-Check 'ticks: match by match' $RG @('tools\rs_dt_diag.py', 'ghost', '--delay', '21') | Out-Null
    }
}

Write-Both ("== summary ({0}). How to read it: docs\SPARRING_RUNBOOK.md section 19, 'Why the second model lost'. For reference, second_model.ps1 measured: A live_big_hog 46.0% (33.3 plays), B dt_big_hog 29.3% (28.7 plays), both at --delay 21; C dt_big_hog without delay 21.0%." -f (Get-Date -Format 'HH:mm:ss')) 'Green'
foreach ($line in $summary) { Write-Both $line }
if ($failed.Count -gt 0) {
    Write-Both ("{0} check(s) failed: {1}. Send this log back: {2}" -f $failed.Count, ($failed -join '; '), $log) 'Red'
    $writer.Close()
    exit 1
}
Write-Both ("== done. Send back the log: {0}" -f $log) 'Green'
$writer.Close()
