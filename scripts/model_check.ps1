<#
.SYNOPSIS
    Checks one or more checkpoints in the simulator, unattended: tools\rs_eval.py at each
    wait_keep and decision rule asked for, the champion under the same settings as the
    control, and with -League the promotion gate of tools\rs_league_eval.py.

.DESCRIPTION
    For each checkpoint (-Ckpt), each wait_keep (-Keeps) and each decision rule (-Rules), one
    rs_eval run: -Matches held-out Hog 2.6 matches against the reactive -Opponent at a -Delay
    tick command delay, the instrument of docs\SPARRING_RUNBOOK.md section 19. Right after each
    setting the champion (-Champion, default live_big_hog, the live model) runs under the same
    setting as the control, so every number has its control in the same log (-NoBaseline
    leaves the control out). A run that was already made earlier in the same run is not made
    again; the summary repeats its numbers.

    wait_keep: rs_eval and the live runner add log(1 / wait_keep) to the wait logit, with
    wait_keep read from the args.json next to the checkpoint (tools\rs_decide.py wait_keep_for;
    0.25 gives +1.386, 1.0 gives 0). 'own' is the checkpoint's own args.json, and a checkpoint
    without one is refused for 'own'. A number plays a copy: best.pt copied to
    runs\tokens\<name>_wkNNN\ with an args.json that holds that wait_keep (wk100 = 1.0), as
    scripts\dt_diagnose.ps1 made them. Before anything runs, each copy is read back through
    rs_decide.wait_keep_for. scripts\vision_play.py reads the same file, so a copy is also what
    its --ckpt points at to play live at that wait_keep. An existing copy is used only when it
    holds the same best.pt as its source; when the source changed, or another checkpoint of the
    same name made it, the script stops instead of replacing it (it may be the live model).

    Rules: 'plain' is rs_eval's own decision rule. 'live' is --decide live, the live runner's
    rule, which rs_eval gives both seats (the opponent too): compare live runs with live runs.

    -League: after the rs_eval runs, tools\rs_league_eval.py run plays the first -Ckpt (at its
    own wait_keep, or at -LeagueKeep) as the candidate against the -Champion, with live_big
    epoch 1 as the anchor, at the settings of scripts\third_model.ps1's league step: paired
    games on both seats, a 24-tick delay, the -Noise profile on every seat, 1,000 games against
    the champion and 400 against the anchor for each (--full), at most 6 workers. Its log is
    -LeagueLog; a rerun plays only the games the log lacks. The gate's decision is repeated in
    the summary. A checkpoint is a league player named after its folder (best.pt) or its folder
    and file name (other files). The script stops at the start when -LeagueLog already uses a
    name of this run for another checkpoint, or another league log uses the candidate's; for
    the champion's or the anchor's name in another log it prints a warning. Every candidate
    plays the current champion (the last model that passed), never a former one: give it with
    -Champion once live_big_hog is no longer the champion.

    Everything printed also goes to reports\model_check_<time>.log; its last block is a summary
    table. A run that fails is logged and the script goes on with the next; it exits 1 at the
    end when any failed. Close MuMu and the browser first. Nothing here touches the emulator.

        powershell -NoProfile -ExecutionPolicy Bypass -File C:\hcr\scripts\model_check.ps1 -League

    Time: an rs_eval run of 300 matches takes about 10 minutes at 8 workers, so the default (2
    wait_keeps x 2 rules, for the checkpoint and the control) takes about 80 minutes. The league
    takes about 1-1.5 hours for its 1,800 games on a new log (an estimate until a league has
    been timed on the PC), less when the log holds some.

.PARAMETER Ckpt
    One or more checkpoints, comma-separated (default runs\tokens\dt_big_hog_np\best.pt).
    Relative paths are read from the repo root. Example: -Ckpt runs\tokens\v2_hog_20k\best.pt

.PARAMETER Keeps
    The wait_keeps, comma-separated (default 'own,1.0'): 'own' or numbers above 0 and at most 10
    with at most two decimals. Example: -Keeps 'own,0.5,1.0'

.PARAMETER Rules
    The decision rules, comma-separated (default 'plain,live'): plain (rs_eval's own rule) and
    live (--decide live).

.PARAMETER Delay
    rs_eval's command delay in ticks (default 21). The league always plays at 24.

.PARAMETER Matches
    rs_eval matches per run (default 300).

.PARAMETER Workers
    rs_eval processes (default 8). Use 6 if RAM runs short. The league uses at most 6, and
    tools\rs_league_eval.py lowers that further when free memory is short.

.PARAMETER Opponent
    rs_eval's reactive opponent (default runs\tokens\live_big\ep01.pt).

.PARAMETER League
    After the rs_eval runs, the league step (above).

.PARAMETER LeagueOnly
    The league step alone, no rs_eval runs (implies -League).

.PARAMETER LeagueKeep
    The candidate's wait_keep in the league: 'own' (default) or a number, which plays the
    runs\tokens\<name>_wkNNN copy under the copy's name.

.PARAMETER LeagueName
    The candidate's league name instead of the one made from its path (letters, digits, '.',
    '_' and '-'), for when a league log already uses that name for another checkpoint.

.PARAMETER LeagueLog
    The league's JSONL log (default runs\league\model_check.jsonl). Its pairings and report go
    next to it. runs\league\third_model.jsonl reuses the games scripts\third_model.ps1 played.

.PARAMETER Champion
    The league's champion and the rs_eval control (default runs\tokens\live_big_hog\best.pt).
    Give the current champion's checkpoint: the last model whose league said PASS.

.PARAMETER Noise
    The league's perception-noise profile (default data\noise\default_prior.json, the prior, as
    in scripts\third_model.ps1).

.PARAMETER NoBaseline
    No control runs of the champion.
#>
param(
    [string[]]$Ckpt = @('runs\tokens\dt_big_hog_np\best.pt'),
    [string]$Keeps = 'own,1.0',
    [string]$Rules = 'plain,live',
    [ValidateRange(0, 200)][int]$Delay = 21,
    [ValidateRange(1, 100000)][int]$Matches = 300,
    [ValidateRange(1, 64)][int]$Workers = 8,
    [ValidateNotNullOrEmpty()][string]$Opponent = 'runs\tokens\live_big\ep01.pt',
    [switch]$League,
    [switch]$LeagueOnly,
    [ValidateNotNullOrEmpty()][string]$LeagueKeep = 'own',
    [ValidatePattern('^[A-Za-z0-9_.-]*$')][string]$LeagueName = '',
    [ValidateNotNullOrEmpty()][string]$LeagueLog = 'runs\league\model_check.jsonl',
    [ValidateNotNullOrEmpty()][string]$Champion = 'runs\tokens\live_big_hog\best.pt',
    [ValidateNotNullOrEmpty()][string]$Noise = 'data\noise\default_prior.json',
    [switch]$NoBaseline
)

# Continue, not Stop: native programs write warnings to stderr, and under Stop
# Windows PowerShell would end the script on the first one. Failures are caught
# from each run's exit code instead.
$ErrorActionPreference = 'Continue'

# PowerShell's regex match operators and the regex switch statement store their groups in the
# automatic variable of the same name as the -Matches parameter. This script never uses them
# ([regex]::Match and [regex]::IsMatch instead) and reads the count once, here.
$matchCount = $Matches

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$RG = Join-Path $root '.venvs\royalegym\Scripts\python.exe'
if (-not (Test-Path $RG)) {
    Write-Host "ERROR: $RG is missing. Run scripts\setup_windows.ps1 first." -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path (Join-Path $root 'reports') | Out-Null
$log = Join-Path $root ("reports\model_check_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$writer = New-Object System.IO.StreamWriter($log, $true, (New-Object System.Text.UTF8Encoding($false)))
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

$failedSteps = New-Object System.Collections.Generic.List[string]

# One run: its output to the console and the log, every line kept for the caller. A failure
# (an exit code other than 0) is logged and the script goes on.
function Invoke-Check([string]$StepName, [string]$Exe, [string[]]$ExeArgs) {
    Write-Both ("== {0}  ({1})  {2} {3}" -f $StepName, (Get-Date -Format 'HH:mm:ss'), $Exe, ($ExeArgs -join ' ')) 'Cyan'
    $kept = New-Object System.Collections.Generic.List[string]
    & $Exe @ExeArgs 2>&1 | ForEach-Object {
        $outLine = "$_"
        Write-Both $outLine
        $kept.Add($outLine)
    }
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Both ("ERROR: '{0}' failed with exit code {1}; going on with the next step." -f $StepName, $code) 'Red'
        $script:failedSteps.Add($StepName)
    }
    return [pscustomobject]@{ Ok = ($code -eq 0); Code = $code; Lines = $kept }
}

# The non-empty parts of the given strings split at $SplitPattern. -File passes "own,1.0" as it
# is, and PowerShell itself joins an array given to a [string] parameter with spaces.
function Split-List([string[]]$ListItems, [string]$SplitPattern) {
    $parts = @()
    foreach ($item in $ListItems) {
        if ($null -eq $item) { continue }
        foreach ($piece in ($item -split $SplitPattern)) {
            $trimmed = $piece.Trim()
            if ($trimmed) { $parts += $trimmed }
        }
    }
    return $parts
}

# A number written with a decimal point whatever Windows' decimal separator (python prints 0.25),
# or $null.
function ConvertTo-Number([string]$NumberText) {
    $parsed = 0.0
    if ([double]::TryParse($NumberText, [System.Globalization.NumberStyles]::Float, $script:invariant, [ref]$parsed)) {
        if (-not ([double]::IsNaN($parsed) -or [double]::IsInfinity($parsed))) { return $parsed }
    }
    return $null
}

function Format-Keep([double]$KeepValue) {
    return $KeepValue.ToString('0.0#', $script:invariant)
}

# A wait_keep given as a number: above 0, at most 10, at most two decimals (a copy's folder name
# holds it in hundredths: wk100 = 1.0, wk025 = 0.25).
function Test-KeepValue($KeepCandidate) {
    if ($null -eq $KeepCandidate) { return $false }
    $x = [double]$KeepCandidate
    if ($x -le 0 -or $x -gt 10) { return $false }
    $hundredths = $x * 100
    return ([Math]::Abs($hundredths - [Math]::Round($hundredths)) -lt 1e-6)
}

# A path as the tools are given it: relative to the repo root when it is under it (with
# backslashes, as scripts\third_model.ps1 passes them, so that a league log names a checkpoint
# the same way from every script), else absolute. Name: the checkpoint's folder for best.pt,
# else folder_file (runs\rl\hog_rl1\it0010.pt: hog_rl1_it0010).
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

# What rs_eval and the live runner read as a checkpoint's wait_keep: rs_decide.wait_keep_for,
# but 'none' when the args.json next to it is missing or has no wait_keep (wait_keep_for would
# fall back to 0.35 without a word). No double quotes in the probe: PowerShell 5.1 passes them
# to python unescaped.
$keepProbe = 'import sys, json, pathlib; sys.path[:0] = [''tools'']; import rs_decide; p = pathlib.Path(sys.argv[1]).with_name(''args.json''); d = json.loads(p.read_text(encoding=''utf-8'')) if p.is_file() else {}; print(rs_decide.wait_keep_for(sys.argv[1]) if isinstance(d, dict) and ''wait_keep'' in d else ''none'')'
$ownKeeps = @{}
function Read-WaitKeep([string]$ModelPath) {
    $outLines = @(& $script:RG -c $script:keepProbe $ModelPath 2>&1 | ForEach-Object { "$_".Trim() })
    $answer = ''
    for ($i = $outLines.Count - 1; $i -ge 0; $i--) {
        if ($outLines[$i] -eq 'none' -or $null -ne (ConvertTo-Number $outLines[$i])) { $answer = $outLines[$i]; break }
    }
    if (-not $answer -and $outLines.Count -gt 0) { $answer = $outLines[$outLines.Count - 1] }
    $value = $null
    if ($answer -and $answer -ne 'none') { $value = ConvertTo-Number $answer }
    return [pscustomobject]@{ Text = $answer; Value = $value }
}

function Get-OwnKeep($OwnModel) {
    $ownKey = $OwnModel.Full.ToLowerInvariant()
    if (-not $script:ownKeeps.ContainsKey($ownKey)) { $script:ownKeeps[$ownKey] = Read-WaitKeep $OwnModel.Full }
    return $script:ownKeeps[$ownKey]
}

# The checkpoint at another wait_keep: best.pt copied to runs\tokens\<name>_wkNNN\ with an
# args.json that holds the wait_keep (no byte-order mark: json.loads refuses one, and rs_decide
# would fall back to 0.35), read back through rs_decide.wait_keep_for. A folder that is not such a
# copy (no args.json from this script or dt_diagnose.ps1) is never overwritten. An existing copy
# is used only when it holds the same best.pt (SHA-256) and, if this script made it, the same
# source path: a copy may be the live model (vision_play --ckpt), so a retrained or another
# source of the same name stops the run instead of replacing the copy's weights.
$copies = @{}
function New-KeepCopy($SourceModel, [double]$KeepValue) {
    $copyKey = '{0}|{1}' -f $SourceModel.Full.ToLowerInvariant(), (Format-Keep $KeepValue)
    if ($script:copies.ContainsKey($copyKey)) { return $script:copies[$copyKey] }
    $keepText = Format-Keep $KeepValue
    $dirRel = 'runs\tokens\{0}_wk{1:000}' -f $SourceModel.Name, [int][Math]::Round($KeepValue * 100)
    $dir = Join-Path $script:root $dirRel
    $argsPath = Join-Path $dir 'args.json'
    $bestPath = Join-Path $dir 'best.pt'
    $sourceRel = $SourceModel.Rel.Replace('\', '/')
    if (Test-Path -LiteralPath $dir) {
        $isCopy = $false
        $oldSource = ''
        if (Test-Path -LiteralPath $argsPath) {
            $old = [System.IO.File]::ReadAllText($argsPath)
            $isCopy = $old.Contains('"made_by": "scripts/model_check.ps1"') -or [regex]::IsMatch($old.Trim(), '^\{"wait_keep": "[0-9.]+"\}$')
            if ($old.Contains('"made_by": "scripts/model_check.ps1"')) {
                try { $oldSource = [string](($old | ConvertFrom-Json).copy_of) } catch { $oldSource = '' }
            }
        } else {
            $isCopy = (@(Get-ChildItem -LiteralPath $dir -Force)).Count -eq 0
        }
        if (-not $isCopy) {
            Stop-Run ("ERROR: {0} exists and is not a wait_keep copy (its args.json is not this script's or dt_diagnose.ps1's), so it is not overwritten. Move it away, or leave {1} out of -Keeps / -LeagueKeep." -f $dir, $keepText)
        }
        if ($oldSource -and $oldSource -ine $sourceRel) {
            Stop-Run ("ERROR: {0} is a copy of {1}, not of {2}: two checkpoints have the name '{3}'. Nothing has run yet. Check them in separate runs, or move one into a folder with another name." -f $dir, $oldSource, $sourceRel, $SourceModel.Name)
        }
        if (Test-Path -LiteralPath $bestPath -PathType Leaf) {
            $oldDigest = Get-CkptDigest $bestPath
            $sourceDigest = Get-CkptDigest $SourceModel.Full
            if ($oldDigest -ne $sourceDigest) {
                Stop-Run ("ERROR: {0} holds another best.pt (digest {1}) than {2} (digest {3}): the source changed since this copy was made. The copy may be the live model (vision_play --ckpt), so it is not replaced. Nothing has run yet. If it is not the live model, move it away (Move-Item {4} {4}_old) and run again." -f $bestPath, $oldDigest, $SourceModel.Rel, $sourceDigest, $dir)
            }
        }
    }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    if (-not (Test-Path -LiteralPath $bestPath -PathType Leaf)) {
        Copy-Item -LiteralPath $SourceModel.Full -Destination $bestPath -Force
    }
    $json = '{"wait_keep": "' + $keepText + '", "made_by": "scripts/model_check.ps1", "copy_of": "' + $SourceModel.Rel.Replace('\', '/') + '"}'
    [System.IO.File]::WriteAllText($argsPath, $json, (New-Object System.Text.UTF8Encoding($false)))
    $copy = Resolve-Model ($dirRel + '\best.pt')
    $back = Read-WaitKeep $copy.Full
    if ($null -eq $back.Value -or [Math]::Abs($back.Value - $KeepValue) -gt 1e-9) {
        Stop-Run ("ERROR: {0}: rs_decide.wait_keep_for read '{1}', not {2}; a run with it would not have the wait bias asked for." -f $copy.Rel, $back.Text, $keepText)
    }
    Write-Both ("{0}: a copy of {1}; rs_decide reads wait_keep {2}" -f $copy.Rel, $SourceModel.Rel, $back.Text)
    $script:copies[$copyKey] = $copy
    return $copy
}

# The checkpoint to play at a wait_keep spec: Model (the checkpoint itself, or its copy), Value
# (the wait_keep it plays with) and Text (for the label: own(0.25), 1.0).
function Get-EvalModel($BaseModel, $KeepSpec) {
    $own = Get-OwnKeep $BaseModel
    if ($KeepSpec.Spec -eq 'own') {
        if ($null -eq $own.Value) {
            Stop-Run ("ERROR: {0} has no args.json with a wait_keep next to it (the probe read '{1}'), so 'own' has no value: rs_eval and the live runner would play it at 0.35 (the live runner refuses taps only when args.json is missing). sim\train_tokens.py writes that file; copy it next to the checkpoint, or give numbers only in -Keeps / -LeagueKeep." -f $BaseModel.Rel, $own.Text)
        }
        return [pscustomobject]@{ Model = $BaseModel; Value = $own.Value; Text = ('own({0})' -f $own.Text) }
    }
    if ($null -ne $own.Value -and [Math]::Abs($own.Value - $KeepSpec.Value) -lt 1e-9) {
        return [pscustomobject]@{ Model = $BaseModel; Value = $KeepSpec.Value; Text = $KeepSpec.Spec }
    }
    $copy = New-KeepCopy $BaseModel $KeepSpec.Value
    return [pscustomobject]@{ Model = $copy; Value = $KeepSpec.Value; Text = $KeepSpec.Spec }
}

function Read-KeepSpec([string]$SpecText, [string]$ParamName) {
    if ($SpecText.ToLowerInvariant() -eq 'own') { return [pscustomobject]@{ Spec = 'own'; Value = $null } }
    $number = ConvertTo-Number $SpecText
    if (-not (Test-KeepValue $number)) {
        Stop-Run ("ERROR: {0}: '{1}' is neither 'own' nor a wait_keep above 0 and at most 10 with at most two decimals." -f $ParamName, $SpecText)
    }
    return [pscustomobject]@{ Spec = (Format-Keep $number); Value = [double]$number }
}

# 16 hex digits of SHA-256 over the file: tools\rs_league_eval.py's ckpt_digest.
function Get-CkptDigest([string]$DigestPath) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $DigestPath).Hash.ToLowerInvariant().Substring(0, 16)
}

# The players' names in the league logs (runs\league\*.jsonl, runs\rl\*\gate.jsonl and
# -LeagueLog): a name that one of them uses for another checkpoint (another digest) is a clash.
# In -LeagueLog itself another path is one too. Each clash has Text and Fatal: every clash in
# -LeagueLog (tools\rs_league_eval.py would refuse those games or pool them) and the candidate's
# name in any log stop the run; the champion's or the anchor's name in another log is only a
# warning (an old log may name an earlier checkpoint of that folder, and an RL run's gate.jsonl
# is part of its resumable state).
function Find-NameClashes($PlayerList, [string]$OwnLogFull) {
    $logFiles = @()
    $logFiles += @(Get-ChildItem -Path (Join-Path $script:root 'runs\league') -Filter '*.jsonl' -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    $logFiles += @(Get-ChildItem -Path (Join-Path $script:root 'runs\rl\*\gate.jsonl') -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    if (Test-Path -LiteralPath $OwnLogFull) { $logFiles += $OwnLogFull }
    $found = New-Object System.Collections.Generic.List[object]
    $seenClashes = @{}
    $readLogs = @{}
    foreach ($logFile in $logFiles) {
        $logKey = $logFile.ToLowerInvariant()
        if ($readLogs.ContainsKey($logKey)) { continue }
        $readLogs[$logKey] = $true
        $isOwn = ($logKey -eq $OwnLogFull.ToLowerInvariant())
        $hits = @(Select-String -LiteralPath $logFile -SimpleMatch -Pattern '"kind":"league_run"' -ErrorAction SilentlyContinue)
        foreach ($hit in $hits) {
            $header = $null
            try { $header = $hit.Line | ConvertFrom-Json } catch { $header = $null }
            if ($null -eq $header -or $null -eq $header.players) {
                Write-Both ("warning: {0} line {1}: a run header that could not be read; its player names are not checked" -f $logFile, $hit.LineNumber) 'Yellow'
                continue
            }
            foreach ($prop in $header.players.PSObject.Properties) {
                foreach ($player in $PlayerList) {
                    if ($prop.Name -cne $player.Name) { continue }
                    $oldDigest = [string]$prop.Value.ckpt_digest
                    $oldCkpt = [string]$prop.Value.ckpt
                    $why = ''
                    if ($oldDigest -and $oldDigest -ne $player.Digest) {
                        $why = 'another checkpoint ({0}, digest {1}; this one is {2})' -f $oldCkpt, $oldDigest, $player.Digest
                    } elseif ($isOwn -and $oldCkpt -cne $player.Rel) {
                        $why = 'another path ({0}; this run gives {1})' -f $oldCkpt, $player.Rel
                    }
                    if ($why) {
                        $entry = '{0} ({1}): {2} names {3}' -f $player.Name, $player.Role, $logFile, $why
                        if (-not $seenClashes.ContainsKey($entry)) {
                            $seenClashes[$entry] = $true
                            $found.Add([pscustomobject]@{ Text = $entry; Fatal = ($isOwn -or $player.Role -eq 'candidate') })
                        }
                    }
                }
            }
        }
    }
    return $found
}

# --- what to run ---------------------------------------------------------------------------
$runLeague = [bool]($League -or $LeagueOnly)
$runSweep = -not $LeagueOnly
$useBaseline = -not $NoBaseline

$models = New-Object System.Collections.Generic.List[object]
$modelPaths = @{}
foreach ($ckptText in @(Split-List $Ckpt ',')) {
    $model = Resolve-Model $ckptText
    if ($modelPaths.ContainsKey($model.Full.ToLowerInvariant())) { continue }
    $modelPaths[$model.Full.ToLowerInvariant()] = $true
    $models.Add($model)
}
if ($models.Count -eq 0) { Stop-Run "ERROR: -Ckpt names no checkpoint." }

$keepSpecs = New-Object System.Collections.Generic.List[object]
foreach ($keepText in @(Split-List @($Keeps) '[,\s]+')) {
    $spec = Read-KeepSpec $keepText '-Keeps'
    if (-not (@($keepSpecs | Where-Object { $_.Spec -eq $spec.Spec }).Count)) { $keepSpecs.Add($spec) }
}
$ruleList = @()
foreach ($ruleText in @(Split-List @($Rules) '[,\s]+')) {
    $ruleName = $ruleText.ToLowerInvariant()
    if (@('plain', 'live') -notcontains $ruleName) {
        Stop-Run ("ERROR: unknown rule '{0}' in -Rules. The rules: plain (rs_eval's own), live (--decide live)." -f $ruleText)
    }
    if ($ruleList -notcontains $ruleName) { $ruleList += $ruleName }
}
if ($runSweep -and ($keepSpecs.Count -eq 0 -or $ruleList.Count -eq 0)) {
    Stop-Run "ERROR: -Keeps or -Rules names nothing to run (-LeagueOnly runs the league alone)."
}
$leagueSpec = Read-KeepSpec $LeagueKeep '-LeagueKeep'

$opponentModel = Resolve-Model $Opponent
$championModel = Resolve-Model $Champion
# The anchor of scripts\third_model.ps1's league, under its name there.
$anchorModel = Resolve-Model 'runs\tokens\live_big\ep01.pt'
$anchorModel.Name = 'live_big'
$leagueLogModel = Resolve-Model $LeagueLog

# What the runs read, checked now rather than an hour in.
$need = @()
$need += @($models | ForEach-Object { $_ })
if ($runSweep) { $need += $opponentModel }
if (($runSweep -and $useBaseline) -or $runLeague) { $need += $championModel }
if ($runLeague) { $need += $anchorModel }
foreach ($m in $need) {
    if (-not (Test-Path -LiteralPath $m.Full -PathType Leaf)) {
        Stop-Run ("ERROR: {0} is missing. live_big_hog and live_big come from scripts\first_model.ps1 (runbook section 14), dt_big_hog_np from scripts\dt_diagnose.ps1 (section 19), v2_hog_20k and v2_hog from scripts\third_model.ps1 (section 20)." -f $m.Full)
    }
}
foreach ($m in @($models) + @($championModel)) {
    if (-not [regex]::IsMatch($m.Name, '^[A-Za-z0-9_.-]+$')) {
        Stop-Run ("ERROR: {0}: the name '{1}' made from its path is not letters, digits, '.', '_' and '-' only; move the checkpoint into a folder with such a name." -f $m.Full, $m.Name)
    }
}
# One name, one checkpoint: the labels and the wait_keep copies (runs\tokens\<name>_wkNNN) are
# named after it, so two files with one name would share a copy folder.
$byName = @{}
foreach ($m in @($models) + @($championModel)) {
    $nameKey = $m.Name.ToLowerInvariant()
    if ($byName.ContainsKey($nameKey) -and $byName[$nameKey].Full -ine $m.Full) {
        Stop-Run ("ERROR: {0} and {1} both have the name '{2}' (made from their paths): their labels and wait_keep copies (runs\tokens\{2}_wkNNN) would mix. Check them in separate runs, or move one into a folder with another name." -f $byName[$nameKey].Rel, $m.Rel, $m.Name)
    }
    $byName[$nameKey] = $m
}
if ($runLeague) {
    # The names tools\perception_noise.py reads as the prior or as no noise; anything else is a file.
    if (@('default', 'prior', 'default_prior', 'none', 'off', 'zero', '0') -notcontains $Noise.Trim().ToLowerInvariant()) {
        $noisePath = $(if ([System.IO.Path]::IsPathRooted($Noise)) { $Noise } else { Join-Path $root $Noise })
        if (-not (Test-Path $noisePath)) { Stop-Run ("ERROR: the noise profile {0} is missing (-Noise)." -f $noisePath) }
    }
}

Write-Both ("model_check.ps1: checkpoints {0}; wait_keep {1}; rules {2}; rs_eval {3} matches against {4} at a {5}-tick delay, {6} workers; control {7}; league {8}. Log: {9}" -f
    (($models | ForEach-Object { $_.Rel }) -join ', '), (($keepSpecs | ForEach-Object { $_.Spec }) -join ','),
    ($ruleList -join ','), $matchCount, $opponentModel.Rel, $Delay, $Workers,
    $(if ($useBaseline -and $runSweep) { $championModel.Rel } else { 'none' }),
    $(if ($runLeague) { 'yes' } else { 'no' }), $log) 'Cyan'

# The rs_eval runs, in order: for each wait_keep and rule, every checkpoint, then the control.
# A run with the same checkpoint, wait_keep and rule as an earlier one is listed, not run again.
$plan = New-Object System.Collections.Generic.List[object]
$planned = @{}           # run key -> the plan entry that runs it
$controlOf = @{}         # "wait_keep spec|rule" -> the control's run key
function Add-Plan($PlanModel, $PlanSpec, [string]$PlanRule, [string]$PlanRole) {
    $em = Get-EvalModel $PlanModel $PlanSpec
    $runKey = '{0}|{1}|{2}' -f $PlanModel.Full.ToLowerInvariant(), $em.Value.ToString('R', $script:invariant), $PlanRule
    $label = '{0} wk {1} {2} d{3}' -f $PlanModel.Name, $em.Text, $PlanRule, $script:Delay
    if ($PlanRole -eq 'control') { $script:controlOf[$PlanSpec.Spec + '|' + $PlanRule] = $runKey }
    $same = ''
    if ($script:planned.ContainsKey($runKey)) { $same = $script:planned[$runKey].Label }
    $entry = [pscustomobject]@{ Key = $runKey; Label = $label; Role = $PlanRole; Spec = $PlanSpec.Spec;
                                Rule = $PlanRule; Ckpt = $em.Model.Rel; Same = $same }
    if (-not $same) { $script:planned[$runKey] = $entry }
    $script:plan.Add($entry)
}
if ($runSweep) {
    foreach ($spec in $keepSpecs) {
        foreach ($ruleName in $ruleList) {
            foreach ($model in $models) { Add-Plan $model $spec $ruleName 'candidate' }
            if ($useBaseline) { Add-Plan $championModel $spec $ruleName 'control' }
        }
    }
}
$toRun = @($plan | Where-Object { -not $_.Same }).Count

# The league's players, their names and checkpoints checked now rather than after the runs.
$leagueWorkers = [Math]::Min($Workers, 6)
if ($runLeague) {
    $candEval = Get-EvalModel $models[0] $leagueSpec
    $candModel = $candEval.Model
    $candName = $(if ($LeagueName) { $LeagueName } else { $candModel.Name })
    $champOwn = Get-OwnKeep $championModel
    if ($null -eq $champOwn.Value) {
        Stop-Run ("ERROR: the champion {0} has no args.json with a wait_keep next to it (the probe read '{1}'): the league would play it at 0.35." -f $championModel.Rel, $champOwn.Text)
    }
    $leaguePlayers = @(
        [pscustomobject]@{ Role = 'candidate'; Name = $candName; Rel = $candModel.Rel; Full = $candModel.Full; Digest = (Get-CkptDigest $candModel.Full) },
        [pscustomobject]@{ Role = 'champion'; Name = $championModel.Name; Rel = $championModel.Rel; Full = $championModel.Full; Digest = (Get-CkptDigest $championModel.Full) },
        [pscustomobject]@{ Role = 'anchor'; Name = $anchorModel.Name; Rel = $anchorModel.Rel; Full = $anchorModel.Full; Digest = (Get-CkptDigest $anchorModel.Full) })
    for ($i = 0; $i -lt $leaguePlayers.Count; $i++) {
        for ($j = $i + 1; $j -lt $leaguePlayers.Count; $j++) {
            $a = $leaguePlayers[$i]
            $b = $leaguePlayers[$j]
            if ($a.Full -ieq $b.Full) {
                Stop-Run ("ERROR: the league's {0} and {1} are the same checkpoint, {2}. To test the champion at another wait_keep, give -LeagueKeep (for example 1.0): its copy is another player." -f $a.Role, $b.Role, $a.Rel)
            }
            if ($a.Name -ieq $b.Name) {
                Stop-Run ("ERROR: the league's {0} and {1} would both be named '{2}'. Give the candidate another name with -LeagueName." -f $a.Role, $b.Role, $a.Name)
            }
        }
    }
    $clashes = @(Find-NameClashes $leaguePlayers $leagueLogModel.Full)
    foreach ($c in @($clashes | Where-Object { -not $_.Fatal })) {
        Write-Both ("warning: name clash in another log: {0}. This run's games go to {1}, which the league checks; that other log keeps its own games." -f $c.Text, $leagueLogModel.Rel) 'Yellow'
    }
    $fatalClashes = @($clashes | Where-Object { $_.Fatal })
    if ($fatalClashes.Count -gt 0) {
        foreach ($c in $fatalClashes) { Write-Both ("name clash: {0}" -f $c.Text) 'Red' }
        Stop-Run ("ERROR: a player name of this run already means another checkpoint (above): in {0}, where the league would refuse it, or for the candidate in another log, where one name must mean one checkpoint. For the candidate give another name with -LeagueName. For the champion or the anchor in {0}, its checkpoint changed since those games were played: restore it, or give another -LeagueLog (or another -Champion). Nothing has run yet." -f $leagueLogModel.Rel)
    }
    Write-Both ("league: {0}={1} (wait_keep {2}) against the champion {3}={4} (wait_keep {5}), anchor {6}={7}; 24-tick delay, noise {8}, 1,000 + 400 games (--full), at most {9} workers; log {10}" -f
        $candName, $candModel.Rel, $candEval.Text, $championModel.Name, $championModel.Rel,
        $champOwn.Text, $anchorModel.Name, $anchorModel.Rel, $Noise, $leagueWorkers,
        $leagueLogModel.Rel)
}

$evalMinutes = [Math]::Ceiling($toRun * 10.0 * ($matchCount / 300.0) * (8.0 / $Workers))
Write-Both ("plan: {0} rs_eval run(s), about {1} min (about 10 min per 300 matches at 8 workers){2}" -f
    $toRun, $evalMinutes,
    $(if ($runLeague) { '; then the league, about 1-1.5 h for 1,800 games on a new log (the games its log holds are not played again)' } else { '' }))
foreach ($e in $plan) {
    $note = $(if ($e.Same) { '  the same run as ' + $e.Same + ': not run again' } else { '' })
    Write-Both ("  {0}  ({1}){2}" -f $e.Label, $e.Ckpt, $note)
}

# --- the rs_eval runs ----------------------------------------------------------------------
# rs_eval's result line, e.g. "dt_big_hog_np delay 21 vs reactive live_big: 242/300 wins (80.7%,
# 95% [75.8%, 84.7%]) 1 draws, crown diff +0.79, agent plays/match 37.4 (586s)".
$evalPattern = ': (\d+)/(\d+) wins \(([0-9.]+)%, 95% \[([0-9.]+)%, ([0-9.]+)%\]\) (\d+) draws, crown diff ([+-]?[0-9.]+), agent plays/match ([0-9.]+)'
function Read-EvalResult($CheckResult) {
    foreach ($resultLine in $CheckResult.Lines) {
        $found = [regex]::Match($resultLine, $script:evalPattern)
        if ($found.Success) {
            return [pscustomobject]@{
                Wins = $found.Groups[1].Value; N = $found.Groups[2].Value; Pct = $found.Groups[3].Value
                PctValue = (ConvertTo-Number $found.Groups[3].Value); Lo = $found.Groups[4].Value
                Hi = $found.Groups[5].Value; Draws = $found.Groups[6].Value
                Crowns = $found.Groups[7].Value; Plays = $found.Groups[8].Value }
        }
    }
    return $null
}

#region rs_eval flags (tests\pipeline\test_model_check_ps1.py checks each against tools\rs_eval.py)
function Get-EvalArgs([string]$EvalCkpt, [string]$EvalRule) {
    $evalArgs = @('tools\rs_eval.py', '--ckpt', $EvalCkpt, '--opponent-ckpt', $script:opponentModel.Rel,
                  '--matches', "$($script:matchCount)", '--workers', "$($script:Workers)",
                  '--delay', "$($script:Delay)")
    if ($EvalRule -eq 'live') { $evalArgs += @('--decide', 'live') }
    return $evalArgs
}
#endregion

$results = @{}           # run key -> Read-EvalResult, $null for a failed run
foreach ($e in $plan) {
    if ($e.Same) {
        Write-Both ("== {0}: the same run as {1}; not run again" -f $e.Label, $e.Same) 'Cyan'
        continue
    }
    $runTitle = $(if ($e.Role -eq 'control') { $e.Label + ' (control)' } else { $e.Label })
    $check = Invoke-Check $runTitle $RG @(Get-EvalArgs $e.Ckpt $e.Rule)
    $parsed = $null
    if ($check.Ok) {
        $parsed = Read-EvalResult $check
        if ($null -eq $parsed) {
            Write-Both ("ERROR: '{0}' printed no result line; going on with the next step." -f $runTitle) 'Red'
            $failedSteps.Add($runTitle)
        }
    }
    $results[$e.Key] = $parsed
}

# --- the league ----------------------------------------------------------------------------
$leagueCheck = $null
if ($runLeague) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $leagueLogModel.Full) | Out-Null
#region rs_league_eval flags (tests\pipeline\test_model_check_ps1.py checks each against tools\rs_league_eval.py)
    # scripts\third_model.ps1's league step: 1,000 paired games against the champion (both seats,
    # both decks), 400 against the anchor for each of the candidate and the champion; --full
    # plays the anchor games even when the candidate does not beat the champion.
    $leagueArgs = @('tools\rs_league_eval.py', 'run',
                    '--candidate', ('{0}={1}' -f $candName, $candModel.Rel),
                    '--champion', ('{0}={1}' -f $championModel.Name, $championModel.Rel),
                    '--anchor', ('{0}={1}' -f $anchorModel.Name, $anchorModel.Rel),
                    '--delay', '24', '--noise', $Noise,
                    '--champion-games', '1000', '--anchor-games', '400', '--full',
                    '--workers', "$leagueWorkers", '--log', $leagueLogModel.Rel)
#endregion
    $leagueCheck = Invoke-Check ("league: {0} vs {1}, anchor {2}" -f $candName, $championModel.Name, $anchorModel.Name) $RG $leagueArgs
}

# --- the summary ---------------------------------------------------------------------------
Write-Both ("== summary ({0}). rs_eval: {1} matches against {2} at a {3}-tick delay, {4} workers; 'plain' is rs_eval's own rule, 'live' is --decide live (both seats). How to read it: docs\SPARRING_RUNBOOK.md section 19, 'Checking a checkpoint: model_check.ps1'." -f
    (Get-Date -Format 'HH:mm:ss'), $matchCount, $opponentModel.Rel, $Delay, $Workers) 'Green'
if ($plan.Count -gt 0) {
    $width = 5
    foreach ($e in $plan) { if ($e.Label.Length -gt $width) { $width = $e.Label.Length } }
    $row = '{0,-' + $width + '}  {1,7}  {2,5}  {3,6}  {4,-14}  {5,6}  {6,5}  {7,10}'
    Write-Both ($row -f 'label', 'wins', 'draws', 'win %', '95% CI', 'crowns', 'plays', 'vs control')
    foreach ($e in $plan) {
        $res = $results[$e.Key]
        if ($null -eq $res) {
            Write-Both (('{0,-' + $width + '}  {1}') -f $e.Label, 'failed: see its == line above')
            continue
        }
        $vs = ''
        if ($e.Role -eq 'control') {
            $vs = 'control'
        } else {
            $controlKey = $controlOf[$e.Spec + '|' + $e.Rule]
            if ($controlKey -and $controlKey -ne $e.Key -and $null -ne $results[$controlKey]) {
                $vs = ($res.PctValue - $results[$controlKey].PctValue).ToString('+0.0;-0.0;0.0', $invariant)
            }
        }
        $line = $row -f $e.Label, ('{0}/{1}' -f $res.Wins, $res.N), $res.Draws, $res.Pct,
                        ('[{0}, {1}]' -f $res.Lo, $res.Hi), $res.Crowns, $res.Plays, $vs
        if ($e.Same) { $line += '  (the run of ' + $e.Same + ')' }
        Write-Both $line
    }
    Write-Both "vs control: win points above the control at the same wait_keep and rule. At 300 matches an interval is about +-5.6 points, so a difference under about 8 points is not conclusive."
} else {
    Write-Both "no rs_eval runs (-LeagueOnly)."
}

if ($runLeague) {
    Write-Both ("league: {0} ({1}) against {2}, anchor {3}, log {4}:" -f $candName, $candModel.Rel, $championModel.Name, $anchorModel.Name, $leagueLogModel.Rel)
    $decision = ''
    if ($null -ne $leagueCheck) {
        $inBlock = $false
        foreach ($leagueLine in $leagueCheck.Lines) {
            # rs_league_eval's header lines before the games, and its report path after them.
            if ($leagueLine.StartsWith('protocol:') -or $leagueLine.StartsWith('pairs:') -or
                $leagueLine.StartsWith('workers:') -or $leagueLine.StartsWith('report:')) { Write-Both $leagueLine; continue }
            if ($leagueLine.StartsWith('league evaluation:')) { $inBlock = $true }
            if ($inBlock) { Write-Both $leagueLine }
            if ($leagueLine.StartsWith('best:')) { $inBlock = $false }
            # format_report's gate line: "  NAME: PASS (vs champion CHAMPION)".
            $gateLine = [regex]::Match($leagueLine, '^\s+(\S+): (PASS|FAIL|CYCLE|INCOMPLETE|PRIVILEGED) \(vs champion (\S+)\)\s*$')
            if ($gateLine.Success -and $gateLine.Groups[1].Value -ceq $candName) { $decision = $gateLine.Groups[2].Value }
        }
    }
    $rule = "The gate: a Wilson 95% lower bound above 52% over 1,000 games against the champion (at least 551 points; a win is 1, a draw 0.5), and no more than 2 points below the champion against live_big."
    if ($decision -eq 'PASS') {
        Write-Both ("league verdict: PASS. {0} is the new champion: every later candidate plays it (-Champion {2}). It may replace {1} in the live runner after a section 17 dry run with scripts\vision_play.py --ckpt {2} (runbook section 19, 'Result (2026-10-30)', Decisions). {3}" -f $candName, $championModel.Name, $candModel.Rel, $rule) 'Green'
    } elseif ($decision -eq 'INCOMPLETE') {
        Write-Both ("league verdict: INCOMPLETE: games are missing (see the error lines above). Run it again with -LeagueOnly and the same -Ckpt, -LeagueKeep, -LeagueName, -LeagueLog and -Champion: the league plays only the games its log lacks. {0}" -f $rule) 'Yellow'
    } elseif ($decision) {
        Write-Both ("league verdict: {0}. Keep {1} in the live runner. The beats_champion and anchor lines above say which condition decided it. {2}" -f $decision, $championModel.Name, $rule) 'Yellow'
    } else {
        Write-Both "league verdict: none: the league step gave no gate line for the candidate (see its output above)." 'Red'
        if ($null -eq $leagueCheck -or $leagueCheck.Ok) { $failedSteps.Add('league: no gate decision') }
    }
}

if ($failedSteps.Count -gt 0) {
    Write-Both ("{0} step(s) failed: {1}. Send this log back: {2}" -f $failedSteps.Count, ($failedSteps -join '; '), $log) 'Red'
    $writer.Close()
    exit 1
}
Write-Both ("== done ({0}). Send back the log: {1}" -f (Get-Date -Format 'HH:mm:ss'), $log) 'Green'
$writer.Close()
