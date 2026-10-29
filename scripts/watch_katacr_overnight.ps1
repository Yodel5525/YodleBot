$ErrorActionPreference = "Continue"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$pythonExe = Join-Path $projectRoot ".venvs\buildabot\Scripts\python.exe"
$runner = Join-Path $projectRoot "scripts\mumu_katacr.py"
# adb and serial: HASTYCR_ADB / HASTYCR_SERIAL, machine.json, defaults (src\hastycr\machine.py).
$machinePy = Join-Path $projectRoot "src\hastycr\machine.py"
$adbExe = (& $pythonExe -I $machinePy --get adb | Select-Object -Last 1)
$serial = (& $pythonExe -I $machinePy --get serial | Select-Object -Last 1)
$checkpoint = Join-Path $projectRoot "vendor\KataCR\logs\Policy\StARformer_3L_v0.8_golem_ai_cnn_blocks__nbc128__ep30__step50__0__20240512_181646\ckpt"
$logDir = Join-Path $projectRoot "tmp\live"
$runLog = Join-Path $logDir "katacr_overnight.log"
$supervisorLog = Join-Path $logDir "katacr_supervisor.log"
$deadline = (Get-Date).AddHours(8)

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $projectRoot

while ((Get-Date) -lt $deadline) {
    $remainingHours = ($deadline - (Get-Date)).TotalHours
    Add-Content -Path $supervisorLog -Value "$(Get-Date -Format s) START remaining_hours=$([math]::Round($remainingHours, 3))"
    & $pythonExe $runner `
        --adb $adbExe `
        --serial $serial `
        --checkpoint $checkpoint `
        --epoch 3 `
        --hours $remainingHours `
        --log $runLog
    $exitCode = $LASTEXITCODE
    Add-Content -Path $supervisorLog -Value "$(Get-Date -Format s) EXIT code=$exitCode"
    if ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
    }
}

Add-Content -Path $supervisorLog -Value "$(Get-Date -Format s) COMPLETE"
