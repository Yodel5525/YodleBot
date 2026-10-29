$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venvs\buildabot\Scripts\python.exe"
# adb and serial: HASTYCR_ADB / HASTYCR_SERIAL, machine.json, defaults (src\hastycr\machine.py).
$machinePy = Join-Path $projectRoot "src\hastycr\machine.py"
$adb = (& $python -I $machinePy --get adb | Select-Object -Last 1)
$serial = (& $python -I $machinePy --get serial | Select-Object -Last 1)
$frame = Join-Path $projectRoot "tmp\live\clash-royale.png"

& $python (Join-Path $PSScriptRoot "live_probe.py") `
    --adb $adb `
    --serial $serial `
    --save-frame $frame

exit $LASTEXITCODE
