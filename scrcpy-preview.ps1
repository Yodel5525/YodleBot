# Read-only native-resolution preview of the verified Clash Royale instance.
param()
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venvs\buildabot\Scripts\python.exe'
$scrcpy = Join-Path $root 'tools\scrcpy\scrcpy-win64-v4.1\scrcpy.exe'
if (-not (Test-Path -LiteralPath $scrcpy)) {
    throw 'scrcpy v4.1 is missing from tools/scrcpy.'
}
& $python (Join-Path $root 'scripts\emulator.py')
if ($LASTEXITCODE -ne 0) { throw 'Clash Royale device verification failed.' }
# adb and serial as every script sees them: HASTYCR_ADB / HASTYCR_SERIAL, machine.json, defaults.
$machinePy = Join-Path $root 'src\hastycr\machine.py'
$adb = (& $python -I $machinePy --get adb | Select-Object -Last 1)
$serial = (& $python -I $machinePy --get serial | Select-Object -Last 1)
$previousAdb = $env:ADB
try {
    $env:ADB = $adb
    & $scrcpy "--serial=$serial" '--no-control' '--no-audio' `
        '--no-downsize-on-error' '--video-codec=h264' '--video-bit-rate=24M' `
        '--print-fps' '--window-title=Clash Royale - scrcpy native' '--window-height=900'
}
finally { $env:ADB = $previousAdb }
