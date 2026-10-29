<#
.SYNOPSIS
    Create or update the Python 3.12 venvs this repo uses on Windows.

.DESCRIPTION
    .venvs\royalegym   Engine venv: CPU torch and royalesim (requirements\engine.txt).
                       Used for building rows, evaluation and RL rollout workers.
    .venvs\buildabot   GPU venv: CUDA torch and torchvision, royalesim and the vision
                       packages (requirements\gpu.txt). Used for training, YOLO and live play.
    .venvs\rs022       Optional (-Reference): the original author's royalesim 0.1.22, used
                       only to reproduce his re-drive numbers (requirements\rs022-reference.txt).

    For each venv the script:
      1. creates it with Python 3.12 if it is missing (an existing venv is reused);
      2. upgrades pip;
      3. installs torch (and torchvision) FIRST, from PyTorch's own index: -CpuIndexUrl
         for royalegym, -GpuIndexUrl for buildabot. PyPI's Windows torch is CPU-only.
         The versions come from the requirements file, which holds every pin;
      4. installs the requirements file from PyPI;
      5. checks torch again, and force-reinstalls the right build if a package replaced it;
      6. installs this repo into the venv (pip install --no-deps -e .);
      7. writes reports\freeze-<venv>.txt (pip freeze --all);
      8. verifies torch with a python one-liner (in buildabot, CUDA must work);
      9. runs scripts\selfcheck.py --offline for that venv.
    Running it again is safe. Anything already right is kept, so a second run only
    re-checks.

    Nothing here touches the emulator or the game.

.PARAMETER Only
    all (default), royalegym or buildabot.

.PARAMETER GpuIndexUrl
    PyTorch index for the CUDA build of torch and torchvision. The default is cu130: it has
    torch 2.14.1 for Windows and is verified on this PC. The cu128 index stops at 2.11.0.
    If you change it, the index must offer the torch / torchvision versions pinned in
    requirements\gpu.txt (open <index>/torch/ in a browser to see the list).

.PARAMETER CpuIndexUrl
    PyTorch index for the CPU build used in royalegym.

.PARAMETER Python
    A Python 3.12 python.exe to build the venvs from. The default is the py launcher (py -3.12).

.PARAMETER Recreate
    Delete the selected venvs first and build them from scratch.

.PARAMETER Reference
    Also create .venvs\rs022 from requirements\rs022-reference.txt.

.PARAMETER SkipSelfcheck
    Do not run scripts\selfcheck.py --offline at the end.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Only buildabot -GpuIndexUrl https://download.pytorch.org/whl/cu130

    Only the GPU venv, with an explicit CUDA index.
#>
[CmdletBinding()]
param(
    [ValidateSet('all', 'royalegym', 'buildabot')]
    [string]$Only = 'all',
    [string]$GpuIndexUrl = 'https://download.pytorch.org/whl/cu130',
    [string]$CpuIndexUrl = 'https://download.pytorch.org/whl/cpu',
    [string]$Python = '',
    [switch]$Recreate,
    [switch]$Reference,
    [switch]$SkipSelfcheck
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$RequirementsDir = Join-Path $Root 'requirements'
$ReportsDir = Join-Path $Root 'reports'
$VenvRoot = Join-Path $Root '.venvs'
$script:BasePython = ''
$script:Failed = @()

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

function Write-Step([string]$Text) {
    Write-Host ''
    Write-Host ('== ' + $Text) -ForegroundColor Cyan
}

function Write-Note([string]$Text) {
    Write-Host ('   ' + $Text)
}

function Invoke-Native {
    # Run a program and throw a clear error when it exits non-zero. Call it as a
    # statement: its output is the program's output, shown on the console.
    param([string]$Exe, [string[]]$Arguments, [string]$What)
    # The exit code decides. With 'Stop', Windows PowerShell 5.1 would abort on the first
    # stderr line (a pip warning) whenever this script's output is redirected.
    $ErrorActionPreference = 'Continue'
    Write-Host ('   > ' + $What) -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw ('{0} failed (exit code {1}). The command was: "{2}" {3}' -f $What, $LASTEXITCODE, $Exe, ($Arguments -join ' '))
    }
}

function Get-Pin([string]$File, [string]$Package) {
    # The version on a 'package==version' line of a requirements file.
    $pattern = '^\s*' + [regex]::Escape($Package) + '\s*==\s*([^\s#;]+)'
    foreach ($line in [System.IO.File]::ReadAllLines($File)) {
        $match = [regex]::Match($line, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
        if ($match.Success) { return $match.Groups[1].Value }
    }
    throw ('{0} has no "{1}==<version>" line.' -f $File, $Package)
}

function Get-PythonVersion([string]$Py) {
    # 'major.minor' of an interpreter, or 'broken' when it does not run.
    $ErrorActionPreference = 'Continue'
    $out = @(& $Py -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null)
    if ($LASTEXITCODE -ne 0 -or $out.Count -lt 1) { return 'broken' }
    return ([string]$out[0]).Trim()
}

function Get-DistVersion([string]$Py, [string]$Name) {
    # Installed version of a distribution, or '' when it is not installed.
    $ErrorActionPreference = 'Continue'
    $out = @(& $Py -c "from importlib.metadata import version; print(version('$Name'))" 2>$null)
    if ($LASTEXITCODE -ne 0 -or $out.Count -lt 1) { return '' }
    return ([string]$out[0]).Trim()
}

function Get-TorchState([string]$Py) {
    # torch's version and CUDA build ('none' for CPU) plus torchvision's version, or $null
    # when torch does not import.
    $ErrorActionPreference = 'Continue'
    $out = @(& $Py -c "import torch; print(torch.__version__); print(torch.version.cuda or 'none')" 2>$null)
    if ($LASTEXITCODE -ne 0 -or $out.Count -lt 2) { return $null }
    return [pscustomobject]@{
        Version = ([string]$out[0]).Trim()
        Cuda    = ([string]$out[1]).Trim()
        Vision  = (Get-DistVersion $Py 'torchvision')
    }
}

function Format-TorchState($State) {
    if ($null -eq $State) { return 'no working torch' }
    $text = 'torch {0} (CUDA {1})' -f $State.Version, $State.Cuda
    if ($State.Vision) { $text += (', torchvision ' + $State.Vision) }
    return $text
}

function Test-TorchOk($State, [string]$TorchPin, [string]$VisionPin, [bool]$WantCuda) {
    if ($null -eq $State) { return $false }
    if ($State.Version.Split('+')[0] -ne $TorchPin) { return $false }
    if ($WantCuda) {
        if ($State.Cuda -eq 'none') { return $false }
        if ($VisionPin) {
            if ($State.Vision.Split('+')[0] -ne $VisionPin) { return $false }
            if ($State.Vision -notmatch '\+cu') { return $false }
        }
    } elseif ($State.Cuda -ne 'none') {
        return $false
    }
    return $true
}

function Install-Torch([string]$Py, [string[]]$Specs, [string]$IndexUrl, [bool]$Force) {
    $pipArgs = @('-m', 'pip', 'install', '--prefer-binary')
    if ($Force) { $pipArgs += @('--force-reinstall', '--no-deps') }
    $pipArgs += $Specs
    $pipArgs += @('--index-url', $IndexUrl)
    $how = 'pip install'
    if ($Force) { $how = 'pip install --force-reinstall --no-deps' }
    Invoke-Native $Py $pipArgs ('{0} {1} --index-url {2}' -f $how, ($Specs -join ' '), $IndexUrl)
}

function Repair-Torch {
    # Make sure the venv has torch at the pin, as the CUDA or the CPU build (and, for
    # CUDA, a CUDA torchvision at its pin). Installs only when something is wrong.
    param([string]$Py, [string]$TorchPin, [string]$VisionPin, [string]$IndexUrl, [bool]$WantCuda, [string]$When)
    $state = Get-TorchState $Py
    if (Test-TorchOk $state $TorchPin $VisionPin $WantCuda) {
        Write-Note ('{0}: {1} is right; keeping it' -f $When, (Format-TorchState $state))
        return
    }
    $specs = @('torch==' + $TorchPin)
    if ($VisionPin) { $specs += ('torchvision==' + $VisionPin) }
    Write-Note ('{0}: found {1}; installing {2} from {3}' -f $When, (Format-TorchState $state), ($specs -join ' '), $IndexUrl)
    if ($WantCuda) { Write-Note 'the CUDA torch download is about 2.5 GB; this can take a while' }
    Install-Torch $Py $specs $IndexUrl $false
    $state = Get-TorchState $Py
    if (-not (Test-TorchOk $state $TorchPin $VisionPin $WantCuda)) {
        Write-Note ('still {0}; forcing the build from {1}' -f (Format-TorchState $state), $IndexUrl)
        Install-Torch $Py $specs $IndexUrl $true
        $state = Get-TorchState $Py
    }
    if (-not (Test-TorchOk $state $TorchPin $VisionPin $WantCuda)) {
        $want = 'the CPU build'
        if ($WantCuda) { $want = 'a CUDA build' }
        throw ('torch is still not {0} of {1} after installing from {2}: found {3}. Check that the index has torch {1} for Python 3.12 on Windows (open {2}/torch/ in a browser), or pass another index with -GpuIndexUrl / -CpuIndexUrl.' -f $want, $TorchPin, $IndexUrl, (Format-TorchState $state))
    }
}

function Resolve-BasePython {
    # The full path of a Python 3.12 to build venvs from.
    if ($Python) {
        if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
            throw ('-Python "{0}" does not exist.' -f $Python)
        }
        $exe = (Resolve-Path -LiteralPath $Python).Path
        $pre = @()
    } else {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if (-not $launcher) {
            throw 'The Python launcher "py" is not on PATH. Install Python 3.12 (winget install -e --id Python.Python.3.12), open a NEW terminal and run this again, or pass -Python C:\path\to\python.exe (a 3.12).'
        }
        $exe = $launcher.Source
        $pre = @('-3.12')
    }
    $ErrorActionPreference = 'Continue'
    $out = @(& $exe @pre -c "import sys; print(sys.executable); print('%d.%d' % sys.version_info[:2])")
    if ($LASTEXITCODE -ne 0 -or $out.Count -lt 2) {
        throw ('Could not run Python 3.12 with "{0}" {1}. "py -0p" lists the installed versions; install 3.12 with: winget install -e --id Python.Python.3.12' -f $exe, ($pre -join ' '))
    }
    if (([string]$out[1]).Trim() -ne '3.12') {
        throw ('{0} is Python {1}; this repo needs 3.12 (pyproject.toml).' -f $out[0], $out[1])
    }
    return ([string]$out[0]).Trim()
}

function Test-DiskSpace {
    try {
        $drive = (Get-Item -LiteralPath $Root).PSDrive
        $freeGb = [math]::Round($drive.Free / 1GB, 1)
        Write-Note ('{0} GB free on {1}:' -f $freeGb, $drive.Name)
        if ($freeGb -lt 20) {
            Write-Warning ('Only {0} GB free on {1}:. The two venvs take about 8 GB, and the data needs much more (docs\SPARRING_RUNBOOK.md).' -f $freeGb, $drive.Name)
        }
    } catch {
        Write-Note 'could not read the free disk space'
    }
}

function Write-Freeze([string]$Py, [string]$Name) {
    $ErrorActionPreference = 'Continue'
    $lines = @(& $Py -m pip freeze --all)
    if ($LASTEXITCODE -ne 0) { throw ('pip freeze failed for .venvs\' + $Name) }
    New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
    $path = Join-Path $ReportsDir ('freeze-' + $Name + '.txt')
    $stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm')
    $header = @(
        ('# pip freeze --all of .venvs\{0}, written by scripts\setup_windows.ps1 on {1}' -f $Name, $stamp),
        ('# python {0}' -f (Get-PythonVersion $Py))
    )
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($path, [string[]]($header + $lines), $utf8)
    Write-Note ('wrote ' + $path)
}

function Test-Venv([string]$Py, [string]$Kind, [string]$RequirementsFile) {
    # The python one-liner check. Throws with advice when it fails.
    if ($Kind -eq 'train') {
        $code = "import torch, torchvision, royalesim; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', ok); print('device', torch.cuda.get_device_name(0) if ok else '-', 'bf16', ok and torch.cuda.is_bf16_supported()); print('torchvision', torchvision.__version__, 'royalesim', royalesim.__version__); raise SystemExit(0 if ok and torch.cuda.is_bf16_supported() else 3)"
        $advice = 'CUDA torch is installed but CUDA (or bf16) does not work. Update the NVIDIA driver: nvidia-smi must list the GPU with a "CUDA Version" at least as new as the torch build above (13.0 for cu130). Or re-run with -GpuIndexUrl for a CUDA version your driver supports, with matching pins in requirements\gpu.txt.'
    } elseif ($Kind -eq 'engine') {
        $code = "import torch, royalesim; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('royalesim', royalesim.__version__); raise SystemExit(0 if torch.version.cuda is None else 3)"
        $advice = 'The engine venv must have the CPU build of torch. Re-run with -Only royalegym -Recreate.'
    } else {
        $pin = Get-Pin (Join-Path $RequirementsDir $RequirementsFile) 'royalesim'
        $code = "import royalesim; print('royalesim', royalesim.__version__); raise SystemExit(0 if royalesim.__version__ == '$pin' else 3)"
        $advice = ('royalesim is not {0} in this venv. Re-run with -Reference -Recreate.' -f $pin)
    }
    $ErrorActionPreference = 'Continue'
    Write-Host '   > verify' -ForegroundColor DarkGray
    & $Py -c $code
    if ($LASTEXITCODE -ne 0) { throw ('verification failed (exit code {0}). {1}' -f $LASTEXITCODE, $advice) }
}

function Install-Venv {
    param([string]$Name, [string]$RequirementsFile, [string]$Kind, [string]$IndexUrl)
    # Kind: engine (CPU torch), train (CUDA torch) or reference (no torch, no repo install).
    Write-Step ('.venvs\{0}  (requirements\{1})' -f $Name, $RequirementsFile)
    $dir = Join-Path $VenvRoot $Name
    $py = Join-Path $dir 'Scripts\python.exe'
    $req = Join-Path $RequirementsDir $RequirementsFile
    if (-not (Test-Path -LiteralPath $req -PathType Leaf)) { throw ($req + ' is missing.') }

    if ($Recreate -and (Test-Path -LiteralPath $dir)) {
        Write-Note ('-Recreate: deleting ' + $dir)
        Remove-Item -LiteralPath $dir -Recurse -Force
    }
    if (Test-Path -LiteralPath $py -PathType Leaf) {
        $version = Get-PythonVersion $py
        if ($version -ne '3.12') {
            throw ('{0} exists, but its python is "{1}", not 3.12. Re-run with -Recreate to rebuild it.' -f $dir, $version)
        }
        Write-Note ('reusing ' + $dir)
    } elseif (Test-Path -LiteralPath $dir) {
        throw ('{0} exists but has no Scripts\python.exe (a half-made venv?). Re-run with -Recreate to rebuild it.' -f $dir)
    } else {
        Invoke-Native $script:BasePython @('-m', 'venv', $dir) ('create ' + $dir)
    }

    Invoke-Native $py @('-m', 'pip', 'install', '--upgrade', 'pip') 'upgrade pip'

    $withTorch = ($Kind -eq 'engine') -or ($Kind -eq 'train')
    $wantCuda = ($Kind -eq 'train')
    $torchPin = ''
    $visionPin = ''
    if ($withTorch) {
        $torchPin = Get-Pin $req 'torch'
        if ($wantCuda) { $visionPin = Get-Pin $req 'torchvision' }
        Repair-Torch $py $torchPin $visionPin $IndexUrl $wantCuda 'torch first'
    }

    Invoke-Native $py @('-m', 'pip', 'install', '--prefer-binary', '-r', $req) ('pip install -r requirements\' + $RequirementsFile)

    if ($withTorch) {
        Repair-Torch $py $torchPin $visionPin $IndexUrl $wantCuda ('after requirements\' + $RequirementsFile)
        Invoke-Native $py @('-m', 'pip', 'install', '--no-deps', '-e', $Root) 'pip install --no-deps -e . (this repo)'
    }

    Write-Freeze $py $Name
    Test-Venv $py $Kind $RequirementsFile

    if ($withTorch -and -not $SkipSelfcheck) {
        if (-not (Invoke-Selfcheck $py $Kind)) { $script:Failed += ('.venvs\' + $Name + ' selfcheck') }
    }
}

function Invoke-Selfcheck([string]$Py, [string]$Kind) {
    # scripts\selfcheck.py --offline for one venv; $true when it passes.
    $ErrorActionPreference = 'Continue'
    Write-Host '   > selfcheck --offline' -ForegroundColor DarkGray
    & $Py (Join-Path $Root 'scripts\selfcheck.py') --offline --role $Kind | Out-Host
    return ($LASTEXITCODE -eq 0)
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

$envNames = @('PIP_DISABLE_PIP_VERSION_CHECK', 'PIP_NO_INPUT', 'PYTHONUTF8')
$savedEnv = @{}
foreach ($envName in $envNames) { $savedEnv[$envName] = [Environment]::GetEnvironmentVariable($envName, 'Process') }
$pushed = $false
$exitCode = 0
try {
    if ($env:OS -ne 'Windows_NT') { throw 'setup_windows.ps1 is for Windows.' }
    foreach ($url in @($GpuIndexUrl, $CpuIndexUrl)) {
        if ($url -notmatch '^https?://') { throw ('index URL "{0}" must start with https://' -f $url) }
    }
    Push-Location -LiteralPath $Root
    $pushed = $true
    $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
    $env:PIP_NO_INPUT = '1'
    $env:PYTHONUTF8 = '1'

    Write-Step ('HastyCR setup in ' + $Root)
    Write-Note ('CPU torch index: ' + $CpuIndexUrl)
    Write-Note ('GPU torch index: ' + $GpuIndexUrl)
    Test-DiskSpace
    $script:BasePython = Resolve-BasePython
    Write-Note ('base Python 3.12: ' + $script:BasePython)
    New-Item -ItemType Directory -Force -Path $VenvRoot | Out-Null

    if ($Only -ne 'buildabot') { Install-Venv -Name 'royalegym' -RequirementsFile 'engine.txt' -Kind 'engine' -IndexUrl $CpuIndexUrl }
    if ($Only -ne 'royalegym') { Install-Venv -Name 'buildabot' -RequirementsFile 'gpu.txt' -Kind 'train' -IndexUrl $GpuIndexUrl }
    if ($Reference) { Install-Venv -Name 'rs022' -RequirementsFile 'rs022-reference.txt' -Kind 'reference' -IndexUrl '' }

    Write-Step 'Done'
    if ($script:Failed.Count -gt 0) {
        Write-Host ('   selfcheck failed in: ' + ($script:Failed -join ', ') + '. Read the FAIL lines above.') -ForegroundColor Red
        $exitCode = 1
    } else {
        Write-Note 'Next, with MuMu running: .venvs\buildabot\Scripts\python.exe scripts\selfcheck.py'
        Write-Note 'Then the tests: .venvs\royalegym\Scripts\python.exe -m pytest tests/pipeline -q'
    }
} catch {
    Write-Host ''
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    $exitCode = 1
} finally {
    if ($pushed) { Pop-Location }
    foreach ($envName in $envNames) { [Environment]::SetEnvironmentVariable($envName, $savedEnv[$envName], 'Process') }
}
exit $exitCode
