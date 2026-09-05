<#
.SYNOPSIS
  Reproducible RetinaPrep environment setup on Windows (PowerShell 5.1+).

.DESCRIPTION
  This is documentation-as-script: every step below is what was actually run
  to build the project's dev environment. It creates a self-contained .venv,
  installs everything in requirements.txt EXCEPT torch/torchvision, and
  editable-installs the retinaprep package itself. That base environment is
  enough to run `ingest`, `split`, and the full pytest suite -- torch is only
  load-bearing for `train` and the embedding path in `dedupe`, and both
  import it lazily so its absence is not an error (see scripts/doctor.py).

  The CUDA-enabled torch/torchvision install is a separate, opt-in step
  (-IncludeCudaTorch) because it is a 2-3 GB download and you may prefer to
  run it on its own, on a better connection, or skip it entirely on a
  CPU-only box.

.PARAMETER IncludeCudaTorch
  Also run the CUDA torch/torchvision install (Step 3). Off by default.

.PARAMETER TorchCudaTag
  PyTorch's CUDA wheel tag, e.g. "cu130". Verified against
  https://download.pytorch.org/whl/<tag>/torch/ on 2026-09-05 as the tag
  carrying torch 2.14.0 for cp313-win_amd64. PyTorch's supported tags change
  over time -- if this 404s, check that page (or
  https://pytorch.org/get-started/locally/) for the current tag and pass it
  here, e.g. -TorchCudaTag cu126.

.PARAMETER TorchVersion / -TorchvisionVersion
  Must be a matched pair published under the same CUDA tag. Defaults are the
  versions verified alongside the tag above.

.EXAMPLE
  .\scripts\setup_env.ps1
  Base environment only: venv + requirements (no torch) + editable install.

.EXAMPLE
  .\scripts\setup_env.ps1 -IncludeCudaTorch
  Base environment, then the CUDA torch/torchvision install on top.

.NOTES
  Anaconda users: run `conda deactivate` (possibly more than once, if envs
  are stacked) before running this script. An active conda environment can
  shadow the .venv interpreter on PATH for anything that relies on bare
  `python`/`pip` rather than the explicit .venv path this script itself uses.
#>
param(
    [switch]$IncludeCudaTorch,
    [string]$TorchCudaTag = "cu130",
    [string]$TorchVersion = "2.14.0",
    [string]$TorchvisionVersion = "0.29.0"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($env:CONDA_PREFIX) {
    Write-Warning "A conda environment is active ($env:CONDA_PREFIX). Run 'conda deactivate' first -- this script always invokes .venv by explicit path so it will still work, but leaving conda active invites accidentally installing into it later out of habit."
}

# --- Step 1: create .venv -----------------------------------------------
# Prefer a standalone python.org interpreter via the `py` launcher; fall
# back to whatever `python` resolves to. Either way, `python -m venv`
# produces a fully self-contained environment with its own site-packages --
# nothing installs into the base interpreter's site-packages from here on,
# even if that base interpreter happens to be Anaconda's.
Write-Output "=== Step 1: create .venv ==="

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    $baseExe = "py"
    $baseArgs = @("-3")
} else {
    $baseCmd = Get-Command python -ErrorAction Stop
    $baseExe = $baseCmd.Source
    $baseArgs = @()
    Write-Warning "No 'py' launcher found; using '$baseExe' as the venv base interpreter."
}

if (Test-Path ".venv") {
    Write-Output ".venv already exists, skipping creation."
} else {
    & $baseExe @baseArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit $LASTEXITCODE)" }
}

$venvPy = Join-Path $repoRoot ".venv\Scripts\python.exe"
& $venvPy --version

# --- Step 2: requirements (torch/torchvision excluded) + editable install ---
Write-Output ""
Write-Output "=== Step 2: requirements.txt (excluding torch/torchvision) ==="

& $venvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip self-upgrade failed (exit $LASTEXITCODE)" }

$filtered = Get-Content (Join-Path $repoRoot "requirements.txt") |
    Where-Object { $_ -notmatch '^\s*(torch|torchvision)\s*[><=]' }
$tmpReq = New-TemporaryFile
try {
    $filtered | Set-Content -Path $tmpReq -Encoding utf8
    & $venvPy -m pip install -r $tmpReq
    if ($LASTEXITCODE -ne 0) { throw "requirements install failed (exit $LASTEXITCODE)" }
} finally {
    Remove-Item $tmpReq -ErrorAction SilentlyContinue
}

Write-Output ""
Write-Output "=== Step 2b: editable install of retinaprep ==="
& $venvPy -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw "editable install failed (exit $LASTEXITCODE)" }

Write-Output ""
Write-Output "Base environment ready. torch/torchvision are NOT installed."
Write-Output "  Verify: .\.venv\Scripts\python.exe -m pytest"
Write-Output "  Report: .\.venv\Scripts\python.exe scripts\doctor.py"

if (-not $IncludeCudaTorch) {
    Write-Output ""
    Write-Output "Skipping the CUDA torch install (pass -IncludeCudaTorch to run it too)."
    return
}

# ============================================================
# Step 3 (separate, opt-in): CUDA-enabled torch + torchvision
# ============================================================
# A 2-3 GB download. $TorchCudaTag/$TorchVersion/$TorchvisionVersion above
# were verified as a matched, currently-published trio -- re-check them at
# https://download.pytorch.org/whl/<tag>/torch/ if this has gone stale.
#
# This machine's GPU is a GTX 1050 (Pascal, compute capability 6.1 /
# "sm_61"). PyTorch's default wheels have dropped older compute
# capabilities before; a wheel installing cleanly proves nothing about
# whether it actually shipped sm_61 kernels. That is exactly what
# scripts/doctor.py checks after this step -- always run it once this
# finishes, and do not assume success just because pip exits 0.
Write-Output ""
Write-Output "=== Step 3: CUDA torch/torchvision ($TorchCudaTag) ==="

$indexUrl = "https://download.pytorch.org/whl/$TorchCudaTag"
& $venvPy -m pip install "torch==$TorchVersion" "torchvision==$TorchvisionVersion" --index-url $indexUrl

if ($LASTEXITCODE -ne 0) {
    Write-Warning "pip install from $indexUrl failed or timed out (common for a 2-3GB wheel on a slow connection)."
    Write-Output "Falling back to a resumable direct download via curl.exe -L -C - ..."

    # Direct wheel URLs, verified against download.pytorch.org's index pages.
    # If TorchVersion/TorchvisionVersion/TorchCudaTag are overridden, these
    # are best-effort -- check the index page for the real filename if a
    # 404 comes back.
    $pyTagOutput = & $venvPy -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
    $pyTag = "$pyTagOutput-$pyTagOutput-win_amd64"

    $torchFile = "torch-$TorchVersion+$TorchCudaTag-$pyTag.whl"
    $visionFile = "torchvision-$TorchvisionVersion+$TorchCudaTag-$pyTag.whl"
    # download-r2.pytorch.org is the actual CDN host the whl/<tag>/ index
    # pages link to; '+' is percent-encoded as %2B in the real URL.
    $torchUrl = "https://download-r2.pytorch.org/whl/$TorchCudaTag/$($torchFile -replace '\+', '%2B')"
    $visionUrl = "https://download-r2.pytorch.org/whl/$TorchCudaTag/$($visionFile -replace '\+', '%2B')"

    $downloadDir = Join-Path $env:TEMP "retinaprep_torch_wheels"
    New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
    $torchDest = Join-Path $downloadDir $torchFile
    $visionDest = Join-Path $downloadDir $visionFile

    Write-Output "Downloading torch wheel (re-run this command as-is if it drops -- '-C -' resumes):"
    Write-Output "  curl.exe -L -C - -o `"$torchDest`" `"$torchUrl`""
    curl.exe -L -C - -o "$torchDest" "$torchUrl"
    if ($LASTEXITCODE -ne 0) { throw "torch wheel download failed (exit $LASTEXITCODE) -- re-run the curl.exe line above to resume" }

    Write-Output "Downloading torchvision wheel (re-run this command as-is if it drops -- '-C -' resumes):"
    Write-Output "  curl.exe -L -C - -o `"$visionDest`" `"$visionUrl`""
    curl.exe -L -C - -o "$visionDest" "$visionUrl"
    if ($LASTEXITCODE -ne 0) { throw "torchvision wheel download failed (exit $LASTEXITCODE) -- re-run the curl.exe line above to resume" }

    & $venvPy -m pip install "$torchDest" "$visionDest"
    if ($LASTEXITCODE -ne 0) { throw "installing downloaded wheels failed (exit $LASTEXITCODE)" }
}

Write-Output ""
Write-Output "torch/torchvision installed. Now run: .\.venv\Scripts\python.exe scripts\doctor.py"
Write-Output "Check its sm_61 line specifically -- a clean install is not proof the GPU kernels are actually there."
