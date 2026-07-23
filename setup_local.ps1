# ============================================================
# WeNet AISHELL - Local Windows Setup Script
# Usage: powershell -ExecutionPolicy Bypass -File .\setup_local.ps1
# ============================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = if ($env:WENET_ROOT) { $env:WENET_ROOT } else { $PSScriptRoot }
Set-Location $ProjectRoot

function Say-Step  { param($msg) Write-Host ("=== " + $msg + " ===") -ForegroundColor Cyan }
function Say-OK    { param($msg) Write-Host ("[OK] " + $msg) -ForegroundColor Green }
function Say-Skip  { param($msg) Write-Host ("[SKIP] " + $msg) -ForegroundColor Gray }
function Say-Err   { param($msg) Write-Host ("[FAIL] " + $msg) -ForegroundColor Red }
function Say-Info  { param($msg) Write-Host $msg -ForegroundColor Yellow }

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  WeNet AISHELL - Local Environment Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# ----------------------------------------------------------
# 1. Check Git Bash
# ----------------------------------------------------------
$GitBashCandidates = @(
    "C:\Program Files\Git\bin\bash.exe",
    "C:\Program Files\Git\usr\bin\bash.exe",
    "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe"
)
$GitBash = $null
foreach ($candidate in $GitBashCandidates) {
    if (Test-Path $candidate) { $GitBash = $candidate; break }
}
if (-not $GitBash) {
    Say-Err "Git Bash not found. Tried: $($GitBashCandidates -join ', ')"
    Say-Err "Please install Git for Windows first."
    exit 1
}
Say-OK "Git Bash found at: $GitBash"

# 2. Check Python
$pyCmd = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $pyCmd) {
    Say-Err "Python not found in PATH"
    exit 1
}
Say-OK ("Python " + (python --version 2>&1))

# Check PyTorch (write to temp file to avoid quoting issues)
$torchCheckFile = Join-Path $env:TEMP "wenet_torch_check.py"
@'
import torch
print(f"torch {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
'@ | Set-Content -Path $torchCheckFile -Encoding UTF8
$torchInfo = python $torchCheckFile 2>&1 | Out-String
Remove-Item $torchCheckFile -ErrorAction SilentlyContinue
if ($LASTEXITCODE -ne 0) {
    Say-Err "PyTorch not installed. Please install: pip install torch torchaudio"
    exit 1
}
Say-Info ($torchInfo -join " ")
if ($torchInfo -match "CPU-only" -or $torchInfo -match "CUDA: False") {
    Say-Info "CPU-only mode detected. Training will be slow but functional."
}

# ----------------------------------------------------------
# 3. Extract AISHELL data
# ----------------------------------------------------------
Say-Step "Step 1/3: Extracting AISHELL-1 Data"
$DataTarget = Join-Path $ProjectRoot "datasets\aishell"

if ((Test-Path "$DataTarget\data_aishell\wav") -or (Test-Path "$DataTarget\data_aishell")) {
    Say-Skip "AISHELL data already exists: $DataTarget"
} else {
    $extractPy = @"
import tarfile, os, sys
src = sys.argv[1]
dst = sys.argv[2]
os.makedirs(dst, exist_ok=True)
print(f'Extracting {src} ...')
with tarfile.open(src, 'r:gz') as tf:
    tf.extractall(dst, filter='data')
print(f'Done: {len(tf.getnames())} files')
"@

    $DataTarball = Join-Path $ProjectRoot "data\data_aishell.tgz"
    if (Test-Path $DataTarball) {
        python -c $extractPy "$DataTarball" "$DataTarget"
        Say-OK "data_aishell.tgz extracted"
    } else {
        Say-Err "Data tarball not found: $DataTarball"
        exit 1
    }

    $ResourceTarball = Join-Path $ProjectRoot "data\resource_aishell.tgz"
    if (Test-Path $ResourceTarball) {
        python -c $extractPy "$ResourceTarball" "$DataTarget"
        Say-OK "resource_aishell.tgz extracted"
    } else {
        Say-Err "Resource tarball not found: $ResourceTarball"
        exit 1
    }
}

# ----------------------------------------------------------
# 4. Clone WeNet
# ----------------------------------------------------------
Say-Step "Step 2/3: Clone WeNet Repository"
$WenetDir = Join-Path $ProjectRoot "wenet"

if (Test-Path "$WenetDir\.git") {
    Say-Skip "WeNet already exists: $WenetDir"
    Set-Location $WenetDir
    git pull origin main 2>&1 | Out-Null
    Set-Location $ProjectRoot
} else {
    Say-Info "Cloning WeNet (depth=1)..."
    git clone --depth 1 https://github.com/wenet-e2e/wenet.git $WenetDir 2>&1
    Say-OK "WeNet cloned"
}

# ----------------------------------------------------------
# 5. Install WeNet dependencies
# ----------------------------------------------------------
Say-Step "Step 3/3: Install WeNet Dependencies"
Set-Location $WenetDir

# Install base packages
python -m pip install -U pip setuptools wheel -q 2>&1 | Out-Null

# Install WeNet requirements (skip torch/torchaudio/deepspeed)
$reqFile = Join-Path $WenetDir "requirements.txt"
if (Test-Path $reqFile) {
    $reqNoGpu = Join-Path $env:TEMP "wenet_req_nogpu.txt"
    $reqFilterScript = @"
lines = open(r'$reqFile', encoding='utf-8').readlines()
skip_prefixes = ('torch', 'torchaudio', 'deepspeed')
filtered = [l for l in lines if l.strip() and not l.strip().startswith(skip_prefixes)]
with open(r'$reqNoGpu', 'w', encoding='utf-8') as f:
    f.writelines(filtered)
print(f'Package count: {len(filtered)}')
"@
    python -c $reqFilterScript
    Say-Info "Installing WeNet dependencies..."
    python -m pip install -r $reqNoGpu -i https://pypi.tuna.tsinghua.edu.cn/simple -q 2>&1 | Out-Null
}

# Install wenet package (no-deps to avoid overriding torch)
Say-Info "Installing wenet..."
python -m pip install -e . --no-deps -q 2>&1 | Out-Null
Say-OK "WeNet Python package installed"

Set-Location $ProjectRoot

# ----------------------------------------------------------
# Verify
# ----------------------------------------------------------
Say-Step "Verification"
$verifyScript = "import wenet; print('wenet version:', wenet.__version__ if hasattr(wenet, '__version__') else 'ok')"
$verifyResult = python -c $verifyScript 2>&1
if ($LASTEXITCODE -eq 0) {
    Say-OK "WeNet import successful"
} else {
    Say-Err "WeNet import failed: $verifyResult"
    Say-Info "This might be ok if you use WeNet via its run.sh scripts"
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Say-Info "Next steps:"
Write-Host "  1. Test environment:  powershell .\run_pipeline.ps1 -Step 00" -ForegroundColor White
Write-Host "  2. Quick pipeline:   .\run_pipeline.ps1" -ForegroundColor White
Write-Host ""
Write-Host ("Data location:  " + $DataTarget) -ForegroundColor White
Write-Host ("WeNet location: " + $WenetDir) -ForegroundColor White
Write-Host ("Project root:   " + $ProjectRoot) -ForegroundColor White
