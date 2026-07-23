# ============================================================
# WeNet AISHELL Local Pipeline Runner
# Usage: .\run_pipeline.ps1 [-Fast|-Data|-Train|-Full|-Step N]
# ============================================================
param(
    [switch]$Fast,
    [switch]$Data,
    [switch]$Train,
    [switch]$Full,
    [string]$Step = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = if ($env:WENET_ROOT) { $env:WENET_ROOT } else { "D:\wenet" }
$GitBash = "C:\Program Files\Git\bin\bash.exe"

# Colors
function Say-Step  { Write-Host ("`n=== " + $args[0] + " ===") -ForegroundColor Cyan }
function Say-OK    { Write-Host ("Done: " + $args[0]) -ForegroundColor Green }
function Say-Err   { Write-Host ("FAIL: " + $args[0]) -ForegroundColor Red }
function Say-Info  { Write-Host $args[0] -ForegroundColor Yellow }

# Validate
if (-not (Test-Path $GitBash)) {
    Say-Err "Git Bash not found. Run setup_local.ps1 first."
    exit 1
}

# Run bash script through Git Bash
function Invoke-Bash {
    param([string]$ScriptPath, [string]$Description)
    Say-Step $Description
    $bashScript = "cd /$($ProjectRoot.Replace(':','').Replace('\','/')); source ./env_autodl.sh; bash $ScriptPath"
    & $GitBash -c $bashScript
    if ($LASTEXITCODE -ne 0) {
        Say-Err "Script failed: $ScriptPath"
        exit 1
    }
    Say-OK $Description
}

# Determine steps
$steps = @()
if ($Step -ne "") {
    $found = Get-ChildItem "$ProjectRoot\scripts\" -Name "$Step*.sh" | Select-Object -First 1
    if (-not $found) {
        Say-Err "No script matching '$Step'. Available:"
        Get-ChildItem "$ProjectRoot\scripts\" -Name | ForEach-Object { Write-Host "  $_" }
        exit 1
    }
    $steps = @("scripts/$found")
}
elseif ($Data) {
    $steps = @("scripts/00_prepare_autodl.sh", "scripts/01_fetch_wenet.sh", "scripts/02_prepare_aishell.sh")
}
elseif ($Train) {
    $steps = @("scripts/03_train_course_fast.sh")
}
elseif ($Full) {
    $steps = @(
        "scripts/00_prepare_autodl.sh", "scripts/01_fetch_wenet.sh", "scripts/02_prepare_aishell.sh",
        "scripts/03_train_course_fast.sh", "scripts/04_decode_eval.sh", "scripts/05_export_model.sh",
        "scripts/06_package_runtime_model.sh"
    )
}
else {
    # Default: Fast mode
    $steps = @(
        "scripts/00_prepare_autodl.sh", "scripts/01_fetch_wenet.sh", "scripts/02_prepare_aishell.sh",
        "scripts/03_train_course_fast.sh", "scripts/04_decode_eval.sh", "scripts/05_export_model.sh"
    )
}

# Display plan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  WeNet AISHELL Pipeline" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Say-Info "Mode: $(if ($Fast) { 'Fast Verify' } elseif ($Data) { 'Data Only' } elseif ($Train) { 'Train Only' } elseif ($Full) { 'Full Pipeline' } elseif ($Step) { "Single Step: $Step" } else { 'Fast (default)' })"
Write-Host "Steps: $($steps.Count)" -ForegroundColor White
foreach ($s in $steps) { Write-Host "  - $s" -ForegroundColor Gray }
Write-Host ""

# Execute
$startTime = Get-Date
for ($i = 0; $i -lt $steps.Count; $i++) {
    $step = $steps[$i]
    $stepNum = $i + 1
    Invoke-Bash -ScriptPath $step -Description "Step $stepNum/$($steps.Count): $step"
}

$elapsed = (Get-Date) - $startTime
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Pipeline Complete! Time: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results:" -ForegroundColor White
Write-Host "  CER Report: $ProjectRoot\results_summary.md" -ForegroundColor Yellow
Write-Host "  Checkpoints: $ProjectRoot\wenet\examples\aishell\s0\exp\" -ForegroundColor Yellow
Write-Host "  Runtime Models: $ProjectRoot\wenet_runtime_models\" -ForegroundColor Yellow
