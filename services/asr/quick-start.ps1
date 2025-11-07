# WhisperX Quick Start Script
# Run this after getting your HuggingFace token

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  WhisperX Migration Quick Start" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
$expectedPath = "services\asr"
$currentPath = (Get-Location).Path

if ($currentPath -notlike "*$expectedPath*") {
    Write-Host "Error: Please run this script from the services/asr directory" -ForegroundColor Red
    Write-Host "Current directory: $currentPath" -ForegroundColor Yellow
    Write-Host "Expected to be in: *\services\asr" -ForegroundColor Yellow
    exit 1
}

# Step 1: Check for .env file
Write-Host "[Step 1/6] Checking configuration..." -ForegroundColor Yellow

if (-not (Test-Path ".env")) {
    Write-Host "  Error: .env file not found!" -ForegroundColor Red
    Write-Host "  Please create .env file from .env.example" -ForegroundColor Yellow
    exit 1
}

# Check if HF_TOKEN is set
$envContent = Get-Content ".env" -Raw
if ($envContent -match "hf_YOUR_ACTUAL_TOKEN_HERE") {
    Write-Host "  Warning: HF_TOKEN not configured!" -ForegroundColor Red
    Write-Host "  Please update .env with your real HuggingFace token" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Get token from: https://huggingface.co/settings/tokens" -ForegroundColor Cyan
    Write-Host "  Accept models:" -ForegroundColor Cyan
    Write-Host "    - https://huggingface.co/pyannote/segmentation-3.0" -ForegroundColor Cyan
    Write-Host "    - https://huggingface.co/pyannote/speaker-diarization-3.1" -ForegroundColor Cyan
    Write-Host ""
    
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

Write-Host "  Configuration OK" -ForegroundColor Green
Write-Host ""

# Step 2: Install WhisperX
Write-Host "[Step 2/6] Installing WhisperX dependencies..." -ForegroundColor Yellow
Write-Host "  This may take 2-5 minutes..." -ForegroundColor Gray

& .\install_whisperx.ps1

if ($LASTEXITCODE -ne 0) {
    Write-Host "  Error: Installation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "  Installation complete" -ForegroundColor Green
Write-Host ""

# Step 3: Backup original app.py
Write-Host "[Step 3/6] Backing up original service..." -ForegroundColor Yellow

if (Test-Path "app.py") {
    $backupName = "app_vanilla_whisper_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').py"
    Copy-Item "app.py" $backupName
    Write-Host "  Backup saved: $backupName" -ForegroundColor Green
} else {
    Write-Host "  No existing app.py found (this is okay for new installations)" -ForegroundColor Gray
}

Write-Host ""

# Step 4: Deploy WhisperX service
Write-Host "[Step 4/6] Deploying WhisperX service..." -ForegroundColor Yellow

if (-not (Test-Path "app_whisperx.py")) {
    Write-Host "  Error: app_whisperx.py not found!" -ForegroundColor Red
    exit 1
}

Copy-Item "app_whisperx.py" "app.py" -Force
Write-Host "  WhisperX service deployed" -ForegroundColor Green
Write-Host ""

# Step 5: Check Python packages
Write-Host "[Step 5/6] Verifying installation..." -ForegroundColor Yellow

$packages = @("whisperx", "transformers", "pyannote.audio", "torch")
$missing = @()

foreach ($package in $packages) {
    $installed = pip list 2>$null | Select-String -Pattern "^$package\s"
    if (-not $installed) {
        $missing += $package
    }
}

if ($missing.Count -gt 0) {
    Write-Host "  Warning: Missing packages: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "  Run: .\install_whisperx.ps1" -ForegroundColor Yellow
} else {
    Write-Host "  All packages installed" -ForegroundColor Green
}

Write-Host ""

# Step 6: Summary
Write-Host "[Step 6/6] Migration Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Next Steps" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Start the service:" -ForegroundColor White
Write-Host "   python app.py" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Test with test audio:" -ForegroundColor White
Write-Host "   cd ..\..\" -ForegroundColor Gray
Write-Host "   python test_asr.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Check for improvements:" -ForegroundColor White
Write-Host "   - No hallucinations (البروستاتا, الحمل)" -ForegroundColor Gray
Write-Host "   - No repetitions" -ForegroundColor Gray
Write-Host "   - Speaker labels (SPEAKER_00, SPEAKER_01)" -ForegroundColor Gray
Write-Host "   - 4-70x faster processing" -ForegroundColor Gray
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Documentation" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Full Guide: WHISPERX_MIGRATION.md" -ForegroundColor Gray
Write-Host "  Summary:    MIGRATION_SUMMARY.md" -ForegroundColor Gray
Write-Host "  Rollback:   Copy-Item app_vanilla_whisper_backup*.py app.py" -ForegroundColor Gray
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Optional: Start service
$startNow = Read-Host "Start ASR service now? (y/n)"
if ($startNow -eq "y") {
    Write-Host ""
    Write-Host "Starting ASR service..." -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
    Write-Host ""
    python app.py
}
