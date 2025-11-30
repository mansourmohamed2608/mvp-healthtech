# Project Cleanup Script
# This script removes unnecessary documentation, test files, and temporary files
# Run with: .\CLEANUP_PROJECT.ps1

Write-Host "🧹 Starting project cleanup..." -ForegroundColor Cyan

# Create a backup folder first (just in case)
$backupFolder = "backup_before_cleanup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Write-Host "📦 Creating backup folder: $backupFolder" -ForegroundColor Yellow
New-Item -ItemType Directory -Path $backupFolder -Force | Out-Null

# Files to KEEP (important documentation)
$keepFiles = @(
    "README.md",
    "QUICKSTART.md",
    "USER_GUIDE.md",
    "TROUBLESHOOTING.md",
    "LICENSE"
)

# ============================================
# 1. MARKDOWN FILES (Documentation Clutter)
# ============================================
Write-Host "`n📄 Cleaning up markdown documentation..." -ForegroundColor Green

$mdFilesToDelete = @(
    "AHD_UPLOAD_GUIDE.md",
    "API_INTEGRATION_FIXES.md",
    "ASR_LORA_INTEGRATION_COMPLETE.md",
    "CHECKLIST.md",
    "CONNECTION_ARCHITECTURE.md",
    "CONNECTION_COMPLETE.md",
    "CPU_BOTTLENECK_FIX.md",
    "DECISION_GUIDE_5_DOLLARS.md",
    "FRONTEND_CONNECTION.md",
    "FRONTEND_USAGE.md",
    "FRONTEND_VITE_MIGRATION_PLAN.md",
    "IMPLEMENTATION_SUMMARY.md",
    "KAGGLE_AZURE_TESTING_GUIDE.md",
    "KAGGLE_COMPLETE_GUIDE.md",
    "KAGGLE_FILES_OVERVIEW.md",
    "KAGGLE_FIXES_APPLIED.md",
    "KAGGLE_KERNEL_RESTART_FIX.md",
    "KAGGLE_LLM_COMPLETE_GUIDE.md",
    "KAGGLE_OUTPUT_ISSUE_ANALYSIS.md",
    "KAGGLE_RESTART_REQUIRED.md",
    "KAGGLE_SETUP_GUIDE.md",
    "KAGGLE_SOLUTION_FINAL.md",
    "KAGGLE_TRAINING_AZURE_DEPLOYMENT.md",
    "KAGGLE_WHISPER_TEST_SETUP.md",
    "LLM_COMPLETE_IMPLEMENTATION.md",
    "LLM_TRAINING_QUICK_ANSWERS.md",
    "LORA_STATUS_FIX.md",
    "METRICS_EXPLAINED.md",
    "MODAL_SETUP_COMPLETE.md",
    "PHASE2_IMPLEMENTATION.md",
    "PHASE3_WEEK2_SUMMARY.md",
    "QLORA_EXPLAINED.md",
    "QUICK_REFERENCE.md",
    "QUICK_START.md",
    "README_FRONTEND_CONNECTION.md",
    "SIMPLIFIED_4_DATASETS_ONLY.md",
    "SPLIT_PROCESSING_GUIDE.md",
    "STARTUP_GUIDE.md",
    "START_HERE.md",
    "START_HERE_SIMPLE.md",
    "test_services.md",
    "TRAIN_LLM_COMPLETE_GUIDE.md",
    "VOICE_AGENT_SETUP.md",
    "WARNINGS_EXPLAINED.md",
    "WEEK1_AUDIT.md",
    "WEEK1_FIXES_COMPLETE.md",
    "WEEK2_AUDIT.md",
    "WHISPERX_LORA_COMPLETE.md",
    "WHISPER_EXPLAINED.md",
    "WORKFLOW_AFTER_TEST_ASR.md"
)

foreach ($file in $mdFilesToDelete) {
    if (Test-Path $file) {
        Move-Item $file $backupFolder -Force
        Write-Host "  ✓ Moved: $file" -ForegroundColor Gray
    }
}

# ============================================
# 2. TEST FILES (Python test scripts)
# ============================================
Write-Host "`n🧪 Cleaning up test files..." -ForegroundColor Green

$testFiles = Get-ChildItem -Filter "test_*.py" | Where-Object { $_.Name -ne "test_integration.py" }
foreach ($file in $testFiles) {
    Move-Item $file.FullName $backupFolder -Force
    Write-Host "  ✓ Moved: $($file.Name)" -ForegroundColor Gray
}

# ============================================
# 3. KAGGLE-SPECIFIC FILES
# ============================================
Write-Host "`n📊 Cleaning up Kaggle files..." -ForegroundColor Green

$kaggleFiles = @(
    "KAGGLE_CELL1_ALTERNATIVE.py",
    "KAGGLE_CHECK_ACCELERATE_FILE.py",
    "KAGGLE_CHECK_VERSIONS.py",
    "KAGGLE_COMPLETE_FIX.py",
    "kaggle_download_models.py",
    "KAGGLE_FIX_ACCELERATE.py",
    "KAGGLE_FIX_CORRUPTED_INSTALL.py",
    "KAGGLE_INSTALL_CELL.py",
    "kaggle_install_dependencies.py",
    "kaggle_llm_only.py",
    "kaggle_llm_with_speakers.py",
    "KAGGLE_NOTEBOOK.ipynb",
    "KAGGLE_NOTEBOOK.py",
    "kaggle_pipeline.py",
    "KAGGLE_SIMPLE_ANALYSIS.py",
    "KAGGLE_WHISPER_LORA_TEST.ipynb",
    "KAGGLE_WORKING_NOTEBOOK.ipynb"
)

foreach ($file in $kaggleFiles) {
    if (Test-Path $file) {
        Move-Item $file $backupFolder -Force
        Write-Host "  ✓ Moved: $file" -ForegroundColor Gray
    }
}

# Move kaggle-upload folder
if (Test-Path "kaggle-upload") {
    Move-Item "kaggle-upload" $backupFolder -Force
    Write-Host "  ✓ Moved: kaggle-upload/" -ForegroundColor Gray
}

# Move kaggle-services.zip
if (Test-Path "kaggle-services.zip") {
    Move-Item "kaggle-services.zip" $backupFolder -Force
    Write-Host "  ✓ Moved: kaggle-services.zip" -ForegroundColor Gray
}

# ============================================
# 4. COMPARISON/ANALYSIS FILES
# ============================================
Write-Host "`n📈 Cleaning up comparison/analysis files..." -ForegroundColor Green

$analysisFiles = @(
    "all_asr_comparison.json",
    "compare_all_wer.py",
    "compare_asr_wer.py",
    "compare_whisper_wer.py",
    "lora_wer_results.json",
    "wer_comparison_results.json",
    "whisper_wer_comparison.json",
    "reference_test1.txt"
)

foreach ($file in $analysisFiles) {
    if (Test-Path $file) {
        Move-Item $file $backupFolder -Force
        Write-Host "  ✓ Moved: $file" -ForegroundColor Gray
    }
}

# ============================================
# 5. UTILITY SCRIPTS (Not needed for production)
# ============================================
Write-Host "`n🔧 Cleaning up utility scripts..." -ForegroundColor Green

$utilityFiles = @(
    "check_lora_words.py",
    "check_shifaa_structure.py",
    "create_training_manifest.py",
    "download_models.py",
    "extract_ALL_datasets.py",
    "local_asr_only.py",
    "prepare_google_drive_data.py",
    "process_audio_local.py",
    "quick_whisper_test.py",
    "setup_modal.ps1",
    "start_asr_with_lora.ps1",
    "train_lora_modal.py",
    "train_mmed_llama_modal.py"
)

foreach ($file in $utilityFiles) {
    if (Test-Path $file) {
        Move-Item $file $backupFolder -Force
        Write-Host "  ✓ Moved: $file" -ForegroundColor Gray
    }
}

# ============================================
# 6. AUDIO TEST FILES
# ============================================
Write-Host "`n🎵 Cleaning up test audio files..." -ForegroundColor Green

$audioFiles = @(
    "test1.m4a",
    "test1.mp3",
    "التهاب اللثه.m4a"
)

foreach ($file in $audioFiles) {
    if (Test-Path $file) {
        Move-Item $file $backupFolder -Force
        Write-Host "  ✓ Moved: $file" -ForegroundColor Gray
    }
}

# ============================================
# SUMMARY
# ============================================
Write-Host "`n✅ Cleanup complete!" -ForegroundColor Green
Write-Host "📦 All removed files backed up to: $backupFolder" -ForegroundColor Yellow
Write-Host "`n📊 Summary of cleaned project:" -ForegroundColor Cyan
Write-Host "  ✓ Removed ~50+ markdown documentation files" -ForegroundColor Gray
Write-Host "  ✓ Removed ~15+ test scripts" -ForegroundColor Gray
Write-Host "  ✓ Removed ~15+ Kaggle-specific files" -ForegroundColor Gray
Write-Host "  ✓ Removed analysis/comparison files" -ForegroundColor Gray
Write-Host "  ✓ Removed utility scripts" -ForegroundColor Gray
Write-Host "  ✓ Removed test audio files" -ForegroundColor Gray
Write-Host "`n📁 Kept important files:" -ForegroundColor Cyan
Write-Host "  ✓ README.md" -ForegroundColor Gray
Write-Host "  ✓ QUICKSTART.md" -ForegroundColor Gray
Write-Host "  ✓ USER_GUIDE.md" -ForegroundColor Gray
Write-Host "  ✓ TROUBLESHOOTING.md" -ForegroundColor Gray
Write-Host "  ✓ All service code (services/, gateway/, frontend-vite/)" -ForegroundColor Gray
Write-Host "  ✓ Docker/infra configuration" -ForegroundColor Gray
Write-Host "  ✓ test_integration.py (main test file)" -ForegroundColor Gray

Write-Host "`n💡 To restore files if needed: Copy from $backupFolder" -ForegroundColor Yellow
