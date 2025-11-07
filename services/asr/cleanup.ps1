# Cleanup Script - Remove Unnecessary Migration Files
# Run this AFTER successful migration to keep your project clean

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Cleanup WhisperX Migration Files" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$filesKept = @(
    "app.py (WhisperX version)",
    ".env (your configuration)",
    ".env.example (template)",
    "SIMPLE_SETUP.md (quick reference)"
)

$filesToDelete = @(
    "app_whisperx.py (already copied to app.py)",
    "WHISPERX_MIGRATION.md (detailed guide - not needed after migration)",
    "MIGRATION_SUMMARY.md (reference doc - not needed after migration)",
    "README_WHISPERX.md (overview - not needed after migration)",
    "requirements_whisperx.txt (info only)",
    "install_whisperx.ps1 (already ran)",
    "quick-start.ps1 (already ran)",
    "cleanup.ps1 (this file)"
)

Write-Host "This will delete the following files:" -ForegroundColor Yellow
Write-Host ""
foreach ($file in $filesToDelete) {
    Write-Host "  ❌ $file" -ForegroundColor Gray
}

Write-Host ""
Write-Host "These files will be kept:" -ForegroundColor Green
Write-Host ""
foreach ($file in $filesKept) {
    Write-Host "  ✅ $file" -ForegroundColor Gray
}

Write-Host ""
$confirm = Read-Host "Continue with cleanup? (y/n)"

if ($confirm -ne "y") {
    Write-Host "Cleanup cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Cleaning up..." -ForegroundColor Yellow

# Delete files if they exist
$deleted = 0
$files = @(
    "app_whisperx.py",
    "WHISPERX_MIGRATION.md",
    "MIGRATION_SUMMARY.md",
    "README_WHISPERX.md",
    "requirements_whisperx.txt",
    "install_whisperx.ps1",
    "quick-start.ps1"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  ✓ Deleted: $file" -ForegroundColor Green
        $deleted++
    }
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Cleanup Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Deleted: $deleted files" -ForegroundColor White
Write-Host "  Kept: app.py, .env, .env.example, SIMPLE_SETUP.md" -ForegroundColor White
Write-Host ""
Write-Host "Your ASR service is now clean and production-ready! 🎉" -ForegroundColor Green
Write-Host ""

# Self-delete
Remove-Item "cleanup.ps1" -Force
