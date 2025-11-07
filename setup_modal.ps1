# Install Modal CLI and Setup
# Run this after extract_ALL_datasets.py completes

Write-Host "Setting up Modal for training..." -ForegroundColor Green
Write-Host ""

# Install Modal CLI
Write-Host "📦 Installing Modal CLI..." -ForegroundColor Cyan
pip install modal

Write-Host ""
Write-Host "🔑 Authenticating with Modal..." -ForegroundColor Cyan
Write-Host "   This will open a browser window. Follow the steps to get $30 free credits." -ForegroundColor Yellow
modal token new

Write-Host ""
Write-Host "📁 Creating Modal volume..." -ForegroundColor Cyan
modal volume create mmed-llama-qlora-training

Write-Host ""
Write-Host "✅ Modal setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Upload training data:" -ForegroundColor Cyan
Write-Host "   modal volume put mmed-llama-qlora-training training_data_combined_ALL.json" -ForegroundColor White
Write-Host ""
Write-Host "2. Upload AHD file (if you have it):" -ForegroundColor Cyan
Write-Host "   modal volume put mmed-llama-qlora-training AHD.xlsx" -ForegroundColor White
Write-Host ""
Write-Host "3. Start training:" -ForegroundColor Cyan
Write-Host "   modal run train_mmed_llama_modal.py" -ForegroundColor White
Write-Host ""
