# Install Local Dependencies for Dataset Extraction
# Run this before running extract_ALL_datasets.py

Write-Host "Installing packages for dataset extraction..." -ForegroundColor Green
Write-Host ""

# Core packages with specific versions (tested for compatibility)
pip install datasets==3.6.0
pip install huggingface_hub==0.34.2
pip install pandas==2.2.0
pip install openpyxl==3.1.5
pip install tqdm==4.66.5

Write-Host ""
Write-Host "✅ All packages installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run: python extract_ALL_datasets.py" -ForegroundColor Cyan
Write-Host "2. Wait 30-60 minutes for extraction to complete" -ForegroundColor Cyan
Write-Host "3. Then setup Modal and upload data" -ForegroundColor Cyan
Write-Host ""
