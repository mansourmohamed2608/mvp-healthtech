# WhisperX Installation Script for Windows
# Run this in PowerShell from services/asr directory

Write-Host "Installing WhisperX and dependencies..." -ForegroundColor Green

# Install WhisperX
pip install git+https://github.com/m-bain/whisperx.git

# Install additional dependencies for Arabic support
pip install transformers>=4.30.0
pip install pyannote.audio>=3.1.0

Write-Host "`nInstallation complete!" -ForegroundColor Green
Write-Host "Next: Set your HuggingFace token in .env file" -ForegroundColor Yellow
