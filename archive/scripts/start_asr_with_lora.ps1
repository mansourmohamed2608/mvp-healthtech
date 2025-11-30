# Quick Start: WhisperX + LoRA Integration
# Test your updated ASR service

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "WhisperX + LoRA ASR Service - Quick Start" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Check if in correct directory
if (-not (Test-Path "services\asr\app_whisperx.py")) {
    Write-Host "❌ Please run this script from the project root:" -ForegroundColor Red
    Write-Host "   cd d:\Downloads\HealthTech\mvp-healthtech" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Checking environment..." -ForegroundColor Green

# Check if .env exists
if (-not (Test-Path "services\asr\.env")) {
    Write-Host ""
    Write-Host "⚠️  No .env file found. Creating from .env.example..." -ForegroundColor Yellow
    
    if (Test-Path "services\asr\.env.example") {
        Copy-Item "services\asr\.env.example" "services\asr\.env"
        Write-Host "✅ Created .env file" -ForegroundColor Green
        Write-Host ""
        Write-Host "📝 Please edit services\asr\.env and set:" -ForegroundColor Cyan
        Write-Host "   - USE_LORA=true" -ForegroundColor White
        Write-Host "   - LORA_ADAPTER_PATH=./lora_ckpt" -ForegroundColor White
        Write-Host "   - HF_TOKEN=your_token (optional, for diarization)" -ForegroundColor White
        Write-Host ""
        Read-Host "Press Enter after updating .env file"
    } else {
        Write-Host "❌ .env.example not found!" -ForegroundColor Red
        Write-Host "Creating basic .env..." -ForegroundColor Yellow
        
        @"
DEVICE=cuda
COMPUTE_TYPE=float16
WHISPER_MODEL=large-v3
USE_LORA=true
LORA_ADAPTER_PATH=./lora_ckpt
ENABLE_DIARIZATION=true
ENABLE_VAD=true
HF_TOKEN=
PORT=8001
"@ | Out-File -FilePath "services\asr\.env" -Encoding utf8
        
        Write-Host "✅ Created basic .env" -ForegroundColor Green
        Write-Host "⚠️  Add your HF_TOKEN for diarization (optional)" -ForegroundColor Yellow
    }
}

# Check for LoRA adapters
Write-Host ""
Write-Host "📦 Checking LoRA adapters..." -ForegroundColor Cyan
if (Test-Path "services\asr\lora_ckpt\adapter_config.json") {
    Write-Host "✅ LoRA adapters found!" -ForegroundColor Green
    
    # Show LoRA config
    $config = Get-Content "services\asr\lora_ckpt\adapter_config.json" | ConvertFrom-Json
    Write-Host "   Base model: $($config.base_model_name_or_path)" -ForegroundColor White
    Write-Host "   LoRA rank: $($config.r)" -ForegroundColor White
    Write-Host "   LoRA alpha: $($config.lora_alpha)" -ForegroundColor White
} else {
    Write-Host "❌ LoRA adapters NOT found at: services\asr\lora_ckpt\" -ForegroundColor Red
    Write-Host ""
    Write-Host "The service will run without LoRA (base WhisperX only)" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

# Check Python dependencies
Write-Host ""
Write-Host "📦 Checking Python dependencies..." -ForegroundColor Cyan
python -c "import whisperx; import peft; import transformers" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ All dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠️  Some dependencies missing. Installing..." -ForegroundColor Yellow
    pip install whisperx peft transformers torch
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "🚀 Starting ASR Service..." -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The service will:" -ForegroundColor White
Write-Host "  1. Load WhisperX (base model)" -ForegroundColor White
Write-Host "  2. Load your LoRA adapters (if found)" -ForegroundColor White
Write-Host "  3. Load diarization model (if HF_TOKEN set)" -ForegroundColor White
Write-Host ""
Write-Host "Service will be available at: http://localhost:8001" -ForegroundColor Green
Write-Host "Health check: http://localhost:8001/health" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Start the service with LoRA support
cd services\asr
python app_whisperx_lora.py
