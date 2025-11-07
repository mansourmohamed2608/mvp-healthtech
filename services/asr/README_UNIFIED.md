# ✅ ASR Service Unified - LoRA Integrated!

## What Changed

**Merged `app_whisperx_lora.py` into `app.py`**

Now `app.py` has **everything in one place**:
- ✅ Base WhisperX transcription
- ✅ Optional LoRA enhancement
- ✅ Speaker diarization
- ✅ Word-level timestamps
- ✅ Clear status logging

## How to Use

### 1. Start the Service

**Just run (no scripts needed):**
```bash
cd services/asr
python app.py
```

Or with uvicorn:
```bash
cd services/asr
uvicorn app:app --host 0.0.0.0 --port 5000
```

### 2. Check Startup Logs

You'll see:
```
================================================================================
✓ ASR SERVICE READY!
================================================================================
📊 CONFIGURATION:
   Base model: WhisperX large-v3
   Device: cuda
   Compute type: float16

🔧 LORA STATUS:
   ✅ LoRA ADAPTERS LOADED AND ACTIVE!
   📁 Path: ./lora_ckpt
   🎯 Enhanced Arabic medical transcription enabled

🎤 DIARIZATION:
   ✅ Enabled
================================================================================
```

### 3. Test It

```bash
python test_asr_complete.py
```

You'll see during transcription:
```
============================================================
🔥 USING LORA-ENHANCED MODEL!
============================================================
  📁 LoRA adapters: ./lora_ckpt
  🎯 Enhanced Arabic medical transcription active
============================================================
```

## Configuration (.env)

```bash
# Device
DEVICE=cuda
COMPUTE_TYPE=float16

# Model
WHISPER_MODEL=large-v3

# LoRA (Optional - auto-detects if available)
USE_LORA=true
LORA_ADAPTER_PATH=./lora_ckpt

# Diarization (Optional)
ENABLE_DIARIZATION=true
HF_TOKEN=your_token_here
```

## Features

### Automatic LoRA Detection
- **If LoRA adapters exist** → Loads automatically
- **If LoRA not available** → Falls back to base model gracefully
- **If PEFT not installed** → Works without LoRA (base model only)

### Per-Request Control
Send in your request:
```json
{
  "audio": "base64_audio",
  "language": "ar",
  "use_lora": true  // ← Control LoRA usage per request
}
```

### Response Includes Model Info
```json
{
  "text": "...",
  "model_used": "Whisper Large v3 + LoRA",  // ← Shows which model was used
  ...
}
```

## Health Check

```bash
curl http://localhost:5000/health
```

Returns:
```json
{
  "status": "healthy",
  "model": "large-v3",
  "lora_enabled": true,
  "lora_path": "./lora_ckpt",
  "device": "cuda"
}
```

## What Happens if LoRA Not Available?

**No problem!** The service:
1. ✅ Starts normally
2. ✅ Uses base WhisperX model
3. ✅ Logs why LoRA isn't loaded
4. ✅ Still works perfectly

Example log:
```
ℹ️  LoRA adapter path not found: ./lora_ckpt
✓ Base Whisper model loaded
```

## Clean Up

**Can delete now:**
- ❌ `app_whisperx_lora.py` (merged into app.py)
- ❌ `start_asr_with_lora.ps1` (not needed)
- ❌ Any other `app_*.py` variants

**Keep:**
- ✅ `app.py` (has everything)
- ✅ `.env` (configuration)
- ✅ `lora_ckpt/` (your trained adapters)
- ✅ `test_asr_complete.py` (testing)

## Summary

🎯 **One file does it all:** `app.py`  
🔥 **LoRA works automatically** if adapters exist  
⚡ **No scripts needed** - just `python app.py`  
📊 **Clear logging** - always know which model is active  
🛡️ **Graceful fallback** - works with or without LoRA  

**Just start the service and go!** 🚀
