# WhisperX + LoRA Integration Guide

## ✅ What You Have

- **Base Model**: Whisper Large v3
- **LoRA Adapters**: Trained for Arabic medical transcription
- **Location**: `d:\Downloads\HealthTech\mvp-healthtech\services\asr\lora_ckpt\`
- **WhisperX**: Already configured

## 🎯 What's New

### 1. New Service: `app_whisperx_lora.py`

This combines your fine-tuned LoRA adapters with WhisperX features:

**Features:**
- ✅ Uses your fine-tuned Whisper Large v3 + LoRA
- ✅ Speaker diarization (WhisperX)
- ✅ Word-level timestamps (WhisperX)
- ✅ VAD preprocessing (WhisperX)
- ✅ Toggle LoRA on/off per request
- ✅ Fallback to base model if needed

### 2. LoRA Configuration

**From your `adapter_config.json`:**
```json
{
  "base_model": "openai/whisper-large-v3",
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "target_modules": [
    "q_proj", "v_proj", "k_proj",
    "fc1", "fc2", "out_proj"
  ]
}
```

## 🚀 How to Use

### Step 1: Set Environment Variables

Create `.env` file in `services/asr/`:

```bash
# Model Configuration
DEVICE=cuda
COMPUTE_TYPE=float16
WHISPER_MODEL=large-v3

# LoRA Configuration
USE_LORA=true
LORA_ADAPTER_PATH=./lora_ckpt

# HuggingFace Token (for diarization)
HF_TOKEN=your_token_here

# Features
ENABLE_DIARIZATION=true
ENABLE_VAD=true

# Server
PORT=8001
```

### Step 2: Install Dependencies

```powershell
cd services\asr
pip install -r requirements_whisperx.txt
pip install peft transformers
```

### Step 3: Run the Service

```powershell
cd services\asr
python app_whisperx_lora.py
```

### Step 4: Test the Integration

```python
import requests
import base64

# Read audio file
with open("test_audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Transcribe with LoRA
response = requests.post("http://localhost:8001/transcribe", json={
    "audio": audio_base64,
    "language": "ar",
    "dialect": "egypt",
    "use_lora": True,  # Use your fine-tuned model!
    "enable_diarization": True
})

result = response.json()
print(f"Model used: {result['model_used']}")
print(f"Text: {result['text']}")
print(f"Speakers: {result['speakers']}")
```

## 🔄 Comparison: With vs Without LoRA

### Without LoRA (Base WhisperX):
```json
{
  "use_lora": false
}
```
- Uses standard Whisper Large v3
- General-purpose Arabic transcription
- No medical vocabulary specialization

### With LoRA (Your Fine-tuned Model):
```json
{
  "use_lora": true
}
```
- Uses YOUR fine-tuned Whisper Large v3 + LoRA
- Specialized for Arabic medical terminology
- Better accuracy on medical conversations
- Same LoRA adapters you already trained!

## 📊 Expected Improvements

With your LoRA adapters, you should see:

1. **Better medical term recognition**
   - Before: "مرض السكر" → "مرض السكري" (correct)
   - Before: "الكلسترول" → "الكوليسترول" (correct)

2. **Improved Arabic dialect handling**
   - Your adapters were trained on medical conversations
   - Better handling of Egyptian/Gulf Arabic medical terms

3. **Reduced hallucinations**
   - Medical context awareness
   - More accurate symptom descriptions

## 🔧 Integration with Existing Code

### Update `local_asr_only.py`:

```python
# OLD:
response = requests.post("http://localhost:8001/transcribe", ...)

# NEW: Add use_lora parameter
response = requests.post("http://localhost:8001/transcribe", json={
    "audio": audio_base64,
    "language": "ar",
    "dialect": "egypt",
    "use_lora": True,  # Enable your fine-tuned model
    "enable_diarization": True
})
```

### Update `test_asr.py`:

```python
# Compare with and without LoRA
results = []

for use_lora in [False, True]:
    response = requests.post("http://localhost:8001/transcribe", json={
        "audio": audio_base64,
        "use_lora": use_lora,
        ...
    })
    results.append({
        "model": "LoRA" if use_lora else "Base",
        "text": response.json()["text"]
    })

# Compare results
for r in results:
    print(f"{r['model']}: {r['text']}")
```

## 🎯 Kaggle AHD Dataset Integration

### Updated `download_free_data.py`:

Now checks multiple locations for AHD:
1. `ahd_dataset.xlsx` (local)
2. `/kaggle/input/ahd-dataset/AHD.xlsx` (Kaggle)
3. `../input/ahd-dataset/AHD.xlsx` (Kaggle relative)

**In Kaggle Notebook:**
```python
# AHD will be automatically found in Kaggle input!
# Just add ahd-dataset to your notebook inputs
```

**On Kaggle, the script will:**
1. Auto-detect AHD in `/kaggle/input/ahd-dataset/`
2. Process ALL 808k+ examples
3. Save to `/kaggle/working/training_data_ahd.json`

## 📂 File Structure

```
services/asr/
├── app_whisperx.py              # Original WhisperX (base model)
├── app_whisperx_lora.py         # NEW: WhisperX + Your LoRA
├── lora_ckpt/                   # Your LoRA adapters
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── ...
├── requirements_whisperx.txt
└── .env                         # Configuration

training/
├── download_free_data.py        # Updated: Auto-finds Kaggle AHD
├── training_data_shifaa.json    # 84,422 examples
├── training_data_ahd.json       # 808k+ examples (when processed)
├── training_data_mmedc.json     # 167 examples
└── training_data_all_combined.json  # ALL examples
```

## ✅ Quick Start Commands

```powershell
# 1. Set up environment
cd d:\Downloads\HealthTech\mvp-healthtech\services\asr
cp .env.example .env
# Edit .env with your settings

# 2. Start LoRA-enhanced service
python app_whisperx_lora.py

# 3. Test it
cd ../../
python test_asr.py

# 4. Compare models
python test_asr_lora_comparison.py  # We can create this!
```

## 🎉 Summary

**You now have:**
- ✅ WhisperX + Your LoRA adapters integrated
- ✅ Toggle between base and fine-tuned model
- ✅ All WhisperX features (diarization, timestamps, VAD)
- ✅ Kaggle AHD dataset auto-detection
- ✅ Ready for 893k+ training examples!

**Total Cost: $0** (Everything uses your existing resources!)
