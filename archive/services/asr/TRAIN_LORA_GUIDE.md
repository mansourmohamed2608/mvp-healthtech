# Training LoRA for Arabic Medical ASR - Complete Guide

## Overview
This guide will help you train a LoRA adapter for Whisper Large-V3 to improve Arabic medical transcription accuracy.

## Prerequisites

### 1. Training Data Requirements
You need:
- **Audio files**: Egyptian Arabic medical conversations (MP3/WAV format)
- **Transcriptions**: Accurate text for each audio file
- **Minimum**: 100-500 audio samples (30min - 3hrs total)
- **Recommended**: 1000+ samples (10+ hours) for best results

### 2. Data Format
Create a CSV file (`manifest.csv`) with two columns:

```csv
audio_filepath,text
audio/sample1.wav,السلام عليكم يا دكتور
audio/sample2.wav,عندي ألم في اللثة
audio/sample3.wav,بغسل أسناني كل يوم
```

**Important**: 
- `audio_filepath`: Path to audio file (relative or absolute)
- `text`: Exact transcription in Arabic

## Training Options

### Option 1: Train on Kaggle (Recommended - FREE GPU)

**Why Kaggle?**
- ✅ Free GPU (30 hrs/week)
- ✅ No setup needed
- ✅ Fast training (2-4 hours for 1000 samples)

**Steps:**

1. **Prepare your data**:
   - Create `manifest.csv` with your audio files and transcriptions
   - Zip all audio files: `audio_files.zip`

2. **Upload to Kaggle**:
   - Go to kaggle.com → Datasets → New Dataset
   - Upload `manifest.csv` and `audio_files.zip`
   - Make dataset public or private

3. **Create Kaggle Notebook**:
   - Copy the script from `train_lora_whisper.py`
   - Update the CONFIG section:
     ```python
     CONFIG = {
         "csv_path": "/kaggle/input/YOUR-DATASET-NAME/manifest.csv",
         "dataset_root_fallback": "/kaggle/input/YOUR-DATASET-NAME",
         "output_dir": "/kaggle/working/lora_ckpt",
         "base_model": "openai/whisper-large-v3",
         "language": "arabic",
         "task": "transcribe",
         "num_epochs": 3,  # Increase to 3-5 for better results
         "batch_size": 1,
         "grad_accum": 16,
         "lr": 1e-4,
         "train_max_rows": None,  # Use all data
         "use_hint": True,
         "hint_prefix": "ملاحظة طبية:",  # Medical note prefix
         "save_steps": 400,
         "logging_steps": 25,
     }
     ```

4. **Run training**:
   - Enable GPU: Settings → Accelerator → GPU T4 x2
   - Run all cells
   - Training will take 2-4 hours depending on data size

5. **Download trained LoRA**:
   - After training completes, download `/kaggle/working/lora_ckpt` folder
   - This contains your LoRA adapters!

### Option 2: Train Locally (If you have GPU)

**Requirements:**
- NVIDIA GPU with 16GB+ VRAM
- CUDA installed
- Python 3.9+

**Steps:**

1. **Install dependencies**:
```powershell
pip install transformers peft accelerate datasets librosa jiwer soundfile bitsandbytes
```

2. **Prepare data structure**:
```
services/asr/training_data/
├── manifest.csv
├── audio/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
```

3. **Update `train_lora_whisper.py` CONFIG**:
```python
CONFIG = {
    "csv_path": "./training_data/manifest.csv",
    "dataset_root_fallback": "./training_data",
    "output_dir": "./lora_ckpt_new",
    "num_epochs": 3,
    ...
}
```

4. **Run training**:
```powershell
cd services/asr
python train_lora_whisper.py
```

## Using Your Trained LoRA

After training completes, you'll have a `lora_ckpt` folder with these files:
```
lora_ckpt/
├── adapter_config.json
├── adapter_model.safetensors  ← Your LoRA weights!
├── preprocessor_config.json
├── tokenizer.json
└── ...
```

### Replace Old LoRA:

1. **Backup old LoRA**:
```powershell
cd services/asr
mv lora_ckpt lora_ckpt_old
```

2. **Copy new LoRA**:
```powershell
# If trained on Kaggle: extract downloaded zip
# If trained locally: copy the output folder
cp -r lora_ckpt_new lora_ckpt
```

3. **Restart ASR service**:
```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 5000
```

4. **Test with your script**:
```powershell
python compare_asr_wer.py test1.mp3 reference_test1.txt ar egypt
```

## Data Collection Tips

### Where to Get Training Data?

1. **Record Real Consultations** (Best quality):
   - Record doctor-patient conversations (with consent!)
   - Manually transcribe or use professional transcription service
   - Aim for diverse speakers, accents, medical topics

2. **Use Existing Datasets**:
   - Mozilla Common Voice Arabic (general Arabic)
   - MGB-2 Arabic (broadcast news)
   - QASR Arabic (Quranic recitation)
   - ⚠️ Medical-specific datasets are rare!

3. **Generate Synthetic Data**:
   - Use TTS (Text-to-Speech) to create audio from medical texts
   - Not as good as real data but can help
   - Arabic TTS: gTTS, Coqui TTS, or commercial APIs

### Data Quality Tips:
- ✅ Clear audio (no background noise)
- ✅ Accurate transcriptions (spell-check!)
- ✅ Diverse speakers (male/female, young/old)
- ✅ Varied medical topics (dental, cardiology, etc.)
- ❌ Avoid very short clips (< 2 seconds)
- ❌ Avoid very long clips (> 30 seconds, split them)

## Training Hyperparameters Explained

```python
"num_epochs": 3           # How many times to go through data (3-5 recommended)
"batch_size": 1           # Keep at 1 for large models
"grad_accum": 16          # Effective batch = 16 (higher = more stable)
"lr": 1e-4               # Learning rate (1e-4 is safe, try 5e-5 if overfitting)
"r": 8                   # LoRA rank (8 = good balance, 16 = more capacity)
"lora_alpha": 16         # LoRA scaling (typically 2x rank)
"lora_dropout": 0.05     # Prevent overfitting
```

## Expected Results

After training on **good quality data**:
- **Before LoRA**: 22% WER (baseline WhisperX)
- **After LoRA**: 10-15% WER (40-50% improvement!)
- **Training time**: 2-4 hours on Kaggle GPU
- **Model size**: ~30MB (just the LoRA adapters)

## Troubleshooting

### Issue: WER got worse after training
**Causes:**
- Poor quality training data
- Too few epochs (undertrained)
- Too many epochs (overfitted)
- Wrong language/dialect in training data

**Solutions:**
- Check data quality manually
- Try training for 3-5 epochs
- Use more diverse training data
- Add validation set to monitor overfitting

### Issue: Out of memory during training
**Solutions:**
- Reduce `batch_size` to 1
- Reduce `grad_accum` to 8
- Use `load_in_4bit=True` (already in script)
- Use shorter audio clips (< 20 seconds)

### Issue: Training too slow
**Solutions:**
- Use Kaggle GPU (free and fast!)
- Reduce `train_max_rows` for testing
- Use `fp16=True` (already enabled)
- Remove `gradient_checkpointing` (uses more memory but faster)

## Next Steps

1. **Collect/prepare training data** (most important!)
2. **Train on Kaggle** (easiest and free)
3. **Test with your audio** (`compare_asr_wer.py`)
4. **Iterate**: More data = better results!

## Need Help?

Common questions:
- **How much data do I need?** Minimum 100 samples, ideal 1000+
- **Can I use non-medical Arabic?** Yes, but medical terms won't improve
- **How long does training take?** 2-4 hours on GPU, days on CPU
- **Do I need to retrain often?** Only when you get new data or want better results

---

**Summary**: The current LoRA is producing 41% WER (worse than baseline 22%). You need to train a new LoRA with proper Arabic medical data to see improvements. Kaggle is the easiest way with free GPUs!
