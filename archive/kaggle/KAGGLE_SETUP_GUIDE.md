# Kaggle Setup Guide - Medical Audio Processing

## Quick Start (5 Steps)

### Step 1: Create Kaggle Notebook
1. Go to https://www.kaggle.com/
2. Click **"New Notebook"**
3. Enable **GPU**: Settings → Accelerator → **GPU T4 x2**
4. Set Internet: **On** (to download models)

### Step 2: Upload Your Audio Files
1. Click **"+ Add Data"** (right panel)
2. Click **"Upload"** → Select your MP3/WAV files
3. Or use existing Kaggle dataset

### Step 3: Install Dependencies
Create a new code cell and run (IMPORTANT - Copy the ENTIRE cell from `kaggle_install_dependencies.py`):

```python
# Fix numpy/scipy compatibility (CRITICAL!)
!pip install -q --upgrade numpy==1.24.3 scipy==1.11.4

# Install PyTorch with CUDA
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ML packages
!pip install -q transformers==4.36.0 bitsandbytes==0.41.3 accelerate==0.25.0 whisperx

# Verify
import torch
print(f"GPU: {torch.cuda.is_available()}")
```

**Why specific versions?** Kaggle's default numpy/scipy have a compatibility bug with transformers.

### Step 4: Pre-Download Models (RECOMMENDED)
This saves time on subsequent runs:

```python
# Copy and paste contents of kaggle_download_models.py
# This will download:
# - WhisperX large-v3 (~3GB, 2-5 mins)
# - Alignment model (~300MB, 30s)
# - MMed-Llama-3-8B (~8GB on GPU, 5-10 mins)
#
# Total: ~11GB, ~10-15 minutes first time
# Cached for next runs!
```

### Step 5: Copy the Pipeline Code
1. Create a new code cell
2. Copy entire contents of `kaggle_pipeline.py`
3. Paste into the cell

### Step 6: Run the Pipeline
```python
# The script will automatically:
# 1. Detect GPU (T4)
# 2. Find all audio files in /kaggle/input
# 3. Load models (~5 minutes)
# 4. Process all audio files
# 5. Save results to /kaggle/working

# Just run the cell - it handles everything!
```

---

## What Happens on Kaggle

### Model Loading

**First Run (downloads models):**
```
📥 Downloading WhisperX large-v3 (~3GB)... 2-5 minutes
📥 Downloading Alignment model (~300MB)... 30 seconds  
📥 Downloading MMed-Llama-3-8B (~8GB)... 5-10 minutes
Total: ~10-15 minutes
```

**Subsequent Runs (from cache):**
```
📥 Loading WhisperX model (large-v3)...
✅ WhisperX loaded in 45.2s

📥 Loading LLM model (MMed-Llama-3-8B)...
✅ Using GPU with 4-bit quantization
✅ LLM loaded in 243.5s (4.1 minutes)

Total: ~5 minutes
```

### Processing Each Audio File

**Example: 134s audio file**

| Step | Time on GPU | Time on CPU |
|------|-------------|-------------|
| ASR Transcription | ~30s | ~900s |
| LLM Correction | ~5s | ~1400s |
| SOAP Generation | ~15s | ~3000s |
| **Total** | **~50s** | **~5300s (88 min)** |

**Speed:** 500x faster on Kaggle GPU!

---

## File Structure on Kaggle

```
/kaggle/
├── input/              # Your uploaded audio files
│   ├── test1.mp3
│   ├── test2.mp3
│   └── recording.wav
│
└── working/            # Results (downloadable)
    ├── test1_result.json
    ├── test2_result.json
    ├── recording_result.json
    └── processing_summary.json
```

---

## Output Format

### Individual Result: `test1_result.json`
```json
{
  "audio_file": "/kaggle/input/test1.mp3",
  "dialect": "egypt",
  "device": "cuda",
  "asr_result": {
    "segments": [
      {
        "start": 0.0,
        "end": 3.5,
        "text": "السلام عليكم يا دكتور",
        "speaker": "SPEAKER_00"
      }
    ],
    "full_text": "السلام عليكم يا دكتور. عندي الم في الصدر."
  },
  "corrected_text": "السلام عليكم يا دكتور. عندي ألم في الصدر.",
  "soap_note": "S: المريض يشكو من ألم في الصدر...",
  "status": "success"
}
```

### Summary: `processing_summary.json`
```json
{
  "total_files": 3,
  "successful": 3,
  "failed": 0,
  "dialect": "egypt",
  "device": "cuda",
  "results": [...]
}
```

---

## Download Results

1. After notebook finishes running
2. Go to **Output** tab (right panel)
3. Click **Download All** to get ZIP with all results
4. Or download individual JSON files

---

## Optional: Speaker Diarization

To enable speaker detection (who said what):

1. Get HuggingFace token: https://huggingface.co/settings/tokens
2. Accept pyannote terms: https://huggingface.co/pyannote/speaker-diarization-3.1
3. In Kaggle: **Settings** → **Secrets** → Add **HF_TOKEN**
4. The script will automatically use it

Without token: Script runs fine, just no speaker labels

---

## Customization

### Change Dialect
Edit in the code:
```python
dialect = "egypt"  # Options: egypt, gulf, levantine, maghreb, msa
```

### Process Specific Files
Modify `find_audio_files()` to filter:
```python
audio_files = [
    "/kaggle/input/test1.mp3",
    "/kaggle/input/test2.mp3"
]
```

### Adjust LLM Settings
Change token limits for faster/slower but better quality:
```python
max_new_tokens=64   # Correction (fast)
max_new_tokens=128  # More detailed correction
max_new_tokens=256  # SOAP notes (detailed)
max_new_tokens=512  # Very detailed SOAP notes
```

---

## Kaggle Quota

- **Free Tier**: 30 hours/week GPU time
- **T4 GPU**: 16GB VRAM
- **Session limit**: Up to 12 hours continuous

### Example Usage:
- Model loading: ~5 minutes (one-time per session)
- Process 1 audio file (2 min): ~1 minute
- Process 10 audio files: ~10 minutes
- Process 100 audio files: ~100 minutes

**You can process ~150 audio files per session (12 hours)**

---

## Troubleshooting

### "No audio files found"
- Make sure audio is uploaded to dataset
- Check file extensions (.mp3, .wav, .m4a)
- Verify dataset is added to notebook inputs

### "CUDA out of memory"
- Reduce `batch_size=16` → `batch_size=8` in transcribe
- Use shorter audio files
- Restart notebook to clear memory

### "Model loading too slow"
- Enable GPU: Settings → Accelerator → GPU T4 x2
- Enable Internet: Settings → Internet → On
- First run is slower (downloads models)

### "Generation taking too long"
- Check GPU is enabled (should be ~5-10s per generation)
- If using CPU by mistake, will take 20+ minutes
- Verify: Look for "✅ Using GPU" in output

---

## Next Steps After Processing

1. **Download** all results from Kaggle
2. **Extract** JSON files
3. **Upload** to your local system
4. **Use** in your frontend/gateway
5. **Analyze** transcriptions and SOAP notes

---

## Cost Comparison

| Platform | Cost | Speed | Setup |
|----------|------|-------|-------|
| **Kaggle** | **FREE** | **Fast (T4)** | **5 min** |
| Azure NC16as_T4_v3 | $1.20/hour | Fast (T4) | 30 min + blocked |
| Local CPU | Free | Very slow | Already set up |
| Google Colab Pro | $10/month | Fast | 10 min |

**Recommendation: Use Kaggle (FREE + FAST)**

---

## Summary

✅ **Upload audio files** to Kaggle dataset  
✅ **Copy `kaggle_pipeline.py`** to notebook  
✅ **Enable GPU T4**  
✅ **Run the notebook**  
✅ **Download results** from Output tab  

**Total time: ~5 minutes setup + processing time**

Processing is **500x faster** than your local CPU!
