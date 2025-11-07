# Kaggle Files Overview

## 📦 Files for Kaggle

### 1. **`kaggle_download_models.py`** - Pre-download Models
**Purpose:** Download all models to cache before running pipeline  
**When to use:** First time setup, or when models need updating  
**Runtime:** 10-15 minutes (one-time)

**What it downloads:**
- ✅ WhisperX large-v3 (~3GB)
- ✅ Arabic alignment model (~300MB)
- ✅ MMed-Llama-3-8B (~8GB on GPU, ~16GB on CPU)

**How to use on Kaggle:**
1. Create new code cell
2. Copy entire script
3. Run once
4. Models cached in `/kaggle/working/models`

---

### 2. **`kaggle_pipeline.py`** - Main Processing Pipeline
**Purpose:** Process audio files end-to-end  
**When to use:** After models are downloaded  
**Runtime:** ~50 seconds per audio file (with GPU)

**What it does:**
1. 🎤 ASR transcription (WhisperX)
2. 👥 Speaker diarization (optional)
3. ✏️ LLM correction
4. 📋 SOAP note generation
5. 💾 Save results as JSON

**How to use on Kaggle:**
1. Upload audio files to Kaggle dataset
2. Add dataset as input
3. Copy this script to notebook
4. Run (finds all audio automatically)
5. Download results from `/kaggle/working`

---

### 3. **`KAGGLE_SETUP_GUIDE.md`** - Complete Instructions
**Purpose:** Step-by-step setup guide  
**Includes:**
- Kaggle notebook setup
- GPU enablement
- Model download instructions
- Output format examples
- Troubleshooting tips
- Cost comparison

---

## 🚀 Quick Start Workflow

### Option A: Download Models First (RECOMMENDED)
```
1. Run kaggle_download_models.py (10-15 mins)
   ↓
2. Models cached
   ↓
3. Run kaggle_pipeline.py (fast - 50s per file)
   ↓
4. Download results
```

### Option B: Run Directly
```
1. Run kaggle_pipeline.py
   ↓
2. Auto-downloads models on first use (10-15 mins)
   ↓
3. Processes audio (50s per file)
   ↓
4. Download results
```

**Recommendation:** Use Option A if processing multiple files

---

## 📊 Time Comparison

### First Run (No Cache)
| Step | Time |
|------|------|
| Download models | 10-15 mins |
| Load models | 5 mins |
| Process 1 audio | 50s |
| **Total for 1 file** | **~20 mins** |

### Subsequent Runs (With Cache)
| Step | Time |
|------|------|
| Load models | 5 mins |
| Process 1 audio | 50s |
| **Total for 1 file** | **~6 mins** |

### Batch Processing (10 files)
| Step | Time |
|------|------|
| Load models (once) | 5 mins |
| Process 10 audios | 8-10 mins |
| **Total for 10 files** | **~15 mins** |

---

## 💾 Disk Space on Kaggle

```
/kaggle/working/models/         ~11 GB (cached models)
/kaggle/working/*.json          ~10 KB per audio file
/kaggle/input/                  Your audio files
```

**Kaggle limits:**
- Disk: 73 GB (plenty of space)
- RAM: 16 GB
- GPU VRAM: 16 GB (T4)

---

## 🔧 Model Specifications

### WhisperX Large-v3
- **Size:** ~3 GB
- **Purpose:** ASR (speech-to-text)
- **Language:** Arabic optimized
- **Speed:** ~30s for 2-minute audio on GPU

### MMed-Llama-3-8B
- **Size:** 8 GB (4-bit) or 16 GB (full)
- **Purpose:** Medical LLM
- **Tasks:** Correction + SOAP notes
- **Speed:** ~5s correction + ~15s SOAP on GPU

### Total Cache Size: ~11 GB

---

## 📤 Output Files

Each audio produces:
```json
{
  "audio_file": "/kaggle/input/test1.mp3",
  "dialect": "egypt",
  "device": "cuda",
  "asr_result": {
    "segments": [...],
    "full_text": "..."
  },
  "corrected_text": "...",
  "soap_note": "...",
  "status": "success"
}
```

Plus summary:
```json
{
  "total_files": 10,
  "successful": 10,
  "failed": 0,
  "device": "cuda",
  "results": [...]
}
```

---

## 🎯 When to Use What

### Use `kaggle_download_models.py` when:
- ✅ First time setup
- ✅ Processing many files (saves time later)
- ✅ Want to separate download from processing
- ✅ Testing model availability

### Use `kaggle_pipeline.py` when:
- ✅ Ready to process audio files
- ✅ Models already cached (or first run)
- ✅ Want full end-to-end pipeline
- ✅ Need JSON outputs

---

## ⚡ Performance: Kaggle vs Local

| Task | Local CPU | Kaggle GPU | Speed Gain |
|------|-----------|------------|------------|
| Load models | 20 mins | 5 mins | 4x faster |
| ASR (134s audio) | 927s | 30s | 31x faster |
| LLM correction | 1415s | 5s | 283x faster |
| SOAP generation | 3572s | 15s | 238x faster |
| **Total pipeline** | **~105 mins** | **~6 mins** | **18x faster** |

---

## 💡 Pro Tips

1. **Batch Processing:** Upload multiple audio files, process all at once
2. **Cache Models:** Use download script first to save time
3. **HF Token:** Add to Kaggle secrets for speaker diarization
4. **Save Session:** Keep models cached between runs (same session)
5. **Download Results:** Always download from `/kaggle/working` before closing

---

## 🆘 Common Issues

### "No GPU detected"
→ Settings → Accelerator → GPU T4 x2

### "Model download timeout"
→ Settings → Internet → On  
→ Try again (Kaggle has download quotas)

### "CUDA out of memory"
→ Restart notebook  
→ Reduce batch_size in code

### "No audio files found"
→ Add dataset as input  
→ Check file extensions (.mp3, .wav)

---

## 📚 Further Reading

- **KAGGLE_SETUP_GUIDE.md** - Detailed setup instructions
- **kaggle_pipeline.py** - Main script with comments
- **kaggle_download_models.py** - Pre-download script with comments

---

## ✅ Checklist

Before running on Kaggle:
- [ ] Kaggle account created
- [ ] GPU enabled (T4)
- [ ] Internet enabled
- [ ] Audio files uploaded as dataset
- [ ] Dataset added to notebook inputs
- [ ] Dependencies installed
- [ ] Models downloaded (optional but recommended)
- [ ] Pipeline script copied
- [ ] Ready to run!

Estimated total time for first complete run: **20-25 minutes**  
Estimated time for subsequent runs: **6-10 minutes per audio**
