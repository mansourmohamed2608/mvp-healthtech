# Split Processing Guide: Local ASR + Kaggle LLM

This approach lets you run ASR locally (fast) and LLM on Kaggle GPU (500x faster than local CPU).

## 🎯 Why Use This?

- **ASR Local**: Your machine handles this fine (~30 seconds)
- **LLM Kaggle**: GPU processes in 5-10 seconds vs 44 minutes on your CPU
- **Best of Both**: Fast end-to-end without uploading large audio files

---

## 📋 Quick Start

### Step 1: Run ASR Locally (30 seconds)

```powershell
cd D:\Downloads\HealthTech\mvp-healthtech

# Edit local_asr_only.py - set your audio file path
# AUDIO_FILE = "test1.mp3"  # or your file

python local_asr_only.py
```

**Output:**
- Transcription text printed to console
- Saved to `test1_transcription.txt`
- Saved to `test1_transcription.json`

**Copy the Arabic transcription text from the output!**

---

### Step 2: Paste Text into Kaggle Script

1. Open `kaggle_llm_only.py`
2. Find the `INPUT_TEXT` variable (line ~31)
3. **Paste your transcription:**

```python
INPUT_TEXT = """
المريض يشكو من صداع مستمر منذ ثلاثة أيام...
[YOUR TRANSCRIPTION HERE]
"""
```

---

### Step 3: Run on Kaggle (15-20 seconds)

1. **Upload to Kaggle:**
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"
   - Click "File" → "Upload" → Select `kaggle_llm_only.py`
   - Or copy/paste the code into a Python notebook

2. **Enable GPU:**
   - Right sidebar → "Accelerator"
   - Select "GPU T4 x2"
   - Click "Save"

3. **Run:**
   - Click "Run All" (or Ctrl+Enter)
   - Wait ~15-20 seconds

4. **Download Results:**
   - Right sidebar → "Output"
   - Download `llm_result.json`

---

## 📊 Performance Comparison

| Method | ASR Time | LLM Time | Total | Automation |
|--------|----------|----------|-------|------------|
| **All Local (CPU)** | 30s | 44 mins | 44.5 mins | ✅ Full |
| **Split (Local ASR + Kaggle LLM)** | 30s | 15s | 45s + manual | ⚠️ Manual |
| **All Kaggle (GPU)** | 50s | 15s | 65s | ✅ Full |

**Recommendation:**
- **Testing LLM output?** Use split method (this guide)
- **Production/batch?** Use all-Kaggle method (KAGGLE_NOTEBOOK.ipynb)

---

## 📁 Files Overview

### Local Files (Run on Your Machine)
- **`local_asr_only.py`**: ASR transcription script
  - Input: Audio file path
  - Output: Transcription text + JSON
  - Time: ~30 seconds

### Kaggle Files (Run on Kaggle GPU)
- **`kaggle_llm_only.py`**: LLM processing script
  - Input: Transcription text (pasted)
  - Output: Corrected text + SOAP note
  - Time: ~15 seconds

### Full Pipeline (Alternative)
- **`KAGGLE_NOTEBOOK.ipynb`**: Complete automated pipeline
  - Input: Audio files uploaded to Kaggle
  - Output: Full results (ASR + LLM)
  - Time: ~65 seconds
  - Automation: Full

---

## 🔧 Troubleshooting

### Local ASR Issues

**Error: "WhisperX not installed"**
```powershell
pip install git+https://github.com/m-bain/whisperx.git
```

**Error: "Audio file not found"**
- Update `AUDIO_FILE` path in `local_asr_only.py`
- Use absolute path: `D:\\Downloads\\HealthTech\\mvp-healthtech\\test1.mp3`

### Kaggle LLM Issues

**Error: "Transformers version mismatch"**
- Add installation cell to notebook:
```python
!pip install -q transformers==4.44.0 accelerate bitsandbytes
```

**Slow processing (20+ mins)**
- Check GPU is enabled: Settings → Accelerator → GPU T4
- Restart kernel and re-run

---

## 💡 Tips

1. **Save transcriptions**: Keep the `*_transcription.json` files for reference
2. **Batch processing**: Run multiple audio files locally, then process all texts on Kaggle
3. **Kaggle quota**: Free tier = 30 hours/week GPU (plenty for LLM processing)
4. **Model caching**: First run downloads models (~8GB), subsequent runs are fast

---

## 🚀 Next Steps

**Once you verify LLM output is good:**
- Switch to full Kaggle pipeline (`KAGGLE_NOTEBOOK.ipynb`)
- Upload audio files as Kaggle dataset
- Run fully automated batch processing
- Download all results at once

**For production use:**
- Consider Azure VM with GPU (when you upgrade account)
- Or continue using Kaggle for free GPU access
