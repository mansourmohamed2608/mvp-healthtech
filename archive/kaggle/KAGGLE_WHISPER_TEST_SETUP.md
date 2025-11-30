# 🧪 Kaggle Whisper vs LoRA Test - Complete Setup Guide

## 📋 Prerequisites

### 1. Kaggle Account
- Free account at [kaggle.com](https://kaggle.com)
- **GPU Quota**: 30 hours/week (free tier)
- **Internet**: Must be enabled

### 2. Test Materials Needed
- ✅ **2-3 real Arabic medical audio files** (.mp3, .m4a, .wav)
- ✅ **Your LoRA adapters** folder (`lora_ckpt/`)
- ⚠️ **NOT synthetic TTS audio** - must be real human speech!

---

## 🚀 Step-by-Step Setup

### Step 1: Upload LoRA Adapters to Kaggle

1. **Zip your LoRA folder locally:**
   ```powershell
   # In your project root
   cd services/asr
   Compress-Archive -Path lora_ckpt -DestinationPath whisper-lora.zip
   ```

2. **Create Kaggle Dataset:**
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload `whisper-lora.zip`
   - Title: "Whisper LoRA Adapters"
   - Click "Create"
   - **Note the dataset URL** (e.g., `username/whisper-lora-adapters`)

### Step 2: Upload Test Audio Files

**Option A: As a separate dataset (recommended)**
1. Create another dataset at https://www.kaggle.com/datasets
2. Upload your 2-3 Arabic medical audio files
3. Title: "Arabic Medical Audio Test"
4. Click "Create"

**Option B: Direct upload to notebook**
- You can upload directly when running the notebook
- See "Add Data" → "Upload" in notebook interface

### Step 3: Create Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. **Settings:**
   - Title: "Whisper vs LoRA Test"
   - Accelerator: **GPU T4 x2** ✅
   - Internet: **ON** ✅
   - Persistence: OFF (not needed)

4. **Add Input Datasets:**
   - Click "Add Data" → "Your Datasets"
   - Select your "Whisper LoRA Adapters" dataset
   - Select your "Arabic Medical Audio Test" dataset
   - Click "Add"

5. **Upload the notebook:**
   - File → Import Notebook
   - Upload `KAGGLE_WHISPER_LORA_TEST.ipynb`

### Step 4: Configure Paths in Notebook

In **Cell 2**, update the `LORA_PATH`:

```python
# Find your dataset path - check the input browser on right →
# Should be something like:
LORA_PATH = "/kaggle/input/whisper-lora-adapters/lora_ckpt"
```

**How to find the correct path:**
1. In Kaggle notebook, look at right sidebar "Input"
2. Expand your LoRA dataset
3. Right-click any file → "Copy path"
4. Extract the directory path

### Step 5: Run the Notebook

**Cell by Cell:**

1. **Cell 1** (Install): 
   - Click "Run"
   - Wait ~3-5 minutes
   - Should see "✅ All packages installed!"
   - **Do NOT restart kernel**

2. **Cell 2** (Load Models):
   - Click "Run"
   - Wait ~2-3 minutes (downloads Whisper first time)
   - Should see:
     ```
     Base Whisper: ✅ Loaded
     LoRA Whisper: ✅ Loaded (or ❌ if path wrong)
     ```

3. **Cell 3** (Test):
   - Click "Run"
   - Wait ~1-2 minutes per audio file
   - Watch the comparison output!

---

## 📦 Package Versions & Compatibility

### ✅ Tested Kaggle Configuration:

| Package | Version | Reason |
|---------|---------|--------|
| **transformers** | 4.36.2 | Stable, works with CUDA 12.2 |
| **peft** | 0.7.1 | LoRA loading, no `clear_device_cache` |
| **accelerate** | 0.25.0 | Compatible with peft 0.7.1 |
| **librosa** | 0.10.1 | Audio loading, tested on Kaggle |
| **soundfile** | 0.12.1 | Required by librosa |
| **jiwer** | 3.0.3 | WER calculation |
| **torch** | 2.1.0 (pre-installed) | Kaggle default |
| **CUDA** | 12.2 (pre-installed) | Kaggle default |

### ⚠️ Known Issues & Solutions:

#### Issue 1: LoRA Path Not Found
```
⚠️  LoRA path not found: /kaggle/input/...
```
**Solution:**
- Check right sidebar → Input → Your dataset
- Copy the exact path
- Update `LORA_PATH` in Cell 2

#### Issue 2: Audio Files Not Found
```
Found 0 audio file(s)
```
**Solution:**
- Make sure audio files are uploaded as dataset
- Or manually specify paths in Cell 3:
  ```python
  test_files = [
      '/kaggle/input/my-audio-dataset/test1.mp3',
      '/kaggle/input/my-audio-dataset/test2.m4a',
  ]
  ```

#### Issue 3: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Make sure you selected **GPU T4 x2** (not CPU)
- Kaggle T4 has 15GB VRAM (enough for Whisper large-v3)
- If still failing, try base Whisper only (comment out LoRA loading)

#### Issue 4: Import Error
```
ModuleNotFoundError: No module named 'peft'
```
**Solution:**
- Make sure Cell 1 completed successfully
- Check for "✅ All packages installed!"
- **Do NOT restart kernel** after Cell 1

---

## 🎯 Expected Results

### Scenario 1: LoRA trained on synthetic TTS (Salma's case)
```
⚠️  IDENTICAL TRANSCRIPTIONS
LoRA adapters made NO difference!

❌ RECOMMENDATION: DO NOT USE LORA
Use base Whisper-large-v3 instead!
```

### Scenario 2: LoRA trained on real medical audio
```
✅ DIFFERENT TRANSCRIPTIONS
Word differences:
  Position 3: 'الم' → 'ألم'
  Position 7: 'معده' → 'المعدة'
  
✅ LoRA IS MAKING CHANGES
Next step: Manually review if improvements are meaningful
```

### Scenario 3: LoRA makes it worse
```
Base WER: 15.3%
LoRA WER: 28.7%

❌ LoRA is WORSE by 13.4% WER
```

---

## 📊 Interpreting Results

### If Transcriptions are IDENTICAL:
**Verdict:** ❌ **LoRA is useless**

**Why:**
- LoRA adapters not being applied
- OR trained on completely different domain (TTS vs real speech)
- OR LoRA rank/alpha too small to matter

**Action:**
- Remove LoRA from your ASR service
- Use base Whisper-large-v3
- Save memory and complexity

### If Transcriptions are DIFFERENT:
**Verdict:** 🤔 **Needs manual review**

**Check:**
1. Are medical terms more accurate?
2. Is dialect handling better?
3. Are there fewer hallucinations?
4. Is overall fluency improved?

**If YES** → Keep LoRA ✅  
**If NO** → Remove LoRA ❌

---

## 🔧 Troubleshooting Commands

### Check GPU:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### List Available Paths:
```python
import os
print("Input datasets:")
for root, dirs, files in os.walk('/kaggle/input'):
    print(root)
    if files:
        for f in files[:3]:  # Show first 3 files
            print(f"  - {f}")
```

### Test Audio Loading:
```python
import librosa
audio, sr = librosa.load('/path/to/audio.mp3', sr=16000)
print(f"Loaded {len(audio)/sr:.1f}s of audio")
```

### Check LoRA Adapters:
```python
import os
lora_path = "/kaggle/input/whisper-lora-adapters/lora_ckpt"
if os.path.exists(lora_path):
    files = os.listdir(lora_path)
    print(f"LoRA files found: {files}")
else:
    print("LoRA path not found!")
```

---

## 💡 Pro Tips

1. **Start with 1 audio file** - Test quickly before running all files
2. **Use short clips** (10-30s) - Faster testing, easier to evaluate
3. **Include diverse examples:**
   - Different speakers
   - Different dialects
   - Background noise levels
   - Medical vs general content

4. **Provide ground truth** - Even for 1-2 files, helps calculate WER
5. **Save notebook version** - After successful run, save for future reference

---

## 📝 After Testing

### If LoRA is NOT helpful (expected):

**Update your ASR service:**

1. Edit `services/asr/app.py`:
   ```python
   # Change this line:
   USE_LORA = os.getenv("USE_LORA", "false").lower() == "true"  # Default to false!
   ```

2. Update `.env`:
   ```
   USE_LORA=false
   ```

3. Restart ASR service:
   ```powershell
   cd services/asr
   python app.py
   ```

4. **Benefits:**
   - Faster startup (no LoRA loading)
   - Less memory usage
   - Simpler codebase
   - Same or better accuracy!

### If LoRA IS helpful (surprising):

**Document what improved:**
- Medical terms: Before → After
- Dialect handling: Examples
- WER improvements: Numbers
- Save this in your docs!

---

## 🎓 Why This Test Matters

### The Problem with TTS Training Data:

**Synthetic TTS audio:**
- ❌ Too clean (no background noise)
- ❌ Perfect pronunciation
- ❌ No hesitations, false starts, or corrections
- ❌ Missing dialectal variations
- ❌ Unnatural prosody

**Real medical consultations:**
- ✅ Background noise (clinic environment)
- ✅ Accents and dialects
- ✅ Speech disfluencies
- ✅ Overlapping speech
- ✅ Natural conversation patterns

**If LoRA was trained on TTS, it learned the WRONG patterns!**

---

## 📞 Need Help?

If you get stuck:

1. **Check Kaggle Logs:**
   - Look for error messages in cell outputs
   - Red text indicates errors

2. **Common Error Patterns:**
   - "Path not found" → Update paths
   - "CUDA OOM" → Select GPU, not CPU
   - "Module not found" → Cell 1 didn't complete

3. **Share Your Results:**
   - Copy the "FINAL VERDICT" section
   - Include sample transcriptions
   - Note if identical or different

---

## ⏱️ Time Budget

| Task | Time | Total |
|------|------|-------|
| Upload datasets | 5 min | 5 min |
| Create notebook | 2 min | 7 min |
| Cell 1 (install) | 3-5 min | 10-12 min |
| Cell 2 (load models) | 2-3 min | 12-15 min |
| Cell 3 (test 3 files) | 3-6 min | 15-21 min |
| **Total** | | **~20 minutes** |

**GPU Quota Used:** ~0.5 hours (out of 30h/week free quota)

---

## 🎯 Expected Outcome

**My prediction:** LoRA adapters will show **IDENTICAL** transcriptions to base Whisper.

**This means:**
- ✅ Salma's TTS training data was wrong approach
- ✅ Base Whisper-large-v3 is already excellent
- ✅ You should remove LoRA complexity
- ✅ Simpler = Better!

**Run the test to confirm!** 🚀
