# Workflow: After Running test_asr.py

## 📋 Overview

You have `test_asr.py` which transcribes audio using your ASR service (localhost:5000). After getting the transcription, you need to process it with the LLM to get corrections, SOAP notes, and speaker identification.

## 🔄 Complete Workflow

```
Step 1: Run ASR            Step 2: Copy Text         Step 3: Process with LLM
┌─────────────────┐        ┌─────────────────┐       ┌─────────────────────┐
│  test_asr.py    │   →    │  Copy the       │   →   │  kaggle_llm_only.py │
│  (Local, 30s)   │        │  transcription  │       │  (Kaggle GPU, 15s)  │
│                 │        │  text output    │       │                     │
└─────────────────┘        └─────────────────┘       └─────────────────────┘
```

---

## Step 1: Run test_asr.py (You Already Know This)

### Start ASR Service First

```powershell
# Terminal 1: Start ASR service
cd d:\Downloads\HealthTech\mvp-healthtech\services\asr
python app.py
```

Wait for: `✓ ASR service running on http://0.0.0.0:5000`

### Run test_asr.py

```powershell
# Terminal 2: Test with your audio file
cd d:\Downloads\HealthTech\mvp-healthtech
python test_asr.py "path\to\your\audio.mp3"
```

### Example Output

```
✅ Transcription successful!

================================================================================
FULL TRANSCRIPT:
================================================================================
المريض يشكو من صداع مستمر منذ ثلاثة أيام مع ارتفاع في درجة الحرارة
الطبيب يفحص المريض ويجد احتقان في الحلق واحمرار في اللوزتين
الضغط طبيعي والنبض منتظم
التشخيص التهاب في الحلق والعلاج المقترح مضاد حيوي وخافض للحرارة

================================================================================
TRANSCRIPT WITH SPEAKERS:
================================================================================

[0.0s - 4.5s] SPEAKER_00:
  المريض يشكو من صداع مستمر منذ ثلاثة أيام مع ارتفاع في درجة الحرارة

[4.5s - 9.2s] SPEAKER_01:
  الطبيب يفحص المريض ويجد احتقان في الحلق واحمرار في اللوزتين
  
[9.2s - 12.8s] SPEAKER_01:
  الضغط طبيعي والنبض منتظم

[12.8s - 17.5s] SPEAKER_01:
  التشخيص التهاب في الحلق والعلاج المقترح مضاد حيوي وخافض للحرارة

================================================================================
SUMMARY:
================================================================================
Duration: 17.50s
Processing time: 28.34s
Speed: 1.62x RTF
Language: ar
Speakers detected: ['SPEAKER_00', 'SPEAKER_01']
Total segments: 4
```

---

## Step 2: Copy the Transcription Text ✂️

### Option A: Copy FULL TRANSCRIPT (Recommended)

Copy everything from the **"FULL TRANSCRIPT"** section:

```
المريض يشكو من صداع مستمر منذ ثلاثة أيام مع ارتفاع في درجة الحرارة
الطبيب يفحص المريض ويجد احتقان في الحلق واحمرار في اللوزتين
الضغط طبيعي والنبض منتظم
التشخيص التهاب في الحلق والعلاج المقترح مضاد حيوي وخافض للحرارة
```

**Use this for**: SOAP generation, correction (tasks: `correct`, `soap`, `full`)

### Option B: Copy TRANSCRIPT WITH SPEAKERS

Copy with speaker labels:

```
SPEAKER_00: المريض يشكو من صداع مستمر منذ ثلاثة أيام مع ارتفاع في درجة الحرارة
SPEAKER_01: الطبيب يفحص المريض ويجد احتقان في الحلق واحمرار في اللوزتين
SPEAKER_01: الضغط طبيعي والنبض منتظم
SPEAKER_01: التشخيص التهاب في الحلق والعلاج المقترح مضاد حيوي وخافض للحرارة
```

**Use this for**: Speaker role identification (task: `identify_speakers`, `full`)

---

## Step 3: Process with LLM on Kaggle 🚀

### Open kaggle_llm_only.py

1. Open the file: `d:\Downloads\HealthTech\mvp-healthtech\kaggle_llm_only.py`

2. **Choose your task**:

```python
# Line 32-38 in kaggle_llm_only.py

# For basic correction only:
TASK = "correct"

# For SOAP note only:
TASK = "soap"

# For speaker identification only:
TASK = "identify_speakers"

# For everything (correction + SOAP + speakers):
TASK = "full"  # ← RECOMMENDED
```

3. **Paste the transcription**:

```python
# Line 40-48 in kaggle_llm_only.py

INPUT_TEXT = """
PASTE YOUR TRANSCRIPTION HERE (from Step 2)

Example:
المريض يشكو من صداع مستمر منذ ثلاثة أيام مع ارتفاع في درجة الحرارة
الطبيب يفحص المريض ويجد احتقان في الحلق واحمرار في اللوزتين
"""
```

4. **Save the file**

### Upload to Kaggle

1. Go to [Kaggle.com](https://www.kaggle.com)
2. Click **"New Notebook"** → **"Upload"**
3. Upload `kaggle_llm_only.py`
4. **IMPORTANT**: Enable GPU
   - Click **"Settings"** (top right)
   - Accelerator → Select **"GPU T4"**
   - Click **"Save"**

### Run the Script

Click **"Run All"** or press `Shift + Enter`

**Expected output**:

```
================================================================================
TASK: full
================================================================================

Loading model Henrychur/MMed-Llama-3-8B on cuda...
✅ Model loaded with 4-bit quantization (~5-6GB VRAM)

================================================================================
STEP 1: TRANSCRIPTION CORRECTION
================================================================================
Input length: 187 characters
🤖 Generating correction...
✅ Generated in 6.8s

Corrected text (185 chars):
المريض يشكو من صداع مستمر منذ ثلاثة أيام مع ارتفاع في درجة الحرارة...

================================================================================
STEP 2: SOAP NOTE GENERATION
================================================================================
🤖 Generating SOAP note...
✅ Generated in 13.2s

SOAP Note (456 chars):
S: المريض يشكو من صداع مستمر منذ ثلاثة أيام مع ارتفاع في درجة الحرارة
O: الفحص السريري يظهر احتقان في الحلق واحمرار في اللوزتين. الضغط طبيعي والنبض منتظم
A: التهاب في الحلق (Pharyngitis)
P: مضاد حيوي (أموكسيسيلين 500 ملغ) وخافض للحرارة. متابعة خلال 3-5 أيام

================================================================================
STEP 3: SPEAKER ROLE IDENTIFICATION
================================================================================
🤖 Analyzing speaker roles...
✅ Generated in 15.1s

Identified Roles:
  SPEAKER_00: Patient (confidence: 0.92)
    Reasoning: Describes symptoms and personal experiences
  SPEAKER_01: Doctor (confidence: 0.95)
    Reasoning: Uses medical terminology, performs examination, provides diagnosis

================================================================================
COMPLETE
================================================================================
✅ Total processing time: 37.6s (0.6 mins)
✅ Results saved to: /kaggle/working/result.json

📥 Download from Kaggle Output tab
================================================================================
```

### Download Results

1. In Kaggle, click **"Output"** tab (top right)
2. You'll see `result.json`
3. Click download icon
4. Open the file to see structured results

---

## Step 4: View Your Results 📊

The downloaded `result.json` contains:

```json
{
  "task": "full",
  "input_text": "المريض يشكو من صداع مستمر منذ ثلاثة أيام...",
  "corrected_text": "المريض يشكو من صداع مستمر منذ ثلاثة أيام...",
  "soap_note": "S: المريض يشكو...\nO: الفحص السريري...\nA: التهاب...\nP: مضاد حيوي...",
  "speaker_roles": [
    {
      "speaker_id": "SPEAKER_00",
      "role": "Patient",
      "confidence": 0.92,
      "reasoning": "Describes symptoms and personal experiences"
    },
    {
      "speaker_id": "SPEAKER_01",
      "role": "Doctor",
      "confidence": 0.95,
      "reasoning": "Uses medical terminology, performs examination"
    }
  ],
  "dialect": "egypt",
  "device": "cuda",
  "processing_time_seconds": 37.6,
  "status": "success"
}
```

---

## 🎯 Quick Reference: Task Selection

### Use `TASK = "correct"` when:
- You just want to fix ASR errors
- Testing transcription quality
- **Time**: ~6-8 seconds

### Use `TASK = "soap"` when:
- You need structured clinical notes
- Generating medical documentation
- **Time**: ~12-15 seconds

### Use `TASK = "identify_speakers"` when:
- You want to know who is Doctor/Patient
- Analyzing conversation roles
- **Time**: ~14-16 seconds

### Use `TASK = "full"` when:
- You want everything (RECOMMENDED)
- Complete medical record processing
- **Time**: ~35-40 seconds

### Use `TASK = "chat"` when:
- You have a medical question (not transcription)
- Set `CHAT_MESSAGE` instead of `INPUT_TEXT`
- **Time**: ~8-10 seconds

---

## 💡 Pro Tips

### Tip 1: Save Transcription to File

Instead of copying manually, save the output:

```powershell
python test_asr.py "audio.mp3" > transcription.txt
```

Then open `transcription.txt` and copy the relevant section.

### Tip 2: Batch Processing

Process multiple audio files:

```powershell
# Run ASR on all files
python test_asr.py "audio1.mp3" > trans1.txt
python test_asr.py "audio2.mp3" > trans2.txt
python test_asr.py "audio3.mp3" > trans3.txt

# Then paste each into kaggle_llm_only.py and run
```

### Tip 3: Test Locally First

Before uploading to Kaggle, verify your transcription looks good:

```powershell
# If transcription has obvious errors, adjust ASR settings
python test_asr.py "audio.mp3" "egyptian"  # Try different dialect
```

### Tip 4: Keep ASR Service Running

Don't restart the ASR service for each file:

```powershell
# Terminal 1: Keep this running
cd services\asr
python app.py

# Terminal 2: Run multiple tests
python test_asr.py "audio1.mp3"
python test_asr.py "audio2.mp3"
python test_asr.py "audio3.mp3"
```

---

## 📊 Performance Summary

| Step | Tool | Where | Time | GPU |
|------|------|-------|------|-----|
| 1. ASR | test_asr.py | Local | ~30s | No (CPU) |
| 2. Copy | Manual | - | ~5s | - |
| 3. LLM | kaggle_llm_only.py | Kaggle | ~15-40s | Yes (T4) |
| **TOTAL** | **Full workflow** | **Hybrid** | **~50-75s** | **Mixed** |

**vs Full Local**: Would take ~45 minutes on CPU!

**Speedup**: ~36x faster with this hybrid approach! 🚀

---

## 🔧 Troubleshooting

### Issue: "Connection refused" when running test_asr.py

**Solution**: Start the ASR service first!

```powershell
cd services\asr
python app.py
```

### Issue: test_asr.py shows empty transcription

**Solutions**:
1. Check audio file is valid (not corrupted)
2. Try different dialect: `python test_asr.py "audio.mp3" "levant"`
3. Check audio is Arabic (model trained on Arabic)

### Issue: Transcription has many errors

**Solutions**:
1. Use better quality audio (clear recording, no background noise)
2. Try different WhisperX model (in services/asr/app.py)
3. Process with LLM correction (`TASK = "correct"`) to fix errors

### Issue: Can't paste Arabic text into kaggle_llm_only.py

**Solutions**:
1. Open file in VS Code (better Unicode support)
2. Or use Notepad++ with UTF-8 encoding
3. Avoid Windows Notepad (poor Arabic support)

---

## 📚 Related Documentation

- **`KAGGLE_LLM_COMPLETE_GUIDE.md`** - Detailed LLM task documentation
- **`SPLIT_PROCESSING_GUIDE.md`** - Complete split workflow guide
- **`LLM_COMPLETE_IMPLEMENTATION.md`** - Technical implementation details

---

## ✅ Summary: Your Workflow

```
1. Start ASR service (services/asr/app.py)
   ↓
2. Run: python test_asr.py "audio.mp3"
   ↓
3. Copy the "FULL TRANSCRIPT" section
   ↓
4. Open kaggle_llm_only.py
   ↓
5. Set TASK = "full"
   ↓
6. Paste transcription into INPUT_TEXT
   ↓
7. Upload to Kaggle
   ↓
8. Enable GPU (Settings → GPU T4)
   ↓
9. Run the script
   ↓
10. Download result.json
    ↓
11. Done! ✅
```

**Total time: ~50-75 seconds vs 45 minutes locally!** 🎉
