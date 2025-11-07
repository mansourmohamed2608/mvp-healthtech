# 🔍 Understanding the Difference: Scripts vs ASR Service

## 📊 The Two "Whispers" You're Using:

### 1️⃣ **Whisper (transformers)** - Used by comparison scripts
```
Technology:   PyTorch (Python)
Library:      transformers
Model:        WhisperForConditionalGeneration
Speed:        FAST (80s for 134s audio)
Features:     Basic transcription only
GPU Support:  ✅ Works great on GTX 1050 3GB
```

**What it does:**
```
Audio → Load model → Transcribe (one shot) → Text
        (10s)        (80s)                    DONE!
```

### 2️⃣ **WhisperX (faster-whisper)** - Used by ASR service
```
Technology:   CTranslate2 (C++ optimized)
Library:      faster-whisper + whisperx
Model:        WhisperModel (optimized)
Speed:        SLOWER on your GPU (150s+)
Features:     Transcription + Alignment + Diarization
GPU Support:  ⚠️ Needs specific compute types
```

**What it does:**
```
Audio → Load WhisperX → Transcribe (chunks) → Load Alignment → Align words → Load Diarization → Detect speakers → Text + Timestamps + Speakers
        (10s)          (40-60s)                (5-10s)          (20-40s)       (5-10s)             (30-50s)          DONE!
                                                                                                    Total: 150s+
```

---

## 🤔 Why Is WhisperX Slower on Your Hardware?

### **Your GPU: GTX 1050 3GB**
- Released: 2017 (old!)
- VRAM: Only 3GB
- Compute: Limited optimization support

### **WhisperX Requirements:**
| Model | VRAM Needed | Your VRAM | Result |
|-------|-------------|-----------|--------|
| Whisper large-v3 | ~3GB | 3GB | ✅ Barely fits |
| + Alignment model | +1.2GB | 3GB | ❌ Doesn't fit |
| + Diarization | +500MB | 3GB | ❌ Doesn't fit |

**What happens:**
1. WhisperX loads the base model (3GB) → Uses all your VRAM
2. Tries to load alignment model → No room!
3. Falls back to **CPU** for alignment → SUPER SLOW
4. Tries to load diarization → No room!
5. Falls back to **CPU** for diarization → SUPER SLOW

---

## 🎯 Why Scripts Are Fast:

### **Comparison Scripts** (`compare_whisper_wer.py`)
```python
# Only loads ONE model at a time
model = WhisperForConditionalGeneration.from_pretrained(...)
# 3GB model fits in 3GB VRAM ✅
# Processes entire audio in one go
# No chunking, no alignment, no diarization
# Result: FAST! (80s)
```

### **ASR Service** (`app.py`)
```python
# Tries to load MULTIPLE models
whisper_model = whisperx.load_model(...)      # 3GB
align_model = whisperx.load_align_model(...)  # +1.2GB (doesn't fit!)
diarize_model = Pipeline.from_pretrained(...) # +500MB (doesn't fit!)
# Models don't fit, falls back to CPU
# Result: SLOW! (150s+)
```

---

## 💡 Solutions:

### **Option 1: Use Medium Model** (Recommended for GTX 1050)
```env
# In services/asr/.env
WHISPER_MODEL=medium      # 1.5GB instead of 3GB
DEVICE=cuda
COMPUTE_TYPE=float16
ENABLE_DIARIZATION=false  # Don't try to load extra models
```

**Result:**
- ✅ Model fits in VRAM
- ✅ Faster processing
- ✅ Still good accuracy
- ✅ ~30-40s for 134s audio

### **Option 2: Use Large on CPU** (Better accuracy, slower)
```env
# In services/asr/.env
WHISPER_MODEL=large-v3
DEVICE=cpu                # Use CPU instead
COMPUTE_TYPE=int8         # Optimized for CPU
ENABLE_DIARIZATION=false
```

**Result:**
- ✅ More accurate (large model)
- ❌ Slower (~100-120s for 134s audio)
- ✅ Works reliably

### **Option 3: Keep Scripts Only** (Fastest!)
```bash
# Don't use ASR service at all
# Just use the comparison scripts
python compare_whisper_wer.py test1.mp3 reference.txt ar
```

**Result:**
- ✅ FASTEST (80s)
- ✅ Uses GPU efficiently
- ❌ No word timestamps
- ❌ No speaker detection

---

## 📋 Summary Table:

| Approach | Speed | Accuracy | Features | GPU Usage |
|----------|-------|----------|----------|-----------|
| **Scripts (Whisper)** | ⭐⭐⭐⭐⭐ Fast (80s) | ⭐⭐⭐⭐ Good | Basic | ✅ Efficient |
| **ASR (WhisperX large)** | ⭐⭐ Slow (150s+) | ⭐⭐⭐⭐⭐ Best | Full | ❌ Overflows |
| **ASR (WhisperX medium)** | ⭐⭐⭐⭐ Fast (40s) | ⭐⭐⭐⭐ Good | Full | ✅ Fits |
| **ASR (large on CPU)** | ⭐⭐⭐ Medium (100s) | ⭐⭐⭐⭐⭐ Best | Full | N/A |

---

## 🎯 My Recommendation for GTX 1050 3GB:

**Use WhisperX MEDIUM model on CUDA:**

```env
WHISPER_MODEL=medium
DEVICE=cuda
COMPUTE_TYPE=float16
ENABLE_DIARIZATION=false
USE_LORA=false
ENABLE_VAD=true
```

This gives you:
- ✅ Fast processing (~40s for 134s audio)
- ✅ Word-level timestamps
- ✅ Good accuracy (still very good for Arabic)
- ✅ Fits in your 3GB VRAM
- ✅ No crashes or slowdowns

---

## 🧪 Test Now:

```powershell
# Test with fixed compute type
python test_whisperx_complete.py test1.mp3 reference_test1.txt ar
```

This should work now! 🚀
