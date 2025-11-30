# ASR Service Warnings & Errors - Explained

## ⚠️ Critical Warnings (Affect Performance)

### 1. **`The attention mask is not set...`**
**Status:** ✅ FIXED

**Impact:** 
- Can cause unexpected behavior in LoRA generation
- May lead to poor quality transcriptions
- Can cause early cutoff in long audio

**Solution Applied:**
```python
attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=DEVICE)
whisper_model_lora.generate(inputs, attention_mask=attention_mask, ...)
```

---

## 🟡 Medium Warnings (Can Be Ignored or Fixed)

### 2. **`TRANSFORMERS_CACHE is deprecated, use HF_HOME instead`**
**Impact:** Low - still works, just deprecated

**Fix:** Already using `HF_HOME` in `.env`, can ignore this warning

---

### 3. **`on_event is deprecated, use lifespan event handlers instead`**
**Impact:** Low - will break in future FastAPI v5

**Fix (Optional):**
```python
# Replace @app.on_event("startup") with:
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_models()
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan)
```

---

### 4. **`torch_dtype is deprecated! Use dtype instead`**
**Impact:** Low - still works, just deprecated

**Fix:** In `app.py` line 135-139, change:
```python
# From:
torch_dtype=torch.float16

# To:
dtype=torch.float16
```

---

### 5. **`gradient_checkpointing to config is deprecated`**
**Impact:** None - this is from the wav2vec2 model config, not our code

**Fix:** Can ignore - it's in the pre-trained model files

---

## 🟢 Info Warnings (Can Ignore Safely)

### 6. **`pkg_resources is deprecated`**
**Impact:** None - internal ctranslate2 warning

**Fix:** Ignore - wait for ctranslate2 to update

---

### 7. **`Module 'speechbrain.pretrained' was deprecated`**
**Impact:** None - automatically redirects to new module

**Fix:** Ignore - SpeechBrain handles this internally

---

### 8. **`Model was trained with pyannote.audio 0.0.1, yours is 3.4.0`**
**Impact:** Low - model still works fine across versions

**Fix:** Ignore - Pyannote models are generally backward compatible

---

### 9. **`Model was trained with torch 1.10.0, yours is 2.7.1`**
**Impact:** Low - PyTorch is backward compatible

**Fix:** Ignore - newer PyTorch versions support older models

---

### 10. **`std(): degrees of freedom is <= 0`**
**Impact:** None - internal Pyannote calculation warning

**Fix:** Ignore - doesn't affect diarization results

---

## 🎯 Priority Actions

### ✅ Already Fixed:
1. Attention mask warning (critical)
2. Max length issue for LoRA (using `max_new_tokens=1024`)

### 🔧 Optional Improvements:
1. Replace `on_event` with `lifespan` (low priority)
2. Change `torch_dtype` to `dtype` (low priority)

### ❌ Can Ignore:
- All other warnings are from dependencies and don't affect functionality

---

## 📊 Current Status After Fixes

**LoRA Generation:**
- ✅ Attention mask set properly
- ✅ `max_new_tokens=1024` for long audio
- ✅ Should generate full 134s transcription now

**Expected Results:**
- No more attention mask warning
- LoRA should transcribe full audio (not just 30s)
- Should see ~274 words instead of ~52 words
- Should have ~19 segments instead of 5 segments

---

## 🔬 Test Again

Restart service and run:
```bash
cd services/asr
python -m uvicorn app:app --host 0.0.0.0 --port 5000

# In another terminal:
python compare_asr_wer.py test1.mp3 reference_test1.txt ar egypt
```

Expected improvement:
- ✅ No attention mask warning
- ✅ Full transcription from LoRA
- 🤞 Better medical term accuracy
