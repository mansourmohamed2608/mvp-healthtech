# Week 1-5 Testing Guide
## Testing Environment Comparison

### **RECOMMENDATION: Start LOCAL, then KAGGLE for training** ✅

---

## 🏆 Best Testing Strategy:

### **Phase 1: Local Testing (Week 1-4 Features)** ⭐ RECOMMENDED FIRST
**Best for**: E2E workflow, debugging, UI testing

```powershell
# Start all services locally
cd D:\Downloads\HealthTech\mvp-healthtech
.\start-all.ps1

# Services will run on:
# - Frontend: http://localhost:3000
# - Gateway: http://localhost:3000/api
# - Voice Client: http://localhost:3001/voice
# - ASR: http://localhost:5000
# - LLM: http://localhost:5001
# - TTS: http://localhost:5002
# - SOAP: http://localhost:5003
# - FHIR: http://localhost:5004
```

**Why Local First:**
- ✅ **Full UI testing** (voice client, clinical notes, metrics dashboard)
- ✅ **Real-time debugging** (inspect logs, network requests)
- ✅ **No internet dependency** (works offline)
- ✅ **Fast iteration** (edit code → refresh)
- ✅ **GTX 1050 4GB sufficient** for inference (already confirmed)
- ✅ **Test all 7 microservices** together

**What to Test Locally:**
1. ✅ Voice client (Twilio WebRTC) → ASR → LLM → TTS loop
2. ✅ Clinical notes recording → SOAP generation
3. ✅ Metrics dashboard (acceptance rate, edit distance)
4. ✅ RAG integration (few-shot examples, FAQs)
5. ✅ FHIR writeback (mock mode)
6. ✅ Dialect selector UI (manual selection)

**Limitations:**
- ❌ Cannot train LoRA adapters (needs more VRAM for training, not inference)
- ❌ No dialect auto-detection testing (needs trained adapters)
- ⚠️ Limited to ~2-3 concurrent users (local hardware)

---

### **Phase 2: Kaggle (LoRA Training ONLY)** ⭐ RECOMMENDED FOR TRAINING
**Best for**: Training dialect-specific adapters

**Why Kaggle:**
- ✅ **Free GPU**: Tesla T4 (16GB VRAM) or P100 (16GB)
- ✅ **30-40 hours/week** free GPU time
- ✅ **Pre-installed**: PyTorch, transformers, PEFT
- ✅ **Perfect for training**: 2-3 hours per dialect adapter
- ✅ **Upload datasets**: Can upload your audio data
- ✅ **Download adapters**: Export trained LoRA weights

**Setup:**
```python
# Kaggle Notebook Steps:
1. Create new notebook
2. Enable GPU (Settings → Accelerator → GPU T4 x2)
3. Upload your code:
   - train_dialect_lora.py
   - dialect_adapter.py
4. Upload datasets:
   - data/dialects/egyptian/audio/*.wav
   - data/dialects/egyptian/metadata.csv
5. Install dependencies:
   !pip install transformers peft datasets jiwer soundfile
6. Run training:
   !python train_dialect_lora.py --dialect egyptian --epochs 5
7. Download adapters:
   - lora_ckpt/egy/adapter_config.json
   - lora_ckpt/egy/adapter_model.bin
8. Copy to local: D:\Downloads\HealthTech\mvp-healthtech\services\asr\lora_ckpt\
```

**Kaggle Training Time:**
- Egyptian adapter: ~2-3 hours (T4 GPU)
- Levantine adapter: ~2-3 hours
- Gulf adapter: ~2-3 hours
- **Total: ~6-9 hours** (fits in free quota)

**What to Train on Kaggle:**
1. ✅ Egyptian dialect LoRA adapter
2. ✅ Levantine dialect LoRA adapter
3. ✅ Gulf dialect LoRA adapter
4. ✅ Evaluate WER improvements

**Download & Test Locally:**
After training, download adapters and test locally:
```powershell
# Copy adapters to local
# Then test dialect auto-detection locally
curl -X POST http://localhost:5000/transcribe \
  -F "audio=@test_egyptian.wav" \
  -F "auto_detect=true"
```

---

### **Phase 3: Free Online Hosting (Optional)** ⚠️ NOT RECOMMENDED YET

**Why NOT recommended for Weeks 1-5:**
- ❌ **Requires GPU**: Free tiers (Render, Railway, Fly.io) are CPU-only
- ❌ **Large models**: MMed-Llama-3-8B (8GB) + Whisper-large-v2 (3GB) = too big
- ❌ **Memory limits**: Free tiers have 512MB-2GB RAM limits
- ❌ **Cold starts**: Models load slowly (30s+)
- ❌ **No WebRTC**: Voice client needs real-time audio (Twilio webhook)

**Free Options (If you simplify):**
1. **Hugging Face Spaces** (Free GPU for students)
   - ✅ 16GB T4 GPU
   - ✅ Good for gradio/streamlit demos
   - ❌ Limited to single app (not 7 microservices)

2. **Google Colab** (Similar to Kaggle)
   - ✅ Free T4 GPU
   - ✅ Good for training
   - ❌ Sessions timeout after 12 hours
   - ❌ Not persistent hosting

3. **Render/Railway** (Free tier)
   - ✅ Good for frontend + gateway
   - ❌ CPU-only (no GPU)
   - ❌ Cannot run LLM/ASR models

**When to use online hosting:**
- **Week 7+**: After pilot testing
- **Simplified version**: Deploy frontend + gateway only
- **External API**: Use OpenAI Whisper API instead of local ASR
- **Paid tier**: Railway Pro ($20/mo) or Render Team ($7/mo)

---

## 📋 Recommended Testing Workflow:

### **Step 1: Local E2E Testing (2-3 days)** ✅
```powershell
# Day 1: Setup & Basic Testing
cd D:\Downloads\HealthTech\mvp-healthtech
.\start-all.ps1

# Test services individually
curl http://localhost:5000/health  # ASR
curl http://localhost:5001/health  # LLM
curl http://localhost:5002/health  # TTS

# Test voice client
# Open browser: http://localhost:3001/voice
# Record audio → Check transcription

# Day 2: Clinical Notes Testing
# Open browser: http://localhost:3000/clinical-notes
# Record audio → Generate SOAP → Test metrics dashboard
# Test FHIR writeback (mock mode)
# Test dialect selector (manual selection)

# Day 3: Integration Testing
# Test full workflow 10+ times
# Check metrics dashboard updates
# Verify RAG responses (ask FAQ questions)
# Test all error cases
```

**Expected Results:**
- ✅ All services start successfully
- ✅ Voice client records and transcribes
- ✅ SOAP notes generated with 4 sections
- ✅ Metrics dashboard shows data
- ✅ RAG provides context-aware responses
- ⚠️ Dialect auto-detection will NOT work (no trained adapters yet)

---

### **Step 2: Kaggle Training (1 week)** ✅
```python
# Week 1: Collect datasets
# - Record 10+ hours Egyptian audio (or download public datasets)
# - Record 10+ hours Levantine audio
# - Record 10+ hours Gulf audio
# - Create metadata.csv for each

# Week 2: Train on Kaggle
# Day 1: Upload datasets to Kaggle
# Day 2: Train Egyptian adapter (2-3 hours)
# Day 3: Train Levantine adapter (2-3 hours)
# Day 4: Train Gulf adapter (2-3 hours)
# Day 5: Evaluate WER improvements
# Day 6: Download adapters and integrate locally
# Day 7: Test dialect auto-detection locally
```

**Expected Results:**
- ✅ 3 trained LoRA adapters (egy, lev, gulf)
- ✅ WER improvement: 18% → 12.5% (Egyptian)
- ✅ Dialect auto-detection works locally
- ✅ Quality report shows >70% intent accuracy

---

### **Step 3: Local Testing with Trained Adapters (2 days)** ✅
```powershell
# After copying adapters from Kaggle
cd D:\Downloads\HealthTech\mvp-healthtech\services\asr
# Verify adapters exist:
ls lora_ckpt/egy/
ls lora_ckpt/lev/
ls lora_ckpt/gulf/

# Restart ASR service
python app.py

# Test auto-detection
# Frontend: Select "كشف تلقائي" (auto-detect)
# Record Egyptian audio → Should detect "egyptian"
# Record Levantine audio → Should detect "levantine"
```

---

## 🎯 Final Recommendation:

| Phase | Environment | Duration | Purpose |
|-------|-------------|----------|---------|
| **Phase 1** | 💻 **Local** | 2-3 days | E2E testing, UI testing, debugging |
| **Phase 2** | 📊 **Kaggle** | 1 week | Train LoRA adapters (GPU needed) |
| **Phase 3** | 💻 **Local** | 2 days | Test with trained adapters |
| **Phase 4** | ☁️ **Online** | Week 7+ | Pilot deployment (paid tier) |

**For Weeks 1-5 testing:**
1. ✅ **Start local** (test everything except dialect training)
2. ✅ **Use Kaggle** (train dialect adapters only)
3. ✅ **Back to local** (test with trained adapters)
4. ❌ **Skip online hosting** (not feasible for free tier with GPU models)

---

## 💡 Quick Start Commands:

```powershell
# LOCAL TESTING (START HERE)
cd D:\Downloads\HealthTech\mvp-healthtech

# Install all dependencies (first time only)
cd frontend; pnpm install; cd ..
cd gateway; pnpm install; cd ..
cd services/asr; pip install -r requirements.txt; cd ../..
cd services/llm; pip install -r requirements.txt; cd ../..
cd services/tts; pip install -r requirements.txt; cd ../..
cd services/soap; pip install -r requirements.txt; cd ../..
cd services/fhir; pip install -r requirements.txt; cd ../..

# Start all services
.\start-all.ps1

# Open in browser
# Voice: http://localhost:3001/voice
# Clinical Notes: http://localhost:3000/clinical-notes
```

Need help with any specific testing phase? 🚀
