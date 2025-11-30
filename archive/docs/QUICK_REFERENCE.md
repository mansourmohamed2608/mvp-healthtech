# 🎯 Quick Reference Card

## 🚀 One Command to Start Everything

```powershell
cd D:\Downloads\HealthTech\mvp-healthtech
.\start-all.ps1
```

---

## 🌐 URLs to Remember

| Service | URL | What It Does |
|---------|-----|--------------|
| **Frontend** | http://localhost:5173 | Main UI (all pages) |
| **Voice Agent** | http://localhost:5173/voice-agent | Talk to medical AI |
| **Clinical Notes** | http://localhost:5173/features/clinical-notes | Generate SOAP notes |
| **Gateway API** | http://localhost:3001 | Backend API |
| **ASR Metrics** | http://localhost:5000/metrics | Speech recognition stats |
| **LLM Metrics** | http://localhost:5001/metrics | AI response stats |
| **Orchestrator** | http://localhost:5006/metrics | Intent routing stats |

---

## 🧪 Test Commands

```powershell
# Test individual services
python test_asr.py              # Speech recognition
python test_llm.py              # AI responses  
python test_tts.py              # Text-to-speech
python test_soap.py             # SOAP generation
python test_fhir.py             # FHIR export

# Test complete workflow
python test_integration.py      # Full pipeline

# Test Week 2 features
python test_orchestrator.py     # Intent classification
python services/llm/guardrails.py  # Safety policies
```

---

## 📱 What Each Page Does

### ✅ Ready to Use:

1. **Home** (`/`) - Landing page
2. **Features** (`/features`) - Feature overview
3. **Voice Agent** (`/voice-agent`) ⭐ - Real-time voice chat
4. **Clinical Notes** (`/features/clinical-notes`) ⭐ - SOAP generation
5. **Voice Transcription** (`/features/voice-transcription`) - Speech-to-text
6. **SOAP Generation** (`/features/soap-generation`) - Clinical notes
7. **FHIR Integration** (`/features/fhir-integration`) - Health records
8. **Dashboard** (`/dashboard`) - System metrics
9. **About** (`/about`) - Project info
10. **Pricing** (`/pricing`) - Subscription plans
11. **Service Test** (`/test`) - Developer testing

---

## 🎤 Voice Agent - Quick Start

```
1. Open http://localhost:5173/voice-agent
2. Click "Start Conversation"
3. Allow microphone access
4. Speak Arabic: "عندي صداع منذ يومين"
5. Wait for AI response (text + audio)
6. Continue conversation
7. Click "Stop" when done
```

---

## 📝 Clinical Notes - Quick Start

```
1. Open http://localhost:5173/features/clinical-notes
2. Select dialect (Auto recommended)
3. Click "Start Recording" OR "Upload Audio File"
4. Speak consultation in Arabic
5. Click "Stop Recording"
6. Review transcript (edit if needed)
7. Click "Generate SOAP Note"
8. Review SOAP (S, O, A, P sections)
9. Click "Export to FHIR" (optional)
10. Click "Accept" to save
```

---

## 🧪 Test Orchestrator (Intent Classification)

```powershell
python test_orchestrator.py
```

**Expected Results:**
- ✅ 25 test cases
- ✅ 90%+ pass rate
- ✅ <50ms average latency

**What it tests:**
- Symptom intent ("عندي صداع")
- Emergency intent ("نوبة قلبية")
- Appointment intent ("أريد حجز موعد")
- Prescription intent ("أحتاج دواء")
- Medical history ("عندي حساسية")
- General queries ("مرحبا")

---

## 🛡️ Test Guardrails (Safety)

```powershell
python services/llm/guardrails.py
```

**What it tests:**
- ✅ Medical disclaimers
- 🚨 Emergency detection (13 keywords)
- ❌ Harmful content blocking
- ⏱️ Rate limiting (10 req/min)
- 🔄 Turn limiting (max 20)

---

## 📊 Check Metrics

```powershell
# ASR performance
curl http://localhost:5000/metrics | grep asr_rtf

# LLM performance  
curl http://localhost:5001/metrics | grep llm_first_token

# Orchestrator performance
curl http://localhost:5006/metrics | grep orchestrator
```

**Target Metrics:**
- ASR RTF: ≤0.5 (faster is better)
- LLM First Token: <300ms
- LLM Total: <1.5s
- Orchestrator: <50ms

---

## 🔧 Troubleshooting

### Services won't start?
```powershell
# Check if ports are in use
netstat -ano | findstr :5000

# Kill process
taskkill /PID <PID> /F

# Restart
.\start-all.ps1
```

### Microphone not working?
1. Check browser → Settings → Allow microphone
2. Check Windows → Privacy → Microphone
3. Try Chrome browser
4. Refresh page

### CUDA out of memory?
```powershell
# Check GPU usage
nvidia-smi

# Close other programs using GPU
# Or restart computer
```

### AI response not relevant?
1. Check LLM service is running
2. Try rephrasing in Arabic
3. Check logs in `services/llm/logs/`

---

## 📚 Full Documentation

- **Complete Guide:** `USER_GUIDE.md`
- **Architecture:** `docs/ARCHITECTURE_OVERVIEW.md`
- **Testing:** `docs/TESTING_GUIDE_WEEK1-5.md`
- **Guardrails:** `docs/GUARDRAILS.md`
- **Setup:** `docs/SETUP.md`

---

## ✅ Week 1-4 Checklist

- [x] Week 1: Core services (ASR, LLM, TTS, Gateway)
- [x] Week 2: Metrics (RTF, latency) + Orchestrator + Guardrails
- [x] Week 3: SOAP service + Clinical Notes UI
- [x] Week 4: FHIR integration + Export
- [ ] Week 5: True streaming ASR (next!)

---

**Status:** All features ready for testing! 🎉  
**Next:** Test everything using the commands above.
