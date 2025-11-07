# 🎨 Frontend Usage Guide

**Date:** October 30, 2025  
**Gateway Port:** 3001  
**Frontend Port:** 5173

---

## ✅ Quick Start

### 1. **Start All Services** (if not running)

```powershell
# In mvp-healthtech folder
.\start-all.ps1
```

This starts:
- ✅ Gateway on **http://localhost:3001**
- ✅ Frontend on **http://localhost:5173**
- ✅ ASR Service on http://localhost:5000
- ✅ LLM Service on http://localhost:5001
- ✅ TTS Service on http://localhost:5002
- ✅ Orchestrator on http://localhost:5006

---

### 2. **Open Frontend**

```
http://localhost:5173
```

---

## 📱 What You Can Do on Each Page

### 🏠 **Home Page** (`/`)
- Overview of the system
- Quick navigation to all features
- Status indicators

**How to use:**
- Click on any feature card to navigate
- Switch language (AR/EN) using the toggle in Navbar

---

### 🎤 **Voice Agent** (`/voice-agent`)

**What it does:** Real-time voice conversations with AI doctor

**How to use:**
1. Click **"Start Call"** button
2. Allow microphone access when prompted
3. Speak your symptoms (Arabic or English)
4. AI will respond with voice + transcript
5. Click **"End Call"** when done

**Features:**
- ✅ Real-time speech recognition (Whisper)
- ✅ AI medical responses (MMed-Llama)
- ✅ Text-to-speech playback
- ✅ Live transcript
- ✅ Mute/unmute button

**Current Status:**
- ⚠️ Requires Twilio setup for real voice calls
- ✅ Can test transcription separately on Service Test page

**Example conversation:**
```
You: "عندي صداع شديد منذ يومين"
AI:  "يمكن علاج الصداع بتناول المسكنات مثل الباراسيتامول..."
```

---

### 📝 **Clinical Notes** (`/features/clinical-notes`)

**What it does:** Generate SOAP notes from conversations

**How to use:**
1. Patient fills conversation form:
   - Chief Complaint: "صداع شديد"
   - History: "منذ يومين"
   - Symptoms: Select from checkboxes
2. Click **"Generate SOAP Note"**
3. AI generates structured SOAP format:
   - **S**ubjective: Patient's description
   - **O**bjective: Vitals, examination
   - **A**ssessment: Diagnosis
   - **P**lan: Treatment recommendations
4. Save as PDF or send to FHIR

**Example Output:**
```
SOAP Note - October 30, 2025

S (Subjective):
- Chief Complaint: صداع شديد
- History: منذ يومين
- Associated symptoms: dizziness

O (Objective):
- Vitals: Normal
- Physical Exam: No findings

A (Assessment):
- Likely tension headache

P (Plan):
- Paracetamol 500mg PRN
- Rest, hydration
- Follow-up if worsens
```

---

### 🧪 **Service Test** (`/service-test`)

**What it does:** Test all backend services individually

**How to use:**

#### **ASR (Speech Recognition)**
1. Upload `.wav` or `.mp3` file
2. Click **"Test ASR"**
3. See transcription result
4. View metrics (RTF, duration)

**Example:**
```
Input: [audio file with "عندي صداع"]
Output: "عندي صداع"
RTF: 0.35 ✅ (fast)
Duration: 1.2s
```

#### **LLM (AI Inference)**
1. Type medical question in Arabic/English
2. Click **"Test LLM"**
3. See AI response
4. View metrics (tokens/s, latency)

**Example:**
```
Input: "ما هو علاج الصداع؟"
Output: "يمكن علاج الصداع بتناول المسكنات..."
Tokens/s: 22.5
Latency: 180ms first token
```

#### **TTS (Text-to-Speech)**
1. Type text in Arabic/English
2. Click **"Test TTS"**
3. Audio player appears
4. Click play to hear voice

**Example:**
```
Input: "مرحبا كيف حالك"
Output: [Audio player with Arabic voice]
```

#### **Orchestrator (Intent Classification)**
1. Type patient message
2. Click **"Test Orchestrator"**
3. See detected intent + confidence

**Example:**
```
Input: "عندي صداع منذ يومين"
Intent: symptom_inquiry (85% confidence)
Entities: {symptoms: ["صداع"], durations: ["يومين"]}
```

#### **Integration (End-to-End)**
1. Upload audio file
2. Click **"Test Full Pipeline"**
3. See: ASR → Orchestrator → LLM → TTS
4. View complete metrics

---

### 🔒 **Guardrails Test** (`/guardrails-test`)

**What it does:** Test content safety filters

**How to use:**
1. Type test message (try safe/unsafe content)
2. Click **"Test Guardrails"**
3. See if message passes filters

**Example:**
```
✅ Safe: "ما هو علاج الصداع؟"
❌ Blocked: "how to hurt someone"
❌ Blocked: "give me drugs without prescription"
```

---

### 📊 **Metrics Dashboard** (`/metrics`)

**What it does:** Real-time performance monitoring

**How to use:**
1. Page loads automatically
2. Shows live metrics for:
   - ASR: RTF, transcription duration
   - LLM: Token generation rate, latency
   - Orchestrator: Intent classification speed
3. Auto-refreshes every 5 seconds

**What to look for:**
- ✅ RTF < 0.5 (fast ASR)
- ✅ First token < 300ms (responsive AI)
- ✅ Tokens/s > 20 (smooth generation)

---

### 🚀 **Quick Demo** (`/quick-demo`)

**What it does:** Interactive demo of voice conversation

**How to use:**
1. Click **"Start Demo"**
2. Simulates voice conversation:
   - Shows ASR transcription
   - Shows LLM response
   - Shows TTS playback
3. Watch metrics update in real-time

---

### 🗂️ **Dashboard** (`/dashboard`)

**What it does:** Overview of all features + status

**How to use:**
- View system health
- See recent conversations
- Access all features from cards
- Monitor service status

---

### 🏥 **FHIR Resources** (`/fhir`)

**What it does:** View/manage FHIR patient records

**How to use:**
1. View list of patients
2. Click patient to see details
3. View encounters, observations
4. Search by patient name/ID

**Current Status:**
- ⚠️ Requires FHIR server running (port 5004)
- ✅ Can view sample data

---

### ⚙️ **Settings** (`/settings`)

**What it does:** Configure system preferences

**How to use:**
- Switch language (AR/EN)
- Change theme (light/dark)
- Configure API endpoints
- Adjust audio settings

---

## 🎯 Complete Testing Flow

### **Week 1-2 Features Test:**

```powershell
# 1. Start all services
.\start-all.ps1

# 2. Open frontend
# http://localhost:5173

# 3. Test ASR
# Go to /service-test
# Upload audio file → Test ASR
# ✅ Should see transcription + RTF < 0.5

# 4. Test LLM
# Go to /service-test
# Type "ما هو علاج الصداع؟" → Test LLM
# ✅ Should see AI response + metrics

# 5. Test Orchestrator
# Go to /service-test
# Type "عندي صداع" → Test Orchestrator
# ✅ Should see intent="symptom" + entities

# 6. Test Guardrails
# Go to /guardrails-test
# Try safe and unsafe messages
# ✅ Should block harmful content

# 7. View Metrics
# Go to /metrics
# ✅ Should see live performance data
```

---

### **Week 3-4 Features Test:**

```powershell
# 1. Test Clinical Notes
# Go to /features/clinical-notes
# Fill patient form
# Click "Generate SOAP Note"
# ✅ Should generate structured SOAP format

# 2. Test Integration
# Go to /service-test → Integration tab
# Upload audio → Test Full Pipeline
# ✅ Should see ASR → LLM → TTS

# 3. Test Voice Agent (if Twilio configured)
# Go to /voice-agent
# Click "Start Call"
# Speak symptoms
# ✅ Should hear AI response
```

---

## 🔧 Troubleshooting

### **Issue 1: Frontend shows "Cannot connect to backend"**

**Solution:**
```powershell
# Check if gateway is running
curl http://localhost:3001/health

# If not, start it
cd gateway
pnpm run start:dev
```

---

### **Issue 2: Voice Agent shows "Device not initialized"**

**Causes:**
- Gateway not running on port 3001
- Twilio credentials not configured

**Solution:**
```powershell
# 1. Make sure gateway is running
curl http://localhost:3001/health

# 2. Check Twilio credentials in gateway/.env
# Look for TWILIO_ACCOUNT_SID, TWILIO_API_KEY, etc.

# 3. If missing, Voice Agent won't work (requires Twilio account)
# You can still test ASR/LLM separately on Service Test page
```

---

### **Issue 3: Services return 500 errors**

**Solution:**
```powershell
# Check if Python services are running
curl http://localhost:5000/health  # ASR
curl http://localhost:5001/health  # LLM
curl http://localhost:5006/health  # Orchestrator

# If not, start them
cd services/asr
python app.py

cd services/llm
python app.py

cd services/llm
python orchestrator.py
```

---

### **Issue 4: Frontend can't find backend (port mismatch)**

**Solution:**
```powershell
# Check frontend .env
cd frontend-vite
cat .env

# Should show:
# VITE_API_URL=http://localhost:3001

# If not, create/update .env:
echo "VITE_API_URL=http://localhost:3001" > .env

# Restart frontend
pnpm run dev
```

---

## 🎯 Configuration Summary

| Component | Port | URL | Status |
|-----------|------|-----|--------|
| **Frontend** | 5173 | http://localhost:5173 | ✅ Running |
| **Gateway** | 3001 | http://localhost:3001 | ✅ Running |
| ASR Service | 5000 | http://localhost:5000 | Should be running |
| LLM Service | 5001 | http://localhost:5001 | Should be running |
| TTS Service | 5002 | http://localhost:5002 | Optional |
| SOAP Service | 5003 | http://localhost:5003 | Optional |
| FHIR Service | 5004 | http://localhost:5004 | Optional |
| Orchestrator | 5006 | http://localhost:5006 | Should be running |

---

## 📚 Related Documentation

- **USER_GUIDE.md** - Complete user guide for all features
- **METRICS_EXPLAINED.md** - How metrics are calculated
- **QUICK_REFERENCE.md** - Quick command reference
- **TROUBLESHOOTING.md** - Common issues and fixes

---

## 🎉 Quick Win Test

**Test this right now:**

1. Open http://localhost:5173
2. Go to **Service Test** page
3. Type in LLM test box: `"ما هو علاج الصداع؟"`
4. Click **"Test LLM"**
5. ✅ You should see AI response in Arabic!

If this works, your system is ready! 🚀
