# 🚀 Complete Step-by-Step Guide to Run HealthTech MVP

## 📋 Prerequisites Check

Before starting, verify you have installed:

1. **Python 3.11** - Check version:
   ```powershell
   python --version
   # Should show: Python 3.11.x
   ```

2. **Node.js 18+** - Check version:
   ```powershell
   node --version
   # Should show: v18.x or higher
   ```

3. **pnpm** - Check version:
   ```powershell
   pnpm --version
   # Should show: 8.x or higher
   # If not installed: npm install -g pnpm
   ```

4. **Docker Desktop** - Check if running:
   ```powershell
   docker --version
   # Should show: Docker version 24.x or higher
   # Make sure Docker Desktop is running
   ```

---

## 🔧 Step 1: Install All Dependencies (First Time Only)

### 1.1 Navigate to Project
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech
```

### 1.2 Install Gateway Dependencies
```powershell
cd gateway
pnpm install
cd ..
```
⏱️ Takes ~2-3 minutes

### 1.3 Install Frontend Dependencies
```powershell
cd frontend
pnpm install
cd ..
```
⏱️ Takes ~2-3 minutes

### 1.4 Install ASR Service (Python)
```powershell
cd services\asr
pip install -r requirements.txt
cd ..\..
```
⏱️ Takes ~5-10 minutes (downloads Whisper model ~3GB)

### 1.5 Install LLM Service (Python)
```powershell
cd services\llm
pip install -r requirements.txt
cd ..\..
```
⏱️ Takes ~10-15 minutes (downloads MMed-Llama-3-8B ~8GB)

### 1.6 Install TTS Service (Python)
```powershell
cd services\tts
pip install -r requirements.txt
cd ..\..
```
⏱️ Takes ~1-2 minutes

### 1.7 Install SOAP Service (Python)
```powershell
cd services\soap
pip install -r requirements.txt
cd ..\..
```
⏱️ Takes ~1 minute

### 1.8 Install FHIR Service (Python - Week 5)
```powershell
cd services\fhir
pip install -r requirements.txt
cd ..\..
```
⏱️ Takes ~1 minute

---

## 🚀 Step 2: Start Redis (Required)

Redis is needed for caching and session management.

```powershell
# Check if Redis is already running
docker ps --filter "name=healthtech-redis"

# If not running, start Redis:
docker run -d -p 6379:6379 --name healthtech-redis redis:7-alpine

# Verify it's running:
docker ps
```

✅ You should see `healthtech-redis` in the list

---

## 🎬 Step 3: Start All Services (Manual Method)

You need to open **8 separate PowerShell windows** (one for each service).

### 3.1 Terminal 1: ASR Service (Whisper)
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\services\asr
python app.py
```
⏱️ Takes ~30-60 seconds to load Whisper model  
🌐 Runs on: http://localhost:5000  
✅ Wait until you see: `Uvicorn running on http://0.0.0.0:5000`

---

### 3.2 Terminal 2: LLM Service (MMed-Llama-3-8B)
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\services\llm
python app.py
```
⏱️ Takes ~5-10 minutes FIRST TIME (downloads 8GB model)  
⏱️ Takes ~2-3 minutes subsequent times (loads from cache)  
🌐 Runs on: http://localhost:5001  
✅ Wait until you see: `Uvicorn running on http://0.0.0.0:5001`

**⚠️ IMPORTANT**: This is the slowest service to start. Let it finish before testing!

---

### 3.3 Terminal 3: TTS Service (edge-tts)
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\services\tts
python app.py
```
⏱️ Takes ~5-10 seconds  
🌐 Runs on: http://localhost:5002  
✅ Wait until you see: `Uvicorn running on http://0.0.0.0:5002`

---

### 3.4 Terminal 4: SOAP Generator Service
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\services\soap
python app.py
```
⏱️ Takes ~5-10 seconds  
🌐 Runs on: http://localhost:5003  
✅ Wait until you see: `Uvicorn running on http://0.0.0.0:5003`

---

### 3.5 Terminal 5: FHIR Service (Week 5)
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\services\fhir
python app.py
```
⏱️ Takes ~5-10 seconds  
🌐 Runs on: http://localhost:5004  
✅ Wait until you see: `Uvicorn running on http://0.0.0.0:5004`

---

### 3.6 Terminal 6: Gateway (NestJS)
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\gateway
pnpm start:dev
```
⏱️ Takes ~30-60 seconds  
🌐 Runs on: http://localhost:3000  
✅ Wait until you see: `Nest application successfully started`

---

### 3.7 Terminal 7: Frontend (Next.js)
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\frontend
pnpm dev
```
⏱️ Takes ~10-20 seconds  
🌐 Runs on: http://localhost:3001  
✅ Wait until you see: `Ready in 5s` or similar

---

## ✅ Step 4: Verify All Services Are Running

Open a **NEW PowerShell window** and run these tests:

### 4.1 Test ASR Service
```powershell
curl http://localhost:5000/health
```
✅ Should return: `{"status":"healthy"}`

### 4.2 Test LLM Service
```powershell
curl http://localhost:5001/health
```
✅ Should return: `{"status":"healthy"}`

### 4.3 Test TTS Service
```powershell
curl http://localhost:5002/health
```
✅ Should return: `{"status":"healthy"}`

### 4.4 Test SOAP Service
```powershell
curl http://localhost:5003/health
```
✅ Should return: `{"status":"healthy"}`

### 4.5 Test FHIR Service
```powershell
curl http://localhost:5004/health
```
✅ Should return: `{"status":"healthy"}`

### 4.6 Test Gateway
```powershell
curl http://localhost:3000/health
```
✅ Should return JSON with `"status":"ok"`

### 4.7 Test Frontend
Open browser and go to:
```
http://localhost:3001
```
✅ Should load the home page

---

## 🎯 Step 5: Access Your Application

### Main Application Pages:

1. **Voice Client** (Week 1-3 Implementation)
   ```
   http://localhost:3001/voice
   ```
   - Test voice calls
   - Record audio
   - Real-time transcription
   - TTS responses

2. **Clinical Notes UI** (Week 4 Implementation)
   ```
   http://localhost:3001/clinical-notes
   ```
   - Record patient consultations
   - Generate SOAP notes
   - View metrics dashboard
   - Save to EHR (FHIR)
   - Select dialect (Egyptian/Levantine/Gulf)

### API Endpoints:

3. **Gateway API**
   ```
   http://localhost:3000/api
   ```

4. **Metrics Dashboard** (Week 5 Implementation)
   - Go to Clinical Notes page
   - Click "Show Metrics" button
   - View acceptance rate, edit distance, review time

---

## 🧪 Step 6: Test the Full Workflow

### Test 1: Voice Client
1. Open http://localhost:3001/voice
2. Click "Start Call"
3. Allow microphone access
4. Speak in Arabic: "مرحبا، أريد حجز موعد"
5. ✅ Should see transcription appear
6. ✅ Should hear TTS response

### Test 2: Clinical Notes
1. Open http://localhost:3001/clinical-notes
2. Select dialect: "كشف تلقائي" (auto-detect)
3. Click "Record Audio" or upload audio file
4. Speak a medical consultation in Arabic
5. Click "Generate SOAP Note"
6. ✅ Should see SOAP note with 4 sections (S/O/A/P)
7. Click "Show Metrics" to see dashboard
8. Edit the SOAP note if needed
9. Click "Accept & Save"
10. ✅ Metrics should update

### Test 3: FHIR Writeback (Week 5)
1. After generating SOAP note
2. Click "Save to EHR"
3. ✅ Should see success message with document ID
4. (Note: Currently in mock mode unless you have real FHIR server)

---

## 🛑 Step 7: Stop All Services

### Stop Services (Easy Method):
1. Go to each PowerShell window (1-7)
2. Press `Ctrl+C` in each window
3. Type `exit` to close

### Stop Redis:
```powershell
docker stop healthtech-redis
```

### To Remove Redis Completely (Optional):
```powershell
docker rm healthtech-redis
```

---

## 🔧 Troubleshooting

### Problem 1: "Port already in use"
**Solution**: Another service is using the port
```powershell
# Find process using port 3000 (example)
netstat -ano | findstr :3000

# Kill process by PID
taskkill /PID <PID_NUMBER> /F
```

### Problem 2: "Module not found" (Python)
**Solution**: Install missing package
```powershell
pip install <package-name>
```

### Problem 3: "pnpm: command not found"
**Solution**: Install pnpm
```powershell
npm install -g pnpm
```

### Problem 4: LLM Service takes too long
**Solution**: First time downloads 8GB model
- Wait 5-10 minutes for download
- Models cached in: `~/.cache/huggingface/`
- Subsequent starts are faster (~2-3 min)

### Problem 5: "Docker daemon not running"
**Solution**: Start Docker Desktop
- Open Docker Desktop application
- Wait for it to fully start
- Try command again

### Problem 6: Out of Memory
**Solution**: Close other applications
- LLM service needs ~8GB RAM
- ASR service needs ~2GB RAM
- Total system should have 16GB+ RAM

### Problem 7: GPU Not Detected
**Solution**: GTX 1050 should work
```powershell
# Check CUDA
nvidia-smi

# If not working, services will use CPU (slower but works)
```

---

## 📊 Expected Resource Usage

| Service | CPU | RAM | GPU VRAM | Startup Time |
|---------|-----|-----|----------|--------------|
| ASR | 10-20% | 2GB | 1.5GB | 30-60s |
| LLM | 20-30% | 8GB | 3.8GB | 2-10min |
| TTS | 5-10% | 100MB | 0GB | 5-10s |
| SOAP | 5-10% | 100MB | 0GB | 5-10s |
| FHIR | 5-10% | 100MB | 0GB | 5-10s |
| Gateway | 10-15% | 200MB | 0GB | 30-60s |
| Frontend | 10-15% | 300MB | 0GB | 10-20s |
| Redis | 1-2% | 50MB | 0GB | 5s |
| **TOTAL** | **60-90%** | **~11GB** | **~5.3GB** | **First: ~10min / Later: ~3min** |

**Your GTX 1050 4GB**: ⚠️ Might run out of VRAM  
**Solution**: Use 8-bit quantization (already enabled in code)

---

## 📝 Summary of All Commands

### One-Time Setup (First Time):
```powershell
# 1. Gateway
cd gateway; pnpm install; cd ..

# 2. Frontend
cd frontend; pnpm install; cd ..

# 3. ASR
cd services\asr; pip install -r requirements.txt; cd ..\..

# 4. LLM
cd services\llm; pip install -r requirements.txt; cd ..\..

# 5. TTS
cd services\tts; pip install -r requirements.txt; cd ..\..

# 6. SOAP
cd services\soap; pip install -r requirements.txt; cd ..\..

# 7. FHIR
cd services\fhir; pip install -r requirements.txt; cd ..\..

# 8. Start Redis
docker run -d -p 6379:6379 --name healthtech-redis redis:7-alpine
```

### Every Time You Want to Run:
```powershell
# Terminal 1: ASR
cd D:\Downloads\HealthTech\mvp-healthtech\services\asr; python app.py

# Terminal 2: LLM (wait for this!)
cd D:\Downloads\HealthTech\mvp-healthtech\services\llm; python app.py

# Terminal 3: TTS
cd D:\Downloads\HealthTech\mvp-healthtech\services\tts; python app.py

# Terminal 4: SOAP
cd D:\Downloads\HealthTech\mvp-healthtech\services\soap; python app.py

# Terminal 5: FHIR
cd D:\Downloads\HealthTech\mvp-healthtech\services\fhir; python app.py

# Terminal 6: Gateway
cd D:\Downloads\HealthTech\mvp-healthtech\gateway; pnpm start:dev

# Terminal 7: Frontend
cd D:\Downloads\HealthTech\mvp-healthtech\frontend; pnpm dev
```

### Access URLs:
- Voice Client: http://localhost:3001/voice
- Clinical Notes: http://localhost:3001/clinical-notes
- Gateway API: http://localhost:3000

---

## 🎉 Success Checklist

- [ ] All 7 services started without errors
- [ ] All health checks return "healthy"
- [ ] Voice client loads and can record audio
- [ ] Clinical notes page loads
- [ ] Can generate SOAP notes
- [ ] Metrics dashboard shows data
- [ ] No console errors in browser
- [ ] Total time: ~10 minutes first run, ~3 minutes subsequent runs

**You're ready to test! 🚀**
