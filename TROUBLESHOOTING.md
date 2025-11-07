# 🔧 Troubleshooting: Missing Dependencies

## Error: "No module named 'prometheus_client'"

**Problem:** Python package not installed

**Solution:**
```powershell
# Install single package
pip install prometheus_client

# OR install all dependencies at once
pip install -r requirements.txt
```

---

## Complete Setup (First Time Only)

If you haven't run setup before, do this:

```powershell
# Step 1: Run setup script (installs everything)
.\setup.ps1

# Step 2: Start services
.\start-all.ps1
```

---

## Quick Fix for Common Missing Packages

```powershell
# All required packages
pip install prometheus_client
pip install torch transformers peft
pip install fastapi uvicorn
pip install openai-whisper soundfile
pip install edge-tts
pip install redis
pip install colorama
```

---

## Check What's Installed

```powershell
# List installed packages
pip list | Select-String "prometheus|torch|transformers|fastapi"

# Check specific package
pip show prometheus_client
```

---

## If pip install fails:

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Then try again
pip install -r requirements.txt
```

---

## Service-Specific Dependencies

### ASR Service (services/asr/app.py)
```powershell
pip install prometheus_client
pip install openai-whisper
pip install soundfile
pip install jiwer
```

### LLM Service (services/llm/app.py)
```powershell
pip install prometheus_client
pip install torch transformers peft
pip install sentence-transformers faiss-cpu
```

### TTS Service (services/tts/app.py)
```powershell
pip install edge-tts
```

### SOAP Service (services/soap/app.py)
```powershell
pip install fastapi uvicorn
```

### FHIR Service (services/fhir/app.py)
```powershell
pip install fastapi uvicorn
```

### Orchestrator (services/llm/orchestrator.py)
```powershell
pip install prometheus_client
pip install fastapi uvicorn
```

---

## Now Try Starting ASR Again

```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\services\asr
python app.py
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5000
```

---

## Verify It's Working

```powershell
# In another terminal
curl http://localhost:5000/health
```

**Expected Response:**
```json
{"status": "healthy", "service": "ASR"}
```

---

## Full Startup After Installing Dependencies

```powershell
# Go back to project root
cd D:\Downloads\HealthTech\mvp-healthtech

# Start all services
.\start-all.ps1
```

This will open 7 PowerShell windows for:
1. ASR (Port 5000)
2. LLM (Port 5001)
3. TTS (Port 5002)
4. SOAP (Port 5003)
5. FHIR (Port 5004)
6. Orchestrator (Port 5006)
7. Gateway (Port 3001)

Plus frontend in your current window.

---

## Still Having Issues?

**Check Python version:**
```powershell
python --version
# Should be: Python 3.10, 3.11, 3.12, or 3.13
```

**Check if in virtual environment:**
```powershell
# If you're using a venv, activate it first
.\venv\Scripts\Activate.ps1

# Then install packages
pip install -r requirements.txt
```

**Check pip path:**
```powershell
pip --version
# Should match your Python version
```

---

## Next Steps After Fixing

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Start services: `.\start-all.ps1`
3. ✅ Open frontend: http://localhost:5173
4. ✅ Test voice agent: http://localhost:5173/voice-agent
5. ✅ Test clinical notes: http://localhost:5173/features/clinical-notes

See **USER_GUIDE.md** for complete testing instructions! 🚀
