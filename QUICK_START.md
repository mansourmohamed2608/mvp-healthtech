# 🚀 COMPLETE STARTUP - Copy & Paste Commands

## PowerShell Commands (Windows)

### Terminal 1: ASR (Already Running ✅)
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\services\asr
python app.py
```
**Wait for:** ✅ Model with LoRA adapter loaded successfully on cpu!
**Wait for:** Starting ASR service on http://0.0.0.0:5000...

---

### Terminal 2: LLM Service
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\services\llm
python app.py
```
**Wait for:** Loading model...
**Wait for:** Model loaded successfully!

---

### Terminal 3: TTS Service
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\services\tts
python app.py
```
**Wait for:** TTS Service starting...
**Wait for:** ✅ TTS Service ready

---

### Terminal 4: SOAP Service
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\services\soap
python app.py
```
**Wait for:** Application startup complete

---

### Terminal 5: FHIR Service
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\services\fhir
python app.py
```
**Wait for:** Application startup complete

---

### Terminal 6: Gateway (Optional - only if not using direct mode)
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\gateway

# First time only
pnpm install

# Start gateway
pnpm run start:dev
```
**Wait for:** Gateway listening on port 3001

---

### Terminal 7: Frontend
```powershell
cd D:\Downloads\HealthTech\mvp-healthtech\frontend-vite

# First time only
pnpm install

# Start frontend
pnpm run dev
```
**Wait for:** Local: http://localhost:5173/

---

## 🧪 Test Everything

### Open Browser
1. Navigate to: http://localhost:5173/test
2. Click "Check All Services" button
3. All services should show 🟢 online
4. Click individual test buttons to test functionality

### Or use these curl commands:
```powershell
# Test all services
curl http://localhost:5000/health
curl http://localhost:5001/health
curl http://localhost:5002/health
curl http://localhost:5003/health
curl http://localhost:5004/health

# If using gateway
curl http://localhost:3001/health
```

---

## 📝 Configuration for Direct Mode (Recommended)

### Edit: frontend-vite/.env
```env
# Use direct service connections (no gateway needed)
VITE_USE_DIRECT_SERVICES=true

# Gateway URL (not used in direct mode)
VITE_API_URL=http://localhost:3001

# Direct service URLs
VITE_ASR_URL=http://localhost:5000
VITE_LLM_URL=http://localhost:5001
VITE_TTS_URL=http://localhost:5002
VITE_SOAP_URL=http://localhost:5003
VITE_FHIR_URL=http://localhost:5004
```

**With this config, you DON'T need to start the Gateway (Terminal 6)!**

---

## 📝 Configuration for Gateway Mode

### Edit: frontend-vite/.env
```env
# Use gateway
VITE_USE_DIRECT_SERVICES=false

# Gateway URL
VITE_API_URL=http://localhost:3001

# Direct service URLs (not used in gateway mode)
VITE_ASR_URL=http://localhost:5000
VITE_LLM_URL=http://localhost:5001
VITE_TTS_URL=http://localhost:5002
VITE_SOAP_URL=http://localhost:5003
VITE_FHIR_URL=http://localhost:5004
```

### Create/Edit: gateway/.env
```env
PORT=3001
NODE_ENV=development

# Service URLs
ASR_SERVICE_URL=http://localhost:5000
LLM_SERVICE_URL=http://localhost:5001
TTS_SERVICE_URL=http://localhost:5002
SOAP_SERVICE_URL=http://localhost:5003
FHIR_SERVICE_URL=http://localhost:5004
```

**With this config, you MUST start the Gateway (Terminal 6)!**

---

## 🎯 Recommended Approach for Testing

**Use Direct Mode** - It's simpler and better for development:

1. ✅ ASR already running
2. Start LLM, TTS, SOAP, FHIR (Terminals 2-5)
3. Skip Gateway (Terminal 6)
4. Set `VITE_USE_DIRECT_SERVICES=true` in frontend-vite/.env
5. Start Frontend (Terminal 7)
6. Test at http://localhost:5173/test

---

## 🔍 Check What's Running

```powershell
# See all your services
netstat -an | Select-String "5000|5001|5002|5003|5004|3001|5173"
```

Expected output:
```
TCP    0.0.0.0:5000    ...    LISTENING    # ASR
TCP    0.0.0.0:5001    ...    LISTENING    # LLM
TCP    0.0.0.0:5002    ...    LISTENING    # TTS
TCP    0.0.0.0:5003    ...    LISTENING    # SOAP
TCP    0.0.0.0:5004    ...    LISTENING    # FHIR
TCP    0.0.0.0:3001    ...    LISTENING    # Gateway (optional)
TCP    0.0.0.0:5173    ...    LISTENING    # Frontend
```

---

## 🐛 If Something Goes Wrong

### Port Already in Use
```powershell
# Find what's using a port
Get-NetTCPConnection -LocalPort 5000 | Select-Object OwningProcess

# Kill it
Stop-Process -Id <ProcessId> -Force
```

### Service Won't Start
1. Check Python is installed: `python --version`
2. Check dependencies: `pip list | Select-String fastapi`
3. Try running from project root
4. Check for error messages in terminal

### Frontend Can't Connect
1. Make sure services show "ready" in their terminals
2. Check .env file configuration
3. Try direct mode first (easier to debug)
4. Check browser console for errors (F12)

---

## ✅ Success Checklist

- [ ] All 5 backend services running (or 6 with gateway)
- [ ] Frontend running on http://localhost:5173
- [ ] Test page at http://localhost:5173/test works
- [ ] All services show green/online on test page
- [ ] No errors in browser console (F12)
- [ ] No errors in service terminals

---

## 🎉 You're Ready!

Once all services are green on the test page, you can:
- Use the main application at http://localhost:5173
- Test voice transcription features
- Generate SOAP notes
- Integrate with FHIR
- Use all backend services from the UI

**The test page URL:** http://localhost:5173/test

Keep it open to monitor service status while developing!
