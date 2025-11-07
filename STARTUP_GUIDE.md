# 🚀 Complete Startup Guide - All Services

## Service Ports
- **ASR (Speech Recognition)**: http://localhost:5000
- **LLM (Language Model)**: http://localhost:5001
- **TTS (Text-to-Speech)**: http://localhost:5002
- **SOAP (Notes Generator)**: http://localhost:5003
- **FHIR (EHR Integration)**: http://localhost:5004
- **Gateway (API Gateway)**: http://localhost:3001
- **Frontend (Vite)**: http://localhost:5173

---

## 🔧 Option 1: Start All Services (Recommended)

### Terminal 1 - ASR Service
```powershell
cd services/asr
python app.py
```
✅ Wait for: "Starting ASR service on http://0.0.0.0:5000..."

### Terminal 2 - LLM Service
```powershell
cd services/llm
python app.py
```
✅ Wait for: "Model loaded successfully!"

### Terminal 3 - TTS Service
```powershell
cd services/tts
python app.py
```
✅ Wait for: "TTS Service ready"

### Terminal 4 - SOAP Service
```powershell
cd services/soap
python app.py
```
✅ Wait for: "Application startup complete"

### Terminal 5 - FHIR Service
```powershell
cd services/fhir
python app.py
```
✅ Wait for: "Application startup complete"

### Terminal 6 - Gateway (NestJS)
```powershell
cd gateway
pnpm install  # First time only
pnpm run start:dev
```
✅ Wait for: "Gateway listening on port 3001"

### Terminal 7 - Frontend (Vite)
```powershell
cd frontend-vite
pnpm install  # First time only
pnpm run dev
```
✅ Wait for: "Local: http://localhost:5173/"

---

## 🎯 Option 2: Direct Service Connection (No Gateway)

If you want to test services directly without the gateway:

1. Update `frontend-vite/.env`:
```env
VITE_USE_DIRECT_SERVICES=true
```

2. Start only the services you need (ASR, LLM, TTS, SOAP, FHIR)
3. Start the frontend
4. Frontend will connect directly to each service

---

## 🧪 Testing Connections

### Health Check URLs

**Via Gateway:**
- http://localhost:3001/health
- http://localhost:3001/asr/health (Not yet implemented in gateway)
- http://localhost:3001/llm/health (Not yet implemented in gateway)

**Direct Services:**
- http://localhost:5000/health (ASR)
- http://localhost:5001/health (LLM)
- http://localhost:5002/health (TTS)
- http://localhost:5003/health (SOAP)
- http://localhost:5004/health (FHIR)

### Quick Test Commands (PowerShell)

```powershell
# Test ASR
curl http://localhost:5000/health

# Test LLM
curl http://localhost:5001/health

# Test TTS
curl http://localhost:5002/health

# Test SOAP
curl http://localhost:5003/health

# Test FHIR
curl http://localhost:5004/health

# Test Gateway
curl http://localhost:3001/health
```

---

## 📝 Test from Frontend

Once all services are running, navigate to:
- http://localhost:5173/test (Service Test Page - Create this)
- http://localhost:5173 (Main Application)

---

## 🐛 Troubleshooting

### Service Won't Start
- Check if port is already in use
- Verify Python/Node dependencies are installed
- Check .env files are configured

### Connection Refused
- Ensure service is fully started (check terminal output)
- Verify correct port numbers
- Check CORS settings allow localhost

### Gateway Can't Reach Services
- Verify `gateway/.env` has correct service URLs
- Check services are running on expected ports
- Test direct service URLs first

---

## 📊 Service Status Check

Create this test in your frontend to verify all connections:

```typescript
// Test all services
const testServices = async () => {
  const results = {
    asr: await api.checkASRHealth(),
    llm: await api.checkLLMHealth(),
    tts: await api.checkTTSHealth(),
    soap: await api.checkSOAPHealth(),
    fhir: await api.checkFHIRHealth(),
  };
  console.log('Service Status:', results);
};
```

---

## 🚦 Startup Order (Important!)

1. **Start Services First** (ASR, LLM, TTS, SOAP, FHIR)
   - Wait for each to fully initialize

2. **Start Gateway** (if using gateway mode)
   - Verify it can connect to all services

3. **Start Frontend Last**
   - It will connect to gateway or services directly

---

## 💡 Quick Start Script

For convenience, use the existing scripts:

```powershell
# Start all Python services
./start-all.ps1

# Or start manually for better control
./start-manual.ps1
```

But you'll still need to start Gateway and Frontend separately!
