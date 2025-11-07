# 🔗 Frontend-Backend Connection Guide

## ✅ Current Status

Your services are running on these ports:
- **ASR** (Speech Recognition): `http://localhost:5000` ✅
- **LLM** (Language Model): `http://localhost:5001`
- **TTS** (Text-to-Speech): `http://localhost:5002`
- **SOAP** (Notes Generator): `http://localhost:5003`
- **FHIR** (EHR Integration): `http://localhost:5004`
- **Gateway** (API Gateway): `http://localhost:3001`
- **Frontend** (Vite): `http://localhost:5173`

---

## 🚀 Quick Start

### Step 1: Start Backend Services (Already Running)
You mentioned ASR is already running on port 5000. Start the others:

```powershell
# Terminal 2 - LLM
cd services/llm
python app.py

# Terminal 3 - TTS
cd services/tts
python app.py

# Terminal 4 - SOAP
cd services/soap
python app.py

# Terminal 5 - FHIR
cd services/fhir
python app.py
```

### Step 2: Start Gateway (Optional)
```powershell
# Terminal 6
.\start-gateway.ps1
# OR
cd gateway
pnpm install
pnpm run start:dev
```

### Step 3: Start Frontend
```powershell
# Terminal 7
.\start-frontend.ps1
# OR
cd frontend-vite
pnpm install
pnpm run dev
```

---

## 🎯 Two Connection Modes

### Mode 1: Via Gateway (Recommended for Production)
```env
# frontend-vite/.env
VITE_USE_DIRECT_SERVICES=false
VITE_API_URL=http://localhost:3001
```

**Flow:** Frontend → Gateway (3001) → Services (5000-5004)

### Mode 2: Direct Services (Better for Testing)
```env
# frontend-vite/.env
VITE_USE_DIRECT_SERVICES=true
VITE_ASR_URL=http://localhost:5000
VITE_LLM_URL=http://localhost:5001
VITE_TTS_URL=http://localhost:5002
VITE_SOAP_URL=http://localhost:5003
VITE_FHIR_URL=http://localhost:5004
```

**Flow:** Frontend → Services directly (5000-5004)

**For your testing, I recommend Mode 2** since you're running services individually!

---

## 🧪 Testing

### 1. Open the Test Page
Once frontend is running, visit:
```
http://localhost:5173/test
```

This page will:
- Check health status of all services
- Run functional tests for each service
- Display connection status and results

### 2. Manual API Tests

**Test ASR:**
```powershell
curl http://localhost:5000/health
```

**Test LLM:**
```powershell
curl http://localhost:5001/health
```

**Test TTS:**
```powershell
curl http://localhost:5002/health
```

**Test SOAP:**
```powershell
curl http://localhost:5003/health
```

**Test FHIR:**
```powershell
curl http://localhost:5004/health
```

---

## 📝 What I Changed

### 1. Updated `frontend-vite/.env`
Added support for both gateway and direct service modes:
```env
VITE_API_URL=http://localhost:3001
VITE_USE_DIRECT_SERVICES=false
VITE_ASR_URL=http://localhost:5000
VITE_LLM_URL=http://localhost:5001
VITE_TTS_URL=http://localhost:5002
VITE_SOAP_URL=http://localhost:5003
VITE_FHIR_URL=http://localhost:5004
```

### 2. Updated `frontend-vite/src/utils/api.ts`
- Added dual-mode support (gateway vs direct)
- Added health check methods for all services
- Routes requests based on configuration

### 3. Created `gateway/.env`
Configured gateway to point to all services:
```env
PORT=3001
ASR_SERVICE_URL=http://localhost:5000
LLM_SERVICE_URL=http://localhost:5001
TTS_SERVICE_URL=http://localhost:5002
SOAP_SERVICE_URL=http://localhost:5003
FHIR_SERVICE_URL=http://localhost:5004
```

### 4. Created Test Page
- `frontend-vite/src/pages/ServiceTest.tsx`
- Route: `/test`
- Features:
  - Health check for all services
  - Functional tests for each service
  - Real-time status display
  - Configuration info display

### 5. Added Helper Scripts
- `start-gateway.ps1` - Start the NestJS gateway
- `start-frontend.ps1` - Start the Vite frontend
- `STARTUP_GUIDE.md` - Comprehensive startup documentation

---

## 🎨 Using in Your Frontend

The API client is already configured and ready to use in any component:

```typescript
import api from '@utils/api';

// Transcribe audio
const result = await api.transcribeAudio(audioBase64, 'call-sid', 'egyptian');

// Generate response
const response = await api.inferMessage('Hello', 'session-123');

// Synthesize speech
const audio = await api.synthesizeSpeech('مرحبا', 'ar-EG-SalmaNeural');

// Create SOAP note
const soap = await api.createSOAPNote({
  transcript: 'Patient transcript...',
  sessionId: 'session-123'
});

// FHIR operations
const patient = await api.createFHIRResource('Patient', patientData);
```

---

## 🐛 Troubleshooting

### Services Not Connecting
1. Check if all services are running (look for success messages in terminals)
2. Verify ports are not in use by other applications
3. Check .env files have correct URLs
4. Test direct service URLs with curl/browser

### CORS Errors
All services already have CORS enabled for localhost. If you see CORS errors:
1. Check service is actually running
2. Verify URL is correct in .env
3. Check browser console for exact error

### Gateway Not Starting
1. Make sure Node.js is installed
2. Run `pnpm install` in gateway directory
3. Check gateway/.env exists
4. Verify no other process is using port 3001

---

## 📊 Service Endpoints Reference

### ASR (Port 5000)
- `GET /health` - Health check
- `POST /transcribe` - Transcribe audio
- `POST /stream` - Stream transcription

### LLM (Port 5001)
- `GET /health` - Health check
- `POST /infer` - Generate response

### TTS (Port 5002)
- `GET /health` - Health check
- `POST /synthesize` - Synthesize speech

### SOAP (Port 5003)
- `GET /health` - Health check
- `POST /generate` - Generate SOAP note
- `GET /notes` - Get all notes

### FHIR (Port 5004)
- `GET /health` - Health check
- `POST /{resourceType}` - Create resource
- `GET /{resourceType}/{id}` - Get resource
- `GET /{resourceType}` - Search resources

### Gateway (Port 3001)
- Proxies all above endpoints with `/asr/`, `/llm/`, etc. prefixes

---

## 🎯 Next Steps

1. **Start remaining services** (LLM, TTS, SOAP, FHIR)
2. **Choose your mode** (Direct or Gateway)
3. **Update .env** accordingly
4. **Start frontend** with `.\start-frontend.ps1`
5. **Visit test page** at http://localhost:5173/test
6. **Run tests** to verify all connections work

---

## 💡 Pro Tips

- Use **Direct Mode** for development/testing (faster, easier debugging)
- Use **Gateway Mode** for production (better security, rate limiting, monitoring)
- Keep all service terminals visible to see logs
- Use the test page to quickly verify all services before coding
- Check service health endpoints if anything fails

---

Need help? Check the logs in each terminal for detailed error messages!
