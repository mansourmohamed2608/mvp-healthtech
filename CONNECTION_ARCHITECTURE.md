# 📊 Frontend-Backend Connection Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Vite + React)                    │
│                        http://localhost:5173                         │
│                                                                       │
│  Components: VoiceTranscription, SOAPGeneration, FHIR Integration   │
│  API Client: src/utils/api.ts                                       │
│  Test Page: /test (ServiceTest.tsx)                                 │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      │ Two Connection Modes:
                      │
        ┌─────────────┴──────────────┐
        │                            │
        │ MODE 1: Direct             │ MODE 2: Via Gateway
        │ (VITE_USE_DIRECT_          │ (VITE_USE_DIRECT_
        │  SERVICES=true)            │  SERVICES=false)
        │                            │
        ▼                            ▼
┌───────────────────┐    ┌─────────────────────────┐
│  Direct to        │    │   Gateway (NestJS)      │
│  Services         │    │   http://localhost:3001 │
│                   │    │                         │
│  No middleware    │    │  - Rate limiting        │
│  Faster           │    │  - Authentication       │
│  Better for       │    │  - Request routing      │
│  development      │    │  - Better for prod      │
└────────┬──────────┘    └────────┬────────────────┘
         │                        │
         │                        │ Routes to:
         │                        │ /asr/*
         │                        │ /llm/*
         │                        │ /tts/*
         │                        │ /soap/*
         │                        │ /fhir/*
         │                        │
         └────────┬───────────────┘
                  │
                  │ HTTP Requests
                  │
         ┌────────┴─────────────────────────────────┐
         │                                           │
         │        Backend Microservices              │
         │        (Python FastAPI)                   │
         │                                           │
         └───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬──────────────┬─────────────┐
    │             │              │              │             │
    ▼             ▼              ▼              ▼             ▼
┌────────┐  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  ASR   │  │  LLM   │  │   TTS    │  │   SOAP   │  │   FHIR   │
│ :5000  │  │ :5001  │  │  :5002   │  │  :5003   │  │  :5004   │
│        │  │        │  │          │  │          │  │          │
│ Speech │  │Language│  │   Text   │  │  Note    │  │   EHR    │
│   to   │  │ Model  │  │    to    │  │Generator │  │Integration│
│  Text  │  │        │  │  Speech  │  │          │  │          │
└────────┘  └────────┘  └──────────┘  └──────────┘  └──────────┘
    │            │            │             │             │
    │            │            │             │             │
    ▼            ▼            ▼             ▼             ▼
  Whisper    Llama 3-8B    Coqui TTS    LLM-based    FHIR R4
   LoRA       Medical      or edge-tts   Analysis      API
  Adapter      Model
```

---

## Request Flow Examples

### Example 1: Voice Transcription (ASR)

**Direct Mode:**
```
User speaks → Frontend records audio → Base64 encode
                                           │
                                           ▼
                     POST http://localhost:5000/transcribe
                     { audio: "base64...", callSid: "123" }
                                           │
                                           ▼
                              ASR Service (Whisper LoRA)
                                           │
                                           ▼
                     { text: "Patient complains of..." }
                                           │
                                           ▼
                              Frontend displays text
```

**Gateway Mode:**
```
User speaks → Frontend records audio → Base64 encode
                                           │
                                           ▼
                     POST http://localhost:3001/asr/transcribe
                                           │
                                           ▼
                           Gateway (NestJS) validates & routes
                                           │
                                           ▼
                     POST http://localhost:5000/transcribe
                                           │
                                           ▼
                              ASR Service (Whisper LoRA)
                                           │
                                           ▼
                                      Text result
                                           │
                                           ▼
                            Gateway → Frontend displays
```

---

### Example 2: SOAP Note Generation

```
Transcript → Frontend → API call
                            │
                            ▼
              POST /soap/generate (3003 or 3001/soap)
              { transcript: "...", sessionId: "..." }
                            │
                            ▼
              SOAP Service processes with LLM
                            │
                            ▼
              {
                subjective: "...",
                objective: "...",
                assessment: "...",
                plan: "...",
                icd_codes: ["..."],
                cpt_codes: ["..."]
              }
                            │
                            ▼
              Frontend displays structured note
```

---

## API Endpoints Mapping

### ASR Service (Port 5000)
| Endpoint          | Method | Description              | Gateway Route    |
|-------------------|--------|--------------------------|------------------|
| `/health`         | GET    | Health check             | `/asr/health`    |
| `/transcribe`     | POST   | Batch transcription      | `/asr/transcribe`|
| `/stream`         | POST   | Streaming transcription  | `/asr/stream`    |

### LLM Service (Port 5001)
| Endpoint          | Method | Description              | Gateway Route    |
|-------------------|--------|--------------------------|------------------|
| `/health`         | GET    | Health check             | `/llm/health`    |
| `/infer`          | POST   | Generate response        | `/llm/infer`     |

### TTS Service (Port 5002)
| Endpoint          | Method | Description              | Gateway Route    |
|-------------------|--------|--------------------------|------------------|
| `/health`         | GET    | Health check             | `/tts/health`    |
| `/synthesize`     | POST   | Text to speech           | `/tts/synthesize`|

### SOAP Service (Port 5003)
| Endpoint          | Method | Description              | Gateway Route    |
|-------------------|--------|--------------------------|------------------|
| `/health`         | GET    | Health check             | `/soap/health`   |
| `/generate`       | POST   | Generate SOAP note       | `/soap/generate` |
| `/notes`          | GET    | Get all notes            | `/soap/notes`    |

### FHIR Service (Port 5004)
| Endpoint              | Method | Description          | Gateway Route        |
|-----------------------|--------|----------------------|----------------------|
| `/health`             | GET    | Health check         | `/fhir/health`       |
| `/{resourceType}`     | POST   | Create resource      | `/fhir/{type}`       |
| `/{resourceType}/{id}`| GET    | Get resource         | `/fhir/{type}/{id}`  |
| `/{resourceType}`     | GET    | Search resources     | `/fhir/{type}?...`   |

---

## Port Allocation

```
Port 5000 ────► ASR Service        (Python FastAPI)
Port 5001 ────► LLM Service        (Python FastAPI)
Port 5002 ────► TTS Service        (Python FastAPI)
Port 5003 ────► SOAP Service       (Python FastAPI)
Port 5004 ────► FHIR Service       (Python FastAPI)
Port 3001 ────► Gateway            (NestJS) [Optional]
Port 5173 ────► Frontend           (Vite + React)
```

---

## Configuration Files

### Frontend Configuration
**File:** `frontend-vite/.env`
```env
VITE_USE_DIRECT_SERVICES=true|false    # Choose mode
VITE_API_URL=http://localhost:3001    # Gateway URL
VITE_ASR_URL=http://localhost:5000    # Direct ASR
VITE_LLM_URL=http://localhost:5001    # Direct LLM
VITE_TTS_URL=http://localhost:5002    # Direct TTS
VITE_SOAP_URL=http://localhost:5003   # Direct SOAP
VITE_FHIR_URL=http://localhost:5004   # Direct FHIR
```

### Gateway Configuration
**File:** `gateway/.env`
```env
PORT=3001
ASR_SERVICE_URL=http://localhost:5000
LLM_SERVICE_URL=http://localhost:5001
TTS_SERVICE_URL=http://localhost:5002
SOAP_SERVICE_URL=http://localhost:5003
FHIR_SERVICE_URL=http://localhost:5004
```

---

## Technology Stack

### Frontend
- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite
- **Routing:** React Router
- **Styling:** Tailwind CSS
- **State:** Zustand
- **API Client:** Fetch API (custom wrapper)

### Backend Services
- **Framework:** FastAPI (Python)
- **Server:** Uvicorn
- **CORS:** Enabled for all origins (dev mode)
- **Validation:** Pydantic

### Gateway (Optional)
- **Framework:** NestJS (TypeScript)
- **Features:**
  - Rate limiting (Throttler)
  - CORS management
  - Request validation
  - Service routing

---

## Development vs Production

### Development (Current Setup)
- ✅ Direct service connections
- ✅ All CORS origins allowed
- ✅ Detailed error messages
- ✅ Hot reload enabled
- ✅ Console logging

### Production (Future)
- ✅ Gateway required
- ✅ Restricted CORS origins
- ✅ Generic error messages
- ✅ Rate limiting enabled
- ✅ Structured logging
- ✅ HTTPS only
- ✅ Authentication/Authorization

---

## Files Changed/Created

### Modified Files
1. `frontend-vite/.env` - Added service URLs and mode flag
2. `frontend-vite/src/utils/api.ts` - Dual-mode support
3. `frontend-vite/src/App.tsx` - Added test route
4. `gateway/.env` - Created with service URLs

### New Files
1. `frontend-vite/src/pages/ServiceTest.tsx` - Test page
2. `start-gateway.ps1` - Gateway startup script
3. `start-frontend.ps1` - Frontend startup script
4. `STARTUP_GUIDE.md` - Comprehensive startup guide
5. `FRONTEND_CONNECTION.md` - Connection documentation
6. `CHECKLIST.md` - Pre-flight checklist
7. `QUICK_START.md` - Quick reference commands
8. `CONNECTION_ARCHITECTURE.md` - This file

---

## Next Steps

1. ✅ Start remaining services (LLM, TTS, SOAP, FHIR)
2. ✅ Start frontend with `.\start-frontend.ps1`
3. ✅ Visit http://localhost:5173/test
4. ✅ Check all services are online
5. ✅ Run functional tests
6. ✅ Start building features!

---

## Support & Debugging

### Health Check URLs
- ASR: http://localhost:5000/health
- LLM: http://localhost:5001/health
- TTS: http://localhost:5002/health
- SOAP: http://localhost:5003/health
- FHIR: http://localhost:5004/health
- Gateway: http://localhost:3001/health
- Frontend Test: http://localhost:5173/test

### Logs Location
- **Service logs:** Check each terminal window
- **Frontend logs:** Browser console (F12)
- **Gateway logs:** Gateway terminal window

### Common Issues
See `FRONTEND_CONNECTION.md` troubleshooting section for detailed solutions.
