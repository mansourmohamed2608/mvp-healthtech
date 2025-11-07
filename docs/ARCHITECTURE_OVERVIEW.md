# 🏗️ MVP HealthTech Architecture Overview

**Project**: Arabic-first HealthTech MVP
**Status**: Week 4/14 Complete (29%)
**Target**: Dec 31, 2025
**Performance**: ✅ 70%+ accuracy, ✅ <2s latency, ✅ GTX 1050 4GB

---

## 📊 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  Vite Frontend (React 18)          │  Next.js Frontend (React 19)│
│  Port: 3000                        │  Port: 3001                 │
│  - Demo Page (All services)        │  - Voice Client (WebRTC)    │
│  - Voice Transcription             │  - Clinical Notes           │
│  - About/Pricing/Features          │  - Production UI            │
└──────────────────┬──────────────────┴───────────┬────────────────┘
                   │                              │
                   ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       API GATEWAY (NestJS)                       │
├─────────────────────────────────────────────────────────────────┤
│  Port: 3001 (for Vite) | 3000 (for Next.js)                     │
│                                                                  │
│  Controllers:                                                    │
│  - ASR Controller      → /asr/transcribe, /asr/stream          │
│  - LLM Controller      → /llm/infer                            │
│  - TTS Controller      → /tts/synthesize                       │
│  - SOAP Controller     → /soap/generate                        │
│  - FHIR Controller     → /fhir/{resourceType}                  │
│  - Twilio Controller   → /twilio/voice, /twilio/token         │
│                                                                  │
│  Services:                                                       │
│  - AsrService          → Proxies to ASR microservice           │
│  - LlmService          → Proxies to LLM microservice           │
│  - TtsService          → Proxies to TTS microservice           │
│  - ConversationService → Manages session state                 │
│  - VectorCacheService  → In-memory vector search               │
│  - KvCacheService      → Key-value cache                       │
└──────┬─────┬─────┬─────┬─────┬──────────────────────────────────┘
       │     │     │     │     │
       ▼     ▼     ▼     ▼     ▼
┌──────────────────────────────────────────────────────────────────┐
│                   MICROSERVICES LAYER (Python FastAPI)           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │
│  │  ASR Service   │  │  LLM Service   │  │  TTS Service    │   │
│  │  Port: 5000    │  │  Port: 5001    │  │  Port: 5002     │   │
│  ├────────────────┤  ├────────────────┤  ├─────────────────┤   │
│  │ Model:         │  │ Model:         │  │ Engine:         │   │
│  │ Whisper-large- │  │ MMed-Llama-3-  │  │ edge-tts (free) │   │
│  │ v3 + LoRA      │  │ 8B (4-bit)     │  │ ar-EG-Salma     │   │
│  │                │  │                │  │                 │   │
│  │ VRAM: 2GB      │  │ VRAM: 3.8GB    │  │ VRAM: 0GB       │   │
│  │                │  │                │  │                 │   │
│  │ Features:      │  │ Features:      │  │ Features:       │   │
│  │ - Dialect      │  │ - Medical      │  │ - Natural       │   │
│  │   detection    │  │   intent       │  │   Arabic TTS    │   │
│  │ - 16kHz output │  │ - RAG store    │  │ - 16kHz (Twilio)│   │
│  │ - Base64 I/O   │  │ - Few-shot     │  │ - Base64 audio  │   │
│  └────────────────┘  └────────────────┘  └─────────────────┘   │
│                                                                   │
│  ┌────────────────┐  ┌────────────────┐                         │
│  │ SOAP Generator │  │ FHIR Service   │                         │
│  │  Port: 5003    │  │  Port: 5004    │                         │
│  ├────────────────┤  ├────────────────┤                         │
│  │ Function:      │  │ Function:      │                         │
│  │ - Transcript   │  │ - FHIR R4 API  │                         │
│  │   → SOAP note  │  │ - DocumentRef  │                         │
│  │ - Calls LLM    │  │ - Encounter    │                         │
│  │ - Structured   │  │ - Composition  │                         │
│  │   parsing      │  │                │                         │
│  │                │  │ VRAM: 0GB      │                         │
│  │ VRAM: 0GB      │  │                │                         │
│  └────────────────┘  └────────────────┘                         │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                  │
├──────────────────────────────────────────────────────────────────┤
│  Redis (Port: 6379)                                              │
│  - Session management                                            │
│  - Conversation history                                          │
│  - Key-value cache                                               │
│                                                                   │
│  PostgreSQL (Future)                                             │
│  - Clinical data persistence                                     │
│  - Patient records                                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Request Flow Examples

### Example 1: Voice Transcription (ASR)

```
User (Browser)
  │ 1. Record audio via MediaRecorder API
  │    (audio/webm or audio/wav)
  ▼
Vite Frontend (Demo.tsx or VoiceTranscription.tsx)
  │ 2. Convert Blob → Base64
  │ 3. POST /asr/transcribe { audio: "base64...", dialect: "egyptian" }
  ▼
Gateway (ASR Controller)
  │ 4. Forward to ASR service
  │    POST http://localhost:5000/transcribe
  ▼
ASR Service (services/asr/app.py)
  │ 5. Decode base64 → audio bytes
  │ 6. Load audio with soundfile
  │ 7. Resample to 16kHz
  │ 8. Process with Whisper-large-v3 + LoRA
  │ 9. Generate with forced Arabic decoder
  │ 10. Return { text: "...", dialect: "egyptian", auto_detected: false }
  ▼
Gateway → Frontend
  │ 11. Display transcription
  │ 12. Show audio playback controls
```

**Response Format:**
```json
{
  "text": "أنا أشعر بألم في الصدر",
  "dialect": "egyptian",
  "auto_detected": false
}
```

---

### Example 2: Medical Query (LLM)

```
User (Browser)
  │ 1. Type or speak medical question
  ▼
Frontend
  │ 2. POST /llm/infer { message: "ما هو علاج الضغط؟", sessionId: "session-123" }
  ▼
Gateway (LLM Controller)
  │ 3. Forward to LLM service
  │    POST http://localhost:5001/infer
  ▼
LLM Service (services/llm/app.py)
  │ 4. Build RAG-augmented prompt:
  │    - Get few-shot examples for intent
  │    - Retrieve relevant FAQs from rag_store
  │    - Construct prompt with context
  │ 5. Tokenize and generate with MMed-Llama-3-8B (4-bit)
  │ 6. Extract intent (e.g., "symptom_inquiry", "medication_info")
  │ 7. Parse reply from model output
  │ 8. Return { intent: "medication_info", reply: "..." }
  ▼
Gateway → Frontend
  │ 9. Display AI response
  │ 10. Log intent for analytics
```

**Response Format:**
```json
{
  "intent": "medication_info",
  "reply": "علاج ارتفاع ضغط الدم يشمل تغيير نمط الحياة والأدوية..."
}
```

---

### Example 3: SOAP Note Generation

```
User (Clinical Notes Page)
  │ 1. Record consultation or upload audio
  ▼
Frontend
  │ 2. Transcribe audio via ASR
  │ 3. POST /soap/generate { transcript: "...", sessionId: "...", patientContext: {...} }
  ▼
Gateway (SOAP Controller)
  │ 4. Forward to SOAP service
  │    POST http://localhost:5003/generate
  ▼
SOAP Generator (services/soap/app.py)
  │ 5. Build SOAP extraction prompt with medical instructions
  │ 6. Call LLM service:
  │    POST http://localhost:5001/infer
  ▼
LLM Service
  │ 7. Generate structured medical note
  │ 8. Return formatted text
  ▼
SOAP Generator
  │ 9. Parse sections:
  │    - Subjective (patient complaints)
  │    - Objective (observations)
  │    - Assessment (diagnosis)
  │    - Plan (treatment)
  │ 10. Extract ICD/CPT codes (optional)
  │ 11. Return structured JSON
  ▼
Gateway → Frontend
  │ 12. Display SOAP note in organized format
  │ 13. Allow review and editing
  │ 14. Enable FHIR writeback
```

**Response Format:**
```json
{
  "subjective": "المريض يشكو من ألم في الصدر منذ يومين...",
  "objective": "ضغط الدم: 140/90، نبض: 85، حرارة: 37.2",
  "assessment": "احتمال ارتفاع ضغط الدم",
  "plan": "وصف دواء للضغط، متابعة بعد أسبوع",
  "icd_codes": ["I10"],
  "cpt_codes": ["99213"]
}
```

---

### Example 4: Text-to-Speech (TTS)

```
User (TTS Tab)
  │ 1. Enter Arabic text
  │ 2. Click "Synthesize"
  ▼
Frontend
  │ 3. POST /tts/synthesize { text: "مرحبا بك في العيادة", voice: "ar-EG-SalmaNeural" }
  ▼
Gateway (TTS Controller)
  │ 4. Forward to TTS service
  │    POST http://localhost:5002/synthesize
  ▼
TTS Service (services/tts/app.py)
  │ 5. Use edge-tts (Microsoft cloud, free)
  │ 6. Generate audio with selected voice
  │ 7. Convert to 16kHz WAV (Twilio-compatible)
  │ 8. Encode to base64
  │ 9. Calculate duration and sample rate
  │ 10. Return { audio: "base64...", duration: 3.2, sampleRate: 16000 }
  ▼
Gateway → Frontend
  │ 11. Decode base64 → Blob
  │ 12. Create audio URL with URL.createObjectURL()
  │ 13. Display audio player
```

**Response Format:**
```json
{
  "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "duration": 3.2,
  "sampleRate": 16000
}
```

---

### Example 5: FHIR Writeback

```
User (Clinical Notes)
  │ 1. Review generated SOAP note
  │ 2. Click "Save to EHR"
  ▼
Frontend
  │ 3. POST /fhir/DocumentReference {
  │      soapNote: {...},
  │      patientId: "Patient/123",
  │      practitionerId: "Practitioner/456"
  │    }
  ▼
Gateway (FHIR Controller)
  │ 4. Forward to FHIR service
  │    POST http://localhost:5004/write
  ▼
FHIR Service (services/fhir/app.py)
  │ 5. Authenticate with FHIR server (OAuth2 or API key)
  │ 6. Build FHIR R4 resources:
  │    - Encounter (consultation record)
  │    - DocumentReference (SOAP note)
  │    - Composition (structured sections)
  │ 7. POST to external FHIR server
  │    POST https://fhir-server.com/DocumentReference
  │ 8. Return resource IDs
  ▼
Gateway → Frontend
  │ 9. Show success message with resource ID
  │ 10. Update UI with "Saved" status
```

**Response Format:**
```json
{
  "success": true,
  "documentReferenceId": "DocumentReference/abc123",
  "encounterId": "Encounter/xyz789",
  "error": null
}
```

---

## 🗂️ File Structure & Responsibilities

### Frontend (Vite - Development/Testing)

```
frontend-vite/src/
├── pages/
│   ├── Demo.tsx                    # Multi-tab testing page (ALL services)
│   │   ├── ASR Tab                 → Tests transcription with push-to-talk
│   │   ├── LLM Tab                 → Tests medical Q&A
│   │   ├── SOAP Tab                → Tests note generation
│   │   ├── FHIR Tab                → Tests resource creation
│   │   └── TTS Tab                 → Tests speech synthesis
│   │
│   ├── VoiceTranscription.tsx      # Dedicated ASR page
│   │   ├── Dialect selection       (Egyptian, Levantine, Gulf, MSA)
│   │   ├── Audio recording         (MediaRecorder API)
│   │   └── Audio playback          (HTML5 audio element)
│   │
│   ├── About.tsx                   # Company info, team, mission
│   ├── Pricing.tsx                 # Service tiers, billing
│   └── Features.tsx                # Product capabilities
│
├── utils/
│   └── api.ts                      # REST API client
│       ├── transcribeAudio()       → POST /asr/transcribe
│       ├── inferMessage()          → POST /llm/infer
│       ├── createSOAPNote()        → POST /soap/generate
│       ├── synthesizeSpeech()      → POST /tts/synthesize
│       └── createFHIRResource()    → POST /fhir/{type}
│
└── App.tsx                         # Routes, navigation
```

### Frontend (Next.js - Production)

```
frontend/src/app/
├── voice/
│   └── page.tsx                    # Real-time voice client (Twilio WebRTC)
│       ├── Call controls           (start, mute, end)
│       ├── Transcript display      (ASR stream)
│       └── Connection status       (WebSocket)
│
├── clinical-notes/
│   └── page.tsx                    # Clinical documentation
│       ├── Live recording          (MediaRecorder)
│       ├── File upload             (drag & drop)
│       ├── SOAP review interface   (editable sections)
│       └── FHIR writeback          (save to EHR)
│
└── api/
    ├── twilio/token/route.ts       # JWT token generator (1-hour TTL)
    ├── clinical/transcribe/route.ts# File → ASR proxy
    └── clinical/soap/route.ts      # Transcript → SOAP proxy
```

### Gateway (NestJS)

```
gateway/src/
├── main.ts                         # Bootstrap, CORS, middleware
│
├── app.module.ts                   # Module imports, DI container
│   ├── AuthModule                  → JWT strategy
│   ├── SessionModule               → Redis-backed sessions
│   ├── TwilioModule                → Webhooks, call handling
│   ├── ClinicalModule              → Metrics, quality tracking
│   └── RAGModule                   → Knowledge base
│
├── controllers/
│   ├── asr.controller.ts           # /asr/transcribe, /asr/stream
│   ├── llm.controller.ts           # /llm/infer, /llm/soap
│   ├── tts.controller.ts           # /tts/synthesize
│   ├── soap.controller.ts          # /soap/generate
│   ├── fhir.controller.ts          # /fhir/{resourceType}
│   ├── metrics.controller.ts       # /metrics (API health)
│   └── health.controller.ts        # /health (liveness)
│
├── services/
│   ├── asr.service.ts              # HTTP → ASR microservice
│   ├── llm.service.ts              # HTTP → LLM microservice
│   ├── tts.service.ts              # HTTP → TTS microservice
│   ├── conversation.service.ts     # Session state management
│   ├── vector-cache.service.ts     # In-memory vector search
│   └── kv-cache.service.ts         # Key-value cache
│
└── auth/
    ├── jwt.strategy.ts             # Passport JWT validation
    └── jwt.guard.ts                # Route protection
```

### Microservices (Python FastAPI)

```
services/
├── asr/
│   ├── app.py                      # FastAPI server (port 5000)
│   ├── main.py                     # Whisper model loading
│   ├── dialect_adapter.py          # LoRA adapter switching
│   ├── train_lora_whisper.py       # Fine-tuning script
│   ├── eval_wer.py                 # WER benchmarking
│   └── lora_ckpt/                  # Trained LoRA weights
│
├── llm/
│   ├── app.py                      # FastAPI server (port 5001)
│   ├── orchestrator.py             # Multi-turn conversation
│   ├── rag_store.py                # In-memory RAG (FAQs, few-shot)
│   ├── vector_rag.py               # Vector similarity search
│   └── data/                       # Medical knowledge base
│
├── tts/
│   ├── app.py                      # FastAPI server (port 5002)
│   ├── main.py                     # edge-tts integration
│   └── requirements.txt            # edge-tts, fastapi, uvicorn
│
├── soap/
│   ├── app.py                      # FastAPI server (port 5003)
│   └── requirements.txt            # httpx, pydantic
│
└── fhir/
    ├── app.py                      # FastAPI server (port 5004)
    └── requirements.txt            # httpx, pydantic, fhir.resources
```

---

## 🔑 Key Technologies

### Frontend
- **React 18** (Vite) / **React 19** (Next.js)
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **@tabler/icons-react** - Icons
- **MediaRecorder API** - Audio recording
- **Twilio Voice SDK** - WebRTC calling

### Backend Gateway
- **NestJS** - Node.js framework
- **TypeScript** - Type safety
- **Axios** - HTTP client
- **Passport JWT** - Authentication
- **Redis** - Sessions, cache
- **Throttler** - Rate limiting

### Microservices
- **FastAPI** - Python web framework
- **PyTorch** - Deep learning
- **Transformers** - Hugging Face models
- **PEFT (LoRA)** - Parameter-efficient fine-tuning
- **Soundfile** - Audio I/O
- **edge-tts** - Microsoft TTS (free)

### AI Models
- **Whisper-large-v3** - ASR (2GB VRAM with 8-bit)
- **MMed-Llama-3-8B** - Medical LLM (3.8GB VRAM with 4-bit)
- **LoRA adapters** - Dialect-specific fine-tuning

---

## 📡 Communication Patterns

### 1. REST API (Primary)
- Frontend ↔ Gateway: JSON over HTTP
- Gateway ↔ Microservices: JSON over HTTP
- **Pros**: Simple, stateless, easy debugging
- **Cons**: Higher latency for real-time features

### 2. WebSocket (Voice Client)
- Frontend ↔ Gateway: Twilio WebSocket
- **Use case**: Real-time voice streaming
- **Pros**: Low latency, bidirectional
- **Cons**: More complex state management

### 3. Redis Pub/Sub (Future)
- Microservices ↔ Gateway: Event-driven
- **Use case**: Async job processing, notifications
- **Planned**: Week 7-8

---

## 💾 Data Management

### Session State (Redis)
```typescript
interface Session {
  id: string;
  userId: string;
  callSid?: string;
  conversationHistory: Array<{
    role: 'user' | 'assistant';
    content: string;
    timestamp: number;
  }>;
  context: {
    patientId?: string;
    intent?: string;
    dialect?: string;
  };
  createdAt: Date;
  expiresAt: Date;
}
```

### Cache Strategy
- **Vector Cache** (in-memory): FAQ embeddings, cosine similarity search
- **KV Cache** (Redis): Session data, conversation history, user preferences
- **TTL**: 1 hour for sessions, 24 hours for user data

---

## 🚦 Performance Metrics

### Current Performance (Week 4)
- **ASR WER**: 12.5% ✅ (target: <15%)
- **LLM Intent Accuracy**: 83.8% ✅ (target: >70%)
- **End-to-End Latency**: 1.8s ✅ (target: <2s)
- **GPU Usage**: Peak 3.8GB ✅ (limit: 4GB)

### Latency Breakdown
```
User Speech → ASR → LLM → TTS → User Hears
    0ms        800ms   600ms  400ms    1800ms
             (Whisper) (Llama) (edge)
```

### Throughput
- **ASR**: ~5 concurrent requests (GPU bottleneck)
- **LLM**: ~3 concurrent requests (GPU bottleneck)
- **TTS**: ~20 concurrent requests (cloud-based, no GPU)

---

## 🔐 Security & Auth

### Authentication Flow
1. User logs in → Gateway issues JWT (1-hour expiry)
2. Frontend stores JWT in localStorage
3. All API requests include `Authorization: Bearer <token>`
4. Gateway validates JWT with Passport strategy
5. Protected routes use `@UseGuards(JwtGuard)`

### CORS Configuration
```typescript
allowedOrigins: [
  'http://localhost:3000',  // Vite dev server
  'http://localhost:5173',  // Vite alt port
  'http://localhost:3001',  // Next.js dev server
]
```

### Rate Limiting
- **Throttler**: 50 requests per 60 seconds per IP
- **Applied to**: All gateway endpoints
- **Override**: Premium users (future)

---

## 🧪 Testing Strategy

### Unit Tests
- **Gateway**: Jest (NestJS default)
- **Microservices**: pytest (Python)
- **Frontend**: Vitest (Vite)

### Integration Tests
- **Files**: `test_asr.py`, `test_llm.py`, `test_soap.py`, `test_tts.py`, `test_fhir.py`
- **Strategy**: Mock-free, test against real services
- **Run**: `python test_integration.py`

### End-to-End Tests
- **Demo Page**: Manual testing of all tabs
- **Voice Client**: Real Twilio call testing

---

## 📈 Future Enhancements (Weeks 5-14)

### Week 5-6: EHR Integration
- ✅ FHIR writeback (implemented)
- ⏳ Patient context enrichment
- ⏳ Bi-directional sync

### Week 7-8: Quality & Scale
- ⏳ Model fine-tuning (Egyptian dialect)
- ⏳ Async job queue (Redis)
- ⏳ Horizontal scaling (Docker Swarm)

### Week 9-10: Production Readiness
- ⏳ HIPAA compliance audit
- ⏳ E2E encryption
- ⏳ Audit logging

### Week 11-12: Advanced Features
- ⏳ Multi-language support
- ⏳ Voice biometrics
- ⏳ Clinical decision support

### Week 13-14: Launch
- ⏳ Beta testing with 10 clinics
- ⏳ Performance optimization
- ⏳ Documentation & training

---

## 🔧 Development Commands

### Start All Services (PowerShell)
```powershell
.\start-all.ps1
```
This starts:
1. Redis (Docker)
2. ASR service (port 5000)
3. LLM service (port 5001)
4. TTS service (port 5002)
5. SOAP service (port 5003)
6. FHIR service (port 5004)
7. Gateway (port 3001)
8. Frontend (port 3000 or 3001)

### Individual Service Startup
```powershell
# Redis
docker run -d -p 6379:6379 --name healthtech-redis redis:7-alpine

# ASR
cd services\asr
python app.py

# LLM
cd services\llm
python app.py

# TTS
cd services\tts
python app.py

# SOAP
cd services\soap
python app.py

# FHIR
cd services\fhir
python app.py

# Gateway
cd gateway
pnpm dev

# Frontend (Vite)
cd frontend-vite
npm run dev

# Frontend (Next.js)
cd frontend
pnpm dev
```

### Testing
```powershell
# Test all services
python test_integration.py

# Test individual services
python test_asr.py
python test_llm.py
python test_soap.py
python test_tts.py
python test_fhir.py
```

---

## 🐛 Troubleshooting

### Issue: Gateway can't connect to microservices
**Solution**: Check `.env` file has correct service URLs:
```env
ASR_SERVICE_URL=http://localhost:5000
LLM_SERVICE_URL=http://localhost:5001
TTS_SERVICE_URL=http://localhost:5002
SOAP_SERVICE_URL=http://localhost:5003
FHIR_SERVICE_URL=http://localhost:5004
```

### Issue: ASR service out of memory
**Solution**: Lower batch size or use 8-bit quantization:
```python
model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,  # Reduces VRAM from 3GB to 2GB
    device_map="auto"
)
```

### Issue: LLM service takes forever to load
**Solution**: Model downloads on first run (16GB). Use cached model:
```python
# Check if model exists locally
import os
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
print(os.listdir(cache_dir))
```

### Issue: Frontend can't record audio
**Solution**: HTTPS or localhost required for MediaRecorder API:
```javascript
// Check browser support
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  console.error('MediaRecorder API not supported');
}
```

### Issue: CORS errors in browser
**Solution**: Verify frontend origin is in gateway's CORS whitelist (main.ts)

---

## 📚 Additional Resources

- **Week 1-4 Reports**: See `docs/Week1_Report.md` through `docs/Week4_Report.md`
- **GPU Requirements**: See `docs/GPU_REQUIREMENTS.md`
- **Setup Guide**: See `docs/SETUP.md`
- **Testing Guide**: See `docs/TESTING_GUIDE_WEEK1-5.md`
- **Implementation Guide**: See `docs/IMPLEMENTATION_GUIDE_W2-4.md`

---

**Last Updated**: Week 4 Day 28 (Oct 22, 2025)
**Maintained By**: Project Team
