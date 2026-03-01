# Technical Documentation

## Overview
This project is a modular health technology platform for voice-driven clinical documentation and AI-powered medical note generation. It consists of backend microservices (Python FastAPI), a frontend (React/Vite), and a gateway (NestJS/Node.js) for orchestration.
---
## 1. System Architecture

- **Backend Services (Python/FastAPI):**
  - **ASR (Automatic Speech Recognition):**
    - Uses WhisperX for fast, dialect-aware Arabic medical transcription.
    - Supports diarization, alignment, and medical vocabulary injection.
    - Exposes `/transcribe` and `/stream/chunk` endpoints.
  - **TTS (Text-to-Speech):**
    - Synthesizes Arabic speech using Coqui TTS, XTTS, or Edge TTS.
    - Supports multiple voices and models.
    - Exposes `/synthesize` endpoint.
  - **SOAP Note Generator:**
    - Converts transcripts to structured SOAP notes using LLMs.
    - Persists notes in Postgres and supports FHIR writeback.
    - Exposes endpoints for note generation and template management.
    - Provides medical text post-processing, speaker role identification, and RAG (Retrieval-Augmented Generation).
    - Integrates with transformers and custom rules.

  - Orchestrates requests between frontend and backend services.
  - Handles authentication, session management, and API rate limiting.
  - Provides REST endpoints for ASR, TTS, LLM, SOAP, FHIR, and more.
- **Frontend (React/Vite/TypeScript):**
  - Main user interface for voice agent and clinical notes.

---
## 2. Backend Services

### ASR Service (`services/asr/app.py`)
- **Tech:** FastAPI, WhisperX, PyTorch, Prometheus
- **Endpoints:**
- **Features:**
  - Diarization-first and diarize-last modes
  - LLM-based correction (optional)
  - Prometheus metrics

- **Endpoints:**
  - `/synthesize` (POST): Accepts text, returns audio (wav/mulaw)
  - Multiple voice/model support
  - Device selection (CUDA/CPU)
  - Prometheus metrics

- **Endpoints:**
  - `/generate` (POST): Generates SOAP note from transcript
- **Features:**
  - LLM-based field generation
  - PDF parsing (optional)
- **Tech:** FastAPI, transformers, Prometheus
- **Endpoints:**
  - `/identify-speakers` (POST): Assigns roles to speaker segments
  - `/postprocess` (POST): Applies corrections and normalization

## 3. Gateway (NestJS)
- **Main Module:** `gateway/src/app.module.ts`
  - SOAP: `/soap/generate`
- **Guards:** JWT, Tenant, Throttler
- **Services:** Internal HTTP client for backend calls, session and conversation management

---
## 4. Frontend (React/Vite)
- **State Management:** Zustand stores for theme and auth

   - User uploads/records audio → ASR Service → Transcript → SOAP Service → Structured note → Review/edit in frontend.
- `.env` and `.env.local` files for all services (API keys, DB URIs, model paths, etc.)
---

## 7. Testing
- **Backend:** `test_*.py` scripts for each service
- **Gateway:** Jest e2e and unit tests
- **Frontend:** Component and integration tests (React Testing Library)

---

## 8. Extensibility
- Add new voices/models to TTS by updating config and model files
- Add new templates to SOAP service via template store
- Extend LLM service with new correction or validation rules
- Frontend supports new pages/routes via React Router


## 10. Observability
## 11. Quick Start (Development)
2. **Set up environment variables:**
3. **Run backend services:**
   - `uvicorn services.asr.app:app --reload`
   - `uvicorn services.tts.app:app --reload`
   - `uvicorn services.soap.app:app --reload`
   - `uvicorn services.llm.app:app --reload`
4. **Run gateway:**
   - `pnpm start:dev` in `gateway/`
5. **Run frontend:**
   - `pnpm dev` in `frontend-vite/`

---

## 12. File Structure (Key Parts)

- `services/asr/` — ASR microservice (WhisperX, FastAPI)
- `services/tts/` — TTS microservice (Coqui/XTTS/Edge, FastAPI)
- `services/soap/` — SOAP note generator (LLM, FastAPI)
- `services/llm/` — LLM service (transformers, FastAPI)
- `gateway/` — API gateway (NestJS)
- `frontend-vite/` — Frontend (React/Vite)

---

## 13. Contribution
- Use feature branches for new work
- Write clear commit messages
- Ensure all tests pass before PR
- Document new features in code and markdown

---

## 14. Troubleshooting
- Check logs for errors (suppressed PHI)
- Ensure all services are running and reachable
- Validate `.env` files and API keys

---

## 15. License
- See `LICENSE` file
