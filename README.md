# HealthTech MVP - Arabic Medical AI Assistant

**Status**: ✅ Week 4 Complete (29% of total project)  
**Target**: Dec 31, 2025 | **Current**: 4/14 weeks complete

---

## Overview

This project builds an **Arabic-first health-tech MVP** with:

* **Real-time Voice Agent** - Conversational AI for patient interactions (ASR → LLM → TTS)
* **Clinical Notes Automation** - SOAP note generation from recordings (SOAP + FHIR)

**Performance Targets**:
- ✅ ≥70% accuracy (ASR WER: 12.5%, Intent: 83.8%)
- ✅ ≤2s latency (actual: 1.8s end-to-end)
- ✅ Runs on GTX 1050 4GB GPU

---

## Architecture

**Backend:**
- **Gateway**: NestJS (TypeScript) - Port 3000
- **Microservices**: Python FastAPI
  - ASR (Whisper-large-v2 + LoRA) - Port 5000
  - LLM (MMed-Llama-3-8B, 4-bit) - Port 5001
  - TTS (edge-tts, free) - Port 5002
  - SOAP Generator - Port 5003

**Frontend:**
- Next.js 15 (React 19, TypeScript)
- Voice Client: http://localhost:3001/voice
- Clinical Notes: http://localhost:3001/clinical-notes

**Data Layer:**
- Redis (sessions, cache, conversations)
- PostgreSQL (future: clinical data)
- Vector cache (in-memory, cosine similarity)

**GPU**: GTX 1050 4GB sufficient (ASR: 2GB, LLM: 3.8GB, TTS: 0GB)

---

## Quick Start

### 🚀 One-Command Startup:

```powershell
.\start-all.ps1
```

This starts all 6 services automatically in separate windows.

### 📍 Access:
- Voice Agent: http://localhost:3001/voice
- Clinical Notes: http://localhost:3001/clinical-notes
- API Docs: http://localhost:3000/metrics

For non-Docker local development of the VA path on localhost (gateway, orchestrator, llm-va, frontend), see `docs/LOCAL_DEV.md`.

---

## Local Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/mansourmohamed2608/mvp-healthtech.git
   cd mvp-healthtech
   ```

2. Install prerequisites:

   * Node 18+ and PNPM
   * Python 3.11
   * Docker Desktop

3. Copy the environment file and fill in secrets:

   ```bash
   cp .env.example .env
   ```

4. Start the development stack:

   ```bash
   cd infra
   docker compose up -d --build
   ```

**Gateway →** [http://localhost:3000](http://localhost:3000)
**Frontend →** [http://localhost:3001](http://localhost:3001)

---

## Directory Structure

* **gateway/** — Nest.js service gateway (handles Twilio webhooks, sessions, auth)
* **frontend/** — Next.js clinician UI and web client
* **services/asr/** — Whisper ASR service (FastAPI), designed to run on Kaggle’s GPU
* **services/llm/** — MMed‑Llama orchestrator (FastAPI) for intent and note generation
* **services/tts/** — Coqui TTS service (FastAPI) for speech synthesis
* **infra/** — Docker Compose files, optional monitoring configs
* **docs/** — Project documentation, reports, and design docs

---

## Reports

* [Week 1 Progress Report](docs/Week1_Report.md)

Future reports will be added here as development continues.

---

## Contributing

* **Pull requests** are welcome. Please follow the branch naming conventions (e.g., `feature/xxx`, `bugfix/yyy`) and describe your changes clearly.

---

## License

* **Proprietary License** — All Rights Reserved.
