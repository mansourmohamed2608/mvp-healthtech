# Mvp HealthTech MVP



---

## Overview

This project aims to build an Arabic‑first health‑tech MVP with:

* **Real‑time Voice Agent** for patient interaction.
* **Clinical Notes Automation** (SOAP + FHIR) for clinicians.

Target completion: **31 Dec 2025**, with ≥ 70 % precision and ≤ 2 s latency.

---

## Architecture

* **Backend:** Nest.js gateway + Python micro‑services (FastAPI for ASR/LLM/TTS)
* **Frontend:** Next.js (TypeScript) web client
* **Data Layer:** PostgreSQL + Redis (vector cache later)
* **Compute:** Local machines for gateway/UI; free Kaggle T4 GPU for ASR/LLM/TTS

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
