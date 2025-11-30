# Production-Ready Notes (streaming + SOAP async + OTEL)

## Critical env vars
- `JWT_SECRET`, `INTERNAL_SECRET` – auth/secrets for gateway + services.
- Service URLs: `ASR_SERVICE_URL`, `LLM_SERVICE_URL`, `TTS_SERVICE_URL`, `SOAP_SERVICE_URL`, `FHIR_SERVICE_URL`.
- DB/Redis: `DATABASE_URL`, `REDIS_HOST`/`REDIS_PORT` or `SOAP_QUEUE_URL`.
- HTTP client hardening: `INTERNAL_HTTP_TIMEOUT_MS`, `INTERNAL_HTTP_RETRIES`, optional circuit breaker (`INTERNAL_HTTP_CB_ENABLED`, `INTERNAL_HTTP_CB_THRESHOLD`, `INTERNAL_HTTP_CB_COOLDOWN_MS`).
- Streaming ASR tuning: `STREAM_SILENCE_MS`, `STREAM_SILENCE_RMS`.
- SOAP async queue: `SOAP_ASYNC_ENABLED`, `SOAP_QUEUE_URL`, `SOAP_QUEUE_KEY`, `SOAP_JOB_MAX_ATTEMPTS`, `SOAP_JOB_BACKOFF_MS`.
- OTEL: `OTEL_ENABLED`, `OTEL_EXPORTER_OTLP_ENDPOINT` (optional).

## Running with streaming ASR + voice agent
1. Ensure ASR, gateway, and frontend are running; set streaming envs above.
2. Twilio Media Stream hits gateway `/twilio/ws/{callSid}`; gateway forwards μ-law frames to ASR `/stream/chunk`.
3. Partial/final transcripts are emitted; final triggers LLM/TTS pipeline. Silence thresholds are governed by `STREAM_SILENCE_*`.

## SOAP async queue + worker
1. Enable async: `SOAP_ASYNC_ENABLED=1` and set `SOAP_QUEUE_URL`.
2. Bring up worker: `docker compose up soap-worker` (needs Redis and Postgres).
3. Jobs recorded in `soap_jobs` table and Redis queue; status via `GET /soap/job/:id`.

## OpenTelemetry
- Toggle with `OTEL_ENABLED=true`; set `OTEL_EXPORTER_OTLP_ENDPOINT` for OTLP.
- Safe to leave disabled; services run as before when unset.

## PHI-safe logging
- Gateway uses redaction helper; only IDs/status/durations are logged.
- Python services suppress raw transcripts; correlation/session IDs stay in logs for troubleshooting.

## Minimal voice-agent TTS test (no LLM)
- Set `VOICE_AGENT_LLM_ENABLED=0` to bypass LLM and return a canned reply for TTS.
- Bring up only the essentials (example):
  ```bash
  VOICE_AGENT_LLM_ENABLED=0 \
  docker compose up gateway asr tts redis postgres
  # frontend-vite optional for UI
  ```
- Not required for this audio-only test: `llm`, `soap`, `fhir`, `soap-worker`.
- Ensure Twilio env vars (account SID/auth token/app SID, etc.) are set per the existing Twilio docs; the voice reply will be the static canned phrase (not smart).
