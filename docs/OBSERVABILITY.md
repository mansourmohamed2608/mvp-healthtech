# Observability

Minimal guidance to monitor the MVP stack (gateway + ASR/LLM/TTS/SOAP/FHIR).

## Metrics

- Gateway: `GET /metrics` (Prometheus text format)
  - Histograms: `asr_latency_seconds`, `llm_latency_seconds`, `tts_latency_seconds`, `soap_latency_seconds`, `fhir_latency_seconds`
  - Counters/Gauges: `active_conversations_total`, `messages_processed_total`, `twilio_calls_total`
  - Latency middleware also exposes `gateway_request_duration_ms`
- Services:
  - ASR/LLM/TTS/SOAP/FHIR each expose `GET /health` and `GET /metrics`.
  - ASR: `asr_transcription_duration_seconds`, `asr_rtf_ratio`
  - LLM: `llm_first_token_latency_ms`, `llm_complete_response_duration_ms`, `llm_tokens_per_second`
  - TTS: synthesis histogram via middleware + service metrics endpoint (after recent hardening)
  - SOAP/FHIR: latency/error metrics exported to gateway registry; service-level metrics on `/metrics`

Scrape all `:port/metrics` endpoints with Prometheus. Suggested scrape targets (docker-compose defaults):
`gateway:3000`, `asr:5000`, `llm:5001`, `tts:5002`, `soap:5003`, `fhir:5004`.

## Correlation IDs

- Gateway middleware sets `x-correlation-id` on every request if missing.
- All outbound calls to internal services include `x-correlation-id`; services log it when present.
- Include this header when debugging or replaying requests to tie logs/metrics together.

## Tracing (optional)

- Gateway: `ENABLE_OTEL=true` triggers a best-effort OpenTelemetry init (see `src/observability/otel.ts`). If OTEL deps are absent, it logs a warning and continues normally.
- Python services: `ENABLE_OTEL=true` will try to initialize OTLP tracing via `services/otel_setup.py`. Missing deps/config are logged and ignored.
- Configure `OTEL_EXPORTER_OTLP_ENDPOINT` to point at your collector.

## Dashboards / Alerts (suggested)

- Voice pipeline: p50/p95 `asr_latency_seconds`, `llm_latency_seconds`, `tts_latency_seconds`; error-rate alerts >5% over 5m.
- SOAP/FHIR: `soap_latency_seconds`, `fhir_latency_seconds`, `soap_errors_total`, `fhir_errors_total`.
- Gateway health: `gateway_request_duration_ms` 95th percentile, `active_conversations_total`.
- Alerts: any `/health` failing, sustained 5xx on core endpoints, rising RTF (`asr_rtf_ratio`) above 1.5x.

## Load testing

- Synthetic harness: `services/eval/load_test.py`
  - Example: `python services/eval/load_test.py --base-url http://localhost:3000 --iterations 10 --concurrency 3 --jwt <token>`
  - Exercises /asr/transcribe, /llm/chat, /tts/synthesize, /soap/generate, /soap/notes/:id/approve with non-PHI payloads.
  - Watch gateway + service `/metrics` during the run. Keeps PHI out of logs by design.
