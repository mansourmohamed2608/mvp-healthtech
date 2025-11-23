# Deployment Overview (public-facing posture)

## Logical architecture
- Gateway (NestJS) exposed publicly (TLS terminates at LB/ingress); Twilio WS + REST endpoints.
- Internal services (private network/VPC only): ASR (Whisper v3), LLM (MMed-Llama-3-8B), TTS, SOAP, FHIR.
- Data plane: Postgres (sessions, notes, audit), Redis (cache/conversation state).
- Frontend (Vite SPA) served separately (points to gateway).

## Secrets / env
- See `.env.example` for required variables: JWT_SECRET, WS_SHARED_SECRET/TWILIO_AUTH_TOKEN, INTERNAL_SECRET, FHIR auth, DB/Redis URLs, model tokens, service URLs.
- Use a secret manager (Vault/SSM/KMS); never bake secrets into images. Plan rotations for JWT, WS secret, INTERNAL_SECRET, FHIR creds.

## Security edge requirements
- TLS termination at ingress/LB; redirect HTTP→HTTPS.
- WAF + rate limiting in front of gateway (DoS, SQLi/XSS filters, IP throttling).
- Internal services must NOT be internet-accessible; restrict to private subnets/namespaces. Optionally add mTLS/service mesh.

## Backup/DR basics
- Postgres: schedule `pg_dump` backups; test restores (`psql < dump.sql`). Keep retention aligned with PHI policy.
- Redis: treat as ephemeral cache; persistence not required.
- Logs: ship to centralized logging with PHI-safe retention (see PHI_LOGGING_POLICY.md).

## Deployment patterns
- Dev: `docker-compose` (infra/), metrics on `:port/metrics`.
- Prod (outline):
  - Containerize each service; deploy to k8s/containers.
  - One public Service/Ingress for gateway; ClusterIP for internal services + DB/Redis or managed equivalents.
  - Configure secrets via k8s Secrets/SSM; mount env vars only.
  - Use HPA on gateway and stateless services; pin GPU nodes for ASR/LLM if required.

## Operations checklist
- Health endpoints: `/health` on all services; monitor via probes.
- Metrics: `/metrics` on gateway + services; scrape with Prometheus.
- Tracing (optional): `ENABLE_OTEL=true` + OTLP collector.
- Retention: run `services/eval/retention_job.py` via cron/CronJob (soft-delete/archival).
- E2E smoke: after `docker-compose up`, run `JWT_SECRET=<jwt> INTERNAL_SECRET=<secret> node scripts/e2e-smoke.js` (gateway only; uses synthetic payloads).
