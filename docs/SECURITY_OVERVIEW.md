# Security Overview

Secrets (env-only, rotate regularly via secret manager):
- `JWT_SECRET`: signs/verifies JWTs for gateway auth.
- `TWILIO_AUTH_TOKEN` or `WS_SHARED_SECRET`: HMAC for Twilio WS (`sig = HMAC_SHA256(secret, \`${callSid}:${ts}\`)`).
- `INTERNAL_SECRET`: shared header `x-internal-secret` for gateway → services and enforced by all services.
- FHIR auth: `FHIR_BEARER_TOKEN` or `FHIR_BASIC_AUTH_USER`/`FHIR_BASIC_AUTH_PASSWORD`.
- Model/HF tokens: `HUGGINGFACE_HUB_TOKEN` (if required).
- DB/Redis credentials supplied via env.

WebSocket auth (Twilio media):
- Requires valid JWT (Bearer) signed with `JWT_SECRET`.
- Requires `sig` and `ts` query params; `sig = HMAC_SHA256(TWILIO_AUTH_TOKEN|WS_SHARED_SECRET, \`${callSid}:${ts}\`)`.
- Replay window: 5 minutes; older timestamps are rejected. Rotate WS secret regularly.

Internal-only services:
- ASR/LLM/TTS/SOAP/FHIR must not be internet-exposed; deploy inside private network/VPC/cluster.
- All internal calls carry `x-internal-secret`; services reject requests without the correct secret (health/metrics are exempt).

Edge requirements:
- Terminate TLS at ingress/LB.
- Place WAF and rate limiting in front of the gateway (DoS/IP throttling/SQLi-XSS protections).
- Keep JWT and WS/Twilio secrets in a secret manager; rotate on a schedule with rolling deploys.
