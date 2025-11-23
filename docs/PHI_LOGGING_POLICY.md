# PHI Logging Policy

What you **can** log:
- IDs only: sessionId, callSid, noteId, patientId, clinicianId.
- Statuses, counts, and timings (latency, durations, token counts).
- Error codes and HTTP statuses.
- Correlation IDs for traceability.
- Audit metadata must be IDs/status/httpStatus only (never free text or payload bodies).

Frontend notes:
- Do not log transcripts, SOAP notes, or FHIR payloads to the browser console.
- Rely on gateway error envelopes (`{message, code, correlationId}`) for user-facing errors.
- Use correlationId when reporting bugs instead of copying PHI into tickets.

What you **must not** log:
- Transcripts or raw audio/base64.
- SOAP note text or FHIR payload JSON.
- Names, addresses, or any free-text clinical content.
- Full request/response bodies that may contain PHI.

Retention guidance:
- Ship logs to a centralized, access-controlled store.
- Apply PHI-safe retention (e.g., minimum necessary, bounded window per compliance).
- Ensure redaction/filters in place for any unexpected PHI (see TODOs in code where deeper refactor may be needed).
