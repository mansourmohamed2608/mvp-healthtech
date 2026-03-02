-- Migration 006: Create FHIR outbox table for reliable FHIR writes
-- Implements the transactional outbox pattern: SOAP note approval writes to
-- this table in the same DB transaction, a background worker delivers to FHIR
-- with exponential backoff.  Idempotent — safe to run multiple times.

BEGIN;

CREATE TABLE IF NOT EXISTS fhir_outbox (
    id                 uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id          text        NOT NULL DEFAULT 'default',
    soap_note_id       uuid        NOT NULL REFERENCES soap_notes(id) ON DELETE CASCADE,
    idempotency_key    text        NOT NULL UNIQUE,
    payload            jsonb       NOT NULL,
    status             text        NOT NULL DEFAULT 'pending'
                                   CHECK (status IN ('pending', 'delivered', 'failed', 'dead')),
    attempt_count      int         NOT NULL DEFAULT 0,
    max_attempts       int         NOT NULL DEFAULT 8,
    next_retry_at      timestamptz NOT NULL DEFAULT now(),
    last_error         text,
    created_at         timestamptz NOT NULL DEFAULT now(),
    delivered_at       timestamptz
);

-- Worker polls: pending items where next retry time has passed
CREATE INDEX IF NOT EXISTS idx_fhir_outbox_worker
    ON fhir_outbox (status, next_retry_at)
    WHERE status IN ('pending', 'failed');

-- Tenant isolation for multi-tenant audits
CREATE INDEX IF NOT EXISTS idx_fhir_outbox_tenant ON fhir_outbox (tenant_id);

COMMIT;
