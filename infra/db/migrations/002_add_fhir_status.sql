-- Migration 002: Add FHIR writeback status columns
-- PR-8: FHIR writeback integrity
-- Run with: psql $DATABASE_URL -f infra/db/migrations/002_add_fhir_status.sql

BEGIN;

-- ============================================
-- Add FHIR status columns to soap_notes
-- Separate from approval_status (which is the existing 'status' column)
-- ============================================

-- FHIR write status: tracks EHR sync state
ALTER TABLE soap_notes
  ADD COLUMN IF NOT EXISTS fhir_status text NOT NULL DEFAULT 'not_requested'
    CHECK (fhir_status IN ('not_requested', 'pending', 'success', 'failed'));

-- Retry tracking
ALTER TABLE soap_notes
  ADD COLUMN IF NOT EXISTS fhir_attempts integer NOT NULL DEFAULT 0;

-- Last error message for debugging
ALTER TABLE soap_notes
  ADD COLUMN IF NOT EXISTS fhir_last_error text;

-- When FHIR write succeeded
ALTER TABLE soap_notes
  ADD COLUMN IF NOT EXISTS fhir_written_at timestamptz;

-- Idempotency key to prevent duplicate writes
ALTER TABLE soap_notes
  ADD COLUMN IF NOT EXISTS fhir_idempotency_key text;

-- External FHIR resource ID (e.g., Encounter/abc123)
ALTER TABLE soap_notes
  ADD COLUMN IF NOT EXISTS fhir_resource_id text;

-- Indexes for FHIR retry worker queries
CREATE INDEX IF NOT EXISTS idx_soap_notes_fhir_status ON soap_notes(fhir_status);
CREATE INDEX IF NOT EXISTS idx_soap_notes_fhir_pending ON soap_notes(fhir_status, fhir_attempts)
  WHERE fhir_status IN ('pending', 'failed');
CREATE UNIQUE INDEX IF NOT EXISTS idx_soap_notes_fhir_idempotency
  ON soap_notes(fhir_idempotency_key) WHERE fhir_idempotency_key IS NOT NULL;

-- ============================================
-- FHIR Outbox table for reliable writes
-- Implements transactional outbox pattern
-- ============================================
CREATE TABLE IF NOT EXISTS fhir_outbox (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  tenant_id text NOT NULL DEFAULT 'default',
  soap_note_id uuid NOT NULL REFERENCES soap_notes(id) ON DELETE CASCADE,
  idempotency_key text NOT NULL UNIQUE,
  payload jsonb NOT NULL,
  status text NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'processing', 'success', 'failed', 'dead_letter')),
  attempts integer NOT NULL DEFAULT 0,
  max_attempts integer NOT NULL DEFAULT 3,
  last_error text,
  next_retry_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  processed_at timestamptz
);

-- Indexes for outbox worker
CREATE INDEX IF NOT EXISTS idx_fhir_outbox_pending ON fhir_outbox(status, next_retry_at)
  WHERE status IN ('pending', 'failed');
CREATE INDEX IF NOT EXISTS idx_fhir_outbox_tenant ON fhir_outbox(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fhir_outbox_soap_note ON fhir_outbox(soap_note_id);

-- ============================================
-- FHIR reconciliation tracking
-- ============================================
CREATE TABLE IF NOT EXISTS fhir_reconciliation (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  tenant_id text NOT NULL DEFAULT 'default',
  started_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz,
  status text NOT NULL DEFAULT 'running'
    CHECK (status IN ('running', 'completed', 'failed')),
  notes_checked integer NOT NULL DEFAULT 0,
  notes_fixed integer NOT NULL DEFAULT 0,
  notes_failed integer NOT NULL DEFAULT 0,
  errors jsonb DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_fhir_reconciliation_tenant ON fhir_reconciliation(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fhir_reconciliation_status ON fhir_reconciliation(status);

INSERT INTO schema_migrations (version) VALUES ('002_add_fhir_status')
ON CONFLICT (version) DO NOTHING;

COMMIT;
