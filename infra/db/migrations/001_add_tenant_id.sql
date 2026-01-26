-- Migration 001: Add tenant_id to all PHI tables
-- PR-6: Multi-tenant isolation
-- Run with: psql $DATABASE_URL -f infra/db/migrations/001_add_tenant_id.sql

BEGIN;

-- ============================================
-- Add tenant_id column to PHI tables
-- Default 'default' for single-tenant backward compat
-- ============================================

-- Sessions table
ALTER TABLE sessions
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id);

-- SOAP notes table
ALTER TABLE soap_notes
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_soap_notes_tenant ON soap_notes(tenant_id);

-- Appointments table
ALTER TABLE appointments
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_appointments_tenant ON appointments(tenant_id);

-- Patients table
ALTER TABLE patients
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_patients_tenant ON patients(tenant_id);

-- Patient documents table
ALTER TABLE patient_documents
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_patient_documents_tenant ON patient_documents(tenant_id);

-- Patient RAG items table
ALTER TABLE patient_rag_items
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_patient_rag_tenant ON patient_rag_items(tenant_id);

-- Conversation messages table
ALTER TABLE conversation_messages
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_conv_messages_tenant ON conversation_messages(tenant_id);

-- Audit log table
ALTER TABLE audit_log
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id);

-- SOAP jobs table
ALTER TABLE soap_jobs
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_soap_jobs_tenant ON soap_jobs(tenant_id);

-- Doctors table (for scheduling)
ALTER TABLE doctors
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_doctors_tenant ON doctors(tenant_id);

-- Doctor schedules (inherits tenant from doctor, but add for query efficiency)
ALTER TABLE doctor_schedules
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_doctor_schedules_tenant ON doctor_schedules(tenant_id);

-- Clinicians table
ALTER TABLE clinicians
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_clinicians_tenant ON clinicians(tenant_id);

-- SOAP note audit
ALTER TABLE soap_note_audit
  ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_soap_note_audit_tenant ON soap_note_audit(tenant_id);

-- ============================================
-- Create composite indexes for common queries
-- ============================================
CREATE INDEX IF NOT EXISTS idx_sessions_tenant_patient ON sessions(tenant_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_sessions_tenant_clinician ON sessions(tenant_id, clinician_id);
CREATE INDEX IF NOT EXISTS idx_soap_notes_tenant_session ON soap_notes(tenant_id, session_id);
CREATE INDEX IF NOT EXISTS idx_soap_notes_tenant_patient ON soap_notes(tenant_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_actor ON audit_log(tenant_id, actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_created ON audit_log(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_appointments_tenant_doctor ON appointments(tenant_id, doctor_id);

-- ============================================
-- Migration tracking table
-- ============================================
CREATE TABLE IF NOT EXISTS schema_migrations (
  version text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO schema_migrations (version) VALUES ('001_add_tenant_id')
ON CONFLICT (version) DO NOTHING;

COMMIT;
