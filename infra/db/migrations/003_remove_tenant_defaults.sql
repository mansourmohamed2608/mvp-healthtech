-- Migration 003: Remove tenant_id defaults for production hardening
-- PR-6/E: Run AFTER backfilling all rows with proper tenant_id
-- 
-- PREREQUISITES:
--   1. All rows have been backfilled with actual tenant_id values
--   2. No rows remain with tenant_id = 'default' (or they are intentionally orphaned)
--   3. Application code is updated to always provide tenant_id
--
-- Run with: psql $DATABASE_URL -f infra/db/migrations/003_remove_tenant_defaults.sql
--
-- WARNING: This will FAIL if any rows have tenant_id = 'default' and MULTI_TENANT is enforced.
-- Verify first: SELECT table_name, COUNT(*) FROM (
--   SELECT 'sessions' as table_name FROM sessions WHERE tenant_id = 'default'
--   UNION ALL SELECT 'soap_notes' FROM soap_notes WHERE tenant_id = 'default'
--   ...
-- ) t GROUP BY table_name;

BEGIN;

-- ============================================
-- Step 1: Verify no orphaned 'default' tenants exist
-- (Uncomment to enforce - will fail migration if any found)
-- ============================================
-- DO $$
-- DECLARE
--   default_count integer;
-- BEGIN
--   SELECT COUNT(*) INTO default_count FROM (
--     SELECT 1 FROM sessions WHERE tenant_id = 'default'
--     UNION ALL SELECT 1 FROM soap_notes WHERE tenant_id = 'default'
--     UNION ALL SELECT 1 FROM appointments WHERE tenant_id = 'default'
--     UNION ALL SELECT 1 FROM patients WHERE tenant_id = 'default'
--     UNION ALL SELECT 1 FROM patient_documents WHERE tenant_id = 'default'
--     UNION ALL SELECT 1 FROM patient_rag_items WHERE tenant_id = 'default'
--     UNION ALL SELECT 1 FROM conversation_messages WHERE tenant_id = 'default'
--     UNION ALL SELECT 1 FROM soap_jobs WHERE tenant_id = 'default'
--     UNION ALL SELECT 1 FROM doctors WHERE tenant_id = 'default'
--     UNION ALL SELECT 1 FROM doctor_schedules WHERE tenant_id = 'default'
--     UNION ALL SELECT 1 FROM clinicians WHERE tenant_id = 'default'
--   ) t;
--   IF default_count > 0 THEN
--     RAISE EXCEPTION 'Cannot remove defaults: % rows still have tenant_id=default', default_count;
--   END IF;
-- END $$;

-- ============================================
-- Step 2: Remove DEFAULT constraint from all PHI tables
-- tenant_id remains NOT NULL but no default - forces explicit value
-- ============================================

-- Sessions table
ALTER TABLE sessions ALTER COLUMN tenant_id DROP DEFAULT;

-- SOAP notes table  
ALTER TABLE soap_notes ALTER COLUMN tenant_id DROP DEFAULT;

-- Appointments table
ALTER TABLE appointments ALTER COLUMN tenant_id DROP DEFAULT;

-- Patients table
ALTER TABLE patients ALTER COLUMN tenant_id DROP DEFAULT;

-- Patient documents table
ALTER TABLE patient_documents ALTER COLUMN tenant_id DROP DEFAULT;

-- Patient RAG items table
ALTER TABLE patient_rag_items ALTER COLUMN tenant_id DROP DEFAULT;

-- Conversation messages table
ALTER TABLE conversation_messages ALTER COLUMN tenant_id DROP DEFAULT;

-- Audit log table
ALTER TABLE audit_log ALTER COLUMN tenant_id DROP DEFAULT;

-- SOAP jobs table
ALTER TABLE soap_jobs ALTER COLUMN tenant_id DROP DEFAULT;

-- Doctors table
ALTER TABLE doctors ALTER COLUMN tenant_id DROP DEFAULT;

-- Doctor schedules table
ALTER TABLE doctor_schedules ALTER COLUMN tenant_id DROP DEFAULT;

-- Clinicians table
ALTER TABLE clinicians ALTER COLUMN tenant_id DROP DEFAULT;

-- SOAP note audit table
ALTER TABLE soap_note_audit ALTER COLUMN tenant_id DROP DEFAULT;

-- FHIR outbox table
ALTER TABLE fhir_outbox ALTER COLUMN tenant_id DROP DEFAULT;

-- FHIR reconciliation table
ALTER TABLE fhir_reconciliation ALTER COLUMN tenant_id DROP DEFAULT;

-- ============================================
-- Step 3: Add CHECK constraint to prevent 'default' in production
-- This is a safety net - application should reject first
-- ============================================
-- Uncomment if you want DB-level enforcement:
-- ALTER TABLE sessions ADD CONSTRAINT chk_sessions_no_default_tenant 
--   CHECK (tenant_id <> 'default');
-- ALTER TABLE soap_notes ADD CONSTRAINT chk_soap_notes_no_default_tenant 
--   CHECK (tenant_id <> 'default');
-- ... (repeat for all tables)

INSERT INTO schema_migrations (version) VALUES ('003_remove_tenant_defaults')
ON CONFLICT (version) DO NOTHING;

COMMIT;
