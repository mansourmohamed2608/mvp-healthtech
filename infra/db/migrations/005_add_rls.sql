-- Migration 005: Enable Row Level Security on all PHI tables
-- Tenant isolation is enforced at the DB layer so an application bug cannot
-- leak cross-tenant data.  The app must call:
--
--   SET LOCAL app.tenant_id = '<tenant>';
--
-- at the beginning of every transaction / connection before issuing queries.
-- Rows are invisible (not just blocked) when app.tenant_id is unset.
--
-- Run with: psql $DATABASE_URL -f infra/db/migrations/005_add_rls.sql
--
-- IMPORTANT: After applying, update the DATABASE_URL to connect as the
-- `healthtech_app` role (non-superuser) so RLS is not bypassed.

BEGIN;

-- -------------------------------------------------------------------
-- 1. Create a least-privilege application role
-- -------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'healthtech_app') THEN
        -- Role is created without a password; use connection pooler credentials.
        CREATE ROLE healthtech_app NOLOGIN;
    END IF;
END
$$;

-- Grant DML on PHI tables to the app role
GRANT SELECT, INSERT, UPDATE, DELETE
    ON soap_notes, sessions, patients, audit_log,
       patient_documents, patient_rag_items, conversation_messages,
       appointments, users
    TO healthtech_app;

GRANT USAGE ON SCHEMA public TO healthtech_app;

-- -------------------------------------------------------------------
-- 2. Enable RLS and create per-tenant policies
-- -------------------------------------------------------------------

-- soap_notes
ALTER TABLE soap_notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE soap_notes FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS soap_notes_tenant_isolation ON soap_notes;
CREATE POLICY soap_notes_tenant_isolation ON soap_notes
    USING (tenant_id = current_setting('app.tenant_id', true));

-- sessions
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS sessions_tenant_isolation ON sessions;
CREATE POLICY sessions_tenant_isolation ON sessions
    USING (tenant_id = current_setting('app.tenant_id', true));

-- patients
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE patients FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS patients_tenant_isolation ON patients;
CREATE POLICY patients_tenant_isolation ON patients
    USING (tenant_id = current_setting('app.tenant_id', true));

-- audit_log
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS audit_log_tenant_isolation ON audit_log;
CREATE POLICY audit_log_tenant_isolation ON audit_log
    USING (tenant_id = current_setting('app.tenant_id', true));

-- patient_documents
ALTER TABLE patient_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE patient_documents FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS patient_documents_tenant_isolation ON patient_documents;
CREATE POLICY patient_documents_tenant_isolation ON patient_documents
    USING (tenant_id = current_setting('app.tenant_id', true));

-- patient_rag_items
ALTER TABLE patient_rag_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE patient_rag_items FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS patient_rag_items_tenant_isolation ON patient_rag_items;
CREATE POLICY patient_rag_items_tenant_isolation ON patient_rag_items
    USING (tenant_id = current_setting('app.tenant_id', true));

-- conversation_messages
ALTER TABLE conversation_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_messages FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS conv_messages_tenant_isolation ON conversation_messages;
CREATE POLICY conv_messages_tenant_isolation ON conversation_messages
    USING (tenant_id = current_setting('app.tenant_id', true));

-- appointments
ALTER TABLE appointments ENABLE ROW LEVEL SECURITY;
ALTER TABLE appointments FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS appointments_tenant_isolation ON appointments;
CREATE POLICY appointments_tenant_isolation ON appointments
    USING (tenant_id = current_setting('app.tenant_id', true));

-- users (each tenant can only see their own user records)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE users FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS users_tenant_isolation ON users;
CREATE POLICY users_tenant_isolation ON users
    USING (tenant_id = current_setting('app.tenant_id', true));

COMMIT;
