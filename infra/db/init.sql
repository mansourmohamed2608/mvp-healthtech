CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS clinicians (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  email text UNIQUE,
  display_name text,
  roles text[] DEFAULT '{clinician}',
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS patients (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  external_id text,
  display_name text,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id text NOT NULL,
  patient_id text NOT NULL,
  clinician_id text NOT NULL,
  started_at timestamptz NOT NULL DEFAULT now(),
  ended_at timestamptz,
  deleted_at timestamptz,
  archived_at timestamptz
);

CREATE TABLE IF NOT EXISTS soap_notes (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id text NOT NULL,
  patient_id text NOT NULL,
  clinician_id text NOT NULL,
  status text NOT NULL CHECK (status IN ('pending', 'approved', 'rejected')),
  raw_transcript text NOT NULL,
  soap_json jsonb NOT NULL,
  subjective text,
  objective text,
  assessment text,
  plan text,
  icd_codes text[] DEFAULT '{}'::text[],
  cpt_codes text[] DEFAULT '{}'::text[],
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  archived_at timestamptz
);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_patient_id ON sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_sessions_clinician_id ON sessions(clinician_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_soap_notes_session ON soap_notes(session_id);
CREATE INDEX IF NOT EXISTS idx_soap_notes_patient ON soap_notes(patient_id);
CREATE INDEX IF NOT EXISTS idx_soap_notes_clinician ON soap_notes(clinician_id);
CREATE INDEX IF NOT EXISTS idx_soap_notes_status ON soap_notes(status);

CREATE TABLE IF NOT EXISTS soap_jobs (
  job_id text PRIMARY KEY,
  session_id text,
  patient_id text,
  clinician_id text,
  status text NOT NULL CHECK (status IN ('pending','processing','done','failed')),
  attempts integer NOT NULL DEFAULT 0,
  note_id uuid NULL,
  error_code text,
  last_error text,
  correlation_id text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_soap_jobs_status ON soap_jobs(status);

CREATE TABLE IF NOT EXISTS audit_log (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  actor_id text NOT NULL,
  action text NOT NULL,
  resource_type text NOT NULL,
  resource_id text NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audit_actor_id ON audit_log(actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_resource_type ON audit_log(resource_type);
CREATE INDEX IF NOT EXISTS idx_audit_resource_id ON audit_log(resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_log(created_at);

-- TODO (retention/audit):
-- - Define PHI retention policy (e.g., archive or delete after X days) using archived_at/deleted_at.
-- - Add audit trail table (note_id, action, actor, at, metadata) for approvals/edits.
-- - Consider soft-delete vs hard-delete hooks in application services.
