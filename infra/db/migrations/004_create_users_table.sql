-- Migration 004: Create users table for local bcrypt authentication
-- Provides a real user store so credentials are NOT stored as plain-text env vars.
-- In production the primary auth path is OIDC; this table supports dev/demo logins
-- and serves as a fallback for deployments without an external IdP.
--
-- Run with: psql $DATABASE_URL -f infra/db/migrations/004_create_users_table.sql

BEGIN;

CREATE TABLE IF NOT EXISTS users (
    id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     text        NOT NULL DEFAULT 'default',
    username      text        NOT NULL,
    email         text,
    password_hash text        NOT NULL,  -- bcrypt hash; NEVER store plain-text
    roles         text[]      NOT NULL DEFAULT ARRAY['clinician'],
    active        boolean     NOT NULL DEFAULT true,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, username)
);

CREATE INDEX IF NOT EXISTS idx_users_tenant_username ON users (tenant_id, username);
CREATE INDEX IF NOT EXISTS idx_users_tenant_email    ON users (tenant_id, email) WHERE email IS NOT NULL;

-- Keep updated_at current automatically
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS users_set_updated_at ON users;
CREATE TRIGGER users_set_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------
-- Seed a dev user using the gateway's bcrypt helper:
--
--   node -e "require('bcrypt').hash('YOUR_PASSWORD', 12).then(h => \
--     console.log(\"INSERT INTO users (tenant_id,username,email,password_hash,roles) VALUES ('default','dev','dev@healthtech.local','\" + h + \"',ARRAY['clinician','admin']);\"))"
--
-- Copy the output and run it against your DB.  Do NOT embed plain-text
-- passwords or weak hashes in this file.
-- ---------------------------------------------------------------

COMMIT;
