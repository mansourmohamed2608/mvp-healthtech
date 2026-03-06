-- Add unique constraint on sessions.session_id required for ON CONFLICT upsert
-- Idempotent: no-op if the constraint already exists
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'sessions_session_id_unique'
      AND conrelid = 'sessions'::regclass
  ) THEN
    ALTER TABLE sessions ADD CONSTRAINT sessions_session_id_unique UNIQUE (session_id);
  END IF;
END $$;
