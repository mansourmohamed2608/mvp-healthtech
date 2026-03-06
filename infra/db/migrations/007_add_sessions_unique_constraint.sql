-- Add unique constraint on sessions.session_id required for ON CONFLICT upsert
ALTER TABLE sessions ADD CONSTRAINT sessions_session_id_unique UNIQUE (session_id);
