#!/bin/bash
sudo docker exec infra-postgres-1 psql -U postgres -d healthtech -c "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';"
