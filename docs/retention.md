# PHI Retention Job

Purpose: delete stale PHI rows from Postgres (`sessions`, `soap_notes`, `audit_log`) after a retention window.

- Script: `services/eval/retention_job.py`
- Env: `DATABASE_URL` (required), `PHI_RETENTION_DAYS` (default 90), `DRY_RUN=true` to preview without deleting.

## Run locally
```bash
cd services/eval
PHI_RETENTION_DAYS=90 DATABASE_URL=postgresql://postgres:postgres@localhost:5432/healthtech python retention_job.py
```

## Run via docker-compose
```bash
cd infra
PHI_RETENTION_DAYS=90 docker compose run --rm retention
```

The compose service sets `RETENTION_DRY_RUN` to control dry-run behavior (`DRY_RUN` inside the container).
