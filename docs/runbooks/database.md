# Database Operations Runbook

## Overview

This runbook covers database administration, backup/restore, and disaster recovery procedures.

---

## Connection Information

| Environment | Host | Port | Database |
|-------------|------|------|----------|
| Local | localhost | 5432 | healthtech |
| Staging | staging-db.example.com | 5432 | healthtech |
| Production | prod-db.example.com | 5432 | healthtech |

---

## Daily Operations

### Health Check

```bash
# Check PostgreSQL is running
docker-compose exec postgres pg_isready

# Check connection count
docker-compose exec postgres psql -U healthtech -c \
  "SELECT count(*) FROM pg_stat_activity"

# Check database size
docker-compose exec postgres psql -U healthtech -c \
  "SELECT pg_size_pretty(pg_database_size('healthtech'))"

# Check table sizes
docker-compose exec postgres psql -U healthtech -c \
  "SELECT schemaname, tablename, 
   pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
   FROM pg_tables WHERE schemaname = 'public' ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC LIMIT 10"
```

### Performance Check

```bash
# Check slow queries
docker-compose exec postgres psql -U healthtech -c \
  "SELECT query, calls, mean_time, total_time 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC LIMIT 10"

# Check for locks
docker-compose exec postgres psql -U healthtech -c \
  "SELECT pid, relation::regclass, mode, granted
   FROM pg_locks WHERE NOT granted"

# Check for long-running queries
docker-compose exec postgres psql -U healthtech -c \
  "SELECT pid, now() - pg_stat_activity.query_start AS duration, query
   FROM pg_stat_activity
   WHERE state = 'active' AND now() - pg_stat_activity.query_start > interval '1 minute'"
```

---

## Backup Procedures

### Automated Backups

Backups are configured to run automatically:
- Full backup: Daily at 2 AM UTC
- WAL archiving: Continuous
- Retention: 30 days

### Manual Backup

```bash
# Full database backup
docker-compose exec postgres pg_dump -U healthtech healthtech > backup_$(date +%Y%m%d_%H%M%S).sql

# Compressed backup
docker-compose exec postgres pg_dump -U healthtech healthtech | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Custom format (recommended for large databases)
docker-compose exec postgres pg_dump -U healthtech -Fc healthtech > backup_$(date +%Y%m%d_%H%M%S).dump

# Backup specific tables
docker-compose exec postgres pg_dump -U healthtech -t soap_notes -t encounters healthtech > soap_backup.sql
```

### Backup to Cloud Storage

```bash
# Backup to S3
docker-compose exec postgres pg_dump -U healthtech -Fc healthtech | \
  aws s3 cp - s3://healthtech-backups/db/backup_$(date +%Y%m%d).dump

# Backup to Azure Blob
docker-compose exec postgres pg_dump -U healthtech -Fc healthtech | \
  az storage blob upload --container-name backups --name backup_$(date +%Y%m%d).dump --data @-
```

### Verify Backup

```bash
# List backup contents
pg_restore -l backup.dump

# Test restore to temp database
createdb healthtech_restore_test
pg_restore -d healthtech_restore_test backup.dump
dropdb healthtech_restore_test
```

---

## Restore Procedures

### Full Database Restore

```bash
# ⚠️ WARNING: This will overwrite all data

# 1. Stop application services
docker-compose stop gateway asr llm tts soap fhir

# 2. Drop and recreate database
docker-compose exec postgres psql -U healthtech -c "DROP DATABASE IF EXISTS healthtech"
docker-compose exec postgres psql -U healthtech -c "CREATE DATABASE healthtech"

# 3. Restore from backup
docker-compose exec -T postgres pg_restore -U healthtech -d healthtech < backup.dump

# Or from SQL file
docker-compose exec -T postgres psql -U healthtech healthtech < backup.sql

# 4. Start application services
docker-compose start gateway asr llm tts soap fhir
```

### Point-in-Time Recovery (PITR)

Requires WAL archiving enabled:

```bash
# 1. Stop PostgreSQL
docker-compose stop postgres

# 2. Clear data directory (backup first!)
rm -rf /var/lib/postgresql/data/*

# 3. Restore base backup
pg_basebackup -D /var/lib/postgresql/data

# 4. Create recovery.conf
cat > /var/lib/postgresql/data/recovery.conf << EOF
restore_command = 'cp /path/to/wal/%f %p'
recovery_target_time = '2024-01-15 14:30:00'
EOF

# 5. Start PostgreSQL
docker-compose start postgres
```

### Partial Restore (Specific Tables)

```bash
# Extract specific table from backup
pg_restore -t soap_notes backup.dump > soap_notes.sql

# Restore to database
docker-compose exec -T postgres psql -U healthtech healthtech < soap_notes.sql
```

---

## Migration Procedures

### Run Migrations

```bash
# Using gateway migration
cd gateway
npm run migration:run

# Or with Docker
docker-compose exec gateway npm run migration:run

# Check migration status
npm run migration:status
```

### Create New Migration

```bash
# Generate migration from entity changes
npm run migration:generate -- -n CreateNewTable

# Create empty migration
npm run migration:create -- -n ManualChanges
```

### Rollback Migration

```bash
# Revert last migration
npm run migration:revert

# Revert to specific migration
npm run migration:revert -- -t MigrationName
```

---

## Troubleshooting

### Connection Issues

```bash
# Check PostgreSQL logs
docker-compose logs postgres --tail=100

# Test connection
docker-compose exec postgres pg_isready -h localhost -p 5432

# Check max connections
docker-compose exec postgres psql -U healthtech -c "SHOW max_connections"

# Kill idle connections
docker-compose exec postgres psql -U healthtech -c \
  "SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE state = 'idle' AND query_start < now() - interval '10 minutes'"
```

### Performance Issues

```bash
# Analyze tables
docker-compose exec postgres psql -U healthtech -c "ANALYZE"

# Vacuum database
docker-compose exec postgres psql -U healthtech -c "VACUUM ANALYZE"

# Rebuild indexes
docker-compose exec postgres psql -U healthtech -c \
  "REINDEX DATABASE healthtech"

# Check for bloat
docker-compose exec postgres psql -U healthtech -c \
  "SELECT schemaname, tablename, 
   pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as size
   FROM pg_tables WHERE schemaname = 'public'"
```

### Disk Space Issues

```bash
# Check disk usage
df -h

# Check table sizes
docker-compose exec postgres psql -U healthtech -c \
  "SELECT schemaname, tablename, 
   pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
   FROM pg_tables ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC"

# Clean up old data
docker-compose exec postgres psql -U healthtech -c \
  "DELETE FROM audit_logs WHERE created_at < now() - interval '90 days'"

# Run vacuum to reclaim space
docker-compose exec postgres psql -U healthtech -c "VACUUM FULL"
```

### Deadlock Detection

```bash
# Find deadlocks
docker-compose exec postgres psql -U healthtech -c \
  "SELECT blocked_locks.pid AS blocked_pid,
   blocked_activity.usename AS blocked_user,
   blocking_locks.pid AS blocking_pid,
   blocking_activity.usename AS blocking_user,
   blocked_activity.query AS blocked_statement
   FROM pg_catalog.pg_locks blocked_locks
   JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
   JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
   JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
   WHERE NOT blocked_locks.granted"

# Kill blocking query
docker-compose exec postgres psql -U healthtech -c \
  "SELECT pg_terminate_backend(<blocking_pid>)"
```

---

## Disaster Recovery

### Complete Database Loss

1. **Notify stakeholders**
2. **Identify most recent backup**
3. **Provision new database instance**
4. **Restore from backup**
5. **Apply WAL logs if available (PITR)**
6. **Verify data integrity**
7. **Update connection strings**
8. **Resume services**

### Corrupted Data

```bash
# 1. Identify corruption scope
docker-compose exec postgres psql -U healthtech -c \
  "SELECT * FROM pg_catalog.pg_stat_user_tables WHERE n_dead_tup > 10000"

# 2. Stop writes to affected tables
# Application-level: disable related endpoints

# 3. Export uncorrupted data
pg_dump -U healthtech -t good_table healthtech > good_table.sql

# 4. Restore affected tables from backup
pg_restore -t corrupted_table -d healthtech backup.dump

# 5. Reconcile with exported data if needed

# 6. Resume operations
```

---

## Maintenance Windows

### Weekly Maintenance (Sunday 3 AM UTC)

```bash
#!/bin/bash
# weekly-maintenance.sh

# Run vacuum
docker-compose exec postgres psql -U healthtech -c "VACUUM ANALYZE"

# Update statistics
docker-compose exec postgres psql -U healthtech -c "ANALYZE"

# Check for unused indexes
docker-compose exec postgres psql -U healthtech -c \
  "SELECT schemaname, tablename, indexname, idx_scan
   FROM pg_stat_user_indexes WHERE idx_scan = 0"
```

### Monthly Maintenance (First Sunday 2 AM UTC)

```bash
#!/bin/bash
# monthly-maintenance.sh

# Full vacuum
docker-compose exec postgres psql -U healthtech -c "VACUUM FULL"

# Reindex
docker-compose exec postgres psql -U healthtech -c "REINDEX DATABASE healthtech"

# Archive old logs
docker-compose exec postgres psql -U healthtech -c \
  "DELETE FROM audit_logs WHERE created_at < now() - interval '90 days'"
```

---

## Security

### Access Control

```bash
# Create read-only user
docker-compose exec postgres psql -U healthtech -c \
  "CREATE USER readonly WITH PASSWORD 'secure_password';
   GRANT CONNECT ON DATABASE healthtech TO readonly;
   GRANT USAGE ON SCHEMA public TO readonly;
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;"

# Rotate passwords
docker-compose exec postgres psql -U healthtech -c \
  "ALTER USER healthtech WITH PASSWORD 'new_secure_password';"
```

### Audit Queries

```bash
# Enable query logging (temporarily)
docker-compose exec postgres psql -U healthtech -c \
  "ALTER SYSTEM SET log_statement = 'all';
   SELECT pg_reload_conf();"

# Disable after investigation
docker-compose exec postgres psql -U healthtech -c \
  "ALTER SYSTEM SET log_statement = 'none';
   SELECT pg_reload_conf();"
```
