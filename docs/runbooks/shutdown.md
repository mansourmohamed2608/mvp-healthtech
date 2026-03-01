# Shutdown Runbook

## Overview

This runbook covers graceful shutdown procedures for the HealthTech platform.

---

## Graceful Shutdown Principles

1. **Drain connections** - Stop accepting new requests
2. **Complete in-flight requests** - Allow active requests to finish
3. **Close database connections** - Clean up connection pools
4. **Persist state** - Save any in-memory data
5. **Notify dependencies** - Let upstream services know

---

## Local Development Shutdown

### Stop All Services

```bash
# Graceful stop (30s timeout)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop, remove containers, and remove volumes (⚠️ destroys data)
docker-compose down -v
```

### Stop Individual Services

```bash
# Stop specific service
docker-compose stop <service-name>

# Examples
docker-compose stop gateway
docker-compose stop asr
docker-compose stop frontend
```

### Stop Development Servers

```bash
# For processes running in terminals
# Press Ctrl+C in the terminal

# Kill by port
npx kill-port 3000 5173 8001 8002 8003 8004 8005
```

---

## Staging Environment Shutdown

### Maintenance Shutdown

```bash
# SSH to staging
ssh deploy@staging.healthtech.example.com

# Enable maintenance mode (returns 503 to new requests)
./scripts/maintenance-mode.sh enable

# Wait for connections to drain (check active connections)
docker-compose exec gateway sh -c 'curl localhost:3000/metrics | grep http_active_connections'

# Stop services gracefully
docker-compose -f docker-compose.staging.yml stop

# Disable maintenance mode when ready to restart
./scripts/maintenance-mode.sh disable
```

---

## Production Environment Shutdown

### ⚠️ Pre-Shutdown Checklist

- [ ] Announce maintenance window to users
- [ ] Notify on-call team
- [ ] Verify backup completed
- [ ] Confirm rollback plan is ready
- [ ] Get approval from platform lead

### Planned Maintenance Shutdown

```bash
# 1. Enable maintenance page
./scripts/maintenance-mode.sh enable "Scheduled maintenance in progress"

# 2. Remove from load balancer (if applicable)
./scripts/lb-drain.sh production

# 3. Wait for active connections to complete
# Monitor metrics: http_active_connections
watch -n 5 'curl -s http://localhost:3000/metrics | grep active_connections'

# 4. Stop application services (not databases)
docker-compose -f docker-compose.prod.yml stop gateway asr llm tts soap fhir frontend

# 5. Perform maintenance tasks...

# 6. Start services
docker-compose -f docker-compose.prod.yml up -d

# 7. Verify health
./scripts/health-check.sh production

# 8. Add back to load balancer
./scripts/lb-enable.sh production

# 9. Disable maintenance page
./scripts/maintenance-mode.sh disable
```

### Emergency Shutdown

```bash
# ⚠️ USE ONLY IN EMERGENCIES
# This immediately stops all services without graceful drain

# Stop all containers immediately
docker-compose -f docker-compose.prod.yml kill

# Or stop specific service
docker-compose -f docker-compose.prod.yml kill <service-name>
```

---

## Service-Specific Shutdown Notes

### Gateway (NestJS)

The gateway handles graceful shutdown automatically:

```typescript
// Configured in main.ts
app.enableShutdownHooks();
```

On SIGTERM:
- Stops accepting new connections
- Waits for in-flight requests (30s timeout)
- Closes database connections
- Closes Redis connections

### ASR Service

```bash
# ASR service may have long-running transcription jobs
# Check for active jobs before shutdown
curl http://localhost:8001/metrics | grep asr_active_jobs

# Wait for jobs to complete or set timeout
docker-compose stop -t 120 asr  # 2 minute timeout
```

### LLM Service

```bash
# LLM inference can take 30-60 seconds
# Allow extra time for graceful shutdown
docker-compose stop -t 120 llm
```

### Database (PostgreSQL)

```bash
# ⚠️ Never force-stop PostgreSQL

# Graceful shutdown
docker-compose stop postgres

# Check for active connections before stop
docker-compose exec postgres psql -U healthtech -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
```

---

## Shutdown Verification

### Verify All Services Stopped

```bash
# Check no containers running
docker-compose ps

# Check ports are free
netstat -tulpn | grep -E '3000|5173|8001|8002|8003|8004|8005'
```

### Verify Clean Shutdown

```bash
# Check logs for errors during shutdown
docker-compose logs --tail=50 | grep -i -E 'error|warning|killed'

# Check for orphaned processes
ps aux | grep -E 'node|python|uvicorn'
```

---

## Troubleshooting Shutdown Issues

### Container Won't Stop

```bash
# Check what's keeping it alive
docker-compose logs <service-name> --tail=50

# Force stop after timeout
docker-compose stop -t 60 <service-name>

# Last resort: kill container
docker-compose kill <service-name>
```

### Database Connections Not Closing

```bash
# Check active connections
docker-compose exec postgres psql -U healthtech -c \
  "SELECT pid, state, query FROM pg_stat_activity WHERE datname = 'healthtech'"

# Terminate idle connections
docker-compose exec postgres psql -U healthtech -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'healthtech' AND state = 'idle'"
```

### Orphaned Processes

```bash
# Find orphaned processes
ps aux | grep -E 'node|python|uvicorn' | grep -v grep

# Kill specific process
kill -TERM <pid>

# Force kill if needed
kill -9 <pid>
```

---

## Post-Shutdown Tasks

1. **Verify logs are persisted**
   - Check log files exist in /var/log/healthtech
   - Verify logs shipped to Loki/Grafana

2. **Backup verification**
   - Confirm database backup completed
   - Verify backup integrity

3. **Clear temporary files**
   ```bash
   # Clear upload temp directory
   rm -rf /tmp/healthtech-uploads/*
   
   # Clear cache
   docker-compose exec redis redis-cli FLUSHDB
   ```

4. **Update status page**
   - Mark services as offline
   - Update estimated restore time

5. **Notify stakeholders**
   - Send shutdown confirmation
   - Provide maintenance timeline
