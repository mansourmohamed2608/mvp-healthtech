# Incident Response Playbook

## Overview

This playbook provides structured procedures for responding to production incidents.

---

## GCP Demo VM – Quick Triage Commands

Run these first for any incident. VM: `healthtech-demo`, project: `healthtech-482409`.

```bash
# 1. SSH in
gcloud compute ssh healthtech-demo --zone=us-east1-b --project=healthtech-482409 \
  --strict-host-key-checking=no

# 2. Container health at a glance
docker ps --format 'table {{.Names}}\t{{.Status}}' | sort

# 3. Gateway logs (most frequent failure point)
docker logs infra-gateway-1 --tail 60

# 4. Check for TypeScript compilation errors (kills gateway startup)
docker logs infra-gateway-1 2>&1 | grep 'error TS\|Found [0-9]* errors'

# 5. Nginx/SSL
sudo systemctl status nginx
curl -sf http://localhost:3000/health && echo "Gateway OK"

# 6. GPU memory (ASR/LLM OOM is common)
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**Common quick-fixes:**

| Symptom | Fix |
|---|---|
| 502 on all routes | `docker compose -f infra/docker-compose.demo.yml restart gateway` |
| `Found N errors` in gateway logs | Fix TS errors, push, rebuild gateway |
| ASR/LLM `unhealthy` | `docker compose restart asr` or `restart llm` |
| Nginx 502 but gateway OK | `sudo systemctl reload nginx` |
| DB connection refused | `docker compose restart postgres && restart gateway` |

---

## Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| **SEV1** | Complete outage | 15 minutes | All services down, data loss |
| **SEV2** | Partial outage | 30 minutes | Critical feature unavailable |
| **SEV3** | Degraded service | 2 hours | Slow response, intermittent errors |
| **SEV4** | Minor issue | 24 hours | Non-critical bug, cosmetic issue |

---

## Incident Response Process

### 1. Detection

**Automated Alerts**
- Prometheus AlertManager
- Uptime monitoring (Pingdom/StatusPage)
- Error rate thresholds
- Latency thresholds

**Manual Detection**
- User reports
- Support tickets
- Internal testing

### 2. Acknowledgment

```bash
# Acknowledge alert in PagerDuty
# Or manually notify team

# Join incident channel
# Slack: #incident-response
```

### 3. Assessment

Quickly assess:
- **Impact**: How many users affected?
- **Scope**: Which services are affected?
- **Severity**: Assign severity level

### 4. Communication

**Internal**
- Create incident channel: `#incident-YYYY-MM-DD-brief`
- Notify on-call engineers
- Update stakeholders

**External (if needed)**
- Update status page
- Notify affected customers

### 5. Mitigation

Follow relevant playbook below.

### 6. Resolution

- Verify fix in production
- Update status page
- Begin post-mortem

---

## Incident Playbooks

### Gateway Down (SEV1)

**Symptoms**
- 502/503 errors from load balancer
- No response from gateway health endpoint
- Metrics showing zero requests

**Investigation**

```bash
# Check gateway container status
docker-compose ps gateway
docker-compose logs gateway --tail=100

# Check resource usage
docker stats gateway

# Check health endpoint
curl -v http://localhost:3000/health

# Check for OOM kills
dmesg | grep -i oom
```

**Mitigation**

```bash
# Restart gateway
docker-compose restart gateway

# If restart fails, recreate
docker-compose up -d --force-recreate gateway

# Scale horizontally if load issue
docker-compose up -d --scale gateway=3

# Check dependencies
docker-compose ps postgres redis
```

**Rollback**

```bash
# If recent deployment caused issue
docker-compose pull gateway  # Get previous image
docker-compose up -d gateway

# Or deploy specific version
IMAGE_TAG=v1.2.3 docker-compose up -d gateway
```

---

### Database Connection Errors (SEV1/SEV2)

**Symptoms**
- "Connection refused" errors
- Timeout on database queries
- Pool exhaustion warnings

**Investigation**

```bash
# Check PostgreSQL status
docker-compose ps postgres
docker-compose logs postgres --tail=100

# Check connections
docker-compose exec postgres psql -U healthtech -c \
  "SELECT count(*) as total, state FROM pg_stat_activity GROUP BY state"

# Check max connections
docker-compose exec postgres psql -U healthtech -c \
  "SHOW max_connections"

# Check disk space
docker-compose exec postgres df -h
```

**Mitigation**

```bash
# Kill idle connections
docker-compose exec postgres psql -U healthtech -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
   WHERE state = 'idle' AND query_start < now() - interval '1 hour'"

# Restart PostgreSQL (⚠️ causes brief downtime)
docker-compose restart postgres

# Increase connection pool in application
# Edit gateway .env: DATABASE_POOL_SIZE=50
docker-compose restart gateway
```

---

### ASR Service Failures (SEV2)

**Symptoms**
- Transcription requests failing
- ASR health check failing
- High latency on ASR requests

**Investigation**

```bash
# Check ASR service
docker-compose ps asr
docker-compose logs asr --tail=100

# Check GPU
nvidia-smi

# Check memory
docker stats asr

# Test ASR directly
curl http://localhost:8001/health
```

**Mitigation**

```bash
# Restart ASR service
docker-compose restart asr

# If GPU memory exhausted
# Restart to clear GPU memory
docker-compose down asr
docker-compose up -d asr

# Reduce batch size temporarily
# Edit services/asr/.env: BATCH_SIZE=1
docker-compose restart asr
```

---

### LLM Service Timeout (SEV2)

**Symptoms**
- LLM inference timing out
- High memory usage on LLM container
- GPU utilization at 100%

**Investigation**

```bash
# Check LLM service
docker-compose logs llm --tail=100

# Check GPU memory
nvidia-smi

# Check pending requests
curl http://localhost:8002/metrics | grep pending_requests
```

**Mitigation**

```bash
# Enable request queuing
# Edit services/llm/.env: MAX_CONCURRENT_REQUESTS=2

# Restart LLM service
docker-compose restart llm

# If persistent, scale down to smaller model
# Edit services/llm/.env: MODEL_NAME=qwen-7b
docker-compose restart llm
```

---

### High Error Rate (SEV2/SEV3)

**Symptoms**
- Error rate above threshold (>1%)
- Increase in 5xx responses
- Alerts from monitoring

**Investigation**

```bash
# Check error distribution
curl http://localhost:3000/metrics | grep http_requests_total

# Check recent logs for errors
docker-compose logs --tail=500 | grep -i error

# Check specific service logs
docker-compose logs gateway --tail=100 | grep -E '(error|Error|ERROR)'

# Query Loki for error patterns
# In Grafana: {job="gateway"} |= "error" | json
```

**Mitigation**

```bash
# Identify failing endpoint
# Check Grafana dashboard for error rate by endpoint

# Enable circuit breaker if downstream service failing
# Restart affected service
docker-compose restart <service-name>

# Scale up if load-related
docker-compose up -d --scale gateway=3
```

---

### Memory Leak (SEV3)

**Symptoms**
- Gradual memory increase over time
- Container restarts due to OOM
- Performance degradation

**Investigation**

```bash
# Monitor memory over time
watch -n 5 docker stats

# Check for OOM events
dmesg | grep -i oom

# Capture heap dump (Node.js)
docker-compose exec gateway kill -USR2 1

# Check garbage collection (Python)
docker-compose logs llm | grep gc
```

**Mitigation**

```bash
# Short-term: restart service
docker-compose restart <service-name>

# Set memory limits
# Edit docker-compose.yml:
#   deploy:
#     resources:
#       limits:
#         memory: 4G

# Schedule periodic restarts
# Add to cron:
# 0 4 * * * docker-compose restart gateway
```

---

### Authentication Failures (SEV2)

**Symptoms**
- Users unable to login
- JWT validation failing
- 401 errors on authenticated endpoints

**Investigation**

```bash
# Check auth service logs
docker-compose logs gateway --tail=100 | grep auth

# Verify JWT secret is set
docker-compose exec gateway printenv | grep JWT

# Check Redis (session store)
docker-compose exec redis redis-cli ping

# Test auth endpoint
curl -X POST http://localhost:3000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test"}'
```

**Mitigation**

```bash
# If JWT secret issue
# Verify secret matches across services
# Restart gateway
docker-compose restart gateway

# If Redis issue
docker-compose restart redis
docker-compose restart gateway

# Clear all sessions (force re-login)
docker-compose exec redis redis-cli FLUSHDB
```

---

## Post-Incident

### Immediate Actions

1. **Verify resolution**
   - All health checks passing
   - Error rate back to normal
   - User-reported issues resolved

2. **Update status page**
   - Mark incident as resolved
   - Provide resolution summary

3. **Notify stakeholders**
   - Send resolution notification
   - Share initial timeline

### Post-Mortem Template

Schedule post-mortem within 48 hours:

```markdown
# Incident Post-Mortem: [TITLE]

## Summary
- **Date**: YYYY-MM-DD
- **Duration**: X hours Y minutes
- **Severity**: SEV-X
- **Impact**: [Number of affected users/requests]

## Timeline
- HH:MM - Incident detected
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Mitigation applied
- HH:MM - Incident resolved

## Root Cause
[Detailed explanation]

## Resolution
[What was done to fix it]

## Lessons Learned
- What went well
- What could be improved

## Action Items
- [ ] Action 1 (Owner, Due Date)
- [ ] Action 2 (Owner, Due Date)
```

---

## Emergency Contacts

| Role | Contact |
|------|---------|
| On-Call Engineer | PagerDuty: healthtech-oncall |
| Platform Lead | +1-XXX-XXX-XXXX |
| Database Admin | dba@example.com |
| Security Team | security@example.com |
| Cloud Provider Support | AWS Support / Azure Support |
