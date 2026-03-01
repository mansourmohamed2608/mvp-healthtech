# Monitoring Runbook

## Overview

This runbook covers monitoring setup, daily operations, and alert response for the HealthTech platform.

---

## Quick Reference

| Action | Command/URL |
|--------|------------|
| Grafana | http://localhost:3001 (admin/admin) |
| Prometheus | http://localhost:9090 |
| Prometheus Alerts | http://localhost:9090/alerts |
| Service Health | `curl http://localhost:3000/health` |

---

## 1. Monitoring Stack Setup

### 1.1 Start Observability Stack

```bash
cd mvp-healthtech

# Start with observability profile
docker compose --profile observability up -d

# Verify services
docker compose ps | grep -E "(prometheus|grafana|loki)"
```

### 1.2 Verify Prometheus Targets

1. Open http://localhost:9090/targets
2. Verify all targets are UP:
   - gateway:3000
   - asr:5000
   - llm:5001
   - tts:5002
   - soap:5003
   - fhir:5004
   - postgres-exporter:9187
   - redis-exporter:9121

### 1.3 Import Dashboards

Dashboards are auto-provisioned via Grafana provisioning. Manual import:

1. Open Grafana → Dashboards → Import
2. Upload JSON from `infra/observability/grafana/dashboards/`:
   - `platform-overview.json`
   - `ml-services.json`

---

## 2. Daily Monitoring Checklist

### 2.1 Morning Health Check (5 min)

```bash
# Check all services
curl -s http://localhost:3000/health | jq .

# Check Prometheus alerts
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts | length'

# Quick error check (last hour)
docker compose logs --since 1h --no-log-prefix 2>&1 | grep -c -i "error"
```

### 2.2 Grafana Dashboard Review

Open **Platform Overview** dashboard and verify:

| Panel | Expected | Action if Abnormal |
|-------|----------|-------------------|
| Service Status | All green | Check service logs |
| Error Rate | < 1% | Check error logs |
| P99 Latency | < 2s | Check slow queries |
| Active Requests | < 100 | Check for traffic spike |

### 2.3 Weekly Metrics Review

| Metric | Query | Target |
|--------|-------|--------|
| Availability | `avg_over_time(up{job="gateway"}[7d])` | > 99.9% |
| Error Budget | See SLO query below | > 0 |
| P50 Latency Trend | `histogram_quantile(0.5, sum(rate(http_request_duration_seconds_bucket[7d])) by (le))` | Stable |

---

## 3. Alert Response Procedures

### 3.1 ServiceDown Alert

**Severity:** Critical

**Symptoms:**
- Prometheus target shows DOWN
- Health endpoint unreachable

**Response:**

```bash
# 1. Check service status
docker compose ps <service-name>

# 2. Check service logs
docker compose logs --tail=100 <service-name>

# 3. Restart if needed
docker compose restart <service-name>

# 4. Verify recovery
curl http://localhost:<port>/health
```

### 3.2 HighErrorRate Alert

**Severity:** Critical (>10%), Warning (>5%)

**Response:**

```bash
# 1. Check error distribution
curl -s 'http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total{status=~"5.."}[5m]))by(path)' | jq .

# 2. Check recent errors in logs
docker compose logs --since 15m gateway | grep -i error

# 3. Check downstream services
for svc in asr llm soap fhir; do
  echo "=== $svc ==="
  docker compose logs --since 15m $svc | grep -i error | head -5
done

# 4. Common causes:
#    - Database connection issues → Check Postgres
#    - Memory pressure → Check container stats
#    - Upstream service failure → Check dependency health
```

### 3.3 HighLatency Alert

**Severity:** Warning (>2s P99), Critical (>5s P99)

**Response:**

```bash
# 1. Identify slow endpoints
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,sum(rate(http_request_duration_seconds_bucket[5m]))by(le,path))' | jq .

# 2. Check for resource constraints
docker stats --no-stream

# 3. Check database slow queries
docker compose exec postgres psql -U postgres healthtech -c "
SELECT query, calls, mean_time, max_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"

# 4. Check ASR/LLM queue depth
curl -s http://localhost:5000/metrics | grep queue
curl -s http://localhost:5001/metrics | grep queue
```

### 3.4 HighGPUMemory Alert

**Severity:** Warning (>90%)

**Response:**

```bash
# 1. Check GPU usage
nvidia-smi

# 2. Check ML service batch sizes
docker compose logs llm | grep -i "batch\|memory"

# 3. Reduce batch size if needed
docker compose exec llm /bin/bash -c "
export MAX_BATCH_SIZE=4
"

# 4. Consider scaling horizontally
# See: docs/runbooks/scaling.md
```

### 3.5 AuthenticationFailureSpike Alert

**Severity:** Warning

**Response:**

```bash
# 1. Check authentication failures by IP
docker compose logs gateway | grep -i "auth\|401" | awk '{print $NF}' | sort | uniq -c | sort -rn | head

# 2. Check for brute force patterns
docker compose logs gateway --since 1h | grep 401 | wc -l

# 3. If attack suspected:
#    - Enable enhanced rate limiting
#    - Consider IP blocking at firewall level
#    - Review audit logs

# 4. Check for legitimate issues:
#    - Expired tokens
#    - Clock skew
#    - Configuration changes
```

---

## 4. Custom Queries

### 4.1 SLO Queries

```promql
# Error Budget (99.9% SLO)
1 - (
  sum(increase(http_requests_total{status=~"5.."}[30d]))
  /
  sum(increase(http_requests_total[30d]))
) / 0.001

# Burn rate (how fast error budget is being consumed)
sum(rate(http_requests_total{status=~"5.."}[1h]))
/
sum(rate(http_requests_total[1h]))
/
0.001  # 0.1% error budget

# Time remaining at current burn rate (hours)
# If burn_rate > 1, budget exhausted before 30 days
```

### 4.2 Business Metrics

```promql
# SOAP notes per hour
sum(increase(soap_notes_generated_total[1h]))

# Transcription minutes processed
sum(increase(asr_audio_seconds_processed_total[1h])) / 60

# Active voice sessions
sum(active_voice_sessions)

# FHIR push success rate
sum(rate(fhir_pushes_total{status="success"}[5m])) 
/ sum(rate(fhir_pushes_total[5m]))
```

### 4.3 Capacity Planning

```promql
# Projected request rate (linear regression)
predict_linear(http_requests_total[7d], 30*24*3600)

# Memory growth rate
deriv(container_memory_usage_bytes{container="gateway"}[1h]) * 3600

# Database connection pool utilization
pg_stat_activity_count / pg_settings_max_connections
```

---

## 5. Log Analysis

### 5.1 Loki Queries (LogQL)

```logql
# All errors across services
{job=~"gateway|asr|llm|soap|fhir"} |= "error"

# Slow requests (>1s)
{job="gateway"} | json | duration > 1000

# Failed transcriptions
{job="asr"} |= "error" |= "transcription"

# FHIR push failures
{job="fhir"} |= "push" |= "failed"

# Count errors by service (last hour)
sum(count_over_time({job=~".+"} |= "error" [1h])) by (job)
```

### 5.2 Docker Logs Quick Reference

```bash
# Recent errors
docker compose logs --since 30m 2>&1 | grep -i error

# Follow logs for service
docker compose logs -f gateway

# Export logs to file
docker compose logs --no-log-prefix > logs_$(date +%Y%m%d_%H%M%S).txt

# Logs with timestamps
docker compose logs -t gateway | tail -100
```

---

## 6. Prometheus Maintenance

### 6.1 Check Storage

```bash
# Check Prometheus data size
docker compose exec prometheus df -h /prometheus

# Check retention
curl -s http://localhost:9090/api/v1/status/runtimeinfo | jq '.data.storageRetention'
```

### 6.2 Reload Configuration

```bash
# Reload without restart
curl -X POST http://localhost:9090/-/reload

# Verify config loaded
curl -s http://localhost:9090/api/v1/status/config | jq '.status'
```

### 6.3 Backup Metrics

```bash
# Snapshot Prometheus data
curl -X POST http://localhost:9090/api/v1/admin/tsdb/snapshot

# Copy snapshot
docker cp prometheus:/prometheus/snapshots/<snapshot-name> ./prometheus-backup/
```

---

## 7. Grafana Maintenance

### 7.1 Backup Dashboards

```bash
# Export all dashboards
mkdir -p grafana-backup
curl -s http://admin:admin@localhost:3001/api/search | jq -r '.[].uid' | while read uid; do
  curl -s "http://admin:admin@localhost:3001/api/dashboards/uid/$uid" | jq '.dashboard' > "grafana-backup/$uid.json"
done
```

### 7.2 User Management

```bash
# Create service account
curl -X POST -H "Content-Type: application/json" \
  -d '{"name":"automation","role":"Viewer"}' \
  http://admin:admin@localhost:3001/api/serviceaccounts

# Reset admin password
docker compose exec grafana grafana-cli admin reset-admin-password newpassword
```

---

## 8. Troubleshooting

### 8.1 Prometheus Not Scraping

```bash
# Check target status
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'

# Common fixes:
# 1. Check network connectivity
docker compose exec prometheus wget -q -O- http://gateway:3000/metrics

# 2. Check firewall rules
# 3. Verify service is exposing /metrics endpoint
# 4. Check scrape_configs in prometheus.yml
```

### 8.2 Grafana Dashboard Not Loading

```bash
# Check datasource connectivity
curl -s http://admin:admin@localhost:3001/api/datasources/proxy/1/api/v1/query?query=up

# Check provisioning errors
docker compose logs grafana | grep -i error

# Restart Grafana
docker compose restart grafana
```

### 8.3 Alerts Not Firing

```bash
# Check alertmanager connectivity
curl -s http://localhost:9090/api/v1/alertmanagers | jq .

# Check alert rules are loaded
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[].rules | length'

# Check alert state
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts'
```

---

## Related Documentation

- [Incident Response](./incident-response.md)
- [Scaling Guide](./scaling.md)
- [Observability Overview](../OBSERVABILITY.md)
- [Alerting Rules](../../infra/observability/alerting-rules.yml)
