# Runbook: Startup & Shutdown Procedures

**Platform:** HealthTech MVP  
**VM:** GCP `healthtech-demo` · zone `us-east1-b` · static IP `34.26.235.26`  
**Last updated:** 2026-03-01

---

## 1. Full Stack Startup

### Prerequisites
- GCP project `healthtech-482409` access
- `gcloud` CLI authenticated (`gcloud auth login`)
- `.env` file present on VM at `~/mvp-healthtech/infra/.env` with all required secrets

### Start the VM (if stopped)
```bash
gcloud compute instances start healthtech-demo \
  --zone=us-east1-b --project=healthtech-482409

# Wait ~30s for SSH to become available, then verify
gcloud compute instances describe healthtech-demo \
  --zone=us-east1-b --project=healthtech-482409 \
  --format='get(status)'   # should print RUNNING
```

### SSH onto the VM
```bash
gcloud compute ssh healthtech-demo \
  --zone=us-east1-b --project=healthtech-482409 \
  --strict-host-key-checking=no
```

### Start all services
```bash
cd ~/mvp-healthtech/infra
docker compose -f docker-compose.demo.yml up -d
```

### Verify all services are healthy (allow ~60 s for GPU models to load)
```bash
watch -n5 'docker ps --format "table {{.Names}}\t{{.Status}}" | sort'
```

Expected healthy states:

| Container | Expected Status |
|---|---|
| infra-gateway-1 | Up (healthy) |
| infra-frontend-vite-1 | Up |
| infra-asr-1 | Up (healthy) |
| infra-llm-1 | Up (healthy) |
| infra-llm-va-1 | Up (healthy) |
| infra-tts-1 | Up (healthy) |
| infra-soap-1 | Up (healthy) |
| infra-fhir-1 | Up (healthy) |
| infra-postgres-1 | Up (healthy) |
| infra-redis-1 | Up |
| infra-otel-collector-1 | Up |
| infra-prometheus-1 | Up |
| infra-grafana-1 | Up |

### Verify nginx and HTTPS
```bash
sudo systemctl status nginx
curl -sf https://34-26-235-26.nip.io/health | python3 -m json.tool
```

### Confirm login works
```bash
curl -s -X POST https://34-26-235-26.nip.io/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"userId":"dev","password":"changeme"}' | python3 -m json.tool
# Should return {"access_token": "eyJ..."}
```

---

## 2. Graceful Shutdown

### Stop all containers (data preserved in Docker volumes)
```bash
cd ~/mvp-healthtech/infra
docker compose -f docker-compose.demo.yml down
```

### Stop the VM (saves ~$1.50/hr GPU cost)
```bash
# From local machine — NOT from inside the SSH session
gcloud compute instances stop healthtech-demo \
  --zone=us-east1-b --project=healthtech-482409
```

> **Important:** The static IP `34.26.235.26` is **reserved** and persists across stop/start.  
> SSL cert at `/etc/letsencrypt/live/34-26-235-26.nip.io/` also persists.

---

## 3. Restart a Single Service

```bash
# Restart gateway only (e.g. after a config change)
cd ~/mvp-healthtech/infra
docker compose -f docker-compose.demo.yml restart gateway

# Or rebuild + restart
docker compose -f docker-compose.demo.yml up -d --build gateway
```

---

## 4. Startup Checklist (Investor Demo)

- [ ] VM status: RUNNING
- [ ] `docker ps` shows all containers healthy
- [ ] `https://34-26-235-26.nip.io` loads the React frontend (no SSL error)
- [ ] Login with `dev` / `changeme` returns JWT
- [ ] ASR demo: upload audio → transcript appears
- [ ] SOAP note generation: clinical note renders within 15 s
- [ ] Knowledge Base: RAG search returns results

---

## 5. Monitoring URLs (once running)

| Service | URL |
|---|---|
| Frontend | https://34-26-235-26.nip.io |
| Gateway health | https://34-26-235-26.nip.io/health |
| Grafana | http://34.26.235.26:3002 (admin / $GRAFANA_PASSWORD) |
| Prometheus | http://34.26.235.26:9090 |
