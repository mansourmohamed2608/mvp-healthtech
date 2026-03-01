# Startup Runbook

## Overview

This runbook covers starting the HealthTech platform in various environments.

## Prerequisites

- Docker and Docker Compose installed
- Access to environment secrets/configuration
- Network access to required services (PostgreSQL, Redis)

---

## Local Development Startup

### 1. Quick Start (All Services)

```bash
# Navigate to project root
cd mvp-healthtech

# Start all services with Docker Compose
docker-compose up -d

# Verify all containers are running
docker-compose ps

# Check logs for any startup errors
docker-compose logs -f --tail=100
```

### 2. Selective Service Startup

```bash
# Start only infrastructure (DB, Redis, observability)
docker-compose up -d postgres redis prometheus grafana loki

# Start gateway only
docker-compose up -d gateway

# Start specific microservices
docker-compose up -d asr llm tts

# Start frontend
docker-compose up -d frontend
```

### 3. Development Mode (Hot Reload)

```bash
# Gateway (NestJS)
cd gateway
npm install
npm run start:dev

# Frontend (Vite)
cd frontend-vite
pnpm install
pnpm dev

# Python services (individual terminals)
cd services/asr
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8001
```

---

## Staging Environment Startup

### 1. Deploy from CI/CD

Staging deployments are triggered automatically when pushing to `develop` branch.

### 2. Manual Deployment

```bash
# SSH to staging server
ssh deploy@staging.healthtech.example.com

# Pull latest images
docker-compose -f docker-compose.staging.yml pull

# Deploy with zero downtime
docker-compose -f docker-compose.staging.yml up -d --remove-orphans

# Verify deployment
./scripts/health-check.sh staging
```

---

## Production Environment Startup

### 1. Standard Deployment

```bash
# Production deployments should always go through CI/CD
# Manual deployment requires approval

# SSH to production server (requires MFA)
ssh -i ~/.ssh/healthtech-prod deploy@prod.healthtech.example.com

# Verify current state
docker-compose -f docker-compose.prod.yml ps

# Pull new images (do this before maintenance window)
docker-compose -f docker-compose.prod.yml pull

# Deploy during maintenance window
docker-compose -f docker-compose.prod.yml up -d --remove-orphans
```

### 2. Blue-Green Deployment

```bash
# Start new (green) environment
docker-compose -f docker-compose.green.yml up -d

# Run smoke tests against green
./scripts/smoke-test.sh green

# Switch traffic (update load balancer)
./scripts/switch-traffic.sh green

# Verify green is serving traffic
./scripts/verify-traffic.sh green

# Stop old (blue) environment after confirmation
docker-compose -f docker-compose.blue.yml down
```

---

## Service-Specific Startup Notes

### ASR Service (WhisperX)

```bash
# Requires GPU for optimal performance
# First startup downloads model (~2GB)

# Check GPU availability
nvidia-smi

# Start with GPU support
docker-compose up -d asr

# Verify model loaded (may take 2-3 minutes on first run)
docker-compose logs asr | grep "Model loaded"
```

### LLM Service

```bash
# Requires significant GPU memory (16GB+ recommended)
# Model loading takes 3-5 minutes

# Check GPU memory
nvidia-smi --query-gpu=memory.free --format=csv

# Start LLM service
docker-compose up -d llm

# Monitor model loading
docker-compose logs -f llm
```

### TTS Service

```bash
# Requires voice model files
# First startup downloads models

docker-compose up -d tts

# Verify voices are available
curl http://localhost:8003/voices
```

---

## Health Verification Checklist

After startup, verify all services are healthy:

```bash
#!/bin/bash
# save as scripts/verify-startup.sh

SERVICES=(
  "gateway:3000"
  "asr:8001"
  "llm:8002"
  "tts:8003"
  "soap:8004"
  "fhir:8005"
)

echo "Checking service health..."

for SERVICE in "${SERVICES[@]}"; do
  NAME="${SERVICE%%:*}"
  PORT="${SERVICE##*:}"
  
  if curl -s "http://localhost:$PORT/health" > /dev/null; then
    echo "✅ $NAME is healthy"
  else
    echo "❌ $NAME is NOT healthy"
  fi
done

# Check database connectivity
echo "Checking database..."
docker-compose exec -T postgres pg_isready && echo "✅ PostgreSQL is ready" || echo "❌ PostgreSQL is NOT ready"

# Check Redis
echo "Checking Redis..."
docker-compose exec -T redis redis-cli ping | grep -q PONG && echo "✅ Redis is ready" || echo "❌ Redis is NOT ready"
```

---

## Troubleshooting Startup Issues

### Container Won't Start

```bash
# Check container logs
docker-compose logs <service-name>

# Check for port conflicts
netstat -tulpn | grep <port>

# Check resource limits
docker stats
```

### Database Connection Failed

```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Check connection string
echo $DATABASE_URL

# Test connection manually
docker-compose exec postgres psql -U healthtech -d healthtech -c "SELECT 1"
```

### GPU Not Detected

```bash
# Verify NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Restart Docker daemon if needed
sudo systemctl restart docker
```

### Out of Memory

```bash
# Check memory usage
free -h

# Check container memory limits
docker stats --no-stream

# Increase Docker memory (Docker Desktop)
# Settings > Resources > Memory
```

---

## Post-Startup Verification

1. **Run smoke tests**
   ```bash
   npm run test:e2e:smoke
   ```

2. **Check Grafana dashboards**
   - Navigate to http://localhost:3001
   - Verify metrics are being collected

3. **Test critical paths**
   ```bash
   # Test authentication
   curl -X POST http://localhost:3000/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"test@example.com","password":"test"}'
   
   # Test ASR endpoint
   curl http://localhost:3000/asr/health
   ```

4. **Monitor logs for errors**
   ```bash
   docker-compose logs -f --tail=100 | grep -i error
   ```
