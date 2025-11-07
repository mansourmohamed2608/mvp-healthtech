# HealthTech MVP - Docker Compose Setup
# Complete 8-service architecture

## 🚀 Quick Start with Docker

### Prerequisites:
- Docker Desktop installed and running
- 16GB+ RAM recommended
- GPU optional (will use CPU if not available)

---

## ⚡ One-Command Startup

```bash
# Navigate to project
cd D:\Downloads\HealthTech\mvp-healthtech\infra

# Start all services with Docker Compose
docker-compose up --build
```

That's it! All 8 services will start:
- ✅ Gateway (NestJS) - Port 3000
- ✅ Frontend (Next.js) - Port 3001  
- ✅ ASR Service (Whisper) - Port 5000
- ✅ LLM Service (MMed-Llama-3-8B) - Port 5001
- ✅ TTS Service (edge-tts) - Port 5002
- ✅ SOAP Service - Port 5003
- ✅ FHIR Service - Port 5004
- ✅ Redis - Port 6379

---

## 📋 Step-by-Step Docker Setup

### Step 1: First Time Build (15-20 minutes)

```bash
cd D:\Downloads\HealthTech\mvp-healthtech\infra

# Build all Docker images
docker-compose build

# This will:
# - Build 7 Docker images
# - Download base images (~5GB)
# - Install all dependencies
# - Cache models in volumes
```

**⏱️ Build Time:**
- Gateway: ~2 min
- Frontend: ~2 min
- ASR: ~5 min (downloads Whisper ~3GB)
- LLM: ~10 min (downloads MMed-Llama ~8GB)
- TTS: ~1 min
- SOAP: ~1 min
- FHIR: ~1 min

---

### Step 2: Start All Services

```bash
# Start in foreground (see all logs)
docker-compose up

# OR start in background (detached)
docker-compose up -d
```

**⏱️ Startup Time:**
- First time: ~5-10 min (models load)
- Subsequent: ~2-3 min

---

### Step 3: Check Service Health

```bash
# Check all running containers
docker-compose ps

# Expected output:
# NAME                STATUS              PORTS
# infra-gateway-1     Up 2 minutes        0.0.0.0:3000->3000/tcp
# infra-frontend-1    Up 2 minutes        0.0.0.0:3001->3000/tcp
# infra-asr-1         Up 2 minutes        0.0.0.0:5000->5000/tcp
# infra-llm-1         Up 2 minutes        0.0.0.0:5001->5001/tcp
# infra-tts-1         Up 2 minutes        0.0.0.0:5002->5002/tcp
# infra-soap-1        Up 2 minutes        0.0.0.0:5003->5003/tcp
# infra-fhir-1        Up 2 minutes        0.0.0.0:5004->5004/tcp
# infra-redis-1       Up 2 minutes        0.0.0.0:6379->6379/tcp
```

---

### Step 4: Test Health Endpoints

```bash
# Gateway
curl http://localhost:3000/health

# ASR
curl http://localhost:5000/health

# LLM
curl http://localhost:5001/health

# TTS
curl http://localhost:5002/health

# SOAP
curl http://localhost:5003/health

# FHIR
curl http://localhost:5004/health
```

All should return `{"status":"healthy"}` or similar.

---

### Step 5: Access Your Application

**Frontend:**
```
http://localhost:3001
```

**Voice Client:**
```
http://localhost:3001/voice
```

**Clinical Notes:**
```
http://localhost:3001/clinical-notes
```

**Gateway API:**
```
http://localhost:3000/api
```

---

## 🛑 Stop Services

```bash
# Stop all services (keeps data)
docker-compose stop

# Stop and remove containers (keeps images/volumes)
docker-compose down

# Remove everything including volumes (fresh start)
docker-compose down -v
```

---

## 📊 View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f gateway
docker-compose logs -f llm
docker-compose logs -f asr

# Last 100 lines
docker-compose logs --tail=100
```

---

## 🔄 Restart a Single Service

```bash
# Restart just one service
docker-compose restart gateway

# Rebuild and restart
docker-compose up -d --build gateway
```

---

## 🧹 Cleanup Commands

```bash
# Remove stopped containers
docker-compose rm

# Remove all unused images
docker image prune -a

# Remove all unused volumes
docker volume prune

# See disk usage
docker system df

# Full cleanup (WARNING: removes everything)
docker system prune -a --volumes
```

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file in `infra/` directory:

```bash
# infra/.env

# FHIR Configuration (Week 5)
FHIR_BASE_URL=https://your-fhir-server.com/api
FHIR_CLIENT_ID=your_client_id
FHIR_CLIENT_SECRET=your_client_secret

# Gateway Configuration
JWT_SECRET=your_jwt_secret_here
TWILIO_AUTH_TOKEN=your_twilio_token

# Optional: GPU Configuration
NVIDIA_VISIBLE_DEVICES=0  # Use GPU 0
```

Then use in docker-compose:
```bash
docker-compose --env-file .env up
```

---

## 🐛 Troubleshooting

### Problem 1: "Port already in use"
```bash
# Find process using port
netstat -ano | findstr :3000

# Stop Docker containers
docker-compose down

# Or change port in docker-compose.yml
ports:
  - "3002:3000"  # Use port 3002 instead
```

### Problem 2: Out of Memory
```bash
# Limit service memory in docker-compose.yml
services:
  llm:
    deploy:
      resources:
        limits:
          memory: 10G

# Or increase Docker Desktop memory:
# Docker Desktop → Settings → Resources → Memory → 12GB+
```

### Problem 3: GPU Not Detected
```bash
# Check if Docker can see GPU
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# If not working, remove GPU requirements:
# Edit docker-compose.yml and remove:
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

# Services will use CPU (slower but works)
```

### Problem 4: Build Fails
```bash
# Clean build
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### Problem 5: Service Won't Start
```bash
# Check logs for specific service
docker-compose logs llm

# Common issues:
# - Missing dependencies: Check Dockerfile
# - Wrong port: Check docker-compose.yml ports
# - Model download failed: Check internet connection
```

---

## 📈 Resource Usage

| Service | CPU | RAM | GPU VRAM | Disk |
|---------|-----|-----|----------|------|
| Gateway | 10% | 200MB | - | 100MB |
| Frontend | 10% | 300MB | - | 200MB |
| ASR | 20% | 2GB | 1.5GB | 3GB |
| LLM | 30% | 8GB | 3.8GB | 8GB |
| TTS | 10% | 100MB | - | 100MB |
| SOAP | 5% | 100MB | - | 50MB |
| FHIR | 5% | 100MB | - | 50MB |
| Redis | 2% | 50MB | - | 50MB |
| **TOTAL** | **~90%** | **~11GB** | **~5.3GB** | **~12GB** |

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 16GB
- Disk: 20GB free
- GPU: GTX 1050 4GB (optional, will use CPU if not available)

---

## 🎯 Docker vs Manual Comparison

| Aspect | Docker Compose | Manual (7 Terminals) |
|--------|---------------|---------------------|
| **Setup** | 1 command | 7 separate commands |
| **Startup** | ~3 min | ~5 min |
| **Management** | Easy (docker-compose) | Complex (track 7 windows) |
| **Logs** | Centralized | Scattered |
| **Stop** | 1 command | Close 7 windows |
| **Networking** | Automatic | Manual configuration |
| **Production Ready** | ✅ Yes | ❌ No |
| **Portability** | ✅ Runs anywhere | ❌ Local only |
| **Recommended** | ✅ **YES** | ⚠️ Only for debugging |

---

## 🚀 Production Deployment

### Deploy to Cloud with Docker:

```bash
# Push images to Docker Hub
docker tag infra-gateway username/healthtech-gateway:latest
docker push username/healthtech-gateway:latest

# Deploy with Docker Compose on server
ssh your-server
git clone your-repo
cd mvp-healthtech/infra
docker-compose -f docker-compose.prod.yml up -d
```

---

## ✅ Success Checklist

- [ ] Docker Desktop installed and running
- [ ] `docker-compose build` completed successfully
- [ ] `docker-compose up` shows no errors
- [ ] All 8 services show "Up" status
- [ ] Health endpoints return healthy
- [ ] http://localhost:3001 loads
- [ ] Voice client page loads
- [ ] Clinical notes page loads
- [ ] Can record audio and generate SOAP notes

**Your Docker setup is ready! 🎉**

---

## 📝 Quick Reference

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down

# Restart single service
docker-compose restart gateway

# Rebuild after code changes
docker-compose up -d --build

# Check status
docker-compose ps

# Access shell in container
docker-compose exec gateway sh
docker-compose exec llm bash
```

**Docker makes everything 10x easier! 🐳**
