# HealthTech MVP - Development Setup Guide

## Prerequisites

### Required Software
- **Node.js**: 18.x or higher ([Download](https://nodejs.org/))
- **pnpm**: 8.x or higher (install via `corepack enable`)
- **Python**: 3.11 ([Download](https://www.python.org/downloads/))
- **Docker Desktop**: Latest version ([Download](https://www.docker.com/products/docker-desktop))
- **Git**: Latest version

### Optional (for GPU workloads)
- **CUDA Toolkit**: 12.1 (if running ML services locally with GPU)
- **Kaggle Account**: Free GPU access for training/inference

---

## Quick Start (Local Development)

### 1. Clone Repository
```bash
git clone https://github.com/mansourmohamed2608/mvp-healthtech.git
cd mvp-healthtech
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env and fill in your values:
# - Twilio credentials (get from console.twilio.com)
# - JWT secret (generate strong random string)
# - Service URLs (use defaults for Docker Compose)
```

### 3. Install Gateway Dependencies
```bash
cd gateway
corepack enable
pnpm install
```

### 4. Start Services with Docker Compose
```bash
cd ../infra
docker compose up -d
```

This starts:
- Gateway (port 3000)
- Frontend (port 3001)
- Redis (port 6379)
- PostgreSQL (port 5432)

### 5. Verify Services
```bash
# Check gateway health
curl http://localhost:3000/health

# Expected response:
# {"status":"ok","timestamp":1729900000000,"uptime":5000,"environment":"development","version":"1.0.0","services":{"gateway":"up"}}

# Check metrics
curl http://localhost:3000/metrics
```

---

## Running Services Individually

### Gateway (NestJS)
```bash
cd gateway
pnpm install
pnpm start:dev
```
Available at: http://localhost:3000

### Frontend (Next.js)
```bash
cd frontend
pnpm install
pnpm dev
```
Available at: http://localhost:3001

---

## Kaggle Setup (for ML Training/Inference)

### 1. Upload LoRA Adapters
After training Whisper on Kaggle:
```bash
# Download adapter files from Kaggle:
# - adapter_config.json
# - adapter_model.safetensors
# - tokenizer files

# Place in project:
mkdir -p services/asr/lora_ckpt
# Copy adapter files to services/asr/lora_ckpt/
```

### 2. Kaggle Notebook for Inference
See `services/asr/kaggle_inference.ipynb` (added in Day 8)

---

## Development Workflow

### Run Tests
```bash
cd gateway
pnpm test
pnpm test:e2e
```

### Linting
```bash
pnpm lint
```

### Format Code
```bash
pnpm format
```

---

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 3000
# Windows:
netstat -ano | findstr :3000
# Kill process by PID
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:3000 | xargs kill -9
```

### Docker Issues
```bash
# Reset Docker containers
docker compose down -v
docker compose up -d --build
```

### pnpm Install Fails
```bash
# Clear cache
pnpm store prune
rm -rf node_modules
pnpm install
```

---

## Next Steps

- **Day 2**: Session management and authentication
- **Day 3**: Twilio webhook integration
- **Day 4-5**: ASR service setup
- **Day 6-7**: LLM integration

See `tech plan.md` for complete roadmap.
