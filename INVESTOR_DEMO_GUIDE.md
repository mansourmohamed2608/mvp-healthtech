# 🚀 Investor Demo Deployment Guide

Deploy HealthTech to Google Compute Engine with L4 GPU for live investor demonstrations.

## Quick Start (15 minutes)

### Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated
- Twilio account (for voice calls)
- HuggingFace account (for model downloads)

### Step 1: Deploy VM

```bash
# From your local machine
cd mvp-healthtech/scripts
chmod +x deploy-gce-demo.sh
./deploy-gce-demo.sh
```

This creates a GCE VM with:
- **Machine**: g2-standard-8 (8 vCPU, 32GB RAM)
- **GPU**: NVIDIA L4 (24GB VRAM)
- **Cost**: ~$1.50/hour
- **Region**: us-central1-a

### Step 2: Configure Credentials

SSH into the VM:
```bash
gcloud compute ssh healthtech-demo --zone=us-central1-a
```

Edit the environment file:
```bash
nano ~/mvp-healthtech/infra/.env
```

**Required credentials:**
| Variable | Get from |
|----------|----------|
| `HUGGINGFACE_HUB_TOKEN` | https://huggingface.co/settings/tokens |
| `TWILIO_ACCOUNT_SID` | https://console.twilio.com |
| `TWILIO_AUTH_TOKEN` | Twilio Console → Account Info |
| `TWILIO_API_KEY` | Twilio Console → API Keys |
| `TWILIO_API_SECRET` | Twilio Console → API Keys |
| `TWILIO_PHONE_NUMBER` | Your Twilio number |
| `TWILIO_TWIML_APP_SID` | Twilio Console → TwiML Apps |

### Step 3: Start Services

```bash
cd ~/mvp-healthtech/infra
docker-compose -f docker-compose.demo.yml up -d
```

**First startup takes 10-15 minutes** (downloading WhisperX large-v3 and MMed-Llama-3-8B).

Monitor progress:
```bash
docker-compose -f docker-compose.demo.yml logs -f
```

### Step 4: Configure Twilio Webhooks

Get your VM's external IP:
```bash
curl ifconfig.me
```

In Twilio Console → Phone Numbers → Your Number:
- **Voice URL**: `http://YOUR_IP:3000/api/twilio/voice/incoming`
- **Status Callback**: `http://YOUR_IP:3000/api/twilio/voice/status`

---

## Access Points

| Service | URL |
|---------|-----|
| **Frontend** | http://YOUR_IP (port 80) |
| **API Gateway** | http://YOUR_IP:3000 |
| **API Docs** | http://YOUR_IP:3000/api |
| **Grafana** | http://YOUR_IP:3002 (admin/demo123) |

---

## Demo Flow

### 1. Web Interface Demo
1. Open http://YOUR_IP in browser
2. Create a patient session
3. Use browser microphone for voice input
4. Show real-time transcription
5. Generate SOAP note

### 2. Phone Call Demo
1. Call your Twilio phone number
2. Speak medical conversation
3. Show real-time transcription in dashboard
4. Display generated SOAP note

### 3. Latency Showcase
Expected performance with L4 GPU:
| Component | Target | Actual |
|-----------|--------|--------|
| ASR (WhisperX) | <500ms | ~250ms |
| LLM (SOAP Gen) | <2s | ~1.2s |
| TTS (edge-tts) | <500ms | ~320ms |
| **Total E2E** | <3s | ~1.8s |

---

## GPU Memory Usage

L4 has 24GB VRAM, we use ~8GB:
```
+-------------+--------+-----------+
| Model       | VRAM   | Load Time |
+-------------+--------+-----------+
| WhisperX    | ~3.5GB | 30s       |
| MMed-Llama  | ~4.0GB | 60s       |
| (Headroom)  | 16.5GB | -         |
+-------------+--------+-----------+
```

---

## Troubleshooting

### Models not loading
```bash
# Check GPU is visible
nvidia-smi

# Check container logs
docker-compose -f docker-compose.demo.yml logs asr
docker-compose -f docker-compose.demo.yml logs llm
```

### HuggingFace rate limited
Add your token to `.env`:
```
HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxx
```

### Services unhealthy
```bash
# Restart with longer startup time
docker-compose -f docker-compose.demo.yml down
docker-compose -f docker-compose.demo.yml up -d

# Wait 5 minutes for models to load
```

### Twilio calls not working
1. Verify webhook URLs are HTTP (not HTTPS) for demo
2. Check firewall allows port 3000
3. Verify Twilio credentials in `.env`

---

## Cleanup

**IMPORTANT**: Delete VM after demo to stop billing!

```bash
# From local machine
gcloud compute instances delete healthtech-demo --zone=us-central1-a
```

Or stop to keep data but stop billing:
```bash
gcloud compute instances stop healthtech-demo --zone=us-central1-a
```

---

## Cost Estimate

| Resource | Per Hour | 8-Hour Demo Day |
|----------|----------|-----------------|
| g2-standard-8 (L4) | $1.33 | $10.64 |
| Boot Disk (100GB SSD) | $0.02 | $0.16 |
| Network Egress | ~$0.10 | ~$0.80 |
| **Total** | **~$1.50** | **~$12** |

💡 **Tip**: Stop the VM when not in use to save costs!
