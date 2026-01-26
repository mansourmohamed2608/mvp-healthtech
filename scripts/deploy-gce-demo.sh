#!/bin/bash
# ============================================================
# GCE L4 GPU Deployment Script for Investor Demo
# ============================================================
# This script creates a Google Compute Engine VM with L4 GPU
# and deploys the HealthTech Voice Agent system.
#
# Requirements:
#   - Google Cloud SDK (gcloud) installed and authenticated
#   - A GCP project with billing enabled
#   - GPU quota for L4 in your region
#
# Usage:
#   chmod +x deploy-gce-demo.sh
#   ./deploy-gce-demo.sh
# ============================================================

set -e

# ============ CONFIGURATION ============
# CHANGE THESE VALUES
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="healthtech-demo"
MACHINE_TYPE="g2-standard-8"  # 8 vCPUs, 32GB RAM, 1x L4 GPU (24GB VRAM)
# For 2x L4 GPUs, use: g2-standard-16 with --accelerator=count=2

# Boot disk
BOOT_DISK_SIZE="100GB"
BOOT_DISK_TYPE="pd-ssd"

# Image - Ubuntu 22.04 with NVIDIA drivers pre-installed
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

# ============ COLORS ============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  HealthTech GCE L4 Demo Deployment${NC}"
echo -e "${GREEN}============================================${NC}"

# ============ PRE-FLIGHT CHECKS ============
echo -e "\n${YELLOW}[1/7] Pre-flight checks...${NC}"

# Check gcloud
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found. Install from https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

# Check project
if [ "$PROJECT_ID" == "your-project-id" ]; then
    echo -e "${RED}Error: Set GCP_PROJECT_ID environment variable${NC}"
    echo "  export GCP_PROJECT_ID=your-actual-project-id"
    exit 1
fi

gcloud config set project "$PROJECT_ID"
echo -e "${GREEN}✓ Using project: $PROJECT_ID${NC}"

# Check GPU quota
echo -e "\n${YELLOW}Checking L4 GPU quota in $ZONE...${NC}"
QUOTA=$(gcloud compute regions describe "${ZONE%-*}" \
    --format="value(quotas[name=NVIDIA_L4_GPUS].limit)" 2>/dev/null || echo "0")
if [ "$QUOTA" == "0" ] || [ -z "$QUOTA" ]; then
    echo -e "${YELLOW}Warning: L4 GPU quota may not be set. Request quota at:${NC}"
    echo "  https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID"
fi

# ============ CREATE FIREWALL RULES ============
echo -e "\n${YELLOW}[2/7] Creating firewall rules...${NC}"

# Allow HTTP/HTTPS
gcloud compute firewall-rules create healthtech-allow-web \
    --allow=tcp:80,tcp:443,tcp:3000,tcp:5173 \
    --target-tags=healthtech-demo \
    --description="Allow web traffic to HealthTech demo" \
    --quiet 2>/dev/null || echo "Firewall rule already exists"

echo -e "${GREEN}✓ Firewall rules configured${NC}"

# ============ CREATE VM INSTANCE ============
echo -e "\n${YELLOW}[3/7] Creating GCE instance with L4 GPU...${NC}"

# Check if instance exists
if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &>/dev/null; then
    echo -e "${YELLOW}Instance $INSTANCE_NAME already exists. Delete it first or use a different name.${NC}"
    read -p "Delete existing instance? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
    else
        exit 1
    fi
fi

# Create the VM
gcloud compute instances create "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=nvidia-l4,count=1" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --maintenance-policy=TERMINATE \
    --tags=healthtech-demo \
    --metadata=startup-script='#!/bin/bash
# This runs on first boot
touch /var/log/healthtech-setup.log
echo "Setup started at $(date)" >> /var/log/healthtech-setup.log
'

echo -e "${GREEN}✓ VM created: $INSTANCE_NAME${NC}"

# Wait for VM to be ready
echo -e "\n${YELLOW}Waiting for VM to be ready...${NC}"
sleep 30

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

echo -e "${GREEN}✓ External IP: $EXTERNAL_IP${NC}"

# ============ SETUP VM ============
echo -e "\n${YELLOW}[4/7] Installing Docker and NVIDIA drivers on VM...${NC}"

gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command='
#!/bin/bash
set -e

echo "=== Installing NVIDIA drivers ==="
# Install NVIDIA driver
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r)

# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA driver and container toolkit
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 nvidia-container-toolkit

echo "=== Installing Docker ==="
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Configure Docker for NVIDIA
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "=== Installing Docker Compose ==="
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

echo "=== Setup complete! Rebooting for driver activation ==="
sudo reboot
'

echo -e "${YELLOW}VM is rebooting to activate NVIDIA drivers...${NC}"
echo "Waiting 60 seconds..."
sleep 60

# Wait for VM to come back online
echo -e "${YELLOW}Waiting for VM to come back online...${NC}"
for i in {1..30}; do
    if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="echo 'VM is ready'" 2>/dev/null; then
        break
    fi
    echo "Attempt $i/30..."
    sleep 10
done

# ============ VERIFY GPU ============
echo -e "\n${YELLOW}[5/7] Verifying GPU setup...${NC}"

gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command='
nvidia-smi
'

echo -e "${GREEN}✓ GPU verified${NC}"

# ============ CLONE AND SETUP PROJECT ============
echo -e "\n${YELLOW}[6/7] Setting up HealthTech project...${NC}"

gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command='
#!/bin/bash
set -e

# Clone the repository
cd ~
if [ ! -d "mvp-healthtech" ]; then
    git clone https://github.com/mansourmohamed2608/mvp-healthtech.git
fi

echo "Project cloned. Waiting for .env upload..."
'

echo -e "${GREEN}✓ Project cloned${NC}"

# Upload your local .env file
echo -e "\n${YELLOW}Uploading your .env file...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../infra/.env"

if [ -f "$ENV_FILE" ]; then
    gcloud compute scp "$ENV_FILE" "$INSTANCE_NAME":~/mvp-healthtech/infra/.env --zone="$ZONE"
    echo -e "${GREEN}✓ .env file uploaded${NC}"
else
    echo -e "${RED}✗ No .env file found at $ENV_FILE${NC}"
    echo "You'll need to manually create ~/mvp-healthtech/infra/.env on the VM"
fi

# ============ PRINT NEXT STEPS ============
echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}  DEPLOYMENT COMPLETE!${NC}"
echo -e "${GREEN}============================================${NC}"

echo -e "\n${YELLOW}NEXT STEPS:${NC}"
echo ""
echo "1. SSH into the VM:"
echo -e "   ${GREEN}gcloud compute ssh $INSTANCE_NAME --zone=$ZONE${NC}"
echo ""
echo "2. Start the services (first time takes 10-15 min for model downloads):"
echo -e "   ${GREEN}cd ~/mvp-healthtech/infra${NC}"
echo -e "   ${GREEN}docker-compose -f docker-compose.demo.yml up -d${NC}"
echo ""
echo "4. Access the demo:"
echo -e "   Frontend: ${GREEN}http://$EXTERNAL_IP${NC}"
echo -e "   API:      ${GREEN}http://$EXTERNAL_IP:3000${NC}"
echo -e "   Grafana:  ${GREEN}http://$EXTERNAL_IP:3002${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT: Update your Twilio webhook URLs to:${NC}"
echo "   Voice URL: http://$EXTERNAL_IP:3000/api/twilio/voice/incoming"
echo "   Status URL: http://$EXTERNAL_IP:3000/api/twilio/voice/status"
echo ""
echo -e "${YELLOW}To stop the demo:${NC}"
echo "   docker-compose -f docker-compose.demo.yml down"
echo ""
echo -e "${YELLOW}To delete the VM when done:${NC}"
echo "   gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
echo -e "${YELLOW}Estimated cost: ~\$1.50/hour (L4 GPU + VM)${NC}"
