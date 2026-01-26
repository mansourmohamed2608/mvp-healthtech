# services/asr/Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install system dependencies including CA certificates for SSL
# pkg-config and libav* are required for building PyAV
# curl is needed for health checks
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        build-essential \
        ffmpeg libsndfile1 git \
        ca-certificates curl \
        pkg-config \
        libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
        && ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install Python packages (use system pip to avoid issues)
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Download VAD model during build using requests (handles redirects properly)
# The S3 URL whisperx uses returns 301 redirect that urllib doesn't handle
RUN mkdir -p /root/.cache/torch/hub && \
    python3 -c "import requests; r = requests.get('https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin', allow_redirects=True); open('/root/.cache/torch/hub/whisperx-vad-segmentation.bin', 'wb').write(r.content); print(f'VAD model: {len(r.content)} bytes')"

COPY . /app

# Set environment variable to use the pre-downloaded VAD model
ENV VAD_MODEL_PATH=/root/.cache/torch/hub/whisperx-vad-segmentation.bin
ENV TORCH_HOME=/root/.cache/torch

EXPOSE 5000
CMD ["python", "app.py"]
