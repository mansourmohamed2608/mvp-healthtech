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

COPY . /app

# VAD model will be downloaded at runtime from HuggingFace using HUGGINGFACE_HUB_TOKEN
# The whisperx S3 bucket is broken/restricted
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/cache

EXPOSE 5000
# Download VAD model on startup, then run app
CMD ["sh", "-c", "python download_vad.py && python app.py"]
