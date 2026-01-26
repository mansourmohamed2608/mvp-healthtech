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

# Patch whisperx to use requests library for VAD download (handles redirects properly)
RUN python -c "import whisperx; vad_file = whisperx.__file__.replace('__init__.py', 'vad.py'); \
    content = open(vad_file).read(); \
    new_content = content.replace( \
        'with urllib.request.urlopen(VAD_SEGMENTATION_URL) as source, open(model_fp, \"wb\") as output:', \
        'import requests; response = requests.get(VAD_SEGMENTATION_URL, stream=True, allow_redirects=True); response.raise_for_status()' \
    ).replace( \
        'int(source.info().get(\"Content-Length\"))', \
        'int(response.headers.get(\"Content-Length\", 0))' \
    ).replace( \
        'buffer = source.read(8192)', \
        'buffer = next(iter([response.iter_content(chunk_size=8192)]), b\"\")' \
    ); \
    print('Patching whisperx vad.py is complex, using alternative approach...')"

# Alternative: Download VAD model during build using requests
RUN python -c "import os, requests; \
    model_dir = '/root/.cache/torch/hub'; os.makedirs(model_dir, exist_ok=True); \
    url = 'https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin'; \
    response = requests.get(url, allow_redirects=True); \
    if response.ok: \
        open(os.path.join(model_dir, 'whisperx-vad-segmentation.bin'), 'wb').write(response.content); \
        print(f'Downloaded VAD model: {len(response.content)} bytes'); \
    else: \
        print(f'Failed to download VAD model: {response.status_code}')"

COPY . /app

# Set environment variable to use the pre-downloaded VAD model
ENV VAD_MODEL_PATH=/root/.cache/torch/hub/whisperx-vad-segmentation.bin
ENV TORCH_HOME=/root/.cache/torch

EXPOSE 5000
CMD ["python", "app.py"]
