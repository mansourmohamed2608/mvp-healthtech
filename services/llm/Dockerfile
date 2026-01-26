FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python/pip
RUN rm -f /usr/bin/python /usr/bin/pip && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5001
CMD ["python", "app.py"]
