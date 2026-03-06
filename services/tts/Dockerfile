# Python base for FastAPI services
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install edge-tts as a separate layer so base deps stay cached
RUN pip install --no-cache-dir edge-tts==6.1.12

# Copy application code
COPY . .

# Expose port
EXPOSE 5002

# Run TTS service
CMD ["python", "app.py"]
