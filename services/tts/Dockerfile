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

# Install gTTS (lightweight, no conflicting deps)
RUN pip install --no-cache-dir gTTS==2.5.3

# Ensure click stays >=8.2 for typer/XTTS compatibility
# (some transitive dep can downgrade it during the gTTS step above)
RUN pip install --no-cache-dir "click>=8.2.0,<9.0"

# Copy application code
COPY . .

# Expose port
EXPOSE 5002

# Run TTS service
CMD ["python", "app.py"]
