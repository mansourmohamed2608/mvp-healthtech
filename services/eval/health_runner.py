#!/usr/bin/env python3
"""Simple smoke to hit all /health endpoints via docker-compose defaults."""
import httpx

services = {
    "gateway": "http://localhost:3000/health",
    "asr": "http://localhost:5000/health",
    "llm": "http://localhost:5001/health",
    "tts": "http://localhost:5002/health",
    "soap": "http://localhost:5003/health",
    "fhir": "http://localhost:5004/health",
}

for name, url in services.items():
    try:
        resp = httpx.get(url, timeout=5)
        print(name, resp.status_code, resp.json())
    except Exception as e:
        print(name, "ERROR", e)
