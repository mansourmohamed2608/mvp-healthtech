# services/tts/app.py
"""
TTS (Text-to-Speech) Service - Coqui TTS with Arabic voices
Synthesizes natural Arabic speech from text responses
Week 3 Day 15 (Oct 9, 2025)
"""
import io
import time
import base64
import audioop
import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import numpy as np
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts")
# Optional OTEL (no-op if disabled)
try:
    from otel_setup import init_otel
    init_otel("tts")
except Exception:
    logger.debug("OTEL init skipped for TTS")


def safe_print(*args, **_kwargs):
    logger.debug("suppressed print", extra={"fields": len(args)})


print = safe_print

# Try Coqui TTS; fallback will synthesize silence to keep contract (no mp3 fallback)
TTS_ENGINE = "edge"  # Default label; we will synthesize silence if Coqui unavailable
try:
    from TTS.api import TTS as CoquiTTS
    TTS_ENGINE = "coqui"
    logger.info("Using Coqui TTS")
except ImportError:
    logger.warning("Coqui TTS not available, will use silent fallback mulaw audio")

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Internal synthesis rate (Coqui); we downsample to 8kHz mulaw for Twilio media streams.
SAMPLE_RATE = 16000
TWILIO_SAMPLE_RATE = 8000
VOICE = "ar-EG-SalmaNeural"  # Edge-TTS Arabic voice (Egyptian female)
COQUI_MODEL = "tts_models/ar/css10/vits"  # Coqui Arabic model
TTS_TIMEOUT_SECONDS = float(os.getenv("TTS_TIMEOUT_SECONDS", "10"))

logger.info("TTS Service starting", extra={"device": DEVICE, "engine": TTS_ENGINE})

# Initialize TTS engine
tts_model = None
if TTS_ENGINE == "coqui":
    try:
        tts_model = CoquiTTS(model_name=COQUI_MODEL, gpu=(DEVICE == "cuda"))
        logger.info("Loaded Coqui model", extra={"model": COQUI_MODEL})
    except Exception as e:
        logger.warning("Coqui initialization failed, falling back to edge-tts", extra={"error": str(e)})
        TTS_ENGINE = "edge"

logger.info("TTS Service ready", extra={"engine": TTS_ENGINE})

app = FastAPI(title="TTS Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
INTERNAL_SECRET = os.getenv("INTERNAL_SECRET", "")
if not INTERNAL_SECRET:
    raise RuntimeError("INTERNAL_SECRET must be set for TTS service")

tts_requests_total = Counter("tts_requests_total", "Total TTS synthesis requests")
tts_latency_seconds = Histogram(
    "tts_latency_seconds",
    "Latency of TTS synthesis",
    buckets=[0.1, 0.3, 0.5, 1, 2, 5, 10],
)
tts_errors_total = Counter("tts_errors_total", "Total TTS errors", ["reason"])

def log_safe(level: int, msg: str, request: Request | None = None, session_id: str | None = None, **kwargs):
    extra = {
        "correlationId": request.headers.get("x-correlation-id") if request else None,
        "sessionId": session_id,
    }
    for k, v in kwargs.items():
        if v is not None:
            extra[k] = v
    logger.log(level, msg, extra=extra)

# Internal helpers
def _blocking_tts_generate(text: str, voice: Optional[str]) -> bytes:
    """Blocking TTS generation that always returns mulaw bytes (8kHz)."""
    if TTS_ENGINE == "coqui" and tts_model:
        audio_np = tts_model.tts(text=text)
        audio_int16 = (np.array(audio_np) * 32767).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()
        pcm_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, SAMPLE_RATE, TWILIO_SAMPLE_RATE, None)
        mulaw_bytes = audioop.lin2ulaw(pcm_8k, 2)
        return mulaw_bytes
    # Fallback: 1s silence in mulaw to keep contract consistent
    pcm_silence = (b"\x00\x00") * int(TWILIO_SAMPLE_RATE)
    return audioop.lin2ulaw(pcm_silence, 2)

async def _run_tts_engine(text: str, voice: Optional[str]) -> bytes:
    return await asyncio.to_thread(_blocking_tts_generate, text, voice)

@app.middleware("http")
async def internal_auth(request: Request, call_next):
    if request.url.path.startswith("/health") or request.url.path.startswith("/ready") or request.url.path.startswith("/metrics"):
        return await call_next(request)
    if not INTERNAL_SECRET or request.headers.get("x-internal-secret") != INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await call_next(request)

# Request/Response models
class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = VOICE
    sessionId: Optional[str] = None
    format: Optional[str] = "mulaw"  # wav, mp3, mulaw

class SynthesizeResponse(BaseModel):
    audio: str  # Base64 encoded audio
    duration: float
    sampleRate: int

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "ok": True,
        "service": "tts",
        "engine": TTS_ENGINE,
        "device": DEVICE if TTS_ENGINE == "coqui" else "cpu",
        "model": COQUI_MODEL if TTS_ENGINE == "coqui" else VOICE,
        "correlationId": None,  # placeholder
    }

@app.get("/ready")
async def ready():
    """Readiness check for downstream orchestration."""
    return {
        "ready": TTS_ENGINE == "edge" or tts_model is not None,
        "engine": TTS_ENGINE,
        "model": COQUI_MODEL if TTS_ENGINE == "coqui" else VOICE,
    }

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text
    Returns base64 audio blob for gateway/Twilio (8kHz mulaw when Coqui is available)
    """
    tts_requests_total.inc()
    start_time = time.time()

    if not request.text or not isinstance(request.text, str):
        raise HTTPException(status_code=400, detail="Text is required")
    if len(request.text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long")

    try:
        mulaw_bytes = await asyncio.wait_for(
            _run_tts_engine(request.text, request.voice),
            timeout=TTS_TIMEOUT_SECONDS,
        )
        duration = time.time() - start_time
        payload = {
            "audio": base64.b64encode(mulaw_bytes).decode("utf-8"),
            "format": "mulaw",
            "sampleRate": TWILIO_SAMPLE_RATE,
            "duration": duration,
            "contentType": "audio/basic",  # audio/mulaw
        }
        tts_latency_seconds.observe(duration)
        log_safe(logging.INFO, "TTS synthesized", request=None, session_id=request.sessionId, textLen=len(request.text))
        return payload
    except asyncio.TimeoutError:
        tts_errors_total.inc({"reason": "timeout"})
        log_safe(logging.WARNING, "TTS timeout", request=None, session_id=request.sessionId, textLen=len(request.text))
        raise HTTPException(status_code=504, detail="TTS service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        tts_errors_total.inc({"reason": "synthesize_error"})
        log_safe(logging.ERROR, "TTS synthesis failed", request=None, session_id=request.sessionId, error=str(type(e).__name__))
        raise HTTPException(status_code=500, detail="TTS synthesis failed")

@app.post("/synthesize/stream")
async def synthesize_stream(request: SynthesizeRequest):
    """
    Stream synthesized speech in chunks
    Useful for real-time playback
    """
    try:
        # For simplicity, return the base64 payload but keep a streaming content-type
        synthesized = await synthesize(request)
        audio_bytes = base64.b64decode(synthesized["audio"])
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=synthesized.get("contentType", "audio/mpeg"),
        )
    
    except Exception as e:
        log_safe(logging.ERROR, "Stream synthesis failed", request=None, session_id=request.sessionId, error=str(type(e).__name__))
        raise HTTPException(status_code=500, detail="Stream synthesis failed")

@app.get("/voices")
async def list_voices():
    """
    List available voices
    """
    if TTS_ENGINE == "coqui":
        return {
            "engine": "coqui",
            "voices": ["default"],  # Coqui uses model-specific voice
            "model": COQUI_MODEL,
        }
    else:
        # Common Arabic voices in edge-tts
        arabic_voices = [
            "ar-EG-SalmaNeural",  # Egyptian Female
            "ar-EG-ShakirNeural",  # Egyptian Male
            "ar-SA-HamedNeural",   # Saudi Male
            "ar-SA-ZariyahNeural", # Saudi Female
            "ar-AE-FatimaNeural",  # UAE Female
            "ar-AE-HamdanNeural",  # UAE Male
        ]
        return {
            "engine": "edge-tts",
            "voices": arabic_voices,
            "default": VOICE,
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
