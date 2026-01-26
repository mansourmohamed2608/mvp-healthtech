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
import inspect
import wave
from pathlib import Path
from threading import Lock
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
def safe_print(*args, **_kwargs):
    logger.debug("suppressed print", extra={"fields": len(args)})


print = safe_print

# XTTS (EGTTS + Saudi) helpers
XTTS_AVAILABLE = False
try:
    from huggingface_hub import hf_hub_download
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    XTTS_AVAILABLE = True
except Exception:
    hf_hub_download = None
    XttsConfig = None
    Xtts = None
    XttsAudioConfig = None
    XttsArgs = None
    BaseDatasetConfig = None
    XTTS_AVAILABLE = False

# Try Coqui TTS; fallback to Edge only if explicitly allowed
TTS_ENGINE = os.getenv("TTS_ENGINE", "xtts").lower()
try:
    from TTS.api import TTS as CoquiTTS
    logger.info("Coqui TTS available")
except ImportError:
    CoquiTTS = None
    logger.warning("Coqui TTS not available, attempting Edge TTS")

EDGE_TTS_ALLOWED = os.getenv("EDGE_TTS_ALLOWED", "false").lower() == "true"
try:
    if EDGE_TTS_ALLOWED:
        import edge_tts
        EDGE_AVAILABLE = True
    else:
        EDGE_AVAILABLE = False
except ImportError:
    EDGE_AVAILABLE = False
    logger.warning("edge-tts not available, will use silent fallback mulaw audio")

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Internal synthesis rate (Coqui); we downsample to 8kHz mulaw for Twilio media streams.
SAMPLE_RATE = 16000
TWILIO_SAMPLE_RATE = 8000
XTTS_SAMPLE_RATE = 24000
VOICE = os.getenv("TTS_EDGE_VOICE", "ar-EG-SalmaNeural")  # Edge-TTS Arabic voice (only if EDGE_TTS_ALLOWED)
COQUI_MODEL = "tts_models/ar/css10/vits"  # Coqui Arabic model
EGTTS_VOICE_ID = os.getenv("EGTTS_VOICE_ID", "egtts")
SAUDI_VOICE_ID = os.getenv("SAUDI_VOICE_ID", "saudi-tts")
DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", EGTTS_VOICE_ID)
EGTTS_REPO_ID = os.getenv("EGTTS_REPO_ID", "OmarSamir/EGTTS-V0.1")
EGTTS_REVISION = os.getenv("EGTTS_REVISION") or None
EGTTS_SPEAKER_FILE = os.getenv("EGTTS_SPEAKER_FILE", "speaker_reference.wav")
EGTTS_TEMPERATURE = float(os.getenv("EGTTS_TEMPERATURE", "0.55"))
SAUDI_REPO_ID = os.getenv("SAUDI_TTS_REPO_ID", "AhmedEladl/saudi-tts")
SAUDI_REVISION = os.getenv("SAUDI_TTS_REVISION", "f99ffe0")
SAUDI_SPEAKER_FILE = os.getenv("SAUDI_TTS_SPEAKER_FILE", "speaker.wav")
SAUDI_TEMPERATURE = float(os.getenv("SAUDI_TEMPERATURE", "0.50"))
TTS_MODEL_DIR = Path(os.getenv("TTS_MODEL_DIR", str(Path(__file__).parent / "models")))
TTS_TIMEOUT_SECONDS = float(os.getenv("TTS_TIMEOUT_SECONDS", "10"))
EDGE_OUTPUT_FORMAT = "riff-16khz-16bit-mono-pcm"

if TTS_ENGINE == "xtts" and not XTTS_AVAILABLE:
    logger.warning("XTTS not available, falling back", extra={"engine": TTS_ENGINE})
    TTS_ENGINE = "coqui" if CoquiTTS else "none"
if TTS_ENGINE == "coqui" and not CoquiTTS:
    TTS_ENGINE = "none"
if TTS_ENGINE == "edge" and not EDGE_AVAILABLE:
    TTS_ENGINE = "none"

logger.info("TTS Service starting", extra={"device": DEVICE, "engine": TTS_ENGINE})

# Initialize TTS engine
tts_model = None
if TTS_ENGINE == "coqui" and CoquiTTS:
    try:
        tts_model = CoquiTTS(model_name=COQUI_MODEL, gpu=(DEVICE == "cuda"))
        logger.info("Loaded Coqui model", extra={"model": COQUI_MODEL})
    except Exception as e:
        logger.warning("Coqui initialization failed, disabling TTS engine", extra={"error": str(e)})
        TTS_ENGINE = "none"

logger.info("TTS Service ready", extra={"engine": TTS_ENGINE})

app = FastAPI(title="TTS Service", version="1.0.0")

# CORS: configurable via env, default to localhost only
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ALLOWED_ORIGINS],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "x-internal-secret", "x-correlation-id"],
)

# Optional OTEL (no-op if disabled)
try:
    from otel_setup import init_otel
    init_otel("tts", app=app)
except Exception:
    logger.debug("OTEL init skipped for TTS")
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
def _silence_mulaw() -> bytes:
    pcm_silence = (b"\x00\x00") * int(TWILIO_SAMPLE_RATE)
    return audioop.lin2ulaw(pcm_silence, 2)

EDGE_VOICES = {
    "ar-EG-SalmaNeural",
    "ar-EG-ShakirNeural",
    "ar-SA-HamedNeural",
    "ar-SA-ZariyahNeural",
    "ar-AE-FatimaNeural",
    "ar-AE-HamdanNeural",
}

_egtts_state = {"model": None, "latents": None, "embedding": None, "device": None}
_saudi_state = {"model": None, "latents": None, "embedding": None, "device": None}
_xtts_lock = Lock()


def _register_safe_globals() -> None:
    if not hasattr(torch, "serialization") or not hasattr(torch.serialization, "add_safe_globals"):
        return
    try:
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
    except Exception:
        return


def _hf_download(repo_id: str, filename: str, revision: str | None, local_dir: Path) -> Path:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub not installed")
    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id,
            filename,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
    )


def _load_xtts_state(kind: str) -> dict:
    if not XTTS_AVAILABLE:
        raise RuntimeError("XTTS dependencies not available")
    if kind == "egtts":
        state = _egtts_state
        repo_id = EGTTS_REPO_ID
        revision = EGTTS_REVISION
        speaker_file = EGTTS_SPEAKER_FILE
        model_dir = TTS_MODEL_DIR / "egtts_v01"
    else:
        state = _saudi_state
        repo_id = SAUDI_REPO_ID
        revision = SAUDI_REVISION
        speaker_file = SAUDI_SPEAKER_FILE
        model_dir = TTS_MODEL_DIR / "saudi_tts"

    if state["model"] is not None:
        return state

    with _xtts_lock:
        if state["model"] is not None:
            return state
        _register_safe_globals()
        config_path = _hf_download(repo_id, "config.json", revision, model_dir)
        vocab_path = _hf_download(repo_id, "vocab.json", revision, model_dir)
        _hf_download(repo_id, "model.pth", revision, model_dir)
        speaker_path = _hf_download(repo_id, speaker_file, revision, model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = XttsConfig()
        config.load_json(str(config_path))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=str(model_dir),
            vocab_path=str(vocab_path),
            use_deepspeed=False,
            eval=True,
        )
        model.to(device)
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[str(speaker_path)]
        )
        state.update(
            {
                "model": model,
                "latents": gpt_cond_latent,
                "embedding": speaker_embedding,
                "device": device,
            }
        )
    return state


def _xtts_generate(text: str, kind: str, temperature: float) -> bytes:
    state = _load_xtts_state(kind)
    model = state["model"]
    gpt_cond_latent = state["latents"]
    speaker_embedding = state["embedding"]
    if model is None or gpt_cond_latent is None or speaker_embedding is None:
        return _silence_mulaw()
    out = model.inference(
        text=text,
        language="ar",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=temperature,
    )
    wav = out.get("wav", [])
    if not wav:
        return _silence_mulaw()
    audio_np = np.array(wav)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()
    pcm_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, XTTS_SAMPLE_RATE, TWILIO_SAMPLE_RATE, None)
    return audioop.lin2ulaw(pcm_8k, 2)


def _coqui_generate(text: str) -> bytes:
    audio_np = tts_model.tts(text=text)
    audio_int16 = (np.array(audio_np) * 32767).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()
    pcm_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, SAMPLE_RATE, TWILIO_SAMPLE_RATE, None)
    return audioop.lin2ulaw(pcm_8k, 2)

def _normalize_voice_id(voice: Optional[str]) -> str:
    if not voice:
        return DEFAULT_VOICE
    raw = voice.strip()
    lowered = raw.lower()
    if lowered == "auto":
        return DEFAULT_VOICE
    if lowered in {"egypt", "egyptian", "eg", EGTTS_VOICE_ID.lower()}:
        return EGTTS_VOICE_ID
    if lowered in {"saudi", "ksa", "gulf", "gcc", SAUDI_VOICE_ID.lower(), "saudi_tts"}:
        return SAUDI_VOICE_ID
    if lowered in {v.lower() for v in EDGE_VOICES}:
        return DEFAULT_VOICE
    return raw


def _edge_kwargs() -> dict:
    if not EDGE_AVAILABLE:
        return {}
    try:
        params = inspect.signature(edge_tts.Communicate.__init__).parameters
        if "output_format" in params:
            return {"output_format": EDGE_OUTPUT_FORMAT}
        if "format" in params:
            return {"format": EDGE_OUTPUT_FORMAT}
    except Exception:
        return {}
    return {}


def _decode_wave_bytes(wav_bytes: bytes) -> tuple[bytes, int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    if channels > 1:
        pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
    return pcm, rate, sample_width


async def _edge_generate(text: str, voice: Optional[str]) -> bytes:
    if not EDGE_AVAILABLE:
        return _silence_mulaw()
    voice_id = voice if voice in EDGE_VOICES else VOICE
    audio_bytes = b""
    communicate = edge_tts.Communicate(text=text, voice=voice_id, **_edge_kwargs())
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            audio_bytes += chunk.get("data", b"")
    if not audio_bytes:
        return _silence_mulaw()
    try:
        pcm_bytes, rate, sample_width = _decode_wave_bytes(audio_bytes)
        pcm_8k, _ = audioop.ratecv(pcm_bytes, sample_width, 1, rate, TWILIO_SAMPLE_RATE, None)
        return audioop.lin2ulaw(pcm_8k, sample_width)
    except Exception:
        return _silence_mulaw()


def _should_use_edge(voice: Optional[str]) -> bool:
    if not EDGE_AVAILABLE:
        return False
    voice_id = _normalize_voice_id(voice)
    if voice_id in EDGE_VOICES:
        return True
    if TTS_ENGINE == "edge":
        return True
    return False


async def _run_tts_engine(text: str, voice: Optional[str]) -> bytes:
    voice_id = _normalize_voice_id(voice)
    if voice_id == EGTTS_VOICE_ID:
        try:
            return await asyncio.to_thread(_xtts_generate, text, "egtts", EGTTS_TEMPERATURE)
        except Exception:
            if EDGE_AVAILABLE:
                return await _edge_generate(text, VOICE)
            return _silence_mulaw()
    if voice_id == SAUDI_VOICE_ID:
        try:
            return await asyncio.to_thread(_xtts_generate, text, "saudi", SAUDI_TEMPERATURE)
        except Exception:
            if EDGE_AVAILABLE:
                return await _edge_generate(text, VOICE)
            return _silence_mulaw()
    if _should_use_edge(voice_id):
        try:
            return await _edge_generate(text, voice_id)
        except Exception:
            return _silence_mulaw()
    if TTS_ENGINE == "coqui" and tts_model:
        return await asyncio.to_thread(_coqui_generate, text)
    return _silence_mulaw()

@app.middleware("http")
async def internal_auth(request: Request, call_next):
    if request.url.path.startswith("/health") or request.url.path.startswith("/ready") or request.url.path.startswith("/metrics"):
        return await call_next(request)
    # Use constant-time comparison to prevent timing attacks
    import hmac
    provided_secret = request.headers.get("x-internal-secret") or ""
    if not INTERNAL_SECRET or not hmac.compare_digest(provided_secret, INTERNAL_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await call_next(request)

# Request/Response models
class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = DEFAULT_VOICE
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
        "device": DEVICE if TTS_ENGINE in {"coqui", "xtts"} else "cpu",
        "model": COQUI_MODEL if TTS_ENGINE == "coqui" else VOICE,
        "voices": {
            "egtts": EGTTS_REPO_ID,
            "saudi": SAUDI_REPO_ID,
        },
        "correlationId": None,  # placeholder
    }

@app.get("/ready")
async def ready():
    """Readiness check for downstream orchestration."""
    xtts_ready = XTTS_AVAILABLE
    return {
        "ready": TTS_ENGINE == "edge" or tts_model is not None or xtts_ready,
        "engine": TTS_ENGINE,
        "model": COQUI_MODEL if TTS_ENGINE == "coqui" else VOICE,
        "xttsReady": xtts_ready,
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
    xtts_voices = [
        EGTTS_VOICE_ID,
        SAUDI_VOICE_ID,
    ]
    if TTS_ENGINE == "coqui":
        return {
            "engine": "coqui",
            "voices": ["default"] + xtts_voices,
            "model": COQUI_MODEL,
        }
    # Common Arabic voices in edge-tts (only if allowed)
    arabic_voices = sorted(EDGE_VOICES) if EDGE_AVAILABLE else []
    return {
        "engine": "edge-tts" if EDGE_AVAILABLE else "xtts",
        "voices": xtts_voices + arabic_voices,
        "default": DEFAULT_VOICE,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
