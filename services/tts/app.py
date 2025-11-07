# services/tts/app.py
"""
TTS (Text-to-Speech) Service - Coqui TTS with Arabic voices
Synthesizes natural Arabic speech from text responses
Week 3 Day 15 (Oct 9, 2025)
"""
import io
import time
import wave
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import numpy as np
import uvicorn

# Try Coqui TTS, fallback to edge-tts if not available
TTS_ENGINE = "edge"  # Default to edge-tts (free, no GPU needed)

try:
    from TTS.api import TTS as CoquiTTS
    TTS_ENGINE = "coqui"
    print("✅ Using Coqui TTS")
except ImportError:
    print("⚠️  Coqui TTS not available, using edge-tts")
    import edge_tts

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000  # 16kHz for Twilio compatibility
VOICE = "ar-EG-SalmaNeural"  # Edge-TTS Arabic voice (Egyptian female)
COQUI_MODEL = "tts_models/ar/css10/vits"  # Coqui Arabic model

print(f"🚀 TTS Service starting on {DEVICE}...")
print(f"Engine: {TTS_ENGINE}")

# Initialize TTS engine
tts_model = None
if TTS_ENGINE == "coqui":
    try:
        tts_model = CoquiTTS(model_name=COQUI_MODEL, gpu=(DEVICE == "cuda"))
        print(f"✅ Loaded Coqui model: {COQUI_MODEL}")
    except Exception as e:
        print(f"⚠️  Coqui initialization failed: {e}, falling back to edge-tts")
        TTS_ENGINE = "edge"

print(f"✅ TTS Service ready")

app = FastAPI(title="TTS Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = VOICE
    sessionId: Optional[str] = None
    format: Optional[str] = "wav"  # wav, mp3, mulaw

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
    }

@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text
    Returns WAV audio as streaming response
    """
    start_time = time.time()
    
    try:
        if TTS_ENGINE == "coqui" and tts_model:
            # Use Coqui TTS
            audio_np = tts_model.tts(text=request.text)
            
            # Convert to 16-bit PCM
            audio_int16 = (np.array(audio_np) * 32767).astype(np.int16)
            
            # Create WAV file in memory
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())
            
            wav_io.seek(0)
            duration = time.time() - start_time
            
            return StreamingResponse(
                wav_io,
                media_type="audio/wav",
                headers={
                    "X-Duration": str(duration),
                    "X-Sample-Rate": str(SAMPLE_RATE),
                }
            )
        
        else:
            # Use edge-tts (Microsoft Azure TTS - free tier)
            communicate = edge_tts.Communicate(request.text, request.voice or VOICE)
            
            # Generate audio
            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
            
            # Combine chunks
            audio_data = b"".join(audio_chunks)
            audio_io = io.BytesIO(audio_data)
            
            duration = time.time() - start_time
            
            return StreamingResponse(
                audio_io,
                media_type="audio/mpeg",  # edge-tts returns MP3
                headers={
                    "X-Duration": str(duration),
                    "X-Engine": "edge-tts",
                }
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.post("/synthesize/stream")
async def synthesize_stream(request: SynthesizeRequest):
    """
    Stream synthesized speech in chunks
    Useful for real-time playback
    """
    try:
        if TTS_ENGINE == "coqui" and tts_model:
            # Coqui doesn't support streaming, return full audio
            return await synthesize(request)
        
        else:
            # edge-tts supports streaming
            async def audio_generator():
                communicate = edge_tts.Communicate(request.text, request.voice or VOICE)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        yield chunk["data"]
            
            return StreamingResponse(
                audio_generator(),
                media_type="audio/mpeg",
                headers={"X-Engine": "edge-tts-stream"}
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream synthesis failed: {str(e)}")

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
