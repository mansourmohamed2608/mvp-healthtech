# services/asr/app.py
import base64
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import soundfile as sf

app = FastAPI(title="ASR Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class TranscribeRequest(BaseModel):
    audio: str
    callSid: str | None = None

class TranscribeResponse(BaseModel):
    text: str

class StreamRequest(BaseModel):
    callSid: str
    audio: str  # base64 chunk

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/whisper-large-v2"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(DEVICE)

# Load LoRA adapters from services/asr/lora_ckpt if present
try:
    model = PeftModel.from_pretrained(model, "./lora_ckpt")
except Exception:
    pass

# In-memory buffers for streaming audio per call
streams: dict[str, list[float]] = {}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest):
    try:
        audio_bytes = base64.b64decode(req.audio)
        waveform, sample_rate = sf.read(io.BytesIO(audio_bytes))
        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            predicted_ids = model.generate(**inputs)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream")
async def stream(req: StreamRequest):
    chunk = base64.b64decode(req.audio)
    waveform, sample_rate = sf.read(io.BytesIO(chunk))
    buf = streams.setdefault(req.callSid, [])
    buf.extend(waveform.tolist())
    # decode every 0.3 s worth of samples
    if len(buf) / sample_rate > 0.3:
        audio_tensor = torch.tensor(buf, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        inputs = processor(audio_tensor, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            predicted_ids = model.generate(**inputs)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        streams[req.callSid] = []
        return {"partial": text}
    return {"partial": ""}
