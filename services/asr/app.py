# services/asr/app.py
import asyncio
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
import torch
import whisper
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
base_model = whisper.load_model("large-v2", device=DEVICE)

# Load LoRA adapters if they exist
if (Path("lora_ckpt") / "adapter_config.json").exists():
    MODEL = PeftModel.from_pretrained(base_model, "lora_ckpt")
else:
    MODEL = base_model

app = FastAPI()

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"ok": True, "svc": "asr"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> Dict[str, str]:
    audio_bytes = await file.read()
    result = MODEL.transcribe(audio_bytes, language="ar", fp16=(DEVICE=="cuda"))
    return {"text": result["text"]}

@app.websocket("/ws")
async def transcribe_stream(ws: WebSocket):
    await ws.accept()
    buffer = b""
    try:
        while True:
            data = await ws.receive_bytes()
            buffer += data
            if len(buffer) >= 9600:
                result = MODEL.transcribe(buffer, language="ar", fp16=(DEVICE=="cuda"))
                await ws.send_json({"partial": result["text"]})
                buffer = b""
    except WebSocketDisconnect:
        return
