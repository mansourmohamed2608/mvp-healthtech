# services/asr/app.py
import base64
import io
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import soundfile as sf
import numpy as np
from scipy import signal
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(title="ASR Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Prometheus metrics
transcription_duration = Histogram(
    'asr_transcription_duration_seconds',
    'Time taken to transcribe audio',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
)
rtf_ratio = Histogram(
    'asr_rtf_ratio',
    'Real-Time Factor (processing time / audio duration)',
    buckets=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
)
transcriptions_total = Counter(
    'asr_transcriptions_total',
    'Total number of transcription requests'
)
slow_transcriptions = Counter(
    'asr_slow_transcriptions_total',
    'Number of transcriptions with RTF > 0.5'
)
partial_transcript_latency = Histogram(
    'asr_partial_transcript_latency_ms',
    'Latency for partial transcript generation in streaming',
    buckets=[50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]
)

class TranscribeRequest(BaseModel):
    audio: str
    callSid: str | None = None
    dialect: str | None = None  # Optional: "egyptian", "levantine", "gulf", "msa"
    auto_detect: bool = False   # Auto-detect dialect from audio

class TranscribeResponse(BaseModel):
    text: str
    dialect: str | None = None
    auto_detected: bool = False

class StreamRequest(BaseModel):
    callSid: str
    audio: str  # base64 chunk
    dialect: str | None = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the trained LoRA adapter (simpler approach - use your trained model directly)
ADAPTER_PATH = "./lora_ckpt"
BASE_MODEL = "openai/whisper-large-v3"  # Match the adapter's base model

try:
    print(f"Loading Whisper model with LoRA adapter from {ADAPTER_PATH}...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)

    # Load base model
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.to(DEVICE)
    model.eval()

    print(f"✅ Model with LoRA adapter loaded successfully on {DEVICE}!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    import traceback
    traceback.print_exc()
    raise

# In-memory buffers for streaming audio per call
streams: dict[str, list[float]] = {}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest):
    transcriptions_total.inc()
    start_time = time.time()

    try:
        audio_bytes = base64.b64decode(req.audio)
        waveform, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # Calculate audio duration in seconds
        audio_duration = len(waveform) / sample_rate

        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        # Resample to 16kHz if needed (Whisper requirement)
        TARGET_SAMPLE_RATE = 16000
        if sample_rate != TARGET_SAMPLE_RATE:
            print(f"Resampling from {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz...")
            # Calculate number of samples after resampling
            num_samples = int(len(waveform) * TARGET_SAMPLE_RATE / sample_rate)
            waveform = signal.resample(waveform, num_samples)
            sample_rate = TARGET_SAMPLE_RATE

        # Start transcription timing
        transcription_start = time.time()

        # For long audio, split into 30-second chunks (Whisper's optimal window)
        # Use aggressive anti-hallucination settings
        
        if audio_duration > 30:
            print(f"Long audio detected ({audio_duration:.1f}s). Using 30s chunks with anti-hallucination...")
            
            # Use 30-second chunks with 3-second overlap (balanced)
            CHUNK_DURATION = 30
            OVERLAP_DURATION = 3
            CHUNK_SAMPLES = CHUNK_DURATION * sample_rate
            OVERLAP_SAMPLES = OVERLAP_DURATION * sample_rate
            STEP = CHUNK_SAMPLES - OVERLAP_SAMPLES  # Move forward 27 seconds
            
            transcriptions = []
            previous_text = ""
            
            for i in range(0, len(waveform), STEP):
                chunk_end = min(i + CHUNK_SAMPLES, len(waveform))
                chunk = waveform[i:chunk_end]
                chunk_duration = len(chunk) / sample_rate
                
                print(f"Chunk {len(transcriptions) + 1}: {i/sample_rate:.1f}s-{chunk_end/sample_rate:.1f}s...")
                
                # Process chunk
                inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
                input_features = inputs.input_features.to(DEVICE)
                attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=DEVICE)
                
                # BALANCED settings for Arabic speech (production-tested):
                # - num_beams=1: Greedy decoding (fastest, prevents beam collapse)
                # - temperature=0.2: Slight randomness for Arabic diacritic variations
                # - compression_ratio_threshold=2.4: Standard Whisper value (not too strict)
                # - logprob_threshold=None: Don't skip uncertain segments
                # - no_speech_threshold=0.6: Standard silence detection
                # - condition_on_previous_text=True: Use context from previous chunks
                with torch.no_grad():
                    predicted_ids = model.generate(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        language="ar",
                        task="transcribe",
                        num_beams=1,  # Greedy decoding - simpler, more reliable
                        temperature=0.2,  # Allow slight variations for Arabic dialects
                        compression_ratio_threshold=2.4,  # Standard Whisper value
                        condition_on_previous_text=True,  # Use context
                        no_speech_threshold=0.6,
                    )
                
                chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                
                # Intelligent overlap removal: compare with previous chunk
                if i > 0 and len(transcriptions) > 0 and chunk_text:
                    prev_text = transcriptions[-1]
                    prev_words = prev_text.split()
                    curr_words = chunk_text.split()
                    
                    # Look for overlap: check up to 20 words (covers 3-second overlap at normal speed)
                    max_overlap = min(20, len(prev_words), len(curr_words))
                    overlap_found = 0
                    
                    # First: Try exact matching
                    for overlap_size in range(max_overlap, 3, -1):  # At least 4 words
                        prev_tail = " ".join(prev_words[-overlap_size:])
                        curr_head = " ".join(curr_words[:overlap_size])
                        
                        # Exact match found
                        if prev_tail == curr_head:
                            overlap_found = overlap_size
                            chunk_text = " ".join(curr_words[overlap_size:])
                            print(f"  → Removed {overlap_size}-word exact overlap")
                            break
                    
                    # Second: If no exact match, try fuzzy matching (85% similarity - strict)
                    if overlap_found == 0 and max_overlap >= 8:
                        for overlap_size in range(min(15, max_overlap), 7, -1):
                            prev_tail_words = prev_words[-overlap_size:]
                            curr_head_words = curr_words[:overlap_size]
                            
                            # Count matching words
                            matches = sum(1 for p, c in zip(prev_tail_words, curr_head_words) if p == c)
                            similarity = matches / overlap_size
                            
                            if similarity >= 0.85:  # 85% similarity threshold (stricter)
                                chunk_text = " ".join(curr_words[overlap_size:])
                                print(f"  → Removed {overlap_size}-word fuzzy overlap ({similarity:.0%} match)")
                                break
                
                if chunk_text:
                    transcriptions.append(chunk_text)
                    print(f"  → {chunk_text[:100]}...")
                
                # Early stop if we reached the end
                if chunk_end >= len(waveform):
                    break
            
            text = " ".join(transcriptions)
        else:
            # Short audio - process normally
            inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(DEVICE)
            attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=DEVICE)
            
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    language="ar",
                    task="transcribe",
                    num_beams=5,
                    temperature=0.0,
                )
            
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Calculate metrics
        processing_time = time.time() - transcription_start
        rtf_value = processing_time / audio_duration if audio_duration > 0 else 0

        # Record metrics
        transcription_duration.observe(processing_time)
        rtf_ratio.observe(rtf_value)

        # Track slow transcriptions (RTF > 0.5)
        if rtf_value > 0.5:
            slow_transcriptions.inc()
            print(f"⚠️ Slow transcription: RTF={rtf_value:.3f} (audio={audio_duration:.2f}s, processing={processing_time:.2f}s)")
        else:
            print(f"✅ Fast transcription: RTF={rtf_value:.3f} (audio={audio_duration:.2f}s, processing={processing_time:.2f}s)")

        # Return with dialect info if provided (even though we're using single adapter)
        return TranscribeResponse(
            text=text,
            dialect=req.dialect if req.dialect else None,
            auto_detected=False
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream")
async def stream(req: StreamRequest):
    stream_start = time.time()

    chunk = base64.b64decode(req.audio)
    waveform, sample_rate = sf.read(io.BytesIO(chunk))
    buf = streams.setdefault(req.callSid, [])
    buf.extend(waveform.tolist())

    # decode every 0.3 s worth of samples
    if len(buf) / sample_rate > 0.3:
        transcription_start = time.time()

        audio_tensor = torch.tensor(buf, dtype=torch.float32).unsqueeze(0)
        inputs = processor(audio_tensor, sampling_rate=sample_rate, return_tensors="pt")

        # Move to device properly
        input_features = inputs.input_features.to(DEVICE)

        # Create attention mask explicitly (all 1s for valid audio)
        attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=DEVICE)

        # Use keyword arguments for generate() with attention_mask
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                language="ar",
                task="transcribe"
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Calculate and record partial transcript latency
        latency_ms = (time.time() - transcription_start) * 1000
        partial_transcript_latency.observe(latency_ms)

        print(f"🔄 Partial transcript latency: {latency_ms:.1f}ms (buffer: {len(buf)/sample_rate:.2f}s)")

        streams[req.callSid] = []
        return {"partial": text, "latency_ms": latency_ms}

    return {"partial": ""}

if __name__ == "__main__":
    import uvicorn
    print("Starting ASR service on http://0.0.0.0:5000...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=5000)
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
