# services/asr/app_whisperx_lora.py
"""
WhisperX-based ASR Service with LoRA Adapters
Integrates your fine-tuned Whisper Large v3 LoRA adapters with WhisperX
"""
import base64
import io
import time
import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisperx
import gc
import torch
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load environment variables
load_dotenv()

app = FastAPI(title="ASR Service (WhisperX + LoRA)")
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
transcriptions_total = Counter('asr_transcriptions_total', 'Total number of transcriptions')

# Configuration from environment
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "./lora_ckpt")
HF_TOKEN = os.getenv("HF_TOKEN", None)
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"
ENABLE_VAD = os.getenv("ENABLE_VAD", "true").lower() == "true"
USE_LORA = os.getenv("USE_LORA", "true").lower() == "true"

# Global models (loaded once at startup)
whisper_model = None
whisper_model_lora = None  # LoRA-enhanced model
diarize_model = None
processor = None

# Language-specific alignment models
ALIGNMENT_MODELS = {
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",  # Arabic
    "en": "WAV2VEC2_ASR_LARGE_LV60K_960H"  # English
}


class TranscriptionRequest(BaseModel):
    audio: str  # base64 encoded audio
    dialect: Optional[str] = "egypt"
    language: Optional[str] = "ar"
    enable_diarization: Optional[bool] = True
    use_lora: Optional[bool] = True  # NEW: Allow per-request LoRA toggle
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    score: Optional[float] = None


class TranscriptionSegment(BaseModel):
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    words: Optional[List[WordTimestamp]] = None


class TranscriptionResponse(BaseModel):
    text: str
    segments: List[TranscriptionSegment]
    language: str
    duration: float
    processing_time: float
    rtf: float
    speakers: Optional[List[str]] = None
    model_used: str  # NEW: Show if LoRA was used


@app.on_event("startup")
async def load_models():
    """Load WhisperX models with LoRA adapters at startup"""
    global whisper_model, whisper_model_lora, diarize_model, processor
    
    print(f"Loading WhisperX model: {WHISPER_MODEL} on {DEVICE}...")
    
    # Load base WhisperX model (for comparison or fallback)
    whisper_model = whisperx.load_model(
        WHISPER_MODEL,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language="ar"  # Default to Arabic
    )
    print("✓ Base Whisper model loaded")
    
    # Load LoRA-enhanced model if enabled
    if USE_LORA and os.path.exists(LORA_ADAPTER_PATH):
        print(f"Loading LoRA adapters from: {LORA_ADAPTER_PATH}")
        try:
            # Load base Whisper model from HuggingFace
            base_model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v3",
                torch_dtype=torch.float16 if COMPUTE_TYPE == "float16" else torch.float32,
                device_map=DEVICE
            )
            
            # Load LoRA adapters
            whisper_model_lora = PeftModel.from_pretrained(
                base_model,
                LORA_ADAPTER_PATH,
                torch_dtype=torch.float16 if COMPUTE_TYPE == "float16" else torch.float32
            )
            whisper_model_lora.eval()
            
            # Load processor for LoRA model
            processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            
            print("✓ LoRA adapters loaded and integrated!")
            print(f"  LoRA rank: 8")
            print(f"  LoRA alpha: 16")
            print(f"  Target modules: q_proj, v_proj, k_proj, fc1, fc2, out_proj")
        except Exception as e:
            print(f"⚠️ Could not load LoRA adapters: {e}")
            print("Falling back to base WhisperX model")
            whisper_model_lora = None
    else:
        if not os.path.exists(LORA_ADAPTER_PATH):
            print(f"⚠️ LoRA adapter path not found: {LORA_ADAPTER_PATH}")
        else:
            print("ℹ️ LoRA disabled by configuration")
        whisper_model_lora = None
    
    # Load diarization model if enabled and token provided
    if ENABLE_DIARIZATION and HF_TOKEN:
        print("Loading diarization model...")
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=HF_TOKEN,
                device=DEVICE
            )
            print("✓ Diarization model loaded")
        except Exception as e:
            print(f"⚠️ Could not load diarization model: {e}")
            print("Diarization will be disabled. Check your HF_TOKEN.")
            diarize_model = None
    else:
        if not HF_TOKEN:
            print("⚠️ No HF_TOKEN provided. Diarization disabled.")
        else:
            print("ℹ️ Diarization disabled by configuration")
    
    print("=" * 80)
    print("✓ ASR SERVICE READY!")
    print("=" * 80)
    print(f"📊 CONFIGURATION:")
    print(f"   Base model: WhisperX {WHISPER_MODEL}")
    print(f"   Device: {DEVICE}")
    print(f"   Compute type: {COMPUTE_TYPE}")
    print()
    print(f"🔧 LORA STATUS:")
    if whisper_model_lora:
        print(f"   ✅ LoRA ADAPTERS LOADED AND ACTIVE!")
        print(f"   📁 Path: {LORA_ADAPTER_PATH}")
        print(f"   🎯 Enhanced Arabic medical transcription enabled")
    else:
        print(f"   ❌ LoRA NOT LOADED - Using base model only")
        if not USE_LORA:
            print(f"   ℹ️  Reason: USE_LORA=False in config")
        elif not os.path.exists(LORA_ADAPTER_PATH):
            print(f"   ⚠️  Reason: Adapter path not found")
    print()
    print(f"🎤 DIARIZATION:")
    print(f"   {'✅ Enabled' if diarize_model else '❌ Disabled'}")
    print("=" * 80)


def decode_audio(audio_base64: str) -> tuple[np.ndarray, int]:
    """Decode base64 audio to numpy array"""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        return audio_data, sample_rate
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {str(e)}")


def transcribe_with_lora(audio_data: np.ndarray, sample_rate: int, language: str = "ar") -> Dict[str, Any]:
    """
    Transcribe using LoRA-enhanced Whisper model
    Returns WhisperX-compatible format
    """
    if whisper_model_lora is None or processor is None:
        raise ValueError("LoRA model not loaded")
    
    # Resample if needed (Whisper expects 16kHz)
    if sample_rate != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    # Prepare inputs
    inputs = processor(
        audio_data,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_features.to(DEVICE)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = whisper_model_lora.generate(inputs)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    # Format as WhisperX-compatible output
    # Note: LoRA model doesn't provide word-level timestamps
    # We'll use WhisperX for alignment after getting the transcription
    return {
        "text": transcription,
        "segments": [{
            "text": transcription,
            "start": 0.0,
            "end": len(audio_data) / sample_rate
        }],
        "language": language
    }


def format_segments_for_frontend(
    segments: List[Dict[str, Any]], 
    language: str
) -> List[TranscriptionSegment]:
    """
    Convert WhisperX segments to frontend-compatible format
    """
    formatted_segments = []
    
    for seg in segments:
        # Extract words if available
        words = None
        if "words" in seg and seg["words"]:
            words = [
                WordTimestamp(
                    word=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    score=w.get("score", None)
                )
                for w in seg["words"]
            ]
        
        formatted_segments.append(TranscriptionSegment(
            text=seg.get("text", "").strip(),
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            speaker=seg.get("speaker", None),
            words=words
        ))
    
    return formatted_segments


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """
    Transcribe audio with WhisperX + optional LoRA adapters
    Features:
    - Fine-tuned Arabic medical transcription (with LoRA)
    - Fast batched inference (4-70x realtime)
    - VAD preprocessing (reduces hallucinations)
    - Word-level timestamps
    - Optional speaker diarization
    """
    transcriptions_total.inc()
    overall_start = time.time()
    
    try:
        # Decode audio
        audio_data, sample_rate = decode_audio(request.audio)
        audio_duration = len(audio_data) / sample_rate
        
        print(f"\n{'='*60}")
        print(f"Transcription Request:")
        print(f"  Language: {request.language}")
        print(f"  Dialect: {request.dialect}")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Use LoRA: {request.use_lora and whisper_model_lora is not None}")
        print(f"  Diarization: {request.enable_diarization and diarize_model is not None}")
        
        # Choose model based on request
        use_lora_model = request.use_lora and whisper_model_lora is not None
        model_used = "Whisper Large v3 + LoRA" if use_lora_model else "WhisperX Large v3"
        
        if use_lora_model:
            print("=" * 60)
            print("🔥 USING LORA-ENHANCED MODEL!")
            print("=" * 60)
            print("  📁 LoRA adapters loaded from:", LORA_ADAPTER_PATH)
            print("  🎯 Enhanced Arabic medical transcription active")
            print("=" * 60)
            start_transcribe = time.time()
            
            # Transcribe with LoRA model
            result = transcribe_with_lora(audio_data, sample_rate, request.language)
            
            # Use WhisperX for alignment (word-level timestamps)
            if ENABLE_VAD:
                print("  Aligning with WhisperX...")
                alignment_model_name = ALIGNMENT_MODELS.get(request.language, "WAV2VEC2_ASR_LARGE_LV60K_960H")
                model_a, metadata = whisperx.load_align_model(
                    language_code=request.language,
                    device=DEVICE
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio_data,
                    DEVICE,
                    return_char_alignments=False
                )
            
            transcribe_time = time.time() - start_transcribe
            
        else:
            print("=" * 60)
            print("⚠️  USING BASE MODEL (No LoRA)")
            print("=" * 60)
            if not request.use_lora:
                print("  ℹ️  Reason: use_lora=false in request")
            elif whisper_model_lora is None:
                print("  ⚠️  Reason: LoRA adapters not loaded")
            print("=" * 60)
            start_transcribe = time.time()
            
            # Standard WhisperX transcription
            result = whisper_model.transcribe(
                audio_data,
                batch_size=16,
                language=request.language
            )
            
            # Alignment for word-level timestamps
            if ENABLE_VAD:
                alignment_model_name = ALIGNMENT_MODELS.get(request.language, "WAV2VEC2_ASR_LARGE_LV60K_960H")
                model_a, metadata = whisperx.load_align_model(
                    language_code=request.language,
                    device=DEVICE
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio_data,
                    DEVICE,
                    return_char_alignments=False
                )
            
            transcribe_time = time.time() - start_transcribe
        
        print(f"  Transcription time: {transcribe_time:.2f}s")
        
        # Speaker diarization (if requested and available)
        speakers_list = None
        if request.enable_diarization and diarize_model is not None:
            print("  Running speaker diarization...")
            start_diarize = time.time()
            
            diarize_segments = diarize_model(
                audio_data,
                min_speakers=request.min_speakers,
                max_speakers=request.max_speakers
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Extract unique speakers
            speakers_set = set()
            for segment in result["segments"]:
                if "speaker" in segment:
                    speakers_set.add(segment["speaker"])
            speakers_list = sorted(list(speakers_set))
            
            diarize_time = time.time() - start_diarize
            print(f"  Diarization time: {diarize_time:.2f}s")
            print(f"  Detected speakers: {len(speakers_list)}")
        
        # Format response
        full_text = " ".join([seg["text"] for seg in result["segments"]])
        formatted_segments = format_segments_for_frontend(result["segments"], request.language)
        
        overall_time = time.time() - overall_start
        rtf = overall_time / audio_duration if audio_duration > 0 else 0
        
        # Update metrics
        transcription_duration.observe(overall_time)
        rtf_ratio.observe(rtf)
        
        print(f"  Total time: {overall_time:.2f}s")
        print(f"  RTF: {rtf:.2f}x")
        print(f"  Model used: {model_used}")
        print(f"{'='*60}\n")
        
        return TranscriptionResponse(
            text=full_text,
            segments=formatted_segments,
            language=request.language,
            duration=audio_duration,
            processing_time=overall_time,
            rtf=rtf,
            speakers=speakers_list,
            model_used=model_used
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": WHISPER_MODEL,
        "lora_enabled": whisper_model_lora is not None,
        "diarization_enabled": diarize_model is not None,
        "device": DEVICE
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
