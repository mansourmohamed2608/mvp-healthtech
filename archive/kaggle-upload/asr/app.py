# services/asr/app_whisperx.py
"""
WhisperX-based ASR Service with Speaker Diarization
Migrated from vanilla Whisper to WhisperX for better performance
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
import librosa
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pathlib import Path

# Load environment variables from root .env
root_env = Path(__file__).parent.parent.parent / ".env"
load_dotenv(root_env)

app = FastAPI(title="ASR Service (WhisperX)")
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
HF_TOKEN = os.getenv("HF_TOKEN", None)
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"
ENABLE_VAD = os.getenv("ENABLE_VAD", "true").lower() == "true"

# Global models (loaded once at startup)
whisper_model = None
diarize_model = None

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


@app.on_event("startup")
async def load_models():
    """Load WhisperX models at startup"""
    global whisper_model, diarize_model
    
    print(f"Loading WhisperX model: {WHISPER_MODEL} on {DEVICE}...")
    whisper_model = whisperx.load_model(
        WHISPER_MODEL,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language="ar"  # Default to Arabic
    )
    print("✓ Whisper model loaded")
    
    # Load diarization model if enabled and token provided
    if ENABLE_DIARIZATION and HF_TOKEN:
        print("Loading diarization model...")
        try:
            # WhisperX 3.x API changed - use load_align_model instead
            from pyannote.audio import Pipeline
            diarize_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN
            )
            if DEVICE != "cpu":
                diarize_model.to(torch.device(DEVICE))
            print("✓ Diarization model loaded")
        except Exception as e:
            print(f"⚠️ Could not load diarization model: {e}")
            print("Diarization will be disabled. Check your HF_TOKEN and model agreements.")
            print("Accept models at: https://huggingface.co/pyannote/speaker-diarization-3.1")
            diarize_model = None
    else:
        if not HF_TOKEN:
            print("⚠️ No HF_TOKEN provided. Diarization disabled.")
        else:
            print("ℹ️ Diarization disabled by configuration")
    
    print("✓ ASR Service ready!")


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


def format_segments_for_frontend(
    segments: List[Dict[str, Any]], 
    language: str
) -> List[TranscriptionSegment]:
    """
    Convert WhisperX segments to frontend-compatible format
    Mitigates format differences between vanilla Whisper and WhisperX
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
    Transcribe audio with WhisperX
    Features:
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
        print(f"  Duration: {audio_duration:.2f}s")
        print(f"  Sample Rate: {sample_rate}Hz")
        print(f"  Diarization: {request.enable_diarization and diarize_model is not None}")
        print(f"{'='*60}\n")
        
        # Convert audio to 16kHz mono (WhisperX requirement)
        # No temp files needed - process audio directly in memory
        print(f"Resampling audio to 16kHz mono...")
        if sample_rate != 16000:
            audio = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        else:
            audio = audio_data.copy()
        
        # Ensure mono and float32
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Convert to float32 for WhisperX
        audio = audio.astype(np.float32)
        
        print(f"Audio prepared successfully, shape: {audio.shape}, dtype: {audio.dtype}")
        
        # 1. TRANSCRIPTION (with VAD if enabled)
        print("Step 1: Transcription...")
        transcription_start = time.time()
        
        # NOTE: FasterWhisper (WhisperX backend) supports minimal parameters
        # - NO initial_prompt (use LLM post-processing instead)
        # - NO condition_on_previous_text
        # - NO vad_filter
        # Only batch_size and language are reliably supported
        
        result = whisper_model.transcribe(
            audio,
            batch_size=16,
            language=request.language
        )
        
        transcription_time = time.time() - transcription_start
        print(f"  ✓ Transcribed in {transcription_time:.2f}s")
        print(f"  Detected language: {result.get('language', 'unknown')}")
        
        # 2. ALIGNMENT (for accurate word timestamps)
        print("\nStep 2: Word-level alignment...")
        alignment_start = time.time()
        
        # Select alignment model for detected language
        detected_lang = result.get("language", request.language)
        align_model_name = ALIGNMENT_MODELS.get(detected_lang, ALIGNMENT_MODELS["en"])
        
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_lang,
                device=DEVICE,
                model_name=align_model_name if detected_lang == "ar" else None
            )
            
            aligned_result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                DEVICE,
                return_char_alignments=False
            )
            
            # Update result with aligned segments
            result["segments"] = aligned_result["segments"]
            
            alignment_time = time.time() - alignment_start
            print(f"  ✓ Aligned in {alignment_time:.2f}s")
            
            # Clear alignment model from memory
            del model_a
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ⚠️ Alignment failed: {e}")
            print("  Continuing without word-level timestamps...")
            alignment_time = 0
        
        # 3. DIARIZATION (optional - identify speakers)
        speakers_list = None
        if request.enable_diarization and diarize_model is not None:
            print("\nStep 3: Speaker diarization...")
            diarization_start = time.time()
            
            try:
                # Prepare audio for diarization (ensure correct format)
                waveform = torch.from_numpy(audio).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # Add batch dimension
                
                print(f"  Waveform shape: {waveform.shape}, dtype: {waveform.dtype}")
                print(f"  Min speakers: {request.min_speakers}, Max speakers: {request.max_speakers}")
                
                # Run diarization with Pyannote 3.x API
                diarize_annotation = diarize_model(
                    {"waveform": waveform, "sample_rate": 16000},
                    min_speakers=request.min_speakers,
                    max_speakers=request.max_speakers
                )
                
                print(f"  Diarization complete, type: {type(diarize_annotation)}")
                
                # Convert pyannote Annotation to DataFrame (required by assign_word_speakers)
                import pandas as pd
                diarize_df = pd.DataFrame(
                    diarize_annotation.itertracks(yield_label=True),
                    columns=['segment', 'label', 'speaker']
                )
                diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
                diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
                
                print(f"  Converted to DataFrame: {len(diarize_df)} diarization segments")
                
                # Assign speakers to words using WhisperX helper
                result_with_speakers = whisperx.assign_word_speakers(diarize_df, result)
                
                # Update result with speaker-labeled segments
                if isinstance(result_with_speakers, dict) and "segments" in result_with_speakers:
                    result["segments"] = result_with_speakers["segments"]
                
                # Extract unique speakers
                speakers_list = list(set(
                    seg.get("speaker") 
                    for seg in result["segments"] 
                    if seg.get("speaker")
                ))
                speakers_list.sort()
                
                diarization_time = time.time() - diarization_start
                print(f"  ✓ Diarized in {diarization_time:.2f}s")
                print(f"  Detected speakers: {speakers_list}")
                
            except Exception as e:
                import traceback
                print(f"  ⚠️ Diarization failed: {e}")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Full traceback:")
                traceback.print_exc()
                print("  Continuing without speaker labels...")
        else:
            print("\nStep 3: Diarization skipped")
        
        # 4. FORMAT RESULTS
        print("\nStep 4: Formatting results...")
        formatted_segments = format_segments_for_frontend(
            result["segments"],
            detected_lang
        )
        
        # Combine all text
        full_text = " ".join(seg.text for seg in formatted_segments)
        
        # Calculate metrics
        total_time = time.time() - overall_start
        rtf_value = total_time / audio_duration
        
        # Update Prometheus metrics
        transcription_duration.observe(total_time)
        rtf_ratio.observe(rtf_value)
        
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Total segments: {len(formatted_segments)}")
        print(f"  Processing time: {total_time:.2f}s")
        print(f"  RTF: {rtf_value:.2f}x")
        print(f"  Speed: {audio_duration/total_time:.1f}x realtime")
        print(f"{'='*60}\n")
        
        return TranscriptionResponse(
            text=full_text,
            segments=formatted_segments,
            language=detected_lang,
            duration=audio_duration,
            processing_time=total_time,
            rtf=rtf_value,
            speakers=speakers_list
        )
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": WHISPER_MODEL,
        "device": DEVICE,
        "diarization_enabled": diarize_model is not None,
        "vad_enabled": ENABLE_VAD
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ASR_PORT", 5000))
    host = os.getenv("ASR_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
