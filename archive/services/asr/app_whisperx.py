# services/asr/app_whisperx.py
"""
WhisperX-based ASR Service with Speaker Diarization + LoRA
Migrated from vanilla Whisper to WhisperX for better performance
Now includes fine-tuned LoRA adapters for medical transcription
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
USE_LORA = os.getenv("USE_LORA", "true").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN", None)
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"
ENABLE_VAD = os.getenv("ENABLE_VAD", "true").lower() == "true"

# Global models (loaded once at startup)
whisper_model = None
whisper_model_with_lora = None  # WhisperX model with LoRA adapters
diarize_model = None
lora_enabled = False

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
    """Load WhisperX models with LoRA adapters at startup"""
    global whisper_model, whisper_model_with_lora, diarize_model, lora_enabled
    
    print("=" * 60)
    print("LOADING ASR MODELS")
    print("=" * 60)
    
    # Load base WhisperX model
    print(f"📥 Loading WhisperX model: {WHISPER_MODEL} on {DEVICE}...")
    whisper_model = whisperx.load_model(
        WHISPER_MODEL,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language="ar"  # Default to Arabic
    )
    print("✅ Base Whisper model loaded")
    
    # Try to load LoRA adapters
    if USE_LORA and os.path.exists(LORA_ADAPTER_PATH):
        print(f"📥 Loading LoRA adapters from: {LORA_ADAPTER_PATH}")
        try:
            from peft import PeftModel
            from transformers import WhisperForConditionalGeneration
            
            # Load base Whisper model for LoRA
            base_model_hf = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v3",
                torch_dtype=torch.float16 if COMPUTE_TYPE == "float16" else torch.float32,
                device_map=DEVICE
            )
            
            # Load LoRA adapters
            whisper_model_with_lora = PeftModel.from_pretrained(
                base_model_hf,
                LORA_ADAPTER_PATH,
                torch_dtype=torch.float16 if COMPUTE_TYPE == "float16" else torch.float32
            )
            whisper_model_with_lora.eval()
            
            lora_enabled = True
            print("✅ LoRA adapters loaded successfully!")
            print(f"   Adapter path: {LORA_ADAPTER_PATH}")
            print(f"   LoRA will be used for initial transcription")
            print(f"   WhisperX features still available (diarization, alignment)")
        except Exception as e:
            print(f"⚠️  Could not load LoRA adapters: {e}")
            print("   Falling back to base WhisperX model only")
            whisper_model_with_lora = None
            lora_enabled = False
    else:
        if not os.path.exists(LORA_ADAPTER_PATH):
            print(f"ℹ️  LoRA adapter path not found: {LORA_ADAPTER_PATH}")
        if not USE_LORA:
            print("ℹ️  LoRA disabled by configuration (USE_LORA=false)")
        whisper_model_with_lora = None
        lora_enabled = False
    
    # Load diarization model if enabled and token provided
    if ENABLE_DIARIZATION and HF_TOKEN:
        print("📥 Loading diarization model...")
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=HF_TOKEN,
                device=DEVICE
            )
            print("✅ Diarization model loaded")
        except Exception as e:
            print(f"⚠️  Could not load diarization model: {e}")
            print("   Diarization will be disabled. Check your HF_TOKEN.")
            diarize_model = None
    else:
        if not HF_TOKEN:
            print("ℹ️  No HF_TOKEN provided. Diarization disabled.")
        if not ENABLE_DIARIZATION:
            print("ℹ️  Diarization disabled by configuration")
        diarize_model = None
    
    print()
    print("=" * 60)
    print("✅ ASR SERVICE READY!")
    print("=" * 60)
    print(f"  Model: WhisperX {WHISPER_MODEL}")
    print(f"  LoRA: {'✅ Enabled (Fine-tuned for medical)' if lora_enabled else '❌ Disabled'}")
    print(f"  Diarization: {'✅ Enabled' if diarize_model else '❌ Disabled'}")
    print(f"  VAD: {'✅ Enabled' if ENABLE_VAD else '❌ Disabled'}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)
    print()


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
    Returns WhisperX-compatible format for alignment
    """
    if whisper_model_with_lora is None:
        raise ValueError("LoRA model not loaded")
    
    from transformers import WhisperProcessor
    import librosa
    
    # Resample to 16kHz if needed (Whisper expects 16kHz)
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    # Load processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    # Prepare inputs
    inputs = processor(
        audio_data,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_features.to(DEVICE)
    
    # Generate transcription with LoRA model
    with torch.no_grad():
        predicted_ids = whisper_model_with_lora.generate(inputs)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    # Format as WhisperX-compatible output
    audio_duration = len(audio_data) / sample_rate
    return {
        "text": transcription,
        "segments": [{
            "text": transcription,
            "start": 0.0,
            "end": audio_duration
        }],
        "language": language
    }


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
        
        # Save to temporary WAV file (WhisperX expects file path)
        temp_path = "/tmp/temp_audio.wav"
        sf.write(temp_path, audio_data, sample_rate)
        
        # Load audio with WhisperX
        audio = whisperx.load_audio(temp_path)
        
        # 1. TRANSCRIPTION (with LoRA if enabled, otherwise WhisperX)
        print("Step 1: Transcription...")
        transcription_start = time.time()
        
        if lora_enabled and whisper_model_with_lora is not None:
            print("  Using LoRA-enhanced model (fine-tuned for medical)...")
            try:
                # Transcribe with LoRA model
                result = transcribe_with_lora(audio_data, sample_rate, request.language)
                print(f"  ✓ LoRA transcription successful!")
            except Exception as e:
                print(f"  ⚠️ LoRA transcription failed: {e}")
                print("  Falling back to base WhisperX model...")
                result = whisper_model.transcribe(
                    audio,
                    batch_size=16,
                    language=request.language,
                    condition_on_previous_text=False,
                    vad_filter=ENABLE_VAD,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=400
                    ) if ENABLE_VAD else None
                )
        else:
            print("  Using base WhisperX model...")
            result = whisper_model.transcribe(
                audio,
                batch_size=16,  # Adjust based on GPU memory
                language=request.language,
                condition_on_previous_text=False,  # Reduces hallucination
                vad_filter=ENABLE_VAD,  # VAD preprocessing
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Minimum silence to split
                    speech_pad_ms=400  # Padding around speech
                ) if ENABLE_VAD else None
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
            
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                DEVICE,
                return_char_alignments=False
            )
            
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
                # Run diarization
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=request.min_speakers,
                    max_speakers=request.max_speakers
                )
                
                # Assign speakers to words
                result = whisperx.assign_word_speakers(diarize_segments, result)
                
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
                print(f"  ⚠️ Diarization failed: {e}")
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
        
        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass
        
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
        "lora_enabled": lora_enabled,
        "lora_path": LORA_ADAPTER_PATH if lora_enabled else None,
        "diarization_enabled": diarize_model is not None,
        "vad_enabled": ENABLE_VAD,
        "features": {
            "transcription": "✅",
            "lora_fine_tuning": "✅ Medical Arabic" if lora_enabled else "❌",
            "word_timestamps": "✅",
            "speaker_diarization": "✅" if diarize_model else "❌",
            "vad_preprocessing": "✅" if ENABLE_VAD else "❌"
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
