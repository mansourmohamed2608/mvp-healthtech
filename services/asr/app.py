#!/usr/bin/env python3
"""
ASR Service (WhisperX large-v3) — Arabic Medical, Dialect-aware

What’s inside:
- WhisperX (CTranslate2) fast inference (GPU-first, CPU fallback)
- VAD (handled internally by WhisperX when available)
- Single cached Arabic aligner for word/char timestamps
- Two diarization modes:
    * diarize-last  (default): Transcribe → Align → Diarize → assign speakers
    * diarize-first (optional, better attribution): Diarize → per-speaker transcribe → Align & merge
- Medical vocabulary injection (prompt) + Arabic post-processing corrector
- Egyptian/Levant prompts supported; custom vocab file supported
- pyannote diarization: prefers 3.2+, falls back to 3.1 if needed
- Prometheus metrics

Note: LoRA/Transformers are intentionally NOT used.
"""

import base64
import io
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import re
import json

import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisperx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ---------------------------
# Load environment
# ---------------------------
root_env = Path(__file__).parent.parent.parent / ".env"
load_dotenv(root_env)

# ---------------------------
# Metrics
# ---------------------------
transcription_duration = Histogram(
    "asr_transcription_duration_seconds",
    "Time taken to transcribe audio",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0],
)
rtf_ratio = Histogram(
    "asr_rtf_ratio",
    "Real-Time Factor (processing time / audio duration)",
    buckets=[0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
)
transcriptions_total = Counter("asr_transcriptions_total", "Total number of transcriptions")

# ---------------------------
# Config & helpers
# ---------------------------
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = os.getenv("DEVICE") or pick_device()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")

def pick_compute_type() -> str:
    env_ct = (os.getenv("COMPUTE_TYPE") or "").lower()
    return env_ct if env_ct else "float32"

COMPUTE_TYPE = pick_compute_type()

HF_TOKEN = os.getenv("HF_TOKEN", None)
ENABLE_VAD = os.getenv("ENABLE_VAD", "true").lower() == "true"
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"

# diarization strategy
DIARIZE_FIRST = os.getenv("DIARIZE_FIRST", "false").lower() == "true"
PYANNOTE_MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.2")

# optional external vocab
VOCAB_FILE = os.getenv("MEDICAL_VOCAB_FILE", "").strip()  # .json or .txt (one term per line)
VOCAB_ENABLE = os.getenv("VOCAB_ENABLE", "true").lower() == "true"

# Alignment
ALIGNMENT_MODELS = {
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "en": "WAV2VEC2_ASR_LARGE_LV60K_960H",
}

# ---------------------------
# Built-in medical vocabulary & confusion map
# ---------------------------
BUILTIN_MEDICAL_TERMS = [
    "لثة", "اللثة", "اللسّة", "جيوب لثوية", "التهاب اللثة", "نزيف اللثة",
    "سنان", "أسنان", "ضرس", "ضرس العقل",
    "خيط أسنان", "الخيط الطبي", "مضمضة", "حشو", "خلع ضرس",
    "أشعة", "تحاليل", "تشخيص", "علاج", "جرعة", "متابعة", "استشارة",
    "ضغط الدم", "سكر", "حساسية الأسنان", "حساسية", "تورم",
]

CONFUSION_MAP = {
    "السة": "اللثة",
    "اللسة": "اللثة",
    "القراصيم": "الجراثيم",
    "مرار الوقت": "مرور الوقت",
    "الخيط": "خيط",
    "خط طبي": "خيط طبي",
    "خط الأسنان": "خيط أسنان",
    "ممش": "مش",
    "اللسويه": "اللثوية",
    "اللسم": "اللثة",
}

def load_extra_vocab() -> List[str]:
    if not VOCAB_FILE:
        return []
    path = Path(VOCAB_FILE)
    if not path.exists():
        print(f"ℹ️ MEDICAL_VOCAB_FILE not found: {path}")
        return []
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "terms" in data:
                return [str(t).strip() for t in data["terms"] if str(t).strip()]
            if isinstance(data, list):
                return [str(t).strip() for t in data if str(t).strip()]
            return []
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as e:
        print(f"⚠️ Failed to load MEDICAL_VOCAB_FILE: {e}")
        return []

EXTRA_TERMS = load_extra_vocab()

def make_initial_prompt(dialect_hint: str = "egypt") -> str:
    if not VOCAB_ENABLE:
        return ""
    core = list(dict.fromkeys(BUILTIN_MEDICAL_TERMS + EXTRA_TERMS))
    vocab_str = "، ".join(core[:256])
    if (dialect_hint or "").lower().startswith("egypt"):
        style = "محادثة بين طبيب ومريضة باللهجة المصرية."
    else:
        style = "حوار طبي بين طبيب ومريض."
    return f"{style} مصطلحات طبية: {vocab_str}"

ARABIC_CHAR_NORM = [
    (re.compile(r"[إأٱآا]"), "ا"),
    (re.compile(r"[يى]"), "ي"),
    (re.compile(r"[ة]"), "ه"),
    (re.compile(r"[ًٌٍَُِّْ]"), ""),
]

def normalize_arabic(s: str) -> str:
    out = s
    for pat, repl in ARABIC_CHAR_NORM:
        out = pat.sub(repl, out)
    return out

def post_process_text(text: str) -> str:
    if not text:
        return text
    t = text
    neighborhoods = ["اسنان", "سنان", "لث", "ضرس", "طبيب", "دكتور", "علاج", "مضمض", "خيط", "اشعه", "تحليل"]
    norm_t = normalize_arabic(t)
    for wrong, right in CONFUSION_MAP.items():
        w_norm = normalize_arabic(wrong)
        if any(nb in norm_t for nb in neighborhoods):
            t = re.sub(rf"(?<!\w){re.escape(wrong)}(?!\w)", right, t)
    t = t.replace("سنان", "أسنان")
    t = t.replace("اللسة", "اللثة")
    t = t.replace("السة", "اللثة")
    t = t.replace("خط طبي", "خيط طبي")
    t = t.replace("خط الأسنان", "خيط أسنان")
    return t

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="ASR Service (WhisperX)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------
# Models (globals, loaded once)
# ---------------------------
whisper_model = None
align_model_a = None
align_metadata = None
diarize_model = None

# ---------------------------
# Schemas
# ---------------------------
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

class TranscriptionRequest(BaseModel):
    audio: str
    dialect: Optional[str] = "egypt"
    language: Optional[str] = "ar"
    enable_diarization: Optional[bool] = True
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    diarize_first: Optional[bool] = None

class TranscriptionResponse(BaseModel):
    text: str
    segments: List[TranscriptionSegment]
    language: str
    duration: float
    processing_time: float
    rtf: float
    speakers: Optional[List[str]] = None
    model_used: str
    pipeline_mode: str  # "diarize-first" or "diarize-last"

# ---------------------------
# Utilities
# ---------------------------
def decode_audio(audio_base64: str) -> Tuple[np.ndarray, int]:
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        return audio_data, sample_rate
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {str(e)}")

def format_segments_for_frontend(segments: List[Dict[str, Any]]) -> List[TranscriptionSegment]:
    formatted: List[TranscriptionSegment] = []
    for seg in segments or []:
        words = None
        if "words" in seg and seg["words"]:
            words = [
                WordTimestamp(
                    word=w.get("word", ""),
                    start=float(w.get("start", 0.0) or 0.0),
                    end=float(w.get("end", 0.0) or 0.0),
                    score=w.get("score", None),
                )
                for w in seg["words"]
            ]
        formatted.append(
            TranscriptionSegment(
                text=(seg.get("text") or "").strip(),
                start=float(seg.get("start", 0.0) or 0.0),
                end=float(seg.get("end", 0.0) or 0.0),
                speaker=seg.get("speaker"),
                words=words,
            )
        )
    return formatted

def _safe_transcribe(model, audio: np.ndarray, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call model.transcribe(audio, **kwargs) but automatically strip any kwargs
    that this installation of faster-whisper/whisperx doesn't recognize.
    """
    try_kwargs = dict(base_kwargs)
    while True:
        try:
            return model.transcribe(audio, **try_kwargs)
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument" in msg:
                bad = None
                if "'" in msg:
                    parts = msg.split("'")
                    if len(parts) >= 2:
                        bad = parts[1]
                if bad and bad in try_kwargs:
                    print(f"↻ Removing unsupported kwarg: {bad!r} and retrying...")
                    try_kwargs.pop(bad, None)
                    continue
            print("↻ Falling back to minimal transcribe() args (compat mode)...")
            return model.transcribe(
                audio,
                batch_size=base_kwargs.get("batch_size", 1),
                language=base_kwargs.get("language", "ar"),
                task=base_kwargs.get("task", "transcribe"),
            )

def transcribe_chunk(
    audio_f32: np.ndarray,
    language: str,
    dialect_hint: str,
    batch_sz: int,
    start_offset_s: float = 0.0,
) -> Dict[str, Any]:
    decode_kwargs = dict(
        batch_size=batch_sz,
        language=language or "ar",
        task="transcribe",
        initial_prompt=make_initial_prompt(dialect_hint),
        temperature=0.0,
        condition_on_previous_text=False,
        beam_size=5,
    )
    result = _safe_transcribe(whisper_model, audio_f32, decode_kwargs)

    for s in result.get("segments", []) or []:
        s["start"] = float(s.get("start", 0.0) or 0.0) + start_offset_s
        s["end"] = float(s.get("end", 0.0) or 0.0) + start_offset_s
    return result

def align_segments(
    result_segments: List[Dict[str, Any]],
    audio_f32: np.ndarray,
    language_code: str,
) -> List[Dict[str, Any]]:
    if not result_segments or align_model_a is None:
        return result_segments
    aligned = whisperx.align(
        result_segments,
        align_model_a,
        align_metadata,
        audio_f32,
        DEVICE,
        return_char_alignments=False,
    )
    return aligned.get("segments", result_segments)

# ---------------------------
# Startup: load & cache models once
# ---------------------------
@app.on_event("startup")
async def load_models():
    global whisper_model, align_model_a, align_metadata, diarize_model

    print(f"Loading WhisperX model: {WHISPER_MODEL} on {DEVICE} (compute_type={COMPUTE_TYPE})...")
    whisper_model = whisperx.load_model(
        WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE, language="ar"
    )
    print("✓ Base WhisperX model loaded")

    # Cache Arabic aligner
    try:
        if not os.getenv("HUGGINGFACE_HUB_TOKEN") and HF_TOKEN:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
        align_model_a, align_metadata = whisperx.load_align_model(
            language_code="ar",
            device=DEVICE,
            model_name=ALIGNMENT_MODELS["ar"],
        )
        print("✓ Alignment model cached")
    except Exception as e:
        align_model_a = None
        align_metadata = None
        print(f"⚠️ Could not cache Arabic aligner: {e}")

    # Diarization
    if ENABLE_DIARIZATION and HF_TOKEN:
        try:
            from pyannote.audio import Pipeline
            try:
                diarize_model = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=HF_TOKEN)
            except Exception as e_v32:
                print(f"⚠️ {PYANNOTE_MODEL} failed: {e_v32} — falling back to 3.1")
                diarize_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
                )
            if DEVICE == "cuda":
                diarize_model.to(torch.device("cuda"))
            print("✓ Diarization model loaded")
        except Exception as e:
            diarize_model = None
            print(f"⚠️ Diarization disabled: {e}")
    else:
        diarize_model = None
        print("ℹ️ Diarization disabled by config or missing HF_TOKEN")

    print("=" * 80)
    print("✓ ASR SERVICE READY")
    print(f"Model: WhisperX {WHISPER_MODEL} | Device: {DEVICE} | ComputeType: {COMPUTE_TYPE}")
    print(f"Aligner: {'loaded' if align_model_a is not None else 'not loaded'}")
    print(f"Diarization: {'enabled' if diarize_model is not None else 'disabled'}")
    print("=" * 80)

# ---------------------------
# Main endpoint
# ---------------------------
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    transcriptions_total.inc()
    overall_start = time.time()

    try:
        # 0) Decode
        audio_data, sample_rate = decode_audio(request.audio)
        audio_duration = len(audio_data) / float(sample_rate + 1e-12)

        print(f"\n{'='*60}")
        print("Transcription Request")
        print(f"  Language: {request.language}")
        print(f"  Duration: {audio_duration:.2f}s | Sample Rate: {sample_rate} Hz")
        print(f"  Diarization: {request.enable_diarization and (diarize_model is not None)}")
        print(f"  Mode: {'diarize-first' if (request.diarize_first if request.diarize_first is not None else DIARIZE_FIRST) else 'diarize-last'}")
        print(f"{'-'*60}")

        # 1) Resample to 16kHz mono (fast)
        wave = torch.from_numpy(audio_data).float()
        if wave.dim() > 1:
            wave = wave.mean(dim=1)
        wave = wave.unsqueeze(0)  # [1, T]
        if sample_rate != 16000:
            wave = torchaudio.functional.resample(wave, sample_rate, 16000)
        wave = wave.squeeze(0)
        audio = wave.numpy().astype(np.float32)
        sample_rate = 16000

        batch_sz = 1 if DEVICE in ("cpu", "mps") else 16
        use_diarize_first = (request.diarize_first if request.diarize_first is not None else DIARIZE_FIRST) \
                            and (request.enable_diarization and diarize_model is not None)

        segments: List[Dict[str, Any]] = []
        detected_lang = request.language or "ar"

        if use_diarize_first:
            # -------- DIARIZE FIRST → per-speaker transcription
            print("Diarize-first mode: running diarization...")
            diarization_start = time.time()
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            diarize_annotation = diarize_model(
                {"waveform": waveform, "sample_rate": 16000},
                min_speakers=request.min_speakers,
                max_speakers=request.max_speakers,
            )
            diarize_time = time.time() - diarization_start
            print(f"  ✓ Diarized in {diarize_time:.2f}s")

            diar_segments: List[Tuple[float, float, str]] = [
                (seg.start, seg.end, spk) for seg, _, spk in diarize_annotation.itertracks(yield_label=True)
            ]
            diar_segments.sort(key=lambda x: x[0])

            print(f"Transcribing {len(diar_segments)} diarized chunks ...")
            t_start = time.time()
            for (s, e, spk) in diar_segments:
                s = float(s); e = float(e)
                s_samp = int(max(0.0, s) * sample_rate)
                e_samp = int(min(len(audio) / sample_rate, e) * sample_rate)
                if e_samp <= s_samp:
                    continue
                chunk = audio[s_samp:e_samp].copy()
                result = transcribe_chunk(
                    chunk, request.language or "ar", request.dialect or "egypt", batch_sz, start_offset_s=s
                )
                for seg in result.get("segments", []) or []:
                    seg["speaker"] = spk
                segments.extend(result.get("segments", []))

            transcribe_time = time.time() - t_start
            print(f"  ✓ Per-speaker transcription done in {transcribe_time:.2f}s")

            if align_model_a is not None and segments:
                print("Word-level alignment (global)...")
                segments = align_segments(segments, audio, request.language or "ar")
                print("  ✓ Alignment complete")

            detected_lang = request.language or "ar"
            full_text = " ".join([(s.get("text") or "").strip() for s in segments])
            full_text = post_process_text(full_text)

            formatted_segments = format_segments_for_frontend(segments)

            total_time = time.time() - overall_start
            rtf_value = (total_time / audio_duration) if audio_duration > 0 else 0.0
            transcription_duration.observe(total_time)
            rtf_ratio.observe(rtf_value)

            speakers_list = sorted(list({s.get("speaker") for s in segments if s.get("speaker")}))

            print(f"{'-'*60}")
            print(f"Segments: {len(formatted_segments)} | Total time: {total_time:.2f}s | RTF: {rtf_value:.2f}x")
            print(f"{'='*60}\n")

            return TranscriptionResponse(
                text=full_text,
                segments=formatted_segments,
                language=detected_lang,
                duration=audio_duration,
                processing_time=total_time,
                rtf=rtf_value,
                speakers=speakers_list,
                model_used=f"WhisperX {WHISPER_MODEL} ({DEVICE}, {COMPUTE_TYPE})",
                pipeline_mode="diarize-first",
            )

        else:
            # -------- DIARIZE LAST (classic WhisperX flow)
            print("Diarize-last mode: transcribing full audio...")
            t0 = time.time()
            result = transcribe_chunk(audio, request.language or "ar", request.dialect or "egypt", batch_sz, 0.0)
            detected_lang = result.get("language", request.language or "ar")
            t1 = time.time() - t0
            print(f"  ✓ Transcribed in {t1:.2f}s")

            if align_model_a is not None and result.get("segments"):
                print("Word-level alignment...")
                result["segments"] = align_segments(result["segments"], audio, detected_lang)
                print("  ✓ Alignment complete")

            speakers_list = None
            if request.enable_diarization and diarize_model is not None:
                print("Assigning speakers with diarization...")
                from pandas import DataFrame
                waveform = torch.from_numpy(audio).float().unsqueeze(0)
                diarize_annotation = diarize_model(
                    {"waveform": waveform, "sample_rate": 16000},
                    min_speakers=request.min_speakers,
                    max_speakers=request.max_speakers,
                )
                diarize_df = DataFrame(
                    diarize_annotation.itertracks(yield_label=True),
                    columns=["segment", "label", "speaker"],
                )
                diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
                diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
                with_spk = whisperx.assign_word_speakers(diarize_df, result)
                if isinstance(with_spk, dict) and "segments" in with_spk:
                    result["segments"] = with_spk["segments"]
                speakers_list = sorted(list({s.get("speaker") for s in result["segments"] if s.get("speaker")}))

            segments = result.get("segments", [])
            full_text = " ".join([(s.get("text") or "").strip() for s in segments])
            full_text = post_process_text(full_text)
            formatted_segments = format_segments_for_frontend(segments)

            total_time = time.time() - overall_start
            rtf_value = (total_time / audio_duration) if audio_duration > 0 else 0.0
            transcription_duration.observe(total_time)
            rtf_ratio.observe(rtf_value)

            print(f"{'-'*60}")
            print(f"Segments: {len(formatted_segments)} | Total time: {total_time:.2f}s | RTF: {rtf_value:.2f}x")
            print(f"{'='*60}\n")

            return TranscriptionResponse(
                text=full_text,
                segments=formatted_segments,
                language=detected_lang,
                duration=audio_duration,
                processing_time=total_time,
                rtf=rtf_value,
                speakers=speakers_list,
                model_used=f"WhisperX {WHISPER_MODEL} ({DEVICE}, {COMPUTE_TYPE})",
                pipeline_mode="diarize-last",
            )

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# ---------------------------
# Health & metrics
# ---------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": WHISPER_MODEL,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "aligner_cached": align_model_a is not None,
        "diarization_enabled": diarize_model is not None,
        "vad_enabled": ENABLE_VAD,
        "diarize_first_default": DIARIZE_FIRST,
        "pyannote_model": PYANNOTE_MODEL,
        "vocab_terms_builtin": len(BUILTIN_MEDICAL_TERMS),
        "vocab_terms_extra": len(EXTRA_TERMS),
    }

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ASR_PORT", os.getenv("PORT", 5000)))
    host = os.getenv("ASR_HOST", os.getenv("HOST", "0.0.0.0"))
    uvicorn.run(app, host=host, port=port)
