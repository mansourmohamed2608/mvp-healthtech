#!/usr/bin/env python3
"""
ASR Service (WhisperX large-v3) — Arabic Medical, Dialect-aware

What's inside:
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

Note: Adapter tuning is intentionally not used.
"""

import base64
import io
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import re
import json
import inspect
from functools import lru_cache
import traceback
import asyncio
import audioop

import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisperx
from pandas import DataFrame
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import difflib as _difflib
from pydantic import BaseModel
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr")

def safe_print(*args, **_kwargs):
    # Avoid leaking PHI via stdout; count fields instead.
    logger.debug("suppressed print", extra={"fields": len(args)})


def log_safe(level: int, msg: str, session_id: str | None = None, **kwargs):
    extra = {"sessionId": session_id}
    extra.update({k: v for k, v in kwargs.items() if v is not None})
    logger.log(level, msg, extra=extra)


print = safe_print

# AraBART for grammar correction
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    ARABART_AVAILABLE = True
except ImportError:
    ARABART_AVAILABLE = False
    logger.warning("transformers not installed - AraBART GEC disabled")

# LLM Corrector (medical-aware)
try:
    from services.asr.llm_corrector import correct_segments as llm_correct_segments
    LLM_CORRECTOR_AVAILABLE = True
except ImportError:
    try:
        from llm_corrector import correct_segments as llm_correct_segments
        LLM_CORRECTOR_AVAILABLE = True
    except ImportError:
        LLM_CORRECTOR_AVAILABLE = False
        logger.warning("llm_corrector not found - LLM correction disabled")

# Prefer shared functions from text_fix_ar.py to avoid duplication
try:
    from services.asr.text_fix_ar import (
        normalize_arabic,
        collapse_repeats,
        DIALECT_PRESERVE
    )
    _shared_normalize = normalize_arabic
    _shared_collapse = collapse_repeats
except Exception:
    try:
        from text_fix_ar import (
            normalize_arabic,
            collapse_repeats,
            DIALECT_PRESERVE
        )
        _shared_normalize = normalize_arabic
        _shared_collapse = collapse_repeats
    except Exception:
        # Emergency fallback
        def normalize_arabic(s: str) -> str:
            return s
        def collapse_repeats(s: str) -> str:
            return s
        DIALECT_PRESERVE = []
        _shared_normalize = normalize_arabic
        _shared_collapse = collapse_repeats

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
ENABLE_ALIGNMENT = os.getenv("ENABLE_ALIGNMENT", "false").lower() == "true"

# diarization strategy
DIARIZE_FIRST = os.getenv("DIARIZE_FIRST", "false").lower() == "true"
PYANNOTE_MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.2")

# optional external vocab + replacements
VOCAB_FILE = os.getenv("MEDICAL_VOCAB_FILE", "").strip()
VOCAB_ENABLE = os.getenv("VOCAB_ENABLE", "true").lower() == "true"

# Debug toggle to print ambiguous-replacement decision traces
DEBUG_AMBIGUOUS = os.getenv("DEBUG_AMBIGUOUS", "false").lower() == "true"

# AraBART GEC toggle
ENABLE_ARABART = os.getenv("ENABLE_ARABART", "false").lower() == "true"
ARABART_MODEL_NAME = os.getenv("ARABART_MODEL", "aubmindlab/arabart-text-corrector")

# Alignment
ALIGNMENT_MODELS = {
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "en": "WAV2VEC2_ASR_LARGE_LV60K_960H",
}

# ---------------------------
# Built-in medical vocabulary & minimal confusion map
# ---------------------------
BUILTIN_MEDICAL_TERMS = [
    "لثة", "اللثة", "جيوب لثوية", "التهاب اللثة", "نزيف اللثة",
    "أسنان", "ضرس", "ضرس العقل", "خيط أسنان", "مضمضة", "حشو",
    "أشعة", "تحاليل", "تشخيص", "علاج", "جرعة", "متابعة",
    "ضغط الدم", "سكر", "حساسية", "تورم",
]

# Minimal CONFUSION_MAP - only unique entries NOT in medical_vocab_ar_en.json
CONFUSION_MAP = {
    "الثة": "اللثة",
    "اللتة": "اللثة",
    "الليثة": "اللثة",
    "لتوية": "لثوية",
    "تكلسات": "تكلس",
    "هيوجة": "هيوجع",
    "دم": "دماء",
}

def _read_text_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def _read_json_any(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def load_extra_vocab() -> List[str]:
    if not VOCAB_FILE:
        return []
    path = Path(VOCAB_FILE)
    if not path.exists():
        logger.info("MEDICAL_VOCAB_FILE not found", extra={"path": str(path)})
        return []
    try:
        if path.suffix.lower() == ".json":
            data = _read_json_any(path)
            if isinstance(data, dict) and "terms" in data:
                return [str(t).strip() for t in data["terms"] if str(t).strip()]
            if isinstance(data, list):
                return [str(t).strip() for t in data if str(t).strip()]
            return []
        return _read_text_lines(path)
    except Exception as e:
        logger.warning("Failed to load MEDICAL_VOCAB_FILE terms", extra={"error": str(e)})
        return []

def load_replacements_map() -> Dict[str, str]:
    """Load replacements dict from the same VOCAB_FILE if JSON provides 'replacements'."""
    if not VOCAB_FILE:
        return {}
    path = Path(VOCAB_FILE)
    if not path.exists() or path.suffix.lower() != ".json":
        return {}
    try:
        data = _read_json_any(path)
        repl = data.get("replacements", {})
        if isinstance(repl, dict):
            cleaned = {str(k): str(v) for k, v in repl.items()}
            return cleaned
    except Exception as e:
        logger.warning("Failed to load replacements from MEDICAL_VOCAB_FILE", extra={"error": str(e)})
    return {}

def load_ambiguous_map() -> Dict[str, str]:
    """Load ambiguous replacements from VOCAB_FILE JSON under 'ambiguous_replacements'."""
    if not VOCAB_FILE:
        return {}
    path = Path(VOCAB_FILE)
    if not path.exists() or path.suffix.lower() != ".json":
        return {}
    try:
        data = _read_json_any(path)
        amb = data.get("ambiguous_replacements", {})
        if isinstance(amb, dict):
            return {str(k): str(v) for k, v in amb.items()}
    except Exception as e:
        logger.warning("Failed to load ambiguous_replacements from MEDICAL_VOCAB_FILE", extra={"error": str(e)})
    return {}

def load_medical_keywords() -> List[str]:
    """Load configurable medical keywords from VOCAB_FILE JSON under 'medical_keywords'."""
    if not VOCAB_FILE:
        return []
    path = Path(VOCAB_FILE)
    if not path.exists() or path.suffix.lower() != ".json":
        return []
    try:
        data = _read_json_any(path)
        keys = data.get("medical_keywords", [])
        if isinstance(keys, list):
            return [str(k) for k in keys if str(k).strip()]
    except Exception as e:
        logger.warning("Failed to load medical_keywords from MEDICAL_VOCAB_FILE", extra={"error": str(e)})
    return []

EXTRA_TERMS = load_extra_vocab()
EXTRA_REPLACEMENTS = load_replacements_map()
EXTRA_AMBIGUOUS = load_ambiguous_map()
EXTRA_MEDICAL_KEYWORDS = load_medical_keywords()

# Define normalizer BEFORE using it
_norm_func = _shared_normalize if _shared_normalize else normalize_arabic
EXTRA_MEDICAL_KEYWORDS_NORM = set(_norm_func(str(k).lower()) for k in EXTRA_MEDICAL_KEYWORDS)

def load_dialect_config() -> Tuple[bool, List[str]]:
    """Load optional dialect-preservation settings from VOCAB_FILE JSON."""
    if not VOCAB_FILE:
        return False, []
    path = Path(VOCAB_FILE)
    if not path.exists() or path.suffix.lower() != ".json":
        return False, []
    try:
        data = _read_json_any(path)
        preserve = bool(data.get("preserve_dialect", False))
        terms = data.get("dialect_terms", []) or []
        if isinstance(terms, list):
            terms = [str(t) for t in terms if str(t).strip()]
        else:
            terms = []
        return preserve, terms
    except Exception as e:
        logger.warning("Failed to load dialect config from MEDICAL_VOCAB_FILE", extra={"error": str(e)})
    return False, []

PRESERVE_DIALECT, EXTRA_DIALECT_TERMS = load_dialect_config()

def _write_vocab_json_safe(path: Path, data: dict) -> None:
    """Write JSON to path with a timestamped backup of the previous file."""
    try:
        if path.exists():
            backup = path.with_suffix(path.suffix + f".bak_{int(time.time())}")
            path.replace(backup)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def reload_vocab_from_file() -> dict:
    """Reload replacements and ambiguous maps from VOCAB_FILE into memory."""
    global EXTRA_TERMS, EXTRA_REPLACEMENTS, EXTRA_AMBIGUOUS, EXTRA_MEDICAL_KEYWORDS
    global COMPILED_EXTRA_REPLACEMENTS, COMPILED_AMBIGUOUS_REPLACEMENTS
    global PRESERVE_DIALECT, EXTRA_DIALECT_TERMS, EXTRA_MEDICAL_KEYWORDS_NORM

    path = Path(VOCAB_FILE) if VOCAB_FILE else Path(__file__).parent / "medical_vocab_ar_en.json"
    if not path.exists():
        return {"error": f"vocab file not found: {path}"}
    try:
        data = _read_json_any(path)
        if not isinstance(data, dict):
            data = {"terms": [], "replacements": {}, "ambiguous_replacements": {}}
        EXTRA_TERMS = data.get("terms", []) if isinstance(data, dict) else []
        EXTRA_REPLACEMENTS = {str(k): str(v) for k, v in (data.get("replacements", {}) or {}).items()}
        EXTRA_AMBIGUOUS = {str(k): str(v) for k, v in (data.get("ambiguous_replacements", {}) or {}).items()}
        EXTRA_MEDICAL_KEYWORDS = [str(k) for k in (data.get("medical_keywords", []) or [])]
        PRESERVE_DIALECT = bool(data.get("preserve_dialect", PRESERVE_DIALECT))
        EXTRA_DIALECT_TERMS = [str(t) for t in (data.get("dialect_terms", []) or [])]

        COMPILED_EXTRA_REPLACEMENTS = _compile_replacements(EXTRA_REPLACEMENTS)
        COMPILED_AMBIGUOUS_REPLACEMENTS = _compile_replacements(EXTRA_AMBIGUOUS)

        _normf = _shared_normalize if _shared_normalize else normalize_arabic
        EXTRA_MEDICAL_KEYWORDS_NORM = set(_normf(str(k).lower()) for k in EXTRA_MEDICAL_KEYWORDS)

        return {
            "terms": len(EXTRA_TERMS),
            "replacements": len(EXTRA_REPLACEMENTS),
            "ambiguous": len(EXTRA_AMBIGUOUS),
            "medical_keywords": len(EXTRA_MEDICAL_KEYWORDS),
            "preserve_dialect": PRESERVE_DIALECT,
        }
    except Exception as e:
        logger.exception("Failed to reload vocab from file", extra={"error": str(e)})
        return {"error": "reload_failed"}

def make_initial_prompt(dialect_hint: str = "egypt") -> str:
    """Create initial prompt for WhisperX with dialect awareness and medical vocabulary.

    The prompt style significantly affects transcription accuracy:
    - Egyptian dialect: use colloquial markers like 'ازاي', 'عاملة ايه'
    - Medical terms: inject key vocabulary to guide recognition
    """
    if not VOCAB_ENABLE:
        return ""

    # Select top medical terms (prioritize common ones)
    core = list(dict.fromkeys(BUILTIN_MEDICAL_TERMS + EXTRA_TERMS))
    vocab_str = "، ".join(core[:64])  # Reduced from 256 - shorter is better

    if (dialect_hint or "").lower().startswith("egypt"):
        # Egyptian dialect prompt with natural conversational markers
        style = (
            "محادثة عيادة أسنان باللهجة المصرية. "
            "الدكتور بيسأل: ازيك؟ عاملة ايه؟ حاسة بإيه؟ "
            "المريضة بتقول: والله حاسة بألم، اللثة بتنزف، الأسنان حساسة. "
        )
    else:
        style = "حوار طبي بين طبيب ومريض. "

    return f"{style}مصطلحات: {vocab_str}"

# --- dynamic replacements compiled once ---
def _compile_replacements(repl_map: Dict[str, str]) -> List[Tuple[re.Pattern, str]]:
    compiled = []
    for wrong, right in repl_map.items():
        pat = re.compile(rf"(?<!\w){re.escape(wrong)}(?!\w)")
        compiled.append((pat, right))
    return compiled

COMPILED_EXTRA_REPLACEMENTS = _compile_replacements(EXTRA_REPLACEMENTS)
COMPILED_AMBIGUOUS_REPLACEMENTS = _compile_replacements(EXTRA_AMBIGUOUS)

def post_process_text(text: str) -> str:
    """Apply medical ASR corrections with dialect preservation."""
    if not text:
        return text

    # 1) Normalize once upfront
    t = _shared_collapse(text) if _shared_collapse else text
    t = re.sub(r"[ًٌٍَُِّْ]+", "", t)

    # Collapse alif/hamza duplicates (preserve الله)
    try:
        HAMZA_DUP_RE = re.compile(r"(?<!الل)([أا])\1+")
        t = HAMZA_DUP_RE.sub(r"\1", t)
    except Exception:
        t = t.replace("أأ", "أ")

    # 2) Get normalized form for context detection (reuse for all checks)
    norm_t = _norm_func(t.lower()) if _norm_func else t.lower()

    # 3) Detect medical context
    MEDICAL_KEYWORDS = set(["لثة", "أسنان", "التهاب", "جيوب", "جير",
                           "خيط", "حساسية", "حشو", "خلع", "جذر"])
    MEDICAL_KEYWORDS.update(EXTRA_MEDICAL_KEYWORDS)
    has_medical = any(kw in norm_t for kw in MEDICAL_KEYWORDS)

    # 4) Detect dialect
    has_dialect = False
    if PRESERVE_DIALECT and EXTRA_DIALECT_TERMS:
        has_dialect = any(dt and dt in t.lower() for dt in EXTRA_DIALECT_TERMS)

    # 5) Apply replacements in ONE pass
    # Always apply: built-in confusion map (minimal now)
    for wrong, right in CONFUSION_MAP.items():
        t = re.sub(rf"(?<!\w){re.escape(wrong)}(?!\w)", right, t)

    # Always apply: vocab file replacements (medical terms, safe)
    for pat, right in COMPILED_EXTRA_REPLACEMENTS:
        t = pat.sub(right, t)

    # Conditionally apply ambiguous (only when safe)
    if has_medical and not has_dialect:
        # Full context → apply all ambiguous
        for pat, right in COMPILED_AMBIGUOUS_REPLACEMENTS:
            if DEBUG_AMBIGUOUS:
                logger.debug("AMBIG apply (global)", extra={"pattern": pat.pattern})
            t = pat.sub(right, t)
    elif has_medical:
        # Medical but dialect present → windowed application
        tokens = re.findall(r"[\w\u0600-\u06FF]+", norm_t)
        window = 3

        for (pat, right), (wrong_raw, _) in zip(COMPILED_AMBIGUOUS_REPLACEMENTS,
                                                  EXTRA_AMBIGUOUS.items()):
            try:
                wrong_norm = _norm_func(str(wrong_raw).lower())
                indices = [i for i, tok in enumerate(tokens) if tok == wrong_norm]

                apply = False
                for idx in indices:
                    start = max(0, idx - window)
                    end = min(len(tokens), idx + window + 1)
                    if set(tokens[start:end]) & EXTRA_MEDICAL_KEYWORDS_NORM:
                        apply = True
                        break

                if apply:
                    t = pat.sub(right, t)
            except Exception:
                continue

    return t

def correct_segments_inplace(segments: List[Dict[str, Any]]) -> None:
    """Apply post-processing to each segment text in-place."""
    if not segments:
        return
    for seg in segments:
        txt = (seg.get("text") or "").strip()
        if txt:
            seg["text"] = post_process_text(txt)

def apply_arabart_gec(text: str) -> str:
    """Apply AraBART grammar error correction.

    Args:
        text: Input Arabic text

    Returns:
        Corrected text or original if model unavailable
    """
    if not arabart_model or not arabart_tokenizer or not text:
        return text

    try:
        # AraBART expects sentences to be corrected individually for best results
        # Split on sentence boundaries
        sentences = re.split(r'[.!?؟،]\s*', text)
        corrected = []

        for sent in sentences:
            if not sent.strip():
                continue

            # Tokenize
            inputs = arabart_tokenizer(
                sent,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )

            if DEVICE == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate correction
            with torch.no_grad():
                outputs = arabart_model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )

            # Decode
            corrected_sent = arabart_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            corrected.append(corrected_sent.strip())

        return " ".join(corrected)

    except Exception as e:
        print(f"⚠️ AraBART correction failed: {e}")
        return text

def correct_segments_with_arabart(segments: List[Dict[str, Any]]) -> None:
    """Apply both rule-based and AraBART corrections to segments.

    Pipeline: ASR output → rule-based fixes → AraBART GEC
    """
    if not segments:
        return

    for seg in segments:
        txt = (seg.get("text") or "").strip()
        if txt:
            # Step 1: Rule-based corrections
            txt = post_process_text(txt)

            # Step 2: AraBART GEC (if enabled)
            if arabart_model:
                txt = apply_arabart_gec(txt)

            seg["text"] = txt

# ---------------------------
# Speaker Role Identification
# ---------------------------

def identify_speaker_roles(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify speaker roles (Doctor vs Patient) using heuristic patterns."""
    if not segments:
        return segments

    # Doctor question patterns
    doctor_patterns = [
        r'\bإيه\b', r'\bايه\b', r'\bفين\b', r'\bمين\b',
        r'\bامتى\b', r'\bمتى\b', r'\bليه\b', r'\bلماذا\b',
        r'\bإزاي\b', r'\bازاي\b', r'\bكيف\b',
        r'\bعندك\b', r'\bعندك أي\b', r'\bبتحس\b', r'\bتحس\b',
        r'\bبيوجع\b', r'\bيوجع\b',
        r'\bافتح\b', r'\bافتحي\b', r'\bورني\b', r'\bوريني\b',
        r'\bخليني أشوف\b', r'\bهشوف\b', r'\bفاحص\b', r'\bبفحص\b',
        r'\bعندك\b.*\b(التهاب|تسوس|مشكلة|جيوب)\b',
        r'\bواضح\b', r'\bبشوف\b', r'\bموجود\b',
        r'\bالتشخيص\b', r'\bالحالة\b',
        r'\bلازم\b.*\b(تعمل|نعمل|تاخد|ننضف)\b',
        r'\bهنعمل\b', r'\bهعمل\b', r'\bهكتب\b',
        r'\bالعلاج\b', r'\bالخطة\b', r'\bوصفة\b', r'\bدواء\b', r'\bمضاد\b',
        r'\bجيوب لثوية\b', r'\bقناة الجذر\b', r'\bالتكلس\b',
        r'\bScaling\b', r'\bRoot planing\b',
        r'\bالأشعة\b.*\b(بانوراما|سينية)\b',
        r'\bهنعمل\b', r'\bهنشوف\b', r'\bنتابع\b',
    ]

    patient_patterns = [
        r'\bبحس\b.*\b(بألم|بوجع|حاسس)\b',
        r'\bعندي\b.*\b(ألم|وجع|تورم|نزيف)\b',
        r'\bبيوجعني\b', r'\bموجعني\b', r'\bمؤلم\b',
        r'\bمش قادر\b', r'\bمش عارف\b',
        r'\bمن\b.*\b(أسبوع|شهر|يوم|ساعة)\b',
        r'\bبقالي\b', r'\bبقاله\b', r'\bمن زمان\b', r'\bمن فترة\b',
        r'\bحاسس\b', r'\bشاعر\b', r'\bبحس إن\b',
        r'\bمش مرتاح\b', r'\bمتضايق\b',
        r'^(آه|أيوة|لا|مش|ممكن|طيب)$',
        r'^\w{1,5}$',
    ]

    first_person_patterns = [
        r'\bأنا\b', r'\bانا\b', r'\bأني\b', r'\bعندي\b',
        r'\bبحس\b', r'\bحاسة\b', r'\bحسيت\b',
        r'\bبغسل\b', r'\bبستخدم\b'
    ]

    doctor_regexes = [re.compile(p, re.IGNORECASE) for p in doctor_patterns]
    patient_regexes = [re.compile(p, re.IGNORECASE) for p in patient_patterns]
    first_person_regexes = [re.compile(p, re.IGNORECASE) for p in first_person_patterns]

    # Score each segment ONCE
    role_scores = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            role_scores.append({"doctor": 0, "patient": 0})
            continue

        doctor_score = sum(1 for r in doctor_regexes if r.search(text))
        patient_score = sum(1 for r in patient_regexes if r.search(text))
        fp_count = sum(1 for r in first_person_regexes if r.search(text))

        if fp_count:
            patient_score += fp_count * 2
        if len(text.split()) > 15:
            doctor_score += 1
        if len(text.split()) <= 3:
            patient_score += 1

        role_scores.append({"doctor": doctor_score, "patient": patient_score})

    # Assign roles
    speaker_ids = [seg.get("speaker", "SPEAKER_00") for seg in segments]
    unique_speakers = list(dict.fromkeys(speaker_ids))
    role_map = {}

    if len(unique_speakers) == 2:
        speaker_totals = {spk: {"doctor": 0, "patient": 0} for spk in unique_speakers}
        fp_counts = {spk: 0 for spk in unique_speakers}

        for i, seg in enumerate(segments):
            spk = speaker_ids[i]
            speaker_totals[spk]["doctor"] += role_scores[i]["doctor"]
            speaker_totals[spk]["patient"] += role_scores[i]["patient"]

            text = seg.get("text", "")
            fp_counts[spk] += sum(1 for r in first_person_regexes if r.search(text))

        for spk in unique_speakers:
            doc_total = speaker_totals[spk]["doctor"]
            pat_total = speaker_totals[spk]["patient"]
            other = unique_speakers[1] if unique_speakers[0] == spk else unique_speakers[0]

            if doc_total > pat_total:
                if speaker_totals[other]["doctor"] > speaker_totals[other]["patient"] \
                   and fp_counts[spk] > fp_counts[other]:
                    role_map[spk] = "مريض"
                else:
                    role_map[spk] = "طبيب"
            elif pat_total > doc_total:
                role_map[spk] = "مريض"
            else:
                role_map[spk] = "مريض" if fp_counts[spk] >= fp_counts[other] else "طبيب"

    elif len(unique_speakers) == 1:
        role_map = {unique_speakers[0]: "طبيب"}

    else:
        for spk in unique_speakers:
            spk_segments = [i for i, s in enumerate(speaker_ids) if s == spk]
            total_doc = sum(role_scores[i]["doctor"] for i in spk_segments)
            total_pat = sum(role_scores[i]["patient"] for i in spk_segments)
            role_map[spk] = "طبيب" if total_doc >= total_pat else "مريض"

    # Apply role mapping
    for seg in segments:
        original_speaker = seg.get("speaker", "Unknown")
        seg["speaker"] = role_map.get(original_speaker, original_speaker)

    return segments

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="ASR Service (WhisperX)")

# CORS: configurable via env, default to localhost only
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ALLOWED_ORIGINS],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "x-internal-secret", "x-correlation-id"],
)

# Optional OTEL
try:
    from otel_setup import init_otel
    init_otel("asr", app=app)
except Exception:
    logger.debug("OTEL init skipped for ASR")

INTERNAL_SECRET = os.getenv("INTERNAL_SECRET", "")

@app.middleware("http")
async def internal_auth(request: Request, call_next):
    # Allow health/ready/metrics without auth
    if request.url.path.startswith("/health") or request.url.path.startswith("/ready") or request.url.path.startswith("/metrics"):
        return await call_next(request)
    # Use constant-time comparison to prevent timing attacks
    import hmac
    provided_secret = request.headers.get("x-internal-secret") or ""
    if not INTERNAL_SECRET or not hmac.compare_digest(provided_secret, INTERNAL_SECRET):
        raise HTTPException(status_code=403, detail="Unauthorized")
    return await call_next(request)

@app.middleware("http")
async def _asr_postprocess_middleware(request: Request, call_next):
    """Middleware that post-processes /transcribe JSON responses."""
    response = await call_next(request)
    try:
        if request.url.path == "/transcribe" and response.status_code == 200 \
           and response.headers.get("content-type", "").startswith("application/json"):

            body = b""
            if hasattr(response, 'body_iterator'):
                async for chunk in response.body_iterator:
                    body += chunk
            else:
                body = getattr(response, 'body', b"") or b""

            if not body:
                return response

            try:
                data = json.loads(body.decode('utf-8'))
            except Exception:
                return response

            if os.getenv("ENABLE_ASR_POSTPROCESS", "true").lower() == "true":
                data["text_cleaned"] = post_process_text(data.get("text", ""))
                segs = data.get("segments", [])
                for s in segs:
                    s["clean_text"] = post_process_text(s.get("text", ""))

                    orig_seg_text = s.get("text", "") or ""
                    cleaned_seg_text = s.get("clean_text", "") or ""
                    for token in DIALECT_PRESERVE:
                        if token and token in orig_seg_text and token not in cleaned_seg_text:
                            cleaned_tokens = cleaned_seg_text.split()
                            match = _difflib.get_close_matches(token, cleaned_tokens, n=1, cutoff=0.6)
                            if match:
                                cleaned_seg_text = cleaned_seg_text.replace(match[0], token)
                    s["clean_text"] = cleaned_seg_text

                data["segments"] = segs

                orig_top = data.get("text", "") or ""
                top_clean = data.get("text_cleaned", "") or ""
                for token in DIALECT_PRESERVE:
                    if token and token in orig_top and token not in top_clean:
                        top_tokens = top_clean.split()
                        match = _difflib.get_close_matches(token, top_tokens, n=1, cutoff=0.6)
                        if match:
                            top_clean = top_clean.replace(match[0], token)
                data["text_cleaned"] = top_clean

            new_body = json.dumps(data, ensure_ascii=False).encode('utf-8')
            return Response(content=new_body, status_code=response.status_code, media_type="application/json")
    except Exception:
        return response
    return response

# ---------------------------
# Models (globals, loaded once)
# ---------------------------
whisper_model = None
align_model_a = None
align_metadata = None
diarize_model = None
arabart_tokenizer = None
arabart_model = None
stream_buffers: Dict[str, List[bytes]] = {}
stream_lock = asyncio.Lock()
stream_meta: Dict[str, Dict[str, Any]] = {}

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
    audio: str  # base64
    dialect: Optional[str] = "egypt"
    language: Optional[str] = "ar"
    enable_diarization: Optional[bool] = True
    enable_alignment: Optional[bool] = False
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    diarize_first: Optional[bool] = None

class NormalizeRequest(BaseModel):
    text: str

class TranscriptionResponse(BaseModel):
    text: str
    segments: List[TranscriptionSegment]
    language: str
    duration: float
    processing_time: float
    rtf: float
    speakers: Optional[List[str]] = None
    model_used: str
    pipeline_mode: str


class StreamChunkRequest(BaseModel):
    audio: str  # base64 chunk
    sessionId: str
    format: Optional[str] = "mulaw"
    sampleRate: Optional[int] = 8000
    isFinal: Optional[bool] = False


class StreamChunkResponse(BaseModel):
    partial: str = ""
    final: Optional[str] = None
    isFinal: bool = False

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

def decode_mulaw_chunk(audio_base64: str, sample_rate: int = 8000) -> np.ndarray:
    """Decode mulaw 8k chunk to float32 16k mono."""
    try:
        pcm_mulaw = base64.b64decode(audio_base64)
        pcm16 = audioop.ulaw2lin(pcm_mulaw, 2)
        wave = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != 16000:
            wave = torchaudio.functional.resample(torch.from_numpy(wave), sample_rate, 16000).numpy()
        return wave
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode mulaw chunk: {str(e)}")

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

# ==== One-time kwarg filter ====
_SUPPORTED_TRANSCRIBE_KW: Optional[set] = None

def _prepare_supported_kwargs(model) -> set:
    global _SUPPORTED_TRANSCRIBE_KW
    if _SUPPORTED_TRANSCRIBE_KW is not None:
        return _SUPPORTED_TRANSCRIBE_KW
    try:
        sig = inspect.signature(model.transcribe)
        _SUPPORTED_TRANSCRIBE_KW = {p.name for p in sig.parameters.values()}
    except Exception:
        _SUPPORTED_TRANSCRIBE_KW = {"audio", "batch_size", "language", "task"}
    return _SUPPORTED_TRANSCRIBE_KW

def _transcribe_filtered(model, audio: np.ndarray, **kwargs) -> Dict[str, Any]:
    supported = _prepare_supported_kwargs(model)
    clean = {k: v for k, v in kwargs.items() if k in supported}
    return model.transcribe(audio, **clean)

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
    result = _transcribe_filtered(whisper_model, audio_f32, **decode_kwargs)

    for s in result.get("segments", []) or []:
        s["start"] = float(s.get("start", 0.0) or 0.0) + start_offset_s
        s["end"] = float(s.get("end", 0.0) or 0.0) + start_offset_s

    # Apply corrections (rule-based + AraBART if enabled)
    if ENABLE_ARABART and arabart_model:
        correct_segments_with_arabart(result.get("segments", []))
    else:
        correct_segments_inplace(result.get("segments", []))

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
    segs = aligned.get("segments", result_segments)
    # Re-apply corrections after alignment (alignment may change text slightly)
    if ENABLE_ARABART and arabart_model:
        correct_segments_with_arabart(segs)
    else:
        correct_segments_inplace(segs)
    return segs

# ---------------------------
# Startup: load & cache models once
# ---------------------------
@app.on_event("startup")
async def load_models():
    global whisper_model, align_model_a, align_metadata, diarize_model

    logger.info(
        "Loading WhisperX model",
        extra={"model": WHISPER_MODEL, "device": DEVICE, "computeType": COMPUTE_TYPE},
    )
    # VAD model path - can be pre-downloaded to avoid broken S3 URL issue
    vad_model_fp = os.getenv("VAD_MODEL_PATH", None)
    whisper_model = whisperx.load_model(
        WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE, language="ar",
        vad_model_fp=vad_model_fp
    )
    logger.info("Base WhisperX model loaded")

    if ENABLE_ALIGNMENT:
        try:
            if not os.getenv("HUGGINGFACE_HUB_TOKEN") and HF_TOKEN:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
            align_model_a, align_metadata = whisperx.load_align_model(
                language_code="ar",
                device=DEVICE,
                model_name=ALIGNMENT_MODELS["ar"],
            )
            logger.info("Alignment model cached", extra={"language": "ar"})
        except Exception as e:
            align_model_a = None
            align_metadata = None
            logger.warning("Could not cache Arabic aligner", extra={"error": str(e)})
    else:
        align_model_a = None
        align_metadata = None
        logger.info("Alignment disabled; skipping align model load")

    diar_loaded = False
    if ENABLE_DIARIZATION:
        if not HF_TOKEN:
            logger.info("Diarization disabled due to missing HF_TOKEN")
        else:
            try:
                from pyannote.audio import Pipeline
                try:
                    diarize_model = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=HF_TOKEN)
                except Exception as e_v32:
                    logger.warning(
                        "Primary diarization model failed, falling back",
                        extra={"model": PYANNOTE_MODEL, "error": str(e_v32)},
                    )
                    diarize_model = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
                    )
                if DEVICE == "cuda":
                    diarize_model.to(torch.device("cuda"))
                diar_loaded = True
            except Exception as e:
                diarize_model = None
                logger.warning("Diarization disabled", extra={"error": str(e)})
    else:
        logger.info("Diarization disabled via config")

    logger.info(
        "ASR service ready",
        extra={
            "model": WHISPER_MODEL,
            "device": DEVICE,
            "computeType": COMPUTE_TYPE,
            "alignerCached": align_model_a is not None,
            "diarization": bool(diar_loaded and diarize_model is not None),
        },
    )

# ---------------------------
# API endpoints
# ---------------------------
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    transcriptions_total.inc()
    overall_start = time.time()

    try:
        audio_data, sample_rate = decode_audio(request.audio)
        audio_duration = len(audio_data) / float(sample_rate + 1e-12)

        log_safe(
            logging.INFO,
            "ASR transcription request",
            duration_sec=round(audio_duration, 2),
            sample_rate=sample_rate,
            language=request.language,
            diarization=bool(request.enable_diarization and (diarize_model is not None)),
            mode="diarize-first" if (request.diarize_first if request.diarize_first is not None else DIARIZE_FIRST) else "diarize-last",
        )

        wave = torch.from_numpy(audio_data).float()
        if wave.dim() > 1:
            wave = wave.mean(dim=1)
        wave = wave.unsqueeze(0)
        if sample_rate != 16000:
            wave = torchaudio.functional.resample(wave, sample_rate, 16000)
        wave = wave.squeeze(0)
        audio = wave.numpy().astype(np.float32)
        sample_rate = 16000

        batch_sz = 1 if DEVICE in ("cpu", "mps") else 16
        use_diarize_first = (request.diarize_first if request.diarize_first is not None else DIARIZE_FIRST) \
                            and (request.enable_diarization and diarize_model is not None)
        use_alignment = request.enable_alignment if request.enable_alignment is not None else ENABLE_ALIGNMENT
        use_alignment = bool(use_alignment) and ENABLE_ALIGNMENT

        detected_lang = request.language or "ar"

        if use_diarize_first:
            logger.info("ASR diarize-first path")
            diarization_start = time.time()
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            diarize_annotation = diarize_model(
                {"waveform": waveform, "sample_rate": 16000},
                min_speakers=request.min_speakers,
                max_speakers=request.max_speakers,
            )
            diarize_time = time.time() - diarization_start
            logger.debug("Diarization complete", extra={"durationSec": round(diarize_time, 2)})

            diar_segments: List[Tuple[float, float, str]] = [
                (seg.start, seg.end, spk) for seg, _, spk in diarize_annotation.itertracks(yield_label=True)
            ]
            diar_segments.sort(key=lambda x: x[0])

            segments: List[Dict[str, Any]] = []
            logger.info("Transcribing diarized chunks", extra={"chunks": len(diar_segments)})
            t_start = time.time()
            for (s, e, spk) in diar_segments:
                s = float(s); e = float(e)
                s_samp = int(max(0.0, s) * sample_rate)
                e_samp = int(min(len(audio) / sample_rate, e) * sample_rate)
                if e_samp <= s_samp:
                    continue
                chunk = audio[s_samp:e_samp].copy()
                result = transcribe_chunk(
                    chunk, detected_lang, request.dialect or "egypt", batch_sz, start_offset_s=s
                )
                for seg in result.get("segments", []) or []:
                    seg["speaker"] = spk
                segments.extend(result.get("segments", []))
            transcribe_time = time.time() - t_start
            logger.debug("Per-speaker transcription done", extra={"durationSec": round(transcribe_time, 2)})

            if use_alignment and align_model_a is not None and segments:
                logger.debug("Running global alignment")
                segments = align_segments(segments, audio, detected_lang)
                logger.debug("Alignment complete")

                diarize_df = DataFrame(
                    diarize_annotation.itertracks(yield_label=True),
                    columns=["segment", "label", "speaker"],
                )
                diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
                diarize_df["end"]   = diarize_df["segment"].apply(lambda x: x.end)
                with_spk = whisperx.assign_word_speakers(diarize_df, {"segments": segments})
                if isinstance(with_spk, dict) and "segments" in with_spk:
                    segments = with_spk["segments"]

            segments = identify_speaker_roles(segments)

            full_text = " ".join([(s.get("text") or "").strip() for s in segments])
            formatted_segments = format_segments_for_frontend(segments)

            total_time = time.time() - overall_start
            rtf_value = (total_time / audio_duration) if audio_duration > 0 else 0.0
            transcription_duration.observe(total_time)
            rtf_ratio.observe(rtf_value)

            speakers_list = sorted(list({s.get("speaker") for s in segments if s.get("speaker")})) or None
            log_safe(
                logging.INFO,
                "ASR diarize-first complete",
                duration_sec=round(total_time, 2),
                rtf=round(rtf_value, 3),
                segments=len(formatted_segments),
            )

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
            logger.info("ASR diarize-last path")
            t0 = time.time()
            result = transcribe_chunk(audio, detected_lang, request.dialect or "egypt", batch_sz, 0.0)
            detected_lang = result.get("language", detected_lang)
            t1 = time.time() - t0
            logger.debug("Transcription done", extra={"durationSec": round(t1, 2)})

            if use_alignment and align_model_a is not None and result.get("segments"):
                logger.debug("Word-level alignment")
                result["segments"] = align_segments(result["segments"], audio, detected_lang)
                logger.debug("Alignment complete")

            speakers_list = None
            if request.enable_diarization and diarize_model is not None:
                logger.debug("Assigning speakers via diarization")
                waveform = torch.from_numpy(audio).float().unsqueeze(0)
                diarize_annotation = diarize_model(
                    {"waveform": waveform, "sample_rate": 16000},
                    min_speakers=request.min_speakers,
                    max_speakers=request.max_speakers,
                )
                if use_alignment:
                    diarize_df = DataFrame(
                        diarize_annotation.itertracks(yield_label=True),
                        columns=["segment", "label", "speaker"],
                    )
                    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
                    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
                    with_spk = whisperx.assign_word_speakers(diarize_df, result)
                    if isinstance(with_spk, dict) and "segments" in with_spk:
                        result["segments"] = with_spk["segments"]
                    speakers_list = sorted(list({s.get("speaker") for s in result["segments"] if s.get("speaker")})) or None

            segments = result.get("segments", [])
            segments = identify_speaker_roles(segments)

            # LLM correction (if enabled)
            if ENABLE_LLM_CORRECTION and LLM_CORRECTOR_AVAILABLE:
                logger.debug("Applying LLM medical correction")
                try:
                    segments = llm_correct_segments(segments, context="محادثة طبية في عيادة أسنان")
                    for seg in segments:
                        if "text_llm_corrected" in seg:
                            seg["text"] = seg["text_llm_corrected"]
                    logger.debug("LLM correction complete")
                except Exception as e:
                    logger.warning("LLM correction failed", extra={"error": str(e)})

            full_text = " ".join([(s.get("text") or "").strip() for s in segments])
            formatted_segments = format_segments_for_frontend(segments)

            total_time = time.time() - overall_start
            rtf_value = (total_time / audio_duration) if audio_duration > 0 else 0.0
            transcription_duration.observe(total_time)
            rtf_ratio.observe(rtf_value)

            log_safe(
                logging.INFO,
                "ASR diarize-last complete",
                duration_sec=round(total_time, 2),
                rtf=round(rtf_value, 3),
                segments=len(formatted_segments),
            )

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
        logger.exception("ASR transcription failed", extra={"error": str(type(e).__name__)})
        raise HTTPException(status_code=500, detail="Transcription failed")


@app.post("/stream/chunk", response_model=StreamChunkResponse)
async def stream_chunk(request: StreamChunkRequest):
    """
    Streaming endpoint for low-latency partial transcripts.
    - Accepts small mulaw chunks (8k) from gateway.
    - Buffers per session with simple back-pressure (max 10 chunks).
    - Returns best-effort partial transcript; on isFinal=true flushes final.
    """
    if not request.sessionId:
        raise HTTPException(status_code=400, detail="sessionId required")

    now = time.time()
    silence_window_ms = float(os.getenv("STREAM_SILENCE_MS", "1200"))
    rms_threshold = float(os.getenv("STREAM_SILENCE_RMS", "300"))

    async with stream_lock:
        buf = stream_buffers.get(request.sessionId) or []
        if len(buf) > 10:
            # back-pressure: drop oldest chunk
            buf = buf[-10:]
        buf.append(base64.b64decode(request.audio))
        stream_buffers[request.sessionId] = buf
        meta = stream_meta.get(request.sessionId) or {"last_ts": now, "silence_count": 0}
        stream_meta[request.sessionId] = meta

    # Merge buffered audio
    chunks = stream_buffers.get(request.sessionId, [])
    merged = b"".join(chunks)

    # Decode and run quick transcription
    audio_np = decode_mulaw_chunk(base64.b64encode(merged).decode("utf-8"), request.sampleRate or 8000)
    # Simple silence detection
    rms = audioop.rms(base64.b64decode(request.audio), 2) if request.audio else 0
    final_due_to_silence = False
    async with stream_lock:
        meta = stream_meta.get(request.sessionId) or {"last_ts": now, "silence_count": 0}
        last_ts = meta.get("last_ts", now)
        if rms < rms_threshold:
            meta["silence_count"] = meta.get("silence_count", 0) + 1
        else:
            meta["silence_count"] = 0
        # Time-based
        if (now - last_ts) * 1000 > silence_window_ms:
            final_due_to_silence = True
        # Consecutive silence
        if meta.get("silence_count", 0) >= 3:
            final_due_to_silence = True
        meta["last_ts"] = now
        stream_meta[request.sessionId] = meta

    batch_sz = 1 if DEVICE in ("cpu", "mps") else 4
    result = transcribe_chunk(audio_np, "ar", "egypt", batch_sz, 0.0)
    partial_text = " ".join([(s.get("text") or "").strip() for s in result.get("segments", []) or []]).strip()

    if request.isFinal or final_due_to_silence:
        async with stream_lock:
            stream_buffers.pop(request.sessionId, None)
            stream_meta.pop(request.sessionId, None)
        return StreamChunkResponse(partial=partial_text, final=partial_text, isFinal=True)

    return StreamChunkResponse(partial=partial_text, isFinal=False)

# ---------------------------
# Health & aux endpoints
# ---------------------------
@app.get("/health")
async def health_check():
    return {"ok": True, "service": "asr"}


@app.get("/ready")
async def readiness():
    return {
        "ready": whisper_model is not None,
        "model": WHISPER_MODEL,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "aligner_cached": align_model_a is not None,
        "diarization_enabled": diarize_model is not None and ENABLE_DIARIZATION,
        "vocab_terms_builtin": len(BUILTIN_MEDICAL_TERMS),
        "extra_replacements": len(EXTRA_REPLACEMENTS),
    }

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/normalize")
async def normalize_endpoint(req: NormalizeRequest):
    """Expose normalization and correction as a service."""
    raw = req.text or ""
    norm = post_process_text(raw.strip())
    return {"input": raw, "normalized": norm}

class VocabUpdateRequest(BaseModel):
    wrong: str
    right: str
    ambiguous: Optional[bool] = False

@app.post("/vocab/update")
async def vocab_update(req: VocabUpdateRequest):
    """Accept a user correction (wrong->right)."""
    path = Path(VOCAB_FILE) if VOCAB_FILE else Path(__file__).parent / "medical_vocab_ar_en.json"
    if not path.exists():
        return {"error": f"vocab file not found: {path}"}
    try:
        data = _read_json_any(path)
        if not isinstance(data, dict):
            data = {"terms": [], "replacements": {}, "ambiguous_replacements": {}}
        if req.ambiguous:
            amb = data.get("ambiguous_replacements", {}) or {}
            amb[req.wrong] = req.right
            data["ambiguous_replacements"] = amb
        else:
            repl = data.get("replacements", {}) or {}
            repl[req.wrong] = req.right
            data["replacements"] = repl

        _write_vocab_json_safe(path, data)
        status = reload_vocab_from_file()
        return {"status": "ok", "updated": {req.wrong: req.right}, "reload": status}
    except Exception as e:
        logger.exception("Failed to update vocab", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Failed to update vocab")

@app.get("/vocab")
async def vocab_info():
    """Quick helper to inspect loaded vocab & replacements."""
    return {
        "builtin_terms": len(BUILTIN_MEDICAL_TERMS),
        "extra_terms": EXTRA_TERMS[:64],
        "extra_terms_count": len(EXTRA_TERMS),
        "replacements_count": len(EXTRA_REPLACEMENTS),
    }

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ASR_PORT", os.getenv("PORT", 5000)))
    host = os.getenv("ASR_HOST", os.getenv("HOST", "0.0.0.0"))
    uvicorn.run(app, host=host, port=port)
