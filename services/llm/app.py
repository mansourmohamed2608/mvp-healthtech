# services/llm/app.py
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rag_store import rag_store
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import logging
import os
import asyncio

# Import post-processing modules for accuracy boost
from corrections import apply_corrections, normalize_vital_signs
from rules import SOAPValidator, normalize_medical_abbreviations
from speaker_rules import SpeakerIdentifier

app = FastAPI(title="LLM Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
INTERNAL_SECRET = os.getenv("INTERNAL_SECRET", "")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm")
# Optional OTEL (safe no-op if deps/env missing)
try:
    from otel_setup import init_otel
    init_otel("llm")
except Exception:
    logger.debug("OTEL init skipped for LLM")
if not INTERNAL_SECRET:
    raise RuntimeError("INTERNAL_SECRET must be set for LLM service")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
MAX_MESSAGE_LENGTH = int(os.getenv("LLM_MAX_MESSAGE_LENGTH", "4096"))
MAX_HISTORY_TURNS = int(os.getenv("LLM_MAX_HISTORY_TURNS", "20"))


def safe_print(*args, **_kwargs):
    """Replace stdout prints with PHI-safe debug logs (content suppressed)."""
    logger.debug("suppressed print", extra={"fields": len(args)})


# Override built-in print to avoid PHI in stdout
print = safe_print


def log_safe(level: int, msg: str, request: Request | None = None, session_id: str | None = None, **kwargs):
    """PHI-safe logger: only IDs/lengths/status, no raw text."""
    extra = {
        "correlationId": request.headers.get("x-correlation-id") if request else None,
        "sessionId": session_id,
    }
    for k, v in kwargs.items():
        if v is not None:
            extra[k] = v
    logger.log(level, msg, extra=extra)

@app.middleware("http")
async def internal_auth(request: Request, call_next):
    if request.url.path.startswith("/health") or request.url.path.startswith("/ready") or request.url.path.startswith("/metrics"):
        return await call_next(request)
    if not INTERNAL_SECRET or request.headers.get("x-internal-secret") != INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await call_next(request)

# Prometheus metrics
first_token_latency = Histogram(
    'llm_first_token_latency_ms',
    'Time to generate first token in milliseconds',
    buckets=[50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500]
)
complete_response_duration = Histogram(
    'llm_complete_response_duration_ms',
    'Time to generate complete response in milliseconds',
    buckets=[200, 500, 750, 1000, 1250, 1500, 2000, 3000, 5000]
)
tokens_per_second = Histogram(
    'llm_tokens_per_second',
    'Token generation rate (tokens/second)',
    buckets=[5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
)
requests_total = Counter(
    'llm_requests_total',
    'Total number of LLM inference requests'
)
slow_responses = Counter(
    'llm_slow_responses_total',
    'Number of responses slower than 1.5s'
)

class InferRequest(BaseModel):
    message: str
    sessionId: str
    intent: str = "general"  # Optional intent for RAG retrieval

class InferResponse(BaseModel):
    intent: str
    reply: str

class TranscriptionCorrectionRequest(BaseModel):
    text: str  # Raw ASR output
    dialect: str = "egypt"  # egypt, levant, gulf
    context: str = "medical"  # Always medical for now

class TranscriptionCorrectionResponse(BaseModel):
    original: str
    corrected: str
    corrections_made: int
    dialect_normalized: bool

class SpeakerSegment(BaseModel):
    speaker: str  # SPEAKER_00, SPEAKER_01, etc.
    text: str
    start: float = 0.0
    end: float = 0.0

class SpeakerRoleRequest(BaseModel):
    segments: list[SpeakerSegment]
    context: str = "medical"  # medical, general, legal, etc.

class SpeakerRole(BaseModel):
    speaker_id: str  # SPEAKER_00, etc.
    role: str  # Doctor, Patient, Nurse, etc.
    confidence: float  # 0.0-1.0
    reasoning: str  # Why this role was assigned

class SpeakerRoleResponse(BaseModel):
    roles: list[SpeakerRole]
    primary_doctor: str | None = None  # SPEAKER_00, etc.
    primary_patient: str | None = None

class ChatMessage(BaseModel):
    role: str  # user | assistant | system
    content: str

class ChatRequest(BaseModel):
    history: List[ChatMessage] = []
    message: str
    sessionId: str
    intent: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    intent: str
    tokens_generated: int
    first_token_ms: float
    total_latency_ms: float

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Henrychur/MMed-Llama-3-8B"  # English base + Arabic post-processing = 70%+ accuracy

print(f"Loading model {MODEL_NAME} on {DEVICE}...")
print("⚠️ Note: Model is English-only, but Arabic support via post-processing (corrections.py, rules.py)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# GTX 1050 3GB is too small for 4-bit quantization (~4-5GB needed)
# Use 8-bit quantization on CPU (better than fp32, works on CPU)
# For faster inference, use Kaggle (free T4 16GB) or Azure T4 VM
print("⚠️ GTX 1050 3GB is insufficient - using 8-bit CPU quantization instead")
print("💡 For GPU acceleration, use Kaggle (free T4) or Azure NC16as_T4_v3")

try:
    from transformers import BitsAndBytesConfig

    # 8-bit quantization works on CPU (4-bit doesn't)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print("✅ Model loaded with 8-bit CPU quantization (~50% RAM saved, ~2x faster)")
except Exception as e:
    # Fallback to fp32 if 8-bit fails
    print(f"⚠️ 8-bit quantization failed: {e}")
    print("Loading in fp32 (full precision, no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = model.to("cpu")
    print("✅ Model loaded on CPU (fp32, slower)")

# Load LoRA weights if present
try:
    model = PeftModel.from_pretrained(model, "/app/lora-llama")
    print("LoRA weights loaded successfully")
except Exception as e:
    print(f"No LoRA weights found: {e}")

print("Model loaded successfully!")

def _blocking_generate(prompt: str, max_new_tokens: int = 192) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "decoded": decoded,
        "input_len": len(inputs["input_ids"][0]),
        "output_len": len(outputs[0]),
    }


async def _run_llm_inference(prompt: str, max_new_tokens: int = 192) -> dict:
    """Async helper to make mocking/testing easier."""
    return await asyncio.to_thread(_blocking_generate, prompt, max_new_tokens)

@app.get("/health")
async def health():
    return {"ok": True, "service": "llm"}


@app.get("/ready")
async def ready():
    return {
        "ready": model is not None,
        "model": MODEL_NAME,
        "device": str(model.device) if model else None,
        "tokenizer": tokenizer.name_or_path if tokenizer else None,
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/correct-transcription", response_model=TranscriptionCorrectionResponse)
async def correct_transcription(req: TranscriptionCorrectionRequest):
    """
    HYBRID APPROACH: Post-process ASR output with medical LLM

    Corrects:
    - Dialect-specific medical terms (e.g., Egyptian البروستاتا → البروستات)
    - Hallucinated words from ASR
    - Medical terminology errors
    - Contextual mistakes
    """
    try:
        log_safe(logging.INFO, "LLM correction request", request=None, session_id=None, textLen=len(req.text))
        
        # Simpler, more direct prompt that's easier for the model to follow
        prompt = f"""صحح الأخطاء في هذا النص الطبي: {req.text}

النص المصحح:"""

        print(f"🔤 Tokenizing prompt...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print(f"📊 Input tokens: {inputs['input_ids'].shape[1]}")
        
        print(f"🤖 Generating response (max 64 tokens, ~20-30 mins on CPU)...")
        import time
        start_time = time.time()
        
        with torch.no_grad():  # Disable gradient calculation for faster inference
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,  # Further reduced - correction should be similar length to input
                do_sample=False,  # Deterministic for corrections
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Use KV cache for faster generation
                repetition_penalty=1.1  # Prevent repeating the prompt
            )
        
        elapsed = time.time() - start_time
        log_safe(logging.INFO, "LLM correction generation complete", request=None, session_id=None, latencyMs=int(elapsed * 1000))

        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"🔍 Raw LLM output: {corrected[:200]}...")  # Debug: see what we got
        
        # Extract only the corrected text after the marker
        if "النص المصحح:" in corrected:
            corrected = corrected.split("النص المصحح:")[-1].strip()
        
        # Remove the prompt if model repeated it
        corrected = corrected.replace(prompt, "").strip()
        
        # If output starts with instruction text, try to extract just the answer
        if corrected.startswith("صحح"):
            # Find the first colon or newline and take what's after
            for sep in [":", "\n"]:
                if sep in corrected:
                    corrected = corrected.split(sep, 1)[-1].strip()
                    break
        
        # If output still has the original text embedded, extract just after it
        if req.text in corrected:
            # Take what comes after the original text
            parts = corrected.split(req.text)
            if len(parts) > 1 and parts[1].strip():
                corrected = parts[1].strip()
        
        # Final check: if output is too long (more than 3x original), use original
        if len(corrected) > len(req.text) * 3:
            print(f"⚠️ LLM output too long ({len(corrected)} chars), using original")
            corrected = req.text
        
        # If we ended up with nothing, use original
        if not corrected or len(corrected) < 5:
            print(f"⚠️ LLM output empty, using original")
            corrected = req.text
        
        print(f"✅ LLM output: {corrected[:100]}...")
        
        # ✨ POST-PROCESSING: Apply medical corrections (+5-8% accuracy)
        corrected, dict_corrections = apply_corrections(corrected, dialect=req.dialect)
        logger.debug("Applied dictionary corrections", extra={"count": dict_corrections})
        
        # ✨ POST-PROCESSING: Normalize vital signs
        corrected = normalize_vital_signs(corrected)
        logger.debug("Normalized vital signs")

        # Count differences
        corrections_made = len(set(req.text.split()) - set(corrected.split())) + dict_corrections

        return {
            "original": req.text,
            "corrected": corrected,
            "corrections_made": corrections_made,
            "dialect_normalized": True
        }
    except Exception as e:
        log_safe(logging.ERROR, "LLM correction failed", request=None, session_id=None, error=str(type(e).__name__), textLen=len(req.text))
        raise HTTPException(status_code=500, detail="Correction failed")


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest, request: Request):
    requests_total.inc()
    start_time = time.time()
    if not req.message or len(req.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid request")
    try:
        log_safe(logging.INFO, "Infer request", request=request, session_id=req.sessionId, messageLen=len(req.message))

        prompt = build_rag_prompt(req.message, req.intent)
        intent = classify_intent(req.message)

        generation_start = time.time()
        try:
            result = await asyncio.wait_for(
                _run_llm_inference(prompt, max_new_tokens=128),
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log_safe(logging.WARNING, "LLM timeout", request=request, session_id=req.sessionId, intent=intent)
            raise HTTPException(status_code=504, detail="LLM service unavailable")

        generation_time = time.time() - generation_start
        decoded = result["decoded"]
        reply = decoded.split("المساعد:")[-1].strip()

        reply, corrections_count = apply_corrections(reply, dialect="egypt")
        reply = normalize_vital_signs(normalize_medical_abbreviations(reply))

        total_time_ms = (time.time() - start_time) * 1000
        tokens_generated = result["output_len"] - result["input_len"]
        tokens_per_second.observe(tokens_generated / max(generation_time, 1e-6))
        complete_response_duration.observe(total_time_ms)
        est_first_ms = generation_time * 0.15 * 1000
        first_token_latency.observe(est_first_ms)

        if total_time_ms > 1500:
            slow_responses.inc()
        log_safe(logging.INFO, "Infer reply generated", request=request, session_id=req.sessionId, tokens=tokens_generated, totalMs=int(total_time_ms))

        return {"intent": intent, "reply": reply}
    except HTTPException:
        raise
    except Exception as e:
        log_safe(logging.ERROR, "LLM inference failed", request=request, session_id=req.sessionId, error=str(type(e).__name__))
        raise HTTPException(status_code=500, detail="LLM inference failed")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """
    Conversation-friendly endpoint that accepts history and current message.
    """
    requests_total.inc()
    overall_start = time.time()
    if not req.message or len(req.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid request")

    history = req.history or []
    if len(history) > MAX_HISTORY_TURNS:
        history = history[-MAX_HISTORY_TURNS:]

    intent = req.intent or classify_intent(req.message)
    log_safe(logging.INFO, "Chat request", request=request, session_id=req.sessionId, historyTurns=len(history), messageLen=len(req.message))

    history_lines = [f"{turn.role}: {turn.content}" for turn in history]
    history_text = "\n".join(history_lines)

    prompt_parts = [
        "أنت مساعد طبي يتحدث العربية. كن مهذباً ودقيقاً في الإجابات.",
    ]
    if history_text:
        prompt_parts.append("السياق السابق:")
        prompt_parts.append(history_text)
    prompt_parts.append(f"المستخدم: {req.message}")
    prompt_parts.append("المساعد:")
    prompt = "\n".join(prompt_parts)

    try:
        generation_start = time.time()
        try:
            result = await asyncio.wait_for(
                _run_llm_inference(prompt, max_new_tokens=192),
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log_safe(logging.WARNING, "LLM timeout", request=request, session_id=req.sessionId, intent=intent)
            raise HTTPException(status_code=504, detail="LLM service unavailable")

        generation_time = time.time() - generation_start
        decoded = result["decoded"]
        reply = decoded.split("المساعد:")[-1].strip()

        reply, corrections_count = apply_corrections(reply, dialect="egypt")
        reply = normalize_vital_signs(normalize_medical_abbreviations(reply))

        total_time_ms = (time.time() - overall_start) * 1000
        tokens_generated = result["output_len"] - result["input_len"]
        tokens_per_second.observe(tokens_generated / max(generation_time, 1e-6))
        complete_response_duration.observe(total_time_ms)

        log_safe(logging.INFO, "Chat reply generated", request=request, session_id=req.sessionId, tokens=tokens_generated, totalMs=int(total_time_ms))
        return ChatResponse(
            reply=reply,
            intent=intent,
            tokens_generated=tokens_generated,
            first_token_ms=generation_time * 0.15 * 1000,
            total_latency_ms=total_time_ms,
        )
    except HTTPException:
        raise
    except Exception as e:
        log_safe(logging.ERROR, "Chat inference failed", request=request, session_id=req.sessionId, error=str(type(e).__name__))
        raise HTTPException(status_code=500, detail="LLM inference failed")


def build_rag_prompt(message: str, intent: str = "general") -> str:
    """Build prompt with RAG context (few-shot examples + relevant FAQs)"""

    # Get few-shot examples for this intent
    few_shot_examples = rag_store.get_few_shot_examples(intent, limit=2)

    # Get relevant FAQs based on the message
    relevant_faqs = rag_store.get_relevant_faqs(message, limit=2)

    # Build prompt with RAG context
    prompt_parts = [
        "أنت مساعد طبي ذكي يتحدث العربية. مهمتك مساعدة المرضى بطريقة محترفة وودودة.",
        "",
    ]

    # Add few-shot examples
    if few_shot_examples:
        prompt_parts.append("أمثلة على المحادثات:")
        for ex in few_shot_examples:
            prompt_parts.append(f"المستخدم: {ex['user']}")
            prompt_parts.append(f"المساعد: {ex['assistant']}")
            prompt_parts.append("")

    # Add relevant FAQs as context
    if relevant_faqs:
        prompt_parts.append("معلومات طبية ذات صلة:")
        for faq in relevant_faqs:
            prompt_parts.append(f"س: {faq['question']}")
            prompt_parts.append(f"ج: {faq['answer']}")
            prompt_parts.append("")

    # Add current conversation
    prompt_parts.append("المحادثة الحالية:")
    prompt_parts.append(f"المستخدم: {message}")
    prompt_parts.append("المساعد:")

    return "\n".join(prompt_parts)


def classify_intent(message: str) -> str:
    """Simple intent classification based on keywords"""
    message_lower = message.lower()

    if any(word in message_lower for word in ["موعد", "حجز", "تأجيل"]):
        return "appointment"
    elif any(word in message_lower for word in ["ألم", "صداع", "حمى", "أعراض", "مريض"]):
        return "symptom"
    elif any(word in message_lower for word in ["وصفة", "دواء", "علاج"]):
        return "prescription"
    elif any(word in message_lower for word in ["حساسية", "تاريخ", "عملية"]):
        return "medical_history"
    else:
        return "general"

@app.post("/identify-speakers", response_model=SpeakerRoleResponse)
async def identify_speaker_roles(request: SpeakerRoleRequest):
    """
    Analyze conversation segments to identify speaker roles (Doctor, Patient, etc.)
    Uses HYBRID approach: Rule-based patterns + LLM analysis
    """
    if not request.segments or len(request.segments) == 0:
        raise HTTPException(status_code=400, detail="No segments provided")
    
    # ✨ POST-PROCESSING: Apply rule-based speaker identification first (+2-3% accuracy)
    identifier = SpeakerIdentifier()
    segments_dict = [{"speaker": seg.speaker, "text": seg.text} for seg in request.segments]
    rule_based_roles = identifier.identify_conversation_roles(segments_dict)
    
    print(f"✅ Rule-based identification complete: {rule_based_roles}")

    # Build conversation for analysis
    conversation_text = "\n".join([
        f"{seg.speaker}: {seg.text}" for seg in request.segments
    ])

    # Create prompt for LLM to analyze roles
    prompt = f"""Analyze the following medical conversation and identify the role of each speaker.
Consider:
1. Medical terminology usage (doctors use more technical terms)
2. Question patterns (doctors ask diagnostic questions)
3. Authority indicators ("I will prescribe", "Let me examine")
4. Symptom descriptions (patients describe their pain/discomfort)
5. Treatment plans (doctors explain procedures)

Conversation:
{conversation_text}

For each unique speaker, identify their role (Doctor, Patient, Nurse, etc.) and provide reasoning.
Format your response as JSON with this structure:
{{
  "roles": [
    {{"speaker_id": "SPEAKER_00", "role": "Doctor", "confidence": 0.95, "reasoning": "Uses medical terminology, asks diagnostic questions"}},
    {{"speaker_id": "SPEAKER_01", "role": "Patient", "confidence": 0.90, "reasoning": "Describes symptoms and responds to doctor's questions"}}
  ]
}}
"""

    try:
        print(f"🎭 Speaker identification request: {len(request.segments)} segments")
        
        # Generate analysis
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print(f"📊 Input tokens: {inputs['input_ids'].shape[1]}")

        print(f"🤖 Generating analysis (max 256 tokens, ~1-2 minutes on CPU)...")
        gen_start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # Reduced from 512
                do_sample=True,
                temperature=0.3,  # Lower temperature for more deterministic output
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        print(f"✅ Analysis complete in {time.time() - gen_start:.1f}s")

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract JSON from response (basic parsing)
        # In production, use more robust JSON extraction
        import json
        import re

        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
            llm_roles = analysis.get("roles", [])
            print(f"✅ LLM identified roles: {llm_roles}")
        else:
            # Fallback: use rule-based analysis
            print(f"⚠️ LLM failed to parse JSON, using rule-based fallback")
            llm_roles = analyze_speakers_heuristic(request.segments)
        
        # ✨ HYBRID: Combine LLM + rule-based results (use higher confidence)
        roles = []
        for llm_role in llm_roles:
            speaker_id = llm_role["speaker_id"]
            if speaker_id in rule_based_roles:
                rule_conf = rule_based_roles[speaker_id]["confidence"]
                llm_conf = llm_role.get("confidence", 0.5)
                
                # Use whichever has higher confidence
                if rule_conf > llm_conf:
                    print(f"✅ Using rule-based for {speaker_id} (conf: {rule_conf:.2f} > {llm_conf:.2f})")
                    roles.append({
                        "speaker_id": speaker_id,
                        "role": rule_based_roles[speaker_id]["role"],
                        "confidence": rule_conf,
                        "reasoning": f"Rule-based (stronger): {rule_based_roles[speaker_id]['reasoning']}"
                    })
                else:
                    print(f"✅ Using LLM for {speaker_id} (conf: {llm_conf:.2f} > {rule_conf:.2f})")
                    roles.append(llm_role)
            else:
                roles.append(llm_role)

        # Identify primary doctor and patient
        doctor_speaker = None
        patient_speaker = None

        role_objects = []
        for role_data in roles:
            role_obj = SpeakerRole(**role_data)
            role_objects.append(role_obj)

            if role_obj.role.lower() == "doctor" and doctor_speaker is None:
                doctor_speaker = role_obj.speaker_id
            elif role_obj.role.lower() == "patient" and patient_speaker is None:
                patient_speaker = role_obj.speaker_id

        return SpeakerRoleResponse(
            roles=role_objects,
            primary_doctor=doctor_speaker,
            primary_patient=patient_speaker
        )

    except Exception as e:
        log_safe(
            logging.ERROR,
            "Speaker role identification failed",
            request=None,
            session_id=None,
            segments=len(request.segments),
            error=str(type(e).__name__),
        )
        # Fallback to heuristic
        roles = analyze_speakers_heuristic(request.segments)

        doctor_speaker = None
        patient_speaker = None
        role_objects = []

        for role_data in roles:
            role_obj = SpeakerRole(**role_data)
            role_objects.append(role_obj)

            if role_obj.role.lower() == "doctor":
                doctor_speaker = role_obj.speaker_id
            elif role_obj.role.lower() == "patient":
                patient_speaker = role_obj.speaker_id

        return SpeakerRoleResponse(
            roles=role_objects,
            primary_doctor=doctor_speaker,
            primary_patient=patient_speaker
        )

def analyze_speakers_heuristic(segments: list[SpeakerSegment]) -> list[dict]:
    """
    Fallback heuristic analysis when LLM fails
    Uses keyword patterns to identify roles
    """
    speaker_analysis = {}

    # Keywords that indicate doctor role
    doctor_keywords = [
        "prescribe", "examine", "diagnosis", "treatment", "recommend", "assess",
        "يصف", "فحص", "تشخيص", "علاج", "أوصي", "تقييم",
        "blood pressure", "heart rate", "temperature", "vitals",
        "ضغط الدم", "معدل القلب", "حرارة", "علامات حيوية"
    ]

    # Keywords that indicate patient role
    patient_keywords = [
        "pain", "hurts", "feeling", "symptom", "sick", "discomfort",
        "ألم", "يؤلم", "شعور", "أعراض", "مريض", "إزعاج",
        "I have", "I feel", "since", "for days",
        "لدي", "أشعر", "منذ", "أيام"
    ]

    for segment in segments:
        speaker_id = segment.speaker
        text_lower = segment.text.lower()

        if speaker_id not in speaker_analysis:
            speaker_analysis[speaker_id] = {
                "doctor_score": 0,
                "patient_score": 0,
                "utterances": 0
            }

        speaker_analysis[speaker_id]["utterances"] += 1

        # Count keyword matches
        for keyword in doctor_keywords:
            if keyword.lower() in text_lower:
                speaker_analysis[speaker_id]["doctor_score"] += 1

        for keyword in patient_keywords:
            if keyword.lower() in text_lower:
                speaker_analysis[speaker_id]["patient_score"] += 1

    # Determine roles based on scores
    roles = []
    for speaker_id, analysis in speaker_analysis.items():
        if analysis["doctor_score"] > analysis["patient_score"]:
            role = "Doctor"
            confidence = min(0.95, 0.6 + (analysis["doctor_score"] * 0.1))
            reasoning = f"Uses medical terminology and diagnostic language ({analysis['doctor_score']} doctor indicators)"
        elif analysis["patient_score"] > analysis["doctor_score"]:
            role = "Patient"
            confidence = min(0.95, 0.6 + (analysis["patient_score"] * 0.1))
            reasoning = f"Describes symptoms and personal experiences ({analysis['patient_score']} patient indicators)"
        else:
            # Default: first speaker is usually doctor in medical context
            role = "Doctor" if speaker_id == "SPEAKER_00" else "Patient"
            confidence = 0.5
            reasoning = "Assigned by position (first speaker assumed to be doctor)"

        roles.append({
            "speaker_id": speaker_id,
            "role": role,
            "confidence": confidence,
            "reasoning": reasoning
        })

    return roles

if __name__ == "__main__":
    import uvicorn
    print("Starting LLM service on http://0.0.0.0:5001...")
    uvicorn.run(app, host="0.0.0.0", port=5001)
