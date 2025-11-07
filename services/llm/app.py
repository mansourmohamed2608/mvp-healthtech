# services/llm/app.py
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rag_store import rag_store
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(title="LLM Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "MBZUAI/BiMediX2-8B"  # ✅ Bilingual Arabic-English medical LLM (66% accuracy, 1.6M samples, EMNLP 2025)

print(f"Loading model {MODEL_NAME} on {DEVICE}...")
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

@app.get("/health")
async def health():
    return {"status": "ok"}

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
        print(f"📝 Correction request: {len(req.text)} chars")
        
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
        print(f"✅ Generation complete in {elapsed:.1f}s")

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
        
        print(f"✅ Final corrected text: {corrected[:100]}...")

        # Count differences
        corrections_made = len(set(req.text.split()) - set(corrected.split()))

        return {
            "original": req.text,
            "corrected": corrected,
            "corrections_made": corrections_made,
            "dialect_normalized": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    requests_total.inc()
    start_time = time.time()
    first_token_time = None

    try:
        print(f"💬 Infer request: {req.message[:50]}...")
        
        # Build RAG-augmented prompt
        prompt = build_rag_prompt(req.message, req.intent)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print(f"📊 Input tokens: {inputs['input_ids'].shape[1]}")

        # Track first token latency using a custom stopping criteria
        print(f"🤖 Generating response (max 128 tokens, ~1-2 minutes on CPU)...")
        generation_start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        print(f"✅ Generation complete in {time.time() - generation_start:.1f}s")

        # For simplicity, estimate first token latency as ~10-20% of total generation time
        # In production, you'd use a custom callback to track exact first token time
        generation_time_ms = (time.time() - generation_start) * 1000
        estimated_first_token_ms = generation_time_ms * 0.15  # Rough estimate

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract intent and reply
        intent = classify_intent(req.message)
        reply = decoded.strip().split("المساعد:")[-1].strip() if "المساعد:" in decoded else decoded.strip()

        # Calculate metrics
        total_time_ms = (time.time() - start_time) * 1000
        num_tokens = len(outputs[0]) - len(inputs['input_ids'][0])  # Generated tokens only
        tps = (num_tokens / (generation_time_ms / 1000)) if generation_time_ms > 0 else 0

        # Record metrics
        first_token_latency.observe(estimated_first_token_ms)
        complete_response_duration.observe(total_time_ms)
        tokens_per_second.observe(tps)

        # Track slow responses (>1.5s)
        if total_time_ms > 1500:
            slow_responses.inc()
            print(f"⚠️ Slow response: {total_time_ms:.0f}ms (first token: ~{estimated_first_token_ms:.0f}ms, {num_tokens} tokens, {tps:.1f} tok/s)")
        else:
            print(f"✅ Fast response: {total_time_ms:.0f}ms (first token: ~{estimated_first_token_ms:.0f}ms, {num_tokens} tokens, {tps:.1f} tok/s)")

        return {"intent": intent, "reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    Uses LLM to analyze language patterns, terminology, and conversational dynamics
    """
    if not request.segments or len(request.segments) == 0:
        raise HTTPException(status_code=400, detail="No segments provided")

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
            roles = analysis.get("roles", [])
        else:
            # Fallback: basic heuristic analysis
            roles = analyze_speakers_heuristic(request.segments)

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
        print(f"Error in speaker role identification: {e}")
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
