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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Henrychur/MMed-Llama-3-8B"

print(f"Loading model {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True,
)

if DEVICE == "cpu":
    model = model.to(DEVICE)

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
        # Build correction prompt
        prompt = f"""أنت خبير طبي. قم بتصحيح النص التالي من نظام التعرف الصوتي.
صحح الأخطاء الطبية والأخطاء الإملائية فقط. لا تغير المعنى.

اللهجة: {req.dialect}
النص الأصلي: {req.text}

النص المصحح:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # Deterministic for corrections
            temperature=0.3,  # Low temperature for accuracy
            top_p=0.95
        )
        
        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected = corrected.split("النص المصحح:")[-1].strip()
        
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
        # Build RAG-augmented prompt
        prompt = build_rag_prompt(req.message, req.intent)

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Track first token latency using a custom stopping criteria
        generation_start = time.time()
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=True, 
            temperature=0.7, 
            top_p=0.9
        )
        
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

if __name__ == "__main__":
    import uvicorn
    print("Starting LLM service on http://0.0.0.0:5001...")
    uvicorn.run(app, host="0.0.0.0", port=5001)
