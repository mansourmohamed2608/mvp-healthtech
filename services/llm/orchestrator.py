# services/llm/orchestrator.py
"""
LLM Orchestrator Service - Week 2 Day 12
Handles intent extraction, entity recognition, and routing for medical conversations
Port: 5006
"""
import re
from dotenv import load_dotenv, find_dotenv

# Prefer .env.local for local dev, fallback to .env
load_dotenv(find_dotenv(".env.local", usecwd=True), override=True)
load_dotenv(find_dotenv(".env", usecwd=True), override=False)
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(title="LLM Orchestrator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

INTERNAL_SECRET = os.environ.get("INTERNAL_SECRET", "")
CLINICAL_LLM_URL = os.environ.get("CLINICAL_LLM_URL", os.environ.get("LLM_ENDPOINT", "http://localhost:5001"))
VA_LLM_URL = os.environ.get("VA_LLM_URL", "http://localhost:5007")

@app.middleware("http")
async def internal_auth(request, call_next):
  if request.url.path.startswith("/health") or request.url.path.startswith("/metrics"):
    return await call_next(request)
  if not INTERNAL_SECRET or request.headers.get("x-internal-secret") != INTERNAL_SECRET:
    raise HTTPException(status_code=403, detail="Unauthorized")
  return await call_next(request)

# Prometheus metrics
orchestration_requests = Counter('orchestrator_requests_total', 'Total orchestration requests')
intent_classification_duration = Histogram(
    'orchestrator_intent_classification_ms',
    'Time taken for intent classification',
    buckets=[10, 25, 50, 75, 100, 150, 200, 300]
)
entity_extraction_duration = Histogram(
    'orchestrator_entity_extraction_ms',
    'Time taken for entity extraction',
    buckets=[5, 10, 20, 30, 50, 75, 100]
)

class OrchestrateRequest(BaseModel):
    transcript: str
    sessionId: str
    mode: str | None = "clinical_soap"
    history: list[dict] | None = None
    context: dict = {}
    slots: dict = {}
    dialect: str | None = None
    tenantId: str | None = None

class OrchestrateResponse(BaseModel):
    intent: str
    entities: dict
    reply: str
    confidence: float
    routing: str
    slots: dict | None = None

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", f"{CLINICAL_LLM_URL}/infer")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "orchestrator"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(req: OrchestrateRequest):
    """
    Orchestrate LLM request with intent classification and entity extraction
    
    Intents:
    - appointment: حجز موعد، تأجيل، إلغاء
    - symptom: أعراض، ألم، صداع، حمى
    - prescription: وصفة طبية، دواء، علاج
    - medical_history: تاريخ مرضي، حساسية، عمليات سابقة
    - emergency: حالة طارئة، نوبة قلبية، نزيف
    - general: استفسارات عامة
    
    Entities:
    - dates: تواريخ ومواعيد
    - symptoms: أعراض محددة
    - medications: أدوية
    - body_parts: أجزاء الجسم
    - durations: مدة الأعراض
    """
    orchestration_requests.inc()
    import time
    start_time = time.time()
    
    try:
        # Step 1: Intent Classification (keyword-based + confidence)
        intent_start = time.time()
        intent, confidence = classify_intent(req.transcript)
        intent_duration_ms = (time.time() - intent_start) * 1000
        intent_classification_duration.observe(intent_duration_ms)
        
        # Step 2: Entity Extraction
        entity_start = time.time()
        entities = extract_entities(req.transcript, intent)
        slots = extract_slots(req.transcript, req.slots or {})
        dialect = normalize_dialect(req.dialect)
        entity_duration_ms = (time.time() - entity_start) * 1000
        entity_extraction_duration.observe(entity_duration_ms)
        
        # Step 3: Determine routing strategy
        routing = determine_routing(intent, entities, confidence)
        
        # Step 4: Call LLM for response generation
        if req.mode == "voice_agent_va":
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{VA_LLM_URL}/chat",
                    json={
                        "message": req.transcript,
                        "sessionId": req.sessionId,
                        "mode": req.mode,
                        "history": req.history or [],
                        "slots": slots,
                        "dialect": dialect,
                        "tenantId": req.tenantId,
                    },
                    headers={"x-internal-secret": INTERNAL_SECRET} if INTERNAL_SECRET else {},
                    timeout=30.0,
                )
            data = response.json()
            reply = data.get("reply", "")
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    LLM_ENDPOINT,
                    json={
                        "message": req.transcript,
                        "sessionId": req.sessionId,
                        "intent": intent
                    },
                    headers={"x-internal-secret": INTERNAL_SECRET} if INTERNAL_SECRET else {},
                    timeout=30.0,
                )
            data = response.json()
            reply = data.get("reply", "")
        
        print(f"🎯 Orchestration: intent={intent} ({confidence:.2f}), entities={len(entities)}, routing={routing}, latency={int((time.time()-start_time)*1000)}ms")
        
        return {
            "intent": intent,
            "entities": data.get("slots", entities),
            "reply": reply,
            "confidence": confidence,
            "routing": routing,
            "slots": data.get("slots", slots),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def classify_intent(message: str) -> tuple[str, float]:
    """
    Classify intent with confidence score
    Returns: (intent, confidence)
    """
    message_lower = message.lower()
    
    # Emergency keywords (highest priority)
    emergency_keywords = ["نوبة قلبية", "heart attack", "صعوبة تنفس", "نزيف شديد", "فقدان وعي", "جلطة", "سكتة"]
    if any(kw in message_lower for kw in emergency_keywords):
        return ("emergency", 0.95)
    
    # Appointment keywords
    appointment_keywords = ["موعد", "حجز", "تأجيل", "إلغاء", "appointment", "book", "schedule"]
    appointment_score = sum(1 for kw in appointment_keywords if kw in message_lower)
    
    # Symptom keywords
    symptom_keywords = ["ألم", "صداع", "حمى", "أعراض", "مريض", "pain", "fever", "headache", "symptom"]
    symptom_score = sum(1 for kw in symptom_keywords if kw in message_lower)
    
    # Prescription keywords
    prescription_keywords = ["وصفة", "دواء", "علاج", "prescription", "medication", "drug"]
    prescription_score = sum(1 for kw in prescription_keywords if kw in message_lower)
    
    # Medical history keywords
    history_keywords = ["حساسية", "تاريخ", "عملية", "allergy", "history", "surgery"]
    history_score = sum(1 for kw in history_keywords if kw in message_lower)
    
    # Calculate intent and confidence
    scores = {
        "appointment": appointment_score,
        "symptom": symptom_score,
        "prescription": prescription_score,
        "medical_history": history_score
    }
    
    max_intent = max(scores, key=scores.get)
    max_score = scores[max_intent]
    
    if max_score == 0:
        return ("general", 0.5)
    
    # Confidence based on keyword matches
    confidence = min(0.6 + (max_score * 0.15), 0.95)
    
    return (max_intent, confidence)


def extract_entities(message: str, intent: str) -> dict:
    """
    Extract medical entities from message
    Returns dict with entity types and values
    """
    entities = {}
    
    # Extract dates (Arabic and numbers)
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # 01/01/2025
        r'غدا|اليوم|أمس',  # tomorrow, today, yesterday
        r'الأحد|الإثنين|الثلاثاء|الأربعاء|الخميس|الجمعة|السبت',  # days
    ]
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, message, re.IGNORECASE)
        dates.extend(matches)
    if dates:
        entities['dates'] = dates
    
    # Extract symptoms (intent-specific)
    if intent == "symptom":
        symptom_keywords = ["ألم", "صداع", "حمى", "غثيان", "قيء", "إسهال", "سعال", "رشح"]
        symptoms = [kw for kw in symptom_keywords if kw in message.lower()]
        if symptoms:
            entities['symptoms'] = symptoms
    
    # Extract body parts
    body_parts = ["رأس", "صدر", "بطن", "ظهر", "يد", "رجل", "عين", "أذن", "أنف", "حلق"]
    found_parts = [part for part in body_parts if part in message]
    if found_parts:
        entities['body_parts'] = found_parts
    
    # Extract durations
    duration_pattern = r'(\d+)\s*(يوم|ساعة|أسبوع|شهر|day|hour|week|month)'
    durations = re.findall(duration_pattern, message, re.IGNORECASE)
    if durations:
        entities['durations'] = [f"{num} {unit}" for num, unit in durations]
    
    # Extract medications (intent-specific)
    if intent == "prescription":
        # Common Arabic medication names
        med_keywords = ["باراسيتامول", "أسبرين", "أيبوبروفين", "أموكسيسيلين", "أنتيبيوتك"]
        medications = [med for med in med_keywords if med in message]
        if medications:
            entities['medications'] = medications
    
    return entities


def normalize_dialect(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in ["egypt", "egyptian", "eg"]:
        return "egypt"
    if normalized in ["saudi", "ksa", "gulf", "gcc"]:
        return "saudi"
    return normalized


def extract_slots(message: str, current: dict) -> dict:
    slots = {**(current or {})}
    text = message or ""

    if not slots.get("phone"):
        phone_match = re.search(r"(?:\\+?966|0)?\\d{8,12}", text.replace(" ", ""))
        if phone_match:
            slots["phone"] = phone_match.group(0)

    if not slots.get("name"):
        name_match = re.search(r"(?:اسمي|انا|أنا|اسمى)\\s+([\\w\\u0600-\\u06FF]+(?:\\s+[\\w\\u0600-\\u06FF]+){0,2})", text)
        if name_match:
            slots["name"] = name_match.group(1).strip()

    if not slots.get("doctor_name"):
        doc_match = re.search(r"(?:دكتور|دكتورة)\\s+([\\w\\u0600-\\u06FF]+)", text)
        if doc_match:
            slots["doctor_name"] = doc_match.group(1).strip()

    if not slots.get("specialty"):
        specialties = [
            "جلدية",
            "باطنة",
            "أطفال",
            "اطفال",
            "أسنان",
            "اسنان",
            "نساء",
            "ولادة",
            "عظام",
            "عيون",
            "انف",
            "اذن",
            "انف واذن",
        ]
        for spec in specialties:
            if spec in text:
                slots["specialty"] = spec
                break

    if not slots.get("date"):
        date_match = re.search(r"\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}", text)
        if date_match:
            slots["date"] = date_match.group(0)

    if not slots.get("time"):
        time_match = re.search(r"(\\d{1,2})[:٫](\\d{2})", text)
        if time_match:
            slots["time"] = f"{time_match.group(1)}:{time_match.group(2)}"
        else:
            hour_match = re.search(r"(?:الساعة|ساعه)\\s*(\\d{1,2})", text)
            if hour_match:
                slots["time"] = f"{hour_match.group(1)}:00"

    if slots.get("no_marketing") is None:
        if "لا" in text and ("رسائل" in text or "تسويق" in text):
            slots["no_marketing"] = True
        elif "اريد" in text and ("رسائل" in text or "تسويق" in text):
            slots["no_marketing"] = False

    return slots


def determine_routing(intent: str, entities: dict, confidence: float) -> str:
    """
    Determine routing strategy based on intent and entities
    
    Routing strategies:
    - direct: Direct response from LLM
    - rag: Retrieve relevant medical knowledge
    - escalate: Human handoff required
    - appointment_system: Route to scheduling system
    - pharmacy: Route to prescription system
    """
    if intent == "emergency":
        return "escalate"
    
    if intent == "appointment" and confidence > 0.7:
        return "appointment_system"
    
    if intent == "prescription" and confidence > 0.7:
        return "pharmacy"
    
    if intent in ["symptom", "medical_history"] and confidence > 0.6:
        return "rag"
    
    return "direct"


if __name__ == "__main__":
    import uvicorn
    print("Starting LLM Orchestrator service on http://0.0.0.0:5006...")
    uvicorn.run(app, host="0.0.0.0", port=5006)
