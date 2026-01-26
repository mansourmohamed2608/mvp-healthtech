"""
VA LLM Service (Qwen) - Arabic Voice Agent booking
Scaffold: FastAPI + internal-secret auth; replace stubbed generate_reply with real model wiring.
"""
import os
import time
import re
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pathlib import Path
import asyncio
import httpx

from prompt_builder import build_va_prompt
from slot_extractor import extract_slots
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv, find_dotenv


def load_env():
    """Load env before pulling values so secrets override defaults."""
    local_path = find_dotenv(".env.local", usecwd=True)
    if local_path:
        load_dotenv(local_path, override=True)
    load_dotenv(find_dotenv(".env", usecwd=True), override=False)


load_env()

VA_MODEL = os.getenv("VA_MODEL", "Qwen/Qwen2.5-3B-Instruct")
INTERNAL_SECRET = os.getenv("INTERNAL_SECRET", "")
VA_DEVICE = os.getenv("VA_DEVICE", "cpu")
VA_DTYPE = os.getenv("VA_DTYPE", "float16")
# Enforce dialect output (reject MSA)
VA_DIALECT_ENFORCE = os.getenv("VA_DIALECT_ENFORCE", "true").lower() == "true"
VA_DIALECT_REWRITE = os.getenv("VA_DIALECT_REWRITE", "true").lower() == "true"
VA_DIALECT_REWRITE_TOKENS = int(os.getenv("VA_DIALECT_REWRITE_TOKENS", "120"))
# Hospital RAG (shared policies/FAQ) hosted by LLM service
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:5001").rstrip("/")
RAG_TIMEOUT_SECONDS = float(os.getenv("RAG_TIMEOUT_SECONDS", "2.5"))
# Accept both VA_MAX_CONCURRENT and VA_MAX_CONCURRENCY for compatibility
MAX_CONCURRENT = int(os.getenv("VA_MAX_CONCURRENT") or os.getenv("VA_MAX_CONCURRENCY") or "4")

app = FastAPI(title="VA LLM Service")

# CORS: configurable via env, default to localhost only
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ALLOWED_ORIGINS],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "x-internal-secret", "x-correlation-id", "x-tenant-id"],
)

# Optional OTEL
try:
    from otel_setup import init_otel
    init_otel("llm-va", app=app)
except Exception:
    import logging
    logging.getLogger("llm-va").debug("OTEL init skipped for VA")

requests_total = Counter("va_llm_requests_total", "Total VA LLM requests")
latency_seconds = Histogram(
    "va_llm_latency_seconds",
    "Latency of VA LLM responses",
    buckets=[0.1, 0.3, 0.5, 1, 2, 3, 5],
)
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

_SYSTEM_PROMPT = None
_MODEL = None
_TOKENIZER = None
_MODEL_AVAILABLE = False

EGYPTIAN_MARKERS = [
    "ازاي", "إزاي", "عايز", "عاوز", "عايزة", "مش", "ليه", "أيوه", "ايوه",
    "لسه", "دلوقتي", "كده", "حاجة", "مفيش", "فين", "يا فندم",
]
SAUDI_MARKERS = [
    "ايش", "إيش", "وش", "وشلون", "ليش", "ابغى", "أبغى", "يبغى", "تبي",
    "تبغى", "ودي", "الحين", "مره", "ترى", "حيل", "ياخي",
]
MSA_MARKERS = [
    "يرجى", "فضلا", "فضلاً", "سوف", "سنقوم", "سيتم", "نرجو", "الرجاء",
    "من فضلك", "من فضلكم", "هل ترغب", "هل تود", "يمكنك", "بالإمكان",
    "سوف يتم", "حيث", "بإمكان", "بإمكانك", "لديكم", "يرجى منك",
    "أود", "أرغب", "نود", "نرجو منك", "يُرجى", "فضلك", "فضلًا",
]
MSA_REGEXES = [
    r"\b(يرجى|الرجاء|نرجو|فضلا|فضلاً)\b",
    r"\b(سوف|سيتم|سنقوم)\b",
    r"\b(من فضلك|من فضلكم)\b",
    r"\b(هل ترغب|هل تود)\b",
    r"\b(بالإمكان|بإمكانك|يمكنك)\b",
    r"\b(أود|أرغب|نود)\b",
    r"\b(يرجى منك|نرجو منك)\b",
]


def _score_markers(text: str, markers: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for marker in markers if marker in lowered)


def _contains_msa_marker(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in MSA_MARKERS):
        return True
    for pattern in MSA_REGEXES:
        if re.search(pattern, lowered):
            return True
    return False


def _is_msa_like(text: str) -> bool:
    if not text:
        return False
    if _contains_msa_marker(text):
        return True
    msa_score = _score_markers(text, MSA_MARKERS)
    dialect_score = max(_score_markers(text, EGYPTIAN_MARKERS), _score_markers(text, SAUDI_MARKERS))
    if msa_score >= 2 and dialect_score == 0:
        return True
    if msa_score >= 3 and dialect_score <= 1:
        return True
    return False


def _is_dialect_ok(text: str, dialect: Optional[str]) -> bool:
    if not text:
        return False
    if _contains_msa_marker(text):
        return False
    if not dialect:
        return not _is_msa_like(text)
    if dialect == "egypt":
        return _score_markers(text, EGYPTIAN_MARKERS) > 0 or not _is_msa_like(text)
    if dialect == "saudi":
        return _score_markers(text, SAUDI_MARKERS) > 0 or not _is_msa_like(text)
    return not _is_msa_like(text)


def _strip_assistant_prefix(text: str) -> str:
    cleaned = text.strip()
    for token in ["المساعد:", "Assistant:", "assistant:"]:
        if token in cleaned:
            cleaned = cleaned.split(token)[-1].strip()
    return cleaned


def _rewrite_prompt(dialect: str, text: str) -> list[dict]:
    label = "مصرية عامية" if dialect == "egypt" else "سعودية محكية"
    return [
        {
            "role": "system",
            "content": (
                f"أعد صياغة النص التالي إلى لهجة {label} فقط. ممنوع الفصحى. "
                "اجعل الرد قصيراً وودوداً وينتهي بسؤال واضح."
            ),
        },
        {"role": "user", "content": text},
    ]


def _generate_from_prompt(prompt: str, max_new_tokens: int, temperature: float) -> str:
    if not _MODEL_AVAILABLE or _MODEL is None or _TOKENIZER is None:
        return ""
    inputs = _TOKENIZER(prompt, return_tensors="pt").to(_MODEL.device)
    with torch.no_grad():
        output = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )
    return _TOKENIZER.decode(output[0], skip_special_tokens=True)


def _generate_from_messages(messages: list[dict], max_new_tokens: int, temperature: float) -> str:
    if _TOKENIZER is None:
        return ""
    prompt = _TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return _generate_from_prompt(prompt, max_new_tokens, temperature)


def load_system_prompt(dialect: Optional[str] = None) -> str:
    normalized = normalize_dialect(dialect)
    if normalized == "egypt":
        prompt_path = Path(__file__).parent / "prompts" / "va_booking_egypt_system.txt"
        return prompt_path.read_text(encoding="utf-8")
    if normalized == "saudi":
        prompt_path = Path(__file__).parent / "prompts" / "va_booking_saudi_system.txt"
        return prompt_path.read_text(encoding="utf-8")
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT:
        return _SYSTEM_PROMPT
    prompt_path = Path(__file__).parent / "prompts" / "va_booking_arb_system.txt"
    _SYSTEM_PROMPT = prompt_path.read_text(encoding="utf-8")
    return _SYSTEM_PROMPT


def dtype_from_env():
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(VA_DTYPE.lower(), torch.float16)


def load_model():
    global _MODEL, _TOKENIZER, _MODEL_AVAILABLE
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(VA_MODEL)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            VA_MODEL,
            torch_dtype=dtype_from_env(),
            device_map="auto" if VA_DEVICE != "cpu" else None,
        )
        if VA_DEVICE == "cpu":
            _MODEL.to("cpu")
        _MODEL_AVAILABLE = True
    except Exception as e:
        import logging
        logging.error(f"VA model load failed: {e}")
        _MODEL_AVAILABLE = False


@app.middleware("http")
async def internal_auth(request: Request, call_next):
    if request.url.path.startswith("/health") or request.url.path.startswith("/metrics") or request.url.path.startswith("/ready"):
        return await call_next(request)
    # Use constant-time comparison to prevent timing attacks
    import hmac
    provided_secret = request.headers.get("x-internal-secret") or ""
    if not INTERNAL_SECRET or not hmac.compare_digest(provided_secret, INTERNAL_SECRET):
        raise HTTPException(status_code=403, detail="Unauthorized")
    return await call_next(request)


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatTurn]] = []
    sessionId: str
    mode: str
    slots: Optional[Dict[str, Any]] = {}
    dialect: Optional[str] = None
    tenantId: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    slots: Dict[str, Any]
    intent: str
    latency_ms: float


def normalize_dialect(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in ["egypt", "egyptian", "eg"]:
        return "egypt"
    if normalized in ["saudi", "ksa", "gulf", "gcc"]:
        return "saudi"
    if normalized == "auto":
        return None
    return normalized


def _is_missing(slots: Dict[str, Any], key: str) -> bool:
    if key not in slots:
        return True
    value = slots.get(key)
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _first_missing_label(slots: Dict[str, Any]) -> Optional[str]:
    order = [
        ("name", "الاسم الكامل"),
        ("phone", "رقم الجوال"),
        ("dob", "تاريخ الميلاد"),
        ("doctor_name", "اسم الطبيب"),
        ("specialty", "التخصص المطلوب"),
        ("date", "التاريخ المناسب"),
        ("time", "الوقت المناسب"),
    ]
    for key, label in order:
        if _is_missing(slots, key):
            return label
    return None


def _canned_reply(dialect: Optional[str], slots: Dict[str, Any]) -> str:
    missing = _first_missing_label(slots)
    if dialect == "saudi":
        if missing:
            return f"يعطيك العافية، ممكن {missing}؟"
        return "تم تمام، بأكد لك الحجز وأرسله لك بعد قليل."
    # default egyptian
    if missing:
        return f"تمام، ممكن {missing}؟"
    return "تمام كده، هأكد لك الحجز وأبعت لك التفاصيل."


async def fetch_rag_context(query: str, tenant_id: Optional[str]) -> Dict[str, Any]:
    if not query or not RAG_SERVICE_URL:
        return {}
    headers = {"x-internal-secret": INTERNAL_SECRET} if INTERNAL_SECRET else {}
    try:
        async with httpx.AsyncClient(timeout=RAG_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{RAG_SERVICE_URL}/rag/query",
                json={"query": query, "limit": 3, "tenantId": tenant_id},
                headers=headers,
            )
        if response.status_code != 200:
            return {}
        data = response.json()
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


@app.get("/health")
async def health():
    return {"ok": True, "service": "llm-va", "model": VA_MODEL}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.mode != "voice_agent_va":
        raise HTTPException(status_code=400, detail="Invalid mode for VA service")
    requests_total.inc()
    start = time.time()
    dialect = normalize_dialect(req.dialect)
    async with semaphore:
        # Extract slots heuristically before building prompt
        current_slots = req.slots or {}
        extracted_slots = extract_slots(req.message, current_slots)
        rag_context = await fetch_rag_context(req.message, req.tenantId)
        system_prompt = load_system_prompt(dialect)
        prompt = build_va_prompt(
            system_prompt=system_prompt,
            history=req.history or [],
            slots=extracted_slots,
            user_message=req.message,
            dialect=dialect,
            rag_context=rag_context,
        )

        reply = _canned_reply(dialect, extracted_slots)
        if _MODEL_AVAILABLE and _MODEL and _TOKENIZER:
            try:
                raw = _generate_from_prompt(prompt, max_new_tokens=160, temperature=0.25)
                reply = _strip_assistant_prefix(raw)
            except Exception as e:
                import logging
                logging.error(f"VA generation failed, using fallback: {e}")

        if VA_DIALECT_ENFORCE and dialect in {"egypt", "saudi"}:
            if not _is_dialect_ok(reply, dialect):
                rewritten = ""
                if VA_DIALECT_REWRITE:
                    try:
                        rewrite_msgs = _rewrite_prompt(dialect, reply)
                        rewritten_raw = _generate_from_messages(
                            rewrite_msgs,
                            max_new_tokens=VA_DIALECT_REWRITE_TOKENS,
                            temperature=0.2,
                        )
                        rewritten = _strip_assistant_prefix(rewritten_raw)
                    except Exception:
                        rewritten = ""
                if rewritten and _is_dialect_ok(rewritten, dialect):
                    reply = rewritten
                else:
                    reply = _canned_reply(dialect, extracted_slots)

        if VA_DIALECT_ENFORCE and _contains_msa_marker(reply):
            reply = _canned_reply(dialect, extracted_slots)

        updated_slots = extracted_slots
        duration = (time.time() - start) * 1000
        latency_seconds.observe(duration / 1000)

        return ChatResponse(
            reply=reply,
            slots=updated_slots,
            intent="book_appointment",
            latency_ms=duration,
        )

@app.on_event("startup")
async def _startup():
    load_model()


if __name__ == "__main__":
    import uvicorn

    load_model()
    uvicorn.run(app, host="0.0.0.0", port=5007)
