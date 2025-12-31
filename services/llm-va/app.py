"""
VA LLM Service (Qwen) - Arabic Voice Agent booking
Scaffold: FastAPI + internal-secret auth; replace stubbed generate_reply with real model wiring.
"""
import os
import time
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
# Hospital RAG (shared policies/FAQ) hosted by LLM service
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:5001").rstrip("/")
RAG_TIMEOUT_SECONDS = float(os.getenv("RAG_TIMEOUT_SECONDS", "2.5"))
# Accept both VA_MAX_CONCURRENT and VA_MAX_CONCURRENCY for compatibility
MAX_CONCURRENT = int(os.getenv("VA_MAX_CONCURRENT") or os.getenv("VA_MAX_CONCURRENCY") or "4")

app = FastAPI(title="VA LLM Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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


def load_system_prompt() -> str:
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
    if not INTERNAL_SECRET or request.headers.get("x-internal-secret") != INTERNAL_SECRET:
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


async def fetch_rag_context(query: str) -> Dict[str, Any]:
    if not query or not RAG_SERVICE_URL:
        return {}
    headers = {"x-internal-secret": INTERNAL_SECRET} if INTERNAL_SECRET else {}
    try:
        async with httpx.AsyncClient(timeout=RAG_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{RAG_SERVICE_URL}/rag/query",
                json={"query": query, "limit": 3},
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
        rag_context = await fetch_rag_context(req.message)
        system_prompt = load_system_prompt()
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
                inputs = _TOKENIZER(prompt, return_tensors="pt").to(_MODEL.device)
                with torch.no_grad():
                    output = _MODEL.generate(
                        **inputs,
                        max_new_tokens=160,
                        temperature=0.25,
                        top_p=0.9,
                        do_sample=True,
                    )
                reply = _TOKENIZER.decode(output[0], skip_special_tokens=True)
            except Exception as e:
                import logging
                logging.error(f"VA generation failed, using fallback: {e}")

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
