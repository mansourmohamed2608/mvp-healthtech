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

from prompt_builder import build_va_prompt, build_slot_summary
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

# Egyptian → Saudi substitutions applied to every LLM reply
_EG_TO_SA: list[tuple[str, str]] = [
    # Egyptian → Saudi
    ("عاوز ", "أبغى "), ("عاوزة ", "أبغى "),
    ("عايز ", "أبغى "), ("عايزة ", "أبغى "),
    ("دلوقتي", "الحين"), ("دلوقت", "الحين"),
    (" مش ", " مو "), ("مشكلة", "مشكلة"),  # keep مشكلة
    ("ازاي", "كيف"), ("إزاي", "كيف"),
    ("إيه", "إيش"), (" ايه", " إيش"),
    ("كده", "كذا"), ("حاجة", "شي"), ("حاجه", "شي"),
    ("مفيش", "ما فيه"), ("يا فندم", "أستاذ"),
    ("بتاع", "حق"),
    # Levantine → Saudi
    ("شو اللي", "وش اللي"), ("شو ال", "وش ال"),
    (" شو ", " وش "), (" شو؟", " وش؟"),
    (" بدي ", " أبغى "), (" بده ", " يبغى "), (" بدها ", " تبغى "),
    (" بدنا ", " نبغى "), (" بدكم ", " تبغون "),
    (" هيدا ", " هذا "), (" هيدي ", " هذي "),
    (" كيفك", " كيف حالك"), (" منيح", " زين"),
    ("مرحبا", "هلا"),
    # Greetings not in Saudi dialect
    ("أهلاً", "هلا"), ("أهلا", "هلا"),
    # Egyptian demonstrative pronoun
    (" ده ", " هذا "), (" ده.", " هذا."), (" ده،", " هذا،"), (" ده؟", " هذا؟"),
    # Egyptian affirmations
    ("أيوه", "إيه"), ("ايوة", "إيه"),
]


def _saudi_post_filter(text: str) -> str:
    """Replace Egyptian/Levantine dialect words with Saudi equivalents."""
    for eg, sa in _EG_TO_SA:
        text = text.replace(eg, sa)
    return text


_LATIN_TOKEN_RE = re.compile(r'[a-zA-Z]')


def _strip_english(text: str) -> str:
    """Remove any Latin/English words that slip into the Arabic VA reply."""
    tokens = text.split()
    clean = [t for t in tokens if not _LATIN_TOKEN_RE.search(t)]
    result = ' '.join(clean)
    return re.sub(r'\s{2,}', ' ', result).strip()


def _enforce_single_question(text: str) -> str:
    """If the reply has multiple question marks (؟), keep only up to the first."""
    parts = text.split('؟')
    if len(parts) > 2:  # More than one ؟
        return parts[0].rstrip() + '؟'
    return text


def _dedup_words(text: str) -> str:
    """Remove consecutive duplicate Arabic words like 'تفضل تفضل'."""
    return re.sub(r'(\b[\u0600-\u06FF]+\b)\s+\1', r'\1', text)


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


# Phrases that signal a YouTube/social-media sign-off hallucination from the model's
# training data. Strip anything from these tokens to end-of-string.
_HALLUCINATION_STOPS = [
    "شكرا لمشاهدة", "شكراً لمشاهدة", "شكرًا لمشاهدة",
    "شكرا على المشاهدة", "شكراً على المشاهدة",
    "لا تنسى الاشتراك", "لا تنسوا الاشتراك",
    "اشترك في القناة", "اشتركوا في القناة",
    "لايك وشير", "لايك وسيبسكرايب",
    "مع السلامة وإلى اللقاء", "ووداعاً",
]

def _strip_assistant_prefix(text: str) -> str:
    cleaned = text.strip()
    # Strip leading role prefix if the model echoed it at the start of its output
    for token in ["المساعد:", "مساعد:", "Assistant:", "assistant:"]:
        if cleaned.startswith(token):
            cleaned = cleaned[len(token):].strip()
            break
    # Truncate any hallucinated continuation (turn separator or role switch)
    for stop in ["\nuser:", "\n user:", "\nمستخدم:", "\nالمستخدم:",
                  " user:", " مستخدم:", "\nassistant:", "\nAssistant:"]:
        idx = cleaned.find(stop)
        if idx != -1:
            cleaned = cleaned[:idx]
    # Remove YouTube/social-media sign-off hallucinations anywhere in the text
    for phrase in _HALLUCINATION_STOPS:
        idx = cleaned.find(phrase)
        if idx != -1:
            cleaned = cleaned[:idx]
    return cleaned.strip()


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
    input_len = inputs["input_ids"].shape[1]  # track prompt length to extract only new tokens
    with torch.no_grad():
        output = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )
    return _TOKENIZER.decode(output[0][input_len:], skip_special_tokens=True)


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

        # Inject today's date so the LLM never hallucinates past years
        from datetime import date as _date
        _today = _date.today()
        _day_ar = ["الاثنين","الثلاثاء","الأربعاء","الخميس","الجمعة","السبت","الأحد"][_today.weekday()]
        _date_str = _today.strftime("%d/%m/%Y")

        # Build the system message: base prompt + dialect hint + slot state
        slot_block = build_slot_summary(extracted_slots)
        if dialect == "saudi":
            dialect_hint = (
                "استخدمي لهجة سعودية محكية دافئة فقط (خليجية سعودية بسيطة). ممنوع الفصحى. "
                "أمثلة: \"هلا والله، يسعدنا نخدمك!\"، \"وش اسمك الكريم؟\"، \"تفضل، رقم جوالك؟\"."
            )
        elif dialect == "egypt":
            dialect_hint = (
                "استخدمي لهجة مصرية عامية دافئة فقط. ممنوع الفصحى. "
                "أمثلة: \"أهلاً وسهلاً بيك!\"، \"ممكن نعرف اسمك الكريم؟\"."
            )
        else:
            dialect_hint = ""
        full_system = system_prompt.strip()
        # Always inject current date/day so the model never hallucinates past years
        full_system += f"\n\nتاريخ اليوم: {_day_ar} {_date_str} (السنة {_today.year}). جميع المواعيد المحجوزة يجب أن تكون في المستقبل."
        if dialect_hint:
            full_system += "\n\n" + dialect_hint
        full_system += "\n\nحالة الحقول:\n" + (slot_block or "لا توجد حقول بعد.")
        history_turns = req.history or []

        # Doctor schedule reference for next-step guidance
        _DR_SCHEDULE = {
            "جلدية":  "دكتور علي الغامدي – الثلاثاء ٢–٤ عصراً والخميس ١٠–١٢",
            "طب عام": "دكتورة سارة الزهراني – الأحد والثلاثاء ٩ صباحاً–١ ظهراً",
            "باطنية": "دكتور محمد المالكي – الاثنين والأربعاء ٣–٦ والسبت ١٠–١",
        }

        def _next_step(slots: dict) -> str:
            if _is_missing(slots, "specialty"):
                return "اسألي: «وش لي عندنا اليوم؟ كشف ولا متابعة؟»"
            spec = slots.get("specialty", "")
            if _is_missing(slots, "doctor_name"):
                dr = _DR_SCHEDULE.get(spec, "")
                dr_short = dr.split(" –")[0] if " –" in dr else dr
                return (
                    f"عرّفي الطبيب المتاح لـ{spec}: {dr}. "
                    f"ثم اسألي: «تبي نحجز مع {dr_short}؟» — لا تسألي عن تاريخ أو وقت قبل الموافقة."
                )
            if _is_missing(slots, "name"):
                return "اسألي فقط: «وش اسمك الكريم؟»"
            if _is_missing(slots, "phone"):
                return "اسألي فقط: «ما رقم جوالك؟»"
            if _is_missing(slots, "dob"):
                return "اسألي فقط: «ما تاريخ ميلادك؟»"
            if _is_missing(slots, "date"):
                dr = _DR_SCHEDULE.get(spec, "")
                avail = dr.split(" – ")[1] if " – " in dr else ""
                return f"اسألي عن التاريخ وذكّريه بالأيام المتاحة: {avail}."
            if _is_missing(slots, "time"):
                dr = _DR_SCHEDULE.get(spec, "")
                avail = dr.split(" – ")[1] if " – " in dr else ""
                return f"اسألي عن الوقت ضمن دوام الطبيب ({avail})."
            return "أكدي الحجز: الاسم والتاريخ والوقت واسم الطبيب."

        # Build strong instruction that prevents re-asking filled slots
        filled_keys = [k for k in ["name","phone","dob","specialty","doctor_name","date","time"]
                       if not _is_missing(extracted_slots, k)]
        filled_block = ""
        if filled_keys:
            from prompt_builder import CANONICAL_KEYS as _CK
            filled_labels = "، ".join(_CK.get(k, k) for k in filled_keys)
            filled_block = f" الخانات التالية مكتملة ولا تُسأل عنها أبداً: {filled_labels}."

        next_step = _next_step(extracted_slots)
        cont_instruction = (
            f"واصلي بصفتك ليان.{filled_block} "
            f"الخطوة التالية الآن: {next_step} "
            "استهدفي خانة واحدة فقط في كل رد. "
            "اجعلي الرد ٢-٣ جمل طبيعية وانتهي بسؤال واضح ينتهي بـ «؟»."
        )
        if history_turns:
            cont_instruction += " لا تبدأي الرد بتحية أو تقديم نفسك لأن المحادثة قائمة بالفعل."
        full_system += "\n\nالتعليمات: " + cont_instruction
        # Final hard-command appended LAST so the model sees it at max attention
        full_system += f"\n\n===الأمر الإلزامي النهائي===\nاسألي عن هذه الخانة فقط ولا شيء غيرها: {next_step}"

        # Build messages list using proper Qwen chat template
        messages: list[dict] = [{"role": "system", "content": full_system}]
        for turn in history_turns[-6:]:
            r = turn.role if hasattr(turn, "role") else turn.get("role", "")
            c = turn.content if hasattr(turn, "content") else turn.get("content", "")
            if r in ("user", "assistant"):
                messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": req.message})

        reply = _canned_reply(dialect, extracted_slots)
        if _MODEL_AVAILABLE and _MODEL and _TOKENIZER:
            try:
                raw = _generate_from_messages(messages, max_new_tokens=120, temperature=0.25)
                reply = _strip_assistant_prefix(raw)
                # Guard: if specialty is unknown but the LLM asked about a different slot, override
                if _is_missing(extracted_slots, "specialty"):
                    _off_track = ("جوال", "هاتف", "رقم جوال", "موبايل", "اسمك", "اسمكم", "ميلاد", "مواليد")
                    if any(w in reply for w in _off_track):
                        _n = extracted_slots.get("name", "")
                        _first = _n.split()[0] if _n else ""
                        _call = f" يا {_first}" if _first else ""
                        reply = f"هلا{_call}! وش اللي جاك بيه اليوم؟ إيش التخصص اللي تبيه؟"
            except Exception as e:
                import logging
                logging.error(f"VA generation failed, using fallback: {e}")

        # Post-process: fix Egyptian words, remove duplicate words, one question only
        reply = _saudi_post_filter(reply)
        reply = _strip_english(reply)
        reply = _dedup_words(reply)
        reply = _enforce_single_question(reply)

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
