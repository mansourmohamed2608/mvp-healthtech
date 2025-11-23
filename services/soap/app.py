# services/soap/app.py
"""
SOAP Note Generator Service
Generates structured clinical notes from transcripts using LLM
Persists notes in Postgres for clinician review and FHIR writeback.
"""
import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("soap")
# Optional OTEL
try:
    from otel_setup import init_otel
    init_otel("soap")
except Exception:
    logger.debug("OTEL init skipped for SOAP")

def safe_print(*args, **_kwargs):
    logger.debug("suppressed print", extra={"fields": len(args)})

print = safe_print

app = FastAPI(title="SOAP Generator Service", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INTERNAL_SECRET = os.getenv("INTERNAL_SECRET", "")
if not INTERNAL_SECRET:
    raise RuntimeError("INTERNAL_SECRET must be set for SOAP service")

@app.middleware("http")
async def internal_auth(request: Request, call_next):
    if request.url.path.startswith("/health"):
        return await call_next(request)
    if not INTERNAL_SECRET or request.headers.get("x-internal-secret") != INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await call_next(request)

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:5001")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/healthtech")
_pool: Optional[asyncpg.pool.Pool] = None

# ---------------------------
# Models
# ---------------------------
class SOAPRequest(BaseModel):
    transcript: str
    sessionId: str
    patientContext: Optional[dict] = None
    patientId: str
    clinicianId: str
    encounterId: Optional[str] = None

class SOAPResponse(BaseModel):
    id: str
    session_id: str
    patient_id: Optional[str]
    clinician_id: Optional[str]
    status: str
    subjective: str
    objective: str
    assessment: str
    plan: str
    icd_codes: Optional[List[str]] = None
    cpt_codes: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    raw_transcript: Optional[str] = None

class ApproveRejectRequest(BaseModel):
    action: str  # approve|reject

# ---------------------------
# Startup / DB
# ---------------------------
@app.on_event("startup")
async def startup():
    global _pool
    try:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        async with _pool.acquire() as conn:
            await conn.execute("SELECT 1")
        logger.info("SOAP DB pool ready")
    except Exception as e:
        logger.warning("SOAP DB init failed", extra={"error": str(e)})
        _pool = None

@app.on_event("shutdown")
async def shutdown():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None

# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "soap-generator",
        "llm_url": LLM_SERVICE_URL,
        "db_connected": _pool is not None,
    }

@app.post("/generate", response_model=SOAPResponse)
async def generate_soap(req: SOAPRequest):
    if _pool is None:
        raise HTTPException(status_code=503, detail="DB not available")

    prompt = build_soap_prompt(req.transcript, req.patientContext)
    session_id = req.sessionId or f"soap-{int(datetime.utcnow().timestamp())}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{LLM_SERVICE_URL}/infer",
                json={"message": prompt, "sessionId": session_id},
            )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"LLM service error: {response.text}")
        llm_output = response.json()
        note_struct = parse_soap_sections(llm_output.get("reply", ""))
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM service timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SOAP generation failed: {e}")

    record = await save_note({
        "session_id": session_id,
        "patient_id": req.patientId,
        "clinician_id": req.clinicianId,
        "encounter_id": req.encounterId,
        "status": "pending",
        "raw_transcript": req.transcript,
        "soap_json": note_struct,
        "subjective": note_struct["subjective"],
        "objective": note_struct["objective"],
        "assessment": note_struct["assessment"],
        "plan": note_struct["plan"],
        "icd_codes": note_struct.get("icd_codes") or [],
        "cpt_codes": note_struct.get("cpt_codes") or [],
    })

    return record

@app.get("/notes")
async def list_notes(status: Optional[str] = None, clinicianId: Optional[str] = None):
    if _pool is None:
        raise HTTPException(status_code=503, detail="DB not available")
    rows = await fetch_notes(status=status, clinician_id=clinicianId)
    return {"notes": rows}

@app.get("/notes/{note_id}", response_model=SOAPResponse)
async def get_note(note_id: str):
    note = await fetch_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note

@app.patch("/notes/{note_id}/approve", response_model=SOAPResponse)
async def approve_note(note_id: str):
    updated = await update_status(note_id, "approved")
    if not updated:
        raise HTTPException(status_code=404, detail="Note not found")
    return updated

@app.patch("/notes/{note_id}/reject", response_model=SOAPResponse)
async def reject_note(note_id: str):
    updated = await update_status(note_id, "rejected")
    if not updated:
        raise HTTPException(status_code=404, detail="Note not found")
    return updated

# ---------------------------
# Helpers
# ---------------------------
def build_soap_prompt(transcript: str, context: Optional[dict] = None) -> str:
    prompt = f"""أنت طبيب متخصص في توثيق السجلات الطبية. قم بتحويل النص التالي إلى ملاحظة SOAP منظمة:

النص الطبي:
{transcript}

قم بإنشاء ملاحظة SOAP مفصلة باللغة العربية بالتنسيق التالي:

[الذاتي - Subjective]
(اكتب شكوى المريض الرئيسية والأعراض التي يصفها)

[الموضوعي - Objective]
(اكتب نتائج الفحص السريري والعلامات الحيوية)

[التقييم - Assessment]
(اكتب التشخيص الطبي والتقييم السريري)

[الخطة - Plan]
(اكتب خطة العلاج والمتابعة)
"""
    if context:
        prompt += f"\n\nمعلومات إضافية عن المريض: {context}"
    return prompt

def parse_soap_sections(llm_reply: str) -> Dict[str, Any]:
    sections = {"subjective": "", "objective": "", "assessment": "", "plan": ""}
    current = None
    for line in llm_reply.split("\n"):
        line = line.strip()
        low = line.lower()
        if "subjective" in low or "الذاتي" in line:
            current = "subjective"
            continue
        if "objective" in low or "الموضوعي" in line:
            current = "objective"
            continue
        if "assessment" in low or "التقييم" in line:
            current = "assessment"
            continue
        if "plan" in low or "الخطة" in line:
            current = "plan"
            continue
        if current and line and not line.startswith("["):
            sections[current] += line + "\n"
    if not any(sections.values()):
        parts = llm_reply.split("\n\n")
        sections["subjective"] = parts[0] if len(parts) > 0 else "غير متوفر"
        sections["objective"] = parts[1] if len(parts) > 1 else "غير متوفر"
        sections["assessment"] = parts[2] if len(parts) > 2 else "غير متوفر"
        sections["plan"] = parts[3] if len(parts) > 3 else "غير متوفر"
    return sections

async def save_note(note: Dict[str, Any]) -> SOAPResponse:
    if not note.get("patient_id") or not note.get("clinician_id") or not note.get("session_id"):
        raise HTTPException(status_code=400, detail="patient_id, clinician_id, session_id required")
    icd = note.get("icd_codes") or []
    cpt = note.get("cpt_codes") or []
    soap_json = note.get("soap_json") or {}
    async with _pool.acquire() as conn:  # type: ignore
        row = await conn.fetchrow(
            """
            INSERT INTO soap_notes (session_id, patient_id, clinician_id, status, raw_transcript, soap_json,
                                    subjective, objective, assessment, plan, icd_codes, cpt_codes)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            RETURNING id, session_id, patient_id, clinician_id, status, subjective, objective, assessment, plan, icd_codes, cpt_codes, created_at, updated_at
            """,
            note.get("session_id"),
            note.get("patient_id"),
            note.get("clinician_id"),
            note.get("status", "pending"),
            note.get("raw_transcript"),
            json.dumps(soap_json),
            note.get("subjective"),
            note.get("objective"),
            note.get("assessment"),
            note.get("plan"),
            icd,
            cpt,
        )
    return record_to_model(row)

async def fetch_notes(status: Optional[str] = None, clinician_id: Optional[str] = None) -> List[Dict[str, Any]]:
    query = "SELECT * FROM soap_notes"
    conds = []
    params: List[Any] = []
    if status:
        params.append(status)
        conds.append(f"status = ${len(params)}")
    if clinician_id:
        params.append(clinician_id)
        conds.append(f"clinician_id = ${len(params)}")
    if conds:
        query += " WHERE " + " AND ".join(conds)
    query += " ORDER BY created_at DESC"
    async with _pool.acquire() as conn:  # type: ignore
        rows = await conn.fetch(query, *params)
    return [record_to_model(r).dict() for r in rows]

async def fetch_note(note_id: str) -> Optional[SOAPResponse]:
    async with _pool.acquire() as conn:  # type: ignore
        row = await conn.fetchrow("SELECT * FROM soap_notes WHERE id = $1", note_id)
    if not row:
        return None
    return record_to_model(row)

async def update_status(note_id: str, status: str) -> Optional[SOAPResponse]:
    async with _pool.acquire() as conn:  # type: ignore
        row = await conn.fetchrow(
            "UPDATE soap_notes SET status=$2, updated_at=now() WHERE id=$1 RETURNING *",
            note_id,
            status,
        )
    if not row:
        return None
    return record_to_model(row)


def record_to_model(row: asyncpg.Record) -> SOAPResponse:
    return SOAPResponse(
        id=str(row["id"]),
        session_id=row.get("session_id"),
        patient_id=row.get("patient_id"),
        clinician_id=row.get("clinician_id"),
        status=row.get("status"),
        subjective=row.get("subjective") or "",
        objective=row.get("objective") or "",
        assessment=row.get("assessment") or "",
        plan=row.get("plan") or "",
        icd_codes=row.get("icd_codes"),
        cpt_codes=row.get("cpt_codes"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        raw_transcript=row.get("raw_transcript"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003, log_level="info")
