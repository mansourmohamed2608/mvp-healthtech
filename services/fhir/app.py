# services/fhir/app.py
"""
FHIR R4 Writeback Service
Week 5 Day 29 (Oct 23, 2025)
Writes SOAP notes to EHR using FHIR R4 API
"""
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import httpx
import os
import time
from datetime import datetime
import json
import logging
from html import escape
import re
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fhir")


def log_safe(level: int, msg: str, request: Request | None = None, session_id: str | None = None, **kwargs):
    extra = {
        "correlationId": request.headers.get("x-correlation-id") if request else None,
        "sessionId": session_id,
    }
    for k, v in kwargs.items():
        if v is not None:
            extra[k] = v
    logger.log(level, msg, extra=extra)

app = FastAPI(title="FHIR Writeback Service", version="1.0.0")

# CORS: configurable via env, default to localhost only
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ALLOWED_ORIGINS],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "x-internal-secret", "x-correlation-id", "Idempotency-Key"],
)

# Optional OTEL
try:
    from otel_setup import init_otel
    init_otel("fhir", app=app)
except Exception:
    logger.debug("OTEL init skipped for FHIR")

fhir_requests_total = Counter(
    "fhir_requests_total",
    "Total FHIR service requests",
    ["endpoint", "status"],
)
fhir_latency_seconds = Histogram(
    "fhir_latency_seconds",
    "FHIR service request latency",
    ["endpoint", "status"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10],
)
INTERNAL_SECRET = os.getenv("INTERNAL_SECRET")
if not INTERNAL_SECRET:
    raise RuntimeError("INTERNAL_SECRET must be set for FHIR service")

FHIR_BASE_URL = os.getenv("FHIR_BASE_URL")
if not FHIR_BASE_URL:
    raise RuntimeError("FHIR_BASE_URL must be set for FHIR service")

@app.middleware("http")
async def internal_auth(request: Request, call_next):
    if (
        request.url.path.startswith("/health")
        or request.url.path.startswith("/ready")
        or request.url.path.startswith("/metrics")
    ):
        return await call_next(request)
    # Use constant-time comparison to prevent timing attacks
    import hmac
    provided_secret = request.headers.get("x-internal-secret") or ""
    if not INTERNAL_SECRET or not hmac.compare_digest(provided_secret, INTERNAL_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await call_next(request)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    if request.url.path.startswith("/metrics"):
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    start = time.time()
    response = await call_next(request)
    status = "ok" if response.status_code < 400 else "error"
    fhir_requests_total.labels(endpoint=request.url.path, status=status).inc()
    fhir_latency_seconds.labels(endpoint=request.url.path, status=status).observe(time.time() - start)
    return response

FHIR_CLIENT_ID = os.getenv("FHIR_CLIENT_ID", "")
FHIR_CLIENT_SECRET = os.getenv("FHIR_CLIENT_SECRET", "")
FHIR_BEARER_TOKEN = os.getenv("FHIR_BEARER_TOKEN", "")
FHIR_BASIC_AUTH_USER = os.getenv("FHIR_BASIC_AUTH_USER", "")
FHIR_BASIC_AUTH_PASSWORD = os.getenv("FHIR_BASIC_AUTH_PASSWORD", "")

class SOAPNote(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str
    icd_codes: Optional[list[str]] = Field(default=None, alias="icdCodes")
    cpt_codes: Optional[list[str]] = Field(default=None, alias="cptCodes")
    soap_json: Optional[Dict[str, Any]] = Field(default=None, alias="soapJson")

    class Config:
        allow_population_by_field_name = True

class FHIRWriteRequest(BaseModel):
    soapNote: SOAPNote
    patientId: str
    practitionerId: str
    encounterId: Optional[str] = None
    sessionId: str

class FHIRWriteResponse(BaseModel):
    success: bool
    documentReferenceId: Optional[str] = None
    encounterId: Optional[str] = None
    compositionId: Optional[str] = None
    observationIds: Optional[list[str]] = None
    error: Optional[str] = None

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"ok": True, "service": "fhir-writeback"}


@app.get("/ready")
async def ready():
    """Readiness check: validates FHIR connectivity."""
    connected = await check_fhir_connection()
    return {"ready": connected, "fhir_base_url": FHIR_BASE_URL, "connected": connected}

async def check_fhir_connection() -> bool:
    """Check if FHIR server is reachable"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{FHIR_BASE_URL}/metadata")
            return response.status_code == 200
    except Exception:
        return False

@app.post("/write", response_model=FHIRWriteResponse)
async def write_soap_to_fhir(
    req: FHIRWriteRequest,
    authorization: Optional[str] = Header(None),
    idempotency_key: Optional[str] = Header(None),
):
    """
    Write SOAP note to FHIR server as DocumentReference
    
    Maps SOAP sections to FHIR R4 resources:
    - DocumentReference: Contains the complete SOAP note
    - Encounter: Links to the clinical encounter
    - Composition: Structured document with sections
    """
    log_safe(
        logging.INFO,
        "FHIR write request",
        request=None,
        session_id=req.sessionId,
        patientId=req.patientId,
        practitionerId=req.practitionerId,
    )
    try:
        # Get access token (if using OAuth2)
        access_token = None
        basic_auth = None
        if authorization:
            access_token = authorization.replace("Bearer ", "")
        elif FHIR_BEARER_TOKEN:
            access_token = FHIR_BEARER_TOKEN
        elif FHIR_BASIC_AUTH_USER and FHIR_BASIC_AUTH_PASSWORD:
            import base64
            basic_token = base64.b64encode(f"{FHIR_BASIC_AUTH_USER}:{FHIR_BASIC_AUTH_PASSWORD}".encode()).decode()
            basic_auth = f"Basic {basic_token}"
        elif FHIR_CLIENT_ID and FHIR_CLIENT_SECRET:
            access_token = await get_fhir_access_token()
        
        # Build FHIR resources
        encounter_resource = build_encounter_resource(
            req.patientId,
            req.practitionerId,
            req.encounterId
        )
        
        # Write to FHIR server
        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        elif basic_auth:
            headers["Authorization"] = basic_auth
        headers["Content-Type"] = "application/fhir+json"
        headers["Idempotency-Key"] = idempotency_key or generate_idempotency_key(req)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Create/Update Encounter
            encounter_response = await client.post(
                f"{FHIR_BASE_URL}/Encounter",
                json=encounter_resource,
                headers=headers
            )
            
            if encounter_response.status_code not in [200, 201]:
                log_safe(
                    logging.ERROR,
                    "Failed to create Encounter",
                    request=None,
                    session_id=req.sessionId,
                    status=encounter_response.status_code,
                )
                raise HTTPException(status_code=502, detail="Failed to create Encounter")
            
            encounter_id = encounter_response.json().get("id") or req.encounterId

            observation_ids: list[str] = []
            observation_resources = build_observation_resources(
                req.soapNote,
                req.patientId,
                req.practitionerId,
                encounter_id,
            )
            for obs in observation_resources:
                obs_response = await client.post(
                    f"{FHIR_BASE_URL}/Observation",
                    json=obs,
                    headers=headers,
                )
                if obs_response.status_code not in [200, 201]:
                    log_safe(
                        logging.ERROR,
                        "Failed to create Observation",
                        request=None,
                        session_id=req.sessionId,
                        status=obs_response.status_code,
                    )
                    raise HTTPException(status_code=502, detail="Failed to create Observation")
                obs_id = obs_response.json().get("id")
                if obs_id:
                    observation_ids.append(obs_id)

            document_reference = build_document_reference(
                req.soapNote,
                req.patientId,
                req.practitionerId,
                encounter_id,
            )

            # Create DocumentReference
            doc_response = await client.post(
                f"{FHIR_BASE_URL}/DocumentReference",
                json=document_reference,
                headers=headers
            )
            
            if doc_response.status_code not in [200, 201]:
                log_safe(
                    logging.ERROR,
                    "Failed to create DocumentReference",
                    request=None,
                    session_id=req.sessionId,
                    status=doc_response.status_code,
                )
                raise HTTPException(status_code=502, detail="Failed to create DocumentReference")
            
            doc_id = doc_response.json().get("id")

            composition_id = None
            composition_resource = build_composition_resource(
                req.soapNote,
                req.patientId,
                req.practitionerId,
                encounter_id,
                doc_id,
                observation_ids,
            )
            comp_response = await client.post(
                f"{FHIR_BASE_URL}/Composition",
                json=composition_resource,
                headers=headers,
            )
            if comp_response.status_code not in [200, 201]:
                log_safe(
                    logging.ERROR,
                    "Failed to create Composition",
                    request=None,
                    session_id=req.sessionId,
                    status=comp_response.status_code,
                )
                raise HTTPException(status_code=502, detail="Failed to create Composition")
            composition_id = comp_response.json().get("id")
        
        logger.info(
            "FHIR write ok",
            extra={
                "docId": doc_id,
                "encounterId": encounter_id,
                "compositionId": composition_id,
                "sessionId": req.sessionId,
            },
        )
        
        return FHIRWriteResponse(
            success=True,
            documentReferenceId=doc_id,
            encounterId=encounter_id,
            compositionId=composition_id,
            observationIds=observation_ids or None,
        )
        
    except httpx.TimeoutException:
        log_safe(logging.ERROR, "FHIR server timeout", request=None, session_id=req.sessionId)
        raise HTTPException(status_code=504, detail="FHIR server timeout")
    except Exception as e:
        logger.error("FHIR write error", extra={"sessionId": req.sessionId, "error": str(e)})
        return FHIRWriteResponse(
            success=False,
            error="FHIR write failed"
        )

async def get_fhir_access_token() -> str:
    """
    Get OAuth2 access token for FHIR server
    Uses client credentials grant
    """
    token_url = os.getenv("FHIR_TOKEN_URL", f"{FHIR_BASE_URL}/oauth/token")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": FHIR_CLIENT_ID,
                "client_secret": FHIR_CLIENT_SECRET,
                "scope": "system/*.write"
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=401,
                detail="Failed to obtain FHIR access token"
            )
        
        return response.json().get("access_token")

def build_encounter_resource(
    patient_id: str,
    practitioner_id: str,
    encounter_id: Optional[str] = None
) -> Dict[str, Any]:
    """Build FHIR R4 Encounter resource"""
    resource = {
        "resourceType": "Encounter",
        "status": "finished",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "AMB",
            "display": "ambulatory"
        },
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "participant": [
            {
                "individual": {
                    "reference": f"Practitioner/{practitioner_id}"
                }
            }
        ],
        "period": {
            "start": datetime.utcnow().isoformat() + "Z"
        }
    }
    
    if encounter_id:
        resource["id"] = encounter_id
    
    return resource

def build_document_reference(
    soap_note: SOAPNote,
    patient_id: str,
    practitioner_id: str,
    encounter_id: Optional[str]
) -> Dict[str, Any]:
    """Build FHIR R4 DocumentReference with SOAP note"""
    
    # Format SOAP note as text
    soap_text = f"""SOAP Clinical Note

SUBJECTIVE:
{soap_note.subjective}

OBJECTIVE:
{soap_note.objective}

ASSESSMENT:
{soap_note.assessment}

PLAN:
{soap_note.plan}
"""
    
    if soap_note.icd_codes:
        soap_text += f"\nICD-10 Codes: {', '.join(soap_note.icd_codes)}"
    if soap_note.cpt_codes:
        soap_text += f"\nCPT Codes: {', '.join(soap_note.cpt_codes)}"
    
    # Base64 encode the note
    import base64
    note_base64 = base64.b64encode(soap_text.encode()).decode()
    
    document = {
        "resourceType": "DocumentReference",
        "status": "current",
        "type": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "11506-3",
                    "display": "Progress note"
                }
            ]
        },
        "category": [
            {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "47039-3",
                        "display": "Hospital Admission evaluation note"
                    }
                ]
            }
        ],
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "date": datetime.utcnow().isoformat() + "Z",
        "author": [
            {
                "reference": f"Practitioner/{practitioner_id}"
            }
        ],
        "context": {},
        "content": [
            {
                "attachment": {
                    "contentType": "text/plain",
                    "data": note_base64,
                    "title": "SOAP Clinical Note"
                }
            }
        ]
    }
    if encounter_id:
        document["context"]["encounter"] = [{"reference": f"Encounter/{encounter_id}"}]
    
    return document

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5004, log_level="info")

def generate_idempotency_key(req: FHIRWriteRequest) -> str:
    # Deterministic idempotency key: note/session/practitioner/patient
    return f"note:{req.sessionId}:{req.patientId}:{req.practitionerId}"

def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())

def _get_nested_ci(data: Any, path: list[str]) -> Any:
    cur = data
    for part in path:
        if not isinstance(cur, dict):
            return None
        target = _normalize_key(part)
        match_key = None
        for key in cur.keys():
            if _normalize_key(key) == target:
                match_key = key
                break
        if match_key is None:
            return None
        cur = cur[match_key]
    return cur

def _to_xhtml(text: str) -> str:
    safe = escape(text or "")
    return f'<div xmlns="http://www.w3.org/1999/xhtml">{safe}</div>'

def _extract_vitals(soap_json: Dict[str, Any] | None) -> Dict[str, str]:
    if not isinstance(soap_json, dict):
        return {}
    vitals_block = _get_nested_ci(
        soap_json,
        ["Objective", "Clinical Examination Findings", "Vital Signs"],
    )
    if not isinstance(vitals_block, dict):
        return {}
    vitals: Dict[str, str] = {}
    for key, value in vitals_block.items():
        if not isinstance(value, str) or not value.strip():
            continue
        norm = _normalize_key(key)
        if norm in {"bp", "hr", "temp", "rr", "spo2"}:
            vitals[norm] = value.strip()
    return vitals

def build_observation_resources(
    soap_note: SOAPNote,
    patient_id: str,
    practitioner_id: str,
    encounter_id: Optional[str],
) -> list[Dict[str, Any]]:
    vitals = _extract_vitals(soap_note.soap_json)
    if not vitals:
        return []
    now = datetime.utcnow().isoformat() + "Z"
    loinc = {
        "bp": ("85354-9", "Blood pressure panel"),
        "hr": ("8867-4", "Heart rate"),
        "temp": ("8310-5", "Body temperature"),
        "rr": ("9279-1", "Respiratory rate"),
        "spo2": ("59408-5", "Oxygen saturation in Arterial blood by Pulse oximetry"),
    }
    observations: list[Dict[str, Any]] = []
    for key, value in vitals.items():
        code, display = loinc[key]
        obs: Dict[str, Any] = {
            "resourceType": "Observation",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": code,
                        "display": display,
                    }
                ]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": now,
            "valueString": value,
        }
        if encounter_id:
            obs["encounter"] = {"reference": f"Encounter/{encounter_id}"}
        if practitioner_id:
            obs["performer"] = [{"reference": f"Practitioner/{practitioner_id}"}]
        observations.append(obs)
    return observations

def build_composition_resource(
    soap_note: SOAPNote,
    patient_id: str,
    practitioner_id: str,
    encounter_id: Optional[str],
    document_reference_id: Optional[str],
    observation_ids: list[str],
) -> Dict[str, Any]:
    now = datetime.utcnow().isoformat() + "Z"
    sections = []
    for title, content in (
        ("Subjective", soap_note.subjective or "Not mentioned."),
        ("Objective", soap_note.objective or "Not mentioned."),
        ("Assessment", soap_note.assessment or "Not mentioned."),
        ("Plan", soap_note.plan or "Not mentioned."),
    ):
        section: Dict[str, Any] = {
            "title": title,
            "text": {"status": "generated", "div": _to_xhtml(content)},
        }
        if title == "Objective" and observation_ids:
            section["entry"] = [
                {"reference": f"Observation/{obs_id}"}
                for obs_id in observation_ids
            ]
        sections.append(section)
    if document_reference_id:
        sections.append(
            {
                "title": "Full Note",
                "text": {"status": "generated", "div": _to_xhtml("See attached document.")},
                "entry": [{"reference": f"DocumentReference/{document_reference_id}"}],
            }
        )
    composition: Dict[str, Any] = {
        "resourceType": "Composition",
        "status": "final",
        "type": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "11506-3",
                    "display": "Progress note",
                }
            ]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "date": now,
        "author": [{"reference": f"Practitioner/{practitioner_id}"}],
        "title": "SOAP Clinical Note",
        "section": sections,
    }
    if encounter_id:
        composition["encounter"] = {"reference": f"Encounter/{encounter_id}"}
    return composition
