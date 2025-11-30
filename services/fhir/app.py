# services/fhir/app.py
"""
FHIR R4 Writeback Service
Week 5 Day 29 (Oct 23, 2025)
Writes SOAP notes to EHR using FHIR R4 API
"""
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import httpx
import os
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fhir")
# Optional OTEL
try:
    from otel_setup import init_otel
    init_otel("fhir")
except Exception:
    logger.debug("OTEL init skipped for FHIR")


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
INTERNAL_SECRET = os.getenv("INTERNAL_SECRET", "")
if not INTERNAL_SECRET:
    raise RuntimeError("INTERNAL_SECRET must be set for FHIR service")
if not FHIR_BASE_URL:
    raise RuntimeError("FHIR_BASE_URL must be set for FHIR service")

@app.middleware("http")
async def internal_auth(request: Request, call_next):
    if request.url.path.startswith("/health") or request.url.path.startswith("/ready"):
        return await call_next(request)
    if not INTERNAL_SECRET or request.headers.get("x-internal-secret") != INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await call_next(request)

# Configuration
FHIR_BASE_URL = os.getenv("FHIR_BASE_URL", "http://localhost:8080/fhir")
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
    icd_codes: Optional[list[str]] = None
    cpt_codes: Optional[list[str]] = None

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
        
        document_reference = build_document_reference(
            req.soapNote,
            req.patientId,
            req.practitionerId,
            encounter_resource.get("id")
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
            
            encounter_id = encounter_response.json().get("id")
            
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
        
        logger.info("FHIR write ok", extra={"docId": doc_id, "encounterId": encounter_id, "sessionId": req.sessionId})
        
        return FHIRWriteResponse(
            success=True,
            documentReferenceId=doc_id,
            encounterId=encounter_id
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
    encounter_id: str
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
        "context": {
            "encounter": [
                {
                    "reference": f"Encounter/{encounter_id}"
                }
            ]
        },
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
    
    return document

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5004, log_level="info")

def generate_idempotency_key(req: FHIRWriteRequest) -> str:
    # Deterministic idempotency key: note/session/practitioner/patient
    return f"note:{req.sessionId}:{req.patientId}:{req.practitionerId}"
