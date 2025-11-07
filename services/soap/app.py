# services/soap/app.py
"""
SOAP Note Generator Service
Week 4 Day 27 (Oct 21, 2025)
Generates structured clinical notes from transcripts using LLM
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import os

app = FastAPI(title="SOAP Generator Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:5001")

class SOAPRequest(BaseModel):
    transcript: str
    sessionId: str
    patientContext: Optional[dict] = None

class SOAPResponse(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str
    icd_codes: Optional[list[str]] = None
    cpt_codes: Optional[list[str]] = None

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "ok": True,
        "service": "soap-generator",
        "llm_url": LLM_SERVICE_URL,
    }

@app.post("/generate", response_model=SOAPResponse)
async def generate_soap(req: SOAPRequest):
    """
    Generate SOAP note from clinical transcript
    
    Process:
    1. Construct specialized medical prompt
    2. Call LLM service with SOAP extraction instructions
    3. Parse structured output
    4. Extract ICD/CPT codes (optional)
    """
    try:
        # Build SOAP extraction prompt
        prompt = build_soap_prompt(req.transcript, req.patientContext)
        
        # Call LLM service
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{LLM_SERVICE_URL}/infer",
                json={
                    "message": prompt,
                    "sessionId": req.sessionId,
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM service error: {response.text}"
                )
            
            llm_output = response.json()
            reply = llm_output.get("reply", "")
        
        # Parse SOAP sections from LLM output
        soap_note = parse_soap_sections(reply)
        
        print(f"✅ Generated SOAP note for session: {req.sessionId}")
        
        return soap_note
        
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="LLM service timeout"
        )
    except Exception as e:
        print(f"❌ SOAP generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"SOAP generation failed: {str(e)}"
        )

def build_soap_prompt(transcript: str, context: Optional[dict] = None) -> str:
    """
    Build specialized prompt for SOAP note extraction
    """
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

تأكد من أن كل قسم واضح ومفصل ومبني على المعلومات الواردة في النص."""

    if context:
        prompt += f"\n\nمعلومات إضافية عن المريض: {context}"
    
    return prompt

def parse_soap_sections(llm_reply: str) -> SOAPResponse:
    """
    Parse SOAP sections from LLM output
    Uses section markers to extract structured data
    """
    sections = {
        "subjective": "",
        "objective": "",
        "assessment": "",
        "plan": "",
    }
    
    # Split by section markers
    current_section = None
    lines = llm_reply.split("\n")
    
    for line in lines:
        line = line.strip()
        
        # Detect section headers
        if "subjective" in line.lower() or "الذاتي" in line:
            current_section = "subjective"
            continue
        elif "objective" in line.lower() or "الموضوعي" in line:
            current_section = "objective"
            continue
        elif "assessment" in line.lower() or "التقييم" in line:
            current_section = "assessment"
            continue
        elif "plan" in line.lower() or "الخطة" in line:
            current_section = "plan"
            continue
        
        # Append to current section
        if current_section and line and not line.startswith("["):
            sections[current_section] += line + "\n"
    
    # Fallback: if parsing failed, use simple split
    if not any(sections.values()):
        parts = llm_reply.split("\n\n")
        sections["subjective"] = parts[0] if len(parts) > 0 else "غير متوفر"
        sections["objective"] = parts[1] if len(parts) > 1 else "غير متوفر"
        sections["assessment"] = parts[2] if len(parts) > 2 else "غير متوفر"
        sections["plan"] = parts[3] if len(parts) > 3 else "غير متوفر"
    
    return SOAPResponse(
        subjective=sections["subjective"].strip() or "لم يتم ذكر شكوى محددة",
        objective=sections["objective"].strip() or "الفحص السريري طبيعي",
        assessment=sections["assessment"].strip() or "يتطلب مزيد من التقييم",
        plan=sections["plan"].strip() or "متابعة حسب الحاجة",
        icd_codes=None,  # TODO: Implement ICD code extraction
        cpt_codes=None,  # TODO: Implement CPT code extraction
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003, log_level="info")
