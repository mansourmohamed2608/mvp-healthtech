from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="LLM Orchestrator")

class OrchestrateRequest(BaseModel):
    transcript: str
    sessionId: str

class OrchestrateResponse(BaseModel):
    intent: str
    entities: list[str]
    reply: str

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://llm:8000/infer")

@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(req: OrchestrateRequest):
    # Build a detailed prompt in Arabic requesting intent and entities
    prompt = (
        "أنت مساعد طبي عربي. استخرج نية المستخدم والكيانات الطبية، ثم قدم ردًا مختصرًا.\n"
        f"المستخدم: {req.transcript}\n"
        "المساعد:"
    )
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                LLM_ENDPOINT,
                json={"message": prompt, "sessionId": req.sessionId},
                timeout=30.0,
            )
        data = response.json()
        # Parse reply; split on semicolons to extract intent and entities.
        # Adjust according to your prompt format.
        parts = data["reply"].split(";")
        intent = parts[0].strip() if parts else ""
        entities = (
            [p.strip() for p in parts[1].split(",")] if len(parts) > 1 else []
        )
        reply = parts[-1].strip()
        return {"intent": intent, "entities": entities, "reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
