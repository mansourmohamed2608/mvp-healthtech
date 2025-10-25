# services/llm/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="LLM Service")

class InferRequest(BaseModel):
    message: str
    sessionId: str

class InferResponse(BaseModel):
    intent: str
    reply: str

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mmedu/mmed-llama-3-8b-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
)

# Load LoRA weights if present
try:
    model = PeftModel.from_pretrained(model, "/app/lora-llama")
except Exception:
    pass

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    try:
        prompt = f"You are a helpful Arabic medical assistant. Respond to: {req.message}"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # For simplicity, split the last line into intent and reply
        intent = decoded.strip().split(" ")[0]
        return {"intent": intent, "reply": decoded.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
