# Week 2 Audit - ASR and LLM Foundations (Oct 2-8, 2025)
**Audited by: Code Analysis Only (No MD files trusted)**
**Date: Oct 29, 2025**

---

## 📋 WEEK 2 TASKS OVERVIEW

| Day | Task | Status |
|-----|------|--------|
| Day 8 (Oct 2) | ASR environment & containerization | ✅ COMPLETE |
| Day 9 (Oct 3) | Arabic ASR adaptation & dataset | ⚠️ PARTIAL (LoRA exists, dataset gathering unclear) |
| Day 10 (Oct 4) | ASR microservice & integration | ✅ COMPLETE |
| Day 11 (Oct 5) | LLM deployment (MMed-Llama-3-8B) | ✅ COMPLETE |
| Day 12 (Oct 6) | LLM orchestrator service | ⚠️ PARTIAL (exists but not fully integrated) |
| Day 13 (Oct 7) | ASR ↔ LLM integration | ✅ COMPLETE (via Gateway) |
| Day 14 (Oct 8) | WER & intent evaluation | ⚠️ PARTIAL (eval script exists, not automated) |

**Overall Week 2: 78% Complete**

---

## ✅ DAY 8 (Oct 2) - ASR Environment & Containerization

### Status: ✅ **COMPLETE**

### Evidence Found:

**1. Dockerfile**
```dockerfile
# services/asr/Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
- Python 3 + pip ✅
- ffmpeg, libsndfile1 ✅
- GPU support (CUDA 12.1) ✅
EXPOSE 5000 ✅
CMD ["python", "app.py"] ✅
```

**2. Requirements.txt**
```python
fastapi==0.115.0 ✅
uvicorn[standard]==0.30.6 ✅
soundfile==0.12.1 ✅
librosa==0.10.2 ✅
openai-whisper==20231117 ✅
torch==2.3.1 ✅
transformers==4.44.2 ✅
peft==0.13.2 (LoRA) ✅
```

**3. GPU Support**
- DEVICE detection: `"cuda" if torch.cuda.is_available() else "cpu"` ✅
- Model loads on GPU: `device_map="auto"` ✅

**4. Real-Time Factor (RTF) Target: ≤0.5**
- ❌ **NOT MEASURED** - No RTF calculation in code
- ❌ **NOT LOGGED** - No performance metrics exported

### Missing:
- ❌ RTF measurement/logging
- ❌ Latency monitoring (target: partial transcripts every 200-300ms)

---

## ⚠️ DAY 9 (Oct 3) - Arabic ASR Adaptation & Dataset

### Status: ⚠️ **PARTIAL (70% Complete)**

### Evidence Found:

**1. LoRA Adapter EXISTS**
```python
# services/asr/app.py (lines 34-52)
ADAPTER_PATH = "./lora_ckpt" ✅
BASE_MODEL = "openai/whisper-large-v3" ✅
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH) ✅
```

**2. Training Scripts EXISTS**
```
services/asr/
  - train_lora_whisper.py ✅
  - train_dialect_lora.py ✅
  - dialect_adapter.py ✅
  - DIALECT_TRAINING.md ✅
```

**3. Evaluation Script EXISTS**
```python
# services/asr/eval_wer.py
from jiwer import wer ✅
def main(ref_jsonl, hyp_jsonl):
    scores = [wer(r['text'], h['text']) for r, h in zip(refs, hyps)]
    print(f"Average WER: {sum(scores)/len(scores)*100:.2f}%") ✅
```

**4. Dialect Detection EXISTS**
```python
# services/asr/dialect_adapter.py
def detect_dialect_from_text(text: str) -> str:
    - egyptian_markers ✅
    - levantine_markers ✅
    - gulf_markers ✅
```

### Missing:

#### ❌ **Arabic Medical Speech Dataset**
**Plan Required**: "Gather Arabic medical speech data (public corpora or existing internal data)"

**Current State:**
- `services/asr/data/` directory exists
- `download_dataset.py` exists
- `make_synth_audio.py` exists (synthetic data generation)
- ❌ **NO EVIDENCE** of actual medical speech corpus
- ❌ **NO EVIDENCE** of data collection pipeline
- ❌ **NO EVIDENCE** of dataset size/quality metrics

**What's Needed:**
```python
# Download/prepare Arabic medical speech dataset:
1. CommonVoice Arabic dataset
2. Medical terminology audio corpus
3. Egyptian/Levantine/Gulf dialect samples
4. Target: 10+ hours per dialect minimum
```

#### ❌ **WER Evaluation Not Automated**
- eval_wer.py exists but requires manual invocation
- No automated CI/CD evaluation
- No golden test set defined in repo

---

## ✅ DAY 10 (Oct 4) - ASR Microservice & Integration

### Status: ✅ **COMPLETE**

### Evidence Found:

**1. FastAPI Microservice**
```python
# services/asr/app.py
app = FastAPI(title="ASR Service") ✅
CORS enabled ✅
Port: 5000 ✅
```

**2. Endpoints**
```python
GET  /health ✅
POST /transcribe (response_model=TranscribeResponse) ✅
POST /stream (streaming endpoint) ✅
```

**3. Features**
```python
- Base64 audio decoding ✅
- Mono conversion (stereo → mono) ✅
- Resampling to 16kHz ✅
- Whisper-large-v3 + LoRA inference ✅
- Dialect parameter support ✅
- Error handling with HTTPException ✅
```

**4. Gateway Integration**
```typescript
// gateway/src/asr/asr.controller.ts
@Controller('asr')
export class AsrController {
  @Post('transcribe') ✅
  @Post('stream') ✅
}

// gateway/src/asr/asr.service.ts
async transcribe(audioData: string, callSid?: string, dialect?: string) ✅
```

**5. Latency Target: Partial transcripts every 200-300ms**
- ❌ **NOT IMPLEMENTED** - /stream endpoint exists but doesn't emit partials
- ❌ **NOT MEASURED** - No timestamp logging

### Missing:
- ❌ True streaming with partial transcripts every 200-300ms
- ❌ Latency measurement/logging

---

## ✅ DAY 11 (Oct 5) - LLM Deployment (MMed-Llama-3-8B)

### Status: ✅ **COMPLETE**

### Evidence Found:

**1. Model Deployment**
```python
# services/llm/app.py
MODEL_NAME = "Henrychur/MMed-Llama-3-8B" ✅
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" ✅

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32, ✅
    device_map="auto", ✅
    low_cpu_mem_usage=True, ✅
)
```

**2. LoRA Adapter Support**
```python
try:
    model = PeftModel.from_pretrained(model, "/app/lora-llama") ✅
except Exception as e:
    print(f"No LoRA weights found: {e}") ✅
```

**3. GPU Allocation**
- ✅ Automatic GPU detection
- ✅ float16 for CUDA, float32 for CPU
- ✅ device_map="auto" handles memory

**4. Dockerfile**
```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 ✅
EXPOSE 5001 ✅
```

**5. Prompt Templates**
```python
def build_rag_prompt(message: str, intent: str) -> str:
    prompt_parts = [
        "أنت مساعد طبي ذكي يتحدث العربية..." ✅
    ]
    # Few-shot examples ✅
    # Relevant FAQs ✅
```

### Missing:
- ❌ **First token latency <300ms** - NOT MEASURED
- ❌ **Complete response <1.5s** - NOT MEASURED
- ❌ **Memory allocation monitoring** - No Prometheus metrics

---

## ⚠️ DAY 12 (Oct 6) - LLM Orchestrator Service

### Status: ⚠️ **PARTIAL (60% Complete)**

### Evidence Found:

**1. Orchestrator EXISTS**
```python
# services/llm/orchestrator.py
app = FastAPI(title="LLM Orchestrator") ✅

@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(req: OrchestrateRequest):
    - Intent extraction ✅
    - Entity extraction ✅
    - Prompt building ✅
```

**2. RAG Store EXISTS**
```python
# services/llm/rag_store.py
rag_store.get_few_shot_examples(intent, limit=2) ✅
rag_store.get_relevant_faqs(message, limit=2) ✅
```

**3. Intent Classification EXISTS**
```python
# services/llm/app.py
def classify_intent(message: str) -> str:
    # appointment, symptom, prescription, medical_history ✅
```

### Missing:

#### ❌ **Orchestrator NOT Running as Separate Service**
**Plan Required**: "Build the LLM orchestrator service that accepts transcripts and returns structured intents or actions"

**Current State:**
- orchestrator.py exists but NOT exposed as a service
- No separate port (should be 5001 or separate)
- ❌ NOT in docker-compose.yml
- ❌ NOT in gateway routing

**What's Needed:**
```yaml
# infra/docker-compose.yml (ADD)
llm-orchestrator:
  build: ../services/llm
  command: python orchestrator.py
  ports:
    - "5006:8000"
  environment:
    - LLM_ENDPOINT=http://llm:5001/infer
```

#### ❌ **Safe Prompting & Tool Invocation**
**Plan Required**: "Implement safe prompting and tool invocation"

**Missing:**
- ❌ Prompt injection prevention
- ❌ Output sanitization
- ❌ Tool routing (calendar, appointments, prescriptions)
- ❌ Guardrails for medical advice

#### ❌ **Policy Guardrails**
**Plan Required**: "policy guardrails"

**Missing:**
- ❌ Medical liability guardrails
- ❌ Escalation triggers
- ❌ Conversation policies (max turns, timeout)

---

## ✅ DAY 13 (Oct 7) - ASR ↔ LLM Integration

### Status: ✅ **COMPLETE**

### Evidence Found:

**1. Gateway Integration**
```typescript
// gateway/src/conversation/conversation.service.ts
@Injectable()
export class ConversationService {
  async appendMessage(sessionId, role, content, metadata) ✅
  async getMessages(sessionId, limit=10) ✅
  async getState(sessionId) ✅
  async updateContext(sessionId, context) ✅
}
```

**2. Stateful Conversation Manager**
```typescript
- Redis-backed message storage ✅
- MAX_MESSAGES = 20 ✅
- CONVERSATION_TTL = 7200 (2 hours) ✅
- Retry logic with MAX_RETRIES = 3 ✅
- Exponential backoff ✅
```

**3. Session Metadata Storage**
```typescript
// gateway/src/session/session.service.ts
async create(dto: CreateSessionDto): Promise<CreateSessionResponseDto> ✅
async findById(sessionId: string) ✅
```

**4. Data Flow**
```
User Speech → ASR Service (/transcribe) ✅
     ↓
Gateway → ConversationService.appendMessage('user', text) ✅
     ↓
Gateway → LLM Service (/infer) with conversation history ✅
     ↓
Gateway → ConversationService.appendMessage('assistant', reply) ✅
```

### Missing:
- ❌ **Partial transcript streaming** - ASR doesn't send partial results
- ❌ **Context window optimization** - No semantic compression

---

## ⚠️ DAY 14 (Oct 8) - WER & Intent Evaluation

### Status: ⚠️ **PARTIAL (50% Complete)**

### Evidence Found:

**1. WER Evaluation Script**
```python
# services/asr/eval_wer.py
from jiwer import wer ✅
def main(ref_jsonl, hyp_jsonl):
    scores = [wer(r['text'], h['text']) for r, h in zip(refs, hyps)] ✅
```

**2. JIWER Dependency**
```
requirements.txt: jiwer==3.0.4 ✅
```

**3. Intent Classification Code**
```python
# services/llm/app.py
def classify_intent(message: str) -> str:
    keywords = {
        "appointment": [...], ✅
        "symptom": [...], ✅
        "prescription": [...], ✅
        "medical_history": [...] ✅
    }
```

### Missing:

#### ❌ **Automated WER Evaluation**
**Plan Required**: "Compute WER on development data and measure intent accuracy"

**Missing:**
- ❌ Golden test set not committed to repo
- ❌ No CI/CD integration for WER evaluation
- ❌ No benchmark results documented
- ❌ No automated regression testing

**What's Needed:**
```bash
# .github/workflows/test-asr.yml
- name: Evaluate WER
  run: |
    cd services/asr
    python eval_wer.py data/golden/refs.jsonl outputs/hyps.jsonl
    # Assert WER < 15%
```

#### ❌ **Intent Accuracy Measurement**
**Plan Required**: "measure intent accuracy"

**Missing:**
- ❌ No intent evaluation script
- ❌ No intent test dataset
- ❌ No accuracy metrics logged
- ❌ No benchmark (target: ≥70% precision)

**What's Needed:**
```python
# services/llm/eval_intent.py
from sklearn.metrics import accuracy_score, f1_score

def evaluate_intent(test_jsonl: str):
    correct = 0
    total = 0
    for line in open(test_jsonl):
        data = json.loads(line)
        predicted = classify_intent(data['message'])
        if predicted == data['intent']:
            correct += 1
        total += 1
    accuracy = correct / total
    print(f"Intent Accuracy: {accuracy * 100:.1f}%")
    # Assert accuracy >= 70%
```

#### ❌ **LoRA Weight Adjustment**
**Plan Required**: "Adjust prompts, hyper-parameters and LoRA weights to aim for ≥70 % precision"

**Missing:**
- ❌ No hyperparameter tuning documentation
- ❌ No LoRA weight versioning
- ❌ No A/B testing framework

---

## 📊 WEEK 2 COMPLETION SUMMARY

| Day | Task | Status | Completion % |
|-----|------|--------|--------------|
| 8 | ASR containerization | ✅ | 90% (missing RTF measurement) |
| 9 | Arabic dataset & LoRA | ⚠️ | 70% (LoRA ✅, dataset unclear) |
| 10 | ASR microservice | ✅ | 90% (missing true streaming) |
| 11 | LLM deployment | ✅ | 95% (missing latency metrics) |
| 12 | LLM orchestrator | ⚠️ | 60% (exists but not deployed) |
| 13 | ASR ↔ LLM integration | ✅ | 95% (working via gateway) |
| 14 | WER & intent evaluation | ⚠️ | 50% (scripts exist, not automated) |

**Overall Week 2: 78% Complete**

---

## 🔧 CRITICAL MISSING ITEMS - Week 2

### High Priority (Must Fix):

#### 1. **Arabic Medical Speech Dataset** (Day 9)
**Status**: ❌ NOT VERIFIED
```bash
# What's needed:
- CommonVoice Arabic: 10+ hours
- Medical terminology corpus
- Dialect-specific samples (Egyptian/Levantine/Gulf)
- Dataset statistics documented
```

#### 2. **LLM Orchestrator Deployment** (Day 12)
**Status**: ❌ NOT DEPLOYED AS SERVICE
```yaml
# Add to docker-compose.yml:
llm-orchestrator:
  build: ./services/llm
  command: python orchestrator.py
  ports:
    - "5006:8000"
```

#### 3. **Automated WER/Intent Evaluation** (Day 14)
**Status**: ❌ NOT AUTOMATED
```bash
# Create CI/CD pipeline:
.github/workflows/
  - test-asr-wer.yml
  - test-llm-intent.yml
```

#### 4. **Policy Guardrails** (Day 12)
**Status**: ❌ NOT IMPLEMENTED
```python
# Add to services/llm/guardrails.py:
- Medical advice disclaimers
- Escalation triggers
- Conversation policies
```

#### 5. **Latency Measurements** (Days 10-11)
**Status**: ❌ NOT MEASURED
```python
# Add Prometheus metrics:
- ASR RTF (target: ≤0.5)
- LLM first token latency (target: <300ms)
- LLM complete response (target: <1.5s)
- Partial transcript timing (target: every 200-300ms)
```

### Medium Priority:

6. **True Streaming ASR** - Partial transcripts every 200-300ms
7. **Tool Routing** - Calendar, appointments, prescriptions
8. **Prompt Injection Prevention**
9. **Context Window Optimization**
10. **LoRA Weight Versioning**

---

## 🎯 NEXT STEPS

1. **Document Arabic Dataset** - Verify data sources and quality metrics
2. **Deploy LLM Orchestrator** - Add to docker-compose and gateway routing
3. **Automate Evaluation** - CI/CD for WER and intent accuracy
4. **Add Policy Guardrails** - Medical liability and escalation
5. **Implement Latency Monitoring** - Prometheus metrics for all services
6. **Continue Week 3 Audit** - TTS, core platform services

---

**End of Week 2 Audit**
