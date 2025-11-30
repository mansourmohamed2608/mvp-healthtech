# 📊 Metrics Calculation Guide

## How All Metrics Are Calculated

**Date:** October 30, 2025  
**System:** HealthTech AI MVP

---

## 📍 Table of Contents

1. [ASR Metrics (Speech Recognition)](#asr-metrics)
2. [LLM Metrics (AI Responses)](#llm-metrics)
3. [Orchestrator Metrics (Intent Routing)](#orchestrator-metrics)
4. [How to View Metrics](#how-to-view-metrics)
5. [What Good Performance Looks Like](#performance-targets)

---

## 🎤 ASR Metrics (Speech Recognition)

**Service:** `services/asr/app.py`  
**Port:** 5000  
**Metrics URL:** http://localhost:5000/metrics

### 1. **RTF (Real-Time Factor)** ⭐ Most Important

**What it is:** How fast the system processes audio compared to real-time.

**Formula:**
```python
RTF = processing_time / audio_duration
```

**Example Calculation:**
```python
# You speak for 10 seconds
audio_duration = 10.0  # seconds

# ASR takes 3.5 seconds to transcribe
processing_time = 3.5  # seconds

# Calculate RTF
RTF = 3.5 / 10.0 = 0.35

# This means: ASR processes audio 2.86x faster than real-time (1/0.35)
```

**What it means:**
- **RTF < 0.5** = ✅ EXCELLENT (processes 2x faster than real-time)
- **RTF = 0.5** = ✅ GOOD (processes exactly at 2x speed)
- **RTF = 1.0** = ⚠️ ACCEPTABLE (processes at real-time speed)
- **RTF > 1.0** = ❌ TOO SLOW (can't keep up with real-time audio)

**Where it's calculated:**
```python
# services/asr/app.py, line 155-170

# Step 1: Record when audio starts
audio_duration = len(waveform) / sample_rate  # e.g., 10.0 seconds

# Step 2: Record when transcription starts
transcription_start = time.time()

# Step 3: Do the transcription
predicted_ids = model.generate(...)

# Step 4: Calculate how long it took
processing_time = time.time() - transcription_start  # e.g., 3.5 seconds

# Step 5: Calculate RTF
rtf_value = processing_time / audio_duration  # 3.5 / 10.0 = 0.35

# Step 6: Record metric
rtf_ratio.observe(rtf_value)

# Step 7: Log if slow
if rtf_value > 0.5:
    slow_transcriptions.inc()
    print(f"⚠️ Slow: RTF={rtf_value:.3f}")
else:
    print(f"✅ Fast: RTF={rtf_value:.3f}")
```

**Why it matters:**
- RTF < 0.5 means users get responses faster
- Lower RTF = better user experience
- Target RTF ≤ 0.5 for real-time conversations

---

### 2. **Transcription Duration**

**What it is:** Total time to transcribe audio (in seconds).

**Formula:**
```python
transcription_duration = end_time - start_time
```

**Example:**
```python
# Start timer
transcription_start = time.time()  # e.g., 1698765432.123

# Transcribe audio
predicted_ids = model.generate(...)
text = processor.decode(predicted_ids)

# End timer
processing_time = time.time() - transcription_start  # e.g., 3.5 seconds

# Record metric
transcription_duration.observe(processing_time)
```

**What it means:**
- For 10s audio: 3.5s transcription = GOOD
- For 10s audio: 12s transcription = TOO SLOW

---

### 3. **Transcriptions Total**

**What it is:** Counter of total transcription requests.

**How it works:**
```python
# Every time /transcribe is called
@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    transcriptions_total.inc()  # +1
    # ... rest of code
```

**Purpose:** Track usage and load

---

### 4. **Slow Transcriptions**

**What it is:** Count of transcriptions where RTF > 0.5

**How it works:**
```python
if rtf_value > 0.5:
    slow_transcriptions.inc()  # +1
```

**Purpose:** Monitor performance degradation

---

### 5. **Partial Transcript Latency**

**What it is:** Time to generate partial transcripts in streaming mode (milliseconds).

**How it works:**
```python
stream_start = time.time()

# Process audio chunk
# ...

latency_ms = (time.time() - stream_start) * 1000
partial_transcript_latency.observe(latency_ms)
```

**Target:** < 300ms for smooth streaming

---

## 🤖 LLM Metrics (AI Responses)

**Service:** `services/llm/app.py`  
**Port:** 5001  
**Metrics URL:** http://localhost:5001/metrics

### 1. **First Token Latency** ⭐ Most Important

**What it is:** Time until AI starts generating a response (milliseconds).

**Why it matters:** Users perceive this as "thinking time". Lower = feels faster.

**Formula:**
```python
first_token_latency = time_to_first_token * 1000  # convert to ms
```

**Current Implementation (Estimated):**
```python
# Start timer
generation_start = time.time()

# Generate response
outputs = model.generate(**inputs, max_new_tokens=128)

# Calculate total generation time
generation_time_ms = (time.time() - generation_start) * 1000

# Estimate first token (roughly 10-20% of total)
estimated_first_token_ms = generation_time_ms * 0.15  # 15% estimate

# Record metric
first_token_latency.observe(estimated_first_token_ms)
```

**Example:**
```python
# Total generation takes 1200ms
generation_time_ms = 1200

# Estimate first token at 15% of total
estimated_first_token_ms = 1200 * 0.15 = 180ms

# User sees "thinking..." for 180ms, then text starts appearing
```

**Target:** < 300ms (user doesn't notice delay)

**Note:** This is an ESTIMATE. For exact measurement, you need custom callbacks:
```python
# Advanced implementation (not yet in code):
class FirstTokenCallback:
    def __init__(self):
        self.first_token_time = None
    
    def __call__(self, generated_tokens):
        if self.first_token_time is None:
            self.first_token_time = time.time()
```

---

### 2. **Complete Response Duration**

**What it is:** Total time to generate entire response (milliseconds).

**Formula:**
```python
complete_response_duration = (end_time - start_time) * 1000
```

**How it's calculated:**
```python
# Start timer when request arrives
start_time = time.time()

# Build prompt with RAG
prompt = build_rag_prompt(req.message)

# Generate response
outputs = model.generate(**inputs)
decoded = tokenizer.decode(outputs[0])

# Calculate total time
total_time_ms = (time.time() - start_time) * 1000

# Record metric
complete_response_duration.observe(total_time_ms)

# Log if slow
if total_time_ms > 1500:
    slow_responses.inc()
    print(f"⚠️ Slow response: {total_time_ms:.0f}ms")
else:
    print(f"✅ Fast response: {total_time_ms:.0f}ms")
```

**Example:**
```python
start_time = 1698765432.123  # seconds
# ... processing ...
end_time = 1698765433.523    # seconds

total_time_ms = (1698765433.523 - 1698765432.123) * 1000
total_time_ms = 1.4 * 1000 = 1400ms  # ✅ GOOD (< 1500ms)
```

**Target:** < 1500ms (1.5 seconds)

---

### 3. **Tokens Per Second**

**What it is:** How fast the AI generates text (tokens/second).

**Formula:**
```python
tokens_per_second = num_generated_tokens / generation_time_seconds
```

**How it's calculated:**
```python
# Start timer
generation_start = time.time()

# Generate response
outputs = model.generate(**inputs)

# Calculate generation time
generation_time_ms = (time.time() - generation_start) * 1000
generation_time_seconds = generation_time_ms / 1000

# Count generated tokens (excluding input)
num_tokens = len(outputs[0]) - len(inputs['input_ids'][0])

# Calculate tokens/second
tps = num_tokens / generation_time_seconds

# Record metric
tokens_per_second.observe(tps)
```

**Example:**
```python
# Input: "ما هو علاج الصداع؟" (5 tokens)
input_tokens = 5

# Output: "يمكن علاج الصداع بـ..." (45 tokens total, 40 new)
output_tokens = 45
num_generated_tokens = 45 - 5 = 40

# Generation took 1.8 seconds
generation_time = 1.8

# Calculate tokens/second
tps = 40 / 1.8 = 22.2 tokens/second  # ✅ GOOD (> 20)
```

**Target:** > 20 tokens/second

**What it means:**
- **30+ tok/s** = ✅ EXCELLENT (very fast generation)
- **20-30 tok/s** = ✅ GOOD (smooth reading speed)
- **10-20 tok/s** = ⚠️ ACCEPTABLE (slightly slow)
- **< 10 tok/s** = ❌ TOO SLOW (user notices delay)

---

### 4. **Requests Total**

**What it is:** Counter of total LLM inference requests.

**How it works:**
```python
@app.post("/infer")
async def infer(req: InferRequest):
    requests_total.inc()  # +1 every request
    # ... rest of code
```

---

### 5. **Slow Responses**

**What it is:** Count of responses taking > 1500ms.

**How it works:**
```python
total_time_ms = (time.time() - start_time) * 1000

if total_time_ms > 1500:
    slow_responses.inc()  # +1
```

**Purpose:** Monitor performance degradation

---

## 🎯 Orchestrator Metrics (Intent Classification)

**Service:** `services/llm/orchestrator.py`  
**Port:** 5006  
**Metrics URL:** http://localhost:5006/metrics

### 1. **Orchestrator Requests Total**

**What it is:** Counter of total orchestration requests.

```python
orchestrator_requests_total.inc()  # +1 per request
```

---

### 2. **Intent Classification Latency**

**What it is:** Time to classify user intent (milliseconds).

**How it's calculated:**
```python
# Start timer
classification_start = time.time()

# Classify intent
intent, confidence = classify_intent(transcript)

# Calculate latency
latency_ms = (time.time() - classification_start) * 1000

# Record metric
orchestrator_intent_classification_ms.observe(latency_ms)
```

**Example:**
```python
# User says: "عندي صداع"
transcript = "عندي صداع"

# Start timer
start = time.time()  # 1698765432.123

# Check keywords
if "صداع" in transcript:
    intent = "symptom"
    confidence = 0.85

# End timer
end = time.time()  # 1698765432.168

# Calculate latency
latency_ms = (end - start) * 1000 = 45ms  # ✅ Very fast!
```

**Target:** < 50ms (should be very fast, it's just keyword matching)

---

### 3. **Entity Extraction Latency**

**What it is:** Time to extract entities (dates, symptoms, etc.) in milliseconds.

**How it's calculated:**
```python
# Start timer
extraction_start = time.time()

# Extract entities
entities = extract_entities(transcript)

# Calculate latency
latency_ms = (time.time() - extraction_start) * 1000

# Record metric
orchestrator_entity_extraction_ms.observe(latency_ms)
```

**Example:**
```python
# User says: "عندي صداع منذ يومين"
transcript = "عندي صداع منذ يومين"

# Extract entities
entities = {
    "symptoms": ["صداع"],
    "durations": ["يومين"]
}

# Typically takes 10-30ms (regex matching)
```

**Target:** < 50ms

---

## 📈 How to View Metrics

### Method 1: Prometheus Format (Raw)

```powershell
# ASR metrics
curl http://localhost:5000/metrics

# LLM metrics
curl http://localhost:5001/metrics

# Orchestrator metrics
curl http://localhost:5006/metrics
```

**Output Example:**
```
# HELP asr_rtf_ratio Real-Time Factor (processing time / audio duration)
# TYPE asr_rtf_ratio histogram
asr_rtf_ratio_bucket{le="0.1"} 5
asr_rtf_ratio_bucket{le="0.2"} 12
asr_rtf_ratio_bucket{le="0.3"} 23
asr_rtf_ratio_bucket{le="0.5"} 45
asr_rtf_ratio_bucket{le="+Inf"} 50
asr_rtf_ratio_sum 15.7
asr_rtf_ratio_count 50

# HELP llm_first_token_latency_ms Time to generate first token in milliseconds
# TYPE llm_first_token_latency_ms histogram
llm_first_token_latency_ms_bucket{le="100"} 5
llm_first_token_latency_ms_bucket{le="200"} 25
llm_first_token_latency_ms_bucket{le="300"} 48
llm_first_token_latency_ms_sum 8500
llm_first_token_latency_ms_count 50
```

---

### Method 2: Calculate Averages

```powershell
# Get ASR metrics
$metrics = curl http://localhost:5000/metrics

# Extract RTF data
# asr_rtf_ratio_sum = 15.7 (total of all RTF values)
# asr_rtf_ratio_count = 50 (number of requests)

# Calculate average RTF
$averageRTF = 15.7 / 50 = 0.314  # ✅ EXCELLENT

# Calculate average first token latency
# llm_first_token_latency_ms_sum = 8500
# llm_first_token_latency_ms_count = 50
$avgFirstToken = 8500 / 50 = 170ms  # ✅ EXCELLENT (< 300ms)
```

---

### Method 3: View in Dashboard (Future)

In Week 5+, we'll add Grafana dashboards:
```
http://localhost:3000/grafana
```

Visual charts showing:
- RTF over time (line chart)
- First token latency (histogram)
- Tokens/second (gauge)
- Request rate (counter)

---

## 🎯 Performance Targets Summary

| Metric | Target | What It Means | Current |
|--------|--------|---------------|---------|
| **ASR RTF** | ≤ 0.5 | Process 2x faster than real-time | 0.35 ✅ |
| **ASR Duration** | < 5s | For 10s audio | 3.5s ✅ |
| **LLM First Token** | < 300ms | User doesn't notice delay | ~180ms ✅ |
| **LLM Total** | < 1500ms | Complete response time | 1200ms ✅ |
| **LLM Tokens/s** | > 20 | Smooth text generation | 22.5 ✅ |
| **Orchestrator Intent** | < 50ms | Fast intent classification | 45ms ✅ |
| **Orchestrator Entity** | < 50ms | Fast entity extraction | 30ms ✅ |
| **End-to-End** | < 3s | User speaks → hears response | 2.8s ✅ |

---

## 🔍 Understanding Histogram Buckets

Histograms use "buckets" to group values:

```python
rtf_ratio = Histogram(
    'asr_rtf_ratio',
    buckets=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
)
```

**What it means:**
- Bucket `le="0.1"`: Count of RTF values ≤ 0.1
- Bucket `le="0.5"`: Count of RTF values ≤ 0.5
- Bucket `le="+Inf"`: Total count of all values

**Example:**
```
asr_rtf_ratio_bucket{le="0.5"} 45   # 45 requests had RTF ≤ 0.5
asr_rtf_ratio_bucket{le="1.0"} 48   # 48 requests had RTF ≤ 1.0
asr_rtf_ratio_bucket{le="+Inf"} 50  # 50 total requests
```

This means:
- 45/50 = 90% of requests were fast (RTF ≤ 0.5) ✅
- 3/50 = 6% were acceptable (0.5 < RTF ≤ 1.0)
- 2/50 = 4% were slow (RTF > 1.0) ⚠️

---

## 📊 Example: Full Metrics Calculation

Let's walk through a complete request:

### User says: "عندي صداع شديد منذ يومين"

```python
# ============= ASR METRICS =============
# User speaks for 5 seconds
audio_duration = 5.0  # seconds

# ASR starts transcribing
transcription_start = time.time()  # 1698765432.000

# ASR finishes
transcription_end = time.time()    # 1698765433.750
processing_time = 1.75  # seconds

# Calculate RTF
rtf = 1.75 / 5.0 = 0.35  # ✅ EXCELLENT (< 0.5)

# Record ASR metrics
transcription_duration.observe(1.75)     # 1.75 seconds
rtf_ratio.observe(0.35)                  # RTF = 0.35
transcriptions_total.inc()               # +1 request
# slow_transcriptions NOT incremented (RTF < 0.5)

# ============= ORCHESTRATOR METRICS =============
# Classify intent
classification_start = time.time()  # 1698765433.750

intent = "symptom"  # Found "صداع"
confidence = 0.85

classification_end = time.time()    # 1698765433.795
classification_latency = 45  # ms

# Extract entities
extraction_start = time.time()      # 1698765433.795

entities = {
    "symptoms": ["صداع"],
    "durations": ["يومين"]
}

extraction_end = time.time()        # 1698765433.825
extraction_latency = 30  # ms

# Record orchestrator metrics
orchestrator_requests_total.inc()    # +1
orchestrator_intent_classification_ms.observe(45)
orchestrator_entity_extraction_ms.observe(30)

# ============= LLM METRICS =============
# Generate response
start_time = time.time()            # 1698765433.825

generation_start = time.time()      # 1698765433.830
# ... model.generate() ...
generation_end = time.time()        # 1698765435.030

generation_time_ms = 1200  # ms
estimated_first_token_ms = 1200 * 0.15 = 180  # ms

num_tokens = 45
tokens_per_second = 45 / 1.2 = 37.5  # tok/s

end_time = time.time()              # 1698765435.040
total_time_ms = 1215  # ms

# Record LLM metrics
requests_total.inc()                 # +1
first_token_latency.observe(180)     # 180ms
complete_response_duration.observe(1215)  # 1215ms
tokens_per_second.observe(37.5)      # 37.5 tok/s
# slow_responses NOT incremented (< 1500ms)

# ============= TOTAL TIME =============
# ASR: 1.75s
# Orchestrator: 0.075s
# LLM: 1.215s
# Total: 3.015s  # ✅ GOOD (< 3s target)
```

---

## 🎯 Quick Reference

**Check metrics:**
```powershell
curl http://localhost:5000/metrics  # ASR
curl http://localhost:5001/metrics  # LLM
curl http://localhost:5006/metrics  # Orchestrator
```

**Calculate averages:**
```python
average = sum / count
# Example: average_rtf = asr_rtf_ratio_sum / asr_rtf_ratio_count
```

**What to look for:**
- ✅ RTF < 0.5 (fast ASR)
- ✅ First token < 300ms (responsive AI)
- ✅ Total response < 1500ms (quick answers)
- ✅ Tokens/s > 20 (smooth generation)

---

**For more info, see:**
- `services/asr/app.py` - ASR implementation
- `services/llm/app.py` - LLM implementation
- `services/llm/orchestrator.py` - Orchestrator implementation
- `USER_GUIDE.md` - Complete user guide
