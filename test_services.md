# Service Testing Guide

## Service Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  - Web Browser (http://localhost:3000)                          │
│  - Twilio Voice SDK (for phone calls)                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GATEWAY (Port 3001)                           │
│  - NestJS API Gateway                                            │
│  - Authentication & Session Management                           │
│  - Routes requests to microservices                              │
└──────┬──────┬──────┬──────┬──────┬─────────────────────────────┘
       │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│   ASR    │ │   LLM    │ │   TTS    │ │   SOAP   │ │   FHIR   │
│ Port 5000│ │ Port 5001│ │ Port 5002│ │ Port 5003│ │ Port 5004│
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

## Data Flow Examples

### Voice Agent Flow:
```
User speaks in phone
    ↓
Twilio → Gateway (3001) → ASR (5000) → transcribed text
    ↓
Gateway → LLM (5001) → AI response text
    ↓
Gateway → TTS (5002) → audio response
    ↓
Twilio → User hears response
```

### Clinical Notes Flow:
```
Doctor records audio
    ↓
Frontend → Gateway (3001) → ASR (5000) → transcript
    ↓
Gateway → LLM (5001) → extract medical entities
    ↓
Gateway → SOAP (5003) → generate SOAP note
    ↓
Doctor reviews in UI
    ↓
Gateway → FHIR (5004) → write to EHR system
```

## Service Connection Details

### Gateway → ASR (Port 5000)
- **Endpoint**: `POST http://localhost:5000/transcribe`
- **Purpose**: Convert audio to text
- **Used by**: Voice calls, clinical dictation

### Gateway → LLM (Port 5001)
- **Endpoint**: `POST http://localhost:5001/generate`
- **Purpose**: Medical reasoning, conversation, entity extraction
- **Used by**: Voice agent responses, SOAP generation

### Gateway → TTS (Port 5002)
- **Endpoint**: `POST http://localhost:5002/synthesize`
- **Purpose**: Convert text to Arabic speech
- **Used by**: Voice agent (speak responses to patient)

### Gateway → SOAP (Port 5003)
- **Endpoint**: `POST http://localhost:5003/generate-note`
- **Purpose**: Generate structured clinical documentation
- **Used by**: Clinical notes automation

### Gateway → FHIR (Port 5004)
- **Endpoint**: `POST http://localhost:5004/create-encounter`
- **Purpose**: Write notes to EHR in FHIR format
- **Used by**: Final step of clinical notes workflow

## Session Flow (Redis)
```
Gateway creates session → stores in Redis
  ↓
Each service call includes sessionId
  ↓
Gateway tracks conversation context
  ↓
Session expires or ends → cleanup
```
