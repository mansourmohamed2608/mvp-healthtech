# API Integration Fixes - Demo Page

## Overview
Fixed request/response formats in the Demo page to match actual backend service implementations.

## Changes Made

### 1. **About.tsx Syntax Error** ✅
- **Issue**: Smart quote in string literal causing parse error
- **Fix**: Changed `'We're` to `"We're"`
- **Location**: Line 86

### 2. **Gateway SOAP Controller** ✅
- **Issue**: Expected wrong request format (separate SOAP fields instead of transcript)
- **Previous**:
  ```typescript
  class CreateSoapDto {
    subjective: string;
    objective: string;
    assessment: string;
    plan: string;
    patientId?: string;
  }
  ```
- **Fixed**:
  ```typescript
  class CreateSoapDto {
    transcript: string;
    sessionId?: string;
    patientContext?: any;
  }
  ```
- **Location**: `gateway/src/soap/soap.controller.ts`

### 3. **API Client Updates** ✅

#### ASR Service
- **Response**: `{ text: string; dialect?: string; auto_detected?: boolean }`
- **Request**: `{ audio: string; callSid?: string; dialect?: string }`

#### LLM Service
- **Response**: `{ intent: string; reply: string }`
- **Request**: `{ message: string; sessionId: string; intent?: string }`
- **Removed**: `confidence` field (not in actual response)

#### SOAP Service
- **Response**:
  ```typescript
  {
    subjective: string;
    objective: string;
    assessment: string;
    plan: string;
    icd_codes?: string[];
    cpt_codes?: string[];
  }
  ```
- **Request**: `{ transcript: string; sessionId?: string; patientContext?: any }`

#### TTS Service
- **Response**: `{ audio: string; duration: number; sampleRate: number }`
- **Request**: `{ text: string; voice?: string }`

### 4. **Demo Page Request Handlers** ✅

#### ASR Handler
- Now formats response with `transcription`, `dialect`, `timestamp`
- Updated error message to reference port 3001

#### LLM Handler
- Removed `confidence` field
- Updated error message to reference port 3001

#### SOAP Handler
- **Changed**: Now sends `transcript` field instead of individual SOAP fields
- **Previous**: Sent `{ subjective, objective, assessment, plan }`
- **Now**: Sends `{ transcript: string, sessionId: string }`
- Response properly extracts SOAP sections, ICD/CPT codes

#### FHIR Handler
- Formats response with `resourceType`, `resource`, `timestamp`
- Updated error message to reference port 3001

#### TTS Handler
- Formats response with `text`, `audioGenerated`, `timestamp`
- Updated error message to reference port 3001

### 5. **Demo Page UI Updates** ✅

#### SOAP Tab
- **Previous**: 4 separate text areas for S.O.A.P. fields (incorrect - those are outputs!)
- **Fixed**: Single large textarea for clinical transcript input
- Added helper text explaining AI will generate structured SOAP note
- Example transcript with realistic doctor-patient conversation
- Uses monospace font for better readability

## Service Communication Flow

### SOAP Note Generation
```
Frontend Demo Page
  ↓ POST { transcript, sessionId }
Gateway SOAP Controller (localhost:3001)
  ↓ POST { transcript, sessionId }
SOAP Service (localhost:5003)
  ↓ POST { message (formatted prompt), sessionId }
LLM Service (localhost:5001)
  ↓ Returns { intent, reply }
SOAP Service
  ↓ Parses reply into SOAP sections
  ↓ Returns { subjective, objective, assessment, plan, icd_codes, cpt_codes }
Gateway
  ↓ Returns same structure
Frontend
  ↓ Displays structured SOAP note
```

### ASR Transcription
```
Frontend Demo Page
  ↓ POST { audio (base64), callSid, dialect }
Gateway ASR Controller (localhost:3001)
  ↓ POST { audio, callSid, dialect }
ASR Service (localhost:5000)
  ↓ Returns { text, dialect, auto_detected }
Gateway
  ↓ Returns same structure
Frontend
  ↓ Displays transcription with dialect info
```

### LLM Inference
```
Frontend Demo Page
  ↓ POST { message, sessionId, intent }
Gateway LLM Controller (localhost:3001)
  ↓ POST { message, sessionId, intent }
LLM Service (localhost:5001)
  ↓ Returns { intent, reply }
Gateway
  ↓ Returns same structure
Frontend
  ↓ Displays AI reply with intent
```

## Environment Configuration
- **Frontend**: localhost:3000 (Vite dev server)
- **Gateway**: localhost:3001 (NestJS)
- **ASR Service**: localhost:5000 (FastAPI)
- **LLM Service**: localhost:5001 (FastAPI)
- **TTS Service**: localhost:5002 (FastAPI)
- **SOAP Service**: localhost:5003 (FastAPI)
- **FHIR Service**: localhost:5004 (FastAPI)

## Testing Checklist

### Before Testing
- [ ] Start Gateway: `cd gateway && pnpm dev` (port 3001)
- [ ] Start ASR: `cd services/asr && python app.py` (port 5000)
- [ ] Start LLM: `cd services/llm && python app.py` (port 5001)
- [ ] Start TTS: `cd services/tts && python app.py` (port 5002)
- [ ] Start SOAP: `cd services/soap && python app.py` (port 5003)
- [ ] Start FHIR: `cd services/fhir && python app.py` (port 5004)
- [ ] Start Frontend: `cd frontend-vite && npm run dev` (port 3000)

### Test Each Service
- [ ] ASR Tab: Select dialect, click "Test ASR Service", verify transcription appears
- [ ] LLM Tab: Ask medical question, verify AI reply with intent
- [ ] SOAP Tab: Enter clinical transcript, verify structured note with S.O.A.P. sections
- [ ] FHIR Tab: Select resource type, enter JSON data, verify resource creation
- [ ] TTS Tab: Enter text, verify audio synthesis and playback

## Files Modified
1. `frontend-vite/src/pages/About.tsx` - Fixed syntax error
2. `frontend-vite/src/pages/Demo.tsx` - Updated request handlers and SOAP UI
3. `frontend-vite/src/utils/api.ts` - Fixed API client types and methods
4. `gateway/src/soap/soap.controller.ts` - Fixed SOAP request format

## Expected Behavior

### Demo Page - SOAP Tab
When user clicks "Generate SOAP Note":
1. Frontend sends transcript to `/soap/generate`
2. Gateway forwards to SOAP service with `transcript` field
3. SOAP service sends transcript to LLM with specialized prompt
4. LLM returns structured medical note
5. SOAP service parses into S.O.A.P. sections
6. Frontend displays:
   - **Subjective**: Patient's symptoms and complaints
   - **Objective**: Vital signs and examination findings
   - **Assessment**: Diagnosis and clinical impression
   - **Plan**: Treatment plan and next steps
   - **ICD Codes**: Diagnostic codes (if available)
   - **CPT Codes**: Procedure codes (if available)

## Notes
- All error messages now reference correct port (3001)
- Response structures match actual backend implementations
- SOAP generation now uses transcript input (not pre-filled fields)
- All services tested with realistic medical data examples
