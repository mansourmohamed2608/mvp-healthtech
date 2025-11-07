# Testing the HealthTech Frontend with Backend Services

## Quick Start

### 1. Start Backend Services

```powershell
# From the root directory
cd infra
docker-compose up -d
```

This will start:
- Gateway (NestJS) on `http://localhost:3000`
- ASR Service on `http://localhost:5000`
- LLM Service on `http://localhost:5001`
- TTS Service on `http://localhost:5002`
- SOAP Service on `http://localhost:5003`
- FHIR Service on `http://localhost:5004`

### 2. Start Frontend

```powershell
# In frontend-vite directory
pnpm dev
# or
npx vite
```

Frontend will run on `http://localhost:3000` or `http://localhost:5173`

### 3. Test All Services

Navigate to **http://localhost:3000/demo** (or 5173)

This page lets you test ALL backend services in one place:

#### Available Tests:

1. **ASR (Voice to Text)**
   - Select dialect (Egyptian, Levantine, Gulf, MSA, English)
   - Enter text to simulate transcription
   - Click "Test ASR Service"
   - View transcription result

2. **LLM (AI Assistant)**
   - Ask medical questions
   - Get AI-powered responses
   - Session tracking included

3. **SOAP Notes Generation**
   - Enter Subjective, Objective, Assessment, Plan
   - Generate structured SOAP notes
   - View formatted output

4. **FHIR Integration**
   - Create FHIR resources (Patient, Observation, Condition, etc.)
   - Test EHR integration
   - View created resource

5. **TTS (Text to Speech)**
   - Enter text
   - Generate audio
   - Play audio in browser

## Other Pages

### Features Page
**URL:** `/features`

- Overview of all platform capabilities
- Links to individual feature pages
- Stats and benefits

### Voice Transcription Page
**URL:** `/features/voice-transcription`

- Record audio directly in browser
- Real-time transcription
- Multi-dialect support
- Download transcripts

### Clinical Notes
**URL:** `/features/clinical-notes`

- View and manage clinical notes
- Search and filter
- Export capabilities

### SOAP Generation
**URL:** `/features/soap-generation`

- Generate SOAP notes from transcripts
- Edit and refine
- Save to database

### FHIR Integration
**URL:** `/features/fhir-integration`

- Connect to EHR systems
- View FHIR resources
- Create/Read/Update/Delete operations

### Dashboard
**URL:** `/dashboard`

- Analytics and metrics
- Patient statistics
- Performance insights

## API Configuration

The frontend connects to the backend via the API client in `src/utils/api.ts`.

Default API URL: `http://localhost:3000` (Gateway)

To change this, create or edit `.env`:

```env
VITE_API_URL=http://your-backend-url:port
```

## Troubleshooting

### Backend Not Responding

```powershell
# Check if services are running
docker ps

# View logs
docker-compose logs gateway
docker-compose logs asr
docker-compose logs llm
# etc...

# Restart services
docker-compose restart
```

### CORS Errors

Make sure your gateway (NestJS) has CORS enabled for `localhost:3000` or `localhost:5173`.

In `gateway/src/main.ts`, ensure:

```typescript
app.enableCors({
  origin: ['http://localhost:3000', 'http://localhost:5173'],
  credentials: true,
});
```

### Port Conflicts

If ports 3000 or 5173 are in use:

```powershell
# Frontend - use different port
vite --port 3001

# Backend - modify docker-compose.yml ports
```

## Demo Workflow

1. Go to `/demo`
2. Select "Voice to Text" tab
3. Choose dialect → Enter text → Click Test
4. Switch to "LLM" tab
5. Ask a question → Click Test
6. Switch to "SOAP Notes" tab
7. Fill SOAP fields → Generate
8. Switch to "FHIR" tab
9. Select resource type → Click Create
10. Switch to "TTS" tab
11. Enter text → Synthesize → Listen

## Features Overview

| Feature | Page | Backend Service | Status |
|---------|------|----------------|--------|
| Voice Transcription | `/features/voice-transcription` | ASR (port 5000) | ✅ |
| AI Assistant | `/dashboard` | LLM (port 5001) | ✅ |
| SOAP Notes | `/features/soap-generation` | SOAP (port 5003) | ✅ |
| FHIR Integration | `/features/fhir-integration` | FHIR (port 5004) | ✅ |
| Text-to-Speech | `/demo` (TTS tab) | TTS (port 5002) | ✅ |
| Clinical Notes | `/features/clinical-notes` | Gateway /clinical | ✅ |

## Development Tips

### Adding New API Endpoints

Edit `src/utils/api.ts`:

```typescript
async myNewEndpoint(data: any) {
  return this.request<ResponseType>('/my-endpoint', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}
```

### Testing with Mock Data

The Demo page uses mock data when services aren't available. Check the console for detailed error messages.

### Hot Reload

Both frontend (Vite) and backend (NestJS with `start:dev`) support hot reload. Changes will reflect automatically.

## Next Steps

1. **Test each service individually** via the Demo page
2. **Try the dedicated feature pages** for full workflows
3. **Check browser console** for detailed API requests/responses
4. **View Docker logs** if services fail

## Support

- Frontend runs on Vite + React + TypeScript + Tailwind
- Backend uses NestJS gateway + Python microservices
- All services communicate through the gateway on port 3000

Happy testing! 🚀
