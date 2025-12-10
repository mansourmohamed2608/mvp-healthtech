# LOCAL DEVELOPMENT (no Docker) — VA path first

Targets on localhost:
- Gateway (Nest) :3001
- Orchestrator (FastAPI) :5006
- VA LLM (FastAPI/Qwen) :5007 (CPU)
- Postgres :5432 (local install or a one-off container)
- Redis :6379 (local install or a one-off container)
- Frontend (Vite) :5173 (or default Vite dev port)

## 0) Postgres + Redis
Use local services or quick containers (optional, keeps app services local):
```
docker run --name pg-local -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_USER=postgres -e POSTGRES_DB=healthtech -d postgres:16
docker run --name redis-local -p 6379:6379 -d redis:7
```

## 1) VA LLM (services/llm-va)
```
cd services/llm-va
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.local.example .env.local   # VA_DEVICE=cpu, VA_DTYPE=float32, INTERNAL_SECRET=dev-internal-secret-change-me
uvicorn app:app --host 0.0.0.0 --port 5007 --reload
```

## 2) Orchestrator (services/llm)
```
cd services/llm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.orchestrator.local.example .env.local   # INTERNAL_SECRET matches gateway/llm-va
uvicorn orchestrator:app --host 0.0.0.0 --port 5006 --reload
```

## 3) Gateway (Nest) (gateway/)
```
cd gateway
npm install
cp .env.local.example .env.local   # set DB/Redis to localhost, INTERNAL_SECRET/JWT_SECRET
npm run start:dev
```

## 4) Frontend (frontend-vite/)
```
cd frontend-vite
npm install
cp .env.local.example .env.local   # VITE_API_BASE_URL=http://localhost:3001
npm run dev
```

## 5) Manual HTTP smoke (VA only, no Twilio/ASR/TTS)

VA service direct:
```
curl -X POST http://localhost:5007/chat \
  -H "Content-Type: application/json" \
  -H "x-internal-secret: dev-internal-secret-change-me" \
  -d '{ "message": "حابب أحجز دكتور جلدية الأسبوع الجاي بعد الشغل", "history": [], "sessionId": "test-1", "mode": "voice_agent_va", "slots": {} }'
```

Orchestrator:
```
curl -X POST http://localhost:5006/orchestrate \
  -H "Content-Type: application/json" \
  -H "x-internal-secret: dev-internal-secret-change-me" \
  -d '{ "transcript": "حابب أحجز دكتور جلدية الأسبوع الجاي بعد الشغل", "sessionId": "test-1", "mode": "voice_agent_va", "slots": {} }'
```

Gateway (get a dev JWT first):
```
curl -X POST http://localhost:3001/auth/login \
  -H "Content-Type: application/json" \
  -d '{ "userId": "dev", "password": "changeme" }'
# use the returned access_token:
curl -X POST http://localhost:3001/llm/orchestrate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{ "transcript": "حابب أحجز دكتور جلدية الأسبوع الجاي بعد الشغل", "sessionId": "test-1", "mode": "voice_agent_va", "slots": {} }'
```

Ports summary (localhost):
- Gateway: 3001
- Orchestrator: 5006
- VA LLM: 5007
- Clinical LLM (if needed): 5001
- Postgres: 5432
- Redis: 6379
- Frontend (Vite): 5173 (default)
