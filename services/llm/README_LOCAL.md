# Local Orchestrator (no Docker)

1) Python env
```
cd services/llm
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2) Env
```
cp .env.orchestrator.local.example .env.local
```
Edit if needed for localhost URLs and INTERNAL_SECRET.

3) Run (reload)
```
uvicorn orchestrator:app --host 0.0.0.0 --port 5006 --reload
```

Expected ports (default): orchestrator 5006. Ensure llm-va is running on localhost:5007 and clinical LLM on 5001 if you want both routes to work.
