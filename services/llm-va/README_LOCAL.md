# Local VA LLM (no Docker)

1) Python env
```
cd services/llm-va
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2) Env
```
cp .env.local.example .env.local
```
Keep `VA_DEVICE=cpu` and `VA_DTYPE=float32` for low-VRAM GPUs/CPU.

3) Run (reload)
```
uvicorn app:app --host 0.0.0.0 --port 5007 --reload
```

Health: http://localhost:5007/health  
Chat: POST /chat with `x-internal-secret` header. See LOCAL_DEV.md for curl examples.
