# services/llm/main.py - TODO: migrate to app.py
from fastapi import FastAPI
app = FastAPI()

@app.get('/health')
def health():
    return {'ok': True, 'svc': 'llm'}
