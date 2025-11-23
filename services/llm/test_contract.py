import importlib
import sys
import types
import pytest
from fastapi.testclient import TestClient


def build_client(monkeypatch, mock_inference=None):
    """Build a TestClient with mocked model/tokenizer/inference to avoid heavy loads."""
    monkeypatch.setenv("INTERNAL_SECRET", "test-secret")

    class DummyTokenizer:
        pad_token = "<pad>"
        eos_token = "</s>"
        eos_token_id = 1

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            import torch

            return {"input_ids": torch.tensor([[0, 1]])}

        def decode(self, ids, skip_special_tokens=True):
            return "المساعد: رد تجريبي"

    class DummyModel:
        device = "cpu"

        def generate(self, **kwargs):
            import torch

            return torch.tensor([[0, 1, 2]])

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: DummyTokenizer())
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: DummyModel())

    if "app" in sys.modules:
        importlib.reload(sys.modules["app"])
    import app as appmod

    if mock_inference:
        monkeypatch.setattr(appmod, "_run_llm_inference", mock_inference)

    return TestClient(appmod.app)


def test_missing_internal_secret_rejected(monkeypatch):
    client = build_client(monkeypatch)
    resp = client.post("/chat", json={"message": "hi", "sessionId": "s1"})
    assert resp.status_code == 401


def test_invalid_body_returns_400(monkeypatch):
    client = build_client(monkeypatch, mock_inference=None)
    resp = client.post("/chat", headers={"x-internal-secret": "test-secret"}, json={"message": "", "sessionId": "s1"})
    assert resp.status_code == 400


def test_chat_happy_path_shape(monkeypatch):
    async def fake_infer(prompt: str, max_new_tokens: int = 192):
        return {"decoded": "المساعد: رد تجريبي", "input_len": 1, "output_len": 4}

    client = build_client(monkeypatch, mock_inference=fake_infer)
    resp = client.post(
        "/chat",
        headers={"x-internal-secret": "test-secret"},
        json={"message": "hello", "sessionId": "s1", "history": []},
    )
    assert resp.status_code == 200
    data = resp.json()
    for key in ["reply", "intent", "tokens_generated", "first_token_ms", "total_latency_ms"]:
        assert key in data
    assert isinstance(data["reply"], str)


def test_chat_inference_error_500(monkeypatch):
    async def boom(prompt: str, max_new_tokens: int = 192):
        raise Exception("boom")

    client = build_client(monkeypatch, mock_inference=boom)
    resp = client.post(
        "/chat",
        headers={"x-internal-secret": "test-secret"},
        json={"message": "hello", "sessionId": "s1"},
    )
    assert resp.status_code == 500
    assert "LLM" in resp.json().get("detail", "")


def test_health(monkeypatch):
    client = build_client(monkeypatch)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body and "model" in body
