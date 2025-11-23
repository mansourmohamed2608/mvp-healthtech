import base64
import importlib
import sys
import pytest
from fastapi.testclient import TestClient


def build_client(monkeypatch, engine_bytes=b"\x00\x01"):
    monkeypatch.setenv("INTERNAL_SECRET", "test-secret")

    # Reload module to apply env
    if "app" in sys.modules:
        importlib.reload(sys.modules["app"])
    import app as appmod

    async def fake_run_tts(text: str, voice: str | None):
        return engine_bytes

    monkeypatch.setattr(appmod, "_run_tts_engine", fake_run_tts)
    return TestClient(appmod.app)


def test_missing_internal_secret_rejected(monkeypatch):
    # Do not set header
    client = build_client(monkeypatch)
    resp = client.post("/synthesize", json={"text": "hello"})
    assert resp.status_code == 401


def test_invalid_payload_400(monkeypatch):
    client = build_client(monkeypatch)
    resp = client.post("/synthesize", headers={"x-internal-secret": "test-secret"}, json={"text": ""})
    assert resp.status_code == 400


def test_synthesize_happy_path(monkeypatch):
    client = build_client(monkeypatch, engine_bytes=b"\x00\x00" * 100)
    resp = client.post(
        "/synthesize",
        headers={"x-internal-secret": "test-secret"},
        json={"text": "مرحبا", "sessionId": "s1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "mulaw"
    assert data["sampleRate"] == 8000
    audio = base64.b64decode(data["audio"])
    assert len(audio) > 0


def test_tts_error_500(monkeypatch):
    async def boom(text: str, voice: str | None):
        raise Exception("boom")

    monkeypatch.setenv("INTERNAL_SECRET", "test-secret")
    if "app" in sys.modules:
        importlib.reload(sys.modules["app"])
    import app as appmod
    monkeypatch.setattr(appmod, "_run_tts_engine", boom)
    client = TestClient(appmod.app)

    resp = client.post(
        "/synthesize",
        headers={"x-internal-secret": "test-secret"},
        json={"text": "hello"},
    )
    assert resp.status_code in (500, 504)
    assert "synthesis" in resp.json().get("detail", "").lower() or "service unavailable" in resp.json().get("detail", "").lower()


def test_health(monkeypatch):
    client = build_client(monkeypatch)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "service" in body and "model" in body
