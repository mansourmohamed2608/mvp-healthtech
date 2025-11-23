#!/usr/bin/env python3
import os
import base64
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("INTERNAL_SECRET", "test_secret")

import services.asr.app as asr_app  # noqa: E402

client = TestClient(asr_app.app)


def fake_audio_b64():
    # Very small silent wav header base64 (RIFF)
    return base64.b64encode(b"RIFF....data").decode()


def test_missing_internal_secret_returns_401():
    resp = client.post("/transcribe", json={"audio": fake_audio_b64()})
    assert resp.status_code in (401, 403)


def test_invalid_audio_returns_400():
    resp = client.post("/transcribe", headers={"x-internal-secret": "test_secret"}, json={"audio": "!!notb64"})
    assert resp.status_code == 400
    assert "Invalid audio payload" in resp.text or "Invalid audio payload" in resp.text


def test_happy_path_shape(monkeypatch):
    def fake_core(req):
        return {
            "text": "hello",
            "segments": [{"text": "hello", "start": 0.0, "end": 1.0, "speaker": None}],
            "language": "en",
            "duration": 1.0,
            "processing_time": 0.1,
            "rtf": 0.1,
            "speakers": [],
            "model_used": "fake-model",
            "pipeline_mode": "diarize-last",
        }

    monkeypatch.setattr(asr_app, "_transcription_core", fake_core)
    resp = client.post("/transcribe", headers={"x-internal-secret": "test_secret"}, json={"audio": fake_audio_b64()})
    assert resp.status_code == 200
    body = resp.json()
    for key in ["text", "segments", "language", "duration", "processing_time", "rtf", "model_used", "pipeline_mode"]:
        assert key in body
    assert body["segments"][0]["text"] == "hello"


def test_error_path_returns_phi_safe(monkeypatch):
    def fake_core(req):
        raise RuntimeError("boom")

    monkeypatch.setattr(asr_app, "_transcription_core", fake_core)
    resp = client.post("/transcribe", headers={"x-internal-secret": "test_secret"}, json={"audio": fake_audio_b64()})
    assert resp.status_code == 500
    assert "ASR service unavailable" in resp.text
