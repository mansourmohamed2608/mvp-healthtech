#!/usr/bin/env python3
"""
Contract test for FHIR service /write endpoint.
Verifies idempotency header handling and auth passthrough with PHI-safe logging.
"""
import os
from fastapi.testclient import TestClient
from types import SimpleNamespace

# Set env before import
os.environ.setdefault("INTERNAL_SECRET", "test_secret")
os.environ.setdefault("FHIR_BASE_URL", "http://fhir-backend")
os.environ.setdefault("FHIR_BEARER_TOKEN", "bearer_token")

import app as fhir_app  # type: ignore
from app import app  # type: ignore


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 201

    def json(self):
        return self._payload


class FakeHTTPX:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url):
        return SimpleNamespace(status_code=200)

    async def post(self, url, json=None, headers=None):
        # Capture headers for assertion
        FakeHTTPX.last_headers = headers
        if "Encounter" in url:
            return FakeResponse({"id": "enc1"})
        return FakeResponse({"id": "doc1"})


def run_tests():
    # Inject fake httpx client
    fhir_app.httpx.AsyncClient = FakeHTTPX  # type: ignore

    client = TestClient(app)
    unauth = client.post("/write", json={"soapNote": {}, "patientId": "p1", "practitionerId": "c1", "sessionId": "s1"})
    assert unauth.status_code == 401

    payload = {
        "soapNote": {"subjective": "s", "objective": "o", "assessment": "a", "plan": "p"},
        "patientId": "p1",
        "practitionerId": "c1",
        "sessionId": "s1",
    }
    res = client.post(
        "/write",
        json=payload,
        headers={"x-internal-secret": "test_secret", "Idempotency-Key": "note:s1:p1:c1"},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["success"] is True
    # Ensure idempotency and auth headers were sent downstream
    assert "Idempotency-Key" in FakeHTTPX.last_headers
    assert "Authorization" in FakeHTTPX.last_headers
    print("✓ FHIR contract OK")


if __name__ == "__main__":
    run_tests()
