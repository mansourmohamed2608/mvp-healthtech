#!/usr/bin/env python3
"""
Contract test for SOAP service:
- /generate creates a note (persisted in fake pool)
- /notes/{id}/approve updates status
FHIR call is mocked via httpx.AsyncClient.
"""
import os
import asyncio
from fastapi.testclient import TestClient
from types import SimpleNamespace

# Ensure middleware secrets are present before importing the app
os.environ.setdefault("INTERNAL_SECRET", "test_secret")
os.environ.setdefault("LLM_SERVICE_URL", "http://llm")  # mocked

from app import app, _pool  # type: ignore
import app as soap_app


class FakeConn:
    def __init__(self, store):
        self.store = store

    async def fetchrow(self, query, *params):
        # INSERT
        if query.strip().lower().startswith("insert"):
            note_id = f"note{len(self.store)+1}"
            row = {
                "id": note_id,
                "session_id": params[0],
                "patient_id": params[1],
                "clinician_id": params[2],
                "status": params[3],
                "raw_transcript": params[4],
                "soap_json": params[5],
                "subjective": params[6],
                "objective": params[7],
                "assessment": params[8],
                "plan": params[9],
                "icd_codes": params[10],
                "cpt_codes": params[11],
                "created_at": "now",
                "updated_at": "now",
            }
            self.store[note_id] = row
            return row
        # UPDATE
        if query.lower().startswith("update"):
            note_id = params[0]
            status = params[1]
            row = self.store.get(note_id)
            if not row:
                return None
            row["status"] = status
            self.store[note_id] = row
            return row
        # SELECT WHERE
        if "where id" in query.lower():
            note_id = params[0]
            return self.store.get(note_id)
        return None

    async def fetch(self, query, *params):
        # simple filter on status/clinician
        rows = list(self.store.values())
        if "where" in query.lower():
            if "status" in query.lower():
                want = params[0]
                rows = [r for r in rows if r["status"] == want]
            if "clinician_id" in query.lower():
                want = params[-1]
                rows = [r for r in rows if r["clinician_id"] == want]
        return rows


class FakeAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self):
        self.store = {}
        self.acquire_conn = FakeConn(self.store)

    def acquire(self):
        return FakeAcquire(self.acquire_conn)

    async def close(self):
        return


class FakeHTTPXClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None, headers=None):
        # mimic llm infer returning reply
        if "infer" in url:
            return SimpleNamespace(status_code=200, json=lambda: {"reply": "subjective\nobjective\nassessment\nplan"})
        # default success
        return SimpleNamespace(status_code=200, json=lambda: {"ok": True})


def setup_fakes():
    # Inject fake pool and httpx client before TestClient runs startup
    soap_app._pool = FakePool()
    soap_app.httpx.AsyncClient = FakeHTTPXClient  # type: ignore


def run_tests():
    setup_fakes()
    client = TestClient(app)

    # Generate
    resp = client.post(
        "/generate",
        json={"transcript": "t", "sessionId": "s1", "patientId": "p1", "clinicianId": "c1"},
        headers={"x-internal-secret": "test_secret"},
    )
    assert resp.status_code == 200, resp.text
    note = resp.json()
    assert note["status"] == "pending"
    note_id = note["id"]

    # Approve
    approve = client.patch(f"/notes/{note_id}/approve", headers={"x-internal-secret": "test_secret"})
    assert approve.status_code == 200
    assert approve.json()["status"] == "approved"

    # List
    notes = client.get("/notes", headers={"x-internal-secret": "test_secret"})
    assert notes.status_code == 200
    assert len(notes.json()["notes"]) >= 1

    print("✓ SOAP contract OK")


if __name__ == "__main__":
    run_tests()
