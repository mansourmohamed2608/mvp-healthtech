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
                "template_id": params[3],
                "status": params[4],
                "raw_transcript": params[5],
                "soap_json": params[6],
                "subjective": params[7],
                "objective": params[8],
                "assessment": params[9],
                "plan": params[10],
                "icd_codes": params[11],
                "cpt_codes": params[12],
                "created_at": "now",
                "updated_at": "now",
            }
            self.store[note_id] = row
            return row
        # UPDATE
        if query.lower().startswith("update"):
            note_id = params[0]
            row = self.store.get(note_id)
            if not row:
                return None
            lower = query.lower()
            if "set status" in lower:
                status = params[1]
                row["status"] = status
            elif "set soap_json" in lower:
                row["soap_json"] = params[1]
                row["subjective"] = params[2]
                row["objective"] = params[3]
                row["assessment"] = params[4]
                row["plan"] = params[5]
                row["raw_transcript"] = params[6]
            elif "set subjective" in lower:
                row["subjective"] = params[1]
                row["objective"] = params[2]
                row["assessment"] = params[3]
                row["plan"] = params[4]
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

    async def execute(self, query, *params):
        return ""


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
        if "generate" in url:
            messages = (json or {}).get("messages", [])
            system = messages[0]["content"] if messages else ""
            if "chief_complaint" in system:
                return SimpleNamespace(
                    status_code=200,
                    json=lambda: {"text": '{"chief_complaint":"pain","hpi":"2 days","ros":""}'},
                )
            if "patient_education" in system:
                return SimpleNamespace(
                    status_code=200,
                    json=lambda: {"text": '{"instructions":["rest"],"follow_up":"in 1 week","patient_education":["hydration"]}'},
                )
            if "clinical note editor" in system:
                return SimpleNamespace(
                    status_code=200,
                    json=lambda: {"text": '{"value":"updated field"}'},
                )
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "text": (
                        "Subjective: tooth pain for 2 days.\n"
                        "Objective: mild swelling noted.\n"
                        "Assessment: dental pain.\n"
                        "Plan: analgesics and follow-up."
                    )
                },
            )
        return SimpleNamespace(status_code=200, json=lambda: {"ok": True})


def setup_fakes():
    # Inject fake pool and httpx client before TestClient runs startup
    soap_app._pool = FakePool()
    import llm_client as llm_client_module
    llm_client_module.httpx.AsyncClient = FakeHTTPXClient  # type: ignore


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

    # Update sections
    sections = client.patch(
        f"/notes/{note_id}/sections",
        json={"soapText": "Subjective: s\nObjective: o\nAssessment: a\nPlan: p"},
        headers={"x-internal-secret": "test_secret"},
    )
    assert sections.status_code == 200
    assert sections.json()["plan"] == "p"

    # Update field
    field = client.patch(
        f"/notes/{note_id}/field",
        json={"fieldPath": "Subjective.Chief Complaint", "transcript": "اضافة"},
        headers={"x-internal-secret": "test_secret"},
    )
    assert field.status_code == 200
    assert field.json()["subjective"]

    # List
    notes = client.get("/notes", headers={"x-internal-secret": "test_secret"})
    assert notes.status_code == 200
    assert len(notes.json()["notes"]) >= 1

    print("✓ SOAP contract OK")


if __name__ == "__main__":
    run_tests()
