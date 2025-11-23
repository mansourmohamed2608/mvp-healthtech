"""
Lightweight load test harness for the MVP.
Uses synthetic (non-PHI) payloads to exercise the main gateway flows.
"""
import argparse
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


def fake_audio_base64() -> str:
    # 1 second of silence (wav header pre-encoded)
    return "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="


def run_once(base_url: str, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    summary = {}

    def post(path: str, payload: dict):
        start = time.time()
        resp = requests.post(f"{base_url}{path}", json=payload, headers=headers, timeout=10)
        latency = time.time() - start
        return resp.status_code, latency

    summary["asr"] = post("/asr/transcribe", {"audio": fake_audio_base64(), "callSid": "load-test"})
    summary["llm"] = post("/llm/chat", {"message": "اختبار النظام", "sessionId": "load-test"})
    summary["tts"] = post("/tts/synthesize", {"text": "مرحبا دكتور"})
    soap_status, soap_latency = post(
        "/soap/generate",
        {
            "transcript": "اختبار سريري قصير بدون محتوى حقيقي",
            "sessionId": "load-test",
            "patientId": "patient-load",
            "practitionerId": "clinician-load",
        },
    )
    summary["soap"] = (soap_status, soap_latency)

    note_id = None
    if soap_status == 200:
        try:
            resp = requests.get(f"{base_url}/soap/notes", headers=headers, timeout=5)
            if resp.ok:
                items = resp.json() or []
                if items:
                    note_id = items[-1].get("id") or items[-1].get("note_id")
        except Exception:
            note_id = None

    if note_id:
        summary["approve"] = post(f"/soap/notes/{note_id}/approve", {})
    else:
        summary["approve"] = (0, 0.0)
    return summary


def run_load(base_url: str, iterations: int, concurrency: int, token: str | None):
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_once, base_url, token) for _ in range(iterations)]
        for fut in as_completed(futures):
            results.append(fut.result())

    report = {}
    for key in ["asr", "llm", "tts", "soap", "approve"]:
        codes = [r[key][0] for r in results]
        lats = [r[key][1] for r in results]
        report[key] = {
            "count": len(codes),
            "success": sum(1 for c in codes if 200 <= c < 300),
            "min_ms": round(min(lats) * 1000, 2),
            "max_ms": round(max(lats) * 1000, 2),
            "avg_ms": round(sum(lats) / len(lats) * 1000, 2),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="MVP load tester (synthetic, non-PHI)")
    parser.add_argument("--base-url", default="http://localhost:3000", help="Gateway base URL")
    parser.add_argument("--iterations", type=int, default=5, help="Total requests to run")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent workers")
    parser.add_argument("--jwt", help="Bearer token for protected routes", default=None)
    args = parser.parse_args()

    print(f"Running load with {args.iterations} iterations (concurrency={args.concurrency}) against {args.base_url}")
    report = run_load(args.base_url, args.iterations, args.concurrency, args.jwt)
    print("=== Summary (status counts / latency ms) ===")
    for k, v in report.items():
        print(f"{k}: successes={v['success']}/{v['count']} min={v['min_ms']} avg={v['avg_ms']} max={v['max_ms']}")


if __name__ == "__main__":
    main()
