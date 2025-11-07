#!/usr/bin/env python3
"""
Simple LLM Health Check
Just checks if the service is running - no slow inference tests
"""
import requests

LLM_URL = "http://localhost:5001"

print("\n" + "="*80)
print("LLM SERVICE HEALTH CHECK")
print("="*80)

try:
    response = requests.get(f"{LLM_URL}/health", timeout=None)
    response.raise_for_status()
    print(f"\n✅ LLM Service is running at {LLM_URL}")
    print(f"   Status: {response.json()}")
    print("\n🎉 Service is healthy! You can run the full pipeline now:")
    print("   python test_full_pipeline.py test1.mp3 egypt")
    print("\n⚠️  Note: LLM inference on CPU is VERY slow (1-2 min per request)")
    print("   The full pipeline will take a long time. Be patient!")
except Exception as e:
    print(f"\n❌ LLM Service not running: {e}")
    print("\nPlease start the LLM service:")
    print("  cd services/llm")
    print("  python app.py")
    exit(1)
