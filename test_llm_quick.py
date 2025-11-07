#!/usr/bin/env python3
"""
Quick LLM Service Test
Tests all LLM endpoints to catch errors before running full pipeline
"""
import requests
import json

LLM_URL = "http://localhost:5001"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)

    try:
        response = requests.get(f"{LLM_URL}/health", timeout=None)
        response.raise_for_status()
        print("✅ Health check passed!")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_correct_transcription():
    """Test transcription correction endpoint"""
    print("\n" + "="*80)
    print("TEST 2: Transcription Correction")
    print("="*80)

    # Simple Arabic text to correct
    test_text = "السلام عليكم يا دكتور. عندي الم في الصدر."

    try:
        response = requests.post(
            f"{LLM_URL}/correct-transcription",
            json={
                "text": test_text,
                "dialect": "egypt",
                "context": "medical"
            },
            timeout=None  # No timeout - let it run as long as needed
        )
        response.raise_for_status()
        result = response.json()

        print("✅ Transcription correction passed!")
        print(f"   Original: {result['original']}")
        print(f"   Corrected: {result['corrected']}")
        print(f"   Corrections made: {result['corrections_made']}")
        return True

    except Exception as e:
        print(f"❌ Transcription correction failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Error detail: {error_detail}")
            except:
                print(f"   Response text: {e.response.text}")
        return False


def test_infer():
    """Test LLM inference endpoint"""
    print("\n" + "="*80)
    print("TEST 3: LLM Inference")
    print("="*80)

    try:
        response = requests.post(
            f"{LLM_URL}/infer",
            json={
                "message": "What are the symptoms of diabetes?",
                "sessionId": "test-session-123",
                "intent": "symptom"
            },
            timeout=None  # No timeout - let it run as long as needed
        )
        response.raise_for_status()
        result = response.json()

        print("✅ LLM inference passed!")
        print(f"   Intent: {result['intent']}")
        print(f"   Reply: {result['reply'][:100]}...")  # First 100 chars
        return True

    except Exception as e:
        print(f"❌ LLM inference failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Error detail: {error_detail}")
            except:
                print(f"   Response text: {e.response.text}")
        return False


def test_identify_speakers():
    """Test speaker role identification endpoint"""
    print("\n" + "="*80)
    print("TEST 4: Speaker Role Identification")
    print("="*80)

    segments = [
        {
            "speaker": "SPEAKER_00",
            "text": "Good morning. What brings you in today?",
            "start": 0.0,
            "end": 3.0
        },
        {
            "speaker": "SPEAKER_01",
            "text": "I have chest pain.",
            "start": 3.5,
            "end": 5.0
        }
    ]

    try:
        response = requests.post(
            f"{LLM_URL}/identify-speakers",
            json={
                "segments": segments,
                "context": "medical"
            },
            timeout=None  # No timeout - let it run as long as needed
        )
        response.raise_for_status()
        result = response.json()

        print("✅ Speaker identification passed!")
        print(f"   Primary Doctor: {result.get('primary_doctor')}")
        print(f"   Primary Patient: {result.get('primary_patient')}")
        for role in result.get('roles', []):
            print(f"   - {role['speaker_id']}: {role['role']} ({role['confidence']:.0%})")
        return True

    except Exception as e:
        print(f"❌ Speaker identification failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Error detail: {error_detail}")
            except:
                print(f"   Response text: {e.response.text}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("LLM SERVICE QUICK TEST")
    print("Testing all endpoints to catch errors early")
    print("="*80)

    # Check if service is running
    try:
        response = requests.get(f"{LLM_URL}/health", timeout=None)
        print(f"\n✅ LLM Service is running at {LLM_URL}")
    except:
        print(f"\n❌ LLM Service not running at {LLM_URL}")
        print("Please start the LLM service first:")
        print("  cd services/llm")
        print("  python app.py")
        exit(1)

    # Run all tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Transcription Correction", test_correct_transcription()))
    results.append(("LLM Inference", test_infer()))
    results.append(("Speaker Identification", test_identify_speakers()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! LLM service is working correctly.")
        print("You can now safely run the full pipeline:")
        print("  python test_full_pipeline.py test1.mp3 egypt")
    else:
        print("\n⚠️  Some tests failed. Please fix the errors before running the full pipeline.")
        exit(1)
