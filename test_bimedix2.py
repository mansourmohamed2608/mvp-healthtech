# test_bimedix2.py
"""
Quick test script for BiMediX2-8B + post-processing modules
Run this after starting the LLM service to verify everything works
"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_health():
    """Test if LLM service is running"""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ LLM service is running")
            return True
        else:
            print(f"❌ Service returned {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Service not reachable: {e}")
        print("💡 Start the service: cd services/llm && python app.py")
        return False

def test_correction():
    """Test Arabic medical correction endpoint"""
    print("\n" + "=" * 60)
    print("TEST 2: Arabic Medical Correction")
    print("=" * 60)
    
    test_cases = [
        {
            "text": "المريض يشكو من الام في البروستاتا",
            "dialect": "egypt",
            "expected_fixes": ["الام→ألم", "البروستاتا→البروستات"]
        },
        {
            "text": "عنده حمى وسعال وضغط الدم 120 على 80",
            "dialect": "gulf",
            "expected_fixes": ["120 على 80→120/80 mmHg"]
        },
        {
            "text": "الدكتور وصف مضاد حيوي ومسكن للوجع",
            "dialect": "egypt",
            "expected_fixes": ["وجع→ألم"]
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Input:    {test['text']}")
        print(f"  Dialect:  {test['dialect']}")
        print(f"  Expected: {', '.join(test['expected_fixes'])}")
        
        try:
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/correct-transcription",
                json={"text": test['text'], "dialect": test['dialect']},
                timeout=1800  # 30 minutes (CPU inference is slow)
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                print(f"  Output:   {data['corrected']}")
                print(f"  Changes:  {data['corrections_made']} corrections")
                print(f"  Time:     {elapsed:.1f}s")
                print(f"  ✅ SUCCESS")
            else:
                print(f"  ❌ FAILED: {response.status_code}")
                print(f"  Error: {response.text}")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")

def test_soap_generation():
    """Test SOAP note generation endpoint"""
    print("\n" + "=" * 60)
    print("TEST 3: SOAP Note Generation")
    print("=" * 60)
    
    test_message = "مريض عمره 45 سنة يشكو من صداع شديد وحمى منذ يومين. ضغط الدم 140 على 90"
    
    print(f"Input: {test_message}")
    
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/infer",
            json={
                "message": test_message,
                "sessionId": "test-123",
                "intent": "symptom"
            },
            timeout=1800
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nIntent: {data['intent']}")
            print(f"\nSOAP Note:\n{data['reply']}")
            print(f"\nTime: {elapsed:.1f}s")
            print(f"✅ SUCCESS")
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_speaker_identification():
    """Test speaker role identification endpoint"""
    print("\n" + "=" * 60)
    print("TEST 4: Speaker Role Identification")
    print("=" * 60)
    
    segments = [
        {"speaker": "SPEAKER_00", "text": "أهلاً، ما الذي يؤلمك اليوم؟", "start": 0.0, "end": 2.5},
        {"speaker": "SPEAKER_01", "text": "عندي صداع شديد منذ يومين", "start": 2.5, "end": 5.0},
        {"speaker": "SPEAKER_00", "text": "دعني أفحص ضغط الدم. هل تأخذ أدوية؟", "start": 5.0, "end": 8.0},
        {"speaker": "SPEAKER_01", "text": "لا، مافيش أدوية", "start": 8.0, "end": 9.5},
        {"speaker": "SPEAKER_00", "text": "سأصف لك مسكن، خذ حبة كل 6 ساعات", "start": 9.5, "end": 13.0},
        {"speaker": "SPEAKER_01", "text": "تمام، شكراً يا دكتور", "start": 13.0, "end": 15.0}
    ]
    
    print("Conversation:")
    for seg in segments:
        print(f"  {seg['speaker']}: {seg['text']}")
    
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/identify-speakers",
            json={"segments": segments, "context": "medical"},
            timeout=1800
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nIdentified Roles:")
            for role in data['roles']:
                print(f"  {role['speaker_id']}: {role['role']} ({role['confidence']:.2f})")
                print(f"    Reasoning: {role['reasoning']}")
            
            print(f"\nPrimary Doctor: {data.get('primary_doctor')}")
            print(f"Primary Patient: {data.get('primary_patient')}")
            print(f"\nTime: {elapsed:.1f}s")
            print(f"✅ SUCCESS")
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def main():
    """Run all tests"""
    print("\n" + "🧪 " * 20)
    print("BiMediX2-8B + Post-Processing Test Suite")
    print("🧪 " * 20 + "\n")
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Service not running. Please start it first:")
        print("   cd d:\\Downloads\\HealthTech\\mvp-healthtech\\services\\llm")
        print("   python app.py")
        return
    
    time.sleep(1)
    
    # Test 2: Correction endpoint
    test_correction()
    
    time.sleep(2)
    
    # Test 3: SOAP generation
    # test_soap_generation()  # Comment out if too slow
    
    # time.sleep(2)
    
    # Test 4: Speaker identification
    # test_speaker_identification()  # Comment out if too slow
    
    print("\n" + "=" * 60)
    print("🎉 Test Suite Complete!")
    print("=" * 60)
    print("\n💡 Tips:")
    print("  - First run downloads BiMediX2-8B (~8GB, 5-10 mins)")
    print("  - CPU inference takes ~20-30 mins per request")
    print("  - Uncomment SOAP/Speaker tests after model loads")
    print("  - Use Kaggle T4 for faster testing (free GPU)")

if __name__ == "__main__":
    main()
