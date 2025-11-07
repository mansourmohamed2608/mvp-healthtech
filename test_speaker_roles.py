#!/usr/bin/env python3
"""
Test script for LLM Speaker Role Detection
Tests the /identify-speakers endpoint with various conversation patterns
"""
import requests
import json

LLM_URL = "http://localhost:5001"

def test_typical_consultation():
    """Test typical doctor-patient consultation"""
    print("\n" + "="*80)
    print("TEST 1: Typical Medical Consultation")
    print("="*80)

    segments = [
        {
            "speaker": "SPEAKER_00",
            "text": "Good morning. What brings you in today? Tell me about your symptoms.",
            "start": 0.0,
            "end": 5.0
        },
        {
            "speaker": "SPEAKER_01",
            "text": "I've been having chest pain for the past two days. It hurts when I breathe deeply.",
            "start": 5.5,
            "end": 12.0
        },
        {
            "speaker": "SPEAKER_00",
            "text": "Let me examine you. I'll check your blood pressure and heart rate first.",
            "start": 12.5,
            "end": 17.0
        },
        {
            "speaker": "SPEAKER_01",
            "text": "Okay, thank you doctor.",
            "start": 17.5,
            "end": 19.0
        },
        {
            "speaker": "SPEAKER_00",
            "text": "Your blood pressure is 140 over 90, slightly elevated. I'll prescribe you medication.",
            "start": 19.5,
            "end": 25.0
        }
    ]

    try:
        response = requests.post(
            f"{LLM_URL}/identify-speakers",
            json={"segments": segments, "context": "medical"},
            timeout=None
        )
        response.raise_for_status()
        result = response.json()

        print("\n✅ Role Detection Complete!")
        print(f"\nPrimary Doctor: {result['primary_doctor']}")
        print(f"Primary Patient: {result['primary_patient']}\n")

        for role in result['roles']:
            print(f"🎭 {role['speaker_id']}: {role['role']}")
            print(f"   Confidence: {role['confidence']:.0%}")
            print(f"   Reasoning: {role['reasoning']}\n")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_patient_speaks_first():
    """Test when patient initiates conversation"""
    print("\n" + "="*80)
    print("TEST 2: Patient Speaks First (Emergency)")
    print("="*80)

    segments = [
        {
            "speaker": "SPEAKER_00",
            "text": "Help! I'm having severe chest pain and shortness of breath!",
            "start": 0.0,
            "end": 4.0
        },
        {
            "speaker": "SPEAKER_01",
            "text": "Calm down. Sit down here. When did the pain start? Any radiation to your arm?",
            "start": 4.5,
            "end": 9.0
        },
        {
            "speaker": "SPEAKER_00",
            "text": "About 20 minutes ago. Yes, it's going down my left arm.",
            "start": 9.5,
            "end": 13.0
        },
        {
            "speaker": "SPEAKER_01",
            "text": "I'm calling emergency services. This could be a heart attack. Take aspirin immediately.",
            "start": 13.5,
            "end": 18.0
        }
    ]

    try:
        response = requests.post(
            f"{LLM_URL}/identify-speakers",
            json={"segments": segments, "context": "medical"},
            timeout=None
        )
        response.raise_for_status()
        result = response.json()

        print("\n✅ Role Detection Complete!")
        print(f"\nPrimary Doctor: {result['primary_doctor']}")
        print(f"Primary Patient: {result['primary_patient']}\n")

        for role in result['roles']:
            print(f"🎭 {role['speaker_id']}: {role['role']}")
            print(f"   Confidence: {role['confidence']:.0%}")
            print(f"   Reasoning: {role['reasoning']}\n")

        # Verify correct detection
        if result['primary_patient'] == 'SPEAKER_00' and result['primary_doctor'] == 'SPEAKER_01':
            print("✅ Correctly identified patient speaking first!\n")
            return True
        else:
            print("⚠️ Incorrect role assignment!\n")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_arabic_conversation():
    """Test with Arabic medical conversation"""
    print("\n" + "="*80)
    print("TEST 3: Arabic Medical Conversation")
    print("="*80)

    segments = [
        {
            "speaker": "SPEAKER_00",
            "text": "صباح الخير. ما الذي جاء بك اليوم؟",
            "start": 0.0,
            "end": 3.0
        },
        {
            "speaker": "SPEAKER_01",
            "text": "دكتور، عندي ألم في الصدر منذ يومين. الألم يزيد لما أتنفس بعمق.",
            "start": 3.5,
            "end": 9.0
        },
        {
            "speaker": "SPEAKER_00",
            "text": "خليني أفحصك. هقيس ضغط الدم ومعدل نبضات القلب الأول.",
            "start": 9.5,
            "end": 14.0
        },
        {
            "speaker": "SPEAKER_01",
            "text": "حاضر يا دكتور.",
            "start": 14.5,
            "end": 16.0
        },
        {
            "speaker": "SPEAKER_00",
            "text": "ضغط الدم مرتفع شوية. هاكتبلك علاج ينظم الضغط.",
            "start": 16.5,
            "end": 21.0
        }
    ]

    try:
        response = requests.post(
            f"{LLM_URL}/identify-speakers",
            json={"segments": segments, "context": "medical"},
            timeout=None
        )
        response.raise_for_status()
        result = response.json()

        print("\n✅ Role Detection Complete!")
        print(f"\nPrimary Doctor: {result['primary_doctor']}")
        print(f"Primary Patient: {result['primary_patient']}\n")

        for role in result['roles']:
            print(f"🎭 {role['speaker_id']}: {role['role']}")
            print(f"   Confidence: {role['confidence']:.0%}")
            print(f"   Reasoning: {role['reasoning']}\n")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_multi_speaker():
    """Test conversation with nurse present"""
    print("\n" + "="*80)
    print("TEST 4: Multi-Speaker (Doctor, Patient, Nurse)")
    print("="*80)

    segments = [
        {
            "speaker": "SPEAKER_00",
            "text": "Good morning. I understand you have concerns about your blood sugar levels.",
            "start": 0.0,
            "end": 4.0
        },
        {
            "speaker": "SPEAKER_01",
            "text": "Yes doctor, I've been feeling dizzy and very thirsty lately.",
            "start": 4.5,
            "end": 8.0
        },
        {
            "speaker": "SPEAKER_02",
            "text": "Doctor, I've taken the patient's vitals. Blood pressure 130/85, temperature 98.6°F.",
            "start": 8.5,
            "end": 14.0
        },
        {
            "speaker": "SPEAKER_00",
            "text": "Thank you nurse. Let's run a glucose test and HbA1c.",
            "start": 14.5,
            "end": 18.0
        },
        {
            "speaker": "SPEAKER_01",
            "text": "Will I need insulin?",
            "start": 18.5,
            "end": 20.0
        },
        {
            "speaker": "SPEAKER_00",
            "text": "Let's see the results first. We'll discuss treatment options afterward.",
            "start": 20.5,
            "end": 24.0
        }
    ]

    try:
        response = requests.post(
            f"{LLM_URL}/identify-speakers",
            json={"segments": segments, "context": "medical"},
            timeout=None
        )
        response.raise_for_status()
        result = response.json()

        print("\n✅ Role Detection Complete!")
        print(f"\nPrimary Doctor: {result['primary_doctor']}")
        print(f"Primary Patient: {result['primary_patient']}\n")

        for role in result['roles']:
            print(f"🎭 {role['speaker_id']}: {role['role']}")
            print(f"   Confidence: {role['confidence']:.0%}")
            print(f"   Reasoning: {role['reasoning']}\n")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SPEAKER ROLE DETECTION TEST SUITE")
    print("Testing LLM semantic analysis of conversation roles")
    print("="*80)

    # Check service health
    try:
        response = requests.get(f"{LLM_URL}/health", timeout=5)
        print(f"\n✅ LLM Service is running at {LLM_URL}")
    except:
        print(f"\n❌ LLM Service not running at {LLM_URL}")
        print("Please start the LLM service first:")
        print("cd services/llm && python app.py")
        exit(1)

    # Run tests
    results = []
    results.append(("Typical Consultation", test_typical_consultation()))
    results.append(("Patient Speaks First", test_patient_speaks_first()))
    results.append(("Arabic Conversation", test_arabic_conversation()))
    results.append(("Multi-Speaker", test_multi_speaker()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {passed}/{len(results)} tests passed")
