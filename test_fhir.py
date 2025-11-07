#!/usr/bin/env python3
"""
Test FHIR Service (Port 5004)
Tests FHIR resource creation for EHR integration
"""
import requests
import json

def test_fhir_health():
    """Test if FHIR service is running"""
    try:
        response = requests.get("http://localhost:5004/health")
        if response.status_code == 200:
            print("✅ FHIR service is running!")
            return True
        else:
            print(f"❌ FHIR health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to FHIR service: {e}")
        return False

def test_fhir_create_encounter():
    """Test FHIR Encounter resource creation"""
    payload = {
        "patient_id": "patient-12345",
        "encounter_id": "enc-001",
        "soap_note": {
            "subjective": "المريض يشكو من صداع شديد وحمى",
            "objective": "درجة الحرارة 38.5، ضغط الدم 130/80",
            "assessment": "احتمال عدوى فيروسية",
            "plan": "راحة، سوائل، متابعة بعد 48 ساعة"
        },
        "practitioner_id": "dr-smith",
        "date": "2025-10-28T12:00:00Z"
    }
    
    try:
        print("\n📤 Creating FHIR Encounter resource...")
        response = requests.post(
            "http://localhost:5004/create-encounter",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200 or response.status_code == 201:
            result = response.json()
            print("\n✅ FHIR Resource Created:")
            print(f"Resource Type: {result.get('resourceType', 'N/A')}")
            print(f"Encounter ID: {result.get('id', 'N/A')}")
            print(f"Status: {result.get('status', 'N/A')}")
            print(f"\nFull FHIR JSON:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"\n❌ FHIR creation failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing FHIR Service (EHR Integration)")
    print("=" * 60)
    
    if test_fhir_health():
        test_fhir_create_encounter()
    else:
        print("\n⚠️  Make sure FHIR service is running:")
        print("cd D:\\Downloads\\HealthTech\\mvp-healthtech\\services\\fhir; python app.py")
