#!/usr/bin/env python3
"""
Test FHIR Service (Port 5004)
Tests FHIR resource creation for EHR integration
"""
import os
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

def test_fhir_write():
    """Test FHIR writeback (/write) with Composition + Observations"""
    payload = {
        "soapNote": {
            "subjective": "المريض يشكو من صداع شديد وحمى",
            "objective": "درجة الحرارة 38.5، ضغط الدم 130/80",
            "assessment": "احتمال عدوى فيروسية",
            "plan": "راحة، سوائل، متابعة بعد 48 ساعة",
            "icdCodes": ["R51"],
            "soapJson": {
                "Objective": {
                    "Clinical Examination Findings": {
                        "Vital Signs": {
                            "BP": "130/80",
                            "HR": "78",
                            "Temp": "38.5",
                            "RR": "20",
                            "SpO2": "96%",
                        }
                    }
                }
            }
        },
        "patientId": "patient-12345",
        "practitionerId": "dr-smith",
        "encounterId": "enc-001",
        "sessionId": "session-001",
    }
    
    try:
        print("\n📤 Writing SOAP note to FHIR...")
        internal_secret = os.getenv("INTERNAL_SECRET", "")
        headers = {"Content-Type": "application/json"}
        if internal_secret:
            headers["x-internal-secret"] = internal_secret
        response = requests.post(
            "http://localhost:5004/write",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200 or response.status_code == 201:
            result = response.json()
            print("\n✅ FHIR Write Response:")
            print(f"Encounter ID: {result.get('encounterId', 'N/A')}")
            print(f"DocumentReference ID: {result.get('documentReferenceId', 'N/A')}")
            print(f"Composition ID: {result.get('compositionId', 'N/A')}")
            print(f"Observation IDs: {result.get('observationIds', [])}")
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
        test_fhir_write()
    else:
        print("\n⚠️  Make sure FHIR service is running:")
        print("cd D:\\Downloads\\HealthTech\\mvp-healthtech\\services\\fhir; python app.py")
