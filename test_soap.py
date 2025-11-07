#!/usr/bin/env python3
"""
Test SOAP Service (Port 5003)
Tests SOAP note generation from clinical text
"""
import requests

def test_soap_health():
    """Test if SOAP service is running"""
    try:
        response = requests.get("http://localhost:5003/health")
        if response.status_code == 200:
            print("✅ SOAP service is running!")
            return True
        else:
            print(f"❌ SOAP health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to SOAP service: {e}")
        return False

def test_soap_generate():
    """Test SOAP note generation"""
    payload = {
        "transcript": "المريض يبلغ من العمر 45 عاما يشكو من صداع شديد منذ يومين مع حمى وألم في الجسم. الفحص السريري: درجة الحرارة 38.5، ضغط الدم 130/80، النبض 90",
        # Patient is 45 years old, complaining of severe headache for two days with fever and body pain. Clinical exam: temp 38.5, BP 130/80, pulse 90
        "patient_id": "12345",
        "encounter_id": "enc-001"
    }
    
    try:
        print("\n📤 Generating SOAP note from transcript...")
        response = requests.post(
            "http://localhost:5003/generate-note",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SOAP Note Generated:")
            print(f"\n📋 Subjective:\n{result.get('subjective', 'N/A')}")
            print(f"\n🔍 Objective:\n{result.get('objective', 'N/A')}")
            print(f"\n💡 Assessment:\n{result.get('assessment', 'N/A')}")
            print(f"\n📝 Plan:\n{result.get('plan', 'N/A')}")
            return True
        else:
            print(f"\n❌ SOAP generation failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SOAP Service (Clinical Note Generation)")
    print("=" * 60)
    
    if test_soap_health():
        test_soap_generate()
    else:
        print("\n⚠️  Make sure SOAP service is running:")
        print("cd D:\\Downloads\\HealthTech\\mvp-healthtech\\services\\soap; python app.py")
