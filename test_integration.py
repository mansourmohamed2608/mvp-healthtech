#!/usr/bin/env python3
"""
End-to-End Integration Test
Tests the complete flow: Audio → ASR → LLM → SOAP → FHIR
"""
import os
import requests
import base64
import json
import time

def test_all_health():
    """Check if all services are running"""
    services = {
        "ASR": "http://localhost:5000/health",
        "LLM": "http://localhost:5001/health",
        "TTS": "http://localhost:5002/health",
        "SOAP": "http://localhost:5003/health",
        "FHIR": "http://localhost:5004/health",
        "Gateway": "http://localhost:3001/health"
    }
    
    print("🔍 Checking all services...\n")
    all_running = True
    
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ {name:10} → Running")
            else:
                print(f"❌ {name:10} → Error {response.status_code}")
                all_running = False
        except Exception as e:
            print(f"❌ {name:10} → Not reachable")
            all_running = False
    
    return all_running

def test_clinical_notes_flow():
    """Test complete clinical notes workflow"""
    print("\n" + "=" * 70)
    print("Testing Clinical Notes Flow (Audio → SOAP → FHIR)")
    print("=" * 70)
    
    # Step 1: Read audio file
    audio_file = "services/asr/tts_ar_med/tts_ar_med/000aa669aaa3421ea768a134c25ff7db.mp3"
    try:
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        print(f"\n📁 Step 1: Audio file loaded ({len(audio_bytes)} bytes)")
    except Exception as e:
        print(f"❌ Cannot load audio file: {e}")
        return False
    
    # Step 2: Transcribe with ASR
    print("\n🎤 Step 2: Transcribing audio...")
    try:
        asr_response = requests.post(
            "http://localhost:5000/transcribe",
            json={"audio": audio_base64, "dialect": "egyptian"}
        )
        if asr_response.status_code == 200:
            transcript = asr_response.json()['text']
            print(f"✅ Transcript: {transcript}")
        else:
            print(f"❌ ASR failed: {asr_response.text}")
            return False
    except Exception as e:
        print(f"❌ ASR error: {e}")
        return False
    
    # Step 3: Generate SOAP note
    print("\n📝 Step 3: Generating SOAP note...")
    time.sleep(1)  # Small delay
    try:
        soap_response = requests.post(
            "http://localhost:5003/generate-note",
            json={
                "transcript": transcript,
                "patient_id": "test-patient-001",
                "encounter_id": "test-enc-001"
            }
        )
        if soap_response.status_code == 200:
            soap_note = soap_response.json()
            print("✅ SOAP Note Generated:")
            print(f"   S: {soap_note.get('subjective', 'N/A')[:100]}...")
            print(f"   O: {soap_note.get('objective', 'N/A')[:100]}...")
            print(f"   A: {soap_note.get('assessment', 'N/A')[:100]}...")
            print(f"   P: {soap_note.get('plan', 'N/A')[:100]}...")
        else:
            print(f"❌ SOAP failed: {soap_response.text}")
            return False
    except Exception as e:
        print(f"❌ SOAP error: {e}")
        return False
    
    # Step 4: Create FHIR resource
    print("\n🏥 Step 4: Writing FHIR resources...")
    time.sleep(1)
    try:
        internal_secret = os.getenv("INTERNAL_SECRET", "")
        headers = {"Content-Type": "application/json"}
        if internal_secret:
            headers["x-internal-secret"] = internal_secret
        fhir_response = requests.post(
            "http://localhost:5004/write",
            json={
                "soapNote": soap_note,
                "patientId": "test-patient-001",
                "encounterId": "test-enc-001",
                "practitionerId": "dr-test",
                "sessionId": "session-test-001",
            },
            headers=headers,
        )
        if fhir_response.status_code in [200, 201]:
            fhir_resource = fhir_response.json()
            print(f"✅ FHIR write OK: {fhir_resource.get('documentReferenceId', 'N/A')}")
            print(f"   Encounter: {fhir_resource.get('encounterId', 'N/A')}")
            print(f"   Composition: {fhir_resource.get('compositionId', 'N/A')}")
        else:
            print(f"❌ FHIR failed: {fhir_response.text}")
            return False
    except Exception as e:
        print(f"❌ FHIR error: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ END-TO-END TEST SUCCESSFUL!")
    print("=" * 70)
    return True

def test_voice_agent_flow():
    """Test voice agent conversation flow"""
    print("\n" + "=" * 70)
    print("Testing Voice Agent Flow (ASR → LLM → TTS)")
    print("=" * 70)
    
    # Step 1: Transcribe (already tested)
    print("\n✅ Step 1: ASR transcription (already working)")
    transcript = "المريض يشكو من صداع شديد"
    
    # Step 2: LLM generates response
    print(f"\n🤖 Step 2: LLM generating response for: '{transcript}'")
    try:
        llm_response = requests.post(
            "http://localhost:5001/generate",
            json={"text": transcript, "context": "clinical", "max_tokens": 100}
        )
        if llm_response.status_code == 200:
            ai_response = llm_response.json()['generated_text']
            print(f"✅ LLM Response: {ai_response}")
        else:
            print(f"❌ LLM failed: {llm_response.text}")
            return False
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return False
    
    # Step 3: TTS synthesizes response
    print(f"\n🔊 Step 3: TTS synthesizing: '{ai_response[:50]}...'")
    try:
        tts_response = requests.post(
            "http://localhost:5002/synthesize",
            json={"text": ai_response, "language": "ar"}
        )
        if tts_response.status_code == 200:
            audio_data = tts_response.json().get('audio')
            if audio_data:
                audio_bytes = base64.b64decode(audio_data)
                print(f"✅ TTS Audio Generated: {len(audio_bytes)} bytes")
            else:
                print("❌ No audio in TTS response")
                return False
        else:
            print(f"❌ TTS failed: {tts_response.text}")
            return False
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ VOICE AGENT FLOW SUCCESSFUL!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("END-TO-END INTEGRATION TEST")
    print("=" * 70)
    
    # Check all services first
    if not test_all_health():
        print("\n⚠️  Not all services are running. Please start missing services.")
        print("\nStartup commands:")
        print("ASR:  $env:HF_HOME='D:\\huggingface_cache'; cd services/asr; python app.py")
        print("LLM:  $env:HF_HOME='D:\\huggingface_cache'; cd services/llm; python app.py")
        print("TTS:  cd services/tts; python app.py")
        print("SOAP: cd services/soap; python app.py")
        print("FHIR: cd services/fhir; python app.py")
        exit(1)
    
    # Test both flows
    print("\n")
    test_clinical_notes_flow()
    print("\n")
    test_voice_agent_flow()
