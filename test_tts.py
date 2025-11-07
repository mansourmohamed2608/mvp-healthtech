#!/usr/bin/env python3
"""
Test TTS Service (Port 5002)
Tests Arabic Text-to-Speech synthesis
"""
import requests
import base64

def test_tts_health():
    """Test if TTS service is running"""
    try:
        response = requests.get("http://localhost:5002/health")
        if response.status_code == 200:
            print("✅ TTS service is running!")
            return True
        else:
            print(f"❌ TTS health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to TTS service: {e}")
        return False

def test_tts_synthesize():
    """Test TTS synthesis"""
    payload = {
        "text": "مرحبا بك في النظام الطبي الذكي",  # Welcome to the smart medical system
        "language": "ar",
        "voice": "default"
    }
    
    try:
        print("\n📤 Sending text to TTS for synthesis...")
        response = requests.post(
            "http://localhost:5002/synthesize",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            audio_data = result.get('audio')
            
            if audio_data:
                # Save audio file for testing
                audio_bytes = base64.b64decode(audio_data)
                output_file = "test_tts_output.wav"
                with open(output_file, "wb") as f:
                    f.write(audio_bytes)
                
                print(f"\n✅ TTS Synthesis successful!")
                print(f"Audio saved to: {output_file}")
                print(f"Audio size: {len(audio_bytes)} bytes")
                return True
            else:
                print("\n❌ No audio data in response")
                return False
        else:
            print(f"\n❌ TTS synthesis failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing TTS Service (Arabic Text-to-Speech)")
    print("=" * 60)
    
    if test_tts_health():
        test_tts_synthesize()
    else:
        print("\n⚠️  Make sure TTS service is running:")
        print("cd D:\\Downloads\\HealthTech\\mvp-healthtech\\services\\tts; python app.py")
