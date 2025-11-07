#!/usr/bin/env python3
"""
Test LLM Service (Port 5001)
Tests medical Arabic LLM for generating responses
"""
import requests
import json

def test_llm_health():
    """Test if LLM service is running"""
    try:
        response = requests.get("http://localhost:5001/health")
        if response.status_code == 200:
            print("✅ LLM service is running!")
            return True
        else:
            print(f"❌ LLM health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to LLM service: {e}")
        return False

def test_llm_generate():
    """Test LLM text generation"""
    payload = {
        "message": "المريض يشكو من صداع شديد وحمى",  # Patient complains of severe headache and fever
        "sessionId": "test-session-001",
        "intent": "clinical"
    }
    
    try:
        print("\n📤 Sending medical text to LLM...")
        response = requests.post(
            "http://localhost:5001/infer",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ LLM Response:")
            print(f"Intent: {result.get('intent', 'N/A')}")
            print(f"Reply: {result.get('reply', 'N/A')}")
            return True
        else:
            print(f"\n❌ LLM generation failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing LLM Service (Medical Arabic)")
    print("=" * 60)
    
    if test_llm_health():
        test_llm_generate()
    else:
        print("\n⚠️  Make sure LLM service is running:")
        print("$env:HF_HOME = \"D:\\huggingface_cache\"; $env:TRANSFORMERS_CACHE = \"D:\\huggingface_cache\"; cd D:\\Downloads\\HealthTech\\mvp-healthtech\\services\\llm; python app.py")
