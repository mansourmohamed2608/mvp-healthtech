#!/usr/bin/env python3
"""
Simple LoRA Test Script
Tests if LoRA adapters are loading and working correctly
"""
import sys
import base64
import requests
import time
from pathlib import Path
import json

def test_lora_loading():
    """Check if LoRA is loaded via health endpoint"""
    print("=" * 80)
    print("STEP 1: Checking LoRA Status")
    print("=" * 80)
    
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code != 200:
            print(f"❌ ASR service not responding (status {response.status_code})")
            print("   Make sure to start the service first:")
            print("   cd services/asr && python -m uvicorn app:app --host 0.0.0.0 --port 5000")
            return False
        
        health = response.json()
        print(f"\n✓ ASR Service is running")
        print(f"  Model: {health.get('model')}")
        print(f"  Device: {health.get('device')}")
        print(f"  LoRA Enabled: {health.get('lora_enabled')}")
        print(f"  LoRA Path: {health.get('lora_path')}")
        print(f"  Diarization: {health.get('diarization_enabled')}")
        print(f"  VAD: {health.get('vad_enabled')}")
        
        if not health.get('lora_enabled'):
            print("\n⚠️  WARNING: LoRA is NOT loaded!")
            print("   Check services/asr/.env - USE_LORA should be 'true'")
            print("   Check that lora_ckpt directory exists and has adapter files")
            return False
        
        print("\n✅ LoRA adapters are loaded!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to ASR service at http://localhost:5000")
        print("   Start the service first:")
        print("   cd services/asr && python -m uvicorn app:app --host 0.0.0.0 --port 5000")
        return False


def test_transcription_with_lora(audio_path: str):
    """Test transcription with LoRA enabled"""
    print("\n" + "=" * 80)
    print("STEP 2: Testing Transcription WITH LoRA")
    print("=" * 80)
    
    # Load audio
    print(f"\nLoading audio: {audio_path}")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    print(f"✓ Audio loaded: {len(audio_base64)/(1024*1024):.2f}MB (base64)")
    
    # Transcribe with LoRA
    payload = {
        "audio": audio_base64,
        "language": "ar",
        "dialect": "egypt",
        "enable_diarization": True,
        "use_lora": True  # Enable LoRA
    }
    
    print("\n🔥 Sending request with use_lora=True...")
    start_time = time.time()
    
    try:
        response = requests.post("http://localhost:5000/transcribe", json=payload)
        processing_time = time.time() - start_time
        
        if response.status_code != 200:
            print(f"❌ Transcription failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
        
        result = response.json()
        
        print(f"\n✅ Transcription completed!")
        print(f"  Model used: {result.get('model_used')}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Audio duration: {result.get('duration', 0):.2f}s")
        print(f"  RTF: {result.get('rtf', 0):.2f}x")
        print(f"  Total segments: {len(result.get('segments', []))}")
        print(f"  Speakers detected: {result.get('speakers', [])}")
        
        # Show first 3 segments
        print(f"\n📝 First 3 segments:")
        for i, seg in enumerate(result.get('segments', [])[:3]):
            print(f"\n  [{i+1}] {seg['start']:.2f}s - {seg['end']:.2f}s | {seg.get('speaker', 'N/A')}")
            print(f"      {seg['text']}")
        
        # Full transcript
        print(f"\n📄 Full transcript:")
        print(f"  {result.get('text', '')[:500]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return None


def test_transcription_without_lora(audio_path: str):
    """Test transcription without LoRA for comparison"""
    print("\n" + "=" * 80)
    print("STEP 3: Testing Transcription WITHOUT LoRA (for comparison)")
    print("=" * 80)
    
    # Load audio
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    # Transcribe without LoRA
    payload = {
        "audio": audio_base64,
        "language": "ar",
        "dialect": "egypt",
        "enable_diarization": True,
        "use_lora": False  # Disable LoRA
    }
    
    print("\n📊 Sending request with use_lora=False...")
    start_time = time.time()
    
    try:
        response = requests.post("http://localhost:5000/transcribe", json=payload)
        processing_time = time.time() - start_time
        
        if response.status_code != 200:
            print(f"❌ Transcription failed: {response.status_code}")
            return None
        
        result = response.json()
        
        print(f"\n✅ Transcription completed!")
        print(f"  Model used: {result.get('model_used')}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  RTF: {result.get('rtf', 0):.2f}x")
        print(f"  Total segments: {len(result.get('segments', []))}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_lora.py <audio_file>")
        print("Example: python test_lora.py test1.mp3")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    print("\n" + "🔬" * 40)
    print(" " * 15 + "LoRA TEST SUITE")
    print("🔬" * 40 + "\n")
    
    # Step 1: Check if LoRA is loaded
    if not test_lora_loading():
        print("\n❌ Cannot proceed - LoRA not loaded")
        sys.exit(1)
    
    # Step 2: Test with LoRA
    result_with_lora = test_transcription_with_lora(audio_path)
    if not result_with_lora:
        print("\n❌ LoRA transcription failed")
        sys.exit(1)
    
    # Step 3: Test without LoRA
    result_without_lora = test_transcription_without_lora(audio_path)
    if not result_without_lora:
        print("\n⚠️  Baseline transcription failed, but LoRA worked")
    
    # Step 4: Compare results
    if result_with_lora and result_without_lora:
        print("\n" + "=" * 80)
        print("STEP 4: Comparison Summary")
        print("=" * 80)
        
        print(f"\n📊 Performance Comparison:")
        print(f"  With LoRA:")
        print(f"    - Processing time: {result_with_lora.get('processing_time', 0):.2f}s")
        print(f"    - Segments: {len(result_with_lora.get('segments', []))}")
        print(f"    - RTF: {result_with_lora.get('rtf', 0):.2f}x")
        
        print(f"\n  Without LoRA:")
        print(f"    - Processing time: {result_without_lora.get('processing_time', 0):.2f}s")
        print(f"    - Segments: {len(result_without_lora.get('segments', []))}")
        print(f"    - RTF: {result_without_lora.get('rtf', 0):.2f}x")
        
        print(f"\n📝 Text Comparison (first 300 chars):")
        print(f"\n  With LoRA:")
        print(f"    {result_with_lora.get('text', '')[:300]}...")
        print(f"\n  Without LoRA:")
        print(f"    {result_without_lora.get('text', '')[:300]}...")
        
        # Check if texts are identical
        if result_with_lora.get('text') == result_without_lora.get('text'):
            print(f"\n⚠️  WARNING: Transcripts are IDENTICAL!")
            print(f"   This suggests LoRA may not actually be applied.")
            print(f"   Check the transcribe_with_lora() function in services/asr/app.py")
        else:
            print(f"\n✅ Transcripts are DIFFERENT - LoRA is having an effect!")
    
    print("\n" + "=" * 80)
    print("✅ LoRA TEST COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
