"""
Test script to compare Whisper Base vs LoRA-enhanced models
Shows the difference your fine-tuned model makes!
"""
import requests
import base64
import json
import os
from pathlib import Path

# Configuration
ASR_SERVICE_URL = "http://localhost:8001"
TEST_AUDIO_PATH = "test_data/medical_consultation.wav"  # Update with your test file

def load_audio_base64(audio_path: str) -> str:
    """Load audio file and encode as base64"""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def transcribe_audio(audio_base64: str, use_lora: bool, enable_diarization: bool = True):
    """Transcribe audio with specified model"""
    response = requests.post(
        f"{ASR_SERVICE_URL}/transcribe",
        json={
            "audio": audio_base64,
            "language": "ar",
            "dialect": "egypt",
            "use_lora": use_lora,
            "enable_diarization": enable_diarization
        }
    )
    response.raise_for_status()
    return response.json()

def compare_models(audio_path: str):
    """Compare base model vs LoRA-enhanced model"""
    print("=" * 80)
    print("WHISPER MODEL COMPARISON: Base vs LoRA")
    print("=" * 80)
    print()
    
    # Check service health
    health = requests.get(f"{ASR_SERVICE_URL}/health").json()
    print(f"Service Status: {health['status']}")
    print(f"LoRA Available: {health['lora_enabled']}")
    print(f"Diarization Available: {health['diarization_enabled']}")
    print()
    
    if not health['lora_enabled']:
        print("⚠️ WARNING: LoRA model not loaded!")
        print("Make sure:")
        print("  1. USE_LORA=true in .env")
        print("  2. LORA_ADAPTER_PATH points to ./lora_ckpt")
        print("  3. LoRA adapter files exist")
        return
    
    # Load test audio
    print(f"Loading test audio: {audio_path}")
    audio_base64 = load_audio_base64(audio_path)
    print("✓ Audio loaded")
    print()
    
    # Test with base model
    print("-" * 80)
    print("TEST 1: Base Whisper Large v3 (No LoRA)")
    print("-" * 80)
    result_base = transcribe_audio(audio_base64, use_lora=False)
    print(f"Model Used: {result_base['model_used']}")
    print(f"Processing Time: {result_base['processing_time']:.2f}s")
    print(f"RTF: {result_base['rtf']:.2f}x")
    print(f"Detected Speakers: {len(result_base.get('speakers', []))}")
    print()
    print("Transcription:")
    print(result_base['text'])
    print()
    
    # Test with LoRA model
    print("-" * 80)
    print("TEST 2: Whisper Large v3 + LoRA (Fine-tuned)")
    print("-" * 80)
    result_lora = transcribe_audio(audio_base64, use_lora=True)
    print(f"Model Used: {result_lora['model_used']}")
    print(f"Processing Time: {result_lora['processing_time']:.2f}s")
    print(f"RTF: {result_lora['rtf']:.2f}x")
    print(f"Detected Speakers: {len(result_lora.get('speakers', []))}")
    print()
    print("Transcription:")
    print(result_lora['text'])
    print()
    
    # Comparison
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()
    
    # Character-level difference
    if result_base['text'] != result_lora['text']:
        print("✅ Models produced DIFFERENT transcriptions")
        print(f"   Base length: {len(result_base['text'])} chars")
        print(f"   LoRA length: {len(result_lora['text'])} chars")
        print(f"   Difference: {abs(len(result_base['text']) - len(result_lora['text']))} chars")
        print()
        
        # Word-level difference
        base_words = set(result_base['text'].split())
        lora_words = set(result_lora['text'].split())
        
        unique_to_base = base_words - lora_words
        unique_to_lora = lora_words - base_words
        
        if unique_to_base:
            print(f"Words only in Base model ({len(unique_to_base)}):")
            print(f"   {', '.join(list(unique_to_base)[:10])}")
            print()
        
        if unique_to_lora:
            print(f"Words only in LoRA model ({len(unique_to_lora)}):")
            print(f"   {', '.join(list(unique_to_lora)[:10])}")
            print()
    else:
        print("⚠️ Models produced IDENTICAL transcriptions")
        print("   This might happen if:")
        print("   - Audio is very clear and simple")
        print("   - No medical terminology in this sample")
        print("   - Try with more complex medical audio")
        print()
    
    # Performance comparison
    speedup = result_base['processing_time'] / result_lora['processing_time']
    if speedup > 1.1:
        print(f"⚡ LoRA model is {speedup:.1f}x FASTER")
    elif speedup < 0.9:
        print(f"🐌 LoRA model is {1/speedup:.1f}x SLOWER")
    else:
        print(f"⚖️  Similar processing speed ({speedup:.2f}x)")
    
    print()
    
    # Segment-level comparison
    print("-" * 80)
    print("SEGMENT COMPARISON")
    print("-" * 80)
    print()
    print(f"Base model segments: {len(result_base['segments'])}")
    print(f"LoRA model segments: {len(result_lora['segments'])}")
    print()
    
    # Show first 3 segments from each
    print("First 3 segments (Base):")
    for i, seg in enumerate(result_base['segments'][:3]):
        print(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
    print()
    
    print("First 3 segments (LoRA):")
    for i, seg in enumerate(result_lora['segments'][:3]):
        print(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
    print()
    
    # Save results
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "base_result.json", "w", encoding="utf-8") as f:
        json.dump(result_base, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "lora_result.json", "w", encoding="utf-8") as f:
        json.dump(result_lora, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Full results saved to: {output_dir}/")
    print()
    print("=" * 80)
    print("DONE!")
    print("=" * 80)

if __name__ == "__main__":
    # Update this path to your test audio file
    test_audio = "test_data/medical_consultation.wav"
    
    if not os.path.exists(test_audio):
        print(f"❌ Test audio not found: {test_audio}")
        print()
        print("Please provide a medical consultation audio file.")
        print("You can:")
        print("  1. Record a sample conversation")
        print("  2. Use existing test audio")
        print("  3. Update TEST_AUDIO_PATH in this script")
        print()
        print("Example:")
        print(f"  python {__file__} path/to/your/audio.wav")
        exit(1)
    
    try:
        compare_models(test_audio)
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to ASR service!")
        print()
        print("Make sure the service is running:")
        print("  cd services/asr")
        print("  python app_whisperx_lora.py")
        exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
