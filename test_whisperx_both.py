"""
Test WhisperX with and without LLM correction
Shows both outputs side-by-side
"""
import requests
import base64
import json

# Test audio file path
AUDIO_FILE = "test1.mp3"  # Your 2:14 audio

print("=" * 80)
print("WHISPERX + LLM POST-PROCESSING TEST")
print("=" * 80)

# Read and encode audio
with open(AUDIO_FILE, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# ============================================================================
# STEP 1: Transcribe with WhisperX (with prompt engineering)
# ============================================================================
print("\n[1/2] Transcribing with WhisperX + Prompt Engineering...")
print("      Services: ASR (http://localhost:5000)")
print()

try:
    asr_response = requests.post(
        "http://localhost:5000/transcribe",
        json={
            "audio": audio_b64,
            "dialect": "egypt",
            "language": "ar",
            "enable_diarization": True
        },
        timeout=120
    )
    asr_response.raise_for_status()
    asr_result = asr_response.json()
    
    print("✅ WhisperX Transcription:")
    print("-" * 80)
    print(f"Duration: {asr_result.get('duration', 'N/A'):.2f}s")
    print(f"Processing time: {asr_result.get('processing_time', 'N/A'):.2f}s")
    print(f"Speed: {asr_result.get('rtf', 'N/A'):.2f}x realtime")
    print(f"Speakers detected: {len(asr_result.get('speakers', []))}")
    print()
    print(f"RAW TEXT (WhisperX only):")
    print(asr_result['text'])
    print()
    
    if asr_result.get('segments'):
        print(f"SEGMENTS WITH SPEAKERS:")
        for seg in asr_result['segments'][:5]:  # Show first 5
            speaker = seg.get('speaker', 'Unknown')
            text = seg.get('text', '')
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            print(f"  [{start:.1f}s - {end:.1f}s] {speaker}: {text}")
        if len(asr_result['segments']) > 5:
            print(f"  ... and {len(asr_result['segments']) - 5} more segments")
    print()

except requests.exceptions.ConnectionError:
    print("❌ ERROR: ASR service not running!")
    print("   Start it with: cd services/asr && python app.py")
    exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    exit(1)

# ============================================================================
# STEP 2: Correct with Medical LLM
# ============================================================================
print("\n[2/2] Correcting with Medical LLM (MMed-Llama-3-8B)...")
print("      Services: LLM (http://localhost:5001)")
print()

try:
    llm_response = requests.post(
        "http://localhost:5001/correct-transcription",
        json={
            "text": asr_result['text'],
            "dialect": "egypt",
            "context": "medical"
        },
        timeout=60
    )
    llm_response.raise_for_status()
    llm_result = llm_response.json()
    
    print("✅ LLM-Corrected Transcription:")
    print("-" * 80)
    print(f"CORRECTED TEXT (WhisperX + LLM):")
    print(llm_result['corrected'])
    print()
    print(f"Corrections made: {llm_result.get('corrections_made', 0)}")
    print(f"Dialect normalized: {llm_result.get('dialect_normalized', False)}")
    print()

except requests.exceptions.ConnectionError:
    print("❌ ERROR: LLM service not running!")
    print("   Start it with: cd services/llm && python app.py")
    print()
    print("⚠️  Showing WhisperX output only (without LLM correction)")
except Exception as e:
    print(f"❌ ERROR: {e}")
    print()
    print("⚠️  Showing WhisperX output only (without LLM correction)")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON:")
print("=" * 80)
print()
print("WITHOUT LLM (WhisperX only):")
print(asr_result['text'][:200] + "..." if len(asr_result['text']) > 200 else asr_result['text'])
print()

if 'llm_result' in locals():
    print("WITH LLM (WhisperX + Medical LLM):")
    print(llm_result['corrected'][:200] + "..." if len(llm_result['corrected']) > 200 else llm_result['corrected'])
    print()
    
    # Highlight differences
    if asr_result['text'] != llm_result['corrected']:
        print("✨ LLM made corrections! Check the differences above.")
    else:
        print("ℹ️  No corrections needed - WhisperX was already accurate!")

print("\n" + "=" * 80)
print("To compare in detail, check the full outputs above.")
print("=" * 80)
