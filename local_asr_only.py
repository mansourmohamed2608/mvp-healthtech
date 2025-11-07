"""
Local ASR Script
================
Run this on your local machine to transcribe audio.
Copy the output text and paste it into kaggle_llm_only.py

Usage:
1. Update AUDIO_FILE path below
2. Run: python local_asr_only.py
3. Copy the transcription text from output
4. Paste into kaggle_llm_only.py INPUT_TEXT variable
5. Run kaggle_llm_only.py on Kaggle with GPU

Requirements (local):
- Python with whisperx installed
- Audio file to transcribe
"""

import os
import sys
import time
import json
from pathlib import Path

# Add services path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'asr'))

try:
    import whisperx
except ImportError:
    print("❌ WhisperX not installed!")
    print("Install with: pip install git+https://github.com/m-bain/whisperx.git")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update this path to your audio file
AUDIO_FILE = "test1.mp3"  # or "التهاب اللثه.m4a"

# Dialect setting (optional, helps with better context)
DIALECT = "egypt"  # egypt, gulf, levant, morocco, etc.

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("LOCAL ASR TRANSCRIPTION")
print("=" * 80)
print()

# Check if audio file exists
if not os.path.exists(AUDIO_FILE):
    print(f"❌ Audio file not found: {AUDIO_FILE}")
    print()
    print("Available audio files in current directory:")
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            print(f"  - {file}")
    sys.exit(1)

print(f"Audio file: {AUDIO_FILE}")
print(f"Dialect: {DIALECT}")
print()

# ============================================================================
# TRANSCRIPTION
# ============================================================================

def transcribe_audio(audio_path):
    """Transcribe audio using WhisperX"""
    print("=" * 80)
    print("TRANSCRIPTION")
    print("=" * 80)
    print()
    
    # Load model
    print("📥 Loading WhisperX large-v3...")
    print("   (This may take 1-2 mins first time)")
    start_load = time.time()
    
    model = whisperx.load_model(
        "large-v3",
        device="cpu",  # Force CPU since GPU is limited
        compute_type="int8",
        language="ar"
    )
    
    print(f"✅ Model loaded in {time.time()-start_load:.1f}s")
    print()
    
    # Load audio
    print("📂 Loading audio file...")
    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / 16000
    print(f"   Duration: {duration:.1f}s ({duration/60:.1f} mins)")
    print()
    
    # Transcribe
    print("🎤 Transcribing (this may take a few minutes)...")
    start_transcribe = time.time()
    
    result = model.transcribe(audio, language="ar", batch_size=16)
    
    elapsed = time.time() - start_transcribe
    print(f"✅ Transcribed in {elapsed:.1f}s ({elapsed/duration:.2f}x RT)")
    print()
    
    # Align (optional but improves accuracy)
    print("🔍 Aligning timestamps...")
    try:
        model_a, metadata = whisperx.load_align_model(language_code="ar", device="cpu")
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            "cpu", 
            return_char_alignments=False
        )
        print("✅ Alignment complete")
    except Exception as e:
        print(f"⚠️  Alignment failed (non-critical): {e}")
        print("   Continuing with basic transcription...")
    
    print()
    
    # Extract text
    segments = result.get("segments", [])
    full_text = " ".join([seg["text"] for seg in segments])
    
    print(f"✅ Transcription complete!")
    print(f"   Segments: {len(segments)}")
    print(f"   Text length: {len(full_text)} characters")
    print("=" * 80)
    print()
    
    return {
        "segments": segments,
        "full_text": full_text,
        "duration": duration,
        "processing_time": elapsed
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    start_total = time.time()
    
    # Transcribe
    result = transcribe_audio(AUDIO_FILE)
    
    # Display result
    print("=" * 80)
    print("TRANSCRIPTION RESULT")
    print("=" * 80)
    print()
    print(result["full_text"])
    print()
    print("=" * 80)
    print()
    
    # Save to file
    output_file = f"{Path(AUDIO_FILE).stem}_transcription.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["full_text"])
    
    json_file = f"{Path(AUDIO_FILE).stem}_transcription.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "audio_file": AUDIO_FILE,
            "dialect": DIALECT,
            "transcription": result["full_text"],
            "segments": result["segments"],
            "duration_seconds": result["duration"],
            "processing_time_seconds": result["processing_time"]
        }, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_total
    
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Copy the transcription text above")
    print("2. Open kaggle_llm_only.py")
    print("3. Paste into INPUT_TEXT variable")
    print("4. Upload kaggle_llm_only.py to Kaggle")
    print("5. Enable GPU and run")
    print("6. Download results in ~15-20 seconds!")
    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)
    print(f"✅ Text: {output_file}")
    print(f"✅ JSON: {json_file}")
    print(f"✅ Total time: {total_time:.1f}s ({total_time/60:.1f} mins)")
    print("=" * 80)

if __name__ == "__main__":
    main()
