"""
Script to auto-transcribe Google Drive audio files using baseline WhisperX
This generates Arabic transcriptions that you can then manually correct
"""

import os
import requests
import json
from pathlib import Path
from typing import List, Dict
import argparse

def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse filename to extract metadata
    Format: egyptian_Q_question_S_answer_L_answer
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    
    # Extract dialect (egyptian or emirati)
    dialect = parts[0] if parts else "unknown"
    
    # Extract English answer from L_ section
    try:
        l_index = parts.index("L")
        english_answer = " ".join(parts[l_index + 1:]).replace("_", " ")
    except (ValueError, IndexError):
        english_answer = ""
    
    # Extract question from Q_ section
    try:
        q_index = parts.index("Q")
        s_index = parts.index("S")
        english_question = " ".join(parts[q_index + 1:s_index]).replace("_", " ")
    except (ValueError, IndexError):
        english_question = ""
    
    return {
        "dialect": dialect,
        "english_question": english_question,
        "english_answer": english_answer,
        "original_filename": filename
    }

def transcribe_audio(audio_path: str, asr_url: str = "http://localhost:5000/asr", 
                     language: str = "ar", dialect: str = "egypt") -> Dict:
    """
    Transcribe audio file using baseline WhisperX ASR service
    """
    print(f"  Transcribing: {Path(audio_path).name}")
    
    try:
        with open(audio_path, 'rb') as f:
            files = {"audio": f}
            data = {
                "language": language,
                "dialect": dialect,
                "use_lora": "false"  # Use baseline WhisperX only!
            }
            
            response = requests.post(asr_url, files=files, data=data, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            return {
                "success": True,
                "transcription": result.get("transcription", ""),
                "confidence": result.get("confidence", 0.0)
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "transcription": "",
            "confidence": 0.0
        }

def process_audio_directory(audio_dir: str, output_file: str = "transcriptions_to_review.json",
                           asr_url: str = "http://localhost:5000/asr"):
    """
    Process all audio files in directory and generate transcriptions
    """
    audio_dir_path = Path(audio_dir)
    
    if not audio_dir_path.exists():
        print(f"❌ Directory not found: {audio_dir}")
        return
    
    # Find all audio files
    audio_extensions = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir_path.glob(f"*{ext}"))
    
    if not audio_files:
        print(f"❌ No audio files found in: {audio_dir}")
        return
    
    print(f"📁 Found {len(audio_files)} audio files")
    print(f"🔄 Starting auto-transcription with baseline WhisperX...\n")
    
    results = []
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        # Parse filename
        metadata = parse_filename(audio_file.name)
        
        # Transcribe with baseline WhisperX
        dialect = "egypt" if metadata["dialect"] == "egyptian" else "uae"
        transcription_result = transcribe_audio(str(audio_file), asr_url, "ar", dialect)
        
        # Combine metadata and transcription
        entry = {
            "audio_path": str(audio_file),
            "dialect": metadata["dialect"],
            "english_question": metadata["english_question"],
            "english_answer": metadata["english_answer"],
            "auto_transcription_arabic": transcription_result["transcription"],
            "confidence": transcription_result["confidence"],
            "success": transcription_result["success"],
            "corrected_transcription_arabic": "",  # To be filled manually
            "needs_review": True,
            "reviewed": False
        }
        
        if not transcription_result["success"]:
            entry["error"] = transcription_result.get("error", "Unknown error")
        
        results.append(entry)
        print(f"  ✅ Auto-transcription: {transcription_result['transcription'][:100]}...")
        print()
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Processed {len(results)} files")
    print(f"📄 Results saved to: {output_file}")
    print(f"\n📝 Next steps:")
    print(f"1. Open {output_file} in a text editor")
    print(f"2. Review each 'auto_transcription_arabic' field")
    print(f"3. Correct errors and paste into 'corrected_transcription_arabic'")
    print(f"4. Set 'reviewed': true for corrected entries")
    print(f"5. Run create_training_manifest.py to generate training CSV")

def main():
    parser = argparse.ArgumentParser(
        description="Auto-transcribe Google Drive audio files for LoRA training"
    )
    parser.add_argument(
        "audio_dir",
        help="Directory containing audio files (Egyptian/Emirati folders)"
    )
    parser.add_argument(
        "--output",
        default="transcriptions_to_review.json",
        help="Output JSON file for review (default: transcriptions_to_review.json)"
    )
    parser.add_argument(
        "--asr-url",
        default="http://localhost:5000/asr",
        help="ASR service URL (default: http://localhost:5000/asr)"
    )
    
    args = parser.parse_args()
    
    process_audio_directory(args.audio_dir, args.output, args.asr_url)

if __name__ == "__main__":
    main()
