#!/usr/bin/env python3
"""
Test full pipeline: ASR (WhisperX) + LLM (Medical Correction)
Shows speaker-labeled transcription and LLM-corrected medical terms
"""
import sys
import base64
import requests
import json

def test_full_pipeline(audio_file_path, dialect="egypt"):
    print(f"\n{'='*80}")
    print("FULL MEDICAL TRANSCRIPTION PIPELINE TEST")
    print(f"{'='*80}\n")

    # Read and encode audio file
    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

    # ============================================================================
    # STEP 1: ASR (WhisperX with Speaker Diarization)
    # ============================================================================
    print(f"[1/2] Transcribing with WhisperX + Speaker Diarization...")
    print(f"      Audio file: {audio_file_path}")
    print(f"      Dialect: {dialect}")
    print(f"      Service: http://localhost:5000\n")

    try:
        asr_response = requests.post(
            "http://localhost:5000/transcribe",
            json={
                "audio": audio_base64,
                "dialect": dialect,
                "language": "ar",
                "enable_diarization": True
            },
            timeout=None  # No timeout - let it run as long as needed
        )
        asr_response.raise_for_status()
        asr_result = asr_response.json()

        print("✅ ASR Complete!")
        print(f"   Duration: {asr_result.get('duration', 0):.2f}s")
        print(f"   Processing time: {asr_result.get('processing_time', 0):.2f}s")
        print(f"   Speakers: {asr_result.get('speakers', [])}")
        print(f"   Segments: {len(asr_result.get('segments', []))}\n")

    except requests.exceptions.ConnectionError:
        print("❌ ERROR: ASR service not running!")
        print("   Start it with: cd services/asr && python app.py")
        return
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return

    # ============================================================================
    # STEP 2: LLM Medical Correction
    # ============================================================================
    print(f"[2/2] Correcting medical terms with LLM...")
    print(f"      Service: http://localhost:5001")
    print(f"      Model: MMed-Llama-3-8B\n")

    try:
        llm_response = requests.post(
            "http://localhost:5001/correct-transcription",
            json={
                "text": asr_result['text'],
                "dialect": dialect,
                "context": "medical"
            },
            timeout=None  # No timeout - let it run as long as needed
        )
        llm_response.raise_for_status()
        llm_result = llm_response.json()

        print("✅ LLM Correction Complete!")
        print(f"   Corrections made: {llm_result.get('corrections_made', 0)}")
        print(f"   Dialect normalized: {llm_result.get('dialect_normalized', False)}\n")

    except requests.exceptions.ConnectionError:
        print("⚠️  WARNING: LLM service not running!")
        print("   Start it with: cd services/llm && python app.py")
        print("   Showing ASR-only results...\n")
        llm_result = None
    except Exception as e:
        print(f"⚠️  WARNING: LLM correction failed: {e}")
        print("   Showing ASR-only results...\n")
        llm_result = None

    # ============================================================================
    # DISPLAY RESULTS
    # ============================================================================
    print(f"\n{'='*80}")
    print("RESULTS: RAW ASR TRANSCRIPTION (WhisperX)")
    print(f"{'='*80}\n")
    print(asr_result['text'])

    if llm_result:
        print(f"\n{'='*80}")
        print(f"RESULTS: CORRECTED TRANSCRIPTION (ASR + LLM)")
        print(f"{'='*80}\n")
        print(llm_result['corrected'])

    print(f"\n{'='*80}")
    print("TRANSCRIPT WITH SPEAKERS")
    print(f"{'='*80}\n")

    # Map speaker IDs to roles (simple heuristic: first speaker is usually doctor)
    speakers = asr_result.get('speakers', [])
    speaker_map = {}
    if len(speakers) == 2:
        # Assume SPEAKER_00 is doctor, SPEAKER_01 is patient
        # (In production, you'd use ML or manual labeling)
        speaker_map = {
            speakers[0]: "🩺 Doctor",
            speakers[1]: "👤 Patient"
        }
    elif len(speakers) > 0:
        for i, spk in enumerate(speakers):
            speaker_map[spk] = f"Speaker {i+1}"

    for seg in asr_result.get('segments', []):
        speaker_id = seg.get('speaker', 'Unknown')
        speaker_label = speaker_map.get(speaker_id, speaker_id)
        text = seg.get('text', '').strip()
        start = seg.get('start', 0)
        end = seg.get('end', 0)

        print(f"[{start:>6.1f}s - {end:>6.1f}s] {speaker_label}:")
        print(f"  {text}\n")

    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Audio duration:      {asr_result.get('duration', 0):.2f}s")
    print(f"Processing time:     {asr_result.get('processing_time', 0):.2f}s")
    print(f"Real-time factor:    {asr_result.get('rtf', 0):.2f}x")
    print(f"Language:            {asr_result.get('language', 'N/A')}")
    print(f"Speakers detected:   {len(speakers)}")
    print(f"Total segments:      {len(asr_result.get('segments', []))}")

    if llm_result:
        print(f"LLM corrections:     {llm_result.get('corrections_made', 0)}")

    print(f"{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_full_pipeline.py <audio_file.mp3> [dialect]")
        print("\nDialects: egypt (default), levant, gulf")
        sys.exit(1)

    audio_path = sys.argv[1]
    dialect = sys.argv[2] if len(sys.argv) > 2 else "egypt"

    test_full_pipeline(audio_path, dialect)
