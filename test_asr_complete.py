#!/usr/bin/env python3
"""
Complete ASR Test with Speaker Role Identification
Tests: WhisperX → Speaker Diarization → LLM Role Identification → Transcription Correction
"""
import sys
import base64
import requests
import json
from typing import List, Dict

def identify_speaker_roles(segments: List[Dict], text: str) -> Dict:
    """Use LLM to identify which speaker is doctor vs patient"""
    
    # Prepare conversation for LLM analysis
    conversation = "\n".join([
        f"SPEAKER_{seg.get('speaker', 'UNKNOWN')}: {seg['text']}"
        for seg in segments
    ])
    
    prompt = f"""أنت خبير في تحليل المحادثات الطبية. قم بتحديد من هو الطبيب ومن هو المريض في المحادثة التالية.

المحادثة:
{conversation}

قم بالرد بصيغة JSON فقط:
{{
    "doctor_speaker": "SPEAKER_XX",
    "patient_speaker": "SPEAKER_XX",
    "confidence": "high/medium/low",
    "reasoning": "السبب باختصار"
}}"""

    try:
        response = requests.post(
            "http://localhost:5001/chat",
            json={"messages": [{"role": "user", "content": prompt}]},
            timeout=30
        )
        
        if response.status_code == 200:
            llm_result = response.json()
            llm_text = llm_result.get('response', '{}')
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', llm_text)
            if json_match:
                role_mapping = json.loads(json_match.group())
                return role_mapping
        
    except Exception as e:
        print(f"⚠️  Role identification failed: {e}")
    
    # Fallback: Simple heuristic (first speaker is usually doctor)
    unique_speakers = list(set(seg.get('speaker') for seg in segments if seg.get('speaker')))
    if len(unique_speakers) >= 2:
        return {
            "doctor_speaker": unique_speakers[0],
            "patient_speaker": unique_speakers[1],
            "confidence": "low",
            "reasoning": "Fallback heuristic"
        }
    
    return {
        "doctor_speaker": "SPEAKER_00",
        "patient_speaker": "SPEAKER_01",
        "confidence": "unknown",
        "reasoning": "Default assignment"
    }


def relabel_speakers(segments: List[Dict], role_mapping: Dict) -> List[Dict]:
    """Replace SPEAKER_XX with Doctor/Patient labels"""
    doctor_id = role_mapping.get('doctor_speaker', 'SPEAKER_00')
    patient_id = role_mapping.get('patient_speaker', 'SPEAKER_01')
    
    for seg in segments:
        if seg.get('speaker') == doctor_id:
            seg['speaker'] = 'Doctor'
        elif seg.get('speaker') == patient_id:
            seg['speaker'] = 'Patient'
    
    return segments


def test_complete_pipeline(audio_file_path: str, dialect: str = "egypt"):
    """Complete test: ASR → Diarization → Role ID → LLM Correction"""
    
    print("="*80)
    print("COMPLETE ASR + DIARIZATION + ROLE IDENTIFICATION + LLM TEST")
    print("="*80)
    
    # Read and encode audio
    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # ==============================================================
    # STEP 1: ASR + Diarization
    # ==============================================================
    print("\n[1/4] Transcribing with WhisperX + Speaker Diarization...")
    print("      Service: ASR (http://localhost:5000)")
    
    try:
        asr_response = requests.post(
            "http://localhost:5000/transcribe",
            json={
                "audio": audio_base64,
                "dialect": dialect,
                "language": "ar",
                "enable_diarization": True
            },
            timeout=900  # 15 minutes
        )
        asr_response.raise_for_status()
        asr_result = asr_response.json()
        
        print("✅ Transcription complete!")
        print(f"   Duration: {asr_result.get('duration', 0):.2f}s")
        print(f"   Processing time: {asr_result.get('processing_time', 0):.2f}s")
        print(f"   Segments: {len(asr_result.get('segments', []))}")
        print(f"   Speakers: {asr_result.get('speakers', [])}")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: ASR service not running!")
        print("   Start it with: cd services/asr && python app.py")
        return
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return
    
    # ==============================================================
    # STEP 2: Identify Speaker Roles (Doctor vs Patient)
    # ==============================================================
    print("\n[2/4] Identifying speaker roles (Doctor vs Patient)...")
    print("      Service: LLM (http://localhost:5001)")
    
    role_mapping = identify_speaker_roles(
        asr_result.get('segments', []),
        asr_result['text']
    )
    
    print(f"✅ Role identification complete!")
    print(f"   Doctor: {role_mapping.get('doctor_speaker')}")
    print(f"   Patient: {role_mapping.get('patient_speaker')}")
    print(f"   Confidence: {role_mapping.get('confidence')}")
    print(f"   Reasoning: {role_mapping.get('reasoning', 'N/A')}")
    
    # Relabel speakers
    segments_with_roles = relabel_speakers(
        asr_result.get('segments', []).copy(),
        role_mapping
    )
    
    # ==============================================================
    # STEP 3: Correct with Medical LLM
    # ==============================================================
    print("\n[3/4] Correcting transcription with Medical LLM...")
    print("      Service: LLM (http://localhost:5001)")
    
    try:
        llm_response = requests.post(
            "http://localhost:5001/correct-transcription",
            json={
                "text": asr_result['text'],
                "dialect": dialect,
                "context": "medical"
            },
            timeout=60
        )
        llm_response.raise_for_status()
        llm_result = llm_response.json()
        
        print("✅ LLM correction complete!")
        print(f"   Corrections made: {llm_result.get('corrections_made', 0)}")
        print(f"   Dialect normalized: {llm_result.get('dialect_normalized', False)}")
        
    except requests.exceptions.ConnectionError:
        print("⚠️  LLM service not running, skipping correction")
        print("   Start it with: cd services/llm && python app.py")
        llm_result = None
    except Exception as e:
        print(f"⚠️  LLM correction failed: {e}")
        llm_result = None
    
    # ==============================================================
    # STEP 4: Display Results
    # ==============================================================
    print("\n" + "="*80)
    print("[4/4] RESULTS")
    print("="*80)
    
    print("\n📝 ORIGINAL TRANSCRIPTION (WhisperX):")
    print("-" * 80)
    print(asr_result['text'])
    
    if llm_result:
        print("\n✨ CORRECTED TRANSCRIPTION (WhisperX + LLM):")
        print("-" * 80)
        print(llm_result['corrected'])
    
    print("\n👥 SPEAKER-LABELED CONVERSATION:")
    print("-" * 80)
    for seg in segments_with_roles[:10]:  # Show first 10
        speaker = seg.get('speaker', 'Unknown')
        text = seg.get('text', '').strip()
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        print(f"[{start:.1f}s - {end:.1f}s] {speaker}: {text}")
    
    if len(segments_with_roles) > 10:
        print(f"... and {len(segments_with_roles) - 10} more segments")
    
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"Audio duration: {asr_result.get('duration', 0):.2f}s")
    print(f"Processing time: {asr_result.get('processing_time', 0):.2f}s")
    print(f"Speed: {asr_result.get('rtf', 0):.2f}x realtime factor")
    print(f"Speakers detected: {len(asr_result.get('speakers', []))}")
    print(f"Total segments: {len(segments_with_roles)}")
    if llm_result:
        print(f"LLM corrections: {llm_result.get('corrections_made', 0)}")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_asr_complete.py <audio_file.mp3> [dialect]")
        print("Example: python test_asr_complete.py test1.mp3 egypt")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    dialect = sys.argv[2] if len(sys.argv) > 2 else "egypt"
    
    test_complete_pipeline(audio_path, dialect)
