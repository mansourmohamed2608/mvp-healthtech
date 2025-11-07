#!/usr/bin/env python3
"""
WER Comparison Script: Test ASR with and without LoRA
Calculates Word Error Rate to measure accuracy
"""
import sys
import base64
import requests
import time
from pathlib import Path
import json
from typing import Dict, List
import jiwer

def load_audio_file(audio_path: str) -> str:
    """Load audio file and encode to base64"""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


def transcribe_with_config(
    audio_base64: str,
    use_lora: bool,
    language: str = "ar",
    dialect: str = "egypt",
    asr_url: str = "http://localhost:5000"
) -> Dict:
    """Transcribe audio with specified configuration"""
    
    payload = {
        "audio": audio_base64,
        "language": language,
        "dialect": dialect,
        "enable_diarization": True,
        "use_lora": use_lora
    }
    
    start_time = time.time()
    response = requests.post(f"{asr_url}/transcribe", json=payload)
    processing_time = time.time() - start_time
    
    if response.status_code != 200:
        raise Exception(f"Transcription failed: {response.status_code} - {response.text}")
    
    result = response.json()
    result["client_processing_time"] = processing_time
    
    return result


def calculate_wer(reference: str, hypothesis: str) -> Dict:
    """Calculate Word Error Rate and related metrics"""
    
    # Clean and normalize text
    reference = reference.strip()
    hypothesis = hypothesis.strip()
    
    # Calculate WER using process_words for detailed metrics
    output = jiwer.process_words(reference, hypothesis)
    
    return {
        "wer": output.wer,
        "mer": output.mer,
        "wil": output.wil,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
        "reference_words": len(reference.split()),
        "hypothesis_words": len(hypothesis.split())
    }


def compare_asr_modes(
    audio_path: str,
    reference_text: str,
    language: str = "ar",
    dialect: str = "egypt",
    asr_url: str = "http://localhost:5000"
):
    """Test ASR with LoRA and calculate WER"""
    
    print("=" * 80)
    print("ASR WER TEST: LoRA-Enhanced Model")
    print("=" * 80)
    print(f"\n📁 Audio file: {audio_path}")
    print(f"🌍 Language: {language}")
    print(f"🗣️  Dialect: {dialect}")
    print(f"📊 Reference text length: {len(reference_text.split())} words\n")
    
    # Load audio
    print("Loading audio file...")
    audio_base64 = load_audio_file(audio_path)
    audio_size_mb = len(audio_base64) / (1024 * 1024)
    print(f"✓ Audio loaded: {audio_size_mb:.2f}MB (base64)\n")
    
    results = {}
    
    # Test with LoRA
    print("=" * 80)
    print("ASR WITH LoRA (Enhanced Model)")
    print("=" * 80)
    try:
        result_with_lora = transcribe_with_config(
            audio_base64,
            use_lora=True,
            language=language,
            dialect=dialect,
            asr_url=asr_url
        )
        
        print(f"✓ Transcription complete")
        print(f"  Duration: {result_with_lora['duration']:.2f}s")
        print(f"  Processing time: {result_with_lora['processing_time']:.2f}s")
        print(f"  RTF: {result_with_lora['rtf']:.2f}x")
        print(f"  Segments: {len(result_with_lora['segments'])}")
        
        # Calculate WER
        wer_with_lora = calculate_wer(reference_text, result_with_lora['text'])
        
        print(f"\n📊 WER Metrics:")
        print(f"  WER: {wer_with_lora['wer']:.2%}")
        print(f"  MER: {wer_with_lora['mer']:.2%}")
        print(f"  WIL: {wer_with_lora['wil']:.2%}")
        print(f"  Errors: {wer_with_lora['substitutions']} subst, {wer_with_lora['deletions']} del, {wer_with_lora['insertions']} ins")
        
        results['with_lora'] = {
            'transcription': result_with_lora,
            'wer_metrics': wer_with_lora
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        results['with_lora'] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    if 'error' not in results.get('with_lora', {}):
        wer_with = results['with_lora']['wer_metrics']['wer']
        time_with = results['with_lora']['transcription']['processing_time']
        seg_with = len(results['with_lora']['transcription']['segments'])
        words_with = results['with_lora']['wer_metrics']['hypothesis_words']
        
        print(f"\n📊 LoRA Model Performance:")
        print(f"  WER:        {wer_with:.2%}")
        print(f"  MER:        {results['with_lora']['wer_metrics']['mer']:.2%}")
        print(f"  WIL:        {results['with_lora']['wer_metrics']['wil']:.2%}")
        print(f"  Time:       {time_with:.2f}s")
        print(f"  Segments:   {seg_with}")
        print(f"  Words:      {words_with}")
        
        print(f"\n🎯 Error Breakdown:")
        print(f"  Substitutions: {results['with_lora']['wer_metrics']['substitutions']}")
        print(f"  Deletions:     {results['with_lora']['wer_metrics']['deletions']}")
        print(f"  Insertions:    {results['with_lora']['wer_metrics']['insertions']}")
        print(f"  Correct:       {results['with_lora']['wer_metrics']['hits']}")
        
        # Baseline comparison (known values)
        baseline_wer = 0.2210  # 22.10% from WhisperX
        if wer_with < baseline_wer:
            improvement = ((baseline_wer - wer_with) / baseline_wer) * 100
            print(f"\n✅ LoRA improves WER by {improvement:.1f}% vs baseline WhisperX (22.10%)")
        elif wer_with > baseline_wer:
            degradation = ((wer_with - baseline_wer) / baseline_wer) * 100
            print(f"\n⚠️  LoRA WER is {degradation:.1f}% worse than baseline WhisperX (22.10%)")
        else:
            print(f"\n⚖️  LoRA WER matches baseline WhisperX (22.10%)")
    
    # Save detailed results
    output_file = "lora_wer_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_asr_wer.py <audio_file> <reference_text_file> [language] [dialect]")
        print("\nExample:")
        print("  python compare_asr_wer.py test1.mp3 reference.txt ar egypt")
        print("\nNote: Install jiwer first: pip install jiwer")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    reference_file = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) > 3 else "ar"
    dialect = sys.argv[4] if len(sys.argv) > 4 else "egypt"
    
    # Check if audio file exists
    if not Path(audio_file).exists():
        print(f"❌ Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Load reference text
    if not Path(reference_file).exists():
        print(f"❌ Error: Reference text file not found: {reference_file}")
        sys.exit(1)
    
    with open(reference_file, "r", encoding="utf-8") as f:
        reference_text = f.read().strip()
    
    if not reference_text:
        print(f"❌ Error: Reference text file is empty")
        sys.exit(1)
    
    # Run comparison
    try:
        compare_asr_modes(audio_file, reference_text, language, dialect)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
