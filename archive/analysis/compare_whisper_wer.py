#!/usr/bin/env python3
"""
WER Comparison Script: Whisper Base vs Whisper + LoRA
Tests vanilla Whisper (not WhisperX) with and without LoRA adapters
Calculates Word Error Rate to measure accuracy
"""
import sys
import time
from pathlib import Path
import json
from typing import Dict
import torch
import warnings
warnings.filterwarnings('ignore')

def calculate_wer(reference: str, hypothesis: str) -> Dict:
    """Calculate Word Error Rate and related metrics"""
    try:
        import jiwer
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
    except ImportError:
        print("⚠️  jiwer not installed. Install with: pip install jiwer")
        return None


def transcribe_with_base_whisper(audio_path: str, language: str = "ar") -> Dict:
    """Transcribe using base Whisper model"""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import librosa
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"  Loading base Whisper large-v3 on {device}...")
    start_load = time.time()
    
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        local_files_only=True  # Use cached model
    )
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-large-v3",
        local_files_only=True  # Use cached model
    )
    
    load_time = time.time() - start_load
    print(f"  ✓ Model loaded in {load_time:.1f}s")
    
    print(f"  Transcribing...")
    start_time = time.time()
    
    # Load audio with librosa
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    
    # Prepare input
    input_features = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features
    
    # Match dtype with model
    if device == "cuda":
        input_features = input_features.to(device).to(torch.float16)
    else:
        input_features = input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language=language,
            task="transcribe",
        )
    
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    
    processing_time = time.time() - start_time
    
    return {
        "text": text,
        "processing_time": processing_time,
        "duration": duration,
        "rtf": processing_time / duration if duration > 0 else 0,
        "load_time": load_time
    }


def transcribe_with_lora_whisper(audio_path: str, lora_path: str, language: str = "ar") -> Dict:
    """Transcribe using Whisper + LoRA adapters"""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel
    import librosa
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"  Loading Whisper + LoRA on {device}...")
    start_load = time.time()
    
    # Load base model
    base_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        local_files_only=True  # Use cached model
    )
    
    # Load LoRA adapters
    lora_model = PeftModel.from_pretrained(
        base_model, 
        lora_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    lora_model.eval()
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-large-v3",
        local_files_only=True  # Use cached model
    )
    
    load_time = time.time() - start_load
    print(f"  ✓ Model + LoRA loaded in {load_time:.1f}s")
    
    print(f"  Transcribing...")
    start_time = time.time()
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    
    # Prepare input
    input_features = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features
    
    # Match dtype with model
    if device == "cuda":
        input_features = input_features.to(device).to(torch.float16)
    else:
        input_features = input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = lora_model.generate(
            input_features=input_features,
            language=language,
            task="transcribe",
        )
    
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    
    processing_time = time.time() - start_time
    
    return {
        "text": text,
        "processing_time": processing_time,
        "duration": duration,
        "rtf": processing_time / duration if duration > 0 else 0,
        "load_time": load_time
    }


def compare_whisper_modes(
    audio_path: str,
    reference_text: str,
    language: str = "ar",
    lora_path: str = "./services/asr/lora_ckpt"
):
    """Compare Whisper base vs Whisper + LoRA and calculate WER"""
    
    print("\n" + "=" * 80)
    print("WHISPER WER COMPARISON: Base vs LoRA")
    print("=" * 80)
    print(f"\n📁 Audio file: {audio_path}")
    print(f"🌍 Language: {language}")
    print(f"📂 LoRA path: {lora_path}")
    print(f"📊 Reference text length: {len(reference_text.split())} words")
    print(f"💻 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    results = {}
    
    # Test BASE Whisper
    print("=" * 80)
    print("TEST 1: BASE WHISPER (No LoRA)")
    print("=" * 80)
    try:
        result_base = transcribe_with_base_whisper(audio_path, language)
        
        print(f"\n✓ Transcription complete")
        print(f"  Duration: {result_base['duration']:.2f}s")
        print(f"  Processing time: {result_base['processing_time']:.2f}s")
        print(f"  RTF: {result_base['rtf']:.2f}x")
        print(f"\n📝 Transcription:")
        print(f"  {result_base['text']}")
        
        # Calculate WER
        wer_base = calculate_wer(reference_text, result_base['text'])
        
        if wer_base:
            print(f"\n📊 WER Metrics:")
            print(f"  WER: {wer_base['wer']:.2%}")
            print(f"  MER: {wer_base['mer']:.2%}")
            print(f"  WIL: {wer_base['wil']:.2%}")
            print(f"  Errors: {wer_base['substitutions']} subst, {wer_base['deletions']} del, {wer_base['insertions']} ins")
            print(f"  Hits: {wer_base['hits']}/{wer_base['reference_words']} words")
        
        results['base'] = {
            'transcription': result_base,
            'wer_metrics': wer_base
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['base'] = {'error': str(e)}
        return results
    
    print()
    
    # Test Whisper + LoRA
    print("=" * 80)
    print("TEST 2: WHISPER + LoRA")
    print("=" * 80)
    
    # Check if LoRA path exists
    if not Path(lora_path).exists():
        print(f"\n⚠️  LoRA path not found: {lora_path}")
        print("Skipping LoRA test.")
        results['lora'] = {'error': 'LoRA path not found'}
    else:
        try:
            result_lora = transcribe_with_lora_whisper(audio_path, lora_path, language)
            
            print(f"\n✓ Transcription complete")
            print(f"  Duration: {result_lora['duration']:.2f}s")
            print(f"  Processing time: {result_lora['processing_time']:.2f}s")
            print(f"  RTF: {result_lora['rtf']:.2f}x")
            print(f"\n📝 Transcription:")
            print(f"  {result_lora['text']}")
            
            # Calculate WER
            wer_lora = calculate_wer(reference_text, result_lora['text'])
            
            if wer_lora:
                print(f"\n📊 WER Metrics:")
                print(f"  WER: {wer_lora['wer']:.2%}")
                print(f"  MER: {wer_lora['mer']:.2%}")
                print(f"  WIL: {wer_lora['wil']:.2%}")
                print(f"  Errors: {wer_lora['substitutions']} subst, {wer_lora['deletions']} del, {wer_lora['insertions']} ins")
                print(f"  Hits: {wer_lora['hits']}/{wer_lora['reference_words']} words")
            
            results['lora'] = {
                'transcription': result_lora,
                'wer_metrics': wer_lora
            }
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results['lora'] = {'error': str(e)}
    
    # Detailed Comparison
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    if 'error' not in results.get('base', {}) and 'error' not in results.get('lora', {}):
        base_text = results['base']['transcription']['text']
        lora_text = results['lora']['transcription']['text']
        
        print(f"\n📝 Side-by-Side Transcriptions:")
        print(f"\n  Base: {base_text}")
        print(f"  LoRA: {lora_text}")
        
        if base_text == lora_text:
            print(f"\n⚠️  IDENTICAL TRANSCRIPTIONS!")
            print(f"  LoRA adapters made NO difference to the output.")
            print(f"  This suggests LoRA was trained on incompatible data (e.g., synthetic TTS).")
        else:
            print(f"\n✅ DIFFERENT TRANSCRIPTIONS")
            print(f"  LoRA adapters ARE affecting the output.")
            
            # Word-level comparison
            base_words = base_text.split()
            lora_words = lora_text.split()
            
            print(f"\n  Word counts: Base={len(base_words)}, LoRA={len(lora_words)}")
            
            if len(base_words) == len(lora_words):
                diffs = []
                for i, (b, l) in enumerate(zip(base_words, lora_words)):
                    if b != l:
                        diffs.append((i+1, b, l))
                
                if diffs:
                    print(f"\n  📝 Word-level differences ({len(diffs)} changes):")
                    for pos, base_w, lora_w in diffs[:10]:
                        print(f"    Position {pos}: '{base_w}' → '{lora_w}'")
                    if len(diffs) > 10:
                        print(f"    ... and {len(diffs) - 10} more")
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    if 'error' not in results.get('base', {}) and 'error' not in results.get('lora', {}):
        wer_base = results['base']['wer_metrics']
        wer_lora = results['lora']['wer_metrics']
        time_base = results['base']['transcription']['processing_time']
        time_lora = results['lora']['transcription']['processing_time']
        
        if wer_base and wer_lora:
            print(f"\n📊 WER Comparison:")
            print(f"  Base Whisper:  {wer_base['wer']:.2%}")
            print(f"  LoRA Whisper:  {wer_lora['wer']:.2%}")
            
            wer_diff = wer_lora['wer'] - wer_base['wer']
            wer_diff_pct = (wer_diff / wer_base['wer'] * 100) if wer_base['wer'] > 0 else 0
            
            if abs(wer_diff) < 0.001:  # Less than 0.1% difference
                print(f"\n  ⚖️  IDENTICAL WER - No measurable difference")
                print(f"  ❌ VERDICT: LoRA adapters are NOT helping")
            elif wer_diff < 0:
                print(f"\n  ✅ LoRA IMPROVED WER by {abs(wer_diff):.2%} ({abs(wer_diff_pct):.1f}%)")
                if abs(wer_diff_pct) > 5:
                    print(f"  🎯 VERDICT: LoRA adapters provide SIGNIFICANT improvement!")
                else:
                    print(f"  🤔 VERDICT: LoRA provides minor improvement ({abs(wer_diff_pct):.1f}%)")
            else:
                print(f"\n  ❌ LoRA WORSENED WER by {wer_diff:.2%} ({wer_diff_pct:.1f}%)")
                print(f"  ⚠️  VERDICT: LoRA adapters are HARMFUL - remove them!")
            
            print(f"\n⏱️  Processing Time:")
            print(f"  Base: {time_base:.2f}s")
            print(f"  LoRA: {time_lora:.2f}s")
            time_diff = time_lora - time_base
            print(f"  Difference: {'+' if time_diff > 0 else ''}{time_diff:.2f}s")
            
            print(f"\n🎯 Error Breakdown:")
            print(f"  {'Metric':<15} {'Base':<10} {'LoRA':<10} {'Diff':<10}")
            print(f"  {'-'*45}")
            print(f"  {'Substitutions':<15} {wer_base['substitutions']:<10} {wer_lora['substitutions']:<10} {wer_lora['substitutions'] - wer_base['substitutions']:<10}")
            print(f"  {'Deletions':<15} {wer_base['deletions']:<10} {wer_lora['deletions']:<10} {wer_lora['deletions'] - wer_base['deletions']:<10}")
            print(f"  {'Insertions':<15} {wer_base['insertions']:<10} {wer_lora['insertions']:<10} {wer_lora['insertions'] - wer_base['insertions']:<10}")
            print(f"  {'Correct (Hits)':<15} {wer_base['hits']:<10} {wer_lora['hits']:<10} {wer_lora['hits'] - wer_base['hits']:<10}")
            
            print(f"\n📋 Recommendation:")
            if abs(wer_diff) < 0.001:
                print(f"  ❌ REMOVE LoRA adapters - they're not doing anything")
                print(f"  Reason: Likely trained on synthetic TTS data, not real speech")
            elif wer_diff < -0.05:  # 5% improvement
                print(f"  ✅ KEEP LoRA adapters - significant improvement!")
            elif wer_diff < 0:
                print(f"  🤔 LoRA shows minor improvement - consider trade-off:")
                print(f"     + WER improvement: {abs(wer_diff_pct):.1f}%")
                print(f"     - Added complexity: LoRA loading + maintenance")
                print(f"     - Slower startup: ~{results['lora']['transcription']['load_time']:.1f}s vs {results['base']['transcription']['load_time']:.1f}s")
            else:
                print(f"  ❌ REMOVE LoRA adapters - they're making it worse!")
    
    elif 'error' in results.get('lora', {}):
        print(f"\n⚠️  Could not test LoRA adapters")
        print(f"  Error: {results['lora'].get('error', 'Unknown')}")
        print(f"\n  Base Whisper WER: {results['base']['wer_metrics']['wer']:.2%}")
    
    # Save detailed results
    output_file = "whisper_wer_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        # Make results JSON serializable
        json_results = {
            'base': {
                'transcription': results['base']['transcription'],
                'wer_metrics': results['base'].get('wer_metrics') if results['base'].get('wer_metrics') else None
            } if 'error' not in results['base'] else results['base'],
            'lora': {
                'transcription': results['lora']['transcription'],
                'wer_metrics': results['lora'].get('wer_metrics') if results['lora'].get('wer_metrics') else None
            } if 'error' not in results.get('lora', {}) else results.get('lora', {})
        }
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_whisper_wer.py <audio_file> <reference_text_file> [language] [lora_path]")
        print("\nExample:")
        print("  python compare_whisper_wer.py test1.mp3 reference_test1.txt ar")
        print("  python compare_whisper_wer.py test1.mp3 reference_test1.txt ar ./services/asr/lora_ckpt")
        print("\nNote: Requires transformers, peft, librosa, jiwer, soundfile, torch")
        print("Install: pip install transformers peft librosa jiwer soundfile torch")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    reference_file = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) > 3 else "ar"
    lora_path = sys.argv[4] if len(sys.argv) > 4 else "./services/asr/lora_ckpt"
    
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
        compare_whisper_modes(audio_file, reference_text, language, lora_path)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
