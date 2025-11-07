#!/usr/bin/env python3
"""
3-WAY WER COMPARISON: Whisper vs WhisperX vs WhisperX+LoRA
Complete comparison of all ASR variants
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
        reference = reference.strip()
        hypothesis = hypothesis.strip()
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


def transcribe_with_whisper(audio_path: str, language: str = "ar") -> Dict:
    """1. Base Whisper (transformers)"""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import librosa
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"  Loading Whisper large-v3...")
    start_load = time.time()
    
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        local_files_only=True
    )
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-large-v3",
        local_files_only=True
    )
    
    load_time = time.time() - start_load
    print(f"  ✓ Loaded in {load_time:.1f}s")
    
    print(f"  Transcribing...")
    start_time = time.time()
    
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    
    input_features = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features
    
    if device == "cuda":
        input_features = input_features.to(device).to(torch.float16)
    else:
        input_features = input_features.to(device)
    
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


def transcribe_with_whisperx(audio_path: str, language: str = "ar") -> Dict:
    """2. WhisperX (faster-whisper with alignment)"""
    import whisperx
    import librosa
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # GTX 1050 doesn't support float16 in CTranslate2 - use int8
    compute_type = "int8"
    
    print(f"  Loading WhisperX large-v3...")
    start_load = time.time()
    
    model = whisperx.load_model(
        "large-v3",
        device,
        compute_type=compute_type,
        language=language
    )
    
    load_time = time.time() - start_load
    print(f"  ✓ Loaded in {load_time:.1f}s")
    
    print(f"  Transcribing...")
    start_time = time.time()
    
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    
    result = model.transcribe(audio, language=language)
    text = result["text"].strip()
    
    processing_time = time.time() - start_time
    
    return {
        "text": text,
        "processing_time": processing_time,
        "duration": duration,
        "rtf": processing_time / duration if duration > 0 else 0,
        "load_time": load_time,
        "segments": len(result.get("segments", []))
    }


def transcribe_with_whisperx_lora(
    audio_path: str, 
    lora_path: str,
    language: str = "ar"
) -> Dict:
    """3. WhisperX + LoRA hybrid approach"""
    import whisperx
    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # GTX 1050 doesn't support float16 in CTranslate2 - use int8
    compute_type = "int8"
    
    print(f"  Loading WhisperX + LoRA...")
    start_load = time.time()
    
    # Load WhisperX for alignment
    whisperx_model = whisperx.load_model(
        "large-v3",
        device,
        compute_type=compute_type,
        language=language
    )
    
    # Load LoRA model
    base_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        local_files_only=True
    )
    lora_model = PeftModel.from_pretrained(
        base_model, 
        lora_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    lora_model.eval()
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-large-v3",
        local_files_only=True
    )
    
    load_time = time.time() - start_load
    print(f"  ✓ Loaded in {load_time:.1f}s")
    
    print(f"  Transcribing...")
    start_time = time.time()
    
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    
    # Get segments from WhisperX
    whisperx_result = whisperx_model.transcribe(audio, language=language)
    
    # Transcribe with LoRA
    input_features = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features
    
    if device == "cuda":
        input_features = input_features.to(device).to(torch.float16)
    else:
        input_features = input_features.to(device)
    
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
        "load_time": load_time,
        "segments": len(whisperx_result.get("segments", []))
    }


def compare_all_models(
    audio_path: str,
    reference_text: str,
    language: str = "ar",
    lora_path: str = "./services/asr/lora_ckpt"
):
    """Compare all three ASR approaches"""
    
    print("\n" + "=" * 80)
    print("3-WAY ASR COMPARISON: Whisper vs WhisperX vs WhisperX+LoRA")
    print("=" * 80)
    print(f"\n📁 Audio file: {audio_path}")
    print(f"🌍 Language: {language}")
    print(f"📊 Reference text length: {len(reference_text.split())} words")
    print(f"💻 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    results = {}
    
    # Test 1: Base Whisper
    print("=" * 80)
    print("TEST 1: BASE WHISPER (transformers)")
    print("=" * 80)
    try:
        result_whisper = transcribe_with_whisper(audio_path, language)
        
        print(f"\n✓ Complete")
        print(f"  Duration: {result_whisper['duration']:.2f}s")
        print(f"  Processing: {result_whisper['processing_time']:.2f}s")
        print(f"  RTF: {result_whisper['rtf']:.2f}x")
        print(f"\n📝 Text: {result_whisper['text'][:100]}...")
        
        wer_whisper = calculate_wer(reference_text, result_whisper['text'])
        if wer_whisper:
            print(f"\n📊 WER: {wer_whisper['wer']:.2%}")
        
        results['whisper'] = {
            'transcription': result_whisper,
            'wer_metrics': wer_whisper
        }
    except Exception as e:
        print(f"\n❌ Error: {e}")
        results['whisper'] = {'error': str(e)}
    
    print()
    
    # Test 2: WhisperX
    print("=" * 80)
    print("TEST 2: WHISPERX (faster-whisper + alignment)")
    print("=" * 80)
    try:
        result_whisperx = transcribe_with_whisperx(audio_path, language)
        
        print(f"\n✓ Complete")
        print(f"  Duration: {result_whisperx['duration']:.2f}s")
        print(f"  Processing: {result_whisperx['processing_time']:.2f}s")
        print(f"  RTF: {result_whisperx['rtf']:.2f}x")
        print(f"  Segments: {result_whisperx['segments']}")
        print(f"\n📝 Text: {result_whisperx['text'][:100]}...")
        
        wer_whisperx = calculate_wer(reference_text, result_whisperx['text'])
        if wer_whisperx:
            print(f"\n📊 WER: {wer_whisperx['wer']:.2%}")
        
        results['whisperx'] = {
            'transcription': result_whisperx,
            'wer_metrics': wer_whisperx
        }
    except Exception as e:
        print(f"\n❌ Error: {e}")
        results['whisperx'] = {'error': str(e)}
    
    print()
    
    # Test 3: WhisperX + LoRA
    print("=" * 80)
    print("TEST 3: WHISPERX + LoRA (hybrid)")
    print("=" * 80)
    
    if not Path(lora_path).exists():
        print(f"\n⚠️  LoRA path not found: {lora_path}")
        results['whisperx_lora'] = {'error': 'LoRA path not found'}
    else:
        try:
            result_lora = transcribe_with_whisperx_lora(audio_path, lora_path, language)
            
            print(f"\n✓ Complete")
            print(f"  Duration: {result_lora['duration']:.2f}s")
            print(f"  Processing: {result_lora['processing_time']:.2f}s")
            print(f"  RTF: {result_lora['rtf']:.2f}x")
            print(f"  Segments: {result_lora['segments']}")
            print(f"\n📝 Text: {result_lora['text'][:100]}...")
            
            wer_lora = calculate_wer(reference_text, result_lora['text'])
            if wer_lora:
                print(f"\n📊 WER: {wer_lora['wer']:.2%}")
            
            results['whisperx_lora'] = {
                'transcription': result_lora,
                'wer_metrics': wer_lora
            }
        except Exception as e:
            print(f"\n❌ Error: {e}")
            results['whisperx_lora'] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    # Create comparison table
    models = []
    if 'error' not in results.get('whisper', {}):
        models.append(('Whisper', results['whisper']))
    if 'error' not in results.get('whisperx', {}):
        models.append(('WhisperX', results['whisperx']))
    if 'error' not in results.get('whisperx_lora', {}):
        models.append(('WhisperX+LoRA', results['whisperx_lora']))
    
    if len(models) >= 2:
        print(f"\n📊 WER Comparison:")
        print(f"  {'Model':<20} {'WER':<12} {'Time':<10} {'RTF':<10}")
        print(f"  {'-'*52}")
        
        for name, data in models:
            wer = data['wer_metrics']
            trans = data['transcription']
            if wer:
                print(f"  {name:<20} {wer['wer']:>10.2%}  {trans['processing_time']:>8.1f}s  {trans['rtf']:>8.2f}x")
        
        # Find best model
        best_wer = min(models, key=lambda x: x[1]['wer_metrics']['wer'] if x[1]['wer_metrics'] else float('inf'))
        fastest = min(models, key=lambda x: x[1]['transcription']['processing_time'])
        
        print(f"\n🏆 Results:")
        print(f"  Best WER: {best_wer[0]} ({best_wer[1]['wer_metrics']['wer']:.2%})")
        print(f"  Fastest:  {fastest[0]} ({fastest[1]['transcription']['processing_time']:.1f}s)")
        
        # Detailed comparison
        print(f"\n🎯 Error Breakdown:")
        print(f"  {'Model':<20} {'Subst':<8} {'Del':<8} {'Ins':<8} {'Hits':<8}")
        print(f"  {'-'*52}")
        
        for name, data in models:
            wer = data['wer_metrics']
            if wer:
                print(f"  {name:<20} {wer['substitutions']:<8} {wer['deletions']:<8} {wer['insertions']:<8} {wer['hits']:<8}")
        
        # Recommendation
        print(f"\n📋 Recommendation:")
        
        if len(models) == 3:
            whisper_wer = results['whisper']['wer_metrics']['wer']
            whisperx_wer = results['whisperx']['wer_metrics']['wer']
            lora_wer = results['whisperx_lora']['wer_metrics']['wer']
            
            if lora_wer <= whisperx_wer and lora_wer <= whisper_wer:
                print(f"  ✅ LoRA provides best accuracy!")
                print(f"  Use: WhisperX + LoRA")
            elif whisperx_wer < whisper_wer:
                print(f"  ✅ WhisperX is better than base Whisper")
                if lora_wer > whisperx_wer:
                    diff = ((lora_wer - whisperx_wer) / whisperx_wer * 100)
                    print(f"  ❌ LoRA worsens WER by {diff:.1f}% - DON'T USE IT")
                print(f"  Use: WhisperX (no LoRA)")
            else:
                print(f"  Use: Best model is {best_wer[0]}")
    
    # Save results
    output_file = "all_asr_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_all_wer.py <audio_file> <reference_text_file> [language] [lora_path]")
        print("\nExample:")
        print("  python compare_all_wer.py test1.mp3 reference_test1.txt ar")
        print("\nTests:")
        print("  1. Base Whisper (transformers)")
        print("  2. WhisperX (faster-whisper + alignment)")
        print("  3. WhisperX + LoRA (hybrid approach)")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    reference_file = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) > 3 else "ar"
    lora_path = sys.argv[4] if len(sys.argv) > 4 else "./services/asr/lora_ckpt"
    
    if not Path(audio_file).exists():
        print(f"❌ Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    if not Path(reference_file).exists():
        print(f"❌ Error: Reference text file not found: {reference_file}")
        sys.exit(1)
    
    with open(reference_file, "r", encoding="utf-8") as f:
        reference_text = f.read().strip()
    
    if not reference_text:
        print(f"❌ Error: Reference text file is empty")
        sys.exit(1)
    
    try:
        compare_all_models(audio_file, reference_text, language, lora_path)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
