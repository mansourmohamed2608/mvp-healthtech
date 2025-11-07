#!/usr/bin/env python3
"""
COMPLETE WHISPERX TEST with ALL Features
Tests WhisperX with:
- Word-level timestamps (alignment)
- Speaker diarization
- VAD (Voice Activity Detection)
"""
import sys
import time
from pathlib import Path
import json
from typing import Dict
import torch
import os

def test_whisperx_complete(
    audio_path: str,
    reference_text: str = None,
    language: str = "ar",
    hf_token: str = None
):
    """Test WhisperX with ALL features enabled"""
    
    import whisperx
    import librosa
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # GTX 1050 doesn't support float16 in CTranslate2 - use int8
    compute_type = "int8"
    
    print("\n" + "=" * 80)
    print("COMPLETE WHISPERX TEST - ALL FEATURES")
    print("=" * 80)
    print(f"\n📁 Audio: {audio_path}")
    print(f"🌍 Language: {language}")
    print(f"💻 Device: {device}")
    print(f"🔢 Compute: {compute_type}")
    if reference_text:
        print(f"📊 Reference: {len(reference_text.split())} words")
    print()
    
    # Load audio
    print("📂 Loading audio...")
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    print(f"✓ Audio loaded: {duration:.1f}s")
    print()
    
    # STEP 1: Transcription
    print("=" * 80)
    print("STEP 1: TRANSCRIPTION (WhisperX)")
    print("=" * 80)
    print("Loading WhisperX large-v3...")
    start_load = time.time()
    
    model = whisperx.load_model(
        "large-v3",
        device,
        compute_type=compute_type,
        language=language
    )
    
    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.1f}s")
    
    print("\nTranscribing...")
    start_transcribe = time.time()
    
    result = model.transcribe(
        audio,
        language=language,
        batch_size=16  # Larger batch for speed
    )
    
    transcribe_time = time.time() - start_transcribe
    
    print(f"✓ Transcription complete in {transcribe_time:.1f}s")
    print(f"  RTF: {transcribe_time/duration:.2f}x")
    print(f"  Segments: {len(result['segments'])}")
    print(f"\n📝 Transcription:")
    print(f"  {result['text'][:200]}...")
    
    # Calculate WER if reference provided
    if reference_text:
        try:
            import jiwer
            wer = jiwer.wer(reference_text, result['text'])
            print(f"\n📊 WER: {wer:.2%}")
        except:
            pass
    
    # Clean up model to free VRAM
    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\n🧹 Cleared transcription model from memory")
    
    # STEP 2: Alignment (Word-level timestamps)
    print("\n" + "=" * 80)
    print("STEP 2: ALIGNMENT (Word-level Timestamps)")
    print("=" * 80)
    print("Loading alignment model...")
    start_align = time.time()
    
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
            model_name="jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
        )
        
        align_load_time = time.time() - start_align
        print(f"✓ Alignment model loaded in {align_load_time:.1f}s")
        
        print("\nAligning words...")
        start_align_process = time.time()
        
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )
        
        align_time = time.time() - start_align_process
        print(f"✓ Alignment complete in {align_time:.1f}s")
        
        # Show sample word timestamps
        print("\n📍 Sample word timestamps (first 10 words):")
        word_count = 0
        for seg in result['segments']:
            if 'words' in seg:
                for word in seg['words']:
                    print(f"  {word['word']:<15} [{word['start']:.2f}s - {word['end']:.2f}s]")
                    word_count += 1
                    if word_count >= 10:
                        break
            if word_count >= 10:
                break
        
        # Clean up
        del model_a
        gc.collect()
        torch.cuda.empty_cache()
        print("\n🧹 Cleared alignment model from memory")
        
    except Exception as e:
        print(f"⚠️  Alignment failed: {e}")
        align_time = 0
    
    # STEP 3: Diarization (Speaker Detection)
    print("\n" + "=" * 80)
    print("STEP 3: DIARIZATION (Speaker Detection)")
    print("=" * 80)
    
    if not hf_token:
        print("⚠️  No HuggingFace token provided!")
        print("   Skipping diarization (requires: pyannote/speaker-diarization-3.1)")
        print("\nTo enable diarization:")
        print("  1. Get token from: https://huggingface.co/settings/tokens")
        print("  2. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("  3. Run with: python script.py audio.mp3 ref.txt ar YOUR_TOKEN")
        diarize_time = 0
    else:
        print("Loading diarization model...")
        start_diarize = time.time()
        
        try:
            from pyannote.audio import Pipeline
            
            diarize_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            if device != "cpu":
                diarize_model.to(torch.device(device))
            
            diarize_load_time = time.time() - start_diarize
            print(f"✓ Diarization model loaded in {diarize_load_time:.1f}s")
            
            print("\nDetecting speakers...")
            start_diarize_process = time.time()
            
            diarize_segments = diarize_model(audio_path)
            
            diarize_time = time.time() - start_diarize_process
            print(f"✓ Diarization complete in {diarize_time:.1f}s")
            
            # Assign speakers to words
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Count speakers
            speakers = set()
            for seg in result['segments']:
                if 'speaker' in seg:
                    speakers.add(seg['speaker'])
            
            print(f"\n👥 Detected {len(speakers)} speaker(s): {', '.join(sorted(speakers))}")
            
            # Show sample with speakers
            print("\n🗣️  Sample transcription with speakers:")
            for i, seg in enumerate(result['segments'][:5]):
                speaker = seg.get('speaker', 'Unknown')
                text = seg['text']
                print(f"  [{seg['start']:.1f}s-{seg['end']:.1f}s] {speaker}: {text}")
            
            # Clean up
            del diarize_model
            gc.collect()
            torch.cuda.empty_cache()
            print("\n🧹 Cleared diarization model from memory")
            
        except Exception as e:
            print(f"⚠️  Diarization failed: {e}")
            diarize_time = 0
    
    # SUMMARY
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    total_time = transcribe_time + align_time + diarize_time
    
    print(f"\n⏱️  Timing Breakdown:")
    print(f"  Audio duration:     {duration:.1f}s")
    print(f"  Transcription:      {transcribe_time:.1f}s ({transcribe_time/duration:.2f}x RTF)")
    if align_time > 0:
        print(f"  Alignment:          {align_time:.1f}s ({align_time/duration:.2f}x RTF)")
    if diarize_time > 0:
        print(f"  Diarization:        {diarize_time:.1f}s ({diarize_time/duration:.2f}x RTF)")
    print(f"  ──────────────────────────────────")
    print(f"  TOTAL:              {total_time:.1f}s ({total_time/duration:.2f}x RTF)")
    
    print(f"\n📊 Output Features:")
    print(f"  ✅ Full transcription")
    print(f"  {'✅' if align_time > 0 else '❌'} Word-level timestamps")
    print(f"  {'✅' if diarize_time > 0 else '❌'} Speaker diarization")
    print(f"  Total segments: {len(result['segments'])}")
    
    # Count words with timestamps
    word_count = sum(len(seg.get('words', [])) for seg in result['segments'])
    print(f"  Total words with timestamps: {word_count}")
    
    # Save results
    output_file = "whisperx_complete_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Full results saved to: {output_file}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR YOUR HARDWARE (GTX 1050 3GB)")
    print("=" * 80)
    
    print(f"\n🎯 Performance Analysis:")
    if total_time / duration > 1.5:
        print(f"  ⚠️  Processing is SLOW ({total_time/duration:.1f}x RTF)")
        print(f"  Your GPU is struggling with large-v3 model")
        print(f"\n💡 Try using 'medium' model instead:")
        print(f"  - Faster: ~2-3x speed improvement")
        print(f"  - Smaller: 1.5GB vs 3GB")
        print(f"  - Still accurate for medical Arabic")
    elif total_time / duration > 0.8:
        print(f"  ✅ Processing is ACCEPTABLE ({total_time/duration:.1f}x RTF)")
        print(f"  You can use large-v3, but consider 'medium' for production")
    else:
        print(f"  ✅ Processing is FAST ({total_time/duration:.1f}x RTF)")
        print(f"  Your setup is working well!")
    
    if diarize_time > transcribe_time:
        print(f"\n⚠️  Diarization is your BIGGEST bottleneck:")
        print(f"  - Takes {diarize_time:.1f}s ({diarize_time/total_time*100:.0f}% of total)")
        print(f"  - Consider disabling if you don't need speaker detection")
    
    print("\n" + "=" * 80 + "\n")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_whisperx_complete.py <audio_file> [reference_text_file] [language] [hf_token]")
        print("\nExample:")
        print("  python test_whisperx_complete.py test1.mp3 reference.txt ar YOUR_HF_TOKEN")
        print("\nFeatures tested:")
        print("  ✅ Transcription (WhisperX)")
        print("  ✅ Word-level timestamps (alignment)")
        print("  ✅ Speaker diarization (if HF token provided)")
        print("\nGet HF token:")
        print("  1. Visit: https://huggingface.co/settings/tokens")
        print("  2. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    reference_file = sys.argv[2] if len(sys.argv) > 2 else None
    language = sys.argv[3] if len(sys.argv) > 3 else "ar"
    hf_token = sys.argv[4] if len(sys.argv) > 4 else os.getenv("HF_TOKEN")
    
    if not Path(audio_file).exists():
        print(f"❌ Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    reference_text = None
    if reference_file and Path(reference_file).exists():
        with open(reference_file, "r", encoding="utf-8") as f:
            reference_text = f.read().strip()
    
    try:
        test_whisperx_complete(audio_file, reference_text, language, hf_token)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
