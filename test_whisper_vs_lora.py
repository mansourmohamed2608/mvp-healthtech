"""
Test Whisper (base) vs Whisper + LoRA Adapters
Compare WER and transcription quality on real medical audio
"""
import torch
import time
import json
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import librosa
import soundfile as sf
from jiwer import wer, cer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = "openai/whisper-large-v3"
LORA_ADAPTER_PATH = "./services/asr/lora_ckpt"  # Your LoRA adapters

# Test audio files (add your real medical audio files here)
TEST_AUDIO_FILES = [
    "test1.mp3",
    "test1.m4a",
    "التهاب اللثه.m4a",
    # Add more real medical audio files
]

# Ground truth transcriptions (if available)
# Format: {"filename": "exact transcription"}
GROUND_TRUTH = {
    # "test1.mp3": "النص الصحيح هنا",
    # Add ground truth if you have it
}

# ============================================================================
# LOAD MODELS
# ============================================================================
print("=" * 80)
print("LOADING MODELS")
print("=" * 80)
print()

print("📥 Loading Whisper base model...")
base_model = WhisperForConditionalGeneration.from_pretrained(
    WHISPER_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE,
)
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
print(f"✅ Base model loaded on {DEVICE}")
print()

# Try to load LoRA adapters
lora_model = None
if Path(LORA_ADAPTER_PATH).exists():
    print(f"📥 Loading LoRA adapters from: {LORA_ADAPTER_PATH}")
    try:
        lora_model = PeftModel.from_pretrained(
            base_model,
            LORA_ADAPTER_PATH,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        lora_model.eval()
        print("✅ LoRA model loaded!")
        print()
    except Exception as e:
        print(f"❌ Failed to load LoRA adapters: {e}")
        print("Will only test base Whisper model")
        print()
else:
    print(f"⚠️  LoRA adapters not found at: {LORA_ADAPTER_PATH}")
    print("Will only test base Whisper model")
    print()

# ============================================================================
# TRANSCRIPTION FUNCTION
# ============================================================================
def transcribe_audio(audio_path, model, model_name="Model"):
    """Transcribe audio file and return text + metrics"""
    print(f"🎤 Transcribing with {model_name}...")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    
    # Process audio
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )
    input_features = inputs.input_features.to(DEVICE)
    
    # Generate transcription
    start_time = time.time()
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features=input_features,
            language="ar",
            task="transcribe",
            max_new_tokens=448,  # ~30s of audio
        )
    
    processing_time = time.time() - start_time
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    
    # Calculate RTF (Real-Time Factor)
    rtf = processing_time / duration if duration > 0 else 0
    
    return {
        "text": transcript,
        "duration": duration,
        "processing_time": processing_time,
        "rtf": rtf,
        "model": model_name
    }

# ============================================================================
# RUN TESTS
# ============================================================================
results = []

print("=" * 80)
print("RUNNING TRANSCRIPTION TESTS")
print("=" * 80)
print()

for audio_file in TEST_AUDIO_FILES:
    audio_path = Path(audio_file)
    
    if not audio_path.exists():
        print(f"⚠️  Skipping {audio_file} (not found)")
        print()
        continue
    
    print("=" * 80)
    print(f"📁 Testing: {audio_file}")
    print("=" * 80)
    print()
    
    # Test 1: Base Whisper
    base_result = transcribe_audio(audio_path, base_model, "Whisper (Base)")
    print(f"✅ Base Whisper:")
    print(f"   Text: {base_result['text'][:100]}...")
    print(f"   Duration: {base_result['duration']:.2f}s")
    print(f"   Processing: {base_result['processing_time']:.2f}s")
    print(f"   RTF: {base_result['rtf']:.3f}x")
    print()
    
    # Test 2: Whisper + LoRA (if available)
    lora_result = None
    if lora_model is not None:
        lora_result = transcribe_audio(audio_path, lora_model, "Whisper + LoRA")
        print(f"✅ Whisper + LoRA:")
        print(f"   Text: {lora_result['text'][:100]}...")
        print(f"   Duration: {lora_result['duration']:.2f}s")
        print(f"   Processing: {lora_result['processing_time']:.2f}s")
        print(f"   RTF: {lora_result['rtf']:.3f}x")
        print()
    
    # Calculate WER if ground truth available
    gt_text = GROUND_TRUTH.get(audio_file)
    if gt_text:
        base_wer = wer(gt_text, base_result['text'])
        base_cer = cer(gt_text, base_result['text'])
        print(f"📊 Base Whisper Accuracy:")
        print(f"   WER: {base_wer*100:.2f}%")
        print(f"   CER: {base_cer*100:.2f}%")
        print()
        
        if lora_result:
            lora_wer = wer(gt_text, lora_result['text'])
            lora_cer = cer(gt_text, lora_result['text'])
            print(f"📊 Whisper + LoRA Accuracy:")
            print(f"   WER: {lora_wer*100:.2f}%")
            print(f"   CER: {lora_cer*100:.2f}%")
            print()
            
            # Comparison
            wer_diff = (lora_wer - base_wer) * 100
            print(f"📈 Improvement:")
            if lora_wer < base_wer:
                print(f"   ✅ LoRA is BETTER by {abs(wer_diff):.2f}% WER")
            elif lora_wer > base_wer:
                print(f"   ❌ LoRA is WORSE by {abs(wer_diff):.2f}% WER")
            else:
                print(f"   ⚖️  No difference")
            print()
    
    # Compare transcriptions directly
    if lora_result:
        print("🔍 Direct Comparison:")
        print(f"   Base:  {base_result['text']}")
        print(f"   LoRA:  {lora_result['text']}")
        
        # Check if texts are identical
        if base_result['text'] == lora_result['text']:
            print(f"   ⚠️  IDENTICAL TRANSCRIPTIONS - LoRA may not be working!")
        else:
            # Count different words
            base_words = set(base_result['text'].split())
            lora_words = set(lora_result['text'].split())
            diff_count = len(base_words.symmetric_difference(lora_words))
            print(f"   Different words: {diff_count}")
        print()
    
    # Store results
    result_entry = {
        "file": audio_file,
        "base": base_result,
        "lora": lora_result,
        "ground_truth": gt_text
    }
    results.append(result_entry)

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

if lora_model is None:
    print("⚠️  LoRA adapters were NOT loaded. Only tested base Whisper.")
    print()
    print("Recommendations:")
    print("1. Check if LoRA adapters exist at: ./services/asr/lora_ckpt")
    print("2. If adapters were trained on synthetic TTS data, they may not help")
    print("3. Consider training on REAL medical audio recordings instead")
else:
    # Calculate average metrics
    identical_count = 0
    total_tests = len([r for r in results if r['lora'] is not None])
    
    for r in results:
        if r['lora'] and r['base']['text'] == r['lora']['text']:
            identical_count += 1
    
    print(f"📊 Test Results ({total_tests} files):")
    print(f"   Identical transcriptions: {identical_count}/{total_tests}")
    
    if identical_count == total_tests:
        print()
        print("⚠️  WARNING: LoRA adapters produced IDENTICAL results to base model!")
        print()
        print("Possible reasons:")
        print("1. LoRA adapters are not being applied correctly")
        print("2. LoRA was trained on synthetic TTS data (not real speech)")
        print("3. LoRA rank/alpha too small to make meaningful changes")
        print("4. Training data didn't include medical vocabulary improvements")
        print()
        print("Recommendation: DON'T USE these LoRA adapters")
        print("Use base Whisper-large-v3 instead - it's already very good!")
    else:
        print(f"   Different transcriptions: {total_tests - identical_count}/{total_tests}")
        print()
        print("✅ LoRA adapters ARE making changes to transcriptions")
        print()
        print("Next steps:")
        print("1. Manually review the transcriptions above")
        print("2. Check if LoRA improvements are meaningful")
        print("3. If trained on TTS data, consider retraining on real audio")

# Save results to JSON
output_file = "whisper_vs_lora_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print()
print(f"💾 Results saved to: {output_file}")
print()
print("=" * 80)
