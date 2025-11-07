"""
LOCAL TEST: Whisper Base vs Whisper + LoRA
Quick comparison on your local audio files
"""
import torch
import os
import glob
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("🧪 LOCAL WHISPER VS LORA TEST")
print("=" * 80 + "\n")

# Check CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Device: {device}")
if device == "cpu":
    print("⚠️  Running on CPU - this will be slower!")
print()

# Find audio files
audio_patterns = ["*.mp3", "*.m4a", "*.wav"]
audio_files = []
search_paths = [
    "services/asr",
    "services/asr/data",
    ".",
]

print("🔍 Looking for audio files...")
for path in search_paths:
    for pattern in audio_patterns:
        found = glob.glob(os.path.join(path, pattern))
        audio_files.extend(found)
        found_recursive = glob.glob(os.path.join(path, "**", pattern), recursive=True)
        audio_files.extend(found_recursive)

audio_files = list(set(audio_files))[:5]  # Unique, max 5 files

if not audio_files:
    print("❌ No audio files found!")
    print("\nPlease add some Arabic medical audio files to:")
    print("  - services/asr/")
    print("  - services/asr/data/")
    print("\nSupported formats: .mp3, .m4a, .wav")
    exit(1)

print(f"✅ Found {len(audio_files)} audio file(s):")
for f in audio_files:
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f"   - {os.path.basename(f)} ({size_mb:.1f} MB)")
print()

# Load base Whisper
print("📦 Loading Whisper large-v3 (this may take a few minutes)...")
from transformers import pipeline

try:
    base_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        device=0 if device == "cuda" else -1,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        chunk_length_s=30,
        return_timestamps=False,
    )
    print("✅ Base Whisper loaded\n")
except Exception as e:
    print(f"❌ Failed to load Whisper: {e}")
    exit(1)

# Try loading LoRA
has_lora = False
try:
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import librosa
    import soundfile as sf
    
    lora_path = "./services/asr/lora_ckpt"
    if not os.path.exists(lora_path):
        print(f"⚠️  LoRA path not found: {lora_path}")
        print("   Testing base Whisper only!\n")
    else:
        print("📦 Loading LoRA adapters...")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        lora_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        print("✅ LoRA model loaded\n")
        has_lora = True
except Exception as e:
    print(f"⚠️  Could not load LoRA: {e}")
    print("   Testing base Whisper only!\n")

# Test each audio file
print("=" * 80)
print("🎤 TRANSCRIPTION COMPARISON")
print("=" * 80 + "\n")

for i, audio_file in enumerate(audio_files, 1):
    filename = os.path.basename(audio_file)
    print(f"\n{'='*80}")
    print(f"📁 File {i}/{len(audio_files)}: {filename}")
    print('='*80 + "\n")
    
    try:
        # Base Whisper transcription
        print("🔵 Base Whisper:")
        base_result = base_pipe(audio_file, generate_kwargs={"language": "arabic"})
        base_text = base_result["text"].strip()
        print(f"   {base_text}\n")
        
        # LoRA transcription (if available)
        if has_lora:
            print("🟢 LoRA Whisper:")
            try:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=16000)
                input_features = lora_processor(
                    audio, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(device)
                
                # Generate
                with torch.no_grad():
                    predicted_ids = lora_model.generate(
                        input_features,
                        language="ar",
                        task="transcribe",
                    )
                lora_text = lora_processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0].strip()
                print(f"   {lora_text}\n")
                
                # Compare
                if base_text == lora_text:
                    print("⚠️  IDENTICAL - LoRA made NO difference!")
                else:
                    print("✅ DIFFERENT - LoRA is changing output")
                    
                    # Word-level diff
                    base_words = base_text.split()
                    lora_words = lora_text.split()
                    
                    if len(base_words) == len(lora_words):
                        diffs = []
                        for j, (b, l) in enumerate(zip(base_words, lora_words)):
                            if b != l:
                                diffs.append(f"      Word {j+1}: '{b}' → '{l}'")
                        if diffs:
                            print("\n   📝 Word differences:")
                            for d in diffs[:10]:  # Show first 10
                                print(d)
                    else:
                        print(f"\n   📝 Length difference: {len(base_words)} → {len(lora_words)} words")
                
            except Exception as e:
                print(f"   ❌ LoRA failed: {e}")
        
    except Exception as e:
        print(f"❌ Failed to process: {e}")
    
    print()

# Final verdict
print("\n" + "=" * 80)
print("🎯 FINAL VERDICT")
print("=" * 80 + "\n")

if not has_lora:
    print("❌ Could not test LoRA adapters")
    print("\nReasons:")
    print("  1. LoRA path not found (./services/asr/lora_ckpt)")
    print("  2. Missing dependencies (peft, librosa, soundfile)")
    print("  3. LoRA files corrupted")
    print("\nTo install dependencies:")
    print("  pip install peft librosa soundfile")
else:
    print("✅ Test completed!")
    print("\nNext steps:")
    print("  1. Review the transcriptions above")
    print("  2. If IDENTICAL → LoRA is useless, remove it")
    print("  3. If DIFFERENT → Check if medical terms are more accurate")
    print("  4. Consider testing on more files for statistical significance")

print("\n" + "=" * 80 + "\n")
