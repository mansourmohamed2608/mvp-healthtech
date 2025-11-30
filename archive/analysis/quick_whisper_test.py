"""
QUICK TEST: Whisper Base vs Whisper + LoRA
Simple comparison on your audio files
"""
import torch
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("QUICK WHISPER VS LORA TEST")
print("=" * 80)
print()

# Check CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print()

# Load Whisper base model (simple approach)
print("Loading Whisper large-v3...")
from transformers import pipeline

# Base Whisper pipeline
base_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=0 if device == "cuda" else -1,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
print("✅ Base Whisper loaded")
print()

# Try loading LoRA version
try:
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    
    print("Loading LoRA adapters...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    lora_model = PeftModel.from_pretrained(base_model, "./services/asr/lora_ckpt")
    lora_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    print("✅ LoRA model loaded")
    print()
    has_lora = True
except Exception as e:
    print(f"⚠️  Could not load LoRA: {e}")
    print("Will only test base Whisper")
    print()
    has_lora = False

# Test files
test_files = [
    "test1.mp3",
    "test1.m4a", 
    "التهاب اللثه.m4a"
]

print("=" * 80)
print("TESTING AUDIO FILES")
print("=" * 80)
print()

for audio_file in test_files:
    import os
    if not os.path.exists(audio_file):
        continue
        
    print(f"📁 File: {audio_file}")
    print()
    
    # Test 1: Base Whisper (easy pipeline)
    print("🎤 Transcribing with BASE Whisper...")
    base_result = base_pipe(
        audio_file,
        generate_kwargs={"language": "arabic", "task": "transcribe"}
    )
    base_text = base_result['text'].strip()
    print(f"✅ Base: {base_text}")
    print()
    
    # Test 2: LoRA Whisper (if available)
    if has_lora:
        print("🎤 Transcribing with LoRA Whisper...")
        import librosa
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        inputs = lora_processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        with torch.no_grad():
            predicted_ids = lora_model.generate(
                input_features=input_features,
                language="ar",
                task="transcribe",
            )
        lora_text = lora_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        print(f"✅ LoRA: {lora_text}")
        print()
        
        # Compare
        if base_text == lora_text:
            print("⚠️  IDENTICAL - LoRA made NO difference!")
        else:
            print("✅ Different transcriptions - LoRA IS affecting output")
            
            # Show differences
            base_words = base_text.split()
            lora_words = lora_text.split()
            
            if len(base_words) != len(lora_words):
                print(f"   Word count: Base={len(base_words)}, LoRA={len(lora_words)}")
            
            # Find different words
            for i, (b, l) in enumerate(zip(base_words, lora_words)):
                if b != l:
                    print(f"   Diff at word {i+1}: '{b}' → '{l}'")
        print()
    
    print("-" * 80)
    print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if not has_lora:
    print("ℹ️  Could not test LoRA adapters")
    print("Recommendation: Use base Whisper-large-v3 (it's already excellent)")
else:
    print("Review the transcriptions above:")
    print()
    print("If IDENTICAL:")
    print("  ❌ LoRA adapters are NOT helping")
    print("  💡 Use base Whisper instead")
    print()
    print("If DIFFERENT but WORSE:")
    print("  ❌ LoRA adapters make it worse (trained on synthetic TTS?)")
    print("  💡 Use base Whisper instead")
    print()
    print("If DIFFERENT and BETTER:")
    print("  ✅ Keep using LoRA adapters!")

print()
print("💡 My opinion: If Salma trained on TTS/synthetic data,")
print("   the LoRA adapters are probably NOT helpful for real audio.")
print("   Base Whisper-large-v3 is already excellent for Arabic medical!")
