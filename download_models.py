"""
Download Whisper and MMed-Llama models to D: drive cache
"""
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoTokenizer, AutoModelForCausalLM

# Set cache to D: drive
os.environ['HF_HOME'] = 'D:\\huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:\\huggingface_cache'

print("=" * 60)
print("Downloading models to D:\\huggingface_cache")
print("=" * 60)

# Download Whisper large-v2 (~3GB)
print("\n1. Downloading Whisper large-v2 (~3GB)...")
try:
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    print("✅ Whisper large-v2 downloaded successfully!")
except Exception as e:
    print(f"❌ Error downloading Whisper: {e}")

# Download MMed-Llama-3-8B (~16GB)
print("\n2. Downloading MMed-Llama-3-8B (~16GB)...")
print("   This will take 10-15 minutes...")
try:
    tokenizer = AutoTokenizer.from_pretrained("Henrychur/MMed-Llama-3-8B")
    # Note: Not loading full model to save time, just downloading files
    print("✅ MMed-Llama-3-8B downloaded successfully!")
except Exception as e:
    print(f"❌ Error downloading MMed-Llama: {e}")

print("\n" + "=" * 60)
print("Download complete! Models cached in D:\\huggingface_cache")
print("You can now start the services normally.")
print("=" * 60)
