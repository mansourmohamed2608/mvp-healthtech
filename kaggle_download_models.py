"""
Pre-download Models for Kaggle
Run this first to download and cache all models
Then run the main pipeline without waiting for downloads
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("MODEL DOWNLOAD SCRIPT FOR KAGGLE")
print("Pre-download all models to speed up pipeline execution")
print("=" * 80)

# Configuration
MODEL_CACHE = "/kaggle/working/models"
os.makedirs(MODEL_CACHE, exist_ok=True)

os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE
os.environ['HF_HOME'] = MODEL_CACHE
os.environ['HF_HUB_CACHE'] = MODEL_CACHE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  No GPU detected")

print(f"\n📁 Cache directory: {MODEL_CACHE}")
print()

# ============================================================================
# DOWNLOAD WHISPER
# ============================================================================

print("=" * 80)
print("STEP 1: Download WhisperX Large-v3")
print("=" * 80)
print("Size: ~3GB")
print("Time: ~2-5 minutes")
print()

try:
    print("📥 Downloading WhisperX large-v3...")
    start = time.time()
    
    import whisperx
    
    # This will download the model
    model = whisperx.load_model(
        "large-v3",
        device=DEVICE,
        compute_type="float16" if DEVICE == "cuda" else "int8",
        language="ar",
        download_root=MODEL_CACHE
    )
    
    elapsed = time.time() - start
    print(f"✅ WhisperX large-v3 downloaded successfully in {elapsed:.1f}s!")
    
    # Clean up to free memory
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ Error downloading WhisperX: {e}")

print()

# ============================================================================
# DOWNLOAD ALIGNMENT MODEL
# ============================================================================

print("=" * 80)
print("STEP 2: Download WhisperX Alignment Model (Arabic)")
print("=" * 80)
print("Size: ~300MB")
print("Time: ~30 seconds")
print()

try:
    print("📥 Downloading alignment model...")
    start = time.time()
    
    model_a, metadata = whisperx.load_align_model(
        language_code="ar",
        device=DEVICE
    )
    
    elapsed = time.time() - start
    print(f"✅ Alignment model downloaded successfully in {elapsed:.1f}s!")
    
    # Clean up
    del model_a
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ Error downloading alignment model: {e}")

print()

# ============================================================================
# DOWNLOAD LLM
# ============================================================================

print("=" * 80)
print("STEP 3: Download MMed-Llama-3-8B")
print("=" * 80)

if DEVICE == "cuda":
    print("Size: ~8GB (4-bit quantization)")
    print("Time: ~5-10 minutes")
else:
    print("Size: ~16GB (8-bit quantization)")
    print("Time: ~10-20 minutes")

print()

try:
    print("📥 Downloading MMed-Llama-3-8B...")
    print("⏳ This may take a while, please be patient...")
    start = time.time()
    
    model_name = "Henrychur/MMed-Llama-3-8B"
    
    # Download tokenizer first (fast)
    print("   1/2 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE
    )
    print("   ✅ Tokenizer downloaded")
    
    # Download model (slow)
    print("   2/2 Downloading model weights...")
    
    from transformers import BitsAndBytesConfig
    
    if DEVICE == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=MODEL_CACHE
    )
    
    elapsed = time.time() - start
    print(f"✅ MMed-Llama-3-8B downloaded successfully in {elapsed:.1f}s ({elapsed/60:.1f} minutes)!")
    
    # Clean up
    del model
    del tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ Error downloading MMed-Llama: {e}")

print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)
print()
print("✅ All models have been downloaded and cached!")
print(f"📁 Cache location: {MODEL_CACHE}")
print()
print("📊 Disk usage:")
os.system(f"du -sh {MODEL_CACHE}")
print()
print("🚀 Next step: Run the main pipeline script")
print("   The models will load much faster from cache!")
print("=" * 80)
