"""
Kaggle Audio Processing Pipeline
Upload this as a Kaggle notebook and run with GPU enabled

IMPORTANT: Run this first in a separate cell:
!pip install -q --upgrade numpy scipy
!pip install -q whisperx torch transformers bitsandbytes accelerate
"""

import os
import sys
import time
import json

# Import with error handling
try:
    import torch
except ImportError:
    print("❌ PyTorch not found. Run: !pip install torch")
    sys.exit(1)

try:
    import whisperx
except ImportError:
    print("❌ WhisperX not found. Run: !pip install whisperx")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ImportError as e:
    print(f"❌ Transformers import error: {e}")
    print("Fix: !pip install -q --upgrade numpy scipy transformers")
    sys.exit(1)

from pathlib import Path

# ============================================================================
# KAGGLE SETUP
# ============================================================================

print("=" * 80)
print("KAGGLE AUDIO PROCESSING PIPELINE")
print("GPU-Accelerated Medical Audio Processing")
print("=" * 80)

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
else:
    print("⚠️  No GPU detected - will use CPU (very slow)")
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

print(f"Device: {DEVICE}")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths for Kaggle
INPUT_DIR = "/kaggle/input"  # Upload audio files here
WORKING_DIR = "/kaggle/working"  # Results saved here
MODEL_CACHE = "/kaggle/working/models"  # Cache models here

# Create directories
os.makedirs(MODEL_CACHE, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# Set cache directories
os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE
os.environ['HF_HOME'] = MODEL_CACHE

print(f"📁 Input directory: {INPUT_DIR}")
print(f"📁 Output directory: {WORKING_DIR}")
print(f"📁 Model cache: {MODEL_CACHE}")
print()

# ============================================================================
# STEP 1: Load Models
# ============================================================================

def load_asr_model():
    """Load WhisperX ASR model"""
    print("📥 Loading WhisperX model (large-v3)...")
    print("   First run: Downloading model (~3GB, 2-5 minutes)")
    print("   Subsequent runs: Loading from cache (~30s)")
    start = time.time()
    
    model = whisperx.load_model(
        "large-v3",
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        language="ar",
        download_root=MODEL_CACHE  # Use our cache directory
    )
    
    elapsed = time.time() - start
    print(f"✅ WhisperX loaded in {elapsed:.1f}s\n")
    return model

def load_llm_model():
    """Load LLM model with GPU optimization"""
    print("📥 Loading LLM model (MMed-Llama-3-8B)...")
    print("   Model size: ~8GB (4-bit) or ~16GB (full)")
    
    if DEVICE == "cuda":
        print("✅ Using GPU with 4-bit quantization")
        print("⏱️  First run: Downloading model (~8GB, 5-10 minutes)")
        print("⏱️  Subsequent runs: Loading from cache (~3-5 minutes)")
    else:
        print("⚠️  Using CPU with 8-bit quantization")
        print("⏱️  First run: Downloading model (~16GB, 10-20 minutes)")
        print("⏱️  Subsequent runs: Loading from cache (~13-20 minutes)")
    
    start = time.time()
    
    model_name = "Henrychur/MMed-Llama-3-8B"
    
    # GPU: 4-bit quantization, CPU: 8-bit
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=MODEL_CACHE
    )
    
    elapsed = time.time() - start
    print(f"✅ LLM loaded in {elapsed:.1f}s ({elapsed/60:.1f} minutes)\n")
    return model, tokenizer

# ============================================================================
# STEP 2: Process Audio with ASR
# ============================================================================

def transcribe_audio(audio_path, asr_model, dialect="egypt"):
    """Transcribe audio file with WhisperX"""
    print("=" * 80)
    print("STEP 1: ASR TRANSCRIPTION")
    print("=" * 80)
    print(f"Audio file: {audio_path}")
    print(f"Dialect: {dialect}")
    print()
    
    start = time.time()
    
    # Load audio
    print("📂 Loading audio...")
    audio = whisperx.load_audio(audio_path)
    audio_duration = len(audio) / 16000
    print(f"   Duration: {audio_duration:.1f}s")
    
    # Transcribe
    print("🎤 Transcribing...")
    result = asr_model.transcribe(
        audio,
        language="ar",
        batch_size=16
    )
    
    # Align timestamps
    print("🔍 Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(
        language_code="ar",
        device=DEVICE
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False
    )
    
    # Diarize (speaker detection)
    print("👥 Detecting speakers (diarization)...")
    try:
        # Try to load diarization pipeline
        from pyannote.audio import Pipeline
        
        # Note: Requires HuggingFace token
        # You can set it in Kaggle secrets or skip diarization
        hf_token = os.environ.get("HF_TOKEN", None)
        
        if hf_token:
            diarize_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            diarize_model.to(torch.device(DEVICE))
            
            diarize_segments = diarize_model(audio_path)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print("   ✅ Speaker diarization complete")
        else:
            print("   ⚠️  No HF_TOKEN found - skipping diarization")
            print("   💡 Add HF token to Kaggle secrets for speaker detection")
    except Exception as e:
        print(f"   ⚠️  Diarization failed: {e}")
        print("   Continuing without speaker labels...")
    
    elapsed = time.time() - start
    print(f"\n✅ ASR complete in {elapsed:.1f}s ({elapsed/audio_duration:.2f}x real-time)")
    print(f"   Segments: {len(result['segments'])}")
    
    # Extract full text
    full_text = " ".join([seg["text"] for seg in result["segments"]])
    print(f"   Transcription: {full_text[:100]}...")
    
    return result, full_text

# ============================================================================
# STEP 3: Correct Transcription with LLM
# ============================================================================

def correct_transcription(text, llm_model, tokenizer, dialect="egypt"):
    """Correct transcription using LLM"""
    print("\n" + "=" * 80)
    print("STEP 2: LLM CORRECTION")
    print("=" * 80)
    print(f"Input text: {text[:100]}...")
    print(f"Text length: {len(text)} characters")
    print()
    
    # Simple, direct prompt
    prompt = f"""صحح الأخطاء في هذا النص الطبي: {text}

النص المصحح:"""
    
    print("🔤 Tokenizing...")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}
    
    print(f"📊 Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"🤖 Generating correction (max 64 tokens)...")
    
    if DEVICE == "cpu":
        print("⏱️  CPU: ~20-30 minutes")
    else:
        print("⏱️  GPU: ~5-10 seconds")
    
    start = time.time()
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.1
        )
    
    elapsed = time.time() - start
    print(f"✅ Generation complete in {elapsed:.1f}s")
    
    # Decode and extract
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "النص المصحح:" in corrected:
        corrected = corrected.split("النص المصحح:")[-1].strip()
    
    corrected = corrected.replace(prompt, "").strip()
    
    # Validate output
    if len(corrected) > len(text) * 3 or not corrected or len(corrected) < 5:
        print("⚠️  LLM output malformed, using original text")
        corrected = text
    
    print(f"   Corrected: {corrected[:100]}...")
    
    return corrected

# ============================================================================
# STEP 4: Generate SOAP Note with LLM
# ============================================================================

def generate_soap_note(transcription, llm_model, tokenizer):
    """Generate SOAP note from transcription"""
    print("\n" + "=" * 80)
    print("STEP 3: SOAP NOTE GENERATION")
    print("=" * 80)
    
    prompt = f"""قم بتحويل هذه المحادثة الطبية إلى تقرير SOAP:

المحادثة: {transcription}

التقرير (S.O.A.P):"""
    
    print("🔤 Tokenizing...")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}
    
    print(f"🤖 Generating SOAP note (max 256 tokens)...")
    
    if DEVICE == "cpu":
        print("⏱️  CPU: ~40-60 minutes")
    else:
        print("⏱️  GPU: ~10-20 seconds")
    
    start = time.time()
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    elapsed = time.time() - start
    print(f"✅ Generation complete in {elapsed:.1f}s")
    
    soap_note = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract SOAP note
    if "التقرير" in soap_note:
        soap_note = soap_note.split("التقرير")[-1].strip()
    
    soap_note = soap_note.replace(prompt, "").strip()
    
    print(f"   SOAP Note: {soap_note[:200]}...")
    
    return soap_note

# ============================================================================
# STEP 5: Process All Audio Files
# ============================================================================

def find_audio_files(directory):
    """Find all audio files in directory"""
    extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    audio_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    return audio_files

def process_all_files(audio_files, asr_model, llm_model, llm_tokenizer, dialect="egypt"):
    """Process all audio files"""
    results = []
    
    for i, audio_path in enumerate(audio_files, 1):
        print("\n" + "=" * 80)
        print(f"PROCESSING FILE {i}/{len(audio_files)}")
        print("=" * 80)
        print(f"File: {audio_path}\n")
        
        try:
            # Process
            asr_result, full_text = transcribe_audio(audio_path, asr_model, dialect)
            corrected_text = correct_transcription(full_text, llm_model, llm_tokenizer, dialect)
            soap_note = generate_soap_note(corrected_text, llm_model, llm_tokenizer)
            
            # Save individual result
            output_file = os.path.join(WORKING_DIR, f"{Path(audio_path).stem}_result.json")
            
            result = {
                "audio_file": audio_path,
                "dialect": dialect,
                "device": DEVICE,
                "asr_result": {
                    "segments": asr_result["segments"],
                    "full_text": full_text
                },
                "corrected_text": corrected_text,
                "soap_note": soap_note,
                "status": "success"
            }
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ Results saved to: {output_file}")
            results.append(result)
            
        except Exception as e:
            print(f"\n❌ Error processing {audio_path}: {e}")
            results.append({
                "audio_file": audio_path,
                "status": "error",
                "error": str(e)
            })
    
    return results

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("KAGGLE PIPELINE - MEDICAL AUDIO PROCESSING")
    print("=" * 80)
    print()
    
    # Find audio files
    print("🔍 Searching for audio files...")
    audio_files = find_audio_files(INPUT_DIR)
    
    if not audio_files:
        print(f"❌ No audio files found in {INPUT_DIR}")
        print("💡 Upload audio files to Kaggle dataset and add as input")
        return
    
    print(f"✅ Found {len(audio_files)} audio file(s):")
    for f in audio_files:
        print(f"   - {f}")
    print()
    
    # Get dialect (can modify or pass as parameter)
    dialect = "egypt"  # Change this if needed
    print(f"🌍 Dialect: {dialect}\n")
    
    # Load models
    print("=" * 80)
    print("LOADING MODELS")
    print("=" * 80)
    print()
    
    asr_model = load_asr_model()
    llm_model, llm_tokenizer = load_llm_model()
    
    # Process all files
    results = process_all_files(audio_files, asr_model, llm_model, llm_tokenizer, dialect)
    
    # Save summary
    print("\n" + "=" * 80)
    print("SAVING SUMMARY")
    print("=" * 80)
    
    summary_file = os.path.join(WORKING_DIR, "processing_summary.json")
    summary = {
        "total_files": len(audio_files),
        "successful": len([r for r in results if r.get("status") == "success"]),
        "failed": len([r for r in results if r.get("status") == "error"]),
        "dialect": dialect,
        "device": DEVICE,
        "results": results
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Summary saved to: {summary_file}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"✅ Successful: {summary['successful']}/{summary['total_files']}")
    print(f"❌ Failed: {summary['failed']}/{summary['total_files']}")
    print()
    print("📁 Results available in /kaggle/working/")
    print("   Download all files from the Output tab")
    print("=" * 80)

if __name__ == "__main__":
    main()
