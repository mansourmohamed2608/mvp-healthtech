"""
Local Audio Processing Pipeline - No HTTP Requests
Runs ASR and LLM directly using the Python libraries
"""

import os
import sys
import time
import json
import torch
import whisperx
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

print("=" * 80)
print("LOCAL AUDIO PROCESSING PIPELINE")
print("No HTTP requests - Direct Python execution")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Compute Type: {COMPUTE_TYPE}")
print()

# ============================================================================
# STEP 1: Load Models
# ============================================================================

def load_asr_model():
    """Load WhisperX ASR model"""
    print("📥 Loading WhisperX model (large-v3)...")
    start = time.time()
    
    model = whisperx.load_model(
        "large-v3",
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        language="ar"
    )
    
    elapsed = time.time() - start
    print(f"✅ WhisperX loaded in {elapsed:.1f}s\n")
    return model

def load_llm_model():
    """Load LLM model with optimized settings"""
    print("📥 Loading LLM model (MMed-Llama-3-8B)...")
    print("⚠️  This will take 13-20 minutes on CPU...")
    start = time.time()
    
    model_name = "Henrychur/MMed-Llama-3-8B"
    
    # Check if we can use GPU
    if DEVICE == "cuda":
        print("✅ Using GPU acceleration")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        print("⚠️  Using CPU with 8-bit quantization (slow but works)")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True
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
    
    # Align
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
    print("👥 Detecting speakers...")
    # Note: Diarization requires pyannote.audio token
    # For now, we'll skip it or use a simple implementation
    
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
        print("⏱️  This will take ~20-30 minutes on CPU...")
    else:
        print("⏱️  This should take 5-10 seconds on GPU...")
    
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
    
    # Decode output
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract correction
    if "النص المصحح:" in corrected:
        corrected = corrected.split("النص المصحح:")[-1].strip()
    
    corrected = corrected.replace(prompt, "").strip()
    
    # If output is too long or contains instruction text, use original
    if len(corrected) > len(text) * 3 or not corrected or len(corrected) < 5:
        print("⚠️  LLM output was malformed, using original text")
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
        print("⏱️  This will take ~40-60 minutes on CPU...")
    else:
        print("⏱️  This should take 10-20 seconds on GPU...")
    
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
# MAIN PIPELINE
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_audio_local.py <audio_file> [dialect]")
        print("Example: python process_audio_local.py test1.mp3 egypt")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    dialect = sys.argv[2] if len(sys.argv) > 2 else "egypt"
    
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    print(f"\n🎯 Processing: {audio_path}")
    print(f"🌍 Dialect: {dialect}\n")
    
    # Load models
    print("=" * 80)
    print("LOADING MODELS")
    print("=" * 80)
    
    asr_model = load_asr_model()
    llm_model, llm_tokenizer = load_llm_model()
    
    # Process audio
    result, full_text = transcribe_audio(audio_path, asr_model, dialect)
    
    # Correct transcription
    corrected_text = correct_transcription(full_text, llm_model, llm_tokenizer, dialect)
    
    # Generate SOAP note
    soap_note = generate_soap_note(corrected_text, llm_model, llm_tokenizer)
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    output_file = f"{Path(audio_path).stem}_result.json"
    
    results = {
        "audio_file": audio_path,
        "dialect": dialect,
        "device": DEVICE,
        "asr_result": {
            "segments": result["segments"],
            "full_text": full_text
        },
        "corrected_text": corrected_text,
        "soap_note": soap_note
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\n📝 Transcription: {full_text}\n")
    print(f"✏️  Corrected: {corrected_text}\n")
    print(f"📋 SOAP Note: {soap_note}\n")
    print("=" * 80)

if __name__ == "__main__":
    main()
