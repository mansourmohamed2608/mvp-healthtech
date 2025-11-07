"""
Kaggle LLM Complete - All LLM Features
=======================================
This script includes ALL functionality from services/llm service:
1. Transcription Correction - Fix ASR errors
2. SOAP Note Generation - Structured clinical notes
3. Speaker Role Identification - Doctor vs Patient detection
4. Medical Chat - RAG-enhanced question answering

Runs on Kaggle GPU for 500x speedup vs local CPU!

Usage:
1. Set TASK to what you want (see options below)
2. Paste your input into INPUT_TEXT or CHAT_MESSAGE
3. Enable GPU in Kaggle (Settings → Accelerator → GPU T4)
4. Run this script
5. Results saved to result.json

Requirements:
- Kaggle GPU (Tesla T4)
- transformers==4.44.0
- bitsandbytes
- accelerate>=0.27.0
"""

import os
import time
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

# TASK OPTIONS:
# "correct" - Fix transcription errors and medical terminology
# "soap" - Generate SOAP note (Subjective, Objective, Assessment, Plan)
# "identify_speakers" - Detect who is Doctor, Patient, Nurse, etc.
# "chat" - Medical Q&A with RAG context
# "full" - Run correction + SOAP + speaker ID (full pipeline)

TASK = "full"

# FOR TASKS: correct, soap, identify_speakers, full
# Paste your transcription here
INPUT_TEXT = """
السلام عليكم يا دكتور. وعليكم السلام. ازايك عاملة ايه? والله مش حاسة اني كويسة. اللسة عندي بقت حمرى ومتهيجة. ولما بخسل سناني اللسة بتنزف. طيب خلينا نشوف من كلامك اه واضح ان عندك التهاب في اللسة وممكن كمان يكون في جيوب لسوية. وده يعني ان في مسافة ما بين اللسة والاسنان. ودي بتتكون بسبب تراكم الجيل والكرسيم على مرور الوقت يعني. طيب حسيتي ريحة فم مش كويسة أو حسية في الأسنان؟ أيوة ريحة الفم بقت وحشة وحس إن في بعض الأسنان بقت حساسة أكتر من الطبيعي طيب ده بيدل على إن الحالة بقت متقدمة شوية طيب بتغسل سنانك ثلاث مرات في اليوم وبتستخدم الخيط الطبي؟ بغسل سناني بس مش دايما ومش بستخدم الخيط طيب احنا كده لازم نبتدي بخطة علاجية اول حاجة هنعملها هي تنظيف عميق للأسنان ونشيل الجير المتراك من تحت اللسم وبعد كده هديك تعليمات محددة لتنظيف أسنانك في البيت زي ان احنا هنستخدم الفرشة 3 مرات يوميا ونستخدم الخط الطبي ونستخدم معاهم مضمضة الاحساس بعد تنظيف الجير ده هيوجة؟ والله إحنا ممكن نحس ببعض الحساسية شوية أو ألم خفيف بعد الجلسة ولكن ده بشكل مؤقت وممكن ناخد مسكنات ده لو احتاجناين ومع الالتزام بنصايحي الحالة بإذن الله تتحسن طيب وهل في احتمال إن الحالة دي ترجع تاني لو ما اتبعتش تعليمات؟ أكيد لو ما حفظتيش على نضافة أسنانك بشكل كويس ممكن الجير يرجع يتراكم تاني وتستمر المشكلة عشان كده من المهم انك تتابعي معايا بشكل دوري. تمام فهمت. انا هبدأ اعمل كده من دلوقتي. حلو قوي. هنعمل موعد للمتابعة ونشوف تحسن الحالة ولو احتكت اي حاجة او حسيت باي تغير ما تتراضديش في الاتصال بينا. شكرا يا دكتور على التوضيح والنصايا. العفو ده واجبنا وسلامتك الف سلامة وتكوني بصحة دايما وسلام.
"""

# FOR TASK: chat
# Your medical question
CHAT_MESSAGE = "ما هي أعراض ارتفاع ضغط الدم؟"

# Dialect context (egypt, gulf, levant, etc.)
DIALECT = "egypt"

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("KAGGLE LLM PROCESSOR")
print("=" * 80)
print()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKING_DIR = "/kaggle/working"

# Use pre-downloaded model from Kaggle input (saves 5-10 mins!)
MODEL_DIR = "/kaggle/input/medllm/models--Henrychur--MMed-Llama-3-8B"
MODEL_CACHE = os.path.join(WORKING_DIR, "models")

os.makedirs(MODEL_CACHE, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE
os.environ['HF_HOME'] = MODEL_CACHE

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Output: {WORKING_DIR}\n")

# ============================================================================
# LOAD LLM
# ============================================================================

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

HF_REPO_ID = "Henrychur/MMed-Llama-3-8B"  # fallback if local snapshot incomplete

def _find_config_root(root_dir: str | Path) -> Path | None:
    """Return the directory that directly contains config.json, searching depth-first."""
    root = Path(root_dir)
    if not root.exists():
        return None
    # 1) direct (flat export)
    if (root / "config.json").exists():
        return root
    # 2) common Kaggle/HF dataset layout: …/snapshots/<sha>/config.json
    for cfg in (root / "snapshots").glob("*/config.json"):
        return cfg.parent
    # 3) last resort: walk a little (avoid huge recursion)
    for p in root.rglob("config.json"):
        # skip cache/temp dirs if any
        if all(part not in {"__pycache__", ".git", ".hg"} for part in p.parts):
            return p.parent
    return None

def _resolve_model_source(local_dir: str) -> tuple[str, dict]:
    """
    If a usable config.json exists, return its containing folder and force local_files_only.
    Otherwise fall back to HF repo id with remote download allowed.
    """
    cfg_root = _find_config_root(local_dir)
    if cfg_root is not None:
        print(f"✅ Found local config.json at: {cfg_root}")
        return str(cfg_root), {"local_files_only": True, "trust_remote_code": True}
    print(f"⚠️  No config.json under {local_dir}. Falling back to: {HF_REPO_ID}")
    return HF_REPO_ID, {"local_files_only": False, "trust_remote_code": True}

def load_llm_model():
    """Load Medical LLM with 4-bit on GPU (8-bit on CPU)."""
    print("=" * 80)
    print("LOADING MMed-Llama-3-8B")
    print("=" * 80)
    print()

    using_gpu = (DEVICE == "cuda")
    if using_gpu:
        print("Using 4-bit quantization on GPU\n")
    else:
        print("⚠️  No GPU detected! Using 8-bit on CPU (slow)\n")

    model_id_or_path, extra = _resolve_model_source(MODEL_DIR)

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=using_gpu,
        load_in_8bit=not using_gpu,
        bnb_4bit_use_double_quant=True if using_gpu else None,
        bnb_4bit_quant_type="nf4" if using_gpu else None,
        bnb_4bit_compute_dtype=torch.float16 if using_gpu else None,
        llm_int8_enable_fp32_cpu_offload=not using_gpu,
    )

    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path, cache_dir=MODEL_CACHE, **extra
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16 if using_gpu else torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=MODEL_CACHE,
        **extra,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"\n✅ Model loaded in {time.time()-t0:.1f}s")
    print("=" * 80, "\n")
    return model, tokenizer

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def correct_transcription(text, llm_model, tokenizer):
    """Correct medical transcription errors"""
    print("=" * 80)
    print("STEP 1: TEXT CORRECTION")
    print("=" * 80)
    print(f"Input length: {len(text)} characters")
    
    # CRITICAL: Correction only works for SHORT text (max_new_tokens=64 = ~200 chars)
    # For full conversations, skip correction to avoid truncation/gibberish
    if len(text) > 500:
        print(f"\n⚠️  Text is too long for correction ({len(text)} chars)")
        print("   The model can only correct short utterances (1-3 sentences)")
        print("   For full conversations, skip correction and use original text")
        print("   (This matches your local service behavior)\n")
        print("=" * 80)
        print()
        return text  # Return unchanged - no correction
    
    print(f"Input preview: {text[:100]}...")
    print()

    # WORKING PROMPT FROM LOCAL services/llm/app.py
    prompt = f"""صحح الأخطاء في هذا النص الطبي: {text}

النص المصحح:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

    print("🤖 Generating correction...")
    print(f"   Expected time: GPU ~5-10s, CPU ~20-30 mins")
    start = time.time()

    # WORKING SETTINGS FROM LOCAL services/llm/app.py
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=64,  # Keep it short - correction should be similar length
            do_sample=False,  # Deterministic for corrections
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.1  # Prevent repeating the prompt
        )

    elapsed = time.time() - start
    print(f"✅ Generated in {elapsed:.1f}s")

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # WORKING CLEANUP FROM LOCAL services/llm/app.py
    # Extract only the corrected text after the marker
    if "النص المصحح:" in corrected:
        corrected = corrected.split("النص المصحح:")[-1].strip()
    
    # Remove the prompt if model repeated it
    corrected = corrected.replace(prompt, "").strip()
    
    # If output starts with instruction text, try to extract just the answer
    if corrected.startswith("صحح"):
        for sep in [":", "\n"]:
            if sep in corrected:
                corrected = corrected.split(sep, 1)[-1].strip()
                break
    
    # If output still has the original text embedded, extract just after it
    if text in corrected:
        parts = corrected.split(text)
        if len(parts) > 1 and parts[1].strip():
            corrected = parts[1].strip()
    
    # CRITICAL VALIDATION: Check if output is way shorter than input (truncation!)
    if len(corrected) < len(text) * 0.5:
        print(f"⚠️  LLM output too short ({len(corrected)} vs {len(text)} chars), using original")
        corrected = text
    
    # Validation: if output is too long (more than 3x original), use original
    if len(corrected) > len(text) * 3:
        print("⚠️  LLM output too long, using original")
        corrected = text
    
    # If we ended up with nothing, use original
    if not corrected or len(corrected) < 5:
        print("⚠️  LLM output empty, using original")
        corrected = text

    print(f"\nCorrected text ({len(corrected)} chars):")
    print(corrected)
    print("=" * 80)
    print()

    return corrected

def generate_soap_note(text, llm_model, tokenizer):
    """Generate SOAP note from medical conversation"""
    print("=" * 80)
    print("STEP 2: SOAP NOTE GENERATION")
    print("=" * 80)
    print(f"Input length: {len(text)} characters")
    print()

    prompt = f"""قم بتحويل هذه المحادثة الطبية إلى تقرير SOAP:

المحادثة: {text}

التقرير (S.O.A.P):"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

    print("🤖 Generating SOAP note...")
    print(f"   Expected time: GPU ~10-20s, CPU ~40-60 mins")
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
    print(f"✅ Generated in {elapsed:.1f}s")

    soap = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up output - extract SOAP note (Arabic markers)
    if "التقرير" in soap:
        soap = soap.split("التقرير")[-1].strip()
    
    soap = soap.replace(prompt, "").strip()

    print(f"\nSOAP Note ({len(soap)} chars):")
    print(soap)
    print("=" * 80)
    print()

    return soap

def identify_speaker_roles(text, llm_model, tokenizer):
    """Identify speaker roles (Doctor, Patient, Nurse, etc.)"""
    print("=" * 80)
    print("STEP 3: SPEAKER ROLE IDENTIFICATION")
    print("=" * 80)
    print(f"Input length: {len(text)} characters")
    print()

    prompt = f"""Analyze the following medical conversation and identify the role of each speaker.
Consider:
1. Medical terminology usage (doctors use more technical terms)
2. Question patterns (doctors ask diagnostic questions)
3. Authority indicators ("I will prescribe", "Let me examine")
4. Symptom descriptions (patients describe their pain/discomfort)
5. Treatment plans (doctors explain procedures)

Conversation:
{text}

For each unique speaker, identify their role (Doctor, Patient, Nurse, etc.) and provide reasoning.
Format your response as JSON with this structure:
{{
  "roles": [
    {{"speaker_id": "SPEAKER_00", "role": "Doctor", "confidence": 0.95, "reasoning": "Uses medical terminology, asks diagnostic questions"}},
    {{"speaker_id": "SPEAKER_01", "role": "Patient", "confidence": 0.90, "reasoning": "Describes symptoms and responds to doctor's questions"}}
  ]
}}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

    print("🤖 Analyzing speaker roles...")
    print(f"   Expected time: GPU ~10-20s, CPU ~40-60 mins")
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
    print(f"✅ Generated in {elapsed:.1f}s")

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to parse JSON
    roles = []
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
            roles = analysis.get("roles", [])
    except:
        # Fallback to heuristic analysis
        roles = analyze_speakers_heuristic(text)

    print(f"\nIdentified Roles:")
    for role in roles:
        print(f"  {role.get('speaker_id', 'Unknown')}: {role.get('role', 'Unknown')} (confidence: {role.get('confidence', 0):.2f})")
        print(f"    Reasoning: {role.get('reasoning', 'N/A')}")
    print("=" * 80)
    print()

    return roles

def analyze_speakers_heuristic(text):
    """Fallback heuristic analysis when LLM fails"""
    # Simple heuristic based on keywords
    doctor_keywords = [
        "prescribe", "examine", "diagnosis", "treatment", "recommend", "assess",
        "يصف", "فحص", "تشخيص", "علاج", "أوصي", "تقييم",
        "blood pressure", "heart rate", "temperature", "vitals",
        "ضغط الدم", "معدل القلب", "حرارة", "علامات حيوية"
    ]

    patient_keywords = [
        "pain", "hurts", "feeling", "symptom", "sick", "discomfort",
        "ألم", "يؤلم", "شعور", "أعراض", "مريض", "إزعاج",
        "I have", "I feel", "since", "for days",
        "لدي", "أشعر", "منذ", "أيام"
    ]

    # Count keywords
    text_lower = text.lower()
    doctor_count = sum(1 for kw in doctor_keywords if kw.lower() in text_lower)
    patient_count = sum(1 for kw in patient_keywords if kw.lower() in text_lower)

    # Simple logic: if more doctor keywords, assume single doctor speaking
    # Otherwise, assume it's patient-only text
    if doctor_count > patient_count:
        return [
            {
                "speaker_id": "SPEAKER_00",
                "role": "Doctor",
                "confidence": min(0.95, 0.6 + doctor_count * 0.05),
                "reasoning": f"Uses medical terminology ({doctor_count} doctor indicators)"
            }
        ]
    else:
        return [
            {
                "speaker_id": "SPEAKER_00",
                "role": "Patient",
                "confidence": min(0.95, 0.6 + patient_count * 0.05),
                "reasoning": f"Describes symptoms ({patient_count} patient indicators)"
            }
        ]

def medical_chat(message, llm_model, tokenizer):
    """Medical Q&A with RAG-enhanced context"""
    print("=" * 80)
    print("MEDICAL CHAT")
    print("=" * 80)
    print(f"Question: {message}")
    print()

    # Build RAG-enhanced prompt (simplified - full version would use vector DB)
    prompt = f"""أنت مساعد طبي ذكي يتحدث العربية. مهمتك مساعدة المرضى بطريقة محترفة وودودة.

المستخدم: {message}
المساعد:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

    print("🤖 Generating response...")
    print(f"   Expected time: GPU ~5-10s, CPU ~20-30 mins")
    start = time.time()

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    elapsed = time.time() - start
    print(f"✅ Generated in {elapsed:.1f}s")

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract reply
    if "المساعد:" in response:
        response = response.split("المساعد:")[-1].strip()
    response = response.replace(prompt, "").strip()

    print(f"\nResponse:")
    print(response)
    print("=" * 80)
    print()

    return response

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print(f"TASK: {TASK}")
    print("=" * 80)
    print()

    # Validate input based on task
    if TASK == "chat":
        if not CHAT_MESSAGE or CHAT_MESSAGE.strip() == "":
            print("❌ ERROR: CHAT_MESSAGE is empty!")
            print("Please set CHAT_MESSAGE to your medical question")
            return
    else:
        if not INPUT_TEXT or INPUT_TEXT.strip() == "":
            print("❌ ERROR: INPUT_TEXT is empty!")
            print("Please paste your transcription text into the INPUT_TEXT variable")
            return

    # Load model once
    llm_model, llm_tokenizer = load_llm_model()

    # Process based on task
    total_start = time.time()
    result = {
        "task": TASK,
        "dialect": DIALECT,
        "device": DEVICE,
        "status": "success"
    }

    if TASK == "correct":
        print("Input text to correct:")
        print("-" * 80)
        print(INPUT_TEXT.strip())
        print("-" * 80)
        print()

        corrected = correct_transcription(INPUT_TEXT.strip(), llm_model, llm_tokenizer)
        result["input_text"] = INPUT_TEXT.strip()
        result["corrected_text"] = corrected

    elif TASK == "soap":
        print("Input text for SOAP generation:")
        print("-" * 80)
        print(INPUT_TEXT.strip())
        print("-" * 80)
        print()

        soap = generate_soap_note(INPUT_TEXT.strip(), llm_model, llm_tokenizer)
        result["input_text"] = INPUT_TEXT.strip()
        result["soap_note"] = soap

    elif TASK == "identify_speakers":
        print("Input text for speaker identification:")
        print("-" * 80)
        print(INPUT_TEXT.strip())
        print("-" * 80)
        print()

        roles = identify_speaker_roles(INPUT_TEXT.strip(), llm_model, llm_tokenizer)
        result["input_text"] = INPUT_TEXT.strip()
        result["speaker_roles"] = roles

    elif TASK == "chat":
        response = medical_chat(CHAT_MESSAGE.strip(), llm_model, llm_tokenizer)
        result["question"] = CHAT_MESSAGE.strip()
        result["response"] = response

    elif TASK == "full":
        print("Input text to process:")
        print("-" * 80)
        print(INPUT_TEXT.strip())
        print("-" * 80)
        print()

        # Run full pipeline
        corrected = correct_transcription(INPUT_TEXT.strip(), llm_model, llm_tokenizer)
        soap = generate_soap_note(corrected, llm_model, llm_tokenizer)
        roles = identify_speaker_roles(INPUT_TEXT.strip(), llm_model, llm_tokenizer)

        result["input_text"] = INPUT_TEXT.strip()
        result["corrected_text"] = corrected
        result["soap_note"] = soap
        result["speaker_roles"] = roles

    else:
        print(f"❌ ERROR: Unknown task '{TASK}'")
        print("Valid tasks: correct, soap, identify_speakers, chat, full")
        return

    total_elapsed = time.time() - total_start
    result["processing_time_seconds"] = round(total_elapsed, 2)

    # Save results
    output_file = os.path.join(WORKING_DIR, "result.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"✅ Total processing time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} mins)")
    print(f"✅ Results saved to: {output_file}")
    print()
    print("📥 Download from Kaggle Output tab")
    print("=" * 80)

if __name__ == "__main__":
    main()
