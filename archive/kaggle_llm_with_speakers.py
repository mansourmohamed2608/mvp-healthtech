"""
Kaggle LLM with Speaker-Aware Processing
=========================================
This version accepts ASR output WITH speaker labels and timestamps,
preserving the conversation structure in the output.

Input format:
[   1.2s -    2.7s] 👤 Patient:
  السلام عليكم يا دكتور.

[   2.8s -    3.8s] 🩺 Doctor:
  وعليكم السلام.

Usage:
1. Set INPUT_TEXT_WITH_SPEAKERS to your ASR output (with timestamps and labels)
2. Set TASK = "full"
3. Run on Kaggle GPU
4. Get structured output with speaker attribution
"""

import os
import time
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

TASK = "full"  # correct, soap, identify_speakers, full

# Paste your ASR output with speaker labels here
INPUT_TEXT_WITH_SPEAKERS = """
================================================================================
TRANSCRIPT WITH SPEAKERS
================================================================================

[   1.2s -    2.7s] 👤 Patient:
  السلام عليكم يا دكتور.

[   2.8s -    3.8s] 🩺 Doctor:
  وعليكم السلام.

[   3.8s -    5.3s] 🩺 Doctor:
  ازايك عاملة ايه?

[   5.3s -    7.2s] 👤 Patient:
  والله مش حاسة اني كويسة.

[   7.2s -    9.2s] 👤 Patient:
  اللسة عندي بقت حمرى ومتهيجة.

[   9.2s -   12.0s] 👤 Patient:
  ولما بخسل سناني اللسة بتنزف.

[  12.0s -   20.9s] 🩺 Doctor:
  طيب خلينا نشوف من كلامك اه واضح ان عندك التهاب في اللسة وممكن كمان يكون في جيوب لسوية.

[  21.0s -   24.8s] 🩺 Doctor:
  وده يعني ان في مسافة ما بين اللسة والاسنان.

[  25.0s -   29.3s] 🩺 Doctor:
  ودي بتتكون بسبب تراكم الجيل والكرسيم على مرور الوقت يعني.

[  30.7s -   54.3s] 🩺 Doctor:
  طيب حسيتي ريحة فم مش كويسة أو حسية في الأسنان؟ أيوة ريحة الفم بقت وحشة وحس إن في بعض الأسنان بقت حساسة أكتر من الطبيعي طيب ده بيدل على إن الحالة بقت متقدمة شوية طيب بتغسل سنانك ثلاث مرات في اليوم وبتستخدم الخيط الطبي؟ بغسل سناني بس مش دايما ومش بستخدم الخيط

[  55.5s -   80.1s] 🩺 Doctor:
  طيب احنا كده لازم نبتدي بخطة علاجية اول حاجة هنعملها هي تنظيف عميق للأسنان ونشيل الجير المتراك من تحت اللسم وبعد كده هديك تعليمات محددة لتنظيف أسنانك في البيت زي ان احنا هنستخدم الفرشة 3 مرات يوميا ونستخدم الخط الطبي ونستخدم معاهم مضمضة الاحساس بعد تنظيف الجير ده هيوجة؟

[  81.5s -  109.7s] 🩺 Doctor:
  والله إحنا ممكن نحس ببعض الحساسية شوية أو ألم خفيف بعد الجلسة ولكن ده بشكل مؤقت وممكن ناخد مسكنات ده لو احتاجناين ومع الالتزام بنصايحي الحالة بإذن الله تتحسن طيب وهل في احتمال إن الحالة دي ترجع تاني لو ما اتبعتش تعليمات؟ أكيد لو ما حفظتيش على نضافة أسنانك بشكل كويس ممكن الجير يرجع يتراكم تاني وتستمر المشكلة

[ 110.1s -  114.4s] 🩺 Doctor:
  عشان كده من المهم انك تتابعي معايا بشكل دوري.

[ 114.4s -  115.2s] 👤 Patient:
  تمام فهمت.

[ 115.2s -  117.5s] 👤 Patient:
  انا هبدأ اعمل كده من دلوقتي.

[ 117.6s -  118.4s] 🩺 Doctor:
  حلو قوي.

[ 118.5s -  126.7s] 🩺 Doctor:
  هنعمل موعد للمتابعة ونشوف تحسن الحالة ولو احتكت اي حاجة او حسيت باي تغير ما تتراضديش في الاتصال بينا.

[ 126.8s -  129.2s] 👤 Patient:
  شكرا يا دكتور على التوضيح والنصايا.

[ 129.3s -  133.2s] 🩺 Doctor:
  العفو ده واجبنا وسلامتك الف سلامة وتكوني بصحة دايما وسلام.


"""

DIALECT = "egypt"

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("KAGGLE LLM PROCESSOR (SPEAKER-AWARE)")
print("=" * 80)
print()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKING_DIR = "/kaggle/working"
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
# HELPER FUNCTIONS
# ============================================================================

def parse_speaker_transcript(text):
    """
    Parse ASR transcript with speaker labels into structured format.
    Returns: list of {timestamp, speaker, text} dicts and plain text
    """
    lines = text.strip().split('\n')
    segments = []
    plain_text = []

    current_segment = None
    for line in lines:
        # Check for timestamp line: [   1.2s -    2.7s] 👤 Patient:
        timestamp_match = re.match(r'\[\s*([\d.]+)s\s*-\s*([\d.]+)s\]\s*(.+?):\s*$', line.strip())
        if timestamp_match:
            if current_segment:
                segments.append(current_segment)

            start_time = float(timestamp_match.group(1))
            end_time = float(timestamp_match.group(2))
            speaker_raw = timestamp_match.group(3).strip()

            # Extract speaker role (remove emoji)
            speaker = re.sub(r'[👤🩺💉🏥]', '', speaker_raw).strip()

            current_segment = {
                "start": start_time,
                "end": end_time,
                "speaker": speaker,
                "text": ""
            }
        elif current_segment and line.strip():
            # This is content line for current segment
            current_segment["text"] += line.strip() + " "

    # Add last segment
    if current_segment:
        segments.append(current_segment)

    # Create plain text version
    for seg in segments:
        plain_text.append(seg["text"].strip())

    return segments, " ".join(plain_text)

# ============================================================================
# LOAD LLM (Same as original)
# ============================================================================

HF_REPO_ID = "Henrychur/MMed-Llama-3-8B"

def _find_config_root(root_dir: str | Path) -> Path | None:
    root = Path(root_dir)
    if not root.exists():
        return None
    if (root / "config.json").exists():
        return root
    for cfg in (root / "snapshots").glob("*/config.json"):
        return cfg.parent
    for p in root.rglob("config.json"):
        if all(part not in {"__pycache__", ".git", ".hg"} for part in p.parts):
            return p.parent
    return None

def _resolve_model_source(local_dir: str) -> tuple[str, dict]:
    cfg_root = _find_config_root(local_dir)
    if cfg_root is not None:
        print(f"✅ Found local config.json at: {cfg_root}")
        return str(cfg_root), {"local_files_only": True, "trust_remote_code": True}
    print(f"⚠️  No config.json under {local_dir}. Falling back to: {HF_REPO_ID}")
    return HF_REPO_ID, {"local_files_only": False, "trust_remote_code": True}

def load_llm_model():
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
# PROCESSING FUNCTIONS (Speaker-Aware)
# ============================================================================

def generate_soap_with_speakers(segments, llm_model, tokenizer):
    """Generate SOAP note from structured conversation"""
    print("=" * 80)
    print("SOAP NOTE GENERATION (Speaker-Aware)")
    print("=" * 80)
    print()

    # Build conversation text with clear speaker labels
    conversation = ""
    for seg in segments:
        conversation += f"{seg['speaker']}: {seg['text']}\n"

    prompt = f"""قم بتحويل هذه المحادثة الطبية إلى تقرير SOAP:

المحادثة: {conversation}

التقرير (S.O.A.P):"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

    print("🤖 Generating SOAP note...")
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

    # Extract SOAP note (Arabic marker)
    if "التقرير" in soap:
        soap = soap.split("التقرير")[-1].strip()
    
    soap = soap.replace(prompt, "").strip()

    print(f"\nSOAP Note:")
    print(soap)
    print("=" * 80)
    print()

    return soap

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print(f"TASK: {TASK}")
    print("=" * 80)
    print()

    # Parse input
    segments, plain_text = parse_speaker_transcript(INPUT_TEXT_WITH_SPEAKERS)

    print(f"Parsed {len(segments)} speaker segments")
    print(f"Plain text length: {len(plain_text)} characters\n")

    # Load model
    llm_model, llm_tokenizer = load_llm_model()

    # Process
    total_start = time.time()
    result = {
        "task": TASK,
        "dialect": DIALECT,
        "device": DEVICE,
        "segments": segments,
        "plain_text": plain_text,
        "status": "success"
    }

    if TASK in ["soap", "full"]:
        soap = generate_soap_with_speakers(segments, llm_model, llm_tokenizer)
        result["soap_note"] = soap

    total_elapsed = time.time() - total_start
    result["processing_time_seconds"] = round(total_elapsed, 2)

    # Save results
    output_file = os.path.join(WORKING_DIR, "result_with_speakers.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"✅ Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} mins)")
    print(f"✅ Results saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
