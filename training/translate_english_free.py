"""
Translate English Medical Data to Egyptian Arabic (FREE)
=========================================================

Uses NLLB-200 (Meta's free translator) to convert English medical
examples to Egyptian Arabic.

Cost: $0 (runs on Kaggle free GPU)
Time: ~1-2 hours for 1000 examples

This is a ONE-TIME translation. After training, inference is pure Arabic.
"""

import json
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import torch

print("=" * 80)
print("FREE ENGLISH → EGYPTIAN ARABIC TRANSLATION")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input: English medical examples (SOAP notes, conversations, etc.)
INPUT_FILE = "english_medical_examples.json"  # Your English data
OUTPUT_FILE = "training_data_translated.json"

# Translation model
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Language codes
SRC_LANG = "eng_Latn"  # English
TGT_LANG = "arz_Arab"  # Egyptian Arabic

print(f"Model: {MODEL_NAME}")
print(f"Translation: {SRC_LANG} → {TGT_LANG}")
print()

# ============================================================================
# MEDICAL GLOSSARY (Preserve medical terms)
# ============================================================================

MEDICAL_GLOSSARY = {
    # Acronyms (keep in English)
    "SOAP": "SOAP",
    "BP": "BP",
    "HR": "HR",
    "O2": "O2",
    "CBC": "CBC",
    
    # Common medical terms with standard Arabic translations
    "Subjective": "S (Subjective)",
    "Objective": "O (Objective)",
    "Assessment": "A (Assessment)",
    "Plan": "P (Plan)",
    
    # Diseases/Conditions
    "diabetes": "السكري",
    "hypertension": "ارتفاع ضغط الدم",
    "asthma": "الربو",
    "gingivitis": "التهاب اللثة",
    "headache": "صداع",
    
    # Medications (transliterate)
    "ibuprofen": "ايبوبروفين",
    "paracetamol": "باراسيتامول",
    "aspirin": "أسبرين",
}

def apply_glossary(text, glossary):
    """Apply medical glossary post-translation"""
    for en, ar in glossary.items():
        # Case-insensitive replacement
        text = text.replace(en.lower(), ar)
        text = text.replace(en.upper(), ar)
        text = text.replace(en.capitalize(), ar)
    return text

# ============================================================================
# LOAD TRANSLATOR
# ============================================================================

print("Loading NLLB-200 translator...")
start = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang=SRC_LANG,
    tgt_lang=TGT_LANG,
    max_length=512,
    device=0 if device == "cuda" else -1
)

print(f"✅ Loaded in {time.time()-start:.1f}s")
print()

# ============================================================================
# LOAD ENGLISH DATA
# ============================================================================

print(f"Loading English data from {INPUT_FILE}...")

# Example format:
# [
#   {
#     "instruction": "You are a medical assistant. Write a SOAP note for:",
#     "input": "Doctor: What's wrong?\nPatient: I have tooth pain",
#     "output": "S: Patient complains of tooth pain\nO: ...\nA: ...\nP: ..."
#   }
# ]

try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        english_data = json.load(f)
    
    print(f"✅ Loaded {len(english_data)} examples")
    print()
    
except FileNotFoundError:
    print(f"❌ {INPUT_FILE} not found!")
    print()
    print("Create this file with your English medical examples, or use:")
    print("  - MedQA English dataset")
    print("  - PubMedQA")
    print("  - Your own English SOAP notes")
    print()
    exit(1)

# ============================================================================
# TRANSLATE
# ============================================================================

print("=" * 80)
print("TRANSLATING TO EGYPTIAN ARABIC")
print("=" * 80)
print()

translated_data = []
errors = []

for i, example in enumerate(tqdm(english_data, desc="Translating")):
    try:
        # Translate each field
        translated = {
            "instruction": "",
            "input": "",
            "output": "",
            "metadata": {
                "source": "translated_from_english",
                "task": example.get("task", "medical"),
                "dialect": "egyptian"
            }
        }
        
        # Translate instruction
        if example.get("instruction"):
            result = translator(example["instruction"], max_length=256)
            translated["instruction"] = apply_glossary(
                result[0]["translation_text"], 
                MEDICAL_GLOSSARY
            )
        
        # Translate input
        if example.get("input"):
            result = translator(example["input"], max_length=512)
            translated["input"] = apply_glossary(
                result[0]["translation_text"],
                MEDICAL_GLOSSARY
            )
        
        # Translate output
        if example.get("output"):
            result = translator(example["output"], max_length=512)
            translated["output"] = apply_glossary(
                result[0]["translation_text"],
                MEDICAL_GLOSSARY
            )
        
        translated_data.append(translated)
        
        # Small delay to prevent overheating
        if i % 100 == 0:
            time.sleep(1)
    
    except Exception as e:
        errors.append({"index": i, "error": str(e)})
        print(f"\n⚠️  Error at example {i}: {e}")

print()
print(f"✅ Translated {len(translated_data)}/{len(english_data)} examples")

if errors:
    print(f"⚠️  {len(errors)} errors occurred")

print()

# ============================================================================
# QUALITY CHECK (Round-trip sample)
# ============================================================================

print("=" * 80)
print("QUALITY CHECK (Round-trip on sample)")
print("=" * 80)
print()

if len(translated_data) > 0:
    # Test reverse translation on first example
    sample = translated_data[0]
    
    print("Original English:")
    print(english_data[0]["input"][:200])
    print()
    
    print("Translated to Arabic:")
    print(sample["input"][:200])
    print()
    
    # Reverse translate
    reverse_translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=TGT_LANG,
        tgt_lang=SRC_LANG,
        max_length=512,
        device=0 if device == "cuda" else -1
    )
    
    reverse = reverse_translator(sample["input"][:200])
    print("Back to English:")
    print(reverse[0]["translation_text"])
    print()
    
    print("If the round-trip looks similar, quality is good! ✅")
    print()

# ============================================================================
# SAVE TRANSLATED DATA
# ============================================================================

print("=" * 80)
print("SAVING TRANSLATED DATA")
print("=" * 80)
print()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(translated_data)} examples to {OUTPUT_FILE}")
print()

# Save errors log
if errors:
    with open("translation_errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"⚠️  Saved {len(errors)} errors to translation_errors.json")
    print()

# Statistics
print("Translation summary:")
print(f"  Input examples: {len(english_data)}")
print(f"  Successfully translated: {len(translated_data)}")
print(f"  Errors: {len(errors)}")
print(f"  Success rate: {len(translated_data)/len(english_data)*100:.1f}%")
print()

print("=" * 80)
print("✅ TRANSLATION COMPLETE - 100% FREE!")
print("=" * 80)
print()

print("Next steps:")
print("1. Review translation quality")
print("2. Optionally merge with free Arabic data:")
print("   combined = translated_data + free_arabic_data")
print("3. Upload to Kaggle as dataset")
print("4. Run finetune_kaggle.py")
print()
print("Total cost: $0 🎉")
