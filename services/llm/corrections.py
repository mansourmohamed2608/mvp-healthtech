# services/llm/corrections.py
"""
Arabic Medical Term Corrections
Fixes common ASR/LLM mistakes in Arabic medical terminology
Target: +5-8% accuracy improvement
"""

# Common medical term corrections (ASR artifacts → correct medical term)
MEDICAL_CORRECTIONS = {
    # Anatomy (تشريح)
    "خط": "خيط",  # Suture
    "البروستاتا": "البروستات",  # Prostate
    "الرحام": "الرحم",  # Uterus
    "الكلا": "الكلى",  # Kidney
    "الكبد": "الكبد",  # Liver (sometimes transcribed wrong)
    "القلاب": "القلب",  # Heart
    "المعدا": "المعدة",  # Stomach
    "الامعاء": "الأمعاء",  # Intestines
    "المراره": "المرارة",  # Gallbladder
    "البنكرياس": "البنكرياس",  # Pancreas (standardize)
    
    # Symptoms (أعراض)
    "الام": "ألم",  # Pain
    "الم": "ألم",
    "صداع": "صداع",  # Headache (standardize)
    "حمى": "حمّى",  # Fever
    "حراره": "حرارة",  # Temperature
    "سعال": "سعال",  # Cough (standardize)
    "زكام": "زكام",  # Cold
    "غثيان": "غثيان",  # Nausea (standardize)
    "قيئ": "قيء",  # Vomiting
    "اسهال": "إسهال",  # Diarrhea
    "امساك": "إمساك",  # Constipation
    "دوخه": "دوخة",  # Dizziness
    "دوار": "دوار",  # Vertigo
    
    # Conditions/Diseases (أمراض)
    "سكر": "سكري",  # Diabetes
    "السكر": "السكري",
    "ضغط": "ضغط دم",  # Blood pressure
    "الضغط": "ضغط الدم",
    "سرطان": "سرطان",  # Cancer (standardize)
    "التهاب": "التهاب",  # Inflammation (standardize)
    "عدوى": "عدوى",  # Infection (standardize)
    "حساسيه": "حساسية",  # Allergy
    "ربو": "ربو",  # Asthma (standardize)
    "كورونا": "كوفيد-19",  # COVID-19
    
    # Medications (أدوية)
    "مضاد حيوي": "مضاد حيوي",  # Antibiotic (standardize)
    "مسكن": "مسكّن",  # Painkiller
    "انسولين": "إنسولين",  # Insulin
    "الانسولين": "الإنسولين",
    "باراسيتامول": "باراسيتامول",  # Paracetamol (standardize)
    "ايبوبروفين": "إيبوبروفين",  # Ibuprofen
    
    # Procedures (إجراءات)
    "فحص": "فحص",  # Examination (standardize)
    "تحليل": "تحليل",  # Analysis (standardize)
    "اشعه": "أشعة",  # X-ray/Radiology
    "اشعة": "أشعة",
    "عمليه": "عملية",  # Surgery
    "عملية جراحيه": "عملية جراحية",
    "تطعيم": "تطعيم",  # Vaccination (standardize)
    
    # Vital Signs (علامات حيوية)
    "ضغط الدم": "ضغط الدم",  # Blood pressure (standardize)
    "نبض": "نبض",  # Pulse (standardize)
    "حراره": "حرارة",  # Temperature
    "تنفس": "تنفس",  # Breathing (standardize)
    "اكسجين": "أكسجين",  # Oxygen
    
    # Common typos from Gulf/Egyptian dialects
    "دكتور": "طبيب",  # Doctor (standardize to MSA)
    "مريضه": "مريضة",  # Patient (female, fix hamza)
    "مستشفا": "مستشفى",  # Hospital
    "المستشفا": "المستشفى",
    "عياده": "عيادة",  # Clinic
    "صيدليه": "صيدلية",  # Pharmacy
}

# Dialect-specific corrections (Egyptian → MSA)
EGYPTIAN_TO_MSA = {
    "وجع": "ألم",  # Pain
    "بيوجعني": "يؤلمني",  # It hurts me
    "دماغ": "رأس",  # Head
    "بطن": "معدة",  # Stomach (in medical context)
    "برد": "زكام",  # Cold
    "سخونيه": "حمّى",  # Fever
    "دكتور": "طبيب",  # Doctor
}

# Gulf dialect corrections (Gulf → MSA)
GULF_TO_MSA = {
    "يعورني": "يؤلمني",  # It hurts me
    "حراره": "حرارة",  # Temperature
    "دوا": "دواء",  # Medicine
    "دكتور": "طبيب",  # Doctor
}

# Levantine dialect corrections (Levantine → MSA)
LEVANTINE_TO_MSA = {
    "وجع": "ألم",  # Pain
    "بوجعني": "يؤلمني",  # It hurts me
    "راس": "رأس",  # Head
    "دكتور": "طبيب",  # Doctor
}


def apply_corrections(text: str, dialect: str = "egypt") -> tuple[str, int]:
    """
    Apply medical corrections to Arabic text
    
    Args:
        text: Input text to correct
        dialect: Dialect to normalize from (egypt, gulf, levant)
    
    Returns:
        (corrected_text, num_corrections)
    """
    corrected = text
    corrections_count = 0
    
    # Apply dialect-specific corrections first
    dialect_map = {
        "egypt": EGYPTIAN_TO_MSA,
        "gulf": GULF_TO_MSA,
        "levant": LEVANTINE_TO_MSA,
    }
    
    if dialect in dialect_map:
        for wrong, correct in dialect_map[dialect].items():
            if wrong in corrected:
                corrected = corrected.replace(wrong, correct)
                corrections_count += 1
    
    # Apply general medical corrections
    for wrong, correct in MEDICAL_CORRECTIONS.items():
        if wrong in corrected:
            corrected = corrected.replace(wrong, correct)
            corrections_count += 1
    
    return corrected, corrections_count


def normalize_vital_signs(text: str) -> str:
    """
    Normalize vital signs to standard format
    Examples:
        "120 على 80" → "ضغط الدم: 120/80 mmHg"
        "38 درجه" → "الحرارة: 38°C"
    """
    import re
    
    # Blood pressure: "120 على 80" or "120/80"
    bp_pattern = r"(\d{2,3})\s*(على|\/)\s*(\d{2,3})"
    text = re.sub(bp_pattern, r"ضغط الدم: \1/\3 mmHg", text)
    
    # Temperature: "38 درجه" or "38 درجة"
    temp_pattern = r"(\d{2}\.?\d?)\s*(درجه|درجة|°)"
    text = re.sub(temp_pattern, r"الحرارة: \1°C", text)
    
    # Heart rate: "80 نبضه" or "80 bpm"
    hr_pattern = r"(\d{2,3})\s*(نبضه|نبضة|bpm)"
    text = re.sub(hr_pattern, r"النبض: \1 نبضة/دقيقة", text)
    
    # Oxygen saturation: "98 اكسجين" or "98%"
    o2_pattern = r"(\d{2,3})\s*%?\s*(اكسجين|أكسجين|O2)"
    text = re.sub(o2_pattern, r"الأكسجين: \1%", text)
    
    return text


if __name__ == "__main__":
    # Test corrections
    test_texts = [
        "المريض يشكو من الام في البروستاتا",
        "عنده حمى وسعال وضغط الدم 120 على 80",
        "الدكتور وصف مضاد حيوي ومسكن للوجع",
    ]
    
    print("Testing Medical Corrections:")
    print("=" * 60)
    for text in test_texts:
        corrected, count = apply_corrections(text, dialect="egypt")
        normalized = normalize_vital_signs(corrected)
        print(f"Original:  {text}")
        print(f"Corrected: {normalized}")
        print(f"Changes:   {count} corrections made")
        print("-" * 60)
