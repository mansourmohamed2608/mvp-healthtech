# services/llm/rules.py
"""
SOAP Note Validation and Medical Rules
Ensures clinical documentation follows proper structure
Target: +3-5% accuracy improvement
"""

import re
from typing import Dict, List, Tuple


class SOAPValidator:
    """Validates and enhances SOAP notes structure"""
    
    # SOAP section markers (Arabic and English)
    SOAP_SECTIONS = {
        "subjective": ["Subjective", "الشكوى", "الأعراض", "S:", "الشكوى الرئيسية"],
        "objective": ["Objective", "الفحص", "الفحص السريري", "O:", "الفحص البدني"],
        "assessment": ["Assessment", "التشخيص", "التقييم", "A:", "التشخيص الطبي"],
        "plan": ["Plan", "الخطة", "العلاج", "P:", "خطة العلاج"]
    }
    
    # Required vital signs
    VITAL_SIGNS = [
        "blood_pressure", "temperature", "heart_rate", "respiratory_rate", "oxygen_saturation"
    ]
    
    VITAL_PATTERNS = {
        "blood_pressure": r"(?:ضغط الدم|BP|Blood Pressure):\s*(\d{2,3}/\d{2,3})\s*mmHg",
        "temperature": r"(?:الحرارة|Temp|Temperature):\s*(\d{2}\.?\d?)°C",
        "heart_rate": r"(?:النبض|HR|Heart Rate):\s*(\d{2,3})\s*(?:نبضة/دقيقة|bpm)",
        "respiratory_rate": r"(?:التنفس|RR|Respiratory Rate):\s*(\d{1,2})\s*(?:تنفس/دقيقة|breaths/min)",
        "oxygen_saturation": r"(?:الأكسجين|SpO2|O2 Sat):\s*(\d{2,3})%"
    }
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_soap_structure(self, text: str) -> Dict:
        """
        Validate that SOAP note has all required sections
        Returns: {
            "has_all_sections": bool,
            "missing_sections": list,
            "section_order_correct": bool
        }
        """
        self.errors = []
        self.warnings = []
        
        found_sections = []
        section_positions = []
        
        # Check for each SOAP section
        for section_name, markers in self.SOAP_SECTIONS.items():
            found = False
            position = float('inf')
            
            for marker in markers:
                if marker in text:
                    found = True
                    pos = text.find(marker)
                    if pos < position:
                        position = pos
            
            if found:
                found_sections.append(section_name)
                section_positions.append((section_name, position))
            else:
                self.warnings.append(f"Missing SOAP section: {section_name}")
        
        # Check if sections are in correct order (S → O → A → P)
        section_positions.sort(key=lambda x: x[1])
        expected_order = ["subjective", "objective", "assessment", "plan"]
        actual_order = [s[0] for s in section_positions]
        order_correct = actual_order == expected_order[:len(actual_order)]
        
        if not order_correct:
            self.warnings.append(f"SOAP sections not in standard order. Found: {actual_order}")
        
        return {
            "has_all_sections": len(found_sections) == 4,
            "missing_sections": [s for s in expected_order if s not in found_sections],
            "found_sections": found_sections,
            "section_order_correct": order_correct,
            "warnings": self.warnings
        }
    
    def extract_vital_signs(self, text: str) -> Dict[str, str]:
        """Extract vital signs from SOAP note"""
        vitals = {}
        
        for vital_name, pattern in self.VITAL_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                vitals[vital_name] = match.group(1)
            else:
                self.warnings.append(f"Missing vital sign: {vital_name}")
        
        return vitals
    
    def enhance_soap_structure(self, text: str) -> str:
        """
        Add missing SOAP section headers if content exists but headers are missing
        """
        enhanced = text
        
        # If text has medical content but no SOAP headers, add them
        if "الشكوى" not in text and "Subjective" not in text and "S:" not in text:
            # Try to detect subjective content (patient complaints)
            if any(word in text for word in ["ألم", "شعور", "يشكو", "أعراض"]):
                enhanced = "**الشكوى الرئيسية (Subjective):**\n" + enhanced
        
        return enhanced
    
    def validate_medical_logic(self, soap_text: str) -> List[str]:
        """
        Check for logical inconsistencies in medical data
        Returns list of validation warnings
        """
        warnings = []
        
        # Extract vital signs
        vitals = self.extract_vital_signs(soap_text)
        
        # Validate blood pressure ranges
        if "blood_pressure" in vitals:
            bp = vitals["blood_pressure"]
            try:
                systolic, diastolic = map(int, bp.split("/"))
                if systolic < 60 or systolic > 250:
                    warnings.append(f"Unusual systolic BP: {systolic} mmHg")
                if diastolic < 40 or diastolic > 150:
                    warnings.append(f"Unusual diastolic BP: {diastolic} mmHg")
                if systolic <= diastolic:
                    warnings.append(f"Invalid BP: systolic ≤ diastolic ({systolic}/{diastolic})")
            except:
                warnings.append(f"Invalid BP format: {bp}")
        
        # Validate temperature
        if "temperature" in vitals:
            try:
                temp = float(vitals["temperature"])
                if temp < 32 or temp > 43:
                    warnings.append(f"Unusual temperature: {temp}°C (possibly fatal)")
                elif temp > 39:
                    warnings.append(f"High fever detected: {temp}°C")
            except:
                warnings.append(f"Invalid temperature format")
        
        # Validate heart rate
        if "heart_rate" in vitals:
            try:
                hr = int(vitals["heart_rate"])
                if hr < 40 or hr > 200:
                    warnings.append(f"Unusual heart rate: {hr} bpm")
            except:
                warnings.append(f"Invalid heart rate format")
        
        # Validate oxygen saturation
        if "oxygen_saturation" in vitals:
            try:
                o2 = int(vitals["oxygen_saturation"])
                if o2 < 70 or o2 > 100:
                    warnings.append(f"Unusual O2 saturation: {o2}%")
                elif o2 < 90:
                    warnings.append(f"Low O2 detected: {o2}% (hypoxia)")
            except:
                warnings.append(f"Invalid O2 saturation format")
        
        return warnings


def normalize_medical_abbreviations(text: str) -> str:
    """
    Standardize medical abbreviations to full forms
    Example: "BP" → "ضغط الدم (Blood Pressure)"
    """
    abbreviations = {
        r"\bBP\b": "ضغط الدم (BP)",
        r"\bHR\b": "النبض (HR)",
        r"\bRR\b": "معدل التنفس (RR)",
        r"\bTemp\b": "الحرارة (Temp)",
        r"\bSpO2\b": "الأكسجين (SpO2)",
        r"\bCBC\b": "تحليل الدم الشامل (CBC)",
        r"\bBUN\b": "نيتروجين اليوريا (BUN)",
        r"\bECG\b": "تخطيط القلب (ECG)",
        r"\bCXR\b": "أشعة الصدر (CXR)",
        r"\bMRI\b": "الرنين المغناطيسي (MRI)",
        r"\bCT\b": "الأشعة المقطعية (CT)",
        r"\bIV\b": "وريدي (IV)",
        r"\bPO\b": "فموي (PO)",
    }
    
    for abbrev, full_form in abbreviations.items():
        text = re.sub(abbrev, full_form, text, flags=re.IGNORECASE)
    
    return text


def extract_medications(text: str) -> List[Dict]:
    """
    Extract medication information from SOAP plan section
    Returns: [{"name": str, "dose": str, "frequency": str, "duration": str}]
    """
    medications = []
    
    # Common medication patterns
    # Arabic: "باراسيتامول 500 ملغ كل 6 ساعات لمدة 3 أيام"
    # English: "Paracetamol 500mg every 6 hours for 3 days"
    
    med_pattern = r"([A-Za-z\u0600-\u06FF]+)\s+(\d+\s*(?:mg|ملغ|g|غ))\s+(?:كل|every)\s+(\d+)\s+(?:ساعات|hours)\s+(?:لمدة|for)\s+(\d+)\s+(?:أيام|days)"
    
    matches = re.finditer(med_pattern, text, re.IGNORECASE)
    for match in matches:
        medications.append({
            "name": match.group(1),
            "dose": match.group(2),
            "frequency": f"Every {match.group(3)} hours",
            "duration": f"{match.group(4)} days"
        })
    
    return medications


if __name__ == "__main__":
    # Test SOAP validation
    test_soap = """
    الشكوى الرئيسية: المريض يشكو من ألم في الصدر
    
    الفحص السريري:
    - ضغط الدم: 140/90 mmHg
    - الحرارة: 37.5°C
    - النبض: 85 نبضة/دقيقة
    - الأكسجين: 98%
    
    التشخيص: ارتفاع ضغط الدم
    
    خطة العلاج:
    - باراسيتامول 500 ملغ كل 6 ساعات لمدة 3 أيام
    - مراجعة بعد أسبوع
    """
    
    validator = SOAPValidator()
    
    print("SOAP Validation Test:")
    print("=" * 60)
    structure = validator.validate_soap_structure(test_soap)
    print(f"Has all sections: {structure['has_all_sections']}")
    print(f"Found sections: {structure['found_sections']}")
    print(f"Order correct: {structure['section_order_correct']}")
    
    vitals = validator.extract_vital_signs(test_soap)
    print(f"\nVital Signs: {vitals}")
    
    warnings = validator.validate_medical_logic(test_soap)
    print(f"\nMedical Logic Warnings: {warnings}")
    
    meds = extract_medications(test_soap)
    print(f"\nMedications: {meds}")
