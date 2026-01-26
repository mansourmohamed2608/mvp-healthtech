import re
from typing import Dict, Any

EGYPT_PHONE_RE = re.compile(r"\b01\d{9}\b")
SAUDI_PHONE_RE = re.compile(r"\b05\d{8}\b")
PHONE_RE = re.compile(r"(?:\+?\d{1,3})?\s?\d{9,12}")
DOB_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
DOCTOR_RE = re.compile(r"(?:دكتور|د\.|د\s)[\s\.:]*([\w\u0600-\u06FF]+)")

VISIT_MAP = {
    "كشف جديد": "كشف جديد",
    "جديد": "كشف جديد",
    "متابعة": "متابعة",
    "استشارة": "استشارة أونلاين",
    "أونلاين": "استشارة أونلاين",
}

SPECIALTIES = ["باطنة", "جلدية", "أسنان", "عظام", "قلب", "أنف", "أذن", "حنجرة", "نساء", "ولادة", "أطفال"]
CANONICAL_KEYS = {
    "name": "الاسم",
    "phone": "رقم الهاتف",
    "dob": "تاريخ الميلاد",
    "visit_type": "نوع الزيارة",
    "specialty": "التخصص",
    "doctor_name": "الطبيب",
    "date": "التاريخ المفضل",
    "time": "الوقت المفضل",
    "no_marketing": "المكالمات الدعائية",
}


def is_missing(slots: Dict[str, Any], key: str) -> bool:
    if key not in slots:
        return True
    val = slots[key]
    if isinstance(val, str):
        return val.strip() == ""
    return val is None


def extract_slots(user_text: str, slots: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(slots)
    compact = user_text.replace(" ", "")
    # Phone
    if is_missing(updated, "phone"):
        m = EGYPT_PHONE_RE.search(compact) or SAUDI_PHONE_RE.search(compact) or PHONE_RE.search(compact)
        if m:
            updated["phone"] = m.group(0)
    # DOB
    if is_missing(updated, "dob"):
        m = DOB_RE.search(user_text)
        if m:
            updated["dob"] = m.group(1)
    # Visit type
    if is_missing(updated, "visit_type"):
        for k, v in VISIT_MAP.items():
            if k in user_text:
                updated["visit_type"] = v
                break
    # Specialty
    if is_missing(updated, "specialty"):
        for spec in SPECIALTIES:
            if spec in user_text:
                updated["specialty"] = spec
                break
    # Doctor name
    if is_missing(updated, "doctor_name"):
        m = DOCTOR_RE.search(user_text)
        if m:
            updated["doctor_name"] = m.group(1)
    # No marketing
    if (
        "لا أريد مكالمات دعائية" in user_text
        or "لا تبعتوا" in user_text
        or "بدون دعاية" in user_text
        or "مش عايز إعلانات" in user_text
        or "مش عاوز إعلانات" in user_text
        or "مش حابب إعلانات" in user_text
        or "ما أبغى دعاية" in user_text
        or "ما ابغى دعاية" in user_text
        or "ما أبغى إعلانات" in user_text
        or "ما ابغى إعلانات" in user_text
    ):
        updated["no_marketing"] = True
    elif "اتصلوا" in user_text or "ما عندي مشكلة" in user_text:
        updated["no_marketing"] = False
    # Leave other slots unchanged; empty string represents missing.
    return updated
