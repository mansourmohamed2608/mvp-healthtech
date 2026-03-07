import re
from typing import Dict, Any
from datetime import date as _date, timedelta as _timedelta

# Arabic day names → Python weekday (Mon=0 … Sun=6)
_AR_DAY_WD: dict[str, int] = {
    "الاثنين": 0,
    "الثلاثاء": 1,
    "الأربعاء": 2, "الاربعاء": 2, "الأربعا": 2,
    "الخميس": 3,
    "الجمعة": 4, "الجمعه": 4,
    "السبت": 5,
    "الأحد": 6, "الاحد": 6,
}

_AR_HOUR_MAP: dict[str, int] = {
    "الواحدة": 1, "الواحد": 1,
    "الثانية": 2, "الثاني": 2,
    "الثالثة": 3, "الثالث": 3,
    "الرابعة": 4, "الرابع": 4,
    "الخامسة": 5, "الخامس": 5,
    "السادسة": 6, "السادس": 6,
    "السابعة": 7, "السابع": 7,
    "الثامنة": 8, "الثامن": 8,
    "التاسعة": 9, "التاسع": 9,
    "العاشرة": 10, "العاشر": 10,
    "الحادية عشرة": 11, "الحادي عشر": 11,
    "الثانية عشرة": 12, "الثاني عشر": 12,
}


def _next_date_for_day(day_name: str) -> str | None:
    """Return the next occurrence of an Arabic weekday name as DD/MM/YYYY."""
    target_wd = _AR_DAY_WD.get(day_name)
    if target_wd is None:
        return None
    today = _date.today()
    days_ahead = (target_wd - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7  # always a future date
    return (today + _timedelta(days=days_ahead)).strftime("%d/%m/%Y")


def _parse_arabic_time(text: str) -> str | None:
    """Extract a time string (HH:00) from Arabic speech."""
    pm = any(w in text for w in ["مساء", "مساءً", "عصر", "عصرً", "ظهر"])
    am = any(w in text for w in ["صباح", "صباحً"])
    # Named hours (longest first to avoid partial matches)
    for name in sorted(_AR_HOUR_MAP, key=len, reverse=True):
        if name in text:
            h = _AR_HOUR_MAP[name]
            if pm and h < 12:
                h += 12
            elif not am and h <= 6:
                h += 12  # small hour with no AM context → assume PM
            return f"{h:02d}:00"
    # Numeric: "الساعة 5" or "5 مساءً"
    m = re.search(r'(?:الساعة\s+)?(\d{1,2})', text)
    if m:
        h = int(m.group(1))
        if 1 <= h <= 12:
            if pm and h < 12:
                h += 12
            elif not am and h <= 6:
                h += 12
            return f"{h:02d}:00"
    return None


# Phone: accept digits with optional spaces between them (user may say "0 10 9 5 0")
EGYPT_PHONE_RE = re.compile(r"\b0\s*1\s*\d[\s\d]{8,11}\b")
SAUDI_PHONE_RE = re.compile(r"\b0\s*5\s*\d[\s\d]{7,9}\b")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?\d[\s\d]{8,12}")
DOB_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
DOCTOR_RE = re.compile(r"(?:دكتور|د\.|د\s)[\s\.:]*([\w\u0600-\u06FF]+)")

# Name: "أنا اسمي X" / "اسمي X" / "أنا X" / "انا X"
# "أنا اسمي" must be listed FIRST so the regex engine prefers it over bare "أنا"
# when both prefixes could match (avoids capturing "اسمي X" as the name value).
NAME_RE = re.compile(r"(?:أنا اسمي|أنا انا|اسمي|أنا|انا|اسمى)\s+([\u0600-\u06FF][\u0600-\u06FF\s]{1,30}?)(?:\s*$|\s*[،,.])")

# Arabic word-digits to integer map (for phone numbers spoken as words).
# Both feminine (عشرة) and masculine (عشر) forms are included, as Arabic
# speakers use either depending on grammatical context.
ARABIC_DIGIT_MAP = {
    "صفر": "0", "زيرو": "0",
    "واحد": "1", "وحده": "1",
    "اتنين": "2", "اثنين": "2", "تنين": "2", "اثنان": "2", "اثنتين": "2",
    "تلاتة": "3", "ثلاثة": "3", "تلاتا": "3", "ثلاث": "3",
    "أربعة": "4", "اربعة": "4", "أربع": "4", "اربع": "4",
    "خمسة": "5", "خمس": "5",
    "ستة": "6", "ست": "6",
    "سبعة": "7", "سبع": "7",
    "تمانية": "8", "ثمانية": "8", "تمان": "8", "ثماني": "8", "ثمان": "8",
    "تسعة": "9", "تسع": "9",
    "عشرة": "10", "عشره": "10", "عشر": "10",
}

def _spoken_to_digits(text: str) -> str:
    """Convert spoken Arabic digit-words to numeric digits (e.g. 'زيرو عشر' → '010').

    Also strips the Arabic conjunction 'و' (and) when it appears directly
    before a digit word — e.g. 'وتسع' → 'تسع' → '9'.
    """
    result = text
    # Strip leading Arabic conjunction 'و' glued to a digit word (e.g. 'وتسع' → ' تسع')
    for word in list(ARABIC_DIGIT_MAP.keys()):
        result = result.replace('و' + word, ' ' + word)
    for word, digit in ARABIC_DIGIT_MAP.items():
        result = result.replace(word, digit)
    # After replacement, '10' in a phone context means the digit sequence '1','0' — keep as-is,
    # the regex will handle the full match.
    return result

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
    # Try word->digit conversion first (for spoken phone numbers like "زيرو عشرة ...")
    spoken_converted = _spoken_to_digits(user_text)
    compact = spoken_converted.replace(" ", "")
    # Name — "اسمي منصور" / "أنا منصور"
    if is_missing(updated, "name"):
        # Try explicit pattern first
        m = NAME_RE.search(user_text)
        if m:
            updated["name"] = m.group(1).strip()
        else:
            # Fallback: text is short and looks like just a name (no other slots)
            stripped = user_text.strip()
            words = stripped.split()
            if (
                1 <= len(words) <= 3
                and all(re.match(r'^[\u0600-\u06FF]+$', w) for w in words)
                and not any(k in stripped for k in ["رقم", "موبايل", "هاتف", "دكتور"])
            ):
                updated["name"] = stripped
    # Phone
    if is_missing(updated, "phone"):
        # Try on original (digits with spaces) and spoken-converted
        m = (EGYPT_PHONE_RE.search(user_text)
             or SAUDI_PHONE_RE.search(user_text)
             or EGYPT_PHONE_RE.search(spoken_converted)
             or SAUDI_PHONE_RE.search(spoken_converted)
             or PHONE_RE.search(compact))
        if m:
            # Normalize: strip spaces from matched group
            updated["phone"] = re.sub(r'\s+', '', m.group(0))
            # Clear any partial marker now that we have a full number
            updated.pop("partial_phone", None)
        else:
            # Detect a partial phone attempt (3+ digits in a row) so the model
            # can ask the user to give the complete number
            partial_m = re.search(r'\d[\s\d]{2,}', spoken_converted)
            if partial_m:
                digits_only = re.sub(r'\s+', '', partial_m.group(0))
                if len(digits_only) >= 3:
                    updated["partial_phone"] = digits_only
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
    # Date from Arabic day name (e.g. "يوم الثلاثاء", "الخميس")
    if is_missing(updated, "date"):
        # Try explicit DD/MM/YYYY first (already handled by DOB_RE above, but for date slot)
        m = DOB_RE.search(user_text)
        if m:
            updated["date"] = m.group(1)
        else:
            # Try Arabic day names (longest first to avoid partial matches)
            for day_name in sorted(_AR_DAY_WD, key=len, reverse=True):
                if day_name in user_text:
                    d = _next_date_for_day(day_name)
                    if d:
                        updated["date"] = d
                    break
    # Time from Arabic hour name or "الساعة N"
    if is_missing(updated, "time") and ("ساعة" in user_text or any(h in user_text for h in _AR_HOUR_MAP)):
        t = _parse_arabic_time(user_text)
        if t:
            updated["time"] = t
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
