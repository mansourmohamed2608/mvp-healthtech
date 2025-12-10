from typing import List, Dict


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


def is_missing(slots: Dict[str, any], key: str) -> bool:
    if key not in slots:
        return True
    val = slots[key]
    if isinstance(val, str):
        return val.strip() == ""
    return val is None


def build_slot_summary(slots: Dict[str, any]) -> str:
    known = []
    missing = []
    for k, label in CANONICAL_KEYS.items():
        if not is_missing(slots, k):
            display = slots.get(k)
            if k == "no_marketing":
                display = "رافض" if slots.get(k) else "موافق"
            known.append(f"{label}: {display}")
        else:
            missing.append(label)
    parts = []
    if known:
        parts.append("معلومات متوفرة: " + "; ".join(known))
    if missing:
        parts.append("مطلوب جمع: " + "; ".join(missing))
    return "\n".join(parts)


def build_va_prompt(system_prompt: str, history: List[dict], slots: Dict[str, str], user_message: str) -> str:
    """Compose prompt for VA: system + short history + slot summary + user message."""
    history_lines = []
    for turn in history[-5:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        history_lines.append(f"{role}: {content}")
    slot_summary = build_slot_summary(slots)
    prompt_parts = [
        system_prompt.strip(),
        "",
        "المحادثة السابقة (مختصرة):",
        "\n".join(history_lines) if history_lines else "لا يوجد تاريخ سابق.",
        "",
        "حالة الحقول:",
        slot_summary or "لا توجد حقول معروفة بعد.",
        "",
        "التعليمات الحالية:",
        "واصلي بصفتك ليان من مركز علاجك. استهدفي خانة ناقصة واحدة في هذا الدور، اجعلي الرد ١-٣ جمل قصيرة، وآخر جملة سؤال واضح ينتهي بـ \"؟\".",
        "",
        f"المستخدم: {user_message}",
        "المساعد:",
    ]
    return "\n".join(prompt_parts)
