from typing import List, Dict, Any


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


def build_va_prompt(
    system_prompt: str,
    history: List[dict],
    slots: Dict[str, str],
    user_message: str,
    dialect: str | None = None,
    rag_context: Dict[str, Any] | None = None,
) -> str:
    """Compose prompt for VA: system + short history + slot summary + user message."""
    history_lines = []
    for turn in history[-5:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        history_lines.append(f"{role}: {content}")
    slot_summary = build_slot_summary(slots)
    dialect_hint = ""
    if dialect == "saudi":
        dialect_hint = (
            "استخدمي لهجة سعودية محكية فقط (خليجية سعودية بسيطة). "
            "ممنوع الفصحى. أمثلة: \"هلا\"، \"وش اسمك الكامل؟\"، \"تبي\"، \"ودي\"."
        )
    elif dialect == "egypt":
        dialect_hint = (
            "استخدمي لهجة مصرية عامية فقط. ممنوع الفصحى. "
            "أمثلة: \"أهلاً بحضرتك\"، \"عايز\"، \"ممكن رقم موبايلك؟\"."
        )
    prompt_parts = [
        system_prompt.strip(),
        "",
    ]
    if dialect_hint:
        prompt_parts.extend([dialect_hint, ""])
    if rag_context:
        notes = rag_context.get("notes") or []
        faqs = rag_context.get("faqs") or []
        protocols = rag_context.get("protocols") or {}
        context_lines = []
        for note in notes[:3]:
            title = note.get("title") or "معلومة"
            text = note.get("text") or ""
            if text:
                context_lines.append(f"- {title}: {text}")
        for faq in faqs[:3]:
            question = faq.get("question") or ""
            answer = faq.get("answer") or ""
            if question and answer:
                context_lines.append(f"- سؤال: {question} | إجابة: {answer}")
        if protocols:
            hours = protocols.get("appointment_hours")
            insurance = protocols.get("insurance_accepted")
            if hours:
                context_lines.append(f"- ساعات العمل: {hours}")
            if isinstance(insurance, list) and insurance:
                context_lines.append(f"- التأمين المقبول: {', '.join(insurance)}")
        if context_lines:
            prompt_parts.extend([
                "معلومات عن العيادة:",
                "\n".join(context_lines),
                "",
            ])
    prompt_parts.extend([
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
    ])
    return "\n".join(prompt_parts)
