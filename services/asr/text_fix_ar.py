# services/asr/text_fix_ar.py
"""
Arabic text post-processing for ASR output.
Provides normalization, repetition collapse, and dialect preservation.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

# --- Arabic char normalization rules
_ARABIC_CHAR_NORM: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"[إأٱآا]"), "ا"),
    (re.compile(r"[يى]"), "ي"),
    (re.compile(r"[ًٌٍَُِّْ]"), ""),  # strip diacritics
    (re.compile(r"\u0640+"), ""),    # tatweel
]

# collapse repeated Arabic letters (but keep "الله")
_AR_LETTERS = r"[ء-ي]"
_REPEAT_RE = re.compile(rf"(?<!الل)({_AR_LETTERS})\1{{1,}}")

# domain neighborhoods to gate some fixes
_NEIGHBORHOODS = [
    "اسنان", "سنان", "لث", "ضرس", "طبيب", "دكتور", 
    "علاج", "مضمض", "خيط", "اشعه", "تحليل", "سكر", "جرعة"
]

# Minimal default confusion map (most should be in medical_vocab_ar_en.json)
DEFAULT_CONFUSION_MAP: Dict[str, str] = {
    "السة": "اللثة",
    "اللسة": "اللثة",
    "القراصيم": "الجراثيم",
    "مرار الوقت": "مرور الوقت",
    "الخط": "الخيط",
    "خط طبي": "خيط طبي",
    "خط الأسنان": "خيط أسنان",
    "ممش": "مش",
    "اللسويه": "اللثوية",
    "اللسم": "اللثة",
    "اله": "الله",
}

# Tokens/phrases that should be preserved exactly when present
DIALECT_PRESERVE: List[str] = [
    "الله", "ازايك", "ازيك", "عاملة", "عامل", 
    "ايوة", "ايوه", "ازاي", "حسيت", "بغسل"
]

def normalize_arabic(s: str) -> str:
    """Apply Arabic character normalization (alif/ya variants, diacritics)."""
    out = s
    for pat, repl in _ARABIC_CHAR_NORM:
        out = pat.sub(repl, out)
    return out

def collapse_repeats(s: str) -> str:
    """Collapse repeated Arabic letters (preserves الله)."""
    return _REPEAT_RE.sub(r"\1", s)

def load_vocab_json(path_str: str) -> Tuple[List[str], Dict[str, str]]:
    """Load terms and replacements from JSON vocab file."""
    path = Path(path_str)
    if not path.exists():
        return [], {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        terms = data.get("terms", []) if isinstance(data, dict) else []
        repls = data.get("replacements", {}) if isinstance(data, dict) else {}
        clean_repl = {str(k): str(v) for k, v in repls.items()}
        return [str(t) for t in terms], clean_repl
    except Exception:
        return [], {}

# Load the project medical vocab (if present)
_DEFAULT_VOCAB_PATH = Path(__file__).parent / "medical_vocab_ar_en.json"
_VOCAB_TERMS, _VOCAB_REPLACEMENTS = [], {}
try:
    _VOCAB_TERMS, _VOCAB_REPLACEMENTS = load_vocab_json(str(_DEFAULT_VOCAB_PATH))
except Exception:
    _VOCAB_TERMS, _VOCAB_REPLACEMENTS = [], {}

# Remove duplicate keys from DEFAULT_CONFUSION_MAP when same mapping exists in vocab
for _k in list(DEFAULT_CONFUSION_MAP.keys()):
    if _k in _VOCAB_REPLACEMENTS:
        del DEFAULT_CONFUSION_MAP[_k]

def post_process_text(
    text: str,
    extra_replacements: Dict[str, str] = None
) -> str:
    """
    Apply conservative post-processing to ASR output.
    
    Args:
        text: Raw ASR text
        extra_replacements: Additional replacement mappings to apply
        
    Returns:
        Processed text with corrections applied
    """
    if not text:
        return text

    t = collapse_repeats(text)
    t = re.sub(r"[ًٌٍَُِّْ]+", "", t)  # remove stray diacritics

    # Apply neighborhood-aware confusion fixes
    norm = normalize_arabic(t)
    repl_map = dict(DEFAULT_CONFUSION_MAP)
    if extra_replacements:
        repl_map.update(extra_replacements)

    # Only apply if in medical context
    if any(nb in norm for nb in _NEIGHBORHOODS):
        for wrong, right in repl_map.items():
            t = re.sub(rf"(?<!\w){re.escape(wrong)}(?!\w)", right, t)

    # Canonical preferences (safe across contexts)
    t = t.replace("سنان", "أسنان")
    t = t.replace("خط طبي", "خيط طبي").replace("خط الأسنان", "خيط أسنان")
    t = t.replace("السة", "اللثة").replace("اللسة", "اللثة")
    
    return t