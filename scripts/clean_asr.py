#!/usr/bin/env python3
"""
Simple ASR cleaner for `tmp_asr_test1.txt`.
- Extracts the FULL TRANSCRIPT or speaker segments if available
- Applies lightweight normalization / replacements
- Emits cleaned transcript (speaker-labeled) and a canonical terms JSON

Usage:
  python scripts\clean_asr.py <input_file> [--out-dir outputs]

Outputs (by default written to `outputs/`):
  - <basename>.cleaned.txt
  - <basename>.canonical_terms.json

This is a pragmatic, rule-based cleaner (not a full NLP pipeline). Use results
for quick LLM input; review and correct edge-cases manually before a demo.
"""
import re
import sys
import json
from pathlib import Path

REPLACEMENTS = {
    "واله": "والله",
    "بقت": "أصبحت",
    "بغسل": "أغسل",
    "بغسل": "أغسل",
    "حس": "أشعر",
    "ايوة": "نعم",
    "ايوه": "نعم",
    "ايوة": "نعم",
    "مش": "لا",
    "ما تبعتش": "لم تلتزمي",
    "ممكن": "قد",
    # common ASR artifact fixes
    "الفرشة": "الفرشاة",
}

KEYWORDS = {
    "symptoms": ["التهاب", "نزف", "حساسية", "حسّاس", "رائحة فم", "حساسية الأسنان", "ألم"],
    "findings": ["جيوب لثوية", "جيوب", "تراكم الجير", "جير"],
    "procedures": ["تنظيف عميق", "تنظيف", "إزالة الجير", "مضمضة", "خيط"],
    "meds": ["مسكنات", "مضاد", "مضادات"],
}


def load_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def extract_full_transcript(text: str) -> str:
    # try to extract the block between 'FULL TRANSCRIPT' markers
    m = re.search(r"FULL TRANSCRIPT\s*={2,}\s*(.*?)\s*={2,}", text, flags=re.S)
    if m:
        return m.group(1).strip()
    # fallback: try to find 'TRANSCRIPT WITH SPEAKERS' block
    m2 = re.search(r"TRANSCRIPT WITH SPEAKERS\s*={2,}\s*(.*?)\s*={2,}", text, flags=re.S)
    if m2:
        return m2.group(1).strip()
    # otherwise return whole file
    return text.strip()


def parse_speaker_segments(text: str):
    # Parse patterns like:
    # [1.1s - 2.1s] مريض:\n  السلام عليكم يا دكتور
    segments = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"^\[.*\]\s*([^:]+):\s*$", line)
        if m:
            speaker = m.group(1).strip()
            # next line(s) may be the utterance (indented) or on same line
            j = i + 1
            utter_lines = []
            while j < len(lines) and lines[j].strip().startswith(("",)):
                # take the next non-empty line as utterance, break after one
                if lines[j].strip():
                    utter_lines.append(lines[j].strip())
                    break
                j += 1
            if not utter_lines:
                # maybe the speaker line had the utterance after colon
                rest = re.sub(r"^\[.*\]\s*[^:]+:\s*", "", line)
                if rest:
                    utter_lines = [rest]
            if utter_lines:
                segments.append((speaker, " ".join(utter_lines)))
            i = j
        else:
            # fallback: try to detect lines that start with 'مريض:' or 'طبيب:'
            m2 = re.match(r"^([^:]{2,20}):\s*(.+)$", line)
            if m2 and any(word in m2.group(1) for word in ['مريض','طبيب','دكتور','مريضة','طبيبة']):
                speaker = m2.group(1).strip()
                utter = m2.group(2).strip()
                segments.append((speaker, utter))
            i += 1
    return segments


def apply_replacements(s: str) -> str:
    out = s
    for k, v in REPLACEMENTS.items():
        out = out.replace(k, v)
    # normalize spaces
    out = re.sub(r"\s+", " ", out).strip()
    # fix common punctuation spacing
    out = re.sub(r"\s+([،؟.!])", r"\1", out)
    # ensure sentence ends with punctuation
    if out and out[-1] not in '؟.!':
        # simple heuristic: if line starts with question word, add question mark
        if re.match(r"^(هل|متى|لماذا|كيف|أين|هل)\b", out):
            out += '؟'
        else:
            out += '.'
    return out


def extract_canonical_terms(segments):
    found = {k: [] for k in KEYWORDS}
    for speaker, text in segments:
        for k, kws in KEYWORDS.items():
            for kw in kws:
                if kw in text:
                    if kw not in found[k]:
                        found[k].append(kw)
    return found


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts\\clean_asr.py <input_file> [--out-dir outputs]")
        sys.exit(1)
    infile = Path(sys.argv[1])
    out_dir = Path('outputs')
    if '--out-dir' in sys.argv:
        idx = sys.argv.index('--out-dir')
        if idx+1 < len(sys.argv):
            out_dir = Path(sys.argv[idx+1])
    out_dir.mkdir(parents=True, exist_ok=True)

    txt = load_text(infile)
    block = extract_full_transcript(txt)
    segments = parse_speaker_segments(block)
    if not segments:
        # fallback: split block into sentences by line and treat as patient/doctor alternation
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        segs = []
        for l in lines:
            # try to split first token if it looks like 'مريض:'
            m = re.match(r"^([^:]{2,20}):\s*(.+)$", l)
            if m and any(w in m.group(1) for w in ['مريض','طبيب','دكتور','دكتورة','مريضة','طبيبة']):
                segs.append((m.group(1).strip(), m.group(2).strip()))
            else:
                segs.append(("UNKNOWN", l))
        segments = segs

    cleaned_segments = [(sp, apply_replacements(text)) for sp, text in segments]

    base = infile.stem
    cleaned_path = out_dir / f"{base}.cleaned.txt"
    with cleaned_path.open('w', encoding='utf-8') as fh:
        for sp, text in cleaned_segments:
            fh.write(f"{sp}: {text}\n")

    canonical = extract_canonical_terms([ (sp, text) for sp,text in cleaned_segments ])
    canon_path = out_dir / f"{base}.canonical_terms.json"
    with canon_path.open('w', encoding='utf-8') as fh:
        json.dump(canonical, fh, ensure_ascii=False, indent=2)

    print(f"Wrote cleaned transcript to: {cleaned_path}")
    print(f"Wrote canonical terms to: {canon_path}")


if __name__ == '__main__':
    main()
