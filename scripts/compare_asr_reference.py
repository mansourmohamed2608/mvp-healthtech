#!/usr/bin/env python3
"""
Compare ASR output vs reference transcript and produce a short report with
- sentence alignment (simple heuristic)
- distance scores (Levenshtein ratio)
- suggested replacement map entries for CONFUSION_MAP

Usage:
  python scripts\compare_asr_reference.py <asr_file> <reference_file> [--out outputs/comparison.txt]

This is a lightweight tool (no heavy deps). It uses a simple whitespace/token edit-distance
calculation and heuristic sentence splitting. It's designed to help generate a safe
confusion map for `services/asr/text_fix_ar.py`.
"""
import sys
import argparse
from pathlib import Path
import difflib


def read_text(p: Path) -> str:
    return p.read_text(encoding='utf-8').strip()


def split_sentences_simple(s: str):
    # split on punctuation and long pauses; fallback to newline/phrase chunks
    import re
    parts = re.split(r'[۔.?!؟]\s+|\n+', s)
    parts = [p.strip() for p in parts if p.strip()]
    # if too short, split by commas/pauses
    if len(parts) < 2:
        parts = [p.strip() for p in re.split(r'[،,]\s*', s) if p.strip()]
    return parts


def normalized_tokens(s: str):
    return [t for t in s.split() if t.strip()]


def token_similarity(a: str, b: str) -> float:
    # use SequenceMatcher ratio as a quick similarity measure
    return difflib.SequenceMatcher(None, a, b).ratio()


def make_suggestions(asr_sent, ref_sent):
    # find tokens present in reference but different in ASR and suggest map
    a_tokens = asr_sent.split()
    r_tokens = ref_sent.split()
    suggestions = []
    # naive: for each token in ref, find closest token in asr
    for r in r_tokens:
        best = None
        best_score = 0.0
        for a in a_tokens:
            score = token_similarity(a, r)
            if score > best_score:
                best_score = score
                best = a
        if best and best != r and best_score > 0.5:
            suggestions.append((best, r, best_score))
    return suggestions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('asr')
    ap.add_argument('ref')
    ap.add_argument('--out', default='outputs/tmp_asr_comparison.txt')
    args = ap.parse_args()

    asr_txt = read_text(Path(args.asr))
    ref_txt = read_text(Path(args.ref))

    asr_sents = split_sentences_simple(asr_txt)
    ref_sents = split_sentences_simple(ref_txt)

    # align by best matching sequence using greedy matching
    report_lines = []
    report_lines.append('ASR vs Reference comparison report')
    report_lines.append('')
    i = 0
    for idx, r in enumerate(ref_sents):
        # find best match in ASR sents
        best_j = None
        best_score = 0.0
        for j in range(max(0, i), min(len(asr_sents), i+5)):
            score = token_similarity(asr_sents[j], r)
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is None:
            # fallback: compare to next ASR sentence
            best_j = i if i < len(asr_sents) else len(asr_sents)-1
            best_score = token_similarity(asr_sents[best_j], r)
        report_lines.append(f'Reference [{idx+1}]: {r}')
        report_lines.append(f'ASR match  [{best_j+1}]: {asr_sents[best_j]}')
        report_lines.append(f'  Similarity: {best_score:.3f}')
        # list suggested token-level replacements
        sugg = make_suggestions(asr_sents[best_j], r)
        if sugg:
            report_lines.append('  Suggested mappings (ASR -> REF, score):')
            for a, b, sc in sugg:
                report_lines.append(f'    "{a}" -> "{b}"  ({sc:.2f})')
        else:
            report_lines.append('  No strong token suggestions.')
        report_lines.append('')
        i = best_j + 1

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f'Wrote comparison report to: {outp}')

if __name__ == '__main__':
    main()
