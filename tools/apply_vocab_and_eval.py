import json
from pathlib import Path
import re

json_path = Path('services/asr/medical_vocab_ar_en.json')
ref_path = Path('reference_test1.txt')
hyp_path = Path('tmp_asr_test1.txt')

voc = json.loads(json_path.read_text(encoding='utf-8'))
repl = voc.get('replacements', {})
amb = voc.get('ambiguous_replacements', {})
med_keys = set([k for k in voc.get('medical_keywords', [])])
preserve_dialect = bool(voc.get('preserve_dialect', False))
dialect_terms = [t for t in voc.get('dialect_terms', [])]

ref = ref_path.read_text(encoding='utf-8').strip()
hyp = hyp_path.read_text(encoding='utf-8').strip()

# apply global replacements (word-boundary aware)
for wrong, right in repl.items():
    hyp = re.sub(rf"(?<!\w){re.escape(wrong)}(?!\w)", right, hyp)

# context detection
norm = hyp.lower()
has_med = any(k in norm for k in med_keys)
has_dialect = False
if preserve_dialect and dialect_terms:
    low = hyp.lower()
    for dt in dialect_terms:
        if dt and dt in low:
            has_dialect = True
            break

# emulate windowed ambiguous-apply logic from app.post_process_text
def _norm(s):
    # simple lowercase normalization for this tool (mirrors app normalization roughly)
    return s.lower()

tokens = re.findall(r"[\w\u0600-\u06FF]+", norm)
med_keys_norm = set(_norm(k) for k in med_keys)

if amb:
    # if segment-level medical context and not dialect-preserve, apply all ambiguous
    if has_med and not (preserve_dialect and has_dialect):
        for wrong, right in amb.items():
            hyp = re.sub(rf"(?<!\w){re.escape(wrong)}(?!\w)", right, hyp)
    else:
        # windowed application: for each ambiguous item, apply if a medical keyword exists ±3 tokens
        window = 3
        reports = []
        for wrong, right in amb.items():
            wrong_norm = _norm(wrong)
            indices = [i for i, tok in enumerate(tokens) if tok == wrong_norm]
            applied = False
            for idx in indices:
                start = max(0, idx - window)
                end = min(len(tokens), idx + window + 1)
                neighborhood = tokens[start:end]
                if any(mk in med_keys_norm for mk in neighborhood):
                    # apply replacement globally for simplicity (mirroring app behavior)
                    hyp = re.sub(rf"(?<!\w){re.escape(wrong)}(?!\w)", right, hyp)
                    applied = True
                    reports.append((wrong, right, idx, neighborhood, True))
                    break
            if not applied and indices:
                for idx in indices:
                    start = max(0, idx - window)
                    end = min(len(tokens), idx + window + 1)
                    neighborhood = tokens[start:end]
                    reports.append((wrong, right, idx, neighborhood, False))

        # print a concise report for ambiguous tokens
        if reports:
            print('\nAmbiguous replacements report:')
            for wrong, right, idx, neighborhood, applied in reports:
                status = 'APPLIED' if applied else 'SKIPPED'
                print(f"  {status}: '{wrong}' -> '{right}' at token idx {idx}; neighborhood={neighborhood}")

# write transformed hypothesis
tpath = Path('tmp_asr_test1_postproc.txt')
tpath.write_text(hyp, encoding='utf-8')

# compute WER/CER using previous script logic
import runpy
mod = runpy.run_path('tools/compute_asr_errors.py')
compute_word_errors = mod['compute_word_errors']
compute_char_errors = mod['compute_char_errors']

w = compute_word_errors(ref, hyp)
c = compute_char_errors(ref, hyp)

print('After applying vocab replacements:')
print('WER = {:.2%} (S={}, D={}, I={})'.format(w['WER'], w['S'], w['D'], w['I']))
print('CER = {:.2%} (dist {}/{})'.format(c['CER'], c['dist'], c['len']))

# show top substitutions
subs = []
for op,a,b in w['ops']:
    if op=='S':
        subs.append((a,b))
from collections import Counter
cnt = Counter(subs)
print('\nTop substitutions after postproc:')
for (a,b),c0 in cnt.most_common(20):
    print(f"  '{a}' -> '{b}'  x{c0}")

print('\nIf you want to simulate adding a correction via the new /vocab/update endpoint, call the server with a JSON payload:')
print('  {"wrong":"لسا", "right":"اللثة", "ambiguous": true}')
