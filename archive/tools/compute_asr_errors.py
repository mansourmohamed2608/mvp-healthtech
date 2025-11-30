import sys
from pathlib import Path

def levenshtein(a, b):
    # returns distance matrix and backtrace
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp


def backtrace_ops(dp, a, b):
    i, j = len(a), len(b)
    ops = []  # (op, a_tok, b_tok)
    while i>0 or j>0:
        if i>0 and j>0 and dp[i][j] == dp[i-1][j-1] and a[i-1]==b[j-1]:
            ops.append(('=', a[i-1], b[j-1]))
            i -= 1; j -= 1
        elif i>0 and j>0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(('S', a[i-1], b[j-1])); i-=1; j-=1
        elif i>0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(('D', a[i-1], '')) ; i-=1
        else:
            ops.append(('I', '', b[j-1])); j-=1
    ops.reverse()
    return ops


def compute_word_errors(ref, hyp):
    a = ref.split()
    b = hyp.split()
    dp = levenshtein(a,b)
    ops = backtrace_ops(dp,a,b)
    S = sum(1 for o in ops if o[0]=='S')
    D = sum(1 for o in ops if o[0]=='D')
    I = sum(1 for o in ops if o[0]=='I')
    N = len(a)
    wer = (S + D + I) / N if N>0 else 0.0
    return {'N':N,'S':S,'D':D,'I':I,'WER':wer,'ops':ops}


def compute_char_errors(ref, hyp):
    a = list(ref.replace(' ',''))
    b = list(hyp.replace(' ',''))
    dp = levenshtein(a,b)
    # ops not needed
    dist = dp[len(a)][len(b)]
    cer = dist / max(1, len(a))
    return {'dist':dist, 'len':len(a), 'CER':cer}


def top_substitutions(ops):
    subs = {}
    for op,a,b in ops:
        if op=='S':
            key = (a,b)
            subs[key] = subs.get(key,0)+1
    sorted_subs = sorted(subs.items(), key=lambda x: -x[1])
    return sorted_subs


def main():
    ref_path = Path('reference_test1.txt')
    hyp_path = Path('tmp_asr_test1.txt')
    if not ref_path.exists():
        print('Reference file not found:', ref_path)
        sys.exit(1)
    if not hyp_path.exists():
        print('Hypothesis file not found:', hyp_path)
        sys.exit(1)
    ref = ref_path.read_text(encoding='utf-8').strip()
    hyp = hyp_path.read_text(encoding='utf-8').strip()
    print('Reference length (chars):', len(ref), 'words:', len(ref.split()))
    print('ASR length       (chars):', len(hyp), 'words:', len(hyp.split()))
    print('\nComputing word-level errors...')
    w = compute_word_errors(ref, hyp)
    print('N (ref words):', w['N'])
    print('Substitutions:', w['S'], 'Deletions:', w['D'], 'Insertions:', w['I'])
    print('WER = {:.2%}'.format(w['WER']))
    print('\nComputing char-level errors...')
    c = compute_char_errors(ref, hyp)
    print('CER = {:.2%} (distance {}/{})'.format(c['CER'], c['dist'], c['len']))

    subs = top_substitutions(w['ops'])
    print('\nTop substitutions (ref -> asr)')
    for (a,b),cnt in subs[:30]:
        print(f"  '{a}' -> '{b}'  x{cnt}")

    # show ops with context around errors (print first 120 ops)
    print('\nSample alignment ops (op,ref,asr) [first 200 tokens]:')
    for op,a,b in w['ops'][:200]:
        print(op, a, '->', b)

if __name__=='__main__':
    main()
