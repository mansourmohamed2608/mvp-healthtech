import json
from jiwer import wer

def main(ref_jsonl: str, hyp_jsonl: str):
    refs = [json.loads(line) for line in open(ref_jsonl)]
    hyps = [json.loads(line) for line in open(hyp_jsonl)]
    assert len(refs) == len(hyps)
    scores = [wer(r['text'], h['text']) for r, h in zip(refs, hyps)]
    print(f"Average WER: {sum(scores)/len(scores)*100:.2f}%")

if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])
