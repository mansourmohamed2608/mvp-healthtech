import os, argparse, random
from dataclasses import dataclass
from typing import Any, Dict, List
os.environ.setdefault("HF_HOME", "/kaggle/temp/hfhome")
os.environ.setdefault("HF_DATASETS_CACHE", "/kaggle/temp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/temp/hub")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for p in ("/kaggle/temp/hfhome","/kaggle/temp/hf","/kaggle/temp/hub"):
    os.makedirs(p, exist_ok=True)

import torch
torch.set_num_threads(2)

from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True)
    p.add_argument("--output_dir", default="/kaggle/working/lora_ckpt_med_subset")
    p.add_argument("--base_model", default="openai/whisper-large-v3")
    p.add_argument("--language", default="arabic")
    p.add_argument("--task", default="transcribe", choices=["transcribe","translate"])
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--subset_ratio", type=float, default=0.25)
    p.add_argument("--subset_n", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_fraction", type=float, default=0.01)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--keep_in_memory", action="store_true")
    return p.parse_args()

@dataclass
class Collator:
    processor: WhisperProcessor
    def __call__(self, feats: List[Dict[str, Any]]):
        import torch
        x = torch.tensor([f["input_features"] for f in feats], dtype=torch.float32)
        yb = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        y = yb["input_ids"]
        y = y.masked_fill(y == self.processor.tokenizer.pad_token_id, -100)
        return {"input_features": x, "labels": y}

def resolve(rel, mdir, root):
    if os.path.isabs(rel): return rel
    p1 = os.path.join(mdir, rel)
    return p1 if os.path.exists(p1) else os.path.join(root, rel)

def good_audio(path):
    if not (os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path)>0): return False
    try:
        import torchaudio
        w, sr = torchaudio.load(path)
        return w.numel()>0
    except Exception:
        try:
            import soundfile as sf
            w, sr = sf.read(path);  return (w is not None) and (getattr(w,"size",0)>0)
        except Exception:
            return False

def main():
    a = parse_args()
    print("Loading processor/model…")
    proc = WhisperProcessor.from_pretrained(a.base_model, language=a.language, task=a.task)
    model = WhisperForConditionalGeneration.from_pretrained(a.base_model, load_in_8bit=True, device_map="auto")
    model.generation_config.forced_decoder_ids = proc.get_decoder_prompt_ids(language=a.language, task=a.task)
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM",
                      target_modules=["q_proj","k_proj","v_proj","out_proj"])
    model = get_peft_model(model, lcfg); model.config.use_cache=False; model.gradient_checkpointing_enable()

    mdir = os.path.dirname(os.path.abspath(a.csv_path))
    root = os.path.dirname(mdir)

    print("Loading CSV…")
    ds = load_dataset("csv", data_files={"train": a.csv_path}, keep_in_memory=a.keep_in_memory)["train"]
    if "audio_filepath" in ds.column_names: ds = ds.rename_column("audio_filepath","audio")
    if "text" in ds.column_names: ds = ds.rename_column("text","sentence")

    bad = []
    def keep_ok(ex):
        ap = resolve(ex["audio"], mdir, root)
        ok = good_audio(ap)
        if not ok: bad.append(ex["audio"])
        return ok
    print("Filtering missing/zero/corrupt audio…")
    ds = ds.filter(keep_ok)
    if bad: print(f"Skipped {len(bad)} bad files. First 10:", bad[:10])

    # subset with fresh dense indices
    random.seed(a.seed)
    ridx = list(range(len(ds))); random.shuffle(ridx)
    keep = ridx[: a.subset_n] if a.subset_n>0 else ridx[: max(1,int(len(ds)*a.subset_ratio))]
    ds = ds.select(keep)
    print(f"Training on {len(ds)} examples (after filtering/subset).")

    # split by ranges (no stale indices)
    n_eval = max(1, int(len(ds)*a.eval_fraction))
    train_len = max(1, len(ds)-n_eval)
    train_ds = ds.select(range(train_len))
    eval_ds  = ds.select(range(train_len, len(ds)))

    def preprocess(ex):
        ap = resolve(ex["audio"], mdir, root)
        try:
            import torchaudio, librosa
            w, sr = torchaudio.load(ap)
            if w.ndim>1: w = w.mean(dim=0)
            w = w.numpy()
            if sr != 16000: w = librosa.resample(w, sr, 16000)
        except Exception:
            import soundfile as sf, librosa, numpy as np
            w, sr = sf.read(ap)
            if sr != 16000: w = librosa.resample(w, sr, 16000)
            if getattr(w, "ndim", 1) > 1: w = w.mean(axis=1)
        feats = proc(audio=w, sampling_rate=16000)
        labels = proc.tokenizer(ex["sentence"]).input_ids
        return {"input_features": feats["input_features"][0], "labels": labels}

    print("Preprocess train…")
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names,
                            load_from_cache_file=False, cache_file_name=None, writer_batch_size=64)
    print("Preprocess eval…")
    eval_ds = eval_ds.map(preprocess, remove_columns=eval_ds.column_names,
                          load_from_cache_file=False, cache_file_name=None, writer_batch_size=64)

    coll = Collator(proc)
    targs = TrainingArguments(
        output_dir=a.output_dir, per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.gradient_accumulation, num_train_epochs=a.num_epochs,
        learning_rate=a.learning_rate, logging_steps=10, eval_strategy="steps",
        eval_steps=a.eval_steps, save_steps=0, fp16=torch.cuda.is_available(),
        remove_unused_columns=False, report_to=[]
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      tokenizer=proc.tokenizer, data_collator=coll)
    print("Start training…"); trainer.train()
    model.save_pretrained(a.output_dir); proc.save_pretrained(a.output_dir)
    print("Saved to", a.output_dir)

if __name__=="__main__": main()
