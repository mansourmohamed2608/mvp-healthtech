import os, argparse, random, json, math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_DATASETS_CACHE", "/kaggle/temp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/temp/hub")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
torch.set_num_threads(2)

from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ---------- args ----------
def parse_args():
    p = argparse.ArgumentParser("LoRA Whisper on Kaggle (subset + robust I/O)")
    p.add_argument("--csv_path", required=True, type=str)
    p.add_argument("--output_dir", default="lora_ckpt_med", type=str)
    p.add_argument("--base_model", default="openai/whisper-large-v3", type=str)
    p.add_argument("--language", default="arabic", type=str)
    p.add_argument("--task", default="transcribe", choices=["transcribe","translate"])
    p.add_argument("--num_epochs", default=2, type=int)
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--gradient_accumulation", default=16, type=int)
    p.add_argument("--learning_rate", default=1e-4, type=float)
    p.add_argument("--use_hint", action="store_true")
    p.add_argument("--hint_prefix", default="ملاحظة طبية:", type=str)

    # NEW: disk/RAM friendly knobs
    p.add_argument("--subset_ratio", default=0.25, type=float, help="use this fraction of the *valid* rows (0–1)")
    p.add_argument("--subset_n", default=0, type=int, help="override ratio with exact number of rows if >0")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--eval_fraction", default=0.01, type=float, help="tiny eval slice to monitor")
    p.add_argument("--eval_steps", default=200, type=int)
    p.add_argument("--keep_in_memory", action="store_true", help="load dataset into memory to avoid cache writes")
    return p.parse_args()

# ---------- collator ----------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = torch.tensor([f["input_features"] for f in features], dtype=torch.float32)
        labels_batch = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = labels_batch["input_ids"]
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        return {"input_features": input_features, "labels": labels}

# ---------- helpers ----------
def resolve_path(rel: str, manifest_dir: str, dataset_root: str) -> str:
    if os.path.isabs(rel):
        return rel
    p1 = os.path.join(manifest_dir, rel)         # e.g. /kaggle/working/tts_ar_med/tts_ar_med/FILE.mp3
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(dataset_root, rel)         # e.g. /kaggle/working/tts_ar_med/FILE.mp3
    return p2

def good_audio(path: str) -> bool:
    if not os.path.exists(path) or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    # try torchaudio first (handles mp3); fallback to soundfile
    try:
        import torchaudio
        w, sr = torchaudio.load(path)
        return w.numel() > 0
    except Exception:
        try:
            import soundfile as sf
            w, sr = sf.read(path)
            return w is not None and (getattr(w, "size", 0) > 0)
        except Exception:
            return False

def main():
    args = parse_args()
    assert os.path.exists(args.csv_path), f"CSV not found: {args.csv_path}"

    # Processor + 8-bit base
    print("Loading processor/model…")
    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, load_in_8bit=True, device_map="auto"
    )
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )
    # LoRA
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM",
                      target_modules=["q_proj","k_proj","v_proj","out_proj"])
    model = get_peft_model(model, lcfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Resolve manifest base dirs
    manifest_dir = os.path.dirname(os.path.abspath(args.csv_path))
    dataset_root = os.path.dirname(manifest_dir)

    # Load manifest (strings only; no Audio feature → no torchcodec)
    print("Loading CSV…")
    ds = load_dataset(
        "csv",
        data_files={"train": args.csv_path},
        keep_in_memory=args.keep_in_memory
    )["train"]

    # Normalize column names
    cols = set(ds.column_names)
    if "audio_filepath" in cols:
        ds = ds.rename_column("audio_filepath", "audio")
    if "text" in cols:
        ds = ds.rename_column("text", "sentence")
    assert "audio" in ds.column_names and "sentence" in ds.column_names, "CSV needs 'audio_filepath,text' columns"

    # Filter broken/missing/zero-size
    bad = []
    def keep_ok(ex):
        abs_path = resolve_path(ex["audio"], manifest_dir, dataset_root)
        ok = good_audio(abs_path)
        if not ok:
            bad.append(ex["audio"])
        return ok

    print("Filtering missing/zero/corrupt audio…")
    ds = ds.filter(keep_ok)

    if len(bad) > 0:
        print(f"Skipped {len(bad)} bad files. First 10:", bad[:10])

    # Subset to reduce disk/RAM
    random.seed(args.seed)
    idx = list(range(len(ds)))
    random.shuffle(idx)
    if args.subset_n > 0:
        keep = idx[:args.subset_n]
    else:
        keep = idx[: max(1, int(len(ds) * args.subset_ratio))]
    ds = ds.select(keep)
    print(f"Training on {len(ds)} examples (after filtering/subset).")

    # Tiny eval slice
    n_eval = max(1, int(len(ds) * args.eval_fraction))
    eval_idx = keep[-n_eval:]
    train_idx = keep[:-n_eval] if len(keep) > n_eval else keep
    train_ds = ds.select(range(len(train_idx))) if len(keep) == len(train_idx) else ds.select(train_idx)
    eval_ds  = ds.select(range(len(train_idx), len(keep))) if len(keep) == len(train_idx)+n_eval else ds.select(eval_idx)

    # Preprocess (no cache writes)
    def preprocess(ex):
        import torchaudio, numpy as np
        apath = resolve_path(ex["audio"], manifest_dir, dataset_root)
        # load → mono → resample to 16k if needed
        try:
            w, sr = torchaudio.load(apath)
            if w.ndim > 1: w = w.mean(dim=0)
            w = w.numpy()
            if sr != 16000:
                import librosa
                w = librosa.resample(w, sr, 16000)
        except Exception as e:
            # final fallback
            import soundfile as sf, numpy as np, librosa
            w, sr = sf.read(apath)
            if sr != 16000:
                w = librosa.resample(w, sr, 16000)
            if w.ndim > 1: w = np.mean(w, axis=1)
        feats = processor(audio=w, sampling_rate=16000)
        labels = processor.tokenizer(ex["sentence"]).input_ids
        return {"input_features": feats["input_features"][0], "labels": labels}

    print("Preprocess train…")
    train_ds = train_ds.map(
        preprocess,
        remove_columns=train_ds.column_names,
        load_from_cache_file=False,
        cache_file_name=None,
        writer_batch_size=64,
    )
    print("Preprocess eval…")
    eval_ds = eval_ds.map(
        preprocess,
        remove_columns=eval_ds.column_names,
        load_from_cache_file=False,
        cache_file_name=None,
        writer_batch_size=64,
    )

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    args_train = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=0,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.tokenizer,
        data_collator=collator,
    )

    print("Start training…")
    trainer.train()
    print("Done. Saving…")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Saved to", args.output_dir)

if __name__ == "__main__":
    main()
