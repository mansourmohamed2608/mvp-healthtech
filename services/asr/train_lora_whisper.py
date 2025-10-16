#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA fine-tuning for Whisper on Arabic medical data (robust for Kaggle/Colab).

Key features:
- Robust path resolution for manifest-relative audio.
- Skips missing/zero-byte/corrupt files and logs them.
- Optional subsetting (ratio or count) to fit RAM/disk.
- No datasets disk cache (prevents Arrow .cache from filling disk).
- 8-bit quantized base model via BitsAndBytes + LoRA.
- Optional small eval split with WER metric.
- Works with torchaudio or librosa (new 0.10+ keyword-only API).
"""

from __future__ import annotations
import os
import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
import datasets  # just to toggle caching
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune Whisper on Arabic medical data")
    p.add_argument("--csv_path", type=str, required=True,
                   help="Path to CSV with columns: audio_filepath(or audio), text(or sentence)")
    p.add_argument("--output_dir", type=str, default="lora_ckpt_med",
                   help="Where to save LoRA adapter + processor + logs")
    p.add_argument("--base_model", type=str, default="openai/whisper-large-v3",
                   help="Whisper model ID (e.g. openai/whisper-large-v3)")
    p.add_argument("--language", type=str, default="arabic", help="Language for decoding (e.g. 'arabic' or 'ar')")
    p.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])

    # training
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=20)

    # eval
    p.add_argument("--eval_fraction", type=float, default=0.0,
                   help="Holdout fraction [0..1]. 0 disables eval.")
    p.add_argument("--eval_steps", type=int, default=200,
                   help="Evaluate every N steps if eval_fraction>0")
    p.add_argument("--predict_max_len", type=int, default=225,
                   help="Max generated tokens for eval predictions (WER)")

    # dataset control
    p.add_argument("--subset_ratio", type=float, default=0.0,
                   help="Use only this fraction of (filtered) data. 0 = use all.")
    p.add_argument("--subset_count", type=int, default=0,
                   help="Or use exact count of (filtered) rows. 0 = ignore.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_in_memory", action="store_true",
                   help="Keep datasets in RAM (reduces disk writes on Kaggle).")

    # hinting
    p.add_argument("--use_hint", action="store_true", help="Prepend hint to targets and mask its loss")
    p.add_argument("--hint_prefix", type=str, default="ملاحظة طبية:")

    return p.parse_args()

# ---------- Utilities ----------

def set_env_for_kaggle(output_dir: str) -> None:
    """
    Keep HF caches in temp so we don't blow up /kaggle/working disk.
    Also disable datasets caching (we'll map with load_from_cache_file=False anyway).
    """
    os.makedirs(output_dir, exist_ok=True)
    tmp_home = "/kaggle/temp/hfhome"
    tmp_cache = "/kaggle/temp/hfcache"
    try:
        os.makedirs(tmp_home, exist_ok=True)
        os.makedirs(tmp_cache, exist_ok=True)
    except Exception:
        pass
    os.environ.setdefault("HF_HOME", tmp_home)
    os.environ.setdefault("HF_DATASETS_CACHE", tmp_cache)
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    datasets.disable_caching()  # global switch

def resolve_paths_setup(csv_path: str) -> Tuple[str, str]:
    manifest_dir = os.path.dirname(os.path.abspath(csv_path))
    dataset_root = os.path.dirname(manifest_dir)
    return manifest_dir, dataset_root

def robust_exists_and_audio_ok(path: str) -> bool:
    """Fast checks: exists, isfile, nonzero; then a tiny decode probe via torchaudio.info/load."""
    if not (os.path.exists(path) and os.path.isfile(path)):
        return False
    try:
        if os.path.getsize(path) <= 0:
            return False
    except Exception:
        return False
    # Probe decode (prefer torchaudio)
    try:
        import torchaudio
        # info() is cheap; if it fails, try a short load
        try:
            _ = torchaudio.info(path)
            return True
        except Exception:
            w, sr = torchaudio.load(path, frame_offset=0, num_frames=16000)  # ~1s
            return (w.numel() > 0) and (sr > 0)
    except Exception:
        # Last resort: let soundfile try – often no MP3 support, so don't require it
        try:
            import soundfile as sf
            with sf.SoundFile(path) as _f:
                return True
        except Exception:
            return False

def build_abs_path(rel_or_abs: str, manifest_dir: str, dataset_root: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    c1 = os.path.join(manifest_dir, rel_or_abs)
    return c1 if os.path.exists(c1) else os.path.join(dataset_root, rel_or_abs)

def numpyify(x) -> np.ndarray:
    import torch as _torch
    if isinstance(x, _torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# ---------- Data Collator ----------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    hint_ids: List[int]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # input features are already 80x3000 log-mels; just stack
        inputs = torch.tensor([f["input_features"] for f in features], dtype=torch.float32)
        labels_list = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels_list, return_tensors="pt")
        labels = labels_batch["input_ids"]
        attn = labels_batch["attention_mask"]
        labels = labels.masked_fill(attn.ne(1), -100)
        # mask hint prefix
        for i, f in enumerate(features):
            n = f.get("prefix_len", 0)
            if n > 0:
                labels[i, :n] = -100
        return {"input_features": inputs, "labels": labels}

# ---------- Audio preprocess ----------

def resample_to_16k(w, sr) -> np.ndarray:
    """Return mono float32 @16k. Fast torchaudio → librosa (keyword-only API) → simple decimation."""
    w = numpyify(w)
    if w.ndim > 1:
        w = w.mean(axis=0)
    if sr == 16000:
        return w.astype("float32", copy=False)

    # Try torchaudio (fast & memory-friendly)
    try:
        import torch, torchaudio
        t = torch.tensor(w, dtype=torch.float32)
        t = torchaudio.functional.resample(t, sr, 16000)
        return t.cpu().numpy().astype("float32", copy=False)
    except Exception:
        pass

    # Try librosa (0.10+ keyword-only)
    try:
        import librosa
        y = librosa.resample(y=w, orig_sr=sr, target_sr=16000)
        return y.astype("float32", copy=False)
    except Exception:
        pass

    # Fallback decimation
    step = max(1, int(round(sr / 16000)))
    return w[::step].astype("float32", copy=False)

# ---------- Main ----------

def main():
    args = parse_args()
    set_env_for_kaggle(args.output_dir)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert os.path.exists(args.csv_path), f"CSV not found: {args.csv_path}"
    manifest_dir, dataset_root = resolve_paths_setup(args.csv_path)

    # Load processor & model (8-bit)
    print("Loading processor/model…")
    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task=args.task)
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, quantization_config=bnb_cfg, device_map="auto"
    )
    # Force language/task at generation
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # LoRA
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    # Load CSV as datasets
    print("Loading CSV…")
    ds = load_dataset("csv", data_files={"train": args.csv_path})["train"]
    # Rename columns
    cols = set(ds.column_names)
    if "audio_filepath" in cols:
        ds = ds.rename_column("audio_filepath", "audio")
    if "text" in cols:
        ds = ds.rename_column("text", "sentence")
    assert "audio" in ds.column_names and "sentence" in ds.column_names, \
        f"CSV must have 'audio' and 'sentence' (or 'audio_filepath' and 'text'). Got: {ds.column_names}"

    # Filter missing/zero/corrupt
    print("Filtering missing/zero/corrupt audio…")
    bad_paths: List[str] = []

    def _keep(example: Dict[str, Any]) -> bool:
        rel = example["audio"]
        ap = build_abs_path(rel, manifest_dir, dataset_root)
        ok = robust_exists_and_audio_ok(ap)
        if not ok:
            bad_paths.append(rel)
        return ok

    ds = ds.filter(_keep)
    if bad_paths:
        print(f"Skipped {len(bad_paths)} bad files. First 10: {bad_paths[:10]}")
        try:
            with open(os.path.join(args.output_dir, "skipped_bad_files.txt"), "w", encoding="utf-8") as f:
                for pth in bad_paths:
                    f.write(f"{pth}\n")
        except Exception:
            pass

    # Optional subsetting
    n_total = len(ds)
    n_keep = n_total
    if args.subset_count > 0:
        n_keep = min(n_total, args.subset_count)
    elif args.subset_ratio > 0:
        n_keep = max(1, int(round(n_total * args.subset_ratio)))
    if n_keep < n_total:
        rng = np.random.default_rng(args.seed)
        keep_idx = rng.choice(n_total, size=n_keep, replace=False)
        keep_idx.sort()
        ds = ds.select(keep_idx.tolist())
    print(f"Training on {len(ds)} examples (after filtering/subset).")

    # Optional eval split
    eval_ds = None
    if args.eval_fraction > 0:
        n_eval = max(1, int(math.floor(len(ds) * args.eval_fraction)))
        n_eval = min(n_eval, len(ds) - 1)  # keep at least 1 for train
        perm = np.random.default_rng(args.seed + 1).permutation(len(ds))
        eval_idx = perm[:n_eval].tolist()
        train_idx = perm[n_eval:].tolist()
        eval_ds = ds.select(eval_idx)
        train_ds = ds.select(train_idx)
    else:
        train_ds = ds

    # Pre-tokenize hint IDs (for loss masking)
    hint_ids: List[int] = processor.tokenizer(args.hint_prefix).input_ids if args.use_hint else []

    # Preprocess (load audio → 16k → logmels; tokenize targets)
    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        rel = example["audio"]
        ap = build_abs_path(rel, manifest_dir, dataset_root)

        # load audio (torchaudio first, fallback to soundfile)
        try:
            import torchaudio, torch as _torch
            w, sr = torchaudio.load(ap)
            if w.ndim > 1:
                w = w.mean(dim=0)
            wav = w.cpu().numpy()
            sr_ = sr
        except Exception:
            import soundfile as sf
            wav, sr_ = sf.read(ap)

        wav16 = resample_to_16k(wav, sr_)
        inputs = processor(audio=wav16, sampling_rate=16000)
        text = example["sentence"]
        prefix_len = 0
        if args.use_hint:
            text = f"{args.hint_prefix} {text}"
            prefix_len = len(hint_ids)
        labels = processor.tokenizer(text).input_ids
        return {
            "input_features": inputs["input_features"][0],
            "labels": labels,
            "prefix_len": prefix_len,
        }

    map_kwargs = dict(
        remove_columns=train_ds.column_names,
        load_from_cache_file=False,
        cache_file_name=None,
        keep_in_memory=args.keep_in_memory,
        desc="Preprocess train",
    )
    train_ds = train_ds.map(preprocess, **map_kwargs)
    if eval_ds is not None:
        eval_ds = eval_ds.map(preprocess, **map_kwargs | {"desc": "Preprocess eval"})

    # Collator
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, hint_ids=hint_ids)

    # Compute metrics (WER) if eval
    def compute_metrics(eval_pred):
        try:
            from jiwer import wer
        except Exception:
            return {}
        pred_ids = eval_pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = eval_pred.label_ids
        # replace -100 with pad for decoding
        pad_id = processor.tokenizer.pad_token_id
        label_ids = np.where(label_ids == -100, pad_id, label_ids)
        preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer(refs, preds)}

    # Trainer args
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        fp16=use_fp16,
        bf16=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=[],
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        predict_with_generate=eval_ds is not None,
        generation_max_length=args.predict_max_len,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if eval_ds is not None else None,
        tokenizer=processor.tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )

    # Info dump
    meta = {
        "n_total_after_filter": n_total,
        "n_used_for_training": len(train_ds),
        "n_used_for_eval": 0 if eval_ds is None else len(eval_ds),
        "subset_ratio": args.subset_ratio,
        "subset_count": args.subset_count,
        "eval_fraction": args.eval_fraction,
        "seed": args.seed,
    }
    try:
        with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print("Starting training…")
    trainer.train()

    # Save adapter + processor
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Done. Artifacts saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
