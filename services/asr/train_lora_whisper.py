#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune Whisper with LoRA on Arabic data.

Key features
- Skips missing/zero/corrupt audio safely (and writes a list to disk).
- Robust path resolution for relative audio paths in manifest CSV.
- Custom Trainer that always calls the model with input_features (fixes PEFT/Whisper crash).
- Loads audio on-the-fly in the collator (lower disk usage; no datasets.Audio decode).
- Optional subsetting after filtering to fit Kaggle/Colab resource limits.
"""

import os
import sys
import math
import json
import random
import argparse
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

try:
    from transformers import BitsAndBytesConfig  # for 8-bit load (preferred)
except Exception:
    BitsAndBytesConfig = None

from peft import LoraConfig, get_peft_model


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LoRA fine-tuning for Whisper (Arabic)")
    p.add_argument("--csv_path", required=True, type=str,
                   help="Path to manifest CSV with columns: audio_filepath (or audio) and text (or sentence).")
    p.add_argument("--output_dir", default="lora_ckpt", type=str)
    p.add_argument("--base_model", default="openai/whisper-large-v3", type=str)
    p.add_argument("--language", default="arabic", type=str)
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--num_epochs", default=2, type=int)
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--gradient_accumulation", default=16, type=int)
    p.add_argument("--learning_rate", default=1e-4, type=float)
    p.add_argument("--eval_fraction", default=0.0, type=float,
                   help="Fraction for eval split (0.0 disables eval).")
    p.add_argument("--subset_count", default=None, type=int,
                   help="If set, randomly train on this many rows AFTER filtering.")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--num_proc", default=2, type=int, help="Data loader workers")
    p.add_argument("--save_steps", default=400, type=int)
    p.add_argument("--eval_steps", default=400, type=int)
    p.add_argument("--use_hint", action="store_true")
    p.add_argument("--hint_prefix", default="ملاحظة طبية:", type=str)
    return p.parse_args()


# ----------------------------
# utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_abs_path(rel: str, manifest_dir: str, dataset_root: str) -> str:
    if os.path.isabs(rel):
        return rel
    cand1 = os.path.join(manifest_dir, rel)
    return cand1 if os.path.exists(cand1) else os.path.join(dataset_root, rel)


def probe_ok(path: str) -> bool:
    try:
        # fast stat checks
        if not (os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) > 0):
            return False
        # quick open test with soundfile to catch pipes/broken files
        import soundfile as sf
        with sf.SoundFile(path):
            pass
        return True
    except Exception:
        return False


def filter_manifest(paths: List[str], manifest_dir: str, dataset_root: str) -> Tuple[List[bool], List[str]]:
    keep_mask, bad = [], []
    for rel in paths:
        ap = build_abs_path(rel, manifest_dir, dataset_root)
        ok = probe_ok(ap)
        keep_mask.append(ok)
        if not ok:
            bad.append(rel)
    return keep_mask, bad


# ----------------------------
# Collator: loads audio now
# ----------------------------
class LazyAudioCollator:
    def __init__(self, processor: WhisperProcessor, hint_ids: List[int]):
        self.processor = processor
        self.hint_ids = hint_ids

    def _load_16k(self, path: str) -> np.ndarray:
        # Try torchaudio first (fast resample), fall back to soundfile+librosa
        try:
            import torchaudio
            w, sr = torchaudio.load(path)  # [C, T]
            if w.ndim > 1:
                w = w.mean(dim=0)
            if sr != 16000:
                w = torchaudio.functional.resample(w, sr, 16000)
            return w.cpu().numpy().astype("float32", copy=False)
        except Exception:
            import soundfile as sf
            import librosa
            w, sr = sf.read(path)
            if isinstance(w, np.ndarray) and w.ndim > 1:
                w = w.mean(axis=1)
            if sr != 16000:
                # librosa signature requires keywords
                w = librosa.resample(y=w, orig_sr=sr, target_sr=16000)
            return w.astype("float32", copy=False)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Build log-mels
        feats = []
        for f in features:
            wav16 = self._load_16k(f["audio_path"])
            x = self.processor(audio=wav16, sampling_rate=16000)["input_features"][0]
            feats.append(x)
        # batch stack with numpy → tensor
        input_features = torch.from_numpy(np.stack(feats, axis=0)).float()

        # labels pad + mask
        label_features = [{"input_ids": f["labels"]} for f in features]
        padded = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = padded["input_ids"]
        attn = padded["attention_mask"]
        labels = labels.masked_fill(attn.ne(1), -100)
        for i, f in enumerate(features):
            n = f.get("prefix_len", 0)
            if n > 0:
                labels[i, :n] = -100
        return {"input_features": input_features, "labels": labels}


# ----------------------------
# Custom Trainer to avoid PEFT+Whisper kwarg mismatch
# ----------------------------
class SpeechTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(input_features=inputs["input_features"], labels=inputs.get("labels"))
        loss = out.loss
        return (loss, out) if return_outputs else loss


# ----------------------------
# main
# ----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    assert os.path.exists(args.csv_path), f"CSV not found: {args.csv_path}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    manifest_dir = os.path.dirname(os.path.abspath(args.csv_path))
    dataset_root = os.path.dirname(manifest_dir)

    print("Loading processor/model…")
    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task=args.task)

    # Prefer 8-bit quantization when available (saves VRAM)
    model = None
    if BitsAndBytesConfig is not None:
        try:
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            model = WhisperForConditionalGeneration.from_pretrained(
                args.base_model, quantization_config=bnb, device_map="auto"
            )
        except Exception as e:
            print("8-bit load failed, falling back to full precision:", repr(e))
    if model is None:
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    # force target language/task
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    print("Loading CSV…")
    ds = load_dataset("csv", data_files={"train": args.csv_path})["train"]
    # normalize column names
    if "audio_filepath" in ds.column_names:
        ds = ds.rename_column("audio_filepath", "audio")
    if "text" in ds.column_names:
        ds = ds.rename_column("text", "sentence")

    print("Filtering missing/zero/corrupt audio…")
    keep_mask, bad_files = filter_manifest(ds["audio"], manifest_dir, dataset_root)
    kept = ds.filter(lambda ex, idx: keep_mask[idx], with_indices=True)
    if bad_files:
        bad_path = os.path.join(args.output_dir, "bad_files.txt")
        with open(bad_path, "w", encoding="utf-8") as f:
            for b in bad_files:
                f.write(b + "\n")
        print(f"Skipped {len(bad_files)} bad files. First 10: {bad_files[:10]}")
        print(f"List saved to: {bad_path}")
    print(f"Kept {len(kept):,} rows after filtering.")

    # Optional subset AFTER filtering to avoid index errors
    if args.subset_count is not None and args.subset_count > 0:
        n = min(args.subset_count, len(kept))
        kept = kept.shuffle(seed=args.seed).select(range(n))
        print(f"Training on {n:,} examples (subset).")
    else:
        print(f"Training on {len(kept):,} examples (full after filtering).")

    # Optional eval split
    eval_ds = None
    if args.eval_fraction and args.eval_fraction > 0:
        split = kept.train_test_split(test_size=args.eval_fraction, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
        print(f"Eval set size: {len(eval_ds):,}")
    else:
        train_ds = kept

    # Pre-tokenize targets + absolute audio path (keep audio loading in collator)
    hint_ids = processor.tokenizer(args.hint_prefix).input_ids if args.use_hint else []

    def to_meta(batch: Dict[str, Any]) -> Dict[str, Any]:
        ap = build_abs_path(batch["audio"], manifest_dir, dataset_root)
        txt = batch["sentence"]
        prefix_len = 0
        if args.use_hint:
            txt = f"{args.hint_prefix} {txt}"
            prefix_len = len(hint_ids)
        labels = processor.tokenizer(txt).input_ids
        return {"audio_path": ap, "labels": labels, "prefix_len": prefix_len}

    train_ds = train_ds.map(
        to_meta,
        remove_columns=train_ds.column_names,
        load_from_cache_file=False,
        cache_file_name=None,   # in-memory to reduce disk I/O
    )
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            to_meta,
            remove_columns=eval_ds.column_names,
            load_from_cache_file=False,
            cache_file_name=None,
        )

    collator = LazyAudioCollator(processor=processor, hint_ids=hint_ids)

    # TrainingArguments — version-adaptive (old Transformers may not accept evaluation_strategy)
    sig = inspect.signature(TrainingArguments.__init__)
    ta_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=25,
        save_steps=args.save_steps,
        save_total_limit=1,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=args.num_proc,
        report_to=[],
        label_names=["labels"],
    )
    if eval_ds is not None:
        if "evaluation_strategy" in sig.parameters:
            ta_kwargs["evaluation_strategy"] = "steps"
            if "eval_steps" in sig.parameters:
                ta_kwargs["eval_steps"] = args.eval_steps
        elif "do_eval" in sig.parameters:
            ta_kwargs["do_eval"] = True

    training_args = TrainingArguments(**ta_kwargs)

    # Prefer processing_class if available to silence deprecation
    tr_init_sig = inspect.signature(Trainer.__init__)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    if "processing_class" in tr_init_sig.parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor.tokenizer

    trainer = SpeechTrainer(**trainer_kwargs)

    if bad_files:
        print(f"Warning: skipped {len(bad_files)} unusable audio files. See bad_files.txt in {args.output_dir}")

    print("Starting training…")
    trainer.train()

    print("Saving adapter + processor…")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
