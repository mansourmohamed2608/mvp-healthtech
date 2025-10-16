# === services/asr/train_lora_whisper_full.py ===
import argparse, os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np, torch
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model
from jiwer import wer

def parse_args():
    p = argparse.ArgumentParser("LoRA fine-tune Whisper on Arabic medical data (full dataset, robust)")
    p.add_argument("--csv_path", type=str, required=True,
                   help="Path to manifest.csv (columns: audio_filepath,text)")
    p.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/lora_ckpt_med_full")
    p.add_argument("--base_model", type=str, default="openai/whisper-large-v3")
    p.add_argument("--language", type=str, default="arabic")
    p.add_argument("--task", type=str, default="transcribe", choices=["transcribe","translate"])
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--eval_fraction", type=float, default=0.01,
                   help="small held-out split for WER")    
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--dataloader_workers", type=int, default=2)
    p.add_argument("--keep_in_memory", action="store_true",
                   help="Keep datasets in RAM to avoid disk I/O (good on Colab)")
    p.add_argument("--use_hint", action="store_true")
    p.add_argument("--hint_prefix", type=str, default="ملاحظة طبية:")
    p.add_argument("--resume_from", type=str, default=None)
    return p.parse_args()

@dataclass
class Collator:
    processor: WhisperProcessor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = torch.tensor([f["input_features"] for f in features], dtype=torch.float32)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"]
        attn = labels_batch["attention_mask"]
        labels = labels.masked_fill(attn.ne(1), -100)
        return {"input_features": input_features, "labels": labels}

def build_path(csv_path: str, rel: str) -> str:
    if os.path.isabs(rel): return rel
    mdir = os.path.dirname(os.path.abspath(csv_path))
    c1 = os.path.join(mdir, rel)
    if os.path.exists(c1): return c1
    root = os.path.dirname(mdir)
    return os.path.join(root, rel)

def load_wave(path: str) -> Tuple[np.ndarray, int]:
    # try torchaudio, fallback to soundfile; resample to 16k
    arr, sr = None, 16000
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        if wav.ndim > 1: wav = wav.mean(dim=0)
        arr = wav.numpy()
    except Exception:
        import soundfile as sf
        arr, sr = sf.read(path)
        if arr.ndim > 1: import numpy as _np; arr = _np.mean(arr, axis=1)
    if sr != 16000:
        try:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        except Exception:
            step = max(1, int(sr // 16000)); arr = arr[::step]
    return arr, 16000

def main():
    args = parse_args()
    assert os.path.exists(args.csv_path), f"Manifest not found: {args.csv_path}"

    print("Loading processor and model…")
    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, load_in_8bit=True, device_map="auto"
    )
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )
    lora = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","out_proj"],
        lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print("Loading dataset…")
    ds = load_dataset(
        "csv", data_files={"train": args.csv_path},
        keep_in_memory=args.keep_in_memory
    )["train"]

    # normalize columns
    cols = {c.lower(): c for c in ds.column_names}
    audio_col = cols.get("audio_filepath", cols.get("audio"))
    text_col  = cols.get("text", cols.get("sentence"))
    assert audio_col and text_col, f"Need audio_filepath,text (got: {ds.column_names})"
    if audio_col != "audio": ds = ds.rename_column(audio_col, "audio")
    if text_col  != "sentence": ds = ds.rename_column(text_col, "sentence")

    missing = []
    def _exists(row):
        p = build_path(args.csv_path, row["audio"])
        try:
            if os.path.exists(p) and os.path.isfile(p) and os.path.getsize(p) > 0:
                try:
                    import soundfile as sf
                    with sf.SoundFile(p): pass
                except Exception:
                    missing.append(row["audio"]); return False
                return True
        except Exception: pass
        missing.append(row["audio"]); return False

    print("Filtering missing/zero/corrupt audio …")
    ds = ds.filter(_exists)
    if len(ds) == 0: raise ValueError("All rows filtered out. Check your paths/manifest.")
    if missing: print(f"Skipped {len(missing)} bad files. First 10: {missing[:10]}")

    # small eval split for WER
    eval_size = max(1, int(len(ds) * args.eval_fraction))
    ds = ds.shuffle(seed=1337)
    split = ds.train_test_split(test_size=eval_size, seed=1337)
    train_ds, eval_ds = split["train"], split["test"]

    def preprocess(batch):
        ap = build_path(args.csv_path, batch["audio"])
        wav, sr = load_wave(ap)
        ins = processor(audio=wav, sampling_rate=sr)
        target = batch["sentence"]
        if args.use_hint and args.hint_prefix:
            target = f"{args.hint_prefix} {target}"
        lab = processor.tokenizer(target).input_ids
        return {"input_features": ins["input_features"][0], "labels": lab}

    # avoid arrow caches → less disk I/O / fewer crashes
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names,
                            load_from_cache_file=False, cache_file_name=None,
                            desc="Preprocess train")
    eval_ds  = eval_ds.map(preprocess, remove_columns=eval_ds.column_names,
                           load_from_cache_file=False, cache_file_name=None,
                           desc="Preprocess eval")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple): preds = preds[0]
        pred_ids = np.argmax(preds, axis=-1) if preds.ndim == 3 else preds
        labels_clean = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        pred_txt = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_txt  = processor.tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
        pred_txt = [t.strip() for t in pred_txt]; ref_txt = [t.strip() for t in ref_txt]
        return {"wer": wer(ref_txt, pred_txt)}

    args_training = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=500, save_total_limit=1,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        group_by_length=False,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        report_to=[],
        predict_with_generate=True,
        generation_max_length=128,
    )

    trainer = Trainer(
        model=model, args=args_training,
        train_dataset=train_ds, eval_dataset=eval_ds,
        tokenizer=processor.tokenizer,
        data_collator=Collator(processor),
        compute_metrics=compute_metrics,
    )

    print("Starting training…")
    trainer.train(resume_from_checkpoint=args.resume_from if args.resume_from else None)

    print("Saving adapter + processor…")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Done. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
# === end file ===
