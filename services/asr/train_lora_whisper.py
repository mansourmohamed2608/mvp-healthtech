import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine‑tune Whisper on Arabic medical data with LoRA")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to the CSV manifest containing columns 'audio_filepath' and 'text'.")
    parser.add_argument("--output_dir", type=str, default="lora_ckpt",
                        help="Directory where the LoRA adapter and processor will be saved.")
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3",
                        help="Base Whisper model identifier from Hugging Face Hub.")
    parser.add_argument("--language", type=str, default="arabic",
                        help="Language for the processor.")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="Task for Whisper. Use 'transcribe' for Arabic→Arabic transcription.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per‑device train batch size.")
    parser.add_argument("--gradient_accumulation", type=int, default=16,
                        help="Number of steps to accumulate gradients.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for AdamW.")
    parser.add_argument("--hint_prefix", type=str, default="ملاحظة طبية:",
                        help="Optional hint prefix inserted before each transcription target.")
    parser.add_argument("--use_hint", action="store_true",
                        help="If set, prepend the hint prefix to targets and mask its loss.")
    return parser.parse_args()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    hint_ids: List[int]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack input features (log‑mels) into a tensor
        input_features = torch.tensor([f["input_features"] for f in features], dtype=torch.float32)
        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"]
        attn = labels_batch["attention_mask"]
        labels = labels.masked_fill(attn.ne(1), -100)
        # Mask hint tokens
        if self.hint_ids:
            for i, f in enumerate(features):
                n = f.get("prefix_len", 0)
                if n > 0:
                    labels[i, :n] = -100
        return {"input_features": input_features, "labels": labels}


def main() -> None:
    args = parse_args()
    assert os.path.exists(args.csv_path), f"Manifest CSV not found: {args.csv_path}"

    print("Loading processor and model…")
    processor = WhisperProcessor.from_pretrained(args.base_model,
                                                 language=args.language,
                                                 task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        device_map="auto",
    )
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task)

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
    model.gradient_checkpointing_enable()

    # Load dataset
    print("Loading dataset…")
    ds = load_dataset("csv", data_files={"train": args.csv_path})["train"]
    if "audio_filepath" in ds.column_names:
        ds = ds.rename_column("audio_filepath", "audio")
    if "text" in ds.column_names:
        ds = ds.rename_column("text", "sentence")

    manifest_dir = os.path.dirname(os.path.abspath(args.csv_path))
    dataset_root = os.path.dirname(manifest_dir)
    hint_ids: List[int] = []
    if args.use_hint and args.hint_prefix:
        hint_ids = processor.tokenizer(args.hint_prefix).input_ids
    missing_files: List[str] = []

    def file_exists(batch: Dict[str, Any]) -> bool:
        audio_rel = batch["audio"]
        if os.path.isabs(audio_rel):
            candidate = audio_rel
        else:
            cand1 = os.path.join(manifest_dir, audio_rel)
            candidate = cand1 if os.path.exists(cand1) else os.path.join(dataset_root, audio_rel)
        if os.path.exists(candidate) and os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            try:
                import soundfile as sf
                with sf.SoundFile(candidate):
                    pass
                return True
            except Exception:
                missing_files.append(audio_rel)
                return False
        missing_files.append(audio_rel)
        return False

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio_rel = batch["audio"]
        if os.path.isabs(audio_rel):
            audio_path = audio_rel
        else:
            cand1 = os.path.join(manifest_dir, audio_rel)
            audio_path = cand1 if os.path.exists(cand1) else os.path.join(dataset_root, audio_rel)
        # load audio
        waveform = None
        orig_sr = 16000
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)
            waveform = waveform.numpy()
            orig_sr = sr
        except Exception:
            import soundfile as sf
            waveform, sr = sf.read(audio_path)
            orig_sr = sr
        if orig_sr != 16000:
            try:
                import librosa
                waveform = librosa.resample(waveform, orig_sr, 16000)
            except Exception:
                step = max(1, int(orig_sr // 16000))
                waveform = waveform[::step]
        inputs = processor(audio=waveform, sampling_rate=16000)
        target_text = batch["sentence"]
        prefix_len = 0
        if args.use_hint:
            target_text = f"{args.hint_prefix} {target_text}"
            prefix_len = len(hint_ids)
        labels = processor.tokenizer(target_text).input_ids
        return {
            "input_features": inputs["input_features"][0],
            "labels": labels,
            "prefix_len": prefix_len,
        }

    # Remove missing or corrupt files
    ds = ds.filter(file_exists)
    # Map with caching disabled and a single process
    ds = ds.map(preprocess,
                remove_columns=ds.column_names,
                load_from_cache_file=False,
                cache_file_name=None,
                num_proc=1)

    if missing_files:
        print(f"Warning: {len(missing_files)} audio files were missing or unreadable and have been skipped.\n"
              f"First 10 missing files: {missing_files[:10]}")

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, hint_ids=hint_ids)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=processor.tokenizer,
        data_collator=collator,
    )
    print("Starting training…")
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
