import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine‑tune Whisper on Arabic medical data with LoRA")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV manifest containing columns 'audio_filepath' and 'text'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora_ckpt",
        help="Directory where the LoRA adapter and processor will be saved.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="openai/whisper-large-v3",
        help="Base Whisper model identifier from Hugging Face Hub (e.g. openai/whisper-large-v3).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="arabic",
        help="Language for the processor (e.g. 'arabic' or 'ar').",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task for Whisper. Use 'transcribe' for Arabic→Arabic transcription.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per‑device train batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=16,
        help="Number of steps to accumulate gradients. Effective batch size = batch_size×gradient_accumulation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the AdamW optimizer.",
    )
    parser.add_argument(
        "--hint_prefix",
        type=str,
        default="ملاحظة طبية:",
        help="Optional hint prefix inserted before each transcription target.",
    )
    parser.add_argument(
        "--use_hint",
        action="store_true",
        help="If set, prepend the hint prefix to targets and mask its loss.",
    )
    return parser.parse_args()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Custom collator that pads labels and masks hint tokens."""

    processor: WhisperProcessor
    hint_ids: List[int]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack input features (log‐mels) into a single tensor
        input_features = torch.tensor(
            [f["input_features"] for f in features], dtype=torch.float32
        )
        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"]
        attn = labels_batch["attention_mask"]
        # Mask padding tokens
        labels = labels.masked_fill(attn.ne(1), -100)
        # Additionally mask hint tokens at the beginning
        if self.hint_ids:
            for i, f in enumerate(features):
                # number of tokens from hint
                n = f.get("prefix_len", 0)
                if n > 0:
                    labels[i, :n] = -100
        return {"input_features": input_features, "labels": labels}


def main() -> None:
    args = parse_args()
    assert os.path.exists(args.csv_path), f"Manifest CSV not found: {args.csv_path}"

    # Load processor and model in 8‑bit quantized form
    print("Loading processor and model…")
    processor = WhisperProcessor.from_pretrained(
        args.base_model, language=args.language, task=args.task
    )
    # Load model in 8‑bit with better memory footprint
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        device_map="auto",
    )
    # Force the model to always transcribe in Arabic
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )

    # Apply LoRA to key Whisper layers
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

    # Load dataset from CSV
    print("Loading dataset…")
    ds = load_dataset("csv", data_files={"train": args.csv_path})["train"]
    # Rename columns and cast audio
    if "audio_filepath" in ds.column_names:
        ds = ds.rename_column("audio_filepath", "audio")
    if "text" in ds.column_names:
        ds = ds.rename_column("text", "sentence")
    # Do **not** cast the audio column using datasets.Audio.  The default
    # Audio feature tries to decode audio via TorchCodec/FFmpeg which
    # often fails in restricted environments (e.g. Kaggle) due to
    # missing shared libraries.  Instead we keep the path string and
    # manually load the waveform in preprocess.  If you wish to use
    # datasets.Audio in a different environment, uncomment the line below.
    # ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Determine directory of CSV.  Audio paths in the manifest may be
    # relative; we join them with this base directory to form absolute
    # filepaths.  Without this, torchaudio/soundfile will look in the
    # current working directory and fail to find the files.
    manifest_dir = os.path.dirname(os.path.abspath(args.csv_path))
    # Compute dataset root (the parent of the manifest directory).  Audio
    # filepaths in the CSV may reference paths relative to this root.
    dataset_root = os.path.dirname(manifest_dir)

    # Pre‑tokenize hint once for loss masking.  WhisperProcessor versions
    # after transformers 4.38 no longer implement `as_target_processor()`;
    # instead we use the tokenizer directly to obtain IDs.
    hint_ids: List[int] = []
    if args.use_hint and args.hint_prefix:
        hint_ids = processor.tokenizer(args.hint_prefix).input_ids

    # Keep track of any audio files that cannot be found.  We will
    # filter them out of the training set to avoid crashes during
    # preprocessing.
    missing_files: List[str] = []

    def file_exists(batch: Dict[str, Any]) -> bool:
        """
        Check whether the audio file referenced in the manifest exists
        on disk.  If it is missing, add it to ``missing_files`` and
        return False to drop the example.
        """
        audio_rel = batch["audio"]
        if os.path.isabs(audio_rel):
            candidate = audio_rel
        else:
            # Try manifest_dir first, then dataset_root
            cand1 = os.path.join(manifest_dir, audio_rel)
            if os.path.exists(cand1):
                candidate = cand1
            else:
                candidate = os.path.join(dataset_root, audio_rel)
        # Ensure file exists, is a regular file, and has non‑zero size
        if os.path.exists(candidate) and os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            # Optionally attempt to open the file to catch corrupt or pipe files
            try:
                import soundfile as sf  # type: ignore
                with sf.SoundFile(candidate) as _:
                    pass
                return True
            except Exception:
                # Could not open file; treat as missing/corrupt
                missing_files.append(audio_rel)
                return False
        # Either missing, not a file, or zero‑length
        missing_files.append(audio_rel)
        return False

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        # Convert audio to log‐mel features
        # Load waveform manually to avoid TorchCodec dependencies
        # Build absolute audio path.  If the entry is already absolute,
        # use it directly.  Otherwise, join it with ``dataset_root`` (one
        # directory above the manifest).  This handles cases where the
        # manifest is at ``/path/to/manifest.csv`` and the audio column
        # contains ``subdir/file.mp3`` or ``tts_ar_med/file.mp3``.
        audio_rel = batch["audio"]
        if os.path.isabs(audio_rel):
            # Already absolute
            audio_path = audio_rel
        else:
            # Try resolving relative to the manifest directory.  If the
            # file is not found there, fallback to the parent directory
            # (dataset_root).  This handles both cases where the manifest
            # is in the dataset root and where it is in a subdirectory.
            candidate1 = os.path.join(manifest_dir, audio_rel)
            if os.path.exists(candidate1):
                audio_path = candidate1
            else:
                audio_path = os.path.join(dataset_root, audio_rel)
        waveform = None
        orig_sr = 16000
        try:
            import torchaudio  # type: ignore
            waveform, sr = torchaudio.load(audio_path)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)
            waveform = waveform.numpy()
            orig_sr = sr
        except Exception:
            try:
                import soundfile as sf  # type: ignore
                waveform, sr = sf.read(audio_path)
                orig_sr = sr
            except Exception as e:
                raise RuntimeError(f"Failed to load audio file {audio_path}: {e}")
        # Resample to 16 kHz if necessary
        if orig_sr != 16000:
            try:
                import librosa  # type: ignore
                waveform = librosa.resample(waveform, orig_sr, 16000)
            except Exception:
                import numpy as np  # type: ignore
                step = max(1, int(orig_sr // 16000))
                waveform = waveform[::step]
        inputs = processor(audio=waveform, sampling_rate=16000)
        # Target text with optional hint
        target_text = batch["sentence"]
        prefix_len = 0
        if args.use_hint:
            target_text = f"{args.hint_prefix} {target_text}"
            prefix_len = len(hint_ids)
        # Tokenize target text using the underlying tokenizer.  Do not
        # rely on `as_target_processor()` because it may not be present.
        labels = processor.tokenizer(target_text).input_ids
        return {
            "input_features": inputs["input_features"][0],
            "labels": labels,
            "prefix_len": prefix_len,
        }

    # Filter out rows whose audio files do not exist.  This prevents
    # crashes when loading missing files during preprocessing.
    ds = ds.filter(file_exists)
    # Map and remove unused columns.  Disable caching to avoid disk I/O
    # errors on Kaggle when writing temporary Arrow files.  Setting
    # ``load_from_cache_file=False`` and ``cache_file_name=None`` forces
    # datasets to process in memory.
    ds = ds.map(
        preprocess,
        remove_columns=ds.column_names,
        load_from_cache_file=False,
        cache_file_name=None,
    )

    # Setup data collator
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, hint_ids=hint_ids
    )

    # Define training arguments
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
    # Train
    if missing_files:
        print(f"Warning: {len(missing_files)} audio files were missing and have been skipped.\n"
              f"First 10 missing files: {missing_files[:10]}")
    print("Starting training…")
    trainer.train()
    # Save LoRA adapter and processor
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
