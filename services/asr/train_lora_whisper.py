# services/asr/train_lora_whisper.py
# Fine-tune Whisper with LoRA on Arabic using a CSV manifest (audio_filepath, text)
# pip install -U torch transformers datasets peft soundfile

import os
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from datasets import load_dataset, Audio
from peft import LoraConfig, get_peft_model
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
)

# ---------------- Config ----------------
CSV_PATH   = "data/mixed_train.csv"     # your mixed (medical-heavy) CSV
BASE_MODEL = "openai/whisper-large-v2"
LANGUAGE   = "arabic"
TASK       = "transcribe"

# Use a medical hint as context (ON/OFF)
ADD_HINT_PREFIX = True
HINT = "ملاحظة طبية:"

# -------------- Model & processor --------------
processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

# Ensure Arabic transcription mode
model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=LANGUAGE, task=TASK
)

# LoRA
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

# -------------- Dataset --------------
assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"
ds = load_dataset("csv", data_files={"train": CSV_PATH})["train"]

# rename -> standard names, then decode audio
if "audio_filepath" in ds.column_names:
    ds = ds.rename_column("audio_filepath", "audio")
if "text" in ds.column_names:
    ds = ds.rename_column("text", "sentence")

ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# Pre-tokenize the hint once (used for loss masking)
with processor.as_target_processor():
    HINT_IDS = processor(HINT).input_ids if ADD_HINT_PREFIX else []

def preprocess(batch):
    # audio -> log-mel features
    audio = batch["audio"]["array"]
    inputs = processor(audio=audio, sampling_rate=16000)

    # Build target text with/without hint
    target_text = batch["sentence"]
    if ADD_HINT_PREFIX:
        target_text = f"{HINT} {target_text}"

    # text -> token IDs
    with processor.as_target_processor():
        labels = processor(target_text).input_ids

    # Track how many tokens come from the hint so we can mask them in loss
    prefix_len = len(HINT_IDS) if ADD_HINT_PREFIX else 0

    return {
        "input_features": inputs["input_features"][0],
        "labels": labels,
        "prefix_len": prefix_len,
    }

ds = ds.map(preprocess, remove_columns=ds.column_names)

# -------------- Collator (pads & masks) --------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # stack inputs
        input_features = torch.tensor(
            [f["input_features"] for f in features], dtype=torch.float32
        )
        # pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"]
        attn  = labels_batch["attention_mask"]

        # mask padding
        labels = labels.masked_fill(attn.ne(1), -100)

        # additionally mask the hint prefix tokens so model isn't trained to emit them
        for i, f in enumerate(features):
            n = int(f.get("prefix_len", 0))
            if n > 0:
                labels[i, :n] = -100

        return {"input_features": input_features, "labels": labels}

collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

# -------------- Training --------------
training_args = TrainingArguments(
    output_dir="lora_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=200,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,  # keep speech fields
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=processor.tokenizer,
    data_collator=collator,
)

trainer.train()
model.save_pretrained("lora_ckpt")
processor.save_pretrained("lora_ckpt")
print("LoRA adapter and processor saved to lora_ckpt")
