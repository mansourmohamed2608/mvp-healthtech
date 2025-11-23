# Fine‑tune Whisper Large‑V3 with LoRA on Kaggle/local machine
# This version solves three issues:
# 1. Uses a custom WhisperTuner to remove unsupported 'input_ids' from the forward call【518651146407626†L355-L368】.
# 2. Calls enable_input_require_grads() to ensure inputs have grad_fn to avoid RuntimeError during backward【861473363536959†L106-L115】.
# 3. Disables wandb to prevent API key prompts【454707287818299†L15-L33】.

# ------------------------ 1) Install dependencies ------------------------
# !pip install -q --upgrade \
#   "transformers" \
#   "peft" \
#   "accelerate" \
#   "datasets<2.22,>=2.20" \
#   "evaluate<0.5,>=0.4.2" \
#   "bitsandbytes<0.46,>=0.45.3" \
#   "librosa==0.10.2.post1" \
#   "jiwer==4.0.0" \
#   "soundfile"

# ------------------------ 2) Environment configuration ------------------------
import os
os.environ["WANDB_DISABLED"] = "true"  # disable wandb【454707287818299†L15-L20】
os.environ["WANDB_SILENT"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # quieter logs

import gc
import shutil
from pathlib import Path

import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from datasets import load_dataset, disable_caching
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, PeftType

# Avoid writing large Arrow caches to disk
disable_caching()

# Device and disk info
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
work = Path("/kaggle/working")
work.mkdir(exist_ok=True, parents=True)
print("Disk(working):", {k: round(v / 1e9, 2) for k, v in zip(['total','used','free'], shutil.disk_usage(work))})

# ------------------------ 3) User configuration ------------------------
CONFIG = {
    "csv_path": "/kaggle/input/medical-voice-dataset/tts_ar_med/manifest.csv",
    "dataset_root_fallback": "/kaggle/input/medical-voice-dataset",
    "output_dir": "/kaggle/working/lora_ckpt_med",
    "base_model": "openai/whisper-large-v3",
    "language": "arabic",
    "task": "transcribe",
    "num_epochs": 1,
    "batch_size": 1,
    "grad_accum": 16,
    "lr": 1e-4,
    "train_max_rows": 12000,
    "use_hint": True,
    "hint_prefix": "ملاحظة طبية:",
    "save_steps": 400,
    "logging_steps": 25,
}
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
bad_list_path = Path(CONFIG["output_dir"]) / "bad_files.txt"

# ------------------------ 4) Custom WhisperTuner ------------------------
# Subclass PeftModel to attach LoRA and drop unsupported input_ids【518651146407626†L355-L368】.
class WhisperTuner(PeftModel):
    def __init__(self, model: torch.nn.Module, peft_config: LoraConfig, adapter_name: str = "default") -> None:
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
        self,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids
            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

    def generate(self, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            if not peft_config.is_prompt_learning:
                with self._enable_peft_forward_hooks(**kwargs):
                    kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                    outputs = self.base_model.generate(**kwargs)
        finally:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
        return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: torch.Tensor = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        if (
            model_kwargs.get("past_key_values") is None
            and peft_config.peft_type == PeftType.PREFIX_TUNING
        ):
            batch_size = model_kwargs["decoder_input_ids"].shape[0]
            past_key_values = self.get_prompt(batch_size)
            model_kwargs["past_key_values"] = past_key_values
        return model_kwargs

# ------------------------ 5) Load processor and model ------------------------
print("Loading processor/model…")
processor = WhisperProcessor.from_pretrained(
    CONFIG["base_model"], language=CONFIG["language"], task=CONFIG["task"]
)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = WhisperForConditionalGeneration.from_pretrained(
    CONFIG["base_model"], quantization_config=bnb_cfg, device_map="auto",
)
# Force Arabic transcription
model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=CONFIG["language"], task=CONFIG["task"]
)

# LoRA config and apply WhisperTuner
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
)
model = WhisperTuner(model, lora_cfg)
# Enable gradient on inputs to avoid RuntimeError during backward【861473363536959†L106-L115】
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

model.config.use_cache = False
model.gradient_checkpointing_enable()
torch.backends.cuda.matmul.allow_tf32 = True

# ------------------------ 6) Load and filter CSV ------------------------
print("Loading CSV…")
csv_path = Path(CONFIG["csv_path"]).resolve()
assert csv_path.exists(), f"Manifest not found: {csv_path}"
manifest_dir = csv_path.parent
dataset_root_fallback = Path(CONFIG["dataset_root_fallback"]).resolve()

ds = load_dataset("csv", data_files={"train": str(csv_path)})["train"]
if "audio_filepath" in ds.column_names:
    ds = ds.rename_column("audio_filepath", "audio")
if "text" in ds.column_names:
    ds = ds.rename_column("text", "sentence")
assert "audio" in ds.column_names and "sentence" in ds.column_names

def resolve_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    cand = manifest_dir / p
    if cand.exists():
        return cand
    return dataset_root_fallback / p

bad_files = []

def is_ok(batch):
    p = resolve_path(batch["audio"])
    try:
        if not (p.exists() and p.is_file() and p.stat().st_size > 0):
            bad_files.append(batch["audio"])
            return False
        with sf.SoundFile(str(p)) as _:
            pass
        return True
    except Exception:
        bad_files.append(batch["audio"])
        return False

print("Filtering missing/zero/corrupt audio…")
ds = ds.filter(is_ok)
if bad_files:
    with open(bad_list_path, "w", encoding="utf-8") as f:
        for x in bad_files:
            f.write(str(x) + "\n")
    print(f"Skipped {len(bad_files)} bad files. First 10: {bad_files[:10]}")
else:
    print("No bad files detected.")
print(f"Kept {len(ds)} rows after filtering.")

if CONFIG["train_max_rows"] is not None:
    ds = ds.select(range(min(CONFIG["train_max_rows"], len(ds))))
    print(f"Training on {len(ds)} examples (subset).")
else:
    print(f"Training on full {len(ds)} examples.")

# ------------------------ 7) Preprocess ------------------------
SAMPLE_RATE = 16000

def preprocess(batch):
    path = resolve_path(batch["audio"])
    try:
        w_t, sr = torchaudio.load(str(path))
        w = w_t.mean(0).numpy() if w_t.ndim > 1 else w_t.numpy()[0]
    except Exception:
        w, sr = sf.read(str(path))
        w = np.mean(w, axis=1) if w.ndim > 1 else w
    if sr != SAMPLE_RATE:
        w = librosa.resample(w, orig_sr=sr, target_sr=SAMPLE_RATE)
    inputs = processor(audio=w, sampling_rate=SAMPLE_RATE)
    text = batch["sentence"]
    prefix_len = 0
    if CONFIG["use_hint"] and CONFIG["hint_prefix"]:
        text = f"{CONFIG['hint_prefix']} {text}"
        prefix_len = len(processor.tokenizer(CONFIG["hint_prefix"]).input_ids)
    labels = processor.tokenizer(text).input_ids
    return {"input_features": inputs["input_features"][0], "labels": labels, "prefix_len": prefix_len}

print("Map: preprocessing…")
ds = ds.map(
    preprocess,
    remove_columns=ds.column_names,
    load_from_cache_file=False,
    cache_file_name=None,
)
print("Preprocess done.")

# ------------------------ 8) Data collator ------------------------
class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        xs = [f["input_features"] for f in features]
        input_features = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32)
        label_features = [{"input_ids": f["labels"]} for f in features]
        lb = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels, attn = lb["input_ids"], lb["attention_mask"]
        labels = labels.masked_fill(attn.ne(1), -100)
        for i, f in enumerate(features):
            n = int(f.get("prefix_len", 0))
            if n > 0:
                labels[i, :n] = -100
        return {"input_features": input_features, "labels": labels}

collator = Collator(processor)

# ------------------------ 9) Trainer ------------------------
class WhisperTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(input_features=inputs["input_features"], labels=inputs.get("labels"))
        loss = out.loss
        return (loss, out) if return_outputs else loss

args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["grad_accum"],
    learning_rate=CONFIG["lr"],
    num_train_epochs=CONFIG["num_epochs"],
    logging_steps=CONFIG["logging_steps"],
    save_steps=CONFIG["save_steps"],
    save_total_limit=1,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = WhisperTrainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collator,
    tokenizer=processor.tokenizer,
)

try:
    from transformers.integrations import WandbCallback
    trainer.remove_callback(WandbCallback)
except Exception:
    pass

if bad_files:
    print(f"Warning: {len(bad_files)} unusable files skipped. See {bad_list_path}")

print("Starting training…")
trainer.train()

print("Saving adapter + processor…")
model.save_pretrained(CONFIG["output_dir"])
processor.save_pretrained(CONFIG["output_dir"])
print(f"Done. Artifacts at: {CONFIG['output_dir']}")
