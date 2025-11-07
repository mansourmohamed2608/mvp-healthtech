"""
Phase 2: Continue Training with Shifaa
=======================================

This script continues training from Phase 1 (MMedC) and trains on Shifaa dataset.

Prerequisites:
1. ✅ Completed Phase 1 (MMedC training)
2. ✅ Uploaded MMedC LoRA to Kaggle as dataset
3. ✅ Have training_data_shifaa.json ready

Result: Model trained on MMedC + Shifaa
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
print("=" * 80)
print("PHASE 2: CONTINUE WITH SHIFAA")
print("=" * 80)
print()

!pip install -q transformers accelerate peft bitsandbytes datasets trl

print("✅ Dependencies installed!")
print()

# ============================================================================
# CELL 2: Configuration
# ============================================================================
import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

print("=" * 80)
print("PHASE 2: LOADING MMEDC LORA + TRAINING ON SHIFAA")
print("=" * 80)
print()

CONFIG = {
    # Base model
    "model_name": "Henrychur/MMed-Llama-3-8B",
    "local_model_paths": [
        "/kaggle/input/medllm/models--Henrychur--MMed-Llama-3-8B/snapshots",
        "/kaggle/input/medllm/models--Henrychur--MMed-Llama-3-8B",
    ],
    
    # PHASE 1 LoRA (MMedC)
    "previous_lora_path": [
        "/kaggle/input/mmedc-lora/final_model",
        "/kaggle/input/mmedc-lora-adapters/final_model",
        "/kaggle/working/mmedc_lora/final_model",
    ],
    
    # Training data: Shifaa
    "data_paths": [
        "/kaggle/working/training_data_shifaa.json",
        "/kaggle/input/shifaa-dataset/training_data_shifaa.json",
    ],
    
    # QLoRA Configuration
    "lora_r": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    
    # Training Hyperparameters
    "num_epochs": 1,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 2048,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 0.3,
    
    # QLoRA Optimizations
    "use_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "use_double_quant": True,
    "use_flash_attention": True,
    "use_gradient_checkpointing": True,
    
    # Advanced
    "lr_scheduler": "cosine",
    "optim": "paged_adamw_8bit",
    
    # Output
    "output_dir": "./mmedc_shifaa_lora",
    "save_steps": 500,
    "save_total_limit": 2,
    "logging_steps": 10,
}

print("🔄 PHASE 2 CONFIGURATION:")
print(f"   Base: {CONFIG['model_name']}")
print(f"   Resume from: Phase 1 (MMedC)")
print(f"   New data: Shifaa (84k examples)")
print(f"   Output: MMedC + Shifaa combined")
print()

# ============================================================================
# CELL 3: Load Training Data (Shifaa)
# ============================================================================
print("=" * 80)
print("LOADING SHIFAA DATA")
print("=" * 80)
print()

def load_training_data(data_paths):
    """Load Shifaa training data"""
    for path in data_paths:
        if os.path.exists(path) and os.path.isfile(path):
            print(f"📥 Loading: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            print(f"   ✅ Loaded {len(examples):,} Shifaa examples")
            return examples
    
    print("❌ Shifaa data not found!")
    print("Available files:")
    for parent in ["/kaggle/working", "/kaggle/input"]:
        if os.path.exists(parent):
            for item in os.listdir(parent):
                if item.endswith('.json'):
                    print(f"   - {os.path.join(parent, item)}")
    raise ValueError("No Shifaa training data found")

shifaa_examples = load_training_data(CONFIG["data_paths"])

# Format for Llama 3
def format_instruction(example):
    return {
        "text": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

أنت طبيب مساعد متخصص في الطب. أجب على الأسئلة الطبية بدقة ووضوح باللغة العربية.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
    }

dataset = Dataset.from_list(shifaa_examples)
dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

print()
print(f"✅ Shifaa dataset prepared: {len(dataset):,} examples")
print()

# ============================================================================
# CELL 4: Load Base Model + MMedC LoRA
# ============================================================================
print("=" * 80)
print("LOADING MODEL WITH MMEDC LORA")
print("=" * 80)
print()

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type=CONFIG["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=CONFIG["use_double_quant"],
)

# Find local model
model_path = CONFIG["model_name"]
for path in CONFIG["local_model_paths"]:
    if os.path.exists(path):
        if "snapshots" in path and os.path.isdir(path):
            snapshots = os.listdir(path)
            if snapshots:
                model_path = os.path.join(path, snapshots[0])
                print(f"✅ Found local model: {model_path}")
                break

# Load tokenizer
print(f"📥 Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
except:
    print("   ⚠️  Tokenizer template issue, using base Llama 3 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✅ Tokenizer loaded")

# Load base model
print(f"📥 Loading base model: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if CONFIG["use_flash_attention"] else "sdpa",
    use_cache=False,
    local_files_only=True,
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

print("✅ Base model loaded")
print()

# Load Phase 1 LoRA (MMedC)
print("🔄 Loading Phase 1 (MMedC) LoRA adapters...")
lora_loaded = False
for lora_path in CONFIG["previous_lora_path"]:
    if os.path.exists(lora_path):
        print(f"   Found: {lora_path}")
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            is_trainable=True,  # CRITICAL: Make trainable for continued training
        )
        print(f"✅ Loaded MMedC LoRA from: {lora_path}")
        print(f"   Model now has MMedC knowledge!")
        lora_loaded = True
        break

if not lora_loaded:
    print("⚠️  WARNING: Phase 1 (MMedC) LoRA not found!")
    print("   Available paths checked:")
    for path in CONFIG["previous_lora_path"]:
        print(f"      - {path}")
    print()
    print("   Training from scratch (not incremental)")
    print("   To fix: Upload your MMedC LoRA as a Kaggle dataset")
    print()
    
    # Apply fresh LoRA if previous not found
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=CONFIG["target_modules"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

print()
print("✅ Model ready for Phase 2 training!")
print()

# ============================================================================
# CELL 5: Setup Training
# ============================================================================
print("=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print()

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    lr_scheduler_type=CONFIG["lr_scheduler"],
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    max_grad_norm=CONFIG["max_grad_norm"],
    gradient_checkpointing=CONFIG["use_gradient_checkpointing"],
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim=CONFIG["optim"],
    bf16=True,
    fp16=False,
    logging_steps=CONFIG["logging_steps"],
    save_steps=CONFIG["save_steps"],
    save_total_limit=CONFIG["save_total_limit"],
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    group_by_length=True,
    eval_strategy="no",
    report_to="none",
    seed=42,
)

total_steps = len(dataset) // (CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"])
estimated_hours = total_steps * 1.8 / 3600

print(f"📊 Phase 2 Training Estimates:")
print(f"   Shifaa examples: {len(dataset):,}")
print(f"   Steps: {total_steps:,}")
print(f"   Time: {estimated_hours:.1f} hours")
print()

# ============================================================================
# CELL 6: Train!
# ============================================================================
print("=" * 80)
print("STARTING PHASE 2 TRAINING")
print("=" * 80)
print()

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=CONFIG["max_seq_length"],
    dataset_text_field="text",
    packing=False,
)

print("🚀 Training on Shifaa data (continuing from MMedC)...")
print(f"   Model already knows: MMedC (~167 examples)")
print(f"   Learning now: Shifaa ({len(dataset):,} examples)")
print(f"   Result: Combined knowledge!")
print()

trainer.train()

print()
print("✅ PHASE 2 TRAINING COMPLETE!")
print()

# ============================================================================
# CELL 7: Save Combined Model
# ============================================================================
print("💾 Saving combined model (MMedC + Shifaa)...")

trainer.model.save_pretrained(CONFIG["output_dir"] + "/final_model")
tokenizer.save_pretrained(CONFIG["output_dir"] + "/final_model")

print(f"✅ Saved to: {CONFIG['output_dir']}/final_model")
print()

# Create zip for download
import shutil
shutil.make_archive(
    "/kaggle/working/mmedc_shifaa_lora",
    'zip',
    CONFIG["output_dir"] + "/final_model"
)

print("=" * 80)
print("PHASE 2 COMPLETE!")
print("=" * 80)
print()
print(f"✅ Your model now knows:")
print(f"   - MMedC: ~167 examples (Phase 1)")
print(f"   - Shifaa: {len(dataset):,} examples (Phase 2)")
print(f"   - Total: ~{167 + len(dataset):,} examples!")
print()
print("📥 Download: /kaggle/working/mmedc_shifaa_lora.zip")
print()
print("🎯 Next: Upload this as Kaggle dataset, then run Phase 3 (AHD)")
print()
