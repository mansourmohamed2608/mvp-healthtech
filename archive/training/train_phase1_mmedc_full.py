"""
Phase 1: Train on MMedC FULL (All 70k+ files extracted)
========================================================

This trains on ALL MMedC data (~100k+ examples after extraction)
Estimated time: 14-18 hours (depends on actual example count)

Prerequisites:
1. Run extract_ALL_mmedc.py first to get training_data_mmedc_FULL.json
2. Upload training_data_mmedc_FULL.json to Kaggle
3. Have MMed-Llama-3-8B model in Kaggle input

Result: mmedc_lora.zip - Phase 1 LoRA adapters
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
print("=" * 80)
print("PHASE 1: MMEDC FULL TRAINING")
print("=" * 80)
print()

!pip install -q transformers accelerate peft bitsandbytes datasets trl

print("✅ Dependencies installed!")
print()

# ============================================================================
# CELL 2: Configuration for MMedC Full Training
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("=" * 80)
print("CONFIGURATION: MMEDC FULL")
print("=" * 80)
print()

CONFIG = {
    # Base Model
    "model_name": "Henrychur/MMed-Llama-3-8B",
    "local_model_paths": [
        "/kaggle/input/medllm/models--Henrychur--MMed-Llama-3-8B/snapshots",
        "/kaggle/input/medllm/models--Henrychur--MMed-Llama-3-8B",
    ],
    
    # Training Data - MMedC FULL only
    "data_paths": [
        "/kaggle/working/training_data_mmedc_FULL.json",
        "/kaggle/input/mmedc-full/training_data_mmedc_FULL.json",
        "/kaggle/input/mmedc-dataset/training_data_mmedc_FULL.json",
    ],
    
    # QLoRA Configuration (Optimized for long training)
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
    "gradient_accumulation_steps": 4,  # Effective batch = 16
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
    
    # Optimizer
    "lr_scheduler": "cosine",
    "optim": "paged_adamw_8bit",
    
    # Output
    "output_dir": "./mmedc_lora",
    "save_steps": 1000,  # Save every 1000 steps
    "save_total_limit": 3,  # Keep 3 checkpoints
    "logging_steps": 50,
}

print("📊 PHASE 1: MMEDC FULL")
print(f"   Dataset: ALL 70k+ MMedC files extracted")
print(f"   Expected: ~100,000+ examples")
print(f"   Time estimate: 14-18 hours")
print()

# ============================================================================
# CELL 3: Load Training Data
# ============================================================================
print("=" * 80)
print("LOADING MMEDC FULL DATA")
print("=" * 80)
print()

def load_training_data(data_paths):
    """Load MMedC FULL training data"""
    for path in data_paths:
        if os.path.exists(path) and os.path.isfile(path):
            print(f"📥 Loading: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"   ✅ Loaded {len(examples):,} examples")
            print(f"   📦 File size: {file_size_mb:.1f} MB")
            return examples
    
    print("❌ MMedC FULL data not found!")
    print()
    print("Expected locations:")
    for path in data_paths:
        print(f"   - {path}")
    print()
    print("Did you run extract_ALL_mmedc.py first?")
    print("This extracts ALL 70k+ files into training_data_mmedc_FULL.json")
    print()
    raise ValueError("No MMedC FULL training data found")

mmedc_examples = load_training_data(CONFIG["data_paths"])

# Format for Llama 3
def format_instruction(example):
    """Format example for Llama 3 chat template"""
    return {
        "text": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

أنت طبيب مساعد متخصص في الطب. أجب على الأسئلة الطبية بدقة ووضوح باللغة العربية.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
    }

dataset = Dataset.from_list(mmedc_examples)
dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

print()
print(f"✅ Dataset prepared: {len(dataset):,} examples")
print()

# Estimate training time
steps = len(dataset) // (CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"])
estimated_hours = steps * 1.8 / 3600  # 1.8 sec per step

print(f"📊 TRAINING ESTIMATES:")
print(f"   Total steps: {steps:,}")
print(f"   Steps per epoch: {steps:,}")
print(f"   Estimated time: {estimated_hours:.1f} hours")
print()

if estimated_hours > 20:
    print("⚠️  WARNING: Training will take over 20 hours!")
    print("   Consider:")
    print("   - Reducing examples (sample dataset)")
    print("   - Splitting into multiple sessions")
    print()

# ============================================================================
# CELL 4: Load Model
# ============================================================================
print("=" * 80)
print("LOADING MODEL")
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
    print("✅ Tokenizer loaded")
except Exception as e:
    print(f"   ⚠️  Using fallback tokenizer: {e}")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
    print("✅ Fallback tokenizer loaded")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load base model
print(f"📥 Loading base model...")
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
print("✅ Base model loaded and quantized")

# Apply LoRA
lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=CONFIG["target_modules"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print("✅ LoRA adapters applied")
print()

# Show trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Model Parameters:")
print(f"   Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print(f"   Total: {total_params:,}")
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

print(f"📊 Final Training Setup:")
print(f"   Examples: {len(dataset):,}")
print(f"   Batch size: {CONFIG['batch_size']}")
print(f"   Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
print(f"   Effective batch: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
print(f"   Total steps: {steps:,}")
print(f"   Estimated time: {estimated_hours:.1f} hours")
print()

# ============================================================================
# CELL 6: Train!
# ============================================================================
print("=" * 80)
print("STARTING PHASE 1 TRAINING")
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

print("🚀 Training MMedC FULL dataset...")
print(f"   This will take approximately {estimated_hours:.1f} hours")
print(f"   Progress will be logged every {CONFIG['logging_steps']} steps")
print(f"   Checkpoints saved every {CONFIG['save_steps']} steps")
print()

trainer.train()

print()
print("✅ PHASE 1 TRAINING COMPLETE!")
print()

# ============================================================================
# CELL 7: Save Model
# ============================================================================
print("💾 Saving Phase 1 model (MMedC)...")

trainer.model.save_pretrained(CONFIG["output_dir"] + "/final_model")
tokenizer.save_pretrained(CONFIG["output_dir"] + "/final_model")

print(f"✅ Saved to: {CONFIG['output_dir']}/final_model")
print()

# Create zip for download
import shutil
shutil.make_archive(
    "/kaggle/working/mmedc_lora",
    'zip',
    CONFIG["output_dir"] + "/final_model"
)

print("=" * 80)
print("PHASE 1 COMPLETE!")
print("=" * 80)
print()
print(f"✅ Trained on: {len(dataset):,} MMedC examples")
print(f"⏱️  Training time: {estimated_hours:.1f} hours")
print()
print("📥 Download: /kaggle/working/mmedc_lora.zip")
print()
print("🎯 Next Steps:")
print("1. Download mmedc_lora.zip")
print("2. Upload to Kaggle as dataset: 'mmedc-lora'")
print("3. Run Phase 2: train_phase2_shifaa.py")
print()
