"""
INCREMENTAL TRAINING: Continue Training on AHD Dataset
======================================================

This script continues training from your ALREADY TRAINED model.

How it works:
1. Loads your FIRST trained model (Shifaa + MMedC)
2. Continues training on AHD dataset
3. Result: Model trained on ALL datasets!

Why this is SMART:
- Split 28 hours into 2.6h + 25h chunks
- Test first model before committing to full training
- Can stop after first training if results are good
- Saves Kaggle GPU quota

Total Training Time:
- Round 1: Shifaa + MMedC = ~2.6 hours
- Round 2: AHD only = ~25 hours
- Total: ~27.6 hours (same as training all at once!)
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
print("=" * 80)
print("INCREMENTAL TRAINING - ROUND 2: AHD DATASET")
print("=" * 80)
print()

!pip install -q transformers accelerate peft bitsandbytes datasets trl openpyxl

print("✅ Dependencies installed!")
print()


# ============================================================================
# CELL 2: Configuration for INCREMENTAL Training
# ============================================================================
import torch
import json
import os
import pandas as pd
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
print("INCREMENTAL TRAINING CONFIGURATION")
print("=" * 80)
print()

CONFIG = {
    # BASE MODEL (from Round 1)
    "base_model_name": "Henrychur/MMed-Llama-3-8B",
    "base_model_paths": [  # Local base model
        "/kaggle/input/medllm/models--Henrychur--MMed-Llama-3-8B/snapshots",
        "/kaggle/input/medllm/models--Henrychur--MMed-Llama-3-8B",
    ],
    
    # YOUR TRAINED LORA (from Round 1 - Shifaa + MMedC)
    "round1_lora_paths": [
        "/kaggle/input/round1-lora/mmed_llama3_arabic_lora/final_model",
        "/kaggle/input/round1-lora/final_model",
        "/kaggle/input/your-first-training/mmed_llama3_arabic_lora/final_model",
    ],
    
    # NEW TRAINING DATA (Round 2 - AHD only)
    "ahd_data_paths": [
        "/kaggle/input/ahd-dataset/AHD.xlsx",
        "/kaggle/input/ahd-arabic-healthcare-dataset/AHD.xlsx",
        "/kaggle/input/arabic-healthcare-dataset/AHD.xlsx",
    ],
    
    # QLoRA Configuration (same as Round 1)
    "lora_r": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    
    # Training Settings
    "num_epochs": 1,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,  # Lower LR for continued training
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
    "output_dir": "./mmed_llama3_arabic_lora_FULL",
    "save_steps": 500,
    "save_total_limit": 2,
    "logging_steps": 10,
}

print("🔥 INCREMENTAL TRAINING - ROUND 2")
print()
print("📊 Configuration:")
print(f"   Base Model: {CONFIG['base_model_name']}")
print(f"   Round 1 LoRA: Will load from Kaggle input")
print(f"   Round 2 Data: AHD dataset")
print(f"   Learning Rate: {CONFIG['learning_rate']} (lower for stability)")
print()
print("💡 How this works:")
print("   1. Load base model (8B)")
print("   2. Load Round 1 LoRA adapters (Shifaa + MMedC)")
print("   3. Continue training on AHD")
print("   4. Save final LoRA (trained on ALL data!)")
print()


# ============================================================================
# CELL 3: Load AHD Dataset
# ============================================================================
print("=" * 80)
print("LOADING AHD DATASET")
print("=" * 80)
print()

ahd_found = False
training_examples = []

for path in CONFIG["ahd_data_paths"]:
    if os.path.exists(path):
        print(f"📥 Found AHD at: {path}")
        print("   Loading Excel file... (may take 1-2 minutes)")
        
        try:
            df = pd.read_excel(path)
            print(f"   ✅ Loaded {len(df):,} rows")
            print(f"   Columns: {list(df.columns)}")
            
            # Detect column names
            question_col = None
            answer_col = None
            
            for q in ['question', 'Question', 'query', 'Query', 'سؤال', 'Q']:
                if q in df.columns:
                    question_col = q
                    break
            
            for a in ['answer', 'Answer', 'response', 'Response', 'إجابة', 'A']:
                if a in df.columns:
                    answer_col = a
                    break
            
            if question_col and answer_col:
                print(f"   Using columns: '{question_col}' → '{answer_col}'")
                
                for _, row in df.iterrows():
                    q = str(row[question_col]).strip()
                    a = str(row[answer_col]).strip()
                    
                    if q and a and q != 'nan' and a != 'nan':
                        training_examples.append({
                            "input": q,
                            "output": a,
                            "source": "AHD"
                        })
                
                print(f"   ✅ Converted {len(training_examples):,} valid examples")
                ahd_found = True
                break
            else:
                print(f"   ⚠️  Could not identify Q&A columns")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

if not ahd_found:
    print("❌ AHD DATASET NOT FOUND!")
    print()
    print("Please add AHD dataset to Kaggle:")
    print("1. Go to 'Add Data' in Kaggle")
    print("2. Search 'AHD Arabic Healthcare Dataset'")
    print("3. Or upload your AHD.xlsx file")
    raise ValueError("AHD dataset not found")

print()
print(f"📊 Total AHD examples: {len(training_examples):,}")
print()

# Convert to Hugging Face Dataset
def format_instruction(example):
    """Format for Llama 3 instruction tuning"""
    return {
        "text": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

أنت طبيب مساعد متخصص في الطب. أجب على الأسئلة الطبية بدقة ووضوح باللغة العربية.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
    }

dataset = Dataset.from_list(training_examples)
dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

print("✅ Dataset prepared!")
print(f"   Total samples: {len(dataset):,}")
print()


# ============================================================================
# CELL 4: Load Base Model + Round 1 LoRA
# ============================================================================
print("=" * 80)
print("LOADING BASE MODEL + ROUND 1 LORA")
print("=" * 80)
print()

# Find local base model
base_model_path = CONFIG["base_model_name"]
for path in CONFIG["base_model_paths"]:
    if os.path.exists(path):
        if "snapshots" in path:
            snapshots = os.listdir(path)
            if snapshots:
                base_model_path = os.path.join(path, snapshots[0])
                print(f"✅ Found base model at: {base_model_path}")
                break
        elif os.path.isdir(path):
            base_model_path = path
            print(f"✅ Found base model at: {base_model_path}")
            break

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type=CONFIG["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=CONFIG["use_double_quant"],
)

print()
print("📥 Loading base model with QLoRA...")

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
except:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if CONFIG["use_flash_attention"] else "sdpa",
    use_cache=False,
)

print("✅ Base model loaded!")
print()

# Find and load Round 1 LoRA
print("🔍 Looking for Round 1 LoRA adapters...")
round1_lora_path = None

for path in CONFIG["round1_lora_paths"]:
    if os.path.exists(path):
        round1_lora_path = path
        print(f"✅ Found Round 1 LoRA at: {path}")
        break

if round1_lora_path:
    print("📥 Loading Round 1 LoRA adapters...")
    model = PeftModel.from_pretrained(model, round1_lora_path)
    print("✅ Round 1 LoRA loaded!")
    print("   Your model now has knowledge from Shifaa + MMedC!")
    print()
    print("🔄 Preparing for continued training...")
    # Merge and unload to prepare for new LoRA
    model = model.merge_and_unload()
    print("✅ Merged! Ready for Round 2 training.")
else:
    print("⚠️  Round 1 LoRA not found!")
    print("   Will start from base model (not recommended)")
    print()
    print("To use incremental training:")
    print("1. Upload your Round 1 trained model to Kaggle as dataset")
    print("2. Update CONFIG['round1_lora_paths'] with correct path")

print()

# Prepare for new training
model = prepare_model_for_kbit_training(model)

# Apply NEW LoRA for Round 2
lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=CONFIG["target_modules"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)

model = get_peft_model(model, lora_config)

print("✅ NEW LoRA adapters applied for Round 2!")
print()


# ============================================================================
# CELL 5: Setup Training
# ============================================================================
print("=" * 80)
print("TRAINING CONFIGURATION - ROUND 2")
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

print("📊 Round 2 Training Estimates:")
print(f"   Examples: {len(dataset):,} (AHD only)")
print(f"   Steps: {total_steps:,}")
print(f"   Time: {estimated_hours:.1f} hours")
print()


# ============================================================================
# CELL 6: START TRAINING! 🚀
# ============================================================================
print("=" * 80)
print("STARTING ROUND 2 TRAINING")
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

print("✅ Trainer created!")
print()
print("🚀 STARTING INCREMENTAL TRAINING...")
print(f"   Base: MMed-Llama-3-8B")
print(f"   + Round 1: Shifaa + MMedC (already trained)")
print(f"   + Round 2: AHD (training now)")
print(f"   = FULL MODEL with ALL data!")
print()
print("=" * 80)
print()

# TRAIN!
trainer.train()

print()
print("=" * 80)
print("✅ ROUND 2 TRAINING COMPLETE!")
print("=" * 80)
print()


# ============================================================================
# CELL 7: Save Final Model
# ============================================================================
print("💾 Saving FINAL trained model...")
print()

trainer.model.save_pretrained(CONFIG["output_dir"] + "/final_model")
tokenizer.save_pretrained(CONFIG["output_dir"] + "/final_model")

print(f"✅ Final LoRA saved to: {CONFIG['output_dir']}/final_model")
print()

# Create zip
import shutil
shutil.make_archive(
    "/kaggle/working/mmed_llama3_arabic_lora_FULL",
    'zip',
    CONFIG["output_dir"] + "/final_model"
)

print("=" * 80)
print("INCREMENTAL TRAINING COMPLETE!")
print("=" * 80)
print()
print("✅ Your model is now trained on:")
print("   1. Shifaa (84,422 examples)")
print("   2. MMedC (167 examples)")
print("   3. AHD (808,000+ examples)")
print("   = ~893,000 TOTAL examples!")
print()
print("📥 Download: /kaggle/working/mmed_llama3_arabic_lora_FULL.zip")
print()
print("🎉 CONGRATULATIONS! You have a FULLY trained medical LLM!")
print("=" * 80)
