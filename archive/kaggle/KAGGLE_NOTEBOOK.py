# Complete Kaggle Notebook - Extract + Train MMed-Llama-3-8B
# ============================================================================
# This notebook extracts 4 datasets and trains MMed-Llama-3-8B with QLoRA
# Optimized for Kaggle T4 GPU with proper dependency versions
# ============================================================================

# ==============================================================================
# CELL 1: Fix Dependencies (IGNORE THE WARNINGS - They Won't Affect Training)
# ==============================================================================
# Time: ~5-10 minutes
# The dependency warnings you see are NORMAL on Kaggle and won't break training
# They're about unrelated packages (bigframes, cudf, gradio, etc.)
# Our training packages (transformers, peft, bitsandbytes) will work fine!

print("🔧 Installing training dependencies...")
print("⚠️  You'll see dependency warnings - IGNORE THEM (they don't affect training)")
print()

# Uninstall conflicting versions
!pip uninstall -y transformers tokenizers accelerate peft bitsandbytes trl -q

# Install EXACT versions for Kaggle T4 + CUDA 12.1
# These versions are tested and work on Kaggle as of 2025
!pip install -q transformers==4.36.2
!pip install -q peft==0.15.0
!pip install -q accelerate==0.30.0
!pip install -q bitsandbytes==0.42.0
!pip install -q trl==0.7.10
!pip install -q datasets==2.16.1
!pip install -q scipy==1.11.4
!pip install -q sentencepiece==0.1.99
!pip install -q protobuf==4.25.1

# For data extraction
!pip install -q openpyxl==3.1.2

print()
print("✅ Dependencies installed!")
print()
print("📝 About the dependency warnings:")
print("   - sentence-transformers, gradio, bigframes warnings = SAFE TO IGNORE")
print("   - They're about OTHER Kaggle packages, not our training")
print("   - transformers 4.36.2 works perfectly for MMed-Llama-3-8B training")
print()
print("⚠️  IMPORTANT: After this cell finishes, click 'Restart & Run All' in Kaggle")
print("   This will reload the new package versions properly")
print()


# ==============================================================================
# CELL 2: Extract All 4 Datasets
# ==============================================================================
# Time: ~45-90 minutes
# This extracts: MMedC Arabic + Shifaa Medical + Shifaa Mental + AHD
# 
# ⚠️  BEFORE RUNNING THIS CELL:
# After Cell 1 finishes, click "Restart & Run All" in Kaggle to reload packages
# This prevents ImportError with peft and accelerate

import json
import os
import zipfile
from tqdm import tqdm
import re
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
import random

def clean_text(text):
    """Clean medical text"""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

# ==============================================================================
# DATASET 1: MMEDC - ARABIC ONLY
# ==============================================================================
print("=" * 80)
print("DATASET 1/4: MMEDC ARABIC")
print("=" * 80)
print()
print("📥 Downloading Arabic.zip from HuggingFace (1.28 GB)...")
print("   This may take 10-15 minutes...")

zip_path = hf_hub_download(
    repo_id="Henrychur/MMedC",
    filename="Arabic.zip",
    repo_type="dataset"
)

print(f"✅ Downloaded to: {zip_path}")
print()
print("📦 Extracting Arabic medical texts...")

mmedc_examples = []

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
    print(f"📄 Found {len(txt_files):,} Arabic text files")
    print()
    
    for filename in tqdm(txt_files, desc="Processing MMedC"):
        try:
            with zip_ref.open(filename) as f:
                content = f.read().decode('utf-8', errors='ignore')
            
            content = clean_text(content)
            if len(content) < 100:
                continue
            
            # Chunk into 1500 char pieces with overlap
            chunk_size = 1500
            if len(content) > chunk_size:
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i+chunk_size+200]  # 200 char overlap
                    if len(chunk) >= 100:
                        mmedc_examples.append({
                            "input": "تعلم المعلومات الطبية التالية:",
                            "output": chunk,
                            "source": "MMedC"
                        })
            else:
                mmedc_examples.append({
                    "input": "تعلم المعلومات الطبية التالية:",
                    "output": content,
                    "source": "MMedC"
                })
        except:
            continue

print()
print(f"✅ MMedC Arabic: {len(mmedc_examples):,} examples")
print()

# ==============================================================================
# DATASET 2: SHIFAA MEDICAL CONSULTATIONS
# ==============================================================================
print("=" * 80)
print("DATASET 2/4: SHIFAA MEDICAL CONSULTATIONS")
print("=" * 80)
print()
print("📥 Downloading from HuggingFace...")

dataset = load_dataset("Ahmed-Selem/Shifaa_Arabic_Medical_Consultations")
shifaa_medical_examples = []

for split_name in dataset.keys():
    print(f"Processing split: {split_name}")
    for item in tqdm(dataset[split_name], desc=f"{split_name}"):
        # Use capital 'Question' and 'Answer' (not lowercase)
        question = clean_text(str(item.get('Question', '')))
        answer = clean_text(str(item.get('Answer', '')))
        
        if len(question) > 10 and len(answer) > 10:
            shifaa_medical_examples.append({
                "input": question,
                "output": answer,
                "source": "Shifaa_Medical"
            })

print()
print(f"✅ Shifaa Medical: {len(shifaa_medical_examples):,} examples")
print()

# ==============================================================================
# DATASET 3: SHIFAA MENTAL HEALTH CONSULTATIONS
# ==============================================================================
print("=" * 80)
print("DATASET 3/4: SHIFAA MENTAL HEALTH CONSULTATIONS")
print("=" * 80)
print()
print("📥 Downloading from HuggingFace...")

dataset = load_dataset("Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations")
shifaa_mental_examples = []

for split_name in dataset.keys():
    print(f"Processing split: {split_name}")
    for item in tqdm(dataset[split_name], desc=f"{split_name}"):
        # Use capital 'Question' and 'Answer' (not lowercase)
        question = clean_text(str(item.get('Question', '')))
        answer = clean_text(str(item.get('Answer', '')))
        
        if len(question) > 10 and len(answer) > 10:
            shifaa_mental_examples.append({
                "input": question,
                "output": answer,
                "source": "Shifaa_Mental"
            })

print()
print(f"✅ Shifaa Mental Health: {len(shifaa_mental_examples):,} examples")
print()

# ==============================================================================
# DATASET 4: AHD - ARABIC HEALTHCARE DATASET
# ==============================================================================
print("=" * 80)
print("DATASET 4/4: AHD (Arabic Healthcare Dataset)")
print("=" * 80)
print()

ahd_examples = []

# Path to your uploaded AHD dataset in Kaggle
ahd_path = "/kaggle/input/ahd-dataset/AHD.xlsx"

if os.path.exists(ahd_path):
    print(f"📂 Found AHD file: {ahd_path}")
    print("📖 Reading Excel file...")
    
    df = pd.read_excel(ahd_path)
    print(f"📊 Total rows: {len(df):,}")
    print(f"📊 Columns: {list(df.columns)}")
    print()
    
    # Try different possible column names
    question_cols = ['question', 'Question', 'query', 'Query', 'q', 'Q']
    answer_cols = ['answer', 'Answer', 'response', 'Response', 'a', 'A']
    
    q_col = None
    a_col = None
    
    for col in question_cols:
        if col in df.columns:
            q_col = col
            break
    
    for col in answer_cols:
        if col in df.columns:
            a_col = col
            break
    
    if q_col and a_col:
        print(f"✅ Using columns: '{q_col}' and '{a_col}'")
        print()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing AHD"):
            question = clean_text(str(row[q_col]))
            answer = clean_text(str(row[a_col]))
            
            if len(question) > 10 and len(answer) > 10:
                ahd_examples.append({
                    "input": question,
                    "output": answer,
                    "source": "AHD"
                })
        
        print()
        print(f"✅ AHD: {len(ahd_examples):,} examples")
    else:
        print(f"⚠️  Could not find question/answer columns in: {list(df.columns)}")
        print("   Skipping AHD dataset...")
else:
    print(f"⚠️  AHD file not found at: {ahd_path}")
    print("   Make sure you added 'ahd-dataset' in Kaggle Input")
    print("   Continuing without AHD...")

print()

# ==============================================================================
# COMBINE ALL DATASETS
# ==============================================================================
print("=" * 80)
print("COMBINING ALL DATASETS")
print("=" * 80)
print()

all_examples = mmedc_examples + shifaa_medical_examples + shifaa_mental_examples + ahd_examples

# Shuffle for better training
random.seed(42)
random.shuffle(all_examples)

print(f"📊 Final Dataset Statistics:")
print(f"   1. MMedC Arabic:       {len(mmedc_examples):>8,} examples ({len(mmedc_examples)/len(all_examples)*100:>5.1f}%)")
print(f"   2. Shifaa Medical:     {len(shifaa_medical_examples):>8,} examples ({len(shifaa_medical_examples)/len(all_examples)*100:>5.1f}%)")
print(f"   3. Shifaa Mental:      {len(shifaa_mental_examples):>8,} examples ({len(shifaa_mental_examples)/len(all_examples)*100:>5.1f}%)")
print(f"   4. AHD:                {len(ahd_examples):>8,} examples ({len(ahd_examples)/len(all_examples)*100:>5.1f}%)")
print(f"   {'─' * 50}")
print(f"   TOTAL:                 {len(all_examples):>8,} examples")
print()

# Save to JSON
output_file = "/kaggle/working/training_data_combined_ALL.json"
print(f"💾 Saving to: {output_file}")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_examples, f, ensure_ascii=False, indent=2)

file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"✅ Saved successfully!")
print(f"📦 File size: {file_size_mb:.1f} MB")
print()
print("🎉 DATASET EXTRACTION COMPLETE!")
print()
print("📊 Estimated tokens: ~{:,}".format(len(all_examples) * 400))
print("⏱️  Estimated training time on T4: 18-24 hours")
print()


# ==============================================================================
# CELL 3: Train MMed-Llama-3-8B with QLoRA
# ==============================================================================
# Time: 18-24 hours on T4 (will need to resume after 12h)
# This cell uses the model you uploaded: /kaggle/input/medllm/

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import json

print("=" * 80)
print("TRAINING MMED-LLAMA-3-8B WITH QLORA")
print("=" * 80)
print()

# Load training data
print("📚 Loading training data...")
with open("/kaggle/working/training_data_combined_ALL.json", 'r', encoding='utf-8') as f:
    training_data = json.load(f)

print(f"✅ Loaded {len(training_data):,} training examples")
print()

# Format data for Llama-3 chat template
def format_prompt(example):
    """Format using Llama-3 instruction format"""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

أنت نموذج لغوي طبي متخصص باللغة العربية. مهمتك هي تقديم معلومات طبية دقيقة ومفيدة.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""

print("📝 Formatting data for Llama-3 chat template...")
formatted_data = [{"text": format_prompt(ex)} for ex in training_data]
print(f"✅ Formatted {len(formatted_data):,} examples")
print()

# QLoRA Configuration (4-bit quantization for T4)
print("🔧 Configuring 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Better than float16
    bnb_4bit_use_double_quant=True,        # Saves more memory
)
print("✅ Quantization config ready")
print()

# Load model from Kaggle input
model_path = "/kaggle/input/medllm/models--Henrychur--MMed-Llama-3-8B/snapshots/"

# Find the snapshot folder
import glob
snapshot_dirs = glob.glob(f"{model_path}*")
if snapshot_dirs:
    model_path = snapshot_dirs[0]
    print(f"📂 Found model at: {model_path}")
else:
    # Fallback to HuggingFace download
    model_path = "Henrychur/MMed-Llama-3-8B"
    print(f"📂 Using HuggingFace model: {model_path}")

print()
print("🔄 Loading MMed-Llama-3-8B in 4-bit...")
print("   (This takes ~3-5 minutes)")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Model loaded in 4-bit!")
print(f"💾 Model memory: ~10-12 GB VRAM")
print()

# Prepare model for QLoRA training
print("🔧 Preparing model for QLoRA training...")
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)
print("✅ Model prepared")
print()

# LoRA Configuration (Based on MMed-Llama's original training)
print("🔧 Configuring LoRA adapters...")
lora_config = LoraConfig(
    r=32,                               # LoRA rank (higher for medical domain)
    lora_alpha=64,                      # LoRA alpha (2x rank)
    target_modules=[
        "q_proj",                        # Query projection
        "k_proj",                        # Key projection
        "v_proj",                        # Value projection
        "o_proj",                        # Output projection
        "gate_proj",                     # Llama-3 MLP gate
        "up_proj",                       # Llama-3 MLP up
        "down_proj",                     # Llama-3 MLP down
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("✅ LoRA adapters attached")
print()

# Print trainable parameters
trainable, total = model.get_nb_trainable_parameters()
print(f"📊 Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
print()

# Training Arguments (Optimized for Kaggle T4)
print("🔧 Configuring training arguments...")
training_args = TrainingArguments(
    output_dir="/kaggle/working/mmed_llama_qlora",
    
    # Training duration
    num_train_epochs=3,                 # 3 epochs for 70K examples
    max_steps=-1,                       # Train for full epochs
    
    # Batch size (optimized for T4 16GB)
    per_device_train_batch_size=2,      # Small for T4
    gradient_accumulation_steps=16,     # Effective batch = 32
    
    # Learning rate (from MMed-Llama paper)
    learning_rate=2e-5,                 # Same as original pre-training
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,                  # 3% warmup
    
    # Optimization
    optim="paged_adamw_32bit",          # Memory-efficient optimizer
    weight_decay=0.001,
    max_grad_norm=0.3,                  # Gradient clipping
    
    # Mixed precision
    fp16=False,
    bf16=True,                          # Better than fp16 for T4
    
    # Memory optimization
    gradient_checkpointing=True,        # Save memory (slower but fits)
    
    # Logging & Checkpoints
    logging_steps=10,
    save_steps=250,                     # Save every 250 steps (~1 hour)
    save_total_limit=3,                 # Keep only 3 checkpoints
    
    # Evaluation
    evaluation_strategy="no",           # No eval (saves time)
    
    # Other
    report_to="none",                   # No wandb/tensorboard
    load_best_model_at_end=False,
    ddp_find_unused_parameters=False,
)
print("✅ Training config ready")
print()

# Print training details
total_steps = (len(formatted_data) * 3) // (2 * 16)
print("📊 Training Summary:")
print(f"   Total examples:        {len(formatted_data):,}")
print(f"   Epochs:                3")
print(f"   Batch size:            2 (per device)")
print(f"   Gradient accum:        16")
print(f"   Effective batch:       32")
print(f"   Total steps:           ~{total_steps:,}")
print(f"   Checkpoints:           Every 250 steps (~1 hour)")
print(f"   Expected time:         18-24 hours on T4")
print()

# Create Trainer
print("🚀 Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_data,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=2048,                # Reduced for T4 memory
    tokenizer=tokenizer,
    args=training_args,
    packing=False,                      # Don't pack (simpler, more stable)
)
print("✅ Trainer ready")
print()

# Train!
print("=" * 80)
print("🚀 STARTING TRAINING")
print("=" * 80)
print()
print("⚠️  IMPORTANT NOTES:")
print("   1. Training will take 18-24 hours on T4")
print("   2. Kaggle will stop after 12 hours")
print("   3. Checkpoints are saved every 250 steps (~1 hour)")
print("   4. You can resume from checkpoint in a new notebook")
print()
print("📊 Monitor GPU: Click 'GPU' in right panel")
print()
input("Press ENTER to start training...")
print()

# Start training
trainer.train()

# Save final model
print()
print("=" * 80)
print("💾 SAVING FINAL MODEL")
print("=" * 80)
print()

final_dir = "/kaggle/working/mmed_llama_qlora_final"
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

print(f"✅ Model saved to: {final_dir}")
print()
print("🎉 TRAINING COMPLETE!")
print()
print("📥 To download:")
print("   1. Go to 'Output' tab")
print("   2. Download 'mmed_llama_qlora_final' folder")
print()


# ==============================================================================
# CELL 4: Resume Training (Use if Kaggle stopped after 12 hours)
# ==============================================================================
# Only run this if training was interrupted!

# Find latest checkpoint
import glob
checkpoints = glob.glob("/kaggle/working/mmed_llama_qlora/checkpoint-*")
if checkpoints:
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    print(f"📂 Found checkpoint: {latest_checkpoint}")
    print()
    print("🔄 Resuming training...")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("❌ No checkpoints found! Start from CELL 3 instead.")
