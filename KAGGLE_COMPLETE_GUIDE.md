# Complete Kaggle Training Guide - Everything in One Notebook

## 🎯 Can You Train Everything on Kaggle?

**YES!** ✅ You can do everything on Kaggle:
- Extract all 4 datasets
- Train with QLoRA
- Save trained model

**BUT there are limitations:**
- ⚠️ **GPU Quota:** 30 hours/week (NOT $30 credits)
- ⚠️ **Session Limit:** 12 hours max per session
- ⚠️ **GPUs:** Only T4 (16GB) or P100 (16GB)
- ⚠️ **Slower:** T4/P100 are slower than A100

---

## 💰 Sharing Credits with Others

### ❌ SHORT ANSWER: NO, you CANNOT share Kaggle GPU quota

**How Kaggle GPU Quota Works:**
- Each Kaggle account gets **30 GPU hours per week** (FREE)
- Quota is **per account**, NOT per notebook
- If you share your notebook with someone:
  - They can **copy** your notebook to their account
  - They use **THEIR OWN** 30 hours/week quota
  - Your quota stays with YOU
  - Their quota stays with THEM

**So if your 30 hours run out:**
1. ❌ You CANNOT use someone else's quota on your notebook
2. ✅ Someone else can COPY your notebook and use THEIR quota
3. ✅ Wait until next week for quota reset (every Monday)

### How to Share Notebook (So Others Can Use Their Quota):

**Step 1: Make notebook public**
- In your Kaggle notebook, click "Share" → "Public"

**Step 2: Share the link**
- Send the notebook URL to others
- Example: `https://kaggle.com/YOUR_USERNAME/notebook-name`

**Step 3: They copy it**
- They click "Copy & Edit" in your notebook
- Now it's THEIR notebook using THEIR 30 hours/week
- They can train with their own quota

**Important:** Each person needs their own Kaggle account with verified phone number to get 30 hours/week.

---

## 🖥️ Kaggle GPU Options

| GPU | VRAM | Speed | Quota Cost | Available |
|-----|------|-------|------------|-----------|
| **T4** | 16GB | ⭐⭐ | 1x | ✅ Yes |
| **P100** | 16GB | ⭐ | 1x | ✅ Yes |

**Comparison to Modal:**
- Kaggle T4 = Modal T4 (same GPU)
- Kaggle P100 = Older, slower than T4
- **Modal A100 is 5-6x faster than Kaggle T4**

**Recommended for Kaggle:** Use **T4** (faster than P100)

---

## ⚠️ Kaggle Limitations for Your Training

### Training Time Estimate:
- **70-85K examples on T4:** ~18-24 hours
- **Problem:** Kaggle has 12-hour session limit
- **Solution:** Use checkpointing and resume

### Will It Fit?
- **T4 has 16GB VRAM**
- **QLoRA with 8B model:** Uses ~10-12GB
- ✅ **YES, it will fit!**

### Compatibility Issues on Kaggle:
Kaggle has pre-installed packages that may conflict. Here's what to do:

---

## 📦 Step-by-Step Kaggle Setup

### Step 1: Create New Kaggle Notebook

1. Go to https://kaggle.com/code
2. Click "New Notebook"
3. Turn on **GPU T4 x2** accelerator (Settings → Accelerator → GPU T4 x2)
4. Set **Internet** to ON (Settings → Internet → ON)

---

### Step 2: Install Dependencies (First Cell)

**CRITICAL:** Use these EXACT versions to avoid compatibility issues:

```python
# Cell 1: Install Dependencies
# This takes ~5 minutes

# Uninstall conflicting packages first
!pip uninstall -y transformers tokenizers accelerate peft bitsandbytes trl

# Install exact versions (tested for compatibility)
!pip install -q transformers==4.36.0
!pip install -q peft==0.7.1
!pip install -q accelerate==0.25.0
!pip install -q bitsandbytes==0.41.3
!pip install -q trl==0.7.4
!pip install -q datasets==2.16.0
!pip install -q scipy
!pip install -q sentencepiece
!pip install -q protobuf

# For data extraction
!pip install -q huggingface_hub==0.20.0
!pip install -q openpyxl==3.1.2

print("✅ All dependencies installed!")
```

**Why these specific versions?**
- **transformers 4.36.0:** Stable version compatible with Kaggle's CUDA
- **peft 0.7.1:** QLoRA support without conflicts
- **bitsandbytes 0.41.3:** Works with Kaggle's CUDA 11.8
- **trl 0.7.4:** SFTTrainer for supervised fine-tuning
- **datasets 2.16.0:** Stable HuggingFace datasets

**⚠️ DO NOT use newer versions on Kaggle - they cause CUDA errors!**

---

### Step 3: Extract Datasets (Second Cell)

```python
# Cell 2: Extract All 4 Datasets
# This takes ~30-60 minutes

import json
import os
import zipfile
from tqdm import tqdm
import re
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd

def clean_text(text):
    """Clean medical text"""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

# ==============================================================================
# 1. EXTRACT MMEDC - ARABIC ONLY
# ==============================================================================
print("=" * 80)
print("STEP 1/4: DOWNLOADING MMEDC ARABIC")
print("=" * 80)

# Download Arabic.zip from HuggingFace
zip_path = hf_hub_download(
    repo_id="Henrychur/MMedC",
    filename="Arabic.zip",
    repo_type="dataset"
)

print(f"✅ Downloaded: {zip_path}")
print()

print("EXTRACTING MMEDC...")
examples = []

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
    
    for filename in tqdm(txt_files, desc="Processing MMedC"):
        try:
            with zip_ref.open(filename) as f:
                content = f.read().decode('utf-8', errors='ignore')
            
            content = clean_text(content)
            if len(content) < 100:
                continue
            
            # Chunk into 1500 char pieces
            chunk_size = 1500
            if len(content) > chunk_size:
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i+chunk_size+200]
                    if len(chunk) >= 100:
                        examples.append({
                            "input": "تعلم المعلومات الطبية التالية:",
                            "output": chunk,
                            "source": "MMedC"
                        })
            else:
                examples.append({
                    "input": "تعلم المعلومات الطبية التالية:",
                    "output": content,
                    "source": "MMedC"
                })
        except:
            continue

mmedc_examples = examples
print(f"✅ MMedC: {len(mmedc_examples):,} examples")

# ==============================================================================
# 2. EXTRACT SHIFAA MEDICAL
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 2/4: DOWNLOADING SHIFAA MEDICAL")
print("=" * 80)

dataset = load_dataset("Ahmed-Selem/Shifaa_Arabic_Medical_Consultations")
shifaa_medical_examples = []

for split_name in dataset.keys():
    for item in tqdm(dataset[split_name], desc=f"Processing {split_name}"):
        question = clean_text(item.get('question', ''))
        answer = clean_text(item.get('answer', ''))
        
        if len(question) > 10 and len(answer) > 10:
            shifaa_medical_examples.append({
                "input": question,
                "output": answer,
                "source": "Shifaa_Medical"
            })

print(f"✅ Shifaa Medical: {len(shifaa_medical_examples):,} examples")

# ==============================================================================
# 3. EXTRACT SHIFAA MENTAL HEALTH
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 3/4: DOWNLOADING SHIFAA MENTAL HEALTH")
print("=" * 80)

dataset = load_dataset("Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations")
shifaa_mental_examples = []

for split_name in dataset.keys():
    for item in tqdm(dataset[split_name], desc=f"Processing {split_name}"):
        question = clean_text(item.get('question', ''))
        answer = clean_text(item.get('answer', ''))
        
        if len(question) > 10 and len(answer) > 10:
            shifaa_mental_examples.append({
                "input": question,
                "output": answer,
                "source": "Shifaa_Mental"
            })

print(f"✅ Shifaa Mental Health: {len(shifaa_mental_examples):,} examples")

# ==============================================================================
# 4. EXTRACT AHD (If uploaded)
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 4/4: EXTRACTING AHD (if available)")
print("=" * 80)

ahd_examples = []
ahd_file = "/kaggle/input/ahd-dataset/AHD.xlsx"  # Upload to Kaggle dataset first

if os.path.exists(ahd_file):
    print(f"Found AHD file: {ahd_file}")
    df = pd.read_excel(ahd_file)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing AHD"):
        question = clean_text(str(row.get('question', '')))
        answer = clean_text(str(row.get('answer', '')))
        
        if len(question) > 10 and len(answer) > 10:
            ahd_examples.append({
                "input": question,
                "output": answer,
                "source": "AHD"
            })
    
    print(f"✅ AHD: {len(ahd_examples):,} examples")
else:
    print("⚠️  AHD.xlsx not found. Skipping...")
    print("   To include AHD: Upload as Kaggle dataset first")

# ==============================================================================
# 5. COMBINE ALL
# ==============================================================================
print("\n" + "=" * 80)
print("COMBINING ALL DATASETS")
print("=" * 80)

all_examples = mmedc_examples + shifaa_medical_examples + shifaa_mental_examples + ahd_examples

# Shuffle
import random
random.shuffle(all_examples)

print(f"\n📊 Dataset Breakdown:")
print(f"   MMedC: {len(mmedc_examples):,}")
print(f"   Shifaa Medical: {len(shifaa_medical_examples):,}")
print(f"   Shifaa Mental: {len(shifaa_mental_examples):,}")
print(f"   AHD: {len(ahd_examples):,}")
print(f"   {'─' * 40}")
print(f"   TOTAL: {len(all_examples):,} examples")

# Save
output_file = "/kaggle/working/training_data_combined_ALL.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_examples, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved to: {output_file}")
print(f"📦 File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
print("\n🎉 EXTRACTION COMPLETE!")
```

---

### Step 4: Train with QLoRA (Third Cell)

```python
# Cell 3: Train MMed-Llama with QLoRA
# This takes ~18-24 hours on T4 (will need to resume after 12h limit)

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

# Load training data
print("📚 Loading training data...")
with open("/kaggle/working/training_data_combined_ALL.json", 'r', encoding='utf-8') as f:
    training_data = json.load(f)

print(f"✅ Loaded {len(training_data):,} examples")

# Format data for training
def format_prompt(example):
    return f"### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

formatted_data = [{"text": format_prompt(ex)} for ex in training_data]

# QLoRA Configuration (4-bit quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model in 4-bit
print("🔄 Loading MMed-Llama-3-8B in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    "Henrychur/MMed-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("Henrychur/MMed-Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)

# LoRA Configuration
lora_config = LoraConfig(
    r=32,                          # LoRA rank
    lora_alpha=64,                 # LoRA alpha
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training Arguments (Optimized for T4)
training_args = TrainingArguments(
    output_dir="/kaggle/working/mmed_llama_qlora",
    num_train_epochs=3,
    per_device_train_batch_size=4,      # Small batch for T4
    gradient_accumulation_steps=8,       # Effective batch = 32
    learning_rate=2e-4,
    fp16=True,                           # Mixed precision
    save_steps=500,                      # Save checkpoints often
    logging_steps=10,
    save_total_limit=3,                  # Keep only 3 checkpoints
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="paged_adamw_32bit",           # Memory efficient optimizer
    gradient_checkpointing=True,         # Save memory
    max_grad_norm=0.3,
    weight_decay=0.001,
)

# Create Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_data,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=1024,                 # Reduced for T4
    tokenizer=tokenizer,
    args=training_args,
)

# Train!
print("🚀 Starting training...")
print("⚠️  This will take ~18-24 hours on T4")
print("⚠️  Kaggle will stop after 12 hours - resume from checkpoint")
print()

trainer.train()

# Save final model
print("💾 Saving final model...")
model.save_pretrained("/kaggle/working/mmed_llama_qlora_final")
tokenizer.save_pretrained("/kaggle/working/mmed_llama_qlora_final")

print("✅ Training complete!")
print("📥 Download from: /kaggle/working/mmed_llama_qlora_final")
```

---

### Step 5: Resume Training After 12-Hour Limit

If Kaggle stops after 12 hours, create a NEW notebook and run:

```python
# Cell: Resume Training from Checkpoint

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel
from trl import SFTTrainer
import json

# Load data
with open("/kaggle/working/training_data_combined_ALL.json", 'r') as f:
    training_data = json.load(f)

formatted_data = [{"text": format_prompt(ex)} for ex in training_data]

# Load from checkpoint
checkpoint_dir = "/kaggle/working/mmed_llama_qlora/checkpoint-XXX"  # Replace XXX with latest

# Resume training
trainer.train(resume_from_checkpoint=checkpoint_dir)
```

---

## 📥 Download Trained Model from Kaggle

**Option 1: Download via Kaggle UI**
1. Go to your notebook
2. Click "Output" tab
3. Download `mmed_llama_qlora_final` folder

**Option 2: Save to Kaggle Dataset**
```python
# In your notebook, last cell:
!cp -r /kaggle/working/mmed_llama_qlora_final /kaggle/working/
```
Then download from Output.

---

## ⚖️ Kaggle vs Modal Comparison

| Feature | Kaggle | Modal ($5 budget) |
|---------|--------|-------------------|
| **GPU** | T4 (16GB) | L4 (24GB) |
| **Training Time** | 18-24 hours | 12-14 hours |
| **Session Limit** | 12 hours (must resume) | 12 hours configurable |
| **Cost** | FREE | $5 = ~4.5 hours on L4 |
| **Speed** | Slower | Faster |
| **Ease** | Need checkpointing | One shot |
| **Internet** | Limited | Full |
| **Dependencies** | Compatibility issues | No issues |

### With Your $5 on Modal:

**Option 1: L4 GPU (24GB)** - RECOMMENDED ✅
- **Cost:** $1.10/hour
- **Time:** 12-14 hours needed
- **Your $5:** Covers ~4.5 hours
- **Problem:** Need $13-15 total
- ❌ **Not enough budget**

**Option 2: T4 GPU (16GB)** - Matches Kaggle
- **Cost:** $0.60/hour
- **Time:** 18-24 hours needed
- **Your $5:** Covers ~8 hours
- **Problem:** Need $11-14 total
- ❌ **Not enough budget**

### **RECOMMENDATION with $5:**

**Use Kaggle for FREE! Here's why:**
1. ✅ **FREE** - No cost
2. ✅ **30 hours/week** - Enough for training
3. ✅ **T4 GPU** - Same as Modal T4
4. ⚠️ **Need to resume after 12h** - But still free

**Then if you want faster future training:**
- Save up $15-20 for Modal L4
- Or wait for $30 free credits on Modal

---

## ✅ Final Checklist for Kaggle

### Before Starting:
- [ ] Kaggle account created and phone verified
- [ ] 30 GPU hours available (check weekly quota)
- [ ] New notebook created
- [ ] GPU T4 x2 enabled
- [ ] Internet enabled
- [ ] AHD.xlsx uploaded as Kaggle dataset (optional)

### During Training:
- [ ] Cell 1: Install dependencies (5 min)
- [ ] Cell 2: Extract datasets (30-60 min)
- [ ] Cell 3: Train model (18-24 hours, resume after 12h)
- [ ] Save checkpoints every 500 steps
- [ ] Monitor GPU memory usage
- [ ] Download final model

### After Training:
- [ ] Download `mmed_llama_qlora_final` folder
- [ ] Test model locally
- [ ] Integrate into your LLM service

---

## 🔥 Quick Start Commands for Kaggle

```python
# ==============================================================================
# COMPLETE KAGGLE NOTEBOOK - ALL IN ONE
# ==============================================================================

# CELL 1: Install dependencies (5 min)
!pip uninstall -y transformers tokenizers accelerate peft bitsandbytes trl
!pip install -q transformers==4.36.0 peft==0.7.1 accelerate==0.25.0
!pip install -q bitsandbytes==0.41.3 trl==0.7.4 datasets==2.16.0
!pip install -q scipy sentencepiece protobuf huggingface_hub==0.20.0 openpyxl==3.1.2

# CELL 2: Extract datasets (30-60 min)
# [Paste extraction code from Step 3 above]

# CELL 3: Train model (18-24 hours)
# [Paste training code from Step 4 above]
```

---

## 🆘 Troubleshooting on Kaggle

### "CUDA out of memory"
**Fix:**
- Reduce `per_device_train_batch_size` (4 → 2)
- Reduce `max_seq_length` (1024 → 512)
- Use `gradient_checkpointing=True`

### "Transformers version conflict"
**Fix:**
- Use EXACT versions: `transformers==4.36.0`
- Restart notebook after installing

### "Session expired after 12 hours"
**Fix:**
- This is normal - Kaggle's limit
- Resume from latest checkpoint
- See "Step 5: Resume Training" above

### "GPU quota exceeded"
**Fix:**
- Wait until Monday (weekly reset)
- Or have someone copy your notebook and use their quota

---

## 🎉 Summary

**YES, you can train everything on Kaggle for FREE!**

**Steps:**
1. Create Kaggle notebook with T4 GPU
2. Install dependencies (5 min)
3. Extract 4 datasets (30-60 min)
4. Train with QLoRA (18-24 hours, resume after 12h)
5. Download trained model

**Total Cost:** $0 (FREE)  
**Total Time:** ~20-25 hours  
**GPU Quota:** Uses 20-25 of your 30 hours/week

**Sharing:**
- Others can COPY your notebook
- They use THEIR 30 hours/week quota
- You CANNOT share YOUR quota with them

**With $5 on Modal:**
- Not enough for full training ($11-14 needed)
- Better to use Kaggle for free
- Save $5 for future experiments

**Ready to start?** Open Kaggle and create your first notebook! 🚀
