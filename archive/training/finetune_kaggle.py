"""
Fine-tune MMed-Llama-3-8B on Egyptian Arabic Medical Data
==========================================================

This script fine-tunes MMed-Llama-3-8B using QLoRA (4-bit) on Kaggle FREE GPU.

Requirements:
    - Kaggle GPU T4 (free 30 hours/week)
    - training_data.json uploaded as Kaggle dataset
    
Time: 5-10 hours on T4 GPU
Cost: $0 (FREE!)

Output: LoRA adapters (~100MB) that improve Arabic medical performance

Usage:
    1. Upload training_data.json to Kaggle as dataset
    2. Create new Kaggle notebook with GPU
    3. Add dataset to notebook
    4. Copy this script and run

The fine-tuned model will work EXACTLY like the base model,
just better quality Arabic medical outputs!
"""

import os
import json
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (adjust for Kaggle)
BASE_MODEL = "Henrychur/MMed-Llama-3-8B"
TRAINING_DATA_PATH = "/kaggle/input/egyptian-medical-training/training_data.json"
OUTPUT_DIR = "/kaggle/working/egyptian-medical-lora"
MODEL_CACHE = "/kaggle/working/model_cache"

# Training hyperparameters
BATCH_SIZE = 4  # Fits in T4 14GB VRAM
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 1024

# LoRA configuration
LORA_R = 16  # Rank
LORA_ALPHA = 32  # Scaling
LORA_DROPOUT = 0.05

print("=" * 80)
print("FINE-TUNING MMed-Llama-3-8B FOR EGYPTIAN ARABIC MEDICAL")
print("=" * 80)
print()
print(f"Base model: {BASE_MODEL}")
print(f"Training data: {TRAINING_DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# ============================================================================
# LOAD TRAINING DATA
# ============================================================================

print("Loading training data...")
with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
    training_data = json.load(f)

print(f"✅ Loaded {len(training_data)} examples")
print()

# Sample
print("Sample example:")
print("-" * 80)
print(f"Instruction: {training_data[0]['instruction']}")
print(f"Input: {training_data[0]['input'][:150]}...")
print(f"Output: {training_data[0]['output'][:150]}...")
print("-" * 80)
print()

# ============================================================================
# PREPARE DATASET
# ============================================================================

def format_prompt(example):
    """Format example into instruction prompt"""
    return f"""<s>[INST] {example['instruction']}

{example['input']} [/INST]

{example['output']}</s>"""

def tokenize_function(examples):
    """Tokenize examples for training"""
    prompts = [format_prompt(ex) for ex in examples]
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Labels are same as input_ids for causal LM
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

print("Preparing dataset...")

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(training_data)

print(f"Dataset size: {len(dataset)} examples")
print()

# ============================================================================
# LOAD BASE MODEL WITH 4-BIT QUANTIZATION
# ============================================================================

print("=" * 80)
print("LOADING BASE MODEL (4-bit quantization)")
print("=" * 80)
print()

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Double quantization for memory efficiency
    bnb_4bit_quant_type="nf4",  # Normal Float 4
    bnb_4bit_compute_dtype=torch.float16
)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    cache_dir=MODEL_CACHE,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Required for training

print("✅ Tokenizer loaded")
print()

# Load model
print("Loading model with 4-bit quantization...")
print("(This takes ~2-3 minutes)")
start = time.time()

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE,
    trust_remote_code=True
)

print(f"✅ Model loaded in {time.time()-start:.1f}s")
print(f"   Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print()

# ============================================================================
# PREPARE MODEL FOR TRAINING
# ============================================================================

print("Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)

# Configure LoRA
print("Adding LoRA adapters...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj",  # Query projection
        "k_proj",  # Key projection
        "v_proj",  # Value projection
        "o_proj",  # Output projection
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA adapters added")
print(f"   Trainable params: {trainable_params:,} ({trainable_params/all_params*100:.2f}%)")
print(f"   All params: {all_params:,}")
print()

# ============================================================================
# TOKENIZE DATASET
# ============================================================================

print("Tokenizing dataset...")
print("(This may take 5-10 minutes)")
start = time.time()

# Tokenize in batches to avoid memory issues
tokenized_dataset = dataset.map(
    lambda examples: tokenize_function(examples),
    batched=True,
    batch_size=100,
    remove_columns=dataset.column_names
)

print(f"✅ Tokenized in {time.time()-start:.1f}s")
print()

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

print("=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Training hyperparameters
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
    learning_rate=LEARNING_RATE,
    
    # Optimization
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    fp16=True,  # Mixed precision training
    
    # Logging
    logging_steps=10,
    logging_dir=f"{OUTPUT_DIR}/logs",
    
    # Saving
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,  # Keep only 3 checkpoints
    
    # Other
    warmup_steps=50,
    lr_scheduler_type="cosine",
    report_to="none",  # Disable wandb/tensorboard
)

print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * 4})")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Steps per epoch: {len(tokenized_dataset) // (BATCH_SIZE * 4)}")
print(f"Total steps: {(len(tokenized_dataset) // (BATCH_SIZE * 4)) * EPOCHS}")
print(f"Estimated time: {((len(tokenized_dataset) // (BATCH_SIZE * 4)) * EPOCHS * 2) / 3600:.1f} hours")
print()

# ============================================================================
# TRAINING
# ============================================================================

print("=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
start_time = time.time()
trainer.train()
training_time = time.time() - start_time

print()
print("=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"Time: {training_time/3600:.2f} hours")
print()

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================

print("Saving LoRA adapters...")
final_output = f"{OUTPUT_DIR}/final"
os.makedirs(final_output, exist_ok=True)

model.save_pretrained(final_output)
tokenizer.save_pretrained(final_output)

print(f"✅ Saved to {final_output}")
print()

# Check size
adapter_size = sum(os.path.getsize(os.path.join(final_output, f)) 
                   for f in os.listdir(final_output)) / 1024**2
print(f"Adapter size: {adapter_size:.1f} MB")
print()

# ============================================================================
# TEST THE FINE-TUNED MODEL
# ============================================================================

print("=" * 80)
print("TESTING FINE-TUNED MODEL")
print("=" * 80)
print()

# Clear memory
del trainer
torch.cuda.empty_cache()
gc.collect()

# Load base model + adapters
print("Loading fine-tuned model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE
)

finetuned_model = PeftModel.from_pretrained(base_model, final_output)
finetuned_model.eval()

print("✅ Model loaded")
print()

# Test conversation
test_conversation = """دكتور: ازيك؟ في ايه؟
مريض: والله يا دكتور عندي وجع في اللثة وبتنزف لما بغسل سناني
دكتور: ومن امتى وانت حاسس كده؟
مريض: من حوالي اسبوع
دكتور: طيب، واضح ان عندك التهاب في اللثة. هنعمل تنظيف عميق للأسنان"""

test_prompt = f"""<s>[INST] أنت طبيب مساعد. اكتب تقرير SOAP للمحادثة الطبية التالية:

{test_conversation} [/INST]

"""

print("Test input:")
print(test_conversation)
print()

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

print("Generating SOAP note...")
with torch.no_grad():
    outputs = finetuned_model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
soap = result.split("[/INST]")[-1].strip()

print("Generated SOAP Note:")
print("-" * 80)
print(soap)
print("-" * 80)
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("FINE-TUNING SUMMARY")
print("=" * 80)
print(f"✅ Base model: {BASE_MODEL}")
print(f"✅ Training examples: {len(training_data)}")
print(f"✅ Training time: {training_time/3600:.2f} hours")
print(f"✅ Adapter size: {adapter_size:.1f} MB")
print(f"✅ Output location: {final_output}")
print()
print("📥 Download the adapters from Kaggle Output:")
print(f"   {final_output}/")
print()
print("🚀 To use in production:")
print("   1. Download adapter files")
print("   2. Load with: PeftModel.from_pretrained(base_model, './egyptian-medical-lora')")
print("   3. Use exactly like base model, but with better Arabic quality!")
print("=" * 80)
