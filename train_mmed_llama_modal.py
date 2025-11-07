"""
Train MMed-Llama-3-8B with QLoRA on Modal - 4 Arabic Medical Datasets ONLY
===========================================================================

This script trains your medical LLM on ONLY these 4 datasets:
1. MMedC - Arabic files only (70,024 medical documents)
2. Shifaa Medical Consultations
3. Shifaa Mental Health Consultations
4. AHD - Arabic Healthcare Dataset (XLSX from local/Modal)

Uses QLoRA (Quantized LoRA) for efficient training
GPU Options: T4, L4, A10G, A100-40GB, A100-80GB
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("mmed-llama-qlora-training")

# Define container image with all dependencies
# Versions based on Modal's official unsloth_finetune.py example
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",
        "transformers==4.54.0",      # Latest stable with Llama-3 support
        "peft==0.16.0",               # Latest QLoRA/LoRA implementation
        "datasets==3.6.0",            # HuggingFace datasets
        "accelerate==1.9.0",          # Multi-GPU training support
        "bitsandbytes>=0.41.0",       # For 4-bit quantization
        "trl>=0.7.0",                 # SFTTrainer for supervised fine-tuning
        "scipy",
        "sentencepiece",
        "protobuf",
    )
    .apt_install("git")
)

# Create volume for model and data storage
volume = modal.Volume.from_name("mmed-llama-qlora-training", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",  # GPU options: "T4", "L4", "A10G", "A100", "A100-80GB"
    timeout=3600 * 12,  # 12 hours max
    volumes={"/data": volume},
    memory=40960,  # 40GB RAM
)
def train_mmed_llama_qlora(
    training_data_path: str = "training_data_combined_ALL.json",
    output_dir: str = "mmed_llama_qlora",
    base_model: str = "Henrychur/MMed-Llama-3-8B",
    num_epochs: int = 3,
    batch_size: int = 8,  # Larger batch with QLoRA
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
):
    """
    Train MMed-Llama with QLoRA on combined medical datasets
    QLoRA = 4-bit quantization + LoRA = Efficient training
    """
    import torch
    import json
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    from datasets import Dataset
    
    print("=" * 80)
    print("🚀 MMED-LLAMA QLORA TRAINING ON MODAL")
    print("=" * 80)
    print()
    print(f"📊 Configuration:")
    print(f"   Base model: {base_model}")
    print(f"   Method: QLoRA (4-bit quantization + LoRA)")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: 4 steps")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   Training data: {training_data_path}")
    print()
    
    # Load training data
    print(f"📚 Loading training data from /data/{training_data_path}...")
    with open(f"/data/{training_data_path}", "r", encoding="utf-8") as f:
        training_data = json.load(f)
    
    print(f"   Found {len(training_data):,} examples")
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(training_data)
    print(f"   ✅ Dataset loaded: {len(dataset):,} examples")
    print()
    
    # Load tokenizer
    print(f"📥 Loading tokenizer: {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"   ✅ Tokenizer loaded")
    print()
    
    # 4-bit quantization config (reduces memory)
    print("⚙️  Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    print("   ✅ Quantization configured")
    print()
    
    # Load model
    print(f"📥 Loading base model: {base_model}")
    print("   This may take a few minutes (8B parameters)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    print("   ✅ Model loaded")
    print()
    
    # Prepare for training
    print("🔧 Preparing model for LoRA training...")
    model = prepare_model_for_kbit_training(model)
    
    # QLoRA configuration (optimized for medical domain)
    lora_config = LoraConfig(
        r=32,  # Higher rank for complex medical knowledge
        lora_alpha=64,  # 2x rank
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    print("⚙️  QLoRA Configuration:")
    print(f"   Rank (r): 32")
    print(f"   Alpha: 64")
    print(f"   Target modules: 7 (all attention + FFN)")
    print(f"   Dropout: 0.05")
    print()
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()
    
    # Format data for training
    def format_instruction(example):
        """Format example as instruction-following prompt"""
        text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|><|end_of_text|>"""
        return {"text": text}
    
    print("🔄 Formatting dataset...")
    dataset = dataset.map(format_instruction)
    print(f"   ✅ Formatted {len(dataset):,} examples")
    print()
    
    # Show sample
    print("📝 Sample formatted example:")
    print("-" * 80)
    print(dataset[0]["text"][:500] + "...")
    print("-" * 80)
    print()
    
    # Training arguments (optimized for QLoRA)
    output_path = f"/data/{output_dir}"
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch = 8*4 = 32
        learning_rate=learning_rate,
        fp16=True,
        save_strategy="epoch",
        logging_steps=50,
        warmup_ratio=0.03,  # 3% warmup
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",  # Best for QLoRA
        report_to="none",  # Disable wandb
        save_total_limit=2,
        max_grad_norm=0.3,  # Gradient clipping
        weight_decay=0.001,
        group_by_length=True,  # Efficient batching
    )
    
    print("⚙️  Training Hyperparameters:")
    print(f"   Effective batch size: {batch_size * 4} (batch {batch_size} × accum 4)")
    print(f"   Optimizer: paged_adamw_32bit")
    print(f"   Learning rate: {learning_rate}")
    print(f"   LR scheduler: cosine with 3% warmup")
    print(f"   Gradient clipping: 0.3")
    print()
    
    # Create trainer
    print("🎯 Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
    )
    print("   ✅ Trainer ready")
    print()
    
    # Start training
    print("=" * 80)
    print("🚀 STARTING TRAINING")
    print("=" * 80)
    print()
    
    trainer.train()
    
    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print()
    
    # Save final model
    print(f"💾 Saving LoRA adapters to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("   ✅ LoRA adapters saved")
    print()
    
    print("🎉 Training complete!")
    print(f"📁 QLoRA adapters saved to: {output_path}")
    print()
    print("📊 Training Summary:")
    print(f"   Method: QLoRA (4-bit + LoRA rank 32)")
    print(f"   Trained on: {len(dataset):,} examples")
    print(f"   Epochs: {num_epochs}")
    print(f"   Final checkpoint: {output_path}")
    print()
    print("Next steps:")
    print("1. Download QLoRA adapters from Modal:")
    print(f"   modal volume get mmed-llama-qlora-training {output_dir} ./lora_adapters/")
    print("2. Load in your LLM service with PeftModel")
    print("3. Test with Egyptian medical queries")
    print()
    
    return output_path


@app.local_entrypoint()
def main(
    training_data: str = "training_data_combined_ALL.json",
    epochs: int = 3,
    batch_size: int = 8,
    output_dir: str = "mmed_llama_qlora",
    gpu: str = "A100-40GB",
):
    """
    Run QLoRA training from local machine
    
    Usage:
        modal run train_mmed_llama_modal.py
        
    Or with custom params:
        modal run train_mmed_llama_modal.py --training-data data.json --epochs 5 --gpu A100-80GB
    """
    print("=" * 80)
    print("🌐 MODAL QLORA TRAINING - MMED-LLAMA")
    print("=" * 80)
    print()
    print("📤 Uploading training data to Modal...")
    print()
    
    # Check if data file exists locally
    if not Path(training_data).exists():
        print(f"❌ Error: {training_data} not found!")
        print()
        print("Please run extract_ALL_datasets.py first to generate training data:")
        print("  python extract_ALL_datasets.py")
        print()
        return
    
    # Upload training data to Modal volume
    volume = modal.Volume.from_name("mmed-llama-qlora-training", create_if_missing=True)
    print(f"📤 Uploading {training_data} to Modal...")
    volume.put_file(training_data, training_data)
    print(f"✅ Uploaded: {training_data}")
    print()
    
    # Show file size
    file_size_mb = Path(training_data).stat().st_size / (1024 * 1024)
    print(f"📦 Dataset size: {file_size_mb:.1f} MB")
    print(f"🖥️  GPU: {gpu}")
    print(f"📊 Training config: {epochs} epochs, batch size {batch_size}")
    print()
    
    print("🚀 Starting QLoRA training on Modal GPU...")
    print()
    
    # Run training
    result = train_mmed_llama_qlora.remote(
        training_data_path=training_data,
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=batch_size,
    )
    
    print()
    print("=" * 80)
    print("✅ QLORA TRAINING COMPLETE ON MODAL!")
    print("=" * 80)
    print()
    print(f"📁 QLoRA adapters saved to Modal volume: {result}")
    print()
    print("📥 To download:")
    print(f"   modal volume get mmed-llama-qlora-training {output_dir} ./services/llm/lora_adapters/")
    print()
    print("💰 Cost estimate:")
    print("   A100-40GB: ~$3.50/hour")
    print("   A100-80GB: ~$4.50/hour")
    print()
    print("🎯 QLoRA Benefits:")
    print("   ✅ 4x less memory than full fine-tuning")
    print("   ✅ Same quality as full LoRA")
    print("   ✅ Faster training with larger batches")
    print("   ✅ ~100MB adapters (easy to deploy)")
    print()
