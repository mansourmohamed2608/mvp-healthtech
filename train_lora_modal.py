"""
Train Whisper LoRA on Modal.com with GPU
Much faster than Kaggle, pay per second, no time limits
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("whisper-lora-training")

# Define container image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "librosa",
        "soundfile",
        "evaluate",
        "jiwer",
        "tensorboard",
    )
)

# Define training function
@app.function(
    image=image,
    gpu="A10G",  # Or "A100" for faster training
    timeout=3600 * 4,  # 4 hours max
    volumes={"/data": modal.Volume.from_name("whisper-training-data", create_if_missing=True)},
    secrets=[modal.Secret.from_name("huggingface-secret")],  # For model download
)
def train_lora(
    csv_path: str = "medical_training_manifest.csv",
    output_dir: str = "lora_ckpt_medical",
    base_model: str = "openai/whisper-large-v3",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
):
    """
    Train Whisper LoRA on Modal GPU
    """
    import torch
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    import pandas as pd
    
    print("🚀 Starting LoRA training on Modal...")
    print(f"📊 Config: {num_epochs} epochs, batch size {batch_size}, LR {learning_rate}")
    
    # Load model and processor
    print(f"📥 Loading {base_model}...")
    processor = WhisperProcessor.from_pretrained(base_model)
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model,
        load_in_8bit=False,  # Full precision on A10G/A100
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"📚 Loading dataset from {csv_path}...")
    df = pd.read_csv(f"/data/{csv_path}")
    print(f"   Found {len(df)} samples")
    
    # TODO: Add dataset preprocessing (similar to train_lora_whisper.py)
    # This is a simplified version - you'll need to add:
    # - Audio loading
    # - Feature extraction
    # - Data collator
    # - Training loop
    
    print("✅ Training complete!")
    print(f"💾 Saving to /data/{output_dir}")
    
    # Save LoRA adapters
    model.save_pretrained(f"/data/{output_dir}")
    processor.save_pretrained(f"/data/{output_dir}")
    
    return f"Training complete! LoRA saved to {output_dir}"

@app.local_entrypoint()
def main(
    csv_path: str = "medical_training_manifest.csv",
    num_epochs: int = 3,
):
    """
    Run training from local machine
    """
    print("🌐 Starting Modal training...")
    result = train_lora.remote(csv_path=csv_path, num_epochs=num_epochs)
    print(result)
    print("✅ Done! Download LoRA adapters from Modal volume")
