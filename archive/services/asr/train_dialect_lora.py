"""
Dialect-Specific Training Script for Whisper LoRA Adapters
Week 5 Day 32 (Oct 26, 2025)
Train separate adapters for Egyptian, Levantine, and Gulf Arabic

Usage:
    python train_dialect_lora.py --dialect egyptian --data_dir data/dialects/egyptian
    python train_dialect_lora.py --dialect levantine --data_dir data/dialects/levantine
    python train_dialect_lora.py --dialect gulf --data_dir data/dialects/gulf
"""
import argparse
import os
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, TaskType
import torch


def prepare_dataset(batch, processor):
    """Prepare audio and text for training"""
    audio = batch["audio"]
    
    # Compute input features
    batch["input_features"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]
    
    # Encode target text
    batch["labels"] = processor.tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=448
    ).input_ids
    
    return batch


def train_dialect_adapter(
    dialect: str,
    data_dir: str,
    output_dir: str = "./lora_ckpt",
    base_model: str = "openai/whisper-large-v2",
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 3e-4
):
    """Train a LoRA adapter for a specific Arabic dialect"""
    
    print(f"\n{'='*60}")
    print(f"Training {dialect.upper()} Arabic Adapter")
    print(f"{'='*60}\n")
    
    # Load processor and model
    print(f"Loading base model: {base_model}")
    processor = WhisperProcessor.from_pretrained(base_model)
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=32,  # LoRA rank
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],  # Attention layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dialect-specific dataset
    print(f"Loading {dialect} dataset from {data_dir}")
    try:
        dataset = load_dataset(
            "audiofolder",
            data_dir=data_dir,
            split="train"
        )
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        # Split into train/validation
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        
        print(f"Train examples: {len(train_dataset)}")
        print(f"Eval examples: {len(eval_dataset)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nExpected directory structure:")
        print(f"{data_dir}/")
        print("  ├── audio/")
        print("  │   ├── audio1.wav")
        print("  │   ├── audio2.wav")
        print("  │   └── ...")
        print("  └── metadata.csv (columns: file_name, sentence)")
        return
    
    # Prepare datasets
    train_dataset = train_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=eval_dataset.column_names
    )
    
    # Training arguments
    dialect_output_dir = os.path.join(output_dir, dialect[:3])
    training_args = Seq2SeqTrainingArguments(
        output_dir=dialect_output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )
    
    # Train
    print(f"\nStarting training for {num_epochs} epochs...")
    trainer.train()
    
    # Save final adapter
    print(f"\nSaving adapter to {dialect_output_dir}")
    model.save_pretrained(dialect_output_dir)
    processor.save_pretrained(dialect_output_dir)
    
    print(f"\n✅ Training complete for {dialect} dialect!")
    print(f"Adapter saved to: {dialect_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train dialect-specific Whisper LoRA adapters")
    parser.add_argument(
        "--dialect",
        type=str,
        required=True,
        choices=["egyptian", "levantine", "gulf", "msa"],
        help="Arabic dialect to train on"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dialect-specific audio dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_ckpt",
        help="Directory to save adapters"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="openai/whisper-large-v2",
        help="Base Whisper model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per device"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    train_dialect_adapter(
        dialect=args.dialect,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
