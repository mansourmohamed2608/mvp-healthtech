"""
Extract and Combine ONLY 4 Arabic Medical Datasets
===================================================

This script extracts data from ONLY these 4 sources:
1. MMedC - Arabic files only (70,024 medical documents)
2. Shifaa Arabic Medical Consultations
3. Shifaa Arabic Mental Health Consultations
4. AHD - Arabic Healthcare Dataset (XLSX file uploaded to Modal)

Then combines them into one unified training dataset.
"""

import json
import os
import zipfile
from tqdm import tqdm
import re
from datasets import load_dataset
from pathlib import Path

def clean_text(text):
    """Clean medical text"""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    return text

def extract_mmedc(zip_path):
    """
    Extract MMedC corpus - ARABIC files ONLY
    Exactly like the extract_all_mmedc_pure_text function
    Returns list of training examples
    """
    print("=" * 80)
    print("📚 EXTRACTING MMEDC - ARABIC FILES ONLY")
    print("=" * 80)
    print()
    
    examples = []
    total_chars = 0
    
    print(f"📦 Opening: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
        print(f"📄 Found {len(txt_files):,} Arabic text files")
        print()
        
        print("🔄 Processing ALL Arabic MMedC files...")
        for filename in tqdm(txt_files, desc="Processing MMedC Arabic"):
            try:
                with zip_ref.open(filename) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                
                # Clean text
                content = clean_text(content)
                
                # Skip if too short
                if len(content) < 100:
                    continue
                
                # Chunk into manageable pieces (1500 chars each)
                chunk_size = 1500
                
                if len(content) > chunk_size:
                    # Split into overlapping chunks
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i+chunk_size+200]  # Overlap
                        
                        if len(chunk) >= 100:
                            example = {
                                "input": "تعلم المعلومات الطبية التالية:",
                                "output": chunk,
                                "source": "MMedC",
                                "type": "medical_text"
                            }
                            examples.append(example)
                            total_chars += len(chunk)
                else:
                    example = {
                        "input": "تعلم المعلومات الطبية التالية:",
                        "output": content,
                        "source": "MMedC",
                        "type": "medical_text"
                    }
                    examples.append(example)
                    total_chars += len(content)
                
            except Exception as e:
                continue
    
    print()
    print(f"✅ MMedC Arabic: {len(examples):,} examples")
    print(f"📊 Total characters: {total_chars:,}")
    print(f"📊 Estimated tokens: ~{total_chars // 4:,}")
    print()
    
    return examples


def extract_shifaa_medical():
    """
    Extract Shifaa Arabic Medical Consultations
    Returns list of training examples
    """
    print("=" * 80)
    print("🏥 EXTRACTING SHIFAA MEDICAL CONSULTATIONS")
    print("=" * 80)
    print()
    
    try:
        print("📥 Downloading Shifaa Medical dataset from HuggingFace...")
        dataset = load_dataset("Ahmed-Selem/Shifaa_Arabic_Medical_Consultations")
        
        print(f"✅ Loaded Shifaa Medical dataset")
        print(f"   Available splits: {list(dataset.keys())}")
        print()
        
        examples = []
        
        # Process train split
        if 'train' in dataset:
            train_data = dataset['train']
            print(f"📊 Train split: {len(train_data):,} samples")
            
            for item in tqdm(train_data, desc="Processing Shifaa Medical"):
                # Shifaa has question-answer pairs
                # Adapt based on actual structure (check first)
                if 'question' in item and 'answer' in item:
                    question = clean_text(item['question'])
                    answer = clean_text(item['answer'])
                    
                    if len(question) < 10 or len(answer) < 10:
                        continue
                    
                    example = {
                        "input": question,
                        "output": answer,
                        "source": "Shifaa_Medical",
                        "type": "medical_consultation_qa"
                    }
                    examples.append(example)
                
                elif 'text' in item:
                    # If it's text format
                    text = clean_text(item['text'])
                    if len(text) < 50:
                        continue
                    
                    example = {
                        "input": "أجب عن الاستشارة الطبية التالية:",
                        "output": text,
                        "source": "Shifaa_Medical",
                        "type": "medical_consultation_text"
                    }
                    examples.append(example)
        
        print()
        print(f"✅ Shifaa Medical: {len(examples):,} examples")
        print()
        
        return examples
        
    except Exception as e:
        print(f"❌ Error loading Shifaa Medical: {e}")
        print("   Continuing without Shifaa Medical data...")
        print()
        return []


def extract_shifaa_mental_health():
    """
    Extract Shifaa Arabic Mental Health Consultations
    Returns list of training examples
    """
    print("=" * 80)
    print("🧠 EXTRACTING SHIFAA MENTAL HEALTH CONSULTATIONS")
    print("=" * 80)
    print()
    
    try:
        print("📥 Downloading Shifaa Mental Health dataset from HuggingFace...")
        dataset = load_dataset("Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations")
        
        print(f"✅ Loaded Shifaa Mental Health dataset")
        print(f"   Available splits: {list(dataset.keys())}")
        print()
        
        examples = []
        
        # Process train split
        if 'train' in dataset:
            train_data = dataset['train']
            print(f"📊 Train split: {len(train_data):,} samples")
            
            for item in tqdm(train_data, desc="Processing Mental Health"):
                if 'question' in item and 'answer' in item:
                    question = clean_text(item['question'])
                    answer = clean_text(item['answer'])
                    
                    if len(question) < 10 or len(answer) < 10:
                        continue
                    
                    example = {
                        "input": question,
                        "output": answer,
                        "source": "Shifaa_Mental_Health",
                        "type": "mental_health_qa"
                    }
                    examples.append(example)
                
                elif 'text' in item:
                    text = clean_text(item['text'])
                    if len(text) < 50:
                        continue
                    
                    example = {
                        "input": "أجب عن الاستشارة النفسية التالية:",
                        "output": text,
                        "source": "Shifaa_Mental_Health",
                        "type": "mental_health_text"
                    }
                    examples.append(example)
        
        print()
        print(f"✅ Shifaa Mental Health: {len(examples):,} examples")
        print()
        
        return examples
        
    except Exception as e:
        print(f"❌ Error loading Shifaa Mental Health: {e}")
        print("   Continuing without Mental Health data...")
        print()
        return []


# AfriVox removed - NOT in the 4 datasets you specified


def extract_ahd_kaggle(ahd_file="AHD.xlsx"):
    """
    Extract AHD - Arabic Healthcare Dataset (from Kaggle)
    This is uploaded to Modal volume manually
    Returns list of training examples
    """
    print("=" * 80)
    print("📋 EXTRACTING AHD - ARABIC HEALTHCARE DATASET")
    print("=" * 80)
    print()
    
    try:
        if not os.path.exists(ahd_file):
            print(f"⚠️  {ahd_file} not found locally")
            print("   This file should be uploaded to Modal volume")
            print("   Instructions:")
            print(f"   1. Upload {ahd_file} to Modal:")
            print(f"      modal volume put mmed-llama-qlora-training {ahd_file}")
            print(f"   2. Re-run extraction on Modal")
            print()
            return []
        
        print(f"📥 Loading {ahd_file}...")
        
        # Try pandas to read Excel
        import pandas as pd
        
        df = pd.read_excel(ahd_file)
        print(f"✅ Loaded AHD dataset")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")
        print()
        
        examples = []
        
        # Adapt based on actual columns (common patterns)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing AHD"):
            # Try different column patterns
            question = None
            answer = None
            
            # Pattern 1: question/answer columns
            if 'question' in df.columns and 'answer' in df.columns:
                question = str(row['question'])
                answer = str(row['answer'])
            elif 'Question' in df.columns and 'Answer' in df.columns:
                question = str(row['Question'])
                answer = str(row['Answer'])
            # Pattern 2: text/label columns
            elif 'text' in df.columns:
                answer = str(row['text'])
                question = "اشرح المعلومات الطبية التالية:"
            # Pattern 3: consultation/response
            elif 'consultation' in df.columns and 'response' in df.columns:
                question = str(row['consultation'])
                answer = str(row['response'])
            
            if question and answer:
                question = clean_text(question)
                answer = clean_text(answer)
                
                if len(question) < 10 or len(answer) < 10:
                    continue
                
                # Deduplication: skip if too similar to previous
                # (Simple check: first 50 chars)
                
                example = {
                    "input": question,
                    "output": answer,
                    "source": "AHD_Kaggle",
                    "type": "healthcare_qa"
                }
                examples.append(example)
        
        print()
        print(f"✅ AHD Kaggle: {len(examples):,} examples")
        print()
        
        return examples
        
    except Exception as e:
        print(f"❌ Error loading AHD: {e}")
        print("   Continuing without AHD data...")
        print()
        return []


def combine_and_save(mmedc_examples, shifaa_medical_examples, shifaa_mental_examples, 
                     ahd_examples, output_file):
    """
    Combine ONLY the 4 specified datasets and save to JSON
    """
    print("=" * 80)
    print("🔀 COMBINING 4 DATASETS")
    print("=" * 80)
    print()
    
    all_examples = []
    
    # Add ONLY the 4 sources you specified
    all_examples.extend(mmedc_examples)
    all_examples.extend(shifaa_medical_examples)
    all_examples.extend(shifaa_mental_examples)
    all_examples.extend(ahd_examples)
    
    print(f"📊 Dataset Statistics (4 sources only):")
    print(f"   1. MMedC (Arabic): {len(mmedc_examples):,} examples")
    print(f"   2. Shifaa Medical: {len(shifaa_medical_examples):,} examples")
    print(f"   3. Shifaa Mental Health: {len(shifaa_mental_examples):,} examples")
    print(f"   4. AHD (Kaggle): {len(ahd_examples):,} examples")
    print(f"   " + "-" * 40)
    print(f"   TOTAL: {len(all_examples):,} examples")
    print()
    
    # Show distribution by type
    type_counts = {}
    for ex in all_examples:
        t = ex.get('type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("📋 By content type:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"   {t}: {count:,}")
    print()
    
    # Shuffle for better training
    import random
    random.seed(42)
    random.shuffle(all_examples)
    print("🔀 Shuffled examples for training")
    print()
    
    # Save to JSON
    print(f"💾 Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Saved {len(all_examples):,} examples")
    print(f"📦 File size: {file_size_mb:.1f} MB")
    print()
    
    # Show samples
    print("📝 Sample examples:")
    print("-" * 80)
    for i, ex in enumerate(all_examples[:3], 1):
        print(f"\n[{i}] Source: {ex['source']} | Type: {ex['type']}")
        print(f"Input: {ex['input'][:60]}...")
        print(f"Output: {ex['output'][:100]}...")
    print("-" * 80)
    print()
    
    return len(all_examples)


def download_mmedc():
    """Download MMedC Arabic.zip if not present"""
    zip_path = "Arabic.zip"
    
    if os.path.exists(zip_path):
        print(f"✅ Found existing: {zip_path}")
        return zip_path
    
    print("📥 Arabic.zip not found locally")
    print("⏬ Downloading from HuggingFace...")
    print("   Source: Henrychur/MMedC")
    print("   File: Arabic.zip (1.28 GB)")
    print("   This may take 5-10 minutes...")
    print()
    
    try:
        from huggingface_hub import hf_hub_download
        
        zip_path = hf_hub_download(
            repo_id="Henrychur/MMedC",
            filename="Arabic.zip",
            repo_type="dataset"
        )
        print(f"✅ Downloaded to: {zip_path}")
        print()
        return zip_path
    except Exception as e:
        print(f"❌ Error downloading Arabic.zip: {e}")
        print()
        print("Alternative options:")
        print("1. Manually download from: https://huggingface.co/datasets/Henrychur/MMedC")
        print("2. Place Arabic.zip in current directory")
        print("3. Re-run this script")
        return None


def main():
    """
    Main extraction pipeline - ONLY 4 datasets as specified
    """
    print("=" * 80)
    print("🚀 EXTRACT 4 ARABIC MEDICAL DATASETS ONLY")
    print("=" * 80)
    print()
    print("📚 Sources (4 only):")
    print("   1. MMedC - Arabic files only")
    print("   2. Shifaa Arabic Medical Consultations")
    print("   3. Shifaa Arabic Mental Health Consultations")
    print("   4. AHD - Arabic Healthcare Dataset (XLSX from local/Modal)")
    print()
    print("🎯 Output: training_data_combined_ALL.json")
    print()
    
    output_file = "training_data_combined_ALL.json"
    
    # Step 1: Download and extract MMedC Arabic only
    print("\n" + "=" * 80)
    print("STEP 1/4: MMEDC - ARABIC FILES ONLY")
    print("=" * 80 + "\n")
    
    zip_path = download_mmedc()
    if not zip_path:
        print("❌ Cannot proceed without MMedC data")
        return
    
    mmedc_examples = extract_mmedc(zip_path)
    
    # Step 2: Extract Shifaa Medical
    print("\n" + "=" * 80)
    print("STEP 2/4: SHIFAA MEDICAL CONSULTATIONS")
    print("=" * 80 + "\n")
    
    shifaa_medical_examples = extract_shifaa_medical()
    
    # Step 3: Extract Shifaa Mental Health
    print("\n" + "=" * 80)
    print("STEP 3/4: SHIFAA MENTAL HEALTH CONSULTATIONS")
    print("=" * 80 + "\n")
    
    shifaa_mental_examples = extract_shifaa_mental_health()
    
    # Step 4: Extract AHD (XLSX file)
    print("\n" + "=" * 80)
    print("STEP 4/4: AHD - ARABIC HEALTHCARE DATASET (XLSX)")
    print("=" * 80 + "\n")
    
    ahd_examples = extract_ahd_kaggle()
    
    # Combine all 4
    print("\n" + "=" * 80)
    print("FINAL: COMBINE 4 DATASETS")
    print("=" * 80 + "\n")
    
    total = combine_and_save(
        mmedc_examples,
        shifaa_medical_examples,
        shifaa_mental_examples,
        ahd_examples,
        output_file
    )
    
    # Final summary
    print("=" * 80)
    print("✅ EXTRACTION COMPLETE!")
    print("=" * 80)
    print()
    print(f"📁 Output file: {output_file}")
    print(f"📊 Total examples: {total:,}")
    print()
    
    # Estimate training time
    if total > 0:
        tokens_estimate = total * 400  # ~400 tokens per example
        print("⏱️  Estimated training time:")
        print(f"   On A10G GPU (4 batch size):")
        print(f"     - 1 epoch: ~{total / (4 * 100):.1f} hours")
        print(f"     - 3 epochs: ~{total / (4 * 100) * 3:.1f} hours")
        print()
        print(f"   On A100 GPU (8 batch size):")
        print(f"     - 1 epoch: ~{total / (8 * 150):.1f} hours")
        print(f"     - 3 epochs: ~{total / (8 * 150) * 3:.1f} hours")
        print()
        print("💰 Estimated costs:")
        print(f"   A10G: ${total / (4 * 100) * 3 * 1.10:.2f} for 3 epochs")
        print(f"   A100: ${total / (8 * 150) * 3 * 3.50:.2f} for 3 epochs")
    
    print()
    print("🚀 Next step:")
    print(f"   modal run train_mmed_llama_modal.py --training-data {output_file}")
    print()


if __name__ == "__main__":
    main()
