"""
ONE-CLICK: Download & Extract ALL Training Data
================================================

This script automatically:
1. Downloads ALL datasets (Shifaa, AHD, MMedC)
2. Extracts ALL 70k+ MMedC files (not just 167)
3. Creates separate JSON files for each dataset
4. Creates combined file with ALL data

Total: ~992,000+ training examples!
Cost: $0 (completely free!)
"""

import json
import os
import zipfile
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
from tqdm import tqdm
import re

print("=" * 80)
print("🚀 ONE-CLICK TRAINING DATA PREPARATION")
print("=" * 80)
print()
print("📦 This will download and process:")
print("   1. Shifaa: ~84k medical consultations")
print("   2. AHD: ~808k healthcare Q&A (if available)")
print("   3. MMedC: ALL 70k files → ~100k+ examples")
print()
print("   Total: ~992k+ examples!")
print("=" * 80)
print()

# Output files
OUTPUT_FILES = {
    "shifaa": "training_data_shifaa.json",
    "ahd": "training_data_ahd.json",
    "mmedc": "training_data_mmedc_FULL.json",
    "combined": "training_data_FULL_combined.json"
}

all_examples = []

# ============================================================================
# STEP 1: Download Shifaa
# ============================================================================
print("=" * 80)
print("STEP 1/3: SHIFAA DATASET")
print("=" * 80)
print()

try:
    print("📥 Downloading Shifaa Arabic Medical Consultations...")
    shifaa = load_dataset("Ahmed-Selem/Shifaa_Arabic_Medical_Consultations", split="train")
    print(f"✅ Loaded {len(shifaa):,} consultations")
    print()
    
    shifaa_examples = []
    for i, example in enumerate(tqdm(shifaa, desc="Processing Shifaa")):
        # Find question and answer columns
        question_col = None
        answer_col = None
        
        for col in ['Question', 'question', 'Question Title', 'patient_question', 'Patient', 'patient', 'input', 'query']:
            if col in example:
                question_col = col
                break
        
        for col in ['Answer', 'answer', 'Doctor Answer', 'doctor_answer', 'Doctor', 'doctor', 'output', 'response']:
            if col in example:
                answer_col = col
                break
        
        if question_col and answer_col and example[question_col] and example[answer_col]:
            shifaa_examples.append({
                "input": str(example[question_col]).strip(),
                "output": str(example[answer_col]).strip(),
                "source": "shifaa"
            })
    
    with open(OUTPUT_FILES["shifaa"], "w", encoding="utf-8") as f:
        json.dump(shifaa_examples, f, ensure_ascii=False, indent=2)
    
    all_examples.extend(shifaa_examples)
    
    print(f"✅ Shifaa: {len(shifaa_examples):,} examples")
    print(f"   Saved to: {OUTPUT_FILES['shifaa']}")
    print()

except Exception as e:
    print(f"⚠️  Shifaa error: {e}")
    print()

# ============================================================================
# STEP 2: Download AHD (Optional)
# ============================================================================
print("=" * 80)
print("STEP 2/3: AHD DATASET")
print("=" * 80)
print()

try:
    ahd_paths = ["ahd_dataset.xlsx", "AHD.xlsx", "/kaggle/input/ahd-dataset/AHD.xlsx"]
    
    ahd_path = None
    for path in ahd_paths:
        if os.path.exists(path):
            ahd_path = path
            break
    
    if ahd_path:
        print(f"✅ Found AHD at: {ahd_path}")
        ahd = pd.read_excel(ahd_path)
        print(f"   Loaded {len(ahd):,} records")
        print()
        
        ahd_examples = []
        for i, row in tqdm(ahd.iterrows(), total=len(ahd), desc="Processing AHD"):
            if pd.notna(row.get('question')) and pd.notna(row.get('answer')):
                ahd_examples.append({
                    "input": str(row['question']).strip(),
                    "output": str(row['answer']).strip(),
                    "source": "ahd"
                })
        
        with open(OUTPUT_FILES["ahd"], "w", encoding="utf-8") as f:
            json.dump(ahd_examples, f, ensure_ascii=False, indent=2)
        
        all_examples.extend(ahd_examples)
        
        print(f"✅ AHD: {len(ahd_examples):,} examples")
        print(f"   Saved to: {OUTPUT_FILES['ahd']}")
        print()
    else:
        print("ℹ️  AHD not found (optional)")
        print("   Download from: https://data.mendeley.com/datasets/mgj29ndgrk/5")
        print("   Place as 'ahd_dataset.xlsx' and re-run")
        print()

except Exception as e:
    print(f"ℹ️  AHD skipped: {e}")
    print()

# ============================================================================
# STEP 3: Download & Extract ALL MMedC
# ============================================================================
print("=" * 80)
print("STEP 3/3: MMEDC FULL EXTRACTION")
print("=" * 80)
print()

try:
    # Download Arabic.zip
    zip_path = "Arabic.zip"
    
    if not os.path.exists(zip_path):
        print("📥 Downloading MMedC Arabic.zip (1.28 GB)...")
        print("   This may take several minutes...")
        print()
        
        zip_path = hf_hub_download(
            repo_id="Henrychur/MMedC",
            filename="Arabic.zip",
            repo_type="dataset"
        )
        print(f"✅ Downloaded to: {zip_path}")
        print()
    else:
        print(f"✅ Using existing: {zip_path}")
        print()
    
    # Extract ALL files
    print("🔄 Extracting ALL 70,024 MMedC files...")
    print("   Converting to training format...")
    print()
    
    mmedc_examples = []
    
    def clean_text(text):
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
        print(f"   Found {len(txt_files):,} text files")
        print()
        
        for filename in tqdm(txt_files, desc="Processing MMedC"):
            try:
                with zip_ref.open(filename) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                
                content = clean_text(content)
                
                if len(content) < 100:
                    continue
                
                # Chunk long documents
                chunk_size = 1500
                
                if len(content) > chunk_size:
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i+chunk_size+200]
                        
                        if len(chunk) >= 100:
                            mmedc_examples.append({
                                "input": "تعلم المعلومات الطبية التالية:",
                                "output": chunk,
                                "source": "mmedc"
                            })
                else:
                    mmedc_examples.append({
                        "input": "تعلم المعلومات الطبية التالية:",
                        "output": content,
                        "source": "mmedc"
                    })
            
            except:
                continue
    
    with open(OUTPUT_FILES["mmedc"], "w", encoding="utf-8") as f:
        json.dump(mmedc_examples, f, ensure_ascii=False, indent=2)
    
    all_examples.extend(mmedc_examples)
    
    file_size_mb = os.path.getsize(OUTPUT_FILES["mmedc"]) / (1024 * 1024)
    
    print()
    print(f"✅ MMedC: {len(mmedc_examples):,} examples")
    print(f"   Saved to: {OUTPUT_FILES['mmedc']}")
    print(f"   File size: {file_size_mb:.1f} MB")
    print()

except Exception as e:
    print(f"⚠️  MMedC error: {e}")
    print()

# ============================================================================
# SAVE COMBINED FILE
# ============================================================================
print("=" * 80)
print("CREATING COMBINED FILE")
print("=" * 80)
print()

with open(OUTPUT_FILES["combined"], "w", encoding="utf-8") as f:
    json.dump(all_examples, f, ensure_ascii=False, indent=2)

combined_size_mb = os.path.getsize(OUTPUT_FILES["combined"]) / (1024 * 1024)

print(f"✅ Combined: {len(all_examples):,} examples")
print(f"   Saved to: {OUTPUT_FILES['combined']}")
print(f"   File size: {combined_size_mb:.1f} MB")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("🎉 COMPLETE - ALL DATA READY!")
print("=" * 80)
print()

# Count by source
sources = {}
for ex in all_examples:
    source = ex.get('source', 'unknown')
    sources[source] = sources.get(source, 0) + 1

print("📊 BREAKDOWN BY SOURCE:")
print()
for source, count in sorted(sources.items()):
    print(f"   {source:10s}: {count:>8,} examples")
print("   " + "-" * 30)
print(f"   {'TOTAL':10s}: {len(all_examples):>8,} examples")
print()

print("📁 FILES CREATED:")
print()
for key, filename in OUTPUT_FILES.items():
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"   ✅ {filename:35s} ({size_mb:>6.1f} MB)")
print()

print("⏱️  ESTIMATED TRAINING TIMES:")
print()
if "shifaa" in sources:
    time_shifaa = sources['shifaa'] / 16 / 1.8 / 3600
    print(f"   Phase 1 (Shifaa):     {sources['shifaa']:>8,} examples → {time_shifaa:>5.1f} hours")

if "mmedc" in sources:
    time_mmedc = sources['mmedc'] / 16 / 1.8 / 3600
    print(f"   Phase 1 (MMedC):      {sources['mmedc']:>8,} examples → {time_mmedc:>5.1f} hours")

if "ahd" in sources:
    time_ahd = sources['ahd'] / 16 / 1.8 / 3600
    print(f"   Phase 2 (AHD):        {sources['ahd']:>8,} examples → {time_ahd:>5.1f} hours")

total_time = len(all_examples) / 16 / 1.8 / 3600
print(f"   {'─' * 45}")
print(f"   Total (all phases):   {len(all_examples):>8,} examples → {total_time:>5.1f} hours")
print()

print("🚀 NEXT STEPS:")
print()
print("   1. Upload JSON files to Kaggle as datasets")
print("   2. Follow 3-phase training strategy:")
print("      - Phase 1: training_data_mmedc_FULL.json")
print("      - Phase 2: training_data_shifaa.json")
print("      - Phase 3: training_data_ahd.json")
print("   3. Or use training_data_FULL_combined.json for all-at-once")
print()
print("💰 Total Cost: $0 (FREE!)")
print("🎯 Ready for training!")
print()
