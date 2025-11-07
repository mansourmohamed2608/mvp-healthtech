"""
Combine ALL Training Datasets (Shifaa + MMedC + AHD)
====================================================

This script combines all available datasets into one file for training.
Run this in Kaggle to create the complete combined dataset.

Datasets:
1. Shifaa: 84,422 examples ✅
2. MMedC: 167 examples ✅
3. AHD: 808,000+ examples (if available in Kaggle input)

Total: ~893,000 examples
"""

import json
import os
import pandas as pd

print("=" * 80)
print("COMBINING ALL TRAINING DATASETS")
print("=" * 80)
print()

# Try to load all available datasets
all_examples = []
sources = {}

# ============================================================================
# 1. Load Shifaa + MMedC (already combined)
# ============================================================================
shifaa_mmedc_paths = [
    "/kaggle/working/training_data_all_combined.json",  # Kaggle working dir
    "training_data_all_combined.json",  # Local
]

shifaa_mmedc_found = False
for path in shifaa_mmedc_paths:
    if os.path.exists(path):
        print(f"📥 Loading Shifaa + MMedC from: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            shifaa_mmedc = json.load(f)
            all_examples.extend(shifaa_mmedc)
            sources['Shifaa + MMedC'] = len(shifaa_mmedc)
            print(f"   ✅ Loaded {len(shifaa_mmedc):,} examples")
        shifaa_mmedc_found = True
        break

if not shifaa_mmedc_found:
    print("⚠️  Shifaa + MMedC combined file not found!")
    print("   Looking for individual files...")
    
    # Try individual files
    individual_paths = {
        "Shifaa": [
            "training_data_shifaa.json",
            "/kaggle/working/training_data_shifaa.json"
        ],
        "MMedC": [
            "training_data_mmedc.json",
            "/kaggle/working/training_data_mmedc.json"
        ]
    }
    
    for source_name, paths in individual_paths.items():
        for path in paths:
            if os.path.exists(path):
                print(f"📥 Loading {source_name} from: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    examples = json.load(f)
                    all_examples.extend(examples)
                    sources[source_name] = len(examples)
                    print(f"   ✅ Loaded {len(examples):,} examples")
                break

print()

# ============================================================================
# 2. Load AHD Dataset (if available)
# ============================================================================
print("📥 Looking for AHD dataset...")

ahd_paths = [
    "/kaggle/input/ahd-dataset/AHD.xlsx",
    "/kaggle/input/ahd-dataset/AHD_arabic.xlsx",
    "/kaggle/input/arabic-healthcare-dataset/AHD.xlsx",
    "../input/ahd-dataset/AHD.xlsx",
    "ahd_dataset.xlsx",
    "AHD.xlsx",
]

ahd_found = False
for path in ahd_paths:
    if os.path.exists(path):
        print(f"   Found at: {path}")
        print("   Loading... (this may take a minute)")
        
        try:
            # Load Excel file
            df = pd.read_excel(path)
            print(f"   ✅ Loaded {len(df):,} rows from AHD")
            
            # Convert to training format
            # AHD format: columns might be 'question'/'answer' or 'query'/'response'
            # Need to check actual column names
            print(f"   Columns: {list(df.columns)}")
            
            # Try different column name combinations
            question_col = None
            answer_col = None
            
            for q_name in ['question', 'Question', 'query', 'Query', 'سؤال', 'Q']:
                if q_name in df.columns:
                    question_col = q_name
                    break
            
            for a_name in ['answer', 'Answer', 'response', 'Response', 'إجابة', 'A']:
                if a_name in df.columns:
                    answer_col = a_name
                    break
            
            if question_col and answer_col:
                ahd_count = 0
                for _, row in df.iterrows():
                    question = str(row[question_col]).strip()
                    answer = str(row[answer_col]).strip()
                    
                    # Skip empty or invalid entries
                    if question and answer and question != 'nan' and answer != 'nan':
                        all_examples.append({
                            "input": question,
                            "output": answer,
                            "source": "AHD"
                        })
                        ahd_count += 1
                
                sources['AHD'] = ahd_count
                print(f"   ✅ Converted {ahd_count:,} valid examples from AHD")
                ahd_found = True
                break
            else:
                print(f"   ⚠️  Could not identify question/answer columns")
                print(f"       Please check the Excel file format")
                
        except Exception as e:
            print(f"   ❌ Error loading AHD: {e}")
            continue

if not ahd_found:
    print("   ⚠️  AHD dataset not found in any expected location")
    print()
    print("   To include AHD dataset:")
    print("   1. Go to Kaggle")
    print("   2. Add dataset: Search 'AHD Arabic Healthcare Dataset'")
    print("   3. Or upload your AHD.xlsx file as Kaggle dataset")
    print("   4. Re-run this script")

print()

# ============================================================================
# 3. Save Combined Dataset
# ============================================================================
print("=" * 80)
print("SAVING COMBINED DATASET")
print("=" * 80)
print()

if len(all_examples) == 0:
    print("❌ No training data found!")
    print("   Please ensure at least one dataset is available.")
    exit(1)

# Save to working directory (Kaggle) or current directory (local)
output_path = "/kaggle/working/training_data_FULL_combined.json"
if not os.path.exists("/kaggle/working"):
    output_path = "training_data_FULL_combined.json"

print(f"💾 Saving to: {output_path}")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_examples, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(all_examples):,} examples")
print()

# ============================================================================
# 4. Summary
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print("📊 Dataset Sources:")
for source, count in sources.items():
    percentage = (count / len(all_examples)) * 100
    print(f"   {source}: {count:,} examples ({percentage:.1f}%)")

print()
print(f"✅ Total training examples: {len(all_examples):,}")
print(f"✅ Output file: {output_path}")
print()

if 'AHD' not in sources:
    print("⚠️  AHD dataset NOT included (808k+ examples missing)")
    print("   Add AHD to Kaggle inputs to get the full ~893k dataset")
else:
    print("🎉 ALL datasets included!")

print()
print("=" * 80)
print("NEXT STEP")
print("=" * 80)
print()
print("Update your training script to use:")
print(f'   "{output_path}"')
print()
print("Then run the training!")
print()
