"""
Create Training JSON: Shifaa + MMedC Only (No AHD)
==================================================

This creates a combined dataset WITHOUT AHD for faster initial training.
You can train on AHD later using incremental training!

Output: training_data_shifaa_mmedc_only.json
"""

import json
import os
from pathlib import Path

print("=" * 80)
print("CREATING SHIFAA + MMEDC COMBINED DATASET (NO AHD)")
print("=" * 80)
print()

# Try to find the data files
all_examples = []
sources = {}

# ============================================================================
# 1. Load Shifaa Dataset
# ============================================================================
print("📥 Looking for Shifaa dataset...")

shifaa_paths = [
    "/kaggle/working/training_data_shifaa.json",
    "/kaggle/input/arabic-medical-data/training_data_shifaa.json",
    "/kaggle/input/ahd-arabic-healthcare-dataset/training_data_shifaa.json",
    "training_data_shifaa.json",
    "../training_data_shifaa.json",
]

shifaa_found = False
for path in shifaa_paths:
    if os.path.exists(path) and os.path.isfile(path):
        print(f"   Found at: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            shifaa_data = json.load(f)
            all_examples.extend(shifaa_data)
            sources['Shifaa'] = len(shifaa_data)
            print(f"   ✅ Loaded {len(shifaa_data):,} examples")
        shifaa_found = True
        break

if not shifaa_found:
    print("   ❌ Shifaa not found!")
    print("   Looking in /kaggle/working for available files...")
    if os.path.exists("/kaggle/working"):
        files = [f for f in os.listdir("/kaggle/working") if f.endswith('.json')]
        if files:
            print(f"   Available JSON files: {files}")
        else:
            print("   No JSON files found in /kaggle/working")

print()

# ============================================================================
# 2. Load MMedC Dataset
# ============================================================================
print("📥 Looking for MMedC dataset...")

mmedc_paths = [
    "/kaggle/working/training_data_mmedc.json",
    "/kaggle/input/arabic-medical-data/training_data_mmedc.json",
    "/kaggle/input/ahd-arabic-healthcare-dataset/training_data_mmedc.json",
    "training_data_mmedc.json",
    "../training_data_mmedc.json",
]

mmedc_found = False
for path in mmedc_paths:
    if os.path.exists(path) and os.path.isfile(path):
        print(f"   Found at: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            mmedc_data = json.load(f)
            all_examples.extend(mmedc_data)
            sources['MMedC'] = len(mmedc_data)
            print(f"   ✅ Loaded {len(mmedc_data):,} examples")
        mmedc_found = True
        break

if not mmedc_found:
    print("   ❌ MMedC not found!")
    print("   Looking in /kaggle/working for available files...")
    if os.path.exists("/kaggle/working"):
        files = [f for f in os.listdir("/kaggle/working") if f.endswith('.json')]
        if files:
            print(f"   Available JSON files: {files}")

print()

# ============================================================================
# 2.5. Fallback: Check for combined file (including FULL_combined)
# ============================================================================
if len(all_examples) == 0:
    print("⚠️  Individual files not found, checking for combined file...")
    
    combined_paths = [
        "/kaggle/working/training_data_FULL_combined.json",  # With AHD
        "/kaggle/working/training_data_all_combined.json",   # Without AHD
        "/kaggle/input/arabic-medical-data/training_data_FULL_combined.json",
        "/kaggle/input/arabic-medical-data/training_data_all_combined.json",
        "training_data_FULL_combined.json",
        "training_data_all_combined.json",
    ]
    
    for path in combined_paths:
        if os.path.exists(path) and os.path.isfile(path):
            print(f"   ✅ Found combined file at: {path}")
            print(f"   Loading and filtering out AHD...")
            
            with open(path, 'r', encoding='utf-8') as f:
                combined_data = json.load(f)
            
            print(f"   Total examples in file: {len(combined_data):,}")
            
            # First, check what source values exist in first 10 examples
            print(f"   Checking source field values...")
            unique_sources = set()
            for i, example in enumerate(combined_data[:10]):
                source_val = example.get('source', 'NO_SOURCE_FIELD')
                unique_sources.add(source_val)
            print(f"   Sample sources found: {unique_sources}")
            print()
            
            # Filter out AHD if it exists (keep only Shifaa and MMedC)
            shifaa_count = 0
            mmedc_count = 0
            ahd_count = 0
            other_count = 0
            
            for example in combined_data:
                source = example.get('source', '').lower()
                
                if 'ahd' in source or source == 'ahd':
                    ahd_count += 1
                    continue  # Skip AHD
                elif 'shifaa' in source:
                    all_examples.append(example)
                    shifaa_count += 1
                elif 'mmedc' in source:
                    all_examples.append(example)
                    mmedc_count += 1
                elif source == '':
                    # No source field - keep it (assume it's Shifaa/MMedC)
                    all_examples.append(example)
                    other_count += 1
                else:
                    # Unknown source - keep it
                    all_examples.append(example)
                    other_count += 1
            
            if shifaa_count > 0:
                sources['Shifaa'] = shifaa_count
            if mmedc_count > 0:
                sources['MMedC'] = mmedc_count
            if other_count > 0:
                sources['Other (non-AHD)'] = other_count
            
            print(f"   ✅ Extracted {len(all_examples):,} non-AHD examples")
            print(f"      Shifaa: {shifaa_count:,}, MMedC: {mmedc_count:,}, Other: {other_count:,}")
            print(f"      Skipped AHD: {ahd_count:,}")
            
            if len(all_examples) == 0 and ahd_count == len(combined_data):
                print()
                print(f"   ⚠️  WARNING: File contains ONLY AHD data!")
                print(f"   This means your FULL_combined file is actually just AHD.")
                print(f"   You need the original Shifaa + MMedC data.")
            
            break
    
    print()

# ============================================================================
# 3. Save Combined Dataset
# ============================================================================
if len(all_examples) == 0:
    print("❌ NO DATA FOUND!")
    print()
    print("Available files in /kaggle/working:")
    if os.path.exists("/kaggle/working"):
        for item in os.listdir("/kaggle/working"):
            print(f"   - {item}")
    print()
    print("Please ensure one of these files exists:")
    print("  - training_data_shifaa.json + training_data_mmedc.json")
    print("  - training_data_all_combined.json")
    print()
    exit(1)

output_file = "training_data_shifaa_mmedc_only.json"

print("=" * 80)
print("SAVING COMBINED DATASET")
print("=" * 80)
print()

print(f"💾 Saving to: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
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

print("📊 Dataset Breakdown:")
for source, count in sources.items():
    percentage = (count / len(all_examples)) * 100
    print(f"   {source}: {count:,} examples ({percentage:.1f}%)")

print()
print(f"✅ Total: {len(all_examples):,} examples")
print(f"✅ Output: {output_file}")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1️⃣  FIRST TRAINING (Shifaa + MMedC):")
print("   - Upload this file to Kaggle")
print("   - Train with this dataset (~2.6 hours)")
print("   - Download the trained LoRA adapters")
print()
print("2️⃣  SECOND TRAINING (Add AHD):")
print("   - Upload your first trained model to Kaggle")
print("   - Use it as base model")
print("   - Train on AHD dataset only")
print("   - Result: Model trained on ALL data!")
print()
print("💡 This is called INCREMENTAL TRAINING!")
print("   Benefits:")
print("   - Split long training into manageable chunks")
print("   - Test first model before full training")
print("   - Save Kaggle GPU hours if first model is good enough")
print()
