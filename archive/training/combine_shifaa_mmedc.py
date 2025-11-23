"""
Combine Shifaa + MMedC into One JSON File
==========================================

This loads the individual files and combines them.
"""

import json
import os

print("=" * 80)
print("COMBINING SHIFAA + MMEDC")
print("=" * 80)
print()

all_examples = []
sources = {}

# ============================================================================
# 1. Load Shifaa
# ============================================================================
print("📥 Loading Shifaa dataset...")

shifaa_path = "/kaggle/working/training_data_shifaa.json"

if os.path.exists(shifaa_path):
    print(f"   Found: {shifaa_path}")
    with open(shifaa_path, 'r', encoding='utf-8') as f:
        shifaa_data = json.load(f)
        all_examples.extend(shifaa_data)
        sources['Shifaa'] = len(shifaa_data)
        print(f"   ✅ Loaded {len(shifaa_data):,} examples")
else:
    print(f"   ❌ Not found: {shifaa_path}")

print()

# ============================================================================
# 2. Load MMedC
# ============================================================================
print("📥 Loading MMedC dataset...")

mmedc_path = "/kaggle/working/training_data_mmedc.json"

if os.path.exists(mmedc_path):
    print(f"   Found: {mmedc_path}")
    with open(mmedc_path, 'r', encoding='utf-8') as f:
        mmedc_data = json.load(f)
        all_examples.extend(mmedc_data)
        sources['MMedC'] = len(mmedc_data)
        print(f"   ✅ Loaded {len(mmedc_data):,} examples")
else:
    print(f"   ❌ Not found: {mmedc_path}")

print()

# ============================================================================
# 3. Check Results
# ============================================================================
if len(all_examples) == 0:
    print("❌ NO DATA LOADED!")
    print()
    print("Available files in /kaggle/working:")
    if os.path.exists("/kaggle/working"):
        for item in os.listdir("/kaggle/working"):
            if item.endswith('.json'):
                print(f"   - {item}")
    print()
    print("Make sure these files exist:")
    print("   - training_data_shifaa.json")
    print("   - training_data_mmedc.json")
    exit(1)

# ============================================================================
# 4. Save Combined File
# ============================================================================
output_file = "/kaggle/working/training_data_shifaa_mmedc_combined.json"

print("=" * 80)
print("SAVING COMBINED FILE")
print("=" * 80)
print()

print(f"💾 Saving to: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_examples, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(all_examples):,} examples")
print()

# ============================================================================
# 5. Summary
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
print(f"✅ Output: training_data_shifaa_mmedc_combined.json")
print()

# Calculate training time
estimated_hours = (len(all_examples) / 16) * 1.8 / 3600
print(f"⏱️  Estimated training time: ~{estimated_hours:.1f} hours")
print()

print("=" * 80)
print("READY FOR TRAINING!")
print("=" * 80)
print()
print("✅ Use this file in your training script:")
print(f"   {output_file}")
print()
print("This file contains ONLY Shifaa + MMedC (NO AHD)")
print("Perfect for your first quick training run!")
print()
