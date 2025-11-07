"""
Simple: Just use the existing training_data_all_combined.json
=============================================================

You already have training_data_all_combined.json in /kaggle/working
which contains Shifaa + MMedC (no AHD).

This script just copies/renames it for clarity.
"""

import json
import os
import shutil

print("=" * 80)
print("USING EXISTING SHIFAA + MMEDC COMBINED FILE")
print("=" * 80)
print()

# The file that already exists
source_file = "/kaggle/working/training_data_all_combined.json"
output_file = "/kaggle/working/training_data_shifaa_mmedc_ready.json"

if not os.path.exists(source_file):
    print(f"❌ File not found: {source_file}")
    print()
    print("Available files in /kaggle/working:")
    if os.path.exists("/kaggle/working"):
        for item in os.listdir("/kaggle/working"):
            if item.endswith('.json'):
                print(f"   - {item}")
    exit(1)

print(f"📥 Loading: {source_file}")
with open(source_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"   ✅ Loaded {len(data):,} examples")
print()

# Check for sources
sources = {}
for example in data:
    source = example.get('source', 'Unknown')
    sources[source] = sources.get(source, 0) + 1

print("📊 Dataset Breakdown:")
for source, count in sources.items():
    percentage = (count / len(data)) * 100
    print(f"   {source}: {count:,} examples ({percentage:.1f}%)")
print()

# Save with new name for clarity
print(f"💾 Saving as: {output_file}")
shutil.copy(source_file, output_file)
print(f"✅ Saved {len(data):,} examples")
print()

print("=" * 80)
print("READY TO TRAIN!")
print("=" * 80)
print()
print(f"✅ Use this file for training: {output_file}")
print(f"✅ Total: {len(data):,} examples")
print(f"✅ Estimated training time: ~{len(data) * 1.8 / 3600 / 16:.1f} hours")
print()
print("This file contains Shifaa + MMedC (no AHD)")
print("Perfect for your first training run!")
print()
