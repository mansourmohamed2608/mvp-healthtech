"""
Re-download and Separate Shifaa + MMedC (No AHD)
================================================

This re-downloads the data and keeps only Shifaa + MMedC.
"""

import json
import os

print("=" * 80)
print("DOWNLOADING SHIFAA + MMEDC")
print("=" * 80)
print()

all_examples = []
sources = {}

# ============================================================================
# 1. Download Shifaa Dataset
# ============================================================================
print("📥 Downloading Shifaa dataset from HuggingFace...")

try:
    from datasets import load_dataset
    
    # Load Shifaa dataset
    print("   Loading FreedomIntelligence/medical-o1-reasoning-SFT...")
    shifaa_dataset = load_dataset(
        "FreedomIntelligence/medical-o1-reasoning-SFT",
        "Shifaa",
        split="train"
    )
    
    print(f"   ✅ Downloaded {len(shifaa_dataset):,} Shifaa examples")
    
    # Convert to our format
    for item in shifaa_dataset:
        all_examples.append({
            "input": item["conversations"][0]["value"],
            "output": item["conversations"][1]["value"],
            "source": "Shifaa"
        })
    
    sources['Shifaa'] = len(shifaa_dataset)
    
except Exception as e:
    print(f"   ❌ Error downloading Shifaa: {e}")

print()

# ============================================================================
# 2. Download MMedC Dataset
# ============================================================================
print("📥 Downloading MMedC dataset from HuggingFace...")

try:
    from datasets import load_dataset
    
    # Load MMedC dataset
    print("   Loading Henrychur/MMedC (Arabic subset)...")
    mmedc_dataset = load_dataset(
        "Henrychur/MMedC",
        split="train",
        streaming=False
    )
    
    # Filter for Arabic examples
    arabic_examples = []
    for item in mmedc_dataset:
        text = item.get("text", "")
        # Simple Arabic detection (contains Arabic characters)
        if any('\u0600' <= c <= '\u06FF' for c in text):
            # Extract Q&A if possible
            if len(text) > 50:  # Reasonable length
                arabic_examples.append({
                    "input": text[:len(text)//2],  # First half as question
                    "output": text[len(text)//2:],  # Second half as answer
                    "source": "MMedC"
                })
            
            if len(arabic_examples) >= 200:  # Limit to ~200 examples
                break
    
    print(f"   ✅ Extracted {len(arabic_examples):,} Arabic MMedC examples")
    
    all_examples.extend(arabic_examples)
    sources['MMedC'] = len(arabic_examples)
    
except Exception as e:
    print(f"   ❌ Error downloading MMedC: {e}")
    print(f"   Skipping MMedC - will use Shifaa only")

print()

# ============================================================================
# 3. Check Results
# ============================================================================
if len(all_examples) == 0:
    print("❌ NO DATA DOWNLOADED!")
    print()
    print("Please check your internet connection and try again.")
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
