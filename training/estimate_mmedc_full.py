"""
Quick Estimate: How many examples will we get from ALL MMedC?
==============================================================
"""

import zipfile
import os
from tqdm import tqdm

zip_path = "Arabic.zip"

print("=" * 80)
print("MMEDC EXTRACTION ESTIMATE")
print("=" * 80)
print()

# Download if not exists
if not os.path.exists(zip_path):
    print("📥 Arabic.zip not found locally")
    print("⏬ Downloading from HuggingFace...")
    print("   This may take several minutes (1.28 GB)...")
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
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        print("Please download manually and place in current directory")
        exit(1)

try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
        print(f"📄 Total files: {len(txt_files):,}")
        print()
        
        # Sample first 1000 files to estimate
        print("🔍 Sampling 1000 files to estimate...")
        total_chars = 0
        valid_files = 0
        
        for filename in tqdm(txt_files[:1000], desc="Sampling"):
            try:
                with zip_ref.open(filename) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                
                if len(content) >= 100:
                    valid_files += 1
                    total_chars += len(content)
            except:
                continue
        
        avg_chars = total_chars / valid_files if valid_files > 0 else 0
        print()
        print(f"📊 Sample Results:")
        print(f"   Valid files: {valid_files}/1000")
        print(f"   Avg chars per file: {avg_chars:.0f}")
        print()
        
        # Estimate for all files
        estimated_valid = int((valid_files / 1000) * len(txt_files))
        
        # Chunking strategy: 1500 chars per chunk
        chunk_size = 1500
        estimated_chunks = int((avg_chars / chunk_size) + 1)
        total_examples = estimated_valid * estimated_chunks
        
        print("📈 ESTIMATES FOR ALL 70,024 FILES:")
        print("=" * 80)
        print(f"   Valid files: ~{estimated_valid:,}")
        print(f"   Chunks per file: ~{estimated_chunks}")
        print(f"   Total examples: ~{total_examples:,}")
        print()
        
        # Training time
        training_time_hours = total_examples / 16 / 1.8 / 3600
        print(f"⏱️  Estimated training time: ~{training_time_hours:.1f} hours")
        print()
        
        # Comparison
        print("📊 COMPARISON:")
        print(f"   Previous (Q&A only): 167 examples")
        print(f"   New (all content): ~{total_examples:,} examples")
        print(f"   Increase: {total_examples / 167:.0f}x more data!")
        print()
        
        # Updated totals
        total_with_all = total_examples + 84422 + 808000
        total_time = training_time_hours + 2.6 + 25
        
        print("🎯 NEW TOTAL WITH ALL DATASETS:")
        print("=" * 80)
        print(f"   Phase 1 (MMedC full): ~{total_examples:,} examples ({training_time_hours:.1f} hrs)")
        print(f"   Phase 2 (Shifaa): 84,422 examples (2.6 hrs)")
        print(f"   Phase 3 (AHD): 808,000 examples (25 hrs)")
        print(f"   ─────────────────────────────────────")
        print(f"   TOTAL: ~{total_with_all:,} examples ({total_time:.1f} hrs)")
        print()

except FileNotFoundError:
    print(f"❌ Error: {zip_path} not found!")
    print()
    print("Expected location: Current directory")
    print("Make sure Arabic.zip is in the same folder as this script")
    print()
