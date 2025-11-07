"""
Extract ALL MMedC Data - Complete Medical Corpus
=================================================

This script extracts ALL 70,024 MMedC files and converts them to training format.
Instead of filtering for Q&A pairs, we'll use the medical text directly.

Two approaches:
1. Text completion format (for pre-training style)
2. Convert to Q&A format (extract key info as questions)
"""

import json
import os
import zipfile
from tqdm import tqdm
import re
from huggingface_hub import hf_hub_download

def clean_text(text):
    """Clean medical text"""
    # Remove excessive newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Remove extra spaces
    text = re.sub(r' +', ' ', text)
    # Strip
    text = text.strip()
    return text

def extract_all_mmedc_as_completion(zip_path, output_json):
    """
    Extract ALL MMedC files in text completion format.
    This preserves the medical knowledge in the corpus.
    """
    print("=" * 80)
    print("EXTRACTING ALL MMEDC DATA - TEXT COMPLETION FORMAT")
    print("=" * 80)
    print()
    
    examples = []
    
    print(f"📦 Opening: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
        print(f"📄 Found {len(txt_files):,} text files")
        print()
        
        print("🔄 Processing ALL files into training format...")
        for filename in tqdm(txt_files, desc="Processing MMedC"):
            try:
                with zip_ref.open(filename) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                
                # Clean text
                content = clean_text(content)
                
                # Skip if too short (less than 50 chars)
                if len(content) < 50:
                    continue
                
                # Skip if too long (more than 4000 chars - will be truncated anyway)
                if len(content) > 4000:
                    content = content[:4000]
                
                # Format as instruction: given medical text, continue/explain
                example = {
                    "input": "اقرأ النص الطبي التالي وقدم ملخصاً أو شرحاً:",
                    "output": content
                }
                
                examples.append(example)
                
            except Exception as e:
                continue
    
    print()
    print(f"✅ Extracted {len(examples):,} medical documents")
    
    # Save
    print(f"💾 Saving to: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(examples):,} examples")
    print()
    
    return len(examples)


def extract_all_mmedc_as_knowledge(zip_path, output_json):
    """
    Extract ALL MMedC files as medical knowledge documents.
    Format: "Learn this medical information: [content]"
    """
    print("=" * 80)
    print("EXTRACTING ALL MMEDC DATA - KNOWLEDGE FORMAT")
    print("=" * 80)
    print()
    
    examples = []
    
    print(f"📦 Opening: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
        print(f"📄 Found {len(txt_files):,} text files")
        print()
        
        print("🔄 Processing ALL files into knowledge format...")
        for filename in tqdm(txt_files, desc="Processing MMedC"):
            try:
                with zip_ref.open(filename) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                
                # Clean text
                content = clean_text(content)
                
                # Skip if too short
                if len(content) < 50:
                    continue
                
                # Chunk long documents into smaller pieces
                max_chunk_size = 2000
                if len(content) > max_chunk_size:
                    # Split into chunks
                    chunks = []
                    words = content.split()
                    current_chunk = []
                    current_length = 0
                    
                    for word in words:
                        current_chunk.append(word)
                        current_length += len(word) + 1
                        
                        if current_length >= max_chunk_size:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                            current_length = 0
                    
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    
                    # Create examples from chunks
                    for i, chunk in enumerate(chunks):
                        example = {
                            "input": "اشرح المعلومات الطبية التالية بشكل واضح:",
                            "output": chunk
                        }
                        examples.append(example)
                else:
                    # Single example
                    example = {
                        "input": "اشرح المعلومات الطبية التالية بشكل واضح:",
                        "output": content
                    }
                    examples.append(example)
                
            except Exception as e:
                continue
    
    print()
    print(f"✅ Extracted {len(examples):,} knowledge examples")
    
    # Save
    print(f"💾 Saving to: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(examples):,} examples")
    print()
    
    return len(examples)


def extract_all_mmedc_pure_text(zip_path, output_json):
    """
    Extract ALL MMedC files as pure medical text passages.
    Simple format: Just the medical content as output.
    Best for continued pre-training style.
    """
    print("=" * 80)
    print("EXTRACTING ALL MMEDC DATA - PURE TEXT FORMAT")
    print("=" * 80)
    print()
    
    examples = []
    total_chars = 0
    
    print(f"📦 Opening: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
        print(f"📄 Found {len(txt_files):,} text files")
        print()
        
        print("🔄 Processing ALL files...")
        for filename in tqdm(txt_files, desc="Processing MMedC"):
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
                                "output": chunk
                            }
                            examples.append(example)
                            total_chars += len(chunk)
                else:
                    example = {
                        "input": "تعلم المعلومات الطبية التالية:",
                        "output": content
                    }
                    examples.append(example)
                    total_chars += len(content)
                
            except Exception as e:
                continue
    
    print()
    print(f"✅ Processed: {len(examples):,} text passages")
    print(f"📊 Total characters: {total_chars:,}")
    print(f"📊 Estimated tokens: ~{total_chars // 4:,}")
    print()
    
    # Save
    print(f"💾 Saving to: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(examples):,} examples")
    
    # File size
    file_size_mb = os.path.getsize(output_json) / (1024 * 1024)
    print(f"📦 File size: {file_size_mb:.1f} MB")
    print()
    
    return len(examples)


def main():
    """Extract all MMedC data in best format"""
    
    # Download from HuggingFace if not already present
    zip_path = "Arabic.zip"
    
    if not os.path.exists(zip_path):
        print("📥 Arabic.zip not found locally")
        print("⏬ Downloading from HuggingFace...")
        print("   Source: Henrychur/MMedC")
        print("   File: Arabic.zip (1.28 GB)")
        print("   This may take several minutes...")
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
            print(f"❌ Error downloading Arabic.zip: {e}")
            print()
            print("Alternative options:")
            print("1. Manually download from: https://huggingface.co/datasets/Henrychur/MMedC")
            print("2. Place Arabic.zip in current directory")
            print("3. Re-run this script")
            return
    
    print("=" * 80)
    print("MMEDC COMPLETE EXTRACTION")
    print("=" * 80)
    print()
    print("📚 Extracting ALL 70,024 MMedC files")
    print("🎯 Goal: Maximum medical knowledge for training")
    print()
    
    # Method 3 is best - pure text with good chunking
    output_json = "training_data_mmedc_FULL.json"
    
    print("📋 Using: PURE TEXT FORMAT")
    print("   - Preserves all medical content")
    print("   - Chunks large documents")
    print("   - Optimized for training")
    print()
    
    total_examples = extract_all_mmedc_pure_text(zip_path, output_json)
    
    print("=" * 80)
    print("EXTRACTION COMPLETE!")
    print("=" * 80)
    print()
    print(f"✅ Output: {output_json}")
    print(f"✅ Examples: {total_examples:,}")
    print()
    print("📊 Comparison:")
    print(f"   - Previous (Q&A filter): 167 examples")
    print(f"   - Now (all content): {total_examples:,} examples")
    print(f"   - Increase: {total_examples / 167:.1f}x more data!")
    print()
    print("🎯 This file now contains ALL MMedC medical knowledge!")
    print()
    print("Next steps:")
    print("1. Use this file for Phase 1 training")
    print("2. Training time: ~{:.1f} hours".format(total_examples / 16 / 1.8 / 3600 if total_examples > 0 else 0))
    print()


if __name__ == "__main__":
    main()
