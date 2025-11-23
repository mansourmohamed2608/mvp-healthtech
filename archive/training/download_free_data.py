"""
Download Free Arabic Medical Data for Training
===============================================

This script downloads FREE Arabic medical datasets instead of using GPT-4o-mini.

Total cost: $0 (completely free!)
Time: ~1-2 hours (includes 1.28 GB Arabic.zip download)

Sources:
1. Shifaa Arabic Medical Consultations (HuggingFace)
2. AHD - Arabic Healthcare Dataset (Mendeley Data - manual)
3. MMedC Arabic.zip (HuggingFace - 1.28 GB)

Output: training_data_free.json (1000+ examples)
"""

import json
import os
import glob
import shutil
import zipfile
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
from tqdm import tqdm

print("=" * 80)
print("DOWNLOADING FREE ARABIC MEDICAL DATA")
print("=" * 80)
print()

# Separate output files for each dataset
OUTPUT_FILES = {
    "shifaa": "training_data_shifaa.json",
    "ahd": "training_data_ahd.json",
    "mmedc": "training_data_mmedc.json",
    "combined": "training_data_all_combined.json"
}

all_training_examples = []

# ============================================================================
# SOURCE 1: Shifaa Arabic Medical Consultations
# ============================================================================
print("📥 Downloading Shifaa Arabic Medical Consultations...")
print("   Source: https://huggingface.co/datasets/Shifaa/arabic-medical-consultations")
print("   Target: ALL available examples (no limits)")
print()

try:
    # FIXED: Correct repository name
    shifaa = load_dataset("Ahmed-Selem/Shifaa_Arabic_Medical_Consultations", split="train")
    print(f"✅ Loaded {len(shifaa)} consultations")
    
    # Check column names
    if len(shifaa) > 0:
        print(f"   Columns: {list(shifaa[0].keys())}")
    print()
    
    # Convert to instruction format - NO LIMIT, get ALL data
    shifaa_examples = []
    for i, example in enumerate(tqdm(shifaa, desc="Processing ALL Shifaa data")):
        # Shifaa dataset has different column names - detect them
        question_col = None
        answer_col = None
        
        # Try to find question/patient column (with space and case variations)
        for col in ['Question', 'question', 'Question Title', 'patient_question', 'Patient', 'patient', 'input', 'query']:
            if col in example:
                question_col = col
                break
        
        # Try to find answer/doctor column (with space and case variations)
        for col in ['Answer', 'answer', 'Doctor Answer', 'doctor_answer', 'Doctor', 'doctor', 'output', 'response']:
            if col in example:
                answer_col = col
                break
        
        if question_col and answer_col and example[question_col] and example[answer_col]:
            conversation = f"مريض: {example[question_col]}\n\nدكتور: {example[answer_col]}"
            
            # Create SOAP-style instruction
            shifaa_examples.append({
                "instruction": "أنت طبيب مساعد. اكتب تقرير SOAP للمحادثة الطبية التالية:",
                "input": conversation,
                "output": f"S (Subjective): {example[question_col]}\n\nA (Assessment): يحتاج تقييم طبي\n\nP (Plan): {example[answer_col]}",
                "metadata": {
                    "source": "shifaa",
                    "task": "soap_generation",
                    "dialect": "arabic",
                    "example_id": i,
                    "category": example.get('category', 'general')
                }
            })
    
    # Save Shifaa separately
    with open(OUTPUT_FILES["shifaa"], "w", encoding="utf-8") as f:
        json.dump(shifaa_examples, f, ensure_ascii=False, indent=2)
    
    all_training_examples.extend(shifaa_examples)
    
    print(f"✅ Converted ALL {len(shifaa_examples)} examples from Shifaa")
    print(f"   Saved to: {OUTPUT_FILES['shifaa']}")
    print()

except Exception as e:
    print(f"⚠️  Could not load Shifaa: {e}")
    print("   Continuing with other sources...")
    print()

# ============================================================================
# SOURCE 2: AHD - Arabic Healthcare Dataset (Mendeley Data)
# ============================================================================
print("📥 Note: AHD - Arabic Healthcare Dataset...")
print("   Source: https://data.mendeley.com/datasets/mgj29ndgrk/5")
print()

print("ℹ️  AHD Dataset Info:")
print("   - 808k+ Arabic medical Q&A from Altibbi platform")
print("   - Requires manual download from Mendeley Data")
print("   - File size: 102.5 kB (zipped)")
print()
print("   📝 To use AHD dataset:")
print("   1. Go to: https://data.mendeley.com/datasets/mgj29ndgrk/5")
print("   2. Download 'AHD.xlsx' or 'AHD_english.xlsx'")
print("   3. Place in training/ folder")
print("   4. Rename to 'ahd_dataset.xlsx'")
print("   5. Re-run this script")
print()

try:
    # Try multiple possible locations for AHD dataset
    ahd_paths = [
        "ahd_dataset.xlsx",  # Local
        "/kaggle/input/ahd-dataset/AHD.xlsx",  # Kaggle input
        "../input/ahd-dataset/AHD.xlsx",  # Kaggle relative path
    ]
    
    ahd_path = None
    for path in ahd_paths:
        if os.path.exists(path):
            ahd_path = path
            break
    
    if ahd_path:
        print(f"✅ Found AHD dataset at: {ahd_path}")
        print("   Loading...")
        ahd = pd.read_excel(ahd_path)
        print(f"✅ Loaded {len(ahd)} healthcare records")
        print("   Target: ALL available examples (no limits)")
        print()
        
        # AHD has: question, answer, category - get ALL data
        ahd_examples = []
        for i, row in tqdm(ahd.iterrows(), total=len(ahd), desc="Processing ALL AHD data"):
            # Check for required columns
            if pd.notna(row.get('question')) and pd.notna(row.get('answer')):
                conversation = f"مريض: {row['question']}\n\nدكتور: {row['answer']}"
                
                ahd_examples.append({
                    "instruction": "أنت طبيب مساعد. اكتب تقرير SOAP للمحادثة الطبية التالية:",
                    "input": conversation,
                    "output": f"S (Subjective): {row['question']}\n\nO (Objective): فحص طبي مطلوب\n\nA (Assessment): {row.get('category', 'استشارة عامة')}\n\nP (Plan): {row['answer']}",
                    "metadata": {
                        "source": "ahd",
                        "task": "soap_generation",
                        "dialect": "arabic",
                        "category": row.get('category', 'unknown')
                    }
                })
        
        # Save AHD separately
        with open(OUTPUT_FILES["ahd"], "w", encoding="utf-8") as f:
            json.dump(ahd_examples, f, ensure_ascii=False, indent=2)
        
        all_training_examples.extend(ahd_examples)
        
        print(f"✅ Converted ALL {len(ahd_examples)} examples from AHD")
        print(f"   Saved to: {OUTPUT_FILES['ahd']}")
        print()
    else:
        print(f"⚠️  {ahd_path} not found - skipping AHD dataset")
        print("   (Optional: See instructions above to add it)")
        print()

except Exception as e:
    print(f"⚠️  Could not load AHD: {e}")
    print()

# ============================================================================
# SOURCE 3: MMedC Arabic Data (Arabic.zip)
# ============================================================================
print("📥 Downloading MMedC Arabic.zip...")
print("   Source: https://huggingface.co/datasets/Henrychur/MMedC")
print("   File: Arabic.zip (1.28 GB - Arabic medical Q&A only)")
print()

try:
    import zipfile
    from huggingface_hub import hf_hub_download
    
    # Download Arabic.zip specifically
    print("⏬ Downloading Arabic.zip from HuggingFace...")
    zip_path = hf_hub_download(
        repo_id="Henrychur/MMedC",
        filename="Arabic.zip",
        repo_type="dataset"
    )
    print(f"✅ Downloaded to: {zip_path}")
    
    # Process directly from ZIP without full extraction (more efficient)
    print("📂 Reading Arabic.zip contents...")
    print("   Target: ALL available examples (no limits)")
    
    mmedc_examples = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        print(f"   Found {len(file_list)} files in archive")
        print()
        
        # First, find what file types exist
        json_files = [f for f in file_list if f.endswith('.json')]
        jsonl_files = [f for f in file_list if f.endswith('.jsonl')]
        csv_files = [f for f in file_list if f.endswith('.csv')]
        txt_files = [f for f in file_list if f.endswith('.txt')]
        
        print(f"   File types: {len(json_files)} JSON, {len(jsonl_files)} JSONL, {len(csv_files)} CSV, {len(txt_files)} TXT")
        
        # Sample some filenames to see the pattern
        print(f"   Sample files: {file_list[:5]}")
        
        # Process ALL files directly from ZIP (no limits!)
        all_files = json_files + jsonl_files + csv_files + txt_files
        print(f"   Processing ALL {len(all_files)} files...")
        print()
        
        processed_files = 0
        
        for filename in tqdm(all_files, desc="Processing ALL MMedC data"):
            
            # Skip directories
            if filename.endswith('/'):
                continue
            
            processed_files += 1
            
            try:
                # Read file directly from ZIP
                with zip_ref.open(filename) as f:
                    content = f.read().decode('utf-8', errors='ignore')
                
                items = []
                if filename.endswith('.json'):
                    try:
                        data = json.loads(content)
                        items = data if isinstance(data, list) else [data]
                    except:
                        continue
                elif filename.endswith('.jsonl'):
                    for line in content.strip().split('\n'):
                        if line.strip():
                            try:
                                items.append(json.loads(line))
                            except:
                                continue
                elif filename.endswith('.csv'):
                    try:
                        from io import StringIO
                        df = pd.read_csv(StringIO(content))
                        items = df.to_dict('records')
                    except:
                        continue
                elif filename.endswith('.txt'):
                    # Try to parse as JSON first
                    try:
                        data = json.loads(content)
                        items = data if isinstance(data, list) else [data]
                    except:
                        # If not JSON, treat as plain text Q&A
                        # Look for patterns like "Q:" and "A:" or similar
                        lines = content.split('\n')
                        current_q = ""
                        current_a = ""
                        
                        for line in lines:
                            line = line.strip()
                            if line.startswith(('Q:', 'السؤال:', 'Question:', 'سؤال:')):
                                if current_q and current_a:
                                    items.append({'question': current_q, 'answer': current_a})
                                current_q = line.split(':', 1)[1].strip() if ':' in line else line
                                current_a = ""
                            elif line.startswith(('A:', 'الجواب:', 'Answer:', 'جواب:', 'الإجابة:')):
                                current_a = line.split(':', 1)[1].strip() if ':' in line else line
                            elif current_a:
                                current_a += " " + line
                            elif current_q:
                                current_q += " " + line
                        
                        # Add last item
                        if current_q and current_a:
                            items.append({'question': current_q, 'answer': current_a})
                
                # Process items - NO LIMIT, get ALL data
                for item in items:
                    # Extract question and answer (try many field names)
                    question = (item.get('question') or item.get('input') or 
                               item.get('text') or item.get('query') or 
                               item.get('Question') or item.get('prompt') or '')
                    
                    answer = (item.get('answer') or item.get('output') or 
                             item.get('response') or item.get('Answer') or 
                             item.get('completion') or item.get('target') or '')
                    
                    # Only add if both question and answer are non-empty strings
                    if question and answer and isinstance(question, str) and isinstance(answer, str):
                        if len(question.strip()) > 10 and len(answer.strip()) > 10:  # Reasonable length
                            mmedc_examples.append({
                                "instruction": "أنت طبيب مساعد. أجب على السؤال الطبي التالي:",
                                "input": str(question).strip(),
                                "output": str(answer).strip(),
                                "metadata": {
                                    "source": "mmedc_arabic",
                                    "task": "medical_qa",
                                    "dialect": "msa",  # MMedC is Modern Standard Arabic
                                    "file": os.path.basename(filename)
                                }
                            })
                
            except Exception as e:
                # Skip problematic files
                continue
        
        print(f"   Processed {processed_files} files, extracted {len(mmedc_examples)} examples")
    
    # Save MMedC separately
    with open(OUTPUT_FILES["mmedc"], "w", encoding="utf-8") as f:
        json.dump(mmedc_examples, f, ensure_ascii=False, indent=2)
    
    all_training_examples.extend(mmedc_examples)
    
    print()
    print(f"✅ Converted ALL {len(mmedc_examples)} examples from MMedC Arabic.zip")
    print(f"   Saved to: {OUTPUT_FILES['mmedc']}")
    print()

except Exception as e:
    print(f"⚠️  Could not load MMedC Arabic.zip: {e}")
    print("   (This is optional - continuing without it)")
    print("   Note: Arabic.zip is 1.28 GB, download may take time")
    print()

# ============================================================================
# SAVE TRAINING DATA
# ============================================================================

print("=" * 80)
print("SAVING TRAINING DATA")
print("=" * 80)
print()

if len(all_training_examples) == 0:
    print("❌ No examples downloaded!")
    print("   Please check dataset availability and try again.")
    exit(1)

print(f"Total examples across all sources: {len(all_training_examples)}")
print()

# Save combined file
with open(OUTPUT_FILES["combined"], "w", encoding="utf-8") as f:
    json.dump(all_training_examples, f, ensure_ascii=False, indent=2)

print(f"✅ Saved combined file to: {OUTPUT_FILES['combined']}")
print()

# Show sample (only if we have examples)
if len(all_training_examples) > 0:
    print("Sample example:")
    print("-" * 80)
    print(json.dumps(all_training_examples[0], ensure_ascii=False, indent=2))
    print("-" * 80)
    print()

    # Statistics
    sources = {}
    for ex in all_training_examples:
        source = ex['metadata']['source']
        sources[source] = sources.get(source, 0) + 1

    print("=" * 80)
    print("SUMMARY - ALL DATA DOWNLOADED")
    print("=" * 80)
    print()
    print("Individual files created:")
    for key, filename in OUTPUT_FILES.items():
        if key != "combined" and os.path.exists(filename):
            print(f"  📄 {filename}")
    print()
    print("Combined file:")
    print(f"  📦 {OUTPUT_FILES['combined']}")
    print()
    print("Breakdown by source:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count:,} examples")
    print()
    print(f"TOTAL: {len(all_training_examples):,} examples")

print()
print("=" * 80)
print("✅ COMPLETE - 100% FREE!")
print("=" * 80)
print()
print("You now have:")
print("1. Separate files for each dataset (train on each individually)")
print("2. Combined file with ALL data (train on everything)")
print()
print("Next steps:")
print("1. Choose which file(s) to use for training:")
print("   - training_data_shifaa.json (Shifaa only)")
print("   - training_data_ahd.json (AHD only, if downloaded)")
print("   - training_data_mmedc.json (MMedC only)")
print("   - training_data_all_combined.json (ALL datasets)")
print()
print("2. Upload chosen file(s) to Kaggle as dataset")
print("3. Run finetune_kaggle.py with your chosen dataset")
print()
print("Total cost: $0 🎉")
print("Total training examples available: {:,}".format(len(all_training_examples)))
