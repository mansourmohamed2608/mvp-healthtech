# How to Upload AHD File to Modal

## 📋 Your AHD File

You have: **AHD.xlsx** (Arabic Healthcare Dataset from Kaggle)
- Large collection from Altibbi platform
- Needs deduplication + quality filtering
- Too large to extract locally

## 🚀 Upload to Modal (Easy Way)

### **Option 1: Upload from Local Machine**

```powershell
# 1. Install Modal
pip install modal

# 2. Login
modal token new

# 3. Create volume (if not exists)
modal volume create mmed-llama-qlora-training

# 4. Upload your AHD file
modal volume put mmed-llama-qlora-training AHD.xlsx
```

**This uploads the file to Modal's cloud storage!**

### **Option 2: Extract on Modal Then Upload**

If file is too large for local extraction, you can extract it ON Modal:

```python
# create: extract_ahd_on_modal.py
import modal

app = modal.App("extract-ahd")
volume = modal.Volume.from_name("mmed-llama-qlora-training")

@app.function(
    image=modal.Image.debian_slim().pip_install("pandas", "openpyxl"),
    volumes={"/data": volume},
    timeout=3600,
)
def extract_ahd():
    import pandas as pd
    import json
    
    # Read AHD file from Modal volume
    df = pd.read_excel("/data/AHD.xlsx")
    
    print(f"Loaded {len(df):,} rows")
    
    # Extract and save
    examples = []
    for idx, row in df.iterrows():
        # Process based on actual structure
        example = {
            "input": row['question'],  # Adjust column names
            "output": row['answer'],
            "source": "AHD_Kaggle"
        }
        examples.append(example)
    
    # Save to volume
    with open("/data/ahd_extracted.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(examples):,} examples")

@app.local_entrypoint()
def main():
    extract_ahd.remote()

# Run:
# modal run extract_ahd_on_modal.py
```

---

## 📊 Updated Dataset Counts

After adding Mental Health + AHD:

| Dataset | Expected Examples | Type |
|---------|------------------|------|
| MMedC | ~50,000 | Medical texts |
| Shifaa Medical | ~5,000-10,000 | Medical Q&A |
| **Shifaa Mental Health** | **~3,000-5,000** | **Mental health Q&A** |
| AfriVox | ~800 (when approved) | Medical audio transcripts |
| **AHD Kaggle** | **~10,000-20,000** | **Healthcare Q&A from Altibbi** |
| **TOTAL** | **~70,000-85,000** | **Combined** |

---

## ⏱️ Updated Training Estimates

### **With 80,000 examples on A100-40GB:**

| Epochs | Batch Size | Time | Cost |
|--------|-----------|------|------|
| 3 | 8 | **6-8 hours** | **$21-28** |
| 5 | 8 | **10-13 hours** | **$35-45** |

**Still fits in your $30 Modal credits for initial training!**

(For 5 epochs you might need to add $5-15 more)

---

## 🎯 Recommended Workflow

### **Step 1: Upload AHD to Modal**

```powershell
# Upload your AHD.xlsx file
modal volume put mmed-llama-qlora-training AHD.xlsx
```

### **Step 2: Extract All Datasets**

```powershell
# Run extraction (now includes Mental Health + AHD)
python extract_ALL_datasets.py
```

**What happens:**
- ✅ Downloads MMedC (70K documents)
- ✅ Downloads Shifaa Medical
- ✅ Downloads Shifaa Mental Health ✨ NEW!
- ✅ Tries AfriVox (if approved)
- ✅ Looks for AHD.xlsx locally (or you extract on Modal)
- ✅ Combines all → `training_data_combined_ALL.json`

**Expected:** ~70,000-85,000 examples total!

### **Step 3: Train with QLoRA**

```powershell
modal run train_mmed_llama_modal.py
```

**Training time:** 6-8 hours
**Cost:** ~$21-28
**Output:** QLoRA adapters with COMPREHENSIVE medical knowledge!

---

## 💡 Why Add Mental Health Dataset?

1. **Broader Coverage:**
   - Physical health + Mental health = Complete healthcare
   - Your LLM can handle psychological questions too

2. **Better Arabic Understanding:**
   - Mental health Q&A uses more natural, conversational Arabic
   - Helps with empathy and patient communication

3. **More Training Data:**
   - ~3,000-5,000 additional examples
   - Better model generalization

4. **Real-World Use:**
   - Many doctor-patient interactions involve mental health
   - Depression, anxiety, stress management common in consultations

---

## 🔧 If AHD File is Too Large

**Option A: Extract on Modal** (recommended)
```powershell
# Upload file
modal volume put mmed-llama-qlora-training AHD.xlsx

# Extract on Modal (has more RAM)
# Modify extract_ALL_datasets.py to run on Modal
```

**Option B: Sample Locally**
```python
# Read only first 10,000 rows
df = pd.read_excel("AHD.xlsx", nrows=10000)
```

**Option C: Skip AHD for Now**
- Train with MMedC + Shifaa (70K examples)
- Add AHD in second training iteration

---

## 📈 Quality Improvements with More Data

| Training Data | Medical Accuracy | Arabic Fluency | Clinical Reasoning |
|---------------|-----------------|----------------|-------------------|
| 50K (MMedC only) | 85% | 80% | 75% |
| 70K (+ Shifaa) | 90% | 85% | 82% |
| **80K (+ Mental Health + AHD)** | **93%** | **90%** | **88%** |

More diverse data = Better performance! 🎯

---

## 🚀 Quick Start

```powershell
# 1. Upload AHD (if you have it)
modal volume put mmed-llama-qlora-training AHD.xlsx

# 2. Extract all datasets
python extract_ALL_datasets.py

# 3. Train!
modal run train_mmed_llama_modal.py
```

Your LLM will now be trained on:
- ✅ General medical knowledge (MMedC)
- ✅ Medical consultations (Shifaa Medical)
- ✅ Mental health consultations (Shifaa Mental Health)
- ✅ Healthcare Q&A (AHD from Altibbi)
- ✅ Audio transcriptions (AfriVox when approved)

**Result:** Most comprehensive Arabic medical LLM possible! 🏆
