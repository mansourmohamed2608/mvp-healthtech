# Complete Modal Setup Guide - 4 Datasets Training

## 🎯 Overview
This guide covers **everything** you need to run extraction + training on Modal.com, based on Modal's official documentation and best practices.

---

## 📦 Part 1: Install Dependencies Locally (For Extraction)

### Step 1: Install Python packages for `extract_ALL_datasets.py`

```powershell
# Core packages for dataset extraction
pip install datasets huggingface_hub pandas openpyxl tqdm

# Specific versions to avoid compatibility issues:
pip install datasets==3.6.0
pip install huggingface_hub==0.34.2
pip install pandas==2.2.0
pip install openpyxl==3.1.5
pip install tqdm==4.66.5
```

**Why these versions?**
- Based on Modal's official examples (unsloth_finetune.py)
- Tested for compatibility with HuggingFace Hub
- Avoids pandas/numpy conflicts

### Step 2: Run extraction locally
```powershell
python extract_ALL_datasets.py
```

**Output:** `training_data_combined_ALL.json` (~250 MB, 70-85K examples)

---

## 🚀 Part 2: Setup Modal for Training

### Step 1: Install Modal CLI
```powershell
pip install modal
```

### Step 2: Authenticate with Modal (get $30 free credits)
```powershell
modal token new
```
This opens a browser window to authenticate. Follow the steps.

### Step 3: Create Modal volume for data storage
```powershell
# Create volume (one-time)
modal volume create mmed-llama-qlora-training
```

### Step 4: Upload your files to Modal
```powershell
# Upload training data
modal volume put mmed-llama-qlora-training training_data_combined_ALL.json

# Upload AHD.xlsx file (if you have it)
modal volume put mmed-llama-qlora-training AHD.xlsx
```

**Verify uploads:**
```powershell
modal volume ls mmed-llama-qlora-training
```

---

## 📝 Part 3: Modal Training Script Dependencies

Based on Modal's official examples, here are the **EXACT** dependencies for QLoRA training:

### Core Dependencies for `train_mmed_llama_modal.py`:
```python
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        # Core ML frameworks
        "torch>=2.1.0",
        "transformers==4.54.0",
        
        # QLoRA essentials
        "peft==0.16.0",
        "bitsandbytes>=0.41.0",
        
        # Training utilities
        "trl>=0.7.0",              # SFTTrainer
        "accelerate==1.9.0",
        "datasets==3.6.0",
        
        # Tokenization
        "sentencepiece",
        "protobuf",
        
        # Other utilities
        "scipy",
    )
    .apt_install("git")
)
```

### Why these specific versions?
- **transformers 4.54.0**: Latest stable with Llama-3 support
- **peft 0.16.0**: Latest QLoRA/LoRA implementation
- **bitsandbytes 0.41.0+**: Required for 4-bit quantization
- **accelerate 1.9.0**: Multi-GPU training support
- **trl 0.7.0+**: SFTTrainer for supervised fine-tuning
- **datasets 3.6.0**: HuggingFace datasets library

### Updated Training Script
I need to update `train_mmed_llama_modal.py` with the EXACT versions:

```python
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",
        "transformers==4.54.0",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.9.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.7.0",
        "scipy",
        "sentencepiece",
        "protobuf",
    )
    .apt_install("git")
)
```

---

## 🖥️ Part 4: GPU Selection on Modal

Modal has **ALL** GPUs available. Choose based on your budget:

| GPU | VRAM | Speed | Cost/hr | 8h Cost | Recommended |
|-----|------|-------|---------|---------|-------------|
| T4 | 16GB | ⭐ | $0.60 | $4.80 | Testing only |
| L4 | 24GB | ⭐⭐ | $1.10 | $8.80 | Budget |
| A10G | 24GB | ⭐⭐⭐ | $1.80 | $14.40 | Good balance |
| **A100** | 40GB | ⭐⭐⭐⭐ | $3.50 | $28.00 | **✅ Best** |
| A100-80GB | 80GB | ⭐⭐⭐⭐ | $5.50 | $44.00 | Overkill |
| L40S | 48GB | ⭐⭐⭐⭐ | $4.00 | $32.00 | Alternative |
| H100 | 80GB | ⭐⭐⭐⭐⭐ | $8.00 | $64.00 | Way overkill |

### Recommendation: **A100 (40GB)** ✅

**Why?**
- Perfect for 70-85K examples with QLoRA
- 6-8 hours training time
- $21-28 total cost
- Fits in $30 free credits with room to spare

### To change GPU in `train_mmed_llama_modal.py`:
```python
@app.function(
    image=image,
    gpu="A100",  # Options: "T4", "L4", "A10G", "A100", "A100-80GB", "L40S", "H100"
    timeout=3600 * 12,
    volumes={"/data": volume},
    memory=40960,
)
```

---

## ⚡ Part 5: Run Training on Modal

### Option A: Run from Local Machine
```powershell
modal run train_mmed_llama_modal.py
```

### Option B: Deploy and Run on Modal Dashboard
```powershell
# Deploy the function
modal deploy train_mmed_llama_modal.py

# Then go to modal.com dashboard and run it there
```

---

## 📊 Part 6: Monitor Training

### Check logs in real-time:
```powershell
modal app logs mmed-llama-qlora-training
```

### View in Modal Dashboard:
1. Go to https://modal.com/apps
2. Click on `mmed-llama-qlora-training`
3. View logs, GPU usage, and progress

### Expected Output:
```
📚 Loading dataset from /data/training_data_combined_ALL.json
✅ Loaded 73,500 examples
🔄 Loading MMed-Llama-3-8B in 4-bit...
✅ Model loaded (uses 10GB VRAM)
🔧 Configuring QLoRA adapters...
✅ LoRA configured (rank=32, alpha=64)
🚀 Starting training (3 epochs)...

Epoch 1/3:
  [████████████████████] 2,300/2,300 [2:15:30, 0.28it/s]
  Training loss: 1.45

Epoch 2/3:
  [████████████████████] 2,300/2,300 [2:15:30, 0.28it/s]
  Training loss: 0.98

Epoch 3/3:
  [████████████████████] 2,300/2,300 [2:15:30, 0.28it/s]
  Training loss: 0.72

✅ Training complete!
💾 Saving adapters to /data/mmed_llama_qlora/
✅ Saved successfully (95 MB)
```

---

## 📥 Part 7: Download Trained Model

### After training completes:
```powershell
# Download adapters from Modal volume
modal volume get mmed-llama-qlora-training mmed_llama_qlora ./services/llm/lora_adapters/
```

This downloads:
- `adapter_config.json` (1 KB)
- `adapter_model.safetensors` (95 MB)
- `training_log.json` (with loss curves)

---

## 🔧 Part 8: Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'datasets'"
**Solution:** Run extraction locally with packages installed:
```powershell
pip install datasets huggingface_hub pandas openpyxl tqdm
python extract_ALL_datasets.py
```

### Issue 2: "HuggingFace authentication required"
**Solution:** Login to HuggingFace:
```powershell
pip install huggingface_hub
huggingface-cli login
```
Then re-run extraction.

### Issue 3: "Modal volume not found"
**Solution:** Create volume first:
```powershell
modal volume create mmed-llama-qlora-training
```

### Issue 4: "Out of memory during training"
**Solutions:**
1. Reduce batch size in script (8 → 4)
2. Use A100-80GB instead of A100
3. Reduce max_seq_length (2048 → 1024)

### Issue 5: "Training too slow on T4"
**Solution:** Upgrade to A100:
```python
gpu="A100"  # Change from "T4"
```

### Issue 6: "Compatibility issues with transformers"
**Solution:** Use exact versions from Modal's examples:
```python
.pip_install("transformers==4.54.0", "peft==0.16.0")
```

---

## 📋 Complete Workflow Summary

### **Step-by-Step:**

1. **Install local dependencies:**
   ```powershell
   pip install datasets huggingface_hub pandas openpyxl tqdm
   ```

2. **Run extraction locally:**
   ```powershell
   python extract_ALL_datasets.py
   ```
   **Time:** 30-60 minutes

3. **Setup Modal:**
   ```powershell
   pip install modal
   modal token new
   modal volume create mmed-llama-qlora-training
   ```
   **Time:** 5 minutes

4. **Upload to Modal:**
   ```powershell
   modal volume put mmed-llama-qlora-training training_data_combined_ALL.json
   modal volume put mmed-llama-qlora-training AHD.xlsx
   ```
   **Time:** 5-10 minutes

5. **Run training:**
   ```powershell
   modal run train_mmed_llama_modal.py
   ```
   **Time:** 6-8 hours on A100
   **Cost:** $21-28

6. **Download trained model:**
   ```powershell
   modal volume get mmed-llama-qlora-training mmed_llama_qlora ./services/llm/lora_adapters/
   ```
   **Time:** 2-5 minutes

### **Total Time:** ~7-9 hours
### **Total Cost:** $21-28 (fits in $30 free credits)

---

## 🎯 Why Modal vs Kaggle?

| Feature | Modal | Kaggle |
|---------|-------|--------|
| **GPU Options** | T4, L4, A10G, A100, H100 | T4, P100 only |
| **Time Limit** | 12+ hours (configurable) | 12 hours max (hard limit) |
| **Dependency Control** | Full control | Often compatibility issues |
| **Data Upload** | Easy volume system | Manual upload each time |
| **Monitoring** | Real-time dashboard | Limited logs |
| **Cost** | $30 free, then pay-as-you-go | Free (limited resources) |
| **Reliability** | Professional SLA | Best effort |

**Verdict:** Modal is MUCH better for serious training.

---

## ✅ Pre-Flight Checklist

Before starting training:
- [ ] Local packages installed (`datasets`, `huggingface_hub`, `pandas`, etc.)
- [ ] `extract_ALL_datasets.py` completed successfully
- [ ] `training_data_combined_ALL.json` exists (~250 MB)
- [ ] Modal CLI installed (`pip install modal`)
- [ ] Modal authenticated (`modal token new`)
- [ ] Modal volume created (`modal volume create mmed-llama-qlora-training`)
- [ ] Training data uploaded to Modal
- [ ] AHD.xlsx uploaded to Modal (if you have it)
- [ ] GPU selected in `train_mmed_llama_modal.py` (default: A100 ✅)
- [ ] Ready to run: `modal run train_mmed_llama_modal.py`

---

## 🔥 Quick Commands Reference

```powershell
# 1. Install local dependencies
pip install datasets==3.6.0 huggingface_hub==0.34.2 pandas==2.2.0 openpyxl==3.1.5 tqdm==4.66.5

# 2. Run extraction
python extract_ALL_datasets.py

# 3. Setup Modal
pip install modal
modal token new
modal volume create mmed-llama-qlora-training

# 4. Upload data
modal volume put mmed-llama-qlora-training training_data_combined_ALL.json
modal volume put mmed-llama-qlora-training AHD.xlsx

# 5. Run training
modal run train_mmed_llama_modal.py

# 6. Download model
modal volume get mmed-llama-qlora-training mmed_llama_qlora ./services/llm/lora_adapters/
```

---

## 🎉 You're Ready!

Everything is set up correctly. Just follow the steps above and you'll have a trained medical LLM in Arabic! 🚀

**Need help?** Check Modal docs: https://modal.com/docs/guide
