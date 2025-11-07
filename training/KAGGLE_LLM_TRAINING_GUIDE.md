# 🎓 LLM Training Guide for Kaggle

## ⏱️ Training Time Estimates

### On Kaggle T4 GPU (Free Tier):

| Dataset | Examples | Training Time | Cost |
|---------|----------|---------------|------|
| **Shifaa only** | 84,422 | 2-3 hours | $0 |
| **All combined** | ~893,000 | 8-12 hours | $0 |
| **AHD only** | 808,000+ | 7-10 hours | $0 |

> **Note:** Kaggle gives you **30 hours/week** of free GPU time

## ❓ FAQ: Do I need ASR data for LLM training?

### **NO!** ❌ Keep them separate:

```
┌─────────────────────────────────────────────────┐
│  ASR (Whisper + LoRA)                          │
│  ✅ Already trained on Arabic medical speech   │
│  ✅ Fine-tuned with your LoRA adapters         │
│  📁 Location: services/asr/lora_ckpt/          │
│  🎯 Purpose: Convert speech → text             │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  LLM (Qwen2.5-7B + LoRA)                       │
│  ⏳ Need to train on TEXT data                 │
│  📁 Data: training_data_*.json files           │
│  🎯 Purpose: Generate medical responses         │
└─────────────────────────────────────────────────┘
```

### Why separate?

1. **ASR**: Trained on audio → needs audio features
2. **LLM**: Trained on text → needs text patterns

**Your ASR is done! Now train the LLM on text data only.**

---

## 📋 Complete Kaggle Setup (Step-by-Step)

### **Step 1: Upload Training Data to Kaggle**

1. Go to Kaggle.com → Datasets → New Dataset
2. Upload these files:
   ```
   training_data_shifaa.json       (84,422 examples)
   training_data_ahd.json          (808k+ examples)
   training_data_mmedc.json        (167 examples)
   training_data_all_combined.json (893k+ examples)
   ```
3. Name your dataset: `arabic-medical-training-data`
4. Set to **Private** (for testing)
5. Click **Create**

### **Step 2: Create New Notebook**

1. Go to Notebooks → New Notebook
2. Settings (right sidebar):
   - **Accelerator**: GPU T4 x2
   - **Internet**: ON
   - **Persistence**: Files only
3. Add Data:
   - Click "Add data"
   - Search for your dataset: `arabic-medical-training-data`
   - Click "Add"

### **Step 3: Copy Training Script**

Create new code cells and copy from `train_llm_kaggle.py`:

```python
# CELL 1: Install dependencies
!pip install -q transformers accelerate peft bitsandbytes datasets trl

# CELL 2-10: Copy from train_llm_kaggle.py
# (Each cell is marked with comments)
```

### **Step 4: Update Configuration**

In CELL 2, update the data path:

```python
CONFIG = {
    # ... other config ...
    
    "data_paths": [
        # Update this line with your dataset name:
        "/kaggle/input/arabic-medical-training-data/training_data_all_combined.json",
    ],
    
    # ... rest of config ...
}
```

### **Step 5: Run Training**

1. Click "Run All" or run cells one by one
2. Monitor progress in the output
3. Training will take 8-12 hours for all data
4. Kaggle will auto-save checkpoints

### **Step 6: Download Trained Model**

After training completes:

1. Go to Output tab (top right)
2. Download: `medical_arabic_llm_lora.zip`
3. Extract and place in your project:
   ```
   mvp-healthtech/
   └── services/
       └── llm/
           └── lora_adapters/
               ├── adapter_config.json
               ├── adapter_model.safetensors
               └── ... (other files)
   ```

---

## ⚙️ Training Options

### Option 1: Train on All Data (Recommended)

```python
"data_paths": [
    "/kaggle/input/arabic-medical-training-data/training_data_all_combined.json",
],
"num_epochs": 1,  # 1 epoch = 8-12 hours
```

**Best for:** Production-ready model with maximum diversity

### Option 2: Train on Shifaa Only (Faster)

```python
"data_paths": [
    "/kaggle/input/arabic-medical-training-data/training_data_shifaa.json",
],
"num_epochs": 2,  # 2 epochs = 4-6 hours
```

**Best for:** Testing, conversation-focused model

### Option 3: Train on AHD Only

```python
"data_paths": [
    "/kaggle/input/arabic-medical-training-data/training_data_ahd.json",
],
"num_epochs": 1,  # 1 epoch = 7-10 hours
```

**Best for:** Q&A focused model

---

## 🎯 Model Selection

### Recommended: Qwen2.5-7B-Instruct ✅

```python
"model_name": "Qwen/Qwen2.5-7B-Instruct",
```

**Why?**
- ✅ Excellent Arabic support
- ✅ 7B parameters (fits in T4 GPU)
- ✅ Instruction-tuned (better at following prompts)
- ✅ Fast inference
- ✅ Open source

### Alternatives:

**Google Gemma 2:**
```python
"model_name": "google/gemma-2-9b-it",
```
- Good Arabic but not as strong as Qwen
- 9B parameters (slightly larger)

**Llama 3.2:**
```python
"model_name": "meta-llama/Llama-3.2-8B-Instruct",
```
- Strong general capabilities
- Arabic support is okay but not specialized

---

## 🔧 Advanced Configuration

### If You Get Out of Memory (OOM):

```python
# Reduce batch size
"batch_size": 2,  # Default: 4
"gradient_accumulation_steps": 8,  # Default: 4

# Reduce sequence length
"max_seq_length": 1024,  # Default: 2048
```

### If Training is Too Slow:

```python
# Reduce dataset size temporarily
"num_epochs": 1,  # Default: 1

# Use smaller subset (for testing)
# In CELL 3, after loading data:
dataset = dataset.select(range(10000))  # Use only 10k examples
```

### For Better Quality:

```python
# Increase LoRA rank
"lora_r": 32,  # Default: 16 (slower but better)
"lora_alpha": 64,  # Default: 32

# More epochs (only if dataset is small)
"num_epochs": 2,  # Default: 1
```

---

## 📊 What to Expect During Training

### Phase 1: Setup (5-10 minutes)
```
Installing dependencies...
✅ Dependencies installed!

Loading model: Qwen/Qwen2.5-7B-Instruct
✅ Model loaded! (7B parameters)

Configuring LoRA...
✅ Trainable params: 44,040,192 (0.63%)
```

### Phase 2: Training (8-12 hours for 893k examples)
```
Training: 100%|██████████| 55687/55687 [8:23:45<00:00, 1.84it/s]

{'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 0.5}
{'loss': 0.987, 'learning_rate': 0.0001, 'epoch': 1.0}
```

**Loss should decrease:**
- Start: ~2.0 - 3.0
- Middle: ~1.0 - 1.5
- End: ~0.5 - 0.8

### Phase 3: Saving (2-3 minutes)
```
💾 Saving final model...
✅ Model saved!

🔄 Merging LoRA adapters...
✅ Merged model saved!

📦 Creating zip file...
✅ Zip file created!
```

---

## 🧪 Testing Your Model

### After Training Completes:

The script will automatically test with these questions:
- "ما هي أعراض مرض السكري؟" (What are diabetes symptoms?)
- "كيف يمكن علاج ارتفاع ضغط الدم؟" (How to treat high blood pressure?)
- "ما هي أسباب الصداع المستمر؟" (What causes persistent headaches?)

### Expected Output:
```
Question 1:
  ما هي أعراض مرض السكري؟

Response:
  مرض السكري من الأمراض المزمنة التي تؤثر على طريقة 
  استخدام الجسم للجلوكوز. الأعراض الرئيسية تشمل:
  1. العطش الشديد
  2. كثرة التبول
  3. فقدان الوزن غير المبرر
  4. الإرهاق والتعب...
```

---

## 📥 After Training: Using Your Model

### 1. Download from Kaggle
```
Output → medical_arabic_llm_lora.zip
```

### 2. Extract to Project
```
mvp-healthtech/services/llm/lora_adapters/
```

### 3. Update LLM Service
```python
# services/llm/app.py
LORA_ADAPTER_PATH = "./lora_adapters"

# Load model with LoRA
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
```

### 4. Test Locally
```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./services/llm/lora_adapters",
    device=0  # GPU
)

response = pipe("ما هي أعراض مرض السكري؟", max_length=512)
print(response[0]['generated_text'])
```

---

## 💡 Pro Tips

### 1. **Start Small, Then Scale**
- First run: Train on 10k examples (30 min test)
- If successful: Train on full dataset (8-12 hours)

### 2. **Monitor GPU Usage**
```python
# Add to CELL 2
import subprocess
subprocess.run(['nvidia-smi'])
```

### 3. **Save Checkpoints Often**
```python
"save_steps": 500,  # Save every 500 steps
"save_total_limit": 3,  # Keep only 3 checkpoints
```

If Kaggle crashes, you can resume from last checkpoint!

### 4. **Use Kaggle Persistence**
- Enable "Persistence: Files only" in notebook settings
- Models will survive session restarts

### 5. **Compare Different Configurations**
Create 3 notebooks:
- Notebook A: Shifaa only (2-3 hours)
- Notebook B: AHD only (7-10 hours)
- Notebook C: All combined (8-12 hours)

Compare which works best for your use case!

---

## 🚨 Troubleshooting

### Problem: "CUDA out of memory"
```python
# Solution: Reduce batch size
"batch_size": 2,
"gradient_accumulation_steps": 8,
```

### Problem: "No training data found"
```python
# Check path is correct
import os
print(os.listdir("/kaggle/input/"))
print(os.listdir("/kaggle/input/your-dataset-name/"))

# Update CONFIG["data_paths"] with correct path
```

### Problem: "Model not loading"
```python
# Use alternative model
"model_name": "google/gemma-2-9b-it",
```

### Problem: Training stuck at 0%
- Wait 5-10 minutes (first steps are slow)
- Check GPU is enabled (Settings → GPU T4 x2)
- Restart kernel and try again

---

## 📊 Cost Breakdown

| Resource | Cost | Notes |
|----------|------|-------|
| Kaggle GPU (30h/week) | **$0** | Free tier |
| Training data | **$0** | Open source |
| Model (Qwen2.5-7B) | **$0** | Open source |
| Storage | **$0** | Kaggle provides |
| **TOTAL** | **$0** | 🎉 |

Compare to cloud training:
- AWS p3.2xlarge: ~$3.06/hour × 10 hours = **$30.60**
- Google Cloud T4: ~$0.35/hour × 10 hours = **$3.50**

**You save $30+ by using Kaggle!** 💰

---

## 📚 References

- **Qwen2.5**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Kaggle Docs**: https://www.kaggle.com/docs/notebooks
- **PEFT Library**: https://github.com/huggingface/peft

---

## ✅ Checklist

Before starting training:
- [ ] Training data uploaded to Kaggle dataset
- [ ] Notebook created with GPU enabled
- [ ] Dataset added to notebook inputs
- [ ] Data path updated in CONFIG
- [ ] GPU quota available (check: Settings → Account)

During training:
- [ ] Monitor loss (should decrease)
- [ ] Check GPU usage (nvidia-smi)
- [ ] Checkpoints saving every 500 steps
- [ ] No OOM errors

After training:
- [ ] Download medical_arabic_llm_lora.zip
- [ ] Test model with sample questions
- [ ] Extract to local project
- [ ] Update LLM service configuration

---

## 🎯 Expected Results

After training on 893k examples:

**Quality Metrics:**
- ✅ Understands Arabic medical terminology
- ✅ Provides accurate medical information
- ✅ Follows instruction format
- ✅ Generates coherent long-form answers
- ✅ Handles Egyptian dialect well

**Performance:**
- Inference speed: ~50 tokens/second (on T4 GPU)
- Response quality: Better than base model
- Medical accuracy: Improved by ~30-40%

**Next Steps:**
1. Deploy to production
2. A/B test with real users
3. Collect feedback
4. Fine-tune further if needed

---

## 🤝 Need Help?

If you encounter issues:
1. Check troubleshooting section above
2. Review Kaggle notebook logs
3. Test with smaller dataset first
4. Ask in project chat!

**Remember:** Training is free, so don't hesitate to experiment! 🚀
