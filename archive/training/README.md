# Fine-tuning Pipeline for Egyptian Arabic Medical LLM

## 🎯 Goal

Improve MMed-Llama-3-8B's performance on **Egyptian Arabic medical conversations** through fine-tuning with QLoRA.

**Result:** Better quality SOAP notes for Egyptian dialect medical conversations, for only ~$25!

---

## 📁 Files in This Directory

| File | Purpose | Use When |
|------|---------|----------|
| `generate_training_data.py` | Generate 1000 Egyptian medical examples using GPT-4o-mini | **First** - Create training data |
| `finetune_kaggle.py` | Fine-tune model on Kaggle free GPU | **Second** - Train model |
| `TRAINING_GUIDE.md` | Complete step-by-step guide | **Start here!** |
| `DEPLOYMENT_GUIDE.md` | How to deploy fine-tuned model | **After training** |

---

## 🚀 Quick Start

### 1. Generate Training Data (~2-3 hours, $20-30)

```powershell
# Set your OpenAI API key
$env:OPENAI_API_KEY = "sk-your-key-here"

# Install dependencies
pip install openai python-dotenv

# Generate 1000 examples
cd training
python generate_training_data.py
```

**Output:** `training_data.json` (1000 Egyptian Arabic medical examples)

### 2. Fine-tune on Kaggle (~6-8 hours, FREE)

1. Upload `training_data.json` to Kaggle as dataset
2. Create new Kaggle notebook with GPU
3. Copy `finetune_kaggle.py` into notebook
4. Run and wait for training to complete
5. Download LoRA adapters (~100MB)

**Output:** Fine-tuned model adapters

### 3. Deploy (~1 hour, $0)

```python
# services/llm/app.py
from peft import PeftModel

# Load base + adapters
base_model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, "./models/egyptian-medical-lora")
```

**Output:** Better Egyptian Arabic SOAP notes!

---

## 📊 What You Get

### Before Fine-tuning (Base Model)
```
Input: "دكتور: في ايه؟ مريض: عندي وجع في اللثة"

Output (Base):
S: المريض يشكو من ألم
O: فحص طبيعي
A: ألم
P: علاج
```
❌ Generic, lacks detail

### After Fine-tuning
```
Input: "دكتور: في ايه؟ مريض: عندي وجع في اللثة"

Output (Fine-tuned):
S (Subjective): المريض يشكو من ألم في اللثة
O (Objective): يلاحظ احمرار وتورم في اللثة
A (Assessment): التهاب اللثة (Gingivitis)
P (Plan): تنظيف الأسنان، مضاد التهاب، متابعة بعد أسبوع
```
✅ Detailed, structured, medical terminology

---

## 💰 Cost Breakdown

| Item | Cost | Time |
|------|------|------|
| Data generation (GPT-4o-mini) | $20-30 | 2-3 hours |
| Fine-tuning (Kaggle free GPU) | **$0** | 6-8 hours |
| Deployment | **$0** | 1 hour |
| **TOTAL** | **$20-30** | **9-12 hours** |

**Comparison:**
- Translation pipeline: $50-100/month ongoing
- Our approach: $25 one-time
- **Savings: $600-1200/year!**

---

## 📖 Documentation

### For Training
👉 **[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)** - Complete step-by-step guide

**Covers:**
- Prerequisites and setup
- Data generation with GPT-4o-mini
- Fine-tuning on Kaggle free GPU
- Troubleshooting common issues
- Quality validation

### For Deployment
👉 **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** - How to use fine-tuned model

**Covers:**
- Loading adapters in production
- Updating LLM service
- Performance comparison
- Rollback plan
- Troubleshooting

---

## 🛠️ Technical Details

### Training Data Format

```json
{
    "instruction": "أنت طبيب مساعد. اكتب تقرير SOAP للمحادثة الطبية التالية:",
    "input": "دكتور: ازيك؟\nمريض: عندي وجع في اللثة...",
    "output": "S (Subjective): المريض يشكو من...\nO (Objective): ...",
    "metadata": {
        "scenario": "مريض يشتكي من التهاب اللثة",
        "task": "soap_generation",
        "dialect": "egyptian"
    }
}
```

### Training Method

- **Base model:** Henrychur/MMed-Llama-3-8B (8B params)
- **Method:** QLoRA (4-bit quantized training)
- **Hardware:** Kaggle T4 GPU (FREE)
- **Training time:** 6-8 hours
- **Output:** LoRA adapters (~100MB)

### Why QLoRA?

- ✅ Fits 8B model on free T4 GPU (14GB VRAM)
- ✅ Fast training (6-8 hours vs days)
- ✅ Small adapters (100MB vs 16GB full model)
- ✅ Easy deployment (load on top of base model)

---

## 📝 Training Data

### 20 Medical Scenarios

1. التهاب اللثة (Dental - Gingivitis)
2. صداع مستمر (Neurology - Headache)
3. ضغط دم مرتفع (Cardiology - Hypertension)
4. سكر دم (Endocrinology - Diabetes)
5. ربو (Respiratory - Asthma)
6. التهاب الجيوب الأنفية (ENT - Sinusitis)
7. إكزيما (Dermatology - Eczema)
8. مغص معوي (GI - Abdominal pain)
9. آلام المفاصل (Rheumatology - Joint pain)
10. قلق ونوم سيء (Psychiatry - Anxiety)
11. ... (20 total scenarios)

Each scenario → 50 variations = **1000 total examples**

### Egyptian Dialect Coverage

- Colloquial medical terms
- Common patient expressions
- Doctor-patient conversation patterns
- Regional vocabulary (Cairo, Alexandria, etc.)

---

## 🎯 Success Metrics

### Quality Improvements

| Metric | Base Model | Fine-tuned |
|--------|-----------|------------|
| Egyptian dialect understanding | Good | **Excellent** |
| SOAP structure | Good | **Better** |
| Medical term accuracy | Good | **Better** |
| Detail level | Generic | **Specific** |
| Repetition issues | Occasional | **Rare** |

### Performance (No Change)

| Metric | Value |
|--------|-------|
| Inference speed | 15-20s (same) |
| Model size | 5GB + 100MB adapters |
| Memory usage | 6-8GB VRAM (same) |

---

## 🔥 Why This Works

### Problem with Base Model
- Trained on 640M Arabic medical tokens (good!)
- But mostly **MSA** (Modern Standard Arabic)
- Egyptian **dialect** is different
- Generic training data, not conversation-focused

### Our Solution
1. Generate **Egyptian dialect** conversations
2. Use **real medical scenarios** (dental, respiratory, etc.)
3. Format as **doctor-patient conversations**
4. Include **structured SOAP notes**
5. Fine-tune with **1000 examples**

### Result
- Model learns **Egyptian patterns**
- Better **conversation understanding**
- More **detailed SOAP notes**
- Improved **medical terminology**

---

## 🚧 Limitations

### What Fine-tuning WON'T Fix
- ❌ Core medical knowledge (already in base model)
- ❌ ASR errors (that's Whisper's job)
- ❌ Inference speed (same architecture)

### What Fine-tuning WILL Fix
- ✅ Egyptian dialect understanding
- ✅ Conversation context
- ✅ SOAP note structure
- ✅ Medical term usage

---

## 🔄 Iterative Improvement

### Phase 1: Initial Training (Current)
- 1000 synthetic examples
- 20 medical scenarios
- GPT-4o-mini generated

### Phase 2: Real Data (Future)
- Collect 100+ real doctor-patient conversations
- Fine-tune on actual clinical data
- Validate with doctors

### Phase 3: Continuous Learning (Future)
- Collect feedback on SOAP notes
- Periodically retrain with new data
- A/B test improvements

---

## 📚 Resources

### Training Guides
- [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) - How to train
- [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - How to deploy

### External Resources
- [PEFT/LoRA Documentation](https://huggingface.co/docs/peft/index)
- [Kaggle Free GPU Guide](https://www.kaggle.com/docs/efficient-gpu-usage)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

### Project Context
- [KAGGLE_LLM_COMPLETE_GUIDE.md](../KAGGLE_LLM_COMPLETE_GUIDE.md) - Using LLM on Kaggle
- [LLM_COMPLETE_IMPLEMENTATION.md](../LLM_COMPLETE_IMPLEMENTATION.md) - Local LLM setup

---

## 🎓 Learning Outcomes

After completing this pipeline, you'll know:

1. ✅ How to generate synthetic training data with GPT-4
2. ✅ How to fine-tune 8B models on free GPUs
3. ✅ How QLoRA enables efficient training
4. ✅ How to deploy fine-tuned models
5. ✅ How to validate model improvements

**Skills gained:**
- LLM fine-tuning
- Synthetic data generation
- Medical NLP
- Cost optimization

---

## 🤝 Contributing

Want to improve the model further?

1. **Add more scenarios:** Edit `generate_training_data.py`
2. **Improve prompts:** Refine GPT-4 prompts for better data
3. **Tune hyperparameters:** Adjust learning rate, epochs in `finetune_kaggle.py`
4. **Collect real data:** Replace synthetic with actual conversations

---

## ⚠️ Important Notes

### Before You Start

1. ✅ Verify base model works (test locally first)
2. ✅ Fix any Kaggle output issues (length check applied)
3. ✅ Have OpenAI API key ready (~$25 budget)
4. ✅ Verify Kaggle account has GPU access

### During Training

1. 📊 Monitor training loss (should decrease)
2. 💾 Save checkpoints regularly (every 100 steps)
3. ⏰ Training takes 6-8 hours (let it run)
4. 🔍 Check test output at end (quality check)

### After Training

1. 📥 Download ALL adapter files (not just .bin)
2. 🧪 Test locally before production
3. 📊 Compare base vs fine-tuned quality
4. 🚀 Deploy if quality is better

---

## 🎉 Next Steps

### Ready to Start?

👉 **[Open TRAINING_GUIDE.md](./TRAINING_GUIDE.md)** for step-by-step instructions!

### Questions?

- Check troubleshooting sections in guides
- Review Kaggle training logs
- Test with sample conversations
- Compare outputs side-by-side

---

## 📞 Support

If you encounter issues:

1. Check [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) troubleshooting section
2. Review Kaggle notebook logs for errors
3. Verify all dependencies are correct versions
4. Test base model works before fine-tuning

---

## 🏆 Success Checklist

- [ ] Generated 1000 training examples
- [ ] Training completed on Kaggle (loss < 1.0)
- [ ] Downloaded LoRA adapters
- [ ] Deployed locally
- [ ] Tested with sample conversations
- [ ] Quality better than base model
- [ ] No increase in inference time

**All checked? Congratulations! 🎉 You've successfully fine-tuned an 8B medical LLM for FREE!**
