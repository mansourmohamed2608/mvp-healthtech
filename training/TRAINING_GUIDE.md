# Complete Free Training Pipeline

## Overview

This guide walks you through fine-tuning MMed-Llama-3-8B on Egyptian Arabic medical data **for FREE** (well, ~$25 for data generation).

**Total Cost: $20-30**
**Total Time: ~8-12 hours**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Generate Training Data                             │
│  Tool: GPT-4o-mini ($0.15/1M tokens)                        │
│  Input: 20 medical scenarios                                 │
│  Output: 1000 Egyptian Arabic examples                       │
│  Cost: $20-30 | Time: 2-3 hours                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ training_data.json (1000 examples)
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Step 2: Fine-tune Model                                     │
│  Tool: Kaggle T4 GPU (FREE 30hrs/week)                      │
│  Method: QLoRA (4-bit training)                              │
│  Input: Base model + training data                           │
│  Output: LoRA adapters (~100MB)                              │
│  Cost: $0 | Time: 5-10 hours                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ egyptian-medical-lora/ (adapters)
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Step 3: Deploy Fine-tuned Model                            │
│  Tool: Load base model + adapters                            │
│  Performance: Better Egyptian Arabic quality                 │
│  Cost: $0 extra | Speed: Same as base model                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Local Environment
- Python 3.10+
- OpenAI API key (for data generation)
- ~$25 budget for GPT-4o-mini API calls

### Kaggle Account
- Free account: https://www.kaggle.com/account/login
- Phone verification required for GPU access
- Free tier: 30 GPU hours/week

---

## Step 1: Generate Training Data (~2-3 hours, $20-30)

### 1.1 Set OpenAI API Key

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-your-api-key-here"
```

**Or create `.env` file:**
```bash
cd d:\Downloads\HealthTech\mvp-healthtech\training
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

### 1.2 Install Dependencies

```powershell
pip install openai python-dotenv
```

### 1.3 Run Data Generation

```powershell
cd d:\Downloads\HealthTech\mvp-healthtech\training
python generate_training_data.py
```

**Expected output:**
```
Generating 1000 Egyptian Arabic medical training examples...
Using GPT-4o-mini (cost: ~$20-30)

Progress: [=====>    ] 100/1000 (10%)
Estimated remaining: 1.5 hours
```

**What it generates:**
- 20 medical scenarios (dental, respiratory, chronic, pediatrics, ENT, etc.)
- 50 variations per scenario = 1000 total examples
- Each example:
  - Egyptian dialect conversation (8-12 exchanges)
  - MSA SOAP note (structured medical report)
  - Instruction tuning format (Alpaca style)

**Output:** `training_data.json` (~2-3MB)

### 1.4 Verify Data Quality

```powershell
python -c "import json; data = json.load(open('training_data.json', encoding='utf-8')); print(f'✅ {len(data)} examples'); print(data[0])"
```

Expected: See sample conversation + SOAP note in Egyptian Arabic.

---

## Step 2: Fine-tune on Kaggle (~5-10 hours, FREE)

### 2.1 Upload Training Data to Kaggle

1. Go to: https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload `training_data.json`
4. Title: "Egyptian Medical Training Data"
5. Click **"Create"**

### 2.2 Create New Kaggle Notebook

1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Settings:
   - **Accelerator: GPU T4 x2** (FREE)
   - Language: Python
   - Internet: ON

### 2.3 Add Your Dataset

1. In notebook, click **"+ Add Data"** (right sidebar)
2. Search for your dataset: "Egyptian Medical Training Data"
3. Click **"Add"**

### 2.4 Install Dependencies

**Cell 1:**
```python
# Install required packages
!pip install -q transformers==4.44.0 tokenizers==0.19.1 bitsandbytes==0.48.2 peft==0.7.1 accelerate==1.9.0 datasets
```

### 2.5 Copy Fine-tuning Script

**Cell 2:**
Copy entire contents of `training/finetune_kaggle.py` into cell.

**Update paths:**
```python
# Line 41 - adjust to your dataset path
TRAINING_DATA_PATH = "/kaggle/input/egyptian-medical-training/training_data.json"
```

### 2.6 Run Training

Click **"Run All"** or press **Shift+Enter** on Cell 2.

**Expected timeline:**
```
00:00 - Loading dependencies (2 min)
00:02 - Loading model (3 min)
00:05 - Tokenizing dataset (10 min)
00:15 - Training epoch 1/3 (2 hours)
02:15 - Training epoch 2/3 (2 hours)
04:15 - Training epoch 3/3 (2 hours)
06:15 - Saving adapters (2 min)
06:17 - Testing (5 min)
06:22 - COMPLETE! ✅
```

**Total time: 6-7 hours**

### 2.7 Monitor Progress

Watch for:
- ✅ Model loads successfully (~3 min)
- ✅ Training starts (loss values appear)
- ✅ Loss decreases over time (learning!)
- ✅ Checkpoints saved every 100 steps
- ✅ Final adapters saved

**Training metrics:**
```
Step 100/750 | Loss: 2.31 | ~8 hours remaining
Step 200/750 | Loss: 1.87 | ~6 hours remaining
Step 300/750 | Loss: 1.54 | ~4 hours remaining
...
Step 750/750 | Loss: 0.92 | COMPLETE!
```

Lower loss = better learning!

### 2.8 Download Adapters

After training completes:

1. Click **"Save Version"** (top right)
2. Wait for version to save (~2 min)
3. Go to Output tab
4. Download folder: `egyptian-medical-lora/final/`
5. Save to: `d:\Downloads\HealthTech\mvp-healthtech\models\egyptian-medical-lora\`

Files to download:
- `adapter_config.json`
- `adapter_model.bin` (~100MB)
- `tokenizer_config.json`
- Any other files in that folder

---

## Step 3: Deploy Fine-tuned Model

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed deployment instructions.

**Quick deploy:**

1. **Install PEFT:**
   ```powershell
   pip install peft==0.7.1
   ```

2. **Update LLM service:**
   ```python
   # services/llm/app.py
   from peft import PeftModel
   
   # After loading base model:
   LORA_PATH = "../../models/egyptian-medical-lora"
   llm_model = PeftModel.from_pretrained(base_model, LORA_PATH)
   ```

3. **Test:**
   ```powershell
   cd services\llm
   python app.py
   ```

4. **Verify:**
   ```powershell
   curl -X POST http://localhost:5003/soap -H "Content-Type: application/json" -d "{\"text\": \"دكتور: في ايه؟\nمريض: عندي وجع في اللثة\"}"
   ```

---

## Cost Breakdown

| Item | Cost | Why |
|------|------|-----|
| GPT-4o-mini API | $20-30 | Generate 1000 training examples |
| Kaggle GPU | **FREE** | 30 hours/week free tier |
| Storage | $0 | Adapters are only ~100MB |
| Deployment | $0 | Same infrastructure as base model |
| **TOTAL** | **$20-30** | One-time cost |

**Comparison:**

| Approach | Cost/month | Cost/year |
|----------|-----------|-----------|
| Fine-tuning (our approach) | $0 | $0 (+ $25 one-time) |
| Translation pipeline | $50-100 | $600-1200 |
| Cloud AI service | $200-500 | $2400-6000 |

**Fine-tuning saves $600-6000/year!**

---

## Troubleshooting

### Kaggle Issues

**Issue: "No GPU available"**
- Solution: Wait for GPU quota to reset (weekly)
- Check: Kaggle → Settings → Quotas

**Issue: "Session timeout after 2 hours"**
- Solution: Enable "Persist Outputs" in notebook settings
- Training will continue in background

**Issue: "Out of memory"**
- Solution: Reduce batch size in `finetune_kaggle.py`:
  ```python
  BATCH_SIZE = 2  # Instead of 4
  ```

### Data Generation Issues

**Issue: "OpenAI API rate limit"**
- Solution: Script has built-in delays (0.5s)
- If still hitting limits, increase delay in `generate_training_data.py`:
  ```python
  time.sleep(1.0)  # Instead of 0.5
  ```

**Issue: "Cost higher than expected"**
- Solution: Generate fewer examples first (test with 100):
  ```python
  total_examples = 100  # Instead of 1000
  ```

### Deployment Issues

**Issue: "Module 'peft' not found"**
- Solution: `pip install peft==0.7.1`

**Issue: "Adapter files not loading"**
- Solution: Verify ALL files downloaded from Kaggle
- Check path: `models/egyptian-medical-lora/`

---

## Quality Validation

### Test Cases

Create `test_quality.py`:

```python
import requests

TEST_CONVERSATIONS = [
    # Test 1: Dental issue
    """دكتور: في ايه؟
مريض: عندي وجع في اللثة يا دكتور
دكتور: من امتى؟
مريض: من اسبوع تقريبا""",
    
    # Test 2: Headache
    """دكتور: عامل ايه؟
مريض: عندي صداع مستمر
دكتور: بيزيد امتى؟
مريض: بالليل بيبقى اقوى""",
]

for i, conv in enumerate(TEST_CONVERSATIONS, 1):
    print(f"\nTest {i}:")
    print("Input:", conv)
    
    response = requests.post(
        "http://localhost:5003/soap",
        json={"text": conv}
    )
    
    print("Output:", response.json()["soap"])
    print("-" * 80)
```

### Quality Metrics

Check generated SOAP notes for:
- ✅ Contains all sections (S, O, A, P)
- ✅ Egyptian context understood
- ✅ Medical terms correct
- ✅ No repetition or gibberish
- ✅ Length appropriate (150-300 chars)

---

## Optimization Tips

### Improve Training Quality

1. **More data:** Generate 2000+ examples instead of 1000
2. **Better diversity:** Add more medical scenarios
3. **Longer training:** Increase epochs from 3 to 5
4. **Lower learning rate:** Try 1e-4 instead of 2e-4

### Reduce Cost

1. **Use fewer examples:** 500 examples = ~$10-15
2. **Use GPT-3.5-turbo:** Half the cost, slightly lower quality
3. **Batch API calls:** Process multiple scenarios in one request

### Speed Up Training

1. **Increase batch size:** If VRAM allows (requires 16GB+)
2. **Reduce max_length:** From 1024 to 512 tokens
3. **Fewer training steps:** Set `max_steps=500` instead of full epochs

---

## Next Steps After Training

1. ✅ **A/B Testing:** Compare base vs fine-tuned on 100 real conversations
2. ✅ **Monitor Metrics:** Track SOAP note quality scores
3. ✅ **Collect Feedback:** Ask doctors which outputs are better
4. ✅ **Iterate:** Generate more training data if needed
5. ✅ **Scale:** Deploy to production once quality is validated

---

## Success Criteria

Your fine-tuning is successful if:

1. ✅ Training loss decreases (starts ~2.5, ends <1.0)
2. ✅ Test SOAP notes are coherent and structured
3. ✅ Egyptian dialect is understood better than base model
4. ✅ No increase in inference latency
5. ✅ Doctors prefer fine-tuned outputs in blind tests

---

## Support & Resources

### Documentation
- [Deployment Guide](./DEPLOYMENT_GUIDE.md) - How to deploy fine-tuned model
- [Kaggle Setup Guide](../KAGGLE_SETUP_GUIDE.md) - Kaggle environment setup
- [PEFT Documentation](https://huggingface.co/docs/peft/index) - LoRA fine-tuning

### Kaggle Resources
- Free GPU: https://www.kaggle.com/docs/efficient-gpu-usage
- Troubleshooting: https://www.kaggle.com/discussions

### OpenAI Resources
- Pricing: https://openai.com/api/pricing/
- Rate limits: https://platform.openai.com/docs/guides/rate-limits

---

## Timeline Summary

| Phase | Time | Cost | Output |
|-------|------|------|--------|
| 1. Generate data | 2-3 hours | $20-30 | training_data.json |
| 2. Fine-tune model | 6-8 hours | $0 | LoRA adapters |
| 3. Deploy & test | 1 hour | $0 | Production ready |
| **TOTAL** | **9-12 hours** | **$20-30** | **Fine-tuned model** |

**Most of the time is automated!** You can:
- Generate data: Set it and forget it (2-3 hours)
- Train model: Let Kaggle run overnight (6-8 hours)
- Deploy: Quick setup (1 hour)

**Total active work: ~2 hours**
**Total passive waiting: ~8-10 hours**

---

## Conclusion

You now have a complete FREE pipeline to fine-tune MMed-Llama-3-8B for Egyptian Arabic medical conversations!

**Benefits:**
- ✅ Better quality Egyptian Arabic understanding
- ✅ More accurate medical SOAP notes
- ✅ Same inference speed
- ✅ Only ~$25 one-time cost
- ✅ No ongoing costs

**Start here:**
1. Generate training data with `generate_training_data.py`
2. Fine-tune on Kaggle with `finetune_kaggle.py`
3. Deploy with instructions in `DEPLOYMENT_GUIDE.md`

Good luck! 🚀
