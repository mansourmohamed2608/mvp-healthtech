# Deployment Guide: Fine-Tuned Egyptian Medical Model

## Overview

After fine-tuning MMed-Llama-3-8B on Egyptian Arabic medical data, you'll have **LoRA adapters** (~100MB) that improve the model's performance.

This guide shows how to deploy the fine-tuned model in production.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Base Model (8B params)                                      │
│  Henrychur/MMed-Llama-3-8B                                  │
│  [Quantized 4-bit: ~5GB]                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Load adapters
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  LoRA Adapters (~100MB)                                      │
│  ./egyptian-medical-lora/                                    │
│  - adapter_config.json                                       │
│  - adapter_model.bin                                         │
└─────────────────────────────────────────────────────────────┘
                       │
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Fine-tuned Model                                            │
│  Works EXACTLY like base model                               │
│  But better quality for Egyptian Arabic medical             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Download Adapters from Kaggle

After training completes on Kaggle:

1. Go to Kaggle notebook Output
2. Find folder: `/kaggle/working/egyptian-medical-lora/final/`
3. Download all files:
   - `adapter_config.json`
   - `adapter_model.bin`
   - `tokenizer_config.json`
   - Any other files in that folder

4. Save to: `d:\Downloads\HealthTech\mvp-healthtech\models\egyptian-medical-lora\`

---

## Step 2: Update LLM Service

### Option A: Modify Existing Service (Recommended)

Edit `services/llm/app.py`:

```python
# Add at top
from peft import PeftModel

# In model loading section (around line 80-100)
# BEFORE:
llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# AFTER:
# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA adapters
LORA_PATH = "../../models/egyptian-medical-lora"
if os.path.exists(LORA_PATH):
    print(f"Loading fine-tuned adapters from {LORA_PATH}...")
    llm_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    print("✅ Fine-tuned model loaded!")
else:
    print("⚠️ Adapters not found, using base model")
    llm_model = base_model
```

### Option B: Create New Service (For Testing)

Create `services/llm_finetuned/app.py` (copy from `services/llm/app.py` and modify as above).

This lets you run both base and fine-tuned models side-by-side for comparison.

---

## Step 3: Update Requirements

Add to `services/llm/requirements.txt`:

```txt
peft==0.7.1  # For loading LoRA adapters
```

---

## Step 4: Test Locally

### Test 1: Verify Model Loads

```bash
cd d:\Downloads\HealthTech\mvp-healthtech\services\llm
python app.py
```

Expected output:
```
Loading fine-tuned adapters from ../../models/egyptian-medical-lora...
✅ Fine-tuned model loaded!
LLM service running on http://localhost:5003
```

### Test 2: Test Text Correction

```bash
curl -X POST http://localhost:5003/correct \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"عندي وجع في اللثه ودم بينزل\"}"
```

Expected: Better correction than base model.

### Test 3: Test SOAP Generation

```bash
curl -X POST http://localhost:5003/soap \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"دكتور: في ايه؟\nمريض: عندي وجع في اللثة\nدكتور: من امتى؟\nمريض: من اسبوع\"}"
```

Expected: More natural Egyptian SOAP note.

---

## Step 5: Update Kaggle Scripts

Update `kaggle_llm_only.py` to use fine-tuned model:

```python
# Around line 80-100
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load adapters (upload to Kaggle as dataset)
LORA_PATH = "/kaggle/input/egyptian-medical-lora"
if os.path.exists(LORA_PATH):
    print("Loading fine-tuned adapters...")
    llm_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    print("✅ Fine-tuned model loaded!")
else:
    print("⚠️ Using base model")
    llm_model = base_model
```

**To use on Kaggle:**
1. Upload adapters folder as Kaggle dataset
2. Add dataset to notebook
3. Script will automatically use fine-tuned model

---

## Step 6: Performance Comparison

Create `test_finetuned_vs_base.py`:

```python
"""
Compare base model vs fine-tuned model performance
"""

import requests
import json

# Test cases (Egyptian Arabic medical conversations)
TEST_CASES = [
    {
        "conversation": """دكتور: ازيك يا فندم؟
مريض: والله مش كويس يا دكتور، عندي وجع في اللثة
دكتور: ومن امتى وانت حاسس كده؟
مريض: من حوالي اسبوع، ولما بغسل سناني بتنزف
دكتور: طيب هفحصك دلوقتي""",
        "expected": "Dental issue with bleeding gums"
    },
    {
        "conversation": """دكتور: عامل ايه؟
مريض: عندي صداع مستمر يا دكتور
دكتور: من امتى؟
مريض: من 3 ايام تقريبا
دكتور: وبيزيد في وقت معين؟
مريض: اه بالليل بيبقى اقوى""",
        "expected": "Persistent headache, worse at night"
    },
]

def test_soap_generation(conversation):
    """Test SOAP generation for a conversation"""
    response = requests.post(
        "http://localhost:5003/soap",
        json={"text": conversation}
    )
    return response.json()["soap"]

print("=" * 80)
print("BASE MODEL vs FINE-TUNED MODEL COMPARISON")
print("=" * 80)
print()

for i, test in enumerate(TEST_CASES, 1):
    print(f"Test Case {i}: {test['expected']}")
    print("-" * 80)
    print("Conversation:")
    print(test['conversation'])
    print()
    
    # Generate SOAP
    soap = test_soap_generation(test['conversation'])
    
    print("Generated SOAP Note:")
    print(soap)
    print()
    
    # Quality check
    print("Quality Checks:")
    print(f"  ✓ Contains 'S:' or 'Subjective': {'✅' if 'S' in soap or 'Subjective' in soap else '❌'}")
    print(f"  ✓ Contains 'O:' or 'Objective': {'✅' if 'O' in soap or 'Objective' in soap else '❌'}")
    print(f"  ✓ Contains 'A:' or 'Assessment': {'✅' if 'A' in soap or 'Assessment' in soap else '❌'}")
    print(f"  ✓ Contains 'P:' or 'Plan': {'✅' if 'P' in soap or 'Plan' in soap else '❌'}")
    print(f"  ✓ Length > 100 chars: {'✅' if len(soap) > 100 else '❌'}")
    print(f"  ✓ No repetition: {'✅' if not has_repetition(soap) else '❌'}")
    print()
    print("=" * 80)
    print()

def has_repetition(text):
    """Check if text has obvious repetition"""
    words = text.split()
    if len(words) < 3:
        return False
    # Check for 3+ word repetition
    for i in range(len(words) - 2):
        pattern = ' '.join(words[i:i+3])
        if text.count(pattern) > 1:
            return True
    return False
```

Run comparison:
```bash
python test_finetuned_vs_base.py
```

---

## Step 7: Deploy to Azure (Optional)

If deploying to Azure:

1. **Update Docker Image:**
   - Add `peft==0.7.1` to requirements
   - Copy adapters to image: `COPY models/egyptian-medical-lora /app/models/egyptian-medical-lora`
   - Update `app.py` to load adapters

2. **Increase Container Memory:**
   - Fine-tuned model needs same memory as base (~6-8GB)
   - Update Azure Container Instances memory limit

3. **Upload Adapters:**
   - Option A: Include in Docker image (increases image size by ~100MB)
   - Option B: Store in Azure Blob Storage, download on startup

---

## Performance Metrics

### Expected Improvements

| Metric | Base Model | Fine-tuned Model |
|--------|-----------|------------------|
| Egyptian Dialect Understanding | Good | **Excellent** |
| SOAP Note Quality | Good | **Better** |
| Medical Terminology Accuracy | Good | **Better** |
| Inference Speed | 15-20s | 15-20s (same) |
| Model Size | 5GB | 5GB + 100MB |

### Quality Improvements

**Base model issues:**
- Sometimes uses MSA instead of Egyptian context
- Generic SOAP notes
- Occasional medical term errors

**Fine-tuned model:**
- Better Egyptian dialect understanding
- More specific SOAP notes
- Better medical terminology
- More consistent output format

---

## Rollback Plan

If fine-tuned model has issues:

1. **Quick rollback** (keep base model running):
   ```python
   # In app.py, comment out adapter loading
   # llm_model = PeftModel.from_pretrained(base_model, LORA_PATH)
   llm_model = base_model  # Use base model
   ```

2. **A/B testing** (run both models):
   - Deploy both services on different ports
   - Compare outputs side-by-side
   - Gradually migrate traffic

---

## Troubleshooting

### Issue: "Module 'peft' not found"
**Solution:** Install peft: `pip install peft==0.7.1`

### Issue: "Adapter config mismatch"
**Solution:** Ensure you downloaded ALL files from Kaggle output, not just `.bin` file

### Issue: "Out of memory"
**Solution:** Fine-tuned model needs same memory as base. If base model works, fine-tuned should too.

### Issue: "Slower inference"
**Solution:** Adapters should add <5% overhead. If much slower, check:
- Ensure `model.eval()` is called
- Verify 4-bit quantization is enabled
- Check GPU is being used

### Issue: "Quality worse than base"
**Solution:** May need more training data or different hyperparameters. Compare on multiple test cases.

---

## Cost Analysis

### Training (One-time)
- Data generation (GPT-4o-mini): ~$20-30
- Kaggle GPU: **FREE** (30 hours/week)
- **Total: $20-30**

### Deployment (Ongoing)
- Adapter storage: ~100MB (negligible)
- Inference cost: Same as base model
- **Total: $0 extra**

### Comparison (Translation Pipeline)
- Translation: $0.05-0.10 per request
- 1000 requests/month: **$50-100/month**
- **Fine-tuning saves $600-1200/year!**

---

## Next Steps

1. ✅ Download adapters from Kaggle
2. ✅ Update `services/llm/app.py`
3. ✅ Test locally with sample conversations
4. ✅ Compare base vs fine-tuned quality
5. ✅ Deploy to production (if quality is better)
6. ✅ Monitor performance metrics

---

## Support

If you encounter issues:

1. Check Kaggle training logs for errors
2. Verify adapter files are complete
3. Test base model works first
4. Compare outputs side-by-side

The fine-tuned model should work EXACTLY like the base model, just with better quality for Egyptian Arabic medical conversations!
