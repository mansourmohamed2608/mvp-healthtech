# Quick Start Guide - 3-Phase Training

## 🎯 Training Order: Smallest → Largest

```
Phase 1: MMedC (3 min)
   ↓
Phase 2: Shifaa (2.6 hours)  
   ↓
Phase 3: AHD (25 hours)
   ↓
FINAL MODEL (All data)
```

---

## Phase 1: MMedC (~3 minutes)

### Configuration for `train_YOUR_llm_kaggle.py`:

```python
CONFIG = {
    "data_paths": [
        "/kaggle/working/training_data_mmedc.json",
    ],
    "output_dir": "./mmedc_lora",
    "num_epochs": 1,
}
```

### Steps:
1. ✏️ Edit `train_YOUR_llm_kaggle.py` → Update `data_paths` 
2. ▶️ Run in Kaggle
3. ⏱️ Wait 3 minutes
4. 📥 Download `mmedc_lora.zip`
5. 📤 Upload to Kaggle as dataset named `mmedc-lora`

**Output:** Model trained on MMedC (167 examples)

---

## Phase 2: Shifaa (~2.6 hours)

### Use script: `train_phase2_shifaa.py`

Already configured with:
```python
"previous_lora_path": [
    "/kaggle/input/mmedc-lora/final_model",  # From Phase 1
],
"data_paths": [
    "/kaggle/working/training_data_shifaa.json",
],
```

### Steps:
1. ✅ Ensure Phase 1 LoRA uploaded as `mmedc-lora` dataset
2. ▶️ Run `train_phase2_shifaa.py` in Kaggle
3. ⏱️ Wait ~2.6 hours
4. 📥 Download `mmedc_shifaa_lora.zip`
5. 📤 Upload to Kaggle as dataset named `mmedc-shifaa-lora`

**Output:** Model trained on MMedC + Shifaa (~84,589 examples)

---

## Phase 3: AHD (~25 hours)

### Use script: `train_phase2_ahd_incremental.py`

Update configuration:
```python
"previous_lora_path": [
    "/kaggle/input/mmedc-shifaa-lora/final_model",  # From Phase 2
],
"data_paths": [
    "/kaggle/working/training_data_ahd.json",
],
```

### Steps:
1. ✅ Ensure Phase 2 LoRA uploaded as `mmedc-shifaa-lora` dataset
2. ✏️ Edit script → Update `previous_lora_path`
3. ▶️ Run in Kaggle
4. ⏱️ Wait ~25 hours
5. 📥 Download `all_combined_lora.zip`
6. 🚀 **DEPLOY TO PRODUCTION!**

**Output:** FINAL MODEL trained on ALL data (892,589 examples)

---

## 📊 Summary Table

| Phase | Dataset | Examples | Time | Script | Output Zip |
|-------|---------|----------|------|--------|------------|
| 1 | MMedC | 167 | 3 min | `train_YOUR_llm_kaggle.py` | `mmedc_lora.zip` |
| 2 | Shifaa | 84,422 | 2.6 hrs | `train_phase2_shifaa.py` | `mmedc_shifaa_lora.zip` |
| 3 | AHD | 808,000 | 25 hrs | `train_phase2_ahd_incremental.py` | `all_combined_lora.zip` |
| **TOTAL** | **ALL** | **892,589** | **~28 hrs** | - | **Final model** |

---

## 🔑 Key Configuration for Each Phase

### Phase 1 - Edit `train_YOUR_llm_kaggle.py`:

Find this section (around line 62):
```python
"data_paths": [
    "/kaggle/working/training_data_FULL_combined.json",  # ❌ Comment out
    ...
],
```

Change to:
```python
"data_paths": [
    "/kaggle/working/training_data_mmedc.json",  # ✅ MMedC only
],
```

And around line 112:
```python
"output_dir": "./trained_model",  # ❌ Old
```

Change to:
```python
"output_dir": "./mmedc_lora",  # ✅ Phase 1 output
```

### Phase 2 - Use `train_phase2_shifaa.py`:

**No changes needed!** Already configured for:
- Load: Phase 1 LoRA
- Train: Shifaa data
- Output: mmedc_shifaa_lora

### Phase 3 - Edit `train_phase2_ahd_incremental.py`:

Find line ~50:
```python
"previous_lora_path": [
    "/kaggle/input/shifaa-lora/final_model",  # ❌ Old
],
```

Change to:
```python
"previous_lora_path": [
    "/kaggle/input/mmedc-shifaa-lora/final_model",  # ✅ Phase 2 LoRA
],
```

---

## 📁 File Checklist

### Before Starting:
- [ ] `training_data_mmedc.json` exists in `/kaggle/working/`
- [ ] `training_data_shifaa.json` exists in `/kaggle/working/`
- [ ] `training_data_ahd.json` exists in `/kaggle/working/`
- [ ] MMed-Llama-3-8B model in `/kaggle/input/medllm/`

### After Phase 1:
- [ ] Downloaded `mmedc_lora.zip`
- [ ] Uploaded to Kaggle as dataset: `mmedc-lora`

### After Phase 2:
- [ ] Downloaded `mmedc_shifaa_lora.zip`
- [ ] Uploaded to Kaggle as dataset: `mmedc-shifaa-lora`

### After Phase 3:
- [ ] Downloaded `all_combined_lora.zip`
- [ ] Ready to deploy!

---

## 🧪 Testing After Each Phase

### Quick Test Script:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "Henrychur/MMed-Llama-3-8B",
    torch_dtype="auto",
    device_map="auto"
)

# Load trained LoRA
model = PeftModel.from_pretrained(model, "./LORA_PATH/final_model")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Henrychur/MMed-Llama-3-8B")

# Test
question = "ما هي أعراض السكري؟"
inputs = tokenizer(question, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

Replace `./LORA_PATH/final_model` with:
- Phase 1: `./mmedc_lora/final_model`
- Phase 2: `./mmedc_shifaa_lora/final_model`
- Phase 3: `./all_combined_lora/final_model`

---

## ⚡ Time Management

### Option 1: All at Once
- Run Phase 1 → Phase 2 → Phase 3 consecutively
- Total: ~28 hours
- Good if: You can leave Kaggle running

### Option 2: Split Sessions
- Day 1: Phase 1 + Phase 2 (~3 hours)
- Day 2: Phase 3 (~25 hours)
- Good if: Limited Kaggle GPU time

### Option 3: Test Early
- Run Phase 1 + Phase 2 (~3 hours)
- Test model quality
- Decide if Phase 3 needed
- Good if: Unsure about dataset quality

---

## 🚀 Ready to Start?

**Step 1:** Verify your data files exist  
**Step 2:** Edit Phase 1 script (`train_YOUR_llm_kaggle.py`)  
**Step 3:** Run Phase 1 in Kaggle  
**Step 4:** Follow the workflow above  

**Total time to final model: ~28 hours** ⏱️  
**Total examples trained: 892,589** 📚  
**Cost: FREE** 💰  

Good luck! 🎉
