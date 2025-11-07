# Incremental Training Strategy - ALL Datasets

## Training Order (Smallest to Largest)

Train each dataset separately, then load previous LoRA and continue:

| Phase | Dataset | Examples | Training Time | Output |
|-------|---------|----------|---------------|--------|
| **Phase 1** | MMedC (FULL) | ~100,000+ | ~3-4 hours | `mmedc_lora` |
| **Phase 2** | Shifaa | 84,422 | ~2.6 hours | `mmedc_shifaa_lora` |
| **Phase 3** | AHD | 808,000 | ~25 hours | `all_combined_lora` |

**Total Training Time:** ~30-31 hours  
**Total Examples:** ~992,000+  
**Final Model:** Trained on ALL data progressively

**Note:** With full MMedC extraction (all 70k files), you'll get 100k+ examples instead of just 167!

---

## Phase 1: MMedC (Warmup - 3 minutes)

**Why start here:** Smallest dataset, quick validation

### Configuration:
```python
CONFIG = {
    "data_paths": [
        "/kaggle/working/training_data_mmedc.json",
    ],
    "output_dir": "./mmedc_lora",
    "num_epochs": 1,
}
```

### Run:
```bash
# In train_YOUR_llm_kaggle.py
# Update data_paths to use mmedc only
# Run training
```

**Time:** ~3 minutes  
**Output:** Download `mmedc_lora.zip`

---

## Phase 2: Continue with Shifaa

**Load:** Phase 1 (MMedC) LoRA  
**Train on:** Shifaa (84k examples)  
**Result:** Model knows MMedC + Shifaa

### Steps:

1. **Upload Phase 1 LoRA to Kaggle:**
   - Upload `mmedc_lora.zip` as dataset
   - Name: `mmedc-lora`

2. **Create/Update Script:**
   - Use `train_phase2_shifaa.py` (create new or modify existing)
   - Load MMedC LoRA with `is_trainable=True`
   - Train on Shifaa data

### Configuration:
```python
CONFIG = {
    # Previous LoRA from Phase 1
    "previous_lora_path": [
        "/kaggle/input/mmedc-lora/final_model",
    ],
    
    # Current training data
    "data_paths": [
        "/kaggle/working/training_data_shifaa.json",
    ],
    
    "output_dir": "./mmedc_shifaa_lora",
    "num_epochs": 1,
}
```

**Time:** ~2.6 hours  
**Output:** Download `mmedc_shifaa_lora.zip`

---

## Phase 3: Continue with AHD (Final)

**Load:** Phase 2 (MMedC + Shifaa) LoRA  
**Train on:** AHD (808k examples)  
**Result:** Model knows ALL datasets!

### Steps:

1. **Upload Phase 2 LoRA to Kaggle:**
   - Upload `mmedc_shifaa_lora.zip` as dataset
   - Name: `mmedc-shifaa-lora`

2. **Use Existing Script:**
   - Use `train_phase2_ahd_incremental.py`
   - Update `previous_lora_path` to point to Phase 2 LoRA

### Configuration:
```python
CONFIG = {
    # Previous LoRA from Phase 2
    "previous_lora_path": [
        "/kaggle/input/mmedc-shifaa-lora/final_model",
    ],
    
    # Current training data
    "data_paths": [
        "/kaggle/working/training_data_ahd.json",
    ],
    
    "output_dir": "./all_combined_lora",
    "num_epochs": 1,
}
```

**Time:** ~25 hours  
**Output:** Download `all_combined_lora.zip`

---

## Complete Workflow

```
Phase 1: MMedC (3 min)
├─ Base: MMed-Llama-3-8B
├─ Train: 167 examples
├─ Output: mmedc_lora.zip
└─ Upload to Kaggle as dataset
    ↓
Phase 2: Continue with Shifaa (2.6 hours)
├─ Base: MMed-Llama-3-8B
├─ Load: mmedc_lora (is_trainable=True)
├─ Train: 84,422 examples
├─ Output: mmedc_shifaa_lora.zip
└─ Upload to Kaggle as dataset
    ↓
Phase 3: Continue with AHD (25 hours)
├─ Base: MMed-Llama-3-8B
├─ Load: mmedc_shifaa_lora (is_trainable=True)
├─ Train: 808,000 examples
├─ Output: all_combined_lora.zip
└─ FINAL MODEL - Deploy to production!
```

---

## Key Benefits

✅ **Progressive Learning:** Each phase builds on previous  
✅ **Early Testing:** Test after each phase  
✅ **Flexible:** Can stop at any phase  
✅ **Safe:** Each phase is a valid checkpoint  
✅ **Complete:** Uses ALL your data  

---

## Testing After Each Phase

### After Phase 1 (MMedC):
```python
# Load model
model = AutoModelForCausalLM.from_pretrained("MMed-Llama-3-8B", ...)
model = PeftModel.from_pretrained(model, "./mmedc_lora/final_model")

# Test with medical question
# Expect: Basic medical knowledge
```

### After Phase 2 (MMedC + Shifaa):
```python
# Load model
model = AutoModelForCausalLM.from_pretrained("MMed-Llama-3-8B", ...)
model = PeftModel.from_pretrained(model, "./mmedc_shifaa_lora/final_model")

# Test with Arabic medical questions
# Expect: Good medical Q&A responses (84k examples)
```

### After Phase 3 (ALL DATA):
```python
# Load model
model = AutoModelForCausalLM.from_pretrained("MMed-Llama-3-8B", ...)
model = PeftModel.from_pretrained(model, "./all_combined_lora/final_model")

# Test with various medical questions
# Expect: Excellent responses (892k total examples)
```

---

## Important Notes

### Phase 1 (MMedC):
- Only 167 examples (very small)
- Training completes in ~3 minutes
- Mainly for workflow validation
- Model won't be significantly different

### Phase 2 (Shifaa):
- First substantial training (84k examples)
- This is where real learning happens
- Model becomes useful after this phase
- Can deploy if satisfied

### Phase 3 (AHD):
- Largest dataset (808k examples)
- Takes longest (~25 hours)
- Adds massive knowledge base
- Final production model

---

## Next Steps

1. **Start Phase 1** - Validate workflow with MMedC (3 min)
2. **Test quickly** - Does the pipeline work?
3. **Continue to Phase 2** - Real training with Shifaa (2.6 hours)
4. **Evaluate results** - Is model good enough?
5. **Final Phase 3** - Complete training with AHD (25 hours)

**Ready to start Phase 1?** 🚀
