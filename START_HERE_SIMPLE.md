# 🚀 ULTRA-SIMPLE CHECKLIST

## ✅ What You Need to Do (5 Steps Only)

### Step 1: Install Local Dependencies
```powershell
.\install_local_deps.ps1
```
OR manually:
```powershell
pip install datasets==3.6.0 huggingface_hub==0.34.2 pandas==2.2.0 openpyxl==3.1.5 tqdm==4.66.5
```

---

### Step 2: Extract Datasets (30-60 min)
```powershell
python extract_ALL_datasets.py
```
✅ Creates `training_data_combined_ALL.json` (~250 MB, 70-85K examples)

---

### Step 3: Setup Modal (5 min)
```powershell
.\setup_modal.ps1
```
OR manually:
```powershell
pip install modal
modal token new        # Opens browser, get $30 free credits
modal volume create mmed-llama-qlora-training
```

---

### Step 4: Upload to Modal (5-10 min)
```powershell
# Upload training data
modal volume put mmed-llama-qlora-training training_data_combined_ALL.json

# Upload AHD file (if you have it)
modal volume put mmed-llama-qlora-training AHD.xlsx
```

---

### Step 5: Train! (6-8 hours, $21-28)
```powershell
modal run train_mmed_llama_modal.py
```

Monitor: https://modal.com/apps

---

## 📥 After Training Completes

Download trained model:
```powershell
modal volume get mmed-llama-qlora-training mmed_llama_qlora ./services/llm/lora_adapters/
```

---

## 🎯 Summary

**Total Time:** 7-9 hours (mostly automated)  
**Total Cost:** $21-28 (fits in $30 free credits)  
**Output:** Trained medical LLM with 70-85K examples

**5 commands → Trained model!** 🎉

---

## 🔥 Quick Reference

| Step | Command | Time | Cost |
|------|---------|------|------|
| 1. Install deps | `.\install_local_deps.ps1` | 2 min | Free |
| 2. Extract data | `python extract_ALL_datasets.py` | 30-60 min | Free |
| 3. Setup Modal | `.\setup_modal.ps1` | 5 min | Free |
| 4. Upload | `modal volume put ...` | 5-10 min | Free |
| 5. Train | `modal run train_mmed_llama_modal.py` | 6-8 hours | $21-28 |
| 6. Download | `modal volume get ...` | 2-5 min | Free |

**Total:** ~7-9 hours, $21-28

---

## 📖 Detailed Guide

See `MODAL_SETUP_COMPLETE.md` for full documentation with troubleshooting.

---

## 🆘 Quick Troubleshooting

**Problem:** "ModuleNotFoundError: No module named 'datasets'"  
**Fix:** Run `.\install_local_deps.ps1` first

**Problem:** "HuggingFace authentication required"  
**Fix:** `huggingface-cli login` then retry

**Problem:** "Modal volume not found"  
**Fix:** `modal volume create mmed-llama-qlora-training`

**Problem:** "Training too slow"  
**Fix:** In `train_mmed_llama_modal.py`, change `gpu="T4"` to `gpu="A100"`

---

## ✅ Ready to Start?

1. Open PowerShell
2. Navigate to project folder
3. Run: `.\install_local_deps.ps1`
4. Then: `python extract_ALL_datasets.py`
5. Wait for completion, then follow steps 3-5 above

**Let's go!** 🚀
