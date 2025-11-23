# 🚀 ONE-CLICK DATA PREPARATION

## Quick Start (Easiest Way)

```bash
cd training
python download_and_extract_ALL.py
```

**That's it!** This single script will:
- ✅ Download Shifaa (~84k examples)
- ✅ Download AHD if available (~808k examples)
- ✅ Download MMedC (1.28 GB)
- ✅ Extract ALL 70k MMedC files → ~100k+ examples
- ✅ Create separate files for each dataset
- ✅ Create combined file with ALL data

**Result:** ~992,000+ training examples ready in ~5-10 minutes!

---

## Alternative: Step-by-Step

### Option 1: Just MMedC Full Extraction

```bash
python extract_ALL_mmedc.py
```

**Output:** `training_data_mmedc_FULL.json` with ~100k+ examples

### Option 2: Estimate First

```bash
python estimate_mmedc_full.py
```

See how many examples you'll get before extracting.

---

## Files Created

After running `download_and_extract_ALL.py`:

| File | Examples | Use Case |
|------|----------|----------|
| `training_data_mmedc_FULL.json` | ~100,000+ | Phase 1: Medical knowledge |
| `training_data_shifaa.json` | 84,422 | Phase 2: Q&A instruction tuning |
| `training_data_ahd.json` | 808,000 | Phase 3: Large-scale Q&A |
| `training_data_FULL_combined.json` | ~992,000+ | All-in-one training |

---

## Training Strategy

### Recommended: 3-Phase Incremental

```
Phase 1: MMedC Full  (~100k examples) → 3-4 hours
    ↓
Phase 2: Shifaa      (84k examples)   → 2.6 hours
    ↓
Phase 3: AHD         (808k examples)  → 25 hours
    ↓
FINAL: ~992k examples total
```

See `TRAIN_ALL_INCREMENTAL.md` for complete guide.

### Alternative: All-at-once

Use `training_data_FULL_combined.json` for single training run (~31 hours).

---

## Scripts Overview

| Script | Purpose | Time |
|--------|---------|------|
| `download_and_extract_ALL.py` | **ONE-CLICK** - Does everything | 5-10 min |
| `extract_ALL_mmedc.py` | Extract only MMedC (all 70k files) | 1-2 min |
| `estimate_mmedc_full.py` | Preview MMedC extraction results | 30 sec |

---

## Requirements

```bash
pip install transformers datasets huggingface_hub pandas tqdm openpyxl
```

Already installed if you're using Kaggle!

---

## Comparison: Old vs New

### Old Method (Q&A Filter):
```
MMedC: 167 examples (0.2% of files)
Shifaa: 84,422 examples
AHD: 808,000 examples
────────────────────────
TOTAL: 892,589 examples
```

### New Method (ALL Content):
```
MMedC: ~100,000 examples (99.8% of files)
Shifaa: 84,422 examples
AHD: 808,000 examples
─────────────────────────────────
TOTAL: ~992,422 examples
```

**Gain: 100,000 additional medical knowledge examples!** 🎉

---

## Quick Reference

**To get everything:**
```bash
python download_and_extract_ALL.py
```

**To train Phase 1:**
```python
# In train_YOUR_llm_kaggle.py
"data_paths": ["/kaggle/working/training_data_mmedc_FULL.json"]
```

**To continue to Phase 2:**
```bash
# Use train_phase2_shifaa.py (already configured)
```

**To continue to Phase 3:**
```bash
# Use train_phase2_ahd_incremental.py (update LoRA path)
```

---

## Support

- Training guides: `TRAIN_ALL_INCREMENTAL.md`, `QUICK_START_3_PHASES.md`
- Workflow: `EXTRACT_ALL_WORKFLOW.md`
- Issues: Check Kaggle logs for errors

---

## 💡 Pro Tips

1. **Start with estimation:** Run `estimate_mmedc_full.py` first to see what you'll get
2. **Test early:** After Phase 1, test the model before continuing
3. **Save checkpoints:** Download LoRA after each phase
4. **Monitor Kaggle time:** Split Phase 3 if needed (Kaggle has 30hr/week limit)

---

## 🎯 Ready to Start?

```bash
cd training
python download_and_extract_ALL.py
```

Then follow `TRAIN_ALL_INCREMENTAL.md` for training! 🚀
