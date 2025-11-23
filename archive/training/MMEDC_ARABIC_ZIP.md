# MMedC Arabic.zip Download - Updated Instructions

## 🆕 What Changed?

The script now downloads **Arabic.zip** (1.28 GB) from MMedC instead of streaming and filtering.

### Benefits:
- ✅ Gets ALL Arabic medical data (not just filtered subset)
- ✅ More examples: 300 instead of 200
- ✅ Faster processing (no streaming delays)
- ✅ Direct access to Arabic-only content

---

## 📦 Required Dependencies

```bash
pip install datasets huggingface-hub pandas tqdm openpyxl
```

**New dependency:** `huggingface-hub` (for downloading specific files)

---

## 🚀 Usage

### Quick Start:
```bash
cd training
pip install datasets huggingface-hub pandas tqdm openpyxl
python download_free_data.py
```

### What Happens:
1. **Shifaa:** Downloads 500 consultations (~5 minutes)
2. **AHD:** Checks for manual download (optional)
3. **MMedC:** Downloads Arabic.zip (1.28 GB) → Extracts → Processes 300 examples (~10-15 minutes)

**Total time:** ~15-20 minutes (depending on internet speed)

---

## 📊 Expected Output

### With Auto-download (Shifaa + MMedC):
```
✅ Converted 500 examples from Shifaa
⚠️  ahd_dataset.xlsx not found - skipping AHD dataset
⏬ Downloading Arabic.zip from HuggingFace...
✅ Downloaded to: ~/.cache/huggingface/...
📂 Extracting Arabic.zip...
✅ Extraction complete
📄 Processing data files...
✅ Converted 300 examples from MMedC Arabic.zip
───────────────────────────────────────
Total examples: 800
```

### With AHD Added:
```
✅ Converted 500 examples from Shifaa
✅ Converted 300 examples from AHD
✅ Converted 300 examples from MMedC Arabic.zip
───────────────────────────────────────
Total examples: 1100
```

---

## 🔍 Arabic.zip Contents

From the MMedC dataset page:
- **Size:** 1.28 GB
- **Format:** ZIP archive containing JSON/JSONL files
- **Content:** Arabic medical Q&A data
- **Source:** Henrychur/MMedC dataset

The script will:
1. Download Arabic.zip using HuggingFace Hub
2. Extract to temporary folder
3. Process all JSON/JSONL/CSV files
4. Extract up to 300 examples
5. Clean up temporary files

---

## 💾 Download Location

Arabic.zip is cached by HuggingFace Hub:
```
Windows: C:\Users\<username>\.cache\huggingface\hub\
Linux/Mac: ~/.cache/huggingface/hub/
```

**Note:** File is cached, so re-running script won't re-download!

---

## ⚠️ Troubleshooting

### Issue: "No module named 'huggingface_hub'"
**Fix:**
```bash
pip install huggingface-hub
```

### Issue: Arabic.zip download is slow
**Fix:**
- This is expected (1.28 GB file)
- Download happens once, then cached
- Patience! ☕

### Issue: "Disk space error"
**Fix:**
- Ensure ~3 GB free space (1.28 GB zip + extraction)
- Temp files are auto-cleaned after processing

### Issue: MMedC download fails
**Fix:**
- Check internet connection
- Script continues without it (Shifaa + AHD still work)
- Try again later or use VPN if blocked

---

## 📈 Updated Data Breakdown

| Source | Examples | Download Method | Size |
|--------|----------|-----------------|------|
| **Shifaa** | 500 | HuggingFace API | ~5 MB |
| **AHD** | 300 | Manual (optional) | 102 KB |
| **MMedC** | 300 | Arabic.zip | 1.28 GB |
| **TOTAL** | **800-1100** | - | ~1.3 GB |

---

## ✅ Verification

After running, check:

```bash
# File should exist
ls -lh training_data_free.json

# Check example count
python -c "import json; d=json.load(open('training_data_free.json')); print(f'Total: {len(d)}'); print('Sources:', {ex['metadata']['source'] for ex in d})"
```

**Expected output:**
```
Total: 800
Sources: {'shifaa', 'mmedc_arabic'}
```

Or with AHD:
```
Total: 1100
Sources: {'shifaa', 'ahd', 'mmedc_arabic'}
```

---

## 🆚 Comparison: Old vs New MMedC

| Method | Examples | Time | Quality |
|--------|----------|------|---------|
| **Old (streaming)** | 200 | 15-30 min | Filtered |
| **New (Arabic.zip)** | **300** | **10-15 min** | **All Arabic** |

**Improvement:** +50% more data, faster processing! 🎉

---

## 🔗 Source

Arabic.zip from: https://huggingface.co/datasets/Henrychur/MMedC/tree/main

Direct link: https://huggingface.co/datasets/Henrychur/MMedC/resolve/main/Arabic.zip

---

## 💡 Pro Tips

1. **Run on good internet:** 1.28 GB download takes time
2. **Don't interrupt:** Let Arabic.zip download complete
3. **Check cache:** File is reused on subsequent runs
4. **Combine with AHD:** Get 1100+ total examples

---

## 🎯 Next Steps

Once you have `training_data_free.json`:

1. **Upload to Kaggle:**
   - Create new dataset
   - Upload `training_data_free.json`

2. **Fine-tune:**
   ```python
   # On Kaggle with GPU
   python finetune_kaggle.py
   ```

3. **Cost:** $0 (completely free!)
4. **Time:** 6-8 hours training on T4 GPU

---

**Result:** 1100+ Arabic medical examples for $0 cost! 🚀
