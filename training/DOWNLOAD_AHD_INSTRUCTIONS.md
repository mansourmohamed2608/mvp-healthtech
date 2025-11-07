# How to Download AHD Dataset (808k+ Examples!)

## Step-by-Step Instructions:

### 1. Go to Mendeley Data
**Link:** https://data.mendeley.com/datasets/mgj29ndgrk/5

### 2. Choose Your File

You have 2 options:

- **AHD.xlsx** (118 MB) - ✅ **RECOMMENDED** - Original Arabic data
- **AHD_english.xlsx** (173 MB) - English translation (optional)

### 3. Download the File

Click "Download All" or download **AHD.xlsx** individually (118 MB)

### 4. Place the File

Move the downloaded file to:
```
d:\Downloads\HealthTech\mvp-healthtech\training\
```

### 5. Rename the File

Rename from `AHD.xlsx` to `ahd_dataset.xlsx`

```powershell
# In PowerShell, run:
cd d:\Downloads\HealthTech\mvp-healthtech\training
mv AHD.xlsx ahd_dataset.xlsx
```

### 6. Re-run the Script

```powershell
python download_free_data.py
```

## What You'll Get:

- **808k+ medical Q&A examples** in Arabic
- Separate file: `training_data_ahd.json`
- Updated combined file with **~893k total examples!**

## Expected Total After AHD:

| Dataset | Examples |
|---------|----------|
| Shifaa | 84,422 |
| MMedC | 167 |
| **AHD** | **808k+** |
| **TOTAL** | **~893k examples** |

## File Sizes to Expect:

- `training_data_ahd.json` - Will be large (~500-800 MB)
- `training_data_all_combined.json` - Will be VERY large (~1-2 GB)

## Notes:

- AHD has 90 medical categories
- Data from altibbi.com (trusted medical platform)
- 100% FREE
- Perfect for fine-tuning Arabic medical models!

## Total Cost: $0 🎉

You're getting nearly 900,000 medical training examples for FREE!
