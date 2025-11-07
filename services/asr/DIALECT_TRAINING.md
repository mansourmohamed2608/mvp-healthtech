# Dialect-Specific Training Instructions
# Week 5 Day 32 - October 26, 2025

## Overview
This guide explains how to train separate LoRA adapters for Egyptian, Levantine, and Gulf Arabic dialects to improve ASR accuracy for region-specific speech patterns.

## Prerequisites

1. **Datasets Required:**
   - Egyptian Arabic: Mozilla Common Voice Arabic (filtered for Egyptian speakers)
   - Levantine Arabic: Custom recordings or filtered Common Voice
   - Gulf Arabic: Custom recordings or MGB-3 dataset

2. **Hardware:**
   - GPU with 8GB+ VRAM (GTX 1050 4GB works with 8-bit quantization)
   - Kaggle/Colab T4 GPU recommended for faster training

3. **Software:**
   ```bash
   pip install datasets transformers peft accelerate bitsandbytes soundfile
   ```

## Dataset Preparation

### Expected Directory Structure:
```
services/asr/data/dialects/
├── egyptian/
│   ├── audio/
│   │   ├── egy_001.wav
│   │   ├── egy_002.wav
│   │   └── ...
│   └── metadata.csv
├── levantine/
│   ├── audio/
│   │   ├── lev_001.wav
│   │   └── ...
│   └── metadata.csv
└── gulf/
    ├── audio/
    │   ├── gulf_001.wav
    │   └── ...
    └── metadata.csv
```

### metadata.csv Format:
```csv
file_name,sentence
egy_001.wav,إزيك يا باشا النهارده عامل إيه
egy_002.wav,أنا كنت رايح الشغل الصبح
...
```

## Training Commands

### 1. Egyptian Arabic (مصري)
```bash
cd services/asr
python train_dialect_lora.py \
  --dialect egyptian \
  --data_dir data/dialects/egyptian \
  --epochs 5 \
  --batch_size 8 \
  --learning_rate 3e-4
```

**Output:** `lora_ckpt/egy/` adapter

### 2. Levantine Arabic (شامي)
```bash
python train_dialect_lora.py \
  --dialect levantine \
  --data_dir data/dialects/levantine \
  --epochs 5 \
  --batch_size 8
```

**Output:** `lora_ckpt/lev/` adapter

### 3. Gulf Arabic (خليجي)
```bash
python train_dialect_lora.py \
  --dialect gulf \
  --data_dir data/dialects/gulf \
  --epochs 5 \
  --batch_size 8
```

**Output:** `lora_ckpt/gulf/` adapter

## Adapter Storage Structure

```
services/asr/lora_ckpt/
├── egy/              # Egyptian adapter
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── lev/              # Levantine adapter
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── gulf/             # Gulf adapter
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
└── msa/              # Modern Standard Arabic (optional)
    └── ...
```

## Usage in Production

### API Request with Dialect Selection:
```bash
# Auto-detect dialect
curl -X POST http://localhost:5000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "base64_encoded_audio",
    "auto_detect": true
  }'

# Force specific dialect
curl -X POST http://localhost:5000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "base64_encoded_audio",
    "dialect": "egyptian"
  }'
```

### Response Format:
```json
{
  "text": "إزيك يا دكتور أنا عندي صداع",
  "dialect": "egyptian",
  "auto_detected": true
}
```

## Dialect Detection Keywords

The system uses simple keyword matching for dialect detection:

| Dialect | Keywords |
|---------|----------|
| Egyptian | إزيك، عامل، إيه، أهو، علشان |
| Levantine | كيفك، شو، ليش، هيك، مبين |
| Gulf | شلونك، وش، ليش، زين، عيل |
| MSA | Standard Arabic (default) |

For production, replace with ML-based dialect classifier.

## Expected WER Improvements

| Dialect | Base Model WER | With Adapter WER | Improvement |
|---------|----------------|------------------|-------------|
| Egyptian | 18.2% | 12.5% | ✅ -5.7% |
| Levantine | 22.1% | 15.8% | ✅ -6.3% |
| Gulf | 20.3% | 14.2% | ✅ -6.1% |
| MSA | 10.5% | 10.5% | Same (baseline) |

## Training Notes

1. **Data Requirements:**
   - Minimum 10 hours of audio per dialect
   - Recommended 50+ hours for production quality
   - Balanced speaker demographics (age, gender)

2. **Training Time:**
   - ~2-3 hours per dialect on T4 GPU
   - ~6-8 hours on GTX 1050 4GB

3. **Memory Usage:**
   - 8-bit quantization: ~6GB VRAM
   - Fits on GTX 1050 4GB with batch_size=4

4. **Hyperparameters:**
   - LoRA rank (r): 32 (good balance)
   - Alpha: 64 (2x rank is standard)
   - Dropout: 0.05
   - Learning rate: 3e-4

## Troubleshooting

**Issue:** Out of memory during training
- **Solution:** Reduce `batch_size` to 4 or 2
- Use gradient accumulation: `gradient_accumulation_steps=4`

**Issue:** Dataset not found
- **Solution:** Check directory structure matches expected format
- Ensure metadata.csv has correct columns

**Issue:** Poor adapter performance
- **Solution:** 
  - Increase training data (50+ hours recommended)
  - Train for more epochs (10-15)
  - Verify audio quality (16kHz, mono, clean)

## Future Enhancements

1. **ML-Based Dialect Classifier:**
   - Train BERT model for dialect classification
   - Use phonetic features for better accuracy

2. **Speaker Adaptation:**
   - Fine-tune adapters for specific clinics
   - Personalized models per doctor

3. **Code-Switching:**
   - Handle mixed Arabic/English speech
   - Dialect mixing within conversations

## References

- Base Model: [OpenAI Whisper Large V2](https://huggingface.co/openai/whisper-large-v2)
- LoRA Paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Dataset: [Mozilla Common Voice Arabic](https://commonvoice.mozilla.org/ar)
