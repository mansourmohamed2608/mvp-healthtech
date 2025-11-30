# WhisperX Migration Guide

## Overview
This guide walks you through migrating from vanilla Whisper to WhisperX for:
- **4-70x faster transcription** (faster-whisper + batching)
- **Fewer hallucinations** (VAD preprocessing)
- **Accurate word timestamps** (Wav2Vec2 alignment)
- **Speaker diarization** (identify doctor vs patient)

## Prerequisites Completed ✅
- [x] Installation script created (`install_whisperx.ps1`)
- [x] Environment configuration created (`.env.example`, `.env`)
- [x] New WhisperX service created (`app_whisperx.py`)

## Migration Steps

### Step 1: Get HuggingFace Token (Required for Diarization)

1. **Create HuggingFace account**: https://huggingface.co/join
2. **Generate access token**: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name: "whisperx-diarization"
   - Type: "Read"
   - Copy the token (starts with `hf_...`)

3. **Accept model agreements** (required for diarization):
   - https://huggingface.co/pyannote/segmentation-3.0
     - Click "Agree and access repository"
   - https://huggingface.co/pyannote/speaker-diarization-3.1
     - Click "Agree and access repository"

4. **Update `.env` file**:
   ```powershell
   cd d:\Downloads\HealthTech\mvp-healthtech\services\asr
   notepad .env
   ```
   Replace `hf_YOUR_ACTUAL_TOKEN_HERE` with your real token.

---

### Step 2: Install WhisperX Dependencies

Run the installation script:
```powershell
cd d:\Downloads\HealthTech\mvp-healthtech\services\asr
.\install_whisperx.ps1
```

This installs:
- WhisperX (with faster-whisper backend)
- Transformers ≥4.30.0
- Pyannote.audio ≥3.1.0 (for diarization)
- Arabic alignment models

Expected output:
```
Successfully installed whisperx-3.x.x
Successfully installed transformers-4.x.x
Successfully installed pyannote-audio-3.x.x
```

**If installation fails**, install manually:
```powershell
pip install git+https://github.com/m-bain/whisperx.git
pip install transformers>=4.30.0
pip install pyannote.audio>=3.1.0
pip install python-dotenv
```

---

### Step 3: Test WhisperX Service

#### 3.1 Backup Original Service
```powershell
cd d:\Downloads\HealthTech\mvp-healthtech\services\asr
Copy-Item app.py app_vanilla_whisper_backup.py
```

#### 3.2 Replace with WhisperX Service
```powershell
Copy-Item app_whisperx.py app.py -Force
```

#### 3.3 Start Service
```powershell
# Make sure you're in the asr directory
cd d:\Downloads\HealthTech\mvp-healthtech\services\asr

# Activate virtual environment if you have one
# .\venv\Scripts\Activate.ps1

# Start service
python app.py
```

Expected output:
```
Loading WhisperX model: large-v3 on cuda...
✓ Whisper model loaded
Loading diarization model...
✓ Diarization model loaded
✓ ASR Service ready!
INFO:     Uvicorn running on http://0.0.0.0:5000
```

**Troubleshooting**:
- **"No module named whisperx"**: Run `.\install_whisperx.ps1` again
- **"HF_TOKEN not found"**: Check `.env` file has correct token
- **"Out of memory"**: Reduce `batch_size` in `app_whisperx.py` (line 174)
- **"CUDA not available"**: Change `.env` to `DEVICE=cpu` (will be slower)

#### 3.4 Test with test1.mp3
```powershell
# In a new terminal
cd d:\Downloads\HealthTech\mvp-healthtech
python test_asr.py
```

Check for:
- ✅ No hallucinations (البروستاتا, الحمل, etc.)
- ✅ No repetitions
- ✅ Correct spelling
- ✅ Speaker labels (SPEAKER_00, SPEAKER_01)
- ✅ Fast processing (4-70x realtime)

---

### Step 4: Update Frontend (Handle New Output Format)

WhisperX returns additional fields:
```json
{
  "text": "full transcription text",
  "segments": [
    {
      "text": "segment text",
      "start": 0.0,
      "end": 5.2,
      "speaker": "SPEAKER_00",  // NEW
      "words": [                 // NEW
        {
          "word": "hello",
          "start": 0.1,
          "end": 0.5,
          "score": 0.98
        }
      ]
    }
  ],
  "language": "ar",
  "duration": 134.5,
  "processing_time": 3.2,
  "rtf": 0.024,
  "speakers": ["SPEAKER_00", "SPEAKER_01"]  // NEW
}
```

#### 4.1 Find Frontend ASR Integration
```powershell
cd d:\Downloads\HealthTech\mvp-healthtech
# Search for where ASR response is used
grep -r "transcribe" frontend-vite/src/ --include="*.tsx" --include="*.ts"
```

#### 4.2 Update TypeScript Types
Add to `frontend-vite/src/types/asr.ts` (or create it):
```typescript
export interface WordTimestamp {
  word: string;
  start: number;
  end: number;
  score?: number;
}

export interface TranscriptionSegment {
  text: string;
  start: number;
  end: number;
  speaker?: string;  // NEW
  words?: WordTimestamp[];  // NEW
}

export interface TranscriptionResponse {
  text: string;
  segments: TranscriptionSegment[];
  language: string;
  duration: number;
  processing_time: number;
  rtf: number;
  speakers?: string[];  // NEW
}
```

#### 4.3 Display Speaker Labels in UI
Example React component update:
```tsx
// Before
<div className="segment">
  <span>{segment.text}</span>
</div>

// After (with speaker colors)
<div className="segment">
  {segment.speaker && (
    <span className={`speaker ${segment.speaker === 'SPEAKER_00' ? 'doctor' : 'patient'}`}>
      {segment.speaker === 'SPEAKER_00' ? '👨‍⚕️ Doctor' : '🧑 Patient'}
    </span>
  )}
  <span>{segment.text}</span>
</div>
```

Add CSS:
```css
.speaker.doctor {
  color: #2563eb; /* Blue for doctor */
  font-weight: 600;
}

.speaker.patient {
  color: #059669; /* Green for patient */
  font-weight: 600;
}
```

---

### Step 5: End-to-End Testing

#### 5.1 Test ASR Service Directly
```powershell
python test_asr.py
```

#### 5.2 Test via Gateway
```powershell
# Start all services
.\start-all.ps1

# Test gateway endpoint
curl -X POST http://localhost:3000/api/transcribe `
  -H "Content-Type: application/json" `
  -d '{"audio": "base64_audio_data", "dialect": "egypt"}'
```

#### 5.3 Test Full Voice Call Flow
1. Open frontend: http://localhost:5173
2. Start voice call
3. Speak test phrase in Arabic/English
4. Verify:
   - Fast transcription (4-70x realtime)
   - No hallucinations
   - Speaker labels appear
   - Word-level timestamps work

---

## Configuration Options

Edit `.env` to customize behavior:

```bash
# Model Settings
WHISPER_MODEL=large-v3      # Options: tiny, base, small, medium, large-v3
DEVICE=cuda                 # Options: cuda, cpu
COMPUTE_TYPE=float16        # Options: float16, int8 (int8 faster, less accurate)

# Features
ENABLE_DIARIZATION=true     # Set to false to disable speaker detection
ENABLE_VAD=true             # Set to false to disable VAD (not recommended)

# Languages
LANGUAGES=ar,en             # Comma-separated language codes

# HuggingFace
HF_TOKEN=hf_your_token      # Required for diarization
```

### Performance Tuning

**In `app_whisperx.py`**, adjust these parameters:

1. **Batch Size** (line 174):
   ```python
   batch_size=16  # Increase for faster processing (needs more GPU memory)
                  # Decrease if you get OOM errors
   ```

2. **VAD Parameters** (lines 180-182):
   ```python
   min_silence_duration_ms=500  # Lower = more segments (may split mid-sentence)
   speech_pad_ms=400            # Padding around speech
   ```

3. **Diarization Min/Max Speakers** (lines 236-238):
   ```python
   min_speakers=2  # Minimum expected speakers (doctor + patient)
   max_speakers=3  # Maximum (doctor + patient + nurse)
   ```

---

## Comparison: Before vs After

| Metric | Vanilla Whisper | WhisperX |
|--------|----------------|----------|
| **Speed** | 1x realtime | 4-70x realtime |
| **Hallucinations** | Frequent (البروستاتا, الحمل) | Rare (VAD removes noise) |
| **Repetitions** | Yes (overlap issues) | No (VAD-based chunking) |
| **Speaker Labels** | ❌ No | ✅ Yes |
| **Word Timestamps** | Approximate | Accurate (Wav2Vec2) |
| **Processing** | Fixed 30s chunks | Smart VAD chunks |
| **GPU Memory** | ~6GB | ~7GB |

---

## Rollback Plan

If WhisperX causes issues:

1. **Restore original service**:
   ```powershell
   cd d:\Downloads\HealthTech\mvp-healthtech\services\asr
   Copy-Item app_vanilla_whisper_backup.py app.py -Force
   ```

2. **Restart service**:
   ```powershell
   python app.py
   ```

3. **Report issue** with logs:
   ```powershell
   # Get error logs
   Get-Content logs/asr.log -Tail 100
   ```

---

## Known Issues & Mitigations

### 1. ⚠️ HuggingFace Token Required
**Issue**: Diarization needs HF token  
**Mitigation**: One-time setup (Step 1), token stays in `.env`

### 2. ⚠️ Mid-Sentence Splits
**Issue**: VAD may split long sentences  
**Mitigation**: Adjust `min_silence_duration_ms` to 700-1000ms

### 3. ⚠️ Speaker Confusion
**Issue**: Similar voices may get same label  
**Mitigation**: Use `min_speakers=2` to force 2+ speakers

### 4. ⚠️ Arabic Alignment
**Issue**: Arabic alignment model needs download  
**Mitigation**: Automatically downloads on first use (~1GB)

### 5. ⚠️ Memory Usage
**Issue**: Uses ~1GB more GPU memory  
**Mitigation**: Reduce `batch_size` or use `COMPUTE_TYPE=int8`

---

## Verification Checklist

Before deploying to production:

- [ ] HuggingFace token configured in `.env`
- [ ] WhisperX dependencies installed (`pip list | grep whisperx`)
- [ ] Service starts without errors
- [ ] Test with test1.mp3 shows no hallucinations
- [ ] Speaker labels appear in output
- [ ] Processing speed is 4x+ realtime
- [ ] Frontend displays speaker labels correctly
- [ ] Gateway can reach ASR service
- [ ] End-to-end voice call works
- [ ] Prometheus metrics still working
- [ ] Backup of original `app.py` saved

---

## Support

**Documentation**:
- WhisperX: https://github.com/m-bain/whisperX
- Pyannote: https://github.com/pyannote/pyannote-audio
- Faster-Whisper: https://github.com/SYSTRAN/faster-whisper

**Common Commands**:
```powershell
# Check GPU usage
nvidia-smi

# Monitor service logs
Get-Content -Path "asr.log" -Wait

# Test endpoint
curl http://localhost:5000/health

# View metrics
curl http://localhost:5000/metrics
```

---

## Next Steps

After successful migration:

1. **Monitor Performance**:
   - Track RTF metrics in Prometheus
   - Compare hallucination rates (before/after)
   - Measure user satisfaction

2. **Optimize Further**:
   - Try `compute_type=int8` for 2x speed
   - Adjust VAD thresholds for your audio
   - Experiment with `batch_size`

3. **Add Features**:
   - Save speaker-labeled transcripts to FHIR
   - Add speaker name mapping (SPEAKER_00 → "Dr. Ahmed")
   - Export word-level timestamps for highlighting

---

## Summary

You've successfully migrated from vanilla Whisper to WhisperX! 🎉

**Key Improvements**:
- ✅ 4-70x faster transcription
- ✅ No more hallucinations (البروستاتا, الحمل)
- ✅ No more repetitions
- ✅ Speaker diarization (doctor vs patient)
- ✅ Accurate word timestamps
- ✅ Better handling of long audio (VAD-based chunking)

**What Changed**:
- Backend: `app.py` now uses WhisperX pipeline
- Frontend: Displays speaker labels
- Dependencies: Added whisperx, pyannote.audio
- Config: New `.env` file with HF_TOKEN

**Test with**: `python test_asr.py` using test1.mp3
