from TTS.api import TTS
import torch
import os

# Pick device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load XTTSv2 (will reuse the already downloaded model)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

OUT_DIR = "xtts_voice_samples"
os.makedirs(OUT_DIR, exist_ok=True)

TEXT = "مرحبا، هذا اختبار لنظام الصحة."
N = None  # set an integer (e.g. 20) if you want to limit; None = ALL speakers

for i, speaker in enumerate(tts.speakers):
    if N is not None and i >= N:
        break

    safe_name = speaker.replace(" ", "_")
    filename = f"{i:03d}_{safe_name}.wav"
    path = os.path.join(OUT_DIR, filename)

    print(f"Generating {i}: {speaker} -> {path}")
    tts.tts_to_file(
        text=TEXT,
        speaker=speaker,
        language="ar",
        file_path=path,
    )

print(f"\nDone. Check the '{OUT_DIR}' folder.")
