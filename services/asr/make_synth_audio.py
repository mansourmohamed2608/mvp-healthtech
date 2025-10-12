# services/asr/make_synth_audio.py
import os, asyncio, uuid, pandas as pd
import edge_tts

VOICE = "ar-EG-SalmaNeural"  # Arabic (Egypt) female voice
CSV_IN = "data/medical_text.csv"
OUT_DIR = "data/tts_ar_med"
os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(CSV_IN)

async def synth_one(text: str, path: str):
    communicate = edge_tts.Communicate(text, VOICE)
    with open(path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])

async def main():
    rows = []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue
        out_mp3 = os.path.join(OUT_DIR, f"{uuid.uuid4().hex}.mp3")
        await synth_one(text, out_mp3)
        rows.append({"audio_filepath": out_mp3, "text": text})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)

asyncio.run(main())
