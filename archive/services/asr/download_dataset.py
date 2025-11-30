# services/asr/download_dataset.py
import os
from typing import List
from datasets import load_dataset, Audio

BASE = "hf://datasets/mozilla-foundation/common_voice_16_0@refs/convert/parquet"
SPLIT_PATHS = {
    "train":      f"{BASE}/ar/train/*.parquet",
    "validation": f"{BASE}/ar/validation/*.parquet",
    "test":       f"{BASE}/ar/test/*.parquet",
}

# Expanded medical keyword list
MED_TERMS: List[str] = [
    "طب", "طبي", "عيادة", "مستشفى", "أعراض", "تشخيص", "علاج", "دواء", "جرعة",
    "مضاد", "حيوي", "فيروس", "بكتيريا", "حساسية", "سكر", "ضغط", "قلب", "كبد",
    "كلية", "رئة", "التهاب", "سرطان", "أشعة", "تحاليل", "فحص", "حرارة", "حمى",
    "سعال", "زكام", "صداع", "دوخة", "غثيان", "قيء", "إسهال", "معدة", "أمعاء",
    "عظام", "مفصل", "جلطة", "كورونا", "تلقيح", "لقاح"
]

def is_medical(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return any(term in text for term in MED_TERMS)

def prepare_and_write(split: str, out_csv: str):
    ds = load_dataset("parquet", data_files={split: SPLIT_PATHS[split]}, split=split)
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.filter(lambda x: isinstance(x.get("sentence", ""), str) and x["sentence"].strip())
    ds = ds.filter(lambda x: is_medical(x["sentence"]))
    ds = ds.remove_columns([c for c in ds.column_names if c not in ["path", "sentence"]])
    ds = ds.rename_column("path", "audio_filepath")
    ds = ds.rename_column("sentence", "text")
    if "audio" in ds.column_names:
        ds = ds.remove_columns(["audio"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    ds.to_csv(out_csv)
    print(f"Wrote {len(ds):,} rows → {out_csv}")

def main():
    prepare_and_write("train",      "data/whisper_train.csv")
    prepare_and_write("validation", "data/whisper_validation.csv")
    prepare_and_write("test",       "data/whisper_test.csv")
    print("Done.")

if __name__ == "__main__":
    main()
