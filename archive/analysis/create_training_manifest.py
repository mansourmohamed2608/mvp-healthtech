"""
Create training manifest CSV from reviewed transcriptions
Run this after manually correcting transcriptions_to_review.json
"""

import json
import pandas as pd
import argparse
from pathlib import Path

def create_manifest_from_reviewed_transcriptions(
    review_file: str = "transcriptions_to_review.json",
    output_csv: str = "medical_training_manifest.csv",
    only_reviewed: bool = True
):
    """
    Create training manifest CSV from reviewed transcriptions
    
    Args:
        review_file: JSON file with reviewed transcriptions
        output_csv: Output CSV file for training
        only_reviewed: Only include entries marked as reviewed
    """
    
    if not Path(review_file).exists():
        print(f"❌ Review file not found: {review_file}")
        print("Run prepare_google_drive_data.py first!")
        return
    
    # Load reviewed transcriptions
    with open(review_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"📄 Loaded {len(data)} entries from {review_file}")
    
    # Filter entries
    training_data = []
    skipped = 0
    
    for entry in data:
        # Check if reviewed
        if only_reviewed and not entry.get("reviewed", False):
            skipped += 1
            continue
        
        # Use corrected transcription if available, otherwise auto-transcription
        transcription = entry.get("corrected_transcription_arabic", "").strip()
        if not transcription:
            transcription = entry.get("auto_transcription_arabic", "").strip()
        
        # Skip if no transcription
        if not transcription:
            skipped += 1
            continue
        
        training_data.append({
            "audio": entry["audio_path"],
            "sentence": transcription,
            "dialect": entry["dialect"],
            "english_question": entry["english_question"],
            "english_answer": entry["english_answer"]
        })
    
    if not training_data:
        print(f"❌ No data to create manifest!")
        if only_reviewed:
            print(f"💡 Tip: Set 'reviewed': true in {review_file} after correcting transcriptions")
        return
    
    # Create DataFrame
    df = pd.DataFrame(training_data)
    
    # Save CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")
    
    print(f"\n✅ Created training manifest: {output_csv}")
    print(f"📊 Statistics:")
    print(f"  - Total entries: {len(data)}")
    print(f"  - Included in training: {len(training_data)}")
    print(f"  - Skipped: {skipped}")
    print(f"  - Egyptian dialect: {len(df[df['dialect'] == 'egyptian'])}")
    print(f"  - Emirati dialect: {len(df[df['dialect'] == 'emirati'])}")
    
    # Show sample
    print(f"\n📝 Sample entries:")
    for i, row in df.head(3).iterrows():
        print(f"\n  [{i+1}] {Path(row['audio']).name}")
        print(f"      Arabic: {row['sentence'][:80]}...")
        print(f"      English: {row['english_answer'][:80]}...")
    
    print(f"\n🚀 Next step:")
    print(f"   Update train_lora_whisper.py CONFIG:")
    print(f"   CONFIG = {{")
    print(f"       'csv_path': '{output_csv}',")
    print(f"       'output_dir': './lora_ckpt_medical',")
    print(f"       'num_epochs': 3,")
    print(f"       'use_hint': False,")
    print(f"       ...")
    print(f"   }}")
    print(f"   Then run: python train_lora_whisper.py")

def main():
    parser = argparse.ArgumentParser(
        description="Create training manifest from reviewed transcriptions"
    )
    parser.add_argument(
        "--review-file",
        default="transcriptions_to_review.json",
        help="Input JSON file with reviewed transcriptions"
    )
    parser.add_argument(
        "--output",
        default="medical_training_manifest.csv",
        help="Output CSV file for training"
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Include unreviewed entries (use auto-transcriptions)"
    )
    
    args = parser.parse_args()
    
    create_manifest_from_reviewed_transcriptions(
        args.review_file,
        args.output,
        only_reviewed=not args.include_unreviewed
    )

if __name__ == "__main__":
    main()
