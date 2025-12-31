#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from generate_egtts_omar_samir import synthesize_egtts

DEFAULT_EGTTS_TEXT = "أهلاً، هذا اختبار لنظام المساعد الصحي بالمستشفى. كيف أستطيع مساعدتك اليوم؟"
DEFAULT_XTTS_TEXT = "مرحباً، هذا اختبار للهجة السعودية في المساعد الصحي. كيف يمكنني مساعدتك؟"
DEFAULT_SAUDI_SPEAKER = "Suad Qasim"


def run_egtts(args: argparse.Namespace) -> None:
    text = args.text or DEFAULT_EGTTS_TEXT
    out_path = args.out or "egtts_egyptian_demo.wav"
    synthesize_egtts(
        text=text,
        out_path=out_path,
        ref_wav=args.ref_wav,
        temperature=args.temperature,
    )
    print(f"Saved: {out_path}")


def run_xtts(args: argparse.Namespace) -> None:
    try:
        from TTS.api import TTS
    except Exception as exc:
        raise SystemExit(f"Missing TTS dependency: {exc}") from exc

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    text = args.text or DEFAULT_XTTS_TEXT
    out_path = args.out or "xtts_saudi_demo.wav"

    kwargs = {
        "text": text,
        "language": "ar",
        "file_path": out_path,
    }
    if args.speaker_wav:
        kwargs["speaker_wav"] = args.speaker_wav
    else:
        kwargs["speaker"] = args.speaker or DEFAULT_SAUDI_SPEAKER

    tts.tts_to_file(**kwargs)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS demo runner (EGTTS + XTTS)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    egtts = subparsers.add_parser("egtts", help="Run OmarSamir/EGTTS-V0.1 demo")
    egtts.add_argument("--text", help="Text to synthesize")
    egtts.add_argument("--out", help="Output wav path")
    egtts.add_argument("--ref-wav", help="Optional reference wav for speaker")
    egtts.add_argument("--temperature", type=float, default=0.75)
    egtts.set_defaults(func=run_egtts)

    xtts = subparsers.add_parser("xtts", help="Run XTTS v2 demo (Saudi accent)")
    xtts.add_argument("--text", help="Text to synthesize")
    xtts.add_argument("--out", help="Output wav path")
    xtts.add_argument("--speaker", help="XTTS speaker name (default: Suad Qasim)")
    xtts.add_argument("--speaker-wav", help="Reference wav for Saudi accent")
    xtts.set_defaults(func=run_xtts)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
