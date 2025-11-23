#!/usr/bin/env python3
"""
ASR service tester (matches WhisperX API you're running)

Usage:
  python test_asr.py <audio_file> [--host 127.0.0.1] [--port 5000]
                     [--lang ar] [--dialect egypt]
                     [--diarize / --no-diarize]
                     [--min-speakers 2] [--max-speakers 2]
"""

import sys
import base64
import json
import argparse
import requests
from pathlib import Path

def read_audio_b64(p: Path) -> str:
    """Read audio file and encode as base64."""
    with p.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    ap = argparse.ArgumentParser(description="Test ASR service with audio file")
    ap.add_argument("audio", type=Path, help="path to audio file (wav/mp3/m4a/flac...)")
    ap.add_argument("--host", default="localhost", help="ASR host (default: localhost)")
    ap.add_argument("--port", type=int, default=5000, help="ASR port (default: 5000)")
    ap.add_argument("--lang", default="ar", help="language code (default: ar)")
    ap.add_argument("--dialect", default="egypt", help="dialect hint (default: egypt)")

    diar_group = ap.add_mutually_exclusive_group()
    diar_group.add_argument("--diarize", dest="diarize", action="store_true",
                           help="enable diarization")
    diar_group.add_argument("--no-diarize", dest="diarize", action="store_false",
                           help="disable diarization")
    ap.set_defaults(diarize=True)

    ap.add_argument("--min-speakers", type=int, default=2,
                   help="min speakers (default: 2)")
    ap.add_argument("--max-speakers", type=int, default=2,
                   help="max speakers (default: 2)")
    ap.add_argument("--timeout", type=float, default=None,
                   help="HTTP timeout seconds (default: 300)")

    args = ap.parse_args()

    if not args.audio.exists():
        print(f"❌ audio file not found: {args.audio}")
        sys.exit(1)

    print(f"📁 Reading audio file: {args.audio}")
    audio_b64 = read_audio_b64(args.audio)

    payload = {
        "audio": audio_b64,
        "dialect": args.dialect,
        "language": args.lang,
        "enable_diarization": args.diarize,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers,
    }

    url = f"http://{args.host}:{args.port}/transcribe"
    print(f"\n→ POST {url}")
    print(f"  Language: {args.lang}")
    print(f"  Dialect: {args.dialect}")
    print(f"  Diarization: {args.diarize}")
    print(f"  Speakers: min={args.min_speakers}, max={args.max_speakers}")
    print(f"  Timeout: {args.timeout}s")
    print("\n⏳ Sending request...")

    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=args.timeout
        )
    except requests.exceptions.RequestException as e:
        print(f"\n❌ HTTP error: {e}")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"\n❌ Error: {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    result = resp.json()

    print("\n✅ Transcription successful!")
    print("\n" + "="*80)
    print("FULL TRANSCRIPT")
    print("="*80)
    print(result.get("text", "").strip())

    print("\n" + "="*80)
    print("TRANSCRIPT WITH SPEAKERS")
    print("="*80)
    for i, seg in enumerate(result.get("segments", []), 1):
        speaker = seg.get("speaker", "Unknown")
        text = (seg.get("text") or "").strip()
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        print(f"\n[{start:.1f}s - {end:.1f}s] {speaker}:")
        print(f"  {text}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model used:       {result.get('model_used', 'N/A')}")
    print(f"Language:         {result.get('language', 'N/A')}")
    print(f"Pipeline mode:    {result.get('pipeline_mode', 'N/A')}")
    print(f"Duration:         {result.get('duration', 0):.2f}s")
    print(f"Processing time:  {result.get('processing_time', 0):.2f}s")
    print(f"RTF (slower=↑):   {result.get('rtf', 0):.2f}x")

    speakers = result.get("speakers", [])
    if speakers:
        print(f"Speakers:         {', '.join(speakers)}")

    print(f"Total segments:   {len(result.get('segments', []))}")
    print("="*80)

if __name__ == "__main__":
    main()
