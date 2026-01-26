#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

DEFAULT_EGTTS_TEXT = "أهلًا بيك! معاك خدمة الحجز. تحب تحجز لمين، وعايز معاد إمتى؟"
DEFAULT_SAUDI_TEXT = "هلا والله! معك الاستقبال. تبي تحجز موعد؟ وش اسمك الكامل لو سمحت؟"
EGTTS_REPO_ID = "OmarSamir/EGTTS-V0.1"
SAUDI_REPO_ID = "AhmedEladl/saudi-tts"
SAUDI_REVISION = "f99ffe0"


def _load_xtts(repo_id: str, revision: str | None, model_dir: Path, speaker_file: str):
    try:
        from huggingface_hub import hf_hub_download
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig
    except Exception as exc:
        raise SystemExit(f"Missing TTS dependency: {exc}") from exc

    import torch

    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(
        hf_hub_download(
            repo_id,
            "config.json",
            revision=revision,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    )
    vocab_path = Path(
        hf_hub_download(
            repo_id,
            "vocab.json",
            revision=revision,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    )
    Path(
        hf_hub_download(
            repo_id,
            "model.pth",
            revision=revision,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    )
    speaker_path = Path(
        hf_hub_download(
            repo_id,
            speaker_file,
            revision=revision,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(model_dir),
        vocab_path=str(vocab_path),
        use_deepspeed=False,
        eval=True,
    )
    model.to(device)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[str(speaker_path)])
    return model, gpt_cond_latent, speaker_embedding


def run_egtts(args: argparse.Namespace) -> None:
    import torch
    import torchaudio

    text = args.text or DEFAULT_EGTTS_TEXT
    out_path = args.out or "egtts_egyptian_demo.wav"
    model_dir = BASE_DIR / "egtts_model"
    model, gpt_cond_latent, speaker_embedding = _load_xtts(
        EGTTS_REPO_ID,
        None,
        model_dir,
        "speaker_reference.wav",
    )
    out = model.inference(
        text=text,
        language="ar",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=args.temperature,
    )
    torchaudio.save(out_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    print(f"Saved: {out_path}")


def run_saudi(args: argparse.Namespace) -> None:
    import soundfile as sf

    text = args.text or DEFAULT_SAUDI_TEXT
    out_path = args.out or "saudi_tts_demo.wav"
    model_dir = BASE_DIR / "saudi_tts_model"
    model, gpt_cond_latent, speaker_embedding = _load_xtts(
        SAUDI_REPO_ID,
        SAUDI_REVISION,
        model_dir,
        "speaker.wav",
    )
    out = model.inference(
        text=text,
        language="ar",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=args.temperature,
    )
    sf.write(out_path, out["wav"], 24000)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS demo runner (EGTTS + Saudi XTTS)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    egtts = subparsers.add_parser("egtts", help="Run OmarSamir/EGTTS-V0.1 demo")
    egtts.add_argument("--text", help="Text to synthesize")
    egtts.add_argument("--out", help="Output wav path")
    egtts.add_argument("--temperature", type=float, default=0.55)
    egtts.set_defaults(func=run_egtts)

    saudi = subparsers.add_parser("saudi", help="Run AhmedEladl/saudi-tts demo")
    saudi.add_argument("--text", help="Text to synthesize")
    saudi.add_argument("--out", help="Output wav path")
    saudi.add_argument("--temperature", type=float, default=0.50)
    saudi.set_defaults(func=run_saudi)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
