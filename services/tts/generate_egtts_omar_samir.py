
from pathlib import Path
import torch
import soundfile as sf

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Hugging Face URLs for the EGTTS model
CONFIG_URL = "https://huggingface.co/OmarSamir/EGTTS-V0.1/resolve/main/config.json"
VOCAB_URL = "https://huggingface.co/OmarSamir/EGTTS-V0.1/resolve/main/vocab.json"
MODEL_URL = "https://huggingface.co/OmarSamir/EGTTS-V0.1/resolve/main/model.pth"
SPEAKER_AUDIO_URL = "https://huggingface.co/OmarSamir/EGTTS-V0.1/resolve/main/speaker_reference.wav"

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "egtts_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def download_if_missing(url: str, path: Path) -> None:
    """Download a file from HF only if it's not already present."""
    if path.exists():
        return
    print(f"Downloading {url} -> {path} ...")
    torch.hub.download_url_to_file(url, path)


def load_egtts(device: str | None = None):
    """Load the OmarSamir/EGTTS-V0.1 XTTS model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = MODEL_DIR / "config.json"
    vocab_path = MODEL_DIR / "vocab.json"
    model_path = MODEL_DIR / "model.pth"
    speaker_ref_path = MODEL_DIR / "speaker_reference.wav"

    # Download all needed files
    download_if_missing(CONFIG_URL, config_path)
    download_if_missing(VOCAB_URL, vocab_path)
    download_if_missing(MODEL_URL, model_path)
    download_if_missing(SPEAKER_AUDIO_URL, speaker_ref_path)

    # Load config
    config = XttsConfig()
    config.load_json(str(config_path))

    print(f"Loading EGTTS model on {device} ...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(model_path),
        use_deepspeed=False,
        vocab_path=str(vocab_path),
        eval=True,
    )
    model.to(device)

    return model, str(speaker_ref_path), device


def synthesize_egtts(
    text: str,
    out_path: str = "egtts_omar_samir_default.wav",
    ref_wav: str | None = None,
    temperature: float = 0.75,
) -> str:
    """Generate Egyptian Arabic speech from text using EGTTS."""
    model, default_ref, device = load_egtts()
    speaker_audio_path = ref_wav or default_ref

    print("Computing speaker latents ...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[speaker_audio_path]
    )

    print("Running inference ...")
    out = model.inference(
        text=text,
        language="ar",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=temperature,
    )

    wav = out["wav"]
    sample_rate = 24000  # as used in the HF Space

    print(f"Saving to {out_path} ...")
    sf.write(out_path, wav, sample_rate)
    print("Done.")
    return out_path


if __name__ == "__main__":
    # You can change this text to something more medical (e.g. discharge summary, instructions, etc.)
    default_text = "أهلاً، هذا اختبار لنظام المساعد الصحي بالمستشفى. كيف أستطيع مساعدتك اليوم؟"

    print("Text:")
    print(default_text)
    synthesize_egtts(default_text, "egtts_egyptian_medical_demo.wav")
    print("\nCheck the file: egtts_egyptian_medical_demo.wav")
