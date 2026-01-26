#!/usr/bin/env python3
"""
Download VAD model from HuggingFace if not already present.
The whisperx S3 bucket is broken/restricted, so we use pyannote/segmentation from HuggingFace.
"""
import os
import shutil
import sys

def download_vad_model():
    torch_home = os.environ.get("TORCH_HOME", "/root/.cache/torch")
    vad_model_path = os.path.join(torch_home, "hub", "whisperx-vad-segmentation.bin")
    
    if os.path.exists(vad_model_path):
        size = os.path.getsize(vad_model_path)
        if size > 1_000_000:  # Should be ~18MB, not 473 bytes
            print(f"VAD model already exists at {vad_model_path} ({size} bytes)")
            return vad_model_path
        else:
            print(f"VAD model file is invalid ({size} bytes), re-downloading...")
            os.remove(vad_model_path)
    
    # Get HuggingFace token
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HUGGINGFACE_HUB_TOKEN required for VAD model download")
        print("The pyannote/segmentation model is gated and requires authentication.")
        print("Get a token from: https://huggingface.co/settings/tokens")
        print("And accept the model license at: https://huggingface.co/pyannote/segmentation")
        sys.exit(1)
    
    print("Downloading VAD model from HuggingFace (pyannote/segmentation)...")
    os.makedirs(os.path.dirname(vad_model_path), exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id="pyannote/segmentation",
            filename="pytorch_model.bin",
            token=token
        )
        shutil.copy(downloaded_path, vad_model_path)
        size = os.path.getsize(vad_model_path)
        print(f"VAD model downloaded to {vad_model_path} ({size} bytes)")
        return vad_model_path
    except Exception as e:
        print(f"ERROR downloading VAD model: {e}")
        print("Make sure you have accepted the model license at:")
        print("https://huggingface.co/pyannote/segmentation")
        sys.exit(1)

if __name__ == "__main__":
    path = download_vad_model()
    # Write path to file for app.py to read
    with open("/tmp/vad_model_path.txt", "w") as f:
        f.write(path)
    print(f"VAD_MODEL_PATH={path}")
