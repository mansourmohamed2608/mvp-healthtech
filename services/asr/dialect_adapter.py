"""
Dialect-Specific LoRA Adapter Manager
Week 5 Day 32 (Oct 26, 2025)
Manages multiple LoRA adapters for Egyptian, Levantine, and Gulf Arabic
"""
import os
from typing import Dict, Optional
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig


class DialectAdapterManager:
    """Manages multiple LoRA adapters for different Arabic dialects"""

    SUPPORTED_DIALECTS = {
        "egyptian": "egy",  # Egyptian Arabic
        "levantine": "lev",  # Levantine (Syrian, Lebanese, Jordanian, Palestinian)
        "gulf": "gulf",      # Gulf Arabic (Saudi, UAE, Kuwait, Qatar)
        "msa": "msa",        # Modern Standard Arabic (fallback)
    }

    def __init__(
        self,
        base_model_path: str = "openai/whisper-large-v2",
        adapters_dir: str = "./lora_ckpt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.base_model_path = base_model_path
        self.adapters_dir = adapters_dir
        self.device = device

        # Load base model and processor
        print(f"Loading base Whisper model from {base_model_path}...")
        self.processor = WhisperProcessor.from_pretrained(base_model_path)

        # Load model without quantization to avoid dtype issues
        # (8-bit quantization causes float16/float32 mismatch)
        self.base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cuda":
            self.base_model = self.base_model.to(device)

        # Cache for loaded adapters
        self.loaded_adapters: Dict[str, PeftModel] = {}

        # Current active adapter
        self.current_dialect: Optional[str] = None
        self.current_model: Optional[PeftModel] = None

        # Load available adapters
        self._discover_adapters()

    def _discover_adapters(self):
        """Discover available LoRA adapters in the adapters directory"""
        if not os.path.exists(self.adapters_dir):
            print(f"Adapters directory {self.adapters_dir} not found. Using base model only.")
            return

        for dialect_name, dialect_code in self.SUPPORTED_DIALECTS.items():
            adapter_path = os.path.join(self.adapters_dir, dialect_code)
            if os.path.exists(adapter_path) and os.path.isdir(adapter_path):
                print(f"Found adapter for {dialect_name} at {adapter_path}")
            else:
                print(f"No adapter found for {dialect_name} (expected at {adapter_path})")

    def load_adapter(self, dialect: str) -> PeftModel:
        """Load a specific dialect adapter"""
        if dialect not in self.SUPPORTED_DIALECTS:
            raise ValueError(
                f"Unsupported dialect: {dialect}. "
                f"Supported: {list(self.SUPPORTED_DIALECTS.keys())}"
            )

        # Check if already loaded
        if dialect in self.loaded_adapters:
            print(f"Using cached adapter for {dialect}")
            return self.loaded_adapters[dialect]

        dialect_code = self.SUPPORTED_DIALECTS[dialect]
        adapter_path = os.path.join(self.adapters_dir, dialect_code)

        # Check if adapter exists
        if not os.path.exists(adapter_path):
            print(f"Adapter for {dialect} not found at {adapter_path}. Using base model.")
            return self.base_model

        # Load adapter
        try:
            print(f"Loading {dialect} adapter from {adapter_path}...")
            model_with_adapter = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                is_trainable=False
            )
            self.loaded_adapters[dialect] = model_with_adapter
            print(f"Successfully loaded {dialect} adapter")
            return model_with_adapter
        except Exception as e:
            print(f"Failed to load adapter for {dialect}: {e}")
            return self.base_model

    def switch_dialect(self, dialect: str):
        """Switch to a specific dialect adapter"""
        model = self.load_adapter(dialect)
        self.current_dialect = dialect
        self.current_model = model
        print(f"Switched to {dialect} dialect")

    def detect_dialect(self, text: str) -> str:
        """
        Simple dialect detection based on keywords
        In production, use a proper dialect classifier
        """
        text_lower = text.lower()

        # Egyptian indicators
        egyptian_markers = ["إزيك", "عامل", "إيه", "أهو", "علشان", "يعني"]
        if any(marker in text_lower for marker in egyptian_markers):
            return "egyptian"

        # Gulf indicators
        gulf_markers = ["شلونك", "وش", "ليش", "زين", "عيل"]
        if any(marker in text_lower for marker in gulf_markers):
            return "gulf"

        # Levantine indicators
        levantine_markers = ["كيفك", "شو", "ليش", "هيك", "مبين"]
        if any(marker in text_lower for marker in levantine_markers):
            return "levantine"

        # Default to MSA
        return "msa"

    def transcribe(
        self,
        audio_input,
        dialect: Optional[str] = None,
        auto_detect: bool = False,
        language: str = "ar"
    ) -> Dict:
        """
        Transcribe audio with dialect-specific adapter

        Args:
            audio_input: Audio array or path
            dialect: Specific dialect to use (or None for base model)
            auto_detect: If True, detect dialect from initial transcription
            language: Target language code

        Returns:
            Dict with text, dialect, and confidence
        """
        # If dialect specified, use that adapter
        if dialect:
            self.switch_dialect(dialect)
            model = self.current_model
        else:
            model = self.base_model

        # Process audio
        inputs = self.processor(
            audio_input,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)

        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                language=language,
                task="transcribe"
            )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Auto-detect dialect if requested
        detected_dialect = None
        if auto_detect:
            detected_dialect = self.detect_dialect(text)
            if detected_dialect != dialect and detected_dialect != "msa":
                # Re-transcribe with detected dialect
                print(f"Detected {detected_dialect}, re-transcribing...")
                self.switch_dialect(detected_dialect)

                with torch.no_grad():
                    generated_ids = self.current_model.generate(
                        inputs.input_features,
                        language=language,
                        task="transcribe"
                    )

                text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

        return {
            "text": text,
            "dialect": dialect or detected_dialect or "base",
            "auto_detected": auto_detect and detected_dialect is not None
        }

    def get_available_dialects(self) -> list:
        """Get list of dialects with loaded adapters"""
        available = []
        for dialect_name, dialect_code in self.SUPPORTED_DIALECTS.items():
            adapter_path = os.path.join(self.adapters_dir, dialect_code)
            if os.path.exists(adapter_path):
                available.append(dialect_name)
        return available

    def unload_adapter(self, dialect: str):
        """Unload a specific adapter to free memory"""
        if dialect in self.loaded_adapters:
            del self.loaded_adapters[dialect]
            print(f"Unloaded {dialect} adapter")

    def clear_cache(self):
        """Clear all loaded adapters"""
        self.loaded_adapters.clear()
        self.current_dialect = None
        self.current_model = None
        print("Cleared all adapter cache")


# Global instance
dialect_manager: Optional[DialectAdapterManager] = None


def get_dialect_manager() -> DialectAdapterManager:
    """Get or create global dialect manager instance"""
    global dialect_manager
    if dialect_manager is None:
        dialect_manager = DialectAdapterManager()
    return dialect_manager
