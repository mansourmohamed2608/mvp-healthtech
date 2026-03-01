# services/asr/tests/conftest.py
"""
Pytest configuration and fixtures for ASR service tests
"""
import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Add service root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables"""
    os.environ.setdefault('INTERNAL_SECRET', 'test-secret-for-testing')
    os.environ.setdefault('DEVICE', 'cpu')
    os.environ.setdefault('WHISPER_MODEL', 'tiny')
    yield


@pytest.fixture
def mock_whisper_model():
    """Mock WhisperX model for testing without GPU"""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        'segments': [{'text': 'test', 'start': 0, 'end': 1}],
        'language': 'ar'
    }
    return mock_model


@pytest.fixture
def mock_diarization_pipeline():
    """Mock pyannote diarization pipeline"""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = MagicMock()
    return mock_pipeline


@pytest.fixture
def sample_audio_bytes():
    """Generate sample audio bytes for testing"""
    import struct
    
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    
    samples = []
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Simple sine wave
        import math
        sample = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack('<h', sample))
    
    audio_data = b''.join(samples)
    
    # WAV header
    wav_header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + len(audio_data),
        b'WAVE',
        b'fmt ',
        16,
        1,  # PCM
        1,  # Mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b'data',
        len(audio_data)
    )
    
    return wav_header + audio_data
