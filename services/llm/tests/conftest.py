# services/llm/tests/conftest.py
"""
Pytest configuration and fixtures for LLM service tests
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
    os.environ.setdefault('LLM_MODEL', 'test-model')
    yield


@pytest.fixture
def mock_llm_model():
    """Mock LLM model for testing without GPU"""
    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock()
    return mock_model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer"""
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_tok.decode.return_value = "مرحبا"
    return mock_tok


@pytest.fixture
def sample_conversation():
    """Sample conversation history"""
    return [
        {"role": "user", "content": "مرحبا"},
        {"role": "assistant", "content": "أهلا، كيف يمكنني مساعدتك؟"},
        {"role": "user", "content": "عايز احجز موعد"},
    ]


@pytest.fixture
def sample_segments():
    """Sample diarized segments for speaker role ID"""
    return [
        {"speaker": "SPEAKER_00", "text": "مرحبا دكتور", "start": 0.0, "end": 1.5},
        {"speaker": "SPEAKER_01", "text": "أهلا وسهلا، اتفضل", "start": 1.6, "end": 3.0},
        {"speaker": "SPEAKER_00", "text": "عندي ألم في صدري من امبارح", "start": 3.2, "end": 6.0},
        {"speaker": "SPEAKER_01", "text": "فين الألم بالظبط؟", "start": 6.2, "end": 7.5},
    ]
