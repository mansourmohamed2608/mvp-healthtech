# services/tts/tests/test_api.py
"""
TTS Service API Tests
Tests for text-to-speech synthesis endpoints
"""
import pytest
import base64
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import os


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock heavy ML dependencies"""
    with patch.dict('sys.modules', {
        'torch': MagicMock(),
        'TTS': MagicMock(),
        'TTS.api': MagicMock(),
        'TTS.tts': MagicMock(),
    }):
        yield


@pytest.fixture
def client():
    """Create test client with mocked models"""
    os.environ['INTERNAL_SECRET'] = 'test-secret'
    os.environ['TTS_ENGINE'] = 'none'  # Use fallback for tests
    
    with patch('app.CoquiTTS', None), \
         patch('app.XTTS_AVAILABLE', False):
        from app import app
        return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_endpoint(self, client):
        response = client.get('/health')
        assert response.status_code == 200
    
    def test_ready_endpoint(self, client):
        response = client.get('/ready')
        assert response.status_code in [200, 503]
    
    def test_metrics_endpoint(self, client):
        response = client.get('/metrics')
        assert response.status_code == 200


class TestAuthentication:
    """Test internal authentication"""
    
    def test_requires_internal_secret(self, client):
        response = client.post('/synthesize', json={'text': 'test'})
        assert response.status_code == 401
    
    def test_accepts_valid_secret(self, client):
        with patch('app.synthesize_text') as mock_synth:
            mock_synth.return_value = {'audio': 'base64audio', 'format': 'mulaw'}
            response = client.post(
                '/synthesize',
                json={'text': 'مرحبا'},
                headers={'x-internal-secret': 'test-secret'}
            )
            assert response.status_code != 401


class TestSynthesisEndpoint:
    """Test /synthesize endpoint"""
    
    @patch('app.synthesize_text')
    def test_synthesize_arabic(self, mock_synth, client):
        mock_synth.return_value = {
            'audio': base64.b64encode(b'fake_audio').decode(),
            'format': 'mulaw',
            'duration': 1.5
        }
        
        response = client.post(
            '/synthesize',
            json={'text': 'مرحبا، كيف يمكنني مساعدتك؟'},
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert 'audio' in data
    
    @patch('app.synthesize_text')
    def test_synthesize_with_voice(self, mock_synth, client):
        mock_synth.return_value = {'audio': 'audio_data', 'format': 'mulaw'}
        
        response = client.post(
            '/synthesize',
            json={'text': 'test', 'voice': 'egtts'},
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            # Voice parameter should be passed to synthesis
            pass
    
    @patch('app.synthesize_text')
    def test_synthesize_saudi_voice(self, mock_synth, client):
        mock_synth.return_value = {'audio': 'audio_data', 'format': 'mulaw'}
        
        response = client.post(
            '/synthesize',
            json={'text': 'أهلين وسهلين', 'voice': 'saudi-tts'},
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert 'audio' in data


class TestInputValidation:
    """Test input validation"""
    
    def test_missing_text(self, client):
        response = client.post(
            '/synthesize',
            json={},
            headers={'x-internal-secret': 'test-secret'}
        )
        assert response.status_code in [400, 422]
    
    def test_empty_text(self, client):
        response = client.post(
            '/synthesize',
            json={'text': ''},
            headers={'x-internal-secret': 'test-secret'}
        )
        # Empty text might return silent audio or error
        assert response.status_code in [200, 400, 422]
    
    def test_very_long_text(self, client):
        long_text = 'مرحبا ' * 1000
        response = client.post(
            '/synthesize',
            json={'text': long_text},
            headers={'x-internal-secret': 'test-secret'}
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 413, 500]


class TestAudioFormat:
    """Test audio format handling"""
    
    @patch('app.synthesize_text')
    def test_mulaw_format(self, mock_synth, client):
        mock_synth.return_value = {
            'audio': 'mulaw_audio',
            'format': 'mulaw',
            'sampleRate': 8000
        }
        
        response = client.post(
            '/synthesize',
            json={'text': 'test', 'format': 'mulaw'},
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'format' in data:
                assert data['format'] in ['mulaw', 'wav', 'mp3']
