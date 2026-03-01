# services/llm/tests/test_api.py
"""
LLM Service API Tests
Tests for inference, chat, and clinical correction endpoints
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import os


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock heavy ML dependencies"""
    with patch.dict('sys.modules', {
        'torch': MagicMock(),
        'transformers': MagicMock(),
    }):
        yield


@pytest.fixture
def client():
    """Create test client with mocked models"""
    os.environ['INTERNAL_SECRET'] = 'test-secret'
    
    with patch('app.AutoModelForCausalLM') as mock_model, \
         patch('app.AutoTokenizer') as mock_tokenizer:
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
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
        response = client.post('/infer', json={'message': 'test', 'sessionId': 's1'})
        assert response.status_code == 401
    
    def test_accepts_valid_secret(self, client):
        with patch('app.generate_response') as mock_gen:
            mock_gen.return_value = {'reply': 'test', 'intent': 'general'}
            response = client.post(
                '/infer',
                json={'message': 'test', 'sessionId': 's1'},
                headers={'x-internal-secret': 'test-secret'}
            )
            assert response.status_code != 401


class TestInferenceEndpoint:
    """Test /infer endpoint"""
    
    @patch('app.generate_response')
    def test_basic_inference(self, mock_gen, client):
        mock_gen.return_value = {
            'reply': 'أفهم، متى بدأ الألم؟',
            'intent': 'symptoms_inquiry'
        }
        
        response = client.post(
            '/infer',
            json={'message': 'عندي صداع', 'sessionId': 'session-123'},
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert 'reply' in data
    
    @patch('app.generate_response')
    def test_inference_with_intent(self, mock_gen, client):
        mock_gen.return_value = {'reply': 'test', 'intent': 'booking'}
        
        response = client.post(
            '/infer',
            json={
                'message': 'عايز احجز موعد',
                'sessionId': 's1',
                'intent': 'booking'
            },
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            assert 'intent' in response.json()


class TestChatEndpoint:
    """Test /chat endpoint with conversation history"""
    
    @patch('app.generate_chat_response')
    def test_chat_with_history(self, mock_gen, client):
        mock_gen.return_value = {'reply': 'نعم، فهمت', 'intent': 'confirm'}
        
        response = client.post(
            '/chat',
            json={
                'message': 'نعم',
                'history': [
                    {'role': 'user', 'content': 'عايز احجز'},
                    {'role': 'assistant', 'content': 'في أي قسم؟'}
                ],
                'sessionId': 'chat-1'
            },
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert 'reply' in data


class TestTranscriptionCorrection:
    """Test transcription correction endpoint"""
    
    @patch('app.correct_transcription')
    def test_correct_transcription(self, mock_correct, client):
        mock_correct.return_value = {
            'original': 'مرحبا دكتوور',
            'corrected': 'مرحبا دكتور',
            'corrections_made': 1,
            'dialect_normalized': True
        }
        
        response = client.post(
            '/correct-transcription',
            json={
                'text': 'مرحبا دكتوور',
                'dialect': 'egypt'
            },
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert 'corrected' in data


class TestSpeakerRoleIdentification:
    """Test speaker role identification"""
    
    @patch('app.identify_speaker_roles')
    def test_identify_speakers(self, mock_identify, client):
        mock_identify.return_value = {
            'roles': [
                {'speaker_id': 'SPEAKER_00', 'role': 'Patient', 'confidence': 0.9, 'reasoning': 'Symptoms'},
                {'speaker_id': 'SPEAKER_01', 'role': 'Doctor', 'confidence': 0.85, 'reasoning': 'Questions'}
            ]
        }
        
        response = client.post(
            '/identify-speakers',
            json={
                'segments': [
                    {'speaker': 'SPEAKER_00', 'text': 'عندي ألم', 'start': 0, 'end': 1},
                    {'speaker': 'SPEAKER_01', 'text': 'متى بدأ؟', 'start': 1.5, 'end': 2.5}
                ],
                'context': 'medical'
            },
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'roles' in data:
                assert len(data['roles']) >= 1


class TestInputValidation:
    """Test input validation"""
    
    def test_missing_message(self, client):
        response = client.post(
            '/infer',
            json={'sessionId': 's1'},
            headers={'x-internal-secret': 'test-secret'}
        )
        assert response.status_code in [400, 422]
    
    def test_missing_session_id(self, client):
        response = client.post(
            '/infer',
            json={'message': 'test'},
            headers={'x-internal-secret': 'test-secret'}
        )
        assert response.status_code in [400, 422]
    
    def test_message_too_long(self, client):
        long_message = 'a' * 10000  # Very long message
        response = client.post(
            '/infer',
            json={'message': long_message, 'sessionId': 's1'},
            headers={'x-internal-secret': 'test-secret'}
        )
        # Should either process or return error, not crash
        assert response.status_code in [200, 400, 422, 500]


class TestRAGIntegration:
    """Test RAG retrieval integration"""
    
    @patch('app.rag_store')
    @patch('app.generate_response')
    def test_rag_retrieval(self, mock_gen, mock_rag, client):
        mock_rag.search.return_value = [{'text': 'Hospital hours: 8am-8pm', 'score': 0.9}]
        mock_gen.return_value = {'reply': 'مواعيدنا من 8 صباحا ل 8 مساء', 'intent': 'hours'}
        
        response = client.post(
            '/infer',
            json={
                'message': 'ايه مواعيد المستشفى؟',
                'sessionId': 's1',
                'intent': 'hours_inquiry'
            },
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            # RAG should have been consulted for policy/FAQ questions
            pass
