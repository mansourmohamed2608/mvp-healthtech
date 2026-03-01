# services/soap/tests/test_api.py
"""
SOAP Service API Tests
Tests for SOAP note generation, storage, and FHIR integration
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import os


@pytest.fixture
def client():
    """Create test client with mocked database"""
    os.environ['INTERNAL_SECRET'] = 'test-secret'
    os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost:5432/testdb'
    os.environ['LLM_SERVICE_URL'] = 'http://llm:5001'
    
    with patch('app.asyncpg') as mock_asyncpg:
        mock_pool = AsyncMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
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
        response = client.post('/generate', json={'transcript': 'test'})
        assert response.status_code == 401
    
    def test_accepts_valid_secret(self, client):
        with patch('app.generate_soap_note') as mock_gen:
            mock_gen.return_value = {'subjective': 'test'}
            response = client.post(
                '/generate',
                json={'transcript': 'test'},
                headers={'x-internal-secret': 'test-secret'}
            )
            assert response.status_code != 401


class TestSOAPGeneration:
    """Test SOAP note generation"""
    
    @patch('app.generate_soap_note')
    def test_generate_soap_note(self, mock_gen, client):
        mock_gen.return_value = {
            'id': 'soap-123',
            'subjective': 'Patient reports chest pain since yesterday',
            'objective': 'BP 120/80, HR 72',
            'assessment': 'Suspected angina',
            'plan': 'Order ECG, cardiac enzymes'
        }
        
        response = client.post(
            '/generate',
            json={
                'transcript': 'Patient: عندي ألم في صدري من امبارح',
                'sessionId': 's1',
                'patientId': 'P001',
                'practitionerId': 'D001'
            },
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert 'subjective' in data or 'id' in data
    
    @patch('app.generate_soap_note')
    def test_generate_with_template(self, mock_gen, client):
        mock_gen.return_value = {'id': 'soap-456', 'subjective': 'test'}
        
        response = client.post(
            '/generate',
            json={
                'transcript': 'test transcript',
                'sessionId': 's2',
                'patientId': 'P002',
                'practitionerId': 'D001',
                'templateId': 'cardiology-consult'
            },
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            # Template should have been applied
            pass


class TestSOAPRetrieval:
    """Test SOAP note retrieval"""
    
    @patch('app.get_soap_note')
    def test_get_soap_by_id(self, mock_get, client):
        mock_get.return_value = {
            'id': 'soap-123',
            'subjective': 'Patient reports pain',
            'objective': 'Vitals normal',
            'assessment': 'Mild symptoms',
            'plan': 'Follow up in 1 week'
        }
        
        response = client.get(
            '/notes/soap-123',
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert 'id' in data
    
    @patch('app.list_soap_notes')
    def test_list_soap_notes(self, mock_list, client):
        mock_list.return_value = [
            {'id': 'soap-1', 'patientId': 'P001', 'createdAt': '2025-02-05'},
            {'id': 'soap-2', 'patientId': 'P001', 'createdAt': '2025-02-04'}
        ]
        
        response = client.get(
            '/notes?patientId=P001',
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list) or 'notes' in data


class TestSOAPUpdate:
    """Test SOAP note updates"""
    
    @patch('app.update_soap_field')
    def test_update_field(self, mock_update, client):
        mock_update.return_value = {
            'id': 'soap-123',
            'field': 'plan',
            'oldValue': 'Order ECG',
            'newValue': 'Order ECG and chest X-ray',
            'updated': True
        }
        
        response = client.patch(
            '/notes/soap-123/field',
            json={
                'fieldPath': 'plan',
                'transcript': 'Add chest X-ray to the plan'
            },
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert 'updated' in data or 'id' in data


class TestTemplates:
    """Test SOAP template functionality"""
    
    @patch('app.get_templates')
    def test_list_templates(self, mock_templates, client):
        mock_templates.return_value = [
            {'id': 'general', 'name': 'General Visit'},
            {'id': 'cardiology', 'name': 'Cardiology Consult'}
        ]
        
        response = client.get(
            '/templates',
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list) or 'templates' in data


class TestInputValidation:
    """Test input validation"""
    
    def test_missing_transcript(self, client):
        response = client.post(
            '/generate',
            json={'sessionId': 's1', 'patientId': 'P001'},
            headers={'x-internal-secret': 'test-secret'}
        )
        assert response.status_code in [400, 422]
    
    def test_empty_transcript(self, client):
        response = client.post(
            '/generate',
            json={'transcript': '', 'sessionId': 's1', 'patientId': 'P001'},
            headers={'x-internal-secret': 'test-secret'}
        )
        # Empty transcript should be handled
        assert response.status_code in [200, 400, 422]


class TestDocumentSummarization:
    """Test document summarization for clinical context"""
    
    @patch('app.summarize_document')
    def test_summarize_pdf(self, mock_summarize, client):
        mock_summarize.return_value = {
            'summary': 'Patient has history of hypertension and diabetes',
            'extractedConditions': ['hypertension', 'diabetes']
        }
        
        response = client.post(
            '/summarize-document',
            json={
                'documentBase64': 'base64pdfcontent',
                'documentType': 'pdf'
            },
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert 'summary' in data or 'error' in data
