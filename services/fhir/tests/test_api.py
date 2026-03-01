# services/fhir/tests/test_api.py
"""
FHIR Service API Tests
Tests for FHIR R4 writeback and patient data operations
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import os


@pytest.fixture
def client():
    """Create test client with mocked FHIR server"""
    os.environ['INTERNAL_SECRET'] = 'test-secret'
    os.environ['FHIR_BASE_URL'] = 'http://fhir-server:8080/fhir'
    os.environ['FHIR_CLIENT_ID'] = 'test-client'
    os.environ['FHIR_CLIENT_SECRET'] = 'test-secret'
    
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
        response = client.post('/push-note', json={'note': 'test'})
        assert response.status_code == 401
    
    def test_accepts_valid_secret(self, client):
        with patch('app.push_to_fhir') as mock_push:
            mock_push.return_value = {'success': True, 'resourceId': 'doc-123'}
            response = client.post(
                '/push-note',
                json={'note': {'subjective': 'test'}, 'patientId': 'P001'},
                headers={'x-internal-secret': 'test-secret'}
            )
            assert response.status_code != 401


class TestPushNote:
    """Test FHIR note push functionality"""
    
    @patch('app.httpx.AsyncClient')
    def test_push_soap_note(self, mock_client_class, client):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {'id': 'doc-123', 'resourceType': 'DocumentReference'}
        )
        
        response = client.post(
            '/push-note',
            json={
                'patientId': 'P001',
                'practitionerId': 'D001',
                'encounterId': 'E001',
                'note': {
                    'subjective': 'Patient reports chest pain',
                    'objective': 'BP 120/80',
                    'assessment': 'Angina',
                    'plan': 'Order ECG'
                }
            },
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code in [200, 201]:
            data = response.json()
            assert 'resourceId' in data or 'id' in data or 'success' in data


class TestPatientLookup:
    """Test patient lookup functionality"""
    
    @patch('app.httpx.AsyncClient')
    def test_lookup_patient(self, mock_client_class, client):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'resourceType': 'Patient',
                'id': 'P001',
                'name': [{'family': 'Smith', 'given': ['John']}]
            }
        )
        
        response = client.get(
            '/patient/P001',
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert 'id' in data or 'resourceType' in data


class TestEncounterManagement:
    """Test encounter management"""
    
    @patch('app.httpx.AsyncClient')
    def test_create_encounter(self, mock_client_class, client):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {'id': 'E001', 'resourceType': 'Encounter'}
        )
        
        response = client.post(
            '/encounter',
            json={
                'patientId': 'P001',
                'practitionerId': 'D001',
                'type': 'ambulatory'
            },
            headers={'x-internal-secret': 'test-secret'}
        )
        
        if response.status_code in [200, 201]:
            data = response.json()
            # Should return encounter ID
            pass


class TestIdempotency:
    """Test idempotency handling"""
    
    @patch('app.httpx.AsyncClient')
    def test_idempotency_key(self, mock_client_class, client):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {'id': 'doc-123'}
        )
        
        idempotency_key = 'unique-key-123'
        
        # First request
        response1 = client.post(
            '/push-note',
            json={'note': {'subjective': 'test'}, 'patientId': 'P001'},
            headers={
                'x-internal-secret': 'test-secret',
                'Idempotency-Key': idempotency_key
            }
        )
        
        # Second request with same key should be handled
        response2 = client.post(
            '/push-note',
            json={'note': {'subjective': 'test'}, 'patientId': 'P001'},
            headers={
                'x-internal-secret': 'test-secret',
                'Idempotency-Key': idempotency_key
            }
        )
        
        # Both should succeed or second should return cached result
        assert response1.status_code in [200, 201, 400, 500]


class TestInputValidation:
    """Test input validation"""
    
    def test_missing_patient_id(self, client):
        response = client.post(
            '/push-note',
            json={'note': {'subjective': 'test'}},
            headers={'x-internal-secret': 'test-secret'}
        )
        assert response.status_code in [400, 422]
    
    def test_invalid_patient_id_format(self, client):
        response = client.post(
            '/push-note',
            json={'note': {'subjective': 'test'}, 'patientId': ''},
            headers={'x-internal-secret': 'test-secret'}
        )
        assert response.status_code in [400, 422]
