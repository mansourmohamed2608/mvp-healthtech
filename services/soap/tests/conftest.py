# services/soap/tests/conftest.py
"""
Pytest configuration and fixtures for SOAP service tests
"""
import pytest
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Add service root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables"""
    os.environ.setdefault('INTERNAL_SECRET', 'test-secret-for-testing')
    os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost:5432/testdb')
    os.environ.setdefault('LLM_SERVICE_URL', 'http://llm:5001')
    yield


@pytest.fixture
def mock_db_pool():
    """Mock asyncpg database pool"""
    mock_pool = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = AsyncMock()
    mock_pool.execute = AsyncMock()
    mock_pool.fetch = AsyncMock(return_value=[])
    mock_pool.fetchrow = AsyncMock(return_value=None)
    return mock_pool


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for SOAP generation"""
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value={
        'subjective': 'Test subjective',
        'objective': 'Test objective',
        'assessment': 'Test assessment',
        'plan': 'Test plan'
    })
    return mock_client


@pytest.fixture
def sample_transcript():
    """Sample medical transcript"""
    return """
    Doctor: مرحبا، كيف حالك اليوم؟
    Patient: مش كويس يا دكتور، عندي صداع شديد من 3 أيام
    Doctor: فين الصداع بالظبط؟
    Patient: في الجبهة، وبيزيد مع الضوء
    Doctor: تمام، هنعمل فحص ونشوف
    """


@pytest.fixture
def sample_soap_note():
    """Sample SOAP note"""
    return {
        'id': 'soap-test-123',
        'sessionId': 'session-456',
        'patientId': 'P001',
        'practitionerId': 'D001',
        'subjective': 'Patient reports severe headache for 3 days, frontal location, worsens with light',
        'objective': 'Alert and oriented, no fever, BP 120/80',
        'assessment': 'Migraine headache, possible photophobia',
        'plan': '1. Prescribe analgesics\n2. Advise rest in dark room\n3. Follow up if not improved in 3 days',
        'status': 'draft',
        'createdAt': '2025-02-05T10:00:00Z',
        'updatedAt': '2025-02-05T10:00:00Z'
    }
