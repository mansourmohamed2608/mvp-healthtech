import { useState } from 'react';
import api from '../utils/api';

interface ServiceStatus {
  name: string;
  status: 'checking' | 'online' | 'offline' | 'error';
  message?: string;
  url?: string;
}

export default function ServiceTest() {
  const [services, setServices] = useState<ServiceStatus[]>([
    { name: 'ASR Service', status: 'checking', url: 'http://localhost:5000' },
    { name: 'LLM Service', status: 'checking', url: 'http://localhost:5001' },
    { name: 'TTS Service', status: 'checking', url: 'http://localhost:5002' },
    { name: 'SOAP Service', status: 'checking', url: 'http://localhost:5003' },
    { name: 'FHIR Service', status: 'checking', url: 'http://localhost:5004' },
  ]);

  const [testResults, setTestResults] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(false);

  // Check all service health
  const checkAllServices = async () => {
    setLoading(true);
    const results: ServiceStatus[] = [];

    // ASR
    try {
      await api.checkASRHealth();
      results.push({ name: 'ASR Service', status: 'online', message: 'Service is healthy', url: 'http://localhost:5000' });
    } catch (error) {
      results.push({ name: 'ASR Service', status: 'offline', message: String(error), url: 'http://localhost:5000' });
    }

    // LLM
    try {
      await api.checkLLMHealth();
      results.push({ name: 'LLM Service', status: 'online', message: 'Service is healthy', url: 'http://localhost:5001' });
    } catch (error) {
      results.push({ name: 'LLM Service', status: 'offline', message: String(error), url: 'http://localhost:5001' });
    }

    // TTS
    try {
      await api.checkTTSHealth();
      results.push({ name: 'TTS Service', status: 'online', message: 'Service is healthy', url: 'http://localhost:5002' });
    } catch (error) {
      results.push({ name: 'TTS Service', status: 'offline', message: String(error), url: 'http://localhost:5002' });
    }

    // SOAP
    try {
      await api.checkSOAPHealth();
      results.push({ name: 'SOAP Service', status: 'online', message: 'Service is healthy', url: 'http://localhost:5003' });
    } catch (error) {
      results.push({ name: 'SOAP Service', status: 'offline', message: String(error), url: 'http://localhost:5003' });
    }

    // FHIR
    try {
      await api.checkFHIRHealth();
      results.push({ name: 'FHIR Service', status: 'online', message: 'Service is healthy', url: 'http://localhost:5004' });
    } catch (error) {
      results.push({ name: 'FHIR Service', status: 'offline', message: String(error), url: 'http://localhost:5004' });
    }

    setServices(results);
    setLoading(false);
  };

  // Test ASR
  const testASR = async () => {
    setLoading(true);
    try {
      // Create a simple audio test (silence as base64)
      const testAudio = 'UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=';
      const result = await api.transcribeAudio(testAudio, 'test-call-sid');
      setTestResults(prev => ({ ...prev, asr: { success: true, result } }));
    } catch (error) {
      setTestResults(prev => ({ ...prev, asr: { success: false, error: String(error) } }));
    }
    setLoading(false);
  };

  // Test LLM
  const testLLM = async () => {
    setLoading(true);
    try {
      const result = await api.inferMessage('Hello, how are you?', 'test-session', 'greeting');
      setTestResults(prev => ({ ...prev, llm: { success: true, result } }));
    } catch (error) {
      setTestResults(prev => ({ ...prev, llm: { success: false, error: String(error) } }));
    }
    setLoading(false);
  };

  // Test TTS
  const testTTS = async () => {
    setLoading(true);
    try {
      const result = await api.synthesizeSpeech('مرحبا، كيف حالك؟', 'ar-EG-SalmaNeural');
      setTestResults(prev => ({ ...prev, tts: { success: true, result: { ...result, audio: result.audio.substring(0, 50) + '...' } } }));
    } catch (error) {
      setTestResults(prev => ({ ...prev, tts: { success: false, error: String(error) } }));
    }
    setLoading(false);
  };

  // Test SOAP
  const testSOAP = async () => {
    setLoading(true);
    try {
      const transcript = 'Patient complains of headache for 3 days. Temperature 37.5°C. Prescribed ibuprofen.';
      const result = await api.createSOAPNote({
        transcript,
        sessionId: 'test-session',
      });
      setTestResults(prev => ({ ...prev, soap: { success: true, result } }));
    } catch (error) {
      setTestResults(prev => ({ ...prev, soap: { success: false, error: String(error) } }));
    }
    setLoading(false);
  };

  // Test FHIR
  const testFHIR = async () => {
    setLoading(true);
    try {
      const testPatient = {
        resourceType: 'Patient',
        name: [{ given: ['Test'], family: 'Patient' }],
      };
      const result = await api.createFHIRResource('Patient', testPatient);
      setTestResults(prev => ({ ...prev, fhir: { success: true, result } }));
    } catch (error) {
      setTestResults(prev => ({ ...prev, fhir: { success: false, error: String(error) } }));
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 mb-8">
          🔧 Service Connection Test
        </h1>

        {/* Service Status */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold text-gray-800">Service Health Status</h2>
            <button
              onClick={checkAllServices}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
            >
              {loading ? 'Checking...' : 'Check All Services'}
            </button>
          </div>

          <div className="space-y-3">
            {services.map((service) => (
              <div
                key={service.name}
                className="flex items-center justify-between p-4 border border-gray-200 rounded-lg"
              >
                <div className="flex items-center space-x-3">
                  <div
                    className={`w-3 h-3 rounded-full ${
                      service.status === 'online'
                        ? 'bg-green-500'
                        : service.status === 'offline'
                        ? 'bg-red-500'
                        : service.status === 'checking'
                        ? 'bg-yellow-500 animate-pulse'
                        : 'bg-gray-400'
                    }`}
                  />
                  <div>
                    <p className="font-medium text-gray-900">{service.name}</p>
                    <p className="text-sm text-gray-500">{service.url}</p>
                  </div>
                </div>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${
                    service.status === 'online'
                      ? 'bg-green-100 text-green-800'
                      : service.status === 'offline'
                      ? 'bg-red-100 text-red-800'
                      : 'bg-yellow-100 text-yellow-800'
                  }`}
                >
                  {service.status}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Functional Tests */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">Functional Tests</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* ASR Test */}
            <div className="border border-gray-200 rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-2">ASR (Speech Recognition)</h3>
              <button
                onClick={testASR}
                disabled={loading}
                className="w-full px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:bg-gray-400"
              >
                Test Transcription
              </button>
            </div>

            {/* LLM Test */}
            <div className="border border-gray-200 rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-2">LLM (Language Model)</h3>
              <button
                onClick={testLLM}
                disabled={loading}
                className="w-full px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:bg-gray-400"
              >
                Test Inference
              </button>
            </div>

            {/* TTS Test */}
            <div className="border border-gray-200 rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-2">TTS (Text-to-Speech)</h3>
              <button
                onClick={testTTS}
                disabled={loading}
                className="w-full px-4 py-2 bg-pink-600 text-white rounded hover:bg-pink-700 disabled:bg-gray-400"
              >
                Test Synthesis
              </button>
            </div>

            {/* SOAP Test */}
            <div className="border border-gray-200 rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-2">SOAP Notes</h3>
              <button
                onClick={testSOAP}
                disabled={loading}
                className="w-full px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-400"
              >
                Test SOAP Generation
              </button>
            </div>

            {/* FHIR Test */}
            <div className="border border-gray-200 rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-2">FHIR Integration</h3>
              <button
                onClick={testFHIR}
                disabled={loading}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-400"
              >
                Test FHIR Resource
              </button>
            </div>
          </div>
        </div>

        {/* Test Results */}
        {Object.keys(testResults).length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Test Results</h2>
            <div className="space-y-4">
              {Object.entries(testResults).map(([service, result]: [string, any]) => (
                <div key={service} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-gray-900 uppercase">{service}</h3>
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-medium ${
                        result.success
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}
                    >
                      {result.success ? 'Success' : 'Failed'}
                    </span>
                  </div>
                  <pre className="bg-gray-50 p-3 rounded text-sm overflow-x-auto">
                    {JSON.stringify(result.success ? result.result : result.error, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Configuration Info */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mt-8">
          <h3 className="font-semibold text-blue-900 mb-2">ℹ️ Configuration</h3>
          <p className="text-blue-800 text-sm mb-2">
            <strong>Mode:</strong> {import.meta.env.VITE_USE_DIRECT_SERVICES === 'true' ? 'Direct Services' : 'Gateway'}
          </p>
          <p className="text-blue-800 text-sm">
            <strong>API URL:</strong> {import.meta.env.VITE_API_URL || 'http://localhost:3001'}
          </p>
          {import.meta.env.VITE_USE_DIRECT_SERVICES === 'true' && (
            <div className="mt-2 text-sm text-blue-800">
              <p><strong>Direct Service URLs:</strong></p>
              <ul className="list-disc list-inside ml-2">
                <li>ASR: {import.meta.env.VITE_ASR_URL || 'http://localhost:5000'}</li>
                <li>LLM: {import.meta.env.VITE_LLM_URL || 'http://localhost:5001'}</li>
                <li>TTS: {import.meta.env.VITE_TTS_URL || 'http://localhost:5002'}</li>
                <li>SOAP: {import.meta.env.VITE_SOAP_URL || 'http://localhost:5003'}</li>
                <li>FHIR: {import.meta.env.VITE_FHIR_URL || 'http://localhost:5004'}</li>
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
