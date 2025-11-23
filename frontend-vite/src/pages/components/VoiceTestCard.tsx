import api from '../../utils/api';
import { useState } from 'react';

export default function VoiceTestCard() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const testVoice = async () => {
    setLoading(true);
    setError(null);
    try {
      const testAudio = 'UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=';
      const asr = await api.transcribeAudio(testAudio, 'test-call-sid');
      const llm = await api.inferMessage('Hello, how are you?', 'test-session', 'greeting');
      const tts = await api.synthesizeSpeech('مرحبا، كيف حالك؟', 'ar-EG-SalmaNeural');
      setResult({ asr, llm, tts: { ...tts, audio: (tts.audio || '').substring(0, 50) + '...' } });
    } catch (e: any) {
      setError(e?.message || String(e));
    }
    setLoading(false);
  };

  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <h3 className="font-semibold text-gray-900 mb-2">Voice (ASR → LLM → TTS)</h3>
      <button
        onClick={testVoice}
        disabled={loading}
        className="w-full px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:bg-gray-400"
      >
        {loading ? 'Running...' : 'Run Voice Test'}
      </button>
      {error && <p className="text-sm text-red-600 mt-2">{error}</p>}
      {result && (
        <pre className="bg-gray-50 p-3 rounded text-sm overflow-x-auto mt-2">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}
