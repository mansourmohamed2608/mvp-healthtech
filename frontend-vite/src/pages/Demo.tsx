import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { useThemeStore } from '@store/themeStore';
import {
  IconMicrophone,
  IconBrain,
  IconFileText,
  IconDatabase,
  IconVolume,
  IconCheck,
  IconX,
  IconLoader,
  IconPlayerStop,
  IconPlayerPlay
} from '@tabler/icons-react';
import api from '@utils/api';const Demo = () => {
  const { language } = useThemeStore();
  const [activeTab, setActiveTab] = useState('asr');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // ASR State
  const [asrDialect, setAsrDialect] = useState('egyptian');
  const [isRecording, setIsRecording] = useState(false);
  const [asrAudioURL, setAsrAudioURL] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // LLM State
  const [llmMessage, setLlmMessage] = useState('What are the symptoms of diabetes?');
  const [sessionId] = useState(() => `session-${Date.now()}`);

  // SOAP State
  const [soapTranscript, setSoapTranscript] = useState(
    'Patient: I have been experiencing chest pain for the past 3 days, especially when I exert myself.\n' +
    'Doctor: Can you describe the pain?\n' +
    'Patient: It feels like pressure and sometimes radiates to my left arm.\n' +
    'Doctor: Any shortness of breath?\n' +
    'Patient: Yes, especially when climbing stairs.\n' +
    'Doctor: I see. Let me check your vitals. Blood pressure is 140/90, heart rate 88, temperature 98.6°F. Chest is clear to auscultation.\n' +
    'Doctor: Based on your symptoms, this could be angina. We need to rule out myocardial infarction.\n' +
    'Doctor: I will order an ECG and cardiac enzyme tests. If abnormal, I will refer you to cardiology.'
  );

  // FHIR State
  const [fhirResource, setFhirResource] = useState('Patient');
  const [fhirData, setFhirData] = useState({
    name: 'John Doe',
    gender: 'male',
    birthDate: '1980-01-01'
  });

  // TTS State
  const [ttsText, setTtsText] = useState('Hello, this is a test of text to speech');
  const [ttsAudioURL, setTtsAudioURL] = useState<string | null>(null);

  const tabs = [
    { id: 'asr', label: language === 'ar' ? 'تحويل الصوت' : 'Voice to Text', icon: <IconMicrophone size={20} /> },
    { id: 'llm', label: language === 'ar' ? 'مساعد AI' : 'AI Assistant', icon: <IconBrain size={20} /> },
    { id: 'soap', label: language === 'ar' ? 'ملاحظات SOAP' : 'SOAP Notes', icon: <IconFileText size={20} /> },
    { id: 'fhir', label: language === 'ar' ? 'تكامل FHIR' : 'FHIR Integration', icon: <IconDatabase size={20} /> },
    { id: 'tts', label: language === 'ar' ? 'تحويل النص' : 'Text to Speech', icon: <IconVolume size={20} /> },
  ];

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(audioBlob);
        setAsrAudioURL(url);

        // Convert to WAV format using Web Audio API
        try {
          const arrayBuffer = await audioBlob.arrayBuffer();
          const audioContext = new AudioContext({ sampleRate: 16000 });
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

          // Convert to WAV
          const wavBlob = await audioBufferToWav(audioBuffer);

          // Convert WAV to base64
          const reader = new FileReader();
          reader.readAsDataURL(wavBlob);
          reader.onloadend = async () => {
            const base64Audio = reader.result?.toString().split(',')[1];
            if (base64Audio) {
              await handleASRTest(base64Audio);
            }
          };
        } catch (err) {
          console.error('Audio conversion error:', err);
          setError('Failed to convert audio format. Please try again.');
        }

        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      setError('Microphone access denied. Please allow microphone permissions.');
    }
  };

  // Helper function to convert AudioBuffer to WAV format
  const audioBufferToWav = async (audioBuffer: AudioBuffer): Promise<Blob> => {
    const numberOfChannels = 1; // Mono
    const length = audioBuffer.length * numberOfChannels * 2;
    const buffer = new ArrayBuffer(44 + length);
    const view = new DataView(buffer);
    const channels: Float32Array[] = [];
    let offset = 0;
    let pos = 0;

    // Write WAV header
    const setUint16 = (data: number) => {
      view.setUint16(pos, data, true);
      pos += 2;
    };
    const setUint32 = (data: number) => {
      view.setUint32(pos, data, true);
      pos += 4;
    };

    // "RIFF" chunk descriptor
    setUint32(0x46464952); // "RIFF"
    setUint32(36 + length); // File size - 8
    setUint32(0x45564157); // "WAVE"

    // "fmt " sub-chunk
    setUint32(0x20746d66); // "fmt "
    setUint32(16); // Subchunk1Size (16 for PCM)
    setUint16(1); // AudioFormat (1 for PCM)
    setUint16(numberOfChannels);
    setUint32(audioBuffer.sampleRate);
    setUint32(audioBuffer.sampleRate * 2 * numberOfChannels); // ByteRate
    setUint16(numberOfChannels * 2); // BlockAlign
    setUint16(16); // BitsPerSample

    // "data" sub-chunk
    setUint32(0x61746164); // "data"
    setUint32(length); // Subchunk2Size

    // Write interleaved data
    const channelData = audioBuffer.getChannelData(0);
    for (let i = 0; i < channelData.length; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i]));
      view.setInt16(pos, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      pos += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleASRTest = async (audioBase64: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await api.transcribeAudio(audioBase64, `call-${Date.now()}`, asrDialect);
      setResult({
        transcription: response.text || 'Transcription will appear here',
        dialect: asrDialect,
        timestamp: new Date().toISOString(),
        speakers: response.speakers || [],
        segments: response.segments || [],
        roles: response.roles || [],
        primary_doctor: response.primary_doctor,
        primary_patient: response.primary_patient,
        duration: response.duration,
        processing_time: response.processing_time,
        rtf: response.rtf
      });
    } catch (err: any) {
      setError(err.message || 'ASR service error. Make sure the backend gateway is running on localhost:3001');
    } finally {
      setLoading(false);
    }
  };

  const handleLLMTest = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await api.inferMessage(llmMessage, sessionId);
      setResult({
        message: llmMessage,
        reply: response.reply,
        intent: response.intent,
        sessionId: sessionId
      });
    } catch (err: any) {
      setError(err.message || 'LLM service error. Make sure the backend gateway is running on localhost:3001');
    } finally {
      setLoading(false);
    }
  };

  const handleSOAPTest = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await api.createSOAPNote({
        transcript: soapTranscript,
        sessionId: `soap-${Date.now()}`
      });
      setResult({
        subjective: response.subjective,
        objective: response.objective,
        assessment: response.assessment,
        plan: response.plan,
        icd_codes: response.icd_codes,
        cpt_codes: response.cpt_codes,
        timestamp: new Date().toISOString()
      });
    } catch (err: any) {
      setError(err.message || 'SOAP service error. Make sure the backend gateway is running on localhost:3001');
    } finally {
      setLoading(false);
    }
  };

  const handleFHIRTest = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await api.createFHIRResource(fhirResource, fhirData);
      setResult({
        resourceType: fhirResource,
        resource: response,
        timestamp: new Date().toISOString()
      });
    } catch (err: any) {
      setError(err.message || 'FHIR service error. Make sure the backend gateway is running on localhost:3001');
    } finally {
      setLoading(false);
    }
  };

  const handleTTSTest = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await api.synthesizeSpeech(ttsText);
      setResult({
        text: ttsText,
        audioGenerated: !!response.audio,
        timestamp: new Date().toISOString()
      });
      if (response.audio) {
        // Convert base64 to audio URL
        const audioBlob = new Blob(
          [Uint8Array.from(atob(response.audio), c => c.charCodeAt(0))],
          { type: 'audio/wav' }
        );
        setTtsAudioURL(URL.createObjectURL(audioBlob));
      }
    } catch (err: any) {
      setError(err.message || 'TTS service error. Make sure the backend gateway is running on localhost:3001');
    } finally {
      setLoading(false);
    }
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'asr':
        return (
          <div className="space-y-6">
            {/* Dialect Selection */}
            <div>
              <label className="block text-sm font-medium mb-2">
                {language === 'ar' ? 'اللهجة' : 'Dialect'}
              </label>
              <select
                value={asrDialect}
                onChange={(e) => setAsrDialect(e.target.value)}
                className="w-full px-4 py-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700"
              >
                <option value="egyptian">🇪🇬 Egyptian</option>
                <option value="levantine">🇸🇾 Levantine</option>
                <option value="gulf">🇸🇦 Gulf</option>
                <option value="msa">📖 MSA</option>
                <option value="english">🇬🇧 English</option>
              </select>
            </div>

            {/* Modern Push-to-Talk Button */}
            <div className="flex flex-col items-center justify-center py-12">
              <motion.button
                onMouseDown={!isRecording && !loading ? startRecording : undefined}
                onMouseUp={isRecording ? stopRecording : undefined}
                onTouchStart={!isRecording && !loading ? startRecording : undefined}
                onTouchEnd={isRecording ? stopRecording : undefined}
                disabled={loading}
                className={`relative w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300 ${
                  isRecording
                    ? 'bg-gradient-to-br from-red-500 to-pink-600 shadow-2xl shadow-red-500/50 scale-110'
                    : 'bg-gradient-to-br from-accent-500 to-accent-600 hover:scale-105 shadow-glow'
                } ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                whileTap={{ scale: 0.95 }}
                animate={isRecording ? {
                  boxShadow: [
                    '0 0 0 0 rgba(239, 68, 68, 0.7)',
                    '0 0 0 20px rgba(239, 68, 68, 0)',
                  ]
                } : {}}
                transition={{
                  duration: 1.5,
                  repeat: isRecording ? Infinity : 0,
                  ease: 'easeOut'
                }}
              >
                {loading ? (
                  <IconLoader size={48} className="text-white animate-spin" />
                ) : isRecording ? (
                  <IconPlayerStop size={48} className="text-white" />
                ) : (
                  <IconMicrophone size={48} className="text-white" />
                )}
              </motion.button>

              <motion.p
                className="mt-6 text-lg font-semibold"
                animate={isRecording ? { scale: [1, 1.05, 1] } : {}}
                transition={{ duration: 1, repeat: isRecording ? Infinity : 0 }}
              >
                {loading
                  ? (language === 'ar' ? 'جاري المعالجة...' : 'Processing...')
                  : isRecording
                    ? (language === 'ar' ? '🎙️ جاري التسجيل... اترك للإيقاف' : '🎙️ Recording... Release to stop')
                    : (language === 'ar' ? 'اضغط للتحدث' : 'Hold to Talk')
                }
              </motion.p>
            </div>

            {/* Audio Playback */}
            {asrAudioURL && !isRecording && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-card p-4"
              >
                <label className="block text-sm font-medium mb-2 flex items-center gap-2">
                  <IconPlayerPlay size={18} />
                  {language === 'ar' ? 'التسجيل الخاص بك' : 'Your Recording'}
                </label>
                <audio controls src={asrAudioURL} className="w-full" />
              </motion.div>
            )}
          </div>
        );      case 'llm':
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium mb-2">
                {language === 'ar' ? 'رسالتك' : 'Your Message'}
              </label>
              <textarea
                value={llmMessage}
                onChange={(e) => setLlmMessage(e.target.value)}
                className="w-full px-4 py-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700"
                rows={4}
                placeholder="Ask a medical question..."
              />
            </div>

            <div className="text-sm text-gray-600 dark:text-gray-400">
              Session ID: <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">{sessionId}</code>
            </div>

            <button
              onClick={handleLLMTest}
              disabled={loading}
              className="magnetic-btn w-full"
            >
              {loading ? <IconLoader className="animate-spin" /> : 'Test LLM Service'}
            </button>
          </div>
        );

      case 'soap':
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium mb-2">
                {language === 'ar' ? 'نص المحادثة الطبية' : 'Clinical Transcript'}
              </label>
              <textarea
                value={soapTranscript}
                onChange={(e) => setSoapTranscript(e.target.value)}
                className="w-full px-4 py-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 font-mono text-sm"
                rows={12}
                placeholder="Enter the clinical conversation transcript..."
              />
            </div>

            <div className="text-sm text-gray-600 dark:text-gray-400">
              {language === 'ar'
                ? 'سيقوم الذكاء الاصطناعي بتحويل هذا النص إلى ملاحظة SOAP منظمة'
                : 'AI will convert this transcript into a structured SOAP note'}
            </div>

            <button
              onClick={handleSOAPTest}
              disabled={loading}
              className="magnetic-btn w-full"
            >
              {loading ? <IconLoader className="animate-spin" /> : 'Generate SOAP Note'}
            </button>
          </div>
        );

      case 'fhir':
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium mb-2">
                Resource Type
              </label>
              <select
                value={fhirResource}
                onChange={(e) => setFhirResource(e.target.value)}
                className="w-full px-4 py-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700"
              >
                <option value="Patient">Patient</option>
                <option value="Observation">Observation</option>
                <option value="Condition">Condition</option>
                <option value="MedicationRequest">MedicationRequest</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Resource Data (JSON)
              </label>
              <textarea
                value={JSON.stringify(fhirData, null, 2)}
                onChange={(e) => {
                  try {
                    setFhirData(JSON.parse(e.target.value));
                  } catch {}
                }}
                className="w-full px-4 py-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 font-mono text-sm"
                rows={8}
              />
            </div>

            <button
              onClick={handleFHIRTest}
              disabled={loading}
              className="magnetic-btn w-full"
            >
              {loading ? <IconLoader className="animate-spin" /> : 'Create FHIR Resource'}
            </button>
          </div>
        );

      case 'tts':
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium mb-2">
                {language === 'ar' ? 'النص' : 'Text'}
              </label>
              <textarea
                value={ttsText}
                onChange={(e) => setTtsText(e.target.value)}
                className="w-full px-4 py-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700"
                rows={4}
                placeholder="Enter text to synthesize..."
              />
            </div>

            {ttsAudioURL && (
              <div>
                <label className="block text-sm font-medium mb-2">
                  Generated Audio
                </label>
                <audio controls src={ttsAudioURL} className="w-full" />
              </div>
            )}

            <button
              onClick={handleTTSTest}
              disabled={loading}
              className="magnetic-btn w-full"
            >
              {loading ? <IconLoader className="animate-spin" /> : 'Synthesize Speech'}
            </button>
          </div>
        );

      default:
        return null;
    }
  };

  // Helper function to get semantic role information for a speaker
  const getSpeakerRole = (speakerId: string) => {
    // If we have semantic roles from LLM, use them
    if (result?.roles && result.roles.length > 0) {
      const roleInfo = result.roles.find((r: any) => r.speaker_id === speakerId);
      if (roleInfo) {
        return {
          role: roleInfo.role,
          emoji: getRoleEmoji(roleInfo.role),
          confidence: roleInfo.confidence,
          reasoning: roleInfo.reasoning,
          color: getRoleColor(roleInfo.role)
        };
      }
    }

    // Fallback to simple assumption (old behavior)
    if (speakerId === 'SPEAKER_00') {
      return { role: 'Doctor', emoji: '👨‍⚕️', confidence: 0.5, reasoning: 'Position-based assumption', color: 'blue' };
    } else if (speakerId === 'SPEAKER_01') {
      return { role: 'Patient', emoji: '🧑', confidence: 0.5, reasoning: 'Position-based assumption', color: 'green' };
    }

    return { role: speakerId, emoji: '👤', confidence: 0, reasoning: 'Unknown', color: 'gray' };
  };

  const getRoleEmoji = (role: string) => {
    const roleLower = role.toLowerCase();
    if (roleLower.includes('doctor') || roleLower.includes('physician')) return '👨‍⚕️';
    if (roleLower.includes('patient')) return '🧑';
    if (roleLower.includes('nurse')) return '👩‍⚕️';
    if (roleLower.includes('family')) return '👨‍👩‍👧';
    return '👤';
  };

  const getRoleColor = (role: string) => {
    const roleLower = role.toLowerCase();
    if (roleLower.includes('doctor') || roleLower.includes('physician')) return 'blue';
    if (roleLower.includes('patient')) return 'green';
    if (roleLower.includes('nurse')) return 'purple';
    if (roleLower.includes('family')) return 'orange';
    return 'gray';
  };

  const getColorClasses = (color: string, type: 'bg' | 'border' | 'text' | 'shadow' = 'bg') => {
    const colorMap: Record<string, Record<string, string>> = {
      blue: {
        bg: 'bg-gradient-to-r from-blue-500 to-blue-600',
        border: 'border-blue-500',
        text: 'text-white',
        shadow: 'shadow-blue-500/30'
      },
      green: {
        bg: 'bg-gradient-to-r from-green-500 to-green-600',
        border: 'border-green-500',
        text: 'text-white',
        shadow: 'shadow-green-500/30'
      },
      purple: {
        bg: 'bg-gradient-to-r from-purple-500 to-purple-600',
        border: 'border-purple-500',
        text: 'text-white',
        shadow: 'shadow-purple-500/30'
      },
      orange: {
        bg: 'bg-gradient-to-r from-orange-500 to-orange-600',
        border: 'border-orange-500',
        text: 'text-white',
        shadow: 'shadow-orange-500/30'
      },
      gray: {
        bg: 'bg-gradient-to-r from-gray-500 to-gray-600',
        border: 'border-gray-500',
        text: 'text-white',
        shadow: 'shadow-gray-500/30'
      }
    };

    return colorMap[color]?.[type] || colorMap.gray[type];
  };

  return (
    <div className="min-h-screen py-20">
      <div className="container-custom">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center max-w-4xl mx-auto mb-16"
        >
          <h1 className="text-5xl md:text-7xl font-bold mb-6 kinetic-text">
            <span className="gradient-text">
              {language === 'ar' ? 'جرب النظام' : 'Try the Platform'}
            </span>
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-4">
            {language === 'ar'
              ? 'اختبر جميع خدمات API الخاصة بنا في مكان واحد'
              : 'Test all our API services in one place'}
          </p>
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-accent-100 dark:bg-accent-900/20 rounded-full text-sm">
            <IconCheck size={16} className="text-green-500" />
            <span>Backend services should be running (ports 5000-5004)</span>
          </div>
        </motion.div>

        {/* Tabs */}
        <div className="flex flex-wrap gap-4 mb-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id);
                setResult(null);
                setError(null);
                setAsrAudioURL(null);
                setTtsAudioURL(null);
              }}
              className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-accent-500 to-accent-600 text-white shadow-glow'
                  : 'glass-card hover:scale-105'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content Grid */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Panel */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-8"
          >
            <h2 className="text-2xl font-bold mb-6">
              {language === 'ar' ? 'الإدخال' : 'Input'}
            </h2>
            {renderTabContent()}
          </motion.div>

          {/* Result Panel */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-8"
          >
            <h2 className="text-2xl font-bold mb-6">
              {language === 'ar' ? 'النتيجة' : 'Response'}
            </h2>

            {loading && (
              <div className="flex flex-col items-center justify-center h-64">
                <IconLoader size={48} className="animate-spin text-accent-500 mb-4" />
                <p className="text-gray-600 dark:text-gray-400">
                  {language === 'ar' ? 'جارٍ المعالجة...' : 'Processing...'}
                </p>
              </div>
            )}

            {error && (
              <div className="p-4 bg-red-100 dark:bg-red-900/20 border border-red-300 dark:border-red-700 rounded-xl">
                <div className="flex items-start gap-3">
                  <IconX size={20} className="text-red-500 flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="font-bold text-red-700 dark:text-red-300 mb-1">Error</h3>
                    <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {result && !loading && (
              <div className="space-y-4">
                <div className="p-4 bg-green-100 dark:bg-green-900/20 border border-green-300 dark:border-green-700 rounded-xl">
                  <div className="flex items-center gap-2 text-green-700 dark:text-green-300 font-medium mb-2">
                    <IconCheck size={20} />
                    Success!
                  </div>
                </div>

                {/* ASR Result - Beautiful UI */}
                {activeTab === 'asr' && result.transcription && (
                  <div className="space-y-4">
                    <div className="p-6 bg-gradient-to-br from-accent-50 to-accent-100 dark:from-accent-900/20 dark:to-accent-800/20 rounded-xl border border-accent-200 dark:border-accent-700">
                      <div className="flex items-start gap-3 mb-4">
                        <div className="p-2 bg-accent-500 rounded-lg">
                          <IconMicrophone size={24} className="text-white" />
                        </div>
                        <div className="flex-1">
                          <h3 className="font-bold text-lg mb-1">Transcription Result</h3>
                          <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 flex-wrap">
                            <span className="px-2 py-1 bg-white dark:bg-gray-800 rounded-full">
                              {result.dialect || 'Arabic'}
                            </span>
                            <span>•</span>
                            <span>{new Date(result.timestamp).toLocaleTimeString()}</span>
                            {result.duration && (
                              <>
                                <span>•</span>
                                <span>{result.duration.toFixed(1)}s audio</span>
                              </>
                            )}
                            {result.processing_time && result.duration && (
                              <>
                                <span>•</span>
                                <span className="text-green-600 dark:text-green-400 font-semibold">
                                  {(result.duration / result.processing_time).toFixed(1)}x realtime
                                </span>
                              </>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Speakers Display with Semantic Roles */}
                      {result.speakers && result.speakers.length > 0 && (
                        <motion.div
                          className="mb-4 flex items-center gap-2 flex-wrap"
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.3 }}
                        >
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Speakers:</span>
                          {result.speakers.map((speaker: string, idx: number) => {
                            const roleInfo = getSpeakerRole(speaker);
                            return (
                              <motion.span
                                key={idx}
                                initial={{ scale: 0.8, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                transition={{ delay: idx * 0.1, duration: 0.3 }}
                                className={`px-3 py-1 rounded-full text-sm font-medium shadow-lg transition-all hover:scale-105 cursor-help ${
                                  getColorClasses(roleInfo.color, 'bg')
                                } ${getColorClasses(roleInfo.color, 'shadow')}`}
                                title={`${roleInfo.role} (${(roleInfo.confidence * 100).toFixed(0)}% confidence)\n${roleInfo.reasoning}`}
                              >
                                {roleInfo.emoji} {roleInfo.role}
                                {roleInfo.confidence < 0.7 && <span className="ml-1 text-xs opacity-75">?</span>}
                              </motion.span>
                            );
                          })}
                        </motion.div>
                      )}                      {/* Transcription with Semantic Speaker Labels */}
                      {result.segments && result.segments.length > 0 ? (
                        <div className="space-y-3">
                          {result.segments.map((segment: any, idx: number) => {
                            const roleInfo = segment.speaker ? getSpeakerRole(segment.speaker) : null;
                            const color = roleInfo?.color || 'gray';

                            return (
                              <motion.div
                                key={idx}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: idx * 0.05, duration: 0.3 }}
                                className={`group p-4 rounded-xl border-l-4 transition-all duration-300 hover:scale-[1.02] hover:shadow-xl ${
                                  color === 'blue'
                                    ? 'bg-gradient-to-br from-blue-50 to-blue-100/50 dark:from-blue-900/20 dark:to-blue-800/10 border-blue-500 hover:shadow-blue-500/20'
                                    : color === 'green'
                                    ? 'bg-gradient-to-br from-green-50 to-green-100/50 dark:from-green-900/20 dark:to-green-800/10 border-green-500 hover:shadow-green-500/20'
                                    : color === 'purple'
                                    ? 'bg-gradient-to-br from-purple-50 to-purple-100/50 dark:from-purple-900/20 dark:to-purple-800/10 border-purple-500 hover:shadow-purple-500/20'
                                    : color === 'orange'
                                    ? 'bg-gradient-to-br from-orange-50 to-orange-100/50 dark:from-orange-900/20 dark:to-orange-800/10 border-orange-500 hover:shadow-orange-500/20'
                                    : 'bg-gradient-to-br from-white to-gray-50 dark:from-gray-900 dark:to-gray-800 border-accent-200 dark:border-accent-700'
                                }`}
                              >
                                <div className="flex items-center gap-2 mb-2">
                                  {roleInfo && (
                                    <motion.span
                                      whileHover={{ scale: 1.1 }}
                                      className={`text-xs font-bold px-2 py-1 rounded-full cursor-help ${
                                        color === 'blue'
                                          ? 'bg-blue-500 text-white shadow-md shadow-blue-500/50'
                                          : color === 'green'
                                          ? 'bg-green-500 text-white shadow-md shadow-green-500/50'
                                          : color === 'purple'
                                          ? 'bg-purple-500 text-white shadow-md shadow-purple-500/50'
                                          : color === 'orange'
                                          ? 'bg-orange-500 text-white shadow-md shadow-orange-500/50'
                                          : 'bg-gray-500 text-white shadow-md shadow-gray-500/50'
                                      }`}
                                      title={`${roleInfo.role} (${(roleInfo.confidence * 100).toFixed(0)}% confidence)\n${roleInfo.reasoning}`}
                                    >
                                      {roleInfo.emoji} {roleInfo.role}
                                      {roleInfo.confidence < 0.7 && <span className="ml-1 opacity-75">?</span>}
                                    </motion.span>
                                  )}
                                  <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                                    {segment.start?.toFixed(1)}s - {segment.end?.toFixed(1)}s
                                  </span>
                                </div>
                                <p className="text-lg leading-relaxed text-gray-900 dark:text-gray-100 font-arabic transition-all group-hover:text-gray-950 dark:group-hover:text-white" dir="rtl">
                                  {segment.text}
                                </p>
                              </motion.div>
                            );
                          })}
                        </div>
                      ) : (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="p-4 bg-white dark:bg-gray-900 rounded-lg border border-accent-200 dark:border-accent-700"
                        >
                          <p className="text-lg leading-relaxed text-gray-900 dark:text-gray-100 font-arabic" dir="rtl">
                            {result.transcription}
                          </p>
                        </motion.div>
                      )}
                    </div>
                  </div>
                )}

                {/* LLM Result - Beautiful UI */}
                {activeTab === 'llm' && result.reply && (
                  <div className="space-y-4">
                    <div className="p-6 bg-gradient-to-br from-indigo-50 to-purple-100 dark:from-indigo-900/20 dark:to-purple-800/20 rounded-xl border border-indigo-200 dark:border-indigo-700">
                      <div className="flex items-start gap-3 mb-4">
                        <div className="p-2 bg-indigo-500 rounded-lg">
                          <IconBrain size={24} className="text-white" />
                        </div>
                        <div className="flex-1">
                          <h3 className="font-bold text-lg mb-1">AI Response</h3>
                          <span className="px-2 py-1 bg-indigo-100 dark:bg-indigo-900/50 rounded-full text-xs">
                            Intent: {result.intent}
                          </span>
                        </div>
                      </div>
                      <div className="space-y-3">
                        <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Your Question:</p>
                          <p className="text-gray-900 dark:text-gray-100">{result.message}</p>
                        </div>
                        <div className="p-4 bg-white dark:bg-gray-900 rounded-lg border-l-4 border-indigo-500">
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">AI Response:</p>
                          <p className="text-lg leading-relaxed text-gray-900 dark:text-gray-100">{result.reply}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* SOAP Result - Beautiful UI */}
                {activeTab === 'soap' && result.subjective && (
                  <div className="space-y-4">
                    <div className="p-6 bg-gradient-to-br from-emerald-50 to-teal-100 dark:from-emerald-900/20 dark:to-teal-800/20 rounded-xl border border-emerald-200 dark:border-emerald-700">
                      <div className="flex items-start gap-3 mb-4">
                        <div className="p-2 bg-emerald-500 rounded-lg">
                          <IconFileText size={24} className="text-white" />
                        </div>
                        <div>
                          <h3 className="font-bold text-lg">SOAP Note Generated</h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">Clinical Documentation</p>
                        </div>
                      </div>
                      <div className="space-y-3">
                        <div className="p-4 bg-white dark:bg-gray-900 rounded-lg">
                          <h4 className="font-bold text-emerald-700 dark:text-emerald-400 mb-2">📝 Subjective</h4>
                          <p className="text-gray-800 dark:text-gray-200">{result.subjective}</p>
                        </div>
                        <div className="p-4 bg-white dark:bg-gray-900 rounded-lg">
                          <h4 className="font-bold text-blue-700 dark:text-blue-400 mb-2">🔍 Objective</h4>
                          <p className="text-gray-800 dark:text-gray-200">{result.objective}</p>
                        </div>
                        <div className="p-4 bg-white dark:bg-gray-900 rounded-lg">
                          <h4 className="font-bold text-orange-700 dark:text-orange-400 mb-2">🩺 Assessment</h4>
                          <p className="text-gray-800 dark:text-gray-200">{result.assessment}</p>
                        </div>
                        <div className="p-4 bg-white dark:bg-gray-900 rounded-lg">
                          <h4 className="font-bold text-purple-700 dark:text-purple-400 mb-2">💊 Plan</h4>
                          <p className="text-gray-800 dark:text-gray-200">{result.plan}</p>
                        </div>
                        {(result.icd_codes?.length > 0 || result.cpt_codes?.length > 0) && (
                          <div className="flex gap-3">
                            {result.icd_codes?.length > 0 && (
                              <div className="flex-1 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                                <p className="text-xs font-medium text-blue-700 dark:text-blue-400 mb-2">ICD Codes</p>
                                <div className="flex flex-wrap gap-2">
                                  {result.icd_codes.map((code: string, i: number) => (
                                    <span key={i} className="px-2 py-1 bg-blue-100 dark:bg-blue-800 rounded text-xs">
                                      {code}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                            {result.cpt_codes?.length > 0 && (
                              <div className="flex-1 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                                <p className="text-xs font-medium text-purple-700 dark:text-purple-400 mb-2">CPT Codes</p>
                                <div className="flex flex-wrap gap-2">
                                  {result.cpt_codes.map((code: string, i: number) => (
                                    <span key={i} className="px-2 py-1 bg-purple-100 dark:bg-purple-800 rounded text-xs">
                                      {code}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* FHIR Result - Beautiful UI */}
                {activeTab === 'fhir' && result.resourceType && (
                  <div className="space-y-4">
                    <div className="p-6 bg-gradient-to-br from-blue-50 to-cyan-100 dark:from-blue-900/20 dark:to-cyan-800/20 rounded-xl border border-blue-200 dark:border-blue-700">
                      <div className="flex items-start gap-3 mb-4">
                        <div className="p-2 bg-blue-500 rounded-lg">
                          <IconDatabase size={24} className="text-white" />
                        </div>
                        <div>
                          <h3 className="font-bold text-lg">FHIR Resource Created</h3>
                          <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/50 rounded-full text-xs">
                            {result.resourceType}
                          </span>
                        </div>
                      </div>
                      <div className="p-4 bg-white dark:bg-gray-900 rounded-lg">
                        <pre className="text-xs overflow-x-auto whitespace-pre-wrap text-gray-700 dark:text-gray-300">
                          {JSON.stringify(result.resource, null, 2)}
                        </pre>
                      </div>
                    </div>
                  </div>
                )}

                {/* TTS Result - Beautiful UI */}
                {activeTab === 'tts' && result.audioGenerated && (
                  <div className="space-y-4">
                    <div className="p-6 bg-gradient-to-br from-pink-50 to-rose-100 dark:from-pink-900/20 dark:to-rose-800/20 rounded-xl border border-pink-200 dark:border-pink-700">
                      <div className="flex items-start gap-3 mb-4">
                        <div className="p-2 bg-pink-500 rounded-lg">
                          <IconVolume size={24} className="text-white" />
                        </div>
                        <div>
                          <h3 className="font-bold text-lg">Audio Synthesized</h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">Text-to-Speech Conversion</p>
                        </div>
                      </div>
                      <div className="space-y-3">
                        <div className="p-4 bg-white dark:bg-gray-900 rounded-lg">
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Input Text:</p>
                          <p className="text-gray-900 dark:text-gray-100">{result.text}</p>
                        </div>
                        {ttsAudioURL && (
                          <div className="p-4 bg-white dark:bg-gray-900 rounded-lg">
                            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Generated Audio:</p>
                            <audio controls src={ttsAudioURL} className="w-full" />
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {!loading && !error && !result && (
              <div className="flex items-center justify-center h-64 text-gray-400">
                <p className="text-center">
                  {language === 'ar'
                    ? 'املأ النموذج واضغط على الزر للاختبار'
                    : 'Fill the form and click the button to test'}
                </p>
              </div>
            )}
          </motion.div>
        </div>

        {/* Instructions */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-16 glass-card p-8"
        >
          <h3 className="text-2xl font-bold mb-6">
            {language === 'ar' ? 'تعليمات' : 'Instructions'}
          </h3>
          <ol className="space-y-3 text-gray-600 dark:text-gray-300">
            <li>1. Make sure all backend services are running:
              <ul className="ml-6 mt-2 space-y-1 text-sm">
                <li>• ASR: <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">localhost:5000</code></li>
                <li>• LLM: <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">localhost:5001</code></li>
                <li>• TTS: <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">localhost:5002</code></li>
                <li>• SOAP: <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">localhost:5003</code></li>
                <li>• FHIR: <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">localhost:5004</code></li>
              </ul>
            </li>
            <li>2. Check <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">.env</code> file for <code>VITE_USE_DIRECT_SERVICES=true</code></li>
            <li>3. Select a service tab above to test</li>
            <li>4. Fill in the form and click the test button</li>
            <li>5. View the API response in real-time on the right panel</li>
          </ol>
        </motion.div>
      </div>
    </div>
  );
};

export default Demo;
