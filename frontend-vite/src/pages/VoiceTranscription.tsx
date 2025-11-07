import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { useThemeStore } from '@store/themeStore';
import {
  IconMicrophone,
  IconPlayerStop,
  IconDownload,
  IconCopy,
  IconCheck,
  IconLanguage,
  IconActivity
} from '@tabler/icons-react';
import api from '@utils/api';

const VoiceTranscription = () => {
  const { language } = useThemeStore();
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [selectedDialect, setSelectedDialect] = useState('egyptian');
  const [isProcessing, setIsProcessing] = useState(false);
  const [copied, setCopied] = useState(false);
  const [audioURL, setAudioURL] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const dialects = [
    { value: 'egyptian', label: language === 'ar' ? 'مصري' : 'Egyptian', flag: '🇪🇬' },
    { value: 'levantine', label: language === 'ar' ? 'شامي' : 'Levantine', flag: '🇸🇾' },
    { value: 'gulf', label: language === 'ar' ? 'خليجي' : 'Gulf', flag: '🇸🇦' },
    { value: 'msa', label: language === 'ar' ? 'فصحى' : 'MSA', flag: '📖' },
    { value: 'english', label: language === 'ar' ? 'إنجليزي' : 'English', flag: '🇬🇧' },
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
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const url = URL.createObjectURL(audioBlob);
        setAudioURL(url);

        // Convert to base64 for API
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = async () => {
          const base64Audio = reader.result?.toString().split(',')[1];
          if (base64Audio) {
            await transcribeAudio(base64Audio);
          }
        };
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Could not access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  const transcribeAudio = async (audioData: string) => {
    setIsProcessing(true);
    try {
      const result = await api.transcribeAudio(audioData, undefined, selectedDialect);
      setTranscription(result.text);
    } catch (error) {
      console.error('Transcription error:', error);
      alert('Failed to transcribe audio. Make sure backend is running on http://localhost:3001');
    } finally {
      setIsProcessing(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(transcription);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadTranscription = () => {
    const element = document.createElement('a');
    const file = new Blob([transcription], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `transcription-${Date.now()}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
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
              {language === 'ar' ? 'تحويل الصوت إلى نص' : 'Voice Transcription'}
            </span>
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            {language === 'ar'
              ? 'سجل محادثاتك الطبية واحصل على نص دقيق فورًا'
              : 'Record your medical conversations and get accurate transcriptions instantly'}
          </p>
        </motion.div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Recording Panel */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-card p-8"
          >
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <IconMicrophone size={28} className="text-accent-500" />
              {language === 'ar' ? 'التسجيل' : 'Recording'}
            </h2>

            {/* Dialect Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium mb-3">
                <IconLanguage size={20} className="inline mr-2" />
                {language === 'ar' ? 'اختر اللهجة' : 'Select Dialect'}
              </label>
              <div className="grid grid-cols-2 gap-3">
                {dialects.map((dialect) => (
                  <button
                    key={dialect.value}
                    onClick={() => setSelectedDialect(dialect.value)}
                    disabled={isRecording}
                    className={`p-4 rounded-xl border-2 transition-all ${
                      selectedDialect === dialect.value
                        ? 'border-accent-500 bg-accent-50 dark:bg-accent-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-accent-300'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    <span className="text-2xl mb-2 block">{dialect.flag}</span>
                    <span className="font-medium">{dialect.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Recording Control */}
            <div className="flex flex-col items-center py-12">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  disabled={isProcessing}
                  className="w-32 h-32 rounded-full bg-gradient-to-br from-accent-500 to-accent-600 text-white shadow-glow hover:shadow-glow-lg transition-all hover:scale-110 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <IconMicrophone size={48} className="mx-auto" />
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="w-32 h-32 rounded-full bg-gradient-to-br from-red-500 to-red-600 text-white shadow-glow hover:shadow-glow-lg transition-all animate-pulse"
                >
                  <IconPlayerStop size={48} className="mx-auto" />
                </button>
              )}

              <p className="mt-6 text-lg font-medium">
                {isRecording
                  ? language === 'ar' ? 'جارٍ التسجيل...' : 'Recording... Click to stop'
                  : language === 'ar' ? 'انقر للتسجيل' : 'Click to start recording'}
              </p>
            </div>

            {/* Audio Playback */}
            {audioURL && (
              <div className="mt-6">
                <label className="block text-sm font-medium mb-3">
                  <IconActivity size={20} className="inline mr-2" />
                  {language === 'ar' ? 'التسجيل' : 'Your Recording'}
                </label>
                <audio controls src={audioURL} className="w-full" />
              </div>
            )}

            {/* Processing Indicator */}
            {isProcessing && (
              <div className="mt-6 text-center">
                <div className="inline-flex items-center gap-3 px-6 py-3 bg-accent-100 dark:bg-accent-900/20 rounded-full">
                  <div className="w-4 h-4 border-2 border-accent-500 border-t-transparent rounded-full animate-spin" />
                  <span className="font-medium text-accent-700 dark:text-accent-300">
                    {language === 'ar' ? 'جارٍ التحويل...' : 'Transcribing...'}
                  </span>
                </div>
              </div>
            )}
          </motion.div>

          {/* Transcription Result */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-card p-8"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold">
                {language === 'ar' ? 'النص المحول' : 'Transcription'}
              </h2>

              {transcription && (
                <div className="flex gap-2">
                  <button
                    onClick={copyToClipboard}
                    className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                    title={language === 'ar' ? 'نسخ' : 'Copy'}
                  >
                    {copied ? <IconCheck size={20} className="text-green-500" /> : <IconCopy size={20} />}
                  </button>
                  <button
                    onClick={downloadTranscription}
                    className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                    title={language === 'ar' ? 'تحميل' : 'Download'}
                  >
                    <IconDownload size={20} />
                  </button>
                </div>
              )}
            </div>

            <div className="min-h-[400px] p-6 bg-gray-50 dark:bg-gray-900/50 rounded-xl">
              {transcription ? (
                <p className="text-lg leading-relaxed whitespace-pre-wrap">
                  {transcription}
                </p>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-400">
                  <p className="text-center">
                    {language === 'ar'
                      ? 'ابدأ التسجيل لرؤية النص المحول هنا'
                      : 'Start recording to see transcription here'}
                  </p>
                </div>
              )}
            </div>

            {/* Stats */}
            {transcription && (
              <div className="mt-6 grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-accent-500">
                    {transcription.split(' ').length}
                  </div>
                  <div className="text-sm text-gray-500">
                    {language === 'ar' ? 'كلمات' : 'Words'}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-accent-500">
                    {transcription.length}
                  </div>
                  <div className="text-sm text-gray-500">
                    {language === 'ar' ? 'أحرف' : 'Characters'}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-accent-500">
                    98%
                  </div>
                  <div className="text-sm text-gray-500">
                    {language === 'ar' ? 'دقة' : 'Accuracy'}
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </div>

        {/* Features */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-16 grid md:grid-cols-4 gap-6"
        >
          {[
            {
              icon: '🎯',
              title: language === 'ar' ? 'دقة عالية' : 'High Accuracy',
              desc: language === 'ar' ? '98%+ دقة' : '98%+ accuracy',
            },
            {
              icon: '🌍',
              title: language === 'ar' ? 'متعدد اللهجات' : 'Multi-Dialect',
              desc: language === 'ar' ? '5+ لهجات' : '5+ dialects',
            },
            {
              icon: '⚡',
              title: language === 'ar' ? 'سريع' : 'Fast',
              desc: language === 'ar' ? 'تحويل فوري' : 'Real-time',
            },
            {
              icon: '🔒',
              title: language === 'ar' ? 'آمن' : 'Secure',
              desc: language === 'ar' ? 'مشفر HIPAA' : 'HIPAA encrypted',
            },
          ].map((feature, index) => (
            <div key={index} className="glass-card p-6 text-center">
              <div className="text-4xl mb-3">{feature.icon}</div>
              <h3 className="font-bold mb-2">{feature.title}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">{feature.desc}</p>
            </div>
          ))}
        </motion.div>
      </div>
    </div>
  );
};

export default VoiceTranscription;
