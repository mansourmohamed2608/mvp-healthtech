import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  IconMicrophone, 
  IconPlayerStop, 
  IconUpload, 
  IconFileText,
  IconCheck,
  IconX,
  IconChartBar,
  IconClock,
  IconFileDownload,
  IconLoader2
} from '@tabler/icons-react';
import api from '../utils/api';

interface AudioRecording {
  id: string;
  file: File;
  duration: number;
  timestamp: Date;
  status: 'pending' | 'processing' | 'completed' | 'error';
  transcript?: string;
  soapNote?: string;
  error?: string;
  reviewStartTime?: number;
  editDistance?: number;
  dialect?: string;
  autoDetected?: boolean;
}

interface MetricsDashboard {
  overview: {
    totalNotes: number;
    acceptanceRate: number;
    avgEditDistance: number;
    avgReviewTime: number;
  };
}

export default function ClinicalNotes() {
  const [recordings, setRecordings] = useState<AudioRecording[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [selectedRecording, setSelectedRecording] = useState<AudioRecording | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);
  const [editedSoapNote, setEditedSoapNote] = useState<string>('');
  const [selectedDialect, setSelectedDialect] = useState<string>('auto');
  const [metrics, setMetrics] = useState<MetricsDashboard | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Calculate edit distance (Levenshtein distance approximation)
  const calculateEditDistance = (original: string, edited: string): number => {
    return Math.abs(original.length - edited.length) + 
      (original !== edited ? Math.min(original.length, edited.length) / 10 : 0);
  };

  // Record review metrics
  const recordReviewMetrics = async (
    recording: AudioRecording,
    accepted: boolean,
    editedText: string
  ) => {
    const reviewEndTime = Date.now();
    const reviewStartTime = recording.reviewStartTime || reviewEndTime;
    const timeToReview = Math.round((reviewEndTime - reviewStartTime) / 1000); // seconds
    const editDistance = calculateEditDistance(recording.soapNote || '', editedText);

    try {
      const response = await fetch('http://localhost:3001/clinical/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          recordingId: recording.id,
          accepted,
          editDistance,
          timeToReview,
          clinicianId: 'demo-user',
        }),
      });

      if (response.ok) {
        // Refresh metrics after recording
        loadMetrics();
      }
    } catch (err) {
      console.error('Failed to record metrics:', err);
    }
  };

  // Load metrics dashboard
  const loadMetrics = async () => {
    try {
      const response = await fetch('http://localhost:3001/clinical/metrics/dashboard');
      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
      }
    } catch (err) {
      console.error('Failed to load metrics:', err);
    }
  };

  // Start live recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const audioFile = new File(
          [audioBlob],
          `recording-${Date.now()}.webm`,
          { type: 'audio/webm' }
        );

        const newRecording: AudioRecording = {
          id: `rec-${Date.now()}`,
          file: audioFile,
          duration: recordingTime,
          timestamp: new Date(),
          status: 'pending',
        };

        setRecordings((prev) => [newRecording, ...prev]);
        setRecordingTime(0);

        // Auto-process the recording
        await processRecording(newRecording);
      };

      mediaRecorder.start();
      setIsRecording(true);

      // Start timer
      timerRef.current = window.setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (error: any) {
      console.error('Failed to start recording:', error);
      alert(`خطأ: ${error.message}`);
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      setIsRecording(false);

      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  // Upload audio file
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    for (const file of Array.from(files)) {
      const newRecording: AudioRecording = {
        id: `upload-${Date.now()}-${Math.random()}`,
        file,
        duration: 0,
        timestamp: new Date(),
        status: 'pending',
      };

      setRecordings((prev) => [newRecording, ...prev]);
      await processRecording(newRecording);
    }

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Process recording (ASR + SOAP generation)
  const processRecording = async (recording: AudioRecording) => {
    try {
      // Update status to processing
      setRecordings((prev) =>
        prev.map((r) =>
          r.id === recording.id ? { ...r, status: 'processing' } : r
        )
      );

      // Step 1: Transcribe audio
      const formData = new FormData();
      formData.append('audio', recording.file);
      
      const transcriptResponse = await api.transcribeAudio(formData);
      const transcript = transcriptResponse.transcript || transcriptResponse.text || '';

      // Step 2: Generate SOAP note
      const soapResponse = await api.generateSoapNote(transcript);
      const soapNote = soapResponse.soapNote || soapResponse.soap || '';

      // Update recording with results
      setRecordings((prev) =>
        prev.map((r) =>
          r.id === recording.id
            ? {
                ...r,
                status: 'completed',
                transcript,
                soapNote,
                dialect: selectedDialect,
                autoDetected: selectedDialect === 'auto',
              }
            : r
        )
      );
    } catch (error: any) {
      console.error('Processing error:', error);
      setRecordings((prev) =>
        prev.map((r) =>
          r.id === recording.id
            ? { ...r, status: 'error', error: error.message }
            : r
        )
      );
    }
  };

  // Format time (seconds to MM:SS)
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Save SOAP note to EHR (Week 5 Day 29)
  const saveToEHR = async () => {
    if (!selectedRecording || !selectedRecording.soapNote) return;

    setIsSaving(true);
    try {
      const response = await api.convertToFHIR({
        soapNote: parseSoapNote(selectedRecording.soapNote),
        patientId: 'patient-123',
        practitionerId: 'prac-456',
        sessionId: selectedRecording.id,
      });

      alert(`✅ تم الحفظ بنجاح!\nDocument ID: ${response.documentReferenceId}`);
    } catch (error: any) {
      alert(`❌ خطأ في الحفظ: ${error.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  // Parse SOAP note text into structured object
  const parseSoapNote = (soapText: string) => {
    const sections = {
      subjective: '',
      objective: '',
      assessment: '',
      plan: '',
    };

    const lines = soapText.split('\n');
    let currentSection = '';

    for (const line of lines) {
      if (line.includes('الذاتي') || line.includes('Subjective')) {
        currentSection = 'subjective';
      } else if (line.includes('الموضوعي') || line.includes('Objective')) {
        currentSection = 'objective';
      } else if (line.includes('التقييم') || line.includes('Assessment')) {
        currentSection = 'assessment';
      } else if (line.includes('الخطة') || line.includes('Plan')) {
        currentSection = 'plan';
      } else if (currentSection && line.trim()) {
        sections[currentSection as keyof typeof sections] += line + '\n';
      }
    }

    return sections;
  };

  const statsCards = [
    { label: 'إجمالي التسجيلات', value: recordings.length, color: 'from-purple-500 to-purple-600' },
    { label: 'مكتمل', value: recordings.filter((r) => r.status === 'completed').length, color: 'from-green-500 to-green-600' },
    { label: 'قيد المعالجة', value: recordings.filter((r) => r.status === 'processing').length, color: 'from-yellow-500 to-yellow-600' },
    { label: 'خطأ', value: recordings.filter((r) => r.status === 'error').length, color: 'from-red-500 to-red-600' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8 font-['Tajawal',sans-serif]" dir="rtl">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="backdrop-blur-md bg-white/10 border border-white/20 rounded-3xl p-8 mb-8 shadow-2xl"
        >
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-200 via-pink-200 to-purple-200 bg-clip-text text-transparent mb-3">
                توثيق السجلات الطبية
              </h1>
              <p className="text-purple-200 text-lg">
                نظام تحويل التسجيلات الصوتية إلى ملاحظات SOAP
              </p>
            </div>
            <button
              onClick={() => {
                setShowMetrics(!showMetrics);
                if (!showMetrics && !metrics) loadMetrics();
              }}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-xl font-bold transition-all shadow-lg hover:shadow-blue-500/50 hover:scale-105"
            >
              <IconChartBar className="w-5 h-5" />
              {showMetrics ? 'إخفاء المقاييس' : 'عرض المقاييس'}
            </button>
          </div>
        </motion.div>

        {/* Metrics Dashboard */}
        <AnimatePresence>
          {showMetrics && metrics && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="backdrop-blur-md bg-white/10 border border-white/20 rounded-3xl p-8 mb-8 shadow-2xl"
            >
              <h2 className="text-2xl font-bold text-white mb-6">لوحة المقاييس</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-gradient-to-br from-purple-500/20 to-purple-600/20 border border-purple-500/30 rounded-2xl p-6">
                  <p className="text-purple-200 text-sm mb-2">إجمالي الملاحظات</p>
                  <p className="text-4xl font-bold text-white">{metrics.overview.totalNotes}</p>
                </div>
                <div className="bg-gradient-to-br from-green-500/20 to-green-600/20 border border-green-500/30 rounded-2xl p-6">
                  <p className="text-green-200 text-sm mb-2">معدل القبول</p>
                  <p className="text-4xl font-bold text-white">{(metrics.overview.acceptanceRate * 100).toFixed(1)}%</p>
                </div>
                <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/20 border border-blue-500/30 rounded-2xl p-6">
                  <p className="text-blue-200 text-sm mb-2">متوسط التعديل</p>
                  <p className="text-4xl font-bold text-white">{metrics.overview.avgEditDistance.toFixed(1)}</p>
                </div>
                <div className="bg-gradient-to-br from-yellow-500/20 to-yellow-600/20 border border-yellow-500/30 rounded-2xl p-6">
                  <p className="text-yellow-200 text-sm mb-2">وقت المراجعة (ثانية)</p>
                  <p className="text-4xl font-bold text-white">{metrics.overview.avgReviewTime.toFixed(0)}</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Panel - Recording Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* Live Recording */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="backdrop-blur-md bg-white/10 border border-white/20 rounded-3xl p-8 shadow-2xl"
            >
              <h2 className="text-2xl font-bold text-white mb-6">تسجيل مباشر</h2>
              <div className="text-center">
                {!isRecording ? (
                  <motion.button
                    onClick={startRecording}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="w-24 h-24 bg-gradient-to-br from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white rounded-full flex items-center justify-center mx-auto mb-6 transition-all shadow-2xl shadow-red-500/50"
                  >
                    <IconMicrophone className="w-12 h-12" />
                  </motion.button>
                ) : (
                  <>
                    <motion.button
                      onClick={stopRecording}
                      animate={{ scale: [1, 1.05, 1] }}
                      transition={{ repeat: Infinity, duration: 1.5 }}
                      className="w-24 h-24 bg-gradient-to-br from-gray-500 to-gray-600 hover:from-gray-600 hover:to-gray-700 text-white rounded-full flex items-center justify-center mx-auto mb-6 transition-all shadow-2xl shadow-gray-500/50"
                    >
                      <IconPlayerStop className="w-12 h-12" />
                    </motion.button>
                    <p className="text-3xl font-mono text-red-400 font-bold">
                      {formatTime(recordingTime)}
                    </p>
                  </>
                )}
                <p className="text-sm text-purple-200 mt-4">
                  {isRecording ? 'جاري التسجيل...' : 'انقر للبدء'}
                </p>
              </div>
            </motion.div>

            {/* File Upload */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="backdrop-blur-md bg-white/10 border border-white/20 rounded-3xl p-8 shadow-2xl"
            >
              <h2 className="text-2xl font-bold text-white mb-6">رفع ملف صوتي</h2>
              
              {/* Dialect Selector */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-purple-200 mb-3">
                  اللهجة:
                </label>
                <select
                  value={selectedDialect}
                  onChange={(e) => setSelectedDialect(e.target.value)}
                  className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent backdrop-blur-sm"
                  dir="rtl"
                >
                  <option value="auto">كشف تلقائي</option>
                  <option value="egyptian">مصري</option>
                  <option value="levantine">شامي</option>
                  <option value="gulf">خليجي</option>
                  <option value="msa">فصحى</option>
                </select>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                multiple
                onChange={handleFileUpload}
                className="hidden"
              />
              <motion.button
                onClick={() => fileInputRef.current?.click()}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg hover:shadow-blue-500/50 flex items-center justify-center gap-3"
              >
                <IconUpload className="w-5 h-5" />
                اختر ملفات
              </motion.button>
              <p className="text-xs text-purple-300 mt-3 text-center">
                يدعم: MP3, WAV, M4A, WebM
              </p>
            </motion.div>

            {/* Stats */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="backdrop-blur-md bg-white/10 border border-white/20 rounded-3xl p-8 shadow-2xl"
            >
              <h2 className="text-2xl font-bold text-white mb-6">الإحصائيات</h2>
              <div className="space-y-4">
                {statsCards.map((stat, index) => (
                  <motion.div
                    key={stat.label}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + index * 0.05 }}
                    className={`bg-gradient-to-r ${stat.color} rounded-xl p-4`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-white text-sm font-medium">{stat.label}</span>
                      <span className="text-white text-2xl font-bold">{stat.value}</span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Right Panel - Recordings List & Details */}
          <div className="lg:col-span-2 space-y-6">
            {/* Recordings List */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="backdrop-blur-md bg-white/10 border border-white/20 rounded-3xl p-8 shadow-2xl"
            >
              <h2 className="text-2xl font-bold text-white mb-6">التسجيلات</h2>
              <div className="space-y-3 max-h-96 overflow-y-auto pr-2 custom-scrollbar">
                {recordings.length === 0 ? (
                  <div className="text-center py-12">
                    <IconFileText className="w-16 h-16 text-purple-300/50 mx-auto mb-4" />
                    <p className="text-purple-200">لا توجد تسجيلات بعد</p>
                  </div>
                ) : (
                  recordings.map((recording, index) => (
                    <motion.div
                      key={recording.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      onClick={() => {
                        setSelectedRecording(recording);
                        setEditedSoapNote('');
                      }}
                      className={`p-5 rounded-2xl border-2 cursor-pointer transition-all ${
                        selectedRecording?.id === recording.id
                          ? 'border-purple-400 bg-purple-500/20 shadow-lg shadow-purple-500/20'
                          : 'border-white/20 bg-white/5 hover:border-purple-300 hover:bg-white/10'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <p className="font-semibold text-white mb-1">
                            {recording.file.name}
                          </p>
                          <p className="text-sm text-purple-200">
                            {recording.timestamp.toLocaleString('ar-EG')}
                          </p>
                        </div>
                        <div>
                          <span
                            className={`px-4 py-2 rounded-full text-xs font-bold flex items-center gap-2 ${
                              recording.status === 'completed'
                                ? 'bg-green-500/20 text-green-300 border border-green-500/30'
                                : recording.status === 'processing'
                                ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
                                : recording.status === 'error'
                                ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                                : 'bg-gray-500/20 text-gray-300 border border-gray-500/30'
                            }`}
                          >
                            {recording.status === 'completed' && <><IconCheck className="w-4 h-4" /> مكتمل</>}
                            {recording.status === 'processing' && <><IconLoader2 className="w-4 h-4 animate-spin" /> جاري...</>}
                            {recording.status === 'error' && <><IconX className="w-4 h-4" /> خطأ</>}
                            {recording.status === 'pending' && <><IconClock className="w-4 h-4" /> في الانتظار</>}
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  ))
                )}
              </div>
            </motion.div>

            {/* Recording Details */}
            <AnimatePresence mode="wait">
              {selectedRecording && (
                <motion.div
                  key={selectedRecording.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="backdrop-blur-md bg-white/10 border border-white/20 rounded-3xl p-8 shadow-2xl"
                >
                  <h2 className="text-2xl font-bold text-white mb-6">التفاصيل</h2>

                  {selectedRecording.status === 'processing' && (
                    <div className="text-center py-12">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
                        className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full mx-auto mb-6"
                      />
                      <p className="text-purple-200 text-lg">جاري معالجة التسجيل...</p>
                    </div>
                  )}

                  {selectedRecording.status === 'error' && (
                    <div className="bg-red-500/20 border border-red-500/30 rounded-2xl p-6">
                      <div className="flex items-center gap-3 text-red-300">
                        <IconX className="w-6 h-6" />
                        <p className="font-medium">
                          حدث خطأ: {selectedRecording.error}
                        </p>
                      </div>
                    </div>
                  )}

                  {selectedRecording.status === 'completed' && (
                    <div className="space-y-6">
                      {/* Transcript */}
                      <div>
                        <h3 className="font-semibold text-purple-200 mb-3 flex items-center gap-2">
                          <IconFileText className="w-5 h-5" />
                          النص المكتوب:
                        </h3>
                        <div className="bg-white/5 border border-white/10 rounded-2xl p-5 max-h-48 overflow-y-auto custom-scrollbar backdrop-blur-sm">
                          <p className="text-white leading-relaxed">
                            {selectedRecording.transcript}
                          </p>
                        </div>
                      </div>

                      {/* SOAP Note */}
                      <div>
                        <h3 className="font-semibold text-purple-200 mb-3 flex items-center gap-2">
                          <IconFileText className="w-5 h-5" />
                          ملاحظة SOAP:
                        </h3>
                        <textarea
                          value={editedSoapNote || selectedRecording.soapNote}
                          onChange={(e) => {
                            setEditedSoapNote(e.target.value);
                            // Start tracking review time on first edit
                            if (!selectedRecording.reviewStartTime) {
                              setRecordings((prev) =>
                                prev.map((r) =>
                                  r.id === selectedRecording.id
                                    ? { ...r, reviewStartTime: Date.now() }
                                    : r
                                )
                              );
                            }
                          }}
                          className="w-full bg-white/5 border border-white/20 rounded-2xl p-5 text-white leading-relaxed font-['Tajawal',sans-serif] focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent backdrop-blur-sm custom-scrollbar resize-none"
                          rows={12}
                          dir="rtl"
                        />
                      </div>

                      {/* Actions */}
                      <div className="flex gap-4 pt-4">
                        <motion.button
                          onClick={async () => {
                            const finalText = editedSoapNote || selectedRecording.soapNote || '';
                            await recordReviewMetrics(selectedRecording, true, finalText);
                            await saveToEHR();
                          }}
                          disabled={isSaving}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className="flex-1 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 disabled:from-gray-500 disabled:to-gray-600 disabled:cursor-not-allowed text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg hover:shadow-green-500/50 flex items-center justify-center gap-3"
                        >
                          {isSaving ? (
                            <>
                              <IconLoader2 className="w-5 h-5 animate-spin" />
                              جاري الحفظ...
                            </>
                          ) : (
                            <>
                              <IconCheck className="w-5 h-5" />
                              قبول وحفظ
                            </>
                          )}
                        </motion.button>
                        <motion.button
                          onClick={async () => {
                            const finalText = editedSoapNote || selectedRecording.soapNote || '';
                            await recordReviewMetrics(selectedRecording, false, finalText);
                            alert('تم رفض الملاحظة');
                          }}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className="flex-1 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg hover:shadow-red-500/50 flex items-center justify-center gap-3"
                        >
                          <IconX className="w-5 h-5" />
                          رفض
                        </motion.button>
                        <motion.button
                          onClick={() => {
                            const finalText = editedSoapNote || selectedRecording.soapNote || '';
                            const blob = new Blob([finalText], { type: 'text/plain;charset=utf-8' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `soap-${selectedRecording.id}.txt`;
                            a.click();
                            URL.revokeObjectURL(url);
                          }}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className="bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg hover:shadow-blue-500/50"
                          title="تنزيل"
                        >
                          <IconFileDownload className="w-5 h-5" />
                        </motion.button>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(168, 85, 247, 0.5);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(168, 85, 247, 0.7);
        }
      `}</style>
    </div>
  );
}
