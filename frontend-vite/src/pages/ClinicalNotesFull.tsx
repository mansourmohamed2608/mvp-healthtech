import { useState, useRef, useEffect } from 'react';
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
  IconLoader2,
  IconUser,
  IconPlus,
  IconFolder,
  IconSettings,
  IconBrain,
  IconStethoscope,
  IconCalendar,
  IconTemplate
} from '@tabler/icons-react';
import api from '../utils/api';
import { useAuthStore } from '@store/authStore';

interface AudioRecording {
  id: string;
  file: File;
  duration: number;
  timestamp: Date;
  status: 'pending' | 'processing' | 'completed' | 'error';
  transcript?: string;
  soapNote?: string;
  soapJson?: any;
  noteId?: string;
  templateId?: string;
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

export default function ClinicalNotesFull() {
  const { userId, token } = useAuthStore();
  
  // Core state
  const [recordings, setRecordings] = useState<AudioRecording[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [selectedRecording, setSelectedRecording] = useState<AudioRecording | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [editedSoapNote, setEditedSoapNote] = useState('');
  const [selectedDialect, setSelectedDialect] = useState('auto');
  
  // Patient state
  const [patientId, setPatientId] = useState('');
  const [patients, setPatients] = useState<Array<{ id: string; displayName?: string; externalId?: string }>>([]);
  const [patientStatus, setPatientStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [patientError, setPatientError] = useState('');
  const [newPatientName, setNewPatientName] = useState('');
  const [newPatientExternalId, setNewPatientExternalId] = useState('');
  const [patientName, setPatientName] = useState('');
  
  // Note settings
  const [practitionerId, setPractitionerId] = useState('');
  const [providerName, setProviderName] = useState('');
  const [dateOfVisit, setDateOfVisit] = useState('');
  
  // Templates
  const [templates, setTemplates] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState('');
  const [templateName, setTemplateName] = useState('');
  const [templateError, setTemplateError] = useState('');
  
  // Documents
  const [patientDocs, setPatientDocs] = useState<Array<any>>([]);
  const [docTitle, setDocTitle] = useState('');
  const [docText, setDocText] = useState('');
  const [docFileName, setDocFileName] = useState('');
  const [docFileType, setDocFileType] = useState('');
  const [docFileBase64, setDocFileBase64] = useState('');
  const [docStatus, setDocStatus] = useState<'idle' | 'processing' | 'done' | 'error'>('idle');
  const [docError, setDocError] = useState('');
  const [summarizeDoc, setSummarizeDoc] = useState(true);
  
  // RAG / Knowledge
  const [patientContext, setPatientContext] = useState<any>(null);
  const [patientRagItems, setPatientRagItems] = useState<Array<any>>([]);
  const [ragItemsStatus, setRagItemsStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [ragItemsError, setRagItemsError] = useState('');
  const [ragTitle, setRagTitle] = useState('');
  const [ragText, setRagText] = useState('');
  const [ragStatus, setRagStatus] = useState('');
  const [ragError, setRagError] = useState('');
  
  // Field updates
  const [fieldOptions, setFieldOptions] = useState<Array<{ path: string; valueType: 'string' | 'list' }>>([]);
  const [selectedFieldPath, setSelectedFieldPath] = useState('');
  const [fieldUpdateMode, setFieldUpdateMode] = useState<'append' | 'replace'>('append');
  const [fieldUpdateStatus, setFieldUpdateStatus] = useState('');
  const [fieldUpdateError, setFieldUpdateError] = useState('');
  const [isFieldRecording, setIsFieldRecording] = useState(false);
  const [fieldRecordingTime, setFieldRecordingTime] = useState(0);
  const [expandedPaths, setExpandedPaths] = useState<Record<string, boolean>>({});
  const [fieldUpdateText, setFieldUpdateText] = useState('');
  
  // Metrics
  const [showMetrics, setShowMetrics] = useState(false);
  const [metrics, setMetrics] = useState<MetricsDashboard | null>(null);
  
  // UI state
  const [activeTab, setActiveTab] = useState<'record' | 'settings' | 'documents' | 'knowledge'>('record');
  
  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const templateInputRef = useRef<HTMLInputElement>(null);
  const docInputRef = useRef<HTMLInputElement>(null);
  const fieldRecorderRef = useRef<MediaRecorder | null>(null);
  const fieldAudioChunksRef = useRef<Blob[]>([]);
  const fieldTimerRef = useRef<number | null>(null);

  // Effects
  useEffect(() => {
    if (userId && !practitionerId) setPractitionerId(userId);
  }, [userId, practitionerId]);

  useEffect(() => {
    if (token) {
      loadTemplates();
      loadPatients();
    }
  }, [token]);

  useEffect(() => {
    if (patientId) {
      loadPatientDocuments(patientId);
      loadPatientContext(patientId);
      loadPatientRagItems(patientId);
    } else {
      setPatientDocs([]);
      setPatientContext(null);
      setPatientRagItems([]);
    }
  }, [patientId]);

  useEffect(() => {
    setEditedSoapNote('');
    setFieldUpdateStatus('');
    setFieldUpdateError('');
    setFieldUpdateText('');
    if (selectedRecording?.soapJson) {
      const options = collectFieldOptions(selectedRecording.soapJson);
      setFieldOptions(options);
      if (!selectedFieldPath || !options.find((opt) => opt.path === selectedFieldPath)) {
        setSelectedFieldPath(options[0]?.path || '');
      }
    } else {
      setFieldOptions([]);
      setSelectedFieldPath('');
    }
  }, [selectedRecording?.id, selectedRecording?.soapJson]);

  // Helper functions
  const fileToText = (file: File) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });

  const fileToBase64 = (file: File) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = String(reader.result || '');
        const base64 = result.split(',')[1] || '';
        resolve(base64);
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsDataURL(file);
    });

  const buildSoapNoteText = (note: { subjective?: string; objective?: string; assessment?: string; plan?: string }) => [
    `Subjective:\n${note.subjective || ''}`,
    `Objective:\n${note.objective || ''}`,
    `Assessment:\n${note.assessment || ''}`,
    `Plan:\n${note.plan || ''}`,
  ].join('\n\n');

  const mapDialect = (dialect: string) => {
    if (dialect === 'egyptian') return 'egypt';
    if (dialect === 'levantine') return 'levant';
    if (dialect === 'gulf') return 'gulf';
    if (dialect === 'msa') return 'msa';
    return undefined;
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const calculateEditDistance = (original: string, edited: string): number => {
    return Math.abs(original.length - edited.length) + (original !== edited ? Math.min(original.length, edited.length) / 10 : 0);
  };

  const formatRagDate = (value: any) => {
    if (!value) return '';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return '';
    return parsed.toLocaleString('ar-EG');
  };

  const ragTypeLabel = (type: string) => {
    switch (type) {
      case 'soap_note': return 'SOAP';
      case 'transcript': return 'تفريغ';
      case 'doc_summary': return 'ملخص وثيقة';
      case 'document': return 'وثيقة';
      default: return type || 'عنصر';
    }
  };

  const truncateSnippet = (text: string, limit = 140) => {
    if (!text) return '';
    if (text.length <= limit) return text;
    return `${text.slice(0, limit).trim()}...`;
  };

  const collectFieldOptions = (value: any, prefix = ''): Array<{ path: string; valueType: 'string' | 'list' }> => {
    const results: Array<{ path: string; valueType: 'string' | 'list' }> = [];
    if (typeof value === 'string') {
      if (prefix) results.push({ path: prefix, valueType: 'string' });
      return results;
    }
    if (Array.isArray(value)) {
      const stringItems = value.filter((item) => typeof item === 'string');
      if (stringItems.length === value.length) {
        if (prefix) results.push({ path: prefix, valueType: 'list' });
        return results;
      }
      value.forEach((item, index) => {
        results.push(...collectFieldOptions(item, `${prefix}[${index}]`));
      });
      return results;
    }
    if (value && typeof value === 'object') {
      Object.entries(value).forEach(([key, child]) => {
        results.push(...collectFieldOptions(child, prefix ? `${prefix}.${key}` : key));
      });
    }
    return results;
  };

  // API functions
  const loadMetrics = async () => {
    try {
      const data = await api.getClinicalMetricsDashboard();
      setMetrics(data);
    } catch (err) {
      console.error('Failed to load metrics:', err);
    }
  };

  const loadTemplates = async () => {
    try {
      const data = await api.listSoapTemplates();
      const list = data.templates || [];
      setTemplates(list);
      if (!selectedTemplateId && list.length > 0) setSelectedTemplateId(list[0].id);
    } catch (err) {
      console.error('Failed to load templates:', err);
      setTemplateError('تعذر تحميل القوالب');
    }
  };

  const loadPatients = async () => {
    try {
      setPatientStatus('loading');
      const data = await api.listPatients();
      const list = data.patients || [];
      setPatients(list);
      if (!patientId && list.length > 0) {
        setPatientId(list[0].id);
        if (list[0].displayName) setPatientName(list[0].displayName);
      }
      setPatientStatus('idle');
      setPatientError('');
    } catch (err: any) {
      console.error('Failed to load patients:', err);
      setPatientStatus('error');
      setPatientError('تعذر تحميل قائمة المرضى');
    }
  };

  const createPatient = async () => {
    if (!newPatientName.trim()) {
      setPatientError('يرجى إدخال اسم المريض');
      return;
    }
    try {
      setPatientStatus('loading');
      const created = await api.createPatient({
        displayName: newPatientName.trim(),
        externalId: newPatientExternalId.trim() || undefined,
      });
      const updatedList = [{ id: created.id, displayName: created.displayName, externalId: created.externalId }, ...patients];
      setPatients(updatedList);
      setPatientId(created.id);
      setPatientName(created.displayName || newPatientName.trim());
      setNewPatientName('');
      setNewPatientExternalId('');
      setPatientStatus('idle');
      setPatientError('');
    } catch (err: any) {
      console.error('Create patient failed:', err);
      setPatientStatus('error');
      setPatientError(err.message || 'فشل إنشاء المريض');
    }
  };

  const loadPatientDocuments = async (id: string) => {
    try {
      const data = await api.listPatientDocuments(id);
      setPatientDocs(data.documents || []);
    } catch (err) {
      console.error('Failed to load patient documents:', err);
      setPatientDocs([]);
    }
  };

  const loadPatientContext = async (id: string) => {
    try {
      const data = await api.getPatientContext(id);
      setPatientContext(data);
    } catch (err) {
      console.error('Failed to load patient context:', err);
      setPatientContext(null);
    }
  };

  const loadPatientRagItems = async (id: string) => {
    try {
      setRagItemsStatus('loading');
      const data = await api.listPatientRagItems(id);
      setPatientRagItems(data.items || []);
      setRagItemsStatus('idle');
      setRagItemsError('');
    } catch (err) {
      console.error('Failed to load patient RAG items:', err);
      setPatientRagItems([]);
      setRagItemsStatus('error');
      setRagItemsError('تعذر تحميل سجل المريض');
    }
  };

  const handleDocFile = async (file: File) => {
    try {
      setDocError('');
      setDocStatus('idle');
      setDocFileName(file.name);
      setDocFileType(file.type || '');
      const base64 = await fileToBase64(file);
      setDocFileBase64(base64);
      if (!docTitle) setDocTitle(file.name.replace(/\.[^.]+$/, ''));
      if (file.type.startsWith('text/') || /\.(txt|csv|json)$/i.test(file.name)) {
        const content = await fileToText(file);
        setDocText(content);
      } else {
        setDocText('');
      }
    } catch (err: any) {
      console.error('Doc file read failed:', err);
      setDocError('تعذر قراءة الملف');
    }
  };

  const uploadPatientDocument = async () => {
    if (!patientId) {
      setDocError('اختر المريض أولاً');
      return;
    }
    const trimmedText = docText.trim();
    const hasFile = Boolean(docFileBase64);
    if (!trimmedText && !hasFile) {
      setDocError('أدخل نص الوثيقة أو ارفع ملفاً');
      return;
    }
    try {
      setDocStatus('processing');
      setDocError('');
      const uploaded = await api.uploadPatientDocument(patientId, {
        title: docTitle.trim() || undefined,
        content: trimmedText || undefined,
        contentBase64: hasFile ? docFileBase64 : undefined,
        fileName: docFileName || undefined,
        contentType: docFileType || (trimmedText ? 'text/plain' : undefined),
        source: 'ui',
        summarize: summarizeDoc,
      });
      setPatientDocs((prev) => [uploaded, ...prev]);
      setDocTitle('');
      setDocText('');
      setDocFileName('');
      setDocFileType('');
      setDocFileBase64('');
      setDocStatus('done');
      await loadPatientContext(patientId);
      await loadPatientRagItems(patientId);
    } catch (err: any) {
      console.error('Upload document failed:', err);
      setDocStatus('error');
      setDocError(err.message || 'فشل رفع الوثيقة');
    }
  };

  const handleTemplateUpload = async (file: File) => {
    try {
      setTemplateError('');
      const raw = await fileToText(file);
      const parsed = JSON.parse(raw);
      const name = templateName.trim() || file.name.replace(/\.[^.]+$/, '');
      const result = await api.createSoapTemplate({ name, template: parsed });
      setTemplateName('');
      await loadTemplates();
      setSelectedTemplateId(result.id);
    } catch (err: any) {
      console.error('Template upload failed:', err);
      setTemplateError('قالب غير صالح أو فشل التحميل');
    }
  };

  const submitRagNote = async () => {
    if (!ragText.trim()) {
      setRagError('يرجى إدخال نص المعرفة');
      return;
    }
    try {
      setRagError('');
      setRagStatus('processing');
      await api.addRagNote({
        title: ragTitle.trim() || undefined,
        text: ragText.trim(),
        metadata: { source: 'ui', clinicianId: practitionerId || undefined },
      });
      setRagStatus('done');
      setRagTitle('');
      setRagText('');
    } catch (err: any) {
      console.error('RAG note failed:', err);
      setRagStatus('error');
      setRagError(err.message || 'فشل إضافة المعرفة');
    }
  };

  const recordReviewMetrics = async (recording: AudioRecording, accepted: boolean, editedText: string) => {
    const reviewEndTime = Date.now();
    const reviewStartTime = recording.reviewStartTime || reviewEndTime;
    const timeToReview = Math.round((reviewEndTime - reviewStartTime) / 1000);
    const editDistance = calculateEditDistance(recording.soapNote || '', editedText);
    try {
      await api.recordClinicalReview({
        recordingId: recording.id,
        accepted,
        editDistance,
        timeToReview,
        clinicianId: practitionerId || 'unknown',
      });
      loadMetrics();
    } catch (err) {
      console.error('Failed to record metrics:', err);
    }
  };

  // Recording functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const audioFile = new File([audioBlob], `recording-${Date.now()}.webm`, { type: 'audio/webm' });
        const newRecording: AudioRecording = {
          id: `rec-${Date.now()}`,
          file: audioFile,
          duration: recordingTime,
          timestamp: new Date(),
          status: 'pending',
        };
        setRecordings((prev) => [newRecording, ...prev]);
        setRecordingTime(0);
        await processRecording(newRecording);
      };

      mediaRecorder.start();
      setIsRecording(true);
      timerRef.current = window.setInterval(() => setRecordingTime((prev) => prev + 1), 1000);
    } catch (error: any) {
      console.error('Failed to start recording:', error);
      alert(`خطأ: ${error.message}`);
    }
  };

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
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const processRecording = async (recording: AudioRecording) => {
    try {
      if (!patientId || !practitionerId) {
        throw new Error('الرجاء إدخال معرف المريض والطبيب');
      }
      setRecordings((prev) => prev.map((r) => (r.id === recording.id ? { ...r, status: 'processing' } : r)));

      const audioBase64 = await fileToBase64(recording.file);
      const mappedDialect = mapDialect(selectedDialect);
      const transcriptResponse: any = await api.transcribeAudio(audioBase64, recording.id, mappedDialect, 'ar', true, false);
      const transcript = transcriptResponse.text || '';

      const soapResponse = await api.createSOAPNote({
        transcript,
        sessionId: recording.id,
        patientId,
        practitionerId,
        templateId: selectedTemplateId || undefined,
        patientName,
        providerName,
        dateOfVisit,
      });

      const soapNote = buildSoapNoteText(soapResponse);
      setRecordings((prev) =>
        prev.map((r) =>
          r.id === recording.id
            ? {
                ...r,
                status: 'completed',
                transcript,
                soapNote,
                soapJson: (soapResponse as any).soapJson ?? (soapResponse as any).soap_json,
                noteId: soapResponse.id,
                templateId: (soapResponse as any).templateId ?? (soapResponse as any).template_id,
                dialect: selectedDialect,
                autoDetected: selectedDialect === 'auto',
              }
            : r
        )
      );
    } catch (error: any) {
      console.error('Processing error:', error);
      setRecordings((prev) => prev.map((r) => (r.id === recording.id ? { ...r, status: 'error', error: error.message } : r)));
    }
  };

  // Field recording
  const startFieldRecording = async () => {
    if (!selectedRecording?.noteId) {
      setFieldUpdateError('يرجى اختيار ملاحظة أولاً');
      return;
    }
    if (!selectedFieldPath) {
      setFieldUpdateError('يرجى اختيار الحقل المراد تحديثه');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      fieldRecorderRef.current = mediaRecorder;
      fieldAudioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) fieldAudioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(fieldAudioChunksRef.current, { type: 'audio/webm' });
        const audioFile = new File([audioBlob], `field-${Date.now()}.webm`, { type: 'audio/webm' });
        setFieldRecordingTime(0);
        await processFieldRecording(audioFile);
      };

      mediaRecorder.start();
      setIsFieldRecording(true);
      setFieldUpdateStatus('');
      setFieldUpdateError('');
      fieldTimerRef.current = window.setInterval(() => setFieldRecordingTime((prev) => prev + 1), 1000);
    } catch (error: any) {
      console.error('Failed to start field recording:', error);
      setFieldUpdateError(error.message || 'فشل التسجيل');
    }
  };

  const stopFieldRecording = () => {
    if (fieldRecorderRef.current && isFieldRecording) {
      fieldRecorderRef.current.stop();
      fieldRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      setIsFieldRecording(false);
      if (fieldTimerRef.current) {
        clearInterval(fieldTimerRef.current);
        fieldTimerRef.current = null;
      }
    }
  };

  const processFieldRecording = async (audioFile: File) => {
    if (!selectedRecording?.noteId) return;
    try {
      setFieldUpdateStatus('processing');
      const audioBase64 = await fileToBase64(audioFile);
      const selectedMeta = fieldOptions.find((opt) => opt.path === selectedFieldPath);
      const valueType = selectedMeta?.valueType;
      const mappedDialect = mapDialect(selectedDialect);
      const updated = await api.updateSOAPNoteField(selectedRecording.noteId, {
        fieldPath: selectedFieldPath,
        audio: audioBase64,
        mode: fieldUpdateMode,
        valueType,
        dialect: mappedDialect,
        language: 'ar',
      });
      const updatedNote = {
        ...selectedRecording,
        soapJson: updated.soapJson ?? updated.soap_json ?? selectedRecording.soapJson,
        soapNote: buildSoapNoteText(updated),
      };
      setRecordings((prev) => prev.map((r) => (r.id === selectedRecording.id ? { ...r, ...updatedNote } : r)));
      setSelectedRecording(updatedNote);
      setFieldUpdateStatus('done');
    } catch (error: any) {
      console.error('Field update error:', error);
      setFieldUpdateStatus('error');
      setFieldUpdateError(error.message || 'فشل تحديث الحقل');
    }
  };

  const submitFieldTextUpdate = async () => {
    if (!selectedRecording?.noteId) return;
    if (!selectedFieldPath || !fieldUpdateText.trim()) {
      setFieldUpdateError('يرجى كتابة النص المطلوب إضافته');
      return;
    }
    try {
      setFieldUpdateStatus('processing');
      setFieldUpdateError('');
      const selectedMeta = fieldOptions.find((opt) => opt.path === selectedFieldPath);
      const valueType = selectedMeta?.valueType;
      const updated = await api.updateSOAPNoteField(selectedRecording.noteId, {
        fieldPath: selectedFieldPath,
        transcript: fieldUpdateText.trim(),
        mode: fieldUpdateMode,
        valueType,
      });
      const updatedNote = {
        ...selectedRecording,
        soapJson: updated.soapJson ?? updated.soap_json ?? selectedRecording.soapJson,
        soapNote: buildSoapNoteText(updated),
      };
      setRecordings((prev) => prev.map((r) => (r.id === selectedRecording.id ? { ...r, ...updatedNote } : r)));
      setSelectedRecording(updatedNote);
      setFieldUpdateStatus('done');
      setFieldUpdateText('');
    } catch (error: any) {
      console.error('Field update error:', error);
      setFieldUpdateStatus('error');
      setFieldUpdateError(error.message || 'فشل تحديث الحقل');
    }
  };

  const saveToEHR = async () => {
    if (!selectedRecording || !selectedRecording.noteId) return;
    setIsSaving(true);
    try {
      const finalText = editedSoapNote || selectedRecording.soapNote || '';
      if (finalText && finalText !== selectedRecording.soapNote) {
        const updated = await api.updateSOAPNoteSections(selectedRecording.noteId, { soapText: finalText });
        const updatedNote = {
          ...selectedRecording,
          soapNote: buildSoapNoteText(updated),
          soapJson: updated.soapJson ?? updated.soap_json ?? selectedRecording.soapJson,
        };
        setRecordings((prev) => prev.map((r) => (r.id === selectedRecording.id ? { ...r, ...updatedNote } : r)));
        setSelectedRecording(updatedNote);
        setEditedSoapNote('');
      }
      await api.approveSOAPNote(selectedRecording.noteId);
      alert('✅ تم الحفظ بنجاح!');
    } catch (error: any) {
      alert(`❌ خطأ في الحفظ: ${error.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const toggleExpanded = (path: string) => setExpandedPaths((prev) => ({ ...prev, [path]: !prev[path] }));

  const renderJsonTree = (value: any, path = '', depth = 0): JSX.Element[] => {
    const nodes: JSX.Element[] = [];
    const indent = Math.min(depth * 12, 48);
    const selectablePaths = new Set(fieldOptions.map((opt) => opt.path));

    const renderNode = (label: string, child: any, childPath: string) => {
      const isArray = Array.isArray(child);
      const isStringList = isArray && child.every((item) => typeof item === 'string');
      const isLeaf = typeof child === 'string' || isStringList;
      const isExpanded = expandedPaths[childPath];
      const selectable = selectablePaths.has(childPath);
      const preview = typeof child === 'string' ? child.slice(0, 40) : isStringList ? `${child.length} عناصر` : '';

      nodes.push(
        <button
          key={childPath}
          type="button"
          onClick={() => {
            if (isLeaf && selectable) setSelectedFieldPath(childPath);
            else toggleExpanded(childPath);
          }}
          className={`w-full text-right flex items-center justify-between gap-2 px-3 py-2 rounded-lg border transition-colors ${
            selectedFieldPath === childPath && selectable
              ? 'border-purple-400 bg-purple-500/30'
              : 'border-slate-600 bg-slate-700/50 hover:bg-slate-700'
          }`}
          style={{ paddingRight: indent + 12 }}
        >
          <span className="text-sm text-slate-200">
            {label}
            {selectable && <span className="text-xs text-purple-300 mr-2">قابل للتحديث</span>}
          </span>
          <span className="text-xs text-slate-400">{isLeaf ? preview || 'نص' : isExpanded ? '−' : '+'}</span>
        </button>
      );

      if (!isLeaf && isExpanded) {
        if (isArray) {
          child.forEach((item: any, index: number) => renderNode(`[${index}]`, item, `${childPath}[${index}]`));
        } else if (child && typeof child === 'object') {
          Object.entries(child).forEach(([key, val]) => renderNode(key, val, childPath ? `${childPath}.${key}` : key));
        }
      }
    };

    if (value && typeof value === 'object') {
      Object.entries(value).forEach(([key, val]) => renderNode(key, val, path ? `${path}.${key}` : key));
    }
    return nodes;
  };

  const statsCards = [
    { label: 'إجمالي التسجيلات', value: recordings.length, color: 'bg-purple-600' },
    { label: 'مكتمل', value: recordings.filter((r) => r.status === 'completed').length, color: 'bg-green-600' },
    { label: 'قيد المعالجة', value: recordings.filter((r) => r.status === 'processing').length, color: 'bg-amber-600' },
    { label: 'خطأ', value: recordings.filter((r) => r.status === 'error').length, color: 'bg-red-600' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900" dir="rtl">
      {/* Header */}
      <header className="bg-slate-800/80 backdrop-blur border-b border-slate-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
              <IconStethoscope className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">توثيق السجلات الطبية</h1>
              <p className="text-sm text-slate-400">نظام SOAP الذكي</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* Stats Summary */}
            <div className="hidden md:flex items-center gap-2">
              {statsCards.map((stat) => (
                <div key={stat.label} className={`${stat.color} px-3 py-1.5 rounded-lg`}>
                  <span className="text-white text-sm font-semibold">{stat.value}</span>
                </div>
              ))}
            </div>
            {/* Metrics Toggle */}
            <button
              onClick={() => {
                setShowMetrics(!showMetrics);
                if (!showMetrics && !metrics) loadMetrics();
              }}
              className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
            >
              <IconChartBar className="w-5 h-5" />
              <span className="hidden sm:inline">المقاييس</span>
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* Metrics Dashboard */}
        <AnimatePresence>
          {showMetrics && metrics && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-slate-800 border border-slate-700 rounded-2xl p-6 mb-6"
            >
              <h2 className="text-lg font-bold text-white mb-4">لوحة المقاييس</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-purple-900/30 border border-purple-500/30 rounded-xl p-4">
                  <p className="text-purple-300 text-sm">إجمالي الملاحظات</p>
                  <p className="text-3xl font-bold text-white mt-1">{metrics.overview.totalNotes}</p>
                </div>
                <div className="bg-green-900/30 border border-green-500/30 rounded-xl p-4">
                  <p className="text-green-300 text-sm">معدل القبول</p>
                  <p className="text-3xl font-bold text-white mt-1">{(metrics.overview.acceptanceRate * 100).toFixed(1)}%</p>
                </div>
                <div className="bg-blue-900/30 border border-blue-500/30 rounded-xl p-4">
                  <p className="text-blue-300 text-sm">متوسط التعديل</p>
                  <p className="text-3xl font-bold text-white mt-1">{metrics.overview.avgEditDistance.toFixed(1)}</p>
                </div>
                <div className="bg-amber-900/30 border border-amber-500/30 rounded-xl p-4">
                  <p className="text-amber-300 text-sm">وقت المراجعة</p>
                  <p className="text-3xl font-bold text-white mt-1">{metrics.overview.avgReviewTime.toFixed(0)}s</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Sidebar - Controls */}
          <div className="lg:col-span-4 space-y-4">
            {/* Tabs */}
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-1 flex gap-1">
              {[
                { id: 'record', label: 'تسجيل', icon: IconMicrophone },
                { id: 'settings', label: 'إعدادات', icon: IconSettings },
                { id: 'documents', label: 'وثائق', icon: IconFolder },
                { id: 'knowledge', label: 'معرفة', icon: IconBrain },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    activeTab === tab.id ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white hover:bg-slate-700'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{tab.label}</span>
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="bg-slate-800 border border-slate-700 rounded-2xl p-5">
              {/* Record Tab */}
              {activeTab === 'record' && (
                <div className="space-y-5">
                  {/* Recording Button */}
                  <div className="text-center">
                    <motion.button
                      onClick={isRecording ? stopRecording : startRecording}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className={`w-24 h-24 rounded-full flex items-center justify-center mx-auto shadow-2xl transition-colors ${
                        isRecording
                          ? 'bg-red-500 hover:bg-red-600 shadow-red-500/40'
                          : 'bg-gradient-to-br from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 shadow-purple-500/40'
                      }`}
                    >
                      {isRecording ? <IconPlayerStop className="w-10 h-10 text-white" /> : <IconMicrophone className="w-10 h-10 text-white" />}
                    </motion.button>
                    {isRecording && <p className="text-2xl font-mono text-red-400 font-bold mt-4">{formatTime(recordingTime)}</p>}
                    <p className="text-sm text-slate-400 mt-2">{isRecording ? 'جارٍ التسجيل...' : 'انقر للبدء'}</p>
                  </div>

                  <div className="border-t border-slate-700 pt-5">
                    {/* Dialect Selector */}
                    <label className="block text-sm text-slate-300 mb-2">اللهجة</label>
                    <select
                      value={selectedDialect}
                      onChange={(e) => setSelectedDialect(e.target.value)}
                      className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="auto">كشف تلقائي</option>
                      <option value="egyptian">مصري</option>
                      <option value="levantine">شامي</option>
                      <option value="gulf">خليجي</option>
                      <option value="msa">فصحى</option>
                    </select>
                  </div>

                  {/* File Upload */}
                  <div>
                    <input ref={fileInputRef} type="file" accept="audio/*" multiple onChange={handleFileUpload} className="hidden" />
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="w-full flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 text-white font-medium py-3 px-4 rounded-xl transition-colors"
                    >
                      <IconUpload className="w-5 h-5" />
                      رفع ملف صوتي
                    </button>
                    <p className="text-xs text-slate-500 text-center mt-2">MP3, WAV, M4A, WebM</p>
                  </div>
                </div>
              )}

              {/* Settings Tab */}
              {activeTab === 'settings' && (
                <div className="space-y-4">
                  {/* Patient Selection */}
                  <div>
                    <label className="flex items-center gap-2 text-sm text-slate-300 mb-2">
                      <IconUser className="w-4 h-4" />
                      المريض
                    </label>
                    <select
                      value={patientId}
                      onChange={(e) => {
                        const nextId = e.target.value;
                        setPatientId(nextId);
                        const selected = patients.find((p) => p.id === nextId);
                        if (selected?.displayName) setPatientName(selected.displayName);
                      }}
                      className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="">-- اختر مريضاً --</option>
                      {patients.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.displayName || p.id}
                        </option>
                      ))}
                    </select>
                    {patientStatus === 'loading' && <p className="text-xs text-slate-400 mt-1">جارٍ التحميل...</p>}
                    {patientError && <p className="text-xs text-red-400 mt-1">{patientError}</p>}
                  </div>

                  {/* New Patient */}
                  <div className="bg-slate-700/50 rounded-xl p-4">
                    <p className="text-sm text-slate-300 mb-3 flex items-center gap-2">
                      <IconPlus className="w-4 h-4" />
                      مريض جديد
                    </p>
                    <input
                      value={newPatientName}
                      onChange={(e) => setNewPatientName(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-600 border border-slate-500 rounded-lg text-white text-sm mb-2"
                      placeholder="اسم المريض"
                    />
                    <input
                      value={newPatientExternalId}
                      onChange={(e) => setNewPatientExternalId(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-600 border border-slate-500 rounded-lg text-white text-sm mb-2"
                      placeholder="رقم الملف (اختياري)"
                      dir="ltr"
                    />
                    <button onClick={createPatient} className="w-full bg-green-600 hover:bg-green-500 text-white py-2 rounded-lg text-sm font-medium transition-colors">
                      إضافة المريض
                    </button>
                  </div>

                  {/* Practitioner */}
                  <div>
                    <label className="text-sm text-slate-300 mb-2 block">معرف الطبيب</label>
                    <input
                      value={practitionerId}
                      onChange={(e) => setPractitionerId(e.target.value)}
                      className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-xl text-white"
                      placeholder="clinician-456"
                      dir="ltr"
                    />
                  </div>

                  {/* Names & Date */}
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-sm text-slate-300 mb-2 block">اسم المريض</label>
                      <input
                        value={patientName}
                        onChange={(e) => setPatientName(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-sm text-slate-300 mb-2 block">اسم الطبيب</label>
                      <input
                        value={providerName}
                        onChange={(e) => setProviderName(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="flex items-center gap-2 text-sm text-slate-300 mb-2">
                      <IconCalendar className="w-4 h-4" />
                      تاريخ الزيارة
                    </label>
                    <input
                      type="date"
                      value={dateOfVisit}
                      onChange={(e) => setDateOfVisit(e.target.value)}
                      className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-xl text-white"
                      dir="ltr"
                    />
                  </div>

                  {/* Template */}
                  <div>
                    <label className="flex items-center gap-2 text-sm text-slate-300 mb-2">
                      <IconTemplate className="w-4 h-4" />
                      قالب SOAP
                    </label>
                    <select
                      value={selectedTemplateId}
                      onChange={(e) => setSelectedTemplateId(e.target.value)}
                      className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-xl text-white mb-2"
                    >
                      {templates.map((tpl) => (
                        <option key={tpl.id} value={tpl.id}>
                          {tpl.name}
                        </option>
                      ))}
                    </select>
                    <input ref={templateInputRef} type="file" accept="application/json" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleTemplateUpload(f); if (templateInputRef.current) templateInputRef.current.value = ''; }} className="hidden" />
                    <div className="flex gap-2">
                      <input
                        value={templateName}
                        onChange={(e) => setTemplateName(e.target.value)}
                        className="flex-1 px-3 py-2 bg-slate-600 border border-slate-500 rounded-lg text-white text-sm"
                        placeholder="اسم القالب الجديد"
                      />
                      <button onClick={() => templateInputRef.current?.click()} className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium">
                        رفع JSON
                      </button>
                    </div>
                    {templateError && <p className="text-xs text-red-400 mt-1">{templateError}</p>}
                  </div>
                </div>
              )}

              {/* Documents Tab */}
              {activeTab === 'documents' && (
                <div className="space-y-4">
                  <p className="text-slate-300 text-sm">رفع وثائق للمريض المحدد</p>
                  {!patientId && <p className="text-amber-400 text-xs">اختر المريض أولاً من الإعدادات</p>}

                  <input ref={docInputRef} type="file" accept=".txt,.json,.csv,.pdf,.docx" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleDocFile(f); if (docInputRef.current) docInputRef.current.value = ''; }} className="hidden" />

                  <div className="flex gap-2">
                    <input
                      value={docTitle}
                      onChange={(e) => setDocTitle(e.target.value)}
                      className="flex-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm"
                      placeholder="عنوان الوثيقة"
                    />
                    <button onClick={() => docInputRef.current?.click()} className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm">
                      رفع ملف
                    </button>
                  </div>
                  {docFileName && <p className="text-xs text-slate-400">📎 {docFileName}</p>}

                  <textarea
                    value={docText}
                    onChange={(e) => setDocText(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-xl text-white text-sm resize-none"
                    rows={4}
                    placeholder="أو الصق بيانات المريض هنا..."
                  />

                  <div className="flex items-center justify-between">
                    <label className="flex items-center gap-2 text-xs text-slate-400">
                      <input type="checkbox" checked={summarizeDoc} onChange={(e) => setSummarizeDoc(e.target.checked)} className="rounded" />
                      تلخيص تلقائي
                    </label>
                    <button
                      onClick={uploadPatientDocument}
                      disabled={!patientId}
                      className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium"
                    >
                      حفظ الوثيقة
                    </button>
                  </div>

                  {docStatus === 'processing' && <p className="text-xs text-purple-300 flex items-center gap-2"><IconLoader2 className="w-4 h-4 animate-spin" /> جارٍ الحفظ...</p>}
                  {docStatus === 'done' && <p className="text-xs text-green-400">✓ تم الحفظ</p>}
                  {docError && <p className="text-xs text-red-400">{docError}</p>}

                  {/* Existing Docs */}
                  {patientDocs.length > 0 && (
                    <div className="border-t border-slate-700 pt-4">
                      <p className="text-sm text-slate-300 mb-2">الوثائق المحفوظة ({patientDocs.length})</p>
                      <div className="space-y-2 max-h-48 overflow-y-auto">
                        {patientDocs.slice(0, 5).map((doc) => (
                          <div key={doc.id} className="bg-slate-700/50 rounded-lg p-3">
                            <p className="text-sm text-white font-medium">{doc.title || 'وثيقة'}</p>
                            <p className="text-xs text-slate-400 mt-1">{truncateSnippet(doc.summaryText || doc.summary_text || '', 80)}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Knowledge Tab */}
              {activeTab === 'knowledge' && (
                <div className="space-y-4">
                  <p className="text-slate-300 text-sm">أضف معرفة للعيادة (سياسات، تعليمات)</p>

                  <input
                    value={ragTitle}
                    onChange={(e) => setRagTitle(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm"
                    placeholder="عنوان (اختياري)"
                  />
                  <textarea
                    value={ragText}
                    onChange={(e) => setRagText(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-xl text-white text-sm resize-none"
                    rows={5}
                    placeholder="المعلومات التي تريد إضافتها للنظام..."
                  />
                  <button
                    onClick={submitRagNote}
                    className="w-full bg-purple-600 hover:bg-purple-500 text-white py-2 rounded-lg text-sm font-medium flex items-center justify-center gap-2"
                  >
                    <IconBrain className="w-4 h-4" />
                    إضافة للمعرفة
                  </button>
                  {ragStatus === 'processing' && <p className="text-xs text-purple-300 flex items-center gap-2"><IconLoader2 className="w-4 h-4 animate-spin" /> جارٍ الحفظ...</p>}
                  {ragStatus === 'done' && <p className="text-xs text-green-400">✓ تم الحفظ</p>}
                  {ragError && <p className="text-xs text-red-400">{ragError}</p>}

                  {/* Patient RAG Items */}
                  {patientRagItems.length > 0 && (
                    <div className="border-t border-slate-700 pt-4">
                      <p className="text-sm text-slate-300 mb-2">سجل المريض</p>
                      {ragItemsStatus === 'loading' && <p className="text-xs text-slate-400">جارٍ التحميل...</p>}
                      <div className="space-y-2 max-h-48 overflow-y-auto">
                        {patientRagItems.slice(0, 8).map((item) => (
                          <div key={item.id} className="bg-slate-700/50 rounded-lg p-3">
                            <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                              <span>{ragTypeLabel(item.itemType || item.item_type)}</span>
                              <span>{formatRagDate(item.createdAt || item.created_at)}</span>
                            </div>
                            <p className="text-sm text-white">{item.title || 'عنصر'}</p>
                            <p className="text-xs text-slate-400 mt-1">{truncateSnippet(item.contentText || item.content_text || '', 80)}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Patient Context */}
                  {patientContext && (
                    <div className="border-t border-slate-700 pt-4">
                      <p className="text-sm text-slate-300 mb-2">سياق المريض</p>
                      <pre className="bg-slate-900 rounded-lg p-3 text-xs text-slate-300 overflow-auto max-h-32">
                        {JSON.stringify(patientContext, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Center - Recordings List */}
          <div className="lg:col-span-3">
            <div className="bg-slate-800 border border-slate-700 rounded-2xl p-5 sticky top-24">
              <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <IconFileText className="w-5 h-5 text-purple-400" />
                التسجيلات
              </h2>
              <div className="space-y-2 max-h-[calc(100vh-200px)] overflow-y-auto">
                {recordings.length === 0 ? (
                  <div className="text-center py-12">
                    <IconFileText className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                    <p className="text-slate-500">لا توجد تسجيلات</p>
                  </div>
                ) : (
                  recordings.map((recording) => (
                    <motion.button
                      key={recording.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      onClick={() => {
                        setSelectedRecording(recording);
                        setEditedSoapNote('');
                      }}
                      className={`w-full text-right p-4 rounded-xl border transition-all ${
                        selectedRecording?.id === recording.id
                          ? 'border-purple-500 bg-purple-500/20'
                          : 'border-slate-700 bg-slate-700/50 hover:border-slate-600'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-white text-sm truncate">{recording.file.name}</p>
                          <p className="text-xs text-slate-400 mt-1">{recording.timestamp.toLocaleString('ar-EG')}</p>
                        </div>
                        <span
                          className={`px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1 ${
                            recording.status === 'completed'
                              ? 'bg-green-500/20 text-green-400'
                              : recording.status === 'processing'
                              ? 'bg-amber-500/20 text-amber-400'
                              : recording.status === 'error'
                              ? 'bg-red-500/20 text-red-400'
                              : 'bg-slate-600 text-slate-400'
                          }`}
                        >
                          {recording.status === 'completed' && <><IconCheck className="w-3 h-3" /> OK</>}
                          {recording.status === 'processing' && <><IconLoader2 className="w-3 h-3 animate-spin" /> ...</>}
                          {recording.status === 'error' && <><IconX className="w-3 h-3" /> خطأ</>}
                          {recording.status === 'pending' && <><IconClock className="w-3 h-3" /> انتظار</>}
                        </span>
                      </div>
                    </motion.button>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Right - Recording Details */}
          <div className="lg:col-span-5">
            <AnimatePresence mode="wait">
              {selectedRecording ? (
                <motion.div
                  key={selectedRecording.id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="bg-slate-800 border border-slate-700 rounded-2xl p-6"
                >
                  <h2 className="text-lg font-bold text-white mb-4">تفاصيل التسجيل</h2>

                  {selectedRecording.status === 'processing' && (
                    <div className="text-center py-16">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
                        className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full mx-auto mb-4"
                      />
                      <p className="text-slate-400">جارٍ المعالجة...</p>
                    </div>
                  )}

                  {selectedRecording.status === 'error' && (
                    <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-4">
                      <p className="text-red-400 flex items-center gap-2">
                        <IconX className="w-5 h-5" />
                        {selectedRecording.error}
                      </p>
                    </div>
                  )}

                  {selectedRecording.status === 'completed' && (
                    <div className="space-y-5">
                      {/* Transcript */}
                      <div>
                        <label className="text-sm text-slate-300 mb-2 block">النص المكتوب:</label>
                        <div className="bg-slate-900 border border-slate-700 rounded-xl p-4 max-h-40 overflow-y-auto">
                          <p className="text-slate-200 text-sm leading-relaxed">{selectedRecording.transcript}</p>
                        </div>
                      </div>

                      {/* SOAP Note */}
                      <div>
                        <label className="text-sm text-slate-300 mb-2 block">ملاحظة SOAP:</label>
                        <textarea
                          value={editedSoapNote || selectedRecording.soapNote}
                          onChange={(e) => {
                            setEditedSoapNote(e.target.value);
                            if (!selectedRecording.reviewStartTime) {
                              setRecordings((prev) =>
                                prev.map((r) => (r.id === selectedRecording.id ? { ...r, reviewStartTime: Date.now() } : r))
                              );
                            }
                          }}
                          className="w-full bg-slate-900 border border-slate-700 rounded-xl p-4 text-slate-200 text-sm leading-relaxed resize-none focus:outline-none focus:ring-2 focus:ring-purple-500"
                          rows={10}
                        />
                      </div>

                      {/* JSON Output */}
                      {selectedRecording.soapJson && (
                        <div>
                          <label className="text-sm text-slate-300 mb-2 block">JSON الناتج:</label>
                          <pre className="bg-slate-900 border border-slate-700 rounded-xl p-4 text-xs text-slate-300 overflow-auto max-h-48">
                            {JSON.stringify(selectedRecording.soapJson, null, 2)}
                          </pre>
                        </div>
                      )}

                      {/* Field Update Section */}
                      {selectedRecording.noteId && (
                        <div className="bg-slate-700/50 border border-slate-600 rounded-xl p-4">
                          <h3 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
                            <IconMicrophone className="w-4 h-4 text-purple-400" />
                            تحديث حقل بالصوت أو النص
                          </h3>
                          <div className="grid gap-4 lg:grid-cols-2">
                            <div>
                              <label className="text-xs text-slate-400 mb-2 block">شجرة الحقول</label>
                              <div className="bg-slate-800 rounded-lg p-2 max-h-48 overflow-y-auto space-y-1">
                                {selectedRecording.soapJson ? renderJsonTree(selectedRecording.soapJson) : <p className="text-xs text-slate-500">لا يوجد JSON</p>}
                              </div>
                            </div>
                            <div className="space-y-3">
                              <div>
                                <label className="text-xs text-slate-400 mb-1 block">الحقل المختار</label>
                                <p className="text-sm text-purple-300">{selectedFieldPath || 'غير محدد'}</p>
                              </div>
                              <div>
                                <label className="text-xs text-slate-400 mb-1 block">طريقة التحديث</label>
                                <select
                                  value={fieldUpdateMode}
                                  onChange={(e) => setFieldUpdateMode(e.target.value as 'append' | 'replace')}
                                  className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm"
                                >
                                  <option value="append">إضافة</option>
                                  <option value="replace">استبدال</option>
                                </select>
                              </div>
                              <textarea
                                value={fieldUpdateText}
                                onChange={(e) => setFieldUpdateText(e.target.value)}
                                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm resize-none"
                                rows={3}
                                placeholder="أضف النص هنا..."
                              />
                              <button
                                onClick={submitFieldTextUpdate}
                                className="w-full bg-blue-600 hover:bg-blue-500 text-white py-2 rounded-lg text-sm font-medium"
                              >
                                تحديث بالنص
                              </button>
                            </div>
                          </div>
                          <div className="flex items-center gap-3 mt-4">
                            {!isFieldRecording ? (
                              <button
                                onClick={startFieldRecording}
                                className="bg-purple-600 hover:bg-purple-500 text-white py-2 px-4 rounded-lg text-sm flex items-center gap-2"
                              >
                                <IconMicrophone className="w-4 h-4" />
                                تسجيل صوتي
                              </button>
                            ) : (
                              <button
                                onClick={stopFieldRecording}
                                className="bg-red-600 hover:bg-red-500 text-white py-2 px-4 rounded-lg text-sm flex items-center gap-2"
                              >
                                <IconPlayerStop className="w-4 h-4" />
                                إيقاف ({formatTime(fieldRecordingTime)})
                              </button>
                            )}
                            {fieldUpdateStatus === 'processing' && <span className="text-xs text-purple-300 flex items-center gap-1"><IconLoader2 className="w-4 h-4 animate-spin" /> جارٍ التحديث...</span>}
                            {fieldUpdateStatus === 'done' && <span className="text-xs text-green-400 flex items-center gap-1"><IconCheck className="w-4 h-4" /> تم</span>}
                          </div>
                          {fieldUpdateError && <p className="text-xs text-red-400 mt-2">{fieldUpdateError}</p>}
                        </div>
                      )}

                      {/* Actions */}
                      <div className="flex gap-3 pt-4 border-t border-slate-700">
                        <button
                          onClick={async () => {
                            const finalText = editedSoapNote || selectedRecording.soapNote || '';
                            await recordReviewMetrics(selectedRecording, true, finalText);
                            await saveToEHR();
                          }}
                          disabled={isSaving}
                          className="flex-1 bg-green-600 hover:bg-green-500 disabled:bg-slate-600 text-white font-medium py-3 rounded-xl flex items-center justify-center gap-2 transition-colors"
                        >
                          {isSaving ? <><IconLoader2 className="w-5 h-5 animate-spin" /> جارٍ الحفظ...</> : <><IconCheck className="w-5 h-5" /> قبول وحفظ</>}
                        </button>
                        <button
                          onClick={async () => {
                            const finalText = editedSoapNote || selectedRecording.soapNote || '';
                            await recordReviewMetrics(selectedRecording, false, finalText);
                            if (selectedRecording.noteId) await api.rejectSOAPNote(selectedRecording.noteId);
                            alert('تم رفض الملاحظة');
                          }}
                          className="flex-1 bg-red-600 hover:bg-red-500 text-white font-medium py-3 rounded-xl flex items-center justify-center gap-2 transition-colors"
                        >
                          <IconX className="w-5 h-5" />
                          رفض
                        </button>
                        <button
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
                          className="bg-slate-700 hover:bg-slate-600 text-white py-3 px-4 rounded-xl transition-colors"
                          title="تنزيل"
                        >
                          <IconFileDownload className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  )}
                </motion.div>
              ) : (
                <div className="bg-slate-800 border border-slate-700 rounded-2xl p-12 text-center">
                  <IconFileText className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                  <p className="text-slate-500">اختر تسجيلاً لعرض التفاصيل</p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>
    </div>
  );
}
