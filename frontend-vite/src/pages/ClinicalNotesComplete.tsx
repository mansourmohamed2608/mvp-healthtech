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
  IconStethoscope,
  IconPlus,
  IconFolder,
  IconTemplate,
  IconBrain,
  IconChevronDown,
  IconChevronRight
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
  dialect?: string;
}

interface MetricsDashboard {
  overview: {
    totalNotes: number;
    acceptanceRate: number;
    avgEditDistance: number;
    avgReviewTime: number;
  };
}

export default function ClinicalNotesComplete() {
  const { userId, token } = useAuthStore();
  
  // Core state
  const [recordings, setRecordings] = useState<AudioRecording[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [selectedRecording, setSelectedRecording] = useState<AudioRecording | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);
  const [editedSoapNote, setEditedSoapNote] = useState<string>('');
  const [selectedDialect, setSelectedDialect] = useState<string>('auto');
  const [metrics, setMetrics] = useState<MetricsDashboard | null>(null);
  
  // Patient & Practitioner
  const [patientId, setPatientId] = useState<string>('');
  const [practitionerId, setPractitionerId] = useState<string>('');
  const [patientName, setPatientName] = useState<string>('');
  const [providerName, setProviderName] = useState<string>('');
  const [dateOfVisit, setDateOfVisit] = useState<string>('');
  const [patients, setPatients] = useState<Array<{ id: string; displayName?: string; externalId?: string }>>([]);
  const [patientStatus, setPatientStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [patientError, setPatientError] = useState<string>('');
  const [newPatientName, setNewPatientName] = useState<string>('');
  const [newPatientExternalId, setNewPatientExternalId] = useState<string>('');
  const [showNewPatient, setShowNewPatient] = useState(false);
  
  // Documents
  const [patientDocs, setPatientDocs] = useState<Array<any>>([]);
  const [docTitle, setDocTitle] = useState<string>('');
  const [docText, setDocText] = useState<string>('');
  const [docFileName, setDocFileName] = useState<string>('');
  const [docFileType, setDocFileType] = useState<string>('');
  const [docFileBase64, setDocFileBase64] = useState<string>('');
  const [docStatus, setDocStatus] = useState<'idle' | 'processing' | 'done' | 'error'>('idle');
  const [docError, setDocError] = useState<string>('');
  const [summarizeDoc, setSummarizeDoc] = useState(true);
  
  // RAG & Context
  const [patientContext, setPatientContext] = useState<any>(null);
  const [patientRagItems, setPatientRagItems] = useState<Array<any>>([]);
  const [ragItemsStatus, setRagItemsStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [ragTitle, setRagTitle] = useState<string>('');
  const [ragText, setRagText] = useState<string>('');
  const [ragStatus, setRagStatus] = useState<string>('');
  const [ragError, setRagError] = useState<string>('');
  
  // Templates
  const [templates, setTemplates] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('');
  const [templateName, setTemplateName] = useState<string>('');
  const [templateError, setTemplateError] = useState<string>('');
  
  // Field updates
  const [fieldOptions, setFieldOptions] = useState<Array<{ path: string; valueType: 'string' | 'list' }>>([]);
  const [selectedFieldPath, setSelectedFieldPath] = useState<string>('');
  const [fieldUpdateMode, setFieldUpdateMode] = useState<'append' | 'replace'>('append');
  const [fieldUpdateStatus, setFieldUpdateStatus] = useState<string>('');
  const [fieldUpdateError, setFieldUpdateError] = useState<string>('');
  const [isFieldRecording, setIsFieldRecording] = useState(false);
  const [fieldRecordingTime, setFieldRecordingTime] = useState(0);
  const [expandedPaths, setExpandedPaths] = useState<Record<string, boolean>>({});
  const [fieldUpdateText, setFieldUpdateText] = useState<string>('');
  
  // Collapsible sections
  const [showPatientSection, setShowPatientSection] = useState(true);
  const [showDocSection, setShowDocSection] = useState(false);
  const [showRagSection, setShowRagSection] = useState(false);
  const [showTemplateSection, setShowTemplateSection] = useState(false);
  
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
    if (userId && !practitionerId) {
      setPractitionerId(userId);
    }
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

  // Helpers
  const calculateEditDistance = (original: string, edited: string): number => {
    return Math.abs(original.length - edited.length) + 
      (original !== edited ? Math.min(original.length, edited.length) / 10 : 0);
  };

  const buildSoapNoteText = (note: {
    subjective?: string;
    objective?: string;
    assessment?: string;
    plan?: string;
  }) => [
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

  const toggleExpanded = (path: string) => {
    setExpandedPaths((prev) => ({ ...prev, [path]: !prev[path] }));
  };

  // API calls
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
      if (!selectedTemplateId && list.length > 0) {
        setSelectedTemplateId(list[0].id);
      }
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
      setPatients((prev) => [{ id: created.id, displayName: created.displayName, externalId: created.externalId }, ...prev]);
      setPatientId(created.id);
      setPatientName(created.displayName || newPatientName.trim());
      setNewPatientName('');
      setNewPatientExternalId('');
      setShowNewPatient(false);
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
    } catch (err) {
      console.error('Failed to load patient RAG items:', err);
      setPatientRagItems([]);
      setRagItemsStatus('error');
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

      const transcriptResponse: any = await api.transcribeAudio(
        audioBase64,
        recording.id,
        mappedDialect,
        'ar',
        true,
        false
      );
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
              }
            : r
        )
      );
    } catch (error: any) {
      console.error('Processing error:', error);
      setRecordings((prev) => prev.map((r) => (r.id === recording.id ? { ...r, status: 'error', error: error.message } : r)));
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

  // Render JSON tree for field selection
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
          className={`w-full text-right flex items-center justify-between gap-2 px-3 py-2 rounded-lg border transition-all ${
            selectedFieldPath === childPath && selectable
              ? 'border-blue-400 bg-blue-500/20'
              : 'border-slate-600 bg-slate-700/30 hover:bg-slate-700/50'
          }`}
          style={{ paddingRight: indent + 12 }}
        >
          <span className="text-sm text-slate-200">
            {label}
            {selectable && <span className="text-xs text-blue-300 mr-2">قابل للتحديث</span>}
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

  // Stats
  const statsCards = [
    { label: 'إجمالي', value: recordings.length, color: 'bg-slate-700' },
    { label: 'مكتمل', value: recordings.filter((r) => r.status === 'completed').length, color: 'bg-green-600' },
    { label: 'قيد المعالجة', value: recordings.filter((r) => r.status === 'processing').length, color: 'bg-amber-600' },
    { label: 'خطأ', value: recordings.filter((r) => r.status === 'error').length, color: 'bg-red-600' },
  ];

  // Section toggle component
  const SectionHeader = ({ title, icon: Icon, isOpen, onToggle }: { title: string; icon: any; isOpen: boolean; onToggle: () => void }) => (
    <button
      onClick={onToggle}
      className="w-full flex items-center justify-between p-3 bg-slate-700/50 hover:bg-slate-700 rounded-xl transition-colors"
    >
      <div className="flex items-center gap-2 text-slate-200">
        <Icon className="w-5 h-5" />
        <span className="font-medium">{title}</span>
      </div>
      {isOpen ? <IconChevronDown className="w-5 h-5 text-slate-400" /> : <IconChevronRight className="w-5 h-5 text-slate-400" />}
    </button>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4 md:p-8 font-['Tajawal',sans-serif]" dir="rtl">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-2xl p-6 shadow-xl"
        >
          <div className="flex flex-wrap justify-between items-center gap-4">
            <div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">الملاحظات السريرية</h1>
              <p className="text-slate-400">تحويل التسجيلات الصوتية إلى ملاحظات SOAP</p>
            </div>
            <button
              onClick={() => {
                setShowMetrics(!showMetrics);
                if (!showMetrics && !metrics) loadMetrics();
              }}
              className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors"
            >
              <IconChartBar className="w-5 h-5" />
              {showMetrics ? 'إخفاء المقاييس' : 'المقاييس'}
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
              className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-2xl p-6 shadow-xl"
            >
              <h2 className="text-xl font-bold text-white mb-4">لوحة المقاييس</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-blue-600/20 border border-blue-500/30 rounded-xl p-4">
                  <p className="text-blue-300 text-sm mb-1">إجمالي الملاحظات</p>
                  <p className="text-3xl font-bold text-white">{metrics.overview.totalNotes}</p>
                </div>
                <div className="bg-green-600/20 border border-green-500/30 rounded-xl p-4">
                  <p className="text-green-300 text-sm mb-1">معدل القبول</p>
                  <p className="text-3xl font-bold text-white">{(metrics.overview.acceptanceRate * 100).toFixed(1)}%</p>
                </div>
                <div className="bg-amber-600/20 border border-amber-500/30 rounded-xl p-4">
                  <p className="text-amber-300 text-sm mb-1">متوسط التعديل</p>
                  <p className="text-3xl font-bold text-white">{metrics.overview.avgEditDistance.toFixed(1)}</p>
                </div>
                <div className="bg-purple-600/20 border border-purple-500/30 rounded-xl p-4">
                  <p className="text-purple-300 text-sm mb-1">وقت المراجعة</p>
                  <p className="text-3xl font-bold text-white">{metrics.overview.avgReviewTime.toFixed(0)}ث</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel */}
          <div className="space-y-4">
            {/* Patient & Settings */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-2xl shadow-xl overflow-hidden"
            >
              <SectionHeader title="بيانات المريض" icon={IconUser} isOpen={showPatientSection} onToggle={() => setShowPatientSection(!showPatientSection)} />
              <AnimatePresence>
                {showPatientSection && (
                  <motion.div
                    initial={{ height: 0 }}
                    animate={{ height: 'auto' }}
                    exit={{ height: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="p-4 space-y-3">
                      <div>
                        <label className="block text-sm text-slate-400 mb-1">المريض</label>
                        <select
                          value={patientId}
                          onChange={(e) => {
                            const nextId = e.target.value;
                            setPatientId(nextId);
                            const selected = patients.find((p) => p.id === nextId);
                            if (selected?.displayName) setPatientName(selected.displayName);
                          }}
                          className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          <option value="">-- اختر مريضاً --</option>
                          {patients.map((patient) => (
                            <option key={patient.id} value={patient.id}>
                              {patient.displayName || patient.id}
                            </option>
                          ))}
                        </select>
                        {patientStatus === 'loading' && <p className="text-xs text-slate-400 mt-1">جارٍ التحميل...</p>}
                        {patientError && <p className="text-xs text-red-400 mt-1">{patientError}</p>}
                      </div>

                      <button
                        onClick={() => setShowNewPatient(!showNewPatient)}
                        className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-colors text-sm"
                      >
                        <IconPlus className="w-4 h-4" />
                        إضافة مريض جديد
                      </button>

                      <AnimatePresence>
                        {showNewPatient && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden space-y-2 bg-slate-700/50 rounded-lg p-3"
                          >
                            <input
                              value={newPatientName}
                              onChange={(e) => setNewPatientName(e.target.value)}
                              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                              placeholder="اسم المريض"
                            />
                            <input
                              value={newPatientExternalId}
                              onChange={(e) => setNewPatientExternalId(e.target.value)}
                              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                              placeholder="رقم ملف خارجي (اختياري)"
                              dir="ltr"
                            />
                            <button
                              onClick={createPatient}
                              className="w-full px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium text-sm transition-colors"
                            >
                              إضافة
                            </button>
                          </motion.div>
                        )}
                      </AnimatePresence>

                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className="block text-xs text-slate-400 mb-1">اسم المريض</label>
                          <input
                            value={patientName}
                            onChange={(e) => setPatientName(e.target.value)}
                            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="الاسم"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-slate-400 mb-1">اسم الطبيب</label>
                          <input
                            value={providerName}
                            onChange={(e) => setProviderName(e.target.value)}
                            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="الطبيب"
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className="block text-xs text-slate-400 mb-1">معرف الطبيب</label>
                          <input
                            value={practitionerId}
                            onChange={(e) => setPractitionerId(e.target.value)}
                            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="ID"
                            dir="ltr"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-slate-400 mb-1">تاريخ الزيارة</label>
                          <input
                            type="date"
                            value={dateOfVisit}
                            onChange={(e) => setDateOfVisit(e.target.value)}
                            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            dir="ltr"
                          />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Templates Section */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.05 }}
              className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-2xl shadow-xl overflow-hidden"
            >
              <SectionHeader title="القوالب" icon={IconTemplate} isOpen={showTemplateSection} onToggle={() => setShowTemplateSection(!showTemplateSection)} />
              <AnimatePresence>
                {showTemplateSection && (
                  <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} className="overflow-hidden">
                    <div className="p-4 space-y-3">
                      <div>
                        <label className="block text-sm text-slate-400 mb-1">قالب الملاحظة</label>
                        <select
                          value={selectedTemplateId}
                          onChange={(e) => setSelectedTemplateId(e.target.value)}
                          className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          {templates.map((tpl) => (
                            <option key={tpl.id} value={tpl.id}>{tpl.name}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm text-slate-400 mb-1">رفع قالب JSON</label>
                        <input ref={templateInputRef} type="file" accept="application/json" onChange={(e) => { const file = e.target.files?.[0]; if (file) handleTemplateUpload(file); if (templateInputRef.current) templateInputRef.current.value = ''; }} className="hidden" />
                        <div className="flex gap-2">
                          <input value={templateName} onChange={(e) => setTemplateName(e.target.value)} className="flex-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="اسم القالب" />
                          <button onClick={() => templateInputRef.current?.click()} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium text-sm transition-colors">رفع</button>
                        </div>
                        {templateError && <p className="text-xs text-red-400 mt-1">{templateError}</p>}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Documents Section */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-2xl shadow-xl overflow-hidden"
            >
              <SectionHeader title="ملفات المريض" icon={IconFolder} isOpen={showDocSection} onToggle={() => setShowDocSection(!showDocSection)} />
              <AnimatePresence>
                {showDocSection && (
                  <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} className="overflow-hidden">
                    <div className="p-4 space-y-3">
                      <input ref={docInputRef} type="file" accept=".txt,.json,.csv,.pdf,.docx" onChange={(e) => { const file = e.target.files?.[0]; if (file) handleDocFile(file); if (docInputRef.current) docInputRef.current.value = ''; }} className="hidden" />
                      <div className="flex gap-2">
                        <input value={docTitle} onChange={(e) => setDocTitle(e.target.value)} className="flex-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="عنوان الوثيقة" />
                        <button onClick={() => docInputRef.current?.click()} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium text-sm transition-colors">رفع</button>
                      </div>
                      <textarea value={docText} onChange={(e) => setDocText(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none" placeholder="الصق بيانات المريض هنا" rows={3} />
                      {docFileName && <p className="text-xs text-slate-400">الملف: {docFileName}</p>}
                      <div className="flex items-center justify-between">
                        <label className="flex items-center gap-2 text-xs text-slate-400">
                          <input type="checkbox" checked={summarizeDoc} onChange={(e) => setSummarizeDoc(e.target.checked)} className="rounded" />
                          تلخيص تلقائي
                        </label>
                        <button onClick={uploadPatientDocument} className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium text-sm transition-colors">حفظ</button>
                      </div>
                      {docStatus === 'processing' && <p className="text-xs text-slate-400 flex items-center gap-2"><IconLoader2 className="w-4 h-4 animate-spin" />جارٍ الحفظ...</p>}
                      {docStatus === 'done' && <p className="text-xs text-green-400 flex items-center gap-2"><IconCheck className="w-4 h-4" />تم الحفظ</p>}
                      {docError && <p className="text-xs text-red-400">{docError}</p>}

                      {patientDocs.length > 0 && (
                        <div className="bg-slate-700/50 rounded-lg p-3 max-h-32 overflow-y-auto custom-scrollbar">
                          {patientDocs.slice(0, 5).map((doc) => (
                            <div key={doc.id} className="mb-2 text-xs text-slate-300 border-b border-slate-600 pb-2">
                              <p className="font-medium">{doc.title || 'وثيقة'}</p>
                              {(doc.summaryText || doc.summary_text) && <p className="text-slate-400 mt-1">{doc.summaryText || doc.summary_text}</p>}
                            </div>
                          ))}
                        </div>
                      )}

                      {patientRagItems.length > 0 && (
                        <div className="bg-slate-700/50 rounded-lg p-3 max-h-32 overflow-y-auto custom-scrollbar">
                          <p className="text-xs text-slate-400 mb-2">سجل المريض (RAG)</p>
                          {patientRagItems.slice(0, 5).map((item) => (
                            <div key={item.id} className="mb-2 text-xs text-slate-300 border-b border-slate-600 pb-2">
                              <div className="flex justify-between text-slate-500">
                                <span>{ragTypeLabel(item.itemType || item.item_type)}</span>
                                <span>{formatRagDate(item.createdAt || item.created_at)}</span>
                              </div>
                              <p className="font-medium text-slate-200 mt-1">{item.title || 'عنصر'}</p>
                              <p className="text-slate-400 mt-1">{truncateSnippet(item.contentText || item.content_text || '')}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* RAG Section */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.15 }}
              className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-2xl shadow-xl overflow-hidden"
            >
              <SectionHeader title="معرفة العيادة (RAG)" icon={IconBrain} isOpen={showRagSection} onToggle={() => setShowRagSection(!showRagSection)} />
              <AnimatePresence>
                {showRagSection && (
                  <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} className="overflow-hidden">
                    <div className="p-4 space-y-3">
                      <input value={ragTitle} onChange={(e) => setRagTitle(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="عنوان مختصر (اختياري)" />
                      <textarea value={ragText} onChange={(e) => setRagText(e.target.value)} className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none" placeholder="سياسات أو معلومات مرجعية" rows={3} />
                      <div className="flex items-center gap-3">
                        <button onClick={submitRagNote} className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium text-sm transition-colors">إضافة</button>
                        {ragStatus === 'processing' && <span className="text-xs text-slate-400 flex items-center gap-2"><IconLoader2 className="w-4 h-4 animate-spin" />جارٍ الحفظ...</span>}
                        {ragStatus === 'done' && <span className="text-xs text-green-400 flex items-center gap-2"><IconCheck className="w-4 h-4" />تم</span>}
                      </div>
                      {ragError && <p className="text-xs text-red-400">{ragError}</p>}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Recording Controls */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-2xl p-6 shadow-xl"
            >
              <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <IconMicrophone className="w-5 h-5" />
                التسجيل
              </h3>

              {/* Dialect Selector */}
              <div className="mb-4">
                <label className="block text-sm text-slate-400 mb-2">اللهجة:</label>
                <select
                  value={selectedDialect}
                  onChange={(e) => setSelectedDialect(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="auto">كشف تلقائي</option>
                  <option value="egyptian">مصري</option>
                  <option value="levantine">شامي</option>
                  <option value="gulf">خليجي</option>
                  <option value="msa">فصحى</option>
                </select>
              </div>

              {/* Record Button */}
              <div className="text-center mb-4">
                {!isRecording ? (
                  <motion.button
                    onClick={startRecording}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="w-20 h-20 bg-red-600 hover:bg-red-700 text-white rounded-full flex items-center justify-center mx-auto shadow-lg shadow-red-600/30 transition-colors"
                  >
                    <IconMicrophone className="w-10 h-10" />
                  </motion.button>
                ) : (
                  <>
                    <motion.button
                      onClick={stopRecording}
                      animate={{ scale: [1, 1.05, 1] }}
                      transition={{ repeat: Infinity, duration: 1.5 }}
                      className="w-20 h-20 bg-slate-600 hover:bg-slate-700 text-white rounded-full flex items-center justify-center mx-auto shadow-lg transition-colors"
                    >
                      <IconPlayerStop className="w-10 h-10" />
                    </motion.button>
                    <p className="text-2xl font-mono text-red-400 font-bold mt-3">{formatTime(recordingTime)}</p>
                  </>
                )}
                <p className="text-sm text-slate-400 mt-2">{isRecording ? 'جاري التسجيل...' : 'انقر للبدء'}</p>
              </div>

              {/* File Upload */}
              <input ref={fileInputRef} type="file" accept="audio/*" multiple onChange={handleFileUpload} className="hidden" />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors"
              >
                <IconUpload className="w-5 h-5" />
                رفع ملف صوتي
              </button>
              <p className="text-xs text-slate-500 mt-2 text-center">MP3, WAV, M4A, WebM</p>
            </motion.div>

            {/* Stats */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.25 }}
              className="grid grid-cols-4 gap-2"
            >
              {statsCards.map((stat) => (
                <div key={stat.label} className={`${stat.color} rounded-xl p-3 text-center`}>
                  <p className="text-2xl font-bold text-white">{stat.value}</p>
                  <p className="text-xs text-white/70">{stat.label}</p>
                </div>
              ))}
            </motion.div>
          </div>

          {/* Right Panel - Recordings & Details */}
          <div className="lg:col-span-2 space-y-4">
            {/* Recordings List */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-2xl p-6 shadow-xl"
            >
              <h3 className="text-lg font-bold text-white mb-4">التسجيلات</h3>
              <div className="space-y-2 max-h-64 overflow-y-auto custom-scrollbar">
                {recordings.length === 0 ? (
                  <div className="text-center py-8">
                    <IconFileText className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                    <p className="text-slate-500">لا توجد تسجيلات بعد</p>
                  </div>
                ) : (
                  recordings.map((recording) => (
                    <motion.div
                      key={recording.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      onClick={() => {
                        setSelectedRecording(recording);
                        setEditedSoapNote('');
                      }}
                      className={`p-4 rounded-xl border cursor-pointer transition-all ${
                        selectedRecording?.id === recording.id
                          ? 'border-blue-500 bg-blue-500/10'
                          : 'border-slate-700 bg-slate-700/30 hover:border-slate-600'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-white text-sm">{recording.file.name}</p>
                          <p className="text-xs text-slate-500">{recording.timestamp.toLocaleString('ar-EG')}</p>
                        </div>
                        <span
                          className={`px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1 ${
                            recording.status === 'completed' ? 'bg-green-600/20 text-green-400' :
                            recording.status === 'processing' ? 'bg-amber-600/20 text-amber-400' :
                            recording.status === 'error' ? 'bg-red-600/20 text-red-400' :
                            'bg-slate-600/20 text-slate-400'
                          }`}
                        >
                          {recording.status === 'completed' && <><IconCheck className="w-3 h-3" /> مكتمل</>}
                          {recording.status === 'processing' && <><IconLoader2 className="w-3 h-3 animate-spin" /> جاري...</>}
                          {recording.status === 'error' && <><IconX className="w-3 h-3" /> خطأ</>}
                          {recording.status === 'pending' && <><IconClock className="w-3 h-3" /> انتظار</>}
                        </span>
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
                  className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-2xl p-6 shadow-xl"
                >
                  {selectedRecording.status === 'processing' && (
                    <div className="text-center py-12">
                      <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }} className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
                      <p className="text-slate-400">جاري معالجة التسجيل...</p>
                    </div>
                  )}

                  {selectedRecording.status === 'error' && (
                    <div className="bg-red-600/20 border border-red-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2 text-red-400">
                        <IconX className="w-5 h-5" />
                        <p>حدث خطأ: {selectedRecording.error}</p>
                      </div>
                    </div>
                  )}

                  {selectedRecording.status === 'completed' && (
                    <div className="space-y-6">
                      {/* Transcript */}
                      <div>
                        <h4 className="font-medium text-slate-300 mb-2 flex items-center gap-2">
                          <IconFileText className="w-4 h-4" />
                          النص المكتوب
                        </h4>
                        <div className="bg-slate-700/50 rounded-xl p-4 max-h-32 overflow-y-auto custom-scrollbar">
                          <p className="text-white text-sm leading-relaxed">{selectedRecording.transcript}</p>
                        </div>
                      </div>

                      {/* SOAP Note - Color Coded Cards */}
                      {selectedRecording.soapJson && (
                        <div>
                          <h4 className="font-medium text-slate-300 mb-3">ملاحظة SOAP</h4>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {/* Subjective - Blue */}
                            <div className="bg-blue-600/20 border border-blue-500/30 rounded-xl p-4">
                              <div className="flex items-center gap-2 mb-2">
                                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                                <h5 className="font-bold text-blue-300">Subjective</h5>
                              </div>
                              <p className="text-slate-200 text-sm">{selectedRecording.soapJson.subjective || 'لا توجد بيانات'}</p>
                            </div>
                            {/* Objective - Green */}
                            <div className="bg-green-600/20 border border-green-500/30 rounded-xl p-4">
                              <div className="flex items-center gap-2 mb-2">
                                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                                <h5 className="font-bold text-green-300">Objective</h5>
                              </div>
                              <p className="text-slate-200 text-sm">{selectedRecording.soapJson.objective || 'لا توجد بيانات'}</p>
                            </div>
                            {/* Assessment - Amber */}
                            <div className="bg-amber-600/20 border border-amber-500/30 rounded-xl p-4">
                              <div className="flex items-center gap-2 mb-2">
                                <div className="w-3 h-3 bg-amber-500 rounded-full"></div>
                                <h5 className="font-bold text-amber-300">Assessment</h5>
                              </div>
                              <p className="text-slate-200 text-sm">{selectedRecording.soapJson.assessment || 'لا توجد بيانات'}</p>
                            </div>
                            {/* Plan - Purple */}
                            <div className="bg-purple-600/20 border border-purple-500/30 rounded-xl p-4">
                              <div className="flex items-center gap-2 mb-2">
                                <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                                <h5 className="font-bold text-purple-300">Plan</h5>
                              </div>
                              <p className="text-slate-200 text-sm">{selectedRecording.soapJson.plan || 'لا توجد بيانات'}</p>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Editable SOAP Text */}
                      <div>
                        <h4 className="font-medium text-slate-300 mb-2">تحرير الملاحظة</h4>
                        <textarea
                          value={editedSoapNote || selectedRecording.soapNote}
                          onChange={(e) => {
                            setEditedSoapNote(e.target.value);
                            if (!selectedRecording.reviewStartTime) {
                              setRecordings((prev) => prev.map((r) => (r.id === selectedRecording.id ? { ...r, reviewStartTime: Date.now() } : r)));
                            }
                          }}
                          className="w-full bg-slate-700 border border-slate-600 rounded-xl p-4 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none custom-scrollbar"
                          rows={8}
                        />
                      </div>

                      {/* Field Update Section */}
                      {selectedRecording.noteId && selectedRecording.soapJson && (
                        <div className="bg-slate-700/30 rounded-xl p-4">
                          <h4 className="font-medium text-slate-300 mb-3 flex items-center gap-2">
                            <IconMicrophone className="w-4 h-4" />
                            تحديث حقل بالصوت
                          </h4>
                          <div className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
                            <div>
                              <label className="block text-xs text-slate-400 mb-2">شجرة الحقول</label>
                              <div className="bg-slate-800/50 rounded-lg p-3 max-h-48 overflow-y-auto custom-scrollbar space-y-1">
                                {renderJsonTree(selectedRecording.soapJson)}
                              </div>
                            </div>
                            <div className="space-y-3">
                              <div className="text-xs text-slate-400">
                                الحقل: <span className="text-white font-medium">{selectedFieldPath || 'غير محدد'}</span>
                              </div>
                              <select
                                value={fieldUpdateMode}
                                onChange={(e) => setFieldUpdateMode(e.target.value as 'append' | 'replace')}
                                className="w-full bg-slate-700 border border-slate-600 rounded-lg p-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                              >
                                <option value="append">إضافة</option>
                                <option value="replace">استبدال</option>
                              </select>
                              <textarea
                                value={fieldUpdateText}
                                onChange={(e) => setFieldUpdateText(e.target.value)}
                                className="w-full bg-slate-700 border border-slate-600 rounded-lg p-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                                rows={3}
                                placeholder="أو أضف النص هنا"
                              />
                              <button
                                onClick={submitFieldTextUpdate}
                                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg text-sm transition-colors"
                              >
                                تحديث بالنص
                              </button>
                            </div>
                          </div>
                          <div className="flex items-center gap-3 mt-4">
                            {!isFieldRecording ? (
                              <motion.button
                                onClick={startFieldRecording}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                className="bg-purple-600 hover:bg-purple-700 text-white font-medium py-2 px-4 rounded-lg text-sm transition-colors flex items-center gap-2"
                              >
                                <IconMicrophone className="w-4 h-4" />
                                تسجيل المعلومة
                              </motion.button>
                            ) : (
                              <motion.button
                                onClick={stopFieldRecording}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                className="bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-lg text-sm transition-colors flex items-center gap-2"
                              >
                                <IconPlayerStop className="w-4 h-4" />
                                إيقاف ({formatTime(fieldRecordingTime)})
                              </motion.button>
                            )}
                            {fieldUpdateStatus === 'processing' && <span className="text-xs text-slate-400 flex items-center gap-2"><IconLoader2 className="w-4 h-4 animate-spin" />جارِ التحديث...</span>}
                            {fieldUpdateStatus === 'done' && <span className="text-xs text-green-400 flex items-center gap-2"><IconCheck className="w-4 h-4" />تم</span>}
                          </div>
                          {fieldUpdateError && <p className="text-xs text-red-400 mt-2">{fieldUpdateError}</p>}
                        </div>
                      )}

                      {/* JSON Output */}
                      {selectedRecording.soapJson && (
                        <details className="bg-slate-700/30 rounded-xl">
                          <summary className="cursor-pointer p-4 text-slate-400 text-sm">عرض JSON</summary>
                          <pre className="p-4 pt-0 text-xs text-slate-300 overflow-auto max-h-48 custom-scrollbar">
                            {JSON.stringify(selectedRecording.soapJson, null, 2)}
                          </pre>
                        </details>
                      )}

                      {/* Actions */}
                      <div className="flex flex-wrap gap-3 pt-2">
                        <motion.button
                          onClick={async () => {
                            const finalText = editedSoapNote || selectedRecording.soapNote || '';
                            await recordReviewMetrics(selectedRecording, true, finalText);
                            await saveToEHR();
                          }}
                          disabled={isSaving}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white font-bold py-3 px-6 rounded-xl transition-colors flex items-center justify-center gap-2"
                        >
                          {isSaving ? (
                            <><IconLoader2 className="w-5 h-5 animate-spin" /> جاري الحفظ...</>
                          ) : (
                            <><IconCheck className="w-5 h-5" /> قبول وحفظ</>
                          )}
                        </motion.button>
                        <motion.button
                          onClick={async () => {
                            const finalText = editedSoapNote || selectedRecording.soapNote || '';
                            await recordReviewMetrics(selectedRecording, false, finalText);
                            if (selectedRecording.noteId) await api.rejectSOAPNote(selectedRecording.noteId);
                            alert('تم رفض الملاحظة');
                          }}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className="flex-1 bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-xl transition-colors flex items-center justify-center gap-2"
                        >
                          <IconX className="w-5 h-5" /> رفض
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
                          className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-xl transition-colors"
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
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.05); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(96, 165, 250, 0.4); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(96, 165, 250, 0.6); }
      `}</style>
    </div>
  );
}
