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
  IconLoader2
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

export default function ClinicalNotes() {
  const { userId, token } = useAuthStore();
  const [recordings, setRecordings] = useState<AudioRecording[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [selectedRecording, setSelectedRecording] = useState<AudioRecording | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);
  const [editedSoapNote, setEditedSoapNote] = useState<string>('');
  const [selectedDialect, setSelectedDialect] = useState<string>('auto');
  const [metrics, setMetrics] = useState<MetricsDashboard | null>(null);
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
  const [patientDocs, setPatientDocs] = useState<Array<any>>([]);
  const [docTitle, setDocTitle] = useState<string>('');
  const [docText, setDocText] = useState<string>('');
  const [docFileName, setDocFileName] = useState<string>('');
  const [docFileType, setDocFileType] = useState<string>('');
  const [docFileBase64, setDocFileBase64] = useState<string>('');
  const [docStatus, setDocStatus] = useState<'idle' | 'processing' | 'done' | 'error'>('idle');
  const [docError, setDocError] = useState<string>('');
  const [patientContext, setPatientContext] = useState<any>(null);
  const [patientRagItems, setPatientRagItems] = useState<Array<any>>([]);
  const [ragItemsStatus, setRagItemsStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [ragItemsError, setRagItemsError] = useState<string>('');
  const [summarizeDoc, setSummarizeDoc] = useState(true);
  const [templates, setTemplates] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('');
  const [templateName, setTemplateName] = useState<string>('');
  const [templateError, setTemplateError] = useState<string>('');
  const [fieldOptions, setFieldOptions] = useState<Array<{ path: string; valueType: 'string' | 'list' }>>([]);
  const [selectedFieldPath, setSelectedFieldPath] = useState<string>('');
  const [fieldUpdateMode, setFieldUpdateMode] = useState<'append' | 'replace'>('append');
  const [fieldUpdateStatus, setFieldUpdateStatus] = useState<string>('');
  const [fieldUpdateError, setFieldUpdateError] = useState<string>('');
  const [isFieldRecording, setIsFieldRecording] = useState(false);
  const [fieldRecordingTime, setFieldRecordingTime] = useState(0);
  const [expandedPaths, setExpandedPaths] = useState<Record<string, boolean>>({});
  const [fieldUpdateText, setFieldUpdateText] = useState<string>('');
  const [ragTitle, setRagTitle] = useState<string>('');
  const [ragText, setRagText] = useState<string>('');
  const [ragStatus, setRagStatus] = useState<string>('');
  const [ragError, setRagError] = useState<string>('');
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const templateInputRef = useRef<HTMLInputElement>(null);
  const docInputRef = useRef<HTMLInputElement>(null);
  const fieldRecorderRef = useRef<MediaRecorder | null>(null);
  const fieldAudioChunksRef = useRef<Blob[]>([]);
  const fieldTimerRef = useRef<number | null>(null);
  const recordingDurationRef = useRef(0);

  useEffect(() => {
    if (userId && !practitionerId) {
      setPractitionerId(userId);
    }
  }, [userId, practitionerId]);

  useEffect(() => {
    loadTemplates();
  }, []);

  useEffect(() => {
    if (token) {
      loadTemplates();
    }
  }, [token]);

  useEffect(() => {
    if (token) {
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

  // Calculate edit distance (Levenshtein distance approximation)
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

  // Returns the best supported audio MIME type for the current browser
  const getSupportedMimeType = (): string => {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
    ];
    for (const type of candidates) {
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }
    return '';
  };

  const showToast = (type: 'success' | 'error', message: string) => {
    setToast({ type, message });
    setTimeout(() => setToast(null), 4000);
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

  // Load metrics dashboard
  const loadMetrics = async () => {
    try {
      setMetricsLoading(true);
      const data = await api.getClinicalMetricsDashboard();
      setMetrics(data);
    } catch (err) {
      console.error('Failed to load metrics:', err);
    } finally {
      setMetricsLoading(false);
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
        if (list[0].displayName) {
          setPatientName(list[0].displayName);
        }
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
      if (!docTitle) {
        setDocTitle(file.name.replace(/\.[^.]+$/, ''));
      }
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
      case 'soap_note':
        return 'SOAP';
      case 'transcript':
        return 'تفريغ';
      case 'doc_summary':
        return 'ملخص وثيقة';
      case 'document':
        return 'وثيقة';
      default:
        return type || 'عنصر';
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
      if (prefix) {
        results.push({ path: prefix, valueType: 'string' });
      }
      return results;
    }
    if (Array.isArray(value)) {
      const stringItems = value.filter((item) => typeof item === 'string');
      if (stringItems.length === value.length) {
        if (prefix) {
          results.push({ path: prefix, valueType: 'list' });
        }
        return results;
      }
      value.forEach((item, index) => {
        const nextPrefix = `${prefix}[${index}]`;
        results.push(...collectFieldOptions(item, nextPrefix));
      });
      return results;
    }
    if (value && typeof value === 'object') {
      Object.entries(value).forEach(([key, child]) => {
        const nextPrefix = prefix ? `${prefix}.${key}` : key;
        results.push(...collectFieldOptions(child, nextPrefix));
      });
    }
    return results;
  };

  const toggleExpanded = (path: string) => {
    setExpandedPaths((prev) => ({
      ...prev,
      [path]: !prev[path],
    }));
  };

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
      const preview = typeof child === 'string'
        ? child.slice(0, 40)
        : isStringList
          ? `${child.length} عناصر`
          : '';

      nodes.push(
        <button
          key={childPath}
          type="button"
          onClick={() => {
            if (isLeaf && selectable) {
              setSelectedFieldPath(childPath);
            } else {
              toggleExpanded(childPath);
            }
          }}
          className={`w-full text-right flex items-center justify-between gap-2 px-3 py-2 rounded-lg border ${
            selectedFieldPath === childPath && selectable
              ? 'border-purple-400 bg-purple-500/20'
              : 'border-white/10 bg-white/5 hover:bg-white/10'
          }`}
          style={{ paddingRight: indent + 12 }}
        >
          <span className="text-sm text-purple-100">
            {label}
            {selectable && (
              <span className="text-xs text-purple-300 ml-2">قابل للتحديث</span>
            )}
          </span>
          <span className="text-xs text-purple-300">
            {isLeaf ? preview || 'نص' : isExpanded ? '−' : '+'}
          </span>
        </button>
      );

      if (!isLeaf && isExpanded) {
        if (isArray) {
          child.forEach((item: any, index: number) => {
            renderNode(`[${index}]`, item, `${childPath}[${index}]`);
          });
        } else if (child && typeof child === 'object') {
          Object.entries(child).forEach(([key, val]) => {
            renderNode(key, val, childPath ? `${childPath}.${key}` : key);
          });
        }
      }
    };

    if (value && typeof value === 'object') {
      Object.entries(value).forEach(([key, val]) => {
        const childPath = path ? `${path}.${key}` : key;
        renderNode(key, val, childPath);
      });
    }

    return nodes;
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

  // Start live recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const supportedMime = getSupportedMimeType();
      const mediaRecorder = new MediaRecorder(stream, supportedMime ? { mimeType: supportedMime } : {});

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
          duration: recordingDurationRef.current,
          timestamp: new Date(),
          status: 'pending',
        };

        setRecordings((prev) => [newRecording, ...prev]);
        setRecordingTime(0);
        recordingDurationRef.current = 0;

        // Auto-process the recording
        await processRecording(newRecording);
      };

      mediaRecorder.start();
      setIsRecording(true);

      // Start timer
      timerRef.current = window.setInterval(() => {
        setRecordingTime((prev) => {
          recordingDurationRef.current = prev + 1;
          return prev + 1;
        });
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
      const supportedMime = getSupportedMimeType();
      const mediaRecorder = new MediaRecorder(stream, supportedMime ? { mimeType: supportedMime } : {});

      fieldRecorderRef.current = mediaRecorder;
      fieldAudioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          fieldAudioChunksRef.current.push(event.data);
        }
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

      fieldTimerRef.current = window.setInterval(() => {
        setFieldRecordingTime((prev) => prev + 1);
      }, 1000);
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
      setRecordings((prev) =>
        prev.map((r) => (r.id === selectedRecording.id ? { ...r, ...updatedNote } : r))
      );
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
      setRecordings((prev) =>
        prev.map((r) => (r.id === selectedRecording.id ? { ...r, ...updatedNote } : r))
      );
      setSelectedRecording(updatedNote);
      setFieldUpdateStatus('done');
      setFieldUpdateText('');
    } catch (error: any) {
      console.error('Field update error:', error);
      setFieldUpdateStatus('error');
      setFieldUpdateError(error.message || 'فشل تحديث الحقل');
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
      if (!patientId || !practitionerId) {
        throw new Error('الرجاء إدخال معرف المريض والطبيب');
      }
      // Update status to processing
      setRecordings((prev) =>
        prev.map((r) =>
          r.id === recording.id ? { ...r, status: 'processing' } : r
        )
      );

      // Step 1: Transcribe audio
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

      // Step 2: Generate SOAP note
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

      // Update recording with results
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
    if (!selectedRecording || !selectedRecording.noteId) return;

    setIsSaving(true);
    try {
      const finalText = editedSoapNote || selectedRecording.soapNote || '';
      if (finalText && finalText !== selectedRecording.soapNote) {
        const updated = await api.updateSOAPNoteSections(selectedRecording.noteId, {
          soapText: finalText,
        });
        const updatedNote = {
          ...selectedRecording,
          soapNote: buildSoapNoteText(updated),
          soapJson: updated.soapJson ?? updated.soap_json ?? selectedRecording.soapJson,
        };
        setRecordings((prev) =>
          prev.map((r) => (r.id === selectedRecording.id ? { ...r, ...updatedNote } : r))
        );
        setSelectedRecording(updatedNote);
        setEditedSoapNote('');
      }
      await api.approveSOAPNote(selectedRecording.noteId);
      showToast('success', 'تم الحفظ بنجاح!');
    } catch (error: any) {
      showToast('error', `خطأ في الحفظ: ${error.message}`);
    } finally {
      setIsSaving(false);
    }
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
          {showMetrics && metricsLoading && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="backdrop-blur-md bg-white/10 border border-white/20 rounded-3xl p-8 mb-8 shadow-2xl flex items-center justify-center gap-4"
            >
              <IconLoader2 className="w-6 h-6 text-purple-300 animate-spin" />
              <p className="text-purple-200">جارٍ تحميل المقاييس...</p>
            </motion.div>
          )}
          {showMetrics && !metricsLoading && metrics && (
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
            {/* Note Settings */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.05 }}
              className="backdrop-blur-md bg-white/10 border border-white/20 rounded-3xl p-8 shadow-2xl"
            >
              <h2 className="text-2xl font-bold text-white mb-6">إعدادات الملاحظة</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-purple-200 mb-2">اختيار المريض</label>
                  <select
                    value={patientId}
                    onChange={(e) => {
                      const nextId = e.target.value;
                      setPatientId(nextId);
                      const selected = patients.find((p) => p.id === nextId);
                      if (selected?.displayName) {
                        setPatientName(selected.displayName);
                      }
                    }}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    dir="rtl"
                  >
                    <option value="">-- اختر مريضاً --</option>
                    {patients.map((patient) => (
                      <option key={patient.id} value={patient.id}>
                        {patient.displayName || patient.id}
                      </option>
                    ))}
                  </select>
                  {patientStatus === 'loading' && (
                    <p className="text-xs text-purple-200 mt-2">جارٍ تحميل المرضى...</p>
                  )}
                  {patientError && (
                    <p className="text-xs text-red-300 mt-2">{patientError}</p>
                  )}
                  {patientId && (
                    <p className="text-xs text-purple-200 mt-2" dir="ltr">
                      ID: {patientId}
                    </p>
                  )}
                </div>
                <div className="bg-white/5 border border-white/10 rounded-2xl p-4">
                  <p className="text-sm text-purple-200 mb-3">إضافة مريض جديد</p>
                  <input
                    value={newPatientName}
                    onChange={(e) => setNewPatientName(e.target.value)}
                    className="w-full px-4 py-3 mb-2 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="اسم المريض"
                  />
                  <input
                    value={newPatientExternalId}
                    onChange={(e) => setNewPatientExternalId(e.target.value)}
                    className="w-full px-4 py-3 mb-3 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="رقم ملف خارجي (اختياري)"
                    dir="ltr"
                  />
                  <button
                    onClick={createPatient}
                    className="w-full px-4 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-xl font-bold shadow-lg hover:shadow-emerald-500/50 transition-all"
                  >
                    إضافة المريض
                  </button>
                </div>
                <div>
                  <label className="block text-sm font-medium text-purple-200 mb-2">معرف الطبيب</label>
                  <input
                    value={practitionerId}
                    onChange={(e) => setPractitionerId(e.target.value)}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="clinician-456"
                    dir="ltr"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-purple-200 mb-2">اسم المريض</label>
                  <input
                    value={patientName}
                    onChange={(e) => setPatientName(e.target.value)}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="اسم المريض"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-purple-200 mb-2">اسم الطبيب</label>
                  <input
                    value={providerName}
                    onChange={(e) => setProviderName(e.target.value)}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="اسم الطبيب"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-purple-200 mb-2">تاريخ الزيارة</label>
                  <input
                    type="date"
                    value={dateOfVisit}
                    onChange={(e) => setDateOfVisit(e.target.value)}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    dir="ltr"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-purple-200 mb-2">قالب الملاحظة</label>
                  <select
                    value={selectedTemplateId}
                    onChange={(e) => setSelectedTemplateId(e.target.value)}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    dir="rtl"
                  >
                    {templates.map((tpl) => (
                      <option key={tpl.id} value={tpl.id}>
                        {tpl.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-purple-200 mb-2">رفع قالب JSON</label>
                  <input
                    ref={templateInputRef}
                    type="file"
                    accept="application/json"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleTemplateUpload(file);
                      if (templateInputRef.current) templateInputRef.current.value = '';
                    }}
                    className="hidden"
                  />
                  <div className="flex gap-2">
                    <input
                      value={templateName}
                      onChange={(e) => setTemplateName(e.target.value)}
                      className="flex-1 px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                      placeholder="اسم القالب (اختياري)"
                    />
                    <button
                      onClick={() => templateInputRef.current?.click()}
                      className="px-4 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl font-bold shadow-lg hover:shadow-blue-500/50 transition-all"
                    >
                      رفع
                    </button>
                  </div>
                  {templateError && (
                    <p className="text-sm text-red-300 mt-2">{templateError}</p>
                  )}
                </div>
                <div className="border-t border-white/10 pt-4">
                  <label className="block text-sm font-medium text-purple-200 mb-2">معرفة العيادة (RAG)</label>
                  <input
                    value={ragTitle}
                    onChange={(e) => setRagTitle(e.target.value)}
                    className="w-full px-4 py-3 mb-2 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="عنوان مختصر (اختياري)"
                  />
                  <textarea
                    value={ragText}
                    onChange={(e) => setRagText(e.target.value)}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                    placeholder="سياسات، تعليمات، جداول أو معلومات تريد أن يستخدمها النظام"
                    rows={4}
                  />
                  <div className="flex items-center gap-3 mt-3">
                    <button
                      onClick={submitRagNote}
                      className="px-4 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-xl font-bold shadow-lg hover:shadow-emerald-500/50 transition-all"
                    >
                      إضافة للمعرفة
                    </button>
                    {ragStatus === 'processing' && (
                      <span className="text-xs text-purple-200 flex items-center gap-2">
                        <IconLoader2 className="w-4 h-4 animate-spin" />
                        جارٍ الحفظ...
                      </span>
                    )}
                    {ragStatus === 'done' && (
                      <span className="text-xs text-green-300 flex items-center gap-2">
                        <IconCheck className="w-4 h-4" />
                        تم الحفظ
                      </span>
                    )}
                  </div>
                  {ragError && (
                    <p className="text-sm text-red-300 mt-2">{ragError}</p>
                  )}
                </div>
                <div className="border-t border-white/10 pt-4">
                  <label className="block text-sm font-medium text-purple-200 mb-2">ملفات المريض</label>
                  <input
                    ref={docInputRef}
                    type="file"
                    accept=".txt,.json,.csv,.pdf,.docx"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleDocFile(file);
                      if (docInputRef.current) docInputRef.current.value = '';
                    }}
                    className="hidden"
                  />
                  <div className="flex gap-2 mb-2">
                    <input
                      value={docTitle}
                      onChange={(e) => setDocTitle(e.target.value)}
                      className="flex-1 px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                      placeholder="عنوان الوثيقة (اختياري)"
                    />
                    <button
                      onClick={() => docInputRef.current?.click()}
                      className="px-4 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl font-bold shadow-lg hover:shadow-blue-500/50 transition-all"
                    >
                      رفع ملف
                    </button>
                  </div>
                  <textarea
                    value={docText}
                    onChange={(e) => setDocText(e.target.value)}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                    placeholder="الصق بيانات أو تاريخ المريض هنا"
                    rows={4}
                  />
                  {docFileName && (
                    <p className="text-xs text-purple-200 mt-2">الملف المرفوع: {docFileName}</p>
                  )}
                  <div className="flex items-center justify-between mt-3">
                    <label className="flex items-center gap-2 text-xs text-purple-200">
                      <input
                        type="checkbox"
                        checked={summarizeDoc}
                        onChange={(e) => setSummarizeDoc(e.target.checked)}
                      />
                      تلخيص الوثيقة تلقائياً
                    </label>
                    <button
                      onClick={uploadPatientDocument}
                      className="px-4 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-xl font-bold shadow-lg hover:shadow-emerald-500/50 transition-all"
                    >
                      حفظ للملف
                    </button>
                  </div>
                  {docStatus === 'processing' && (
                    <p className="text-xs text-purple-200 mt-2 flex items-center gap-2">
                      <IconLoader2 className="w-4 h-4 animate-spin" />
                      جارٍ الحفظ...
                    </p>
                  )}
                  {docStatus === 'done' && (
                    <p className="text-xs text-green-300 mt-2 flex items-center gap-2">
                      <IconCheck className="w-4 h-4" />
                      تم حفظ الوثيقة
                    </p>
                  )}
                  {docError && (
                    <p className="text-xs text-red-300 mt-2">{docError}</p>
                  )}
                  {patientDocs.length > 0 && (
                    <div className="mt-4 bg-black/20 border border-white/10 rounded-xl p-3 max-h-48 overflow-y-auto custom-scrollbar">
                      {patientDocs.slice(0, 5).map((doc) => (
                        <div key={doc.id} className="mb-3 text-xs text-purple-100">
                          <p className="font-semibold">{doc.title || 'وثيقة'}</p>
                          {doc.summaryText || doc.summary_text ? (
                            <p className="text-purple-200 mt-1">{doc.summaryText || doc.summary_text}</p>
                          ) : (
                            <p className="text-purple-300 mt-1">لا يوجد ملخص</p>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                  {patientRagItems.length > 0 && (
                    <div className="mt-4 bg-black/20 border border-white/10 rounded-xl p-3 max-h-56 overflow-y-auto custom-scrollbar">
                      <p className="text-xs text-purple-200 mb-2">سجل المريض (RAG)</p>
                      {patientRagItems.slice(0, 8).map((item) => (
                        <div key={item.id} className="mb-3 text-xs text-purple-100">
                          <div className="flex items-center justify-between text-[10px] text-purple-300">
                            <span>{ragTypeLabel(item.itemType || item.item_type)}</span>
                            <span>{formatRagDate(item.createdAt || item.created_at)}</span>
                          </div>
                          <p className="font-semibold text-white mt-1">{item.title || 'عنصر'}</p>
                          <p className="text-purple-200 mt-1">
                            {truncateSnippet(item.contentText || item.content_text || '')}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                  {ragItemsStatus === 'loading' && (
                    <p className="text-xs text-purple-200 mt-2 flex items-center gap-2">
                      <IconLoader2 className="w-4 h-4 animate-spin" />
                      جارٍ تحميل سجل المريض...
                    </p>
                  )}
                  {ragItemsError && (
                    <p className="text-xs text-red-300 mt-2">{ragItemsError}</p>
                  )}
                  {patientContext && (
                    <div className="mt-3 text-xs text-purple-200">
                      <p className="mb-2">ملخص سياق المريض المتاح للنظام:</p>
                      <pre className="bg-black/30 border border-white/10 rounded-xl p-3 text-[10px] text-purple-100 overflow-auto max-h-40">
                        {JSON.stringify(patientContext, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>

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

                      {selectedRecording.soapJson && (
                        <div>
                          <h3 className="font-semibold text-purple-200 mb-3 flex items-center gap-2">
                            <IconFileText className="w-5 h-5" />
                            JSON الناتج:
                          </h3>
                          <pre className="bg-black/30 border border-white/10 rounded-2xl p-4 text-xs text-purple-100 overflow-auto max-h-64">
                            {JSON.stringify(selectedRecording.soapJson, null, 2)}
                          </pre>
                        </div>
                      )}

                      {selectedRecording.noteId && (
                        <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
                          <h3 className="font-semibold text-purple-200 mb-4 flex items-center gap-2">
                            <IconMicrophone className="w-5 h-5" />
                            استكمال حقل ناقص بالصوت
                          </h3>
                          <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
                            <div>
                              <label className="block text-xs text-purple-200 mb-2">شجرة الحقول</label>
                              <div className="bg-black/20 border border-white/10 rounded-xl p-3 max-h-64 overflow-y-auto custom-scrollbar space-y-2">
                                {selectedRecording.soapJson ? (
                                  renderJsonTree(selectedRecording.soapJson)
                                ) : (
                                  <p className="text-xs text-purple-200">لا يوجد JSON متاح</p>
                                )}
                              </div>
                            </div>
                            <div className="space-y-3">
                              <div className="text-xs text-purple-200">
                                الحقل المختار: <span className="text-white">{selectedFieldPath || 'غير محدد'}</span>
                              </div>
                              <div>
                                <label className="block text-xs text-purple-200 mb-2">طريقة التحديث</label>
                                <select
                                  value={fieldUpdateMode}
                                  onChange={(e) => setFieldUpdateMode(e.target.value as 'append' | 'replace')}
                                  className="w-full bg-white/10 border border-white/20 rounded-xl p-3 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                                >
                                  <option value="append" className="text-black">إضافة</option>
                                  <option value="replace" className="text-black">استبدال</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-xs text-purple-200 mb-2">تحديث بالنص</label>
                                <textarea
                                  value={fieldUpdateText}
                                  onChange={(e) => setFieldUpdateText(e.target.value)}
                                  className="w-full bg-white/10 border border-white/20 rounded-xl p-3 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                                  rows={4}
                                  placeholder="أضف المعلومة المطلوبة هنا"
                                />
                              </div>
                              <motion.button
                                onClick={submitFieldTextUpdate}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                className="w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white font-bold py-3 px-5 rounded-xl transition-all shadow-lg hover:shadow-blue-500/50"
                              >
                                تحديث بالنص
                              </motion.button>
                            </div>
                          </div>

                          <div className="flex items-center gap-4 mt-4">
                            {!isFieldRecording ? (
                              <motion.button
                                onClick={startFieldRecording}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                className="bg-gradient-to-r from-purple-500 to-purple-600 text-white font-bold py-3 px-5 rounded-xl transition-all shadow-lg hover:shadow-purple-500/50 flex items-center gap-2"
                              >
                                <IconMicrophone className="w-5 h-5" />
                                تسجيل المعلومة
                              </motion.button>
                            ) : (
                              <motion.button
                                onClick={stopFieldRecording}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                className="bg-gradient-to-r from-red-500 to-red-600 text-white font-bold py-3 px-5 rounded-xl transition-all shadow-lg hover:shadow-red-500/50 flex items-center gap-2"
                              >
                                <IconPlayerStop className="w-5 h-5" />
                                إيقاف ({formatTime(fieldRecordingTime)})
                              </motion.button>
                            )}
                            {fieldUpdateStatus === 'processing' && (
                              <span className="text-xs text-purple-200 flex items-center gap-2">
                                <IconLoader2 className="w-4 h-4 animate-spin" />
                                جارِ التحديث...
                              </span>
                            )}
                            {fieldUpdateStatus === 'done' && (
                              <span className="text-xs text-green-300 flex items-center gap-2">
                                <IconCheck className="w-4 h-4" />
                                تم تحديث الحقل
                              </span>
                            )}
                          </div>

                          {fieldUpdateError && (
                            <p className="text-xs text-red-300 mt-3">{fieldUpdateError}</p>
                          )}
                        </div>
                      )}

                      {/* Actions */}
                      <div className="flex gap-4 pt-4">
                        <motion.button
                          onClick={async () => {
                            const finalText = editedSoapNote || selectedRecording.soapNote || '';
                            await recordReviewMetrics(selectedRecording, true, finalText);
                            await saveToEHR();
                            setRecordings((prev) =>
                              prev.map((r) =>
                                r.id === selectedRecording.id ? { ...r, status: 'completed' } : r
                              )
                            );
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
                            if (selectedRecording.noteId) {
                              await api.rejectSOAPNote(selectedRecording.noteId);
                            }
                            showToast('success', 'تم رفض الملاحظة');
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

      {/* Toast Notifications */}
      <AnimatePresence>
        {toast && (
          <motion.div
            initial={{ opacity: 0, y: 50, x: '-50%' }}
            animate={{ opacity: 1, y: 0, x: '-50%' }}
            exit={{ opacity: 0, y: 50, x: '-50%' }}
            className={`fixed bottom-8 left-1/2 z-50 flex items-center gap-3 px-6 py-4 rounded-2xl shadow-2xl text-white font-bold text-sm ${
              toast.type === 'success'
                ? 'bg-gradient-to-r from-green-500 to-green-600 shadow-green-500/40'
                : 'bg-gradient-to-r from-red-500 to-red-600 shadow-red-500/40'
            }`}
          >
            {toast.type === 'success' ? (
              <IconCheck className="w-5 h-5 shrink-0" />
            ) : (
              <IconX className="w-5 h-5 shrink-0" />
            )}
            {toast.message}
          </motion.div>
        )}
      </AnimatePresence>

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
