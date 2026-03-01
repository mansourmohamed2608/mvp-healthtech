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
  IconNotes,
  IconClipboard,
  IconSettings,
  IconPlus,
  IconFolder,
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

// SOAP section styling
const soapSections = {
  subjective: { 
    label: 'Subjective', 
    labelAr: 'الشكوى الذاتية', 
    color: 'from-blue-500 to-blue-600',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/30',
    icon: IconUser 
  },
  objective: { 
    label: 'Objective', 
    labelAr: 'الفحص السريري', 
    color: 'from-emerald-500 to-emerald-600',
    bgColor: 'bg-emerald-500/10',
    borderColor: 'border-emerald-500/30',
    icon: IconStethoscope 
  },
  assessment: { 
    label: 'Assessment', 
    labelAr: 'التقييم', 
    color: 'from-amber-500 to-amber-600',
    bgColor: 'bg-amber-500/10',
    borderColor: 'border-amber-500/30',
    icon: IconNotes 
  },
  plan: { 
    label: 'Plan', 
    labelAr: 'خطة العلاج', 
    color: 'from-purple-500 to-purple-600',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/30',
    icon: IconClipboard 
  },
};

export default function ClinicalNotesNew() {
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
  const [activeTab, setActiveTab] = useState<'settings' | 'documents' | 'rag'>('settings');
  const [showNewPatient, setShowNewPatient] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const templateInputRef = useRef<HTMLInputElement>(null);
  const docInputRef = useRef<HTMLInputElement>(null);
  const fieldRecorderRef = useRef<MediaRecorder | null>(null);
  const fieldAudioChunksRef = useRef<Blob[]>([]);
  const fieldTimerRef = useRef<number | null>(null);

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

  const recordReviewMetrics = async (
    recording: AudioRecording,
    accepted: boolean,
    editedText: string
  ) => {
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
      setShowNewPatient(false);
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
    setExpandedPaths((prev) => ({ ...prev, [path]: !prev[path] }));
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
          className={`w-full text-right flex items-center justify-between gap-2 px-3 py-2 rounded-lg border transition-all ${
            selectedFieldPath === childPath && selectable
              ? 'border-cyan-400 bg-cyan-500/20 shadow-sm'
              : 'border-slate-700 bg-slate-800/50 hover:bg-slate-700/50'
          }`}
          style={{ paddingRight: indent + 12 }}
        >
          <span className="text-sm text-slate-200">
            {label}
            {selectable && (
              <span className="text-xs text-cyan-400 mr-2">قابل للتحديث</span>
            )}
          </span>
          <span className="text-xs text-slate-400">
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

      timerRef.current = window.setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
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
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });

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

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const processRecording = async (recording: AudioRecording) => {
    try {
      if (!patientId || !practitionerId) {
        throw new Error('الرجاء إدخال معرف المريض والطبيب');
      }
      setRecordings((prev) =>
        prev.map((r) => (r.id === recording.id ? { ...r, status: 'processing' } : r))
      );

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
              autoDetected: selectedDialect === 'auto',
            }
            : r
        )
      );
    } catch (error: any) {
      console.error('Processing error:', error);
      setRecordings((prev) =>
        prev.map((r) =>
          r.id === recording.id ? { ...r, status: 'error', error: error.message } : r
        )
      );
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

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
      alert('✅ تم الحفظ بنجاح!');
    } catch (error: any) {
      alert(`❌ خطأ في الحفظ: ${error.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const parseSoapSections = (soapText: string | undefined) => {
    if (!soapText) return { subjective: '', objective: '', assessment: '', plan: '' };

    const sections: { [key: string]: string } = { subjective: '', objective: '', assessment: '', plan: '' };
    const lines = soapText.split('\n');
    let currentSection = '';

    for (const line of lines) {
      const lowerLine = line.toLowerCase().trim();
      if (lowerLine.startsWith('subjective:')) {
        currentSection = 'subjective';
        sections[currentSection] = line.replace(/^subjective:/i, '').trim();
      } else if (lowerLine.startsWith('objective:')) {
        currentSection = 'objective';
        sections[currentSection] = line.replace(/^objective:/i, '').trim();
      } else if (lowerLine.startsWith('assessment:')) {
        currentSection = 'assessment';
        sections[currentSection] = line.replace(/^assessment:/i, '').trim();
      } else if (lowerLine.startsWith('plan:')) {
        currentSection = 'plan';
        sections[currentSection] = line.replace(/^plan:/i, '').trim();
      } else if (currentSection) {
        sections[currentSection] += (sections[currentSection] ? '\n' : '') + line;
      }
    }

    return sections;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900" dir="rtl">
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-[1800px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white">توثيق السجلات الطبية</h1>
              <p className="text-slate-400 text-sm">نظام SOAP الذكي</p>
            </div>
            <button
              onClick={() => {
                setShowMetrics(!showMetrics);
                if (!showMetrics && !metrics) loadMetrics();
              }}
              className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg font-medium transition-all"
            >
              <IconChartBar className="w-5 h-5" />
              {showMetrics ? 'إخفاء المقاييس' : 'المقاييس'}
            </button>
          </div>
        </div>
      </header>

      {/* Metrics Dashboard */}
      <AnimatePresence>
        {showMetrics && metrics && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-b border-slate-700/50 bg-slate-800/50"
          >
            <div className="max-w-[1800px] mx-auto px-6 py-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-slate-700/30 border border-slate-600/30 rounded-xl p-4">
                  <p className="text-slate-400 text-sm mb-1">إجمالي الملاحظات</p>
                  <p className="text-3xl font-bold text-white">{metrics.overview.totalNotes}</p>
                </div>
                <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4">
                  <p className="text-emerald-400 text-sm mb-1">معدل القبول</p>
                  <p className="text-3xl font-bold text-emerald-400">{(metrics.overview.acceptanceRate * 100).toFixed(1)}%</p>
                </div>
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
                  <p className="text-blue-400 text-sm mb-1">متوسط التعديل</p>
                  <p className="text-3xl font-bold text-blue-400">{metrics.overview.avgEditDistance.toFixed(1)}</p>
                </div>
                <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4">
                  <p className="text-amber-400 text-sm mb-1">وقت المراجعة (ث)</p>
                  <p className="text-3xl font-bold text-amber-400">{metrics.overview.avgReviewTime.toFixed(0)}</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="max-w-[1800px] mx-auto px-6 py-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Left Sidebar - Controls */}
          <div className="col-span-12 lg:col-span-3 space-y-4">
            {/* Recording Control */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">التسجيل</h3>
              <div className="text-center">
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    className="w-20 h-20 bg-gradient-to-br from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white rounded-full flex items-center justify-center mx-auto transition-all shadow-lg shadow-red-500/30 hover:shadow-red-500/50"
                  >
                    <IconMicrophone className="w-10 h-10" />
                  </button>
                ) : (
                  <button
                    onClick={stopRecording}
                    className="w-20 h-20 bg-gradient-to-br from-slate-600 to-slate-700 hover:from-slate-700 hover:to-slate-800 text-white rounded-full flex items-center justify-center mx-auto transition-all animate-pulse"
                  >
                    <IconPlayerStop className="w-10 h-10" />
                  </button>
                )}
                {isRecording && (
                  <p className="text-2xl font-mono text-red-400 font-bold mt-3">
                    {formatTime(recordingTime)}
                  </p>
                )}
                <p className="text-sm text-slate-400 mt-2">
                  {isRecording ? 'جاري التسجيل...' : 'انقر للبدء'}
                </p>
              </div>

              {/* File Upload */}
              <div className="mt-6 pt-4 border-t border-slate-700/50">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  multiple
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full bg-slate-700/50 hover:bg-slate-700 text-white py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2"
                >
                  <IconUpload className="w-5 h-5" />
                  رفع ملف صوتي
                </button>
              </div>

              {/* Dialect Selection */}
              <div className="mt-4">
                <label className="block text-sm text-slate-400 mb-2">اللهجة</label>
                <select
                  value={selectedDialect}
                  onChange={(e) => setSelectedDialect(e.target.value)}
                  className="w-full bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value="auto">كشف تلقائي</option>
                  <option value="egyptian">مصري</option>
                  <option value="levantine">شامي</option>
                  <option value="gulf">خليجي</option>
                  <option value="msa">فصحى</option>
                </select>
              </div>
            </div>

            {/* Settings Tabs */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-2xl overflow-hidden">
              <div className="flex border-b border-slate-700/50">
                {[
                  { id: 'settings', label: 'الإعدادات', icon: IconSettings },
                  { id: 'documents', label: 'المستندات', icon: IconFolder },
                  { id: 'rag', label: 'المعرفة', icon: IconNotes },
                ].map(({ id, label, icon: Icon }) => (
                  <button
                    key={id}
                    onClick={() => setActiveTab(id as any)}
                    className={`flex-1 py-3 px-2 text-sm font-medium transition-all flex items-center justify-center gap-1 ${activeTab === id
                      ? 'bg-cyan-600/20 text-cyan-400 border-b-2 border-cyan-400'
                      : 'text-slate-400 hover:text-white hover:bg-slate-700/30'
                      }`}
                  >
                    <Icon className="w-4 h-4" />
                    {label}
                  </button>
                ))}
              </div>

              <div className="p-4 max-h-[500px] overflow-y-auto">
                {activeTab === 'settings' && (
                  <div className="space-y-4">
                    {/* Patient Selection */}
                    <div>
                      <label className="block text-sm text-slate-400 mb-2">المريض</label>
                      <div className="flex gap-2">
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
                          className="flex-1 bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                        >
                          <option value="">-- اختر مريضاً --</option>
                          {patients.map((p) => (
                            <option key={p.id} value={p.id}>{p.displayName || p.id}</option>
                          ))}
                        </select>
                        <button
                          onClick={() => setShowNewPatient(!showNewPatient)}
                          className="p-2.5 bg-cyan-600 hover:bg-cyan-700 text-white rounded-xl transition-all"
                        >
                          <IconPlus className="w-5 h-5" />
                        </button>
                      </div>
                      {patientStatus === 'loading' && (
                        <p className="text-xs text-slate-400 mt-1">جارٍ التحميل...</p>
                      )}
                      {patientError && (
                        <p className="text-xs text-red-400 mt-1">{patientError}</p>
                      )}
                    </div>

                    {/* New Patient Form */}
                    <AnimatePresence>
                      {showNewPatient && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="bg-slate-700/30 rounded-xl p-4 space-y-3"
                        >
                          <input
                            value={newPatientName}
                            onChange={(e) => setNewPatientName(e.target.value)}
                            className="w-full bg-slate-800/50 border border-slate-600/50 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                            placeholder="اسم المريض"
                          />
                          <input
                            value={newPatientExternalId}
                            onChange={(e) => setNewPatientExternalId(e.target.value)}
                            className="w-full bg-slate-800/50 border border-slate-600/50 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                            placeholder="رقم ملف خارجي (اختياري)"
                            dir="ltr"
                          />
                          <button
                            onClick={createPatient}
                            className="w-full bg-emerald-600 hover:bg-emerald-700 text-white py-2 rounded-lg text-sm font-medium transition-all"
                          >
                            إضافة المريض
                          </button>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    {/* Practitioner */}
                    <div>
                      <label className="block text-sm text-slate-400 mb-2">معرف الطبيب</label>
                      <input
                        value={practitionerId}
                        onChange={(e) => setPractitionerId(e.target.value)}
                        className="w-full bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                        placeholder="clinician-456"
                        dir="ltr"
                      />
                    </div>

                    {/* Patient & Provider Names */}
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-sm text-slate-400 mb-2">اسم المريض</label>
                        <input
                          value={patientName}
                          onChange={(e) => setPatientName(e.target.value)}
                          className="w-full bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm text-slate-400 mb-2">اسم الطبيب</label>
                        <input
                          value={providerName}
                          onChange={(e) => setProviderName(e.target.value)}
                          className="w-full bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                        />
                      </div>
                    </div>

                    {/* Date of Visit */}
                    <div>
                      <label className="block text-sm text-slate-400 mb-2">تاريخ الزيارة</label>
                      <input
                        type="date"
                        value={dateOfVisit}
                        onChange={(e) => setDateOfVisit(e.target.value)}
                        className="w-full bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                        dir="ltr"
                      />
                    </div>

                    {/* Template Selection */}
                    <div>
                      <label className="block text-sm text-slate-400 mb-2">قالب الملاحظة</label>
                      <select
                        value={selectedTemplateId}
                        onChange={(e) => setSelectedTemplateId(e.target.value)}
                        className="w-full bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                      >
                        {templates.map((tpl) => (
                          <option key={tpl.id} value={tpl.id}>{tpl.name}</option>
                        ))}
                      </select>
                    </div>

                    {/* Template Upload */}
                    <div>
                      <label className="block text-sm text-slate-400 mb-2">رفع قالب JSON</label>
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
                          className="flex-1 bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                          placeholder="اسم القالب"
                        />
                        <button
                          onClick={() => templateInputRef.current?.click()}
                          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-medium transition-all"
                        >
                          رفع
                        </button>
                      </div>
                      {templateError && (
                        <p className="text-xs text-red-400 mt-1">{templateError}</p>
                      )}
                    </div>
                  </div>
                )}

                {activeTab === 'documents' && (
                  <div className="space-y-4">
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
                    <div className="flex gap-2">
                      <input
                        value={docTitle}
                        onChange={(e) => setDocTitle(e.target.value)}
                        className="flex-1 bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                        placeholder="عنوان الوثيقة"
                      />
                      <button
                        onClick={() => docInputRef.current?.click()}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-medium transition-all"
                      >
                        رفع
                      </button>
                    </div>
                    <textarea
                      value={docText}
                      onChange={(e) => setDocText(e.target.value)}
                      className="w-full bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500 resize-none"
                      placeholder="الصق بيانات المريض هنا"
                      rows={4}
                    />
                    {docFileName && (
                      <p className="text-xs text-slate-400">الملف: {docFileName}</p>
                    )}
                    <div className="flex items-center justify-between">
                      <label className="flex items-center gap-2 text-xs text-slate-400">
                        <input
                          type="checkbox"
                          checked={summarizeDoc}
                          onChange={(e) => setSummarizeDoc(e.target.checked)}
                          className="rounded bg-slate-700 border-slate-600"
                        />
                        تلخيص تلقائي
                      </label>
                      <button
                        onClick={uploadPatientDocument}
                        className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-sm font-medium transition-all"
                      >
                        حفظ
                      </button>
                    </div>
                    {docStatus === 'processing' && (
                      <p className="text-xs text-slate-400 flex items-center gap-2">
                        <IconLoader2 className="w-4 h-4 animate-spin" />
                        جارٍ الحفظ...
                      </p>
                    )}
                    {docStatus === 'done' && (
                      <p className="text-xs text-emerald-400 flex items-center gap-2">
                        <IconCheck className="w-4 h-4" />
                        تم الحفظ
                      </p>
                    )}
                    {docError && (
                      <p className="text-xs text-red-400">{docError}</p>
                    )}

                    {/* Patient Documents List */}
                    {patientDocs.length > 0 && (
                      <div className="mt-4 space-y-2">
                        <p className="text-sm text-slate-400">المستندات ({patientDocs.length})</p>
                        {patientDocs.slice(0, 5).map((doc) => (
                          <div key={doc.id} className="bg-slate-700/30 rounded-lg p-3">
                            <p className="text-sm text-white font-medium">{doc.title || 'وثيقة'}</p>
                            {(doc.summaryText || doc.summary_text) && (
                              <p className="text-xs text-slate-400 mt-1">{doc.summaryText || doc.summary_text}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Patient RAG Items */}
                    {patientRagItems.length > 0 && (
                      <div className="mt-4 space-y-2">
                        <p className="text-sm text-slate-400">سجل المريض RAG ({patientRagItems.length})</p>
                        {patientRagItems.slice(0, 8).map((item) => (
                          <div key={item.id} className="bg-slate-700/30 rounded-lg p-3">
                            <div className="flex items-center justify-between text-[10px] text-slate-500">
                              <span>{ragTypeLabel(item.itemType || item.item_type)}</span>
                              <span>{formatRagDate(item.createdAt || item.created_at)}</span>
                            </div>
                            <p className="text-sm text-white font-medium mt-1">{item.title || 'عنصر'}</p>
                            <p className="text-xs text-slate-400 mt-1">
                              {truncateSnippet(item.contentText || item.content_text || '')}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                    {ragItemsStatus === 'loading' && (
                      <p className="text-xs text-slate-400 flex items-center gap-2">
                        <IconLoader2 className="w-4 h-4 animate-spin" />
                        جارٍ التحميل...
                      </p>
                    )}
                  </div>
                )}

                {activeTab === 'rag' && (
                  <div className="space-y-4">
                    <p className="text-sm text-slate-400">أضف معرفة للعيادة (سياسات، تعليمات، معلومات)</p>
                    <input
                      value={ragTitle}
                      onChange={(e) => setRagTitle(e.target.value)}
                      className="w-full bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                      placeholder="عنوان (اختياري)"
                    />
                    <textarea
                      value={ragText}
                      onChange={(e) => setRagText(e.target.value)}
                      className="w-full bg-slate-700/50 border border-slate-600/50 text-white rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500 resize-none"
                      placeholder="المعلومات التي تريد إضافتها للنظام"
                      rows={6}
                    />
                    <button
                      onClick={submitRagNote}
                      className="w-full bg-emerald-600 hover:bg-emerald-700 text-white py-2 rounded-xl text-sm font-medium transition-all"
                    >
                      إضافة للمعرفة
                    </button>
                    {ragStatus === 'processing' && (
                      <p className="text-xs text-slate-400 flex items-center gap-2">
                        <IconLoader2 className="w-4 h-4 animate-spin" />
                        جارٍ الحفظ...
                      </p>
                    )}
                    {ragStatus === 'done' && (
                      <p className="text-xs text-emerald-400 flex items-center gap-2">
                        <IconCheck className="w-4 h-4" />
                        تم الحفظ
                      </p>
                    )}
                    {ragError && (
                      <p className="text-xs text-red-400">{ragError}</p>
                    )}

                    {/* Patient Context */}
                    {patientContext && (
                      <div className="mt-4">
                        <p className="text-sm text-slate-400 mb-2">سياق المريض المتاح</p>
                        <pre className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-3 text-[10px] text-slate-300 overflow-auto max-h-40">
                          {JSON.stringify(patientContext, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Middle - Recordings List */}
          <div className="col-span-12 lg:col-span-3">
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-4 h-full">
              <h3 className="text-lg font-semibold text-white mb-4">التسجيلات</h3>
              <div className="space-y-2 max-h-[calc(100vh-280px)] overflow-y-auto pr-2">
                {recordings.length === 0 ? (
                  <div className="text-center py-12">
                    <IconFileText className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                    <p className="text-slate-500">لا توجد تسجيلات</p>
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
                      className={`p-4 rounded-xl border cursor-pointer transition-all ${selectedRecording?.id === recording.id
                        ? 'border-cyan-500 bg-cyan-500/10'
                        : 'border-slate-700/50 bg-slate-700/20 hover:border-slate-600'
                        }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-white truncate">
                            {recording.file.name}
                          </p>
                          <p className="text-xs text-slate-400 mt-1">
                            {recording.timestamp.toLocaleString('ar-EG')}
                          </p>
                        </div>
                        <span
                          className={`px-2 py-1 rounded-full text-[10px] font-medium flex items-center gap-1 ${recording.status === 'completed'
                            ? 'bg-emerald-500/20 text-emerald-400'
                            : recording.status === 'processing'
                              ? 'bg-amber-500/20 text-amber-400'
                              : recording.status === 'error'
                                ? 'bg-red-500/20 text-red-400'
                                : 'bg-slate-500/20 text-slate-400'
                            }`}
                        >
                          {recording.status === 'completed' && <><IconCheck className="w-3 h-3" /> مكتمل</>}
                          {recording.status === 'processing' && <><IconLoader2 className="w-3 h-3 animate-spin" /> جاري</>}
                          {recording.status === 'error' && <><IconX className="w-3 h-3" /> خطأ</>}
                          {recording.status === 'pending' && <><IconClock className="w-3 h-3" /> انتظار</>}
                        </span>
                      </div>
                    </motion.div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Right - Details Panel */}
          <div className="col-span-12 lg:col-span-6">
            <AnimatePresence mode="wait">
              {selectedRecording ? (
                <motion.div
                  key={selectedRecording.id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-6 h-full"
                >
                  {selectedRecording.status === 'processing' && (
                    <div className="text-center py-16">
                      <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                      <p className="text-slate-400">جاري معالجة التسجيل...</p>
                    </div>
                  )}

                  {selectedRecording.status === 'error' && (
                    <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6">
                      <div className="flex items-center gap-3 text-red-400">
                        <IconX className="w-6 h-6" />
                        <p>خطأ: {selectedRecording.error}</p>
                      </div>
                    </div>
                  )}

                  {selectedRecording.status === 'completed' && (
                    <div className="space-y-6">
                      {/* Transcript */}
                      <div>
                        <h3 className="text-sm font-medium text-slate-400 mb-2 flex items-center gap-2">
                          <IconFileText className="w-4 h-4" />
                          النص المكتوب
                        </h3>
                        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 max-h-32 overflow-y-auto">
                          <p className="text-white text-sm leading-relaxed">
                            {selectedRecording.transcript}
                          </p>
                        </div>
                      </div>

                      {/* SOAP Note - Color Coded Sections */}
                      <div>
                        <h3 className="text-sm font-medium text-slate-400 mb-3 flex items-center gap-2">
                          <IconNotes className="w-4 h-4" />
                          ملاحظة SOAP
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {Object.entries(soapSections).map(([key, config]) => {
                            const sections = parseSoapSections(editedSoapNote || selectedRecording.soapNote);
                            const Icon = config.icon;
                            return (
                              <div
                                key={key}
                                className={`${config.bgColor} border ${config.borderColor} rounded-xl p-4`}
                              >
                                <div className="flex items-center gap-2 mb-2">
                                  <div className={`w-8 h-8 bg-gradient-to-br ${config.color} rounded-lg flex items-center justify-center`}>
                                    <Icon className="w-4 h-4 text-white" />
                                  </div>
                                  <div>
                                    <p className="text-white font-medium text-sm">{config.label}</p>
                                    <p className="text-slate-400 text-xs">{config.labelAr}</p>
                                  </div>
                                </div>
                                <p className="text-slate-300 text-sm leading-relaxed whitespace-pre-wrap">
                                  {sections[key] || 'لا توجد بيانات'}
                                </p>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      {/* Full SOAP Editor */}
                      <div>
                        <h3 className="text-sm font-medium text-slate-400 mb-2">تحرير كامل</h3>
                        <textarea
                          value={editedSoapNote || selectedRecording.soapNote}
                          onChange={(e) => {
                            setEditedSoapNote(e.target.value);
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
                          className="w-full bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-white text-sm leading-relaxed focus:outline-none focus:ring-2 focus:ring-cyan-500 resize-none"
                          rows={8}
                        />
                      </div>

                      {/* Field Update Section */}
                      {selectedRecording.noteId && selectedRecording.soapJson && (
                        <div className="bg-slate-700/30 border border-slate-600/30 rounded-xl p-4">
                          <h3 className="text-sm font-medium text-slate-400 mb-3 flex items-center gap-2">
                            <IconMicrophone className="w-4 h-4" />
                            تحديث حقل بالصوت أو النص
                          </h3>
                          <div className="grid gap-4 lg:grid-cols-2">
                            <div>
                              <label className="block text-xs text-slate-400 mb-2">شجرة الحقول</label>
                              <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-3 max-h-48 overflow-y-auto space-y-1">
                                {renderJsonTree(selectedRecording.soapJson)}
                              </div>
                            </div>
                            <div className="space-y-3">
                              <div className="text-xs text-slate-400">
                                الحقل: <span className="text-cyan-400">{selectedFieldPath || 'غير محدد'}</span>
                              </div>
                              <select
                                value={fieldUpdateMode}
                                onChange={(e) => setFieldUpdateMode(e.target.value as 'append' | 'replace')}
                                className="w-full bg-slate-800/50 border border-slate-600/50 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                              >
                                <option value="append">إضافة</option>
                                <option value="replace">استبدال</option>
                              </select>
                              <textarea
                                value={fieldUpdateText}
                                onChange={(e) => setFieldUpdateText(e.target.value)}
                                className="w-full bg-slate-800/50 border border-slate-600/50 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500 resize-none"
                                rows={3}
                                placeholder="أضف النص هنا"
                              />
                              <button
                                onClick={submitFieldTextUpdate}
                                className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded-lg text-sm font-medium transition-all"
                              >
                                تحديث بالنص
                              </button>
                              <div className="flex gap-2">
                                {!isFieldRecording ? (
                                  <button
                                    onClick={startFieldRecording}
                                    className="flex-1 bg-purple-600 hover:bg-purple-700 text-white py-2 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2"
                                  >
                                    <IconMicrophone className="w-4 h-4" />
                                    تسجيل
                                  </button>
                                ) : (
                                  <button
                                    onClick={stopFieldRecording}
                                    className="flex-1 bg-red-600 hover:bg-red-700 text-white py-2 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2"
                                  >
                                    <IconPlayerStop className="w-4 h-4" />
                                    إيقاف ({formatTime(fieldRecordingTime)})
                                  </button>
                                )}
                              </div>
                              {fieldUpdateStatus === 'processing' && (
                                <p className="text-xs text-slate-400 flex items-center gap-2">
                                  <IconLoader2 className="w-4 h-4 animate-spin" />
                                  جارِ التحديث...
                                </p>
                              )}
                              {fieldUpdateStatus === 'done' && (
                                <p className="text-xs text-emerald-400 flex items-center gap-2">
                                  <IconCheck className="w-4 h-4" />
                                  تم التحديث
                                </p>
                              )}
                              {fieldUpdateError && (
                                <p className="text-xs text-red-400">{fieldUpdateError}</p>
                              )}
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Action Buttons */}
                      <div className="flex gap-3 pt-4 border-t border-slate-700/50">
                        <button
                          onClick={async () => {
                            const finalText = editedSoapNote || selectedRecording.soapNote || '';
                            await recordReviewMetrics(selectedRecording, true, finalText);
                            await saveToEHR();
                          }}
                          disabled={isSaving}
                          className="flex-1 bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 disabled:from-slate-600 disabled:to-slate-700 text-white font-medium py-3 px-6 rounded-xl transition-all flex items-center justify-center gap-2"
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
                        </button>
                        <button
                          onClick={async () => {
                            const finalText = editedSoapNote || selectedRecording.soapNote || '';
                            await recordReviewMetrics(selectedRecording, false, finalText);
                            if (selectedRecording.noteId) {
                              await api.rejectSOAPNote(selectedRecording.noteId);
                            }
                            alert('تم رفض الملاحظة');
                          }}
                          className="flex-1 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white font-medium py-3 px-6 rounded-xl transition-all flex items-center justify-center gap-2"
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
                          className="bg-slate-700 hover:bg-slate-600 text-white py-3 px-4 rounded-xl transition-all"
                          title="تنزيل"
                        >
                          <IconFileDownload className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-6 h-full flex items-center justify-center"
                >
                  <div className="text-center">
                    <IconNotes className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                    <p className="text-slate-500">اختر تسجيلاً لعرض التفاصيل</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
